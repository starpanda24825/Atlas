"""
Atlas — structured memory.

:mod:`memory.chroma_store` stores vectors. This layer stores *facts*: something
reads a conversation, decides which statements about the user are worth keeping,
checks them against what is already known, and writes only what is new. That is
what ``mem0`` does, and it is the part that cannot be faked with a vector index —
deduplication and contradiction handling need reasoning, not similarity.

Three things it does that a plain vector store cannot
-----------------------------------------------------

* **Extraction.** "I'm flying to Lisbon on the 14th and I've gone off espresso"
  becomes two memories, not one embedded paragraph.
* **Deduplication.** Saying the same thing on three days leaves one memory, not
  three near-identical vectors that crowd out everything else in a top-5 search.
* **Temporal updates.** "I used to drink flat whites, now I'm on cortados" edits
  the existing preference instead of adding a contradicting one beside it.

What this module wires up
-------------------------

``mem0`` is assembled from three parts, all local:

=================  =========================================================
vector store       ChromaDB, the *same* client and directory the rest of
                   Atlas uses, in its own ``atlas_memory`` collection
LLM                the fast llama.cpp server, via its OpenAI-compatible API
embeddings         this project's own :class:`~memory.chroma_store.EmbeddingProvider`
=================  =========================================================

Each of those is a deliberate substitution, and each one is explained where it
is made. In short: the LLM must be told to **stop thinking** (Qwen3 otherwise
spends ~20x the latency reasoning and then hands back an unusable JSON object);
the embedder must be **the project's own** (mem0's HuggingFace embedder would
load a second copy of the same weights, skip nomic's task prefixes, and skip
normalisation); and the vector store is handed an existing client so nothing
opens a competing handle on the same directory.

Two import-time side effects, handled
-------------------------------------

``mem0`` does two things the moment it is imported, so the environment is
prepared *before* the import at the top of this file:

* telemetry is on by default and posts to PostHog with a hardcoded key. Atlas
  runs on someone's own machine and their conversations are the payload, so
  ``MEM0_TELEMETRY`` is set to ``false`` unless the user opts in.
* ``~/.mem0`` is created for its history database. ``MEM0_DIR`` is pointed at
  ``CHROMA_DIR/mem0`` instead, keeping every derived file in one gitignored
  directory that can be deleted without losing anything the user wrote.

Usage::

    from memory.mem0_manager import get_mem0_manager

    memories = get_mem0_manager()
    memories.add_from_exchange("I've gone off espresso", "Noted.")   # background

    print(memories.search("what coffee does the user like?"))
    # What Atlas remembers about the user:
    # - Prefers cortados over espresso.

Run the built-in check with::

    python3 -m memory.mem0_manager

It needs the fast model server running (``core/llm_manager.py``) because fact
extraction is a real LLM call; it skips those checks with a clear message when
the server is not up.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.config import (
    CHROMA_DIR,
    FAST_MODEL_PATH,
    FAST_MODEL_PORT,
    LLM_HOST,
    MEM0_COLLECTION,
    MEM0_HISTORY_DB,
    MEM0_INCLUDE_RESPONSE,
    MEM0_LLM_MAX_TOKENS,
    MEM0_SEARCH_LIMIT,
    MEM0_USER_ID,
    MEM0_DIR,
)
from memory.chroma_store import (
    ChromaStore,
    EmbeddingProvider,
    build_embedding_provider,
    get_chroma_store,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# mem0's import-time side effects, neutralised before it is imported
# ---------------------------------------------------------------------------
#
# ``setdefault`` rather than assignment: a user who deliberately exports
# MEM0_TELEMETRY=true is opting in to sending their data, and that is their call
# to make. Everything else gets the private, local default.

os.environ.setdefault("MEM0_TELEMETRY", "false")
os.environ.setdefault("MEM0_DIR", str(MEM0_DIR))

import mem0  # noqa: E402  (must follow the environment setup above)
from mem0.configs.base import MemoryConfig  # noqa: E402
from mem0.configs.llms.openai import OpenAIConfig  # noqa: E402
from mem0.embeddings.base import EmbeddingBase  # noqa: E402
from mem0.llms.openai import OpenAILLM  # noqa: E402
from mem0.memory.telemetry import MEM0_TELEMETRY  # noqa: E402
from mem0.utils.factory import EmbedderFactory, LlmFactory  # noqa: E402

__all__ = [
    "CONTEXT_HEADING",
    "AtlasEmbedder",
    "AtlasMem0LLM",
    "Mem0Manager",
    "get_mem0_manager",
    "reset_mem0_manager",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Provider names registered with mem0's factories for the two classes below.
#: Registered under our own names rather than overriding mem0's, so nothing else
#: in the process that uses mem0 gets different behaviour.
ATLAS_EMBEDDER_PROVIDER = "atlas"
ATLAS_LLM_PROVIDER = "atlas"

#: Written verbatim into every prompt. The agent's system prompt refers to this
#: wording, so the two must stay in step.
CONTEXT_HEADING = "What Atlas remembers about the user:"

#: Model name sent to llama.cpp. It ignores the field (the server serves
#: whichever GGUF it was started with), but mem0's reasoning-model heuristic
#: reads it, so it must not look like "o1"/"gpt-5". The real filename is used so
#: logs say which model actually answered.
DEFAULT_LLM_MODEL = FAST_MODEL_PATH.stem

#: mem0's own default search threshold. Its similarity scale is not the same as
#: Chroma's cosine distance, so this is left at mem0's value rather than
#: reusing MEMORY_RELEVANCE_THRESHOLD.
DEFAULT_SEARCH_THRESHOLD = 0.1

#: Generous ceiling for "give me everything", used by get_all().
ALL_MEMORIES_LIMIT = 5000


def default_llm_base_url() -> str:
    """OpenAI-compatible base URL of the fast llama.cpp server."""
    return f"http://{LLM_HOST}:{FAST_MODEL_PORT}/v1"


# ---------------------------------------------------------------------------
# The two substituted parts
# ---------------------------------------------------------------------------


class AtlasEmbedder(EmbeddingBase):
    """mem0's embedding slot, filled by this project's own provider.

    mem0's bundled ``HuggingFaceEmbedding`` would technically work and is the
    wrong choice three times over:

    * it builds its **own** ``SentenceTransformer``, so the same 600 MB of
      weights would sit in memory twice whenever the vault or the agent is also
      loaded;
    * it ignores ``memory_action``, so nomic never receives the
      ``search_document:`` / ``search_query:`` prefixes it is trained on — the
      single cheapest accuracy regression available here;
    * it does not normalise its vectors. mem0 creates its Chroma collection
      without ``hnsw:space``, so the space is L2, and only normalised vectors
      make L2 rank the same way cosine does. Skipping it silently degrades every
      ranking.

    Settings arrive through ``model_kwargs`` because that is the only field on
    mem0's embedder config that accepts arbitrary keys. ``store`` is the one
    entry that is not a model setting: it is the existing
    :class:`~memory.chroma_store.ChromaStore` whose already-loaded provider
    should be reused.
    """

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        options = dict(getattr(self.config, "model_kwargs", None) or {})
        self._store: ChromaStore | None = options.pop("store", None)
        self._provider_choice = options.pop("provider", None)
        self._model_choice = options.pop("model", None) or getattr(
            self.config, "model", None
        )
        self._device_choice = options.pop("device", None)
        self._extra_options = options
        self._provider: EmbeddingProvider | None = None
        self._lock = threading.Lock()

    @property
    def provider(self) -> EmbeddingProvider:
        """The embedder, resolved and loaded on first use."""
        if self._provider is None:
            with self._lock:
                if self._provider is None:
                    if self._store is not None:
                        # Reuse the store's model rather than loading a second
                        # copy of identical weights.
                        self._provider = self._store.embedding_provider
                    else:
                        self._provider = build_embedding_provider(
                            self._provider_choice,
                            model=self._model_choice,
                            device=self._device_choice,
                            **self._extra_options,
                        )
                    if getattr(self.config, "embedding_dims", None) is None:
                        self.config.embedding_dims = self._provider.dimension
                    logger.info("mem0 embeddings: %s", self._provider.describe())
        return self._provider

    def embed(
        self, text: str, memory_action: str | None = None
    ) -> list[float]:
        """Embed one string.

        ``search`` gets the query prefix; ``add`` and ``update`` get the
        document prefix, which is what nomic's asymmetric training expects.
        """
        if memory_action == "search":
            return self.provider.embed_query(text)
        return self.provider.embed_documents([text])[0]

    def embed_batch(
        self, texts: Sequence[str], memory_action: str = "add"
    ) -> list[list[float]]:
        """Embed several strings in one batched pass."""
        cleaned = [str(text) for text in texts]
        if not cleaned:
            return []
        if memory_action == "search":
            return [self.provider.embed_query(text) for text in cleaned]
        return self.provider.embed_documents(cleaned)


class AtlasMem0LLM(OpenAILLM):
    """mem0's LLM slot, pointed at the fast llama.cpp server.

    The only behavioural change is ``extra_body``. Qwen3 reasons before it
    answers unless told not to, and mem0 asks for a JSON object — so reasoning
    would cost roughly twenty times the latency and then spend the token budget
    before producing the JSON that was actually wanted. The agent already turns
    thinking off this exact way; this makes mem0 do the same.

    ``extra_body`` is injected in ``_get_common_params`` because that is the one
    hook on the base class whose result flows into every request, and because
    mem0's own config schema has no field for a per-request body extension.
    """

    def _get_common_params(self, **kwargs: Any) -> dict[str, Any]:
        params = super()._get_common_params(**kwargs)
        params.setdefault("extra_body", {})
        params["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
        return params


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

#: mem0's config validators reject unknown provider names, so any custom class
#: has to be registered under a name that already exists in its factories. The
#: config is built with those names and the provider is then switched to ours —
#: pydantic does not re-validate on plain attribute assignment, which is
#: exactly the seam this needs.
_registration_lock = threading.Lock()
_registered = False


def _register_providers() -> None:
    """Teach mem0's factories about the two classes above. Idempotent."""
    global _registered
    if _registered:
        return
    with _registration_lock:
        if _registered:
            return
        EmbedderFactory.provider_to_class[ATLAS_EMBEDDER_PROVIDER] = (
            "memory.mem0_manager.AtlasEmbedder"
        )
        LlmFactory.register_provider(
            ATLAS_LLM_PROVIDER, "memory.mem0_manager.AtlasMem0LLM", OpenAIConfig
        )
        _registered = True
        logger.debug("registered mem0 providers: %s", ATLAS_EMBEDDER_PROVIDER)


# ---------------------------------------------------------------------------
# The manager
# ---------------------------------------------------------------------------


class Mem0Manager:
    """Fact extraction, deduplication and recall over ChromaDB.

    Construction is cheap; mem0 itself — and with it the embedding model — is
    built on first use, so importing this module does not load torch.

    The ``store`` argument is the important one. Passing the same
    :class:`~memory.chroma_store.ChromaStore` the rest of Atlas uses means mem0
    writes into the same directory through the same client and shares its
    embedding model, instead of opening a second handle on the database and
    loading a second copy of the weights.
    """

    def __init__(
        self,
        user_id: str | None = None,
        *,
        store: ChromaStore | None = None,
        chroma_dir: str | Path | None = None,
        chroma_client: Any | None = None,
        collection: str | None = None,
        mem0_dir: str | Path | None = None,
        history_db: str | Path | None = None,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_max_tokens: int | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        embedding_device: str | None = None,
        context_limit: int | None = None,
        search_threshold: float | None = None,
        **llm_options: Any,
    ) -> None:
        self.user_id = user_id or MEM0_USER_ID
        self.collection = collection or MEM0_COLLECTION

        self._store = store
        self._chroma_dir = Path(chroma_dir) if chroma_dir else None
        self._chroma_client = chroma_client

        self.mem0_dir = Path(mem0_dir or MEM0_DIR)
        self.history_db = Path(history_db or MEM0_HISTORY_DB)

        self.llm_model = llm_model or DEFAULT_LLM_MODEL
        self.llm_base_url = llm_base_url or default_llm_base_url()
        self.llm_max_tokens = llm_max_tokens or MEM0_LLM_MAX_TOKENS
        self.llm_options = llm_options

        self._embedding_provider_choice = embedding_provider
        self._embedding_model_choice = embedding_model
        self._embedding_device_choice = embedding_device

        self.context_limit = context_limit or MEM0_SEARCH_LIMIT
        self.search_threshold = (
            DEFAULT_SEARCH_THRESHOLD if search_threshold is None else search_threshold
        )
        self.include_response = MEM0_INCLUDE_RESPONSE

        self._memory: Any | None = None
        self._build_lock = threading.RLock()
        # Only one extraction at a time: a second conversation turn makes the
        # first one's work redundant, so it is dropped rather than queued.
        self._add_lock = threading.Lock()
        self._pending: set[threading.Thread] = set()
        self._pending_lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"<Mem0Manager user={self.user_id!r} collection={self.collection!r} "
            f"built={self._memory is not None}>"
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @property
    def store(self) -> ChromaStore:
        """The Chroma store whose directory, client and embedder are shared."""
        if self._store is None:
            self._store = get_chroma_store()
        return self._store

    @property
    def memory(self) -> Any:
        """The underlying ``mem0.Memory``, built on first access."""
        if self._memory is None:
            with self._build_lock:
                if self._memory is None:
                    self._memory = self._build()
        return self._memory

    def _build(self) -> Any:
        """Assemble mem0 with local components. Blocking; loads the embedder."""
        _register_providers()

        # mem0 creates this directory for its history database and config.json.
        self.mem0_dir.mkdir(parents=True, exist_ok=True)

        store = self.store
        chroma_dir = self._chroma_dir or Path(store.persist_dir)
        client = self._chroma_client or store.client

        raw: dict[str, Any] = {
            "vector_store": {
                "provider": "chroma",
                # An existing client is passed through so mem0 does not open a
                # second handle on a database this process already has open.
                # Chroma's own default space is L2; our embedder normalises, so
                # L2 ranks like cosine and no hnsw:space override is needed.
                "config": {
                    "collection_name": self.collection,
                    "client": client,
                    "path": str(chroma_dir),
                },
            },
            "llm": {
                # "openai" here only satisfies mem0's validator; the provider is
                # swapped to ours below.
                "provider": "openai",
                "config": {
                    "model": self.llm_model,
                    "openai_base_url": self.llm_base_url,
                    # llama.cpp ignores the key, but the SDK requires one.
                    "api_key": "none",
                    "temperature": 0.1,
                    "max_tokens": self.llm_max_tokens,
                    # Set explicitly rather than relying on mem0's name-based
                    # heuristic, which would silently drop temperature and
                    # top_p for anything that looked like a reasoning model.
                    "is_reasoning_model": False,
                    **self.llm_options,
                },
            },
            "embedder": {
                "provider": "huggingface",  # also just to pass validation
                "config": {
                    "model": self._embedding_model_choice,
                    "model_kwargs": {
                        "store": store,
                        "provider": self._embedding_provider_choice,
                        "model": self._embedding_model_choice,
                        "device": self._embedding_device_choice,
                    },
                },
            },
            "history_db_path": str(self.history_db),
        }

        config = MemoryConfig(**raw)
        # Validators have run; point both providers at our own classes now.
        # Plain assignment is not re-validated, which is what makes this work.
        config.embedder.provider = ATLAS_EMBEDDER_PROVIDER
        config.llm.provider = ATLAS_LLM_PROVIDER

        from mem0 import Memory

        started = time.time()
        memory = Memory(config)
        logger.info(
            "mem0 ready: collection=%s llm=%s (%.1fs)",
            self.collection,
            self.llm_model,
            time.time() - started,
        )
        return memory

    def warm(self) -> None:
        """Build mem0 and load the embedder now rather than on first recall."""
        self.memory
        self.memory.embedding_model.provider.embed_documents(["warm up"])

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_from_exchange(
        self,
        user_input: str,
        agent_response: str = "",
        *,
        include_response: bool | None = None,
        background: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Learn from one exchange, in the background by default.

        Only the user's turn is sent by default. Passing the assistant's reply
        as well is the obvious reading of "the exchange", and it was tried:
        measured against Qwen3-8B it produced memories like "User was reminded
        they drink cortados" — a fact about Atlas's own sentence, competing for
        a recall slot with the real one. Extraction from the user's turn alone
        kept the facts and dropped the noise, so that is the default.

        Set ``include_response=True`` (or ``MEM0_INCLUDE_RESPONSE=true``) to send
        both, which is the right call with a stronger extraction model that can
        reliably attribute a statement to a speaker.

        Returns the created/updated memories when ``background=False``, and
        ``None`` when the work was handed to a thread. Only one extraction runs
        at a time; a turn arriving mid-extraction is dropped, because the newer
        turn supersedes it anyway.
        """
        send_both = (
            self.include_response if include_response is None else include_response
        )
        messages = [{"role": "user", "content": str(user_input)}]
        if send_both and str(agent_response or "").strip():
            messages.append({"role": "assistant", "content": str(agent_response)})
        if not background:
            return self._add(messages, metadata)

        if not self._add_lock.acquire(blocking=False):
            logger.debug("a mem0 extraction is already running — skipping this turn")
            return None

        def worker() -> None:
            try:
                self._add(messages, metadata)
            except Exception:
                # Memory is a nice-to-have; it must never break a conversation.
                logger.exception("mem0 extraction failed")
            finally:
                self._add_lock.release()

        thread = threading.Thread(target=worker, name="atlas-mem0-add", daemon=True)
        with self._pending_lock:
            self._pending = {item for item in self._pending if item.is_alive()}
            self._pending.add(thread)
        thread.start()
        return None

    def _add(
        self,
        messages: Sequence[Mapping[str, str]],
        metadata: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Run mem0's extraction, returning what it decided to keep."""
        options: dict[str, Any] = {"user_id": self.user_id, "infer": True}
        if metadata:
            options["metadata"] = dict(metadata)

        started = time.time()
        result = self.memory.add(list(messages), **options)
        results = (result or {}).get("results", []) if isinstance(result, dict) else []
        logger.info(
            "mem0 processed an exchange in %.1fs -> %d change(s): %s",
            time.time() - started,
            len(results),
            ", ".join(str(item.get("event", "?")) for item in results) or "none",
        )
        return results

    def extract_and_store(
        self, user_input: str, agent_response: str, **metadata: Any
    ) -> list[dict[str, Any]]:
        """The agent's per-turn background hook.

        Runs synchronously, because :class:`~core.agent.AtlasAgent` already
        calls this from its own worker thread after every exchange — spawning a
        second thread here would double the work and let the agent's shutdown
        wait return before the memories were actually written.
        """
        return self.add_from_exchange(
            user_input, agent_response, background=False, metadata=metadata or None
        )

    def save(self, content: str) -> dict[str, Any]:
        """Store one explicit fact.

        Backs the agent's ``save_memory`` tool, which is only reached when the
        user has explicitly asked for something to be remembered. So if
        extraction finds nothing worth keeping — and it does sometimes judge a
        lone sentence unmemorable — the text is stored verbatim rather than
        silently dropped. An explicit request is honoured.

        Extraction still runs first, because that is where deduplication and
        contradiction handling live.
        """
        text = str(content or "").strip()
        if not text:
            raise ValueError("save() needs some content")

        results = self._add([{"role": "user", "content": text}], None)
        if results:
            return {"saved": True, "verbatim": False, "memories": results}

        logger.info("extraction kept nothing from an explicit save — storing verbatim")
        result = self.memory.add(
            [{"role": "user", "content": text}], user_id=self.user_id, infer=False
        )
        stored = (result or {}).get("results", []) if isinstance(result, dict) else []
        return {"saved": bool(stored), "verbatim": True, "memories": stored}

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int | None = None,
        *,
        threshold: float | None = None,
    ) -> str:
        """Return relevant memories as a prompt-ready section.

        The string is what gets injected into the agent's context, heading and
        all, so the model can see where the facts came from. Returns an empty
        string when there is nothing relevant, which lets the caller skip the
        section entirely rather than announce that memory is empty.

        No LLM call happens here — retrieval is one embedding and one vector
        query — so this is cheap enough to run on every turn.
        """
        return format_context(self.search_memories(query, limit, threshold=threshold))

    def search_memories(
        self,
        query: str,
        limit: int | None = None,
        *,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """The structured form of :meth:`search`: one dict per memory."""
        cleaned = str(query or "").strip()
        if not cleaned:
            return []

        top_k = int(limit or self.context_limit)
        if top_k <= 0:
            return []

        try:
            result = self.memory.search(
                cleaned,
                filters={"user_id": self.user_id},
                top_k=top_k,
                threshold=self.search_threshold if threshold is None else threshold,
            )
        except Exception:
            logger.exception("mem0 search failed")
            return []

        raw = (result or {}).get("results", []) if isinstance(result, dict) else []
        return [normalise_memory(item) for item in raw]

    def context_for(self, query: str, limit: int | None = None) -> str:
        """The section to inject into a prompt. Alias of :meth:`search`.

        Exists under an unambiguous name so the agent's duck-typed lookup cannot
        accidentally bind to some other ``search`` on a composite memory object.
        """
        return self.search(query, limit)

    def retrieve(self, query: str) -> str:
        """Recall for the agent's ``retrieve_memory`` tool. Alias of :meth:`search`."""
        return self.search(query)

    def get_all(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Every memory for this user, newest first — for display in a UI."""
        try:
            result = self.memory.get_all(
                filters={"user_id": self.user_id},
                top_k=int(limit or ALL_MEMORIES_LIMIT),
            )
        except Exception:
            logger.exception("mem0 get_all failed")
            return []

        raw = (result or {}).get("results", []) if isinstance(result, dict) else []
        memories = [normalise_memory(item) for item in raw]
        memories.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return memories

    # ------------------------------------------------------------------
    # Removing
    # ------------------------------------------------------------------

    def delete(self, memory_id: str) -> bool:
        """Remove one memory by id. Returns False if it was not there."""
        if not memory_id:
            raise ValueError("delete() needs a memory id")
        try:
            self.memory.delete(str(memory_id))
            logger.info("deleted memory %s", memory_id)
            return True
        except Exception as exc:
            logger.warning("could not delete memory %s: %s", memory_id, exc)
            return False

    def delete_all(self) -> bool:
        """Forget everything about this user.

        The vault is not touched — ``vault/memories/*.md`` are the user's own
        notes and outlive the index. Re-running ``index_vault()`` afterwards
        restores anything that was indexed from them.
        """
        try:
            self.memory.delete_all(user_id=self.user_id)
            logger.info("cleared all memories for %s", self.user_id)
            return True
        except Exception:
            logger.exception("mem0 delete_all failed")
            return False

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def consolidate(self) -> dict[str, Any]:
        """Idle housekeeping: drop memories that are exact duplicates.

        Called automatically by :class:`core.resource_governor.ResourceGovernor`
        during the idle-compute window, hence the deliberately narrow definition
        of "consolidate": it removes a memory only when its *normalised* text is
        identical to a newer one, so no distinct information is ever lost. It
        never merges, rewrites or summarises — an automatic pass that silently
        rewrote the user's memories would be impossible to audit.

        Returns a report dict; never raises.
        """
        try:
            flushed = self.wait_for_pending()
        except Exception:
            logger.debug("wait_for_pending failed before consolidation", exc_info=True)
            flushed = False

        try:
            memories = self.get_all()
        except Exception:
            logger.exception("could not read memories for consolidation")
            return {"flushed": flushed, "total": 0, "duplicates": 0, "removed": 0}

        # get_all() is newest-first, so the first time a text is seen is the
        # copy worth keeping.
        seen: set[str] = set()
        duplicates: list[str] = []
        for item in memories:
            text = _normalise_memory_text(item.get("memory") or "")
            if not text:
                continue
            if text in seen:
                memory_id = str(item.get("id") or "")
                if memory_id:
                    duplicates.append(memory_id)
            else:
                seen.add(text)

        removed = sum(1 for memory_id in duplicates if self.delete(memory_id))
        report = {
            "flushed": flushed,
            "total": len(memories),
            "duplicates": len(duplicates),
            "removed": removed,
            "kept": len(memories) - removed,
        }
        logger.info("memory consolidation: %s", report)
        return report

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def wait_for_pending(self, timeout: float = 60.0) -> bool:
        """Block until any in-flight extraction has finished.

        For shutdown: ``add_from_exchange`` returns immediately, so a daemon
        exiting without this can drop the last turn's memories.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._pending_lock:
                alive = [item for item in self._pending if item.is_alive()]
                self._pending = set(alive)
            if not alive:
                return True
            time.sleep(0.1)
        logger.warning("mem0 extraction still running after %.0fs", timeout)
        return False

    def count(self) -> int:
        """How many memories are stored. `-1` when mem0 cannot say."""
        try:
            return len(self.get_all())
        except Exception:
            return -1

    def health(self) -> dict[str, Any]:
        """Snapshot for the API's status endpoint. Never raises."""
        report: dict[str, Any] = {
            "user_id": self.user_id,
            "collection": self.collection,
            "built": self._memory is not None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "history_db": str(self.history_db),
            # Reported rather than assumed: if this is True, conversations are
            # being sent to PostHog.
            "telemetry": bool(MEM0_TELEMETRY),
            "mem0_version": getattr(mem0, "__version__", "unknown"),
        }
        try:
            report["chroma_dir"] = str(self._chroma_dir or self.store.persist_dir)
            report["count"] = self.count()
        except Exception as exc:
            report["count"] = None
            report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    def close(self) -> None:
        """Drop the mem0 instance. Everything is already on disk."""
        self.wait_for_pending(timeout=5.0)
        with self._build_lock:
            self._memory = None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _normalise_memory_text(text: Any) -> str:
    """Case- and whitespace-insensitive form, for spotting exact duplicates."""
    return " ".join(str(text or "").lower().split())


def normalise_memory(item: Any) -> dict[str, Any]:
    """Flatten one mem0 result into a plain dict with predictable keys."""
    if isinstance(item, Mapping):
        get = item.get
    else:  # a pydantic object from a newer mem0
        get = lambda key, default=None: getattr(item, key, default)  # noqa: E731

    metadata = get("metadata") or {}
    return {
        "id": get("id") or "",
        "memory": get("memory") or get("data") or "",
        "score": get("score"),
        "created_at": get("created_at") or metadata.get("created_at"),
        "updated_at": get("updated_at") or metadata.get("updated_at"),
        "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
    }


def format_context(
    memories: Iterable[Mapping[str, Any]],
    *,
    heading: str = CONTEXT_HEADING,
    max_chars: int = 1200,
) -> str:
    """Render memories as the section injected into agent prompts.

    Capped in length: this shares the context window with the conversation, and
    a wall of recalled facts would crowd out the thing the user just asked.
    """
    lines: list[str] = []
    used = len(heading)
    for item in memories:
        text = str(item.get("memory") or "").strip()
        if not text:
            continue
        if used + len(text) + 3 > max_chars:
            break
        lines.append(f"- {text}")
        used += len(text) + 3

    if not lines:
        return ""
    return "\n".join([heading, *lines])


# ---------------------------------------------------------------------------
# Shared instance
# ---------------------------------------------------------------------------

_manager: Mem0Manager | None = None
_manager_lock = threading.Lock()


def get_mem0_manager(**kwargs: Any) -> Mem0Manager:
    """Process-wide memory manager, built once.

    The agent, the vault and the API should all share it: each instance would
    otherwise build its own mem0 and hold its own SQLite connection.
    """
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = Mem0Manager(**kwargs)
    return _manager


def reset_mem0_manager() -> None:
    """Drop the shared instance so the next call rebuilds it."""
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.close()
        _manager = None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> int:
    """Exercise the manager against a throwaway store and a real server.

    Fact extraction is a genuine LLM call, so the checks that need it are
    skipped with a clear message rather than faked when the fast server is not
    running.
    """
    import shutil
    import tempfile

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s"
    )
    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        mark = "ok  " if condition else "FAIL"
        print(f"  [{mark}] {label}{f' — {detail}' if detail else ''}")
        if not condition:
            failures.append(label)

    def skip(label: str, reason: str) -> None:
        print(f"  [skip] {label} — {reason}")

    workspace = Path(tempfile.mkdtemp(prefix="atlas-mem0-selftest-"))
    store = ChromaStore(persist_dir=workspace / "chroma")
    manager: Mem0Manager | None = None
    try:
        manager = Mem0Manager(
            user_id="atlas_selftest_user",
            store=store,
            mem0_dir=workspace / "mem0",
            history_db=workspace / "mem0" / "history.db",
        )
        print(f"\nchroma: {store.persist_dir}")
        print(f"mem0 dir: {manager.mem0_dir}")

        check("construction is lazy (no mem0 yet)", manager._memory is None)

        built = manager.memory
        check("mem0 built", built is not None)
        check("telemetry is off", MEM0_TELEMETRY is False, f"MEM0_TELEMETRY={MEM0_TELEMETRY}")
        check("mem0 dir redirected out of ~/.mem0",
              str(manager.mem0_dir).startswith(str(workspace)), str(manager.mem0_dir))

        components = manager.health()
        check("llm points at the fast server",
              components["llm_base_url"].endswith(f":{FAST_MODEL_PORT}/v1"),
              components["llm_base_url"])
        check("negative results are suppressed (thinking off)",
              hasattr(manager.memory.llm, "_get_common_params"))
        check("embedder is the project's own",
              type(manager.memory.embedding_model).__name__ == "AtlasEmbedder")
        check("llm is the project's own",
              type(manager.memory.llm).__name__ == "AtlasMem0LLM")
        check("vector store is a chroma collection",
              type(manager.memory.vector_store.collection).__name__ == "Collection")
        check("mem0 shares the store's chroma client",
              manager.memory.vector_store.client is store.client)
        check("mem0 uses its own collection",
              manager.memory.vector_store.collection_name == MEM0_COLLECTION,
              manager.memory.vector_store.collection_name)
        check("collections coexist in one directory",
              store.collection("atlas_semantic") is not None)

        embedding_model = manager.memory.embedding_model
        vector = embedding_model.embed("the user likes tea", "add")
        check("the embedder produces vectors", isinstance(vector, list) and len(vector) > 0,
              f"{len(vector)}d")
        check("embedding width matches the shared provider",
              len(vector) == store.embedding_provider.dimension,
              f"{len(vector)} vs {store.embedding_provider.dimension}")
        check("embeddings are normalised",
              abs(sum(value * value for value in vector) ** 0.5 - 1.0) < 1e-3)
        check("assistant replies are excluded from extraction by default",
              manager.include_response is False)

        # --- formatting, which needs no LLM -------------------------------
        print("\n--- formatting ---")
        check("empty memory set produces no section", format_context([]) == "")
        sample = format_context([{"memory": "Prefers cortados."}, {"memory": ""}])
        check("section uses the agreed heading", sample.startswith(CONTEXT_HEADING))
        check("section is a bulleted list", "- Prefers cortados." in sample, repr(sample))
        long_set = [{"memory": "x" * 400} for _ in range(20)]
        check("section is length-capped", len(format_context(long_set)) <= 1200,
              f"{len(format_context(long_set))} chars")
        check("normalise_memory fills predictable keys",
              set(normalise_memory({"id": "a", "memory": "b", "metadata": {"k": 1}}))
              == {"id", "memory", "score", "created_at", "updated_at", "metadata"})

        # --- the parts that need the fast model ---------------------------
        print("\n--- extraction (needs the fast server) ---")
        if not _port_open(LLM_HOST, FAST_MODEL_PORT):
            skip("add_from_exchange / search / dedup", "fast llama.cpp server is not running")
        else:
            started = time.time()
            results = manager.add_from_exchange(
                "I've gone right off espresso. I'm on cortados now, usually two a day.",
                "Noted — cortados it is.",
                background=False,
            )
            print(f"       extraction took {time.time() - started:.1f}s")
            check("extraction returned changes", bool(results), f"{len(results or [])} change(s)")
            check("extraction produced memories", manager.count() > 0, f"{manager.count()} stored")

            everything = manager.get_all()
            check("get_all returns dicts with an id", all(item.get("id") for item in everything))
            check("get_all returns the memory text",
                  all(isinstance(item.get("memory"), str) for item in everything))

            section = manager.search("what coffee does the user like?", limit=5)
            check("search returns the section", section.startswith(CONTEXT_HEADING), repr(section[:80]))
            check("search found the coffee fact", "cortado" in section.lower(), repr(section[:200]))

            check("no query returns nothing", manager.search("") == "")
            check("unrelated query returns nothing definite",
                  isinstance(manager.search("quantum chromodynamics lecture notes"), str))

            # Deduplication: the same fact again must not add a second memory.
            before = manager.count()
            manager.add_from_exchange(
                "Just so you know, I drink cortados now — about two a day.",
                "Got it.",
                background=False,
            )
            after = manager.count()
            check("a repeated fact does not duplicate", after <= before + 1,
                  f"{before} -> {after} memories")

            # Temporal update: the old preference should be superseded, not kept
            # alongside a contradiction.
            manager.add_from_exchange(
                "Actually forget cortados, I've switched to straight black coffee.",
                "Updated.",
                background=False,
            )
            section = manager.search("what coffee does the user drink?", limit=5)
            check("the newer preference is the one recalled",
                  "black" in section.lower() or "cortado" not in section.lower(),
                  repr(section[:200]))
            check("memory count stayed bounded",
                  manager.count() <= before + 2, f"{manager.count()} memories")

            # --- deletion -------------------------------------------------
            print("\n--- deletion ---")
            victim = manager.get_all()[0]
            check("delete removes a memory", manager.delete(victim["id"]) is True)
            check("the memory is gone",
                  all(item["id"] != victim["id"] for item in manager.get_all()),
                  f"{manager.count()} left")
            check("deleting an unknown id reports failure",
                  manager.delete("00000000-dead-beef-0000-000000000000") is False)

        if _port_open(LLM_HOST, FAST_MODEL_PORT):
            # The flag exists because passing the assistant's reply was measured
            # to inject facts about Atlas's own wording. Confirm it actually
            # changes what mem0 is asked to read.
            print("\n--- include_response ---")
            before_both = manager.count()
            manager.add_from_exchange(
                "My sister is called Priya and she lives in Leeds.",
                "Good to know about Priya in Leeds.",
                include_response=True,
                background=False,
            )
            check("include_response=True sends the exchange and still extracts",
                  manager.count() >= before_both, f"{before_both} -> {manager.count()}")
            manager.delete_all()

        # --- background mode, no LLM needed to observe the plumbing ---------
        print("\n--- background dispatch ---")
        manager.add_from_exchange("Background plumbing check.", "Sure.")
        thread_names = [item.name for item in threading.enumerate()]
        dispatched = "atlas-mem0-add" in thread_names or manager._pending == set()
        check("add_from_exchange returns immediately and dispatches", dispatched)
        if _port_open(LLM_HOST, FAST_MODEL_PORT):
            check("wait_for_pending drains the worker", manager.wait_for_pending(timeout=90.0))

        print("\n--- health ---")
        report = manager.health()
        check("health reports a telemetry flag", report["telemetry"] is False)
        check("health reports the collection", report["collection"] == MEM0_COLLECTION)
        check("health never raises", isinstance(report, dict))
    except Exception as exc:  # pragma: no cover - the test reports its own failure
        logger.exception("self-test crashed")
        failures.append(f"crashed: {type(exc).__name__}: {exc}")
    finally:
        if manager is not None:
            try:
                manager.close()
            except Exception:
                pass
        shutil.rmtree(workspace, ignore_errors=True)

    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for label in failures:
            print(f"  - {label}")
        return 1
    print("\nall checks passed.")
    return 0


def _port_open(host: str, port: int) -> bool:
    """Whether something is listening — used only by the self-test."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(_self_test())
