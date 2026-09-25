"""
Atlas — vector storage.

ChromaDB wrapper that owns every embedding in the system. Anything that needs
semantic recall — durable facts about the user, the skills they have built,
indexed documents, conversation summaries — goes through here and comes back
ranked by cosine similarity.

Two collections, as the brief specifies
---------------------------------------

``atlas_semantic``
    Durable knowledge: facts, preferences, indexed documents and the skill
    library. Records are told apart by the ``type`` metadata field, not by
    living in separate collections, so one query can span all of them.
``atlas_conversations``
    Conversation summaries, one record per exchange worth keeping.

Embeddings are pluggable
------------------------

The brief asks for ``OllamaEmbeddings`` with ``nomic-embed-text``. Ollama is
not installed on this machine and ``langchain_ollama`` is not in the
environment, so taking that literally would produce a module that cannot run.
Instead the embedding backend is an explicit, configurable seam:

======================  ===========================================================
``sentence_transformers``  local transformer model, no daemon (the default)
``ollama``                 Ollama's ``POST /api/embed``
``llama_cpp``              the fast llama.cpp server's ``POST /v1/embeddings``
``openai``                 any OpenAI-compatible ``/v1/embeddings`` service
``chroma``                 Chroma's built-in ONNX all-MiniLM-L6-v2
======================  ===========================================================

``ATLAS_EMBED_PROVIDER=auto`` (the default) picks the first backend that is
actually usable, so the store works out of the box on this machine and still
routes to Ollama the moment it is installed. ``ATLAS_EMBED_MODEL`` names the
model once and is translated per backend — ``nomic-embed-text`` is Ollama's
tag for the very model that ``sentence_transformers`` loads from
``nomic-ai/nomic-embed-text-v1.5``.

Whichever backend wins, the model and its vector width are stamped into the
collection metadata. Vectors from different models are not comparable, so a
later change is detected and reported instead of quietly mixing two embedding
spaces in one index.

Two verified facts about nomic + transformers 5.x
-------------------------------------------------

Both were measured on this machine, not assumed:

* ``nomic-embed-text-v1.5`` ships remote code whose config records
  ``transformers_version: 5.3.0.dev0``, yet it still calls
  ``self.get_extended_attention_mask(...)`` — an inherited helper that
  transformers removed. Without it, loading dies with
  ``AttributeError: 'NomicBertModel' object has no attribute
  'get_extended_attention_mask'``. :func:`_install_extended_attention_mask`
  restores the classic implementation, including the ``dtype`` fallback: the
  remote code passes ``dtype=None``, and ``torch.finfo(None)`` raises
  ``TypeError``, which is how the first attempt at this shim failed.
* The model is asymmetric by design. Documents are embedded with a
  ``search_document: `` prefix and queries with ``search_query: ``; skipping
  the prefixes costs retrieval accuracy, so the prefixes are applied
  automatically. They can be turned off with ``apply_prefixes=False``.

Usage::

    from memory.chroma_store import get_chroma_store

    store = get_chroma_store()
    store.add(
        "The user's favourite coffee is a flat white from Bridge Street.",
        {"type": "preference", "source": "conversation"},
        collection="semantic",
    )

    for hit in store.search("what coffee do they like?", k=3):
        print(f"{hit.score:.2f}  {hit.text}")

    store.delete("conversation")           # by metadata source
    store.list_by_type("preference")       # by metadata type

Run the built-in check (uses a temporary directory, never the real index) with::

    python3 -m memory.chroma_store
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import re
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlparse

from core.config import (
    CHROMA_DIR,
    CONVERSATION_COLLECTION,
    EMBED_BATCH_SIZE,
    EMBED_DEVICE,
    EMBED_MODEL,
    EMBED_PROVIDER,
    EMBED_TIMEOUT,
    FAST_MODEL_PORT,
    LLM_HOST,
    MEMORY_RELEVANCE_THRESHOLD,
    OLLAMA_URL,
    SEMANTIC_COLLECTION,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ChromaStore",
    "EmbeddingProvider",
    "EmbeddingError",
    "SearchResult",
    "SEMANTIC_COLLECTION",
    "CONVERSATION_COLLECTION",
    "build_embedding_provider",
    "get_chroma_store",
    "reset_chroma_store",
    "resolve_collection_name",
]

# ---------------------------------------------------------------------------
# Metadata vocabulary
# ---------------------------------------------------------------------------

#: Every record carries a ``type``. These are the values the rest of Atlas
#: uses; arbitrary strings are accepted so future modules are not blocked.
META_TYPE = "type"
META_SOURCE = "source"
META_TIMESTAMP = "timestamp"
META_TAGS = "tags"
META_MODEL = "embed_model"

RECORD_TYPE_FACT = "fact"
RECORD_TYPE_PREFERENCE = "preference"
RECORD_TYPE_SKILL = "skill"
RECORD_TYPE_DOCUMENT = "document"
RECORD_TYPE_CONVERSATION = "conversation"
RECORD_TYPE_NOTE = "note"

DEFAULT_RECORD_TYPE = RECORD_TYPE_NOTE

#: Friendly names accepted in place of the collection constants.
_COLLECTION_ALIASES: dict[str, str] = {
    "semantic": SEMANTIC_COLLECTION,
    "memory": SEMANTIC_COLLECTION,
    "general": SEMANTIC_COLLECTION,
    "facts": SEMANTIC_COLLECTION,
    "skills": SEMANTIC_COLLECTION,
    "documents": SEMANTIC_COLLECTION,
    "conversations": CONVERSATION_COLLECTION,
    "conversation": CONVERSATION_COLLECTION,
    "summaries": CONVERSATION_COLLECTION,
    "summary": CONVERSATION_COLLECTION,
}

_VALID_COLLECTION_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")

# Chroma stores metadata as scalars or lists of scalars; anything else has to
# be flattened or it is rejected at write time.
_SCALAR_METADATA = (str, int, float, bool)

_PAGE_SIZE = 500


class EmbeddingError(RuntimeError):
    """Raised when no embedding backend can be built or one fails mid-call."""


# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    """Turns text into vectors.

    A provider knows its own model, its vector width and whether it is usable
    on this machine. Everything above this class is backend-agnostic.
    """

    #: Short identifier used in logs and collection metadata.
    provider_id: str = "base"

    #: Set when the backend cannot use ``ATLAS_EMBED_MODEL`` at all, so the
    #: model name reported in logs and collection metadata stays truthful.
    fixed_model: str | None = None

    def __init__(self, model: str) -> None:
        self.model = model
        self._dimension: int | None = None

    # -- capability -----------------------------------------------------

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """``(usable, reason)`` — `reason` explains a False."""
        return True, ""

    # -- interface ------------------------------------------------------

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages for storage."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query.

        Separate from :meth:`embed_documents` because asymmetric retrieval
        models prefix the two differently.
        """
        return self.embed_documents([text])[0]

    # -- introspection --------------------------------------------------

    @property
    def dimension(self) -> int | None:
        """Vector width, once known. ``None`` until the backend has been used."""
        return self._dimension

    def describe(self) -> str:
        width = f", {self._dimension}d" if self._dimension else ""
        return f"{self.provider_id}:{self.model}{width}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} model={self.model!r} dim={self._dimension}>"


# --- nomic / transformers 5.x shim -----------------------------------------


def _install_extended_attention_mask() -> bool:
    """Restore the attention-mask helper transformers 5.x removed.

    ``nomic-embed-text-v1.5`` keeps its modelling code in a remote repository
    and that code still calls ``self.get_extended_attention_mask``. The method
    used to live on every model via ``ModuleUtilsMixin``; in transformers 5.17
    it is gone, so loading the model raises ``AttributeError``.

    This is the original implementation, with one addition: the remote code
    passes ``dtype=None``, and ``torch.finfo(None)`` is a ``TypeError``. When
    no dtype is supplied it falls back to the module's own dtype, then to
    ``torch.float32``.

    Patching a shared base class is a blunt instrument, so it only happens when
    the attribute is genuinely missing, and it does nothing at all for models
    that never call it.
    """
    try:
        import torch
        from transformers import PreTrainedModel
    except ImportError:  # pragma: no cover - transformers is installed
        return False

    if hasattr(PreTrainedModel, "get_extended_attention_mask"):
        return False  # a future transformers puts it back; leave it alone

    def get_extended_attention_mask(
        self: Any,
        attention_mask: Any,
        input_shape: Any,
        device: Any = None,
        dtype: Any = None,
    ) -> Any:
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {tuple(attention_mask.shape)})"
            )

        if dtype is None:
            dtype = getattr(self, "dtype", None) or torch.float32
        extended = extended.to(dtype=dtype)
        # (1 - mask) * dtype.min keeps attended positions at ~0 and floors the
        # masked ones, which is what softmax needs.
        return (1.0 - extended) * torch.finfo(dtype).min

    PreTrainedModel.get_extended_attention_mask = get_extended_attention_mask
    logger.debug("installed get_extended_attention_mask shim for transformers 5.x")
    return True


#: Ollama's short tag -> the Hugging Face repo holding the same weights.
_LOCAL_MODEL_ALIASES: dict[str, str] = {
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
    "nomic-embed-text-v1.5": "nomic-ai/nomic-embed-text-v1.5",
    "nomic-embed-text:latest": "nomic-ai/nomic-embed-text-v1.5",
    "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
    "bge-m3": "BAAI/bge-m3",
    "snowflake-arctic-embed": "Snowflake/snowflake-arctic-embed-m",
}

#: Models that expect a ``search_document:`` / ``search_query:`` prefix.
_PREFIXED_MODELS = ("nomic-embed",)

_DOCUMENT_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


def _resolve_local_model(name: str) -> str:
    """Map an Ollama-style tag onto a Hugging Face repository id."""
    return _LOCAL_MODEL_ALIASES.get(name.strip().lower(), name.strip())


class SentenceTransformerProvider(EmbeddingProvider):
    """Local ``sentence-transformers`` inference. No daemon, no network."""

    provider_id = "sentence_transformers"

    def __init__(
        self,
        model: str = EMBED_MODEL,
        *,
        device: str = EMBED_DEVICE,
        batch_size: int = EMBED_BATCH_SIZE,
        trust_remote_code: bool = False,
        apply_prefixes: bool | None = None,
    ) -> None:
        super().__init__(_resolve_local_model(model))
        self.device = device
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        # nomic wants the task prefixes; a caller can still force them off.
        self.apply_prefixes = (
            any(tag in self.model.lower() for tag in _PREFIXED_MODELS)
            if apply_prefixes is None
            else apply_prefixes
        )
        self._model: Any | None = None
        self._load_lock = threading.Lock()

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if importlib.util.find_spec("sentence_transformers") is None:
            return False, "sentence_transformers is not installed"
        return True, ""

    # -- loading --------------------------------------------------------

    def _load(self) -> Any:
        """Load the model once, on first use.

        Deferred because importing torch costs seconds and most calls to this
        module (``delete``, ``list_by_type``) never need an embedding at all.
        """
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            if self.trust_remote_code or "nomic" in self.model.lower():
                # nomic-embed-text-v1.5 is exactly the model that needs it.
                _install_extended_attention_mask()
            from sentence_transformers import SentenceTransformer

            started = time.time()
            device = self._resolve_device()
            model = SentenceTransformer(
                self.model, device=device, trust_remote_code=self.trust_remote_code
            )
            # Renamed in sentence-transformers 6; accept either spelling.
            width = getattr(model, "get_embedding_dimension", None) or getattr(
                model, "get_sentence_embedding_dimension"
            )
            self._dimension = int(width())
            self._model = model
            logger.info(
                "loaded %s on %s in %.1fs (%dd)",
                self.model,
                device,
                time.time() - started,
                self._dimension,
            )
        return self._model

    def _resolve_device(self) -> str:
        if self.device and self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover - torch ships with the model
            return "cpu"

    # -- embedding ------------------------------------------------------

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [_as_text(t) for t in texts]
        if not cleaned:
            return []
        model = self._load()
        payload = (
            [_DOCUMENT_PREFIX + t for t in cleaned] if self.apply_prefixes else cleaned
        )
        vectors = model.encode(
            payload,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self._dimension = int(vectors.shape[1])
        return [[float(value) for value in row] for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        cleaned = _as_text(text)
        if not cleaned:
            raise EmbeddingError("cannot embed an empty query")
        model = self._load()
        payload = _QUERY_PREFIX + cleaned if self.apply_prefixes else cleaned
        vector = model.encode(
            [payload],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        self._dimension = int(len(vector))
        return [float(value) for value in vector]


class OllamaProvider(EmbeddingProvider):
    """Ollama's native embedding endpoint.

    Talks HTTP directly rather than through ``langchain_ollama``: the module
    is not installed, and one POST is not worth a dependency. Speaks the
    current ``/api/embed`` shape and falls back to the older one-token-at-a-time
    ``/api/embeddings`` if the running server predates it.
    """

    provider_id = "ollama"

    def __init__(
        self,
        model: str = EMBED_MODEL,
        *,
        url: str = OLLAMA_URL,
        timeout: float = EMBED_TIMEOUT,
        batch_size: int = EMBED_BATCH_SIZE,
        **_: Any,
    ) -> None:
        super().__init__(model)
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if importlib.util.find_spec("requests") is None:
            return False, "requests is not installed"
        parsed = urlparse(OLLAMA_URL)
        host, port = parsed.hostname or "127.0.0.1", parsed.port or 11434
        if not _port_open(host, port, timeout=0.25):
            return False, f"no Ollama server on {host}:{port}"
        return True, ""

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - requests is installed
            raise EmbeddingError(f"requests is not installed: {exc}") from exc
        try:
            response = requests.post(
                f"{self.url}{path}", json=payload, timeout=self.timeout
            )
        except Exception as exc:
            logger.warning("Ollama %s failed: %s", path, exc)
            return None
        if response.status_code == 404:
            return None  # endpoint does not exist; caller may try the older one
        if response.status_code >= 400:
            logger.warning(
                "Ollama %s returned %s: %s", path, response.status_code, response.text[:200]
            )
            return None
        try:
            return response.json()
        except ValueError:
            return None

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]] | None:
        payload = self._post("/api/embed", {"model": self.model, "input": list(texts)})
        if payload and isinstance(payload.get("embeddings"), list):
            return [[float(value) for value in row] for row in payload["embeddings"]]

        # Pre-0.1.30 servers: one prompt per request.
        vectors: list[list[float]] = []
        for text in texts:
            single = self._post("/api/embeddings", {"model": self.model, "prompt": text})
            if not single or not isinstance(single.get("embedding"), list):
                return None
            vectors.append([float(value) for value in single["embedding"]])
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [_as_text(t) for t in texts]
        if not cleaned:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(cleaned), self.batch_size):
            chunk = self._embed_batch(cleaned[start : start + self.batch_size])
            if chunk is None:
                raise EmbeddingError(
                    f"Ollama at {self.url} did not return embeddings for model "
                    f"{self.model!r}. Is it pulled? (ollama pull {self.model})"
                )
            vectors.extend(chunk)
        if vectors:
            self._dimension = len(vectors[0])
        return vectors


class OpenAICompatibleProvider(EmbeddingProvider):
    """Any ``POST /v1/embeddings`` service.

    Covers the llama.cpp server (when started with ``--embeddings``) and any
    hosted OpenAI-compatible API.
    """

    provider_id = "openai"

    def __init__(
        self,
        model: str = EMBED_MODEL,
        *,
        base_url: str = "",
        api_key: str | None = None,
        timeout: float = EMBED_TIMEOUT,
        batch_size: int = EMBED_BATCH_SIZE,
        **_: Any,
    ) -> None:
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout
        self.batch_size = batch_size

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if importlib.util.find_spec("requests") is None:
            return False, "requests is not installed"
        if not os.getenv("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY is not set"
        return True, ""

    def _endpoint(self) -> str:
        return f"{self.base_url}/v1/embeddings" if self.base_url else "/v1/embeddings"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [_as_text(t) for t in texts]
        if not cleaned:
            return []
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - requests is installed
            raise EmbeddingError(f"requests is not installed: {exc}") from exc

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        vectors: list[list[float]] = []
        for start in range(0, len(cleaned), self.batch_size):
            chunk = cleaned[start : start + self.batch_size]
            try:
                response = requests.post(
                    self._endpoint(),
                    json={"model": self.model, "input": chunk},
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                raise EmbeddingError(
                    f"embedding request to {self._endpoint()} failed: {exc}"
                ) from exc

            data = payload.get("data")
            if not isinstance(data, list):
                raise EmbeddingError(
                    f"unexpected embedding response from {self._endpoint()}: "
                    f"{str(payload)[:200]}"
                )
            # The API does not promise ordering, but it does send an index.
            for item in sorted(data, key=lambda entry: entry.get("index", 0)):
                vectors.append([float(value) for value in item["embedding"]])
        if vectors:
            self._dimension = len(vectors[0])
        return vectors


class LlamaCppProvider(OpenAICompatibleProvider):
    """The fast llama.cpp server's embedding endpoint.

    Requires the server to have been started with ``--embeddings``; without
    that flag the route does not exist at all, which is why this provider is
    never chosen automatically.
    """

    provider_id = "llama_cpp"

    def __init__(
        self,
        model: str = EMBED_MODEL,
        *,
        base_url: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            base_url=base_url or f"http://{LLM_HOST}:{FAST_MODEL_PORT}",
            **kwargs,
        )

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if importlib.util.find_spec("requests") is None:
            return False, "requests is not installed"
        if not _port_open(LLM_HOST, FAST_MODEL_PORT, timeout=0.25):
            return False, f"fast llama.cpp server is not listening on {FAST_MODEL_PORT}"
        # The server is up, so ask the route directly: without --embeddings it
        # does not exist, and there is no other way to tell from outside.
        try:
            import requests

            response = requests.post(
                f"http://{LLM_HOST}:{FAST_MODEL_PORT}/v1/embeddings",
                json={"model": "probe", "input": ["ping"]},
                timeout=2.0,
            )
        except Exception as exc:
            return False, f"could not probe the embedding route: {exc}"
        if response.status_code == 200:
            return True, ""
        return False, (
            "the llama.cpp server has no working /v1/embeddings route "
            f"(HTTP {response.status_code}) — start it with --embeddings"
        )


class ChromaDefaultProvider(EmbeddingProvider):
    """Chroma's own ONNX all-MiniLM-L6-v2. No extra dependency, 384 dimensions."""

    provider_id = "chroma"

    #: Bundled with Chroma and not swappable, unlike every other backend.
    fixed_model = "all-MiniLM-L6-v2"

    #: Fixed by the bundled ONNX model.
    _KNOWN_DIMENSION = 384

    def __init__(self, model: str = "all-MiniLM-L6-v2", **_: Any) -> None:
        super().__init__(self.fixed_model)
        self._dimension = self._KNOWN_DIMENSION
        self._function: Any | None = None

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if importlib.util.find_spec("chromadb") is None:
            return False, "chromadb is not installed"
        return True, ""

    def _load(self) -> Any:
        if self._function is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            self._function = DefaultEmbeddingFunction()
            logger.info("using Chroma's built-in ONNX %s", self.model)
        return self._function

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [_as_text(t) for t in texts]
        if not cleaned:
            return []
        vectors = self._load()(cleaned)
        result = [[float(value) for value in row] for row in vectors]
        if result:
            self._dimension = len(result[0])
        return result


#: Providers in the order ``auto`` tries them. Availability is checked with
#: :meth:`EmbeddingProvider.is_available`, so an unusable backend is skipped
#: rather than crashing.
_PROVIDER_ORDER: tuple[type[EmbeddingProvider], ...] = (
    OllamaProvider,
    SentenceTransformerProvider,
    ChromaDefaultProvider,
)

_PROVIDER_REGISTRY: dict[str, type[EmbeddingProvider]] = {
    "ollama": OllamaProvider,
    "sentence_transformers": SentenceTransformerProvider,
    "sentence-transformers": SentenceTransformerProvider,
    "sentencetransformer": SentenceTransformerProvider,
    "st": SentenceTransformerProvider,
    "local": SentenceTransformerProvider,
    "huggingface": SentenceTransformerProvider,
    "llama_cpp": LlamaCppProvider,
    "llama-cpp": LlamaCppProvider,
    "llamacpp": LlamaCppProvider,
    "llamaserver": LlamaCppProvider,
    "openai": OpenAICompatibleProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "chroma": ChromaDefaultProvider,
    "chroma_default": ChromaDefaultProvider,
    "default": ChromaDefaultProvider,
    "minilm": ChromaDefaultProvider,
}


def build_embedding_provider(
    provider: str | EmbeddingProvider | None = None,
    *,
    model: str | None = None,
    device: str | None = None,
    strict: bool = False,
    **options: Any,
) -> EmbeddingProvider:
    """Construct the embedding backend.

    ``provider=None`` uses ``ATLAS_EMBED_PROVIDER``; ``"auto"`` walks
    :data:`_PROVIDER_ORDER` and returns the first backend whose preconditions
    hold. Pass ``strict=True`` to make an unavailable choice an error instead
    of falling back — useful for tests and for a daemon that must not silently
    change embedding spaces.
    """
    if isinstance(provider, EmbeddingProvider):
        return provider

    requested = (provider or EMBED_PROVIDER or "auto").strip().lower()
    model = model or EMBED_MODEL
    resolved_device = device or EMBED_DEVICE

    if requested in ("", "auto"):
        reasons: list[str] = []
        for candidate in _PROVIDER_ORDER:
            usable, reason = candidate.is_available()
            if usable:
                logger.info("embedding provider auto-selected: %s", candidate.provider_id)
                return _construct(candidate, model, resolved_device, options)
            reasons.append(f"{candidate.provider_id}: {reason}")
        raise EmbeddingError(
            "no embedding backend is usable. Tried — " + "; ".join(reasons)
        )

    candidate = _PROVIDER_REGISTRY.get(requested)
    if candidate is None:
        known = sorted(set(_PROVIDER_REGISTRY) | {"auto"})
        raise EmbeddingError(
            f"unknown embedding provider {requested!r}. Known: " + ", ".join(known)
        )

    usable, reason = candidate.is_available()
    if not usable:
        message = f"embedding provider {requested!r} is unavailable: {reason}"
        if strict:
            raise EmbeddingError(message)
        logger.warning("%s — falling back to auto", message)
        return build_embedding_provider(None, model=model, device=resolved_device, **options)

    return _construct(candidate, model, resolved_device, options)


def _construct(
    candidate: type[EmbeddingProvider],
    model: str,
    device: str,
    options: Mapping[str, Any],
) -> EmbeddingProvider:
    """Instantiate a provider, passing ``device`` only where it means something."""
    kwargs = dict(options)
    if candidate.fixed_model:
        model = candidate.fixed_model
    if candidate is SentenceTransformerProvider:
        kwargs.setdefault("device", device)
        if "nomic" in model.lower() or "nomic" in str(
            _resolve_local_model(model)
        ).lower():
            # The weights live in a remote-code repository.
            kwargs.setdefault("trust_remote_code", True)
    try:
        return candidate(model, **kwargs)
    except Exception as exc:
        raise EmbeddingError(
            f"could not initialise provider {candidate.provider_id}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Chroma adapter
# ---------------------------------------------------------------------------


def _make_chroma_embedding_function(provider: EmbeddingProvider) -> Any:
    """Wrap a provider in Chroma's ``EmbeddingFunction`` interface.

    The store always hands Chroma pre-computed vectors, so this adapter is not
    on the hot path. It exists so that a raw ``collection`` handle obtained
    from :meth:`ChromaStore.collection` still behaves sensibly — ``query`` with
    ``query_texts``, ``add`` with ``documents`` — and so Chroma records which
    model a collection was built with.

    A per-store subclass is built at runtime because Chroma requires the
    embedded provider instance to be captured, and it expects ``name()``,
    ``get_config()`` and ``build_from_config()`` to be real implementations
    rather than the deprecated defaults.
    """
    from chromadb.api.types import EmbeddingFunction

    class _Adapter(EmbeddingFunction):  # type: ignore[misc]
        def __init__(self, inner: EmbeddingProvider) -> None:
            self._inner = inner

        def __call__(self, input: Any) -> Any:  # noqa: A002 - Chroma's signature
            return self._inner.embed_documents(list(input))

        @staticmethod
        def name() -> str:
            return "atlas"

        def get_config(self) -> dict[str, Any]:
            return {
                "provider": self._inner.provider_id,
                "model": self._inner.model,
            }

        @staticmethod
        def build_from_config(config: Mapping[str, Any]) -> Any:
            # Rebuilt from the live settings: the config is descriptive, not
            # the source of truth for which backend to use.
            inner = build_embedding_provider(
                config.get("provider"), model=config.get("model")
            )
            return _Adapter(inner)

        def default_space(self) -> str:
            return "cosine"

        def supported_spaces(self) -> list[str]:
            return ["cosine", "l2", "ip"]

    return _Adapter(provider)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SearchResult:
    """One ranked hit, with its metadata already unpacked."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: float = 0.0
    collection: str = ""

    @property
    def score(self) -> float:
        """Cosine similarity, ``0.0``–``1.0`` for related text.

        Chroma reports cosine *distance*; the collections are built with a
        cosine space and normalised vectors, so similarity is ``1 - distance``.
        """
        return 1.0 - self.distance

    @property
    def type(self) -> str:
        return str(self.metadata.get(META_TYPE, ""))

    @property
    def source(self) -> str:
        return str(self.metadata.get(META_SOURCE, ""))

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": dict(self.metadata),
            "score": round(self.score, 4),
            "collection": self.collection,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<SearchResult {self.score:.3f} {self.text[:48]!r}>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_text(value: Any) -> str:
    """Coerce anything into the string Chroma will store as the document."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    """Cheap TCP probe — used only to decide whether a service is worth asking."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _normalise_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Flatten metadata into the scalar forms Chroma accepts.

    ``None`` is dropped (Chroma rejects it), ``Path``/``datetime`` become
    strings, and anything nested is JSON-encoded so a caller can store a dict
    without a crash at write time.
    """
    out: dict[str, Any] = {}
    for raw_key, value in (metadata or {}).items():
        key = str(raw_key)
        if value is None:
            continue
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            out[key] = value
        elif isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (list, tuple, set, frozenset)):
            items = [v for v in value if isinstance(v, _SCALAR_METADATA)]
            if items:
                out[key] = items
        else:
            out[key] = json.dumps(value, default=str, ensure_ascii=False)
    return out


def _record_id(text: str, metadata: Mapping[str, Any]) -> str:
    """Deterministic id so re-indexing the same passage updates, not duplicates.

    Hashing ``type``/``source``/``text`` means re-indexing a document that has
    not changed costs an upsert instead of growing the index forever.
    """
    kind = str(metadata.get(META_TYPE, DEFAULT_RECORD_TYPE))
    source = str(metadata.get(META_SOURCE, ""))
    digest = hashlib.sha1(
        f"{kind}\x00{source}\x00{text}".encode("utf-8", "replace")
    ).hexdigest()
    prefix = re.sub(r"[^a-z0-9]+", "-", Path(source).name.lower()).strip("-")[:24]
    return f"{prefix}-{digest[:32]}" if prefix else digest[:40]


def resolve_collection_name(name: str | None) -> str:
    """Turn a friendly name into a real Chroma collection name."""
    if not name:
        return SEMANTIC_COLLECTION
    candidate = name.strip()
    if candidate in (SEMANTIC_COLLECTION, CONVERSATION_COLLECTION):
        return candidate
    alias = _COLLECTION_ALIASES.get(candidate.lower())
    if alias:
        return alias
    if _VALID_COLLECTION_NAME.match(candidate):
        return candidate  # an explicit collection of the caller's own
    raise ValueError(
        f"invalid collection name {name!r}: Chroma requires 3-512 characters of "
        "[a-zA-Z0-9._-] starting and ending alphanumerically"
    )


def _where(**clauses: Any) -> dict[str, Any]:
    """Drop ``None`` clauses so an unfiltered call stays valid."""
    return {key: value for key, value in clauses.items() if value is not None}


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class ChromaStore:
    """Persistent vector storage over ChromaDB.

    Cheap to construct: the Chroma client and collection handles are created
    eagerly, but the embedding model is loaded on first use, so ``delete`` and
    ``list_by_type`` never pay for torch.
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        *,
        provider: str | EmbeddingProvider | None = None,
        model: str | None = None,
        device: str | None = None,
        collections: Iterable[str] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        **provider_options: Any,
    ) -> None:
        """Open (creating if needed) the persistent index at ``persist_dir``.

        ``provider``/``model``/``device`` override the values in
        ``core/config.py``. ``embedding_provider`` accepts an already-built
        provider, which is what the tests and any caller doing its own
        bootstrapping will want.
        """
        self.persist_dir = Path(persist_dir or CHROMA_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._provider_choice = provider
        self._model_choice = model
        self._device_choice = device
        self._provider_options = provider_options
        self._provider: EmbeddingProvider | None = embedding_provider
        self._provider_lock = threading.RLock()

        # Embedding runs are serialised: a persistent Chroma client plus torch
        # already use their own threads, and interleaving only adds contention.
        self._write_lock = threading.RLock()

        self._collection_names = tuple(
            resolve_collection_name(name)
            for name in (collections or (SEMANTIC_COLLECTION, CONVERSATION_COLLECTION))
        )
        self._collection_cache: dict[str, Any] = {}

        import chromadb
        from chromadb.config import Settings

        # Telemetry off: this is a personal assistant on someone's own machine.
        self._settings = Settings(anonymized_telemetry=False, allow_reset=True)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir), settings=self._settings
        )
        logger.info("ChromaDB at %s (provider=%s)", self.persist_dir, self._provider_choice or EMBED_PROVIDER)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<ChromaStore dir={self.persist_dir} collections={list(self._collection_names)} "
            f"provider={self.embedding_provider.describe() if self._provider else 'lazy'}>"
        )

    @property
    def collections(self) -> tuple[str, ...]:
        return self._collection_names

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """The active backend, built on first access."""
        if self._provider is None:
            with self._provider_lock:
                if self._provider is None:
                    self._provider = build_embedding_provider(
                        self._provider_choice,
                        model=self._model_choice,
                        device=self._device_choice,
                        **self._provider_options,
                    )
                    logger.info("embeddings: %s", self._provider.describe())
        return self._provider

    @property
    def embedding_model(self) -> str:
        """Model name, resolved without forcing a load when possible."""
        if self._provider is not None:
            return self._provider.model
        if self._model_choice:
            return self._model_choice
        if str(EMBED_PROVIDER).strip().lower() in ("", "auto"):
            # Avoid building the provider just to answer a question about it.
            return _resolve_local_model(EMBED_MODEL)
        return EMBED_MODEL

    def warm(self) -> EmbeddingProvider:
        """Load the embedding backend up front.

        For the daemon: pays the one-off model load at start-up instead of
        stalling the first user turn by several seconds.
        """
        provider = self.embedding_provider
        provider.embed_documents(["warm up"])
        return provider

    def health(self) -> dict[str, Any]:
        """Snapshot for the API's status endpoint. Never raises."""
        report: dict[str, Any] = {
            "persist_dir": str(self.persist_dir),
            "provider": self._provider_choice or EMBED_PROVIDER,
            "model": EMBED_MODEL,
            "loaded": self._provider is not None,
            "collections": {},
        }
        for name in self._collection_names:
            try:
                report["collections"][name] = {
                    "count": self.collection(name).count(),
                    "metadata": dict(self.collection(name).metadata or {}),
                }
            except Exception as exc:  # pragma: no cover - defensive
                report["collections"][name] = {"error": f"{type(exc).__name__}: {exc}"}
        if self._provider is not None:
            report["provider"] = self._provider.provider_id
            report["model"] = self._provider.model
            report["dimension"] = self._provider.dimension
        return report

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    def collection(self, name: str | None = None) -> Any:
        """Open the named collection, creating it on first use.

        The cosine space is fixed at creation and cannot be changed later, so
        it is recorded in the collection metadata along with the embedding
        model that produced the vectors.
        """
        resolved = resolve_collection_name(name)
        cached = self._collection_cache.get(resolved)
        if cached is not None:
            return cached

        with self._write_lock:
            cached = self._collection_cache.get(resolved)
            if cached is not None:
                return cached

            # Build the provider first: it resolves the real model name (and may
            # fall back to a different backend), which is what gets recorded.
            embedding_function = _make_chroma_embedding_function(self.embedding_provider)
            metadata = {
                "hnsw:space": "cosine",
                "atlas_provider": self.embedding_provider.provider_id,
                "atlas_model": self.embedding_model,
            }

            try:
                handle = self.client.get_or_create_collection(
                    name=resolved,
                    embedding_function=embedding_function,
                    metadata=metadata,
                )
            except Exception:
                # A collection created by an older build (or by hand) can carry
                # metadata this call disagrees with. Opening it without an
                # embedding function still allows every explicit-vector
                # operation, which is all this store uses.
                logger.warning(
                    "could not open %s with the current settings — opening it as-is",
                    resolved,
                    exc_info=True,
                )
                handle = self.client.get_or_create_collection(resolved)

            self._check_model_drift(resolved, handle)
            self._collection_cache[resolved] = handle
            return handle

    def _check_model_drift(self, name: str, handle: Any) -> None:
        """Warn when a collection was built with a different embedding model.

        Vectors from two models are not comparable, so a silent change would
        degrade recall rather than fail loudly. Warning is the right level: the
        fix is to re-index, which is the caller's decision.
        """
        recorded = (handle.metadata or {}).get("atlas_model")
        if recorded and recorded != self.embedding_model:
            logger.warning(
                "collection %s was built with embedding model %r but %r is "
                "configured — old vectors are not comparable; re-index to fix",
                name,
                recorded,
                self.embedding_model,
            )

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        metadata: Mapping[str, Any] | None = None,
        collection: str | None = None,
        *,
        id: str | None = None,
        timestamp: float | None = None,
    ) -> str:
        """Store one passage and return its id.

        ``metadata`` is the only place record structure lives: put
        ``{"type": "preference", "source": "conversation"}`` there and the
        record can later be found by type or deleted by source. ``type``
        defaults to ``note`` and ``timestamp`` to now.

        The id is derived from type, source and text unless one is supplied, so
        adding the same passage twice upserts rather than duplicating it.
        """
        return self.add_many(
            [text],
            [metadata] if metadata is not None else None,
            collection,
            ids=[id] if id is not None else None,
            timestamps=[timestamp] if timestamp is not None else None,
        )[0]

    def add_many(
        self,
        texts: Sequence[str],
        metadatas: Sequence[Mapping[str, Any] | None] | None = None,
        collection: str | None = None,
        *,
        ids: Sequence[str | None] | None = None,
        timestamps: Sequence[float | None] | None = None,
    ) -> list[str]:
        """Store many passages in one batched write. Returns the ids in order."""
        documents = [_as_text(text) for text in texts]
        if not documents:
            return []
        if metadatas is not None and len(metadatas) != len(documents):
            raise ValueError("metadatas must line up with texts")
        if ids is not None and len(ids) != len(documents):
            raise ValueError("ids must line up with texts")

        now = time.time()
        prepared: list[dict[str, Any]] = []
        for index, text in enumerate(documents):
            raw = dict(metadatas[index]) if metadatas and metadatas[index] else {}
            raw.setdefault(META_TYPE, DEFAULT_RECORD_TYPE)
            raw.setdefault(META_TIMESTAMP, now)
            if timestamps is not None and timestamps[index] is not None:
                raw[META_TIMESTAMP] = timestamps[index]
            raw[META_MODEL] = self.embedding_model
            meta = _normalise_metadata(raw)

            supplied = ids[index] if ids is not None else None
            prepared.append(
                {
                    "id": supplied or _record_id(text, meta),
                    "text": text,
                    "metadata": meta,
                }
            )

        with self._write_lock:
            embeddings = self.embedding_provider.embed_documents(documents)
            handle = self.collection(collection)
            handle.upsert(  # type: ignore[attr-defined]
                ids=[record["id"] for record in prepared],
                documents=[record["text"] for record in prepared],
                embeddings=embeddings,
                metadatas=[record["metadata"] for record in prepared],
            )

        logger.debug(
            "stored %d record(s) in %s", len(prepared), resolve_collection_name(collection)
        )
        return [record["id"] for record in prepared]

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        collection: str | None = None,
        *,
        where: Mapping[str, Any] | None = None,
        type_filter: str | Sequence[str] | None = None,
        min_score: float | None = None,
        include_embeddings: bool = False,
    ) -> list[SearchResult]:
        """Return the ``k`` passages closest to ``query``, best first.

        ``collection`` accepts ``"semantic"`` or ``"conversations"`` (or the
        full names). Narrow further with ``type_filter`` — ``"skill"``,
        ``["fact", "preference"]`` — or a raw Chroma ``where`` clause.

        ``min_score`` drops weak matches; ``store.search_relevant`` applies the
        configured ``MEMORY_RELEVANCE_THRESHOLD`` for you.
        """
        cleaned = _as_text(query).strip()
        if not cleaned:
            return []
        if k <= 0:
            return []

        clauses: dict[str, Any] = dict(where or {})
        if type_filter:
            if isinstance(type_filter, str):
                clauses[META_TYPE] = type_filter
            else:
                clauses[META_TYPE] = {"$in": list(type_filter)}
        where_clause = _where(**clauses) or None

        with self._write_lock:
            vector = self.embedding_provider.embed_query(cleaned)
            handle = self.collection(collection)
            include = ["documents", "metadatas", "distances"]
            if include_embeddings:
                include.append("embeddings")
            try:
                raw = handle.query(
                    query_embeddings=[vector],
                    n_results=k,
                    where=where_clause,
                    include=include,
                )
            except Exception as exc:
                logger.warning("query against %s failed: %s", handle.name, exc)
                return []

        return self._to_results(raw, handle.name, min_score)

    def search_relevant(
        self,
        query: str,
        k: int = 5,
        collection: str | None = None,
        *,
        min_score: float | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Search, discarding anything below the configured relevance floor.

        Uses ``MEMORY_RELEVANCE_THRESHOLD`` unless ``min_score`` is given.
        """
        return self.search(
            query,
            k=k,
            collection=collection,
            min_score=MEMORY_RELEVANCE_THRESHOLD if min_score is None else min_score,
            **kwargs,
        )

    @staticmethod
    def _to_results(
        raw: Mapping[str, Any], collection_name: str, min_score: float | None
    ) -> list[SearchResult]:
        """Unpack Chroma's columnar result into ranked objects."""
        ids = (raw.get("ids") or [[]])[0] or []
        documents = (raw.get("documents") or [[]])[0] or []
        metadatas = (raw.get("metadatas") or [[]])[0] or []
        distances = (raw.get("distances") or [[]])[0] or []

        results: list[SearchResult] = []
        for position, record_id in enumerate(ids):
            result = SearchResult(
                id=str(record_id),
                text=documents[position] if position < len(documents) else "",
                metadata=dict(metadatas[position])
                if position < len(metadatas) and metadatas[position]
                else {},
                distance=float(distances[position]) if position < len(distances) else 0.0,
                collection=collection_name,
            )
            if min_score is not None and result.score < min_score:
                continue
            results.append(result)
        return results

    def list_by_type(
        self,
        type_filter: str | Sequence[str] | None = None,
        collection: str | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
        where: Mapping[str, Any] | None = None,
        with_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        """List stored records, optionally only those of a given type.

        Iterates every collection when ``collection`` is omitted, so
        ``list_by_type("skill")`` finds skills wherever they were written.
        ``type_filter=None`` means "everything". Results are dicts with ``id``,
        ``text``, ``metadata`` and ``collection``.
        """
        names = (
            (resolve_collection_name(collection),)
            if collection
            else self._collection_names
        )

        clauses: dict[str, Any] = dict(where or {})
        if type_filter:
            clauses[META_TYPE] = (
                type_filter if isinstance(type_filter, str) else {"$in": list(type_filter)}
            )
        where_clause = _where(**clauses) or None

        include = ["documents", "metadatas"]
        if with_embeddings:
            include.append("embeddings")

        records: list[dict[str, Any]] = []
        for name in names:
            handle = self.collection(name)
            remaining = None if limit is None else max(limit - len(records), 0)
            if remaining == 0:
                break
            for page in self._iter_pages(handle, where_clause, offset, remaining, include):
                records.extend(
                    {
                        "id": str(record_id),
                        "text": page["documents"][index],
                        "metadata": dict(page["metadatas"][index] or {}),
                        "collection": name,
                    }
                    for index, record_id in enumerate(page["ids"])
                )
        return records

    @staticmethod
    def _iter_pages(
        handle: Any,
        where_clause: Mapping[str, Any] | None,
        offset: int,
        limit: int | None,
        include: Sequence[str],
    ) -> Iterator[dict[str, Any]]:
        """Walk a collection in pages.

        Chroma caps how much one ``get`` returns, and an unlimited ``get`` on a
        large index is a memory spike, so results are streamed a page at a time.
        """
        fetched = 0
        position = offset
        while True:
            want = _PAGE_SIZE if limit is None else min(_PAGE_SIZE, limit - fetched)
            if want <= 0:
                return
            page = handle.get(
                where=dict(where_clause) if where_clause else None,
                limit=want,
                offset=position,
                include=list(include),
            )
            page_ids = page.get("ids") or []
            if not page_ids:
                return
            yield page
            fetched += len(page_ids)
            position += len(page_ids)
            if len(page_ids) < want:
                return
            if limit is not None and fetched >= limit:
                return

    def get(self, record_id: str, collection: str | None = None) -> dict[str, Any] | None:
        """Fetch one record by id, or ``None`` if it is not there."""
        names = (
            (resolve_collection_name(collection),)
            if collection
            else self._collection_names
        )
        for name in names:
            page = self.collection(name).get(ids=[record_id])
            if page.get("ids"):
                return {
                    "id": str(page["ids"][0]),
                    "text": (page.get("documents") or [""])[0],
                    "metadata": dict((page.get("metadatas") or [{}])[0] or {}),
                    "collection": name,
                }
        return None

    # ------------------------------------------------------------------
    # Deleting
    # ------------------------------------------------------------------

    def delete(
        self,
        source_path: str | Path,
        collection: str | None = None,
        *,
        where: Mapping[str, Any] | None = None,
    ) -> int:
        """Remove every record whose ``source`` metadata is ``source_path``.

        Deletes across both collections unless one is named. Returns how many
        records went. This is the counterpart to indexing a document: drop the
        file, call ``delete(path)``, and its chunks are gone.
        """
        target = str(source_path)
        if not target:
            raise ValueError("delete() needs a source path")
        clause = _where(source=target, **(dict(where) if where else {}))
        return self.delete_where(clause, collection)

    def delete_where(
        self, where: Mapping[str, Any], collection: str | None = None
    ) -> int:
        """Remove every record matching a metadata clause. Returns the count."""
        if not where:
            raise ValueError("delete_where() needs at least one clause")
        return self._delete([dict(where)], collection)

    def delete_by_id(self, record_id: str, collection: str | None = None) -> int:
        """Remove records by id, across collections unless one is named."""
        if not record_id:
            raise ValueError("delete_by_id() needs an id")
        removed = 0
        for name in self._target_names(collection):
            before = self.collection(name).count()
            try:
                self.collection(name).delete(ids=[record_id])
            except Exception:
                logger.warning("delete of %s in %s failed", record_id, name, exc_info=True)
                continue
            removed += max(before - self.collection(name).count(), 0)
        if removed:
            logger.debug("deleted id %s (%d record(s))", record_id, removed)
        return removed

    def _target_names(self, collection: str | None) -> tuple[str, ...]:
        if collection:
            return (resolve_collection_name(collection),)
        return self._collection_names

    def _delete(
        self, clauses: Sequence[Mapping[str, Any]], collection: str | None
    ) -> int:
        """Delete by id, per collection, because Chroma's ``where`` handles one
        clause per call and a ``$and`` is easy to get wrong on an empty index.
        Counting through ``get`` first keeps the return value honest — Chroma's
        ``delete`` reports nothing about how much it removed.
        """
        removed = 0
        for name in self._target_names(collection):
            handle = self.collection(name)
            for clause in clauses:
                try:
                    page = handle.get(where=dict(clause))
                except Exception:
                    logger.warning(
                        "lookup before delete in %s failed", name, exc_info=True
                    )
                    continue
                ids = [str(value) for value in (page.get("ids") or [])]
                if not ids:
                    continue
                try:
                    handle.delete(ids=ids)
                except Exception:
                    logger.warning("delete in %s failed", name, exc_info=True)
                    continue
                removed += len(ids)
        if removed:
            logger.info("deleted %d record(s)", removed)
        return removed

    def clear(self, collection: str | None = None) -> int:
        """Empty one collection, or both. Returns how many records went."""
        removed = 0
        for name in self._target_names(collection):
            handle = self.collection(name)
            before = handle.count()
            if not before:
                continue
            try:
                self.client.delete_collection(name)
            except Exception:
                logger.warning("could not drop %s — deleting by id instead", name, exc_info=True)
                ids = [
                    str(value)
                    for page in self._iter_pages(handle, None, 0, None, [])
                    for value in page["ids"]
                ]
                if ids:
                    handle.delete(ids=ids)
                else:
                    continue
            self._collection_cache.pop(name, None)
            removed += before
        if removed:
            logger.info("cleared %d record(s)", removed)
        return removed

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def count(self, collection: str | None = None) -> int:
        """Number of records, summed across collections when none is named."""
        return sum(self.collection(name).count() for name in self._target_names(collection))

    def stats(self) -> dict[str, Any]:
        """Counts per collection and a breakdown by record type."""
        report: dict[str, Any] = {"total": 0, "collections": {}, "types": {}}
        for name in self._collection_names:
            size = self.collection(name).count()
            report["collections"][name] = size
            report["total"] += size
            for record in self.list_by_type(None, name):
                kind = str(record["metadata"].get(META_TYPE, "unknown"))
                report["types"][kind] = report["types"].get(kind, 0) + 1
        return report

    def sources(self, collection: str | None = None) -> dict[str, int]:
        """Map ``source`` -> record count, for spotting what has been indexed."""
        tally: dict[str, int] = {}
        for record in self.list_by_type(None, collection):
            source = str(record["metadata"].get(META_SOURCE, "")) or "(none)"
            tally[source] = tally.get(source, 0) + 1
        return tally

    def close(self) -> None:
        """Release handles. The data is already on disk; this just drops state."""
        self._collection_cache.clear()
        self.client = None  # type: ignore[assignment]
        self._provider = None


# ---------------------------------------------------------------------------
# Shared instance
# ---------------------------------------------------------------------------

_store: ChromaStore | None = None
_store_lock = threading.Lock()


def get_chroma_store(**kwargs: Any) -> ChromaStore:
    """Process-wide store, built once.

    The daemon, the agent and the skill registry should share one instance: a
    second Chroma client on the same directory works, but wastes the loaded
    embedding model. Tests should call :func:`reset_chroma_store` — or build
    their own :class:`ChromaStore` on a temporary directory.
    """
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ChromaStore(**kwargs)
    return _store


def reset_chroma_store() -> None:
    """Drop the shared instance so the next call rebuilds it."""
    global _store
    with _store_lock:
        if _store is not None:
            _store.close()
        _store = None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> int:
    """Exercise every public method against a throwaway index.

    Deliberately uses a temporary directory: the real ``chroma_db`` is user
    data and must not be touched by a smoke test.
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

    workdir = Path(tempfile.mkdtemp(prefix="atlas-chroma-selftest-"))
    try:
        started = time.time()
        store = ChromaStore(persist_dir=workdir)
        provider = store.warm()
        print(
            f"\nprovider: {provider.describe()} (loaded in {time.time() - started:.1f}s)"
        )
        check("two collections exist", set(store.collections) == {
            SEMANTIC_COLLECTION,
            CONVERSATION_COLLECTION,
        })

        documents = [
            ("My favourite coffee is a flat white from the shop on Bridge Street.", {
                META_TYPE: RECORD_TYPE_PREFERENCE, META_SOURCE: "conversation", "user_id": "starpanda",
            }),
            ("The user is building a local AI assistant called Atlas.", {
                META_TYPE: RECORD_TYPE_FACT, META_SOURCE: "conversation", "user_id": "starpanda",
            }),
            ("The user's dog is called Biscuit and is a border collie.", {
                META_TYPE: RECORD_TYPE_FACT, META_SOURCE: "conversation", "user_id": "starpanda",
            }),
            ("Trading journal for Q3: mostly index funds, one covered call.", {
                META_TYPE: RECORD_TYPE_DOCUMENT, META_SOURCE: "/home/user/notes/q3.txt",
            }),
        ]
        ids = store.add_many(
            [text for text, _ in documents], [meta for _, meta in documents]
        )
        check("add_many stored every passage", len(ids) == len(documents), f"{len(ids)} ids")
        check("count reflects the writes", store.count("semantic") == len(documents))

        again = store.add(documents[0][0], documents[0][1], collection="semantic")
        check("re-adding the same passage upserts", again == ids[0] and store.count("semantic") == len(documents))

        hits = store.search("what coffee does the user like?", k=3)
        check("search returns ranked hits", bool(hits), f"{len(hits)} hits")
        if hits:
            check("best hit is the coffee preference", "flat white" in hits[0].text,
                  f"score {hits[0].score:.3f}")
            check("results arrive best-first", all(
                hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1)
            ))
        check("collection defaults to atlas_semantic", bool(hits) and hits[0].collection == SEMANTIC_COLLECTION)

        conversations = [
            ("Searching for a new espresso machine; budget around two hundred pounds.", {
                META_TYPE: RECORD_TYPE_CONVERSATION, META_SOURCE: "session-2026-09-25",
            }),
        ]
        store.add_many([t for t, _ in conversations], [m for _, m in conversations], "conversations")
        check("conversations land in the other collection", store.count("conversations") == 1)
        convo_hits = store.search("espresso machine", collection="conversations")
        check("search can target one collection", bool(convo_hits) and convo_hits[0].collection == CONVERSATION_COLLECTION)

        only_facts = store.list_by_type(RECORD_TYPE_FACT)
        check("list_by_type filters", len(only_facts) == 2, f"{len(only_facts)} facts")
        check("list_by_type spans collections by default", store.list_by_type(None) and len(store.list_by_type(None)) == store.count())
        two_types = store.list_by_type([RECORD_TYPE_FACT, RECORD_TYPE_PREFERENCE])
        check("list_by_type accepts a list", len(two_types) == 3, f"{len(two_types)} records")
        limited = store.list_by_type(None, limit=2)
        check("list_by_type honours limit", len(limited) == 2)

        typed = store.search("what pets does the user have?", k=2, type_filter=RECORD_TYPE_FACT)
        check("search accepts a type filter", bool(typed) and all(h.type == RECORD_TYPE_FACT for h in typed))

        unfiltered = store.search("what coffee does the user like?", k=10)
        filtered = store.search_relevant("what coffee does the user like?", k=10)
        check("search_relevant applies the threshold",
              all(h.score >= MEMORY_RELEVANCE_THRESHOLD for h in filtered)
              and len(filtered) <= len(unfiltered),
              f"{len(filtered)}/{len(unfiltered)} above {MEMORY_RELEVANCE_THRESHOLD}")

        score_hits = store.search("dog", k=2, min_score=0.9)
        check("min_score can exclude everything", score_hits == [] or score_hits[0].score >= 0.9)

        removed = store.delete("/home/user/notes/q3.txt")
        check("delete removes by source path", removed == 1, f"removed {removed}")
        check("count drops after delete", store.count() == len(documents), f"{store.count()} left")

        check("delete on an unknown source is a no-op", store.delete("/nope/missing.txt") == 0)

        by_id = store.delete_by_id(ids[2])
        check("delete_by_id works", by_id == 1, f"removed {by_id}")
        check("the record is gone", store.get(ids[2]) is None)
        check("get still finds the others", store.get(ids[0]) is not None)

        broken = store.add("Some text with awkward metadata.", {
            META_TYPE: RECORD_TYPE_NOTE, "nested": {"a": 1}, "empty": None,
            "tags": ["a", "b", None], "when": Path("/tmp/x"),
        })
        stored = store.get(broken)
        check("awkward metadata is normalised", stored is not None and stored["metadata"].get("nested") == '{"a": 1}')
        check("None metadata values are dropped", stored is not None and "empty" not in stored["metadata"])
        check("Path metadata becomes a string", stored is not None and stored["metadata"].get("when") == "/tmp/x")
        store.delete_by_id(broken)

        reopened = ChromaStore(persist_dir=workdir)
        check("data survives reopening the client", reopened.count() == store.count(),
              f"{reopened.count()} records")
        check("a reopened store can still search", bool(reopened.search("coffee", k=2)))

        short = ChromaStore(persist_dir=workdir, collections=(SEMANTIC_COLLECTION,))
        check("a second client on the same dir works", short.count() == reopened.collection(SEMANTIC_COLLECTION).count())

        before = reopened.count()
        cleared = reopened.clear()
        check("clear empties the store", cleared == before and reopened.count() == 0)
    except Exception as exc:  # pragma: no cover - the test reports its own failure
        logger.exception("self-test crashed")
        failures.append(f"crashed: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for label in failures:
            print(f"  - {label}")
        return 1
    print("\nall checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
