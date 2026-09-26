"""
Atlas — search intent router.

Every potential search need passes through :class:`SearchRouter` *before* any
search happens, so the agent only pays for the web when a query actually needs
it. The classifier is deliberately cheap: one tightly constrained completion on
the always-resident fast model, capped at :data:`CLASSIFY_MAX_TOKENS` tokens,
with thinking disabled. A hit on the small LRU cache returns in microseconds.

The categories are mutually exclusive and map one-to-one onto a downstream
strategy:

* ``NONE``           — answerable from the model's own knowledge; do not search.
* ``QUICK``          — needs current facts; a simple lookup is enough.
* ``DEEP_RESEARCH``  — synthesis across multiple sources.
* ``LIVE_FINANCIAL`` — live market prices or data.
* ``ACADEMIC``       — scientific papers or citations.

Example::

    from search.router import SearchIntent, SearchRouter

    router = SearchRouter(llm_manager)
    if router.classify("what's the weather in London?") is not SearchIntent.NONE:
        ...  # run a quick search

"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # imported for typing only; llm_manager is injected
    from core.llm_manager import LLMServerManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

#: Hard cap on the classifier's reply. The answer is one category name, so this
#: leaves headroom for a stray token without ever letting the model ramble.
CLASSIFY_MAX_TOKENS: int = 20

#: Deterministic output: the same query must route to the same strategy.
CLASSIFY_TEMPERATURE: float = 0.0

#: How many recent classifications to remember. Bounded so a long session
#: cannot grow the cache without limit; 50 covers the follow-up questions that
#: actually repeat within a conversation.
CACHE_SIZE: int = 50

#: Safety net for a stuck request. A healthy fast-model classification takes a
#: few hundred milliseconds; this only exists so a wedged server cannot stall
#: the agent indefinitely before it falls back to ``NONE``.
REQUEST_TIMEOUT: float = 5.0

#: The classification prompt. Written as one tightly constrained sentence with
#: every category defined inline, an explicit "reply with only the category
#: name", and no room to negotiate — the goal is a single token of output.
CLASSIFY_PROMPT: str = (
    "Classify this query into exactly one category: "
    "NONE (answerable from knowledge), "
    "QUICK (needs current facts, simple lookup), "
    "DEEP_RESEARCH (needs synthesis across multiple sources, research task), "
    "LIVE_FINANCIAL (needs live market prices or data), "
    "ACADEMIC (needs scientific papers or citations). "
    "Query: {query}. "
    "Reply with only the category name."
)

#: Matches a category name anywhere in the reply. ``\b`` keeps
#: ``DEEP_RESEARCH`` from being confused with a bare ``DEEP`` the model added,
#: and the alternation is ordered longest-first for readability. Used with
#: :func:`re.search`, so a model that answers "Category: QUICK" still parses.
_CATEGORY_RE = re.compile(
    r"\b(DEEP_RESEARCH|LIVE_FINANCIAL|ACADEMIC|QUICK|NONE)\b",
    re.IGNORECASE,
)


class SearchIntent(Enum):
    """Which search strategy, if any, a query needs."""

    NONE = "NONE"
    QUICK = "QUICK"
    DEEP_RESEARCH = "DEEP_RESEARCH"
    LIVE_FINANCIAL = "LIVE_FINANCIAL"
    ACADEMIC = "ACADEMIC"


def _query_hash(query: str) -> str:
    """Stable cache key for a query.

    The query is lowercased and its whitespace collapsed first, so the same
    follow-up asked with different spacing or casing is a cache hit.
    """
    normalized = " ".join(query.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _parse_intent(text: str) -> SearchIntent | None:
    """Extract a category from a model reply, or None if there is none.

    Tolerates surrounding prose and punctuation (``"QUICK."``, ``"Category:
    QUICK"``) rather than requiring an exact string, because a small model does
    not always follow "reply with only the category name" to the letter. The
    first recognised category wins.
    """
    match = _CATEGORY_RE.search(text or "")
    if match is None:
        return None
    return SearchIntent(match.group(1).upper())


class SearchRouter:
    """Classifies a query into a :class:`SearchIntent` before any search runs.

    One fast-model completion per uncached query, with the result memoised in a
    small LRU cache. The router never starts a model: it uses the fast client as
    given, and if that model is not serving yet the query falls back to
    ``NONE`` rather than blocking the conversation.
    """

    def __init__(
        self,
        llm_manager: "LLMServerManager",
        cache_size: int = CACHE_SIZE,
    ) -> None:
        self.llm_manager = llm_manager
        self.cache_size = cache_size
        #: Ordered by recency of use; the leftmost key is evicted first.
        self._cache: OrderedDict[str, SearchIntent] = OrderedDict()
        #: Guards :attr:`_cache`. classify() can be called from several
        #: threads, and OrderedDict is not safe under concurrent mutation.
        self._lock = threading.Lock()
        #: Latency of the last uncached classification, for diagnostics.
        self.last_latency_s: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, query: str) -> SearchIntent:
        """Route one query, using the cache when possible.

        Returns ``NONE`` for an empty query, a cache hit, or any failure to
        classify — an unrecognised reply, a model error, or a timeout. ``NONE``
        is the safe default because it costs nothing: a missed search is
        recoverable, while searching on every utterance is not.
        """
        query = query.strip()
        if not query:
            return SearchIntent.NONE

        key = _query_hash(query)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        intent = self._classify_uncached(query)
        self._cache_put(key, intent)
        return intent

    def cache_clear(self) -> None:
        """Forget every memoised classification."""
        with self._lock:
            self._cache.clear()

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_uncached(self, query: str) -> SearchIntent:
        """Ask the fast model to classify ``query``, falling back to NONE."""
        request = {
            "model": "fast",
            "messages": [
                {"role": "user", "content": CLASSIFY_PROMPT.format(query=query)}
            ],
            "temperature": CLASSIFY_TEMPERATURE,
            "max_tokens": CLASSIFY_MAX_TOKENS,
            "timeout": REQUEST_TIMEOUT,
            # Thinking off: reasoning would multiply the latency for a task
            # whose answer is a single label. Same switch core/agent.py uses.
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }

        started = time.monotonic()
        try:
            response = self.llm_manager.get_fast_client().chat.completions.create(
                **request
            )
        except Exception:  # the SDK raises many shapes; a route must survive all
            logger.exception("search intent classification failed — routing to NONE")
            return SearchIntent.NONE
        finally:
            self.last_latency_s = time.monotonic() - started

        content = getattr(response.choices[0].message, "content", None) or ""
        intent = _parse_intent(content)
        if intent is None:
            logger.warning(
                "search intent reply %r is not a known category — routing to NONE",
                content.strip(),
            )
            return SearchIntent.NONE
        return intent

    # ------------------------------------------------------------------
    # LRU cache
    # ------------------------------------------------------------------

    def _cache_get(self, key: str) -> SearchIntent | None:
        """Look up ``key`` and mark it most-recently used."""
        with self._lock:
            intent = self._cache.get(key)
            if intent is not None:
                self._cache.move_to_end(key)
            return intent

    def _cache_put(self, key: str, intent: SearchIntent) -> None:
        """Store ``intent`` and evict the least-recently used entry over size."""
        with self._lock:
            self._cache[key] = intent
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
