"""
Atlas — quick search (mode 1).

The fast path for "current facts" questions. It hits a local SearXNG instance
for metadata only — title, snippet and URL — and never fetches the full page,
so the whole call stays inside the interactive latency budget (target: under
2 seconds; the per-request timeout is a 4-second ceiling for a wedged server).

Everything returns a formatted string ready to drop into an LLM prompt, so the
caller (see :class:`search.router.SearchRouter`) gets context, not objects.

If the SearXNG container is not running this degrades rather than fails: the
request is retried against DuckDuckGo through the ``ddgs`` library. If that is
missing too, the methods return an honest "no results" string.

Repeated queries are cheap. SearXNG has no result cache of its own
(``searx.cache`` only backs favicons, weather, currency tables and tracker
patterns), so an identical request is served from :class:`SearchCache` for
:data:`core.config.SEARCH_CACHE_TTL_SECONDS` (10 minutes by default) instead of
hitting the network again. Only successful SearXNG responses are cached.

The engine mix comes from :data:`core.config.SEARXNG_ENGINES` and is
configured on the other side in ``services/searxng/settings.yml``.

Three entry points:

* :meth:`QuickSearch.search`          — general lookup.
* :meth:`QuickSearch.search_news`     — news, last 24 hours; briefings.
* :meth:`QuickSearch.search_financial`— news plus a live quote when the query
  names a ticker.

Example::

    from search.quick_search import QuickSearch

    quick = QuickSearch()
    context = quick.search("who won the 2026 super bowl")

"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

from core.config import (
    QUICK_SEARCH_RESULTS,
    SEARCH_CACHE_TTL_SECONDS,
    SEARXNG_ENGINES,
    SEARXNG_URL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

#: Per-request ceiling. Matches `outgoing.request_timeout` in the SearXNG
#: instance's settings.yml, and deliberately above the 2-second target: a
#: healthy SearXNG answers in tens of milliseconds, so this only bounds a hung
#: server. (It is also what lets a slow engine like yahoo contribute: it was
#: measured timing out at the old 3.0s ceiling.)
REQUEST_TIMEOUT: float = 4.0

#: Engines SearXNG fans out to, from :data:`core.config.SEARXNG_ENGINES`.
#: Passing them explicitly keeps the result set predictable instead of
#: depending on the instance's default engine list — and SearXNG honours an
#: explicit list even for engines that are `disabled` in its settings, which is
#: why this list is the real lever for coverage.
DEFAULT_ENGINES: str = SEARXNG_ENGINES

#: How far back :meth:`QuickSearch.search_news` looks, in SearXNG's vocabulary.
NEWS_TIME_RANGE: str = "day"

#: Most result sets kept in the TTL cache at once.
CACHE_MAX_ENTRIES: int = 128


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One normalised search hit, common to the SearXNG and ddgs backends."""

    title: str
    url: str
    snippet: str


class SearchCache:
    """Thread-safe TTL + LRU cache for SearXNG result sets.

    SearXNG has no search-result cache to switch on, so this is where repeated
    queries on the same topic get cheap. It exists for the second identical
    query, not the first: deep research asks many closely-related questions and
    the same query often recurs across a session.

    Only *successful* SearXNG responses are stored. A ``None`` (SearXNG down)
    or a ddgs fallback result is never cached, so a transient outage cannot be
    pinned in place.
    """

    def __init__(self, ttl: float = SEARCH_CACHE_TTL_SECONDS, max_entries: int = CACHE_MAX_ENTRIES) -> None:
        self.ttl = float(ttl)
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple, tuple[float, list[SearchResult]]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.ttl > 0

    def get(self, key: tuple) -> list[SearchResult] | None:
        """Cached results for ``key``, or None when absent or expired."""
        if not self.enabled:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, results = entry
            if now - stored_at >= self.ttl:
                del self._entries[key]
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            # A fresh list: callers own and mutate what they get back.
            return list(results)

    def put(self, key: tuple, results: Iterable[SearchResult]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._entries[key] = (time.monotonic(), list(results))
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, int | float]:
        """Hit/miss counters and entry count, for diagnostics."""
        with self._lock:
            size = len(self._entries)
        return {"entries": size, "hits": self.hits, "misses": self.misses, "ttl": self.ttl}


#: ``$AAPL`` style cashtags, the least ambiguous ticker signal.
_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5})\b")

#: Bare uppercase tokens. Only 1-5 letters, which is the NASDAQ/NYSE norm.
_BARE_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")

#: Uppercase tokens that are ordinary English or finance jargon, not tickers.
#: Without this, "is the US Fed raising rates" would try to quote "US".
_TICKER_STOPWORDS: frozenset[str] = frozenset(
    {
        "A", "AI", "AM", "AN", "AND", "API", "AS", "AT", "BE", "BY", "CD", "CEO",
        "CFO", "CPI", "DD", "DIY", "DO", "EPS", "ETF", "EU", "EUR", "FED", "FOR",
        "FY", "GBP", "GDP", "GO", "HE", "I", "IF", "IN", "IPO", "IS", "IT", "LLC",
        "ME", "MY", "NO", "OF", "ON", "OR", "PE", "QOQ", "SO", "THE", "TO", "UK",
        "US", "USA", "USD", "VS", "WE", "YOY", "YOU",
    }
)


class QuickSearch:
    """Snippet-only web search with a DuckDuckGo fallback.

    The SearXNG URL comes from :data:`core.config.SEARXNG_URL` by default;
    passing another is the seam tests use to avoid a live network.
    """

    def __init__(
        self,
        searxng_url: str = SEARXNG_URL,
        *,
        timeout: float = REQUEST_TIMEOUT,
        engines: str = DEFAULT_ENGINES,
        cache: SearchCache | None = None,
        cache_ttl: float = SEARCH_CACHE_TTL_SECONDS,
    ) -> None:
        self.searxng_url = searxng_url.rstrip("/")
        self.timeout = timeout
        self.engines = engines
        #: Reused across calls, so it must be shareable between threads.
        self.cache = cache if cache is not None else SearchCache(cache_ttl)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def search(self, query: str, n_results: int = QUICK_SEARCH_RESULTS) -> str:
        """General lookup. Returns formatted snippets, never page content."""
        return self._format(
            query, self.search_results(query, n_results), label="Search results"
        )

    def search_results(
        self, query: str, n_results: int = QUICK_SEARCH_RESULTS
    ) -> list[SearchResult]:
        """Structured hits, for callers that need URLs rather than a string.

        Deep research collects candidate links from many queries and scores
        them before fetching anything, so it needs the raw results; the
        formatted-string API above cannot be parsed back reliably.
        """
        results = self._searxng_search(
            {"q": query, "format": "json", "engines": self.engines}
        )
        if results is None:
            results = self._ddg_search(query, n_results, news=False)
        return results[:n_results]

    def search_news(self, query: str, n_results: int = QUICK_SEARCH_RESULTS) -> str:
        """News from the last day. Used by briefings and current-events asks."""
        results = self._news_results(query, n_results)
        return self._format(query, results[:n_results], label="News results")

    def search_financial(
        self, query: str, n_results: int = QUICK_SEARCH_RESULTS
    ) -> str:
        """News plus a live quote, when the query names a ticker.

        A missing quote is not an error: the news section is still returned, so
        a question about a company always yields something usable.
        """
        sections: list[str] = []

        symbol = _detect_ticker(query)
        if symbol is not None:
            quote = self._quote(symbol)
            if quote is not None:
                sections.append(quote)

        news = self._news_results(query, n_results)
        sections.append(self._format(query, news[:n_results], label="News results"))
        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _searxng_search(self, params: dict[str, str]) -> list[SearchResult] | None:
        """Cache-aware SearXNG query, keyed on the exact request.

        Returns None when SearXNG is unreachable or answers with something that
        is not a result list, so the caller knows to fall back. An empty list is
        a real answer ("nothing found"), and is cached like any other —
        genuinely empty results are still results.
        """
        key = self._cache_key(params)
        if key is not None:
            cached = self.cache.get(key)
            if cached is not None:
                logger.debug("search cache hit for %r", params.get("q"))
                return cached

        results = self._fetch_searxng(params)
        if results is not None and key is not None:
            self.cache.put(key, results)
        return results

    @staticmethod
    def _cache_key(params: dict[str, str]) -> tuple | None:
        """Stable cache key for a request, or None if it should not be cached."""
        if not params.get("q"):
            return None
        return tuple(sorted((str(k), str(v)) for k, v in params.items()))

    def _fetch_searxng(self, params: dict[str, str]) -> list[SearchResult] | None:
        """The actual network call — the one seam tests replace."""
        try:
            import requests
        except ImportError:  # pragma: no cover - requests is installed
            logger.warning("requests is not installed — cannot query SearXNG")
            return None

        try:
            response = requests.get(
                f"{self.searxng_url}/search", params=params, timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning(
                "SearXNG at %s unavailable (%s) — falling back to DuckDuckGo",
                self.searxng_url,
                exc,
            )
            return None

        raw = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(raw, list):
            logger.warning("SearXNG returned an unexpected payload — falling back")
            return None
        return [self._from_searxng(item) for item in raw if isinstance(item, dict)]

    def clear_cache(self) -> None:
        """Drop every cached result set (e.g. after a settings change)."""
        self.cache.clear()

    @staticmethod
    def _from_searxng(item: dict) -> SearchResult:
        return SearchResult(
            title=str(item.get("title") or "").strip(),
            url=str(item.get("url") or "").strip(),
            snippet=str(item.get("content") or "").strip(),
        )

    def _ddg_search(
        self, query: str, n_results: int, *, news: bool
    ) -> list[SearchResult]:
        """Fallback backend for when the SearXNG container is not running."""
        try:
            from ddgs import DDGS
        except ImportError:
            logger.warning(
                "SearXNG is down and the 'ddgs' package is not installed — "
                "quick search has no backend (pip install ddgs)"
            )
            return []

        try:
            with DDGS() as ddgs:
                raw = (
                    ddgs.news(query, max_results=n_results)
                    if news
                    else ddgs.text(query, max_results=n_results)
                )
        except Exception as exc:
            logger.warning("DuckDuckGo fallback for %r failed: %s", query, exc)
            return []

        return [
            SearchResult(
                title=str(item.get("title") or "").strip(),
                # ddgs uses 'href' for web results and 'url' for news.
                url=str(item.get("href") or item.get("url") or "").strip(),
                snippet=str(item.get("body") or item.get("excerpt") or "").strip(),
            )
            for item in raw
            if isinstance(item, dict)
        ]

    def _news_results(self, query: str, n_results: int) -> list[SearchResult]:
        """News hits, shared by :meth:`search_news` and :meth:`search_financial`."""
        results = self._searxng_search(
            {
                "q": query,
                "format": "json",
                "engines": self.engines,
                "categories": "news",
                "time_range": NEWS_TIME_RANGE,
            }
        )
        if results is None:
            results = self._ddg_search(query, n_results, news=True)
        return results

    # ------------------------------------------------------------------
    # Financial data
    # ------------------------------------------------------------------

    def _quote(self, symbol: str) -> str | None:
        """One-line live quote from yfinance, or None when it is unavailable.

        ``fast_info`` is used rather than the full ``info`` because it is a
        single lightweight request — the heavyweight ``info`` can take seconds
        and would blow the quick-search budget.
        """
        try:
            import yfinance as yf
        except ImportError:  # pragma: no cover - yfinance is installed
            logger.warning("yfinance is not installed — no live quote for %s", symbol)
            return None

        try:
            info = yf.Ticker(symbol).fast_info
            price = _fast_info_value(info, "last_price")
            previous = _fast_info_value(info, "previous_close")
            currency = _fast_info_value(info, "currency")
            day_high = _fast_info_value(info, "day_high")
            day_low = _fast_info_value(info, "day_low")
        except Exception as exc:
            logger.warning("yfinance quote for %s failed: %s", symbol, exc)
            return None

        if not _is_number(price):
            logger.warning("yfinance returned no price for %s", symbol)
            return None

        parts = [f"last price {_fmt_number(price)}"]
        if currency:
            parts.append(str(currency))
        details: list[str] = []
        if _is_number(previous):
            details.append(f"previous close {_fmt_number(previous)}")
            if previous:
                change_pct = (price - previous) / previous * 100
                details.append(f"{change_pct:+.2f}%")
        if _is_number(day_high) and _is_number(day_low):
            details.append(
                f"day range {_fmt_number(day_low)}–{_fmt_number(day_high)}"
            )
        if details:
            parts.append(f"({', '.join(details)})")
        return f"{symbol.upper()} quote: {' '.join(parts)}."

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format(query: str, results: list[SearchResult], *, label: str) -> str:
        """Render hits as prompt-ready text, URL included for citation."""
        if not results:
            return f"No {label.lower()} found for {query}."

        lines = [f"{label} for {query}:"]
        for index, result in enumerate(results, start=1):
            title = result.title or "Untitled"
            snippet = result.snippet or "(no snippet)"
            line = f"{index}. {title}: {snippet}"
            if result.url:
                line += f" ({result.url})"
            lines.append(line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ticker detection and number helpers
# ---------------------------------------------------------------------------


def _detect_ticker(query: str) -> str | None:
    """Return a ticker symbol named in ``query``, or None.

    Cashtags win outright. Bare uppercase tokens are accepted only when they
    are not in :data:`_TICKER_STOPWORDS`, which trades a little recall for not
    querying yfinance on every capitalised word.
    """
    cashtag = _CASHTAG_RE.search(query)
    if cashtag:
        return cashtag.group(1).upper()

    for match in _BARE_TICKER_RE.finditer(query):
        token = match.group(1)
        if token not in _TICKER_STOPWORDS:
            return token
    return None


def _fast_info_value(info: object, key: str) -> object:
    """Read ``key`` from a yfinance ``FastInfo`` mapping or attribute."""
    try:
        return info[key]  # type: ignore[index]
    except Exception:
        return getattr(info, key, None)


def _is_number(value: object) -> bool:
    """True for a real, finite number — filters out None and pandas NaN."""
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _fmt_number(value: object) -> str:
    return f"{float(value):,.2f}"  # type: ignore[arg-type]
