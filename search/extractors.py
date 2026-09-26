"""
Atlas — full page content extraction.

Quick search (see :mod:`search.quick_search`) deliberately returns snippets
only. This module is the opposite: it pulls the *whole* readable body of a page
for deep research, where synthesis across full sources is the point.

The primary extractor is trafilatura, which strips boilerplate, navigation and
ads far better than a hand-rolled parser. When it fails — no network, a JS-only
page, a blocked user agent — :meth:`PageExtractor.extract` falls back to a raw
``httpx`` fetch with BeautifulSoup tag stripping rather than returning nothing.

Nothing in this module raises on a bad URL: every failure is logged and
returns an empty string, so a research loop over many sources never dies
because one link was dead.

Example::

    from search.extractors import PageExtractor

    extractor = PageExtractor()
    text = extractor.extract("https://example.com/article")

    texts = asyncio.run(extractor.extract_batch(urls))

"""

from __future__ import annotations

import asyncio
import io
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

#: Ceiling for a single page fetch. Deep research is patient, but a hung host
#: must not stall the whole batch — applied to both the trafilatura and the
#: httpx path.
REQUEST_TIMEOUT: float = 8.0

#: How many pages :meth:`PageExtractor.extract_batch` fetches at once. Kept
#: low because each fetch is blocking network I/O and the sites involved are
#: often the same handful of hosts.
DEFAULT_MAX_CONCURRENT: int = 3

#: Sent on the httpx fallback path. Some sites serve a stripped-down page (or
#: refuse outright) to a client that does not look like a browser.
USER_AGENT: str = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class PageExtractor:
    """Fetches and cleans full page text.

    ``timeout`` bounds each request; the default 8 seconds matches the rest of
    Atlas's network calls. All dependencies (trafilatura, httpx, bs4, pypdf)
    are imported lazily so a missing optional package degrades this module
    rather than breaking the import.
    """

    def __init__(
        self,
        *,
        timeout: float = REQUEST_TIMEOUT,
        user_agent: str = USER_AGENT,
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent
        #: Cached trafilatura config carrying our timeout; built on first use.
        self._trafilatura_config: object | None = None

    # ------------------------------------------------------------------
    # Single page
    # ------------------------------------------------------------------

    def extract(self, url: str) -> str:
        """Return the readable text of ``url``, or "" on any failure.

        trafilatura is tried first. If it yields nothing — its download failed,
        or the page cleaned down to empty — the raw httpx + BeautifulSoup
        fallback is tried before giving up.
        """
        if not url or not isinstance(url, str):
            return ""

        text = self._trafilatura_extract(url)
        if text:
            return text
        return self._beautifulsoup_extract(url)

    def extract_batch(
        self, urls: list[str], max_concurrent: int = DEFAULT_MAX_CONCURRENT
    ) -> list[str]:
        """Extract many URLs concurrently, preserving input order.

        Each call to :meth:`extract` is blocking, so it runs in a worker thread
        with a semaphore capping concurrency at ``max_concurrent``. Errors are
        already swallowed by :meth:`extract`, so the returned list always has
        one entry per input URL.
        """
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")

        async def run() -> list[str]:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def one(url: str) -> str:
                async with semaphore:
                    return await asyncio.to_thread(self.extract, url)

            return await asyncio.gather(*(one(url) for url in urls))

        return asyncio.run(run())

    # ------------------------------------------------------------------
    # Academic PDFs
    # ------------------------------------------------------------------

    def extract_academic_pdf(self, url: str) -> str:
        """Extract text from a PDF URL, falling back to ordinary extraction.

        A URL that does not look like a PDF is handed straight to
        :meth:`extract`. For a real PDF the bytes are downloaded once and fed
        to trafilatura (newer builds can read them) and then to pypdf, which is
        the reliable path on this project's pinned trafilatura. Returns "" if
        neither produces text, or if pypdf is not installed.
        """
        if not url or not isinstance(url, str):
            return ""
        if not _is_pdf_url(url):
            return self.extract(url)

        data = self._download_bytes(url)
        if not data:
            return ""

        text = self._trafilatura_pdf(data)
        if text:
            return text
        return self._pypdf_extract(data)

    # ------------------------------------------------------------------
    # trafilatura path
    # ------------------------------------------------------------------

    def _trafilatura_extract(self, url: str) -> str:
        try:
            import trafilatura
        except ImportError:  # pragma: no cover - dependency is installed
            logger.warning("trafilatura is not installed — using the httpx fallback")
            return ""

        try:
            downloaded = trafilatura.fetch_url(url, config=self._timeout_config())
        except Exception as exc:
            logger.warning("trafilatura download failed for %s: %s", url, exc)
            return ""
        if not downloaded:
            return ""

        return self._trafilatura_extract_text(downloaded)

    def _trafilatura_pdf(self, data: bytes) -> str:
        """Best-effort trafilatura read of raw PDF bytes.

        The pinned trafilatura has no PDF parser, so this is expected to return
        "" and hand off to pypdf; it only pays off on a build that adds support.
        """
        try:
            import trafilatura
        except ImportError:  # pragma: no cover - dependency is installed
            return ""
        return self._trafilatura_extract_text(data)

    @staticmethod
    def _trafilatura_extract_text(content: object) -> str:
        """Run trafilatura's cleaner over already-fetched content."""
        try:
            import trafilatura

            text = trafilatura.extract(
                content,
                include_comments=False,
                include_tables=True,
                output_format="markdown",
            )
        except Exception as exc:
            logger.warning("trafilatura extraction failed: %s", exc)
            return ""
        return (text or "").strip()

    def _timeout_config(self) -> object | None:
        """A trafilatura config carrying :attr:`timeout`, or None.

        trafilatura's ``fetch_url`` has no timeout argument; the value lives in
        the ``DEFAULT`` section, so the shared config is deep-copied and
        adjusted once rather than mutated in place.
        """
        if self._trafilatura_config is None:
            try:
                import copy

                from trafilatura.settings import DEFAULT_CONFIG

                config = copy.deepcopy(DEFAULT_CONFIG)
                config.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(int(self.timeout)))
                self._trafilatura_config = config
            except Exception as exc:
                logger.warning("could not build trafilatura config: %s", exc)
                self._trafilatura_config = False  # type: ignore[assignment]
        return None if self._trafilatura_config is False else self._trafilatura_config

    # ------------------------------------------------------------------
    # httpx + BeautifulSoup fallback
    # ------------------------------------------------------------------

    def _beautifulsoup_extract(self, url: str) -> str:
        try:
            import httpx
            from bs4 import BeautifulSoup
        except ImportError:  # pragma: no cover - dependencies are installed
            logger.warning("httpx/bs4 is not installed — cannot extract %s", url)
            return ""

        try:
            response = httpx.get(
                url,
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            )
            response.raise_for_status()
            html = response.text
        except Exception as exc:
            logger.warning("httpx fetch failed for %s: %s", url, exc)
            return ""

        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        except Exception as exc:
            logger.warning("BeautifulSoup parse failed for %s: %s", url, exc)
            return ""

    # ------------------------------------------------------------------
    # PDF helpers
    # ------------------------------------------------------------------

    def _download_bytes(self, url: str) -> bytes:
        try:
            import httpx
        except ImportError:  # pragma: no cover - dependency is installed
            logger.warning("httpx is not installed — cannot download %s", url)
            return b""

        try:
            response = httpx.get(
                url,
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            )
            response.raise_for_status()
            return response.content
        except Exception as exc:
            logger.warning("PDF download failed for %s: %s", url, exc)
            return b""

    @staticmethod
    def _pypdf_extract(data: bytes) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.warning(
                "pypdf is not installed — cannot extract text from the PDF "
                "(pip install pypdf)"
            )
            return ""

        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [(page.extract_text() or "").strip() for page in reader.pages]
        except Exception as exc:
            logger.warning("pypdf extraction failed: %s", exc)
            return ""
        return "\n\n".join(page for page in pages if page).strip()


def _is_pdf_url(url: str) -> bool:
    """True when the URL's path ends in ``.pdf`` (query string ignored)."""
    try:
        return urlparse(url).path.lower().endswith(".pdf")
    except ValueError:
        return False
