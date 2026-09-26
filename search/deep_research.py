"""
Atlas — deep research (mode 2).

Quick search answers "what is true right now". This module answers "what is
actually going on", which is a different shape of problem: the question is too
broad for one lookup, so it is decomposed, searched across many sources, and
synthesised over several iterations.

The loop, per :meth:`DeepResearcher.research`:

1. **Plan** — the deep model splits the question into 4-6 sub-questions.
2. **Sweep** — each sub-question goes through quick search; unique URLs are
   collected across all of them.
3. **Extract selectively** — candidate URLs are scored against the question by
   embedding cosine similarity and only the best handful are fetched in full.
4. **Synthesise** — the deep model reads the evidence and says what is
   answered, what is unclear, and which follow-up searches would close the gap.
5. **Iterate** — steps 2-4 repeat for the follow-up queries, up to ``depth``
   times.
6. **Report** — the deep model writes a structured report; it is returned as a
   :class:`ResearchReport` and persisted to the vault.

Academic questions take a different on-ramp: Semantic Scholar is queried before
(or alongside) the web sweep so papers, abstracts and open-access PDFs enter
the evidence instead of blog posts.

Because a research run can take minutes, :meth:`DeepResearcher.research_async`
is the conversational entry point: it runs the loop in a background thread and
calls back when the report is ready, letting Atlas keep talking.

Example::

    from search.deep_research import DeepResearcher

    researcher = DeepResearcher(llm_manager, quick, extractor, vault)
    report = await researcher.research("how does RAG evaluation work?")
    print(report.summary)

"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Sequence

from core.config import DEEP_RESEARCH_MAX_ITERATIONS, DEEP_RESEARCH_MAX_SOURCES

if TYPE_CHECKING:  # collaborators are injected; imports here are typing-only
    from core.llm_manager import LLMServerManager
    from memory.vault_manager import VaultManager
    from search.extractors import PageExtractor
    from search.quick_search import QuickSearch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

#: Default number of search/extract iterations. Config caps it independently.
DEFAULT_DEPTH: int = 3

#: The prompt asks for 4-6; only the ceiling is enforced in code.
MAX_SUB_QUESTIONS: int = 6

#: Hits requested per sub-question, and the most URLs fetched in full per
#: iteration. Extraction is the expensive step, so it stays deliberately tight.
RESULTS_PER_SUB_QUESTION: int = 4
MAX_EXTRACT_URLS: int = 8

#: Follow-up queries generated from a synthesis pass.
MAX_FOLLOWUP_QUERIES: int = 4

#: Deep-model completion budgets.
PLAN_MAX_TOKENS: int = 512
SYNTHESIS_MAX_TOKENS: int = 768
REPORT_MAX_TOKENS: int = 3072
DEEP_TEMPERATURE: float = 0.3

#: Evidence is the dominant input to the synthesis and report prompts, so it is
#: capped: a few full-length pages will not fit the deep model's context.
PER_SOURCE_CHARS: int = 4000
MAX_EVIDENCE_CHARS: int = 20_000

#: Network and academic-search settings.
REQUEST_TIMEOUT: float = 8.0
S2_ENDPOINT: str = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS: str = "title,abstract,url,openAccessPdf,year"
S2_LIMIT: int = 5
#: Open-access PDFs actually fetched and parsed; parsing is slow, so it is few.
ACADEMIC_PDF_LIMIT: int = 3

USER_AGENT: str = "Atlas/1.0 (local research assistant)"

#: Terms that suggest an academic question even when the router did not say so.
ACADEMIC_TERMS: tuple[str, ...] = (
    "study", "studies", "paper", "papers", "journal", "peer-reviewed",
    "citation", "citations", "scientific", "literature", "meta-analysis",
    "arxiv", "preprint", "hypothesis", "empirical",
)

# --- think-tag handling, matching core/agent.py and skills/builder.py --------
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_LEADING_THINK = re.compile(r"^\s*<think\b[^>]*>", re.IGNORECASE)
_TRAILING_THINK = re.compile(r"</think\s*>\s*$", re.IGNORECASE)

_WORD_RE = re.compile(r"[a-z0-9]+")

#: A URL as formatted by QuickSearch.format(): "1. title: snippet (url)".
_TEXT_URL_RE = re.compile(r"\((https?://[^\s)]+)\)")

_BULLET_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s+")
_LABEL_RE = re.compile(r"^([A-Za-z][A-Za-z _-]*):\s*$")


@dataclass(frozen=True, slots=True)
class ResearchReport:
    """The finished product of one research run."""

    title: str
    summary: str
    full_report: str
    sources: list[dict[str, str]]
    timestamp: datetime
    #: Iterations actually run, for diagnostics. Not part of the public shape
    #: the vault cares about, but cheap to keep.
    iterations: int = field(default=0)


class DeepResearcher:
    """Autonomous multi-hop research over the web and academic literature.

    All four collaborators are injected. Only :class:`~core.llm_manager.LLMServerManager`
    is required to have a ``get_deep_client``/``ensure_deep_available`` pair;
    the search, extraction and vault objects are duck-typed, so tests can hand
    in small stand-ins.
    """

    def __init__(
        self,
        llm_manager: "LLMServerManager",
        quick_search: "QuickSearch",
        extractor: "PageExtractor",
        vault_manager: "VaultManager",
    ) -> None:
        self.llm_manager = llm_manager
        self.quick_search = quick_search
        self.extractor = extractor
        self.vault_manager = vault_manager
        #: Lazily built embedding provider; False means "tried and unavailable".
        self._embedder: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def research(
        self,
        question: str,
        depth: int = DEFAULT_DEPTH,
        *,
        intent: Any = None,
        progress: Callable[[str], None] | None = None,
    ) -> ResearchReport:
        """Run the full loop and return (and persist) a report.

        ``depth`` is clamped to ``1..DEEP_RESEARCH_MAX_ITERATIONS``. ``intent``
        is an optional :class:`~search.router.SearchIntent`; when it is
        ``ACADEMIC`` the Semantic Scholar path is taken. An academic query is
        also detected from the wording, so a caller that has not classified the
        query still gets papers.

        ``progress`` is an optional callback for a UI: it is called with a
        short human-readable stage string as the run advances.
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("research() needs a question")

        iterations = max(1, min(int(depth), DEEP_RESEARCH_MAX_ITERATIONS))
        academic = _is_academic(question, intent)

        # Step 1 — plan.
        self._notify(progress, "Planning research")
        sub_questions = await self._plan(question)
        if not sub_questions:
            # Planning is allowed to fail; searching the raw question is still
            # better than returning nothing.
            sub_questions = [question]

        # Step 2 (academic on-ramp) — papers before the web sweep.
        evidence: list[dict[str, str]] = []
        if academic:
            evidence.extend(await self._academic_evidence(question))

        seen: set[str] = {item["url"] for item in evidence}
        titles: dict[str, str] = {item["url"]: item["title"] for item in evidence}

        pending = list(sub_questions)
        for iteration in range(iterations):
            # Step 2 — search sweep for the current questions.
            head = pending[0] if pending else question
            extra = f" (+{len(pending) - 1} more)" if len(pending) > 1 else ""
            self._notify(progress, f"Searching: {head}{extra}")
            candidates = await self._collect_candidates(pending)
            fresh = [c for c in candidates if c["url"] not in seen]

            # Step 3 — score, then extract only the best.
            ranked = await self._rank(question, fresh)
            chosen = ranked[:MAX_EXTRACT_URLS]
            self._notify(progress, f"Reading {len(chosen)} of {len(fresh)} sources")
            extracted = await self._extract_urls(chosen)
            for item in extracted:
                if item["url"] in seen:
                    continue
                seen.add(item["url"])
                titles[item["url"]] = item["title"]
                evidence.append(item)

            if iteration == iterations - 1:
                break

            # Step 4 — synthesis, Step 5 — follow-up queries.
            self._notify(progress, "Synthesising findings")
            pending = await self._find_gaps(question, sub_questions, evidence)
            if not pending:
                logger.info("deep research found no further gaps after iteration %d", iteration + 1)
                break

        # Step 6 — report.
        self._notify(progress, "Writing report")
        report_text = await self._write_report(question, sub_questions, evidence)
        if not report_text:
            report_text = _fallback_report(question, evidence)
        title, summary = _parse_report(report_text, question)

        sources = [
            {"title": titles.get(url, ""), "url": url}
            for url in titles
        ][:DEEP_RESEARCH_MAX_SOURCES]
        report = ResearchReport(
            title=title,
            summary=summary,
            full_report=report_text,
            sources=sources,
            timestamp=datetime.now().astimezone(),
            iterations=iterations,
        )

        # Step 7 — persist.
        self._persist(report, question)
        return report

    def research_async(
        self,
        question: str,
        callback: Callable[[ResearchReport | None], None],
        depth: int = DEFAULT_DEPTH,
        *,
        intent: Any = None,
        progress: Callable[[str], None] | None = None,
    ) -> threading.Thread:
        """Run :meth:`research` on a background thread.

        This is the conversational path: the caller returns immediately and
        ``callback`` is invoked with the finished report (or ``None`` if the
        run failed). Used so Atlas can say it is researching and keep talking.
        """
        def worker() -> None:
            report: ResearchReport | None = None
            try:
                report = asyncio.run(
                    self.research(question, depth, intent=intent, progress=progress)
                )
            except Exception:
                logger.exception("background deep research on %r failed", question)
            try:
                callback(report)
            except Exception:
                logger.exception("deep research callback raised")

        thread = threading.Thread(
            target=worker, name="atlas-deep-research", daemon=True
        )
        thread.start()
        return thread

    def deep_research(
        self,
        question: str,
        depth: int = DEFAULT_DEPTH,
        *,
        intent: Any = None,
        progress: Callable[[str], None] | None = None,
    ) -> ResearchReport:
        """Synchronous wrapper around :meth:`research`.

        The agent's duck-typed tool contract accepts a ``deep_research`` method;
        it calls it from a synchronous turn, where awaiting is not possible.
        """
        return asyncio.run(
            self.research(question, depth, intent=intent, progress=progress)
        )

    @staticmethod
    def _notify(progress: Callable[[str], None] | None, message: str) -> None:
        """Report a stage to ``progress``, never letting it break the run."""
        if progress is None:
            return
        try:
            progress(message)
        except Exception:
            logger.debug("progress callback raised for %r", message, exc_info=True)

    # ------------------------------------------------------------------
    # Step 1 — planning
    # ------------------------------------------------------------------

    async def _plan(self, question: str) -> list[str]:
        prompt = (
            "Break this research question into 4-6 specific sub-questions that "
            "together would fully answer it: "
            f"{question}\n\n"
            "Reply with one sub-question per line and nothing else."
        )
        text = await self._ask(prompt, max_tokens=PLAN_MAX_TOKENS)
        return _parse_sub_questions(text)

    # ------------------------------------------------------------------
    # Step 2 — search sweep
    # ------------------------------------------------------------------

    async def _collect_candidates(self, queries: Sequence[str]) -> list[dict[str, str]]:
        """Search every query concurrently and merge unique URLs."""
        groups = await asyncio.gather(
            *(self._search(query, RESULTS_PER_SUB_QUESTION) for query in queries),
            return_exceptions=True,
        )
        merged: list[dict[str, str]] = []
        seen: set[str] = set()
        for group in groups:
            if isinstance(group, BaseException):
                logger.warning("a sub-question search failed: %s", group)
                continue
            for candidate in group:
                url = candidate.get("url", "")
                if not url or url in seen:
                    continue
                seen.add(url)
                merged.append(candidate)
        return merged

    async def _search(self, query: str, n_results: int) -> list[dict[str, str]]:
        return await asyncio.to_thread(self._search_blocking, query, n_results)

    def _search_blocking(self, query: str, n_results: int) -> list[dict[str, str]]:
        searcher = self.quick_search
        if searcher is None:
            return []

        structured = getattr(searcher, "search_results", None)
        if callable(structured):
            try:
                raw = structured(query, n_results)
            except Exception as exc:
                logger.warning("quick search failed for %r: %s", query, exc)
                return []
            return [_as_candidate(item) for item in raw or []]

        # Fallback for a quick_search that only exposes the text API.
        try:
            text = searcher.search(query, n_results)
        except Exception as exc:
            logger.warning("quick search failed for %r: %s", query, exc)
            return []
        return _results_from_text(text)

    # ------------------------------------------------------------------
    # Step 3 — selective extraction
    # ------------------------------------------------------------------

    async def _rank(
        self, question: str, candidates: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Order candidates by relevance to ``question``, best first."""
        if not candidates:
            return []
        return await asyncio.to_thread(self._rank_blocking, question, candidates)

    def _rank_blocking(
        self, question: str, candidates: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        labels = [candidate.get("title") or candidate.get("url", "") for candidate in candidates]
        embedder = self._get_embedder()
        if embedder is not None:
            try:
                query_vector = embedder.embed_query(question)
                title_vectors = embedder.embed_documents(labels)
                scores = [_cosine(query_vector, vector) for vector in title_vectors]
            except Exception as exc:
                logger.warning("embedding ranking failed (%s) — using keyword overlap", exc)
                scores = [_overlap(question, label) for label in labels]
        else:
            scores = [_overlap(question, label) for label in labels]

        ranked = sorted(
            zip(scores, candidates), key=lambda pair: pair[0], reverse=True
        )
        return [candidate for _, candidate in ranked]

    async def _extract_urls(
        self, candidates: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Fetch the full text of the chosen candidates, skipping failures."""
        urls = [c["url"] for c in candidates if c.get("url")]
        if not urls:
            return []
        title_for = {c["url"]: c.get("title", "") for c in candidates}

        texts = await asyncio.to_thread(self._extract_blocking, urls)

        out: list[dict[str, str]] = []
        for url, text in zip(urls, texts):
            content = (text or "").strip()
            if not content:
                continue
            out.append({"url": url, "title": title_for.get(url, ""), "content": content})
        return out

    def _extract_blocking(self, urls: list[str]) -> list[str]:
        batch = getattr(self.extractor, "extract_batch", None)
        if callable(batch):
            try:
                results = batch(urls)
                # Normalise to exactly one entry per URL, in order.
                results = list(results or [])
                if len(results) != len(urls):
                    logger.warning(
                        "extractor.extract_batch returned %d results for %d URLs",
                        len(results),
                        len(urls),
                    )
                return results[: len(urls)] + [""] * (len(urls) - len(results))
            except Exception as exc:
                logger.warning("batch extraction failed: %s", exc)
                return [""] * len(urls)

        # Fallback: an extractor that only offers single-page extraction.
        return [self._safe_extract(url) for url in urls]

    def _safe_extract(self, url: str) -> str:
        try:
            return self.extractor.extract(url) or ""
        except Exception as exc:
            logger.warning("extraction failed for %s: %s", url, exc)
            return ""

    def _get_embedder(self) -> Any:
        """Build the embedding provider once, or return None if unavailable."""
        if self._embedder is not None:
            return None if self._embedder is False else self._embedder
        try:
            from memory.chroma_store import build_embedding_provider

            self._embedder = build_embedding_provider()
            logger.debug("deep research embedding ranking via %s", self._embedder.describe())
        except Exception as exc:
            logger.warning(
                "no embedding backend for URL ranking (%s) — falling back to "
                "keyword overlap",
                exc,
            )
            self._embedder = False
        return None if self._embedder is False else self._embedder

    # ------------------------------------------------------------------
    # Steps 4-5 — synthesis and follow-ups
    # ------------------------------------------------------------------

    async def _find_gaps(
        self,
        question: str,
        sub_questions: Sequence[str],
        evidence: Sequence[dict[str, str]],
    ) -> list[str]:
        prompt = (
            f"You are researching: {question}\n\n"
            "Sub-questions:\n"
            + "\n".join(f"- {item}" for item in sub_questions)
            + "\n\nEvidence collected so far:\n"
            + _evidence_block(evidence)
            + "\n\nExplain briefly, then list the follow-up web search queries "
            "still needed to close the remaining gaps. Use these exact labels:\n"
            "ANSWERED:\n- ...\n"
            "UNCLEAR:\n- ...\n"
            "FOLLOW-UP:\n- <one search query per line>\n"
            "If nothing more is needed, write NONE under FOLLOW-UP."
        )
        text = await self._ask(prompt, max_tokens=SYNTHESIS_MAX_TOKENS)
        return _parse_labelled_lines(text, "FOLLOW-UP")[:MAX_FOLLOWUP_QUERIES]

    # ------------------------------------------------------------------
    # Step 6 — report
    # ------------------------------------------------------------------

    async def _write_report(
        self,
        question: str,
        sub_questions: Sequence[str],
        evidence: Sequence[dict[str, str]],
    ) -> str:
        prompt = (
            f"Write a structured research report answering: {question}\n\n"
            "Cover each of these sub-questions:\n"
            + "\n".join(f"- {item}" for item in sub_questions)
            + "\n\nUse only this evidence, and cite it inline as [n]:\n"
            + _evidence_block(evidence)
            + "\n\nFormat the report as markdown with exactly these sections:\n"
            "# <a short title>\n"
            "## Executive Summary\n"
            "## Key Findings\n"
            "### <sub-question>\n"
            "## Confidence\n"
            "<High, Medium or Low, with one line of justification>\n"
            "## Suggested Follow-up Questions\n"
            "Do not invent facts that are not in the evidence."
        )
        return (await self._ask(prompt, max_tokens=REPORT_MAX_TOKENS)).strip()

    # ------------------------------------------------------------------
    # Academic on-ramp
    # ------------------------------------------------------------------

    async def _academic_evidence(self, question: str) -> list[dict[str, str]]:
        """Papers from Semantic Scholar, enriched with open-access PDF text."""
        papers = await asyncio.to_thread(self._semantic_scholar_blocking, question)
        if not papers:
            return []

        pdf_urls = [p["pdf_url"] for p in papers if p.get("pdf_url")][:ACADEMIC_PDF_LIMIT]
        if pdf_urls and getattr(self.extractor, "extract_academic_pdf", None):
            texts = await asyncio.gather(
                *(asyncio.to_thread(self._extract_pdf, url) for url in pdf_urls),
                return_exceptions=True,
            )
            pdf_text = {
                url: (text if isinstance(text, str) else "")
                for url, text in zip(pdf_urls, texts)
            }
        else:
            pdf_text = {}

        evidence: list[dict[str, str]] = []
        for paper in papers:
            body = paper.get("abstract", "")
            full = pdf_text.get(paper.get("pdf_url") or "", "")
            if full:
                body = f"{body}\n\n{full}".strip() if body else full
            evidence.append(
                {
                    "url": paper["url"],
                    "title": paper["title"],
                    "content": body,
                }
            )
        return evidence

    def _semantic_scholar_blocking(self, query: str) -> list[dict[str, str]]:
        try:
            import httpx
        except ImportError:  # pragma: no cover - dependency is installed
            return []

        try:
            response = httpx.get(
                S2_ENDPOINT,
                params={"query": query, "fields": S2_FIELDS, "limit": S2_LIMIT},
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("Semantic Scholar search failed: %s", exc)
            return []

        papers: list[dict[str, str]] = []
        for raw in payload.get("data") or []:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            url = str(raw.get("url") or "").strip()
            open_pdf = raw.get("openAccessPdf")
            pdf_url = ""
            if isinstance(open_pdf, dict):
                pdf_url = str(open_pdf.get("url") or "").strip()
            if not title or not url:
                continue
            papers.append(
                {
                    "title": title,
                    "url": url,
                    "abstract": str(raw.get("abstract") or "").strip(),
                    "pdf_url": pdf_url,
                }
            )
        return papers

    def _extract_pdf(self, url: str) -> str:
        try:
            return self.extractor.extract_academic_pdf(url) or ""
        except Exception as exc:
            logger.warning("PDF extraction failed for %s: %s", url, exc)
            return ""

    # ------------------------------------------------------------------
    # Model plumbing
    # ------------------------------------------------------------------

    async def _ask(self, prompt: str, *, max_tokens: int) -> str:
        return await asyncio.to_thread(self._ask_blocking, prompt, max_tokens)

    def _ask_blocking(self, prompt: str, max_tokens: int) -> str:
        """One completion, preferring the deep model but degrading to fast.

        A research run is useless without *some* model, so when the 30B cannot
        be loaded (no VRAM, weights missing) the fast model is asked instead —
        the report is weaker but the user gets one.
        """
        request = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": DEEP_TEMPERATURE,
            "max_tokens": max_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        for kind in ("deep", "fast"):
            try:
                if kind == "deep":
                    if not self.llm_manager.ensure_deep_available():
                        logger.warning("deep model unavailable — using the fast model")
                        continue
                    client = self.llm_manager.get_deep_client()
                else:
                    client = self.llm_manager.get_fast_client()
                request["model"] = kind
                response = client.chat.completions.create(**request)
                content = getattr(response.choices[0].message, "content", None) or ""
                content = _strip_think(content)
                if content:
                    return content
                request["temperature"] = 0.0
                logger.warning("%s model returned no content", kind)
            except Exception as exc:  # the SDK raises many shapes
                logger.warning("%s model request failed: %s", kind, exc)
        return ""

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, report: ResearchReport, question: str) -> None:
        writer = getattr(self.vault_manager, "write_research_report", None)
        if not callable(writer):
            logger.warning("vault_manager has no write_research_report — not persisted")
            return
        try:
            writer(
                title=report.title,
                content=report.full_report,
                sources=report.sources,
                question=question,
                tags=["research"],
            )
        except Exception as exc:
            # Losing the file is bad; losing the in-memory report as well is
            # worse, so this is logged and swallowed.
            logger.warning("could not persist research report: %s", exc)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _strip_think(text: str) -> str:
    """Remove reasoning that leaked into the answer (see core/agent.py)."""
    if not text:
        return ""
    cleaned = _THINK_BLOCK.sub(" ", text)
    cleaned = _LEADING_THINK.sub("", cleaned)
    cleaned = _TRAILING_THINK.sub("", cleaned)
    return cleaned.strip()


def _is_academic(question: str, intent: Any) -> bool:
    """True when the intent says ACADEMIC, or the wording clearly does."""
    label = getattr(intent, "value", intent)
    if isinstance(label, str) and label.strip().upper() == "ACADEMIC":
        return True
    lowered = question.lower()
    return any(term in lowered for term in ACADEMIC_TERMS)


def _parse_sub_questions(text: str) -> list[str]:
    """Pull ordered sub-questions out of a planning reply."""
    items: list[str] = []
    seen: set[str] = set()
    for line in (text or "").splitlines():
        stripped = _BULLET_RE.sub("", line.strip()).strip().strip("\"'")
        # Prose like "Here are the sub-questions:" is not a sub-question.
        if len(stripped) < 12 or stripped.endswith(":"):
            continue
        key = stripped.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(stripped)
        if len(items) >= MAX_SUB_QUESTIONS:
            break
    # Fewer than MIN_SUB_QUESTIONS is accepted: the model knows best how many
    # the question really splits into, and searching one good sub-question
    # beats padding the list to hit a quota.
    return items


def _parse_labelled_lines(text: str, label: str) -> list[str]:
    """Return the bullet lines under ``LABEL:`` in a synthesis reply."""
    lines = (text or "").splitlines()
    collected: list[str] = []
    label_upper = label.upper()
    collecting = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        heading = _LABEL_RE.match(stripped)
        if heading:
            collecting = heading.group(1).strip().upper() == label_upper
            continue
        inline = re.match(rf"^{re.escape(label)}:\s*(.+)$", stripped, re.IGNORECASE)
        if inline:
            collecting = True
            collected.append(inline.group(1).strip())
            continue
        if collecting:
            collected.append(_BULLET_RE.sub("", stripped).strip())
    return [item for item in collected if item and item.upper() != "NONE"]


def _evidence_block(evidence: Sequence[dict[str, str]]) -> str:
    """Render evidence for a prompt, truncated to fit the context budget."""
    if not evidence:
        return "(no evidence collected)"
    parts: list[str] = []
    used = 0
    for index, item in enumerate(evidence, start=1):
        content = (item.get("content") or "").strip()[:PER_SOURCE_CHARS]
        chunk = f"[{index}] {item.get('title') or item.get('url')} ({item.get('url')})\n{content}"
        if used + len(chunk) > MAX_EVIDENCE_CHARS:
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n\n".join(parts) if parts else "(no evidence collected)"


def _parse_report(text: str, question: str) -> tuple[str, str]:
    """Split a generated report into ``(title, executive summary)``."""
    title = ""
    for line in (text or "").splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    if not title:
        title = question if len(question) <= 80 else question[:77].rstrip() + "..."

    summary = _section(text, "Executive Summary")
    if not summary:
        # No labelled summary: fall back to the first non-heading paragraph.
        for block in re.split(r"\n\s*\n", text or ""):
            candidate = block.strip()
            if candidate and not candidate.startswith("#"):
                summary = candidate
                break
    if len(summary) > 1200:
        summary = summary[:1197].rstrip() + "..."
    return title, summary


def _section(text: str, heading: str) -> str:
    """Body of ``## <heading>`` up to the next ``##`` heading."""
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*$(.*?)(?=^##\s|\Z)",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text or "")
    return match.group(1).strip() if match else ""


def _fallback_report(question: str, evidence: Sequence[dict[str, str]]) -> str:
    """A minimal report used only when the model produces nothing at all."""
    lines = [
        f"# Research: {question}",
        "",
        "## Executive Summary",
        (
            f"Collected {len(evidence)} source(s), but the model did not "
            "produce a synthesis. The raw sources are listed below."
        ),
        "",
        "## Key Findings",
    ]
    for index, item in enumerate(evidence, start=1):
        lines.append(f"- [{index}] {item.get('title') or item.get('url')}")
    return "\n".join(lines)


def _as_candidate(item: Any) -> dict[str, str]:
    """Normalise a search result object or mapping into ``{title, url}``."""
    if isinstance(item, dict):
        url = str(item.get("url") or item.get("link") or item.get("href") or "")
        title = str(item.get("title") or item.get("name") or "")
    else:
        url = str(getattr(item, "url", "") or "")
        title = str(getattr(item, "title", "") or "")
    return {"title": title.strip(), "url": url.strip()}


def _results_from_text(text: str) -> list[dict[str, str]]:
    """Recover URLs from QuickSearch's formatted string, as a last resort."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in (text or "").splitlines():
        match = _TEXT_URL_RE.search(line)
        if not match:
            continue
        url = match.group(1)
        if url in seen:
            continue
        seen.add(url)
        title = _BULLET_RE.sub("", line.strip()).split(":", 1)[0].strip()
        out.append({"title": title, "url": url})
    return out


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, 0.0 for a degenerate vector."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def _overlap(question: str, title: str) -> float:
    """Fraction of the question's words present in the title (0..1).

    The no-embeddings fallback for ranking; crude, but it still floats a
    relevant title above an unrelated one.
    """
    query_words = set(_WORD_RE.findall((question or "").lower()))
    title_words = set(_WORD_RE.findall((title or "").lower()))
    if not query_words or not title_words:
        return 0.0
    return len(query_words & title_words) / len(query_words)
