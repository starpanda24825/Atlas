"""
Atlas — Model United Nations assistant.

MUN is a research-heavy, time-pressured format: a delegate needs their country's
actual position, a speech in that voice, honest feedback on the speech, and
working-paper clauses that survive negotiation. This module does all four, and
leans on two other parts of Atlas for the inputs:

* :mod:`modules.university.rag` supplies whatever the delegate has already
  indexed (background guides, past position papers, country profiles).
* :class:`~search.deep_research.DeepResearcher` is triggered for a structured
  country brief when it is available, so recent statements and alliances are
  grounded in live sources rather than model memory.

Example::

    from modules.university.mun import MUNAssistant

    assistant = MUNAssistant(llm_manager, rag, vault, researcher)
    brief = assistant.research_country_position("France", "UNSC", "maritime security")
    speech = assistant.draft_speech("France", "UNSC", resolution_text, user_notes)
    review = assistant.give_speech_feedback(speech["speech"])
    paper = assistant.generate_working_paper_points("France", "maritime security", allies=["UK"])
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Sequence

from modules.university.rag import (
    DEFAULT_TOP_K,
    UniversityRAG,
    ask_deep,
    parse_json,
)

if TYPE_CHECKING:  # injected collaborators; imports here are typing-only
    from core.llm_manager import LLMServerManager
    from memory.vault_manager import VaultManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

BRIEF_MAX_TOKENS: int = 2200
SPEECH_MAX_TOKENS: int = 1800
FEEDBACK_MAX_TOKENS: int = 1600
WORKING_PAPER_MAX_TOKENS: int = 1600

DEFAULT_SPEECH_WORDS: int = 450
MAX_SPEECH_WORDS: int = 2000

#: The sections the country brief is assembled from, in report order.
BRIEF_SECTIONS: tuple[str, ...] = (
    "Historical Stance",
    "Recent Statements",
    "Alliances",
    "Economic Interests",
    "Likely Negotiating Position",
    "Key Vulnerabilities",
)

#: Where a saved brief or speech goes in the vault.
MUN_FOLDER: str = "research"


class MUNAssistant:
    """Country research, speech drafting, feedback and working-paper clauses."""

    def __init__(
        self,
        llm_manager: "LLMServerManager",
        rag: UniversityRAG | None = None,
        vault_manager: "VaultManager | None" = None,
        researcher: Any = None,
    ) -> None:
        self.llm_manager = llm_manager
        self._rag = rag
        self.vault_manager = vault_manager
        self.researcher = researcher

    @property
    def rag(self) -> UniversityRAG:
        if self._rag is None:
            from modules.university.rag import get_university_rag

            self._rag = get_university_rag()
        return self._rag

    # ------------------------------------------------------------------
    # Country research
    # ------------------------------------------------------------------

    def research_country_position(
        self,
        country: str,
        committee: str,
        topic: str,
        *,
        depth: int = 3,
        use_web: bool = True,
        top_k: int = DEFAULT_TOP_K,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Build a structured brief on one country's position.

        Combines indexed documents with an optional deep-research run, then asks
        the deep model to organise the evidence into a MUN-shaped brief.
        """
        nation = (country or "").strip()
        body = (committee or "").strip()
        subject = (topic or "").strip()
        if not nation or not subject:
            raise ValueError("research_country_position() needs a country and a topic")

        question = _country_question(nation, body, subject)
        context, citations = self._indexed_context(question, top_k)
        web_report, web_sources = self._web_research(question, depth) if use_web else ("", [], )
        # `web_report` is a string; the sources come back separately.

        prompt = _brief_prompt(nation, body, subject, context, web_report)
        raw = ask_deep(self.llm_manager, prompt, max_tokens=BRIEF_MAX_TOKENS)

        sections = _parse_sections(raw)
        brief: dict[str, Any] = {
            "country": nation,
            "committee": body,
            "topic": subject,
            "historical_stance": sections.get("HISTORICAL STANCE", ""),
            "recent_statements": sections.get("RECENT STATEMENTS", ""),
            "alliances": sections.get("ALLIANCES", ""),
            "economic_interests": sections.get("ECONOMIC INTERESTS", ""),
            "likely_position": sections.get("LIKELY NEGOTIATING POSITION", ""),
            "key_vulnerabilities": sections.get("KEY VULNERABILITIES", ""),
            "sections": sections,
            "report": raw,
            "sources": _merge_sources(citations, web_sources),
            "used_documents": bool(context),
            "used_web_research": bool(web_report),
            "generated_at": _now(),
            "error": None if raw else "the deep model produced no brief",
        }
        if persist and raw:
            self._persist(f"Position brief: {nation} on {subject}", raw, ["mun", "brief"])
        return brief

    # ------------------------------------------------------------------
    # Speech drafting
    # ------------------------------------------------------------------

    def draft_speech(
        self,
        country: str,
        committee: str,
        resolution: str = "",
        user_notes: str = "",
        word_limit: int = DEFAULT_SPEECH_WORDS,
        *,
        use_web: bool = True,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Draft a position speech in the country's voice.

        The delegate's own ``user_notes`` are treated as authoritative and mixed
        with a freshly-researched brief; the result is trimmed to ``word_limit``.
        """
        nation = (country or "").strip()
        body = (committee or "").strip()
        if not nation:
            raise ValueError("draft_speech() needs a country")
        limit = _clamp_words(word_limit)
        notes = (user_notes or "").strip()
        text = (resolution or "").strip()

        # A speech needs the country's actual position behind it, so a brief is
        # always built; ``use_web`` only decides whether it is grounded in live
        # research or kept to what is indexed.
        brief: dict[str, Any] | None = None
        try:
            brief = self.research_country_position(
                nation,
                body,
                _topic_from_resolution(text, body),
                use_web=use_web,
                persist=False,
            )
        except Exception as exc:
            logger.warning("country brief for %s failed: %s", nation, exc)

        prompt = _speech_prompt(nation, body, text, notes, limit, brief)
        raw = ask_deep(self.llm_manager, prompt, max_tokens=SPEECH_MAX_TOKENS)

        speech = _clean_speech(raw)
        truncated = False
        if speech and _word_count(speech) > limit:
            speech = _truncate_words(speech, limit)
            truncated = True

        result: dict[str, Any] = {
            "country": nation,
            "committee": body,
            "resolution": text,
            "speech": speech,
            "word_count": _word_count(speech),
            "word_limit": limit,
            "truncated": truncated,
            "notes_used": bool(notes),
            "brief_used": bool(brief and brief.get("report")),
            "sources": (brief or {}).get("sources", []),
            "generated_at": _now(),
            "error": None if speech else "the deep model produced no speech",
        }
        if persist and speech:
            self._persist(f"Speech: {nation} on {body or _topic_from_resolution(text, body)}", speech, ["mun", "speech"])
        return result

    # ------------------------------------------------------------------
    # Speech feedback
    # ------------------------------------------------------------------

    def give_speech_feedback(self, speech_text: str) -> dict[str, Any]:
        """Critique a speech against the five things an adjudicator scores."""
        text = (speech_text or "").strip()
        if not text:
            raise ValueError("give_speech_feedback() needs some speech text")

        raw = ask_deep(
            self.llm_manager,
            _feedback_prompt(text),
            max_tokens=FEEDBACK_MAX_TOKENS,
            temperature=0.2,
        )
        payload = parse_json(raw)
        feedback: dict[str, Any] = {
            "diplomatic_tone": "",
            "factual_accuracy": "",
            "rhetorical_effectiveness": "",
            "logical_structure": "",
            "convention_adherence": "",
            "overall_score": None,
            "strengths": [],
            "improvements": [],
            "verdict": "",
        }
        if isinstance(payload, dict):
            for key in (
                "diplomatic_tone",
                "factual_accuracy",
                "rhetorical_effectiveness",
                "logical_structure",
                "convention_adherence",
                "verdict",
            ):
                value = payload.get(key)
                if value:
                    feedback[key] = str(value).strip()
            feedback["strengths"] = _string_list(payload.get("strengths"))
            feedback["improvements"] = _string_list(
                payload.get("improvements") or payload.get("suggestions")
            )
            feedback["overall_score"] = _score(payload.get("overall_score"))

        if not any(
            feedback[key]
            for key in (
                "diplomatic_tone",
                "factual_accuracy",
                "rhetorical_effectiveness",
                "logical_structure",
                "convention_adherence",
            )
        ) and raw:
            # JSON parsing failed: return the prose rather than nothing.
            feedback["verdict"] = raw.strip()

        feedback["word_count"] = _word_count(text)
        feedback["raw"] = raw
        feedback["error"] = None if raw else "the deep model produced no feedback"
        return feedback

    # ------------------------------------------------------------------
    # Working paper
    # ------------------------------------------------------------------

    def generate_working_paper_points(
        self,
        country: str,
        topic: str,
        allies: Sequence[str] | str | None = None,
        *,
        top_k: int = DEFAULT_TOP_K,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Suggest preambulatory and operative clauses aligned with ``country``."""
        nation = (country or "").strip()
        subject = (topic or "").strip()
        if not nation or not subject:
            raise ValueError("generate_working_paper_points() needs a country and a topic")
        ally_list = _as_list(allies)

        context, citations = self._indexed_context(
            f"{nation} position on {subject}", top_k
        )
        raw = ask_deep(
            self.llm_manager,
            _working_paper_prompt(nation, subject, ally_list, context),
            max_tokens=WORKING_PAPER_MAX_TOKENS,
        )
        payload = parse_json(raw)

        result: dict[str, Any] = {
            "country": nation,
            "topic": subject,
            "allies": ally_list,
            "preambulatory_clauses": [],
            "operative_clauses": [],
            "rationale": "",
            "sources": citations,
            "generated_at": _now(),
            "error": None,
        }
        if isinstance(payload, dict):
            result["preambulatory_clauses"] = _string_list(
                payload.get("preambulatory_clauses") or payload.get("preambulatory")
            )
            result["operative_clauses"] = _string_list(
                payload.get("operative_clauses") or payload.get("operative")
            )
            result["rationale"] = str(payload.get("rationale") or "").strip()
        elif not raw:
            result["error"] = "the deep model produced no working paper"

        if not result["operative_clauses"] and raw:
            # Fall back to bullet extraction so the delegate still gets clauses.
            result["operative_clauses"] = _bullets(raw)
            result["rationale"] = result["rationale"] or (
                "Parsed from the model's prose; treat wording as a draft."
            )

        if persist and (result["operative_clauses"] or result["preambulatory_clauses"]):
            body = "\n".join(
                ["## Preambulatory clauses"]
                + [f"- {item}" for item in result["preambulatory_clauses"]]
                + ["", "## Operative clauses"]
                + [f"- {item}" for item in result["operative_clauses"]]
            )
            self._persist(f"Working paper points: {nation} on {subject}", body, ["mun", "working-paper"])
        return result

    # ------------------------------------------------------------------
    # Collaborator plumbing
    # ------------------------------------------------------------------

    def _indexed_context(self, question: str, top_k: int) -> tuple[str, list[dict[str, Any]]]:
        try:
            return self.rag.query(question, top_k=top_k)
        except Exception as exc:
            logger.warning("indexed lookup for MUN failed: %s", exc)
            return "", []

    def _web_research(self, question: str, depth: int) -> tuple[str, list[dict[str, Any]]]:
        """Run the deep researcher if one is attached. Returns (report, sources)."""
        researcher = self.researcher
        if researcher is None:
            return "", []

        report: Any = None
        try:
            sync = getattr(researcher, "deep_research", None)
            if callable(sync):
                report = sync(question, depth)
            else:
                lane = getattr(researcher, "research", None)
                if callable(lane):
                    import asyncio

                    report = asyncio.run(lane(question, depth))
                elif callable(researcher):
                    report = researcher(question)
        except Exception as exc:
            logger.warning("deep research for MUN failed: %s", exc)
            return "", []

        if report is None:
            return "", []
        text = str(
            getattr(report, "full_report", None)
            or (report.get("full_report") if isinstance(report, dict) else "")
            or ""
        )
        sources = getattr(report, "sources", None)
        if sources is None and isinstance(report, dict):
            sources = report.get("sources")
        return text, _merge_sources([], sources or [])

    def _persist(self, title: str, content: str, tags: Sequence[str]) -> None:
        writer = getattr(self.vault_manager, "write_note", None)
        if not callable(writer):
            return
        try:
            writer(
                title=title,
                content=content,
                folder=MUN_FOLDER,
                note_type="research",
                tags=list(tags) + ["mun"],
            )
        except Exception as exc:
            logger.warning("could not save the MUN note: %s", exc)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _country_question(country: str, committee: str, topic: str) -> str:
    where = f" in the {committee} committee" if committee else ""
    return (
        f"What is {country}'s position{where} on {topic}? Cover the country's "
        "historical stance, recent official statements, alliances and blocs, "
        "economic interests at stake, and its likely negotiating position and "
        "red lines."
    )


def _brief_prompt(
    country: str, committee: str, topic: str, context: str, web_report: str
) -> str:
    material: list[str] = []
    if context:
        material.append("Indexed delegate documents:\n" + context)
    if web_report:
        material.append("Deep research findings:\n" + web_report[:12000])
    evidence = "\n\n".join(material) if material else "(no source material available)"

    sections = "\n".join(f"## {name}" for name in BRIEF_SECTIONS)
    return (
        "You are a Model United Nations coach preparing a delegacy brief. Base "
        "every claim on the material below; where it is silent, say so instead "
        "of inventing positions.\n\n"
        f"Country: {country}\n"
        f"Committee: {committee or 'unspecified'}\n"
        f"Topic: {topic}\n\n"
        f"{evidence}\n\n"
        "Write the brief as markdown using exactly these sections:\n"
        f"{sections}\n\n"
        "Each section should be a short paragraph or a few tight bullets. Be "
        "specific: name treaties, blocs and past resolutions where you can."
    )


def _speech_prompt(
    country: str,
    committee: str,
    resolution: str,
    notes: str,
    limit: int,
    brief: dict[str, Any] | None,
) -> str:
    parts = [
        f"Write a Model United Nations position speech for {country}.",
        f"Committee: {committee or 'unspecified'}",
        f"Length: at most {limit} words.",
    ]
    if resolution:
        parts.append(f"\nResolution under debate:\n{resolution[:6000]}")
    if brief and brief.get("report"):
        parts.append(f"\nCountry brief to draw on:\n{str(brief['report'])[:6000]}")
    if notes:
        parts.append(
            "\nThe delegate's own notes are authoritative — use them and do not "
            f"contradict them:\n{notes}"
        )
    parts.append(
        "\nWrite in the first person plural as the delegation. Open with the "
        "formal address ('Honourable Chair, distinguished delegates'), state the "
        "country's position clearly, reference the country's interests and any "
        "alliances, and close by urging a specific course of action. Keep the "
        "tone diplomatic throughout. Return only the speech text."
    )
    return "\n".join(parts)


def _feedback_prompt(speech: str) -> str:
    return (
        "You are an experienced MUN adjudicator. Critique the speech below on "
        "five axes: diplomatic tone, factual accuracy, rhetorical effectiveness, "
        "logical structure, and adherence to MUN convention and format. Be "
        "specific and honest; point to wording where you can.\n\n"
        f"Speech:\n{speech[:8000]}\n\n"
        "Reply with ONLY JSON in this shape:\n"
        '{"diplomatic_tone": "...", "factual_accuracy": "...", '
        '"rhetorical_effectiveness": "...", "logical_structure": "...", '
        '"convention_adherence": "...", "overall_score": 7, '
        '"strengths": ["..."], "improvements": ["..."], "verdict": "..."}\n'
        "`overall_score` is out of 10."
    )


def _working_paper_prompt(country: str, topic: str, allies: list[str], context: str) -> str:
    alliance_line = (
        f"Align where possible with: {', '.join(allies)}." if allies else "No named allies."
    )
    return (
        "You are drafting working-paper clauses for a Model United Nations "
        f"delegate from {country} on {topic}.\n{alliance_line}\n\n"
        + (f"Delegate documents:\n{context}\n\n" if context else "")
        + "Suggest clauses that advance this country's interests and are likely "
        "to survive negotiation. Preambulatory clauses begin with a participle "
        "(e.g. 'Recognising', 'Deeply concerned'); operative clauses are "
        "numbered actions beginning with a verb (e.g. 'Calls upon', 'Requests').\n\n"
        "Reply with ONLY JSON in this shape:\n"
        '{"preambulatory_clauses": ["..."], "operative_clauses": ["..."], '
        '"rationale": "why these align with the country\'s interests"}'
    )


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.*?)\s*$")
_CAPS_HEADING_RE = re.compile(r"^\s*([A-Z][A-Z0-9 /_&'-]{2,60})\s*:\s*(.*)$")
_LABEL_NORMALISE_RE = re.compile(r"[^a-z0-9]+")


def _parse_sections(text: str) -> dict[str, str]:
    """Split a labelled reply into ``{NORMALISED LABEL: body}``.

    Handles both markdown headings (``## Alliances``) and inline labels
    (``ALLIANCES: text``), because models produce both.
    """
    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current and buffer:
            body = "\n".join(buffer).strip()
            if body:
                sections[current] = body

    for line in (text or "").splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            current = _normalise_label(heading.group(1))
            buffer = []
            continue
        inline = _CAPS_HEADING_RE.match(line)
        if inline and not line.lstrip().startswith(("-", "*", "#")):
            flush()
            current = _normalise_label(inline.group(1))
            buffer = [inline.group(2)] if inline.group(2).strip() else []
            continue
        if current:
            buffer.append(line)
    flush()
    return sections


def _normalise_label(text: str) -> str:
    # Lower-case before collapsing separators: the character class keeps only
    # a-z0-9, so an un-lowered "Historical Stance" would lose its capitals.
    return _LABEL_NORMALISE_RE.sub(" ", text.lower()).strip().upper()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clamp_words(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = DEFAULT_SPEECH_WORDS
    if number <= 0:
        number = DEFAULT_SPEECH_WORDS
    return max(50, min(number, MAX_SPEECH_WORDS))


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _truncate_words(text: str, limit: int) -> str:
    words = re.findall(r"\S+", text or "")
    if len(words) <= limit:
        return text
    # Attach the marker to the final word rather than as its own space-separated
    # token, so the result still counts as exactly ``limit`` words.
    clipped = " ".join(words[:limit]).rstrip(",;:")
    return clipped + "…"


def _clean_speech(text: str) -> str:
    """Strip a leading label or fence from a generated speech."""
    cleaned = (text or "").strip()
    fenced = re.search(r"```(?:text|markdown)?\s*\n(.*?)```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    cleaned = re.sub(r"^(?:speech|position speech)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _topic_from_resolution(resolution: str, committee: str) -> str:
    text = (resolution or "").strip()
    if not text:
        return committee or "the agenda item"
    first_line = text.splitlines()[0].strip()
    topic = re.sub(r"^(?:topic|agenda item|resolution)\s*[:#-]\s*", "", first_line, flags=re.IGNORECASE)
    return topic[:200] or (committee or "the agenda item")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip(" -*•\t") for line in value.splitlines() if line.strip(" -*•\t")]
    if isinstance(value, Sequence):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return [str(value)]


def _bullets(text: str) -> list[str]:
    out: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(("-", "*", "•")) or re.match(r"^\d+[.)]\s+", stripped):
            cleaned = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", stripped).strip()
            if cleaned:
                out.append(cleaned)
    return out[:12]


def _score(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
        if not match:
            return None
        number = float(match.group())
    return max(0.0, min(number, 10.0))


def _as_list(value: Sequence[str] | str | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,;]", value) if part.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _merge_sources(
    indexed: Sequence[dict[str, Any]], web: Sequence[Any]
) -> list[dict[str, str]]:
    """Merge indexed citations and web sources into one de-duplicated list."""
    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in list(indexed or []) + list(web or []):
        if isinstance(item, dict):
            title = str(item.get("title") or item.get("source") or "")
            url = str(item.get("url") or item.get("path") or "")
        else:
            title = str(getattr(item, "title", "") or "")
            url = str(getattr(item, "url", "") or "")
        key = url or title
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append({"title": title, "url": url})
    return merged


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

_default: MUNAssistant | None = None
_default_lock = threading.Lock()


def get_mun_assistant() -> MUNAssistant:
    """Process-wide assistant, wired to the shared RAG, vault and researcher."""
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                from core.llm_manager import LLMServerManager
                from modules.university.rag import get_university_rag

                _default = MUNAssistant(
                    LLMServerManager(), get_university_rag()
                )
    return _default


def research_country_position(
    country: str, committee: str, topic: str, **kwargs: Any
) -> dict[str, Any]:
    return get_mun_assistant().research_country_position(country, committee, topic, **kwargs)


def draft_speech(
    country: str,
    committee: str,
    resolution: str = "",
    user_notes: str = "",
    word_limit: int = DEFAULT_SPEECH_WORDS,
    **kwargs: Any,
) -> dict[str, Any]:
    return get_mun_assistant().draft_speech(
        country, committee, resolution, user_notes, word_limit, **kwargs
    )


def give_speech_feedback(speech_text: str) -> dict[str, Any]:
    return get_mun_assistant().give_speech_feedback(speech_text)


def generate_working_paper_points(
    country: str, topic: str, allies: Sequence[str] | str | None = None, **kwargs: Any
) -> dict[str, Any]:
    return get_mun_assistant().generate_working_paper_points(country, topic, allies, **kwargs)


__all__ = [
    "MUNAssistant",
    "draft_speech",
    "generate_working_paper_points",
    "get_mun_assistant",
    "give_speech_feedback",
    "research_country_position",
]
