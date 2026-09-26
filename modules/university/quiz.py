"""
Atlas — quizzes and flashcards from course material.

:class:`QuizGenerator` sits on top of :mod:`modules.university.rag`: the topic
is used to retrieve the relevant pages, and the questions are written from that
context so they are about the student's own course rather than the model's
general knowledge. When nothing is indexed the model is told to say so in the
quiz's ``sources``.

Two formats are produced:

* :meth:`QuizGenerator.generate_quiz` — multiple choice, four options each, one
  correct answer and an explanation, returned as a structured dict.
* :meth:`QuizGenerator.generate_flashcards` — front/back cards, simpler and
  faster to revise from.

:meth:`QuizGenerator.run_interactive_quiz` drives a quiz by voice: it speaks the
question and options, listens for the answer, grades it, gives feedback and
tallies the score. It talks to the voice pipeline through two duck-typed calls
(``speak`` and ``listen_once``), so it also runs in a plain console and is easy
to test.

Example::

    from modules.university.quiz import QuizGenerator

    quiz = QuizGenerator(llm_manager, rag).generate_quiz("realism", 5, "hard")
    result = QuizGenerator(llm_manager, rag).run_interactive_quiz(quiz, voice)
    print(result["correct"], "/", result["total"])
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Sequence

from modules.university.rag import (
    DEFAULT_TOP_K,
    UniversityRAG,
    ask_deep,
    ask_fast,
    parse_json,
    strip_think,
)

if TYPE_CHECKING:  # injected collaborators; imports here are typing-only
    from core.llm_manager import LLMServerManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

QUIZ_MAX_TOKENS: int = 3000
FLASHCARD_MAX_TOKENS: int = 2000
FEEDBACK_MAX_TOKENS: int = 32

MIN_QUESTIONS: int = 1
MAX_QUESTIONS: int = 25
MIN_CARDS: int = 1
MAX_CARDS: int = 50

DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")

#: A fuzzy option match must clear this fraction of overlap to count.
_OPTION_MATCH_THRESHOLD: float = 0.34

_OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")

_WORD_RE = re.compile(r"[a-z0-9]+")
_LETTER_INLINE_RE = re.compile(r"\b(?:option|answer|choice|it'?s)\s*[:=-]?\s*\(?([a-f])\b")
_LETTER_SPEECH_RE = re.compile(r"[,\s]*(?:the\s+)?(?:answer\s+is\s+|it'?s\s+)?\(?([a-f])\)?[.)]?\s*$")
_LETTER_ALONE_RE = re.compile(r"^\s*\(?([a-f])\)?[.)]*\s*$")
_NUMBER_RE = re.compile(r"\b([1-9])\b")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class QuizGenerator:
    """Writes quizzes and flashcards from indexed course material."""

    def __init__(
        self,
        llm_manager: "LLMServerManager",
        rag: UniversityRAG | None = None,
        voice: Any = None,
    ) -> None:
        self.llm_manager = llm_manager
        self._rag = rag
        self.voice = voice

    @property
    def rag(self) -> UniversityRAG:
        """The RAG module, defaulting to the process-wide instance."""
        if self._rag is None:
            from modules.university.rag import get_university_rag

            self._rag = get_university_rag()
        return self._rag

    # ------------------------------------------------------------------
    # Quiz generation
    # ------------------------------------------------------------------

    def generate_quiz(
        self,
        topic: str,
        n_questions: int = 5,
        difficulty: str = "medium",
        doc_filter: str | Sequence[str] | None = None,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, Any]:
        """Generate ``n_questions`` multiple-choice questions on ``topic``.

        ``doc_filter`` restricts the source material to one filename or a list
        of them. Returns a structured quiz dict; a generation failure is
        reported in ``error`` with an empty ``questions`` list rather than
        raised, so the UI can show something useful.
        """
        subject = (topic or "").strip()
        if not subject:
            raise ValueError("generate_quiz() needs a topic")
        count = _clamp(n_questions, MIN_QUESTIONS, MAX_QUESTIONS)
        level = _normalise_difficulty(difficulty)

        context, citations = self._context(subject, top_k, doc_filter)
        prompt = _quiz_prompt(subject, count, level, context)
        raw = ask_deep(self.llm_manager, prompt, max_tokens=QUIZ_MAX_TOKENS)

        quiz: dict[str, Any] = {
            "topic": subject,
            "difficulty": level,
            "requested": count,
            "n_questions": 0,
            "questions": [],
            "sources": citations,
            "used_documents": bool(context),
            "generated_at": _now(),
            "error": None,
        }
        if not raw:
            quiz["error"] = "the deep model produced no quiz"
            return quiz

        payload = parse_json(raw)
        questions = _normalise_questions(payload, count)
        if not questions:
            quiz["error"] = "the model's quiz could not be parsed"
            logger.warning("quiz generation for %r produced no usable questions", subject)
            return quiz

        quiz["questions"] = questions
        quiz["n_questions"] = len(questions)
        if len(questions) < count:
            quiz["note"] = f"the model returned {len(questions)} of {count} questions"
        return quiz

    # ------------------------------------------------------------------
    # Flashcards
    # ------------------------------------------------------------------

    def generate_flashcards(
        self,
        topic: str,
        n_cards: int = 10,
        doc_filter: str | Sequence[str] | None = None,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, Any]:
        """Generate ``n_cards`` front/back flashcards on ``topic``."""
        subject = (topic or "").strip()
        if not subject:
            raise ValueError("generate_flashcards() needs a topic")
        count = _clamp(n_cards, MIN_CARDS, MAX_CARDS)

        context, citations = self._context(subject, top_k, doc_filter)
        prompt = (
            f"Write {count} revision flashcards for a university student.\n"
            f"Topic: {subject}\n\n"
            + (_source_block(context))
            + "\n\nEach card has a short prompt on the front and a concise, accurate "
            "answer on the back (one or two sentences). Reply with ONLY JSON:\n"
            '{"cards": [{"front": "...", "back": "..."}]}\n'
        )
        raw = ask_deep(self.llm_manager, prompt, max_tokens=FLASHCARD_MAX_TOKENS)

        deck: dict[str, Any] = {
            "topic": subject,
            "requested": count,
            "n_cards": 0,
            "cards": [],
            "sources": citations,
            "used_documents": bool(context),
            "generated_at": _now(),
            "error": None,
        }
        payload = parse_json(raw) if raw else None
        cards = _normalise_cards(payload)
        if not cards:
            deck["error"] = "the model produced no usable flashcards"
            return deck
        deck["cards"] = cards[:count]
        deck["n_cards"] = len(deck["cards"])
        return deck

    # ------------------------------------------------------------------
    # Interactive quiz (voice)
    # ------------------------------------------------------------------

    def run_interactive_quiz(
        self,
        quiz: dict[str, Any],
        voice: Any = None,
        *,
        on_question: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run a quiz aloud, grading each spoken answer.

        ``voice`` is any object with ``speak(text)`` and ``listen_once()`` —
        the voice pipeline, a test double, or ``None`` for a console session.
        Returns a result dict with the score and a per-question transcript.
        """
        questions = list((quiz or {}).get("questions") or [])
        if not questions:
            raise ValueError("run_interactive_quiz() needs a quiz with questions")

        speaker = voice if voice is not None else self.voice
        topic = str((quiz or {}).get("topic") or "quiz")
        total = len(questions)
        responses: list[dict[str, Any]] = []
        correct_count = 0

        self._say(speaker, f"Starting the {topic} quiz. {total} question"
                           f"{'s' if total != 1 else ''}.")
        for number, question in enumerate(questions, start=1):
            if on_question is not None:
                _safe_call(on_question, number, question)
            self._say(speaker, _spoken_question(number, total, question))
            answer = self._listen(speaker)

            chosen = self._resolve_answer(answer, question)
            expected = int(question.get("answer_index", 0))
            is_correct = chosen == expected
            if is_correct:
                correct_count += 1

            feedback = _feedback(question, chosen, is_correct)
            self._say(speaker, feedback)
            responses.append(
                {
                    "number": number,
                    "question": question.get("question", ""),
                    "heard": answer,
                    "chosen_index": chosen,
                    "chosen": _option_text(question, chosen),
                    "expected_index": expected,
                    "expected": _option_text(question, expected),
                    "correct": is_correct,
                    "explanation": question.get("explanation", ""),
                }
            )

        score = correct_count / total if total else 0.0
        summary = (
            f"Quiz complete. You scored {correct_count} out of {total} "
            f"({round(score * 100)} percent)."
        )
        self._say(speaker, summary)
        return {
            "topic": topic,
            "total": total,
            "correct": correct_count,
            "score": round(score, 4),
            "responses": responses,
            "summary": summary,
            "finished_at": _now(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _context(
        self,
        topic: str,
        top_k: int,
        doc_filter: str | Sequence[str] | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Retrieve source material, tolerating a RAG that cannot answer."""
        try:
            return self.rag.query(topic, top_k=top_k, doc_filter=doc_filter)
        except Exception as exc:
            logger.warning("could not retrieve material for %r: %s", topic, exc)
            return "", []

    def _say(self, voice: Any, text: str) -> None:
        speaker = getattr(voice, "speak", None)
        if callable(speaker):
            try:
                speaker(text)
                return
            except Exception as exc:
                logger.warning("voice.speak failed (%s) — printing instead", exc)
        logger.info("quiz says: %s", text)
        print(text)  # noqa: T201 - console fallback when there is no voice

    def _listen(self, voice: Any) -> str:
        listener = getattr(voice, "listen_once", None)
        if callable(listener):
            try:
                return str(listener() or "").strip()
            except Exception as exc:
                logger.warning("voice.listen_once failed: %s", exc)
                return ""
        try:
            return input().strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    def _resolve_answer(self, response: str, question: dict[str, Any]) -> int | None:
        """Best-effort mapping of a spoken answer to an option index."""
        text = (response or "").strip()
        if not text:
            return None
        index = _letter_index(text)
        options = list(question.get("options") or [])
        if index is not None and index < len(options):
            return index

        best, score = _best_option(text, options)
        if best is not None and score >= _OPTION_MATCH_THRESHOLD:
            return best
        return self._judge_with_model(text, question)

    def _judge_with_model(self, response: str, question: dict[str, Any]) -> int | None:
        """Ask the fast model to grade an ambiguous spoken answer."""
        options = list(question.get("options") or [])
        if not options:
            return None
        lettered = "\n".join(
            f"{_OPTION_LABELS[i]}. {option}" for i, option in enumerate(options)
        )
        prompt = (
            "A student answered a multiple-choice question out loud. Decide which "
            "option they chose and reply with ONLY that letter, or NONE if it is "
            "impossible to tell.\n\n"
            f"Question: {question.get('question', '')}\n"
            f"Options:\n{lettered}\n\n"
            f"The student said: {response!r}\n"
            "Answer with a single letter (A, B, C, D) or NONE."
        )
        reply = strip_think(
            ask_fast(
                self.llm_manager,
                prompt,
                max_tokens=FEEDBACK_MAX_TOKENS,
                temperature=0.0,
            )
        )
        match = re.search(r"\b([A-F])\b", reply.upper())
        if not match:
            return None
        index = ord(match.group(1)) - 65
        return index if index < len(options) else None


# ---------------------------------------------------------------------------
# Prompt and parsing helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clamp(value: Any, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(number, high))


def _normalise_difficulty(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in DIFFICULTIES else "medium"


def _source_block(context: str) -> str:
    if context:
        return (
            "Base the questions on this course material, and prefer it over your "
            "general knowledge:\n\n" + context
        )
    return (
        "No course material has been indexed for this topic, so use accurate "
        "general knowledge and keep the questions general."
    )


def _quiz_prompt(topic: str, count: int, difficulty: str, context: str) -> str:
    return (
        "You are writing a multiple-choice quiz for a university student.\n"
        f"Topic: {topic}\n"
        f"Difficulty: {difficulty}\n"
        f"Write exactly {count} question{'s' if count != 1 else ''}.\n\n"
        + _source_block(context)
        + "\n\nRules:\n"
        f"- Exactly {count} questions.\n"
        "- Exactly 4 options per question, on separate array entries.\n"
        "- Exactly one correct option; give its 0-based index as `answer_index`.\n"
        "- Include a one-sentence `explanation` for the correct answer.\n"
        "- Do not number the options in their text; the UI adds letters.\n\n"
        "Reply with ONLY JSON in this shape:\n"
        '{"questions": [{"question": "...", "options": ["...", "...", "...", "..."], '
        '"answer_index": 0, "explanation": "..."}]}'
    )


def _normalise_questions(payload: Any, expected: int) -> list[dict[str, Any]]:
    """Coerce a parsed model payload into well-formed question dicts."""
    if isinstance(payload, dict):
        raw_items = payload.get("questions") or payload.get("quiz") or []
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = []
    if not isinstance(raw_items, list):
        return []

    questions: list[dict[str, Any]] = []
    for raw in raw_items:
        item = _normalise_question(raw)
        if item is not None:
            questions.append(item)
        if len(questions) >= expected:
            break
    return questions


def _normalise_question(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    question = str(raw.get("question") or raw.get("prompt") or "").strip()
    options = raw.get("options") or raw.get("choices") or []
    if not question or not isinstance(options, list):
        return None
    cleaned = [str(_option_value(option)).strip() for option in options]
    cleaned = [option for option in cleaned if option]
    if len(cleaned) < 2:
        return None
    # Clamp to four options, as the brief specifies.
    cleaned = cleaned[:4]
    while len(cleaned) < 4:
        cleaned.append("None of the above")

    index = raw.get("answer_index", raw.get("correct_index", raw.get("answer")))
    if isinstance(index, str):
        index = _letter_index(index)
    try:
        answer_index = int(index)
    except (TypeError, ValueError):
        answer_index = 0
    answer_index = max(0, min(answer_index, len(cleaned) - 1))

    return {
        "question": question,
        "options": cleaned,
        "answer_index": answer_index,
        "answer": _OPTION_LABELS[answer_index],
        "explanation": str(raw.get("explanation") or raw.get("rationale") or "").strip(),
    }


def _option_value(option: Any) -> str:
    """An option may arrive as a bare string or as ``{"text": ...}``."""
    if isinstance(option, dict):
        return str(option.get("text") or option.get("option") or option.get("value") or "")
    return str(option)


def _normalise_cards(payload: Any) -> list[dict[str, str]]:
    if isinstance(payload, dict):
        raw_items = payload.get("cards") or payload.get("flashcards") or []
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = []
    if not isinstance(raw_items, list):
        return []
    cards: list[dict[str, str]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        front = str(raw.get("front") or raw.get("question") or raw.get("term") or "").strip()
        back = str(raw.get("back") or raw.get("answer") or raw.get("definition") or "").strip()
        if front and back:
            cards.append({"front": front, "back": back})
    return cards


# ---------------------------------------------------------------------------
# Spoken-answer helpers
# ---------------------------------------------------------------------------


def _spoken_question(number: int, total: int, question: dict[str, Any]) -> str:
    options = list(question.get("options") or [])
    parts = [f"Question {number} of {total}. {question.get('question', '')}"]
    for index, option in enumerate(options):
        label = _OPTION_LABELS[index] if index < len(_OPTION_LABELS) else str(index + 1)
        parts.append(f"Option {label}: {option}.")
    return " ".join(parts)


def _letter_index(text: str) -> int | None:
    """Map an answer like "B", "option c" or "3" to a 0-based index."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return None
    match = _LETTER_INLINE_RE.search(lowered)
    if match:
        return ord(match.group(1)) - 97
    match = _LETTER_ALONE_RE.match(lowered)
    if match:
        return ord(match.group(1)) - 97
    # "the answer is b"
    match = _LETTER_SPEECH_RE.search(lowered)
    if match:
        return ord(match.group(1)) - 97
    match = _NUMBER_RE.search(lowered)
    if match:
        return int(match.group(1)) - 1
    return None


def _best_option(response: str, options: Sequence[str]) -> tuple[int | None, float]:
    """Fraction of the response's words present in each option, best first."""
    words = set(_WORD_RE.findall(response.lower()))
    if not words or not options:
        return None, 0.0
    best_index: int | None = None
    best_score = 0.0
    for index, option in enumerate(options):
        option_words = set(_WORD_RE.findall(str(option).lower()))
        if not option_words:
            continue
        overlap = len(words & option_words) / len(words)
        # A spoken option is often a substring of the full option text.
        if str(option).lower() in response.lower():
            overlap = max(overlap, 0.95)
        if overlap > best_score:
            best_index, best_score = index, overlap
    return best_index, best_score


def _option_text(question: dict[str, Any], index: int | None) -> str:
    if index is None:
        return ""
    options = list(question.get("options") or [])
    if 0 <= index < len(options):
        return str(options[index])
    return ""


def _feedback(question: dict[str, Any], chosen: int | None, is_correct: bool) -> str:
    explanation = str(question.get("explanation") or "").strip()
    if is_correct:
        return "Correct. " + (explanation or "Well done.")
    expected = int(question.get("answer_index", 0))
    label = _OPTION_LABELS[expected] if expected < len(_OPTION_LABELS) else str(expected + 1)
    answer = _option_text(question, expected)
    prefix = "Not quite." if chosen is not None else "I did not catch an answer."
    return f"{prefix} The answer is {label}: {answer}. {explanation}".strip()


def _safe_call(callback: Callable[..., Any], *args: Any) -> None:
    try:
        callback(*args)
    except Exception:
        logger.warning("quiz callback raised", exc_info=True)


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

_default: QuizGenerator | None = None
_default_lock = threading.Lock()


def get_quiz_generator() -> QuizGenerator:
    """Process-wide generator, built once with a fresh LLM manager and RAG."""
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                from core.llm_manager import LLMServerManager
                from modules.university.rag import get_university_rag

                _default = QuizGenerator(LLMServerManager(), get_university_rag())
    return _default


def generate_quiz(
    topic: str,
    n_questions: int = 5,
    difficulty: str = "medium",
    doc_filter: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    return get_quiz_generator().generate_quiz(topic, n_questions, difficulty, doc_filter)


def generate_flashcards(
    topic: str, n_cards: int = 10, doc_filter: str | Sequence[str] | None = None
) -> dict[str, Any]:
    return get_quiz_generator().generate_flashcards(topic, n_cards, doc_filter)


def run_interactive_quiz(quiz: dict[str, Any], voice: Any = None) -> dict[str, Any]:
    return get_quiz_generator().run_interactive_quiz(quiz, voice)


__all__ = [
    "QuizGenerator",
    "generate_flashcards",
    "generate_quiz",
    "get_quiz_generator",
    "run_interactive_quiz",
]
