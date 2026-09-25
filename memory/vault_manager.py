"""
Atlas — the markdown vault.

Everything Atlas learns or produces is also written to ``vault/`` as ordinary
markdown. The brief is explicit that these files must stay human-readable and
Obsidian-compatible, so this module is a writer of *notes*, not of a database
dump: YAML frontmatter, real headings, links a person can click. The vector
index is a derived copy — delete ``chroma_db/`` and ``index_vault()`` rebuilds
it from these files.

Layout
------

=================  ==========================================================
``conversations/``  one file per exchange, indexed by its summary
``memories/``       durable facts and preferences, indexed in full
``skills/``         documentation for a skill, next to its Python file
``research/``       research reports with their sources cited
``agent_notes/``    anything else the assistant wants to leave itself
``briefings/``      scheduled news/weather briefings
=================  ==========================================================

The folders already exist on disk; this module creates any that are missing.

Two vocabularies, on purpose
----------------------------

The frontmatter ``type`` is written for a person reading the vault in Obsidian
— ``memory``, ``conversation``, ``skill``, ``research``. The vector index needs
something narrower to filter on, so it records the *most specific* type
available:

* a memory note stores its ``memory_type`` (``preference``, ``fact``, ...), so
  ``store.list_by_type("preference")`` finds it rather than returning every
  memory lumped together;
* every other note stores its frontmatter ``type`` unchanged.

The frontmatter ``type`` is always kept alongside as ``vault_type`` in the
index, so either filter works.

Search returns whole notes, not chunks
--------------------------------------

Notes are chunked (by heading, then by paragraph) before being embedded,
because a 4,000-word research report embedded as one vector matches nothing
well. :meth:`VaultManager.find_note` therefore searches over chunks, takes the
best-scoring one, and reads the **whole file** back off disk — the score comes
from the matching chunk, the content is the complete note. This means a hit is
only as good as its best chunk, which is the behaviour a person expects.

Editing a note replaces its chunks: re-indexing deletes every record for that
``source`` path first. Content-derived ids make a same-length edit an upsert,
but without the delete a shortened note would leave orphaned chunks behind.

Usage::

    from memory.vault_manager import get_vault_manager

    vault = get_vault_manager()

    vault.write_conversation(
        "what's the weather like?",
        "Overcast and 22 degrees.",
        "Asked about the weather; overcast, 22C.",
    )
    vault.write_memory("Favourite coffee", "A flat white from Bridge Street.",
                       memory_type="preference")

    path, note, score = vault.find_note("what coffee do they like?")
    vault.index_vault()          # called at startup by the daemon
    vault.delete_note(path)

Run the built-in check (writes to a temporary vault, never the real one) with::

    python3 -m memory.vault_manager
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import frontmatter

from core.config import (
    AGENT_NAME,
    MEMORY_RELEVANCE_THRESHOLD,
    VAULT_DIR,
)
from memory.chroma_store import (
    CONVERSATION_COLLECTION,
    META_SOURCE,
    META_TAGS,
    META_TIMESTAMP,
    META_TYPE,
    RECORD_TYPE_CONVERSATION,
    RECORD_TYPE_FACT,
    RECORD_TYPE_NOTE,
    SEMANTIC_COLLECTION,
    ChromaStore,
    get_chroma_store,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Note",
    "VaultManager",
    "get_vault_manager",
    "reset_vault_manager",
    "safe_filename",
    "CONVERSATION_DIR",
    "MEMORY_DIR",
    "SKILL_DIR",
    "RESEARCH_DIR",
    "NOTE_DIR",
    "BRIEFING_DIR",
]

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------

CONVERSATION_DIR = "conversations"
MEMORY_DIR = "memories"
SKILL_DIR = "skills"
RESEARCH_DIR = "research"
NOTE_DIR = "agent_notes"
BRIEFING_DIR = "briefings"

VAULT_FOLDERS: tuple[str, ...] = (
    CONVERSATION_DIR,
    MEMORY_DIR,
    SKILL_DIR,
    RESEARCH_DIR,
    NOTE_DIR,
    BRIEFING_DIR,
)

#: Frontmatter ``type`` written for each folder (the human-facing vocabulary).
FOLDER_TYPES: dict[str, str] = {
    CONVERSATION_DIR: RECORD_TYPE_CONVERSATION,
    MEMORY_DIR: "memory",
    SKILL_DIR: "skill",
    RESEARCH_DIR: "research",
    NOTE_DIR: RECORD_TYPE_NOTE,
    BRIEFING_DIR: "briefing",
}

#: Which Chroma collection a note of each frontmatter type belongs in.
#: Only conversation summaries live in their own collection per the brief.
_COLLECTION_BY_TYPE: dict[str, str] = {
    RECORD_TYPE_CONVERSATION: CONVERSATION_COLLECTION,
}

#: Record types the index understands directly. A memory's ``memory_type`` is
#: used verbatim when it appears here, so ``preference`` and ``fact`` notes stay
#: distinguishable in the store.
KNOWN_MEMORY_TYPES: tuple[str, ...] = (
    "fact",
    "preference",
    "skill",
    "document",
    "note",
    "project",
    "goal",
    "habit",
    "contact",
    "event",
    "health",
    "finance",
    "location",
    "opinion",
)

DEFAULT_MEMORY_TYPE = RECORD_TYPE_FACT

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

#: Chunks bigger than this lose specificity; smaller ones lose context. ~1400
#: characters is a few paragraphs — comfortably inside nomic's window and small
#: enough that one chunk usually covers one idea.
MAX_CHUNK_CHARS = 1400

#: A trailing fragment shorter than this is folded into the previous chunk
#: rather than stored as a near-meaningless vector of its own.
MIN_CHUNK_CHARS = 120

#: Used when a paragraph has no space to break on.
_HARD_SPLIT_FLOOR = 400

#: ``find_note`` default relevance floor. Higher than
#: ``MEMORY_RELEVANCE_THRESHOLD`` (0.30) on purpose: that one is tuned for
#: recall — better to over-remember than to forget — whereas handing someone the
#: wrong note as though it were the right one is worse than saying nothing.
#: Floored against the configured value so raising the config still takes effect.
DEFAULT_NOTE_THRESHOLD = max(0.45, MEMORY_RELEVANCE_THRESHOLD)

#: Longest filename stem written to disk, before the extension.
MAX_FILENAME_CHARS = 80

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")

# Characters that are awkward in a filename on at least one of Linux, macOS or
# Windows, or that Obsidian treats specially. Everything else is kept.
_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\[\]#^]+')

# Windows refuses to create files with these stems, whatever the extension.
_RESERVED_STEMS = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Note:
    """A markdown file on disk, frontmatter parsed."""

    path: Path
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)
    mtime: float = 0.0

    @property
    def title(self) -> str:
        return str(self.metadata.get("title") or self.path.stem)

    @property
    def type(self) -> str:
        return str(self.metadata.get("type") or "")

    @property
    def text(self) -> str:
        """The frontmatter and body re-joined, for anything that wants the file."""
        return frontmatter.dumps(
            frontmatter.Post(self.body, **self.metadata)
        )

    @property
    def tags(self) -> list[str]:
        value = self.metadata.get("tags") or []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @property
    def summary(self) -> str:
        return str(self.metadata.get("summary") or "")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Note {self.type or '?'} {self.path.name!r}>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def safe_filename(title: str, *, fallback: str = "untitled") -> str:
    """Turn a title into a filename stem that is safe on every platform.

    Path separators and the handful of characters Windows refuses are replaced
    with dashes rather than dropped, so ``"Q3: profit/loss"`` becomes
    ``"Q3- profit-loss"`` and stays readable instead of collapsing into mush.
    Leading dots are stripped so a title cannot produce a hidden file, and the
    Windows reserved device names get a suffix.
    """
    text = unicodedata.normalize("NFKC", str(title or "")).strip()
    text = _UNSAFE_FILENAME_CHARS.sub("-", text)
    # Collapse runs of dashes/whitespace but keep single spaces: "My  Note"
    # reads better as "My Note", and Obsidian is happy with spaces.
    text = re.sub(r"[\s]+", " ", text)
    text = re.sub(r"-{2,}", "-", text)
    text = text.strip(" .-_")
    text = text[:MAX_FILENAME_CHARS].strip(" .-_")

    if not text:
        return fallback
    if text.lower() in _RESERVED_STEMS:
        return f"{text}-note"
    return text


def _now_local() -> datetime:
    return datetime.now().astimezone().replace(microsecond=0)


def _iso(moment: datetime | None = None) -> str:
    """ISO-8601 with an offset, which Obsidian and Dataview both understand."""
    return (moment or _now_local()).isoformat()


def _normalise_sources(sources: Any) -> list[tuple[str, str]]:
    """Flatten ``sources`` into ``(title, url)`` pairs.

    Accepts plain URLs, ``{"url": ..., "title": ...}`` mappings, or anything
    with a ``.url`` attribute — a search module's result object, most likely —
    so the caller does not have to reshape what it already has.
    """
    if sources is None:
        return []
    if isinstance(sources, (str, bytes)):
        sources = [sources]

    pairs: list[tuple[str, str]] = []
    for entry in sources:
        if isinstance(entry, Mapping):
            url = str(entry.get("url") or entry.get("link") or entry.get("href") or "")
            title = str(entry.get("title") or entry.get("name") or "")
        elif isinstance(entry, str):
            # Bare strings are checked first: ``getattr(entry, "title")`` on a
            # str finds ``str.title``, the built-in method, and would happily
            # write "<built-in method title of str object>" into the report.
            url, title = entry, ""
        else:
            url = str(getattr(entry, "url", "") or "")
            title = str(getattr(entry, "title", "") or "")
            if not url:
                url = str(entry)
        url = url.strip()
        if not url:
            continue
        pairs.append((title.strip(), url))
    return pairs


def _paragraphs(text: str) -> list[str]:
    """Split markdown into paragraph-ish blocks, keeping list items together."""
    blocks: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        cleaned = block.strip()
        if cleaned:
            blocks.append(cleaned)
    return blocks


def _hard_split(text: str, size: int) -> list[str]:
    """Break a single oversized block, preferring sentence then word boundaries."""
    pieces: list[str] = []
    remaining = text.strip()
    while len(remaining) > size:
        cut = remaining.rfind(". ", 0, size)
        if cut < size // 2:
            cut = remaining.rfind(" ", 0, size)
        if cut <= 0:
            cut = size
        pieces.append(remaining[: cut + 1].strip())
        remaining = remaining[cut + 1 :].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _pack(blocks: Sequence[str], budget: int) -> list[str]:
    """Greedily pack blocks into strings no longer than ``budget``."""
    packed: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > budget:
            if current:
                packed.append(current)
                current = ""
            packed.extend(_hard_split(block, budget))
            continue
        if not current:
            current = block
        elif len(current) + len(block) + 2 <= budget:
            current = f"{current}\n\n{block}"
        else:
            packed.append(current)
            current = block
    if current:
        packed.append(current)
    return packed


def _sections(body: str) -> Iterator[tuple[str, str]]:
    """Yield ``(heading_line, section_text)`` pairs in document order.

    Heading text is yielded with its section so a chunk can carry the heading it
    belongs to — a paragraph four pages into "## Risks" means something quite
    different without that line in front of it.
    """
    heading = ""
    buffer: list[str] = []
    for line in body.splitlines():
        if _HEADING.match(line):
            yield heading, "\n".join(buffer).strip()
            heading, buffer = line.strip(), []
        else:
            buffer.append(line)
    yield heading, "\n".join(buffer).strip()


def chunk_markdown(
    body: str,
    *,
    max_chars: int = MAX_CHUNK_CHARS,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[str]:
    """Split a markdown body into embeddable chunks.

    Sections are cut at headings, then packed by paragraph, and every chunk is
    prefixed with its heading so it can be understood on its own. Fragments
    below ``min_chars`` are merged forward so the index is not padded out with
    stubs.
    """
    text = (body or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    for heading, section in _sections(text):
        if not section:
            continue
        prefix = f"{heading}\n\n" if heading else ""
        budget = max(max_chars - len(prefix), _HARD_SPLIT_FLOOR)
        for block in _pack(_paragraphs(section), budget):
            pieces.append(f"{prefix}{block}".strip())

    merged: list[str] = []
    for piece in pieces:
        if (
            merged
            and len(piece) < min_chars
            and len(merged[-1]) + len(piece) + 2 <= max_chars
        ):
            merged[-1] = f"{merged[-1]}\n\n{piece}"
        else:
            merged.append(piece)
    return [piece for piece in merged if piece.strip()]


# ---------------------------------------------------------------------------
# The manager
# ---------------------------------------------------------------------------


class VaultManager:
    """Reads and writes the markdown vault, keeping the vector index in step.

    Every ``write_*`` method does the same three things: render a note with
    frontmatter, write it to disk, and replace that note's records in
    :mod:`memory.chroma_store`. Writing is deliberately independent of
    indexing — if the store is unavailable the note still lands on disk, because
    the file is the durable artefact and the index is a rebuildable cache.
    """

    def __init__(
        self,
        vault_dir: str | Path | None = None,
        *,
        store: ChromaStore | None = None,
        index: bool = True,
        **store_options: Any,
    ) -> None:
        """Bind to a vault directory and (lazily) to the vector store.

        ``store`` takes an existing :class:`~memory.chroma_store.ChromaStore`
        — tests pass one on a temporary directory. Otherwise the process-wide
        instance is used, built on first index or search.

        ``index=False`` makes every write disk-only, which is the right choice
        for bulk imports that will call :meth:`index_vault` at the end.
        """
        self.vault_dir = Path(vault_dir or VAULT_DIR)
        self._store = store
        self._store_options = store_options
        self.index_enabled = index
        self._lock = threading.RLock()
        self._ensure_folders()

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------

    def _ensure_folders(self) -> None:
        """Create the vault skeleton. Idempotent, so it is safe on every start."""
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        for name in VAULT_FOLDERS:
            (self.vault_dir / name).mkdir(exist_ok=True)

    @property
    def store(self) -> ChromaStore:
        """The vector store, resolved on first use."""
        if self._store is None:
            self._store = (
                get_chroma_store(**self._store_options)
                if self._store_options
                else get_chroma_store()
            )
        return self._store

    def folder(self, name: str) -> Path:
        """Absolute path of a vault subfolder, created if it went missing."""
        path = self.vault_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __repr__(self) -> str:
        return (
            f"<VaultManager dir={self.vault_dir} "
            f"index={'on' if self.index_enabled else 'off'}>"
        )

    def _write_note(self, folder: str, stem: str, post: frontmatter.Post) -> Path:
        """Serialise ``post`` to ``folder/stem.md`` atomically.

        Written to a temporary file and moved into place: a crash mid-write
        leaves the previous version of the note intact rather than a truncated
        file, which matters because these files are the user's own notes.
        """
        directory = self.folder(folder)
        target = directory / f"{stem}.md"
        text = frontmatter.dumps(post)
        if not text.endswith("\n"):
            text += "\n"

        temporary = directory / f".{stem}.md.tmp"
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, target)
        return target

    def _unique_stem(self, folder: str, title: str, *, upsert: bool) -> tuple[str, bool]:
        """Pick a filename stem for ``title``.

        Returns ``(stem, replacing)``. When ``upsert`` is set, an existing note
        with the same ``title`` frontmatter is reused so the caller's memory is
        updated in place. A file whose title differs is left alone and the new
        note gets a numeric suffix — two different notes must not silently
        overwrite one another just because their titles sanitise the same way.
        """
        stem = safe_filename(title)
        directory = self.folder(folder)
        candidate = stem
        counter = 2
        while True:
            path = directory / f"{candidate}.md"
            if not path.exists():
                return candidate, False
            if upsert:
                existing_title = _read_title(path)
                # A file with no frontmatter at all is treated as a match: it
                # was almost certainly written by an older version of this code.
                if existing_title is None or existing_title == title:
                    return candidate, True
            candidate = f"{stem} {counter}"
            counter += 1

    def read_note(self, path: str | Path) -> Note:
        """Load and parse one note. Raises ``FileNotFoundError`` if it is gone."""
        target = Path(path)
        post = frontmatter.load(target)
        return Note(
            path=target,
            body=post.content,
            metadata=dict(post.metadata),
            mtime=target.stat().st_mtime,
        )

    def _resolve(self, path: str | Path) -> Path:
        """Absolute path, refusing anything outside the vault.

        The guard exists because ``delete_note`` unlinks a file: a caller
        passing a path derived from model output must not be able to delete
        something else on the machine.
        """
        target = Path(path)
        if not target.is_absolute():
            target = self.vault_dir / target
        resolved = target.resolve()
        vault = self.vault_dir.resolve()
        if resolved != vault and vault not in resolved.parents:
            raise ValueError(f"{target} is outside the vault ({vault})")
        return resolved

    # ------------------------------------------------------------------
    # Writing notes
    # ------------------------------------------------------------------

    def write_conversation(
        self,
        user_input: str,
        response: str,
        summary: str,
        *,
        moment: datetime | None = None,
        tags: Sequence[str] | None = None,
        index: bool | None = None,
    ) -> Path:
        """Record one exchange as a timestamped note in ``conversations/``.

        The file keeps the full transcript, but only ``summary`` is embedded.
        Full exchanges are mostly small talk and pronouns; embedding them
        produces vectors that match everything and therefore nothing. The
        summary is the part with retrievable content in it.

        The filename is timestamped to the second, with a numeric suffix if two
        exchanges land in the same second.
        """
        stamp = moment or _now_local()
        title = f"Conversation {stamp.strftime('%Y-%m-%d %H:%M:%S')}"
        summary = (summary or "").strip() or _fallback_summary(user_input, response)

        body = "\n\n".join(
            [
                "## What was said",
                f"**{AGENT_NAME} heard**\n\n{user_input.strip()}",
                f"**{AGENT_NAME} replied**\n\n{response.strip()}",
                "## Summary",
                summary,
            ]
        )
        metadata: dict[str, Any] = {
            "type": RECORD_TYPE_CONVERSATION,
            "title": title,
            "date": stamp.date(),
            "time": stamp.strftime("%H:%M:%S"),
            "created": _iso(stamp),
            "summary": summary,
            "tags": list(tags or ["conversation"]),
        }

        with self._lock:
            stem = _timestamped_stem(self.folder(CONVERSATION_DIR), stamp)
            path = self._write_note(
                CONVERSATION_DIR, stem, frontmatter.Post(body, **metadata)
            )

        logger.info("wrote conversation note %s", path.name)
        # The summary is the document, so the note indexes to exactly one record.
        self._index_note(path, document=summary, note_type=RECORD_TYPE_CONVERSATION, metadata=metadata)
        return path

    def write_memory(
        self,
        title: str,
        content: str,
        memory_type: str = DEFAULT_MEMORY_TYPE,
        *,
        tags: Sequence[str] | None = None,
        source: str | None = None,
        moment: datetime | None = None,
    ) -> Path:
        """Write a durable fact to ``memories/``, replacing one with the same title.

        Indexing covers the **full content**, unlike conversations: a memory is
        short and every sentence in it is the point.

        Re-writing the same title updates the existing note and preserves its
        original ``created`` timestamp, so the vault accumulates a history of
        what is known rather than a pile of near-duplicates.
        """
        clean_type = str(memory_type or DEFAULT_MEMORY_TYPE).strip().lower()
        if clean_type not in KNOWN_MEMORY_TYPES:
            logger.debug("unknown memory_type %r — recording it anyway", clean_type)

        stamp = moment or _now_local()
        body = (content or "").strip()
        if not body:
            raise ValueError("write_memory() needs some content")

        with self._lock:
            stem, replacing = self._unique_stem(MEMORY_DIR, title, upsert=True)
            path = self.folder(MEMORY_DIR) / f"{stem}.md"
            created = _iso(stamp)
            if replacing and path.exists():
                # Keep the first-seen timestamp: that is when Atlas learnt it.
                created = str(_read_metadata(path).get("created") or created)

            metadata: dict[str, Any] = {
                "type": "memory",
                "title": title,
                "memory_type": clean_type,
                "created": created,
                "updated": _iso(stamp),
                "tags": list(tags or [clean_type]),
            }
            if source:
                metadata["source"] = source
            path = self._write_note(
                MEMORY_DIR, stem, frontmatter.Post(body, **metadata)
            )

        logger.info("%s memory note %s", "updated" if replacing else "wrote", path.name)
        self._index_note(path, document=body, note_type=clean_type, metadata=metadata)
        return path

    def write_skill_note(
        self,
        skill_name: str,
        description: str,
        filepath: str | Path,
        *,
        parameters: Mapping[str, str] | None = None,
        example: str | None = None,
        tags: Sequence[str] | None = None,
        moment: datetime | None = None,
    ) -> Path:
        """Document a skill in ``skills/``, separate from its Python file.

        The generated script is code; this is the note a person reads months
        later to remember what the thing does and where it lives. It also gives
        the skill a vector in the index, so "what skills do I have for
        converting files" is answerable.
        """
        name = str(skill_name or "").strip()
        if not name:
            raise ValueError("write_skill_note() needs a skill name")
        location = str(filepath)

        sections = [f"# {name}", (description or "").strip() or "_No description yet._"]

        if parameters:
            table = ["## Parameters", "", "| Parameter | Meaning |", "| --- | --- |"]
            table += [f"| `{key}` | {value} |" for key, value in parameters.items()]
            sections.append("\n".join(table))

        if example:
            sections.append(f"## Example\n\n```\n{example.strip()}\n```")

        sections.append(f"## Implementation\n\n`{location}`")
        body = "\n\n".join(sections)

        stamp = moment or _now_local()
        metadata: dict[str, Any] = {
            "type": "skill",
            "title": name,
            "skill_name": name,
            "skill_file": location,
            "created": _iso(stamp),
            "updated": _iso(stamp),
            "tags": list(tags or ["skill"]),
        }
        if parameters:
            metadata["parameters"] = [str(key) for key in parameters]

        with self._lock:
            stem, replacing = self._unique_stem(SKILL_DIR, name, upsert=True)
            path = self.folder(SKILL_DIR) / f"{stem}.md"
            if replacing and path.exists():
                metadata["created"] = str(
                    _read_metadata(path).get("created") or metadata["created"]
                )
            path = self._write_note(
                SKILL_DIR, stem, frontmatter.Post(body, **metadata)
            )

        logger.info("%s skill note %s", "updated" if replacing else "wrote", path.name)
        self._index_note(path, document=body, note_type="skill", metadata=metadata)
        return path

    def write_research_report(
        self,
        title: str,
        content: str,
        sources: Any = None,
        *,
        question: str | None = None,
        tags: Sequence[str] | None = None,
        moment: datetime | None = None,
    ) -> Path:
        """Write a research report to ``research/`` with its sources cited.

        Every URL is recorded in the frontmatter ``sources`` list *and* listed
        in the body, because a report whose citations only exist in metadata is
        one Obsidian search away from being unverifiable.
        """
        pairs = _normalise_sources(sources)
        stamp = moment or _now_local()

        sections: list[str] = [f"# {title}"]
        if question:
            sections.append(f"> **Question.** {question.strip()}")
        sections.append((content or "").strip() or "_No findings recorded._")

        if pairs:
            listing = ["## Sources", ""]
            for index, (source_title, url) in enumerate(pairs, start=1):
                label = source_title or url
                listing.append(
                    f"{index}. [{label}]({url})"
                    if source_title
                    else f"{index}. <{url}>"
                )
            sections.append("\n".join(listing))

        body = "\n\n".join(sections)
        metadata: dict[str, Any] = {
            "type": "research",
            "title": title,
            "date": stamp.date(),
            "created": _iso(stamp),
            "sources": [url for _, url in pairs],
            "source_count": len(pairs),
            "tags": list(tags or ["research"]),
        }
        if question:
            metadata["question"] = question.strip()

        with self._lock:
            stem, _ = self._unique_stem(RESEARCH_DIR, title, upsert=False)
            path = self._write_note(
                RESEARCH_DIR, stem, frontmatter.Post(body, **metadata)
            )

        logger.info("wrote research report %s (%d sources)", path.name, len(pairs))
        self._index_note(path, document=body, note_type="research", metadata=metadata)
        return path

    def write_note(
        self,
        title: str,
        content: str,
        *,
        folder: str = NOTE_DIR,
        note_type: str | None = None,
        tags: Sequence[str] | None = None,
        upsert: bool = True,
        moment: datetime | None = None,
    ) -> Path:
        """General-purpose writer for anything without a dedicated method.

        Used for briefings and the agent's own scratch notes. ``folder`` is
        treated as a name inside the vault — a path that escapes it is refused.
        """
        if folder not in VAULT_FOLDERS:
            candidate = Path(folder)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"folder must be a name inside the vault: {folder!r}")
            folder = str(candidate).strip("/\\")
            if not folder:
                raise ValueError("folder must not be empty")

        stamp = moment or _now_local()
        kind = note_type or FOLDER_TYPES.get(folder, RECORD_TYPE_NOTE)
        body = (content or "").strip()
        if not body:
            raise ValueError("write_note() needs some content")

        metadata: dict[str, Any] = {
            "type": kind,
            "title": title,
            "created": _iso(stamp),
            "tags": list(tags or []),
        }

        with self._lock:
            stem, _ = self._unique_stem(folder, title, upsert=upsert)
            path = self._write_note(
                folder, stem, frontmatter.Post(body, **metadata)
            )

        logger.debug("wrote note %s", path.name)
        self._index_note(path, document=body, note_type=kind, metadata=metadata)
        return path

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _index_note(
        self,
        path: Path,
        *,
        document: str,
        note_type: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Replace this note's records in the vector store.

        Chunks are replaced wholesale rather than upserted, so an edit that
        makes a note shorter cannot leave stale chunks behind pointing at text
        that no longer exists. Failures are logged and swallowed: the markdown
        has already been written, and an unavailable index must not lose the
        user's note.
        """
        if not self.index_enabled:
            return 0
        text = (document or "").strip()
        if not text:
            return 0

        chunks = chunk_markdown(text)
        if not chunks:
            return 0

        collection = _COLLECTION_BY_TYPE.get(note_type, SEMANTIC_COLLECTION)
        source = str(path)
        meta = dict(metadata or {})

        try:
            with self._lock:
                # Remove first: the previous chunk count is unknown.
                self.store.delete(source, collection)
                self.store.add_many(
                    chunks,
                    [self._chunk_metadata(meta, note_type, source, index, len(chunks))
                     for index in range(len(chunks))],
                    collection,
                )
        except Exception:
            logger.exception("could not index %s — the note is still on disk", path)
            return 0
        return len(chunks)

    @staticmethod
    def _chunk_metadata(
        metadata: Mapping[str, Any],
        note_type: str,
        source: str,
        index: int,
        total: int,
    ) -> dict[str, Any]:
        """Metadata for one chunk of a note.

        ``type`` carries the specific type so ``list_by_type("preference")``
        works; ``vault_type`` keeps the frontmatter spelling for callers that
        think in the human vocabulary.
        """
        tags = metadata.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]

        chunk: dict[str, Any] = {
            META_TYPE: note_type,
            META_SOURCE: source,
            META_TAGS: [str(tag) for tag in tags] or None,
            META_TIMESTAMP: time.time(),
            "title": str(metadata.get("title") or Path(source).stem),
            "vault_type": str(metadata.get("type") or note_type),
            "chunk_index": index,
            "chunk_count": total,
        }
        # Carried through because they are worth having in a search result.
        # ``mtime_ns`` matters most: index_vault() compares it to the file on
        # disk to decide whether a note needs re-embedding, so dropping it here
        # would silently re-index the whole vault on every start-up.
        for key in (
            "mtime_ns",
            "memory_type",
            "summary",
            "date",
            "question",
            "source_count",
        ):
            value = metadata.get(key)
            if value is None or isinstance(value, (list, dict)):
                continue
            if isinstance(value, (date, datetime)):
                value = value.isoformat()
            chunk[key] = value
        return chunk

    def index_vault(
        self,
        *,
        force: bool = False,
        prune: bool = True,
    ) -> dict[str, int]:
        """Bring the vector index in line with the files on disk.

        Called at startup. A note is indexed when it has no records yet, when
        ``force`` is set, or when its modification time has moved on since it
        was last indexed — so editing a note in Obsidian by hand is picked up
        without any extra bookkeeping.

        ``prune`` deletes records whose file has been removed. Only sources
        inside this vault are considered, so records from elsewhere are never
        touched.

        Returns a report: ``indexed``, ``unchanged``, ``pruned``, ``chunks``,
        ``errors``.
        """
        report = {"indexed": 0, "unchanged": 0, "pruned": 0, "chunks": 0, "errors": 0}
        try:
            known = self._indexed_notes()
        except Exception:
            logger.exception("could not read the index — skipping index_vault()")
            report["errors"] += 1
            return report

        for path in self._vault_files():
            source = str(path)
            try:
                # Integer nanoseconds, not a float: a float mtime does not
                # survive Chroma's serialisation exactly (1790359551.4070895
                # comes back as ...4070897), and one ULP in the wrong direction
                # is enough to re-index that note on every single start-up.
                mtime_ns = path.stat().st_mtime_ns
            except OSError:
                continue

            records = known.get(source)
            if records is not None and not force:
                recorded = records.get("mtime_ns")
                if recorded is not None and int(recorded) >= mtime_ns:
                    report["unchanged"] += 1
                    continue

            try:
                note = self.read_note(path)
            except Exception:
                logger.warning("could not parse %s — skipping", path, exc_info=True)
                report["errors"] += 1
                continue

            note_type = self._note_type(note, path)
            # A conversation's summary is its document, matching write_conversation.
            document = note.summary or note.body
            note.metadata["mtime_ns"] = mtime_ns
            written = self._index_note(
                path, document=document, note_type=note_type, metadata=note.metadata
            )
            if written:
                report["indexed"] += 1
                report["chunks"] += written
            else:
                report["errors"] += 1

        if prune:
            report["pruned"] = self._prune_missing(known)

        logger.info(
            "vault index: %d indexed (%d chunks), %d unchanged, %d pruned, %d errors",
            report["indexed"],
            report["chunks"],
            report["unchanged"],
            report["pruned"],
            report["errors"],
        )
        return report

    def _vault_files(self) -> list[Path]:
        """Every ``.md`` file in the vault, skipping Obsidian's own state."""
        found: list[Path] = []
        for path in sorted(self.vault_dir.rglob("*.md")):
            relative = path.relative_to(self.vault_dir)
            if any(part.startswith(".") for part in relative.parts):
                continue  # .obsidian/, .trash/, and our own .tmp files
            if path.is_file():
                found.append(path)
        return found

    @staticmethod
    def _note_type(note: Note, path: Path) -> str:
        """Decide which index type a note gets.

        The most specific label wins: a memory stores its ``memory_type`` so
        preferences and facts stay separable, everything else uses its
        frontmatter ``type``, and a file with no frontmatter falls back to the
        folder it lives in.
        """
        declared = note.type.lower()
        if declared == "memory":
            kind = str(note.metadata.get("memory_type") or DEFAULT_MEMORY_TYPE).lower()
            return kind or DEFAULT_MEMORY_TYPE
        if declared:
            return declared
        return FOLDER_TYPES.get(path.parent.name, RECORD_TYPE_NOTE)

    def _indexed_notes(self) -> dict[str, dict[str, Any]]:
        """Map ``source path -> metadata`` for everything currently indexed.

        Chunks of one note collapse into one entry; ``mtime`` is the maximum
        seen across its chunks, and ``chunk_count`` comes from the first.
        """
        known: dict[str, dict[str, Any]] = {}
        for collection in (SEMANTIC_COLLECTION, CONVERSATION_COLLECTION):
            try:
                records = self.store.list_by_type(None, collection)
            except Exception:
                logger.warning("could not list %s", collection, exc_info=True)
                continue
            for record in records:
                source = str(record.get("metadata", {}).get(META_SOURCE) or "")
                if not source:
                    continue
                entry = known.setdefault(source, {"mtime_ns": None, "chunk_count": 0})
                entry["chunk_count"] = max(
                    entry["chunk_count"], int(record["metadata"].get("chunk_count") or 1)
                )
                recorded = record["metadata"].get("mtime_ns")
                if recorded is not None:
                    entry["mtime_ns"] = max(entry["mtime_ns"] or 0, int(recorded))
        return known

    def _prune_missing(self, known: Mapping[str, Mapping[str, Any]]) -> int:
        """Delete records whose file is gone. Returns how many notes were cleared.

        Restricted to sources inside this vault: a record whose ``source`` is a
        URL or a file elsewhere on the machine is none of this method's
        business.
        """
        vault = self.vault_dir.resolve()
        removed = 0
        for source in known:
            path = Path(source)
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if vault not in resolved.parents:
                continue
            if resolved.exists():
                continue
            try:
                if self.store.delete(source):
                    removed += 1
                    logger.info("pruned index records for missing file %s", resolved.name)
            except Exception:
                logger.warning("could not prune %s", source, exc_info=True)
        return removed

    # ------------------------------------------------------------------
    # Finding and removing
    # ------------------------------------------------------------------

    def find_note(
        self,
        query: str,
        threshold: float | None = None,
        *,
        collection: str | None = None,
        k: int = 8,
        type_filter: str | Sequence[str] | None = None,
    ) -> tuple[Path | None, str | None, float]:
        """Find the vault note that best answers ``query``.

        Returns ``(path, content, score)``, or ``(None, None, 0.0)`` when
        nothing clears ``threshold``. ``score`` is the cosine similarity of the
        best-matching chunk while ``content`` is the complete note read from
        disk, so the caller can show whole context for a chunk-level match.

        ``threshold`` defaults to :data:`DEFAULT_NOTE_THRESHOLD`. Hits whose
        file has since been deleted are skipped rather than returned, so a
        stale index cannot hand back a path that does not exist.
        """
        cleaned = (query or "").strip()
        if not cleaned:
            return None, None, 0.0

        floor = DEFAULT_NOTE_THRESHOLD if threshold is None else float(threshold)
        try:
            hits = self.store.search(
                cleaned,
                k=k,
                collection=collection or SEMANTIC_COLLECTION,
                type_filter=type_filter,
                min_score=floor,
            )
        except Exception:
            logger.exception("vault search failed")
            return None, None, 0.0

        for hit in hits:
            source = hit.metadata.get(META_SOURCE) or hit.metadata.get("vault_path")
            if not source:
                continue
            try:
                path = Path(str(source))
                if not path.is_file():
                    logger.debug("skipping stale index entry %s", source)
                    continue
                return path, path.read_text(encoding="utf-8"), hit.score
            except OSError:
                logger.warning("could not read %s", source, exc_info=True)
                continue
        return None, None, 0.0

    def search_notes(
        self,
        query: str,
        *,
        k: int = 5,
        threshold: float | None = None,
        collection: str | None = None,
        type_filter: str | Sequence[str] | None = None,
    ) -> list[tuple[Path, float, str]]:
        """Rank notes for ``query``, returning ``(path, score, chunk)`` each.

        The lighter companion to :meth:`find_note`: it does not read the files
        back, and it de-duplicates so one long note cannot fill every slot with
        its own chunks.
        """
        cleaned = (query or "").strip()
        if not cleaned:
            return []
        try:
            hits = self.store.search(
                cleaned,
                k=max(k * 4, k),
                collection=collection or SEMANTIC_COLLECTION,
                type_filter=type_filter,
                min_score=DEFAULT_NOTE_THRESHOLD if threshold is None else float(threshold),
            )
        except Exception:
            logger.exception("vault search failed")
            return []

        ranked: list[tuple[Path, float, str]] = []
        seen: set[str] = set()
        for hit in hits:
            source = str(hit.metadata.get(META_SOURCE) or "")
            if not source or source in seen:
                continue
            path = Path(source)
            if not path.is_file():
                continue
            seen.add(source)
            ranked.append((path, hit.score, hit.text))
            if len(ranked) >= k:
                break
        return ranked

    def delete_note(self, filepath: str | Path) -> bool:
        """Remove a note from disk and drop its records from the index.

        Returns ``True`` when the index was updated. A file that does not exist
        still has its records cleared — that is exactly the stale case
        :meth:`index_vault` would otherwise have to prune.
        """
        try:
            target = self._resolve(filepath)
        except ValueError:
            logger.warning("refusing to delete %s — outside the vault", filepath)
            return False

        removed_index = False
        try:
            removed_index = bool(self.store.delete(str(target)))
        except Exception:
            logger.exception("could not remove %s from the index", target)

        try:
            if target.is_file():
                target.unlink()
                logger.info("deleted %s", target.name)
        except OSError:
            logger.exception("could not delete %s", target)
            return False
        return removed_index

    # ------------------------------------------------------------------
    # Listing and reporting
    # ------------------------------------------------------------------

    def list_notes(
        self, folder: str | None = None, *, limit: int | None = None
    ) -> list[Note]:
        """Load every note in the vault, or in one folder, newest first."""
        root = self.folder(folder) if folder else self.vault_dir
        paths = sorted(
            (path for path in root.rglob("*.md") if not path.name.startswith(".")),
            key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
            reverse=True,
        )
        notes: list[Note] = []
        for path in paths:
            if limit is not None and len(notes) >= limit:
                break
            try:
                notes.append(self.read_note(path))
            except Exception:
                logger.warning("could not parse %s", path, exc_info=True)
        return notes

    def stats(self) -> dict[str, Any]:
        """Counts per folder and how much of it is in the index."""
        report: dict[str, Any] = {
            "vault_dir": str(self.vault_dir),
            "folders": {},
            "files": 0,
        }
        for name in VAULT_FOLDERS:
            count = len(list(self.folder(name).glob("*.md")))
            report["folders"][name] = count
            report["files"] += count
        try:
            report["indexed_notes"] = len(self._indexed_notes())
            report["index_records"] = self.store.count()
        except Exception:
            report["indexed_notes"] = None
            report["index_records"] = None
        return report

    def health(self) -> dict[str, Any]:
        """Snapshot for the API's status endpoint. Never raises."""
        try:
            return self.stats()
        except Exception as exc:  # pragma: no cover - defensive
            return {"vault_dir": str(self.vault_dir), "error": str(exc)}


# ---------------------------------------------------------------------------
# Frontmatter reading helpers
# ---------------------------------------------------------------------------


def _read_metadata(path: Path) -> dict[str, Any]:
    """Frontmatter of a note, or ``{}`` if it cannot be read."""
    try:
        return dict(frontmatter.load(path).metadata)
    except Exception:
        logger.debug("no readable frontmatter in %s", path)
        return {}


def _read_title(path: Path) -> str | None:
    """The ``title`` frontmatter value, or ``None`` when there is no frontmatter."""
    try:
        metadata = frontmatter.load(path).metadata
    except Exception:
        return None
    if not metadata:
        return None
    title = metadata.get("title")
    return str(title) if title is not None else None


def _timestamped_stem(directory: Path, moment: datetime) -> str:
    """``2026-09-25 14-30-05``, suffixed if that second is already taken."""
    base = moment.strftime("%Y-%m-%d %H-%M-%S")
    stem = base
    counter = 2
    while (directory / f"{stem}.md").exists():
        stem = f"{base}-{counter}"
        counter += 1
    return stem


def _fallback_summary(user_input: str, response: str) -> str:
    """A summary when the caller has none — the opening of the exchange.

    Deliberately crude: it is only used when no summariser ran, and a plainly
    truncated line is more honest than an invented one.
    """
    text = f"User asked: {user_input.strip()} Atlas replied: {response.strip()}"
    return text[:400]


# ---------------------------------------------------------------------------
# Shared instance
# ---------------------------------------------------------------------------

_manager: VaultManager | None = None
_manager_lock = threading.Lock()


def get_vault_manager(**kwargs: Any) -> VaultManager:
    """Process-wide vault manager, built once.

    The daemon, the agent and the skill builder should share one: two managers
    on the same directory are safe but each would open its own store handle.
    """
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = VaultManager(**kwargs)
    return _manager


def reset_vault_manager() -> None:
    """Drop the shared instance so the next call rebuilds it."""
    global _manager
    with _manager_lock:
        _manager = None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> int:
    """Exercise every public method against a throwaway vault and index.

    Uses temporary directories for both the vault and the Chroma store, so
    neither the user's notes nor the real index are touched.
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

    workspace = Path(tempfile.mkdtemp(prefix="atlas-vault-selftest-"))
    vault_dir = workspace / "vault"
    index_dir = workspace / "chroma"
    try:
        store = ChromaStore(persist_dir=index_dir)
        vault = VaultManager(vault_dir=vault_dir, store=store)

        print(f"\nvault: {vault.vault_dir}")
        check("folders were created", all(
            (vault_dir / name).is_dir() for name in VAULT_FOLDERS
        ))

        # --- write_conversation --------------------------------------
        path = vault.write_conversation(
            "what's the weather like?",
            "Overcast and twenty-two degrees in London.",
            "Asked about the weather; overcast, 22C in London.",
        )
        check("conversation file written", path.is_file(), path.name)
        check("conversation is in conversations/", path.parent.name == CONVERSATION_DIR)
        note = vault.read_note(path)
        check("frontmatter type is conversation", note.type == RECORD_TYPE_CONVERSATION)
        check("frontmatter has a date", note.metadata.get("date") is not None,
              str(note.metadata.get("date")))
        check("frontmatter has a summary", bool(note.summary))
        check("body keeps the full transcript",
              "Overcast and twenty-two degrees" in note.body and "weather like" in note.body)
        check("conversation indexed into its own collection",
              store.count(CONVERSATION_COLLECTION) == 1,
              f"{store.count(CONVERSATION_COLLECTION)} records")
        convo_records = store.list_by_type(None, CONVERSATION_COLLECTION)
        check("indexed document is the summary, not the transcript",
              convo_records[0]["text"] == "Asked about the weather; overcast, 22C in London.",
              repr(convo_records[0]["text"][:40]))
        check("both landed in the semantic collection? no",
              store.count(SEMANTIC_COLLECTION) == 0)

        # --- write_memory --------------------------------------------
        mem = vault.write_memory(
            "Favourite coffee",
            "A flat white from the shop on Bridge Street, no sugar.",
            memory_type="preference",
        )
        check("memory file written", mem.is_file(), mem.name)
        check("memory filename comes from the title", mem.stem == "Favourite coffee")
        mem_note = vault.read_note(mem)
        check("memory frontmatter type is memory", mem_note.type == "memory")
        check("memory_type recorded", mem_note.metadata.get("memory_type") == "preference")
        indexed = store.list_by_type("preference")
        check("indexed under its specific type", len(indexed) == 1, f"{len(indexed)} records")
        check("full content indexed for a memory",
              "flat white" in indexed[0]["text"])
        check("vault_type kept alongside", indexed[0]["metadata"].get("vault_type") == "memory")
        created_before = mem_note.metadata.get("created")

        time.sleep(1.1)
        mem2 = vault.write_memory(
            "Favourite coffee",
            "A flat white from Bridge Street. Also likes a cortado.",
            memory_type="preference",
        )
        check("same title upserts in place", mem2 == mem)
        check("no duplicate file", len(list((vault_dir / MEMORY_DIR).glob("*.md"))) == 1)
        check("created timestamp preserved",
              vault.read_note(mem).metadata.get("created") == created_before)
        check("updated timestamp moved on",
              vault.read_note(mem).metadata.get("updated") != created_before)
        check("re-indexing replaced the old chunk", store.list_by_type("preference")[0]["text"].endswith("cortado."),
              repr(store.list_by_type("preference")[0]["text"][-30:]))
        check("no orphaned records from the edit", len(store.list_by_type("preference")) == 1)

        # A different title that sanitises to the same stem must not clobber.
        clash = vault.write_memory("Favourite coffee?", "A different memory entirely.")
        check("colliding filename gets a suffix", clash != mem, clash.name)

        # --- unsafe titles -------------------------------------------
        weird = vault.write_memory("Q3: profit/loss <draft>", "Numbers pending.")
        check("unsafe characters sanitised",
              "/" not in weird.name and "<" not in weird.name, weird.name)
        check("hidden files not created", not weird.name.startswith("."))
        check("safe_filename handles junk", safe_filename("../../etc/passwd") == "..-..-etc-passwd"
              or ".." not in safe_filename("../../etc/passwd"),
              safe_filename("../../etc/passwd"))
        check("empty title gets a fallback", safe_filename("   ") == "untitled")
        check("windows device names get a suffix", safe_filename("CON") == "CON-note")

        # --- write_skill_note ----------------------------------------
        skill_file = workspace / "skills" / "pdf_to_text.py"
        skill = vault.write_skill_note(
            "PDF to text",
            "Converts a PDF into plain text using pdftotext.",
            skill_file,
            parameters={"path": "PDF to read", "page": "Page number, optional"},
            example='pdf_to_text("/tmp/a.pdf")',
        )
        check("skill note written", skill.is_file(), skill.name)
        skill_note = vault.read_note(skill)
        check("skill note records the python file",
              skill_note.metadata.get("skill_file") == str(skill_file))
        check("skill note is human-readable markdown",
              "Converts a PDF" in skill_note.body and "```" in skill_note.body)
        check("skill note documents parameters", "| `path` |" in skill_note.body)
        check("skill indexed", len(store.list_by_type("skill")) >= 1)

        # --- write_research_report -----------------------------------
        report = vault.write_research_report(
            "Local LLM inference options",
            "llama.cpp remains the fastest CPU path; vLLM wins on throughput.",
            [
                "https://github.com/ggerganov/llama.cpp",
                {"url": "https://docs.vllm.ai", "title": "vLLM docs"},
            ],
            question="What should I run locally in 2026?",
        )
        check("research report written", report.is_file(), report.name)
        report_note = vault.read_note(report)
        check("sources recorded in frontmatter",
              report_note.metadata.get("sources") == [
                  "https://github.com/ggerganov/llama.cpp", "https://docs.vllm.ai"],
              str(report_note.metadata.get("sources")))
        check("sources cited in the body",
              "vLLM docs" in report_note.body and "<https://github.com" in report_note.body)
        check("question recorded", report_note.metadata.get("question") == "What should I run locally in 2026?")

        # --- index_vault ---------------------------------------------
        print("\n--- index_vault ---")
        fresh = VaultManager(vault_dir=vault_dir, store=ChromaStore(persist_dir=workspace / "chroma2"))
        report_counts = fresh.index_vault()
        check("index_vault indexed every file", report_counts["indexed"] == len(fresh._vault_files()),
              f"{report_counts['indexed']} of {len(fresh._vault_files())} — {report_counts}")
        check("index_vault reports chunks", report_counts["chunks"] > 0)
        again = fresh.index_vault()
        check("second run is a no-op", again["indexed"] == 0 and again["unchanged"] > 0, str(again))

        time.sleep(1.1)
        edited = sorted((vault_dir / RESEARCH_DIR).glob("*.md"))[0]
        edited.write_text(edited.read_text() + "\n\nAn added paragraph.\n", encoding="utf-8")
        third = fresh.index_vault()
        check("an edited file is re-indexed", third["indexed"] == 1, str(third))
        check("the edit is searchable", bool(fresh.store.search("An added paragraph")))

        stale = vault_dir / MEMORY_DIR / "ghost.md"
        stale.write_text("---\ntype: memory\ntitle: Ghost\n---\n\nTemporary note.\n", encoding="utf-8")
        fresh.index_vault()
        before_prune = fresh.store.count()
        stale.unlink()
        pruned = fresh.index_vault()
        check("index_vault prunes deleted files", pruned["pruned"] == 1, str(pruned))
        check("pruned records are gone", fresh.store.count() < before_prune)

        # --- find_note -----------------------------------------------
        print("\n--- find_note ---")
        found_path, found_text, score = vault.find_note("what coffee does the user like?")
        check("find_note returns a path", found_path is not None,
              f"{found_path.name if found_path else None} ({score:.3f})")
        check("find_note returns the whole note, not a chunk",
              found_text is not None and "---" in found_text and "Favourite coffee" in found_text)
        check("find_note returns a float score", isinstance(score, float) and score > 0.0, f"{score:.3f}")
        check("find_note respects a high threshold",
              vault.find_note("what coffee does the user like?", threshold=0.99)[0] is None)
        check("find_note on nonsense finds nothing",
              vault.find_note("xylophone quantum bicycle", threshold=0.6)[0] is None)
        again_path, _, again_score = vault.find_note("what coffee does the user like?")
        check("find_note is deterministic", again_path == found_path and abs(again_score - score) < 1e-6)

        results = frontmatter.load(found_path)
        check("returned content is valid Obsidian markdown",
              results.metadata.get("type") == "memory" and bool(results.content))

        # --- delete_note ---------------------------------------------
        print("\n--- delete_note ---")
        check("delete_note reports index removal", vault.delete_note(found_path) is True)
        check("file is gone", not found_path.exists())
        check(
            "every record for the deleted note is gone",
            not [
                record
                for record in vault.store.list_by_type(None)
                if str(record["metadata"].get(META_SOURCE)) == str(found_path)
            ],
        )
        check(
            "find_note no longer returns it",
            vault.find_note("what coffee does the user like?", threshold=0.45)[0]
            != found_path,
        )
        check("deleting again is harmless", vault.delete_note(found_path) is False)
        check("refuses to delete outside the vault", vault.delete_note("/etc/passwd") is False)
        check("outside path was not touched", Path("/etc/passwd").exists())

        # --- search_notes / stats ------------------------------------
        ranked = vault.search_notes("pdf converter")
        check("search_notes ranks notes", bool(ranked), f"{len(ranked)} notes")
        check("search_notes de-duplicates", len({str(p) for p, _, _ in ranked}) == len(ranked))
        report_stats = vault.stats()
        check("stats counts files", report_stats["files"] > 0, str(report_stats["folders"]))
        check("stats reports indexed notes", report_stats["indexed_notes"] is not None)
        check("health() does not raise", "vault_dir" in vault.health())
    except Exception as exc:  # pragma: no cover - the test reports its own failure
        logger.exception("self-test crashed")
        failures.append(f"crashed: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for label in failures:
            print(f"  - {label}")
        return 1
    print("\nall checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
