"""
Atlas — university document retrieval.

Course material is a different problem from a conversation. A lecture handout is
thousands of words that must be searchable by meaning, cited precisely, and
removable again when the term ends, so this module turns files into embedded
chunks in a dedicated Chroma collection and answers questions from them.

Ingestion pipeline
------------------

1. **Extract** — text is pulled from the file by format:

   ===========  ==================================================================
   ``.pdf``     page by page with pypdf, so page numbers survive into citations
   ``.docx``    stdlib ``zipfile`` + XML over ``word/document.xml`` — no dependency
   ``.txt``     read as UTF-8, falling back to latin-1
   ``.md``      read as text; markdown is just text once it is embedded
   ===========  ==================================================================

2. **Chunk** — split at 512 tokens with 64 tokens of overlap, measured with
   tiktoken's ``cl100k_base`` so "512 tokens" is a real number rather than a
   guess. PDF pages are chunked individually so each chunk keeps its page.

3. **Embed and store** — each chunk goes into the ``university_docs`` collection
   with ``source``, ``page``, ``chunk`` and ``date_added`` metadata. The chunk id
   is derived from source and text, so re-ingesting an unchanged file upserts
   instead of duplicating; a changed file has its old chunks deleted first.

4. **Retrieve** — :meth:`UniversityRAG.query` embeds the question, pulls the
   closest chunks, and returns both a formatted context block and structured
   citations (file, page, chunk, score) for whatever the model writes.

Example::

    from modules.university.rag import UniversityRAG

    rag = UniversityRAG()
    rag.ingest_document("~/university/ps208/lecture-04.pdf")
    context, citations = rag.query("what did the lecture say about realism?")
    for source in rag.list_documents():
        print(source["filename"], source["chunks"])
    rag.remove_document("lecture-04.pdf")
"""

from __future__ import annotations

import json
import logging
import re
import threading
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence
from xml.etree import ElementTree

if TYPE_CHECKING:  # injected collaborators; imports here are typing-only
    from core.llm_manager import LLMServerManager
    from memory.chroma_store import ChromaStore, SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

#: The Chroma collection every university document lives in. Kept separate from
#: ``atlas_semantic`` so a course PDF cannot outrank a fact about the user.
UNIVERSITY_COLLECTION: str = "university_docs"

#: Chunk geometry, in real tiktoken tokens.
CHUNK_TOKENS: int = 512
CHUNK_OVERLAP_TOKENS: int = 64

#: Rough characters-per-token used only when tiktoken is unavailable.
_CHARS_PER_TOKEN: int = 4

DEFAULT_TOP_K: int = 6

#: Metadata vocabulary. ``source`` is deliberately the bare filename: it is what
#: a citation shows and what :meth:`UniversityRAG.remove_document` deletes by.
META_TYPE: str = "type"
META_SOURCE: str = "source"
META_PAGE: str = "page"
META_CHUNK: str = "chunk"
META_CHUNK_TOTAL: str = "chunk_total"
META_DATE_ADDED: str = "date_added"
META_PATH: str = "path"

RECORD_TYPE_DOCUMENT: str = "document"

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".txt", ".md", ".markdown", ".text")

# --- think-tag handling, matching the rest of the codebase ------------------
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_LEADING_THINK = re.compile(r"^\s*<think\b[^>]*>", re.IGNORECASE)
_TRAILING_THINK = re.compile(r"</think\s*>\s*$", re.IGNORECASE)

_FENCE_RE = re.compile(r"```(?:json|python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class DocumentError(RuntimeError):
    """Raised when a file cannot be read at all (missing, wrong type, empty)."""


# ---------------------------------------------------------------------------
# LLM plumbing (shared with quiz.py and mun.py)
# ---------------------------------------------------------------------------


def strip_think(text: str) -> str:
    """Remove reasoning that leaked into an answer."""
    if not text:
        return ""
    cleaned = _THINK_BLOCK.sub(" ", text)
    cleaned = _LEADING_THINK.sub("", cleaned)
    cleaned = _TRAILING_THINK.sub("", cleaned)
    return cleaned.strip()


def _complete(
    llm_manager: "LLMServerManager",
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    prefer: str,
) -> str:
    """One completion, trying the preferred model then the other, or "" .

    ``prefer="deep"`` is the default: judgement calls (a critique, a speech, a
    quiz) are worth the 30B. ``prefer="fast"`` is for cheap, latency-sensitive
    work such as grading a spoken answer.
    """
    order = ("deep", "fast") if prefer == "deep" else ("fast", "deep")
    request: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    for kind in order:
        try:
            if kind == "deep":
                if not llm_manager.ensure_deep_available():
                    logger.warning("deep model unavailable — trying the fast model")
                    continue
                client = llm_manager.get_deep_client()
            else:
                client = llm_manager.get_fast_client()
            request["model"] = kind
            response = client.chat.completions.create(**request)
            content = getattr(response.choices[0].message, "content", None) or ""
            content = strip_think(content)
            if content:
                return content
            logger.warning("%s model returned no content", kind)
        except Exception as exc:  # the SDK raises many shapes
            logger.warning("%s model request failed: %s", kind, exc)
    return ""


def ask_deep(
    llm_manager: "LLMServerManager",
    prompt: str,
    *,
    max_tokens: int,
    temperature: float = 0.3,
) -> str:
    """Completion preferring the deep model, degrading to the fast one."""
    return _complete(llm_manager, prompt, max_tokens=max_tokens, temperature=temperature, prefer="deep")


def ask_fast(
    llm_manager: "LLMServerManager",
    prompt: str,
    *,
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    """Completion preferring the fast model — for grading and other cheap work."""
    return _complete(llm_manager, prompt, max_tokens=max_tokens, temperature=temperature, prefer="fast")


def parse_json(text: str) -> Any:
    """Pull a JSON object or array out of a model reply. Returns None on failure.

    Models wrap JSON in prose, fences and trailing commas, so this is forgiving:
    it strips fences, then scans for the outermost ``{...}`` or ``[...]``.
    """
    if not text:
        return None
    cleaned = strip_think(text).strip()
    fenced = _FENCE_RE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()

    candidates: list[str] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            candidates.append(cleaned[start : end + 1])
    candidates.append(cleaned)

    for candidate in candidates:
        for attempt in (candidate, _drop_trailing_commas(candidate)):
            try:
                return json.loads(attempt)
            except (ValueError, TypeError):
                continue
    logger.debug("could not parse JSON from a model reply")
    return None


def _drop_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


# ---------------------------------------------------------------------------
# Document extraction
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExtractedPage:
    """One unit of extracted text, with the page it came from (0 = unknown)."""

    text: str
    page: int = 0


@dataclass(slots=True)
class IngestionResult:
    """Outcome of one :meth:`UniversityRAG.ingest_document` call."""

    filename: str = ""
    chunks: int = 0
    pages: int = 0
    characters: int = 0
    error: str | None = None
    date_added: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None and self.chunks > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "chunks": self.chunks,
            "pages": self.pages,
            "characters": self.characters,
            "error": self.error,
            "date_added": self.date_added,
        }


def extract_pages(path: Path) -> list[ExtractedPage]:
    """Extract text from a supported file, page by page where it makes sense."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        text = _extract_docx(path)
        return [ExtractedPage(text=text, page=0)] if text else []
    if suffix in (".txt", ".md", ".markdown", ".text") or not suffix:
        return [ExtractedPage(text=_read_text(path), page=0)]
    # An unknown extension: try as text rather than refusing outright.
    logger.info("%s has an unrecognised extension — reading it as text", path.name)
    return [ExtractedPage(text=_read_text(path), page=0)]


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    # Binary masquerading as text: decode with replacement rather than crash.
    return path.read_bytes().decode("utf-8", "replace")


def _extract_pdf(path: Path) -> list[ExtractedPage]:
    """Per-page text via pypdf, degrading to other readers if needed."""
    readers: list[tuple[str, Any]] = []
    try:
        from pypdf import PdfReader  # type: ignore

        readers.append(("pypdf", PdfReader))
    except ImportError:
        pass
    try:
        from PyPDF2 import PdfReader as LegacyReader  # type: ignore

        readers.append(("PyPDF2", LegacyReader))
    except ImportError:
        pass

    if not readers:
        raise DocumentError(
            "PDF support needs pypdf — install it with "
            "`.venv/bin/pip install pypdf` (pinned in requirements.txt)"
        )

    last_error: Exception | None = None
    for name, reader_cls in readers:
        try:
            reader = reader_cls(str(path))
            pages: list[ExtractedPage] = []
            for number, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as exc:  # one bad page must not sink the file
                    logger.warning("could not read page %d of %s: %s", number, path.name, exc)
                    text = ""
                if text.strip():
                    pages.append(ExtractedPage(text=text, page=number))
            if pages:
                logger.debug("read %d page(s) of %s with %s", len(pages), path.name, name)
            return pages
        except DocumentError:
            raise
        except Exception as exc:
            last_error = exc
            logger.warning("%s could not open %s: %s", name, path.name, exc)
    raise DocumentError(f"could not read the PDF {path.name}: {last_error}")


#: WordprocessingML namespace inside a .docx.
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _extract_docx(path: Path) -> str:
    """Text from a .docx using only the standard library.

    A ``.docx`` is a zip archive whose ``word/document.xml`` holds the body.
    Reading ``<w:p>`` elements line by line is enough for lecture notes: a
    table cell is a ``<w:p>`` too, so tables come through as well.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open("word/document.xml") as stream:
                tree = ElementTree.parse(stream)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise DocumentError(f"{path.name} is not a readable .docx ({exc})") from exc
    except ElementTree.ParseError as exc:
        raise DocumentError(f"{path.name} has malformed document XML ({exc})") from exc

    paragraphs: list[str] = []
    for paragraph in tree.iter(f"{_W_NS}p"):
        pieces: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{_W_NS}t" and node.text:
                pieces.append(node.text)
            elif node.tag == f"{_W_NS}tab":
                pieces.append("\t")
            elif node.tag == f"{_W_NS}br":
                pieces.append("\n")
        line = "".join(pieces).strip()
        if line:
            paragraphs.append(line)
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_encoder_lock = threading.Lock()
_encoder: Any = None
_encoder_failed = False


def _get_encoder() -> Any:
    """The tiktoken encoder, loaded once; None when unavailable (offline)."""
    global _encoder, _encoder_failed
    if _encoder is not None or _encoder_failed:
        return _encoder
    with _encoder_lock:
        if _encoder is not None or _encoder_failed:
            return _encoder
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # offline, or tiktoken missing
            logger.warning(
                "tiktoken unavailable (%s) — chunking by character estimate", exc
            )
            _encoder_failed = True
    return _encoder


def count_tokens(text: str) -> int:
    """Token count, with a character-based fallback."""
    encoder = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:  # pragma: no cover - defensive
            pass
    return max(1, len(text) // _CHARS_PER_TOKEN)


def chunk_text(
    text: str,
    *,
    tokens: int = CHUNK_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split ``text`` into overlapping windows of about ``tokens`` tokens."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    encoder = _get_encoder()
    if encoder is None:
        return _chunk_by_chars(cleaned, tokens, overlap)

    try:
        ids = encoder.encode(cleaned)
    except Exception:  # pragma: no cover - defensive
        return _chunk_by_chars(cleaned, tokens, overlap)
    if len(ids) <= tokens:
        return [cleaned]

    step = max(tokens - overlap, 1)
    chunks: list[str] = []
    for start in range(0, len(ids), step):
        window = ids[start : start + tokens]
        if not window:
            break
        piece = encoder.decode(window).strip()
        if piece:
            chunks.append(piece)
        if start + tokens >= len(ids):
            break
    return chunks


def _chunk_by_chars(text: str, tokens: int, overlap: int) -> list[str]:
    size = tokens * _CHARS_PER_TOKEN
    step = max(size - overlap * _CHARS_PER_TOKEN, 1)
    if len(text) <= size:
        return [text]
    return [text[start : start + size].strip() for start in range(0, len(text), step) if text[start : start + size].strip()]


# ---------------------------------------------------------------------------
# The RAG module
# ---------------------------------------------------------------------------


class UniversityRAG:
    """Ingest course files into Chroma and answer questions from them."""

    def __init__(
        self,
        store: "ChromaStore | None" = None,
        *,
        collection: str = UNIVERSITY_COLLECTION,
        chunk_tokens: int = CHUNK_TOKENS,
        chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
    ) -> None:
        self._store = store
        self.collection = collection
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap = chunk_overlap
        self._lock = threading.RLock()

    @property
    def store(self) -> "ChromaStore":
        """The vector store, resolved to the process-wide instance on first use."""
        if self._store is None:
            from memory.chroma_store import get_chroma_store

            self._store = get_chroma_store()
        return self._store

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_document(self, filepath: str | Path) -> IngestionResult:
        """Extract, chunk, embed and store one document.

        Returns an :class:`IngestionResult`. A missing file or an unreadable
        document is reported in ``result.error`` rather than raised, so a caller
        can ingest a folder and keep going.
        """
        path = Path(filepath).expanduser()
        filename = path.name
        result = IngestionResult(filename=filename)
        if not path.is_file():
            result.error = f"no such file: {path}"
            logger.warning("cannot ingest %s — %s", path, result.error)
            return result

        try:
            pages = extract_pages(path)
        except DocumentError as exc:
            result.error = str(exc)
            return result
        except Exception as exc:
            result.error = f"could not read {filename}: {exc}"
            logger.warning("extraction of %s failed", path, exc_info=True)
            return result

        if not pages:
            result.error = f"no extractable text in {filename}"
            return result

        date_added = datetime.now(timezone.utc).isoformat(timespec="seconds")
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for page in pages:
            pieces = chunk_text(
                page.text, tokens=self.chunk_tokens, overlap=self.chunk_overlap
            )
            total = len(pieces)
            for index, piece in enumerate(pieces):
                meta: dict[str, Any] = {
                    META_TYPE: RECORD_TYPE_DOCUMENT,
                    META_SOURCE: filename,
                    META_CHUNK: index,
                    META_CHUNK_TOTAL: total,
                    META_DATE_ADDED: date_added,
                    META_PATH: str(path),
                }
                if page.page:
                    meta[META_PAGE] = page.page
                texts.append(piece)
                metadatas.append(meta)

        if not texts:
            result.error = f"no extractable text in {filename}"
            return result

        with self._lock:
            # Delete first: a changed file must not leave stale chunks behind,
            # because the chunk ids are content-derived and would not collide.
            try:
                self.store.delete(filename, self.collection)
            except Exception:
                logger.warning("could not clear old chunks of %s", filename, exc_info=True)
            try:
                self.store.add_many(texts, metadatas, self.collection)
            except Exception as exc:
                result.error = f"could not index {filename}: {exc}"
                logger.warning("indexing of %s failed", path, exc_info=True)
                return result

        result.chunks = len(texts)
        result.pages = len(pages)
        result.characters = sum(len(page.text) for page in pages)
        result.date_added = date_added
        logger.info(
            "ingested %s: %d chunk(s) across %d section(s)", filename, result.chunks, result.pages
        )
        return result

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        *,
        doc_filter: str | Sequence[str] | None = None,
        min_score: float | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Retrieve context for ``question``.

        Returns ``(context_string, citations)``. ``context_string`` is numbered
        and ready to paste into a prompt; ``citations`` is the same material in
        structured form — ``{source, page, chunk, score}`` — for the UI.
        ``doc_filter`` restricts retrieval to one filename or a list of them.
        """
        text = (question or "").strip()
        if not text:
            return "", []

        where = _document_where(doc_filter)
        try:
            hits = self.store.search(
                text,
                k=max(int(top_k), 1),
                collection=self.collection,
                where=where,
                min_score=min_score,
            )
        except Exception as exc:
            logger.warning("university query failed: %s", exc, exc_info=True)
            return "", []

        if not hits:
            return "", []

        blocks: list[str] = []
        citations: list[dict[str, Any]] = []
        for index, hit in enumerate(hits, start=1):
            meta = hit.metadata or {}
            source = str(meta.get(META_SOURCE, "") or "unknown")
            page = meta.get(META_PAGE)
            label = f"[{index}] {source}" + (f", page {page}" if page else "")
            blocks.append(f"{label}\n{hit.text.strip()}")
            citations.append(
                {
                    "index": index,
                    "source": source,
                    "page": page,
                    "chunk": meta.get(META_CHUNK),
                    "path": meta.get(META_PATH, ""),
                    "score": round(hit.score, 4),
                }
            )
        return "\n\n".join(blocks), citations

    # ------------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------------

    def list_documents(self) -> list[dict[str, Any]]:
        """Every indexed source, with chunk and page counts, newest first."""
        try:
            records = self.store.list_by_type(RECORD_TYPE_DOCUMENT, self.collection)
        except Exception as exc:
            logger.warning("could not list university documents: %s", exc, exc_info=True)
            return []

        grouped: dict[str, dict[str, Any]] = {}
        for record in records:
            meta = record.get("metadata") or {}
            source = str(meta.get(META_SOURCE, "") or "")
            if not source:
                continue
            entry = grouped.setdefault(
                source,
                {
                    "filename": source,
                    "chunks": 0,
                    "pages": set(),
                    "date_added": "",
                    "path": str(meta.get(META_PATH, "") or ""),
                },
            )
            entry["chunks"] += 1
            page = meta.get(META_PAGE)
            if isinstance(page, int) and page > 0:
                entry["pages"].add(page)
            added = str(meta.get(META_DATE_ADDED, "") or "")
            if added > entry["date_added"]:
                entry["date_added"] = added

        documents = []
        for entry in grouped.values():
            entry["pages"] = len(entry["pages"])
            documents.append(entry)
        documents.sort(key=lambda item: item["date_added"], reverse=True)
        return documents

    def remove_document(self, filename: str | Path) -> int:
        """Delete every chunk belonging to one file. Returns how many went."""
        source = Path(str(filename)).name
        if not source:
            raise ValueError("remove_document() needs a filename")
        try:
            removed = self.store.delete(source, self.collection)
        except Exception as exc:
            logger.warning("could not remove %s: %s", source, exc, exc_info=True)
            return 0
        if removed:
            logger.info("removed %d chunk(s) of %s", removed, source)
        return removed

    def is_indexed(self, filename: str | Path) -> bool:
        """True when at least one chunk of ``filename`` is stored."""
        source = Path(str(filename)).name
        return any(doc["filename"] == source for doc in self.list_documents())

    def count(self) -> int:
        """Total number of stored chunks."""
        try:
            return self.store.count(self.collection)
        except Exception:
            return 0


def _document_where(doc_filter: str | Sequence[str] | None) -> dict[str, Any] | None:
    """Translate a filename filter into a Chroma ``where`` clause."""
    if not doc_filter:
        return None
    if isinstance(doc_filter, (str, Path)):
        return {META_SOURCE: Path(str(doc_filter)).name}
    names = [Path(str(item)).name for item in doc_filter if str(item).strip()]
    if not names:
        return None
    if len(names) == 1:
        return {META_SOURCE: names[0]}
    return {META_SOURCE: {"$in": names}}


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

_default: UniversityRAG | None = None
_default_lock = threading.Lock()


def get_university_rag() -> UniversityRAG:
    """Process-wide RAG module, built once."""
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = UniversityRAG()
    return _default


def ingest_document(filepath: str | Path) -> IngestionResult:
    return get_university_rag().ingest_document(filepath)


def query(
    question: str, top_k: int = DEFAULT_TOP_K, *, doc_filter: str | Sequence[str] | None = None
) -> tuple[str, list[dict[str, Any]]]:
    return get_university_rag().query(question, top_k, doc_filter=doc_filter)


def list_documents() -> list[dict[str, Any]]:
    return get_university_rag().list_documents()


def remove_document(filename: str | Path) -> int:
    return get_university_rag().remove_document(filename)


__all__ = [
    "UNIVERSITY_COLLECTION",
    "CHUNK_TOKENS",
    "CHUNK_OVERLAP_TOKENS",
    "DocumentError",
    "ExtractedPage",
    "IngestionResult",
    "UniversityRAG",
    "ask_deep",
    "ask_fast",
    "chunk_text",
    "count_tokens",
    "extract_pages",
    "get_university_rag",
    "ingest_document",
    "list_documents",
    "parse_json",
    "query",
    "remove_document",
    "strip_think",
]
