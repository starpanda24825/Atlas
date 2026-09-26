"""
Atlas — the local HTTP/WebSocket bridge.

This is the only way the UI talks to Atlas. It binds to ``127.0.0.1:8765``
(:data:`core.config.FASTAPI_PORT`) and is never exposed to the network: it can
switch modes, read the vault, approve skills and start research, so anything
that can reach it can drive the machine.

The daemon builds the collaborators and hands them in::

    from api.server import create_app, Services
    from api.websocket_manager import get_websocket_manager

    # The manager is a singleton the daemon broadcasts on directly.
    app = create_app(Services(llm_manager=manager, agent=agent, ...))
    uvicorn.run(app, host="127.0.0.1", port=8765)

Every collaborator is optional. When one is missing the endpoint answers with
``503`` and a sentence naming what is unavailable, rather than raising an
``AttributeError`` — the modules this talks to (trading, university) are
implemented separately, and the UI should be able to show "not ready yet".

Endpoint groups: status, mode, conversations, memory, skills, research,
trading, documents (university), system, and ``/ws``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.websocket_manager import WebSocketManager, get_websocket_manager
from core.config import BASE_DIR, FASTAPI_PORT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

VALID_MODES: tuple[str, ...] = ("voice", "study", "trading", "gaming", "idle")
GAMING_MODE = "gaming"

#: Where an uploaded study document is staged before the RAG indexer sees it.
UPLOADS_DIR: Path = BASE_DIR / "uploads" / "university"

#: UI note-type names mapped to the vault folders that hold them.
VAULT_FOLDERS: dict[str, str] = {
    "memory": "memories",
    "conversation": "conversations",
    "research": "research",
    "skill": "skills",
}


# ---------------------------------------------------------------------------
# Service container
# ---------------------------------------------------------------------------


@dataclass
class Services:
    """Every collaborator the API can reach.

    All fields default to None so a bare ``Services()`` is a valid, fully
    degraded server. The daemon fills in what it has.
    """

    llm_manager: Any = None
    agent: Any = None
    memory_manager: Any = None
    skill_registry: Any = None
    skill_builder: Any = None
    vault_manager: Any = None
    search_router: Any = None
    quick_search: Any = None
    deep_researcher: Any = None
    trading: Any = None
    university: Any = None
    voice_pipeline: Any = None
    wake_word: Any = None
    resource_governor: Any = None
    #: The fan-out used for every ``/ws`` client and background notification.
    #: Defaults to the process-wide singleton so the daemon and the routes
    #: push to the same sockets.
    ws_manager: WebSocketManager = field(default_factory=get_websocket_manager)
    #: Last known mode, mirrored by :class:`ModeController`.
    mode: str = "idle"


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int | None = Field(default=None, ge=1, le=50)


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    depth: int = Field(default=3, ge=1, le=5)


class TradeConfirmRequest(BaseModel):
    trade_id: str = Field(..., min_length=1)
    confirmed: bool


class QuizRequest(BaseModel):
    topic: str | None = None
    documents: list[str] = Field(default_factory=list)
    num_questions: int = Field(default=5, ge=1, le=50)


class GamingToggleRequest(BaseModel):
    enabled: bool | None = None


# ---------------------------------------------------------------------------
# Research job tracking
# ---------------------------------------------------------------------------


@dataclass
class ResearchJob:
    """One in-flight or finished deep research run."""

    id: str
    question: str
    depth: int
    status: str = "running"  # running | complete | failed
    progress: str = "queued"  # human-readable current stage, for the UI
    started: float = field(default_factory=time.time)
    finished: float | None = None
    report: dict[str, Any] | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "depth": self.depth,
            "status": self.status,
            "progress": self.progress,
            "started": _iso(self.started),
            "finished": _iso(self.finished) if self.finished else None,
            "report": self.report,
            "error": self.error,
        }


class ResearchJobs:
    """Thread-safe registry of research runs, newest first."""

    def __init__(self) -> None:
        self._jobs: dict[str, ResearchJob] = {}
        self._lock = threading.Lock()

    def create(self, question: str, depth: int) -> ResearchJob:
        job = ResearchJob(id=uuid.uuid4().hex, question=question, depth=depth)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def update(self, job_id: str, **fields: Any) -> ResearchJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in fields.items():
                setattr(job, key, value)
            return job

    def get(self, job_id: str) -> ResearchJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[ResearchJob]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda job: job.started, reverse=True)
        return jobs


# ---------------------------------------------------------------------------
# Mode control
# ---------------------------------------------------------------------------


class ModeController:
    """Owns the current mode and applies the side effects of switching.

    A collaborator that knows how to switch (the resource governor, the agent)
    is preferred; when none does, the built-in behaviour is applied. Gaming is
    the only mode with hard side effects: the voice and model servers are
    stopped to free the machine, while the wake-word daemon is deliberately
    left running so Atlas can still hear "hey atlas" to come back.
    """

    def __init__(self, services: Services) -> None:
        self.services = services
        self._mode = services.mode if services.mode in VALID_MODES else "idle"
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def set_mode(self, mode_name: str) -> dict[str, Any] | None:
        """Switch mode. Returns ``None`` for an invalid name."""
        name = (mode_name or "").strip().lower()
        if name not in VALID_MODES:
            return None
        with self._lock:
            previous = self._mode
            self._mode = name
        self.services.mode = name
        self._apply(name, previous)
        self.services.ws_manager.mode_change(name, previous=previous)
        return {"mode": name, "previous": previous}

    def toggle_gaming(self, enabled: bool | None) -> dict[str, Any]:
        """Turn gaming mode on or off; ``None`` flips the current state."""
        if enabled is None:
            enabled = self.mode != GAMING_MODE
        target = GAMING_MODE if enabled else "idle"
        return self.set_mode(target) or {"mode": self.mode}

    # -- side effects ---------------------------------------------------

    def _apply(self, name: str, previous: str) -> None:
        delegated = False
        for target in (self.services.resource_governor, self.services.agent):
            method = _first_callable(target, ("set_mode", "switch_mode"))
            if method is not None:
                try:
                    _call_any(method, name)
                    delegated = True
                except Exception:
                    logger.exception("collaborator failed to switch to mode %s", name)
        if delegated:
            return
        if name == GAMING_MODE:
            self._enter_gaming()
        elif previous == GAMING_MODE:
            logger.info("left gaming mode — servers are restarted by the daemon")

    def _enter_gaming(self) -> None:
        """Free the machine: stop voice and models, keep the wake word."""
        voice = self.services.voice_pipeline
        if voice is not None:
            method = _first_callable(voice, ("shutdown", "stop", "close"))
            if method is not None:
                try:
                    _call_any(method)
                    logger.info("gaming mode: voice pipeline stopped")
                except Exception:
                    logger.exception("could not stop the voice pipeline")
        manager = self.services.llm_manager
        if manager is not None:
            method = _first_callable(manager, ("stop_all", "shutdown"))
            if method is not None:
                try:
                    _call_any(method)
                    logger.info("gaming mode: model servers stopped")
                except Exception:
                    logger.exception("could not stop the model servers")
        logger.info("gaming mode: wake word left listening")


# ---------------------------------------------------------------------------
# Duck-typing helpers
# ---------------------------------------------------------------------------


def _first_callable(target: Any, names: Sequence[str]) -> Callable[..., Any] | None:
    """First callable attribute of ``target`` matching one of ``names``."""
    if target is None:
        return None
    for name in names:
        method = getattr(target, name, None)
        if callable(method):
            return method
    return None


def _call_any(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn``, dropping keyword arguments its signature does not accept.

    Collaborators are duck-typed and their parameter names vary, so this keeps
    the API from breaking when, say, an indexer names its argument ``file``
    instead of ``path``.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    parameters = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return fn(*args, **kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in parameters}
    return fn(*args, **accepted)


def _safe(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a collaborator call, turning failures into a 502."""
    try:
        return _call_any(fn, *args, **kwargs)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("collaborator call %s failed", getattr(fn, "__name__", fn))
        raise HTTPException(status_code=502, detail=f"{getattr(fn, '__name__', 'call')} failed: {exc}") from exc


def _require(target: Any, feature: str) -> Any:
    if target is None:
        raise HTTPException(status_code=503, detail=f"{feature} is not available")
    return target


def _import_module(path: str) -> Any | None:
    try:
        return __import__(path, fromlist=["*"])
    except ImportError:
        logger.debug("module %s is not implemented yet", path)
        return None


def _trading_target(services: Services, module_name: str, attr: str) -> Any:
    """The live trading collaborator, falling back to the module on disk."""
    namespace = services.trading
    if namespace is not None:
        target = getattr(namespace, attr, None)
        if target is not None:
            return target
        if not isinstance(namespace, Mapping):
            return namespace
    return _import_module(f"modules.trading.{module_name}")


def _university_target(services: Services, module_name: str) -> Any:
    namespace = services.university
    if namespace is not None:
        target = getattr(namespace, module_name, None)
        if target is not None:
            return target
        if not isinstance(namespace, Mapping):
            return namespace
    return _import_module(f"modules.university.{module_name}")


# ---------------------------------------------------------------------------
# Small serialisers
# ---------------------------------------------------------------------------


def _iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")


def _note_brief(note: Any) -> dict[str, Any]:
    return {
        "id": note.path.name,
        "title": note.title,
        "summary": note.summary,
        "type": note.type,
        "tags": note.tags,
        "date": str(note.metadata.get("date") or ""),
        "created": str(note.metadata.get("created") or ""),
        # Present on research reports; the UI shows it as a source count.
        "source_count": note.metadata.get("source_count"),
        "question": str(note.metadata.get("question") or ""),
    }


def _note_detail(note: Any) -> dict[str, Any]:
    detail = _note_brief(note)
    detail.update({"body": note.body, "metadata": dict(note.metadata)})
    return detail


def _report_dict(report: Any) -> dict[str, Any]:
    """Normalise a ``ResearchReport`` (or a mapping) for JSON."""
    if report is None:
        return {}
    if isinstance(report, Mapping):
        data = dict(report)
    else:
        data = {
            "title": getattr(report, "title", ""),
            "summary": getattr(report, "summary", ""),
            "full_report": getattr(report, "full_report", ""),
            "sources": getattr(report, "sources", []),
            "timestamp": getattr(report, "timestamp", None),
        }
    timestamp = data.get("timestamp")
    if isinstance(timestamp, datetime):
        data["timestamp"] = timestamp.isoformat(timespec="seconds")
    elif timestamp is not None:
        data["timestamp"] = str(timestamp)
    return data


def _resolve_note_path(vault: Any, folder: str, note_id: str) -> Path | None:
    """Find a note by id inside one vault folder, refusing to escape it."""
    try:
        base = vault.folder(folder)
        root = vault.vault_dir.resolve()
    except Exception:
        return None

    candidates = [base / note_id]
    if not note_id.endswith(".md"):
        candidates.append(base / f"{note_id}.md")
    if "/" in note_id:
        candidates.append(vault.vault_dir / note_id)

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved != root and root not in resolved.parents:
            continue
        if resolved.is_file():
            return resolved
    return None


# ---------------------------------------------------------------------------
# System status
# ---------------------------------------------------------------------------


def _process_alive(process: Any) -> bool:
    return process is not None and callable(getattr(process, "poll", None)) and process.poll() is None


def _vram_usage() -> dict[str, Any] | None:
    """Used/total VRAM on GPU 0 via GPUtil, or None when unavailable."""
    try:
        import GPUtil

        gpus = GPUtil.getGPUs()
    except Exception as exc:
        logger.debug("VRAM reading unavailable: %s", exc)
        return None
    if not gpus:
        return None
    gpu = gpus[0]
    return {
        "used_mb": round(float(gpu.memoryUsed)),
        "total_mb": round(float(gpu.memoryTotal)),
        "percent": round(float(gpu.memoryUtil) * 100, 1),
    }


def _ram_usage() -> dict[str, Any] | None:
    try:
        import psutil

        memory = psutil.virtual_memory()
    except Exception as exc:
        logger.debug("RAM reading unavailable: %s", exc)
        return None
    return {
        "used_mb": round(memory.used / 1024 / 1024),
        "total_mb": round(memory.total / 1024 / 1024),
        "percent": memory.percent,
    }


def _status_payload(services: Services, modes: ModeController) -> dict[str, Any]:
    """Everything ``GET /status`` and the periodic push report."""
    manager = services.llm_manager
    deep_alive = _process_alive(getattr(manager, "deep_process", None))
    fast_alive = _process_alive(getattr(manager, "fast_process", None))
    active_model = "deep" if deep_alive else ("fast" if fast_alive else "none")

    wake = services.wake_word
    listening = bool(getattr(wake, "running", False)) if wake is not None else False

    return {
        "active_model": active_model,
        "vram_usage": _vram_usage(),
        "ram_usage": _ram_usage(),
        "active_mode": modes.mode,
        "wake_word_listening": listening,
        "deep_server_running": deep_alive,
    }


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    services: Services = app.state.services
    modes: ModeController = app.state.mode_controller
    task = asyncio.create_task(
        services.ws_manager.status_loop(lambda: _status_payload(services, modes))
    )
    logger.info("Atlas API listening on 127.0.0.1:%d", FASTAPI_PORT)
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        services.ws_manager.close()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(services: Services | None = None) -> FastAPI:
    """Build the FastAPI application around a :class:`Services` container."""
    services = services if services is not None else Services()

    app = FastAPI(title="Atlas API", version="1.0", lifespan=lifespan)
    app.state.services = services
    app.state.mode_controller = ModeController(services)
    app.state.research_jobs = ResearchJobs()

    # Localhost only. The dev UI runs on localhost:1420, while the packaged
    # Tauri app serves from a custom protocol whose origin is tauri://localhost
    # (http(s)://tauri.localhost on Windows), so all of those must be allowed.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=(
            r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?"
            r"|tauri://localhost"
            r"|https?://tauri\.localhost)$"
        ),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_services(request: Request) -> Services:
        return request.app.state.services

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @app.get("/status")
    def get_status(request: Request) -> dict[str, Any]:
        """Current system state. Always answers, even with no collaborators."""
        state = request.app.state
        return _status_payload(state.services, state.mode_controller)

    # ------------------------------------------------------------------
    # Mode control
    # ------------------------------------------------------------------

    @app.get("/mode")
    def get_mode(request: Request) -> dict[str, Any]:
        return {"mode": request.app.state.mode_controller.mode}

    @app.post("/mode/{mode_name}")
    def set_mode(mode_name: str, request: Request) -> dict[str, Any]:
        result = request.app.state.mode_controller.set_mode(mode_name)
        if result is None:
            raise HTTPException(
                status_code=400,
                detail=f"unknown mode {mode_name!r}; expected one of {', '.join(VALID_MODES)}",
            )
        return result

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    @app.get("/conversations")
    def list_conversations(
        request: Request,
        limit: int = 20,
        offset: int = 0,
        services: Services = Depends(get_services),
    ) -> dict[str, Any]:
        vault = _require(services.vault_manager, "The conversation vault")
        notes = _safe(_first_callable(vault, ("list_notes", "notes")), folder="conversations")
        notes = list(notes or [])
        window = notes[offset : offset + max(0, limit)]
        return {
            "total": len(notes),
            "limit": limit,
            "offset": offset,
            "items": [_note_brief(note) for note in window],
        }

    @app.get("/conversations/{conversation_id}")
    def get_conversation(
        conversation_id: str,
        services: Services = Depends(get_services),
    ) -> dict[str, Any]:
        vault = _require(services.vault_manager, "The conversation vault")
        path = _resolve_note_path(vault, "conversations", conversation_id)
        if path is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return _note_detail(_safe(vault.read_note, path))

    @app.delete("/conversations/{conversation_id}")
    def delete_conversation(
        conversation_id: str,
        services: Services = Depends(get_services),
    ) -> dict[str, Any]:
        vault = _require(services.vault_manager, "The conversation vault")
        path = _resolve_note_path(vault, "conversations", conversation_id)
        if path is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return {"deleted": bool(_safe(vault.delete_note, path)), "id": conversation_id}

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    @app.get("/memories")
    def list_memories(
        limit: int | None = None,
        services: Services = Depends(get_services),
    ) -> dict[str, Any]:
        memory = _require(services.memory_manager, "The memory store")
        items = _safe(_first_callable(memory, ("get_all", "list_all", "all")), limit=limit)
        items = list(items or [])
        if limit is not None:
            items = items[: max(0, limit)]
        return {"total": len(items), "items": items}

    @app.delete("/memories/{memory_id}")
    def delete_memory(
        memory_id: str, services: Services = Depends(get_services)
    ) -> dict[str, Any]:
        memory = _require(services.memory_manager, "The memory store")
        removed = _safe(_first_callable(memory, ("delete", "remove", "forget")), memory_id)
        if not removed:
            raise HTTPException(status_code=404, detail="memory not found")
        return {"deleted": True, "id": memory_id}

    @app.post("/memories/search")
    def search_memories(
        body: MemorySearchRequest, services: Services = Depends(get_services)
    ) -> dict[str, Any]:
        memory = _require(services.memory_manager, "The memory store")
        search = _first_callable(memory, ("search_memories", "search", "retrieve"))
        if search is None:
            raise HTTPException(status_code=503, detail="The memory store cannot search")
        results = _safe(search, body.query, limit=body.limit)
        return {"query": body.query, "results": results}

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    @app.get("/skills")
    def list_skills(
        include_builtin: bool = True, services: Services = Depends(get_services)
    ) -> dict[str, Any]:
        registry = _require(services.skill_registry, "The skill registry")
        items = _safe(
            _first_callable(registry, ("list_all", "list_skills")),
            include_builtin=include_builtin,
        )
        items = list(items or [])
        # list_skills() returns SkillRecord objects; list_all() returns dicts.
        items = [item.as_dict() if hasattr(item, "as_dict") else item for item in items]
        return {"total": len(items), "items": items}

    @app.get("/skills/pending")
    def list_pending_skills(services: Services = Depends(get_services)) -> dict[str, Any]:
        builder = _require(services.skill_builder, "The skill builder")
        pending = _safe(
            _first_callable(builder, ("pending_refinements", "pending", "pending_proposals"))
        )
        pending = list(pending or [])
        return {"total": len(pending), "items": pending}

    @app.get("/skills/{skill_name}/code")
    def get_skill_code(
        skill_name: str, services: Services = Depends(get_services)
    ) -> dict[str, Any]:
        registry = _require(services.skill_registry, "The skill registry")
        record = _safe(_first_callable(registry, ("get",)), skill_name)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown skill {skill_name!r}")
        path = getattr(record, "path", None)
        if path is None or not Path(path).is_file():
            raise HTTPException(status_code=404, detail=f"skill {skill_name!r} has no file")
        try:
            code = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"could not read skill: {exc}") from exc
        return {"name": skill_name, "path": str(path), "code": code}

    @app.post("/skills/{skill_name}/approve")
    def approve_skill(
        skill_name: str, services: Services = Depends(get_services)
    ) -> dict[str, Any]:
        builder = _require(services.skill_builder, "The skill builder")
        proposal = _find_pending_proposal(builder, skill_name)
        if proposal is None:
            raise HTTPException(
                status_code=404, detail=f"no pending skill named {skill_name!r}"
            )
        approved = _approve_proposal(builder, proposal, skill_name)
        if not approved:
            raise HTTPException(status_code=502, detail="could not register the skill")
        return {"approved": True, "name": skill_name}

    @app.post("/skills/{skill_name}/reject")
    def reject_skill(
        skill_name: str, services: Services = Depends(get_services)
    ) -> dict[str, Any]:
        builder = _require(services.skill_builder, "The skill builder")
        proposal = _find_pending_proposal(builder, skill_name)

        # Preferred path drops the queued refinement; otherwise record it so
        # the same mistake is not proposed again. A builder that only tracks
        # refinements by name still works via the drop call.
        drop = _first_callable(builder, ("drop_refinement",))
        if drop is not None and _safe(drop, skill_name):
            return {"rejected": True, "name": skill_name}
        if proposal is None:
            raise HTTPException(
                status_code=404, detail=f"no pending skill named {skill_name!r}"
            )
        record = _first_callable(builder, ("record_rejection",))
        if record is not None:
            _safe(record, proposal, feedback="rejected from the UI")
        return {"rejected": True, "name": skill_name}

    @app.delete("/skills/{skill_name}")
    def delete_skill(
        skill_name: str, services: Services = Depends(get_services)
    ) -> dict[str, Any]:
        registry = _require(services.skill_registry, "The skill registry")
        remove = _first_callable(registry, ("remove_skill", "remove", "delete"))
        if remove is None:
            raise HTTPException(status_code=503, detail="The skill registry cannot remove skills")
        removed = _safe(remove, skill_name, delete_file=True)
        if not removed:
            raise HTTPException(status_code=404, detail=f"unknown skill {skill_name!r}")
        return {"deleted": True, "name": skill_name}

    # ------------------------------------------------------------------
    # Vault notes (used by the UI's vault browser)
    # ------------------------------------------------------------------

    @app.get("/vault/notes")
    def list_vault_notes(
        note_type: str | None = None,
        limit: int | None = None,
        services: Services = Depends(get_services),
    ) -> dict[str, Any]:
        vault = _require(services.vault_manager, "The vault")
        folder = VAULT_FOLDERS.get((note_type or "").strip().lower())
        if note_type and folder is None:
            raise HTTPException(
                status_code=400,
                detail=f"unknown note type {note_type!r}; expected one of {', '.join(sorted(VAULT_FOLDERS))}",
            )
        notes = _safe(_first_callable(vault, ("list_notes", "notes")), folder=folder, limit=limit)
        notes = list(notes or [])
        return {"total": len(notes), "items": [_note_brief(note) for note in notes]}

    @app.get("/vault/notes/{folder}/{note_id}")
    def get_vault_note(
        folder: str,
        note_id: str,
        services: Services = Depends(get_services),
    ) -> dict[str, Any]:
        vault = _require(services.vault_manager, "The vault")
        if folder not in VAULT_FOLDERS.values():
            raise HTTPException(status_code=400, detail=f"unknown vault folder {folder!r}")
        path = _resolve_note_path(vault, folder, note_id)
        if path is None:
            raise HTTPException(status_code=404, detail="note not found")
        return _note_detail(_safe(vault.read_note, path))

    # ------------------------------------------------------------------
    # Research
    # ------------------------------------------------------------------

    @app.get("/research")
    def list_research(services: Services = Depends(get_services)) -> dict[str, Any]:
        vault = _require(services.vault_manager, "The research vault")
        notes = _safe(_first_callable(vault, ("list_notes", "notes")), folder="research")
        notes = list(notes or [])
        return {"total": len(notes), "items": [_note_brief(note) for note in notes]}

    # Declared before /research/{report_id} so "jobs" is not read as an id.
    @app.get("/research/jobs")
    def list_research_jobs(request: Request) -> dict[str, Any]:
        jobs = request.app.state.research_jobs.list()
        return {"total": len(jobs), "items": [job.as_dict() for job in jobs]}

    @app.get("/research/{report_id}")
    def get_research(
        report_id: str, services: Services = Depends(get_services)
    ) -> dict[str, Any]:
        vault = _require(services.vault_manager, "The research vault")
        path = _resolve_note_path(vault, "research", report_id)
        if path is None:
            raise HTTPException(status_code=404, detail="research report not found")
        return _note_detail(_safe(vault.read_note, path))

    @app.post("/research")
    def start_research(
        body: ResearchRequest,
        request: Request,
        services: Services = Depends(get_services),
    ) -> dict[str, Any]:
        researcher = _require(services.deep_researcher, "Deep research")
        job = request.app.state.research_jobs.create(body.question, body.depth)
        _kick_off_research(researcher, job, request.app.state.research_jobs, services)
        return {"job": job.as_dict()}

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------

    @app.get("/trading/positions")
    def trading_positions(services: Services = Depends(get_services)) -> dict[str, Any]:
        target = _trading_target(services, "execution", "execution")
        fn = _first_callable(target, ("get_positions", "positions", "list_positions"))
        if fn is None:
            raise HTTPException(status_code=503, detail="Trading positions are not available")
        positions = _safe(fn)
        positions = list(positions or []) if not isinstance(positions, Mapping) else positions
        return {"positions": positions}

    @app.get("/trading/history")
    def trading_history(
        limit: int | None = None, services: Services = Depends(get_services)
    ) -> dict[str, Any]:
        target = _trading_target(services, "execution", "execution")
        fn = _first_callable(target, ("get_history", "history", "get_trades", "trades"))
        if fn is None:
            raise HTTPException(status_code=503, detail="Trade history is not available")
        trades = _safe(fn, limit=limit)
        trades = list(trades or []) if not isinstance(trades, Mapping) else trades
        return {"trades": trades}

    @app.get("/trading/journal")
    def trading_journal(
        limit: int | None = None, services: Services = Depends(get_services)
    ) -> dict[str, Any]:
        target = _trading_target(services, "journal", "journal")
        fn = _first_callable(target, ("list_entries", "entries", "get_entries", "read"))
        if fn is None:
            raise HTTPException(status_code=503, detail="The trading journal is not available")
        entries = _safe(fn, limit=limit)
        entries = list(entries or []) if not isinstance(entries, Mapping) else entries
        return {"entries": entries}

    @app.post("/trading/confirm")
    def trading_confirm(
        body: TradeConfirmRequest, services: Services = Depends(get_services)
    ) -> dict[str, Any]:
        target = _trading_target(services, "execution", "execution")
        fn = _first_callable(
            target, ("confirm_trade", "confirm_order", "confirm", "execute_trade")
        )
        if fn is None:
            raise HTTPException(status_code=503, detail="Trade confirmation is not available")
        result = _safe(fn, trade_id=body.trade_id, confirmed=body.confirmed)
        return {"trade_id": body.trade_id, "confirmed": body.confirmed, "result": result}

    @app.get("/trading/backtest/{strategy_name}")
    def trading_backtest(
        strategy_name: str,
        symbol: str | None = None,
        period: str = "1y",
        services: Services = Depends(get_services),
    ) -> dict[str, Any]:
        target = _trading_target(services, "strategy", "strategy")
        fn = _first_callable(target, ("run_backtest", "backtest"))
        if fn is None:
            raise HTTPException(status_code=503, detail="Backtesting is not available")
        result = _safe(fn, strategy=strategy_name, symbol=symbol, period=period)
        return {"strategy": strategy_name, "symbol": symbol, "period": period, "result": result}

    # ------------------------------------------------------------------
    # Documents (university)
    # ------------------------------------------------------------------

    @app.post("/documents/upload")
    async def upload_document(
        request: Request,
        filename: str | None = None,
        services: Services = Depends(get_services),
    ) -> dict[str, Any]:
        rag = _require(_university_target(services, "rag"), "The document index")
        data, name = await _read_upload(request, filename)
        if not data:
            raise HTTPException(status_code=400, detail="no file in the request")

        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        destination = UPLOADS_DIR / _safe_filename(name)
        destination.write_bytes(data)

        indexer = _first_callable(rag, ("add_document", "index_document", "index", "add"))
        if indexer is None:
            raise HTTPException(status_code=503, detail="The document index cannot add files")
        result = _safe(indexer, path=str(destination), filename=name)
        return {"indexed": True, "filename": name, "path": str(destination), "result": result}

    @app.get("/documents")
    def list_documents(services: Services = Depends(get_services)) -> dict[str, Any]:
        rag = _require(_university_target(services, "rag"), "The document index")
        fn = _first_callable(rag, ("list_documents", "documents", "list", "list_indexed"))
        if fn is None:
            raise HTTPException(status_code=503, detail="The document index cannot list files")
        documents = _safe(fn)
        documents = list(documents or []) if not isinstance(documents, Mapping) else documents
        return {"documents": documents}

    @app.delete("/documents/{document_id}")
    def delete_document(
        document_id: str, services: Services = Depends(get_services)
    ) -> dict[str, Any]:
        rag = _require(_university_target(services, "rag"), "The document index")
        fn = _first_callable(rag, ("remove_document", "delete_document", "remove", "delete"))
        if fn is None:
            raise HTTPException(status_code=503, detail="The document index cannot remove files")
        removed = _safe(fn, document_id=document_id, id=document_id)
        if not removed:
            raise HTTPException(status_code=404, detail="document not found")
        return {"deleted": True, "id": document_id}

    @app.post("/documents/quiz")
    def generate_quiz(
        body: QuizRequest, services: Services = Depends(get_services)
    ) -> dict[str, Any]:
        quiz = _require(_university_target(services, "quiz"), "The quiz generator")
        fn = _first_callable(quiz, ("generate_quiz", "generate", "build_quiz"))
        if fn is None:
            raise HTTPException(status_code=503, detail="The quiz generator is not available")
        result = _safe(
            fn,
            topic=body.topic,
            documents=body.documents,
            num_questions=body.num_questions,
        )
        return {"quiz": result}

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------

    @app.post("/system/gaming-mode/toggle")
    def toggle_gaming_mode(
        request: Request,
        enabled: bool | None = None,
        body: GamingToggleRequest | None = None,
    ) -> dict[str, Any]:
        if body is not None and body.enabled is not None:
            enabled = body.enabled
        return request.app.state.mode_controller.toggle_gaming(enabled)

    @app.post("/system/shutdown")
    def system_shutdown(
        background: BackgroundTasks, services: Services = Depends(get_services)
    ) -> dict[str, Any]:
        background.add_task(_shutdown, services)
        return {"status": "shutting_down"}

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        services: Services = websocket.app.state.services
        manager = services.ws_manager
        await manager.connect(websocket)
        try:
            # Send the current state immediately so the UI is not blank until
            # the first periodic push.
            status = await asyncio.to_thread(
                _status_payload, services, websocket.app.state.mode_controller
            )
            await websocket.send_text(json.dumps({"type": "system_status", **status}))
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:  # pragma: no cover - transport-level noise
            logger.debug("websocket closed unexpectedly", exc_info=True)
        finally:
            manager.disconnect(websocket)

    return app


# ---------------------------------------------------------------------------
# Background work
# ---------------------------------------------------------------------------


def _kick_off_research(
    researcher: Any, job: ResearchJob, jobs: "ResearchJobs", services: Services
) -> None:
    """Start a research run on a worker thread and report back over ``/ws``.

    The run is deliberately not awaited: a deep research pass takes minutes,
    and the request that started it must return immediately so the UI can show
    the job under ``GET /research/jobs``.
    """
    def on_progress(message: str) -> None:
        jobs.update(job.id, progress=message)
        # Push as it happens so the UI's progress indicator is live.
        services.ws_manager.broadcast(
            {"type": "research_progress", "id": job.id, "progress": message}
        )

    def worker() -> None:
        try:
            report = _run_research(researcher, job.question, job.depth, on_progress)
        except Exception as exc:
            logger.exception("research job %s failed", job.id)
            failed = jobs.update(
                job.id, status="failed", error=str(exc), finished=time.time()
            )
            services.ws_manager.emit(
                "research_complete", report=(failed or job).as_dict()
            )
            return

        if report is None:
            failed = jobs.update(
                job.id,
                status="failed",
                error="research produced no report",
                finished=time.time(),
            )
            services.ws_manager.emit(
                "research_complete", report=(failed or job).as_dict()
            )
            return

        data = _report_dict(report)
        data.setdefault("id", job.id)
        jobs.update(job.id, status="complete", report=data, finished=time.time())
        services.ws_manager.research_complete(data)

    threading.Thread(
        target=worker, name="atlas-research-job", daemon=True
    ).start()


def _run_research(
    researcher: Any, question: str, depth: int, progress: Callable[[str], None] | None = None
) -> Any:
    """Run a deep researcher's ``research`` whether it is sync or async.

    ``progress`` is forwarded only when the researcher accepts it, so older
    collaborators keep working unchanged.
    """
    if hasattr(researcher, "research_async"):
        done = threading.Event()
        box: dict[str, Any] = {}

        def callback(report: Any) -> None:
            box["report"] = report
            done.set()

        _call_any(researcher.research_async, question, callback, depth=depth, progress=progress)
        # A research pass can legitimately take minutes; this only guards
        # against a researcher that silently never calls back.
        if not done.wait(timeout=3600):
            raise TimeoutError("research did not finish within an hour")
        return box.get("report")

    research = getattr(researcher, "research", None)
    if research is None:
        raise RuntimeError("deep researcher exposes no research method")
    result = _call_any(research, question, depth, progress=progress)
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result


def _shutdown(services: Services) -> None:
    """Best-effort graceful stop of everything the API knows about."""
    for target, names in (
        (services.agent, ("shutdown", "close")),
        (services.voice_pipeline, ("shutdown", "close")),
    ):
        fn = _first_callable(target, names)
        if fn is not None:
            try:
                _call_any(fn)
            except Exception:
                logger.exception("shutdown step failed")
    manager = services.llm_manager
    fn = _first_callable(manager, ("stop_all", "shutdown"))
    if fn is not None:
        try:
            _call_any(fn)
        except Exception:
            logger.exception("could not stop the model servers")
    logger.info("Atlas API shutdown requested")


async def _read_upload(request: Request, filename: str | None) -> tuple[bytes, str]:
    """Read an uploaded file, from multipart when available or raw otherwise.

    FastAPI's multipart parser needs the optional ``python-multipart`` package;
    when it is missing the raw body is accepted instead, which keeps the
    endpoint usable (e.g. ``curl --data-binary``) rather than failing flat.
    """
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/"):
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "multipart upload needs the python-multipart package "
                    "(pip install python-multipart)"
                ),
            ) from exc
        for field_name, value in form.items():
            if hasattr(value, "read"):
                name = filename or getattr(value, "filename", None) or field_name
                return value.file.read(), str(name)
        raise HTTPException(status_code=400, detail="no file field in the multipart body")

    data = await request.body()
    return data if isinstance(data, bytes) else b"", (filename or "upload.bin")


def _safe_filename(name: str) -> str:
    """A conservative filename that cannot escape the uploads directory."""
    cleaned = Path(name or "upload.bin").name
    cleaned = cleaned.replace("\x00", "").strip() or "upload.bin"
    return cleaned


def _find_pending_proposal(builder: Any, skill_name: str) -> Any | None:
    """Find a queued proposal by name, whatever shape the builder returns."""
    pending_fn = _first_callable(
        builder, ("pending_refinements", "pending", "pending_proposals")
    )
    if pending_fn is None:
        return None
    try:
        pending = list(pending_fn() or [])
    except Exception:
        logger.exception("could not read pending skills")
        return None
    for entry in pending:
        if isinstance(entry, Mapping):
            name = str(entry.get("name") or entry.get("skill_name") or "")
            if name == skill_name:
                return entry.get("proposal") or entry
        else:
            name = str(getattr(entry, "name", "") or "")
            if name == skill_name:
                return getattr(entry, "proposal", entry)
    return None


def _approve_proposal(builder: Any, proposal: Any, skill_name: str) -> bool:
    """Register a pending proposal, trying the builder's known entry points."""
    register = _first_callable(builder, ("register",))
    if register is not None:
        try:
            _call_any(register, proposal, name=skill_name, overwrite=True)
            return True
        except Exception:
            logger.exception("register() failed while approving %s", skill_name)

    present = _first_callable(builder, ("present_for_approval",))
    if present is not None:
        try:
            _call_any(present, proposal, name=skill_name, approved=True)
            return True
        except Exception:
            logger.exception("present_for_approval() failed while approving %s", skill_name)
    return False


def run() -> None:
    """Serve the app on localhost. Intended for ``python -m api.server``."""
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=FASTAPI_PORT, log_level="info")


#: Import-time app for ``uvicorn api.server:app``. The daemon should build its
#: own with :func:`create_app` and real :class:`Services`.
app = create_app()


if __name__ == "__main__":
    run()
