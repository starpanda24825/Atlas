"""
Atlas — the daemon.

The single entry point for the whole system. :class:`AtlasDaemon` constructs
every component in dependency order, starts the FastAPI server on a background
thread, arms the resource governor, and then blocks in the wake-word loop,
handing each detected "hey atlas" to the voice pipeline and each utterance to
the agent.

Start-up order (it matters — later components consume earlier ones)::

    config + logging
    LLMServerManager -> start the fast server, wait for /health
    ChromaStore
    VaultManager      -> index_vault()
    Mem0Manager
    SkillRegistry
    SkillSandbox + SkillBuilder (attached back to the registry)
    SearchRouter, QuickSearch, PageExtractor, DeepResearcher
    AtlasAgent (given a search facade exposing classify/quick_search/deep_research)
    WebSocketManager
    FastAPI server (uvicorn, background thread)
    ResourceGovernor
    VoicePipeline
    WakeWordDaemon
    APScheduler (morning briefing)
    startup greeting

Every heavy piece is built through an overridable ``_build_*`` method, so a
test (or a degraded environment) can substitute a stand-in without touching the
sequencing. A component that fails to start is logged and left as ``None`` —
a machine with no microphone, or no model weights, should still bring up the
API rather than dying at import.

Run it with::

    python3 -m core.daemon

or, from code::

    daemon = AtlasDaemon()
    daemon.initialize()
    daemon.run()          # blocks until SIGINT/SIGTERM
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from core.config import (
    AGENT_NAME,
    BRIEFING_HOUR,
    BRIEFING_MINUTE,
    FASTAPI_PORT,
    LOGS_DIR,
    WAKE_WORDS,
    validate_config,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_PATH: Path = LOGS_DIR / "atlas.log"
LOG_MAX_BYTES: int = 50 * 1024 * 1024  # 50 MB per file
LOG_BACKUP_COUNT: int = 5

_HUMAN_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

#: Attributes every LogRecord carries; anything else is a structured extra.
_RESERVED_RECORD_KEYS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime"}

_logging_configured = False
_logging_lock = threading.Lock()


class JsonFormatter(logging.Formatter):
    """One JSON object per line — the format the log file uses."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created).astimezone().isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Anything passed via ``extra=`` becomes a field of its own.
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_KEYS or key.startswith("_"):
                continue
            try:
                json.dumps(value, default=str)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = str(value)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: int = logging.INFO) -> Path:
    """Set up JSON file logging (rotating, 50 MB) plus a human console.

    Idempotent: a second call is a no-op, so importing the daemon and running it
    does not double every log line.
    """
    global _logging_configured
    with _logging_lock:
        if _logging_configured:
            return LOG_PATH

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger()
        root.setLevel(level)

        for handler in list(root.handlers):
            root.removeHandler(handler)

        file_handler = logging.handlers.RotatingFileHandler(
            LOG_PATH,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(JsonFormatter())
        file_handler.setLevel(level)
        root.addHandler(file_handler)

        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_HUMAN_FORMAT))
        console.setLevel(level)
        root.addHandler(console)

        # These are chatty at INFO and drown out our own lines.
        for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "openai", "apscheduler"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        _logging_configured = True
        return LOG_PATH


# ---------------------------------------------------------------------------
# Search facade
# ---------------------------------------------------------------------------


class SearchFacade:
    """Adapts the search stack to the two method names the agent duck-types.

    :class:`~search.router.SearchRouter` only classifies; the agent expects its
    ``search_router`` collaborator to also answer ``quick_search`` and
    ``deep_research``. One small adapter keeps each module's own interface
    intact rather than contorting either.
    """

    def __init__(
        self,
        router: Any = None,
        quick_search: Any = None,
        deep_researcher: Any = None,
    ) -> None:
        self.router = router
        self.quick = quick_search
        self.deep = deep_researcher

    def classify(self, query: str) -> Any:
        fn = getattr(self.router, "classify", None)
        if callable(fn):
            try:
                return fn(query)
            except Exception:
                logger.warning("search classification failed", exc_info=True)
        return None

    def quick_search(self, query: str) -> str:
        fn = getattr(self.quick, "search", None)
        if not callable(fn):
            return "Web search is not available."
        try:
            return str(fn(query))
        except Exception as exc:
            logger.warning("quick search failed: %s", exc)
            return f"Web search failed: {exc}"

    def deep_research(self, question: str) -> str:
        fn = getattr(self.deep, "deep_research", None)
        if not callable(fn):
            return self.quick_search(question)
        try:
            report = fn(question)
        except Exception as exc:
            logger.warning("deep research failed: %s", exc)
            return f"Deep research failed: {exc}"
        # A ResearchReport carries the full text; fall back to str() for a
        # collaborator that returns something simpler.
        return str(getattr(report, "full_report", None) or report)

    def search(self, query: str) -> str:
        return self.quick_search(query)

    def research(self, question: str) -> str:
        return self.deep_research(question)


# ---------------------------------------------------------------------------
# The daemon
# ---------------------------------------------------------------------------


class AtlasDaemon:
    """Builds, runs and tears down the whole of Atlas."""

    def __init__(self, *, log_level: int = logging.INFO) -> None:
        self.log_level = log_level
        self.log_path: Path | None = None

        # Collaborators, filled in by initialize().
        self.llm_manager: Any = None
        self.store: Any = None
        self.vault_manager: Any = None
        self.memory_manager: Any = None
        self.skill_registry: Any = None
        self.skill_sandbox: Any = None
        self.skill_builder: Any = None
        self.search_router: Any = None
        self.quick_search: Any = None
        self.page_extractor: Any = None
        self.deep_researcher: Any = None
        self.search_facade: Any = None
        self.agent: Any = None
        self.ws_manager: Any = None
        self.api_app: Any = None
        self.api_server: Any = None
        self.api_thread: threading.Thread | None = None
        self.resource_governor: Any = None
        self.voice_pipeline: Any = None
        self.wake_word_daemon: Any = None
        self.scheduler: Any = None

        self._shutdown = threading.Event()
        self._initialised = False
        self._shutdown_done = False
        self._shutdown_lock = threading.Lock()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<AtlasDaemon initialised={self._initialised} "
            f"shutting_down={self._shutdown.is_set()}>"
        )

    # ==================================================================
    # Initialisation
    # ==================================================================

    def initialize(self) -> "AtlasDaemon":
        """Build every component in order. Returns ``self``."""
        if self._initialised:
            return self

        self.log_path = configure_logging(self.log_level)
        logger.info("Atlas starting", extra={"event": "startup", "log": str(self.log_path)})

        # 1. Config.
        try:
            self._validate_config()
        except Exception:
            logger.warning("config validation could not complete", exc_info=True)

        # 2-3. Model servers.
        self.llm_manager = self._safe("llm manager", self._build_llm_manager)
        self._start_fast_model()

        # 4. Vector store.
        self.store = self._safe("chroma store", self._build_store)

        # 5. Vault (+ index).
        self.vault_manager = self._safe("vault manager", self._build_vault)
        self._index_vault()

        # 6. Structured memory.
        self.memory_manager = self._safe("memory manager", self._build_memory)

        # 7-8. Skills.
        self.skill_registry = self._safe("skill registry", self._build_registry)
        self.skill_sandbox = self._safe("skill sandbox", self._build_sandbox)
        self.skill_builder = self._safe("skill builder", self._build_builder)
        self._attach_builder()

        # 9. Search.
        self.search_router = self._safe("search router", self._build_search_router)
        self.quick_search = self._safe("quick search", self._build_quick_search)
        self.page_extractor = self._safe("page extractor", self._build_page_extractor)
        self.deep_researcher = self._safe("deep researcher", self._build_deep_researcher)
        self.search_facade = SearchFacade(
            self.search_router, self.quick_search, self.deep_researcher
        )

        # 10. Agent.
        self.agent = self._safe("agent", self._build_agent)

        # 11. WebSocket fan-out.
        self.ws_manager = self._safe("websocket manager", self._build_websocket)

        # 12. Governor object. It is built before the API so the API's Services
        # container can hold a reference to it; its monitor thread is started
        # only once the voice pipeline and wake word exist, further down.
        self.resource_governor = self._safe("resource governor", self._build_governor)

        # 13. HTTP API.
        self._start_api()

        # 14-15. Voice.
        self.voice_pipeline = self._safe("voice pipeline", self._build_voice)
        self.wake_word_daemon = self._safe("wake word daemon", self._build_wake_word)
        self._start_governor()

        # 16. Scheduled work.
        self.scheduler = self._safe("scheduler", self._build_scheduler)

        # 17. Greeting.
        self._safe("startup greeting", self.run_startup_greeting)

        self._initialised = True
        logger.info(
            "Atlas ready",
            extra={"event": "ready", "voice": self.voice_pipeline is not None,
                   "wake_word": self.wake_word_daemon is not None},
        )
        return self

    def _validate_config(self) -> None:
        ok = validate_config(verbose=False)
        if not ok:
            logger.warning(
                "config validation reported problems — continuing in degraded mode",
                extra={"event": "config_warning"},
            )

    # -- builders (overridable for tests) ------------------------------

    def _build_llm_manager(self) -> Any:
        from core.llm_manager import LLMServerManager

        return LLMServerManager()

    def _build_store(self) -> Any:
        from memory.chroma_store import get_chroma_store

        return get_chroma_store()

    def _build_vault(self) -> Any:
        from memory.vault_manager import VaultManager

        return VaultManager(store=self.store)

    def _build_memory(self) -> Any:
        from memory.mem0_manager import Mem0Manager

        return Mem0Manager(store=self.store)

    def _build_registry(self) -> Any:
        from skills.registry import SkillRegistry

        return SkillRegistry(store=self.store, builder=self.skill_builder)

    def _build_sandbox(self) -> Any:
        from skills.sandbox import get_sandbox

        return get_sandbox()

    def _build_builder(self) -> Any:
        from skills.builder import SkillBuilder

        return SkillBuilder(
            self.llm_manager,
            sandbox=self.skill_sandbox,
            registry=self.skill_registry,
        )

    def _build_search_router(self) -> Any:
        from search.router import SearchRouter

        return SearchRouter(self.llm_manager)

    def _build_quick_search(self) -> Any:
        from search.quick_search import QuickSearch

        return QuickSearch()

    def _build_page_extractor(self) -> Any:
        from search.extractors import PageExtractor

        return PageExtractor()

    def _build_deep_researcher(self) -> Any:
        from search.deep_research import DeepResearcher

        return DeepResearcher(
            self.llm_manager, self.quick_search, self.page_extractor, self.vault_manager
        )

    def _build_agent(self) -> Any:
        from core.agent import AtlasAgent

        return AtlasAgent(
            self.llm_manager,
            memory_manager=self.memory_manager,
            skill_registry=self.skill_registry,
            search_router=self.search_facade,
        )

    def _build_websocket(self) -> Any:
        from api.websocket_manager import get_websocket_manager

        return get_websocket_manager()

    def _build_governor(self) -> Any:
        from core.resource_governor import ResourceGovernor

        return ResourceGovernor(
            llm_manager=self.llm_manager,
            ws_manager=self.ws_manager,
            skill_builder=self.skill_builder,
            skill_registry=self.skill_registry,
            memory_manager=self.memory_manager,
            autostart=False,
        )

    def _build_voice(self) -> Any:
        from voice.pipeline import VoicePipeline

        return VoicePipeline()

    def _build_wake_word(self) -> Any:
        from voice.wake_word import WakeWordDaemon

        return WakeWordDaemon()

    def _build_scheduler(self) -> Any:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(
            self._run_morning_briefing,
            CronTrigger(hour=BRIEFING_HOUR, minute=BRIEFING_MINUTE),
            id="morning_briefing",
            replace_existing=True,
        )
        scheduler.start()
        logger.info(
            "morning briefing scheduled for %02d:%02d", BRIEFING_HOUR, BRIEFING_MINUTE
        )
        return scheduler

    # -- wiring helpers ------------------------------------------------

    def _start_fast_model(self) -> None:
        """Launch the fast server and wait for it, without blocking forever."""
        manager = self.llm_manager
        if manager is None:
            return
        try:
            manager.start_fast_server()
        except Exception:
            logger.warning("could not start the fast model server", exc_info=True)
            return
        try:
            ready = manager.wait_for_server_ready(manager.fast_port)
        except Exception:
            logger.warning("waiting for the fast model failed", exc_info=True)
            return
        if ready:
            logger.info("fast model server ready on port %s", manager.fast_port)
        else:
            logger.warning("fast model server did not become ready in time")

    def _index_vault(self) -> None:
        vault = self.vault_manager
        index = getattr(vault, "index_vault", None)
        if not callable(index):
            return
        try:
            report = index()
            logger.info("vault indexed", extra={"event": "index_vault", **(report or {})})
        except Exception:
            logger.warning("vault indexing failed", exc_info=True)

    def _attach_builder(self) -> None:
        registry, builder = self.skill_registry, self.skill_builder
        attach = getattr(registry, "attach_builder", None)
        if callable(attach) and builder is not None:
            try:
                attach(builder)
            except Exception:
                logger.warning("could not attach the builder to the registry", exc_info=True)

    def _start_api(self) -> None:
        """Build the FastAPI app and serve it on a background uvicorn thread."""
        try:
            from api.server import Services, create_app
        except Exception:
            logger.warning("could not build the API", exc_info=True)
            return

        services = Services(
            llm_manager=self.llm_manager,
            agent=self.agent,
            memory_manager=self.memory_manager,
            skill_registry=self.skill_registry,
            skill_builder=self.skill_builder,
            vault_manager=self.vault_manager,
            search_router=self.search_router,
            quick_search=self.quick_search,
            deep_researcher=self.deep_researcher,
            trading=self._trading_namespace(),
            university=self._university_namespace(),
            voice_pipeline=self.voice_pipeline,
            wake_word=self.wake_word_daemon,
            resource_governor=self.resource_governor,
            ws_manager=self.ws_manager,
            mode="idle",
        )
        try:
            self.api_app = create_app(services)
        except Exception:
            logger.warning("could not create the API app", exc_info=True)
            return

        try:
            import uvicorn
        except Exception:
            logger.warning("uvicorn is not available — the API will not serve", exc_info=True)
            return

        config = uvicorn.Config(
            self.api_app,
            host="127.0.0.1",
            port=FASTAPI_PORT,
            log_level="warning",
            workers=1,
        )
        server = uvicorn.Server(config)
        # ``install_signal_handlers`` must not run off the main thread.
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self.api_server = server
        thread = threading.Thread(target=server.run, name="atlas-api", daemon=True)
        self.api_thread = thread
        thread.start()

        deadline = time.time() + 15.0
        while time.time() < deadline and not getattr(server, "started", False):
            if not thread.is_alive():
                logger.warning("the API thread exited before it started serving")
                return
            time.sleep(0.05)
        if getattr(server, "started", False):
            logger.info("API listening on 127.0.0.1:%d", FASTAPI_PORT)
        else:
            logger.warning("API did not report as started within 15s")

    def _trading_namespace(self) -> Any:
        try:
            import modules.trading.execution as execution
            import modules.trading.journal as journal
            import modules.trading.strategy as strategy

            return SimpleNamespace(execution=execution, journal=journal, strategy=strategy)
        except Exception:
            logger.debug("trading modules not importable", exc_info=True)
            return None

    def _university_namespace(self) -> Any:
        try:
            import modules.university.mun as mun
            import modules.university.quiz as quiz
            import modules.university.rag as rag

            return SimpleNamespace(rag=rag, quiz=quiz, mun=mun)
        except Exception:
            logger.debug("university modules not importable", exc_info=True)
            return None

    def _start_governor(self) -> None:
        governor = self.resource_governor
        if governor is None:
            return
        governor.voice_pipeline = self.voice_pipeline
        governor.wake_word = self.wake_word_daemon
        try:
            governor.start()
        except Exception:
            logger.warning("could not start the resource governor", exc_info=True)

    def _safe(self, label: str, builder: Callable[[], Any]) -> Any:
        """Run a builder, logging and swallowing any failure."""
        try:
            return builder()
        except Exception:
            logger.warning("could not initialise the %s", label, exc_info=True)
            return None

    # ==================================================================
    # Main loop
    # ==================================================================

    def run(self) -> None:
        """Block in the wake-word loop until shutdown is requested."""
        if not self._initialised:
            self.initialize()
        self._install_signal_handlers()

        daemon = self.wake_word_daemon
        if daemon is None:
            logger.warning("no wake word daemon — Atlas has no way to be woken")
            # Without an ear, just wait for a signal.
            self._shutdown.wait()
            self.shutdown()
            return

        logger.info("Atlas is listening for '%s'", WAKE_WORDS[0] if WAKE_WORDS else "the wake word")
        while not self._shutdown.is_set():
            listen = getattr(daemon, "listen_forever", None)
            if not callable(listen):
                logger.error("wake word daemon has no listen_forever() — stopping")
                break
            try:
                listen(self._on_wake)
            except Exception:
                logger.exception("wake word loop failed")
                break
            # listen_forever returns when stopped; exit unless it is a spurious
            # return with no shutdown pending.
            if self._shutdown.is_set():
                break

        self.shutdown()

    # -- wake handling -------------------------------------------------

    def _on_wake(self) -> None:
        """A wake word was heard: start a voice session."""
        if self._shutdown.is_set():
            return
        logger.info("wake word detected — starting a session")

        self._play_cue()
        # A fresh session starts clean: any half-confirmed trade or reminder
        # from last time must not be carried onto this conversation.
        self._clear_pending_actions()
        self._enter_voice_mode()

        voice = self.voice_pipeline
        session = getattr(voice, "full_duplex_session", None)
        if not callable(session):
            logger.warning("voice pipeline is unavailable — cannot run a session")
            self._broadcast_idle()
            return
        try:
            session(self._on_utterance)
        except Exception:
            logger.exception("voice session failed")
        finally:
            if not self._shutdown.is_set():
                logger.info("session ended — returning to idle")
                self._broadcast_idle()

    def _on_utterance(self, text: str) -> str | None:
        """Handle one user turn: broadcast it, think, broadcast the reply."""
        cleaned = (text or "").strip()
        if not cleaned:
            return None

        self._broadcast({"type": "transcript", "text": cleaned})
        intent = self._classify(cleaned)
        if intent is not None:
            self._broadcast({"type": "search_intent", "intent": intent})

        agent = self.agent
        think = getattr(agent, "think", None)
        if not callable(think):
            reply = "My reasoning engine is not available right now."
            self._broadcast({"type": "response", "text": reply})
            return reply

        history = getattr(agent, "session_history", None)
        try:
            reply = think(cleaned, history)
        except Exception:
            logger.exception("agent turn failed")
            reply = "Sorry, something went wrong while I was thinking."
        reply = (reply or "").strip() or None
        if reply:
            self._broadcast({"type": "response", "text": reply})
        return reply

    def _classify(self, text: str) -> str | None:
        """Run the search router and return the intent's label, if any."""
        classify = getattr(self.search_router, "classify", None)
        if not callable(classify):
            return None
        try:
            intent = classify(text)
        except Exception:
            logger.debug("intent classification failed", exc_info=True)
            return None
        if intent is None:
            return None
        label = getattr(intent, "value", None) or str(intent)
        if label and label.upper() != "NONE":
            logger.info("search intent: %s", label)
            return str(label)
        return None

    def _clear_pending_actions(self) -> None:
        reset = getattr(self.agent, "reset", None)
        if callable(reset):
            try:
                reset()
                return
            except Exception:
                logger.debug("could not reset the agent", exc_info=True)
        # Fall back to clearing just the pending confirmation.
        pending = getattr(self.agent, "_pending_action", None)
        if pending is not None:
            try:
                self.agent._pending_action = None  # type: ignore[attr-defined]
            except Exception:
                logger.debug("could not clear the pending action", exc_info=True)

    def _play_cue(self) -> None:
        """Play the listening chime, if the pipeline exposes one."""
        voice = self.voice_pipeline
        for name in ("play_cue", "cue", "_play_cue"):
            fn = getattr(voice, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    logger.debug("listening cue failed", exc_info=True)
                return

    def _enter_voice_mode(self) -> None:
        governor = self.resource_governor
        set_mode = getattr(governor, "set_mode", None)
        if callable(set_mode):
            try:
                set_mode("voice")
                return  # the governor broadcasts the change
            except Exception:
                logger.debug("governor mode switch failed", exc_info=True)
        self._broadcast({"type": "mode_change", "mode": "voice", "previous": None})

    def _broadcast_idle(self) -> None:
        governor = self.resource_governor
        set_mode = getattr(governor, "set_mode", None)
        if callable(set_mode):
            try:
                set_mode("idle")
                return
            except Exception:
                logger.debug("governor idle switch failed", exc_info=True)
        self._broadcast({"type": "mode_change", "mode": "idle", "previous": "voice"})

    def _broadcast(self, message: dict[str, Any]) -> None:
        ws = self.ws_manager
        fn = getattr(ws, "broadcast", None)
        if callable(fn):
            try:
                fn(message)
            except Exception:
                logger.debug("broadcast failed", exc_info=True)

    # ==================================================================
    # Scheduled work + greeting
    # ==================================================================

    def _run_morning_briefing(self) -> None:
        """The scheduled briefing: ask the agent and speak/broadcast its answer."""
        if self._shutdown.is_set():
            return
        logger.info("running the morning briefing")
        prompt = (
            "Give me my morning briefing: greet me, the date and time, the "
            "weather, any calendar or reminders you know about, and one line of "
            "top news. Keep it short and speakable."
        )
        agent = self.agent
        think = getattr(agent, "think", None)
        text = ""
        if callable(think):
            try:
                text = (think(prompt) or "").strip()
            except Exception:
                logger.exception("morning briefing failed")
        if not text:
            text = f"Good morning. It's {datetime.now().strftime('%A %d %B, %H:%M')}."
        self._broadcast({"type": "response", "text": text})
        self._speak(text)

    def run_startup_greeting(self) -> None:
        """Say hello once everything is up."""
        name = WAKE_WORDS[0].title() if WAKE_WORDS else AGENT_NAME
        greeting = f"{AGENT_NAME} is online. Say '{name}' when you need me."
        logger.info("startup greeting: %s", greeting)
        self._broadcast({"type": "response", "text": greeting})
        self._speak(greeting, background=True)

    def _speak(self, text: str, *, background: bool = False) -> None:
        """Speak ``text`` if a voice pipeline is available and idle."""
        voice = self.voice_pipeline
        speak = getattr(voice, "speak", None)
        if not callable(speak) or not text or self._shutdown.is_set():
            return
        # Never talk over a game: the governor stops the voice pipeline there
        # anyway, but a greeting should not try to restart it.
        if getattr(self.resource_governor, "mode", None) == "gaming":
            return

        def worker() -> None:
            try:
                speak(text)
            except Exception:
                logger.debug("speaking failed", exc_info=True)

        if background:
            threading.Thread(target=worker, name="atlas-speak", daemon=True).start()
        else:
            worker()

    # ==================================================================
    # Shutdown
    # ==================================================================

    def request_shutdown(self, signum: int | None = None) -> None:
        """Ask the main loop to exit. Safe from a signal handler or a thread."""
        if self._shutdown.is_set():
            return
        logger.info("shutdown requested%s", f" (signal {signum})" if signum else "")
        self._shutdown.set()
        # Unblock listen_forever so the main loop can return.
        stopper = getattr(self.wake_word_daemon, "stop", None)
        if callable(stopper):
            try:
                stopper()
            except Exception:
                logger.debug("wake word stop failed", exc_info=True)

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            logger.debug("not the main thread — skipping signal handlers")
            return
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, lambda signum, _frame: self.request_shutdown(signum))
            except (ValueError, OSError):  # pragma: no cover - platform dependent
                logger.debug("could not install a handler for %s", sig)

    def shutdown(self) -> None:
        """Tear everything down in reverse order. Idempotent."""
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True
        self._shutdown.set()
        logger.info("Atlas shutting down", extra={"event": "shutdown"})

        self._safe("wake word shutdown", self._stop_wake_word)
        self._safe("voice shutdown", self._stop_voice)
        self._safe("scheduler shutdown", self._stop_scheduler)
        self._safe("governor shutdown", self._stop_governor)
        self._safe("API shutdown", self._stop_api)
        self._safe("model shutdown", self._stop_models)
        self._safe("memory flush", self._flush_memory)

        logger.info("Atlas stopped")

    def _stop_wake_word(self) -> None:
        fn = getattr(self.wake_word_daemon, "stop", None)
        if callable(fn):
            fn()

    def _stop_voice(self) -> None:
        fn = getattr(self.voice_pipeline, "shutdown", None)
        if callable(fn):
            fn()

    def _stop_scheduler(self) -> None:
        fn = getattr(self.scheduler, "shutdown", None)
        if callable(fn):
            fn(wait=False)

    def _stop_governor(self) -> None:
        fn = getattr(self.resource_governor, "stop", None)
        if callable(fn):
            fn()

    def _stop_api(self) -> None:
        server = self.api_server
        if server is not None:
            server.should_exit = True
        thread = self.api_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)
        self.api_server = None
        self.api_thread = None

    def _stop_models(self) -> None:
        fn = getattr(self.llm_manager, "stop_all", None)
        if callable(fn):
            fn()

    def _flush_memory(self) -> None:
        """Let in-flight extraction finish before the process exits."""
        agent = self.agent
        fn = getattr(agent, "shutdown", None)
        if callable(fn):
            try:
                fn(wait_for_memory=True)
            except Exception:
                logger.debug("agent flush failed", exc_info=True)
        wait = getattr(self.memory_manager, "wait_for_pending", None)
        if callable(wait):
            try:
                wait()
            except Exception:
                logger.debug("memory manager flush failed", exc_info=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    daemon = AtlasDaemon()
    daemon.initialize()
    daemon.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
