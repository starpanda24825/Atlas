"""
Atlas — the resource governor.

Atlas shares one GPU with whatever else the user is doing, so it has to get out
of the way on its own. This module owns that judgement: a daemon thread samples
the machine every ``RESOURCE_CHECK_INTERVAL`` seconds and moves Atlas between
five modes:

=========  =========================================================================
``voice``    normal operation — the fast model is resident, nothing else is forced
``study``    RAG work — the deep model is loaded and document retrieval is prioritised
``trading``  market hours — the trading modules are imported and market skills noted
``gaming``   a game is running — every model server and the voice pipeline are stopped
``idle``     light monitoring — the deep model is released, the fast one may stay
=========  =========================================================================

Gaming is the interesting one, and it is *bidirectional*: when a process named in
``GAMING_MODE_PROCESSES`` appears, Atlas drops into gaming mode automatically;
when it exits, Atlas restores whatever mode it was in before. The wake-word
daemon is deliberately never stopped, so "hey atlas" still works from inside a
game. A mode the user set by hand is never auto-exited — only an automatic entry
is undone.

Other signals it acts on:

* **Battery** — below 20% and discharging, the deep model is disabled until the
  machine is charging again or back above 25%.
* **Idle compute** — past ``IDLE_COMPUTE_HOUR`` on AC power, once per day, it
  triggers background skill refinement and memory consolidation.

Every mode change is broadcast to the WebSocket manager, and
:meth:`ResourceGovernor.get_system_stats` backs the ``/status`` endpoint.

Example::

    from core.resource_governor import ResourceGovernor

    governor = ResourceGovernor(
        llm_manager=llm, ws_manager=get_websocket_manager(),
        skill_builder=builder,  memory_manager=memory,
    )
    governor.set_mode("study")          # load the deep model now
    print(governor.get_system_stats())
    governor.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core.config import (
    GAMING_MODE_PROCESSES,
    IDLE_COMPUTE_HOUR,
    RESOURCE_CHECK_INTERVAL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary and tuning
# ---------------------------------------------------------------------------

MODES: tuple[str, ...] = ("voice", "study", "trading", "gaming", "idle")

VOICE_MODE = "voice"
STUDY_MODE = "study"
TRADING_MODE = "trading"
GAMING_MODE = "gaming"
IDLE_MODE = "idle"

#: The mode Atlas assumes when nothing else is known.
DEFAULT_MODE: str = IDLE_MODE

#: The mode restored when an automatic gaming session ends.
DEFAULT_RESUME_MODE: str = VOICE_MODE

#: Battery thresholds. Suspension is lower than resumption on purpose, so a
#: battery hovering on the line does not flap the deep model on and off.
BATTERY_LOW_PERCENT: float = 20.0
BATTERY_RESUME_PERCENT: float = 25.0

#: At most this many failing skills are refined in one idle window. Refinement
#: spins up the deep model, so an idle window should not turn into a marathon.
MAX_SKILLS_PER_IDLE_RUN: int = 3

#: Keywords that mark a skill as market data for trading mode.
MARKET_SKILL_KEYWORDS: tuple[str, ...] = ("market", "trad", "stock", "finance", "ticker")

#: ``nvidia-smi`` is comparatively expensive, and ``/status`` can be polled far
#: more often than the monitor runs, so a GPU reading is reused for this long.
VRAM_CACHE_SECONDS: float = 2.0


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
    """Call ``fn``, dropping keyword arguments its signature does not accept."""
    import inspect

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    parameters = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return fn(*args, **kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in parameters}
    return fn(*args, **accepted)


# ---------------------------------------------------------------------------
# The governor
# ---------------------------------------------------------------------------


class ResourceGovernor:
    """Samples the machine and moves Atlas between modes accordingly.

    Every collaborator is optional and duck-typed, so the governor is useful
    with nothing wired in (it still reports system stats) and gains behaviour as
    the daemon fills it in.
    """

    def __init__(
        self,
        *,
        llm_manager: Any = None,
        ws_manager: Any = None,
        skill_builder: Any = None,
        skill_registry: Any = None,
        memory_manager: Any = None,
        voice_pipeline: Any = None,
        wake_word: Any = None,
        interval: float | None = None,
        gaming_processes: Sequence[str] | None = None,
        idle_hour: int | None = None,
        autostart: bool = True,
    ) -> None:
        self.llm_manager = llm_manager
        self.ws_manager = ws_manager
        self.skill_builder = skill_builder
        self.skill_registry = skill_registry
        self.memory_manager = memory_manager
        self.voice_pipeline = voice_pipeline
        self.wake_word = wake_word

        self.interval = float(interval if interval is not None else RESOURCE_CHECK_INTERVAL)
        self.idle_hour = int(idle_hour if idle_hour is not None else IDLE_COMPUTE_HOUR)
        self.gaming_processes = [
            str(name).strip()
            for name in (gaming_processes if gaming_processes is not None else GAMING_MODE_PROCESSES)
            if str(name).strip()
        ]

        self._lock = threading.RLock()
        self._vram_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._mode: str = DEFAULT_MODE
        #: Mode to return to when an *automatic* gaming session ends.
        self._mode_before_gaming: str = DEFAULT_RESUME_MODE
        #: True while gaming mode was entered automatically, so only that is
        #: undone automatically. A mode the user set by hand stays put.
        self._gaming_auto: bool = False
        self._low_battery: bool = False
        self._rag_priority: bool = False
        self._trading_ready: bool = False
        self._idle_ran_on: date | None = None
        self._maintenance_running: bool = False
        self._last_stats: dict[str, Any] = {}

        # GPU reading cache: (monotonic timestamp, used_mb, total_mb, percent).
        self._vram_cache: tuple[float, float | None, float | None, float | None] | None = None

        if autostart:
            self.start()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ResourceGovernor mode={self._mode} running={self.running}>"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        """Start the monitor thread. Idempotent."""
        with self._lock:
            if self.running:
                return
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._run, name="atlas-resource-governor", daemon=True
            )
            self._thread = thread
        thread.start()
        logger.info(
            "resource governor started (interval %ss, gaming processes: %s)",
            self.interval,
            ", ".join(self.gaming_processes) or "none",
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the monitor thread to stop and wait for it."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._lock:
            self._thread = None
        logger.info("resource governor stopped")

    def _run(self) -> None:
        """The monitor loop: sample, act, sleep. Never lets an error kill it."""
        while not self._stop_event.is_set():
            try:
                self.monitor_once()
            except Exception:
                logger.exception("resource governor monitor iteration failed")
            # Wait on the stop event so stop() does not have to block for a
            # whole interval.
            if self._stop_event.wait(self.interval):
                break

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    @property
    def rag_priority(self) -> bool:
        """True in study mode — retrieval should be preferred over improvisation."""
        with self._lock:
            return self._rag_priority

    @property
    def power_saving(self) -> bool:
        """True while the deep model is disabled because the battery is low."""
        with self._lock:
            return self._low_battery

    def set_mode(self, mode: str, *, auto: bool = False) -> str | None:
        """Switch mode and apply its side effects.

        Returns the new mode name, or ``None`` for an unknown mode. ``auto``
        marks an internally-detected transition (used to remember that gaming
        was entered by the governor, not by the user).
        """
        name = (mode or "").strip().lower()
        if name not in MODES:
            logger.warning("ignoring unknown mode %r", mode)
            return None
        with self._lock:
            previous = self._mode
            changed = previous != name
            self._mode = name

        self._apply_mode(name)
        if changed:
            logger.info("mode %s -> %s%s", previous, name, " (auto)" if auto else "")
            self._broadcast_mode(name, previous)
        return name

    def toggle_gaming(self, enabled: bool | None = None) -> str:
        """Enter/leave gaming mode; ``enabled=None`` flips the current state."""
        if enabled is None:
            enabled = self.mode != GAMING_MODE
        return self.set_mode(GAMING_MODE if enabled else IDLE_MODE) or self.mode

    def _apply_mode(self, name: str) -> None:
        """The side effects of being in ``name``. Called on every transition."""
        if name == GAMING_MODE:
            self._stop_voice()
            self._stop_all_models()
            logger.info("gaming mode: wake word left listening")
            return

        if name == STUDY_MODE:
            with self._lock:
                self._rag_priority = True
            self._ensure_fast()
            self._ensure_deep()
            return

        if name == TRADING_MODE:
            with self._lock:
                self._rag_priority = False
            self._ensure_fast()
            self._ensure_trading_ready()
            return

        if name == VOICE_MODE:
            with self._lock:
                self._rag_priority = False
            self._ensure_fast()
            return

        if name == IDLE_MODE:
            with self._lock:
                self._rag_priority = False
            # Light monitoring: release the deep model, keep the fast one.
            self._stop_deep()
            return

    # ------------------------------------------------------------------
    # The monitor
    # ------------------------------------------------------------------

    def monitor_once(self) -> dict[str, Any]:
        """One sampling pass. Returns the stats it observed."""
        gaming = self._gaming_processes_running()
        self._react_to_gaming(gaming)
        self._check_power()
        self._maybe_run_idle_maintenance()

        stats = self.get_system_stats()
        stats["gaming_processes"] = gaming
        stats["gaming_detected"] = bool(gaming)
        stats["rag_priority"] = self.rag_priority
        with self._lock:
            self._last_stats = stats
        return stats

    def _react_to_gaming(self, gaming: Sequence[str]) -> None:
        """Enter gaming when a game appears; leave it when the game closes."""
        running = bool(gaming)
        current = self.mode

        if running and current != GAMING_MODE:
            self._mode_before_gaming = current if current in MODES else DEFAULT_RESUME_MODE
            self._gaming_auto = True
            logger.info("game detected (%s) — entering gaming mode", ", ".join(gaming))
            self.set_mode(GAMING_MODE, auto=True)
            return

        if not running and current == GAMING_MODE and self._gaming_auto:
            resume = self._mode_before_gaming or DEFAULT_RESUME_MODE
            self._gaming_auto = False
            logger.info("game closed — restoring %s mode", resume)
            self.set_mode(resume, auto=True)

    def _check_power(self) -> None:
        """Disable the deep model on a low, discharging battery."""
        percent, charging = self._read_battery()
        if percent is None:
            return  # desktop with no battery: always "on AC"

        low = percent < BATTERY_LOW_PERCENT and not charging
        recovered = charging or percent >= BATTERY_RESUME_PERCENT

        with self._lock:
            was_low = self._low_battery
            if low and not was_low:
                self._low_battery = True
            elif was_low and recovered:
                self._low_battery = False
            else:
                return
            now_low = self._low_battery
            mode = self._mode

        if now_low:
            logger.warning(
                "battery at %.0f%% and discharging — disabling the deep model",
                percent,
            )
            self._stop_deep()
            self._broadcast_power(percent, charging, power_saving=True)
        else:
            logger.info("power restored (%.0f%%, charging=%s) — deep model re-enabled", percent, charging)
            self._broadcast_power(percent, charging, power_saving=False)
            if mode == STUDY_MODE:
                self._ensure_deep()

    def _maybe_run_idle_maintenance(self) -> None:
        """Trigger once-a-day heavy work past the idle hour, on AC power."""
        now = datetime.now()
        if now.hour < self.idle_hour:
            return
        if not self._on_ac_power():
            return
        with self._lock:
            if self._mode == GAMING_MODE:
                return
            if self._maintenance_running or self._idle_ran_on == now.date():
                return
            # Marked before the thread starts so a long run cannot be double
            # triggered while it is still going.
            self._idle_ran_on = now.date()
            self._maintenance_running = True

        logger.info("idle compute window reached — starting background maintenance")
        threading.Thread(
            target=self._idle_worker, name="atlas-idle-maintenance", daemon=True
        ).start()

    def _idle_worker(self) -> None:
        try:
            self._run_idle_maintenance()
        except Exception:
            logger.exception("idle maintenance failed")
        finally:
            with self._lock:
                self._maintenance_running = False

    def _run_idle_maintenance(self) -> dict[str, Any]:
        """Run skill refinement and memory consolidation once."""
        refined = self._refine_skills()
        consolidated = self._consolidate_memory()
        summary = {"refined": refined, "memory_consolidated": consolidated}
        logger.info(
            "idle maintenance complete (refined %d skill(s), memory consolidated=%s)",
            len(refined),
            consolidated,
        )
        return summary

    # ------------------------------------------------------------------
    # Idle work
    # ------------------------------------------------------------------

    def _refine_skills(self) -> list[str]:
        """Refine skills with recorded failures, up to the per-run cap."""
        builder = self.skill_builder
        refine = _first_callable(builder, ("background_skill_refinement",))
        if refine is None:
            logger.debug("idle maintenance: no skill builder wired in")
            return []
        recorded = _first_callable(builder, ("recorded_errors",))

        refined: list[str] = []
        for name in self._skill_names():
            if len(refined) >= MAX_SKILLS_PER_IDLE_RUN:
                break
            if recorded is not None:
                try:
                    if not recorded(name):
                        continue  # nothing to go on; do not churn a working skill
                except Exception:
                    logger.debug("could not read recorded errors for %r", name, exc_info=True)
                    continue
            try:
                proposal = _call_any(refine, name)
            except Exception:
                logger.warning("refinement of %r failed", name, exc_info=True)
                continue
            if proposal is not None:
                refined.append(name)
        return refined

    def _consolidate_memory(self) -> bool:
        """Flush pending extraction, then consolidate the memory store."""
        memory = self.memory_manager
        if memory is None:
            logger.debug("idle maintenance: no memory manager wired in")
            return False

        # Let any in-flight background extraction finish before consolidating.
        flush = _first_callable(memory, ("wait_for_pending",))
        if flush is not None:
            try:
                _call_any(flush)
            except Exception:
                logger.debug("memory flush failed", exc_info=True)

        consolidate = _first_callable(
            memory,
            ("consolidate", "consolidate_memories", "run_consolidation", "prune", "maintenance"),
        )
        if consolidate is None:
            logger.info("memory consolidation: the memory store exposes no consolidation entry point")
            return False
        try:
            _call_any(consolidate)
        except Exception:
            logger.warning("memory consolidation failed", exc_info=True)
            return False
        return True

    def _skill_names(self) -> list[str]:
        """Names of the registered skills, or [] when there is no registry."""
        registry = self.skill_registry
        if registry is None:
            return []
        lister = _first_callable(registry, ("list_skills",))
        if lister is not None:
            try:
                records = lister() or []
                return [str(getattr(record, "name", record)) for record in records]
            except Exception:
                logger.debug("skill registry listing failed", exc_info=True)
        namer = _first_callable(registry, ("skill_names",))
        if namer is not None:
            try:
                return [str(name) for name in (namer() or [])]
            except Exception:
                logger.debug("skill name listing failed", exc_info=True)
        return []

    # ------------------------------------------------------------------
    # Model / subsystem control (duck-typed)
    # ------------------------------------------------------------------

    def _ensure_fast(self) -> None:
        """Make sure the fast server is up, unless a game is running."""
        manager = self.llm_manager
        if manager is None or self.mode == GAMING_MODE:
            return
        process = getattr(manager, "fast_process", None)
        alive = process is not None and callable(getattr(process, "poll", None)) and process.poll() is None
        if alive:
            return
        starter = _first_callable(manager, ("start_fast_server", "ensure_fast_available", "start_fast"))
        if starter is None:
            return
        try:
            _call_any(starter)
            logger.info("fast model server requested")
        except Exception:
            logger.warning("could not start the fast model server", exc_info=True)

    def _ensure_deep(self) -> bool:
        """Load the deep model, unless the battery has disabled it."""
        if self.power_saving:
            logger.info("deep model requested but the battery is low — staying on the fast model")
            return False
        manager = self.llm_manager
        fn = _first_callable(manager, ("ensure_deep_available",))
        if fn is None:
            return False
        try:
            available = bool(_call_any(fn))
            if available:
                logger.info("deep model server is available")
            return available
        except Exception:
            logger.warning("could not make the deep model available", exc_info=True)
            return False

    def _stop_deep(self) -> None:
        manager = self.llm_manager
        fn = _first_callable(manager, ("stop_deep_server", "stop_deep"))
        if fn is None:
            return
        try:
            _call_any(fn)
            logger.info("deep model server stopped")
        except Exception:
            logger.warning("could not stop the deep model server", exc_info=True)

    def _stop_all_models(self) -> None:
        manager = self.llm_manager
        fn = _first_callable(manager, ("stop_all", "shutdown"))
        if fn is None:
            if manager is not None:
                logger.info("gaming mode: no stop_all on the model manager")
            return
        try:
            _call_any(fn)
            logger.info("model servers stopped")
        except Exception:
            logger.warning("could not stop the model servers", exc_info=True)
        with self._lock:
            self._trading_ready = False

    def _stop_voice(self) -> None:
        voice = self.voice_pipeline
        fn = _first_callable(voice, ("shutdown", "stop", "close"))
        if fn is None:
            return
        try:
            _call_any(fn)
            logger.info("voice pipeline stopped")
        except Exception:
            logger.warning("could not stop the voice pipeline", exc_info=True)

    def _ensure_trading_ready(self) -> bool:
        """Import the trading modules and note the market-data skills available."""
        if self._trading_ready:
            return True
        import importlib

        for module_name in (
            "modules.trading.strategy",
            "modules.trading.execution",
            "modules.trading.journal",
        ):
            try:
                importlib.import_module(module_name)
            except ImportError:
                logger.debug("%s is not available yet", module_name)
        with self._lock:
            self._trading_ready = True

        market_skills = [
            name
            for name in self._skill_names()
            if any(keyword in name.lower() for keyword in MARKET_SKILL_KEYWORDS)
        ]
        logger.info(
            "trading mode: market-data modules loaded%s",
            f" (skills: {', '.join(market_skills)})" if market_skills else "",
        )
        return True

    # ------------------------------------------------------------------
    # System readings
    # ------------------------------------------------------------------

    def get_system_stats(self) -> dict[str, Any]:
        """A snapshot for the ``/status`` endpoint. Never raises."""
        used_mb, total_mb, vram_percent = self._read_vram()
        battery_percent, charging = self._read_battery()
        with self._lock:
            mode = self._mode
            power_saving = self._low_battery
            gaming_auto = self._gaming_auto
        return {
            "vram_used_mb": used_mb,
            "vram_total_mb": total_mb,
            "vram_percent": vram_percent,
            "ram_percent": self._read_ram_percent(),
            "cpu_percent": self._read_cpu_percent(),
            "battery_percent": battery_percent,
            "charging_bool": charging,
            "current_mode": mode,
            "power_saving": power_saving,
            "gaming_auto": gaming_auto,
            "idle_hour": self.idle_hour,
        }

    def _read_vram(self) -> tuple[float | None, float | None, float | None]:
        """Used/total VRAM on GPU 0, cached briefly because it shells out."""
        now = time.monotonic()
        with self._vram_lock:
            cached = self._vram_cache
            if cached is not None and now - cached[0] < VRAM_CACHE_SECONDS:
                return cached[1], cached[2], cached[3]

        used: float | None = None
        total: float | None = None
        percent: float | None = None
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
        except Exception as exc:
            logger.debug("VRAM reading unavailable: %s", exc)
            gpus = []
        if gpus:
            gpu = gpus[0]
            try:
                used = round(float(gpu.memoryUsed))
                total = round(float(gpu.memoryTotal))
                percent = round(used / total * 100, 1) if total else None
            except Exception:
                used = total = percent = None

        with self._vram_lock:
            self._vram_cache = (now, used, total, percent)
        return used, total, percent

    def _read_ram_percent(self) -> float | None:
        try:
            import psutil

            return float(psutil.virtual_memory().percent)
        except Exception as exc:
            logger.debug("RAM reading unavailable: %s", exc)
            return None

    def _read_cpu_percent(self) -> float | None:
        try:
            import psutil

            # interval=None is non-blocking; the first call is 0.0 by design.
            return float(psutil.cpu_percent(interval=None))
        except Exception as exc:
            logger.debug("CPU reading unavailable: %s", exc)
            return None

    def _read_battery(self) -> tuple[float | None, bool]:
        """``(percent, charging)``. Desktops report ``(None, True)`` — always AC."""
        try:
            import psutil

            battery = psutil.sensors_battery()
        except Exception as exc:
            logger.debug("battery reading unavailable: %s", exc)
            return None, True
        if battery is None:
            return None, True
        try:
            return round(float(battery.percent), 1), bool(battery.power_plugged)
        except Exception:
            return None, True

    def _on_ac_power(self) -> bool:
        percent, charging = self._read_battery()
        return True if percent is None else charging

    def _gaming_processes_running(self) -> list[str]:
        """Names from ``GAMING_MODE_PROCESSES`` that are currently running."""
        targets = {name.lower() for name in self.gaming_processes}
        if not targets:
            return []
        try:
            import psutil
        except ImportError:
            return []

        running: set[str] = set()
        try:
            for process in psutil.process_iter(["name"]):
                try:
                    raw = process.info.get("name") or ""
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                name = str(raw).lower()
                if not name:
                    continue
                # Steam's launcher is "steam", but a Windows-style "Steam.exe"
                # should match too.
                stem = name[:-4] if name.endswith(".exe") else name
                for target in targets:
                    if target == stem or target in name:
                        running.add(target)
        except Exception as exc:  # process_iter itself can raise
            logger.debug("process scan failed: %s", exc)
        return sorted(running)

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    def _broadcast_mode(self, mode: str, previous: str) -> None:
        """Tell the UI the mode changed. Falls back to a raw broadcast."""
        manager = self.ws_manager
        if manager is None:
            return
        helper = getattr(manager, "mode_change", None)
        if callable(helper):
            try:
                _call_any(helper, mode, previous=previous)
                return
            except Exception:
                logger.warning("mode_change broadcast failed", exc_info=True)
        broadcast = getattr(manager, "broadcast", None)
        if callable(broadcast):
            try:
                broadcast({"type": "mode_change", "mode": mode, "previous": previous})
            except Exception:
                logger.warning("mode_change broadcast failed", exc_info=True)

    def _broadcast_power(self, percent: float | None, charging: bool, *, power_saving: bool) -> None:
        """Announce a power-state change under its own event type.

        Deliberately not ``system_status``: that shape is owned by the API's
        status loop, and a partial payload here would replace the full one.
        """
        manager = self.ws_manager
        if manager is None:
            return
        broadcast = getattr(manager, "broadcast", None)
        if not callable(broadcast):
            return
        try:
            broadcast(
                {
                    "type": "power_status",
                    "power_saving": power_saving,
                    "battery_percent": percent,
                    "charging_bool": charging,
                }
            )
        except Exception:
            logger.warning("power_status broadcast failed", exc_info=True)


# ---------------------------------------------------------------------------
# Shared instance
# ---------------------------------------------------------------------------

_governor: ResourceGovernor | None = None
_governor_lock = threading.Lock()


def get_resource_governor(**kwargs: Any) -> ResourceGovernor:
    """Process-wide governor, built once with the first caller's collaborators.

    The daemon should own this instance: a second governor would run a second
    monitor thread and apply the same mode side effects twice.
    """
    global _governor
    if _governor is None:
        with _governor_lock:
            if _governor is None:
                _governor = ResourceGovernor(**kwargs)
    return _governor


def reset_resource_governor() -> None:
    """Stop and drop the shared instance (used by tests and daemon shutdown)."""
    global _governor
    with _governor_lock:
        if _governor is not None:
            _governor.stop()
        _governor = None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> int:
    """Exercise the mode machine and readings with fakes. Never runs a thread."""
    import shutil
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        mark = "ok  " if condition else "FAIL"
        print(f"  [{mark}] {label}{f' — {detail}' if detail else ''}")
        if not condition:
            failures.append(label)

    class FakeProcess:
        def __init__(self, alive: bool) -> None:
            self._alive = alive

        def poll(self) -> None:
            return None if self._alive else 0

    class FakeLLM:
        def __init__(self) -> None:
            self.fast_process = None
            self.calls: list[str] = []

        def start_fast_server(self) -> None:
            self.calls.append("start_fast")
            self.fast_process = FakeProcess(True)

        def ensure_deep_available(self) -> bool:
            self.calls.append("ensure_deep")
            return True

        def stop_deep_server(self) -> None:
            self.calls.append("stop_deep")

        def stop_all(self) -> None:
            self.calls.append("stop_all")

    class FakeWS:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def mode_change(self, mode: str, *, previous: str | None = None) -> None:
            self.events.append({"type": "mode_change", "mode": mode, "previous": previous})

        def broadcast(self, message: Mapping[str, Any]) -> None:
            self.events.append(dict(message))

    class FakeVoice:
        def __init__(self) -> None:
            self.stopped = False

        def shutdown(self) -> None:
            self.stopped = True

    workdir = Path(tempfile.mkdtemp(prefix="atlas-governor-"))
    try:
        llm, ws, voice = FakeLLM(), FakeWS(), FakeVoice()
        governor = ResourceGovernor(
            llm_manager=llm,
            ws_manager=ws,
            voice_pipeline=voice,
            gaming_processes=[],
            autostart=False,
        )
        check("starts in idle mode", governor.mode == "idle")
        check("no monitor thread when autostart is off", not governor.running)

        # -- mode side effects ---------------------------------------
        governor.set_mode("study")
        check("study ensures the deep model", "ensure_deep" in llm.calls)
        check("study sets rag_priority", governor.rag_priority)
        check("study broadcasts a mode change", ws.events[-1]["mode"] == "study")

        governor.set_mode("trading")
        check("trading loads the market modules", governor._trading_ready)

        governor.set_mode("gaming")
        check("gaming stops every model server", "stop_all" in llm.calls)
        check("gaming stops the voice pipeline", voice.stopped)

        governor.set_mode("voice")
        check("voice restarts the fast server", "start_fast" in llm.calls)
        check("leaving gaming clears rag_priority", not governor.rag_priority)

        governor.set_mode("idle")
        check("idle releases the deep model", llm.calls[-1] == "stop_deep")

        check("unknown modes are refused", governor.set_mode("nonsense") is None)

        # -- gaming is bidirectional ---------------------------------
        governor2 = ResourceGovernor(gaming_processes=["dota2"], autostart=False)
        governor2.set_mode("study")
        governor2._react_to_gaming(["dota2"])
        check("a game auto-enters gaming", governor2.mode == "gaming")
        governor2._react_to_gaming([])
        check("the game closing restores the previous mode", governor2.mode == "study")

        governor3 = ResourceGovernor(gaming_processes=["dota2"], autostart=False)
        governor3.set_mode("gaming")  # manual
        governor3._react_to_gaming([])
        check("a manual gaming mode is not auto-exited", governor3.mode == "gaming")

        # -- power policy --------------------------------------------
        governor4 = ResourceGovernor(autostart=False)
        governor4.llm_manager = llm
        governor4._read_battery = lambda: (15.0, False)  # type: ignore[assignment]
        governor4._check_power()
        check("low battery disables the deep model", governor4.power_saving)
        check("low battery stops the deep server", llm.calls[-1] == "stop_deep")
        governor4._read_battery = lambda: (80.0, True)  # type: ignore[assignment]
        governor4._check_power()
        check("charging re-enables the deep model", not governor4.power_saving)

        # -- stats ----------------------------------------------------
        governor4._read_battery = lambda: (55.0, True)  # type: ignore[assignment]
        stats = governor4.get_system_stats()
        for key in (
            "vram_used_mb",
            "vram_total_mb",
            "ram_percent",
            "cpu_percent",
            "battery_percent",
            "charging_bool",
            "current_mode",
        ):
            check(f"stats include {key}", key in stats)
        check("stats report the current mode", stats["current_mode"] == "idle")

        # -- idle maintenance is once per day -------------------------
        class FakeBuilder:
            def __init__(self) -> None:
                self.refined: list[str] = []

            def recorded_errors(self, name: str) -> list[str]:
                return ["boom"]

            def background_skill_refinement(self, name: str) -> Any:
                self.refined.append(name)
                return object()

        builder = FakeBuilder()
        governor4.skill_builder = builder
        governor4.skill_registry = type("R", (), {"skill_names": lambda self: ["a", "b", "c", "d"]})()
        governor4.idle_hour = 0
        governor4._on_ac_power = lambda: True  # type: ignore[assignment]
        governor4._maybe_run_idle_maintenance()
        deadline = time.time() + 2.0
        while governor4._maintenance_running and time.time() < deadline:
            time.sleep(0.02)
        check("idle maintenance refines failing skills", len(builder.refined) >= 1, str(builder.refined))
        check("idle maintenance is capped", len(builder.refined) <= MAX_SKILLS_PER_IDLE_RUN)
        ran = governor4._idle_ran_on
        governor4._maybe_run_idle_maintenance()
        check("idle maintenance runs once per day", governor4._idle_ran_on == ran)

        check("the monitor iteration returns stats", "current_mode" in governor4.monitor_once())
    except Exception as exc:  # pragma: no cover - the test reports its own failure
        logger.exception("self-test crashed")
        failures.append(f"crashed: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for label in failures:
            print(f"  - {label}")
        return 1
    print("\nall checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
