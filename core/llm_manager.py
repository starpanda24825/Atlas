"""
Atlas — llama.cpp server manager.

Owns the two llama.cpp server processes behind Atlas:

* **fast** (``FAST_MODEL_PORT``) — small model, always resident, used for the
  low-latency conversational path.
* **deep** (``DEEP_MODEL_PORT``) — 30B MoE model. Too heavy to keep around, so
  it is started on demand by :meth:`LLMServerManager.ensure_deep_available` and
  torn down again after ``DEEP_MODEL_IDLE_TIMEOUT`` seconds of inactivity.

Both servers expose llama.cpp's OpenAI-compatible API, so callers talk to them
through the ``openai`` package (see :meth:`get_fast_client` /
:meth:`get_deep_client`).

Example::

    from core.llm_manager import LLMServerManager

    manager = LLMServerManager()
    manager.start_fast_server()
    if manager.wait_for_server_ready(manager.fast_port):
        client = manager.get_fast_client()
        ...

    manager.ensure_deep_available()   # spins the 30B up, arms the idle timer
    ...
    manager.shutdown()                # stops both servers
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import IO, TYPE_CHECKING

from core.config import (
    DEEP_MODEL_CTX,
    DEEP_MODEL_GPU_LAYERS,
    DEEP_MODEL_IDLE_TIMEOUT,
    DEEP_MODEL_MIN_FREE_VRAM_MB,
    DEEP_MODEL_PATH,
    DEEP_MODEL_PORT,
    FAST_MODEL_CTX,
    FAST_MODEL_GPU_LAYERS,
    FAST_MODEL_PATH,
    FAST_MODEL_PORT,
    LLAMA_SERVER_BIN,
    LLM_HOST,
    LOGS_DIR,
    SERVER_POLL_INTERVAL,
    SERVER_READY_TIMEOUT,
)

if TYPE_CHECKING:  # imported lazily at call time; see _client()
    from openai import OpenAI

logger = logging.getLogger(__name__)

# Route every FFN tensor (the MoE expert weights) to system RAM so the 30B
# model fits in ~8GB of VRAM: attention layers stay on the GPU, experts live
# in RAM. Kept as a raw string — the backslashes belong to the regex.
DEEP_TENSOR_OVERRIDE = r"blk\.\d+\.ffn_.*=CPU"

# llama.cpp >= b6xxx made this flag take a value; a bare "--flash-attn" aborts
# with 'expected value for argument', so it must always be passed explicitly.
FLASH_ATTN = "on"

# KV cache dtype for the fast model, used to keep a wide context inside 8GB of
# VRAM. See fast_command() for the measurement behind it.
FAST_KV_CACHE_TYPE = "q8_0"

HEALTH_PATH = "/health"

# Body of a request to the health endpoint; llama.cpp checks the path only.
_HEALTH_TIMEOUT = 2.0


class LLMServerManager:
    """Starts, health-checks and stops the fast and deep llama.cpp servers.

    All process paths, ports and timings come from :mod:`core.config`. The
    instance attributes are plain copies, so a caller (or a test) can override
    one without touching global config.
    """

    def __init__(self) -> None:
        # --- fast model ---
        self.fast_model_path: Path = FAST_MODEL_PATH
        self.fast_port: int = FAST_MODEL_PORT
        self.fast_ctx: int = FAST_MODEL_CTX
        self.fast_gpu_layers: int = FAST_MODEL_GPU_LAYERS

        # --- deep model ---
        self.deep_model_path: Path = DEEP_MODEL_PATH
        self.deep_port: int = DEEP_MODEL_PORT
        self.deep_ctx: int = DEEP_MODEL_CTX
        self.deep_gpu_layers: int = DEEP_MODEL_GPU_LAYERS

        # --- shared ---
        self.host: str = LLM_HOST
        self.server_bin: Path = LLAMA_SERVER_BIN
        self.logs_dir: Path = LOGS_DIR
        self.ready_timeout: float = SERVER_READY_TIMEOUT
        self.poll_interval: float = SERVER_POLL_INTERVAL
        self.idle_timeout: int = DEEP_MODEL_IDLE_TIMEOUT
        self.deep_min_free_vram_mb: float = DEEP_MODEL_MIN_FREE_VRAM_MB

        self._lock = threading.RLock()
        self._fast_process: subprocess.Popen[bytes] | None = None
        self._deep_process: subprocess.Popen[bytes] | None = None
        self._log_files: dict[str, IO[bytes]] = {}
        self._deep_shutdown_timer: threading.Timer | None = None
        #: Set when the fast server was stopped to free VRAM for the deep one,
        #: so it is only restarted if it was actually taken away.
        self._fast_suspended = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def fast_process(self) -> subprocess.Popen[bytes] | None:
        """The running fast server process, or None."""
        return self._fast_process

    @property
    def deep_process(self) -> subprocess.Popen[bytes] | None:
        """The running deep server process, or None."""
        return self._deep_process

    def base_url(self, port: int) -> str:
        """OpenAI-compatible base URL of a llama.cpp server on ``port``."""
        return f"http://{self.host}:{port}/v1"

    def health_url(self, port: int) -> str:
        """URL of the health endpoint for a llama.cpp server on ``port``."""
        return f"http://{self.host}:{port}{HEALTH_PATH}"

    def _log_path(self, label: str) -> Path:
        return self.logs_dir / f"llama_{label}.log"

    # ------------------------------------------------------------------
    # Command lines
    # ------------------------------------------------------------------

    def fast_command(self) -> list[str]:
        """argv used to launch the fast model server.

        The KV cache is quantised because the context is wide for this hardware.
        Measured on the 8GB card this project targets: 24576 tokens of f16 KV
        does not fit alongside the 4.7GB of weights, while q8_0 halves it to
        ~1.8GB and leaves ~1.2GB free. Quantised KV needs flash attention, which
        is already on.
        """
        return [
            str(self.server_bin),
            "-m", str(self.fast_model_path),
            "-ngl", str(self.fast_gpu_layers),
            "--ctx-size", str(self.fast_ctx),
            "--port", str(self.fast_port),
            "--host", self.host,
            "--parallel", "2",
            "--cache-type-k", FAST_KV_CACHE_TYPE,
            "--cache-type-v", FAST_KV_CACHE_TYPE,
            "--flash-attn", FLASH_ATTN,
        ]

    def deep_command(self) -> list[str]:
        """argv used to launch the deep model server.

        ``--override-tensor`` is the load-bearing flag: it pushes every
        ``blk.<n>.ffn_*`` tensor (the MoE expert weights, the bulk of the
        model) into system RAM while the attention layers stay on the GPU.
        That split is what lets a 30B MoE model run in ~8GB of VRAM.
        """
        return [
            str(self.server_bin),
            "-m", str(self.deep_model_path),
            "-ngl", str(self.deep_gpu_layers),
            "--override-tensor", DEEP_TENSOR_OVERRIDE,
            "--ctx-size", str(self.deep_ctx),
            "--port", str(self.deep_port),
            "--host", self.host,
            "--parallel", "1",
            "--flash-attn", FLASH_ATTN,
        ]

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def is_server_ready(self, port: int) -> bool:
        """True when ``/health`` answers 200.

        llama.cpp returns 503 while the model is still loading, so a False here
        means "not ready yet", not necessarily "not running".
        """
        request = urllib.request.Request(self.health_url(port), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=_HEALTH_TIMEOUT) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError):
            # Connection refused (not listening yet), 503 (loading), timeout.
            return False

    def wait_for_server_ready(
        self,
        port: int,
        timeout: float | None = None,
        process: subprocess.Popen[bytes] | None = None,
    ) -> bool:
        """Poll ``/health`` until it returns 200 or ``timeout`` elapses.

        Polls every ``SERVER_POLL_INTERVAL`` (500ms) for up to
        ``SERVER_READY_TIMEOUT`` (60s) by default. If ``process`` is given and
        has already exited, this returns False immediately instead of waiting
        out the full timeout — loading a multi-GB model can fail fast, and the
        server's own log has the reason.
        """
        deadline = time.monotonic() + (self.ready_timeout if timeout is None else timeout)
        while True:
            if process is not None and process.poll() is not None:
                logger.error(
                    "llama.cpp server on port %s exited early (code %s) — see %s",
                    port, process.returncode, self._log_path_for_port(port),
                )
                return False

            if self.is_server_ready(port):
                logger.info("llama.cpp server ready on port %s", port)
                return True

            if time.monotonic() >= deadline:
                logger.error(
                    "timed out waiting for llama.cpp server on port %s after %.0fs",
                    port, self.ready_timeout if timeout is None else timeout,
                )
                return False

            time.sleep(self.poll_interval)

    def _log_path_for_port(self, port: int) -> Path:
        label = "deep" if port == self.deep_port else "fast"
        return self._log_path(label)

    # ------------------------------------------------------------------
    # Starting
    # ------------------------------------------------------------------

    def start_fast_server(self) -> subprocess.Popen[bytes]:
        """Launch the fast model server and store the process reference.

        Idempotent: calling it while the server is already running returns the
        existing process instead of starting a second one.
        """
        with self._lock:
            if self._is_alive(self._fast_process):
                assert self._fast_process is not None
                logger.debug("fast server already running (pid %s)", self._fast_process.pid)
                return self._fast_process
            self._ensure_port_free("fast", self.fast_port)
            self._fast_process = self._spawn("fast", self.fast_model_path, self.fast_command())
            return self._fast_process

    def start_deep_server(self) -> subprocess.Popen[bytes]:
        """Launch the deep model server and store the process reference.

        Idempotent, like :meth:`start_fast_server`. The deep server is *not*
        started at idle — use :meth:`ensure_deep_available` so the inactivity
        timer is armed along with it.
        """
        with self._lock:
            if self._is_alive(self._deep_process):
                assert self._deep_process is not None
                logger.debug("deep server already running (pid %s)", self._deep_process.pid)
                return self._deep_process
            self._ensure_port_free("deep", self.deep_port)
            self._deep_process = self._spawn("deep", self.deep_model_path, self.deep_command())
            return self._deep_process

    def _ensure_port_free(self, label: str, port: int) -> None:
        if self.is_server_ready(port):
            raise RuntimeError(
                f"{label} server port {port} is already serving requests — another "
                "llama.cpp server is running there. Stop it, or change the port in "
                "core/config.py, before starting Atlas."
            )

    def _spawn(
        self,
        label: str,
        model_path: Path,
        command: list[str],
    ) -> subprocess.Popen[bytes]:
        if not self.server_bin.is_file():
            raise FileNotFoundError(
                f"llama.cpp server binary not found at {self.server_bin} — "
                "check LLAMA_CPP_DIR in core/config.py"
            )
        if not model_path.is_file():
            raise FileNotFoundError(f"{label} model not found at {model_path}")

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_path(label)
        log_file = log_path.open("ab", buffering=0)

        logger.info("starting %s llama.cpp server: %s", label, shlex.join(command))
        try:
            # Deliberately *not* start_new_session: these processes hold
            # gigabytes of RAM, so they must not outlive Atlas if it dies.
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            log_file.close()
            raise

        self._log_files[label] = log_file
        logger.info("%s server started (pid %s), logging to %s", label, process.pid, log_path)
        return process

    # ------------------------------------------------------------------
    # Deep model on demand
    # ------------------------------------------------------------------

    @property
    def fast_suspended(self) -> bool:
        """True while the fast server is stopped to make room for the deep one."""
        return self._fast_suspended

    def free_vram_mb(self) -> float | None:
        """Free VRAM on device 0 in MiB, or None when it cannot be determined.

        ``nvidia-smi`` is not guaranteed to exist (and this is not fatal): when
        the answer is unknown the caller simply tries and recovers, rather than
        assuming the worst and stopping a healthy server.
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        try:
            return float(result.stdout.strip().splitlines()[0])
        except (ValueError, IndexError):
            return None

    def _suspend_fast_for_deep(self) -> bool:
        """Stop the fast server when the deep model will not otherwise fit.

        Returns True only when the fast server was actually stopped, so the
        caller knows whether there is anything to restore later. A card that
        can hold both models keeps the fast server running.
        """
        with self._lock:
            free = self.free_vram_mb()
            if free is None:
                return False
            if free >= self.deep_min_free_vram_mb:
                logger.debug(
                    "%.0f MiB free VRAM is enough for the deep model — keeping the "
                    "fast server resident",
                    free,
                )
                return False
            if not self._is_alive(self._fast_process):
                return False
            logger.info(
                "freeing VRAM for the deep model: %.0f MiB free, %.0f MiB needed — "
                "stopping the fast server (it will be restarted when the deep "
                "model goes idle)",
                free,
                self.deep_min_free_vram_mb,
            )
            self.stop_fast_server()
            self._fast_suspended = True
            return True

    def restore_resident_model(self) -> bool:
        """Bring the fast model back after a deep-model swap.

        Does nothing unless the fast server was stopped to make room, so an
        ordinary conversation never pays for this. The deep server is stopped
        first: on hardware where the swap was needed the two cannot be resident
        at once, so restoring the fast model means giving up the deep one.

        Returns True when the fast model is serving again.
        """
        with self._lock:
            if not self._fast_suspended:
                return False

        self.stop_deep_server()

        with self._lock:
            if not self._fast_suspended:
                return False  # another thread got there first
            self._fast_suspended = False
            try:
                process = self.start_fast_server()
            except Exception:
                logger.exception("could not restart the fast model server")
                return False

        if not self.is_server_ready(self.fast_port):
            return self.wait_for_server_ready(self.fast_port, process=process)
        return True

    def _retry_deep_with_vram_freed(self, process: subprocess.Popen[bytes] | None) -> bool:
        """Second attempt at the deep server after freeing the fast one.

        The pre-flight VRAM estimate can be wrong — no ``nvidia-smi``, a stale
        reading, another process holding memory — and the deep server reports
        failure the same way either way: it exits before ``/health`` ever
        answers. Since stopping the fast server is the only lever available,
        this tries it once, and refuses to loop if it was already tried.
        """
        with self._lock:
            if self._fast_suspended:
                return False
            if not self._suspend_fast_for_deep():
                # Either there is no fast server to stop (so there is nothing
                # left to free) or the VRAM reading says there is room, in
                # which case a retry would fail identically.
                return False
            self._deep_process = None
            self._terminate("deep", process, 5.0)
            try:
                process = self.start_deep_server()
            except Exception:
                logger.exception("could not restart the deep server after freeing VRAM")
                return False

        if self.is_server_ready(self.deep_port):
            return True
        return self.wait_for_server_ready(self.deep_port, process=process)

    def ensure_deep_available(self) -> bool:
        """Make sure the deep server is up and (re)arm its idle-shutdown timer.

        Starts the deep server if it is not running, waits for it to become
        ready, then arms a ``DEEP_MODEL_IDLE_TIMEOUT`` timer. Every call
        restarts that timer, so using the deep model during a conversation
        keeps it resident; once the timer fires unnoticed, the server is
        stopped, the VRAM released, and — if it was the fast server that made
        room — the fast model put back.

        The fast and deep servers cannot both fit in this project's 8GB card,
        so this will stop the fast server when the free VRAM says it has to.
        Call :meth:`restore_resident_model` when finished to get the fast model
        back without waiting out the idle timer.

        Returns True when the deep model is ready to serve requests.
        """
        with self._lock:
            process = self._deep_process
            if not self._is_alive(process):
                self._suspend_fast_for_deep()
                try:
                    process = self.start_deep_server()
                except Exception:
                    logger.exception("could not start the deep model server")
                    self.restore_resident_model()
                    return False

        # Waiting happens outside the lock so a concurrent shutdown is not
        # blocked for up to ready_timeout seconds.
        if not self.is_server_ready(self.deep_port):
            if not self.wait_for_server_ready(self.deep_port, process=process):
                if not self._retry_deep_with_vram_freed(process):
                    # Never leave the user without a model at all.
                    self.restore_resident_model()
                    return False

        with self._lock:
            self._schedule_deep_shutdown()
        return True

    def _schedule_deep_shutdown(self) -> None:
        """(Re)arm the inactivity timer that stops the deep server."""
        self._cancel_deep_shutdown()
        timer = threading.Timer(self.idle_timeout, self._on_deep_idle_timeout)
        timer.daemon = True
        timer.name = "atlas-deep-model-idle-timeout"
        self._deep_shutdown_timer = timer
        timer.start()
        logger.debug("deep server idle timeout armed for %ss", self.idle_timeout)

    def _cancel_deep_shutdown(self) -> None:
        timer = self._deep_shutdown_timer
        self._deep_shutdown_timer = None
        if timer is not None:
            timer.cancel()

    def _on_deep_idle_timeout(self) -> None:
        logger.info(
            "deep model idle for %ss — stopping the deep server",
            self.idle_timeout,
        )
        self.stop_deep_server()
        # Backstop for any caller that used the deep model and forgot to hand
        # the machine back: restoring is a no-op unless a swap happened.
        if self.restore_resident_model():
            logger.info("fast model restored after the deep model went idle")

    # ------------------------------------------------------------------
    # Stopping
    # ------------------------------------------------------------------

    def stop_deep_server(self, timeout: float = 15.0) -> None:
        """Terminate the deep server and clean up its timer and log handle.

        Safe to call when the server is not running. Does *not* restore the
        fast server — use :meth:`restore_resident_model` for that.
        """
        with self._lock:
            self._cancel_deep_shutdown()
            process = self._deep_process
            self._deep_process = None
            self._terminate("deep", process, timeout)

    def stop_fast_server(self, timeout: float = 15.0) -> None:
        """Terminate the fast server. Safe to call when it is not running."""
        with self._lock:
            process = self._fast_process
            self._fast_process = None
            self._terminate("fast", process, timeout)

    def stop_all(self, timeout: float = 15.0) -> None:
        """Stop both servers. Intended for daemon shutdown."""
        self.stop_deep_server(timeout=timeout)
        self.stop_fast_server(timeout=timeout)
        with self._lock:
            self._fast_suspended = False

    def shutdown(self, timeout: float = 15.0) -> None:
        """Alias for :meth:`stop_all` — stop everything cleanly."""
        self.stop_all(timeout=timeout)

    def _terminate(
        self,
        label: str,
        process: subprocess.Popen[bytes] | None,
        timeout: float,
    ) -> None:
        try:
            if self._is_alive(process):
                assert process is not None
                logger.info("stopping %s server (pid %s)", label, process.pid)
                process.terminate()
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "%s server ignored SIGTERM for %.0fs — killing it", label, timeout
                    )
                    process.kill()
                    process.wait(timeout=5)
                logger.info("%s server stopped", label)
        finally:
            log_file = self._log_files.pop(label, None)
            if log_file is not None:
                log_file.close()

    # ------------------------------------------------------------------
    # Clients
    # ------------------------------------------------------------------

    def get_fast_client(self) -> "OpenAI":
        """OpenAI client pointed at the fast server.

        Does not start anything — call :meth:`start_fast_server` and
        :meth:`wait_for_server_ready` first.
        """
        return self._client(self.fast_port)

    def get_deep_client(self) -> "OpenAI":
        """OpenAI client pointed at the deep server.

        Does not start anything — call :meth:`ensure_deep_available` first so
        the server is up and the idle timer is armed.
        """
        return self._client(self.deep_port)

    def _client(self, port: int) -> "OpenAI":
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "The 'openai' package is required for Atlas LLM clients — "
                "install it with `pip install openai`."
            ) from exc

        # llama.cpp ignores the key, but the SDK requires one to be present.
        return OpenAI(base_url=self.base_url(port), api_key="none")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_alive(process: subprocess.Popen[bytes] | None) -> bool:
        return process is not None and process.poll() is None
