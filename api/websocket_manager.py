"""
Atlas — WebSocket fan-out for the UI.

One :class:`WebSocketManager` tracks every connected UI client. It is a
*singleton* (:func:`get_websocket_manager`), because the FastAPI routes and the
core daemon must push to the same set of sockets — a second instance would mean
events that silently reach nobody.

The model is push-only. The daemon and its background threads call
:meth:`WebSocketManager.broadcast` whenever state changes; nothing polls. That
is why ``broadcast`` is deliberately *synchronous*: a daemon thread cannot
await, and the underlying send must happen on the event loop. The method
serialises the message to JSON and schedules the send on the loop with
``run_coroutine_threadsafe``, so it is safe to call from any thread or from
inside the loop itself.

Event shapes follow the UI contract, e.g.::

    manager.broadcast({"type": "transcript", "text": "..."})
    manager.broadcast({"type": "mode_change", "mode": "gaming"})
    manager.broadcast({"type": "system_status", "active_model": "fast", ...})
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

#: Default cadence of the periodic ``system_status`` push.
STATUS_INTERVAL: float = 5.0


class WebSocketManager:
    """Maintains the set of active clients and broadcasts JSON to all of them.

    The client set is guarded by a plain :class:`threading.Lock` because it is
    mutated from more than one thread (connections arrive on the loop; a daemon
    shutdown may disconnect). The actual writes always run on the event loop
    captured from the first connection, serialised by an :class:`asyncio.Lock`.
    """

    def __init__(self, *, status_interval: float = STATUS_INTERVAL) -> None:
        self.status_interval = status_interval
        self._clients: set[Any] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._send_lock: asyncio.Lock | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<WebSocketManager clients={self.client_count}>"

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, ws: Any) -> None:
        """Accept a client and capture the loop that can write to it."""
        await ws.accept()
        with self._lock:
            self._clients.add(ws)
            self._loop = asyncio.get_running_loop()
            if self._send_lock is None:
                self._send_lock = asyncio.Lock()
        logger.info("websocket client connected (%d total)", self.client_count)

    def disconnect(self, ws: Any) -> None:
        """Remove a client. Safe to call more than once."""
        with self._lock:
            self._clients.discard(ws)
        logger.info("websocket client disconnected (%d remain)", self.client_count)

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    def broadcast(self, message_dict: Mapping[str, Any]) -> None:
        """Send ``message_dict`` to every connected client.

        Thread-safe and synchronous, so the daemon can call it directly. The
        JSON is serialised once here; clients that fail to receive it are
        dropped silently. When no client has ever connected there is no loop
        to schedule on, and the message is discarded.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            logger.debug("dropping %s message — no connected client", message_dict.get("type"))
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(message_dict), loop)
        except RuntimeError:
            logger.debug("event loop closed while broadcasting %s", message_dict.get("type"))

    async def _broadcast(self, message_dict: Mapping[str, Any]) -> None:
        """The loop-side send: one serialisation, then a write per client."""
        payload = json.dumps(dict(message_dict), ensure_ascii=False, default=str)
        with self._lock:
            clients = list(self._clients)
        if not clients:
            return

        # An async lock keeps two broadcasts from interleaving their writes.
        if self._send_lock is None:
            self._send_lock = asyncio.Lock()
        async with self._send_lock:
            dead: list[Any] = []
            for ws in clients:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(ws)
            if dead:
                with self._lock:
                    for ws in dead:
                        self._clients.discard(ws)
                logger.debug("dropped %d disconnected client(s)", len(dead))

    # ------------------------------------------------------------------
    # Typed convenience helpers (thin wrappers over broadcast)
    # ------------------------------------------------------------------

    def transcript(self, text: str) -> None:
        self.broadcast({"type": "transcript", "text": text})

    def response(self, text: str) -> None:
        self.broadcast({"type": "response", "text": text})

    def mode_change(self, mode: str, *, previous: str | None = None) -> None:
        self.broadcast({"type": "mode_change", "mode": mode, "previous": previous})

    def skill_approval_needed(self, proposal: Mapping[str, Any]) -> None:
        self.broadcast({"type": "skill_approval_needed", "proposal": dict(proposal)})

    def research_complete(self, report: Mapping[str, Any]) -> None:
        self.broadcast({"type": "research_complete", "report": dict(report)})

    def trade_confirmation_needed(self, trade: Mapping[str, Any]) -> None:
        self.broadcast({"type": "trade_confirmation_needed", "trade": dict(trade)})

    def system_status(self, status: Mapping[str, Any]) -> None:
        self.broadcast({"type": "system_status", **dict(status)})

    # ------------------------------------------------------------------
    # Periodic status push
    # ------------------------------------------------------------------

    async def status_loop(self, provider: Callable[[], Mapping[str, Any]]) -> None:
        """Push ``system_status`` every ``status_interval`` seconds.

        This is still push, not poll: the UI never asks for status. ``provider``
        runs in a worker thread so a slow ``nvidia-smi``/``psutil`` read cannot
        stall the event loop.
        """
        while not self._closed:
            try:
                status = await asyncio.to_thread(provider)
                await self._broadcast({"type": "system_status", **dict(status)})
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("system status broadcast failed")
            await asyncio.sleep(self.status_interval)

    def close(self) -> None:
        """Stop the status loop and forget every client."""
        self._closed = True
        with self._lock:
            self._clients.clear()
            self._loop = None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: WebSocketManager | None = None
_instance_lock = threading.Lock()


def get_websocket_manager() -> WebSocketManager:
    """The process-wide manager shared by the API routes and the daemon.

    Built on first use so importing this module never touches the event loop.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = WebSocketManager()
    return _instance


def reset_websocket_manager() -> None:
    """Drop the singleton (used by tests, and on daemon shutdown)."""
    global _instance
    with _instance_lock:
        if _instance is not None:
            _instance.close()
        _instance = None


#: Alias some callers may prefer.
ConnectionManager = WebSocketManager
