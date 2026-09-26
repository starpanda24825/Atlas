"""
Atlas — trade execution, behind a human gate.

Nothing here places an order by itself. A trade must first be *proposed*
(:meth:`TradeExecutor.propose_trade`), which only records an intent and returns
it for the UI to display and Atlas to read aloud. The order is submitted only
when :meth:`TradeExecutor.confirm_and_execute` is called with that trade's id,
which the caller does after the user has explicitly confirmed.

Read-only calls (:meth:`get_positions`, :meth:`get_account`) need no
confirmation — they cannot move money.

Paper trading is the default and the safe state. Live trading happens only when
``ALPACA_PAPER`` is set to False in ``core/config.py`` (or the environment),
which is a conscious, deliberate change rather than an accident.

Example::

    from modules.trading.execution import TradeExecutor

    executor = TradeExecutor()
    pending = executor.propose_trade("AAPL", 10, "buy", limit_price=190.0,
                                     order_type="limit")
    # ... user confirms ...
    result = executor.confirm_and_execute(pending["trade_id"])
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from core.config import (
    ALPACA_API_KEY,
    ALPACA_BASE_URL,
    ALPACA_PAPER,
    ALPACA_SECRET_KEY,
)

logger = logging.getLogger(__name__)

VALID_ORDER_TYPES = ("market", "limit")
_SIDE_ALIASES = {"long": "buy", "short": "sell"}


class TradingError(RuntimeError):
    """Raised for misuse of the executor (not for ordinary API failures)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_dict(value: Any) -> dict[str, Any]:
    """Best-effort plain-dict conversion for an alpaca-py model."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    for method in ("model_dump", "dict", "to_dict"):
        candidate = getattr(value, method, None)
        if callable(candidate):
            try:
                result = candidate(mode="json") if method == "model_dump" else candidate()
                if isinstance(result, dict):
                    return {key: _jsonable(item) for key, item in result.items()}
            except TypeError:
                try:
                    result = candidate()
                    if isinstance(result, dict):
                        return {key: _jsonable(item) for key, item in result.items()}
                except Exception:
                    continue
            except Exception:
                continue
    # Plain objects (including test doubles) still expose their attributes.
    try:
        raw = vars(value)
    except TypeError:
        raw = {}
    if raw:
        return {
            str(key): _jsonable(item)
            for key, item in raw.items()
            if not str(key).startswith("_")
        }
    return {"repr": str(value)}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


class TradeExecutor:
    """Propose -> confirm -> execute, against Alpaca (paper by default)."""

    def __init__(self, client: Any = None, *, paper: bool | None = None) -> None:
        self._client = client
        self._paper = bool(ALPACA_PAPER if paper is None else paper)
        self._pending: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def paper(self) -> bool:
        """True when orders go to the paper endpoint."""
        return self._paper

    def is_configured(self) -> bool:
        return bool(ALPACA_API_KEY and ALPACA_SECRET_KEY)

    def pending_trades(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(trade) for trade in self._pending.values()]

    def _client_or_none(self) -> Any | None:
        if self._client is not None:
            return self._client
        if not self.is_configured():
            logger.warning(
                "Alpaca credentials are not set — trades cannot be submitted "
                "(set ALPACA_API_KEY and ALPACA_SECRET_KEY)"
            )
            return None
        try:
            from alpaca.trading.client import TradingClient
        except ImportError:
            logger.warning("alpaca-py is not installed — trading is unavailable")
            return None
        try:
            self._client = TradingClient(
                ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=self._paper
            )
        except Exception as exc:  # pragma: no cover - credential shape
            logger.warning("could not build the Alpaca client: %s", exc)
            return None
        return self._client

    # ------------------------------------------------------------------
    # Proposal (creates no order)
    # ------------------------------------------------------------------

    def propose_trade(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        """Record a pending trade for the UI and voice announcement.

        Returns the pending dict. It is never submitted until
        :meth:`confirm_and_execute` is called with ``trade_id``.
        """
        normalised_side = _SIDE_ALIASES.get(str(side).strip().lower(), str(side).strip().lower())
        kind = str(order_type).strip().lower() or "market"

        error = self._validate(symbol, qty, normalised_side, kind, limit_price)
        trade_id = uuid.uuid4().hex[:12]
        pending: dict[str, Any] = {
            "trade_id": trade_id,
            "id": trade_id,
            "symbol": str(symbol).strip().upper(),
            "qty": float(qty) if _is_number(qty) else qty,
            "side": normalised_side,
            "order_type": kind,
            "limit_price": float(limit_price) if _is_number(limit_price) else None,
            "status": "rejected" if error else "pending_confirmation",
            "requires_confirmation": True,
            "paper": self._paper,
            "created": _now(),
            "error": error,
        }
        if error:
            logger.warning("rejected trade proposal: %s", error)
            return pending

        with self._lock:
            self._pending[trade_id] = pending
        logger.info(
            "proposed %s %s x%s (%s) paper=%s — awaiting confirmation",
            pending["side"],
            pending["symbol"],
            pending["qty"],
            kind,
            self._paper,
        )
        return dict(pending)

    @staticmethod
    def _validate(
        symbol: str, qty: Any, side: str, order_type: str, limit_price: Any
    ) -> str | None:
        if not str(symbol or "").strip():
            return "a symbol is required"
        if not _is_number(qty) or float(qty) <= 0:
            return "quantity must be a positive number"
        if side not in ("buy", "sell"):
            return "side must be 'buy' or 'sell'"
        if order_type not in VALID_ORDER_TYPES:
            return "order_type must be 'market' or 'limit'"
        if order_type == "limit" and (not _is_number(limit_price) or float(limit_price) <= 0):
            return "a positive limit_price is required for a limit order"
        return None

    # ------------------------------------------------------------------
    # Confirmation (the only path that submits)
    # ------------------------------------------------------------------

    def confirm_and_execute(self, trade_id: str, confirmed: bool = True) -> dict[str, Any]:
        """Submit a pending trade. Call only after the user has confirmed.

        ``confirmed=False`` cancels the proposal instead, so the UI's "Cancel"
        path does not leave a trade stranded in the pending list.
        """
        with self._lock:
            pending = self._pending.get(trade_id)
            if pending is None:
                return _failure(trade_id, "rejected", "no pending trade with that id")
            if pending["status"] != "pending_confirmation":
                return _failure(trade_id, "rejected", f"trade is already {pending['status']}")

            if not confirmed:
                pending["status"] = "cancelled"
                self._pending.pop(trade_id, None)
                logger.info("trade %s cancelled by the user", trade_id)
                return {"trade_id": trade_id, "status": "cancelled", "paper": self._paper}

            pending["status"] = "submitting"

        client = self._client_or_none()
        if client is None:
            with self._lock:
                pending["status"] = "failed"
                self._pending.pop(trade_id, None)
            return _failure(
                trade_id,
                "failed",
                "Alpaca is not configured — set ALPACA_API_KEY and ALPACA_SECRET_KEY",
            )

        try:
            order = client.submit_order(order_data=self._order_request(pending))
        except Exception as exc:
            logger.warning("Alpaca rejected order for %s: %s", trade_id, exc)
            with self._lock:
                pending["status"] = "failed"
                self._pending.pop(trade_id, None)
            return _failure(trade_id, "failed", str(exc))

        with self._lock:
            pending["status"] = "submitted"
            self._pending.pop(trade_id, None)

        order_dict = _to_dict(order)
        logger.info(
            "submitted %s %s x%s (%s order %s)",
            pending["side"],
            pending["symbol"],
            pending["qty"],
            pending["order_type"],
            order_dict.get("id") or trade_id,
        )
        return {
            "trade_id": trade_id,
            "status": "submitted",
            "paper": self._paper,
            "symbol": pending["symbol"],
            "qty": pending["qty"],
            "side": pending["side"],
            "order": order_dict,
        }

    def _order_request(self, pending: dict[str, Any]) -> Any:
        """Build the alpaca-py order request for a pending trade."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        side = OrderSide.BUY if pending["side"] == "buy" else OrderSide.SELL
        common = {
            "symbol": pending["symbol"],
            "qty": pending["qty"],
            "side": side,
            "time_in_force": TimeInForce.DAY,
        }
        if pending["order_type"] == "limit":
            return LimitOrderRequest(limit_price=pending["limit_price"], **common)
        return MarketOrderRequest(**common)

    # ------------------------------------------------------------------
    # Read-only (no confirmation required)
    # ------------------------------------------------------------------

    def get_positions(self) -> list[dict[str, Any]]:
        client = self._client_or_none()
        if client is None:
            return []
        try:
            positions = client.get_all_positions()
        except Exception as exc:
            logger.warning("could not read positions: %s", exc)
            return []
        return [_to_dict(position) for position in positions or []]

    def get_account(self) -> dict[str, Any]:
        client = self._client_or_none()
        if client is None:
            return {
                "error": "alpaca_not_configured",
                "message": "Set ALPACA_API_KEY and ALPACA_SECRET_KEY to read the account.",
                "paper": self._paper,
            }
        try:
            account = client.get_account()
        except Exception as exc:
            logger.warning("could not read the account: %s", exc)
            return {"error": "alpaca_error", "message": str(exc), "paper": self._paper}
        data = _to_dict(account)
        data["paper"] = self._paper
        data["base_url"] = ALPACA_BASE_URL
        return data

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Recent orders, newest first, for the trading history view."""
        client = self._client_or_none()
        if client is None:
            return []
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit or 50)
            orders = client.get_orders(filter=request)
        except Exception as exc:
            logger.warning("could not read order history: %s", exc)
            return []
        return [_to_dict(order) for order in orders or []]


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def _failure(trade_id: str, status: str, message: str) -> dict[str, Any]:
    return {"trade_id": trade_id, "status": status, "error": message}


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

_default: TradeExecutor | None = None
_default_lock = threading.Lock()


def get_executor() -> TradeExecutor:
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = TradeExecutor()
    return _default


def propose_trade(
    symbol: str,
    qty: float,
    side: str,
    order_type: str = "market",
    limit_price: float | None = None,
) -> dict[str, Any]:
    return get_executor().propose_trade(symbol, qty, side, order_type, limit_price)


def confirm_and_execute(trade_id: str, confirmed: bool = True) -> dict[str, Any]:
    return get_executor().confirm_and_execute(trade_id, confirmed)


#: Alias used by the API/agent tool names.
confirm_trade = confirm_and_execute


def get_positions() -> list[dict[str, Any]]:
    return get_executor().get_positions()


def get_account() -> dict[str, Any]:
    return get_executor().get_account()


def get_history(limit: int | None = None) -> list[dict[str, Any]]:
    return get_executor().get_history(limit)
