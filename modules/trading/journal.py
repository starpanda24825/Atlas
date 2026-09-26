"""
Atlas — the trading journal.

Every trade and every strategy idea is written to the vault as ordinary
markdown, in a ``trading/`` folder, so the record is human-readable and does
not depend on a broker staying reachable. This is the "why" behind a trade,
which is the part a broker's order history never keeps.

:meth:`TradingJournal.summarise_performance` reads those entries back and asks
the deep model for a plain-language narrative — what actually worked, what did
not, and what the record suggests doing differently.

Example::

    from modules.trading.journal import TradingJournal

    journal = TradingJournal(vault_manager, llm_manager)
    journal.log_trade(
        {"symbol": "AAPL", "side": "buy", "qty": 10, "status": "submitted"},
        reasoning="Bounce off the 200-day with volume.",
        outcome="Closed +2.1% after three sessions.",
    )
    print(journal.get_journal(limit=10)[0]["title"])
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from modules.trading.strategy import ask_deep

if TYPE_CHECKING:  # injected collaborators; imports here are typing-only
    from core.llm_manager import LLMServerManager
    from memory.vault_manager import VaultManager

logger = logging.getLogger(__name__)

TRADING_FOLDER = "trading"
TRADE_TYPE = "trade"
IDEA_TYPE = "idea"

SUMMARY_MAX_TOKENS = 1200

_PERIOD_DAYS: dict[str, int] = {
    "1w": 7, "2w": 14, "1m": 30, "1mo": 30, "3m": 90, "3mo": 90,
    "6m": 180, "6mo": 180, "1y": 365, "2y": 730, "all": 36500, "max": 36500,
}


def _pick(source: Any, keys: Sequence[str], default: str = "") -> str:
    """First present, non-empty value under ``keys`` (dict or object)."""
    for key in keys:
        value = source.get(key) if isinstance(source, Mapping) else getattr(source, key, None)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _created_after(metadata: Mapping[str, Any], cutoff: datetime) -> bool:
    """True when a note's ``created`` timestamp is on/after ``cutoff``."""
    raw = metadata.get("created") or metadata.get("date")
    if not raw:
        return True  # no timestamp: do not silently drop the entry
    if isinstance(raw, datetime):
        stamped = raw
    else:
        try:
            stamped = datetime.fromisoformat(str(raw))
        except ValueError:
            return True
    if stamped.tzinfo is None:
        stamped = stamped.replace(tzinfo=timezone.utc)
    return stamped >= cutoff


class TradingJournal:
    """Reads and writes trading notes in the vault."""

    def __init__(
        self,
        vault_manager: "VaultManager",
        llm_manager: "LLMServerManager | None" = None,
    ) -> None:
        self.vault_manager = vault_manager
        self.llm_manager = llm_manager

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def log_trade(self, trade: Any, reasoning: str = "", outcome: str = "") -> Path | None:
        """Write one trade as a markdown journal entry."""
        symbol = _pick(trade, ("symbol", "ticker", "asset"), "?")
        side = _pick(trade, ("side", "direction"), "?")
        qty = _pick(trade, ("qty", "quantity", "shares"), "?")
        order_type = _pick(trade, ("order_type", "type"), "market")
        status = _pick(trade, ("status", "state"), "unknown")
        trade_id = _pick(trade, ("trade_id", "id", "order_id"), "—")
        price = _pick(trade, ("limit_price", "price", "filled_avg_price", "avg_price"), "—")
        paper = trade.get("paper") if isinstance(trade, Mapping) else getattr(trade, "paper", None)

        sections = [
            "## Trade",
            f"- **Symbol:** {symbol}",
            f"- **Side:** {side}",
            f"- **Quantity:** {qty}",
            f"- **Order type:** {order_type}",
            f"- **Price:** {price}",
            f"- **Status:** {status}",
            f"- **Trade id:** {trade_id}",
            f"- **Mode:** {'paper' if paper is not False else 'LIVE'}",
            "",
            "## Reasoning",
            (reasoning or "").strip() or "_No reasoning recorded._",
            "",
            "## Outcome",
            (outcome or "").strip() or "_Still open._",
        ]
        title = f"Trade {symbol} {side} {qty}".strip()
        return self._write(title, "\n".join(sections), TRADE_TYPE, tags=["trading", "trade"])

    def log_idea(self, idea: str, analysis: str = "") -> Path | None:
        """Save a strategy idea alongside its analysis."""
        text = (idea or "").strip()
        if not text:
            raise ValueError("log_idea() needs an idea")
        title = f"Idea: {text[:60]}{'…' if len(text) > 60 else ''}"
        body = "\n".join(
            [
                "## Idea",
                text,
                "",
                "## Analysis",
                (analysis or "").strip() or "_Not analysed yet._",
            ]
        )
        return self._write(title, body, IDEA_TYPE, tags=["trading", "idea"])

    def _write(
        self, title: str, content: str, note_type: str, *, tags: Sequence[str]
    ) -> Path | None:
        writer = getattr(self.vault_manager, "write_note", None)
        if not callable(writer):
            logger.warning("vault_manager has no write_note — trading entry not saved")
            return None
        try:
            return writer(
                title=title,
                content=content,
                folder=TRADING_FOLDER,
                note_type=note_type,
                tags=list(tags),
            )
        except Exception as exc:
            logger.warning("could not write the trading journal entry: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get_journal(self, limit: int = 20) -> list[dict[str, Any]]:
        """Recent journal entries, newest first."""
        lister = getattr(self.vault_manager, "list_notes", None)
        if not callable(lister):
            return []
        try:
            notes = lister(folder=TRADING_FOLDER, limit=limit)
        except Exception as exc:
            logger.warning("could not read the trading journal: %s", exc)
            return []
        return [_note_to_dict(note) for note in notes or []]

    #: Alias the API's journal endpoint looks for.
    list_entries = get_journal

    def summarise_performance(self, period: str = "1m") -> str:
        """A plain-language narrative of recent trading performance."""
        entries = self.get_journal(limit=500)
        window = _period_window(period)
        if window is not None:
            cutoff = datetime.now(timezone.utc) - window
            entries = [entry for entry in entries if _created_after(entry, cutoff)]

        if not entries:
            return f"No trading journal entries in the {period} window."

        lines: list[str] = []
        for entry in entries:
            lines.append(f"- [{entry.get('date') or entry.get('created')}] {entry.get('title')}")
            body = str(entry.get("body") or "").strip()
            if body:
                lines.append("  " + body.replace("\n", "\n  ")[:1200])

        prompt = (
            "You are a trading coach reviewing a trader's own journal. Write a "
            f"short performance narrative for the {period} period.\n\n"
            "Journal entries:\n" + "\n".join(lines) + "\n\n"
            "Cover: what worked, what did not, recurring mistakes or strengths, "
            "and two or three concrete adjustments for the next period. Base every "
            "claim on the entries above; do not invent numbers that are not present. "
            "Write in plain prose with short paragraphs."
        )
        if self.llm_manager is None:
            return "The summariser has no model configured; here are the raw entries:\n\n" + "\n".join(lines)
        narrative = ask_deep(self.llm_manager, prompt, max_tokens=SUMMARY_MAX_TOKENS)
        return narrative or "The summary model is unavailable — no narrative produced."


def _note_to_dict(note: Any) -> dict[str, Any]:
    metadata = getattr(note, "metadata", {}) or {}
    path = getattr(note, "path", None)
    return {
        "id": path.name if isinstance(path, Path) else str(path or ""),
        "title": getattr(note, "title", ""),
        "type": getattr(note, "type", ""),
        "summary": getattr(note, "summary", ""),
        "date": str(metadata.get("created") or metadata.get("date") or ""),
        "created": str(metadata.get("created") or ""),
        "tags": list(getattr(note, "tags", []) or []),
        "body": getattr(note, "body", ""),
    }


def _period_window(period: str) -> timedelta | None:
    key = (period or "1m").strip().lower()
    days = _PERIOD_DAYS.get(key)
    return timedelta(days=days) if days else None


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

_default: TradingJournal | None = None


def get_journal(limit: int = 20) -> list[dict[str, Any]]:
    return _default_journal().get_journal(limit=limit)


#: Alias the API's journal endpoint looks for.
list_entries = get_journal


def log_trade(trade: Any, reasoning: str = "", outcome: str = "") -> Path | None:
    return _default_journal().log_trade(trade, reasoning, outcome)


def log_idea(idea: str, analysis: str = "") -> Path | None:
    return _default_journal().log_idea(idea, analysis)


def summarise_performance(period: str = "1m") -> str:
    return _default_journal().summarise_performance(period)


def _default_journal() -> TradingJournal:
    global _default
    if _default is None:
        from core.llm_manager import LLMServerManager
        from memory.vault_manager import get_vault_manager

        _default = TradingJournal(get_vault_manager(), LLMServerManager())
    return _default
