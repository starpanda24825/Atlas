"""
Atlas — trading strategy: critique, codegen and backtesting.

Three jobs, all with the deep model where a judgement call is needed:

* :meth:`TradingStrategyModule.analyze_idea` — pressure-test a trading idea
  before any money is involved. It hunts for the failure modes an idea's author
  is blind to: regime dependence, look-ahead bias, survivorship bias, costs.
* :meth:`TradingStrategyModule.generate_strategy_code` — turn a description into
  a runnable backtest module.
* :meth:`TradingStrategyModule.run_backtest` — fetch history from yfinance and
  execute the generated code **inside the skill sandbox**, then compute the
  headline metrics from the equity curve it returns.

The sandbox has no network, so the price data is downloaded here, written into
the run directory as a CSV, and handed to the code. That is also what makes the
result reproducible: the code cannot quietly re-fetch different data.

Backtest code contract — the generated module must define exactly::

    def run(prices_csv: str, symbol: str = "", **kwargs) -> dict:
        ...
        return {"equity": [...], "dates": [...], "trades": [{"pnl": ...}, ...]}

and must read only that CSV (no network). ``run_backtest`` computes the metrics
itself from ``equity`` so a slightly creative strategy cannot report its own
Sharpe ratio.

Example::

    from modules.trading.strategy import TradingStrategyModule

    module = TradingStrategyModule(llm_manager)
    print(module.analyze_idea("buy when RSI < 30 and above the 200-day"))
    code = module.generate_strategy_code("mean reversion on SPY")
    result = module.run_backtest(code, "SPY", period="2y")
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # injected collaborators; imports here are typing-only
    from core.llm_manager import LLMServerManager
    from skills.sandbox import SkillSandbox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

ANALYSIS_MAX_TOKENS: int = 1200
CODEGEN_MAX_TOKENS: int = 2500
MODEL_TEMPERATURE: float = 0.3

#: Backtests are heavier than a skill: vectorbt and numba reserve a lot of
#: virtual address space, and a multi-year run is not instant.
BACKTEST_TIMEOUT: float = 120.0
BACKTEST_MEMORY_MB: int = 4096
INITIAL_CAPITAL: float = 100_000.0

#: The name of the price file handed to sandboxed strategy code.
PRICES_FILENAME: str = "prices.csv"

# --- think-tag handling, matching the rest of the codebase ------------------
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_LEADING_THINK = re.compile(r"^\s*<think\b[^>]*>", re.IGNORECASE)
_TRAILING_THINK = re.compile(r"</think\s*>\s*$", re.IGNORECASE)
_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

_PERIOD_DAYS: dict[str, int] = {
    "1w": 7, "2w": 14, "1mo": 30, "1m": 30, "3mo": 90, "3m": 90,
    "6mo": 180, "6m": 180, "1y": 365, "2y": 730, "3y": 1095, "5y": 1825, "10y": 3650,
}


@dataclass(slots=True)
class BacktestResult:
    """Headline numbers from one backtest run."""

    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    equity_curve_data: list[dict[str, Any]] = field(default_factory=list)

    symbol: str = ""
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = INITIAL_CAPITAL
    isolation: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "trade_count": self.trade_count,
            "equity_curve_data": self.equity_curve_data,
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "isolation": self.isolation,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Model plumbing
# ---------------------------------------------------------------------------


def strip_think(text: str) -> str:
    """Remove reasoning that leaked into an answer."""
    if not text:
        return ""
    cleaned = _THINK_BLOCK.sub(" ", text)
    cleaned = _LEADING_THINK.sub("", cleaned)
    cleaned = _TRAILING_THINK.sub("", cleaned)
    return cleaned.strip()


def ask_deep(
    llm_manager: "LLMServerManager",
    prompt: str,
    *,
    max_tokens: int,
    temperature: float = MODEL_TEMPERATURE,
) -> str:
    """One completion, preferring the deep model and degrading to the fast one.

    Shared with :mod:`modules.trading.journal`, which needs the same behaviour
    for its performance narrative.
    """
    request: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    for kind in ("deep", "fast"):
        try:
            if kind == "deep":
                if not llm_manager.ensure_deep_available():
                    logger.warning("deep model unavailable — falling back to the fast model")
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
            request["temperature"] = 0.0
            logger.warning("%s model returned no content", kind)
        except Exception as exc:  # the SDK raises many shapes
            logger.warning("%s model request failed: %s", kind, exc)
    return ""


def extract_code(response: str) -> str:
    """Pull the Python module out of a model reply."""
    text = str(response or "").replace("\r\n", "\n")
    blocks = _CODE_BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    # No fence: take everything from the first plausible definition onward.
    match = re.search(r"^(?:from |import |class |def )", text, re.MULTILINE)
    return text[match.start() :].strip() if match else text.strip()


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------


class TradingStrategyModule:
    """Critique, generate and backtest trading strategies."""

    def __init__(
        self,
        llm_manager: "LLMServerManager",
        sandbox: "SkillSandbox | None" = None,
    ) -> None:
        self.llm_manager = llm_manager
        self._sandbox = sandbox

    @property
    def _sandbox_or_default(self) -> "SkillSandbox":
        if self._sandbox is None:
            from skills.sandbox import SkillSandbox

            self._sandbox = SkillSandbox(keep_artifacts=False)
        return self._sandbox

    # -- analysis -------------------------------------------------------

    def analyze_idea(self, description: str) -> str:
        """Critique a trading idea: risks, edge cases, concrete improvements."""
        idea = (description or "").strip()
        if not idea:
            return "No trading idea was given."
        prompt = (
            "You are a cautious quantitative trading risk reviewer. Critique the "
            "following trading idea.\n\n"
            f"Idea: {idea}\n\n"
            "Cover, in order:\n"
            "1. What the supposed edge actually is, and whether it is plausible.\n"
            "2. Edge cases and risks that would break it (regime dependence, "
            "liquidity, gaps, look-ahead or survivorship bias, transaction costs, "
            "slippage, position sizing, leverage, correlation).\n"
            "3. Concrete, testable improvements.\n"
            "4. A blunt one-line verdict: is this worth backtesting?\n\n"
            "Be specific and sceptical. Do not invent statistics."
        )
        text = ask_deep(self.llm_manager, prompt, max_tokens=ANALYSIS_MAX_TOKENS)
        return text or "The analysis model is unavailable — no critique produced."

    # -- code generation ------------------------------------------------

    def generate_strategy_code(self, description: str) -> str:
        """Write a backtest module for a strategy description."""
        idea = (description or "").strip()
        if not idea:
            raise ValueError("generate_strategy_code() needs a description")
        prompt = (
            "Write a single Python module that backtests a trading strategy.\n\n"
            f"Strategy: {idea}\n\n"
            "Hard requirements:\n"
            f"- Define exactly: def run(prices_csv: str, symbol: str = \"\", **kwargs) -> dict\n"
            f"- Read ONLY the local CSV at `prices_csv` (columns: Date,Open,High,Low,"
            "Close,Volume). There is no network access in the sandbox.\n"
            "- Return a JSON-serialisable dict:"
            ' {"equity": [float, ...], "dates": ["YYYY-MM-DD", ...],'
            ' "trades": [{"pnl": float}, ...], "notes": "..."}.\n'
            "- `equity` is the portfolio value over time, one value per bar, "
            "starting from the initial capital you choose.\n"
            "- Use vectorbt if it is importable, otherwise pandas/numpy. Keep it "
            "self-contained; no file writes and no network.\n"
            "- Add a module docstring with a DESCRIPTION: line and a MEMORY: line "
            "if the run needs more than 512 MB.\n\n"
            "Reply with only the code in one fenced block."
        )
        code = extract_code(ask_deep(self.llm_manager, prompt, max_tokens=CODEGEN_MAX_TOKENS))
        if not code:
            raise RuntimeError("the model did not return any strategy code")
        return code

    # -- backtest -------------------------------------------------------

    def run_backtest(
        self,
        strategy_code: str | None = None,
        symbol: str | None = None,
        start_date: Any = None,
        end_date: Any = None,
        *,
        strategy: str | None = None,
        period: str | None = None,
        initial_capital: float = INITIAL_CAPITAL,
    ) -> BacktestResult:
        """Run ``strategy_code`` against ``symbol`` history, in the sandbox.

        Dates may be given as ``start_date``/``end_date`` or as a ``period``
        shorthand (``"1y"``, ``"6mo"``). ``strategy`` is a compatibility alias
        for ``strategy_code``.
        """
        code = (strategy_code or strategy or "").strip()
        ticker = (symbol or "").strip().upper()
        result = BacktestResult(
            symbol=ticker,
            initial_capital=initial_capital,
        )
        if not code:
            result.error = "no strategy code was supplied"
            return result
        if not ticker:
            result.error = "no symbol was supplied"
            return result

        try:
            start, end = _resolve_window(start_date, end_date, period)
        except ValueError as exc:
            result.error = str(exc)
            return result
        result.start_date, result.end_date = start.isoformat(), end.isoformat()

        frame = _fetch_history(ticker, start, end)
        if frame is None or frame.empty:
            result.error = f"no historical data for {ticker} between {result.start_date} and {result.end_date}"
            return result

        csv_text = frame.to_csv()
        sandbox = self._sandbox_or_default
        sandbox_result = sandbox.execute_in_sandbox(
            code,
            {"prices_csv": PRICES_FILENAME, "symbol": ticker},
            function="run",
            files={PRICES_FILENAME: csv_text},
            timeout=BACKTEST_TIMEOUT,
            memory_mb=BACKTEST_MEMORY_MB,
            allow_network=False,
        )
        result.isolation = sandbox_result.isolation

        if not sandbox_result.success:
            result.error = sandbox_result.error.strip().splitlines()[-1] if sandbox_result.error else "the strategy raised"
            logger.warning("backtest for %s failed: %s", ticker, sandbox_result.summary())
            return result

        payload = sandbox_result.return_value
        if not isinstance(payload, dict):
            result.error = "the strategy did not return a result dict"
            return result

        equity = _coerce_floats(payload.get("equity") or payload.get("equity_curve"))
        if len(equity) < 2:
            result.error = "the strategy returned no usable equity curve"
            return result
        dates = [str(item) for item in (payload.get("dates") or [])]
        trades = payload.get("trades") or []

        result.total_return, result.sharpe_ratio, result.max_drawdown, result.win_rate, result.trade_count = _metrics(
            equity, trades
        )
        result.equity_curve_data = [
            {"date": dates[index] if index < len(dates) else str(index), "equity": value}
            for index, value in enumerate(equity)
        ]
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_window(
    start_date: Any, end_date: Any, period: str | None
) -> tuple[date, date]:
    """Turn explicit dates or a period shorthand into a (start, end) window."""
    end = _as_date(end_date) if end_date else date.today()
    if start_date:
        start = _as_date(start_date)
        if start >= end:
            raise ValueError("start_date must be before end_date")
        return start, end

    key = (period or "1y").strip().lower()
    if key in ("max", "all"):
        return end - timedelta(days=3650), end
    if key not in _PERIOD_DAYS:
        raise ValueError(f"unknown period {period!r}; try 1m, 3m, 6m, 1y, 5y or max")
    return end - timedelta(days=_PERIOD_DAYS[key]), end


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"could not parse date {value!r} (expected YYYY-MM-DD)")


def _fetch_history(symbol: str, start: date, end: date) -> Any:
    """Daily OHLCV from yfinance, or None when unavailable."""
    try:
        import yfinance as yf
    except ImportError:  # pragma: no cover - yfinance is installed
        logger.warning("yfinance is not installed — cannot fetch history for %s", symbol)
        return None
    try:
        frame = yf.download(
            symbol,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        logger.warning("yfinance download for %s failed: %s", symbol, exc)
        return None
    if frame is None or frame.empty:
        return None
    # A single ticker can still come back with a (Price, Ticker) MultiIndex.
    try:
        if getattr(frame.columns, "nlevels", 1) > 1:
            frame.columns = frame.columns.get_level_values(0)
    except Exception:  # pragma: no cover - defensive
        pass
    return frame


def _coerce_floats(values: Any) -> list[float]:
    out: list[float] = []
    for value in values or []:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def _metrics(equity: Sequence[float], trades: Any) -> tuple[float, float, float, float, int]:
    """Return (total_return, sharpe, max_drawdown, win_rate, trade_count)."""
    values = [float(value) for value in equity]
    first, last = values[0], values[-1]
    total_return = (last / first - 1.0) if first else 0.0

    returns = [
        (values[index] / values[index - 1] - 1.0)
        for index in range(1, len(values))
        if values[index - 1]
    ]

    sharpe = 0.0
    if len(returns) > 1:
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        if std > 0:
            sharpe = (mean / std) * math.sqrt(252)

    peak = values[0]
    max_drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        if peak:
            max_drawdown = min(max_drawdown, value / peak - 1.0)

    pnls = _trade_pnls(trades)
    if pnls:
        wins = sum(1 for pnl in pnls if pnl > 0)
        win_rate = wins / len(pnls)
        trade_count = len(pnls)
    else:
        win_rate = (sum(1 for value in returns if value > 0) / len(returns)) if returns else 0.0
        # No trade log: count direction changes in the equity curve as a proxy.
        trade_count = sum(
            1
            for index in range(2, len(returns))
            if (returns[index] > 0) != (returns[index - 1] > 0)
        )
    return total_return, sharpe, max_drawdown, win_rate, trade_count


def _trade_pnls(trades: Any) -> list[float]:
    out: list[float] = []
    if not isinstance(trades, (list, tuple)):
        return out
    for trade in trades:
        value = None
        if isinstance(trade, dict):
            for key in ("pnl", "profit", "return", "result"):
                if key in trade:
                    value = trade[key]
                    break
        else:
            value = getattr(trade, "pnl", None)
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

_default: TradingStrategyModule | None = None


def get_strategy_module() -> TradingStrategyModule:
    """A process-wide module using a fresh LLM server manager."""
    global _default
    if _default is None:
        from core.llm_manager import LLMServerManager

        _default = TradingStrategyModule(LLMServerManager())
    return _default


def analyze_idea(description: str) -> str:
    return get_strategy_module().analyze_idea(description)


def generate_strategy_code(description: str) -> str:
    return get_strategy_module().generate_strategy_code(description)


def run_backtest(
    strategy_code: str | None = None,
    symbol: str | None = None,
    start_date: Any = None,
    end_date: Any = None,
    *,
    strategy: str | None = None,
    period: str | None = None,
) -> BacktestResult:
    return get_strategy_module().run_backtest(
        strategy_code,
        symbol,
        start_date,
        end_date,
        strategy=strategy,
        period=period,
    )
