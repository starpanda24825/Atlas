import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { TrendingDown, TrendingUp, Wallet } from "lucide-react";
import { api, type Position, type Trade } from "../lib/api";
import { Markdown } from "../components/Markdown";
import { cx, money, numberText, pickNumber, pickString, shortDate, signedMoney } from "../lib/format";

interface NormalisedPosition {
  symbol: string;
  quantity: number | null;
  avgCost: number | null;
  price: number | null;
  pnl: number | null;
}

interface CurvePoint {
  label: string;
  equity: number;
  pnl: number;
}

const EMPTY = "—";

function normalisePosition(position: Position): NormalisedPosition {
  const record = position as Record<string, unknown>;
  const quantity = pickNumber(record, ["qty", "quantity", "qty_available", "shares"]);
  const avgCost = pickNumber(record, ["avg_entry_price", "avg_cost", "average_cost", "entry_price"]);
  const price = pickNumber(record, ["current_price", "price", "last_price", "market_price"]);
  let pnl = pickNumber(record, ["unrealized_pl", "unrealized_pnl", "pnl", "profit_loss"]);
  if (pnl === null && price !== null && avgCost !== null && quantity !== null) {
    pnl = (price - avgCost) * quantity;
  }
  return {
    symbol: pickString(record, ["symbol", "ticker", "asset"]) || EMPTY,
    quantity,
    avgCost,
    price,
    pnl,
  };
}

function StatCard({
  label,
  value,
  tone,
  hint,
}: {
  label: string;
  value: string;
  tone?: "up" | "down" | "neutral";
  hint?: string;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3">
      <p className="text-[10px] uppercase tracking-wide text-slate-500">{label}</p>
      <p
        className={cx(
          "mt-1 text-lg font-semibold tabular-nums",
          tone === "up" ? "text-emerald-400" : tone === "down" ? "text-rose-400" : "text-slate-100",
        )}
      >
        {value}
      </p>
      {hint ? <p className="mt-0.5 text-[10px] text-slate-600">{hint}</p> : null}
    </div>
  );
}

export function TradingView() {
  const positions = useQuery({ queryKey: ["trading", "positions"], queryFn: api.positions, refetchInterval: 10000 });
  const history = useQuery({ queryKey: ["trading", "history"], queryFn: () => api.history(200) });
  const journal = useQuery({ queryKey: ["trading", "journal"], queryFn: () => api.journal(50) });

  const rows = useMemo(
    () => (positions.data?.positions ?? []).map(normalisePosition),
    [positions.data],
  );

  const totals = useMemo(() => {
    let marketValue = 0;
    let cost = 0;
    let pnl = 0;
    let priced = 0;
    for (const row of rows) {
      if (row.price !== null && row.quantity !== null) {
        marketValue += row.price * row.quantity;
        priced += 1;
      }
      if (row.avgCost !== null && row.quantity !== null) cost += row.avgCost * row.quantity;
      if (row.pnl !== null) pnl += row.pnl;
    }
    return { marketValue, cost, pnl, priced };
  }, [rows]);

  const curve = useMemo<CurvePoint[]>(() => {
    const trades = (history.data?.trades ?? []) as Trade[];
    let cumulative = 0;
    return trades
      .map((trade) => {
        const record = trade as Record<string, unknown>;
        const pnl = pickNumber(record, ["pnl", "realized_pl", "profit_loss", "profit"]);
        const at = pickString(record, ["filled_at", "date", "created_at", "time", "timestamp"]);
        return { at, pnl: pnl ?? 0 };
      })
      .filter((entry) => entry.at)
      .sort((a, b) => new Date(a.at).getTime() - new Date(b.at).getTime())
      .map((entry) => {
        cumulative += entry.pnl;
        return { label: shortDate(entry.at).split(",")[0], equity: cumulative, pnl: entry.pnl };
      });
  }, [history.data]);

  const daily = useMemo(() => {
    const byDay = new Map<string, number>();
    for (const point of curve) byDay.set(point.label, (byDay.get(point.label) ?? 0) + point.pnl);
    return Array.from(byDay, ([label, pnl]) => ({ label, pnl }));
  }, [curve]);

  const unavailable = positions.isError && history.isError && journal.isError;

  return (
    <div className="flex h-full min-h-0 flex-col gap-4 overflow-y-auto">
      {unavailable ? (
        <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4 text-sm text-slate-400">
          The trading module is not available yet.
        </div>
      ) : null}

      {/* Account summary */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard
          label="Portfolio value"
          value={money(totals.marketValue)}
          hint={`${totals.priced} priced position${totals.priced === 1 ? "" : "s"}`}
        />
        <StatCard
          label="Unrealized P&L"
          value={signedMoney(totals.pnl)}
          tone={totals.pnl >= 0 ? "up" : "down"}
          hint={`cost basis ${money(totals.cost)}`}
        />
        <StatCard label="Buying power" value={EMPTY} hint="not reported by the API" />
        <StatCard label="Day P&L" value={EMPTY} hint="not reported by the API" />
      </div>

      {/* Positions */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/40">
        <h2 className="flex items-center gap-2 border-b border-slate-800 px-4 py-2 text-sm font-semibold text-slate-200">
          <Wallet size={14} /> Positions
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-[10px] uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-4 py-2 font-medium">Symbol</th>
                <th className="px-4 py-2 text-right font-medium">Qty</th>
                <th className="px-4 py-2 text-right font-medium">Avg cost</th>
                <th className="px-4 py-2 text-right font-medium">Price</th>
                <th className="px-4 py-2 text-right font-medium">P&L</th>
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-4 py-6 text-center text-slate-600">
                    {positions.isLoading ? "Loading…" : "No open positions."}
                  </td>
                </tr>
              ) : (
                rows.map((row) => (
                  <tr key={row.symbol} className="border-t border-slate-800/70">
                    <td className="px-4 py-2 font-medium text-slate-200">{row.symbol}</td>
                    <td className="px-4 py-2 text-right tabular-nums text-slate-300">{numberText(row.quantity)}</td>
                    <td className="px-4 py-2 text-right tabular-nums text-slate-300">{money(row.avgCost)}</td>
                    <td className="px-4 py-2 text-right tabular-nums text-slate-300">{money(row.price)}</td>
                    <td
                      className={cx(
                        "px-4 py-2 text-right tabular-nums",
                        row.pnl === null ? "text-slate-500" : row.pnl >= 0 ? "text-emerald-400" : "text-rose-400",
                      )}
                    >
                      {row.pnl === null ? EMPTY : signedMoney(row.pnl)}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* P&L charts */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-3">
          <h2 className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-200">
            <TrendingUp size={14} /> Equity curve
          </h2>
          {curve.length === 0 ? (
            <p className="py-10 text-center text-xs text-slate-600">No realized P&L history available.</p>
          ) : (
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={curve}>
                <defs>
                  <linearGradient id="equity" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#0ea5e9" stopOpacity={0.5} />
                    <stop offset="100%" stopColor="#0ea5e9" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fill: "#64748b", fontSize: 10 }} minTickGap={24} />
                <YAxis tick={{ fill: "#64748b", fontSize: 10 }} width={44} />
                <Tooltip
                  contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 11 }}
                  labelStyle={{ color: "#94a3b8" }}
                />
                <Area type="monotone" dataKey="equity" stroke="#0ea5e9" fill="url(#equity)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </section>

        <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-3">
          <h2 className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-200">
            <TrendingDown size={14} /> Daily P&L
          </h2>
          {daily.length === 0 ? (
            <p className="py-10 text-center text-xs text-slate-600">No daily P&L data available.</p>
          ) : (
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={daily}>
                <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fill: "#64748b", fontSize: 10 }} minTickGap={16} />
                <YAxis tick={{ fill: "#64748b", fontSize: 10 }} width={44} />
                <Tooltip
                  contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 11 }}
                  labelStyle={{ color: "#94a3b8" }}
                />
                <Bar dataKey="pnl" fill="#10b981" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </section>
      </div>

      {/* Journal */}
      <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-3">
        <h2 className="mb-2 text-sm font-semibold text-slate-200">Trading journal</h2>
        {journal.isLoading ? (
          <p className="text-xs text-slate-500">Loading…</p>
        ) : (journal.data?.entries ?? []).length === 0 ? (
          <p className="py-6 text-center text-xs text-slate-600">No journal entries.</p>
        ) : (
          <div className="space-y-2">
            {(journal.data?.entries ?? []).map((entry, index) => {
              const record = entry as Record<string, unknown>;
              const title = pickString(record, ["title", "symbol", "summary"]) || `Entry ${index + 1}`;
              const date = pickString(record, ["date", "created_at", "time", "timestamp"]);
              const body = pickString(record, ["entry", "content", "text", "body", "note"]);
              return (
                <article key={index} className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
                  <div className="mb-1 flex items-center justify-between">
                    <span className="text-xs font-medium text-slate-200">{title}</span>
                    <span className="text-[10px] text-slate-600">{date ? shortDate(date) : ""}</span>
                  </div>
                  {body ? <Markdown text={body} /> : null}
                </article>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
