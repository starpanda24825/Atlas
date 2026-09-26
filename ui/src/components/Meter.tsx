import { clampPercent } from "../lib/format";

export function Meter({
  label,
  percent,
  detail,
}: {
  label: string;
  percent: number | null | undefined;
  detail?: string;
}) {
  const value = clampPercent(percent);
  const tone =
    value > 90 ? "bg-rose-500" : value > 70 ? "bg-amber-500" : "bg-emerald-500";
  return (
    <div className="flex items-center gap-2" title={detail ?? `${label} usage`}>
      <span className="w-9 text-[10px] font-medium uppercase tracking-wide text-slate-500">
        {label}
      </span>
      <div className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-800">
        <div
          className={`h-full rounded-full ${tone} transition-[width] duration-500`}
          style={{ width: `${value}%` }}
        />
      </div>
      <span className="w-10 text-right text-[10px] tabular-nums text-slate-500">
        {Math.round(value)}%
      </span>
    </div>
  );
}
