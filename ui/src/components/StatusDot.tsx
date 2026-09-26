import type { AtlasState } from "../store/atlas";
import { cx } from "../lib/format";

const STYLES: Record<AtlasState, { dot: string; label: string }> = {
  listening: { dot: "bg-emerald-500", label: "Listening" },
  thinking: { dot: "bg-amber-400", label: "Thinking" },
  error: { dot: "bg-rose-500", label: "Error" },
  idle: { dot: "bg-slate-500", label: "Idle" },
};

export function StatusDot({
  state,
  connected = true,
  showLabel = false,
}: {
  state: AtlasState;
  connected?: boolean;
  showLabel?: boolean;
}) {
  const effective: AtlasState = connected ? state : "error";
  const style = STYLES[effective];
  return (
    <span className="inline-flex items-center gap-2" title={connected ? style.label : "Disconnected"}>
      <span className="relative flex h-2.5 w-2.5">
        {effective === "thinking" ? (
          <span className={cx("absolute inline-flex h-full w-full animate-ping rounded-full opacity-60", style.dot)} />
        ) : null}
        <span className={cx("relative inline-flex h-2.5 w-2.5 rounded-full", style.dot)} />
      </span>
      {showLabel ? (
        <span className="text-xs text-slate-400">{connected ? style.label : "Disconnected"}</span>
      ) : null}
    </span>
  );
}
