import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Gamepad2 } from "lucide-react";
import { api } from "../lib/api";
import { useAtlasStore } from "../store/atlas";
import { cx } from "../lib/format";
import { Meter } from "./Meter";
import { StatusDot } from "./StatusDot";

export function TopBar() {
  const queryClient = useQueryClient();
  const atlasState = useAtlasStore((state) => state.atlasState);
  const connected = useAtlasStore((state) => state.connected);
  const mode = useAtlasStore((state) => state.mode);

  const { data: status } = useQuery({
    queryKey: ["status"],
    queryFn: api.status,
    refetchInterval: 5000,
    retry: 1,
  });

  const gaming = useMutation({
    mutationFn: (enabled: boolean) => api.gamingToggle(enabled),
    onSuccess: (result) => {
      useAtlasStore.getState().setMode(result.mode);
      void queryClient.invalidateQueries({ queryKey: ["status"] });
    },
  });

  const model = (status?.active_model ?? "none").toUpperCase();
  const gamingOn = mode === "gaming";

  return (
    <header className="flex h-12 shrink-0 items-center gap-4 border-b border-slate-800 bg-slate-950 px-4">
      <StatusDot state={atlasState} connected={connected} showLabel />

      <span
        className={cx(
          "rounded px-1.5 py-0.5 text-[10px] font-semibold tracking-wide",
          model === "DEEP"
            ? "bg-violet-950 text-violet-300"
            : model === "FAST"
              ? "bg-sky-950 text-sky-300"
              : "bg-slate-800 text-slate-400",
        )}
      >
        {model}
      </span>

      <div className="ml-auto flex items-center gap-3">
        <Meter
          label="VRAM"
          percent={status?.vram_usage?.percent}
          detail={
            status?.vram_usage
              ? `${status.vram_usage.used_mb} / ${status.vram_usage.total_mb} MB`
              : "VRAM unavailable"
          }
        />
        <Meter
          label="RAM"
          percent={status?.ram_usage?.percent}
          detail={
            status?.ram_usage
              ? `${status.ram_usage.used_mb} / ${status.ram_usage.total_mb} MB`
              : "RAM unavailable"
          }
        />

        <span className="rounded-full border border-slate-700 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-slate-300">
          {mode}
        </span>

        <button
          type="button"
          title="Toggle gaming mode"
          aria-pressed={gamingOn}
          disabled={gaming.isPending}
          onClick={() => gaming.mutate(!gamingOn)}
          className={cx(
            "flex items-center gap-1.5 rounded-md border px-2 py-1 text-[10px] font-medium uppercase tracking-wide transition-colors disabled:opacity-50",
            gamingOn
              ? "border-emerald-700 bg-emerald-950 text-emerald-300"
              : "border-slate-700 text-slate-400 hover:text-slate-200",
          )}
        >
          <Gamepad2 size={13} />
          Gaming
        </button>
      </div>
    </header>
  );
}
