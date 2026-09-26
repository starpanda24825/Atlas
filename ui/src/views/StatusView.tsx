import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2, Trash2 } from "lucide-react";
import { api } from "../lib/api";
import { useAtlasStore } from "../store/atlas";
import { SkillApprovalPanel } from "../components/SkillApprovalPanel";
import { cx, relativeTime } from "../lib/format";

function Transcript() {
  const transcript = useAtlasStore((state) => state.transcript);
  const clearTranscript = useAtlasStore((state) => state.clearTranscript);
  const bottom = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript]);

  return (
    <section className="flex min-h-0 flex-1 flex-col rounded-xl border border-slate-800 bg-slate-900/40">
      <div className="flex items-center justify-between border-b border-slate-800 px-4 py-2">
        <h2 className="text-sm font-semibold text-slate-200">Live conversation</h2>
        {transcript.length > 0 ? (
          <button
            type="button"
            onClick={clearTranscript}
            title="Clear the transcript view"
            className="flex items-center gap-1 rounded px-1.5 py-1 text-[11px] text-slate-500 hover:text-slate-300"
          >
            <Trash2 size={12} /> Clear
          </button>
        ) : null}
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {transcript.length === 0 ? (
          <p className="mt-10 text-center text-sm text-slate-600">
            The live transcript appears here as you talk to Atlas.
          </p>
        ) : (
          transcript.map((turn) => (
            <div
              key={turn.id}
              className={cx("flex", turn.role === "user" ? "justify-end" : "justify-start")}
            >
              <div
                className={cx(
                  "max-w-[78%] rounded-2xl px-3.5 py-2 text-sm leading-relaxed",
                  turn.role === "user"
                    ? "bg-sky-900/70 text-sky-50"
                    : "bg-slate-800/80 text-slate-100",
                )}
              >
                {turn.role === "atlas" ? (
                  <span className="mb-0.5 block text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                    Atlas
                  </span>
                ) : null}
                <span className="whitespace-pre-wrap">{turn.text}</span>
              </div>
            </div>
          ))
        )}
        <div ref={bottom} />
      </div>
    </section>
  );
}

function ActiveResearch() {
  const progress = useAtlasStore((state) => state.researchProgress);
  const { data } = useQuery({
    queryKey: ["research", "jobs"],
    queryFn: api.researchJobs,
    refetchInterval: 3000,
  });

  const running = (data?.items ?? []).filter((job) => job.status === "running");
  if (running.length === 0) return null;

  return (
    <section className="rounded-xl border border-violet-900/50 bg-violet-950/20 p-3">
      <h2 className="mb-2 text-sm font-semibold text-violet-300">Active research</h2>
      <div className="space-y-2">
        {running.map((job) => {
          const stage = progress[job.id] ?? job.progress ?? "running";
          return (
            <div key={job.id} className="rounded-lg border border-slate-800 bg-slate-900/60 p-2.5">
              <div className="flex items-center justify-between gap-3">
                <p className="truncate text-xs font-medium text-slate-200">{job.question}</p>
                <span className="shrink-0 text-[10px] text-slate-500">
                  started {relativeTime(job.started)}
                </span>
              </div>
              <p className="mt-1 flex items-center gap-1.5 text-[11px] text-violet-300">
                <Loader2 size={11} className="animate-spin" />
                {stage}
              </p>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export function StatusView() {
  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <Transcript />
      <div className="max-h-72 space-y-4 overflow-y-auto">
        <SkillApprovalPanel />
        <ActiveResearch />
      </div>
    </div>
  );
}
