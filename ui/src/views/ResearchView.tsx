import { useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileText, Loader2, Plus } from "lucide-react";
import { api } from "../lib/api";
import { useAtlasStore } from "../store/atlas";
import { toast } from "../store/toasts";
import { Markdown } from "../components/Markdown";
import { relativeTime, shortDate } from "../lib/format";

function StartResearchDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const queryClient = useQueryClient();
  const [question, setQuestion] = useState("");
  const [depth, setDepth] = useState(3);

  const start = useMutation({
    mutationFn: () => api.startResearch(question.trim(), depth),
    onSuccess: () => {
      toast.info("Research started", "Atlas will report back when it's done.");
      void queryClient.invalidateQueries({ queryKey: ["research", "jobs"] });
      setQuestion("");
      onClose();
    },
    onError: (error: Error) => toast.error("Could not start research", error.message),
  });

  return (
    <Dialog.Root open={open} onOpenChange={(next) => (!next ? onClose() : undefined)}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/70 backdrop-blur-sm" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[32rem] max-w-[92vw] -translate-x-1/2 -translate-y-1/2 rounded-xl border border-slate-700 bg-slate-900 p-5 shadow-2xl">
          <Dialog.Title className="text-base font-semibold text-slate-100">Start research</Dialog.Title>
          <Dialog.Description className="mt-1 text-xs text-slate-400">
            Atlas will plan sub-questions, search, read sources and write a report.
          </Dialog.Description>

          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            rows={3}
            autoFocus
            placeholder="What should Atlas research?"
            className="mt-4 w-full resize-none rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-sm text-slate-200 outline-none placeholder:text-slate-600 focus:border-sky-700"
          />

          <div className="mt-4">
            <label className="flex items-center justify-between text-xs text-slate-400">
              <span>Depth</span>
              <span className="tabular-nums text-slate-300">{depth}</span>
            </label>
            <input
              type="range"
              min={1}
              max={5}
              value={depth}
              onChange={(event) => setDepth(Number(event.target.value))}
              className="mt-2 w-full accent-sky-500"
            />
          </div>

          <div className="mt-5 flex justify-end gap-2">
            <Dialog.Close asChild>
              <button
                type="button"
                className="rounded-md border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-800"
              >
                Cancel
              </button>
            </Dialog.Close>
            <button
              type="button"
              disabled={start.isPending || question.trim().length === 0}
              onClick={() => start.mutate()}
              className="rounded-md bg-sky-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50"
            >
              {start.isPending ? "Starting…" : "Start research"}
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function ReportDialog({ id, onClose }: { id: string | null; onClose: () => void }) {
  const { data, isLoading } = useQuery({
    queryKey: ["research", "report", id],
    queryFn: () => api.researchReport(id as string),
    enabled: id !== null,
  });

  const sources = Array.isArray(data?.metadata?.sources) ? (data?.metadata?.sources as string[]) : [];

  return (
    <Dialog.Root open={id !== null} onOpenChange={(open) => (!open ? onClose() : undefined)}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/70 backdrop-blur-sm" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 flex max-h-[85vh] w-[48rem] max-w-[92vw] -translate-x-1/2 -translate-y-1/2 flex-col rounded-xl border border-slate-700 bg-slate-900 p-5 shadow-2xl">
          <Dialog.Title className="text-base font-semibold text-slate-100">
            {data?.title ?? "Research report"}
          </Dialog.Title>
          <div className="mt-3 min-h-0 flex-1 overflow-y-auto rounded-lg border border-slate-800 bg-slate-950/50 p-4">
            {isLoading ? (
              <p className="text-xs text-slate-500">Loading…</p>
            ) : (
              <>
                <Markdown text={data?.body ?? ""} />
                {sources.length > 0 ? (
                  <div className="mt-4 border-t border-slate-800 pt-3">
                    <p className="text-[11px] uppercase tracking-wide text-slate-500">Sources</p>
                    <ol className="mt-1 space-y-0.5 text-[11px] text-slate-400">
                      {sources.map((source, index) => (
                        <li key={index} className="truncate">
                          <a href={source} target="_blank" rel="noreferrer" className="hover:text-sky-400">
                            {source}
                          </a>
                        </li>
                      ))}
                    </ol>
                  </div>
                ) : null}
              </>
            )}
          </div>
          <div className="mt-4 flex justify-end">
            <Dialog.Close asChild>
              <button
                type="button"
                className="rounded-md border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-800"
              >
                Close
              </button>
            </Dialog.Close>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function ActiveJobs() {
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
      <h2 className="mb-2 text-sm font-semibold text-violet-300">In progress</h2>
      <div className="space-y-2">
        {running.map((job) => (
          <div key={job.id} className="rounded-lg border border-slate-800 bg-slate-900/60 p-2.5">
            <div className="flex items-center justify-between gap-3">
              <p className="truncate text-xs font-medium text-slate-200">{job.question}</p>
              <span className="shrink-0 text-[10px] text-slate-500">
                depth {job.depth} · {relativeTime(job.started)}
              </span>
            </div>
            <p className="mt-1 flex items-center gap-1.5 text-[11px] text-violet-300">
              <Loader2 size={11} className="animate-spin" />
              {progress[job.id] ?? job.progress ?? "running"}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}

export function ResearchView() {
  const [startOpen, setStartOpen] = useState(false);
  const [openReport, setOpenReport] = useState<string | null>(null);

  const { data, isLoading } = useQuery({ queryKey: ["research"], queryFn: api.research });

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <div className="flex items-center justify-between">
        <h1 className="text-sm font-semibold text-slate-200">Research reports</h1>
        <button
          type="button"
          onClick={() => setStartOpen(true)}
          className="flex items-center gap-1.5 rounded-md bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500"
        >
          <Plus size={14} /> Start research
        </button>
      </div>

      <ActiveJobs />

      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto">
        {isLoading ? <p className="text-sm text-slate-500">Loading…</p> : null}
        {!isLoading && (data?.items ?? []).length === 0 ? (
          <div className="mt-12 text-center">
            <FileText size={28} className="mx-auto text-slate-700" />
            <p className="mt-2 text-sm text-slate-600">No research reports yet.</p>
          </div>
        ) : null}
        {(data?.items ?? []).map((report) => (
          <button
            key={report.id}
            type="button"
            onClick={() => setOpenReport(report.id)}
            className="flex w-full items-center gap-3 rounded-lg border border-slate-800 bg-slate-900/40 px-3 py-2.5 text-left transition-colors hover:border-slate-700 hover:bg-slate-900"
          >
            <FileText size={16} className="shrink-0 text-slate-600" />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm font-medium text-slate-200">{report.title}</span>
              {report.summary ? (
                <span className="block truncate text-xs text-slate-500">{report.summary}</span>
              ) : null}
            </span>
            <span className="shrink-0 text-right text-[10px] text-slate-500">
              <span className="block">{shortDate(report.created || report.date)}</span>
              <span className="block">
                {report.source_count != null ? `${report.source_count} sources` : ""}
              </span>
            </span>
          </button>
        ))}
      </div>

      <StartResearchDialog open={startOpen} onClose={() => setStartOpen(false)} />
      <ReportDialog id={openReport} onClose={() => setOpenReport(null)} />
    </div>
  );
}
