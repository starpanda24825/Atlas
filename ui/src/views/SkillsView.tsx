import { useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { useQuery } from "@tanstack/react-query";
import { Activity, Boxes } from "lucide-react";
import { api, type SkillInfo } from "../lib/api";
import { useAtlasStore } from "../store/atlas";
import { SkillApprovalPanel } from "../components/SkillApprovalPanel";
import { cx, relativeTime } from "../lib/format";

function SkillCodeDialog({ name, onClose }: { name: string | null; onClose: () => void }) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["skill", "code", name],
    queryFn: () => api.skillCode(name as string),
    enabled: name !== null,
  });

  return (
    <Dialog.Root open={name !== null} onOpenChange={(open) => (!open ? onClose() : undefined)}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/70 backdrop-blur-sm" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 flex max-h-[80vh] w-[46rem] max-w-[92vw] -translate-x-1/2 -translate-y-1/2 flex-col rounded-xl border border-slate-700 bg-slate-900 p-5 shadow-2xl">
          <Dialog.Title className="text-base font-semibold text-slate-100">{name}</Dialog.Title>
          {data?.path ? (
            <Dialog.Description className="mt-0.5 text-[11px] text-slate-500">{data.path}</Dialog.Description>
          ) : null}
          <div className="mt-3 min-h-0 flex-1 overflow-auto rounded-lg border border-slate-800 bg-slate-950 p-3">
            {isLoading ? (
              <p className="text-xs text-slate-500">Loading…</p>
            ) : isError ? (
              <p className="text-xs text-rose-400">Could not read this skill's code.</p>
            ) : (
              <pre className="text-[11px] leading-relaxed text-slate-300">
                <code>{data?.code}</code>
              </pre>
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

function ActivityLog() {
  const activity = useAtlasStore((state) => state.activity);
  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/40 p-3">
      <h2 className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-200">
        <Activity size={14} /> Atlas activity
      </h2>
      {activity.length === 0 ? (
        <p className="text-xs text-slate-600">
          Skill builds, approvals and research runs show up here as they happen.
        </p>
      ) : (
        <ul className="max-h-40 space-y-1 overflow-y-auto">
          {activity.map((entry) => (
            <li key={entry.id} className="flex items-center justify-between gap-3 text-xs">
              <span className="flex items-center gap-2 text-slate-300">
                <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[9px] uppercase text-slate-400">
                  {entry.kind}
                </span>
                {entry.text}
              </span>
              <span className="shrink-0 text-[10px] text-slate-600">{relativeTime(new Date(entry.at).toISOString())}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function SkillsView() {
  const [openSkill, setOpenSkill] = useState<string | null>(null);
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["skills"],
    queryFn: () => api.skills(),
  });

  const skills = (data?.items ?? []).filter((skill) => skill.kind !== "builtin");

  return (
    <div className="flex h-full min-h-0 flex-col gap-4 overflow-y-auto">
      <SkillApprovalPanel />

      {isLoading ? <p className="text-sm text-slate-500">Loading skills…</p> : null}
      {isError ? <p className="text-sm text-rose-400">Could not load skills: {error.message}</p> : null}

      {!isLoading && skills.length === 0 ? (
        <div className="mt-8 text-center">
          <Boxes size={28} className="mx-auto text-slate-700" />
          <p className="mt-2 text-sm text-slate-600">No skills built yet.</p>
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {skills.map((skill: SkillInfo) => (
          <button
            key={skill.name}
            type="button"
            onClick={() => setOpenSkill(skill.name)}
            className="flex flex-col rounded-xl border border-slate-800 bg-slate-900/40 p-3 text-left transition-colors hover:border-slate-700 hover:bg-slate-900"
          >
            <span className="text-sm font-medium text-slate-100">{skill.name}</span>
            <span className="mt-1 line-clamp-2 flex-1 text-xs text-slate-500">
              {skill.description || "No description"}
            </span>
            <span className="mt-3 flex items-center gap-3 text-[10px] text-slate-500">
              <span className={cx(skill.has_test ? "text-emerald-500" : "")}>
                {skill.has_test ? "tested" : "untested"}
              </span>
              <span>{skill.call_count ?? 0} calls</span>
              <span className="ml-auto">{skill.last_used ? relativeTime(skill.last_used) : "never"}</span>
            </span>
          </button>
        ))}
      </div>

      <ActivityLog />

      <SkillCodeDialog name={openSkill} onClose={() => setOpenSkill(null)} />
    </div>
  );
}
