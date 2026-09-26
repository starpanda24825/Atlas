import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, ShieldAlert, X } from "lucide-react";
import { api, type PendingSkill } from "../lib/api";
import { useAtlasStore } from "../store/atlas";
import { toast } from "../store/toasts";

interface NormalisedProposal {
  name: string;
  description: string;
  code: string;
  warnings: string[];
  tests: string;
  blockedOnNetwork: boolean;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : undefined;
}

function normalise(item: PendingSkill): NormalisedProposal {
  const proposal = asRecord(item.proposal) ?? item;
  const code = String(proposal.code ?? item.code ?? "");
  const warnings = proposal.warnings ?? item.warnings;
  const tests = proposal.test_results ?? item.test_results;
  return {
    name: String(proposal.name ?? item.name ?? item.skill_name ?? "unnamed skill"),
    description: String(proposal.description ?? item.description ?? ""),
    code,
    warnings: Array.isArray(warnings) ? warnings.map(String) : [],
    tests: tests ? JSON.stringify(tests, null, 2) : "",
    blockedOnNetwork: Boolean(proposal.blocked_on_network ?? item.blocked_on_network),
  };
}

export function SkillApprovalPanel() {
  const queryClient = useQueryClient();
  const pushActivity = useAtlasStore((state) => state.pushActivity);

  const { data, isLoading } = useQuery({
    queryKey: ["skills", "pending"],
    queryFn: api.pendingSkills,
    refetchInterval: 15000,
  });

  const approve = useMutation({
    mutationFn: (name: string) => api.approveSkill(name),
    onSuccess: (_result, name) => {
      pushActivity("skill", `Approved ${name}`);
      toast.info("Skill approved", name);
      void queryClient.invalidateQueries({ queryKey: ["skills"] });
    },
    onError: (error: Error) => toast.error("Could not approve skill", error.message),
  });

  const reject = useMutation({
    mutationFn: (name: string) => api.rejectSkill(name),
    onSuccess: (_result, name) => {
      pushActivity("skill", `Rejected ${name}`);
      toast.info("Skill rejected", name);
      void queryClient.invalidateQueries({ queryKey: ["skills"] });
    },
    onError: (error: Error) => toast.error("Could not reject skill", error.message),
  });

  const items = (data?.items ?? []).map(normalise);

  if (isLoading || items.length === 0) return null;

  return (
    <section className="rounded-xl border border-amber-900/60 bg-amber-950/20 p-4">
      <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-amber-300">
        <ShieldAlert size={16} />
        Skills awaiting approval
        <span className="rounded-full bg-amber-900/60 px-2 py-0.5 text-[10px] text-amber-200">
          {items.length}
        </span>
      </h2>

      <div className="space-y-4">
        {items.map((item) => (
          <article key={item.name} className="rounded-lg border border-slate-800 bg-slate-900/60 p-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h3 className="text-sm font-medium text-slate-100">{item.name}</h3>
                {item.description ? (
                  <p className="mt-0.5 text-xs text-slate-400">{item.description}</p>
                ) : null}
              </div>
              <div className="flex shrink-0 gap-2">
                <button
                  type="button"
                  disabled={approve.isPending || reject.isPending}
                  onClick={() => approve.mutate(item.name)}
                  className="flex items-center gap-1 rounded-md bg-emerald-700 px-2.5 py-1 text-xs font-medium text-white hover:bg-emerald-600 disabled:opacity-50"
                >
                  <Check size={13} /> Approve
                </button>
                <button
                  type="button"
                  disabled={approve.isPending || reject.isPending}
                  onClick={() => reject.mutate(item.name)}
                  className="flex items-center gap-1 rounded-md border border-slate-700 px-2.5 py-1 text-xs font-medium text-slate-300 hover:bg-slate-800 disabled:opacity-50"
                >
                  <X size={13} /> Reject
                </button>
              </div>
            </div>

            {item.blockedOnNetwork ? (
              <p className="mt-2 rounded bg-amber-950/50 px-2 py-1 text-xs text-amber-300">
                This skill requests network access — approving allows it to reach the listed hosts.
              </p>
            ) : null}

            {item.code ? (
              <pre className="mt-3 max-h-48 overflow-auto rounded border border-slate-800 bg-slate-950 p-2 text-[11px] leading-relaxed text-slate-300">
                <code>{item.code}</code>
              </pre>
            ) : null}

            {item.warnings.length > 0 ? (
              <ul className="mt-2 list-disc space-y-0.5 pl-4 text-[11px] text-amber-300">
                {item.warnings.map((warning, index) => (
                  <li key={index}>{warning}</li>
                ))}
              </ul>
            ) : null}

            {item.tests ? (
              <details className="mt-2">
                <summary className="cursor-pointer text-[11px] text-slate-400">Test results</summary>
                <pre className="mt-1 max-h-40 overflow-auto rounded border border-slate-800 bg-slate-950 p-2 text-[11px] text-slate-400">
                  <code>{item.tests}</code>
                </pre>
              </details>
            ) : null}
          </article>
        ))}
      </div>
    </section>
  );
}
