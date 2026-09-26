import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Search, Trash2 } from "lucide-react";
import { api } from "../lib/api";
import { useAtlasStore } from "../store/atlas";
import { toast } from "../store/toasts";
import { Markdown } from "../components/Markdown";
import { cx, pickString, relativeTime } from "../lib/format";

type NoteType = "memory" | "conversation" | "research" | "skill";

const NOTE_TABS: Array<{ id: NoteType; label: string; folder: string }> = [
  { id: "memory", label: "Memories", folder: "memories" },
  { id: "conversation", label: "Conversations", folder: "conversations" },
  { id: "research", label: "Research", folder: "research" },
  { id: "skill", label: "Skills", folder: "skills" },
];

function MemoryPanel() {
  const queryClient = useQueryClient();
  const pushActivity = useAtlasStore((state) => state.pushActivity);
  const [query, setQuery] = useState("");

  const searching = query.trim().length > 0;
  const all = useQuery({ queryKey: ["memories"], queryFn: () => api.memories(), enabled: !searching });
  const found = useQuery({
    queryKey: ["memories", "search", query],
    queryFn: () => api.searchMemories(query.trim()),
    enabled: searching,
  });

  const remove = useMutation({
    mutationFn: (id: string) => api.deleteMemory(id),
    onSuccess: (_result, id) => {
      pushActivity("memory", `Forgot memory ${id}`);
      toast.info("Memory removed", id);
      void queryClient.invalidateQueries({ queryKey: ["memories"] });
    },
    onError: (error: Error) => toast.error("Could not remove memory", error.message),
  });

  const items = searching ? found.data?.results ?? [] : all.data?.items ?? [];
  const loading = searching ? found.isLoading : all.isLoading;

  return (
    <section className="flex min-h-0 flex-col rounded-xl border border-slate-800 bg-slate-900/40">
      <div className="relative border-b border-slate-800 p-2">
        <Search size={13} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" />
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search memories…"
          className="w-full rounded-md border border-slate-800 bg-slate-950 py-1.5 pl-8 pr-2 text-xs text-slate-200 outline-none placeholder:text-slate-600 focus:border-sky-700"
        />
      </div>
      <div className="min-h-0 flex-1 space-y-1.5 overflow-y-auto p-2">
        {loading ? <p className="p-2 text-xs text-slate-500">Loading…</p> : null}
        {!loading && items.length === 0 ? (
          <p className="p-4 text-center text-xs text-slate-600">No memories.</p>
        ) : null}
        {items.map((item) => {
          const text = pickString(item as Record<string, unknown>, ["memory", "text", "content", "value"]);
          return (
            <div
              key={item.id}
              className="group flex items-start gap-2 rounded-md border border-slate-800/70 bg-slate-950/50 px-2.5 py-2"
            >
              <p className="flex-1 text-xs leading-relaxed text-slate-300">{text || item.id}</p>
              <button
                type="button"
                title="Delete memory"
                disabled={remove.isPending}
                onClick={() => remove.mutate(item.id)}
                className="shrink-0 rounded p-1 text-slate-600 opacity-0 transition-opacity hover:text-rose-400 group-hover:opacity-100 disabled:opacity-50"
              >
                <Trash2 size={13} />
              </button>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function VaultPanel() {
  const [type, setType] = useState<NoteType>("memory");
  const [selected, setSelected] = useState<string | null>(null);
  const tab = NOTE_TABS.find((entry) => entry.id === type) ?? NOTE_TABS[0];

  const notes = useQuery({
    queryKey: ["vault", type],
    queryFn: () => api.vaultNotes(type),
  });

  const detail = useQuery({
    queryKey: ["vault", tab.folder, selected],
    queryFn: () => api.vaultNote(tab.folder, selected as string),
    enabled: selected !== null,
  });

  return (
    <section className="flex min-h-0 flex-col rounded-xl border border-slate-800 bg-slate-900/40">
      <div className="flex items-center gap-1 border-b border-slate-800 px-2 py-1.5">
        {NOTE_TABS.map((entry) => (
          <button
            key={entry.id}
            type="button"
            onClick={() => {
              setType(entry.id);
              setSelected(null);
            }}
            className={cx(
              "rounded px-2 py-1 text-[11px] font-medium transition-colors",
              entry.id === type ? "bg-slate-800 text-sky-400" : "text-slate-500 hover:text-slate-300",
            )}
          >
            {entry.label}
          </button>
        ))}
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,12rem)_1fr]">
        <div className="min-h-0 space-y-1 overflow-y-auto border-r border-slate-800 p-2">
          {notes.isLoading ? <p className="text-xs text-slate-500">Loading…</p> : null}
          {(notes.data?.items ?? []).length === 0 && !notes.isLoading ? (
            <p className="p-2 text-center text-xs text-slate-600">Nothing here.</p>
          ) : null}
          {(notes.data?.items ?? []).map((note) => (
            <button
              key={note.id}
              type="button"
              onClick={() => setSelected(note.id)}
              className={cx(
                "block w-full truncate rounded px-2 py-1.5 text-left text-xs",
                selected === note.id ? "bg-slate-800 text-slate-100" : "text-slate-400 hover:bg-slate-900",
              )}
              title={note.title}
            >
              {note.title || note.id}
            </button>
          ))}
        </div>

        <div className="min-h-0 overflow-y-auto p-3">
          {selected === null ? (
            <p className="mt-8 text-center text-xs text-slate-600">Select a note to read it.</p>
          ) : detail.isLoading ? (
            <p className="text-xs text-slate-500">Loading…</p>
          ) : detail.data ? (
            <>
              <div className="mb-2 text-[10px] text-slate-500">
                {detail.data.type} · {relativeTime(detail.data.created)}
              </div>
              <Markdown text={detail.data.body} />
            </>
          ) : (
            <p className="text-xs text-rose-400">Could not load this note.</p>
          )}
        </div>
      </div>
    </section>
  );
}

export function MemoryView() {
  return (
    <div className="grid h-full min-h-0 grid-cols-2 gap-4">
      <MemoryPanel />
      <VaultPanel />
    </div>
  );
}
