import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronLeft, ChevronRight, Search, Trash2 } from "lucide-react";
import { api } from "../lib/api";
import { useAtlasStore } from "../store/atlas";
import { toast } from "../store/toasts";
import { Markdown } from "../components/Markdown";
import { cx, shortDate } from "../lib/format";

const PAGE_SIZE = 20;

export function ConversationsView() {
  const queryClient = useQueryClient();
  const pushActivity = useAtlasStore((state) => state.pushActivity);
  const [page, setPage] = useState(0);
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);

  const searching = query.trim().length > 0;
  const limit = searching ? 200 : PAGE_SIZE;
  const offset = searching ? 0 : page * PAGE_SIZE;

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["conversations", limit, offset],
    queryFn: () => api.conversations(limit, offset),
  });

  const detail = useQuery({
    queryKey: ["conversation", expanded],
    queryFn: () => api.conversation(expanded as string),
    enabled: expanded !== null,
  });

  const remove = useMutation({
    mutationFn: (id: string) => api.deleteConversation(id),
    onSuccess: (_result, id) => {
      pushActivity("conversation", `Deleted conversation ${id}`);
      toast.info("Conversation deleted", id);
      void queryClient.invalidateQueries({ queryKey: ["conversations"] });
      setExpanded((current) => (current === id ? null : current));
    },
    onError: (mutationError: Error) => toast.error("Delete failed", mutationError.message),
  });

  const items = useMemo(() => {
    const all = data?.items ?? [];
    if (!searching) return all;
    const needle = query.trim().toLowerCase();
    return all.filter(
      (item) =>
        item.title.toLowerCase().includes(needle) ||
        item.summary.toLowerCase().includes(needle),
    );
  }, [data, query, searching]);

  const total = data?.total ?? 0;
  const lastPage = Math.max(0, Math.ceil(total / PAGE_SIZE) - 1);

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <div className="flex items-center gap-3">
        <div className="relative flex-1">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setPage(0);
            }}
            placeholder="Search conversations…"
            className="w-full rounded-lg border border-slate-800 bg-slate-900 py-2 pl-9 pr-3 text-sm text-slate-200 outline-none placeholder:text-slate-600 focus:border-sky-700"
          />
        </div>
        <span className="text-xs text-slate-500">
          {searching ? `${items.length} matches` : `${total} total`}
        </span>
      </div>

      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto">
        {isLoading ? <p className="text-sm text-slate-500">Loading…</p> : null}
        {isError ? (
          <p className="text-sm text-rose-400">Could not load conversations: {error.message}</p>
        ) : null}
        {!isLoading && items.length === 0 ? (
          <p className="mt-10 text-center text-sm text-slate-600">No conversations found.</p>
        ) : null}

        {items.map((item) => {
          const isOpen = expanded === item.id;
          return (
            <article key={item.id} className="rounded-lg border border-slate-800 bg-slate-900/40">
              <div className="flex items-center gap-2 px-3 py-2.5">
                <button
                  type="button"
                  onClick={() => setExpanded(isOpen ? null : item.id)}
                  className="flex min-w-0 flex-1 items-center gap-2 text-left"
                >
                  <ChevronDown
                    size={15}
                    className={cx("shrink-0 text-slate-500 transition-transform", isOpen && "rotate-180")}
                  />
                  <span className="min-w-0">
                    <span className="block truncate text-sm font-medium text-slate-200">{item.title}</span>
                    {item.summary ? (
                      <span className="block truncate text-xs text-slate-500">{item.summary}</span>
                    ) : null}
                  </span>
                </button>
                <span className="shrink-0 text-[10px] text-slate-600">{shortDate(item.created || item.date)}</span>
                <button
                  type="button"
                  title="Delete conversation"
                  disabled={remove.isPending}
                  onClick={() => {
                    if (window.confirm(`Delete "${item.title}"? This cannot be undone.`)) {
                      remove.mutate(item.id);
                    }
                  }}
                  className="shrink-0 rounded p-1 text-slate-600 hover:bg-slate-800 hover:text-rose-400 disabled:opacity-50"
                >
                  <Trash2 size={14} />
                </button>
              </div>

              {isOpen ? (
                <div className="border-t border-slate-800 px-4 py-3">
                  {detail.isLoading ? (
                    <p className="text-xs text-slate-500">Loading exchange…</p>
                  ) : detail.data ? (
                    <Markdown text={detail.data.body} />
                  ) : (
                    <p className="text-xs text-rose-400">Could not load this exchange.</p>
                  )}
                </div>
              ) : null}
            </article>
          );
        })}
      </div>

      {!searching && total > PAGE_SIZE ? (
        <div className="flex items-center justify-center gap-3">
          <button
            type="button"
            disabled={page <= 0}
            onClick={() => setPage((value) => Math.max(0, value - 1))}
            className="flex items-center gap-1 rounded-md border border-slate-800 px-2 py-1 text-xs text-slate-400 hover:bg-slate-800 disabled:opacity-40"
          >
            <ChevronLeft size={13} /> Prev
          </button>
          <span className="text-xs text-slate-500">
            Page {page + 1} of {lastPage + 1}
          </span>
          <button
            type="button"
            disabled={page >= lastPage}
            onClick={() => setPage((value) => value + 1)}
            className="flex items-center gap-1 rounded-md border border-slate-800 px-2 py-1 text-xs text-slate-400 hover:bg-slate-800 disabled:opacity-40"
          >
            Next <ChevronRight size={13} />
          </button>
        </div>
      ) : null}
    </div>
  );
}
