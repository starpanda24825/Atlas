import {
  Activity,
  Blocks,
  Brain,
  FlaskConical,
  MessagesSquare,
  TrendingUp,
  type LucideIcon,
} from "lucide-react";
import type { ViewId } from "../store/atlas";
import { cx } from "../lib/format";

const ITEMS: Array<{ id: ViewId; label: string; Icon: LucideIcon }> = [
  { id: "status", label: "Status", Icon: Activity },
  { id: "conversations", label: "Conversations", Icon: MessagesSquare },
  { id: "memory", label: "Memory", Icon: Brain },
  { id: "skills", label: "Skills", Icon: Blocks },
  { id: "research", label: "Research", Icon: FlaskConical },
  { id: "trading", label: "Trading", Icon: TrendingUp },
];

export function Sidebar({
  view,
  onSelect,
}: {
  view: ViewId;
  onSelect: (view: ViewId) => void;
}) {
  return (
    <nav className="flex w-14 flex-col items-center gap-1 border-r border-slate-800 bg-slate-950 py-3">
      {ITEMS.map(({ id, label, Icon }) => {
        const active = id === view;
        return (
          <button
            key={id}
            type="button"
            title={label}
            aria-label={label}
            aria-current={active ? "page" : undefined}
            onClick={() => onSelect(id)}
            className={cx(
              "flex h-10 w-10 items-center justify-center rounded-lg transition-colors",
              active
                ? "bg-slate-800 text-sky-400"
                : "text-slate-500 hover:bg-slate-900 hover:text-slate-200",
            )}
          >
            <Icon size={19} strokeWidth={1.9} />
          </button>
        );
      })}
    </nav>
  );
}
