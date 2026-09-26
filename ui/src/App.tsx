import type { ComponentType } from "react";
import { useAtlasStore, type ViewId } from "./store/atlas";
import { useAtlasSocket } from "./hooks/useAtlasSocket";
import { useTraySync } from "./hooks/useTraySync";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { Toaster } from "./components/Toaster";
import { TradeConfirmDialog } from "./components/TradeConfirmDialog";
import { StatusView } from "./views/StatusView";
import { ConversationsView } from "./views/ConversationsView";
import { MemoryView } from "./views/MemoryView";
import { SkillsView } from "./views/SkillsView";
import { ResearchView } from "./views/ResearchView";
import { TradingView } from "./views/TradingView";

const VIEWS: Record<ViewId, ComponentType> = {
  status: StatusView,
  conversations: ConversationsView,
  memory: MemoryView,
  skills: SkillsView,
  research: ResearchView,
  trading: TradingView,
};

export default function App() {
  const view = useAtlasStore((state) => state.view);
  const setView = useAtlasStore((state) => state.setView);

  // Opens the socket and mirrors state to the tray. Both are pure observers —
  // nothing here triggers model work, so the window cannot affect voice latency.
  useAtlasSocket();
  useTraySync();

  const View = VIEWS[view];

  return (
    <div className="flex h-screen w-screen select-none overflow-hidden bg-slate-950 text-slate-200">
      <Sidebar view={view} onSelect={setView} />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar />
        <main className="min-h-0 flex-1 overflow-hidden p-4">
          <View />
        </main>
      </div>
      <TradeConfirmDialog />
      <Toaster />
    </div>
  );
}
