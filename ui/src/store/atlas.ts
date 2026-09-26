import { create } from "zustand";
import type { Status } from "../lib/api";

/** The six sidebar destinations. */
export type ViewId = "status" | "conversations" | "memory" | "skills" | "research" | "trading";

/** What the tray dot and top-bar dot reflect. */
export type AtlasState = "listening" | "thinking" | "error" | "idle";

export interface Turn {
  id: string;
  role: "user" | "atlas";
  text: string;
  at: number;
}

export interface ActivityEntry {
  id: string;
  kind: string;
  text: string;
  at: number;
}

export interface WsEvent {
  type?: string;
  [key: string]: unknown;
}

const MAX_TURNS = 200;
const MAX_ACTIVITY = 100;

function uid(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : null;
}

interface AtlasStore {
  // navigation
  view: ViewId;
  setView: (view: ViewId) => void;

  // live connection / state
  connected: boolean;
  atlasState: AtlasState;
  setAtlasState: (state: AtlasState) => void;

  mode: string;
  setMode: (mode: string) => void;

  status: Status | null;
  setStatus: (status: Status) => void;

  // conversation
  transcript: Turn[];
  addUserTurn: (text: string) => void;
  appendAtlas: (text: string) => void;
  clearTranscript: () => void;

  // skill activity log
  activity: ActivityEntry[];
  pushActivity: (kind: string, text: string) => void;

  // research progress, keyed by job id
  researchProgress: Record<string, string>;
  setProgress: (id: string, progress: string) => void;

  // human-in-the-loop gates
  tradeConfirmation: Record<string, unknown> | null;
  setTradeConfirmation: (trade: Record<string, unknown> | null) => void;
  skillApproval: Record<string, unknown> | null;
  setSkillApproval: (proposal: Record<string, unknown> | null) => void;

  applyEvent: (event: WsEvent) => void;
}

export const useAtlasStore = create<AtlasStore>((set, get) => ({
  view: "status",
  setView: (view) => set({ view }),

  connected: false,
  atlasState: "idle",
  setAtlasState: (atlasState) => set({ atlasState }),

  mode: "idle",
  setMode: (mode) => set({ mode }),

  status: null,
  setStatus: (status) => set({ status, mode: status.active_mode ?? get().mode }),

  transcript: [],
  addUserTurn: (text) =>
    set((state) => {
      const turn: Turn = { id: uid(), role: "user", text, at: Date.now() };
      return {
        transcript: [...state.transcript, turn].slice(-MAX_TURNS),
        atlasState: "thinking" as AtlasState,
      };
    }),
  appendAtlas: (text) =>
    set((state) => {
      const transcript = [...state.transcript];
      const last = transcript[transcript.length - 1];
      if (last && last.role === "atlas") {
        transcript[transcript.length - 1] = { ...last, text: last.text + text };
      } else {
        const turn: Turn = { id: uid(), role: "atlas", text, at: Date.now() };
        transcript.push(turn);
      }
      return { transcript: transcript.slice(-MAX_TURNS), atlasState: "listening" as AtlasState };
    }),
  clearTranscript: () => set({ transcript: [] }),

  activity: [],
  pushActivity: (kind, text) =>
    set((state) => ({
      activity: [{ id: uid(), kind, text, at: Date.now() }, ...state.activity].slice(0, MAX_ACTIVITY),
    })),

  researchProgress: {},
  setProgress: (id, progress) =>
    set((state) => ({ researchProgress: { ...state.researchProgress, [id]: progress } })),

  tradeConfirmation: null,
  setTradeConfirmation: (tradeConfirmation) => set({ tradeConfirmation }),
  skillApproval: null,
  setSkillApproval: (skillApproval) => set({ skillApproval }),

  applyEvent: (event) => {
    const type = event.type;
    switch (type) {
      case "socket":
        set({ connected: event.state === "open" });
        return;
      case "transcript":
        if (typeof event.text === "string" && event.text.trim()) get().addUserTurn(event.text);
        return;
      case "response":
        if (typeof event.text === "string" && event.text) get().appendAtlas(event.text);
        return;
      case "mode_change":
        if (typeof event.mode === "string") {
          get().setMode(event.mode);
          get().pushActivity("mode", `Mode → ${event.mode}`);
        }
        return;
      case "system_status": {
        const status = event as unknown as Status;
        set({ status });
        if (typeof status.active_mode === "string") set({ mode: status.active_mode });
        return;
      }
      case "skill_approval_needed": {
        const proposal = asRecord(event.proposal);
        if (proposal) {
          get().setSkillApproval(proposal);
          const name = String(proposal.name ?? proposal.skill_name ?? "skill");
          get().pushActivity("skill", `Approval needed: ${name}`);
        }
        return;
      }
      case "trade_confirmation_needed": {
        const trade = asRecord(event.trade);
        if (trade) {
          get().setTradeConfirmation(trade);
          get().pushActivity("trade", "Trade awaiting confirmation");
        }
        return;
      }
      case "research_progress": {
        if (typeof event.id === "string" && typeof event.progress === "string") {
          get().setProgress(event.id, event.progress);
        }
        return;
      }
      case "research_complete": {
        const report = asRecord(event.report);
        const id = typeof event.id === "string" ? event.id : String(report?.id ?? "");
        if (id) {
          set((state) => {
            const next = { ...state.researchProgress };
            delete next[id];
            return { researchProgress: next };
          });
        }
        const title = String(report?.title ?? report?.question ?? "research");
        get().pushActivity("research", `Research complete: ${title}`);
        return;
      }
      default:
        return;
    }
  },
}));
