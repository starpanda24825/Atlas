/**
 * Typed REST client for the local Atlas API (127.0.0.1:8765).
 *
 * Every call goes straight to the daemon — there is no cloud hop. Failures
 * throw an ApiError carrying the daemon's `detail`, so views can show a real
 * message instead of "something went wrong".
 */

export const API_BASE = "http://127.0.0.1:8765";
export const WS_URL = "ws://127.0.0.1:8765/ws";

export interface Usage {
  used_mb: number;
  total_mb: number;
  percent: number;
}

export interface Status {
  active_model: "fast" | "deep" | "none";
  vram_usage: Usage | null;
  ram_usage: Usage | null;
  active_mode: string;
  wake_word_listening: boolean;
  deep_server_running: boolean;
}

export interface ModeResult {
  mode: string;
  previous?: string;
}

export interface Page<T> {
  total: number;
  items: T[];
  limit?: number;
  offset?: number;
}

export interface NoteBrief {
  id: string;
  title: string;
  summary: string;
  type: string;
  tags: string[];
  date: string;
  created: string;
  source_count?: number | null;
  question?: string;
}

export interface NoteDetail extends NoteBrief {
  body: string;
  metadata: Record<string, unknown>;
}

export interface MemoryItem {
  id: string;
  [key: string]: unknown;
}

export interface SkillInfo {
  name: string;
  description?: string;
  kind?: string;
  call_count?: number;
  last_used?: string;
  has_test?: boolean;
  path?: string | null;
  [key: string]: unknown;
}

export interface PendingSkill {
  name?: string;
  skill_name?: string;
  description?: string;
  code?: string;
  warnings?: string[];
  blocked_on_network?: boolean;
  [key: string]: unknown;
}

export interface ResearchJob {
  id: string;
  question: string;
  depth: number;
  status: string;
  progress?: string;
  started?: string;
  finished?: string | null;
  report?: NoteDetail | null;
  error?: string | null;
}

export type Position = Record<string, unknown>;
export type Trade = Record<string, unknown>;
export type JournalEntry = Record<string, unknown>;
export type DocumentInfo = Record<string, unknown>;

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError(0, `Atlas API unreachable at ${API_BASE}`);
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  const text = await response.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

const json = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const api = {
  // -- status & mode --------------------------------------------------
  status: () => request<Status>("/status"),
  mode: () => request<ModeResult>("/mode"),
  setMode: (mode: string) => request<ModeResult>(`/mode/${mode}`, { method: "POST" }),
  gamingToggle: (enabled?: boolean) =>
    request<ModeResult>("/system/gaming-mode/toggle", json("POST", enabled === undefined ? {} : { enabled })),
  shutdown: () => request<{ status: string }>("/system/shutdown", { method: "POST" }),

  // -- conversations --------------------------------------------------
  conversations: (limit = 20, offset = 0) =>
    request<Page<NoteBrief>>(`/conversations?limit=${limit}&offset=${offset}`),
  conversation: (id: string) => request<NoteDetail>(`/conversations/${encodeURIComponent(id)}`),
  deleteConversation: (id: string) =>
    request<{ deleted: boolean }>(`/conversations/${encodeURIComponent(id)}`, { method: "DELETE" }),

  // -- memory ---------------------------------------------------------
  memories: (limit?: number) =>
    request<Page<MemoryItem>>(`/memories${limit ? `?limit=${limit}` : ""}`),
  deleteMemory: (id: string) =>
    request<{ deleted: boolean }>(`/memories/${encodeURIComponent(id)}`, { method: "DELETE" }),
  searchMemories: (query: string, limit?: number) =>
    request<{ query: string; results: MemoryItem[] }>(
      "/memories/search",
      json("POST", limit ? { query, limit } : { query }),
    ),

  // -- vault ----------------------------------------------------------
  vaultNotes: (noteType?: string, limit?: number) => {
    const params = new URLSearchParams();
    if (noteType) params.set("note_type", noteType);
    if (limit) params.set("limit", String(limit));
    const qs = params.toString();
    return request<Page<NoteBrief>>(`/vault/notes${qs ? `?${qs}` : ""}`);
  },
  vaultNote: (folder: string, id: string) =>
    request<NoteDetail>(`/vault/notes/${folder}/${encodeURIComponent(id)}`),

  // -- skills ---------------------------------------------------------
  skills: (includeBuiltin = true) =>
    request<Page<SkillInfo>>(`/skills?include_builtin=${includeBuiltin}`),
  pendingSkills: () => request<Page<PendingSkill>>("/skills/pending"),
  skillCode: (name: string) =>
    request<{ name: string; path: string; code: string }>(`/skills/${encodeURIComponent(name)}/code`),
  approveSkill: (name: string) =>
    request<{ approved: boolean; name: string }>(`/skills/${encodeURIComponent(name)}/approve`, {
      method: "POST",
    }),
  rejectSkill: (name: string) =>
    request<{ rejected: boolean; name: string }>(`/skills/${encodeURIComponent(name)}/reject`, {
      method: "POST",
    }),
  deleteSkill: (name: string) =>
    request<{ deleted: boolean }>(`/skills/${encodeURIComponent(name)}`, { method: "DELETE" }),

  // -- research -------------------------------------------------------
  research: () => request<Page<NoteBrief>>("/research"),
  researchReport: (id: string) => request<NoteDetail>(`/research/${encodeURIComponent(id)}`),
  researchJobs: () => request<Page<ResearchJob>>("/research/jobs"),
  startResearch: (question: string, depth: number) =>
    request<{ job: ResearchJob }>("/research", json("POST", { question, depth })),

  // -- trading --------------------------------------------------------
  positions: () => request<{ positions: Position[] }>("/trading/positions"),
  history: (limit?: number) =>
    request<{ trades: Trade[] }>(`/trading/history${limit ? `?limit=${limit}` : ""}`),
  journal: (limit?: number) =>
    request<{ entries: JournalEntry[] }>(`/trading/journal${limit ? `?limit=${limit}` : ""}`),
  confirmTrade: (tradeId: string, confirmed: boolean) =>
    request<{ trade_id: string; confirmed: boolean; result: unknown }>(
      "/trading/confirm",
      json("POST", { trade_id: tradeId, confirmed }),
    ),

  // -- documents ------------------------------------------------------
  documents: () => request<{ documents: DocumentInfo[] }>("/documents"),
  uploadDocument: async (file: File) => {
    const response = await fetch(
      `${API_BASE}/documents/upload?filename=${encodeURIComponent(file.name)}`,
      {
        method: "POST",
        headers: { "Content-Type": file.type || "application/octet-stream" },
        body: file,
      },
    );
    if (!response.ok) throw new ApiError(response.status, response.statusText);
    return (await response.json()) as { indexed: boolean; filename: string };
  },
  deleteDocument: (id: string) =>
    request<{ deleted: boolean }>(`/documents/${encodeURIComponent(id)}`, { method: "DELETE" }),
  generateQuiz: (payload: { topic?: string; documents?: string[]; num_questions: number }) =>
    request<{ quiz: unknown }>("/documents/quiz", json("POST", payload)),
};
