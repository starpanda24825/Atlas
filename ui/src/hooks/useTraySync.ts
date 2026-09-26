import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { api } from "../lib/api";
import { useAtlasStore } from "../store/atlas";

/** True only when running inside the Tauri shell rather than a browser. */
function inTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

/**
 * Keeps the native system tray in step with the store, and routes tray menu
 * clicks back into the app.
 *
 * All calls are guarded by {@link inTauri} so the same components run in a
 * plain browser (`npm run dev`) without a Tauri shell.
 */
export function useTraySync(): void {
  const queryClient = useQueryClient();
  const atlasState = useAtlasStore((state) => state.atlasState);
  const connected = useAtlasStore((state) => state.connected);
  const mode = useAtlasStore((state) => state.mode);

  // Store state -> tray icon colour.
  useEffect(() => {
    if (!inTauri()) return;
    const state = connected ? atlasState : "error";
    void invoke("set_atlas_state", { state }).catch(() => undefined);
  }, [atlasState, connected]);

  // Store mode -> tray "Gaming Mode" checkmark.
  useEffect(() => {
    if (!inTauri()) return;
    void invoke("set_gaming_mode", { enabled: mode === "gaming" }).catch(() => undefined);
  }, [mode]);

  // Tray menu -> actions.
  useEffect(() => {
    if (!inTauri()) return;
    const pending: Array<Promise<UnlistenFn>> = [];

    pending.push(
      listen("tray://toggle-gaming", async () => {
        const current = useAtlasStore.getState().mode;
        try {
          const result = await api.gamingToggle(current !== "gaming");
          useAtlasStore.getState().setMode(result.mode);
          void queryClient.invalidateQueries({ queryKey: ["status"] });
        } catch {
          /* tray toggles fail silently; the UI still reflects reality */
        }
      }),
    );

    pending.push(
      listen("tray://shutdown", async () => {
        try {
          await api.shutdown();
        } catch {
          /* the daemon may already be gone */
        }
        try {
          await invoke("quit_app");
        } catch {
          /* not running under Tauri */
        }
      }),
    );

    return () => {
      pending.forEach((ready) => void ready.then((unlisten) => unlisten()));
    };
  }, [queryClient]);
}
