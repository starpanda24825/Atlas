import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { WS_URL } from "../lib/api";
import { useAtlasStore, type WsEvent } from "../store/atlas";

/**
 * Connects to `/ws` on mount and feeds every event into the Zustand store.
 *
 * Reconnects with a capped backoff so a daemon restart does not leave the UI
 * dead. This hook only *reads* from the daemon — it never triggers model work,
 * so opening or closing the window has no effect on voice latency.
 */
export function useAtlasSocket(): void {
  const queryClient = useQueryClient();

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retry: number | undefined;
    let attempt = 0;
    let stopped = false;

    const dispatch = (event: WsEvent) => {
      useAtlasStore.getState().applyEvent(event);
      if (event.type === "skill_approval_needed") {
        void queryClient.invalidateQueries({ queryKey: ["skills", "pending"] });
      } else if (event.type === "research_progress" || event.type === "research_complete") {
        void queryClient.invalidateQueries({ queryKey: ["research", "jobs"] });
        if (event.type === "research_complete") {
          void queryClient.invalidateQueries({ queryKey: ["research"] });
        }
      } else if (event.type === "mode_change") {
        void queryClient.invalidateQueries({ queryKey: ["status"] });
      }
    };

    const connect = () => {
      socket = new WebSocket(WS_URL);
      socket.onopen = () => {
        attempt = 0;
        useAtlasStore.getState().applyEvent({ type: "socket", state: "open" });
      };
      socket.onmessage = (message) => {
        try {
          dispatch(JSON.parse(message.data as string) as WsEvent);
        } catch {
          /* ignore malformed frame */
        }
      };
      socket.onclose = () => {
        useAtlasStore.getState().applyEvent({ type: "socket", state: "closed" });
        if (stopped) return;
        attempt += 1;
        retry = window.setTimeout(connect, Math.min(500 * attempt, 5000));
      };
      socket.onerror = () => socket?.close();
    };

    connect();
    return () => {
      stopped = true;
      if (retry !== undefined) window.clearTimeout(retry);
      socket?.close();
    };
  }, [queryClient]);
}
