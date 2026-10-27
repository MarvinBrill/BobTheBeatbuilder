import { useCallback, useEffect, useRef } from "react";
import type { State } from "../types";

/** One request at a time. Unchanged state never triggers a React render. */
export function useStatePolling(
  onState: (state: State) => void,
  onError: (message: string) => void,
) {
  const request = useRef<() => void>(() => {});
  const refresh = useCallback(() => request.current(), []);

  useEffect(() => {
    let stopped = false;
    let busy = false;
    let pending = false;
    let active = false;
    let tag = "";
    let timer: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();

    async function poll() {
      if (stopped) return;
      if (busy) {
        pending = true;
        return;
      }
      clearTimeout(timer);
      busy = true;
      try {
        const response = await fetch("/api/state", {
          cache: "no-store",
          headers: tag ? { "If-None-Match": tag } : {},
          signal: controller.signal,
        });
        if (response.status !== 304) {
          if (!response.ok)
            throw new Error("Status konnte nicht geladen werden.");
          const state = (await response.json()) as State;
          if (!stopped && !pending) {
            active =
              state.deletion_blocked ||
              state.jobs.some((job) =>
                ["queued", "running"].includes(job.status),
              );
            onState(state);
          }
        }
        if (!pending) tag = response.headers.get("ETag") || "";
      } catch (error) {
        if (!stopped) onError((error as Error).message);
      } finally {
        busy = false;
        if (!stopped) {
          if (pending) {
            pending = false;
            void poll();
          } else if (!document.hidden)
            timer = setTimeout(poll, active ? 2500 : 30000);
        }
      }
    }
    function resume() {
      clearTimeout(timer);
      if (!document.hidden) void poll();
    }
    request.current = () => {
      void poll();
    };
    document.addEventListener("visibilitychange", resume);
    window.addEventListener("focus", resume);
    void poll();
    return () => {
      stopped = true;
      controller.abort();
      clearTimeout(timer);
      request.current = () => {};
      document.removeEventListener("visibilitychange", resume);
      window.removeEventListener("focus", resume);
    };
  }, [onState, onError]);
  return refresh;
}
