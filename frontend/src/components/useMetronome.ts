import { useEffect, useRef } from "react";
import type { SongPlan } from "../types";
import { beatsBetween } from "../features/songPlan";

const LOOKAHEAD = 0.2; // seconds of clicks scheduled ahead of the playhead

/**
 * Clicks on every beat of `plan` while `audio` plays (a higher click on bar starts), so
 * BPM and offset can be checked by ear before generating. Seeking reschedules.
 */
export function useMetronome(
  audio: HTMLAudioElement | null,
  plan: SongPlan | null,
  enabled: boolean,
  volume = 0.5,
) {
  const context = useRef<AudioContext | null>(null);
  const scheduled = useRef<{ until: number; sources: OscillatorNode[] }>({ until: 0, sources: [] });

  useEffect(() => {
    if (!audio || !plan || !enabled) return;
    let frame = 0;
    const stopAll = () => {
      for (const source of scheduled.current.sources) {
        try {
          source.stop();
        } catch {
          /* already stopped */
        }
      }
      scheduled.current = { until: 0, sources: [] };
    };
    const tick = () => {
      frame = requestAnimationFrame(tick);
      if (audio.paused) return;
      if (!context.current) context.current = new AudioContext();
      const ctx = context.current;
      if (ctx.state === "suspended") void ctx.resume();
      const now = audio.currentTime;
      const from = Math.max(now, scheduled.current.until);
      const to = now + LOOKAHEAD;
      if (to <= from) return;
      for (const beat of beatsBetween(plan, from * 1000, to * 1000)) {
        const when = ctx.currentTime + (beat.time / 1000 - now) / (audio.playbackRate || 1);
        if (when < ctx.currentTime) continue;
        const oscillator = ctx.createOscillator();
        const gain = ctx.createGain();
        oscillator.frequency.value = beat.downbeat ? 1760 : 1100;
        gain.gain.setValueAtTime(volume, when);
        gain.gain.exponentialRampToValueAtTime(0.0001, when + 0.04);
        oscillator.connect(gain).connect(ctx.destination);
        oscillator.start(when);
        oscillator.stop(when + 0.05);
        scheduled.current.sources.push(oscillator);
      }
      scheduled.current.until = to;
      if (scheduled.current.sources.length > 64)
        scheduled.current.sources = scheduled.current.sources.slice(-32);
    };
    const reset = () => stopAll();
    audio.addEventListener("seeking", reset);
    audio.addEventListener("pause", reset);
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
      audio.removeEventListener("seeking", reset);
      audio.removeEventListener("pause", reset);
      stopAll();
    };
  }, [audio, plan, enabled, volume]);

  useEffect(
    () => () => {
      void context.current?.close();
      context.current = null;
    },
    [],
  );
}
