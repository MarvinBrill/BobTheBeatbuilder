import type { PreviewSound } from "../types";

interface SkinManifest {
  name: string | null;
  samples: Record<string, string>;
  warning?: string | null;
}

/** Short look-ahead scheduling follows the music clock, including buffering/seeks. */
export class PreviewSoundPlayer {
  private context?: AudioContext;
  private gain?: GainNode;
  private buffers = new Map<string, AudioBuffer>();
  private sources = new Set<AudioBufferSourceNode>();
  private loading?: Promise<void>;
  private abort = new AbortController();
  private timer?: ReturnType<typeof setInterval>;
  private disposed = false;
  private volume = 0.5;
  private cursor = 0;
  private lastPosition = 0;
  private lastClock = 0;
  private listeners: [string, () => void][];

  constructor(
    private audio: HTMLAudioElement,
    private events: PreviewSound[],
    private onStatus: (message: string) => void,
  ) {
    this.listeners = [
      ["playing", this.start],
      ["pause", this.stop],
      ["ended", this.stop],
      ["error", this.stop],
      ["waiting", this.stop],
      ["seeking", this.stop],
      ["seeked", this.start],
      ["ratechange", this.start],
    ];
    this.listeners.forEach(([name, listener]) =>
      audio.addEventListener(name, listener),
    );
  }

  setVolume(volume: number) {
    this.volume = volume;
    if (this.gain && this.context) {
      this.gain.gain.setTargetAtTime(volume, this.context.currentTime, 0.01);
    }
  }

  async prepare() {
    if (this.disposed) return;
    // Called from the play-button gesture, before awaiting any network work.
    this.context ??= new AudioContext({ latencyHint: "interactive" });
    if (!this.gain) {
      this.gain = this.context.createGain();
      this.gain.gain.value = this.volume;
      this.gain.connect(this.context.destination);
    }
    await this.context.resume();
    this.loading ??= this.load();
    await this.loading;
  }

  private async load() {
    try {
      this.onStatus("Skin-Sounds werden geladen …");
      const response = await fetch("/api/preview-skin", {
        signal: this.abort.signal,
      });
      if (!response.ok) throw new Error("Skin konnte nicht geladen werden.");
      const manifest: SkinManifest = await response.json();
      let failed = 0;
      await Promise.all(
        Object.entries(manifest.samples).map(async ([name, url]) => {
          try {
            const sample = await fetch(url, { signal: this.abort.signal });
            if (!sample.ok) throw new Error("Sample fehlt");
            const buffer = await this.context!.decodeAudioData(
              await sample.arrayBuffer(),
            );
            if (!this.disposed) this.buffers.set(name, buffer);
          } catch {
            failed++;
          }
        }),
      );
      if (!this.disposed) {
        const warning = failed
          ? "Einzelne Sounds konnten nicht abgespielt werden."
          : manifest.warning;
        this.onStatus(
          [manifest.name, warning].filter(Boolean).join(" · ") ||
            "Keine Skin-Sounds verfügbar.",
        );
      }
    } catch (error) {
      if (!this.disposed)
        this.onStatus(
          `Skin-Sounds nicht verfügbar: ${(error as Error).message}`,
        );
    }
  }

  private start = () => {
    this.stop();
    if (
      this.disposed ||
      !this.context ||
      this.audio.paused ||
      this.audio.seeking ||
      this.audio.readyState < 3
    )
      return;
    const position = this.audio.currentTime;
    this.cursor = this.events.findIndex((event) => event.time >= position);
    if (this.cursor < 0) this.cursor = this.events.length;
    // Seeking into a slider resumes only its loop, never old hit/tick sounds.
    for (const event of this.events) {
      if (event.end && event.time < position && event.end > position)
        this.schedule(event, position);
    }
    this.lastPosition = position;
    this.lastClock = this.context.currentTime;
    this.pump();
    this.timer = setInterval(this.pump, 25);
  };

  private pump = () => {
    if (!this.context || this.audio.paused || this.audio.seeking) return;
    const position = this.audio.currentTime;
    const clock = this.context.currentTime;
    const expected =
      this.lastPosition + (clock - this.lastClock) * this.audio.playbackRate;
    if (Math.abs(expected - position) > 0.15) {
      this.start();
      return;
    }
    this.lastPosition = position;
    this.lastClock = clock;
    while (
      this.cursor < this.events.length &&
      this.events[this.cursor].time <= position + 0.12
    ) {
      const event = this.events[this.cursor++];
      // A throttled/background tab must not replay a backlog as a loud burst.
      if (
        event.time >= position - 0.04 ||
        (event.end && event.end > position)
      ) {
        this.schedule(event, position);
      }
    }
  };

  private schedule(event: PreviewSound, position: number) {
    const buffer = this.buffers.get(event.sample);
    if (!buffer || !this.context || !this.gain || buffer.duration <= 0) return;
    const source = this.context.createBufferSource();
    const level = this.context.createGain();
    source.buffer = buffer;
    source.loop = event.end !== undefined;
    source.playbackRate.value = this.audio.playbackRate;
    level.gain.value = event.volume;
    source.connect(level).connect(this.gain);
    this.sources.add(source);
    source.onended = () => {
      this.sources.delete(source);
      source.disconnect();
      level.disconnect();
    };
    const rate = this.audio.playbackRate;
    const when =
      this.context.currentTime + Math.max(0, event.time - position) / rate;
    const offset = source.loop
      ? Math.max(0, position - event.time) % buffer.duration
      : 0;
    source.start(when, offset);
    if (event.end !== undefined) {
      source.stop(
        this.context.currentTime + Math.max(0, event.end - position) / rate,
      );
    }
  }

  private stop = () => {
    clearInterval(this.timer);
    this.timer = undefined;
    for (const source of this.sources) source.stop();
    this.sources.clear();
  };

  dispose() {
    this.disposed = true;
    this.stop();
    this.abort.abort();
    this.listeners.forEach(([name, listener]) =>
      this.audio.removeEventListener(name, listener),
    );
    void this.context?.close();
  }
}
