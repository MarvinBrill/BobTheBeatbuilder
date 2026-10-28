import { useEffect, useRef, useState } from "react";
import { Crosshair, Download, Play, Pause } from "lucide-react";
import type { Output, Preview } from "../types";
import { api } from "../api";
import { PreviewObjects } from "./PreviewObjects";
import { PreviewSoundPlayer } from "./PreviewSoundPlayer";
import { PreviewVolume, usePreviewVolume } from "./PreviewVolume";

export function MapPreview({
  output,
  onError,
}: {
  output?: Output;
  onError: (message: string) => void;
}) {
  const [preview, setPreview] = useState<Preview | null>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [volume, setVolume] = usePreviewVolume("bob-preview-volume");
  const [soundVolume, setSoundVolume] = usePreviewVolume(
    "bob-preview-sound-volume",
  );
  const [soundStatus, setSoundStatus] = useState("");
  const [loadingSounds, setLoadingSounds] = useState(false);
  const audio = useRef<HTMLAudioElement>(null);
  const sounds = useRef<PreviewSoundPlayer | null>(null);

  useEffect(() => {
    if (audio.current) audio.current.volume = volume;
  }, [volume, preview, output?.id]);

  useEffect(() => {
    setSoundStatus("");
    setLoadingSounds(false);
    if (!preview || !audio.current) return;
    const player = new PreviewSoundPlayer(
      audio.current,
      preview.sound_events ?? [],
      setSoundStatus,
    );
    sounds.current = player;
    return () => {
      player.dispose();
      sounds.current = null;
    };
  }, [preview]);

  useEffect(() => {
    sounds.current?.setVolume(soundVolume);
  }, [soundVolume, preview]);

  async function togglePlayback() {
    const element = audio.current;
    const player = sounds.current;
    if (!element) return;
    if (!element.paused) {
      element.pause();
      return;
    }
    setLoadingSounds(true);
    try {
      try {
        await player?.prepare();
      } catch {
        if (sounds.current === player)
          setSoundStatus("Skin-Sounds sind in diesem Browser nicht verfügbar.");
      }
      if (audio.current === element && sounds.current === player)
        await element.play();
    } catch (error) {
      if (sounds.current === player) onError((error as Error).message);
    } finally {
      if (sounds.current === player) setLoadingSounds(false);
    }
  }

  useEffect(() => {
    let active = true;
    setPreview(null);
    setTime(0);
    setPlaying(false);
    if (output)
      api<Preview>(`/outputs/${output.id}/preview`)
        .then((result) => {
          if (active) setPreview(result);
        })
        .catch((error) => {
          if (active) onError(error.message);
        });
    return () => {
      active = false;
    };
  }, [output?.id, onError]);

  useEffect(() => {
    if (!playing) return;
    let frame: number;
    const update = () => {
      setTime(audio.current?.currentTime || 0);
      frame = requestAnimationFrame(update);
    };
    frame = requestAnimationFrame(update);
    return () => cancelAnimationFrame(frame);
  }, [playing]);

  return (
    <section className="panel preview-panel">
      <div className="panel-heading">
        <h2>
          <Crosshair size={17} /> Playfield
        </h2>
        <span className="tag">osu!standard</span>
      </div>
      <div className={`playfield ${preview ? "has-map" : ""}`}>
        {preview ? (
          <svg viewBox="0 0 512 384" aria-label="Beatmap-Vorschau">
            <PreviewObjects
              objects={preview.objects}
              time={time * 1000}
              cs={preview.cs ?? 4}
              ar={preview.ar ?? 9}
            />
          </svg>
        ) : (
          <div className="empty-playfield">
            <div className="orbit orbit-one" />
            <div className="orbit orbit-two" />
            <div className="beat-circle">
              b<span>♪</span>
            </div>
            <h3>Every song has a map.</h3>
            <p>Gib deinem Sound ein neues Spielfeld.</p>
            <span className="preview-label">
              Deine generierte Map erscheint hier
            </span>
          </div>
        )}
        <div className="playfield-caption">
          <span>
            {preview ? `${preview.objects.length} Objekte` : "512 × 384"}{" "}
          </span>
          <span>
            {preview
              ? `${preview.bpm.toFixed(1)} BPM`
              : "AUDIO → RHYTHM → PLAY"}
          </span>
        </div>
      </div>
      <div className="waveform" aria-hidden="true">
        {Array.from({ length: 90 }, (_, index) => {
          const value = preview
            ? preview.waveform[
                Math.floor((index * preview.waveform.length) / 90)
              ]
            : -0.6;
          return (
            <i
              key={index}
              style={{
                height: `${Math.max(5, (value + 1) * 24)}px`,
                opacity: preview ? 0.7 : 0.2,
              }}
            />
          );
        })}
      </div>
      {output && preview ? (
        <>
          <audio
            ref={audio}
            src={`/api/outputs/${output.id}/audio`}
            onEnded={() => setPlaying(false)}
            onPause={() => setPlaying(false)}
            onPlay={() => setPlaying(true)}
            onError={() =>
              onError("Audio-Vorschau konnte nicht abgespielt werden.")
            }
          />
          {preview.quality && (
            <p className="fine-print">
              Timing:{" "}
              {{
                manual: "manuell",
                learned: "gelernt",
                "audio-fallback": "Audio-Schätzung",
              }[preview.quality.timing_source] || preview.quality.timing_source}
              {" · "}
              {preview.quality.timing_sections} Tempoabschnitte ·{" "}
              {(preview.quality.heavily_repaired_fraction * 100).toFixed(1)} %
              stärker korrigierte Platzierungen. Die Vorschau ersetzt keinen
              Spieltest in osu!.
            </p>
          )}
          <div className="playback">
            <button
              className="icon-button"
              aria-label={playing ? "Pause" : "Abspielen"}
              disabled={loadingSounds}
              onClick={togglePlayback}
            >
              {playing ? <Pause size={17} /> : <Play size={17} />}
            </button>
            <input
              aria-label="Wiedergabeposition"
              type="range"
              min={0}
              max={preview.duration}
              step={0.01}
              value={time}
              onChange={(event) => {
                const value = Number(event.target.value);
                if (audio.current) audio.current.currentTime = value;
                setTime(value);
              }}
            />
            <span>
              {Math.floor(time / 60)}:
              {String(Math.floor(time % 60)).padStart(2, "0")}
            </span>
            <a
              className="icon-button"
              href={`/api/outputs/${output.id}/download`}
              aria-label="Beatmap herunterladen"
            >
              <Download size={17} />
            </a>
          </div>
          <PreviewVolume label="Musik" value={volume} onChange={setVolume} />
          <PreviewVolume
            label="Hit- & Slider-Sounds"
            value={soundVolume}
            onChange={setSoundVolume}
          />
          {soundStatus && (
            <p className="fine-print preview-sound-status" role="status">
              {soundStatus}
            </p>
          )}
        </>
      ) : (
        <div className="preview-footnote">
          Live-Vorschau mit Audio nach der Generierung
        </div>
      )}
    </section>
  );
}
