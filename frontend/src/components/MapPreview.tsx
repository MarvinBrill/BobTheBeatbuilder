import { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  Crosshair,
  Download,
  LoaderCircle,
  Play,
  Pause,
  RefreshCw,
  Volume2,
  VolumeX,
} from "lucide-react";
import type { Output, Preview } from "../types";
import { api } from "../api";
import { PreviewObjects } from "./PreviewObjects";

export function MapPreview({
  output,
  onError,
}: {
  output?: Output;
  onError: (message: string) => void;
}) {
  const [preview, setPreview] = useState<Preview | null>(null);
  const [loading, setLoading] = useState(false);
  const [failure, setFailure] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [starting, setStarting] = useState(false);
  const [audioFailed, setAudioFailed] = useState(false);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [volume, setVolume] = useState(() => {
    try {
      const saved = localStorage.getItem("bob-preview-volume");
      const value = saved === null ? 0.5 : Number(saved);
      return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0.5;
    } catch {
      return 0.5;
    }
  });
  const audio = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    if (audio.current) audio.current.volume = volume;
    try {
      localStorage.setItem("bob-preview-volume", String(volume));
    } catch {
      /* Storage may be disabled. */
    }
  }, [volume, preview, output?.id]);

  useEffect(() => {
    let active = true;
    setPreview(null);
    setTime(0);
    setPlaying(false);
    setFailure("");
    setAudioFailed(false);
    setLoading(!!output);
    if (output)
      api<Preview>(`/outputs/${output.id}/preview`)
        .then((result) => {
          if (active) setPreview(result);
        })
        .catch((error) => {
          if (active) setFailure(error.message);
        })
        .finally(() => {
          if (active) setLoading(false);
        });
    return () => {
      active = false;
    };
  }, [output?.id, attempt]);

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
        {loading ? (
          <div className="empty-playfield" role="status">
            <LoaderCircle size={34} className="spin" aria-hidden="true" />
            <p>Vorschau für „{output?.name}“ wird geladen …</p>
          </div>
        ) : failure ? (
          <div className="empty-playfield" role="alert">
            <AlertCircle size={34} aria-hidden="true" />
            <h3>Vorschau nicht verfügbar</h3>
            <p>{failure}</p>
            <button
              type="button"
              className="secondary"
              onClick={() => setAttempt((value) => value + 1)}
            >
              <RefreshCw size={15} aria-hidden="true" /> Erneut laden
            </button>
          </div>
        ) : preview ? (
          <svg viewBox="0 0 512 384" role="img" aria-label="Beatmap-Vorschau">
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
            <div className="beat-circle" aria-hidden="true">
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
              : "Audio, Rhythmus, Spiel"}
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
            onError={() => {
              setAudioFailed(true);
              setPlaying(false);
            }}
          />
          {preview.quality?.difficulty && (
            <p className="fine-print">
              <b className="result-stars">
                {preview.quality.difficulty.star_rating.toFixed(2)} ★
              </b>{" "}
              nach offizieller osu!-Berechnung · Aim{" "}
              {preview.quality.difficulty.aim_difficulty?.toFixed(2) ?? "–"} · Speed{" "}
              {preview.quality.difficulty.speed_difficulty?.toFixed(2) ?? "–"}
              {preview.quality.difficulty.reading_difficulty !== undefined &&
                ` · Reading ${preview.quality.difficulty.reading_difficulty.toFixed(2)}`}
              {preview.quality.calibration && preview.quality.calibration.length > 1 &&
                ` · ${preview.quality.calibration.length} Durchläufe (${preview.quality.calibration
                  .map((round) => (round.stars == null ? "–" : round.stars.toFixed(2)))
                  .join(" → ")})`}
            </p>
          )}
          {preview.quality?.patterns && (
            <p className="fine-print">
              Patterns:{" "}
              {[
                preview.quality.patterns.burst_share !== undefined &&
                  `${Math.round(preview.quality.patterns.burst_share * 100)} % Bursts`,
                preview.quality.patterns.stream_share !== undefined &&
                  `${Math.round(preview.quality.patterns.stream_share * 100)} % Streams`,
                preview.quality.patterns.jump_spacing !== undefined &&
                  `1/2-Sprünge ~${Math.round(preview.quality.patterns.jump_spacing)} px`,
                preview.quality.patterns.slider_ratio !== undefined &&
                  `${Math.round(preview.quality.patterns.slider_ratio * 100)} % Slider`,
                preview.quality.patterns.density !== undefined &&
                  `${preview.quality.patterns.density.toFixed(1)} Objekte/s`,
              ]
                .filter(Boolean)
                .join(" · ")}
            </p>
          )}
          {preview.quality && (
            <p className="fine-print">
              Timing:{" "}
              {{
                manual: "manuell",
                learned: "gelernt",
                reference: "vorgegeben",
                "audio-fallback": "Audio-Schätzung",
              }[preview.quality.timing_source] || preview.quality.timing_source}
              {", "}
              {preview.quality.timing_sections} Tempoabschnitte,{" "}
              {(preview.quality.heavily_repaired_fraction * 100).toFixed(1)} %
              stärker korrigierte Platzierungen. Die Vorschau ersetzt keinen
              Spieltest in osu!.
            </p>
          )}
          <div className="playback">
            <button
              type="button"
              className="icon-button"
              aria-label={playing ? "Pause" : "Abspielen"}
              aria-busy={starting}
              aria-describedby={audioFailed ? "preview-audio-hint" : undefined}
              title={audioFailed ? "Audio nicht verfügbar" : undefined}
              disabled={starting || audioFailed}
              onClick={() => {
                if (playing) return audio.current?.pause();
                setStarting(true);
                audio.current
                  ?.play()
                  .catch((error) => onError(error.message))
                  .finally(() => setStarting(false));
              }}
            >
              {starting ? (
                <LoaderCircle size={17} className="spin" />
              ) : playing ? (
                <Pause size={17} />
              ) : (
                <Play size={17} />
              )}
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
          {audioFailed && (
            <p className="preview-audio-hint" id="preview-audio-hint">
              Audio für die Vorschau fehlt. Die Objekte lassen sich mit dem
              Regler trotzdem durchgehen.
            </p>
          )}
          <label className="preview-volume">
            {volume === 0 ? <VolumeX size={18} /> : <Volume2 size={18} />}
            <span>Lautstärke</span>
            <input
              type="range"
              aria-label="Vorschau-Lautstärke"
              min="0"
              max="1"
              step="0.01"
              value={volume}
              onChange={(event) => setVolume(Number(event.target.value))}
            />
            <output>{Math.round(volume * 100)} %</output>
          </label>
        </>
      ) : (
        <div className="preview-footnote">
          {loading || failure
            ? "\u00a0"
            : "Live-Vorschau mit Audio nach der Generierung"}
        </div>
      )}
    </section>
  );
}
