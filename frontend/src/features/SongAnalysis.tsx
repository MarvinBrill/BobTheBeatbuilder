import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Combine,
  Gauge,
  Pause,
  Play,
  ScanSearch,
  Split,
  Timer,
} from "lucide-react";
import { api } from "../api";
import { ActionHint, BusyIcon, ProgressLine } from "../components/Feedback";
import { useMetronome } from "../components/useMetronome";
import { isActive } from "./jobs";
import { usePendingJob } from "./usePendingJob";
import {
  bpmOf,
  clock,
  defaultStyle,
  kindLabels,
  mergeWithNext,
  moveBoundary,
  setTempo,
  shiftTiming,
  splitSection,
  styleLabels,
  updateSection,
} from "./songPlan";
import type { Device, PlanStyle, SectionKind, SongPlan, State } from "../types";

const KIND_COLORS: Record<SectionKind, string> = {
  intro: "var(--plan-calm)",
  verse: "var(--plan-flow)",
  build: "var(--plan-build)",
  drop: "var(--plan-drop)",
  break: "var(--plan-calm)",
  outro: "var(--plan-calm)",
  silence: "var(--plan-silence)",
};

/**
 * Step 02: analyse the song before generating. Timing (BPM, offset, red lines) can be
 * checked against a metronome and corrected; sections get a kind and a pattern style.
 * Generation then uses exactly this plan.
 */
export function SongAnalysis({
  state,
  audio,
  modelId,
  device,
  params,
  targetStars,
  plan,
  onPlan,
  usePlan,
  onUsePlan,
  onError,
  refresh,
}: {
  state: State;
  audio: { id: string; name: string } | null;
  modelId: string;
  device: Device;
  params: { ar: number; cs: number; od: number; hp: number };
  targetStars: number | null;
  plan: SongPlan | null;
  onPlan: (plan: SongPlan | null) => void;
  usePlan: boolean;
  onUsePlan: (value: boolean) => void;
  onError: (message: string) => void;
  refresh: () => void;
}) {
  const pending = usePendingJob(state.jobs);
  const [jobId, setJobId] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [metronome, setMetronome] = useState(true);
  const [element, setElement] = useState<HTMLAudioElement | null>(null);
  const [time, setTime] = useState(0);
  const loaded = useRef("");
  const job = state.jobs.find((item) => item.id === jobId);
  const running = isActive(job) || submitting || pending.waiting;

  useMetronome(element, plan, metronome && playing);

  useEffect(() => {
    const identifier = job?.status === "completed" ? job.result?.analysis_id : undefined;
    if (!identifier || loaded.current === identifier) return;
    loaded.current = identifier;
    api<{ plan: SongPlan }>(`/analyses/${identifier}`)
      .then((record) => {
        onPlan(record.plan);
        onUsePlan(true);
      })
      .catch((error) => onError((error as Error).message));
  }, [job?.status, job?.result?.analysis_id, onPlan, onUsePlan, onError]);

  useEffect(() => {
    if (!playing || !element) return;
    let frame = 0;
    const update = () => {
      setTime(element.currentTime * 1000);
      frame = requestAnimationFrame(update);
    };
    frame = requestAnimationFrame(update);
    return () => cancelAnimationFrame(frame);
  }, [playing, element]);

  async function analyze() {
    if (!audio) return;
    setSubmitting(true);
    try {
      const created = await api<{ id: string }>("/analyze", {
        audio_id: audio.id,
        model_id: modelId || null,
        device,
        ...params,
        target_stars: targetStars,
      });
      setJobId(created.id);
      pending.track(created);
      refresh();
    } catch (error) {
      onError((error as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  const reason = !audio ? "Lade zuerst einen Track hoch." : "";
  const first = plan?.timing[0];
  const info = plan?.timing_info;
  const confidence = info?.confidence;
  const duration = plan?.duration_ms || 1;
  const current = plan?.sections.findIndex(
    (section) => section.start_ms <= time && time < section.end_ms,
  );

  return (
    <section className="panel song-analysis">
      <div className="panel-heading">
        <h2>
          <ScanSearch size={17} /> Song-Analyse
        </h2>
        <span className="section-number">02</span>
      </div>
      <p className="fine-print">
        Erst Timing und Struktur, dann die Objekte: Prüfe BPM und Offset mit dem
        Metronom und lege fest, wie jeder Abschnitt gespielt werden soll.
      </p>
      <button
        type="button"
        className="secondary"
        disabled={running || !!reason}
        aria-busy={running}
        onClick={() => void analyze()}
      >
        <BusyIcon busy={running} icon={ScanSearch} />
        {plan ? "Neu analysieren" : "Song analysieren"}
      </button>
      <ActionHint>{reason}</ActionHint>
      {job && isActive(job) && <ProgressLine value={job.progress} label="Analyse" />}
      {job?.status === "failed" && <ActionHint tone="error">{job.message}</ActionHint>}
      {plan && first && (
        <>
          <div className="timing-review">
            <div>
              <b>{bpmOf(first).toFixed(2)} BPM</b>
              <small>
                Offset {Math.round(first.time)} ms · {plan.timing.length}{" "}
                {plan.timing.length === 1 ? "Red Line" : "Red Lines"}
                {first.meter ? ` · ${first.meter}/4, Takt erkannt` : " · Takt geschätzt"}
              </small>
            </div>
            {confidence !== undefined && (
              <span className={`confidence ${confidence < 0.5 ? "low" : ""}`}>
                <Gauge size={14} aria-hidden="true" /> Sicherheit {Math.round(confidence * 100)} %
              </span>
            )}
          </div>
          {confidence !== undefined && confidence < 0.5 && (
            <ActionHint tone="error">
              Das Timing ist unsicher. Bitte mit dem Metronom prüfen und bei Bedarf BPM oder
              Offset anpassen.
            </ActionHint>
          )}
          <div className="timing-controls" role="group" aria-label="Timing anpassen">
            <span>Offset</span>
            {[-10, -1, 1, 10].map((step) => (
              <button
                key={step}
                type="button"
                className="chip"
                onClick={() => onPlan(shiftTiming(plan, step))}
              >
                {step > 0 ? `+${step}` : step} ms
              </button>
            ))}
            <span>Tempo</span>
            <button type="button" className="chip" onClick={() => onPlan(setTempo(plan, 0, bpmOf(first) / 2))}>
              ½
            </button>
            <button type="button" className="chip" onClick={() => onPlan(setTempo(plan, 0, bpmOf(first) * 2))}>
              ×2
            </button>
            <label className="inline-number">
              BPM
              <input
                type="number"
                min={20}
                max={400}
                step={0.01}
                value={Number(bpmOf(first).toFixed(3))}
                onChange={(event) => onPlan(setTempo(plan, 0, Number(event.target.value)))}
              />
            </label>
          </div>
          {!!info?.alternatives?.length && (
            <p className="fine-print">
              Auch möglich:{" "}
              {info.alternatives.map((option) => (
                <button
                  key={option.bpm}
                  type="button"
                  className="link-button"
                  onClick={() => onPlan(setTempo(plan, 0, option.bpm))}
                >
                  {option.bpm} BPM
                </button>
              ))}
            </p>
          )}
          {audio && (
            <div className="playback">
              <button
                type="button"
                className="icon-button"
                aria-label={playing ? "Pause" : "Mit Metronom abspielen"}
                onClick={() => {
                  if (!element) return;
                  if (element.paused) void element.play();
                  else element.pause();
                }}
              >
                {playing ? <Pause size={17} /> : <Play size={17} />}
              </button>
              <input
                aria-label="Wiedergabeposition"
                type="range"
                min={0}
                max={duration / 1000}
                step={0.01}
                value={time / 1000}
                onChange={(event) => {
                  if (element) element.currentTime = Number(event.target.value);
                  setTime(Number(event.target.value) * 1000);
                }}
              />
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={metronome}
                  onChange={(event) => setMetronome(event.target.checked)}
                />{" "}
                <Timer size={15} aria-hidden="true" /> Metronom
              </label>
              <audio
                ref={setElement}
                src={`/api/audio/${audio.id}`}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onEnded={() => setPlaying(false)}
              />
            </div>
          )}
          <div className="plan-timeline" aria-label="Abschnitte">
            {plan.sections.map((section, index) => (
              <button
                key={`${section.start_ms}-${index}`}
                type="button"
                title={`${kindLabels[section.kind]} · ${clock(section.start_ms)}`}
                className={index === current ? "current" : ""}
                style={{
                  width: `${((section.end_ms - section.start_ms) / duration) * 100}%`,
                  background: KIND_COLORS[section.kind],
                  opacity: 0.45 + 0.55 * section.intensity,
                }}
                onClick={() => {
                  if (element) element.currentTime = section.start_ms / 1000;
                  setTime(section.start_ms);
                }}
              />
            ))}
            <i style={{ left: `${(time / duration) * 100}%` }} aria-hidden="true" />
          </div>
          <ol className="plan-sections">
            {plan.sections.map((section, index) => (
              <li key={`${section.start_ms}-${index}`} className={index === current ? "current" : ""}>
                <span className="plan-time">
                  {clock(section.start_ms)}
                  <small>{section.bars ?? "?"} Takte</small>
                </span>
                <select
                  aria-label="Abschnitt"
                  value={section.kind}
                  onChange={(event) =>
                    onPlan(updateSection(plan, index, { kind: event.target.value as SectionKind }))
                  }
                >
                  {Object.entries(kindLabels).map(([kind, label]) => (
                    <option key={kind} value={kind}>
                      {label}
                    </option>
                  ))}
                </select>
                <select
                  aria-label="Pattern-Stil"
                  value={section.style}
                  onChange={(event) =>
                    onPlan(updateSection(plan, index, { style: event.target.value as PlanStyle }))
                  }
                >
                  {Object.entries(styleLabels).map(([style, label]) => (
                    <option key={style} value={style}>
                      {style === "auto"
                        ? `${label} (${styleLabels[defaultStyle[section.kind]]})`
                        : label}
                    </option>
                  ))}
                </select>
                <span className="intensity" title="Intensität">
                  <i style={{ width: `${Math.round(section.intensity * 100)}%` }} />
                </span>
                {section.repeat_of != null && (
                  <small className="repeat">wie {clock(plan.sections[section.repeat_of]?.start_ms ?? 0)}</small>
                )}
                <span className="plan-actions">
                  <button
                    type="button"
                    className="icon-button"
                    aria-label="Anfang einen Takt früher"
                    disabled={index === 0}
                    onClick={() => onPlan(moveBoundary(plan, index, -1))}
                  >
                    <ArrowLeft size={14} />
                  </button>
                  <button
                    type="button"
                    className="icon-button"
                    aria-label="Anfang einen Takt später"
                    disabled={index === 0}
                    onClick={() => onPlan(moveBoundary(plan, index, 1))}
                  >
                    <ArrowRight size={14} />
                  </button>
                  <button
                    type="button"
                    className="icon-button"
                    aria-label="Abschnitt teilen"
                    onClick={() => onPlan(splitSection(plan, index))}
                  >
                    <Split size={14} />
                  </button>
                  <button
                    type="button"
                    className="icon-button"
                    aria-label="Mit dem nächsten Abschnitt zusammenlegen"
                    disabled={index + 1 >= plan.sections.length}
                    onClick={() => onPlan(mergeWithNext(plan, index))}
                  >
                    <Combine size={14} />
                  </button>
                </span>
              </li>
            ))}
          </ol>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={usePlan}
              onChange={(event) => onUsePlan(event.target.checked)}
            />{" "}
            Diesen Plan für die Generierung verwenden
          </label>
        </>
      )}
    </section>
  );
}
