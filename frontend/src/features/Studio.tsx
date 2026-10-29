import { useEffect, useState } from "react";
import {
  AudioLines,
  ArrowUpRight,
  Check,
  LoaderCircle,
  Sparkles,
  Upload,
} from "lucide-react";
import { acceptsFile, api, upload } from "../api";
import { ActionHint, BusyIcon, ProgressLine, useDropTarget } from "../components/Feedback";
import { isActive } from "./jobs";
import { usePendingJob } from "./usePendingJob";
import { Slider } from "../components/Controls";
import { StudioAdvanced, defaultAdvanced } from "./StudioAdvanced";
import { StudioStyle, defaultTarget, type StyleTarget } from "./StudioStyle";
import { isStyled, starProfile, typicalSettings } from "./styleControls";
import { MapPreview } from "../components/MapPreview";
import type { Device, Hardware, State } from "../types";
import { GeneratedMaps } from "./GeneratedMaps";
import { generationHelp } from "./generationHelp";

const AUDIO_TYPES = [".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus"];
const MAX_UPLOAD = 512 * 1024 * 1024;
const presets = [
  { name: "Easy", ar: 5, od: 3, density: 1.5 },
  { name: "Normal", ar: 7, od: 5, density: 2.5 },
  { name: "Hard", ar: 9, od: 7, density: 4 },
  { name: "Insane", ar: 9.5, od: 9, density: 6 },
];

export function Studio({
  state,
  hardware,
  refresh,
  onError,
  goLibrary,
  onDeleted,
  shownOutput,
}: {
  state: State;
  hardware: Hardware | null;
  refresh: () => void;
  onError: (message: string) => void;
  goLibrary: () => void;
  onDeleted: (kind: "model" | "output", identifier: string) => void;
  /** A finished generation to open in the preview. */
  shownOutput?: string;
}) {
  const [model, setModel] = useState("");
  const [audio, setAudio] = useState<{ id: string; name: string } | null>(null);
  const [uploading, setUploading] = useState<number | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const pendingJob = usePendingJob(state.jobs);
  const starting = submitting || pendingJob.waiting;
  const [title, setTitle] = useState("");
  const [artist, setArtist] = useState("Unknown");
  const [version, setVersion] = useState("AI draft");
  const [device, setDevice] = useState<Device>(state.device);
  const [params, setParams] = useState({
    ar: 9,
    cs: 4,
    od: 7,
    hp: 5,
    density: 3,
    slider_bias: 0.5,
    spacing: 0.3,
    flow: 0.5,
  });
  const [advanced, setAdvanced] = useState(defaultAdvanced);
  const [target, setTarget] = useState<StyleTarget>(defaultTarget);
  const [previewId, setPreviewId] = useState("");
  const outputs = state.outputs.filter((output) => output.objects);
  const selectedModel =
    state.models.find((item) => item.id === model)?.id ||
    state.models[0]?.id ||
    "";
  const modelRecord = state.models.find((item) => item.id === selectedModel);
  const styled = isStyled(modelRecord);
  const calculatorReady = !!state.difficulty_ready;
  const typical =
    (target.useTarget ? starProfile(modelRecord, target.stars)?.style : undefined) || {};

  /** AR, OD, HP and CS typical for a star rating in this model's training data. */
  function applyStars(stars: number) {
    const profile = starProfile(modelRecord, stars)?.style;
    const fallback = typicalSettings(stars);
    const pick = (key: "ar" | "od" | "hp" | "cs") =>
      Math.round((profile?.[key] ?? fallback[key]) * 10) / 10;
    setParams((current) => ({
      ...current,
      ar: pick("ar"),
      od: pick("od"),
      hp: pick("hp"),
      cs: pick("cs"),
    }));
  }
  const deletionBlocked = state.deletion_blocked;
  const selectedOutput =
    outputs.find((output) => output.id === previewId) || outputs[0];
  const generation = state.jobs.find(
    (job) => job.kind === "generation" && isActive(job),
  );
  const duplicate = state.jobs.some(
    (job) =>
      job.kind === "generation" && isActive(job) && job.name === title.trim(),
  );
  const reason =
    uploading !== null
      ? "Warte, bis der Upload fertig ist."
      : !audio
        ? "Lade zuerst einen Track hoch."
        : !selectedModel
          ? "Trainiere oder importiere zuerst ein Modell."
          : !title.trim()
            ? "Gib dem Track einen Titel."
            : duplicate
              ? `Für „${title.trim()}“ läuft bereits eine Generierung.`
              : "";
  // The progress line under the button already says a generation is running.
  const hint = duplicate && generation ? "" : reason;

  useEffect(() => {
    if (shownOutput) setPreviewId(shownOutput);
  }, [shownOutput]);

  const drop = useDropTarget((file) => void selectAudio(file), uploading !== null);

  async function selectAudio(file?: File) {
    if (!file || uploading !== null) return;
    if (!acceptsFile(file, AUDIO_TYPES)) {
      onError(
        `„${file.name}“ ist kein unterstütztes Audioformat. Erlaubt sind MP3, WAV, OGG, FLAC, M4A, AAC und OPUS.`,
      );
      return;
    }
    if (file.size > MAX_UPLOAD) {
      onError(`„${file.name}“ ist größer als 512 MB.`);
      return;
    }
    setUploading(0);
    try {
      const value = await upload<{ id: string; name: string }>(
        "/audio",
        file,
        setUploading,
      );
      setAudio(value);
      setTitle(file.name.replace(/\.[^.]+$/, ""));
    } catch (error) {
      onError((error as Error).message);
    } finally {
      setUploading(null);
    }
  }

  async function generate() {
    setSubmitting(true);
    try {
      const job = await api("/generate", {
        model_id: selectedModel,
        audio_id: audio?.id,
        title,
        artist,
        version,
        device,
        ...params,
        ...advanced,
        bpm: advanced.bpm ? Number(advanced.bpm) : null,
        target_stars: target.useTarget ? target.stars : null,
        calibrate: target.useTarget && target.calibrate && calculatorReady,
        calibration_rounds: target.rounds,
        ...(styled
          ? {
              aim_share: target.aimShare,
              mapper: target.mapper || null,
              style: target.style,
            }
          : {}),
      });
      pendingJob.track(job);
      refresh();
    } catch (error) {
      onError((error as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <>
      <div className="page-heading">
        <div>
          <h1>
            From beat to <span className="accent">beatmap.</span>
          </h1>
          <p>
            Dein Sound. Dein Spielfeld. Verwandle Audio mit deinem eigenen
            Modell in osu!-Beatmaps.
          </p>
        </div>
        <span className="pill">Local-first studio</span>
      </div>
      {!state.models.length && (
        <div className="onboarding">
          <div className="step-number">01</div>
          <div>
            <b>Dein Mapping-Stil beginnt mit deiner Library.</b>
            <p>
              Importiere Maps, stelle Trainingsdaten zusammen und trainiere dein
              erstes Modell.
            </p>
          </div>
          <button className="text-button" onClick={goLibrary}>
            Library öffnen <ArrowUpRight size={17} />
          </button>
        </div>
      )}
      <div className="studio-grid">
        <div className="studio-settings">
          <section className="panel">
            <div className="panel-heading">
              <h2>
                <AudioLines size={18} /> Dein Track
              </h2>
              <span className="section-number">01</span>
            </div>
            <label
              className={`dropzone ${audio ? "loaded" : ""} ${drop.dragging ? "dragging" : ""} ${uploading !== null ? "busy" : ""}`}
              aria-busy={uploading !== null}
              {...drop.handlers}
            >
              <input
                type="file"
                accept={AUDIO_TYPES.join(",")}
                aria-label="Audiodatei auswählen"
                onChange={(event) => {
                  void selectAudio(event.target.files?.[0]);
                  event.target.value = "";
                }}
                disabled={uploading !== null}
              />
              <div className="upload-icon" aria-hidden="true">
                {uploading !== null ? (
                  <LoaderCircle size={22} className="spin" />
                ) : audio && !drop.dragging ? (
                  <Check size={22} />
                ) : (
                  <Upload size={22} />
                )}
              </div>
              <b>
                {uploading !== null
                  ? uploading >= 1
                    ? "Wird vorbereitet …"
                    : "Wird hochgeladen …"
                  : drop.dragging
                    ? "Loslassen zum Hochladen"
                    : audio?.name || "Audio hier ablegen"}
              </b>
              {uploading !== null ? (
                <ProgressLine value={uploading} label="Upload" />
              ) : (
                <>
                  <p>
                    {audio ? "Anderen Track ablegen oder " : "oder "}
                    <span>Datei auswählen</span>
                  </p>
                  <small>MP3, WAV, OGG, FLAC und mehr, bis 512 MB</small>
                </>
              )}
            </label>
            <div className="field-row">
              <label>
                Titel
                <input
                  value={title}
                  onChange={(event) => setTitle(event.target.value)}
                  placeholder="Songtitel"
                />
              </label>
              <label>
                Artist
                <input
                  value={artist}
                  onChange={(event) => setArtist(event.target.value)}
                />
              </label>
            </div>
            <label>
              Mapping-Modell
              <select
                value={selectedModel}
                onChange={(event) => setModel(event.target.value)}
              >
                <option value="" disabled>
                  Trainiere zuerst ein Modell
                </option>
                {state.models.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
            </label>
          </section>
          <section className="panel">
            <div className="panel-heading">
              <h2>
                <Sparkles size={17} /> Dein Mapping-Stil
              </h2>
              <span className="section-number">02</span>
            </div>
            {styled && (
              <StudioStyle
                model={modelRecord}
                value={target}
                onChange={setTarget}
                onStars={applyStars}
                calculatorReady={calculatorReady}
                typical={typical}
              />
            )}
            {!styled && (
            <div
              className="preset-row"
              role="group"
              aria-label="Schwierigkeits-Preset"
            >
              {presets.map((preset) => (
                <button
                  type="button"
                  key={preset.name}
                  aria-pressed={
                    params.ar === preset.ar &&
                    params.od === preset.od &&
                    params.density === preset.density
                  }
                  onClick={() =>
                    setParams({
                      ...params,
                      ar: preset.ar,
                      od: preset.od,
                      density: preset.density,
                    })
                  }
                >
                  {preset.name}
                </button>
              ))}
            </div>
            )}
            <div className="slider-grid">
              {(["ar", "cs", "od", "hp"] as const).map((key) => (
                <Slider
                  key={key}
                  help={generationHelp[key]}
                  label={
                    {
                      ar: "Approach rate",
                      cs: "Circle size",
                      od: "Overall difficulty",
                      hp: "HP drain",
                    }[key]
                  }
                  value={params[key]}
                  step={0.1}
                  onChange={(value) => setParams({ ...params, [key]: value })}
                />
              ))}
            </div>
            {!styled && (
            <>
            <Slider
              label="Objektdichte · pro Sekunde"
              help={generationHelp.density}
              value={params.density}
              min={0.5}
              onChange={(value) => setParams({ ...params, density: value })}
            />
            <div className="slider-grid">
              <Slider
                label="Bewegungsweite"
                help={generationHelp.spacing}
                value={params.spacing}
                min={0}
                max={1}
                step={0.05}
                onChange={(spacing) => setParams({ ...params, spacing })}
              />
              <Slider
                label="Richtungsfluss"
                help={generationHelp.flow}
                value={params.flow}
                min={0}
                max={1}
                step={0.05}
                onChange={(flow) => setParams({ ...params, flow })}
              />
            </div>
            <p className="fine-print">
              Weite und Richtungsfluss steuern den gelernten Stil: niedriger
              Fluss bevorzugt Umkehrungen, höherer Fluss fortlaufende
              Bewegungen. Keine feste Qualitätsbewertung.
            </p>
            <Slider
              label="Slider-Anteil · Stilvorgabe"
              help={generationHelp.sliders}
              value={params.slider_bias}
              min={0}
              max={1}
              step={0.05}
              onChange={(slider_bias) => setParams({ ...params, slider_bias })}
              hint="0: nur Circles. 1: Slider bevorzugen; sehr kurze Abstände können trotzdem Circles erfordern."
            />
            <StudioStyle
              model={modelRecord}
              value={target}
              onChange={setTarget}
              onStars={applyStars}
              calculatorReady={calculatorReady}
              legacy
            />
            <p className="fine-print">
              Dieses Modell wurde vor der Sterne-, Pattern- und Mapper-Steuerung
              trainiert. Trainiere ein neues Modell (gern mit diesem als
              Startpunkt), um Ziel-Sterne, Mapper-Stil und Patterns direkt zu
              steuern.
            </p>
            </>
            )}
            <StudioAdvanced
              styled={styled}
              value={advanced}
              onChange={setAdvanced}
              device={device}
              setDevice={setDevice}
              hardware={hardware}
              version={version}
              setVersion={setVersion}
            />
            <button
              type="button"
              className="primary generate-button"
              disabled={starting || !!reason}
              aria-busy={starting}
              aria-describedby="generate-hint"
              onClick={() => void generate()}
            >
              <BusyIcon busy={starting} icon={Sparkles} size={18} />
              {starting ? "Wird gestartet …" : "Beatmap generieren"}
              <ArrowUpRight size={18} aria-hidden="true" />
            </button>
            <ActionHint id="generate-hint">{hint}</ActionHint>
            {generation && (
              <ProgressLine
                value={generation.progress}
                label={`„${generation.name}“ wird generiert`}
              />
            )}
            <p className="fine-print">
              Erstellt einen Beatmap-Entwurf. Mit Kalibrierung wird die fertige
              Map mit der offiziellen osu!-Berechnung gemessen; bis ±0,15 ★ gilt
              als getroffen, sonst wird der beste Durchlauf behalten.
            </p>
          </section>
        </div>
        <div className="studio-preview">
          <MapPreview output={selectedOutput} onError={onError} />
          <div className="workflow-strip" aria-hidden="true">
            <span>
              <b>01</b> Audio analysieren
            </span>
            <i />
            <span>
              <b>02</b> Muster lernen
            </span>
            <i />
            <span>
              <b>03</b> Losspielen
            </span>
          </div>
          <GeneratedMaps
            outputs={outputs}
            selectedId={selectedOutput?.id}
            blocked={deletionBlocked}
            onSelect={setPreviewId}
            onDeleted={(identifier) => {
              if (identifier === selectedOutput?.id) setPreviewId("");
              onDeleted("output", identifier);
            }}
          />
          <div className="note">
            <span aria-hidden="true">✦</span>
            <p>
              <b>Dein Modell. Dein Stil.</b>
              <br />
              Alles bleibt lokal: deine Songs, deine Trainingsdaten und deine
              Modelle.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
