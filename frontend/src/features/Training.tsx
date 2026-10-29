import { TrainingPerformance } from "./TrainingPerformance";
import { TrainingQuality } from "./TrainingQuality";
import { LossChart } from "../components/LossChart";
import { useState } from "react";
import { ArrowUpRight, BrainCircuit, Cpu, Play, Layers } from "lucide-react";
import { api } from "../api";
import { ActionHint, BusyIcon, ProgressLine } from "../components/Feedback";
import { activeJob, jobMessage, isActive, scrollToActivity, statusLabel } from "./jobs";
import { usePendingJob } from "./usePendingJob";
import { DeviceSelect } from "../components/Controls";
import type { Device, Hardware, State } from "../types";

export function Training({
  state,
  hardware,
  refresh,
  onError,
  goLibrary,
}: {
  state: State;
  hardware: Hardware | null;
  refresh: () => void;
  onError: (message: string) => void;
  goLibrary?: () => void;
}) {
  const [datasetId, setDatasetId] = useState("");
  const [name, setName] = useState("My mapping style");
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(state.device === "cuda" ? 8 : 2);
  const [learningRate, setLearningRate] = useState(0.0003);
  const [modelSize, setModelSize] = useState("standard");
  const [accumulation, setAccumulation] = useState(
    state.device === "cuda" ? 1 : 4,
  );
  const [precision, setPrecision] = useState("auto");
  const [cudaGraphs, setCudaGraphs] = useState(true);
  const [contextRobustness, setContextRobustness] = useState(false);
  const [rolloutValidation, setRolloutValidation] = useState(true);
  const [resumeId, setResumeId] = useState("");
  const [seed, setSeed] = useState(42);
  const [device, setDevice] = useState<Device>(state.device);
  const [submitting, setSubmitting] = useState(false);
  const pendingJob = usePendingJob(state.jobs);
  const starting = submitting || pendingJob.waiting;
  const selected = state.datasets.find(
    (dataset) => dataset.id === (datasetId || state.datasets[0]?.id),
  );
  const latest = state.jobs.find((job) => job.kind === "training");
  const running = activeJob(state.jobs, "training");
  const invalid = {
    epochs: !Number.isInteger(epochs) || epochs < 1 || epochs > 10000,
    batchSize: !Number.isInteger(batchSize) || batchSize < 1 || batchSize > 64,
    accumulation:
      !Number.isInteger(accumulation) || accumulation < 1 || accumulation > 32,
    learningRate:
      !Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 0.01,
    seed: !Number.isInteger(seed) || seed < 0,
  };
  const reason = running
    ? `„${running.name}“ trainiert gerade. Ein weiteres Training startet erst danach; abbrechen kannst du unter Aktivität.`
    : !selected
      ? "Erstelle zuerst Trainingsdaten in der Library."
      : !name.trim()
        ? "Gib dem Modell einen Namen."
        : invalid.epochs
          ? "Epochen: ganze Zahl von 1 bis 10000."
          : invalid.batchSize
            ? "Batch-Größe: ganze Zahl von 1 bis 64."
            : invalid.accumulation
              ? "Updates sammeln: ganze Zahl von 1 bis 32."
              : invalid.learningRate
                ? "Lernrate: größer als 0 und höchstens 0,01."
                : invalid.seed
                  ? "Seed: ganze Zahl ab 0."
                  : "";

  async function start() {
    setSubmitting(true);
    try {
      const job = await api("/train", {
        name,
        dataset_id: selected?.id,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        seed,
        device,
        model_size: modelSize,
        gradient_accumulation: accumulation,
        resume_model_id: resumeId || null,
        precision: device === "cuda" ? precision : "auto",
        cuda_graphs: cudaGraphs,
        context_robustness: contextRobustness,
        rollout_validation: rolloutValidation,
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
            Train your <span className="accent">style.</span>
          </h1>
          <p>
            Lernt von deinen Lieblingsmaps. Ein Modell für Rhythmus, Positionen
            und Slider, trainiert auf deinem Rechner.
          </p>
        </div>
        <span className="pill">
          <Cpu size={14} aria-hidden="true" />{" "}
          {hardware?.name || "Hardware wird geprüft …"}
        </span>
      </div>
      <div className="training-grid">
        <section className="panel">
          <div className="panel-heading">
            <h2>
              <BrainCircuit size={18} /> Neues Training
            </h2>
          </div>
          <label>
            Modellname
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          <label>
            Trainingsdaten
            <select
              value={selected?.id || ""}
              onChange={(event) => {
                setDatasetId(event.target.value);
                setResumeId("");
              }}
            >
              <option value="" disabled>
                Zuerst in der Library erstellen
              </option>
              {state.datasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name} ({dataset.count} Maps)
                </option>
              ))}
            </select>
          </label>
          <div className="dataset-summary">
            <Layers size={21} />
            <div>
              <b>
                {selected
                  ? `${selected.count} Maps · ${selected.songs} Songs`
                  : "Noch keine Trainingsdaten"}
              </b>
              <small>
                {selected && selected.songs > 1
                  ? "Prüfsongs werden nach Audio und Künstler/Titel getrennt."
                  : "Mindestens 2 verschiedene Songs für unabhängige Validierung."}
              </small>
              {!state.datasets.length && goLibrary && (
                <button
                  type="button"
                  className="text-button"
                  onClick={goLibrary}
                >
                  Maps in der Library auswählen <ArrowUpRight size={15} />
                </button>
              )}
            </div>
          </div>
          <DeviceSelect
            value={device}
            onChange={setDevice}
            hardware={hardware}
          />
          {device === "cuda" && (
            <div className="cuda-settings">
              <label>
                CUDA-Rechengenauigkeit
                <select
                  value={precision}
                  onChange={(event) => setPrecision(event.target.value)}
                >
                  <option value="auto">
                    Automatisch (BF16, wenn unterstützt)
                  </option>
                  <option value="float32">Float32 (Vergleichsmodus)</option>
                </select>
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={cudaGraphs}
                  onChange={(event) => setCudaGraphs(event.target.checked)}
                />{" "}
                CUDA Graphs verwenden
              </label>
              <small>
                CUDA Graphs greifen bei „Updates sammeln: 1“. Für den Einstieg
                auf der RTX 5070 Ti: Standard, Batch 8, Updates sammeln 1.
                Gewichte und Lernziele bleiben in Float32.
              </small>
            </div>
          )}
          <div className="field-row">
            <label>
              Modellgröße
              <select
                value={modelSize}
                disabled={!!resumeId}
                onChange={(event) => setModelSize(event.target.value)}
              >
                <option value="compact">Kompakt (weniger Speicher)</option>
                <option value="standard">Standard (empfohlen)</option>
                <option value="large">Groß (mehr Kapazität)</option>
              </select>
              {resumeId && (
                <small>Beim Fortsetzen bleibt die Größe des Trainingsstands.</small>
              )}
            </label>
            <label>
              Updates sammeln
              <input
                type="number"
                min={1}
                max={32}
                value={accumulation}
                aria-invalid={invalid.accumulation}
                onChange={(event) =>
                  setAccumulation(Number(event.target.value))
                }
              />
              <small>Effektive Batch-Größe: {batchSize * accumulation}</small>
            </label>
          </div>
          <label>
            Trainingsstand
            <select
              value={resumeId}
              onChange={(event) => setResumeId(event.target.value)}
            >
              <option value="">Neues Modell trainieren</option>
              {state.models
                .filter(
                  (model) =>
                    model.continuable && model.dataset_id === selected?.id,
                )
                .map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name} fortsetzen
                  </option>
                ))}
            </select>
          </label>
          <div className="field-row">
            <label>
              Epochen
              <input
                type="number"
                min={1}
                max={10000}
                value={epochs}
                aria-invalid={invalid.epochs}
                onChange={(event) => setEpochs(Number(event.target.value))}
              />
            </label>
            <label>
              Batch-Größe
              <input
                type="number"
                min={1}
                max={64}
                value={batchSize}
                aria-invalid={invalid.batchSize}
                onChange={(event) => setBatchSize(Number(event.target.value))}
              />
            </label>
          </div>
          <div className="field-row">
            <label>
              Lernrate
              <input
                type="number"
                min={0.000001}
                max={0.01}
                step={0.0001}
                value={learningRate}
                aria-invalid={invalid.learningRate}
                onChange={(event) =>
                  setLearningRate(Number(event.target.value))
                }
              />
            </label>
            <label>
              Seed
              <input
                type="number"
                min={0}
                value={seed}
                aria-invalid={invalid.seed}
                onChange={(event) => setSeed(Number(event.target.value))}
              />
            </label>
          </div>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={contextRobustness}
              onChange={(event) => setContextRobustness(event.target.checked)}
            />
            Eigene Positionsvorschläge trainieren (experimentell)
          </label>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={rolloutValidation}
              onChange={(event) => setRolloutValidation(event.target.checked)}
            />
            Freie Generierung während des Trainings prüfen
          </label>
          <p className="fine-print">
            Die Lernrate sinkt bei Stillstand automatisch. Das optionale
            Zusatztraining startet ab Epoche 5 und benötigt mehr Rechenzeit. Im
            bisherigen Vergleich war der Qualitätsgewinn nicht eindeutig;
            deshalb bleibt es zunächst aus. Freie Prüfungen nutzen bis zu drei
            feste Validierungssongs in Epoche 1, danach alle fünf Epochen und
            zum Trainingsende.
          </p>
          <button
            type="button"
            className="primary full-width"
            disabled={starting || !!reason}
            aria-busy={starting}
            aria-describedby="training-hint"
            onClick={() => void start()}
          >
            <BusyIcon busy={starting} icon={Play} />
            {starting ? "Wird gestartet …" : "Training starten"}
          </button>
          <ActionHint id="training-hint">
            {reason}
            {running && (
              <>
                {" "}
                <button
                  type="button"
                  className="text-button inline"
                  onClick={scrollToActivity}
                >
                  Zur Aktivität
                </button>
              </>
            )}
          </ActionHint>
          <p className="fine-print">
            Gespeichert wird der Stand mit dem niedrigsten Prüfverlust (ohne
            Prüfsongs: Trainingsverlust). Der letzte Trainingsstand bleibt
            zusätzlich zum Fortsetzen erhalten, auch bei einem Abbruch nach dem
            ersten Update.
          </p>
        </section>
        <div>
          <section className="panel">
            <div className="panel-heading">
              <h2>Lernkurve</h2>
              <span className={`tag status-${latest?.status || "none"}`}>
                {latest ? statusLabel(latest.status) : "Noch kein Training"}
                {isActive(latest) &&
                  ` ${Math.round((latest?.progress || 0) * 100)} %`}
              </span>
            </div>
            {running && (
              <ProgressLine
                value={running.progress}
                label={jobMessage(running) || "Training läuft"}
              />
            )}
            <LossChart metrics={latest?.metrics || []} />
            {latest?.warnings?.map((warning) => (
              <p className="fine-print" key={warning}>
                {warning}
              </p>
            ))}
            <div className="metric-row">
              <div>
                <small>Epoche</small>
                <b>
                  {latest?.metrics.length || "—"}
                  <span>
                    {" "}
                    / {latest?.metrics.length ? "abgeschlossen" : "bereit"}
                  </span>
                </b>
              </div>
              <div>
                <small>Gesamtverlust Training</small>
                <b>{latest?.metrics.at(-1)?.train_loss.toFixed(4) || "—"}</b>
              </div>
              <div>
                <small>Gesamtverlust Validierung</small>
                <b>
                  {latest?.metrics.at(-1)?.validation_loss?.toFixed(4) || "—"}
                </b>
              </div>
            </div>
            <TrainingPerformance metric={latest?.metrics.at(-1)} />
            <TrainingQuality job={latest} />
            {latest?.metrics.at(-1)?.motion_mae !== undefined && (
              <p className="fine-print">
                Bewegungsfehler: {latest.metrics.at(-1)?.motion_mae?.toFixed(3)},
                Objekttyp korrekt:{" "}
                {(
                  (latest.metrics.at(-1)?.object_type_accuracy ?? 0) * 100
                ).toFixed(1)}{" "}
                %. Diese Werte nutzen die richtige Vorgeschichte. Freie
                Generierungen prüfen zusätzlich die eigene Bewegungsfolge.
              </p>
            )}
            {latest?.metrics.at(-1)?.onset_f1 !== undefined && (
              <p className="fine-print">
                Notenzeitpunkte auf Prüfsongs (±35 ms): F1{" "}
                {((latest.metrics.at(-1)?.onset_f1 ?? 0) * 100).toFixed(1)} %,
                Präzision{" "}
                {((latest.metrics.at(-1)?.onset_precision ?? 0) * 100).toFixed(
                  1,
                )}{" "}
                %, Trefferquote{" "}
                {((latest.metrics.at(-1)?.onset_recall ?? 0) * 100).toFixed(1)}{" "}
                %. Gemessen vor Dichtevorgabe und Rasterkorrektur; kein
                Spielspaß-Score.
              </p>
            )}
          </section>
          <section className="panel training-explainer">
            <h2>Was dein Modell lernt</h2>
            <div>
              <span>01</span>
              <p>
                <b>Musik im Zusammenhang</b>Rund 24 Sekunden detailliertes Audio
                plus ein komprimierter Überblick über den gesamten Song.
              </p>
            </div>
            <div>
              <span>02</span>
              <p>
                <b>Folgen statt Einzelpunkte</b>Bis zu 128 Objekte Kontext.
                Gemeinsame Verteilungen für Position und Kurve, Lernziele für
                Abstände, Richtungswechsel, Geschwindigkeit und Wiederholungen.
              </p>
            </div>
            <div>
              <span>03</span>
              <p>
                <b>Kandidaten prüfen</b>Mehrere Platzierungen werden nach
                gelerntem Kontext, Bewegung und Lesbarkeit verglichen. Der
                Export prüft Grenzen, Slider-Zeiten und Überlappungen.
              </p>
            </div>
            <p className="muted">
              Phrase Mapper · gemeinsame Transformer-Architektur mit zwölf
              Lernzielen. Gute Trainingsmaps und unabhängige Spieltests bleiben
              entscheidend. Frühstopp nach zwölf Epochen ohne besseren
              Prüfverlust.
            </p>
          </section>
        </div>
      </div>
    </>
  );
}
