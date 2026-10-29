import { useState } from "react";
import {
  BrainCircuit,
  ArrowUpRight,
  Download,
  LoaderCircle,
  Upload,
} from "lucide-react";
import { acceptsFile, upload } from "../api";
import { ProgressLine, useDropTarget } from "../components/Feedback";
import { activeJob, jobMessage } from "./jobs";
import { LossChart } from "../components/LossChart";
import type { State } from "../types";
import { DeleteArtifactButton } from "../components/DeleteArtifactButton";
import { RolloutQuality } from "./TrainingQuality";

export function Models({
  state,
  goTraining,
  refresh,
  onError,
  onDeleted,
}: {
  state: State;
  goTraining: () => void;
  refresh: () => void;
  onError: (message: string) => void;
  onDeleted: (kind: "model" | "output", identifier: string) => void;
}) {
  const [uploading, setUploading] = useState<number | null>(null);
  const deletionBlocked = state.deletion_blocked;
  const importing = activeJob(state.jobs, "model-import");

  async function importModel(file?: File) {
    if (!file || uploading !== null) return;
    if (!acceptsFile(file, [".zip"])) {
      onError(`„${file.name}“ ist kein Bob-Modell. Erwartet wird eine .zip-Datei.`);
      return;
    }
    setUploading(0);
    try {
      await upload("/models/import", file, setUploading);
      refresh();
    } catch (error) {
      onError((error as Error).message);
    } finally {
      setUploading(null);
    }
  }
  const drop = useDropTarget(
    (file) => void importModel(file),
    uploading !== null,
  );

  return (
    <>
      <div className="page-heading">
        <div>
          <h1>
            Model <span className="accent">collection.</span>
          </h1>
          <p>
            Deine persönlichen Mapper. Jeder Stil in einer Datei, lokal
            trainiert, portabel gespeichert.
          </p>
        </div>
        <button type="button" className="primary" onClick={goTraining}>
          Neues Modell <ArrowUpRight size={17} aria-hidden="true" />
        </button>
      </div>
      <label
        className={`dropzone compact ${drop.dragging ? "dragging" : ""} ${uploading !== null ? "busy" : ""}`}
        aria-busy={uploading !== null}
        {...drop.handlers}
      >
        <input
          type="file"
          accept=".zip"
          aria-label="Bob-Modell (.zip) auswählen"
          disabled={uploading !== null}
          onChange={(event) => {
            void importModel(event.target.files?.[0]);
            event.target.value = "";
          }}
        />
        {uploading !== null ? (
          <LoaderCircle size={20} className="spin" aria-hidden="true" />
        ) : (
          <Upload size={20} aria-hidden="true" />
        )}
        <b>
          {uploading !== null
            ? "Modell wird hochgeladen …"
            : drop.dragging
              ? "Loslassen zum Importieren"
              : "Gespeichertes Bob-Modell importieren"}
        </b>
        {uploading !== null ? (
          <ProgressLine value={uploading} label="Upload" />
        ) : (
          <small>ZIP mit Safetensors-Gewichten und Metadaten</small>
        )}
      </label>
      {importing && (
        <ProgressLine
          value={importing.progress}
          label={jobMessage(importing) || "Modell wird importiert"}
        />
      )}
      {state.models.length ? (
        <div className="model-grid">
          {state.models.map((model) => (
            <section className="panel model-card" key={model.id}>
              <div className="model-art" aria-hidden="true">
                <BrainCircuit size={52} />
                <span>Phrase Mapper / 03</span>
              </div>
              <div className="panel-heading">
                <h2>{model.name}</h2>
                <span
                  className={`tag ${model.training_status === "interrupted" ? "status-interrupted" : "status-completed"}`}
                >
                  {model.training_status === "interrupted"
                    ? "Unterbrochen"
                    : "Bereit"}
                </span>
              </div>
              <dl className="model-stats">
                <div>
                  <dt>Maps</dt>
                  <dd>{model.maps}</dd>
                </div>
                <div>
                  <dt>Trainingssongs</dt>
                  <dd>{model.training_songs}</dd>
                </div>
                <div>
                  <dt>Validierungssongs</dt>
                  <dd>{model.validation_songs}</dd>
                </div>
              </dl>
              <LossChart metrics={model.metrics} />
              {!!model.best_epoch && (
                <p className="fine-print">
                  Gespeicherter Stand: Epoche {model.best_epoch} (niedrigster
                  Verlust).
                </p>
              )}
              <p className="fine-print">
                {model.parameters
                  ? `${(model.parameters / 1e6).toFixed(1)} Mio. Parameter. `
                  : ""}
                {model.continuable
                  ? "Lokaler Trainingsstand kann fortgesetzt werden."
                  : "Portables Modell zur Generierung."}
                {model.test_songs
                  ? ` ${model.test_songs} unabhängige Testsongs.`
                  : ""}
              </p>
              {model.warnings?.map((warning) => (
                <p className="fine-print" key={warning}>
                  {warning}
                </p>
              ))}
              {model.free_generation_suite && (
                <RolloutQuality suite={model.free_generation_suite} />
              )}
              {!model.free_generation_suite && model.free_generation && (
                <p className="fine-print">
                  Freier Prüfabschnitt: {model.free_generation.objects} Objekte,
                  Zeitpunkte-F1{" "}
                  {(model.free_generation.reference_onset_f1 * 100).toFixed(1)}{" "}
                  %,{" "}
                  {(
                    model.free_generation.heavily_repaired_fraction * 100
                  ).toFixed(1)}{" "}
                  % stärker korrigierte Platzierungen,{" "}
                  {model.free_generation.temporal_overlaps} zeitliche
                  Überschneidungen. Diagnostik mit eigener Vorgeschichte, keine
                  Bewertung des Spielgefühls.
                </p>
              )}
              <div className="model-footer">
                <small>
                  {new Date(model.created_at).toLocaleDateString("de")}
                </small>
                <div className="artifact-actions">
                  <a
                    className="secondary"
                    href={`/api/models/${model.id}/download`}
                    download
                  >
                    <Download size={16} aria-hidden="true" /> Modell
                    herunterladen
                  </a>
                  <DeleteArtifactButton
                    kind="model"
                    identifier={model.id}
                    name={model.name}
                    blocked={deletionBlocked}
                    onDeleted={() => onDeleted("model", model.id)}
                  />
                </div>
              </div>
              <p className="fine-print">
                {deletionBlocked
                  ? "Löschen ist nach Abschluss oder Abbruch aller Aufgaben möglich."
                  : "ZIP mit einem Safetensors-Modell und Trainingsmetadaten."}
              </p>
            </section>
          ))}
        </div>
      ) : (
        <section className="panel large-empty">
          <BrainCircuit size={50} aria-hidden="true" />
          <h2>Dein erster Mapper wartet auf dich.</h2>
          <p>Wähle Trainingsdaten und entwickle deinen eigenen Mapping-Stil.</p>
          <button type="button" className="primary" onClick={goTraining}>
            Zum Training <ArrowUpRight size={17} aria-hidden="true" />
          </button>
        </section>
      )}
    </>
  );
}
