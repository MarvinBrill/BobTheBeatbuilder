import { useState } from "react";
import { BrainCircuit, ArrowUpRight, Download, Upload } from "lucide-react";
import { upload } from "../api";
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
  const [busy, setBusy] = useState(false);
  const deletionBlocked = state.deletion_blocked;

  async function importModel(file?: File) {
    if (!file) return;
    setBusy(true);
    try {
      await upload("/models/import", file);
      refresh();
    } catch (error) {
      onError((error as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">DEINE PERSÖNLICHEN MAPPER.</div>
          <h1>
            Model <em>collection.</em>
          </h1>
          <p>
            Jeder Stil in einer Datei. Lokal trainiert, portabel gespeichert.
          </p>
        </div>
        <button className="primary" onClick={goTraining}>
          Neues Modell <ArrowUpRight size={17} />
        </button>
      </div>
      <label className="dropzone compact">
        <input
          type="file"
          accept=".zip"
          disabled={busy}
          onChange={(event) => {
            void importModel(event.target.files?.[0]);
            event.target.value = "";
          }}
        />
        <Upload size={20} />
        <b>
          {busy
            ? "Modell wird hochgeladen …"
            : "Gespeichertes Bob-Modell importieren"}
        </b>
        <small>ZIP mit Safetensors-Gewichten und Metadaten</small>
      </label>
      {state.models.length ? (
        <div className="model-grid">
          {state.models.map((model) => (
            <section className="panel model-card" key={model.id}>
              <div className="model-art">
                <BrainCircuit size={52} />
                <span>PHRASE MAPPER / 03</span>
              </div>
              <div className="panel-heading">
                <h2>{model.name}</h2>
                <span className="tag">
                  {model.training_status === "interrupted"
                    ? "Unterbrochen"
                    : "Bereit"}
                </span>
              </div>
              <p>
                {model.maps} Maps · {model.training_songs} Trainingssongs ·{" "}
                {model.validation_songs} Validierungssongs
              </p>
              <LossChart metrics={model.metrics} />
              {!!model.best_epoch && (
                <p className="fine-print">
                  Gespeicherter Stand: Epoche {model.best_epoch} (niedrigster
                  Verlust).
                </p>
              )}
              <p className="fine-print">
                {model.parameters
                  ? `${(model.parameters / 1e6).toFixed(1)} Mio. Parameter · `
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
                  Freier Prüfabschnitt: {model.free_generation.objects} Objekte
                  · Zeitpunkte-F1{" "}
                  {(model.free_generation.reference_onset_f1 * 100).toFixed(1)}{" "}
                  % ·{" "}
                  {(
                    model.free_generation.heavily_repaired_fraction * 100
                  ).toFixed(1)}{" "}
                  % stärker korrigierte Platzierungen ·{" "}
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
                  >
                    <Download size={16} /> Modell herunterladen
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
          <BrainCircuit size={50} />
          <h2>Dein erster Mapper wartet auf dich.</h2>
          <p>Wähle Trainingsdaten und entwickle deinen eigenen Mapping-Stil.</p>
          <button className="primary" onClick={goTraining}>
            Zum Training <ArrowUpRight size={17} />
          </button>
        </section>
      )}
    </>
  );
}
