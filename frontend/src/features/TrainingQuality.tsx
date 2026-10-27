import type { Job, RolloutSuite } from "../types";

export function RolloutQuality({ suite }: { suite: RolloutSuite }) {
  return (
    <div className="fine-print">
      <p>
        Freie Generierung: {suite.songs}/{suite.attempted} Song-Abschnitte · bis
        zu {suite.max_objects_per_song} Objekte pro Song.
        {suite.epoch ? ` Gespeicherter Stand: Epoche ${suite.epoch}.` : ""}
      </p>
      {suite.songs > 0 && (
        <p>
          Mittelwerte: Zeitpunkte-F1{" "}
          {(suite.means.reference_onset_f1 * 100).toFixed(1)} % ·{" "}
          {(suite.means.heavily_repaired_fraction * 100).toFixed(1)} % stärker
          korrigiert · {suite.means.temporal_overlaps.toFixed(1)} zeitliche
          Überschneidungen · {suite.means.slider_shape_issues.toFixed(1)}{" "}
          auffällige Slider. Referenzvergleich mit eigener Vorgeschichte, kein
          Spielspaß-Score.
        </p>
      )}
      {suite.failures.map((failure) => (
        <p key={failure.map_id}>
          Prüfabschnitt {failure.map_id}: {failure.reason}
        </p>
      ))}
    </div>
  );
}

export function TrainingQuality({ job }: { job?: Job }) {
  const metric = job?.metrics.at(-1);
  const summary = job?.training_summary;
  const rollout = job?.metrics
    .slice()
    .reverse()
    .find((entry) => entry.free_validation);
  return (
    <>
      {summary && (
        <p className="fine-print">
          Gespeichert: Epoche {summary.best_epoch} ·{" "}
          {summary.validation_songs ? "Prüfverlust" : "Trainingsverlust"}{" "}
          {summary.best_loss?.toFixed(4) ?? "—"}. Zuletzt abgeschlossen: Epoche{" "}
          {summary.last_epoch}.
          {summary.stop_reason === "early_stopping"
            ? ` Frühstopp nach ${summary.stale_epochs} Epochen ohne neuen Bestwert.`
            : ` ${summary.stale_epochs}/${summary.patience} Epochen ohne neuen Bestwert.`}
          <br />
          {summary.training_songs} Trainingssongs · {summary.validation_songs}{" "}
          Validierungssongs · {summary.test_songs} separate Testsongs.
        </p>
      )}
      {metric?.learning_rate !== undefined && (
        <p className="fine-print">
          Lernrate: {metric.learning_rate.toExponential(2)} · nächste Epoche:{" "}
          {(metric.next_learning_rate ?? metric.learning_rate).toExponential(2)}
          {metric.learning_rate_reduced ? " · wegen Stillstand reduziert" : ""}.
          {metric.context_probability !== undefined &&
            ` Eigene Positionsvorschläge im Zusatztraining: ${(metric.context_probability * 100).toFixed(0)} % der geeigneten Vorgänger.`}
        </p>
      )}
      {rollout?.free_validation && (
        <details className="quality-details" open>
          <summary>Freie Validierung · Epoche {rollout.epoch}</summary>
          <RolloutQuality suite={rollout.free_validation} />
        </details>
      )}
    </>
  );
}
