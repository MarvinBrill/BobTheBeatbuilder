import { useEffect, useState } from "react";
import {
  AlertCircle,
  Ban,
  Check,
  Download,
  LoaderCircle,
  X,
} from "lucide-react";
import { api } from "../api";
import {
  isActive,
  jobMessage,
  jobTitle,
  statusLabel,
} from "../features/jobs";
import type { Job } from "../types";

export function JobActivity({
  jobs,
  onError,
}: {
  jobs: Job[];
  onError: (message: string) => void;
}) {
  const [cancelling, setCancelling] = useState<Set<string>>(new Set());
  const [confirming, setConfirming] = useState("");

  // A confirm left alone falls back after a few seconds.
  useEffect(() => {
    if (!confirming) return;
    const timer = setTimeout(() => setConfirming(""), 6000);
    return () => clearTimeout(timer);
  }, [confirming]);
  // Forget cancel requests for jobs that have stopped.
  useEffect(() => {
    setCancelling((current) => {
      const next = new Set(
        [...current].filter((id) => isActive(jobs.find((job) => job.id === id))),
      );
      return next.size === current.size ? current : next;
    });
  }, [jobs]);

  async function cancel(job: Job) {
    setConfirming("");
    setCancelling((current) => new Set(current).add(job.id));
    try {
      await api(`/jobs/${job.id}/cancel`, {});
    } catch (error) {
      onError((error as Error).message);
      setCancelling((current) => {
        const next = new Set(current);
        next.delete(job.id);
        return next;
      });
    }
  }

  if (!jobs.length)
    return (
      <div className="activity-empty">
        Bereit, wenn du es bist. Deine Aufgaben erscheinen hier.
      </div>
    );
  return (
    <div className="job-list">
      {jobs.slice(0, 5).map((job) => {
        const active = isActive(job);
        const stopping = cancelling.has(job.id);
        // Long work gets a second click; a stray click should not end hours of training.
        const needsConfirm = job.kind === "training" && job.status === "running";
        return (
          <div className={`job ${job.status}`} key={job.id}>
            <div className="job-icon" aria-hidden="true">
              {active ? (
                <LoaderCircle className="spin" size={18} />
              ) : job.status === "completed" ? (
                <Check size={18} />
              ) : job.status === "cancelled" ? (
                <Ban size={18} />
              ) : (
                <AlertCircle size={18} />
              )}
            </div>
            <div className="job-body">
              <div className="job-title">
                <b>{jobTitle(job)}</b>
                <span>
                  {stopping ? "Wird abgebrochen …" : statusLabel(job.status)}
                  {active && ` ${Math.round(job.progress * 100)} %`}
                </span>
              </div>
              <p>{jobMessage(job)}</p>
              {job.cleanup_warning && (
                <p role="status">{job.cleanup_warning}</p>
              )}
              {active && (
                <progress
                  max={1}
                  value={job.progress}
                  aria-label={`Fortschritt ${job.name}`}
                />
              )}
              {job.result?.imported !== undefined && (
                <small>
                  {job.result.imported} Maps gefunden,{" "}
                  {job.result.skipped_other_modes || 0} andere Spielmodi
                  übersprungen, {job.result.error_count || 0} Fehler
                </small>
              )}
              {!!job.result?.errors?.length && (
                <details>
                  <summary>Import-Hinweise</summary>
                  {job.result.errors.map((error, index) => (
                    <p key={index}>{error}</p>
                  ))}
                </details>
              )}
              {confirming === job.id && (
                <div className="job-confirm" role="group" aria-label="Training abbrechen?">
                  <span>Training wirklich abbrechen?</span>
                  <button
                    type="button"
                    className="danger-button small"
                    autoFocus
                    onClick={() => void cancel(job)}
                  >
                    Ja, abbrechen
                  </button>
                  <button
                    type="button"
                    className="secondary small"
                    onClick={() => setConfirming("")}
                  >
                    Weiterlaufen lassen
                  </button>
                </div>
              )}
            </div>
            {active && (
              <button
                type="button"
                className="icon-button"
                aria-label={
                  stopping ? "Wird abgebrochen" : `${job.name} abbrechen`
                }
                title="Aufgabe abbrechen"
                disabled={stopping || confirming === job.id}
                aria-busy={stopping}
                onClick={() =>
                  needsConfirm ? setConfirming(job.id) : void cancel(job)
                }
              >
                {stopping ? (
                  <LoaderCircle size={16} className="spin" />
                ) : (
                  <X size={16} />
                )}
              </button>
            )}
            {job.result?.output_id && (
              <a
                className="icon-button"
                aria-label={`Ergebnis von ${job.name} herunterladen`}
                title="Herunterladen"
                href={`/api/outputs/${job.result.output_id}/download`}
              >
                <Download size={17} />
              </a>
            )}
          </div>
        );
      })}
    </div>
  );
}
