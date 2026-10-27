import { Check, LoaderCircle, X, AlertCircle, Download } from "lucide-react";
import { api } from "../api";
import type { Job } from "../types";

export function JobActivity({
  jobs,
  onError,
}: {
  jobs: Job[];
  onError: (message: string) => void;
}) {
  if (!jobs.length)
    return (
      <div className="activity-empty">
        <span className="status-dot" /> Bereit, wenn du es bist. Deine Aufgaben
        erscheinen hier.
      </div>
    );
  return (
    <div className="job-list">
      {jobs.slice(0, 5).map((job) => {
        const active = ["running", "queued"].includes(job.status);
        return (
          <div className={`job ${job.status}`} key={job.id}>
            <div className="job-icon">
              {active ? (
                <LoaderCircle className="spin" size={18} />
              ) : job.status === "completed" ? (
                <Check size={18} />
              ) : (
                <AlertCircle size={18} />
              )}
            </div>
            <div className="job-body">
              <div className="job-title">
                <b>{job.name}</b>
                <span>{Math.round(job.progress * 100)}%</span>
              </div>
              <p>{job.message}</p>
              {job.cleanup_warning && (
                <p role="status">{job.cleanup_warning}</p>
              )}
              {active && <progress max={1} value={job.progress} />}
              {job.result?.imported !== undefined && (
                <small>
                  {job.result.imported} Maps gefunden ·{" "}
                  {job.result.skipped_other_modes || 0} andere Spielmodi
                  übersprungen · {job.result.error_count || 0} Fehler
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
            </div>
            {active && (
              <button
                className="icon-button"
                aria-label="Aufgabe abbrechen"
                onClick={() =>
                  api(`/jobs/${job.id}/cancel`, {}).catch((error) =>
                    onError(error.message),
                  )
                }
              >
                <X size={16} />
              </button>
            )}
            {job.result?.output_id && (
              <a
                className="icon-button"
                aria-label="Ergebnis herunterladen"
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
