import type { Job } from "../types";

export const statusLabels: Record<string, string> = {
  queued: "Wartet",
  running: "Läuft",
  completed: "Abgeschlossen",
  failed: "Fehlgeschlagen",
  cancelled: "Abgebrochen",
  interrupted: "Unterbrochen",
};

export function statusLabel(status?: string) {
  return status ? statusLabels[status] || status : "";
}

// The backend names its own jobs in English; jobs named by the user keep their name.
const generic: Record<string, string> = {
  scan: "Library-Scan",
  import: "Beatmap-Import",
  export: "Export",
  "model-import": "Modell-Import",
  stars: "Sterne berechnen",
};
export function jobTitle(job: Job) {
  return generic[job.kind] || job.name;
}

const messages: Record<string, string> = {
  Starting: "Startet …",
  "Waiting for worker": "Wartet auf Start …",
  Completed: "Fertig",
  Cancelled: "Abgebrochen",
};
export function jobMessage(job: Job) {
  return messages[job.message] || job.message;
}

export function isActive(job?: Job) {
  return !!job && (job.status === "queued" || job.status === "running");
}

export function activeJob(jobs: Job[], kind: string) {
  return jobs.find((job) => job.kind === kind && isActive(job));
}

export function scrollToActivity() {
  const panel = document.getElementById("activity");
  panel?.scrollIntoView({
    behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
      ? "auto"
      : "smooth",
    block: "start",
  });
  panel?.focus({ preventScroll: true });
}
