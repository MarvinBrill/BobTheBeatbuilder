import { useEffect, useRef } from "react";
import type { Job, Page } from "../types";
import type { ToastInput } from "../components/Toasts";
import { isActive, jobTitle, scrollToActivity } from "./jobs";

const kindLabels: Record<string, string> = {
  scan: "Library-Scan",
  import: "Import",
  dataset: "Trainingsdaten",
  export: "Export",
  training: "Training",
  generation: "Generierung",
  "model-import": "Modell-Import",
};

function kindLabel(job: Job) {
  return kindLabels[job.kind] || job.name;
}

function importSummary(job: Job) {
  const result = job.result;
  if (result?.imported === undefined) return undefined;
  return `${result.imported} Maps aufgenommen, ${result.skipped_other_modes || 0} andere Spielmodi übersprungen, ${result.error_count || 0} Fehler.`;
}

/**
 * Turns job status changes from polling into toasts: a job that starts, and a
 * job that finishes, fails or is cancelled while the page is open.
 */
/** The screen that already shows a job's progress inline. */
const homePage: Record<string, Page> = {
  scan: "library",
  import: "library",
  dataset: "library",
  export: "library",
  training: "training",
  generation: "studio",
  "model-import": "models",
};

export function useJobNotifications(
  jobs: Job[] | undefined,
  notify: (toast: ToastInput) => void,
  navigate: (page: Page) => void,
  showOutput: (identifier: string) => void,
  page: Page,
) {
  const seen = useRef<Map<string, string> | null>(null);
  const pageRef = useRef(page);
  pageRef.current = page;
  /** Drop a "go there" action when the user is already there. */
  const goTo = (label: string, target: Page) =>
    pageRef.current === target
      ? undefined
      : { label, run: () => navigate(target) };

  useEffect(() => {
    if (!jobs) return;
    const previous = seen.current;
    seen.current = new Map(jobs.map((job) => [job.id, job.status]));
    if (!previous) return; // First load: the history is not news.

    const activity = { label: "Fortschritt ansehen", run: scrollToActivity };
    for (const job of jobs) {
      const before = previous.get(job.id);
      if (before === job.status) continue;
      const label = kindLabel(job);
      if (before === undefined && isActive(job)) {
        // The page the user is on already shows this job's progress inline.
        if (homePage[job.kind] === pageRef.current) continue;
        const waiting =
          job.status === "queued" &&
          jobs.some((other) => other.id !== job.id && other.status === "running");
        notify({
          key: `job:${job.id}`,
          tone: "info",
          title: `${label} gestartet`,
          message:
            jobTitle(job) === job.name
              ? `„${job.name}“ ${waiting ? "wartet, bis die laufende Aufgabe fertig ist." : "läuft im Hintergrund."}`
              : waiting
                ? "Wartet, bis die laufende Aufgabe fertig ist."
                : "Läuft im Hintergrund.",
          action: activity,
        });
        continue;
      }
      if (before !== undefined && !["queued", "running"].includes(before))
        continue;
      if (job.status === "completed") {
        const output = job.result?.output_id;
        if (job.kind === "generation" && output) showOutput(output);
        const toast: Record<string, Omit<ToastInput, "tone" | "key">> = {
          scan: { title: "Library eingelesen", message: importSummary(job) },
          import: { title: "Import abgeschlossen", message: importSummary(job) },
          dataset: {
            title: "Trainingsdaten bereit",
            message: `„${job.name}“ steht im Training zur Auswahl.`,
            action: goTo("Zum Training", "training"),
          },
          export: {
            title: "Export fertig",
            message: "Die Auswahl liegt als .osz zum Download bereit.",
            action: output
              ? { label: "Herunterladen", href: `/api/outputs/${output}/download` }
              : undefined,
          },
          training: {
            title: "Training abgeschlossen",
            message: `„${job.name}“ ist unter Meine Modelle gespeichert.`,
            action: goTo("Modell ansehen", "models"),
          },
          generation: {
            title: "Beatmap fertig",
            message: `„${job.name}“ ist in der Vorschau geöffnet.`,
            action: goTo("Ansehen", "studio"),
          },
          "model-import": {
            title: "Modell importiert",
            action: goTo("Meine Modelle", "models"),
          },
        };
        notify({
          key: `job:${job.id}`,
          tone: "success",
          ...(toast[job.kind] || { title: `${label} abgeschlossen` }),
        });
      } else if (job.status === "failed") {
        notify({
          key: `job:${job.id}`,
          tone: "error",
          title: `${label} fehlgeschlagen`,
          message: job.message,
          action: { label: "Details ansehen", run: scrollToActivity },
        });
      } else if (job.status === "cancelled") {
        notify({
          key: `job:${job.id}`,
          tone: "info",
          title: `${label} abgebrochen`,
          message:
            job.kind === "training"
              ? `„${job.name}“: Ein bereits gespeicherter Trainingsstand bleibt zum Fortsetzen erhalten.`
              : jobTitle(job) === job.name
                ? `„${job.name}“ wurde abgebrochen.`
                : undefined,
        });
      }
    }
  }, [jobs, notify, navigate, showOutput]);
}
