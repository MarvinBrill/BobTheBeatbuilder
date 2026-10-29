import { useEffect, useState } from "react";
import type { Job } from "../types";

/**
 * A submit button stays busy from the click until polling shows the new job,
 * so there is no gap in which a second click starts a duplicate.
 */
export function usePendingJob(jobs: Job[]) {
  const [pending, setPending] = useState<string | null>(null);
  const arrived = !!pending && jobs.some((job) => job.id === pending);
  useEffect(() => {
    if (!pending) return;
    if (arrived) {
      setPending(null);
      return;
    }
    // Fallback: never stay stuck if the job never shows up.
    const timer = setTimeout(() => setPending(null), 8000);
    return () => clearTimeout(timer);
  }, [pending, arrived]);
  return {
    waiting: !!pending && !arrived,
    track: (job: unknown) => {
      const id = (job as { id?: unknown } | null)?.id;
      if (typeof id === "string") setPending(id);
    },
  };
}
