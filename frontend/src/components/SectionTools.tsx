import { useState } from "react";
import { Dices, ThumbsDown, ThumbsUp } from "lucide-react";
import { api } from "../api";
import { clock, kindLabels, styleLabels } from "../features/songPlan";
import type { Output, PlanStyle, SongPlan } from "../types";

/**
 * Feedback during generation, section by section: roll one section again (optionally in
 * another style) or rate it. Every action is stored for later preference training.
 */
export function SectionTools({
  output,
  plan,
  time,
  busy,
  onJob,
  onError,
}: {
  output: Output;
  plan: SongPlan;
  time: number;
  busy: boolean;
  onJob: (job: unknown) => void;
  onError: (message: string) => void;
}) {
  const [styles, setStyles] = useState<Record<number, PlanStyle>>({});
  const [rated, setRated] = useState<Record<number, "good" | "bad">>({});
  const sections = plan.sections.filter((section) => section.kind !== "silence");

  async function regenerate(index: number) {
    const section = plan.sections[index];
    try {
      const job = await api(`/outputs/${output.id}/regenerate`, {
        start_ms: section.start_ms,
        end_ms: section.end_ms,
        seed: Math.floor(Math.random() * 2 ** 31),
        style: styles[index] && styles[index] !== section.style ? styles[index] : null,
      });
      onJob(job);
    } catch (error) {
      onError((error as Error).message);
    }
  }

  async function rate(index: number, rating: "good" | "bad") {
    const section = plan.sections[index];
    try {
      await api(`/outputs/${output.id}/feedback`, {
        start_ms: section.start_ms,
        end_ms: section.end_ms,
        rating,
      });
      setRated((current) => ({ ...current, [index]: rating }));
    } catch (error) {
      onError((error as Error).message);
    }
  }

  if (!sections.length) return null;
  return (
    <details className="ranking-check section-tools">
      <summary>
        <Dices size={15} aria-hidden="true" />
        <span>Abschnitte bewerten & neu würfeln</span>
      </summary>
      <ol className="plan-sections compact">
        {plan.sections.map((section, index) =>
          section.kind === "silence" ? null : (
            <li
              key={`${section.start_ms}-${index}`}
              className={section.start_ms <= time && time < section.end_ms ? "current" : ""}
            >
              <span className="plan-time">
                {clock(section.start_ms)}
                <small>{kindLabels[section.kind]}</small>
              </span>
              <select
                aria-label="Stil für die neue Variante"
                value={styles[index] ?? section.style}
                onChange={(event) =>
                  setStyles((current) => ({ ...current, [index]: event.target.value as PlanStyle }))
                }
              >
                {Object.entries(styleLabels).map(([style, label]) => (
                  <option key={style} value={style}>
                    {label}
                  </option>
                ))}
              </select>
              <span className="plan-actions">
                <button
                  type="button"
                  className="icon-button"
                  aria-label="Gefällt mir"
                  aria-pressed={rated[index] === "good"}
                  onClick={() => void rate(index, "good")}
                >
                  <ThumbsUp size={14} />
                </button>
                <button
                  type="button"
                  className="icon-button"
                  aria-label="Gefällt mir nicht"
                  aria-pressed={rated[index] === "bad"}
                  onClick={() => void rate(index, "bad")}
                >
                  <ThumbsDown size={14} />
                </button>
                <button
                  type="button"
                  className="secondary small"
                  disabled={busy}
                  onClick={() => void regenerate(index)}
                >
                  <Dices size={14} aria-hidden="true" /> Neu würfeln
                </button>
              </span>
            </li>
          ),
        )}
      </ol>
      <p className="fine-print">
        Neu würfeln erzeugt nur diesen Abschnitt neu und speichert das Ergebnis als neue
        Version; der Rest der Map bleibt gleich. Bewertungen werden lokal gesammelt
        (Export: /api/feedback).
      </p>
    </details>
  );
}
