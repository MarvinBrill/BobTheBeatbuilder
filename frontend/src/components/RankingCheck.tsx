import { useState } from "react";
import { ClipboardCopy, ShieldCheck } from "lucide-react";
import type { FinishingSummary, RankingCheck } from "../types";

const SEVERITY = {
  problem: "Problem",
  warning: "Hinweis",
  info: "Info",
} as const;

function clock(ms: number) {
  const total = Math.max(0, Math.round(ms));
  return `${Math.floor(total / 60000)}:${String(Math.floor(total / 1000) % 60).padStart(2, "0")}`;
}

/** One line on what the finishing pass added (hitsounds, kiai, breaks, preview point). */
export function FinishingLine({ summary }: { summary: FinishingSummary }) {
  const parts = [
    summary.hitsounds === "off"
      ? "ohne Hitsound-Additions"
      : `${summary.claps ?? 0} Claps · ${summary.finishs ?? 0} Finishes · ${summary.whistles ?? 0} Whistles`,
    summary.kiai_sections
      ? `${summary.kiai_sections} Kiai (${Math.round(summary.kiai_seconds)} s)`
      : "kein Kiai",
    `${summary.breaks} Breaks`,
    summary.preview_time >= 0 && `Preview ab ${clock(summary.preview_time)}`,
  ];
  return (
    <p className="fine-print">Finishing: {parts.filter(Boolean).join(" · ")}</p>
  );
}

/** Ranking-Criteria checklist with editor timestamps, like a modding post. */
export function RankingCheckPanel({ check }: { check: RankingCheck }) {
  const [copied, setCopied] = useState(false);
  const findings = check.findings.filter((item) => item.severity !== "info");
  const notes = check.findings.filter((item) => item.severity === "info");
  async function copy() {
    const text = findings
      .map((item) =>
        item.times.length
          ? item.times.map((time) => `${time} - ${item.title}: ${item.message}`).join("\n")
          : `${item.title}: ${item.message}`,
      )
      .join("\n");
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  }
  return (
    <details className="ranking-check" open={check.problems > 0}>
      <summary>
        <ShieldCheck size={15} aria-hidden="true" />
        <span>
          RC-Check {check.tier ? `· ${check.tier}` : "· ohne Stufe"}
          {check.stars != null && ` (${check.stars.toFixed(2)} ★)`}
        </span>
        <span className={`rc-count ${check.problems ? "has-problems" : ""}`}>
          {check.problems} Probleme · {check.warnings} Hinweise
        </span>
      </summary>
      {findings.length ? (
        <ul>
          {findings.map((item) => (
            <li key={item.id} className={`rc-finding rc-${item.severity}`}>
              <span className="rc-severity">{SEVERITY[item.severity]}</span>
              <div>
                <b>{item.title}</b>
                <p>{item.message}</p>
                {item.times.length > 0 && (
                  <code>
                    {item.times.join(", ")}
                    {item.count > item.times.length &&
                      ` … (+${item.count - item.times.length})`}
                  </code>
                )}
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <p className="fine-print">Keine messbaren Verstöße für diese Stufe gefunden.</p>
      )}
      {notes.map((item) => (
        <p key={item.id} className="fine-print">
          <b>{item.title}:</b> {item.message}
        </p>
      ))}
      <div className="rc-actions">
        <span className="fine-print">{check.note}</span>
        {findings.length > 0 && (
          <button type="button" className="secondary small" onClick={copy}>
            <ClipboardCopy size={14} aria-hidden="true" />
            {copied ? "Kopiert" : "Als Mod-Text kopieren"}
          </button>
        )}
      </div>
    </details>
  );
}
