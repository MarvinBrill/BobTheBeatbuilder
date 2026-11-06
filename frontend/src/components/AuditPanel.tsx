import { Activity } from "lucide-react";
import type { MapAudit } from "../types";

const ROWS: { key: string; label: string; percent?: boolean; reference?: string }[] = [
  { key: "onset_hit_share", label: "Objekte auf Onsets", percent: true },
  { key: "strong_onset_coverage", label: "Starke Akzente gemappt", percent: true },
  { key: "sharp_share", label: "Scharfe Richtungswechsel", percent: true },
  { key: "star_chain_share", label: "In Stern-Ketten", percent: true },
  { key: "longest_star_chain", label: "Längste Stern-Kette" },
  { key: "slider_half_beat_share", label: "1/2-Beat-Slider", percent: true },
  { key: "slider_velocities", label: "Slider-Geschwindigkeiten" },
  { key: "timing_drift_ms", label: "Raster-Drift (ms)" },
];

function format(value: number | null | undefined, percent?: boolean) {
  if (value == null) return "–";
  return percent ? `${Math.round(value * 100)} %` : value.toFixed(value >= 10 ? 0 : 1);
}

/** Mapper's first checks: on the music? patterns varied? Compared with human maps. */
export function AuditPanel({ audit }: { audit: MapAudit }) {
  return (
    <details className="ranking-check audit-panel" open={audit.warnings.length > 0}>
      <summary>
        <Activity size={15} aria-hidden="true" />
        <span>
          Map-Check
          {audit.warnings.length
            ? ` · ${audit.warnings.length} ${audit.warnings.length === 1 ? "Auffälligkeit" : "Auffälligkeiten"}`
            : " · unauffällig"}
        </span>
      </summary>
      {audit.warnings.length > 0 && (
        <ul className="audit-warnings">
          {audit.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
      <table className="audit-table">
        <thead>
          <tr>
            <th>Messung</th>
            <th>Diese Map</th>
            <th>Menschen (Median)</th>
          </tr>
        </thead>
        <tbody>
          {ROWS.filter((row) => row.key in audit.metrics).map((row) => (
            <tr key={row.key}>
              <td>{row.label}</td>
              <td>{format(audit.metrics[row.key], row.percent)}</td>
              <td>{format(audit.reference[row.key], row.percent)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="fine-print">
        Gemessen wie im Training (Mel-Onsets, Drehwinkel zwischen Sprüngen). Hinweise, keine
        Spielspaß-Bewertung.
      </p>
    </details>
  );
}
