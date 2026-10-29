import type { Metric } from "../types";
import { useState } from "react";

const groups = {
  total: "Gesamtverlust",
  rhythm: "Rhythmus",
  geometry: "Positionen und Kurven",
  structure: "Objekttyp und Slider-Dauer",
  movement: "Bewegung und Zusammenhang",
  pattern: "Rhythmus-Patterns",
};

export function LossChart({ metrics }: { metrics: Metric[] }) {
  const [group, setGroup] = useState("total");
  const separate = metrics.some((metric) => metric.train_groups);
  const available = Object.entries(groups).filter(
    ([key]) =>
      key === "total" || metrics.some((metric) => metric.train_groups?.[key] !== undefined),
  );
  const displayed =
    group === "total" || !separate
      ? metrics
      : metrics
          .filter((metric) => metric.train_groups?.[group] !== undefined)
          .map((metric) => ({
            ...metric,
            train_loss: metric.train_groups![group],
            validation_loss: metric.validation_groups?.[group] ?? null,
          }));
  metrics = displayed;
  if (!metrics.length)
    return (
      <div className="empty-chart">
        <div className="chart-grid" />
        <p>Die Lernkurve erscheint nach der ersten Epoche.</p>
      </div>
    );
  const values = metrics.flatMap((metric) => [
    metric.train_loss,
    metric.validation_loss ?? metric.train_loss,
  ]);
  const minimum = Math.min(0, ...values);
  const maximum = Math.max(0.000001, ...values);
  const span = Math.max(0.000001, maximum - minimum) * 1.1;
  const height = (value: number) => 180 - ((value - minimum) / span) * 160;
  const points = (key: "train_loss" | "validation_loss") =>
    metrics
      .filter((metric) => metric[key] !== null)
      .map(
        (metric, index) =>
          `${20 + (index / Math.max(1, metrics.length - 1)) * 540},${height(metric[key] ?? 0)}`,
      )
      .join(" ");
  return (
    <>
      {separate && (
        <label>
          Lernziel anzeigen
          <select
            value={group}
            onChange={(event) => setGroup(event.target.value)}
          >
            {available.map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        </label>
      )}
      <svg
        className="loss-chart"
        viewBox="0 0 580 210"
        role="img"
        aria-label="Trainings- und Validierungsverlust pro Epoche"
      >
        {[20, 60, 100, 140, 180].map((y) => (
          <line key={y} x1="20" y1={y} x2="560" y2={y} className="chart-gridline" />
        ))}
        <polyline
          points={points("train_loss")}
          className="chart-line train"
        />
        <polyline
          points={points("validation_loss")}
          className="chart-line validation"
        />
        {metrics.map((metric, index) => (
          <circle
            key={metric.epoch}
            cx={20 + (index / Math.max(1, metrics.length - 1)) * 540}
            cy={height(metric.train_loss)}
            r="3"
            className="chart-point"
          />
        ))}
        <text x="20" y="205" className="chart-axis">
          Epoche {metrics[0].epoch}
        </text>
        <text x="510" y="205" className="chart-axis">
          {metrics.at(-1)?.epoch}
        </text>
      </svg>
      <div className="chart-legend">
        <span>Training</span>
        <span>Validierung</span>
      </div>
      {separate && group !== "total" && (
        <p className="fine-print">
          Zuletzt: Training {metrics.at(-1)?.train_loss.toFixed(4)},
          Validierung {metrics.at(-1)?.validation_loss?.toFixed(4) ?? "—"}.
          Kleinere Werte sind besser.
          {group === "geometry"
            ? " Der Wahrscheinlichkeitsverlust darf negativ sein."
            : ""}
        </p>
      )}
    </>
  );
}
