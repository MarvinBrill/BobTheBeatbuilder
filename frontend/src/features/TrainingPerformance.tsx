import type { Metric } from "../types";

export function TrainingPerformance({ metric }: { metric?: Metric }) {
  const performance = metric?.performance;
  if (!performance) return null;
  return (
    <p className="fine-print">
      Durchsatz: {performance.examples_per_second.toFixed(1)} Audiofenster/s,{" "}
      {performance.precision === "bfloat16" ? "BF16" : "Float32"}
      {performance.peak_allocated_mib != null && (
        <>
          {" "}
, GPU-Speicher: {(performance.peak_allocated_mib / 1024).toFixed(
            2,
          )}{" "}
          GiB aktiv, {((performance.peak_reserved_mib ?? 0) / 1024).toFixed(2)}{" "}
          GiB reserviert
        </>
      )}
      . Spitzenwerte der letzten Epoche. Die erste Epoche enthält die einmalige
      Vorbereitung; Werte ohne Validierung und Audioaufbereitung.
    </p>
  );
}
