"""Same-output rhythm ablation; no weights are converted or trained here."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.ml.structured_rhythm import onset_candidates, select_sequence
from app.ml.phrase_metrics import onset_matches
from app.timing import TimingPoint


def comparison(baseline):
    results = []
    for item in baseline["items"]:
        prediction_path = Path(item["prediction_path"])
        if hashlib.sha256(prediction_path.read_bytes()).hexdigest() != item["prediction_sha256"]:
            raise ValueError("Saved prediction changed")
        prediction = np.load(prediction_path, allow_pickle=False)
        mel = np.load(item["mel"], mmap_mode="r", allow_pickle=False)
        reference = np.load(item["events"], allow_pickle=False)[:, 0] * 1000
        settings = dict(density=item["condition"][4] * 10, snap=True, divisor=4, bpm=None)
        # The saved v3 experiment contains beat anchors, no metric information.
        # Local pulse consistency remains the selector's confidence evidence.
        timing = [TimingPoint(time, beat) for time, beat in item["timing"]]
        started = time.perf_counter()
        candidates = onset_candidates(prediction, mel, settings, timing)
        prepare_seconds = time.perf_counter() - started
        variants = [("A_stored_fixed_window", item["selected_times"], None)]
        for method, width in (("B_greedy", 1), ("B_beam8", 8)):
            started = time.perf_counter()
            times = select_sequence(candidates, timing, width)
            variants.append((method, times, time.perf_counter() - started + prepare_seconds))
        for method, times, seconds in variants:
            boundaries = np.linspace(0, item["duration_seconds"] * 1000, 5)
            for section, (start, end) in enumerate([(0, boundaries[-1]), *zip(boundaries[:-1], boundaries[1:])]):
                chosen = [value for value in times if start <= value < end]
                target = reference[(reference >= start) & (reference < end)]
                matches = onset_matches(chosen, target, 35)
                results.append(dict(map_id=item["map_id"], method=method,
                                    section="full" if section == 0 else section,
                                    start_ms=start, end_ms=end, heads=len(chosen),
                                    reference_heads=len(target), reference_f1=2 * matches / max(1, len(chosen) + len(target)),
                                    selection_seconds=seconds if section == 0 else None))
    return dict(schema="rhythm-ablation-v1", baseline_revision=baseline["revision"],
                model_sha256=baseline["model_sha256"], results=results,
                note="Six development/validation songs, fixed model outputs. Deterministic selector. F1 is reference similarity, not musical quality. No final test used.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=ROOT / "data/benchmarks/architecture/baseline.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/benchmarks/architecture/rhythm-ablation.json")
    args = parser.parse_args()
    report = comparison(json.loads(args.baseline.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for method in ("A_stored_fixed_window", "B_greedy", "B_beam8"):
        rows = [row for row in report["results"] if row["method"] == method and row["section"] == "full"]
        print(method, "heads", sum(row["heads"] for row in rows), "mean reference F1", round(np.mean([row["reference_f1"] for row in rows]), 4), "seconds", sum(row["selection_seconds"] or 0 for row in rows))


if __name__ == "__main__":
    main()
