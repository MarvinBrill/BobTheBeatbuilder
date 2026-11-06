"""Where do a dataset's red lines sit relative to the decoded audio, and how well does the
automatic timing detection find them?

For each sampled map: the offset of the mel onsets against its red lines (see
app/ml/timing_audit.py) and the audio-only detection (app/ml/tempo_grid.py) compared with
the map's own timing. No model is loaded; detection uses the audio and the default timing
convention, so octave errors here are what the model's tempo head has to resolve.

Usage (from the repository root, with the app's Python environment):
    python scripts/benchmarks/timing_offset_audit.py --dataset <dataset id> --limit 200
"""

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app import store  # noqa: E402
from app.beatmaps import parse  # noqa: E402
from app.ml.audio import FRAME_MS, features  # noqa: E402
from app.ml.tempo_grid import detect_timing, onset_envelopes  # noqa: E402
from app.ml.timing_audit import map_offset  # noqa: E402
from app.timing import red_points  # noqa: E402


def dominant(points, duration):
    spans = [
        ((points[i + 1].time if i + 1 < len(points) else duration) - max(0.0, point.time), point)
        for i, point in enumerate(points)
    ]
    return max(spans, key=lambda span: span[0])[1]


def audit(records, limit, seed):
    random.Random(seed).shuffle(records)
    seen, rows = set(), []
    for record in records:
        if len(rows) >= limit:
            break
        if record["audio_hash"] in seen:
            continue
        seen.add(record["audio_hash"])
        try:
            beatmap = parse(Path(record["map_path"]))
            mel = np.asarray(features(Path(record["audio_path"]), record["audio_hash"]), np.float32)
        except (OSError, ValueError) as error:
            rows.append({"map_id": record["id"], "error": str(error)})
            continue
        points = red_points(beatmap.timing)
        if not points:
            continue
        envelope = onset_envelopes(mel)[0]
        offset = map_offset(envelope, points)
        duration = len(mel) * FRAME_MS
        reference = dominant(points, duration)
        try:
            detected = detect_timing(mel, FRAME_MS)
            found = dominant(detected.points, duration)
            ratio = reference.beat_length / found.beat_length
            beat = reference.beat_length
            shift = ((found.time - reference.time) % beat + beat / 2) % beat - beat / 2
            result = dict(
                detected_bpm=round(60000 / found.beat_length, 3),
                bpm_error=round(abs(ratio - 1), 5),
                octave=bool(min(abs(ratio - 0.5), abs(ratio - 2)) < 0.02),
                phase_ms=round(shift, 1) if abs(ratio - 1) < 0.002 else None,
                sections=len(detected.points),
                confidence=detected.info.get("confidence"),
            )
        except ValueError as error:
            result = dict(detected_bpm=None, error=str(error))
        rows.append(
            dict(
                map_id=record["id"],
                title=record.get("title"),
                stars=record.get("star_rating"),
                reference_bpm=round(60000 / reference.beat_length, 3),
                red_lines=len(points),
                offset_ms=None if offset is None else round(offset, 1),
                **result,
            )
        )
    return rows


def summary(rows):
    offsets = [row["offset_ms"] for row in rows if row.get("offset_ms") is not None]
    single = [row for row in rows if row.get("red_lines") == 1 and row.get("detected_bpm")]
    exact = [row for row in single if row["bpm_error"] < 0.0005]
    octave = [row for row in single if row["octave"]]
    phases = [abs(row["phase_ms"]) for row in exact if row.get("phase_ms") is not None]
    return {
        "maps": len(rows),
        "offset_median_ms": float(np.median(offsets)) if offsets else None,
        "offset_iqr_ms": float(np.subtract(*np.percentile(offsets, [75, 25]))) if offsets else None,
        "offset_outside_25ms": int(sum(abs(value - np.median(offsets)) > 25 for value in offsets)) if offsets else 0,
        "constant_tempo_maps": len(single),
        "bpm_exact_share": len(exact) / max(1, len(single)),
        "octave_share": len(octave) / max(1, len(single)),
        "phase_error_median_ms": float(np.median(phases)) if phases else None,
        "note": "Phase error is the audio-only grid against the human grid (default convention).",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=ROOT / "data/benchmarks/timing-offset-audit.json")
    args = parser.parse_args()
    records = list(store.get("dataset", args.dataset)["maps"])
    rows = audit(records, args.limit, args.seed)
    report = {"summary": summary(rows), "maps": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
