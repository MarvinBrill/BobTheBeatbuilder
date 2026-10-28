"""Inventory local catalog read-only and reserve unrun song groups for final testing."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
import numpy as np
from app.ml.corpus_audit import inventory, assert_disjoint_groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "data/benchmarks/architecture")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scan-maps",
        action="store_true",
        help="Inspect every .osu file; does not decode audio or run a model.",
    )
    args = parser.parse_args()
    uri = (args.data / "bob.sqlite3").resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:

        def records(kind):
            return [
                json.loads(row[0])
                for row in connection.execute("SELECT body FROM records WHERE kind=?", (kind,))
            ]

        maps, datasets = records("map"), records("dataset")
    models = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.data / "models").glob("*/metadata.json")
    ]
    if args.scan_maps:
        from app.ml.corpus_map_features import inspect_map

        def observe(record):
            path = record["map_path"]
            if path.startswith("data://"):
                path = args.data / path[7:]
            try:
                return {**record, "map_observations": inspect_map(path)}
            except (ValueError, OSError, IndexError) as error:
                return {**record, "map_observations": {"error": str(error)}}

        with ThreadPoolExecutor(max_workers=4) as executor:
            maps = list(executor.map(observe, maps))
    prepared = []
    baseline = args.output / "baseline.json"
    if baseline.exists():
        for source in json.loads(baseline.read_text(encoding="utf-8"))["items"]:
            item = dict(source)
            for name in ("events", "mel"):
                item[name] = np.load(item[name], mmap_mode="r", allow_pickle=False)
            prepared.append(item)
    issues = [issue for model in models for issue in model.get("excluded_maps", [])]
    report = inventory(maps, models, datasets, prepared, issues, args.seed)
    reservation = report["reservation"]
    assert_disjoint_groups(
        maps,
        {
            "test": reservation["final_test_map_ids"],
            "development": reservation["development_map_ids"],
        },
    )
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "holdout-reservation.json"
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous != reservation:
            raise ValueError(
                "Existing holdout reservation differs. Use a new explicit study directory; never silently reshuffle the final test."
            )
    else:
        destination.write_text(json.dumps(reservation, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "corpus-inventory.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "maps",
                    "song_groups",
                    "unique_audio_hashes",
                    "prior_exposed_groups",
                    "prepared_maps",
                )
            }
        )
    )
    print(json.dumps({key: reservation[key] for key in ("status", "unresolved_historical_models")}))
    print(
        f"Reserved {len(reservation['final_test_groups'])} groups / {len(reservation['final_test_map_ids'])} maps; no final-test audio evaluated."
    )


if __name__ == "__main__":
    main()
