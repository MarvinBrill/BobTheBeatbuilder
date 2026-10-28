"""Audit development audio with compact spectral hints; reserved holdouts are never opened."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
import numpy as np
from app.ml.corpus_audit import acoustic_signature, compare_acoustic_signatures
from app.ml.split import song_groups

SCHEMA = "development-acoustic-inventory-v1"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def development_audio(records, inventory, reservation):
    if reservation.get("schema") != "song-reservation-v1":
        raise ValueError("An explicit versioned holdout reservation is required")
    if inventory.get("reservation") != reservation:
        raise ValueError("Inventory and holdout reservation differ; refuse audio analysis")
    rows = inventory["rows"]
    snapshot = {row["map_id"]: row for row in rows}
    development = set(reservation["development_map_ids"])
    held_ids = set(reservation["final_test_map_ids"])
    held_groups = set(reservation["final_test_groups"])
    if development & held_ids:
        raise ValueError("Development and held-out map IDs overlap")
    # Include the frozen inventory in the union: removed/renamed catalog records
    # must not hide a metadata-linked encoding of a reserved song.
    combined = records + rows
    groups = song_groups(combined)
    protected = {
        group
        for row, group in zip(combined, groups)
        if row.get("map_id", row.get("id")) in held_ids or row.get("song_group") in held_groups
    }
    protected_hashes = {
        row["audio_hash"] for row, group in zip(combined, groups) if group in protected
    }
    chosen, excluded, changed = {}, set(), []
    for record, group in zip(records, groups):
        identifier = record.get("map_id", record.get("id"))
        if group in protected or record["audio_hash"] in protected_hashes:
            excluded.add(record["audio_hash"])
            continue
        if identifier not in development or identifier not in snapshot:
            continue
        if record["audio_hash"] != snapshot[identifier]["audio_hash"]:
            changed.append(identifier)
            continue
        entry = chosen.setdefault(
            record["audio_hash"],
            {
                "audio_hash": record["audio_hash"],
                "map_id": identifier,
                "audio_path": record["audio_path"],
                "map_ids": [],
                "song_groups": [],
            },
        )
        entry["map_ids"].append(identifier)
        entry["song_groups"].append(snapshot[identifier]["song_group"])
    for entry in chosen.values():
        entry["map_ids"] = sorted(set(entry["map_ids"]))
        entry["song_groups"] = sorted(set(entry["song_groups"]))
    return sorted(chosen.values(), key=lambda item: item["audio_hash"]), {
        "excluded_holdout_audio_hashes": sorted(excluded),
        "changed_catalog_map_ids_not_analysed": sorted(changed),
        "reserved_groups": len(held_groups),
    }


def valid_mel(path):
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            values.ndim == 2
            and values.shape[1] == 64
            and values.dtype == np.float32
            and 0 < len(values) <= 1200 * 22050 // 256 + 1
            and np.isfinite(values).all()
            and ((values >= -1) & (values <= 1)).all()
        ):
            return values
    except (OSError, ValueError, EOFError):
        pass
    return None


def observe_audio(item, data, cache_roots, feature_loader):
    started = time.perf_counter()
    path = item["audio_path"]
    path = data / path[7:] if path.startswith("data://") else Path(path)
    result = {**item, "audio_path": str(path)}
    try:
        result["source_audio_sha256"] = digest(path)
        if result["source_audio_sha256"] != item["audio_hash"]:
            raise ValueError("Audio content no longer matches the reserved inventory hash")
        mel, used_cache, invalid = None, None, []
        for root in cache_roots:
            candidate = root / f"mel-v2-{item['audio_hash']}.npy"
            if candidate.is_file():
                mel = valid_mel(candidate)
                if mel is not None:
                    used_cache = candidate
                    break
                invalid.append(str(candidate))
        result["invalid_caches_ignored"] = invalid
        if mel is None:
            mel = feature_loader(path, item["audio_hash"])
            result["feature_source"] = "isolated_recompute"
            if getattr(mel, "filename", None):
                used_cache = Path(mel.filename)
        else:
            result["feature_source"] = "existing_cache"
        vector = acoustic_signature(mel)
        result.update(frames=len(mel), duration_seconds=len(mel) * 256 / 22050)
        if used_cache is not None:
            result.update(cache_path=str(used_cache), cache_sha256=digest(used_cache))
        if vector is None:
            result.update(status="uninformative_signature", signature=None)
        else:
            result.update(status="analysed", signature=vector.tolist())
        del mel
    except (OSError, ValueError, RuntimeError, EOFError, subprocess.SubprocessError) as error:
        result.update(status="error", error=str(error), signature=None)
    result["seconds"] = time.perf_counter() - started
    return result


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--study", type=Path, default=ROOT / "data/benchmarks/architecture")
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2, 3, 4))
    args = parser.parse_args()
    data, study = args.data.resolve(), args.study.resolve()
    reservation_path = study / "holdout-reservation.json"
    inventory_path = study / "corpus-inventory.json"
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    uri = (data / "bob.sqlite3").as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        records = [
            json.loads(row[0])
            for row in connection.execute("SELECT body FROM records WHERE kind='map' ORDER BY id")
        ]
    chosen, exclusions = development_audio(records, inventory, reservation)
    runtime = study / "acoustic-runtime"
    if runtime == data:
        raise ValueError("Benchmark runtime must differ from the production data directory")
    os.environ["BOB_DATA"] = str(runtime)
    os.environ["NUMBA_CACHE_DIR"] = str(runtime / "cache/numba")
    os.environ["MPLCONFIGDIR"] = str(runtime / "cache/matplotlib")
    os.environ["TMPDIR"] = str(runtime / "tmp")
    from app.ml.audio import features
    from app.config import DATA

    if DATA != runtime:
        raise ValueError(
            "Audio module is already bound to another data directory; run this CLI separately"
        )

    report = dict(
        schema=SCHEMA,
        status="running",
        **exclusions,
        inventory_sha256=digest(inventory_path),
        reservation_sha256=digest(reservation_path),
        runner_sha256=digest(Path(__file__)),
        acoustic_code_sha256=digest(ROOT / "backend/app/ml/corpus_audit.py"),
        audio_code_sha256=digest(ROOT / "backend/app/ml/audio.py"),
        catalog_records_sha256=hashlib.sha256(
            json.dumps(records, sort_keys=True).encode()
        ).hexdigest(),
        eligible_unique_audio=len(chosen),
        catalog_unique_audio=len({row["audio_hash"] for row in records}),
        workers=args.workers,
        thresholds={
            "cosine_similarity_min": 0.995,
            "relative_duration_difference_max_exclusive": 0.01,
        },
        rows=[],
        candidates=[],
        notes=[
            "Only explicitly reserved development IDs; entire connected holdout groups excluded before audio/cache reads.",
            "Spectral 32x8 similarity gives manual-review hints, not acoustic duplicate proof or genre labels.",
            "No automatic hash merge, split mutation, training, model inference or production-cache writes.",
        ],
    )
    started = time.perf_counter()
    report_path = study / "acoustic-inventory.json"
    write_report(report_path, report)
    cache_roots = [data / "cache", runtime / "cache"]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(
            executor.map(lambda item: observe_audio(item, data, cache_roots, features), chosen), 1
        ):
            report["rows"].append(result)
            if index % 20 == 0 or index == len(chosen):
                report["seconds"] = time.perf_counter() - started
                write_report(report_path, report)
                print(
                    f"Acoustic development audit {index}/{len(chosen)}: {report['seconds']:.1f}s",
                    flush=True,
                )
    fingerprints = [
        (row, row["frames"], np.asarray(row["signature"], dtype=np.float32))
        for row in report["rows"]
        if row["status"] == "analysed"
    ]
    report["candidates"] = compare_acoustic_signatures(fingerprints)
    report.update(
        status="complete",
        seconds=time.perf_counter() - started,
        analysed_unique_audio=len(fingerprints),
        cache_hits=sum(row.get("feature_source") == "existing_cache" for row in report["rows"]),
        recomputed=sum(row.get("feature_source") == "isolated_recompute" for row in report["rows"]),
        errors=[
            {key: row[key] for key in ("map_id", "audio_hash", "error")}
            for row in report["rows"]
            if row["status"] == "error"
        ],
    )
    write_report(report_path, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "eligible_unique_audio",
                    "analysed_unique_audio",
                    "cache_hits",
                    "recomputed",
                    "seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
