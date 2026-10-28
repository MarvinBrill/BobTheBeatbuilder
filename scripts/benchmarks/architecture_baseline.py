"""Capture generation evidence with frozen sources, explicit weights and prepared data.

Prepared JSON: schema=architecture-prepared-v1, event_schema=current model schema,
items=[{map_id,song,map_hash,audio_hash,audio_path,mel,events,labels,summary,condition}].
Array paths refer to verified .npy files. This is a benchmark snapshot, not a legacy
model loader or a production inference branch.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class Context:
    def check(self):
        pass

    def report(self, *args, **kwargs):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seeds", default="42,73")
    args = parser.parse_args()
    metadata = json.loads((args.model_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads(args.prepared.read_text(encoding="utf-8"))
    if manifest.get("schema") != "architecture-prepared-v1":
        raise ValueError(
            "Expected architecture-prepared-v1 manifest; rebuild prepared data for this experiment"
        )
    if manifest.get("event_schema") != metadata.get("event_schema"):
        raise ValueError(
            "Prepared targets and model schema differ; prepare data and retrain together"
        )
    if args.output.exists():
        raise ValueError("Use a fresh output directory; an existing baseline must remain immutable")
    args.output.mkdir(parents=True)
    snapshot = args.output / "source/backend/app"
    for source in (ROOT / "backend/app").rglob("*.py"):
        target = snapshot / source.relative_to(ROOT / "backend/app")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    os.environ["BOB_DATA"] = str(args.output / "runtime")
    sys.path.insert(0, str(snapshot.parent))
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from app.ml.architecture import EVENT_SCHEMA
    from app.ml.phrase_model import create_model
    from app.ml.rhythm_inference import predict_rhythm, timing_map, select_times
    from app.ml.sequence_generation import generate_objects
    from app.ml.rollout_evaluation import rollout_settings, rollout_songs, roundtrip_report
    from app.ml.rollout_sections import (
        section_plan,
        MAX_SECTION_OBJECTS,
        MAX_FULL_OBJECTS,
        MAX_FULL_SECONDS,
    )
    from app.ml.audio import FRAME_MS
    from app.beatmaps import write

    if metadata.get("event_schema") != EVENT_SCHEMA:
        raise ValueError(
            "This source snapshot requires current model/target schema; old weights require retraining"
        )
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = create_model(metadata["architecture"], metadata["model_config"])
    weights = args.model_dir / "model.safetensors"
    model.load_state_dict(load_file(str(weights)))
    model.to(device).eval()
    seeds = list(map(int, args.seeds.split(",")))
    report = dict(
        schema="architecture-baseline-v2",
        metadata=metadata,
        model_sha256=digest(weights),
        prepared_sha256=digest(args.prepared),
        seeds=seeds,
        device=str(device),
        python=platform.python_version(),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        code_hashes={
            str(path.relative_to(snapshot.parent)): digest(path) for path in snapshot.rglob("*.py")
        },
        items=[],
        runs=[],
        failures=[],
        note="Development evidence. Reference agreement, technical validity and human quality are separate.",
    )

    def save():
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )

    items = []
    for raw in manifest["items"]:
        item, recorded = dict(raw), dict(raw)
        for name in ("mel", "events", "labels", "summary"):
            recorded[name + "_sha256"] = digest(raw[name])
            item[name] = np.load(raw[name], mmap_mode="r", allow_pickle=False)
        item["condition"] = np.asarray(raw["condition"], np.float32)
        condition = torch.tensor(item["condition"][None], device=device)
        started = time.perf_counter()
        prediction = predict_rhythm(
            model, item["mel"], item["summary"], condition, device, Context()
        )
        settings = rollout_settings(item["condition"], seeds[0])
        timing, source = timing_map(prediction, item["audio_path"], settings)
        prediction_path = args.output / (item["map_id"] + "-rhythm.npy")
        np.save(prediction_path, prediction, allow_pickle=False)
        recorded.update(
            prediction_path=str(prediction_path),
            prediction_sha256=digest(prediction_path),
            timing=[asdict(point) for point in timing],
            timing_source=source,
            selected_times=select_times(prediction, item["mel"], settings, timing),
            seconds=time.perf_counter() - started,
        )
        report["items"].append(recorded)
        items.append(item)
        save()

    chosen = rollout_songs(items)
    planned = [
        (item, section, MAX_SECTION_OBJECTS)
        for item in chosen
        for section in section_plan(item["mel"])
    ]
    eligible = [
        item for item in chosen if (len(item["mel"]) - 1) * FRAME_MS <= MAX_FULL_SECONDS * 1000
    ]
    if eligible:
        item = eligible[0]
        planned.append(
            (
                item,
                dict(start_ms=0, end_ms=(len(item["mel"]) - 1) * FRAME_MS, probes=["full_song"]),
                MAX_FULL_OBJECTS,
            )
        )
    for seed in seeds:
        for index, (item, section, limit) in enumerate(planned):
            key = f"{item['map_id']}-{seed}-{index}"
            started = time.perf_counter()
            settings = rollout_settings(item["condition"], seed)
            try:
                objects, timing, diagnostics = generate_objects(
                    model,
                    item["mel"],
                    item["audio_path"],
                    settings,
                    torch.tensor(item["condition"][None], device=device),
                    device,
                    Context(),
                    max_objects=limit,
                    evaluation_range=(section["start_ms"], section["end_ms"]),
                )
                path = args.output / (key + ".osu")
                write(
                    path,
                    objects,
                    settings,
                    "audio.ogg",
                    60000 / timing[0].beat_length,
                    timing[0].time,
                    timing,
                )
                (args.output / (key + "-objects.json")).write_text(
                    json.dumps([asdict(obj) for obj in objects], allow_nan=False), encoding="utf-8"
                )
                report["runs"].append(
                    dict(
                        map_id=item["map_id"],
                        seed=seed,
                        section=section,
                        settings=settings,
                        diagnostics=diagnostics,
                        export_path=str(path),
                        export_sha256=digest(path),
                        **roundtrip_report(objects, timing, settings),
                        seconds=time.perf_counter() - started,
                    )
                )
            except ValueError as error:
                report["failures"].append(
                    dict(map_id=item["map_id"], seed=seed, section=section, reason=str(error))
                )
            save()
            print(key, flush=True)


if __name__ == "__main__":
    main()
