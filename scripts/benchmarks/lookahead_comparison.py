"""Compare depth-two search with local selection on frozen v3 code and identical weights."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    source = directory / "inference-lookahead-source"
    os.environ["BOB_DATA"] = str(directory / "lookahead-runtime")
    sys.path.insert(0, str(source / "backend"))
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from app.beatmaps import write, parse
    from app.ml.phrase_model import create_model
    from app.ml.sequence_generation import generate_objects

    class Context:
        def check(self):
            pass

        def report(self, *_args, **_kwargs):
            pass

    baseline = json.loads((directory / "baseline.json").read_text())
    weights = directory.parents[1] / "models" / baseline["model_id"] / "model.safetensors"
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == baseline["model_sha256"]
    config = baseline["metadata"]
    torch.set_num_threads(4)
    model = create_model(config["architecture"], config["model_config"])
    model.load_state_dict(load_file(str(weights)))
    model.to("cuda").eval()
    report = dict(
        schema="lookahead-comparison-v1",
        model_sha256=baseline["model_sha256"],
        source_hashes={
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source.rglob("*.py")
        },
        notes=[
            "Same v3 weights, full validation song, two seeds.",
            "No human quality labels. Shared host load can affect wall time.",
        ],
        runs=[],
    )
    report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    first_run = next(run for run in baseline["runs"] if run["scope"] == "full")
    first_item = next(item for item in baseline["items"] if item["map_id"] == first_run["map_id"])
    warmup_started = time.perf_counter()
    generate_objects(
        model,
        np.load(first_item["mel"], mmap_mode="r", allow_pickle=False),
        first_item["audio_path"],
        first_run["settings"],
        torch.tensor([first_item["condition"]], device="cuda"),
        torch.device("cuda"),
        Context(),
        max_objects=1,
    )
    report["warmup_seconds"] = time.perf_counter() - warmup_started
    for original in (run for run in baseline["runs"] if run["scope"] == "full"):
        item = next(item for item in baseline["items"] if item["map_id"] == original["map_id"])
        mel = np.load(item["mel"], mmap_mode="r", allow_pickle=False)
        # Alternate order by seed to expose rather than hide first-run/warmup costs.
        for enabled in [False, True] if original["seed"] == 42 else [True, False]:
            settings = {**original["settings"], "lookahead": enabled}
            started = time.perf_counter()
            objects, timing, quality = generate_objects(
                model,
                mel,
                item["audio_path"],
                settings,
                torch.tensor([item["condition"]], device="cuda"),
                torch.device("cuda"),
                Context(),
            )
            elapsed = time.perf_counter() - started
            path = directory / f"{item['map_id']}-{original['seed']}-lookahead-{int(enabled)}.osu"
            write(path, objects, settings, "audio.ogg", 60000 / timing[0][1], timing[0][0], timing)
            reparsed = parse(path).objects
            assert len(objects) == len(reparsed)
            assert all(
                a.repeats == b.repeats
                and a.slider == b.slider
                and abs(a.duration - b.duration) < 2
                and np.allclose([a.end_x, a.end_y], [b.end_x, b.end_y], atol=0.75)
                for a, b in zip(objects, reparsed)
            )
            path.with_suffix(".json").write_text(
                json.dumps([asdict(obj) for obj in objects], allow_nan=False)
            )
            report["runs"].append(
                dict(
                    map_id=item["map_id"],
                    seed=original["seed"],
                    lookahead=enabled,
                    seconds=elapsed,
                    diagnostics=quality,
                    export_path=str(path),
                    unique_heads=len({(obj.x, obj.y) for obj in objects}),
                )
            )
            (directory / "lookahead-comparison.json").write_text(
                json.dumps(report, indent=2, allow_nan=False)
            )
            print(original["seed"], enabled, len(objects), round(elapsed, 3), flush=True)
    report["diversity"] = []
    for enabled in (False, True):
        paired = [run for run in report["runs"] if run["lookahead"] == enabled]
        if len(paired) != 2:
            continue
        maps = [
            {
                obj["time"]: obj
                for obj in json.loads(Path(run["export_path"]).with_suffix(".json").read_text())
            }
            for run in paired
        ]
        shared = sorted(set(maps[0]) & set(maps[1]))
        distances = [
            np.hypot(
                maps[0][time]["x"] - maps[1][time]["x"], maps[0][time]["y"] - maps[1][time]["y"]
            )
            for time in shared
        ]
        report["diversity"].append(
            dict(
                lookahead=enabled,
                shared_head_count=len(shared),
                mean_head_distance_between_seeds=float(np.mean(distances)) if shared else None,
                changed_head_fraction=float(np.mean(np.asarray(distances) > 1)) if shared else None,
                type_change_fraction=(
                    float(
                        np.mean(
                            [maps[0][time]["slider"] != maps[1][time]["slider"] for time in shared]
                        )
                    )
                    if shared
                    else None
                ),
            )
        )
    (directory / "lookahead-comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
