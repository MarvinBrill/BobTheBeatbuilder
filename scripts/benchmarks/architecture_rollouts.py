"""Extend fixed C-checkpoint evaluation with all validation probes and one full song.

Imports are taken exclusively from the training source snapshot. Existing exact
section evaluations can be reused; their original runtime is retained separately
from the wall time of this extension. No checkpoint is trained or registered.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(path, report, started):
    report["elapsed_seconds"] = time.perf_counter() - started
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


def section_key(map_id, start, end, limit):
    return map_id, round(start, 6), round(end, 6), limit


class Context:
    def check(self):
        pass

    def report(self, _progress, message, **kwargs):
        if message.startswith("Preparing"):
            print(message, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training", type=Path, default=ROOT / "data/benchmarks/architecture/training"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/benchmarks/architecture/c-training-records.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--rerun-existing", action="store_true")
    args = parser.parse_args()
    directory = args.training.resolve()
    source = directory / "source"
    training_report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if training_report["schema"] != "architecture-training-ablation-v1":
        raise ValueError("Unsupported training report")
    for name, expected in training_report["code_hashes"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    if digest(args.manifest) != training_report["corpus_sha256"]:
        raise ValueError("Training corpus manifest changed")
    helper = "architecture_experiment_data.py"
    if digest(source / helper) != digest(Path(__file__).with_name(helper)):
        raise ValueError("Preparation helper differs from the frozen training helper")

    os.environ["BOB_DATA"] = str(directory / "runtime")
    sys.path.insert(0, str(source))
    import torch
    from safetensors.torch import load_file
    from app.ml.architecture import ModelConfig
    from app.ml.audio import FRAME_MS
    from app.ml.phrase_model import PhraseMapper
    from app.ml.rollout_evaluation import evaluate_rollout, rollout_songs
    from app.ml.rollout_sections import section_plan, MAX_FULL_SECONDS, MAX_FULL_OBJECTS
    from architecture_experiment_data import prepare_manifest

    torch.set_num_threads(4)
    started = time.perf_counter()
    _, validation, _ = prepare_manifest(args.manifest, Context())
    selected = torch.device(args.device)
    songs = rollout_songs(validation, len(validation))
    planned = [(item, section, 128) for item in songs for section in section_plan(item["mel"])]
    eligible = [
        item for item in songs if (len(item["mel"]) - 1) * FRAME_MS <= MAX_FULL_SECONDS * 1000
    ]
    if eligible:
        planned.append((min(eligible, key=lambda item: len(item["mel"])), None, MAX_FULL_OBJECTS))
    plan = [
        dict(
            map_id=item["map_id"],
            song=item["song"],
            max_objects=limit,
            start_ms=section["start_ms"] if section else 0.0,
            end_ms=section["end_ms"] if section else (len(item["mel"]) - 1) * FRAME_MS,
            probes=section["probes"] if section else ["full_song"],
        )
        for item, section, limit in planned
    ]
    report = dict(
        schema="architecture-extended-rollouts-v1",
        status="running",
        training_report_sha256=digest(directory / "report.json"),
        corpus_sha256=training_report["corpus_sha256"],
        code_hashes=training_report["code_hashes"],
        runner_sha256=digest(Path(__file__)),
        preparation_helper_sha256=digest(source / helper),
        final_test_status=training_report["final_test_status"],
        validation_songs=len(songs),
        plan=plan,
        results=[],
        settings=dict(
            temperature=0.8,
            candidates=12,
            snap=True,
            lookahead=False,
            motif_memory=False,
            condition="unchanged reference condition",
        ),
        note="Development probes, not a fresh final test. Existing matching section results are explicitly marked. Fixed coverage is shared by all arms; activity probes are not semantic chorus labels. Reference agreement and repair counts do not establish musical quality.",
    )
    output = directory / "extended-rollouts.json"
    if output.exists():
        raise ValueError("Extended report already exists; preserve it before another run")
    write_report(output, report, started)
    print(f"Planned {len(planned)} evaluations per checkpoint, {len(songs)} songs", flush=True)
    for trained in training_report["results"]:
        arm, seed = trained["arm"], trained["seed"]
        weights = directory / f"{arm}-{seed}" / "model.safetensors"
        if digest(weights) != trained["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed: {arm}-{seed}")
        individual = json.loads(weights.with_name("report.json").read_text(encoding="utf-8"))
        if individual != trained:
            raise ValueError(f"Individual and aggregate report differ: {arm}-{seed}")
        model = PhraseMapper(ModelConfig(**trained["model_config"])).to(selected)
        model.load_state_dict(load_file(str(weights), device=str(selected)), strict=True)
        model.eval()
        existing = {
            section_key(row["map_id"], row["start_ms"], row["end_ms"], row["max_objects"]): row
            for row in trained["free"]
            if row["seed"] == seed
        }
        arm_result = dict(
            arm=arm,
            seed=seed,
            checkpoint_sha256=trained["checkpoint_sha256"],
            best_epoch=trained["best_epoch"],
            model_config=trained["model_config"],
            free=[],
            failures=[],
            reused=0,
            measured=0,
        )
        report["results"].append(arm_result)
        for index, ((item, section, limit), scope) in enumerate(zip(planned, plan), 1):
            key = section_key(item["map_id"], scope["start_ms"], scope["end_ms"], limit)
            if key in existing and not args.rerun_existing:
                row = dict(existing[key], provenance="reused_training_report")
                arm_result["reused"] += 1
            else:
                try:
                    row = evaluate_rollout(
                        model,
                        item,
                        selected,
                        Context(),
                        max_objects=limit,
                        section=section,
                        seed=seed,
                    )
                    row["provenance"] = "measured_extension"
                    arm_result["measured"] += 1
                except ValueError as error:
                    arm_result["failures"].append(dict(**scope, reason=str(error)))
                    write_report(output, report, started)
                    continue
            arm_result["free"].append(row)
            write_report(output, report, started)
            print(
                f"{arm}-{seed} {index}/{len(planned)} {scope['probes']} "
                f"objects={row['generated_objects']} F1={row['reference_onset_f1']:.4f} "
                f"{row['provenance']}",
                flush=True,
            )
        del model
        gc.collect()
        if selected.type == "cuda":
            torch.cuda.empty_cache()
    report["status"] = "complete"
    write_report(output, report, started)
    print(f"Complete: {output}; {report['elapsed_seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
