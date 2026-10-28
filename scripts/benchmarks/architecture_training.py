"""Equal-capacity, equal-update architecture ablations on development songs only.

Historical critic supervision exists here only as an experiment, never as a
production model loader or compatibility path. The final test remains untouched.
"""

import argparse
from dataclasses import replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(os.environ.get("BOB_ARCHITECTURE_EXPERIMENT_DIR", ROOT / "data/benchmarks/architecture/training")).resolve()
os.environ["BOB_DATA"] = str(DIRECTORY / "runtime")
SOURCE = DIRECTORY / "source"
# Imports run from a benchmark snapshot: later working-tree edits cannot alter
# a long-running ablation midway, and a new session can inspect the exact source.
if "--help" not in sys.argv:
    if not (SOURCE / "app").exists():
        SOURCE.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT / "backend/app", SOURCE / "app", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(Path(__file__), SOURCE / "architecture_training.py")
        shutil.copyfile(Path(__file__).with_name("architecture_experiment_data.py"), SOURCE / "architecture_experiment_data.py")
    sys.path.insert(0, str(SOURCE))
else:
    sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
import torch
from safetensors.torch import save_file
from app.ml.architecture import ARCHITECTURE, EVENT_SCHEMA, TARGET_SCHEMA
from app.ml.training_setup import configuration
from app.ml.phrase_model import PhraseMapper
from app.ml.phrase_losses import parts, objective, LossMeter, binary_loss
from app.ml.movement import motion_tensor
from app.ml.sequence_batches import batches, tasks_for
from app.ml.batch_loading import device_batches
from app.ml.training_runtime import TrainingRuntime
from app.ml.training_control import TrainingControl, context_probability
from app.ml.training import evaluate
from app.ml.rollout_evaluation import evaluate_rollout, rollout_songs
from app.ml.rollout_sections import section_plan
from architecture_experiment_data import prepare_manifest


ARMS = {
    "seconds": dict(enable_musical_time=False, enable_plan=False),
    "time": dict(enable_musical_time=True, enable_plan=False),
    "plan": dict(enable_musical_time=True, enable_plan=True),
    "critic": dict(enable_musical_time=True, enable_plan=True),
    "context": dict(enable_musical_time=True, enable_plan=True),
}
MAIN_KEYS = ("onset", "beat", "density", "tempo", "geometry", "kind", "duration", "repeats", "combo", "motion", "flow", "compatibility")


def source_hash():
    digest = hashlib.sha256()
    for path in sorted((SOURCE / "app").rglob("*.py")):
        digest.update(str(path.relative_to(SOURCE)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Context:
    def check(self):
        pass

    def report(self, _progress, message, **kwargs):
        if message.startswith("Preparing"):
            print(message, flush=True)


def historical_critic(model, output, batch):
    geometry = torch.cat([batch["geometry_target"][:, -7:], batch["geometry_target"][:, :-7]], 1)
    motion = motion_tensor(geometry, batch["seconds_target"], batch["repeat_target"] + 1,
                           batch["kind_target"], batch["event_times"], batch["motion_context"])
    positive = model.compatibility(output["hidden"], batch["geometry_target"], batch["motion_target"])
    negative = model.compatibility(output["hidden"], geometry, motion)
    distinct = ((geometry - batch["geometry_target"]).square().sum(-1) > .02).to(geometry.dtype)
    valid = batch["event_valid"] * batch["geometry_valid"]
    donor_valid = torch.cat([valid[:, -7:], valid[:, :-7]], 1)
    mask = batch["event_mask"] * distinct * batch["geometry_valid"] * donor_valid
    return ((binary_loss(positive - negative, torch.ones_like(positive)) * mask).sum() * .1, mask.sum())


def train_arm(arm, seed, training, validation, args, corpus_hash):
    selected = torch.device(args.device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = replace(configuration(args.size), **ARMS[arm])
    model = PhraseMapper(cfg).to(selected)
    runtime = TrainingRuntime(selected, "auto")
    optimizer = runtime.optimizer(model, .0003, .01)
    tasks = tasks_for(training)
    batch_count = math.ceil(len(tasks) / args.batch_size)
    control = TrainingControl(.0003, 1000, min(200, batch_count))
    weight = float(np.clip(sum(item["labels"][:, 4].sum() for item in training) /
                           max(1, 2 * sum(item["labels"][:, 0].sum() for item in training)) - 1, 1, 40))
    best, best_weights, history, updates = math.inf, None, [], 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        meter = LossMeter()
        epoch_started = runtime.begin_measurement()
        count = 0
        with device_batches(batches(training, args.batch_size, rng, True, tasks), selected) as source:
            for batch in source:
                probability = context_probability(epoch, arm == "context")
                if probability:
                    batch["context_probability"] = torch.tensor([probability], device=selected)
                optimizer.zero_grad(set_to_none=True)
                output = runtime.forward(model, batch)
                terms = parts(model, output, batch, weight)
                if arm != "critic":
                    terms["compatibility"] = historical_critic(model, output, batch)
                loss = objective(terms)
                loss.backward()
                runtime.clip_and_check(model, torch.isfinite(loss))
                for group in optimizer.param_groups:
                    group["lr"] = control.step_rate()
                optimizer.step()
                control.updates += 1
                updates += 1
                count += len(batch["audio"])
                meter.add(terms)
        performance = runtime.measurement(epoch_started, count, batch_count)
        loss, quality = evaluate(model, validation, args.batch_size, selected, Context(), weight)
        # All arms use the same current twelve-term validation criterion. Auxiliary
        # plan/context losses are reported separately and never select a checkpoint.
        if loss < best:
            best, best_epoch = loss, epoch
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        control.observe(loss)
        history.append(dict(epoch=epoch, validation_loss=loss, train_loss=meter.value(),
                            validation_without_critic=sum(quality["loss_terms"][key] for key in MAIN_KEYS if key != "compatibility"),
                            train_terms=meter.values(), performance=performance, **quality))
        print(f"{arm} seed={seed} epoch={epoch}/{args.epochs} validation={loss:.5f}", flush=True)
    model.load_state_dict(best_weights)
    directory = DIRECTORY / f"{arm}-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    save_file(best_weights, str(directory / "model.safetensors"))
    checkpoint_hash = hashlib.sha256((directory / "model.safetensors").read_bytes()).hexdigest()
    free, failures = [], []
    for item in rollout_songs(validation, 3):
        for section in section_plan(item["mel"])[:3]:
            try:
                free.append(evaluate_rollout(model, item, selected, Context(), max_objects=128, section=section, seed=seed))
            except ValueError as error:
                failures.append(dict(map_id=item["map_id"], section=section, reason=str(error)))
    result = dict(arm=arm, seed=seed, model_config=cfg.dictionary(), architecture=ARCHITECTURE,
                  event_schema=EVENT_SCHEMA, target_schema=TARGET_SCHEMA, corpus_sha256=corpus_hash,
                  checkpoint_sha256=checkpoint_hash, parameters=sum(p.numel() for p in model.parameters()),
                  source_sha256=source_hash(), runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  epochs=args.epochs, batch_size=args.batch_size, learning_rate=.0003, weight_decay=.01,
                  updates=updates, best_epoch=best_epoch, best_validation_loss=best,
                  seconds=time.perf_counter() - started, history=history, free=free, failures=failures,
                  context_supervision="structure_only", validation_criterion="same_current_twelve_main_losses",
                  training_critic="matched_soft_counterfactuals" if arm == "critic" else "historical_shift7_benchmark_only")
    (directory / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    del model, optimizer, best_weights
    gc.collect()
    if selected.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/benchmarks/architecture/c-training-records.json")
    parser.add_argument("--size", choices=("compact", "standard", "large"), default="large")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seeds", default="42,73")
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Positive epochs and batch size required")
    arms = args.arms.split(",")
    if any(arm not in ARMS for arm in arms):
        parser.error("Unknown experiment arm")
    torch.set_num_threads(4)
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    training, validation, corpus_hash = prepare_manifest(args.manifest, Context())
    report = dict(schema="architecture-training-ablation-v1", corpus_sha256=corpus_hash,
                  training_groups=len({item["song"] for item in training}),
                  validation_groups=len({item["song"] for item in validation}),
                  final_test_status="unavailable_all_groups_previously_exposed", results=[],
                  comparisons={"time": "seconds", "plan": "time", "critic": "plan", "context": "plan"},
                  code_hashes={str(path.relative_to(SOURCE)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted((SOURCE / "app").rglob("*.py"))},
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                  note="Exploratory fixed-budget development comparison. C arms share initial weights, groups, capacity and update budget per seed. A/B pretrained v3 has a different training budget and is not a fair quality ranking against C. No human ratings.")
    for seed in map(int, args.seeds.split(",")):
        for arm in arms:
            saved = DIRECTORY / f"{arm}-{seed}" / "report.json"
            if saved.exists():
                result = json.loads(saved.read_text(encoding="utf-8"))
                if result["corpus_sha256"] != corpus_hash or result["epochs"] != args.epochs or result["batch_size"] != args.batch_size:
                    raise ValueError("Existing experiment has a different data/update budget; use a separate output directory")
                expected_cfg = replace(configuration(args.size), **ARMS[arm]).dictionary()
                if result["model_config"] != expected_cfg or result.get("source_sha256") != source_hash() or result.get("runner_sha256") != report["runner_sha256"]:
                    raise ValueError("Existing experiment has different code/model settings; select a new BOB_ARCHITECTURE_EXPERIMENT_DIR")
                weights_path = saved.with_name("model.safetensors")
                if result["checkpoint_sha256"] != hashlib.sha256(weights_path.read_bytes()).hexdigest():
                    raise ValueError("Saved experiment weights changed")
                print(f"Reusing completed {arm} seed={seed}", flush=True)
            else:
                result = train_arm(arm, seed, training, validation, args, corpus_hash)
            report["results"].append(result)
            (DIRECTORY / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
