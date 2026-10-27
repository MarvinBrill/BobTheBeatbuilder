"""Bounded A/B/C regression on an existing cached corpus; never publishes user models.

Run from the repository root with its Python environment. All output stays in data/.
The held-out songs are used once per selected model, never for epoch selection.
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "data" / "benchmarks" / "training-policy"
os.environ["BOB_DATA"] = str(DIRECTORY / "runtime")
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
import torch
from app.ml.sequence_batches import batches, tasks_for
from app.ml.training_setup import configuration, split_maps
from app.ml.phrase_model import PhraseMapper
from app.ml.phrase_losses import parts, objective, LossMeter
from app.ml.training import evaluate
from app.ml.training_control import TrainingControl, context_probability
from app.ml.training_runtime import TrainingRuntime
from app.ml.cuda_graph import CudaTrainingGraph
from app.ml.batch_loading import device_batches
from app.ml.rollout_evaluation import evaluate_suite


class Context:
    def check(self):
        pass

    def report(self, *args, **kwargs):
        pass


def corpus():
    items = json.loads((ROOT / "data/benchmarks/cuda/prepared.json").read_text())
    for item in items:
        for key in ("mel", "events", "labels", "summary"):
            item[key] = np.load(item[key], mmap_mode="r", allow_pickle=False)
        item["condition"] = np.array(item["condition"], np.float32)
    return items


def run(mode, seed, maps, size, epochs):
    training, validation, test = split_maps(maps, 42)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    selected = torch.device("cuda")
    model = PhraseMapper(configuration(size)).to(selected)
    runtime = TrainingRuntime(selected)
    optimizer = runtime.optimizer(model, 0.0003, 0.01)
    tasks = tasks_for(training)
    batch_count = math.ceil(len(tasks) / 8)
    weight = float(
        np.clip(
            sum(m["labels"][:, 4].sum() for m in training)
            / max(1, 2 * sum(m["labels"][:, 0].sum() for m in training))
            - 1,
            1,
            40,
        )
    )
    control = TrainingControl(0.0003, 12, min(200, batch_count))
    captured, best_weights, history = None, None, []
    best, stale, updates = math.inf, 0, 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        meter = LossMeter()
        start = runtime.begin_measurement()
        count = 0
        with device_batches(batches(training, 8, rng, True, tasks), selected) as loaded:
            for batch in loaded:
                probability = context_probability(epoch, mode == "plateau_context")
                if probability:
                    batch["context_probability"] = torch.full((1,), probability, device=selected)
                if captured is None or captured.static.keys() != batch.keys():
                    captured = CudaTrainingGraph(model, runtime, batch, weight)
                if captured.matches(batch):
                    loss, terms = captured.replay(batch)
                else:
                    terms = parts(model, runtime.forward(model, batch), batch, weight)
                    loss = objective(terms)
                    loss.backward()
                runtime.clip_and_check(model, torch.isfinite(loss))
                if mode == "original":
                    total = 100 * batch_count
                    warmup = min(200, max(1, round(total * 0.05)))
                    rate = (
                        0.0003
                        * min(1, (updates + 1) / warmup)
                        * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * updates / total)))
                    )
                else:
                    rate = control.step_rate()
                for group in optimizer.param_groups:
                    group["lr"] = rate
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                control.updates += 1
                count += len(batch["audio"])
                meter.add(terms)
        performance = runtime.measurement(start, count, batch_count)
        loss, quality = evaluate(model, validation, 8, selected, Context(), weight)
        if mode == "original":
            improved = loss < best
            stale = 0 if improved else stale + 1
            stop = stale >= 12
        else:
            improved, _, stop = control.observe(loss)
        if improved:
            best = loss
            best_epoch = epoch
            best_weights = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        history.append(
            dict(
                epoch=epoch,
                validation_loss=loss,
                learning_rate=rate,
                performance=performance,
                **quality,
            )
        )
        if epoch % 4 == 0 or stop:
            print(f"{mode} seed={seed} epoch={epoch} val={loss:.4f} best={best:.4f}", flush=True)
        if stop:
            break
    model.load_state_dict(best_weights)
    test_loss, quality = evaluate(model, test, 8, selected, Context(), weight)
    suite = evaluate_suite(model, test, selected, Context())
    result = dict(
        mode=mode,
        seed=seed,
        size=size,
        best_epoch=best_epoch,
        best_validation_loss=best,
        epochs=len(history),
        seconds=time.perf_counter() - started,
        split=dict(training=len(training), validation=len(validation), test=len(test)),
        test=dict(loss=test_loss, **quality),
        free_test=suite,
        history=history,
    )
    del captured, model, optimizer, best_weights
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["compact", "standard", "large"], default="large")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seeds", default="42,73")
    args = parser.parse_args()
    torch.set_num_threads(4)
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    maps = corpus()
    report = dict(
        gpu=torch.cuda.get_device_name(0),
        maps=len(maps),
        results=[],
        limitation="Small cached corpus, two seeds; not a proof of human playability or full-corpus non-regression.",
    )
    for seed in map(int, args.seeds.split(",")):
        for mode in ("original", "plateau", "plateau_context"):
            report["results"].append(run(mode, seed, maps, args.size, args.epochs))
            (DIRECTORY / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
            print(f"FINISHED {mode} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
