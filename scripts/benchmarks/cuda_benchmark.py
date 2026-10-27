import argparse
import json
import time
import subprocess
import threading
import statistics
import numpy as np
import torch
from corpus import DIRECTORY, load_corpus, Context
from runner import Runner, ReferenceMeter
from app.ml.phrase_losses import LossMeter, parts, objective
from app.ml.training_setup import split_maps, dataset_signature
from app.ml.rollout_evaluation import evaluate_rollout


class DeviceMonitor:
    def __enter__(self):
        self.samples = []
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
                "-lms",
                "200",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        def read():
            for line in self.process.stdout:
                try:
                    self.samples.append([float(value.strip()) for value in line.split(",")])
                except ValueError:
                    pass

        self.thread = threading.Thread(target=read, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.process.terminate()
        self.process.wait(timeout=5)
        self.thread.join(timeout=5)
        self.process.stdout.close()

    def result(self):
        return dict(
            gpu_utilization_mean=float(np.mean([item[0] for item in self.samples])) if self.samples else None,
            device_memory_peak_mib=max((item[1] for item in self.samples), default=0),
            gpu_power_mean_watts=float(np.mean([item[2] for item in self.samples])) if self.samples else None,
            samples=len(self.samples),
        )


def save(name, result):
    path = DIRECTORY / f"{name}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


def throughput(training, updates, stage="throughput"):
    results = []
    configurations = [
        ("baseline", 2, 4),
        ("fp32", 2, 4),
        ("bf16", 2, 4),
        ("bf16", 4, 2),
        ("bf16", 8, 1),
        ("bf16", 16, 1),
    ]
    if stage == "extended":
        configurations = [
            ("graph", 8, 1),
            ("graph", 16, 1),
            ("graph", 32, 1),
            ("bf16", 32, 1),
            ("bf16", 64, 1),
        ]
    elif stage == "sustained":
        configurations = [("baseline", 2, 4), ("fp32", 2, 4), ("graph", 8, 1), ("graph", 32, 1)]
    # Rotate execution order between repetitions to reduce ordering/temperature bias.
    for repetition in range(3):
        order = configurations[repetition:] + configurations[:repetition]
        for mode, batch_size, accumulation in order:
            runner = Runner(mode, 42)
            meter = ReferenceMeter() if mode == "baseline" else LossMeter()
            startup = runner.runtime.begin_measurement()
            with runner.loaded(training, batch_size, 42) as loaded:
                for _ in range(5):
                    runner.update(loaded, accumulation)
                warmup = runner.runtime.measurement(startup, 0, 5)
                with DeviceMonitor() as monitor:
                    start = runner.runtime.begin_measurement()
                    examples = sum(runner.update(loaded, accumulation, meter=meter) for _ in range(updates))
                    measured = runner.runtime.measurement(start, examples, updates)
            result = dict(
                mode=mode,
                batch_size=batch_size,
                accumulation=accumulation,
                repetition=repetition,
                warmup_seconds=warmup["training_seconds"],
                warmup_peak_allocated_mib=warmup["peak_allocated_mib"],
                warmup_peak_reserved_mib=warmup["peak_reserved_mib"],
                **measured,
                **monitor.result(),
            )
            results.append(result)
            save(f"throughput-{stage}" if stage != "throughput" else "throughput", results)
            print(json.dumps(result), flush=True)
            runner.close()
    return results


def parity(training):
    references, results = {}, []
    for mode in ("baseline", "fp32", "bf16"):
        runner = Runner(mode, 19, dropout=0)
        with runner.loaded(training, 2, 42, shuffle=False) as loaded:
            batch = next(loaded)
            output = runner.runtime.forward(runner.model, batch)
            terms = parts(runner.model, output, batch)
            loss = objective(terms)
            loss.backward()
        gradient = torch.cat(
            [p.grad.flatten().float() for p in runner.model.parameters() if p.grad is not None]
        ).cpu()
        predictions = {key: value.detach().float().cpu() for key, value in output.items() if key != "hidden"}
        if mode == "baseline":
            references = dict(gradient=gradient, predictions=predictions, loss=loss.item())
        else:
            left, right = gradient.double(), references["gradient"].double()
            cosine = float(
                torch.dot(left, right) / (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
            )
            errors = {
                key: float(
                    torch.linalg.vector_norm(value - references["predictions"][key])
                    / torch.linalg.vector_norm(references["predictions"][key]).clamp(min=1e-8)
                )
                for key, value in predictions.items()
            }
            result = dict(
                mode=mode,
                loss=loss.item(),
                reference_loss=references["loss"],
                gradient_cosine=cosine,
                relative_output_errors=errors,
            )
            results.append(result)
            print(json.dumps(result), flush=True)
            assert cosine > (0.99 if mode == "bf16" else 0.9999), result
            assert max(errors.values()) < (0.025 if mode == "bf16" else 0.001), result
        runner.close()
    save("parity", results)


def quality(training, validation, updates):
    weight = float(
        np.clip(
            sum(item["labels"][:, 4].sum() for item in training)
            / max(1, 2 * sum(item["labels"][:, 0].sum() for item in training))
            - 1,
            1,
            40,
        )
    )
    save(
        "quality-protocol",
        dict(
            seeds=[17, 42, 73],
            updates=updates,
            examples_per_update=8,
            baseline_batch=2,
            baseline_accumulation=4,
            graph_batch=8,
            graph_accumulation=1,
            acceptable_mean_loss_increase="max(0.03, 2% of abs(reference))",
            acceptable_mean_onset_f1_drop=0.02,
            acceptable_mean_motion_mae_increase="5%",
            caveat="Bounded regression experiment; not proof of long-run or human mapping quality.",
        ),
    )
    results = []
    for seed in (17, 42, 73):
        for mode in ("baseline", "fp32", "bf16", "graph"):
            runner = Runner(mode, seed)
            history = []
            with runner.loaded(training, 8 if mode == "graph" else 2, seed, complete_batches=True) as loaded:
                start = time.perf_counter()
                for update in range(updates):
                    factor = min(1, (update + 1) / 10) * (
                        0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * update / updates))
                    )
                    for group in runner.optimizer.param_groups:
                        group["lr"] = 0.0003 * factor
                    runner.update(loaded, 1 if mode == "graph" else 4, weight)
                    if (update + 1) % 40 == 0 or update + 1 == updates:
                        metrics = runner.evaluate(validation, weight)
                        history.append(dict(update=update + 1, **metrics))
                        print(
                            json.dumps(
                                dict(
                                    seed=seed,
                                    mode=mode,
                                    update=update + 1,
                                    loss=metrics["loss"],
                                    onset_f1=metrics["onset_f1"],
                                    motion_mae=metrics["motion_mae"],
                                )
                            ),
                            flush=True,
                        )
            # All variants use identical FP32/explicit inference for rollout comparability.
            from app.ml.attention import Attention

            for module in runner.model.modules():
                if isinstance(module, Attention):
                    module.cuda_sdpa = False
            diagnostics = evaluate_rollout(runner.model, validation[0], torch.device("cuda"), Context())
            result = dict(
                seed=seed,
                mode=mode,
                updates=updates,
                seconds=time.perf_counter() - start,
                history=history,
                free_generation=diagnostics,
            )
            results.append(result)
            save("quality", results)
            runner.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("parity", "throughput", "extended", "sustained", "quality"))
    parser.add_argument("--updates", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    # Strict FP32 baseline; neither TF32 nor compilation changes the comparison silently.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prepared = load_corpus()
    training, validation, test = split_maps(prepared, 42)
    save(
        "environment",
        dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
            threads=torch.get_num_threads(),
            dataset_signature=dataset_signature(prepared),
            training_maps=len(training),
            validation_maps=len(validation),
            test_maps=len(test),
            test_not_used=True,
        ),
    )
    if args.stage == "parity":
        parity(training)
    elif args.stage in ("throughput", "extended", "sustained"):
        throughput(training, args.updates or (150 if args.stage == "sustained" else 25), args.stage)
    else:
        quality(training, validation, args.updates or 120)


if __name__ == "__main__":
    main()
