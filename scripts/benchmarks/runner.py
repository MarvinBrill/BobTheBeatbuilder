"""Identical data/weights/objectives; selectable old and accelerated execution paths."""

from contextlib import contextmanager
import gc
import math
import time
import numpy as np
import torch
from app.ml.attention import Attention
from app.ml.phrase_model import PhraseMapper
from app.ml.architecture import ModelConfig
from app.ml.sequence_batches import batches
from app.ml.batch_loading import device_batches
from app.ml.phrase_losses import parts, objective, LossMeter
from app.ml.training_runtime import TrainingRuntime
from app.ml.phrase_metrics import ValidationMetrics
from app.ml.cuda_graph import CudaTrainingGraph
from reference_batches import batches as old_batches


class ReferenceMeter(LossMeter):
    def add(self, terms):
        for key, (numerator, denominator) in terms.items():
            total = self.totals.setdefault(key, [0.0, 0.0])
            total[0] += numerator.detach().cpu().item()
            total[1] += denominator.detach().cpu().item()


class Runner:
    def __init__(self, mode, seed, dropout=0.1):
        self.baseline = mode == "baseline"
        self.mode = mode
        torch.manual_seed(seed)
        self.runtime = TrainingRuntime(
            torch.device("cuda"), "bfloat16" if mode in ("bf16", "graph") else "float32"
        )
        self.graph = None
        self.model = PhraseMapper(ModelConfig(dropout=dropout)).cuda()
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.cuda_sdpa = not self.baseline
        self.optimizer = (
            torch.optim.AdamW(self.model.parameters(), lr=0.0003, weight_decay=0.01, foreach=False)
            if self.baseline
            else self.runtime.optimizer(self.model, 0.0003, 0.01)
        )

    @contextmanager
    def loaded(self, maps, batch_size, seed, shuffle=True, repeat=True, complete_batches=False):
        def values():
            rng = np.random.default_rng(seed)
            carry = None
            while True:
                for value in (old_batches if self.baseline else batches)(maps, batch_size, rng, shuffle):
                    if not complete_batches:
                        yield value
                        continue
                    if carry is not None:
                        value = {key: np.concatenate([carry[key], array]) for key, array in value.items()}
                    if len(value["audio"]) >= batch_size:
                        yield {key: array[:batch_size] for key, array in value.items()}
                        carry = {key: array[batch_size:] for key, array in value.items()}
                    else:
                        carry = value
                if not repeat:
                    break

        if self.baseline:
            yield (
                {key: torch.from_numpy(value).cuda() for key, value in batch.items()} for batch in values()
            )
        else:
            with device_batches(values(), torch.device("cuda")) as loaded:
                yield loaded

    def update(self, loaded, accumulation, weight=20, meter=None):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        finite = torch.ones((), dtype=torch.bool, device="cuda")
        examples = 0
        for _ in range(accumulation):
            batch = next(loaded)
            if self.mode == "graph" and accumulation == 1 and self.graph is None:
                self.graph = CudaTrainingGraph(self.model, self.runtime, batch, weight)
            captured = self.mode == "graph" and self.graph.matches(batch) and accumulation == 1
            if captured:
                loss, terms = self.graph.replay(batch)
            else:
                output = self.runtime.forward(self.model, batch)
                terms = parts(self.model, output, batch, weight)
                loss = objective(terms)
            if self.baseline:
                if not math.isfinite(loss.item()):
                    raise ValueError("Non-finite reference loss")
            else:
                finite = finite & torch.isfinite(loss.detach())
            if not captured:
                (loss / accumulation).backward()
            if meter is not None:
                meter.add(terms)
            examples += batch["audio"].shape[0]
        if self.baseline:
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1, foreach=False)
            if not math.isfinite(norm.item()):
                raise ValueError("Non-finite reference gradient")
        else:
            self.runtime.clip_and_check(self.model, finite)
        self.optimizer.step()
        return examples

    def evaluate(self, maps, weight):
        self.model.eval()
        meter, quality = LossMeter(), ValidationMetrics()
        # Compare every trained model with the SAME full-precision explicit evaluator.
        modules = [module for module in self.model.modules() if isinstance(module, Attention)]
        states = [module.cuda_sdpa for module in modules]
        for module in modules:
            module.cuda_sdpa = False
        try:
            with torch.no_grad(), self.loaded(maps, 2, 0, False, False) as loaded:
                for batch in loaded:
                    output = self.model(batch)
                    meter.add(parts(self.model, output, batch, weight))
                    quality.add(output, batch)
        finally:
            for module, state in zip(modules, states):
                module.cuda_sdpa = state
        return dict(loss=meter.value(), loss_terms=meter.values(), **quality.result())

    def close(self):
        self.graph = None
        del self.model, self.optimizer
        gc.collect()
        torch.cuda.empty_cache()
