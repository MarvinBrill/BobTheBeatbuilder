"""Validation-driven learning rate and early stop, persisted with optimizer state."""

import math


class TrainingControl:
    def __init__(self, learning_rate, patience, warmup_steps, best=math.inf, state=None):
        self.base_lr = learning_rate
        self.lr = learning_rate
        self.patience = patience
        self.warmup_steps = warmup_steps
        self.updates = 0
        self.best = best
        self.stale = self.plateau_bad = self.since_reduction = 0
        if state:
            self.lr = state["lr"] * learning_rate / state["base_lr"]
            self.warmup_steps = state["warmup_steps"]
            self.updates = state["updates"]
            self.best = state["best"] if state["best"] is not None else math.inf
            self.stale = state["stale"]
            self.plateau_bad = state["plateau_bad"]
            self.since_reduction = state["since_reduction"]

    def step_rate(self):
        return self.lr * min(1, (self.updates + 1) / max(1, self.warmup_steps))

    def observe(self, score, validation=True):
        if not math.isfinite(score):
            raise ValueError("Non-finite validation score")
        improved = score < self.best
        significant = score < self.best - 1e-4
        if improved:
            self.best, self.stale = score, 0
        else:
            self.stale += 1
        self.plateau_bad = 0 if significant else self.plateau_bad + 1
        self.since_reduction += 1
        interval = min(3, max(1, self.patience // 3))
        floor = self.base_lr * 0.05
        reduced = (
            validation
            and self.plateau_bad >= interval
            and self.stale < self.patience
            and self.lr > floor * 1.001
        )
        if reduced:
            self.lr = max(floor, self.lr * 0.5)
            self.plateau_bad = self.since_reduction = 0
        stop = validation and self.stale >= self.patience and self.since_reduction >= interval
        return improved, reduced, stop

    def state(self):
        return {
            key: (value if key != "best" or math.isfinite(value) else None)
            for key, value in vars(self).items()
        }


def context_probability(epoch, enabled):
    # Four clean epochs, then a ten-epoch ramp to at most 20% of history positions.
    return min(0.2, max(0, (epoch - 4) * 0.02)) if enabled else 0.0
