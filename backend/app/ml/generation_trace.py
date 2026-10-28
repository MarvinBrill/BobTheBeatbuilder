"""Optional observation of generation, without sampling or changing its inputs."""

import math
from dataclasses import asdict

import numpy as np

TRACE_SCHEMA = "generation-trace-v1"


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def object_snapshot(obj):
    return asdict(obj)


class GenerationTrace:
    def __init__(self):
        self.events = []

    def record(self, stage, **fields):
        # Copy arrays now: later guard adjustments must not rewrite earlier evidence.
        self.events.append(json_value({"stage": stage, **fields}))

    def document(self):
        return {"schema": TRACE_SCHEMA, "events": self.events}
