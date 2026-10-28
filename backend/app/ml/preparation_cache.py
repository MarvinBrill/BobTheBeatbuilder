"""Verified preparation results; bump VERSION when training target semantics change."""

import hashlib
import json
from dataclasses import asdict
from uuid import uuid4
import numpy as np
from ..config import DATA
from ..timing import TimingPoint
from .architecture import TARGET_SCHEMA, GLOBAL_TOKENS
from .phrase_plan import PLAN_FEATURES

VERSION = 3


def fingerprint(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def paths(key):
    root = DATA / "cache"
    return root / f"phrase-prepared-v{VERSION}-{key}.json", {
        "events": root / f"phrase-events-v5-{key}.npy",
        "labels": root / f"phrase-labels-v5-{key}.npy",
        "plan_targets": root / f"phrase-plan-v1-{key}.npy",
        "plan_mask": root / f"phrase-plan-mask-v1-{key}.npy",
    }


def load(key, frames):
    manifest, files = paths(key)
    try:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            meta["frames"] != frames
            or meta["version"] != VERSION
            or meta["target_schema"] != TARGET_SCHEMA
        ):
            return None
        arrays = {
            name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in files.items()
        }
        if (
            arrays["events"].ndim != 2
            or arrays["events"].shape[1] != 47
            or arrays["labels"].shape != (frames, 6)
            or arrays["plan_targets"].shape != (GLOBAL_TOKENS, PLAN_FEATURES)
            or arrays["plan_mask"].shape != (GLOBAL_TOKENS, PLAN_FEATURES)
        ):
            return None
        for name, array in arrays.items():
            if (
                array.dtype != np.float32
                or list(array.shape) != meta["shapes"][name]
                or fingerprint(array) != meta["hashes"][name]
            ):
                return None
        condition = np.asarray(meta["condition"], dtype=np.float32)
        if condition.shape != (8,) or not np.isfinite(condition).all():
            return None
        timing = [TimingPoint(**point) for point in meta["timing"]]
        if any(point.inherited for point in timing) or timing != sorted(
            timing, key=lambda point: point.time
        ):
            return None
        return {
            **arrays,
            "condition": condition,
            "timing": timing,
            "masked_geometry": int(meta["masked_geometry"]),
        }
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        return None


def save(key, frames, prepared):
    manifest, _ = paths(key)
    meta = {
        "version": VERSION,
        "target_schema": TARGET_SCHEMA,
        "timing": [asdict(point) for point in prepared["timing"]],
        "frames": frames,
        "condition": prepared["condition"].tolist(),
        "masked_geometry": prepared["masked_geometry"],
        "shapes": {name: list(prepared[name].shape) for name in paths(key)[1]},
        "hashes": {name: fingerprint(prepared[name]) for name in paths(key)[1]},
    }
    temporary = manifest.with_name(f"{manifest.stem}-{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(meta, allow_nan=False), encoding="utf-8")
        temporary.replace(manifest)
    finally:
        temporary.unlink(missing_ok=True)
