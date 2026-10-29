"""Verified preparation results; bump VERSION when training target semantics change."""

import hashlib
import json
from uuid import uuid4
import numpy as np
from ..config import DATA
from .style import FEATURE_COUNT

VERSION = 2
EVENT_COLUMNS = 49


def fingerprint(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def paths(key):
    root = DATA / "cache"
    return root / f"phrase-prepared-v{VERSION}-{key}.json", {
        "events": root / f"phrase-events-v4-{key}.npy",
        "labels": root / f"phrase-labels-v3-{key}.npy",
    }


def _vector(values, size):
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError("Invalid cached vector")
    return array


def load(key, frames, signature=""):
    manifest, files = paths(key)
    try:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            meta["frames"] != frames
            or meta["version"] != VERSION
            or meta.get("attribute_signature", "") != signature
        ):
            return None
        arrays = {
            name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in files.items()
        }
        if (
            arrays["events"].ndim != 2
            or arrays["events"].shape[1] != EVENT_COLUMNS
            or arrays["labels"].shape != (frames, 6)
        ):
            return None
        for name, array in arrays.items():
            if (
                array.dtype != np.float32
                or list(array.shape) != meta["shapes"][name]
                or fingerprint(array) != meta["hashes"][name]
            ):
                return None
        timing = [[float(time), float(beat)] for time, beat in meta["timing"]]
        return {
            **arrays,
            "condition": _vector(meta["condition"], 8),
            "style": _vector(meta["style"], FEATURE_COUNT),
            "style_mask": _vector(meta["style_mask"], FEATURE_COUNT),
            "mapper_candidate": meta["mapper_candidate"],
            "creator": str(meta["creator"]),
            "timing": timing,
            "attribute_signature": signature,
            "masked_geometry": int(meta["masked_geometry"]),
        }
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        return None


def save(key, frames, prepared):
    manifest, _ = paths(key)
    meta = {
        "version": VERSION,
        "frames": frames,
        "condition": prepared["condition"].tolist(),
        "style": prepared["style"].tolist(),
        "style_mask": prepared["style_mask"].tolist(),
        "mapper_candidate": prepared["mapper_candidate"],
        "creator": prepared["creator"],
        "timing": prepared["timing"],
        "attribute_signature": prepared.get("attribute_signature", ""),
        "masked_geometry": prepared["masked_geometry"],
        "shapes": {name: list(prepared[name].shape) for name in ("events", "labels")},
        "hashes": {name: fingerprint(prepared[name]) for name in ("events", "labels")},
    }
    temporary = manifest.with_name(f"{manifest.stem}-{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(meta, allow_nan=False), encoding="utf-8")
        temporary.replace(manifest)
    finally:
        temporary.unlink(missing_ok=True)
