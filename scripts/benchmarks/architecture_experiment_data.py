"""Rebuild new-format targets from checked source maps; never convert old targets."""

import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from app.beatmaps import parse
from app.config import DATA
from app.library import digest
from app.ml.sequence_data import prepare_serial


def prepare_manifest(path, context):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records, membership = [], {}
    for split in ("training", "validation"):
        for item in manifest[split]:
            for kind in ("audio", "map"):
                if digest(Path(item[f"{kind}_path"])) != item[f"{kind}_hash"]:
                    raise ValueError(f"Changed {kind} for {item['map_id']}")
            info = parse(Path(item["map_path"])).info()
            records.append(dict(id=item["map_id"], artist=info["artist"], title=info["title"],
                                audio_hash=item["audio_hash"], audio_path=item["audio_path"],
                                map_path=item["map_path"]))
            membership[item["map_id"]] = split
            source = Path(item["mel_path"]) if item.get("mel_path") else None
            destination = DATA / "cache" / f"mel-v2-{item['audio_hash']}.npy"
            if source and source.is_file() and not destination.exists():
                values = np.load(source, mmap_mode="r", allow_pickle=False)
                if values.ndim != 2 or values.shape[1] != 64 or values.dtype != np.float32:
                    raise ValueError("Invalid source Mel cache")
                shutil.copyfile(source, destination)
    issues = []
    prepared = prepare_serial(records, context, issues)
    if issues:
        raise ValueError(f"Fixed experiment corpus must not silently change: {issues}")
    training = [item for item in prepared if membership[item["map_id"]] == "training"]
    validation = [item for item in prepared if membership[item["map_id"]] == "validation"]
    if {item["song"] for item in training} & {item["song"] for item in validation}:
        raise ValueError("Song group leakage in experiment manifest")
    return training, validation, hashlib.sha256(path.read_bytes()).hexdigest()
