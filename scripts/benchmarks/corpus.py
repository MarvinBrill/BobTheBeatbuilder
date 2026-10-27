import json
import time
from pathlib import Path
import numpy as np
from app import store
from app.config import DATA
from app.ml.sequence_data import prepare

DIRECTORY = DATA / "benchmarks" / "cuda"


class Context:
    def __init__(self):
        self.last = 0

    def check(self):
        pass

    def report(self, progress, message, **kwargs):
        if time.monotonic() - self.last > 10:
            print(message, flush=True)
            self.last = time.monotonic()


def prepare_corpus(count=32):
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    manifest = DIRECTORY / "manifest.json"
    if manifest.exists():
        identifiers = json.loads(manifest.read_text(encoding="utf-8"))
        selected = [store.get("map", identifier) for identifier in identifiers]
    else:
        candidates = [
            item
            for item in store.all_records("map")
            if 45 <= item["duration"] <= 150
            and 120 <= item["objects"] <= 900
            and 1 <= item["density"] <= 8
            and Path(item["audio_path"]).is_file()
        ]
        np.random.default_rng(42).shuffle(candidates)
        selected, songs, hashes = [], set(), set()
        for item in candidates:
            song = (item["artist"].casefold(), item["title"].casefold())
            if song in songs or item["audio_hash"] in hashes:
                continue
            selected.append(item)
            songs.add(song)
            hashes.add(item["audio_hash"])
            if len(selected) == count:
                break
        manifest.write_text(json.dumps([item["id"] for item in selected]), encoding="utf-8")
    issues = []
    prepared = prepare(selected, Context(), issues)
    descriptors = []
    for index, item in enumerate(prepared):
        descriptor = {
            key: value
            for key, value in item.items()
            if key not in ("mel", "events", "labels", "summary", "condition")
        }
        descriptor.update(
            {key: str(Path(item[key].filename).resolve()) for key in ("mel", "events", "labels")}
        )
        summary_path = DIRECTORY / f"summary-{index}.npy"
        np.save(summary_path, item["summary"], allow_pickle=False)
        descriptor.update(summary=str(summary_path), condition=item["condition"].tolist())
        descriptors.append(descriptor)
    (DIRECTORY / "prepared.json").write_text(json.dumps(descriptors, indent=2), encoding="utf-8")
    (DIRECTORY / "excluded.json").write_text(json.dumps(issues, indent=2), encoding="utf-8")
    print(
        json.dumps(
            dict(
                maps=len(prepared),
                excluded=issues,
                total_objects=sum(len(item["events"]) for item in prepared),
            )
        ),
        flush=True,
    )


def load_corpus():
    prepared = json.loads((DIRECTORY / "prepared.json").read_text(encoding="utf-8"))
    for item in prepared:
        for key in ("mel", "events", "labels", "summary"):
            item[key] = np.load(item[key], mmap_mode="r", allow_pickle=False)
        item["condition"] = np.asarray(item["condition"], np.float32)
    return prepared


if __name__ == "__main__":
    prepare_corpus()
