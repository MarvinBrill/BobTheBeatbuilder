"""Official osu!standard star ratings via the bundled .NET calculator (ppy.osu.Game.Rulesets.Osu).

Training labels and generated maps use the same calculator, so a target of 6 stars means
6 stars as lazer computes them for this package version.
"""

import json
import math
import subprocess
from pathlib import Path
from uuid import uuid4
from . import store
from .config import DATA, DIFFICULTY

ATTRIBUTE_KEYS = (
    "star_rating",
    "aim_difficulty",
    "speed_difficulty",
    "reading_difficulty",
    "flashlight_difficulty",
    "slider_factor",
    "speed_note_count",
    "aim_difficult_strain_count",
    "speed_difficult_strain_count",
    "max_combo",
)
CHUNK = 1500


def available():
    return DIFFICULTY.is_file()


def _clean(result):
    attributes = {
        key: float(result[key])
        for key in ATTRIBUTE_KEYS
        if isinstance(result.get(key), (int, float)) and math.isfinite(result[key])
    }
    if "star_rating" not in attributes:
        return None
    attributes["calculator"] = str(result.get("calculator", "unknown"))
    return attributes


def calculate(paths, context=None, progress=(0.0, 1.0), message="Sterne berechnen"):
    """Attributes for each path (None where the calculator failed), in input order."""
    paths = [str(Path(path)) for path in paths]
    if not paths:
        return []
    if not available():
        raise ValueError(
            "Der Sterne-Rechner fehlt. Führe scripts/setup.ps1 erneut aus, damit er gebaut wird."
        )
    results = []
    start, end = progress
    for offset in range(0, len(paths), CHUNK):
        if context is not None:
            context.report(
                start + (end - start) * offset / len(paths),
                f"{message} · {offset}/{len(paths)}",
            )
        chunk = paths[offset : offset + CHUNK]
        token = uuid4().hex
        source = DATA / "tmp" / f"difficulty-{token}.txt"
        target = DATA / "tmp" / f"difficulty-{token}.jsonl"
        try:
            source.write_text("\n".join(chunk) + "\n", encoding="utf-8")
            process = subprocess.run(
                ["dotnet", str(DIFFICULTY), str(source), str(target)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(120, 2 * len(chunk)),
            )
            if process.returncode:
                raise ValueError(f"Sterne-Rechner fehlgeschlagen: {process.stderr[-1500:]}")
            lines = target.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError as error:
            raise ValueError("dotnet wurde nicht gefunden; der Sterne-Rechner braucht .NET 10.") from error
        finally:
            source.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
        if len(lines) != len(chunk):
            raise ValueError("Sterne-Rechner lieferte eine unvollständige Antwort")
        for line in lines:
            try:
                results.append(_clean(json.loads(line)))
            except (ValueError, TypeError):
                results.append(None)
    return results


def calculate_one(path):
    return calculate([path])[0]


def with_attributes(record, attributes):
    """A map record with calculated stars; lazer's stored value is kept for reference."""
    if not attributes:
        return record
    updated = {**record, "difficulty_attributes": attributes}
    if record.get("star_source") == "lazer" and record.get("star_rating") is not None:
        updated["lazer_star_rating"] = record["star_rating"]
    updated["star_rating"] = round(attributes["star_rating"], 2)
    updated["star_source"] = "calculated"
    return updated


def needs_attributes(record, calculator=None):
    attributes = record.get("difficulty_attributes")
    if not attributes:
        return True
    return calculator is not None and attributes.get("calculator") != calculator


def annotate(records, context=None, progress=(0.0, 1.0), force=False):
    """Records with difficulty attributes added where missing; returns (records, changed)."""
    pending = [
        index for index, record in enumerate(records) if force or needs_attributes(record)
    ]
    if not pending or not available():
        return list(records), 0
    attributes = calculate([records[index]["map_path"] for index in pending], context, progress)
    updated, changed = list(records), 0
    for index, values in zip(pending, attributes):
        if values:
            updated[index] = with_attributes(updated[index], values)
            changed += 1
    return updated, changed


def annotate_library(context, force=False):
    """Job: calculate star ratings for every library map that has none yet."""
    if not available():
        raise ValueError(
            "Der Sterne-Rechner fehlt. Führe scripts/setup.ps1 erneut aus, damit er gebaut wird."
        )
    records = store.all_records("map")
    pending = [record for record in records if force or needs_attributes(record)]
    updated, changed = annotate(pending, context, (0.0, 0.95), force=True)
    context.check()
    store.save_many("map", [record for record in updated if record.get("difficulty_attributes")])
    failed = len(pending) - changed
    return {"calculated": changed, "failed": failed, "total": len(records)}


def annotate_dataset(dataset_id, context, progress=(0.0, 0.05)):
    """Fill missing attributes of a dataset's map copies (and of the library maps they came from)."""
    dataset = store.get("dataset", dataset_id)
    records, changed = annotate(dataset["maps"], context, progress)
    if changed:
        store.save("dataset", {**dataset, "maps": records})
        known = store.get_many("map", [record["id"] for record in records])
        store.save_many(
            "map",
            [
                with_attributes(known[record["id"]], record["difficulty_attributes"])
                for record in records
                if record["id"] in known
                and record.get("difficulty_attributes")
                and needs_attributes(known[record["id"]])
            ],
        )
    return records, changed
