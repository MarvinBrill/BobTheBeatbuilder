"""Bounded free generation with fixed coverage, separate from reference validation."""

import hashlib
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from ..beatmaps import write, parse
from ..config import DATA
from .audio import FRAME_MS
from .sequence_generation import generate_objects
from .phrase_metrics import onset_matches
from .rollout_sections import (
    section_plan,
    MAX_SECTION_OBJECTS,
    MAX_FULL_SECONDS,
    MAX_FULL_OBJECTS,
)

REPORT_SCHEMA = "free-generation-sections-v2"


class RolloutProgress:
    def __init__(self, context, progress):
        self.context, self.progress = context, progress

    def check(self):
        self.context.check()

    def report(self, _progress, message, **_kwargs):
        self.context.report(self.progress, f"Freie Testgenerierung · {message}")


def rollout_settings(condition, seed):
    settings = dict(zip(("ar", "cs", "od", "hp", "density"), (condition[:5] * 10).tolist()))
    settings.update(
        spacing=float(condition[5]),
        flow=float(condition[6]),
        slider_bias=float(condition[7]),
        seed=seed,
        temperature=0.8,
        candidates=12,
        bpm=None,
        offset=0,
        snap=True,
        divisor=4,
        title="Free evaluation",
        artist="Validation corpus",
        version="Diagnostic",
    )
    return settings


def roundtrip_report(objects, timing, settings):
    if not objects:
        return dict(
            roundtrip_objects=0, roundtrip_max_duration_error_ms=0, roundtrip_max_endpoint_error=0
        )
    with tempfile.TemporaryDirectory(prefix="rollout-", dir=DATA / "tmp") as directory:
        path = Path(directory) / "map.osu"
        write(
            path,
            objects,
            settings,
            "audio.ogg",
            60000 / timing[0].beat_length,
            timing[0].time,
            timing,
        )
        parsed = parse(path).objects
    if len(parsed) != len(objects):
        raise ValueError("Free evaluation export changed the object count")
    return dict(
        roundtrip_objects=len(parsed),
        roundtrip_max_duration_error_ms=max(
            abs(a.duration - b.duration) for a, b in zip(objects, parsed)
        ),
        roundtrip_max_endpoint_error=max(
            np.hypot(a.end_x - b.end_x, a.end_y - b.end_y) for a, b in zip(objects, parsed)
        ),
    )


def evaluate_rollout(
    model, item, selected_device, context, max_objects=256, progress=0.97, section=None, seed=42
):
    condition = item["condition"]
    settings = rollout_settings(condition, seed)
    duration = max(0.0, (len(item["mel"]) - 1) * FRAME_MS)
    start, end = (section["start_ms"], section["end_ms"]) if section else (0.0, duration)
    started = time.perf_counter()
    model.eval()
    objects, timing, diagnostics = generate_objects(
        model,
        item["mel"],
        item["audio_path"],
        settings,
        torch.from_numpy(condition.copy())[None].to(selected_device),
        selected_device,
        RolloutProgress(context, progress),
        max_objects=max_objects,
        evaluation_range=(start, end),
    )
    # Coverage is fixed before generation; sparse output cannot shorten the reference.
    predicted = np.array([obj.time for obj in objects])
    events = item["events"]
    events = events[(events[:, 0] * 1000 >= start) & (events[:, 0] * 1000 < end)]
    reference = events[:, 0] * 1000
    matches = onset_matches(predicted, reference, 35)
    diagnostics.update(
        report_schema=REPORT_SCHEMA,
        map_id=item["map_id"],
        song=item["song"],
        seed=seed,
        max_objects=max_objects,
        start_ms=start,
        end_ms=end,
        probes=section["probes"] if section else ["full_song"],
        scope="section" if section else "song",
        completed_full_song=section is None and not diagnostics.get("truncated", False),
        reference_objects=len(reference),
        generated_objects=len(predicted),
        reference_onset_f1=2 * matches / max(1, len(predicted) + len(reference)),
        reference_slider_fraction=float(events[:, 9].mean()) if len(events) else 0,
        reference_jump_p95_pixels=(
            float(np.percentile(events[:, 39] * 512, 95)) if len(events) else 0
        ),
        **roundtrip_report(objects, timing, settings),
    )
    diagnostics["seconds"] = time.perf_counter() - started
    return diagnostics


def rollout_songs(maps, limit=3):
    ordered = sorted(maps, key=lambda item: hashlib.sha256(item["map_id"].encode()).hexdigest())
    chosen, seen = [], set()
    for item in ordered:
        if item["song"] not in seen:
            chosen.append(item)
            seen.add(item["song"])
        if len(chosen) == limit:
            break
    return chosen


def evaluate_suite(model, maps, selected_device, context, progress=0.97, include_full_song=False):
    results, failures = [], []
    chosen = rollout_songs(maps)
    planned = [
        (item, section, MAX_SECTION_OBJECTS)
        for item in chosen
        for section in section_plan(item["mel"])
    ]
    eligible = [
        item for item in chosen if (len(item["mel"]) - 1) * FRAME_MS <= MAX_FULL_SECONDS * 1000
    ]
    if include_full_song and eligible:
        planned.append((eligible[0], None, MAX_FULL_OBJECTS))
    for item, section, limit in planned:
        context.check()
        try:
            results.append(
                evaluate_rollout(
                    model,
                    item,
                    selected_device,
                    context,
                    progress=progress,
                    section=section,
                    max_objects=limit,
                )
            )
        except ValueError as error:
            failures.append(dict(map_id=item["map_id"], section=section, reason=str(error)))
    keys = (
        "reference_onset_f1",
        "heavily_repaired_fraction",
        "mean_geometry_repair",
        "temporal_overlaps",
        "slider_shape_issues",
        "jump_p95_pixels",
        "aim_speed_p95",
    )
    return {
        "report_schema": REPORT_SCHEMA,
        "songs": len({result["song"] for result in results}),
        "attempted": len(planned),
        "sections": len(results),
        "max_objects_per_section": MAX_SECTION_OBJECTS,
        "max_objects_full_song": MAX_FULL_OBJECTS if include_full_song else None,
        "means": (
            {key: float(np.mean([result[key] for result in results])) for key in keys}
            if results
            else {}
        ),
        "results": results,
        "failures": failures,
        "full_song_skip_reason": (
            "No selected song within 300 seconds" if include_full_song and not eligible else None
        ),
        "note": "Fixed audio probes include window transitions; activity does not identify a chorus. Truncation is reported. Reference agreement is not a human quality rating; not used for checkpoint selection.",
    }
