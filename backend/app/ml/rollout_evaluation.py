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
from .sequence_generation import generate_objects, generate_objects_v2
from .phrase_metrics import onset_matches
from .map_audit import movement_audit, rhythm_audit
from .rollout_sections import (
    section_plan,
    MAX_SECTION_OBJECTS,
    MAX_FULL_SECONDS,
    MAX_FULL_OBJECTS,
)
from . import style

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


def styled_rollout_settings(item, seed):
    values = {key: style.decode(key, item["style"][style.INDEX[key]]) for key in style.KEYS}
    triplets = (
        item["style_mask"][style.INDEX["triplet_share"]] > 0 and values["triplet_share"] > 0.01
    )
    settings = {
        **{key: values[key] for key in ("ar", "cs", "od", "hp")},
        "slider_bias": 0.5,
        "seed": seed,
        "temperature": 0.8,
        "candidates": 12,
        "bpm": None,
        "offset": 0,
        "snap": True,
        "divisor": 12 if triplets else 4,
        "title": "Free evaluation",
        "artist": "Validation corpus",
        "version": "Diagnostic",
    }
    if item.get("timing"):
        # The reference timing isolates pattern quality from tempo detection.
        settings.update(timing_override=list(item["timing"]), timing_source="reference")
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


def longest_short_run(objects):
    """Objects in the longest run of <= 1/4-beat gaps (1 when there is none)."""
    longest = length = 1
    for obj, following in zip(objects, objects[1:]):
        short = following.time - obj.time <= style.STREAM_GAP * obj.beat_length + 1e-6
        length = length + 1 if short else 1
        longest = max(longest, length)
    return longest if objects else 0


def covered_ratio(predicted, reference, truncated):
    """Generated / reference objects over the span the generator actually covered."""
    if truncated and len(predicted):
        reference = reference[reference <= predicted[-1]]
    return len(predicted) / max(1, len(reference))


def evaluate_rollout(
    model, item, selected_device, context, max_objects=256, progress=0.97, section=None, seed=42
):
    duration = max(0.0, (len(item["mel"]) - 1) * FRAME_MS)
    start, end = (section["start_ms"], section["end_ms"]) if section else (0.0, duration)
    started = time.perf_counter()
    model.eval()
    if getattr(model, "styled", False):
        settings = styled_rollout_settings(item, seed)
        condition = style.condition_input(item["style"], item["style_mask"])
        mapper = torch.tensor([float(item.get("mapper", 0))], device=selected_device)
        objects, timing, diagnostics = generate_objects_v2(
            model,
            item["mel"],
            item["audio_path"],
            settings,
            torch.from_numpy(condition)[None].to(selected_device),
            selected_device,
            RolloutProgress(context, progress),
            max_objects=max_objects,
            mapper=mapper,
            evaluation_range=(start, end),
        )
    else:
        condition = item["condition"]
        settings = rollout_settings(condition, seed)
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
        # Over- or undermapping against the reference (1 = same count); a stream locked in
        # for a whole section shows up here and in the longest run, not in the losses.
        object_ratio=covered_ratio(predicted, reference, diagnostics.get("truncated", False)),
        longest_short_run=longest_short_run(objects),
        reference_slider_fraction=float(events[:, 9].mean()) if len(events) else 0,
        reference_jump_p95_pixels=(
            float(np.percentile(events[:, 39] * 512, 95)) if len(events) else 0
        ),
        **roundtrip_report(objects, timing, settings),
    )
    # The checks a mapper would do first (see map_audit): star chains, onset agreement.
    measured = {**movement_audit(objects, settings.get("cs", 4)), **rhythm_audit(objects, item["mel"], FRAME_MS)}
    for key in ("star_chain_share", "sharp_share", "onset_hit_share", "strong_onset_coverage"):
        value = measured.get(key)
        if value is not None:
            diagnostics[key] = float(value)
    diagnostics.pop("song_plan", None)
    diagnostics["seconds"] = time.perf_counter() - started
    return diagnostics


def timing_check(model, item, selected_device, context, progress=0.97):
    """Detected timing (no reference) against the reference red lines of one song."""
    from .rhythm_inference import detect, predict_rhythm
    from .sequence_data import summary_features
    from ..timing import red_points

    points = red_points(item.get("timing") or [])
    if not points or not getattr(model, "styled", False):
        return None
    condition = torch.from_numpy(style.condition_input(item["style"], item["style_mask"]))[None]
    mapper = torch.tensor([float(item.get("mapper", 0))], device=selected_device)
    prediction = predict_rhythm(
        model, item["mel"], summary_features(item["mel"]), condition.to(selected_device),
        selected_device, RolloutProgress(context, progress), mapper,
    )  # fmt: skip
    result = detect(prediction, item["mel"], {})
    if result is None:
        return {"map_id": item["map_id"], "detected": False}
    duration = len(item["mel"]) * FRAME_MS
    spans = [
        ((points[i + 1].time if i + 1 < len(points) else duration) - max(0.0, point.time), point)
        for i, point in enumerate(points)
    ]
    reference = max(spans, key=lambda span: span[0])[1]
    found = result.points
    detected = max(
        (
            ((found[i + 1].time if i + 1 < len(found) else duration) - max(0.0, point.time), point)
            for i, point in enumerate(found)
        ),
        key=lambda span: span[0],
    )[1]
    ratio = reference.beat_length / detected.beat_length
    octave = min(abs(ratio - value) for value in (0.5, 2.0)) < 0.02
    error = abs(ratio - 1)
    beat = reference.beat_length
    shift = ((detected.time - reference.time) % beat + beat / 2) % beat - beat / 2
    return {
        "map_id": item["map_id"],
        "detected": True,
        "reference_bpm": 60000 / beat,
        "detected_bpm": 60000 / detected.beat_length,
        "timing_bpm_error": float(min(error, 1.0)),
        "timing_octave_error": bool(octave),
        "timing_offset_ms": float(abs(shift)) if error < 0.002 else None,
    }


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
    timing_results = []
    for item in chosen:
        context.check()
        try:
            checked = timing_check(model, item, selected_device, context, progress)
        except ValueError as error:
            checked = {"map_id": item["map_id"], "detected": False, "reason": str(error)}
        if checked is not None:
            timing_results.append(checked)
    keys = (
        "reference_onset_f1",
        "heavily_repaired_fraction",
        "mean_geometry_repair",
        "temporal_overlaps",
        "slider_shape_issues",
        "jump_p95_pixels",
        "aim_speed_p95",
        "object_ratio",
        "longest_short_run",
        "star_chain_share",
        "sharp_share",
        "onset_hit_share",
        "strong_onset_coverage",
    )
    means = {}
    for key in keys:
        values = [result[key] for result in results if result.get(key) is not None]
        if values:
            means[key] = float(np.mean(values))
    detected = [entry for entry in timing_results if entry.get("detected")]
    if timing_results:
        # A song whose timing was not found at all counts as a full error.
        errors = [entry["timing_bpm_error"] if entry.get("detected") else 1.0 for entry in timing_results]
        means["timing_bpm_error"] = float(np.mean(errors))
        offsets = [entry["timing_offset_ms"] for entry in detected if entry.get("timing_offset_ms") is not None]
        if offsets:
            means["timing_offset_ms"] = float(np.mean(offsets))
    return {
        "report_schema": REPORT_SCHEMA,
        "songs": len({result["song"] for result in results}),
        "attempted": len(planned),
        "sections": len(results),
        "max_objects_per_section": MAX_SECTION_OBJECTS,
        "max_objects_full_song": MAX_FULL_OBJECTS if include_full_song else None,
        "means": means if results else {},
        "results": results,
        "timing": timing_results,
        "failures": failures,
        "full_song_skip_reason": (
            "No selected song within 300 seconds" if include_full_song and not eligible else None
        ),
        "note": "Fixed audio probes include window transitions; activity does not identify a chorus. Truncation is reported. Reference agreement is not a human quality rating. Object ratio, star chains, onset agreement and timing detection feed the free-generation checkpoint score.",
    }
