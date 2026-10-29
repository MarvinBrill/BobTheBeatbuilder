"""A bounded free-running check, separate from teacher-forced validation."""

import numpy as np
import torch
import hashlib
from .sequence_generation import generate_objects, generate_objects_v2
from .phrase_metrics import onset_matches
from . import style


class RolloutProgress:
    def __init__(self, context, progress):
        self.context, self.progress = context, progress

    def check(self):
        self.context.check()

    def report(self, _progress, message, **_kwargs):
        self.context.report(self.progress, f"Freie Testgenerierung · {message}")


def styled_rollout_settings(item):
    values = {key: style.decode(key, item["style"][style.INDEX[key]]) for key in style.KEYS}
    triplets = item["style_mask"][style.INDEX["triplet_share"]] > 0 and values["triplet_share"] > 0.01
    return {
        **{key: values[key] for key in ("ar", "cs", "od", "hp")},
        "slider_bias": 0.5,
        "seed": 42,
        "temperature": 0.8,
        "candidates": 12,
        "bpm": None,
        "offset": 0,
        "snap": True,
        # The reference timing isolates pattern quality from tempo detection.
        "timing": [tuple(point) for point in item.get("timing") or []],
        "divisor": 12 if triplets else 4,
    }


def evaluate_rollout(model, item, selected_device, context, max_objects=256, progress=0.97):
    model.eval()
    if getattr(model, "styled", False):
        condition = style.condition_input(item["style"], item["style_mask"])
        mapper = torch.tensor([float(item.get("mapper", 0))], device=selected_device)
        objects, _, diagnostics = generate_objects_v2(
            model,
            item["mel"],
            item["audio_path"],
            styled_rollout_settings(item),
            torch.from_numpy(condition)[None].to(selected_device),
            selected_device,
            RolloutProgress(context, progress),
            max_objects=max_objects,
            mapper=mapper,
        )
    else:
        condition = item["condition"]
        settings = dict(zip(("ar", "cs", "od", "hp", "density"), (condition[:5] * 10).tolist()))
        settings.update(
            spacing=float(condition[5]),
            flow=float(condition[6]),
            slider_bias=float(condition[7]),
            seed=42,
            temperature=0.8,
            candidates=12,
            bpm=None,
            offset=0,
            snap=True,
            divisor=4,
        )
        objects, _, diagnostics = generate_objects(
            model,
            item["mel"],
            item["audio_path"],
            settings,
            torch.from_numpy(condition.copy())[None].to(selected_device),
            selected_device,
            RolloutProgress(context, progress),
            max_objects=max_objects,
        )
    # Other valid mappings exist; agreement with one reference is only a diagnostic.
    predicted = np.array([obj.time for obj in objects])
    reference = item["events"][:, 0] * 1000
    reference = reference[reference <= predicted[-1] + 35]
    matches = onset_matches(predicted, reference, 35)
    diagnostics.update(
        map_id=item["map_id"],
        max_objects=max_objects,
        prefix_seconds=float(predicted[-1] / 1000),
        reference_onset_f1=2 * matches / max(1, len(predicted) + len(reference)),
        reference_slider_fraction=float(item["events"][:, 9].mean()),
        reference_jump_p95_pixels=float(np.percentile(item["events"][:, 39] * 512, 95)),
    )
    return diagnostics


def rollout_songs(maps, limit=3):
    # Stable, distinct song groups: alternate difficulties cannot fill the panel.
    ordered = sorted(maps, key=lambda item: hashlib.sha256(item["map_id"].encode()).hexdigest())
    chosen, seen = [], set()
    for item in ordered:
        if item["song"] not in seen:
            chosen.append(item)
            seen.add(item["song"])
        if len(chosen) == limit:
            break
    return chosen


def evaluate_suite(model, maps, selected_device, context, progress=0.97):
    results, failures = [], []
    for item in rollout_songs(maps):
        context.check()
        try:
            results.append(
                evaluate_rollout(model, item, selected_device, context, progress=progress)
            )
        except ValueError as error:
            failures.append({"map_id": item["map_id"], "reason": str(error)})
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
        "songs": len(results),
        "attempted": len(results) + len(failures),
        "max_objects_per_song": 256,
        "means": (
            {key: float(np.mean([result[key] for result in results])) for key in keys}
            if results
            else {}
        ),
        "results": results,
        "failures": failures,
        "note": "Fixed song prefixes; reference agreement is not a human playability rating. Not used for checkpoint selection.",
    }
