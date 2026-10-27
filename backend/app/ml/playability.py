import math
import numpy as np
from ..beatmaps import HitObject
from ..curve_geometry import curve_length
from .movement import motion_numpy, history_context, bezier_points
from .architecture import DURATION_BEATS
from .slider_geometry import fit_curve_bounds, curve_diagnostics


def circle_radius(cs):
    return 54.4 - 4.48 * cs


def candidate_object(
    geometry, time, next_time, beat, slider, duration_beats, repeats, combo, settings, history
):
    radius = circle_radius(settings["cs"])
    margin = math.ceil(radius) + 1
    curve = np.asarray(geometry, dtype=float).reshape(4, 2) * [512, 384]
    original = curve.copy()
    if not slider:
        curve[:] = curve[0]
    curve = fit_curve_bounds(curve, margin)
    if history:
        last = history[-1]
        delta = curve[0] - [last.end_x, last.end_y]
        distance = np.linalg.norm(delta)
        free = max(25, time - last.time - last.duration)
        # A generous emergency ceiling; normal spacing is selected by the learned motion/critic scores.
        ceiling = min(600, (350 + settings["od"] * 160) * free / 1000 + radius)
        if distance > ceiling:
            shift = delta * (1 - ceiling / distance)
            curve -= shift
            curve = fit_curve_bounds(curve, margin)
            if slider and np.linalg.norm(curve[0] - [last.end_x, last.end_y]) > ceiling + 1:
                return None
    curve = np.rint(curve)
    length = curve_length(curve.tolist()) if slider else 0
    available = max(0, next_time - time - 35)
    duration = min(duration_beats * beat, available)
    fitting = [value * beat for value in DURATION_BEATS if value * beat <= duration + 1e-6]
    duration = max(fitting, default=0)
    if not slider or length < 20 or duration < beat * DURATION_BEATS[0]:
        curve[:] = curve[0]
        obj = HitObject(
            float(curve[0, 0]),
            float(curve[0, 1]),
            time,
            end_x=float(curve[0, 0]),
            end_y=float(curve[0, 1]),
            new_combo=combo,
            beat_length=beat,
        )
    else:
        if curve_diagnostics(curve, radius)["unsafe"]:
            return None
        velocity = length * repeats * beat / (140 * duration)
        if not 0.1 <= velocity <= 10:
            return None
        tail = curve[-1] if repeats % 2 else curve[0]
        obj = HitObject(
            float(curve[0, 0]),
            float(curve[0, 1]),
            time,
            True,
            float(duration),
            float(tail[0]),
            float(tail[1]),
            True,
            curve.tolist(),
            repeats,
            beat,
            combo,
            length,
        )
    normalized = (curve / [512, 384]).reshape(-1).astype(np.float32)
    motion = motion_numpy(
        normalized,
        obj.duration / 1000,
        obj.repeats,
        obj.slider,
        time / 1000,
        history_context(history),
    )
    compared = 4 if slider else 1
    repair = float(
        np.linalg.norm(curve[:compared] - original[:compared], axis=1).mean() / 512
    ) + float(slider and not obj.slider)
    return obj, normalized, motion, repair


def readability_cost(obj, history, cs, ar):
    radius = circle_radius(cs)
    approach = 1800 - 120 * ar if ar <= 5 else 1200 - 150 * (ar - 5)
    penalty = 0.0
    for previous in history[-24:]:
        gap = obj.time - previous.time
        if gap > approach or gap < 0:
            continue
        distance = math.hypot(obj.x - previous.x, obj.y - previous.y)
        # Intentional stacks remain possible; ambiguous partial overlaps get a soft penalty.
        if radius * 0.4 < distance < radius * 1.8:
            penalty += (1 - gap / approach) * (1 - distance / (radius * 1.8))
    return penalty


def quality_report(objects, settings):
    if not objects:
        return {"objects": 0}
    speeds, jumps, overlaps = [], [], 0
    for index, obj in enumerate(objects[1:], start=1):
        previous = objects[index - 1]
        free = obj.time - previous.time - previous.duration
        distance = math.hypot(obj.x - previous.end_x, obj.y - previous.end_y)
        speeds.append(distance / max(free / 1000, 0.025))
        jumps.append(distance)
        overlaps += int(free < 0)
    curves = [
        curve_diagnostics(obj.curve, circle_radius(settings["cs"]))
        for obj in objects
        if obj.slider and obj.curve
    ]
    return {
        "objects": len(objects),
        "sliders": sum(obj.slider for obj in objects),
        "slider_shape_issues": sum(curve["unsafe"] for curve in curves),
        "slider_detour_p95": (
            float(np.percentile([curve["detour"] for curve in curves], 95)) if curves else 0
        ),
        "temporal_overlaps": overlaps,
        "jump_p95_pixels": float(np.percentile(jumps, 95)) if jumps else 0,
        "aim_speed_p95": float(np.percentile(speeds, 95)) if speeds else 0,
        "partial_overlap_cost": float(
            np.mean(
                [
                    readability_cost(
                        obj, objects[max(0, index - 24) : index], settings["cs"], settings["ar"]
                    )
                    for index, obj in enumerate(objects)
                ]
            )
        ),
        "note": "Diagnostics, not an official star rating or a human playability score.",
    }
