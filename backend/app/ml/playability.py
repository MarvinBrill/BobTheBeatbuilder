import math
import numpy as np
from ..beatmaps import HitObject
from ..curve_geometry import curve_length
from .movement import motion_numpy, history_context, bezier_points
from .architecture import DURATION_BEATS
from .slider_geometry import fit_curve_bounds, curve_diagnostics
from .generation_trace import object_snapshot
from .movement_guidance import min_slider_length, target_velocity, velocity_band
from ..tiers import rules_for, settings_stars


def circle_radius(cs):
    return 54.4 - 4.48 * cs


def candidate_object(
    geometry,
    time,
    next_time,
    beat,
    slider,
    duration_beats,
    repeats,
    combo,
    settings,
    history,
    trace=None,
    candidate_id=None,
):
    radius = circle_radius(settings["cs"])
    margin = math.ceil(radius) + 1
    curve = np.asarray(geometry, dtype=float).reshape(4, 2) * [512, 384]
    original = curve.copy()
    if trace is not None:
        trace.record(
            "candidate_proposal",
            time=time,
            candidate_id=candidate_id,
            kind="slider" if slider else "circle",
            duration_ms=duration_beats * beat,
            duration_beats=duration_beats,
            repeats=repeats,
            geometry_pixels=original,
            next_time=next_time,
            beat_ms=beat,
        )
    if not slider:
        curve[:] = curve[0]
    curve = fit_curve_bounds(curve, margin)
    if trace is not None and not np.array_equal(
        curve, original if slider else np.tile(original[0], (4, 1))
    ):
        trace.record("candidate_guard", time=time, candidate_id=candidate_id, reason="bounds_fit")
    if history:
        last = history[-1]
        delta = curve[0] - [last.end_x, last.end_y]
        distance = np.linalg.norm(delta)
        free = max(25, time - last.time - last.duration)
        # Conservative supported-style ceiling, independent of hit-window difficulty (OD).
        # Normal spacing is selected by the learned motion and compatibility scores.
        ceiling = min(600, 1500 * free / 1000 + radius)
        if distance > ceiling:
            if trace is not None:
                trace.record(
                    "candidate_guard",
                    time=time,
                    candidate_id=candidate_id,
                    reason="movement_ceiling",
                    distance=distance,
                    ceiling=ceiling,
                )
            shift = delta * (1 - ceiling / distance)
            curve -= shift
            curve = fit_curve_bounds(curve, margin)
            if slider and np.linalg.norm(curve[0] - [last.end_x, last.end_y]) > ceiling + 1:
                if trace is not None:
                    trace.record(
                        "candidate_rejected",
                        time=time,
                        candidate_id=candidate_id,
                        reason="movement_ceiling_after_bounds",
                    )
                return None
    curve = np.rint(curve)
    length = curve_length(curve.tolist()) if slider else 0
    available = max(0, next_time - time - 35)
    duration = min(duration_beats * beat, available)
    fitting = [value * beat for value in DURATION_BEATS if value * beat <= duration + 1e-6]
    duration = max(fitting, default=0)
    slowest, fastest = velocity_band(settings)
    wanted = target_velocity(settings)
    if slider and length > 0 and duration > 0 and wanted is not None:
        velocity = length * repeats * beat / (140 * duration)
        if not slowest <= velocity <= fastest:
            # Keep the learned shape but scale it to the requested velocity instead of
            # writing a 30 px slider that crawls for a whole beat at 0.2x. One shared
            # velocity also keeps the number of green lines down.
            scaled = fit_curve_bounds((curve - curve[0]) * (wanted / velocity) + curve[0], margin)
            scaled = np.rint(scaled)
            if trace is not None:
                trace.record(
                    "candidate_guard",
                    time=time,
                    candidate_id=candidate_id,
                    reason="slider_velocity_scaled",
                    velocity=velocity,
                    wanted=wanted,
                )
            curve, length = scaled, curve_length(scaled.tolist())
    if trace is not None and slider and duration < duration_beats * beat - 1e-6:
        trace.record(
            "candidate_guard",
            time=time,
            candidate_id=candidate_id,
            reason="next_head_duration_limit",
            available_ms=available,
            requested_ms=duration_beats * beat,
            final_ms=duration,
        )
    shortest = min_slider_length(settings)
    if not slider or length < shortest or duration < beat * DURATION_BEATS[0]:
        if trace is not None and slider:
            trace.record(
                "slider_fallback",
                time=time,
                candidate_id=candidate_id,
                reason="short_path" if length < shortest else "insufficient_duration",
            )
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
        diagnostics = curve_diagnostics(curve, radius)
        if diagnostics["unsafe"]:
            if trace is not None:
                trace.record(
                    "candidate_rejected",
                    time=time,
                    candidate_id=candidate_id,
                    reason="conservative_slider_shape",
                    diagnostics=diagnostics,
                )
            return None
        velocity = length * repeats * beat / (140 * duration)
        if not slowest - 0.01 <= velocity <= fastest + 0.01:
            if trace is not None:
                trace.record(
                    "candidate_rejected",
                    time=time,
                    candidate_id=candidate_id,
                    reason="slider_velocity",
                    velocity=velocity,
                )
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
    if trace is not None:
        trace.record(
            "candidate_final",
            time=time,
            candidate_id=candidate_id,
            object=object_snapshot(obj),
            geometry_pixels=curve,
            repair=repair,
            motion=motion,
        )
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


LOW_DIFFICULTY_HISTORY = 8


def low_difficulty_cost(obj, history, beat, settings):
    """Readability for Easy/Normal targets (0 for harder maps).

    * Reading order: a note that sits visually closer to the object before its predecessor
      than to the predecessor itself is read in the wrong order by new players, who follow
      the closest circle rather than the numbers (osu! forum poll t/2247525, 2026).
    * Distance snap: the Ranking Criteria ask Easy/Normal for time-distance equality
      within 0.8-1.2x (Easy) / 0.8-1.3x (Normal) of the map's spacing.
    """
    rules = rules_for(settings_stars(settings))
    if rules is None or rules.distance_snap is None or not history:
        return 0.0
    radius = circle_radius(settings["cs"])
    previous = history[-1]
    gap = (obj.time - previous.time - previous.duration) / max(beat, 1.0)
    spacing = math.hypot(obj.x - previous.end_x, obj.y - previous.end_y)
    cost = 0.0
    if len(history) >= 2 and gap <= 2.05 and spacing > 2 * radius:
        first = history[-2]
        decoy = min(
            math.hypot(obj.x - first.x, obj.y - first.y),
            math.hypot(obj.x - first.end_x, obj.y - first.end_y),
        )
        if 3 <= decoy < spacing:
            cost += 1 - decoy / spacing
    if 0.25 <= gap <= rules.distance_snap_max_beats + 0.01 and spacing >= 3:
        ratios = []
        recent = history[-LOW_DIFFICULTY_HISTORY - 1 :]
        for left, right in zip(recent, recent[1:]):
            beats = (right.time - left.time - left.duration) / max(beat, 1.0)
            distance = math.hypot(right.x - left.end_x, right.y - left.end_y)
            if 0.25 <= beats <= rules.distance_snap_max_beats + 0.01 and distance >= 3:
                ratios.append(distance / beats)
        if len(ratios) >= 3:
            typical = float(np.median(ratios))
            low, high = rules.distance_snap
            ratio = spacing / gap / max(typical, 1e-6)
            if ratio < low:
                cost += 3 * math.log(low / ratio)
            elif ratio > high:
                cost += 3 * math.log(ratio / high)
    return float(cost)


def quality_report(objects, settings):
    if not objects:
        return {
            key: 0
            for key in (
                "objects",
                "sliders",
                "slider_shape_issues",
                "slider_detour_p95",
                "temporal_overlaps",
                "jump_p95_pixels",
                "aim_speed_p95",
                "partial_overlap_cost",
            )
        }
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
