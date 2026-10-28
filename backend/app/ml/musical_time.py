"""Timing-derived decoder features, with explicit masks for missing musical evidence."""

from dataclasses import replace
import numpy as np
from ..timing import red_points

MUSICAL_TIME_FEATURES = 11


def disturbed_timing(timing, rng):
    """One coherent disturbance per example; never fabricate a confident bar phase."""
    points = red_points(timing)
    if rng is None or not points:
        return points
    mode = rng.random()
    if mode < 0.2:
        return []
    hide_meter = mode < 0.6
    phase_shift = float(rng.normal(0, 12)) if mode < 0.85 else 0.0
    tempo_scale = float(rng.uniform(0.985, 1.015)) if phase_shift else 1.0
    confidence = 0.75 if phase_shift else 1.0
    return [
        replace(
            point,
            time=point.time + phase_shift,
            beat_length=point.beat_length * tempo_scale,
            beat_confidence=point.beat_confidence * confidence,
            meter=None if hide_meter else point.meter,
            meter_confidence=0.0 if hide_meter else point.meter_confidence * confidence,
            bar_origin=(
                None if hide_meter or point.bar_origin is None else point.bar_origin + phase_shift
            ),
        )
        for point in points
    ]


def musical_time_features(seconds, timing):
    """Continuous beat count; local phase; tempo/confidence; meter phase/confidence.

    Channel 9 is a timing-derived downbeat likelihood, not an audio posterior.
    Neither pulse regularity nor triplet intervals supply metric confidence.
    """
    times = np.asarray(seconds, dtype=np.float64) * 1000
    result = np.zeros((*times.shape, MUSICAL_TIME_FEATURES), np.float32)
    points = red_points(timing)
    cumulative = 0.0
    for index, point in enumerate(points):
        end = points[index + 1].time if index + 1 < len(points) else np.inf
        valid = (times >= point.time) & (times < end)
        local = (times[valid] - point.time) / point.beat_length
        beat_phase = local % 1
        confidence = point.beat_confidence
        result[valid, 0] = (cumulative + local) / 256 if confidence else 0
        result[valid, 1] = np.sin(2 * np.pi * beat_phase) * confidence
        result[valid, 2] = np.cos(2 * np.pi * beat_phase) * confidence
        result[valid, 3] = np.log(point.beat_length / 500) * confidence
        result[valid, 4] = confidence
        if point.meter_confidence and point.meter is not None and point.bar_origin is not None:
            metric = point.meter_confidence
            bar_beats = ((times[valid] - point.bar_origin) / point.beat_length) % point.meter
            result[valid, 5] = np.sin(2 * np.pi * bar_beats / point.meter) * metric
            result[valid, 6] = np.cos(2 * np.pi * bar_beats / point.meter) * metric
            result[valid, 7] = point.meter / 16 * metric
            result[valid, 8] = metric
            distance = np.minimum(bar_beats, point.meter - bar_beats) * point.beat_length
            result[valid, 9] = np.maximum(0, 1 - distance / 35) * metric
        if index:
            result[valid, 10] = np.maximum(0, 1 - local) * confidence
        if np.isfinite(end):
            cumulative += (end - point.time) / point.beat_length
    return result
