"""Where do the training maps' red lines sit relative to the decoded audio?

Measured on 14 training maps (5.6-6.6 stars): the mel-flux onset phase lies 11.8 ms after
the human grid in the median, with a spread of about +-15 ms and single maps 58 ms off.
The model learns this convention (its beat and onset heads are trained on these grids),
so it is measured on the whole dataset and stored with the model:

* `timing_convention_ms` (the negative median) lets audio-only timing detection place the
  grid where this model expects it;
* maps far from the median (other decoder delay, sloppy timing, a different subdivision
  on the red line) teach a blurred onset target; they can be left out of training.

The offset of one map is the phasor phase of the mel onset envelope at each long red line
section's tempo, relative to that red line, weighted by the section length.
"""

import math

import numpy as np

from .audio import FRAME_MS
from .tempo_grid import grid_phase, onset_envelopes
from ..timing import red_points

MIN_SECTION_MS = 15000


def map_offset(envelope, timing, frame_ms=FRAME_MS):
    """Median-weighted onset phase minus red-line phase (ms), None without long sections."""
    points = red_points(timing)
    values, weights = [], []
    duration = len(envelope) * frame_ms
    for index, point in enumerate(points):
        end = points[index + 1].time if index + 1 < len(points) else duration
        if end - max(point.time, 0) < MIN_SECTION_MS or point.beat_length <= 0:
            continue
        first = max(0, int(point.time / frame_ms))
        last = min(len(envelope), int(end / frame_ms))
        phase, lock = grid_phase(envelope, frame_ms, 60000 / point.beat_length, first, last)
        if lock < 0.05:
            continue
        beat = point.beat_length
        values.append(((phase - point.time) % beat + beat / 2) % beat - beat / 2)
        weights.append(end - max(point.time, 0))
    if not values:
        return None
    order = np.argsort(values)
    cumulative = np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(values)[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def audit_offsets(items, context=None):
    """Offset per map id (None when not measurable); one envelope per audio file."""
    envelopes, offsets = {}, {}
    ordered = sorted(items, key=lambda item: str(item.get("audio_hash", "")))
    for index, item in enumerate(ordered):
        if context is not None and index % 200 == 0:
            context.report(0.12, f"Timing der Trainingsmaps prüfen {index}/{len(items)}")
        key = item.get("audio_hash") or id(item["mel"])
        if key not in envelopes:
            envelopes = {key: onset_envelopes(np.asarray(item["mel"], np.float32))[0]}
        offsets[item["map_id"]] = map_offset(envelopes[key], item.get("timing") or [])
    return offsets


def convention(offsets):
    known = [value for value in offsets.values() if value is not None and math.isfinite(value)]
    if len(known) < 5:
        return None
    median = float(np.median(known))
    low, high = np.percentile(known, [25, 75])
    return {"median_ms": median, "iqr_ms": float(high - low), "maps": len(known)}


def outliers(offsets, median, tolerance):
    return {
        map_id
        for map_id, value in offsets.items()
        if value is not None and abs(value - median) > tolerance
    }
