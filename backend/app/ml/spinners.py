"""Rule-based spinners for generated maps.

Training ignores spinners (they are not part of the object sequence the model learns), so
the generator places them itself in the two situations where mappers use them most:

* a build-up before a drop: a quieter, audible stretch followed by a clear loudness jump;
  the spinner covers about one bar and ends with recovery time before the drop note;
* a long audible gap the rhythm left empty (for example a sustained outro), which would
  otherwise only be a break although the music keeps playing.

Spinners are spans (start, end) in milliseconds. They never overlap objects.
"""

import math
import numpy as np
from scipy.ndimage import uniform_filter1d
from .audio import FRAME_MS
from ..timing import red_points, timing_at

DROP_JUMP_DB = 5.0
BUILD_UP_MAX_DB = -4.0
MIN_SPACING_MS = 30000
MIN_SPINNER_MS = 1000
MAX_SPINNER_MS = 3200
LEAD_BEATS = 0.5  # free time between the previous object's end and the spinner
FILL_GAP_BEATS = 8  # empty audible gaps at least this long become spinners
FILL_START_MS = 5000  # no gap spinner within the first seconds after the first object


def recovery_ms(beat, settings):
    """Time between the spinner end and the next object, at least one beat."""
    ar = float(settings.get("ar", 9))
    preempt = 1200 + 600 * (5 - ar) / 5 if ar < 5 else 1200 - 750 * (ar - 5) / 5
    return max(beat, 0.5 * preempt, 300.0)


def grid_point(timing, time):
    """Red line whose grid covers `time`; the first one also extends backwards."""
    points = red_points(timing)
    if points and time < points[0].time:
        return points[0]
    return timing_at(points, time)


def beat_at(timing, time):
    point = grid_point(timing, time)
    return point.beat_length if point.beat_length > 0 else 500.0


def lead_ms(beat):
    """Free time between the previous object's end and a spinner."""
    return max(LEAD_BEATS * beat, 150.0)


def snap_to_beat(time, timing):
    point = grid_point(timing, time)
    return float(point.time + round((time - point.time) / point.beat_length) * point.beat_length)


def drop_spinners(loudness, active, timing, start, end, settings):
    """Spans before loudness jumps; each span is (start, end, resume_at, lead)."""
    if not settings.get("spinners", True) or len(loudness) < 4:
        return []
    second = max(1, round(1000 / FRAME_MS))
    smooth = uniform_filter1d(np.asarray(loudness, np.float64), second, mode="nearest")
    total = np.concatenate([[0.0], np.cumsum(smooth)])
    frames = np.arange(len(smooth))
    left = np.maximum(0, frames - 4 * second)
    right = np.minimum(len(smooth), frames + 2 * second)
    # Mean loudness of the 4 s before and of the 2 s from each frame on.
    before = (total[frames] - total[left]) / np.maximum(1, frames - left)
    after = (total[right] - total[frames]) / np.maximum(1, right - frames)
    jump = after - before
    spans, last = [], -math.inf
    frame = 4 * second
    while frame < len(jump) - 2 * second:
        if not (jump[frame] >= DROP_JUMP_DB and before[frame] <= BUILD_UP_MAX_DB):
            frame += 1
            continue
        # The strongest jump of this rise is the drop.
        peak = frame + int(np.argmax(jump[frame : frame + second]))
        drop = snap_to_beat(peak * FRAME_MS, timing)
        beat = beat_at(timing, drop)
        span_end = drop - recovery_ms(beat, settings)
        length = float(np.clip(4 * beat, MIN_SPINNER_MS, MAX_SPINNER_MS))
        span_start = span_end - length
        first, final = int(span_start / FRAME_MS), int(span_end / FRAME_MS)
        if (
            span_start >= start + 3000
            and drop <= end
            and span_start - last >= MIN_SPACING_MS
            and 0 <= first < final < len(active)
            and active[first:final].mean() > 0.9
        ):
            spans.append((span_start, span_end, drop, lead_ms(beat_at(timing, span_start))))
            last = span_start
        frame = peak + 2 * second
    return spans


def fill_spinners(objects, active, timing, end, settings, existing=()):
    """Spinners in long audible gaps between objects and after the last object."""
    if not settings.get("spinners", True) or not objects:
        return []
    spans = []
    taken = [span[:2] for span in existing]
    boundaries = [(obj.time + obj.duration, following.time) for obj, following in zip(objects, objects[1:])]
    last_end = objects[-1].time + objects[-1].duration
    frame = int(last_end / FRAME_MS)
    while frame < len(active) and active[frame]:
        frame += 1
    audible_end = min(end, frame * FRAME_MS - 100)
    boundaries.append((last_end, audible_end + recovery_ms(beat_at(timing, last_end), settings)))
    for release, following in boundaries:
        beat = beat_at(timing, release)
        if following - release < FILL_GAP_BEATS * beat:
            continue
        span_start = release + lead_ms(beat)
        span_end = min(following - recovery_ms(beat, settings), span_start + MAX_SPINNER_MS * 2)
        if span_start < objects[0].time + FILL_START_MS or span_end - span_start < MIN_SPINNER_MS:
            continue
        first, final = int(span_start / FRAME_MS), int(span_end / FRAME_MS)
        if final > len(active) or active[first:final].mean() < 0.95:
            continue  # silence: a break, not a spinner
        if any(span_start < right + MIN_SPACING_MS / 3 and left < span_end + MIN_SPACING_MS / 3 for left, right in taken):
            continue
        spans.append((span_start, span_end))
        taken.append((span_start, span_end))
    return spans
