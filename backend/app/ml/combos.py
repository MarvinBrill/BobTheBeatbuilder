"""New combos on musical boundaries.

The model's combo head is a per-object binary decision with a positive rate of roughly
one in eight, so ``logit > 0`` almost never fires and whole sections end up in one combo.
Mappers start a combo with the bar or every second bar, after breaks and spinners, and on
the first note after a longer rest. The model's own new combos are kept where they leave
at least two objects per combo.
"""

import math
from ..timing import red_points, timing_at

MIN_COMBO = 2
BAR_BEATS = 4  # the meter is usually unknown here; four beats is the export default too
REST_BEATS = 2.0
DENSE_BAR = 6  # objects in a bar from which every bar gets its own combo


def grid_point(timing, time):
    points = red_points(timing)
    if points and time < points[0].time:
        return points[0]
    return timing_at(points, time)


def bar_index(timing, time):
    """(red line, bar number) of `time`; bars count from the red line or its bar origin."""
    point = grid_point(timing, time)
    meter = point.meter or BAR_BEATS
    origin = point.bar_origin if point.bar_origin is not None else point.time
    return point.time, math.floor((time - origin) / (point.beat_length * meter) + 1e-6)


def assign_combos(objects, timing, spinners=(), settings=None):
    """Set `new_combo` in place; returns the number of combos."""
    if not objects:
        return 0
    starts = sorted(start for start, *_ in spinners)
    length, previous_bar, count = 0, None, 0
    for index, obj in enumerate(objects):
        beat = grid_point(timing, obj.time).beat_length
        bar_ms = beat * BAR_BEATS
        window = [other for other in objects[index : index + 12] if other.time < obj.time + bar_ms]
        bars = 1 if len(window) >= DENSE_BAR else 2
        bar = bar_index(timing, obj.time)
        previous = objects[index - 1] if index else None
        rest = previous is not None and obj.time - (previous.time + previous.duration) >= REST_BEATS * beat
        after_spinner = previous is not None and any(previous.time < start < obj.time for start in starts)
        crossed = previous_bar is not None and bar != previous_bar and (
            bars == 1 or bar[0] != previous_bar[0] or bar[1] % 2 == 0 or bar[1] - previous_bar[1] >= 2
        )
        start = (
            index == 0
            or after_spinner
            or (rest and length >= MIN_COMBO)
            or (crossed and length >= MIN_COMBO)
            or (obj.new_combo and length >= MIN_COMBO)
        )
        obj.new_combo = bool(start)
        if start:
            length, count = 0, count + 1
        length += 1
        previous_bar = bar
    return count
