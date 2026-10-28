"""Explicit timing evidence shared by import, generation and musical-time features."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TimingPoint:
    time: float
    beat_length: float
    inherited: bool = False
    meter: int | None = None
    beat_confidence: float = 1.0
    meter_confidence: float = 0.0
    bar_origin: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.time + self.beat_length) or self.beat_length == 0:
            raise ValueError("Invalid timing point")
        if self.inherited != (self.beat_length < 0):
            raise ValueError("Timing point sign disagrees with inherited flag")
        if self.meter is not None and (type(self.meter) is not int or self.meter < 1):
            raise ValueError("Timing meter must be a positive integer or unknown")
        if self.bar_origin is not None and not math.isfinite(self.bar_origin):
            raise ValueError("Invalid bar origin")
        if not 0 <= self.beat_confidence <= 1 or not 0 <= self.meter_confidence <= 1:
            raise ValueError("Timing confidence must be between zero and one")
        if self.meter_confidence and (self.meter is None or self.bar_origin is None):
            raise ValueError("Metric confidence requires an explicit meter and bar origin")


def red_points(timing):
    return sorted((point for point in timing if not point.inherited), key=lambda point: point.time)


def timing_at(timing, time):
    points = red_points(timing)
    if not points:
        raise ValueError("No beat timing is available")
    if time < points[0].time:
        # Match the parser/osu! timing default before the first red point. This is
        # a technical duration basis, never evidence of a beat phase or a meter.
        return TimingPoint(0, 500, beat_confidence=0)
    return next(point for point in reversed(points) if point.time <= time)


def beat_distance(timing, start, end):
    points = red_points(timing)
    boundaries = [start] + [point.time for point in points if start < point.time < end] + [end]
    return sum(
        (right - left) / timing_at(points, left).beat_length
        for left, right in zip(boundaries, boundaries[1:])
    )
