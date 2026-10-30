"""Playback events for our generated maps (Soft samples, timing-point volume)."""

from bisect import bisect_right
from .beatmaps import parse

ADDITIONS = ((2, "hitwhistle"), (4, "hitfinish"), (8, "hitclap"))


def sound_events(map_path):
    # Use the exported map's rounded timings and actual slider velocity, rather
    # than the model's pre-export floating point predictions.
    beatmap = parse(map_path)
    tick_rate = max(0.1, beatmap.difficulty.get("SliderTickRate", 1))
    points = beatmap.sound_points
    starts = [point[0] for point in points]
    events = []

    def volume_at(time):
        # osu! uses the timing point active at the sound's own time (slider ends too).
        index = bisect_right(starts, time + 1e-6) - 1
        return (points[index][1] if index >= 0 else points[0][1]) / 100 if points else 0.7

    def add(time, sample, end=None):
        event = {"time": time / 1000, "sample": sample, "volume": volume_at(time)}
        if end is not None:
            event["end"] = end / 1000
        events.append(event)

    def hit(time, sound):
        add(time, "hitnormal")
        for bit, sample in ADDITIONS:
            if sound & bit:
                add(time, sample)

    for obj in beatmap.objects:
        if not obj.slider or obj.duration <= 0:
            hit(obj.time, obj.hitsound)
            continue
        edges = obj.edge_sounds or [0] * (obj.repeats + 1)
        hit(obj.time, edges[0])
        add(obj.time, "sliderslide", obj.time + obj.duration)
        span = obj.duration / obj.repeats
        tick_interval = obj.beat_length / tick_rate
        ticks = []
        tick = tick_interval
        # osu! suppresses ticks within 10 ms of the path end. Reverse spans
        # traverse the same tick positions in the opposite order.
        while tick < span - 10:
            ticks.append(tick)
            tick += tick_interval
        for repeat in range(obj.repeats):
            start = obj.time + repeat * span
            for tick in ticks:
                add(start + (span - tick if repeat % 2 else tick), "slidertick")
            hit(start + span, edges[repeat + 1] if repeat + 1 < len(edges) else 0)

    # A spinner gives its feedback when it ends.
    for (_, end), sound in zip(beatmap.ignored, beatmap.spinner_sounds):
        hit(end, sound)

    return sorted(events, key=lambda event: event["time"])
