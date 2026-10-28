"""Playback events for our generated maps (Soft samples, 70% timing volume)."""

from .beatmaps import parse


def sound_events(map_path):
    # Use the exported map's rounded timings and actual slider velocity, rather
    # than the model's pre-export floating point predictions.
    beatmap = parse(map_path)
    tick_rate = max(0.1, beatmap.difficulty.get("SliderTickRate", 1))
    events = []

    def add(time, sample, end=None):
        event = {"time": time / 1000, "sample": sample, "volume": 0.7}
        if end is not None:
            event["end"] = end / 1000
        events.append(event)

    for obj in beatmap.objects:
        add(obj.time, "hitnormal")
        if not obj.slider or obj.duration <= 0:
            continue
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
            add(start + span, "hitnormal")

    return sorted(events, key=lambda event: event["time"])
