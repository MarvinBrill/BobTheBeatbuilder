"""Mapset finishing for generated drafts: kiai, preview point, breaks, hitsounds and volume.

The model only learns hit objects. Everything a mapper adds afterwards in the editor was
missing from the exported drafts: no kiai, no preview point, no break periods (HP kept
draining through long rests), no hitsound additions and one flat 70 % volume. This module
adds those layers with the conventions the osu! mapping community documents:

* Kibbleru, "[Guide] Hitsounding!" (osu! forum t/306177): claps follow the snare on the
  2nd and 4th beat, finishes go on cymbal crashes, whistles mark downbeats or stand in for
  claps where the song has no snare; similar sections share their hitsounds.
* Ranking Criteria (general): kiai "should start on a sound in the music"; the first
  uninherited timing point cannot toggle kiai; the preview point must be set; slider ends,
  reverses and spinner ends should give hitsound feedback; hitsounds must be audible.
* osu!lazer break periods: at least 200 ms after the previous object, at least the
  approach time (minimum 450 ms) before the next one, and at least 650 ms long.

Everything is derived from the audio loudness and the beat grid, so it is a starting point
for the mapper, not a replacement for hitsounding by ear.
"""

from dataclasses import dataclass, field
import math
import numpy as np
from scipy.ndimage import uniform_filter1d
from .audio import FRAME_MS
from .rhythm_guidance import loudness_profile
from ..tiers import preempt_ms
from ..timing import red_points, timing_at

WHISTLE, FINISH, CLAP = 2, 4, 8
HITSOUND_MODES = ("off", "basic", "full")

# Kiai: sustained loud sections, relative to the loud parts of the song (dB).
KIAI_THRESHOLDS = (-3.0, -2.0, -1.5, -1.0)
KIAI_SMOOTH_MS = 4000
KIAI_MIN_MS = 8000
KIAI_MIN_BEATS = 16
KIAI_MERGE_MS = 2000
KIAI_MAX_SHARE = 0.5  # a song that is loud all the time has no section to highlight
KIAI_MIN_CONTRAST_DB = 3.0

# Volume per section (timing point volume in percent).
VOLUME_CALM, VOLUME_NORMAL, VOLUME_KIAI = 50, 65, 80
CALM_DB = -9.0
SECTION_MIN_BARS = 2

# Break periods as osu!lazer creates them in the editor.
BREAK_GAP_BEFORE = 200
BREAK_GAP_AFTER = 450
BREAK_MIN = 650

FINISH_MIN_BEATS = 2.0
PHRASE_BARS = 4
DROP_JUMP_DB = 5.0
ON_BEAT_TOLERANCE = 0.07  # beats
BAR_BEATS = 4  # export default when the meter is unknown


@dataclass
class Finish:
    kiai: list = field(default_factory=list)  # [(start, end)] in ms
    breaks: list = field(default_factory=list)  # [(start, end)] in ms
    preview_time: int = -1
    # State changes [(time, volume, kiai)]; the first entry is the initial state.
    sound_states: list = field(default_factory=list)
    spinner_sounds: list = field(default_factory=list)  # additions per spinner (sorted order)
    summary: dict = field(default_factory=dict)

    def export(self):
        """The fields `beatmaps.write` reads."""
        return {
            "preview_time": self.preview_time,
            "breaks": list(self.breaks),
            "sound_states": list(self.sound_states),
            "spinner_sounds": list(self.spinner_sounds),
        }


def grid_point(timing, time):
    points = red_points(timing)
    if points and time < points[0].time:
        return points[0]
    return timing_at(points, time)


def metric_position(timing, time):
    """(beat in bar, bar number, distance to the nearest beat in beats, beat ms, meter)."""
    point = grid_point(timing, time)
    beat = point.beat_length if point.beat_length > 0 else 500.0
    meter = point.meter or BAR_BEATS
    origin = point.bar_origin if point.bar_origin is not None else point.time
    beats = (time - origin) / beat
    nearest = round(beats)
    return int(nearest % meter), math.floor(nearest / meter), abs(beats - nearest), beat, meter


def is_downbeat(timing, time):
    in_bar, _, distance, _, _ = metric_position(timing, time)
    return in_bar == 0 and distance <= ON_BEAT_TOLERANCE


def frame_value(values, time):
    if not len(values):
        return 0.0
    return float(values[int(np.clip(round(time / FRAME_MS), 0, len(values) - 1))])


def object_spans(objects, spinners):
    spans = [(obj.time, obj.time + obj.duration) for obj in objects]
    spans += [(float(start), float(end)) for start, end in spinners]
    return sorted(spans)


def runs(mask):
    """[(first, last_exclusive)] of True runs."""
    padded = np.concatenate([[False], np.asarray(mask, bool), [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[::2], changes[1::2]))


def nearest_head(objects, timing, target, before, after):
    """Object head near `target`, preferring downbeats (a kiai starts on a sound)."""
    best = None
    for obj in objects:
        if obj.time < target - before or obj.time > target + after:
            continue
        _, _, _, beat, meter = metric_position(timing, obj.time)
        score = abs(obj.time - target) + (0 if is_downbeat(timing, obj.time) else beat * meter / 2)
        if best is None or score < best[0]:
            best = (score, obj.time)
    return None if best is None else best[1]


def kiai_sections(objects, spinners, timing, loudness, audible):
    """Sustained loud sections, starting and ending on object heads near bar lines."""
    if not objects or not len(loudness):
        return []
    spans = object_spans(objects, spinners)
    mapped_start, mapped_end = spans[0][0], max(end for _, end in spans)
    first = int(max(0, mapped_start / FRAME_MS))
    last = int(min(len(loudness), math.ceil(mapped_end / FRAME_MS)))
    if last - first < 4:
        return []
    window = max(1, round(KIAI_SMOOTH_MS / FRAME_MS))
    smooth = uniform_filter1d(np.asarray(loudness, np.float64), window, mode="nearest")
    present = uniform_filter1d(np.asarray(audible, np.float64), window, mode="nearest") > 0.5
    beat = grid_point(timing, mapped_start).beat_length
    minimum = max(KIAI_MIN_MS, KIAI_MIN_BEATS * beat) / FRAME_MS
    merge = KIAI_MERGE_MS / FRAME_MS
    chosen = []
    for threshold in KIAI_THRESHOLDS:
        mask = np.zeros(len(smooth), bool)
        mask[first:last] = (smooth[first:last] >= threshold) & present[first:last]
        merged = []
        for start, end in runs(mask):
            if merged and start - merged[-1][1] < merge:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        merged = [(start, end) for start, end in merged if end - start >= minimum]
        covered = sum(end - start for start, end in merged)
        if not merged:
            return []
        if covered <= KIAI_MAX_SHARE * (last - first):
            outside = np.ones(last - first, bool)
            for start, end in merged:
                outside[start - first : end - first] = False
            rest = smooth[first:last][outside & present[first:last]]
            if len(rest) and np.median(rest) <= threshold - KIAI_MIN_CONTRAST_DB:
                chosen = merged
            break
    sections = []
    for start, end in chosen:
        start_ms, end_ms = start * FRAME_MS, end * FRAME_MS
        _, _, _, beat, meter = metric_position(timing, start_ms)
        bar = beat * meter
        kiai_start = nearest_head(objects, timing, start_ms, 2 * bar, bar)
        if kiai_start is None:
            continue
        kiai_end = nearest_head(objects, timing, end_ms, bar, 2 * bar)
        if kiai_end is None or kiai_end <= kiai_start:
            # No note at the section end (e.g. the map ends loud): one beat after the
            # last object inside the section.
            inside = [
                obj.time + obj.duration for obj in objects if kiai_start <= obj.time <= end_ms
            ]
            kiai_end = max(inside, default=kiai_start) + beat
        if kiai_end - kiai_start < minimum * FRAME_MS / 2:
            continue
        if sections and kiai_start <= sections[-1][1]:
            sections[-1] = (sections[-1][0], max(sections[-1][1], kiai_end))
        else:
            sections.append((float(kiai_start), float(kiai_end)))
    return sections


def preview_point(objects, timing, loudness, kiai):
    """The chorus start: the first kiai, otherwise the loudest 10 s of the mapped part."""
    if kiai:
        return int(round(kiai[0][0]))
    if not objects or not len(loudness):
        return -1
    window = max(1, round(10000 / FRAME_MS))
    smooth = uniform_filter1d(np.asarray(loudness, np.float64), window, mode="nearest")
    first = int(objects[0].time / FRAME_MS)
    last = int(min(len(smooth), (objects[-1].time + objects[-1].duration) / FRAME_MS))
    if last <= first:
        return int(round(objects[0].time))
    # uniform_filter1d centres the window: the loudest window starts half a window earlier.
    centre = first + int(np.argmax(smooth[first:last]))
    target = max(objects[0].time, (centre - window / 2) * FRAME_MS)
    _, _, _, beat, meter = metric_position(timing, target)
    head = nearest_head(objects, timing, target, 2 * beat * meter, 2 * beat * meter)
    return int(round(head if head is not None else target))


def break_periods(objects, spinners, ar):
    """Break periods for long rests, as the osu!lazer editor would insert them."""
    spans = object_spans(objects, spinners)
    after = max(BREAK_GAP_AFTER, preempt_ms(ar))
    breaks = []
    for (_, release), (following, _) in zip(spans, spans[1:]):
        start, end = release + BREAK_GAP_BEFORE, following - after
        if end - start >= BREAK_MIN:
            breaks.append((int(math.ceil(start)), int(math.floor(end))))
    return breaks


def loudness_jumps(loudness):
    """Times (ms) where the next 2 s are at least DROP_JUMP_DB louder than the last 4 s."""
    if len(loudness) < 8:
        return []
    second = max(1, round(1000 / FRAME_MS))
    values = np.asarray(loudness, np.float64)
    total = np.concatenate([[0.0], np.cumsum(values)])
    frames = np.arange(len(values))
    left = np.maximum(0, frames - 4 * second)
    right = np.minimum(len(values), frames + 2 * second)
    before = (total[frames] - total[left]) / np.maximum(1, frames - left)
    after = (total[right] - total[frames]) / np.maximum(1, right - frames)
    jump = after - before
    times, frame = [], 4 * second
    while frame < len(jump) - 2 * second:
        if jump[frame] < DROP_JUMP_DB:
            frame += 1
            continue
        peak = frame + int(np.argmax(jump[frame : frame + second]))
        times.append(peak * FRAME_MS)
        frame = peak + 2 * second
    return times


def sound_points(objects):
    """Every point that plays a hit sound: heads, slider repeats and slider tails."""
    points = []
    for index, obj in enumerate(objects):
        if obj.slider and obj.duration > 0:
            span = obj.duration / max(1, obj.repeats)
            for edge in range(obj.repeats + 1):
                points.append((obj.time + edge * span, index, edge))
        else:
            points.append((obj.time, index, None))
    return sorted(points, key=lambda point: point[0])


def in_spans(time, spans):
    return any(start - 1e-6 <= time < end - 1e-6 for start, end in spans)


def accent_times(points, timing, spinners, kiai, drops):
    """Finish candidates: kiai/phrase starts, the note after a spinner, loudness drops."""
    times = [point[0] for point in points]
    accents = set()

    def first_at_or_after(time, within):
        for value in times:
            if value >= time - 1 and value <= time + within:
                return value
        return None

    for start, _ in kiai:
        accents.add(start)
    for _, end in spinners:
        value = first_at_or_after(end, 4 * grid_point(timing, end).beat_length)
        if value is not None:
            accents.add(value)
    for drop in drops:
        beat = grid_point(timing, drop).beat_length
        # The loudness window sees a drop a little early: take the note on the nearest
        # downbeat, and leave drops that a kiai start or spinner already accents.
        if any(abs(drop - accent) <= 4 * beat for accent in accents):
            continue
        nearby = [value for value in times if drop - beat <= value <= drop + 2 * beat]
        downbeats = [value for value in nearby if is_downbeat(timing, value)]
        if downbeats or nearby:
            accents.add(min(downbeats or nearby, key=lambda value: abs(value - drop)))
    for start, end in kiai:
        _, first_bar, _, _, _ = metric_position(timing, start)
        for value in times:
            if not start <= value < end or not is_downbeat(timing, value):
                continue
            _, bar, _, _, _ = metric_position(timing, value)
            if (bar - first_bar) % PHRASE_BARS == 0:
                accents.add(value)
    return accents


def plan_hitsounds(objects, timing, spinners, loudness, kiai, mode="basic"):
    """Set `hitsound`/`edge_sounds` on the objects; returns counts per addition."""
    for obj in objects:
        obj.hitsound = 0
        obj.edge_sounds = [0] * (obj.repeats + 1) if obj.slider and obj.duration > 0 else None
    counts = {"whistle": 0, "finish": 0, "clap": 0}
    if mode == "off" or not objects:
        return counts
    points = sound_points(objects)
    accents = accent_times(points, timing, spinners, kiai, loudness_jumps(loudness))
    window = max(1, round(KIAI_SMOOTH_MS / FRAME_MS))
    smooth = (
        uniform_filter1d(np.asarray(loudness, np.float64), window, mode="nearest")
        if len(loudness)
        else loudness
    )
    last_finish = -math.inf
    for time, index, edge in points:
        in_bar, _, distance, beat, _ = metric_position(timing, time)
        on_beat = distance <= ON_BEAT_TOLERANCE
        kiai_now = in_spans(time, kiai)
        calm = not kiai_now and frame_value(smooth, time) < CALM_DB
        sound = 0
        if time in accents and time - last_finish >= FINISH_MIN_BEATS * beat - 1:
            sound |= FINISH
            last_finish = time
        if on_beat and in_bar % 2 == 1 and not calm:
            sound |= CLAP  # backbeat: 2 and 4 in 4/4
        if on_beat and in_bar == 0 and not sound & FINISH and (calm or mode == "full"):
            sound |= WHISTLE  # downbeats; replaces the claps in calm sections
        obj = objects[index]
        if edge is None:
            obj.hitsound = sound
        else:
            obj.edge_sounds[edge] = sound
        counts["whistle"] += bool(sound & WHISTLE)
        counts["finish"] += bool(sound & FINISH)
        counts["clap"] += bool(sound & CLAP)
    return counts


def spinner_sounds(spinners, kiai, drops, timing, mode="basic"):
    """A finish at the end of spinners that lead into a drop or a kiai section."""
    sounds = []
    for _, end in sorted(spinners):
        beat = grid_point(timing, end).beat_length
        leads = any(
            end <= time <= end + 4 * beat for time in [start for start, _ in kiai] + list(drops)
        )
        sounds.append(FINISH if mode != "off" and leads else 0)
    return sounds


def section_volumes(objects, timing, loudness, kiai):
    """[(time, volume, kiai)] state changes: calm 50 %, normal 65 %, kiai 80 %."""
    if not objects:
        return []
    window = max(1, round(KIAI_SMOOTH_MS / FRAME_MS))
    smooth = (
        uniform_filter1d(np.asarray(loudness, np.float64), window, mode="nearest")
        if len(loudness)
        else loudness
    )
    classes = [
        VOLUME_CALM if frame_value(smooth, obj.time) < CALM_DB else VOLUME_NORMAL for obj in objects
    ]
    # Runs of one class; runs shorter than SECTION_MIN_BARS join the previous run.
    sections = []  # [start time, class]
    for obj, value in zip(objects, classes):
        if not sections or sections[-1][1] != value:
            sections.append([obj.time, value])
    smoothed = []
    for index, (start, value) in enumerate(sections):
        end = sections[index + 1][0] if index + 1 < len(sections) else math.inf
        _, _, _, beat, meter = metric_position(timing, start)
        if smoothed and end - start < SECTION_MIN_BARS * beat * meter:
            continue
        if smoothed and smoothed[-1][1] == value:
            continue
        smoothed.append([start, value])

    def volume_at(time):
        if in_spans(time, kiai):
            return VOLUME_KIAI
        current = smoothed[0][1]
        for start, value in smoothed:
            if start <= time + 1e-6:
                current = value
        return current

    first_red = red_points(timing)[0].time if red_points(timing) else objects[0].time
    boundaries = sorted(
        {min(first_red, objects[0].time)}
        | {start for start, _ in smoothed[1:]}
        | {value for span in kiai for value in span}
    )
    raw = [(float(time), volume_at(time), in_spans(time, kiai)) for time in boundaries]

    def bars(time, count=SECTION_MIN_BARS):
        _, _, _, beat, meter = metric_position(timing, time)
        return count * beat * meter

    kept = []  # [time, volume, kiai, toggles kiai]
    for index, (time, volume, on) in enumerate(raw):
        if not kept or kept[-1][2] != on:
            kept.append([time, volume, on, bool(kept)])
            continue
        following = raw[index + 1][0] if index + 1 < len(raw) else math.inf
        if following - time < bars(time):
            continue  # a volume change lasting less than two bars is noise
        previous = kept[-1]
        if previous[3] and time - previous[0] < bars(time):
            previous[1] = volume  # settle the volume on the kiai toggle right before
        elif previous[1] != volume:
            kept.append([time, volume, on, False])
    return [(time, volume, on) for time, volume, on, _ in kept]


def finish_map(objects, timing, spinners, mel, settings):
    """All finishing layers for one generated difficulty. Mutates the objects' hitsounds."""
    mel = np.asarray(mel, np.float32)
    loudness = loudness_profile(mel) if len(mel) else np.zeros(0, np.float32)
    audible = mel.max(axis=1) > -0.7 if len(mel) else np.zeros(0, bool)
    spinners = [tuple(span) for span in spinners]
    mode = settings.get("hitsounds", "basic")
    if mode not in HITSOUND_MODES:
        raise ValueError(f"Unknown hitsound mode {mode!r}")
    kiai = (
        kiai_sections(objects, spinners, timing, loudness, audible)
        if settings.get("kiai", True)
        else []
    )
    breaks = (
        break_periods(objects, spinners, settings.get("ar", 9))
        if settings.get("breaks", True)
        else []
    )
    counts = plan_hitsounds(objects, timing, spinners, loudness, kiai, mode)
    if mode == "off" or not objects:
        # Without additions the previous flat 70 % stays; kiai still toggles.
        start = min(red_points(timing)[0].time, objects[0].time) if objects else 0
        states = [(float(start), 70, False)]
        states += [(float(time), 70, on) for span in kiai for time, on in zip(span, (True, False))]
    else:
        states = section_volumes(objects, timing, loudness, kiai)
    result = Finish(
        kiai=kiai,
        breaks=breaks,
        preview_time=preview_point(objects, timing, loudness, kiai),
        sound_states=states,
        spinner_sounds=spinner_sounds(spinners, kiai, loudness_jumps(loudness), timing, mode),
    )
    spans = object_spans(objects, spinners)
    mapped = max(end for _, end in spans) - spans[0][0] if spans else 0
    result.summary = {
        "hitsounds": mode,
        "kiai_sections": len(kiai),
        "kiai_seconds": round(sum(end - start for start, end in kiai) / 1000, 1),
        "kiai_share": round(sum(end - start for start, end in kiai) / mapped, 3) if mapped else 0,
        "breaks": len(breaks),
        "preview_time": result.preview_time,
        "volumes": sorted({volume for _, volume, _ in states}),
        **{f"{name}s": count for name, count in counts.items()},
    }
    return result
