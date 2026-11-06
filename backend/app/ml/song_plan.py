"""The song plan: timing, sections and accents, decided before any hit object exists.

A mapper times the song first, then listens for the structure (intro, verses, build-ups,
drops, breaks) and decides how each part should play. The generator used to do all of it
implicitly, object by object, with loudness rules for kiai and spinners on the side.

`build_plan` makes those decisions explicit and editable:

* timing: the red lines from `tempo_grid` (or the user's), with bar lines;
* sections on bar lines, from a self-similarity novelty curve plus loudness jumps,
  snapped to 4-bar phrases; each gets an intensity (loudness and onset density), a kind
  and a pattern style ("calm", "flow", "jumps", "streams", "mixed");
* repetitions: a later section that sounds like an earlier one (chorus 2 = chorus 1);
* accents: the strongest onsets of each section, which the rhythm should not skip.

The plan is JSON (`to_dict`/`from_dict`). The studio shows it for review; generation reads
it through `SongPlan.section_at` / `style_at` / `accents_between`.
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
import math

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from ..timing import TimingPoint, red_points
from .tempo_grid import onset_envelopes

PLAN_VERSION = 1
KINDS = ("intro", "verse", "build", "drop", "break", "outro", "silence")
STYLES = ("auto", "calm", "flow", "jumps", "streams", "mixed")
DEFAULT_STYLE = {
    "intro": "calm",
    "verse": "flow",
    "build": "streams",
    "drop": "jumps",
    "break": "calm",
    "outro": "calm",
    "silence": "calm",
}
# What a style asks of the rhythm and movement guides (multipliers on the song's targets,
# angle mix and the pattern vocabulary of the movement planner).
STYLE_PARAMS = {
    "calm": dict(
        density=0.7, spacing=0.7, sharp=0.25, linear=0.35, streams=0.5,
        patterns={"flow": 0.45, "linear": 0.25, "back_forth": 0.2, "zigzag": 0.1},
    ),
    "flow": dict(
        density=0.95, spacing=0.85, sharp=0.35, linear=0.25, streams=0.8,
        patterns={"flow": 0.4, "zigzag": 0.2, "triangle": 0.15, "back_forth": 0.15, "linear": 0.1},
    ),
    "jumps": dict(
        density=1.05, spacing=1.1, sharp=0.5, linear=0.15, streams=0.8,
        patterns={
            "back_forth": 0.3, "triangle": 0.2, "square": 0.2, "zigzag": 0.17, "flow": 0.08,
            "star": 0.05,
        },
    ),
    "streams": dict(
        density=1.15, spacing=0.9, sharp=0.35, linear=0.3, streams=1.6,
        patterns={"flow": 0.35, "zigzag": 0.2, "back_forth": 0.2, "square": 0.15, "linear": 0.1},
    ),
    "mixed": dict(
        density=1.0, spacing=1.0, sharp=0.4, linear=0.2, streams=1.0,
        patterns={
            "back_forth": 0.2, "flow": 0.25, "triangle": 0.15, "square": 0.15, "zigzag": 0.15,
            "linear": 0.07, "star": 0.03,
        },
    ),
}  # fmt: skip
PHRASE_BARS = 4
MIN_SECTION_BARS = 4
NOVELTY_KERNEL_BARS = 4
REPEAT_SIMILARITY = 0.6


@dataclass
class Section:
    start_ms: float
    end_ms: float
    kind: str = "verse"
    intensity: float = 0.5  # 0 calm .. 1 loudest, densest part of the song
    style: str = "auto"
    repeat_of: int | None = None  # index of an earlier section this one repeats
    bars: int = 0
    locked: bool = False  # the user marked it as approved; regeneration keeps it

    def resolved_style(self):
        return DEFAULT_STYLE.get(self.kind, "mixed") if self.style == "auto" else self.style


@dataclass
class SongPlan:
    timing: list
    sections: list
    accents: list = field(default_factory=list)  # ms, sorted
    accent_strength: list = field(default_factory=list)
    timing_source: str = "detected"
    timing_info: dict = field(default_factory=dict)
    duration_ms: float = 0.0
    version: int = PLAN_VERSION

    # --- lookups -----------------------------------------------------------------------
    def index_at(self, time):
        starts = [section.start_ms for section in self.sections]
        return max(0, bisect_right(starts, time) - 1) if self.sections else None

    def section_at(self, time):
        index = self.index_at(time)
        return self.sections[index] if index is not None else None

    def style_at(self, time):
        """Style parameters for `time`, scaled by the section's intensity."""
        section = self.section_at(time)
        if section is None:
            return dict(STYLE_PARAMS["mixed"], intensity=0.5, kind="verse", style="mixed")
        style = section.resolved_style()
        params = dict(STYLE_PARAMS.get(style, STYLE_PARAMS["mixed"]))
        scale = 0.85 + 0.3 * float(np.clip(section.intensity, 0, 1))
        params["density"] *= scale
        params["spacing"] *= scale
        params.update(intensity=section.intensity, kind=section.kind, style=style)
        return params

    def accents_between(self, start, end):
        """(time, strength) of accents with start < time < end."""
        first = bisect_right(self.accents, start)
        last = bisect_left(self.accents, end)
        return list(zip(self.accents[first:last], self.accent_strength[first:last]))

    def repeat_source(self, time):
        """(source_section, offset_ms) when `time` lies in a section repeating an earlier one."""
        section = self.section_at(time)
        if section is None or section.repeat_of is None:
            return None
        if not 0 <= section.repeat_of < len(self.sections):
            return None
        source = self.sections[section.repeat_of]
        offset = time - section.start_ms
        if offset >= source.end_ms - source.start_ms:
            return None
        return source, source.start_ms - section.start_ms

    # --- serialisation -----------------------------------------------------------------
    def to_dict(self):
        return {
            "version": self.version,
            "duration_ms": round(self.duration_ms, 1),
            "timing_source": self.timing_source,
            "timing_info": _plain(self.timing_info),
            "timing": [
                {
                    "time": round(float(point.time), 3),
                    "beat_length": float(point.beat_length),
                    "meter": None if point.meter is None else int(point.meter),
                    "bar_origin": None if point.bar_origin is None else float(point.bar_origin),
                    "beat_confidence": float(point.beat_confidence),
                    "meter_confidence": float(point.meter_confidence),
                }
                for point in self.timing
            ],
            "sections": [
                {
                    "start_ms": round(section.start_ms, 1),
                    "end_ms": round(section.end_ms, 1),
                    "kind": section.kind,
                    "intensity": round(float(section.intensity), 3),
                    "style": section.style,
                    "repeat_of": None if section.repeat_of is None else int(section.repeat_of),
                    "bars": int(section.bars),
                    "locked": bool(section.locked),
                }
                for section in self.sections
            ],
            "accents": [round(float(value), 1) for value in self.accents],
            "accent_strength": [round(float(value), 3) for value in self.accent_strength],
        }

    @classmethod
    def from_dict(cls, values):
        timing = []
        for point in values.get("timing") or []:
            meter = point.get("meter")
            origin = point.get("bar_origin")
            known = meter is not None and origin is not None
            timing.append(
                TimingPoint(
                    float(point["time"]),
                    float(point["beat_length"]),
                    meter=int(meter) if meter is not None else None,
                    beat_confidence=float(np.clip(point.get("beat_confidence", 1.0), 0, 1)),
                    meter_confidence=(
                        float(np.clip(point.get("meter_confidence", 0.75), 0, 1)) if known else 0.0
                    ),
                    bar_origin=float(origin) if known else None,
                )
            )
        sections = []
        for item in values.get("sections") or []:
            kind = item.get("kind", "verse")
            style = item.get("style", "auto")
            repeat = item.get("repeat_of")
            sections.append(
                Section(
                    float(item["start_ms"]),
                    float(item["end_ms"]),
                    kind if kind in KINDS else "verse",
                    float(np.clip(item.get("intensity", 0.5), 0, 1)),
                    style if style in STYLES else "auto",
                    int(repeat) if repeat is not None else None,
                    int(item.get("bars", 0)),
                    bool(item.get("locked", False)),
                )
            )
        sections.sort(key=lambda section: section.start_ms)
        accents = [float(value) for value in values.get("accents") or []]
        strength = [float(value) for value in values.get("accent_strength") or []]
        if len(strength) != len(accents):
            strength = [1.0] * len(accents)
        order = np.argsort(accents) if accents else []
        return cls(
            timing=timing,
            sections=sections,
            accents=[accents[i] for i in order],
            accent_strength=[strength[i] for i in order],
            timing_source=values.get("timing_source", "plan"),
            timing_info=dict(values.get("timing_info") or {}),
            duration_ms=float(values.get("duration_ms", 0.0)),
            version=int(values.get("version", PLAN_VERSION)),
        )


def _plain(value):
    """JSON-safe copy (NumPy scalars to Python numbers)."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# --------------------------------------------------------------------------------------
# Construction


def bar_lines(timing, duration_ms):
    """Bar start times over the song, from the red lines (meter 4 when unknown)."""
    points = red_points(timing)
    bars = []
    for index, point in enumerate(points):
        end = points[index + 1].time if index + 1 < len(points) else duration_ms
        origin = point.bar_origin if point.bar_origin is not None else point.time
        length = point.beat_length * (point.meter or 4)
        first = origin + math.ceil((max(point.time, 0.0) - origin) / length - 1e-9) * length
        bars.extend(np.arange(first, end - 1e-6, length).tolist())
    bars = [bar for bar in bars if 0 <= bar < duration_ms]
    if not bars or bars[0] > 0:
        bars = [0.0] + bars
    return np.asarray(bars + [duration_ms], np.float64)


def loudness_db(mel):
    from .rhythm_guidance import loudness_profile

    return loudness_profile(mel)


def bar_features(mel, frame_ms, bars):
    """Per bar: timbre (z-scored mean mel), loudness (dB), onset rate (1/s), activity."""
    mel = np.asarray(mel, np.float32)
    loud = loudness_db(mel)
    full, _, _ = onset_envelopes(mel)
    peaks, _ = find_peaks(full, height=0.2, distance=max(1, int(50 / frame_ms)))
    peak_times = peaks * frame_ms
    active = mel.max(axis=1) > -0.7
    timbre, level, rate, activity = [], [], [], []
    for left, right in zip(bars[:-1], bars[1:]):
        first = int(left / frame_ms)
        last = max(first + 1, min(len(mel), int(right / frame_ms)))
        timbre.append(mel[first:last].mean(axis=0))
        level.append(float(np.median(loud[first:last])))
        count = np.count_nonzero((peak_times >= left) & (peak_times < right))
        rate.append(count / max(0.001, (right - left) / 1000))
        activity.append(float(active[first:last].mean()))
    timbre = np.asarray(timbre, np.float64)
    if len(timbre) > 1:
        timbre = (timbre - timbre.mean(axis=0)) / (timbre.std(axis=0) + 1e-6)
    return timbre, np.asarray(level), np.asarray(rate), np.asarray(activity)


def similarity_matrix(timbre):
    norms = np.linalg.norm(timbre, axis=1, keepdims=True) + 1e-9
    unit = timbre / norms
    return unit @ unit.T


def novelty(similarity, size=NOVELTY_KERNEL_BARS):
    """Foote novelty: a checkerboard kernel slid along the diagonal of the similarity."""
    count = len(similarity)
    if count < 2 * size + 1:
        return np.zeros(count)
    axis = np.arange(-size, size) + 0.5
    taper = np.exp(-0.5 * (axis / (0.5 * size)) ** 2)
    kernel = np.outer(np.sign(axis), np.sign(axis)) * np.outer(taper, taper)
    padded = np.pad(similarity, size, mode="edge")
    result = np.zeros(count)
    for index in range(count):
        block = padded[index : index + 2 * size, index : index + 2 * size]
        result[index] = float((block * kernel).sum())
    result = np.maximum(result, 0)
    return result / (result.max() + 1e-9)


def boundaries(similarity, level, activity, bars_count):
    """Section starts (bar indices), snapped to 4-bar phrases where possible."""
    curve = novelty(similarity)
    if bars_count >= 8:
        smooth = uniform_filter1d(level, 4, mode="nearest")
        jump = np.zeros(bars_count)
        jump[1:] = np.abs(np.diff(smooth))
        curve = curve + 0.5 * np.clip(jump / 6.0, 0, 1)
    # Silence starts and ends are always boundaries.
    quiet = activity < 0.3
    edges = {0}
    for index in range(1, bars_count):
        if quiet[index] != quiet[index - 1]:
            edges.add(index)
    threshold = float(curve.mean() + 0.5 * curve.std()) if bars_count else 0.0
    peaks, _ = find_peaks(curve, height=threshold, distance=MIN_SECTION_BARS)
    for peak in peaks:
        snapped = int(round(peak / PHRASE_BARS) * PHRASE_BARS)
        edges.add(snapped if abs(snapped - peak) <= 1 else int(peak))
    edges = sorted(edge for edge in edges if 0 <= edge < bars_count)
    # Merge sections shorter than the minimum into their weaker neighbour boundary.
    changed = True
    while changed and len(edges) > 1:
        changed = False
        lengths = [b - a for a, b in zip(edges, edges[1:] + [bars_count])]
        for position, length in enumerate(lengths):
            if length >= MIN_SECTION_BARS or len(edges) == 1:
                continue
            candidates = [i for i in (position, position + 1) if 0 < i < len(edges)]
            if not candidates:
                continue
            weakest = min(candidates, key=lambda i: curve[edges[i]] if edges[i] < len(curve) else 0)
            edges.pop(weakest)
            changed = True
            break
    return edges


def classify(sections, rates, levels, activity):
    """Kinds from intensity and position; build-ups lead into drops."""
    count = len(sections)
    for index, section in enumerate(sections):
        if activity[index] < 0.3:
            section.kind = "silence"
        elif section.intensity >= 0.7:
            section.kind = "drop"
        elif index == 0 and section.intensity < 0.5:
            section.kind = "intro"
        elif index == count - 1 and section.intensity < 0.5:
            section.kind = "outro"
        elif section.intensity < 0.4:
            section.kind = "break"
        else:
            section.kind = "verse"
    for index, section in enumerate(sections[:-1]):
        following = sections[index + 1]
        rising = rates[index][1] > 1.2 * max(rates[index][0], 0.5)
        if section.kind in ("verse", "break") and following.kind == "drop" and rising:
            section.kind = "build"


def accent_list(mel, frame_ms, timing, sections, share=0.15):
    """The strongest onsets per section (relative to the section), snapped to 1/4 beats."""
    from ..timing import timing_at

    full, low, _ = onset_envelopes(mel)
    strength = np.maximum(full, 0.8 * low)
    peaks, _ = find_peaks(strength, height=0.15, distance=max(1, int(60 / frame_ms)))
    times, values = [], []
    points = red_points(timing)
    for section in sections:
        if section.kind == "silence":
            continue
        inside = peaks[(peaks * frame_ms >= section.start_ms) & (peaks * frame_ms < section.end_ms)]
        if not len(inside):
            continue
        level = strength[inside]
        cutoff = np.quantile(level, 1 - share)
        for frame, value in zip(inside, level):
            if value < cutoff:
                continue
            time = frame * frame_ms
            if points and time >= points[0].time:
                point = timing_at(points, time)
                grid = point.beat_length / 4
                snapped = point.time + round((time - point.time) / grid) * grid
                if abs(snapped - time) <= 30:
                    time = snapped
            times.append(float(time))
            values.append(float(np.clip(value / max(level.max(), 1e-6), 0, 1)))
    order = np.argsort(times)
    return [times[i] for i in order], [values[i] for i in order]


def build_plan(mel, frame_ms, timing, timing_source="detected", timing_info=None):
    mel = np.asarray(mel, np.float32)
    duration = len(mel) * frame_ms
    bars = bar_lines(timing, duration)
    timbre, level, rate, activity = bar_features(mel, frame_ms, bars)
    count = len(level)
    if count == 0:
        return SongPlan(list(timing), [], timing_source=timing_source, duration_ms=duration)
    similarity = similarity_matrix(timbre) if count > 1 else np.ones((1, 1))
    starts = boundaries(similarity, level, activity, count)
    audible = activity >= 0.3
    if not np.any(audible):
        audible = np.ones(count, bool)
    loud_low, loud_high = np.percentile(level[audible], [10, 90])
    rate_low, rate_high = np.percentile(rate[audible], [10, 90])

    def intensity_of(loud, density):
        """Half relative to this song's own range, half absolute (dB below its loud parts)."""
        relative = 0.6 * np.clip((loud - loud_low) / max(loud_high - loud_low, 1.0), 0, 1)
        relative += 0.4 * np.clip((density - rate_low) / max(rate_high - rate_low, 0.5), 0, 1)
        absolute = 0.6 * np.clip((loud - loud_high + 12) / 12, 0, 1)
        absolute += 0.4 * np.clip(density / max(rate_high, 1e-6), 0, 1)
        return float(0.5 * relative + 0.5 * absolute)

    sections, halves, levels, activities = [], [], [], []
    for position, first in enumerate(starts):
        last = starts[position + 1] if position + 1 < len(starts) else count
        span = slice(first, last)
        loud = float(np.mean(level[span]))
        density = float(np.mean(rate[span]))
        intensity = intensity_of(loud, density)
        middle = first + max(1, (last - first) // 2)
        halves.append((float(np.mean(rate[first:middle])), float(np.mean(rate[middle:last] if middle < last else rate[first:last]))))
        levels.append(loud)
        activities.append(float(np.mean(activity[span])))
        sections.append(
            Section(float(bars[first]), float(bars[last]), intensity=float(intensity), bars=last - first)
        )
    classify(sections, halves, levels, activities)
    # Repetitions: aligned bar-by-bar similarity with an earlier section of similar length.
    for later in range(len(sections)):
        if sections[later].kind == "silence":
            continue
        best, best_score = None, REPEAT_SIMILARITY
        for earlier in range(later):
            a0, a1 = starts[earlier], starts[earlier + 1] if earlier + 1 < len(starts) else count
            b0, b1 = starts[later], starts[later + 1] if later + 1 < len(starts) else count
            length = min(a1 - a0, b1 - b0)
            if length < MIN_SECTION_BARS or abs((a1 - a0) - (b1 - b0)) > 0.25 * max(a1 - a0, b1 - b0):
                continue
            score = float(np.mean([similarity[a0 + i, b0 + i] for i in range(length)]))
            if score > best_score:
                best, best_score = earlier, score
        if best is not None:
            source = sections[best]
            sections[later].repeat_of = source.repeat_of if source.repeat_of is not None else best
            if abs(sections[later].intensity - source.intensity) < 0.15:
                sections[later].kind = source.kind  # the same music plays the same way
    accents, strength = accent_list(mel, frame_ms, timing, sections)
    return SongPlan(
        timing=list(timing),
        sections=sections,
        accents=accents,
        accent_strength=strength,
        timing_source=timing_source,
        timing_info=dict(timing_info or {}),
        duration_ms=duration,
    )


def with_timing(plan, timing, source="manual"):
    """The same sections on new red lines (sections keep their times)."""
    return replace(plan, timing=list(timing), timing_source=source)
