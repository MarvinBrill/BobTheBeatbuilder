"""Beat grid detection: one constant tempo unless the audio clearly says otherwise.

The previous detector picked peaks of the model's beat head, started a new red line
whenever two frame-quantised (11.6 ms) intervals differed by 4 % and fitted every segment
on its own. Nothing checked the result against the audio. Three songs with a constant
tempo came back with 177.5 instead of 175 BPM (the grid wandered half a beat away from
the music every ~23 s), 248.3 instead of 245 BPM, a 167.7 BPM "section" and a red line that
lasted 90 ms.

Here the tempo is chosen by how coherently *all* onsets of the song (and, when available,
the model's beat activation) fall onto one grid. A phasor sum over the whole song resolves
the period far below one frame. Tempo changes are only accepted where windows of the song
lock clearly better to another tempo for a longer stretch. Round tempos (175.000 instead
of 174.993) win when they explain the audio as well, and the bar phase comes from the
accent pattern of the beats.

Everything here is NumPy only: the functions take the app's mel features (frames x 64,
values in -1..1) and optional model predictions, so they can be tested without audio.
"""

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from ..timing import TimingPoint

MIN_BPM, MAX_BPM = 60.0, 320.0
# Where human red lines sit relative to the mel-flux onset phase, in ms (negative: the red
# line is earlier). Measured on 14 training maps at 5.6-6.6 stars (median -11.8 ms, spread
# about +-15 ms). Training stores the dataset's own value in the model metadata
# (`timing_convention_ms`), which replaces this default.
DEFAULT_CONVENTION_MS = -12.0
DEFAULT_PRIOR_BPM = 175.0
PRIOR_OCTAVES = 0.6
# A tempo needs this share of the best octave's coherence to compete for the prior.
OCTAVE_SHARE = 0.75
OCTAVE_RELATIVE_SHARE = 0.3
# Coherence below this means the song does not lock to any grid (noise ~0.01-0.03).
MIN_COHERENCE = 0.06
# Windows for tempo-section detection, in beats of the global tempo.
WINDOW_BEATS, HOP_BEATS = 16, 8
MIN_SECTION_WINDOWS = 4  # ~32 beats: shorter "tempo changes" are rhythm, not timing
LOCAL_RATIO = 2.0  # local tempo must lock this much better than the global one
LOCAL_DEVIATION = 0.02  # ...and differ by at least 2 %
# Simple tempo ratios are rhythm changes (half time, dotted or triplet feel), almost never a
# new red line; they need the global grid to fail completely in that part of the song.
SIMPLE_RATIOS = (0.5, 2.0, 1.5, 2 / 3, 0.75, 4 / 3, 1.25, 0.8)
# Dotted, triplet and half/double feels: their notes sit on the global grid's 1/4 or 1/3
# ticks, so the existing red line plus the snap divisor already covers them.
SUBDIVISION_RATIOS = (0.5, 2.0, 1.5, 2 / 3, 0.75, 4 / 3)
LOW_BANDS = slice(0, 5)  # roughly below 200 Hz (kick, bass)
HIGH_BANDS = slice(30, 64)


# --------------------------------------------------------------------------------------
# Onset envelopes from the mel features


def onset_envelopes(mel):
    """Spectral-flux onset envelopes (full, low, high) per mel frame, roughly 0..1.

    Frame k of the app's mel features is centred on k * FRAME_MS (librosa, center=True);
    the flux of frame k is attributed to frame k. A constant detector latency cancels out
    wherever the same envelope is used on both sides (training labels and generation).
    """
    mel = np.asarray(mel, np.float32)
    if len(mel) < 2:
        zero = np.zeros(len(mel), np.float32)
        return zero, zero, zero
    rise = np.maximum(np.diff(mel, axis=0, prepend=mel[:1]), 0)
    silent = mel.max(axis=1) <= -0.99
    rise[silent] = 0

    def normalised(values):
        values = values - uniform_filter1d(values, 35, mode="nearest")  # ~0.4 s trend
        values = np.maximum(values, 0)
        scale = np.percentile(values[values > 0], 99) if np.any(values > 0) else 1.0
        return np.clip(values / max(scale, 1e-6), 0, 1.5).astype(np.float32)

    return (
        normalised(rise.mean(axis=1)),
        normalised(rise[:, LOW_BANDS].mean(axis=1)),
        normalised(rise[:, HIGH_BANDS].mean(axis=1)),
    )


# --------------------------------------------------------------------------------------
# Coherence of an envelope with a beat grid


def coherence(envelope, frame_ms, bpms, start=0, stop=None):
    """|phasor| / total of `envelope[start:stop]` for every tempo in `bpms` (0..1)."""
    values = np.asarray(envelope[start:stop], np.float64)
    total = values.sum()
    bpms = np.atleast_1d(np.asarray(bpms, np.float64))
    if total <= 1e-12 or not len(values):
        return np.zeros(len(bpms))
    times = (np.arange(len(values)) + start) * frame_ms / 1000
    result = np.empty(len(bpms))
    for first in range(0, len(bpms), 64):
        chunk = bpms[first : first + 64]
        phasors = np.exp(2j * np.pi * np.outer(chunk / 60, times)) @ values
        result[first : first + len(chunk)] = np.abs(phasors) / total
    return result


def grid_phase(envelope, frame_ms, bpm, start=0, stop=None):
    """(phase_ms, coherence): the grid t = phase + k * beat that collects the envelope."""
    values = np.asarray(envelope[start:stop], np.float64)
    total = values.sum()
    beat = 60000 / bpm
    if total <= 1e-12:
        return 0.0, 0.0
    times = (np.arange(len(values)) + start) * frame_ms
    phasor = (values * np.exp(2j * np.pi * times / beat)).sum() / total
    return float((np.angle(phasor) / (2 * np.pi)) % 1 * beat), float(abs(phasor))


def refine(envelope, frame_ms, bpm, tolerance, steps, start=0, stop=None):
    """Best tempo within bpm*(1 +- tolerance) and its coherence (parabolic peak)."""
    candidates = bpm * np.linspace(1 - tolerance, 1 + tolerance, steps)
    strength = coherence(envelope, frame_ms, candidates, start, stop)
    best = int(np.argmax(strength))
    value = float(candidates[best])
    if 0 < best < len(candidates) - 1:
        left, middle, right = strength[best - 1 : best + 2]
        curvature = left - 2 * middle + right
        if curvature < 0:
            step = candidates[1] - candidates[0]
            value += float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5)) * step
    return value, float(coherence(envelope, frame_ms, [value], start, stop)[0])


def periodicity_candidates(envelope, frame_ms, count=6):
    """Coarse tempo candidates from the envelope's autocorrelation."""
    values = np.asarray(envelope, np.float64)
    values = values - values.mean()
    energy = float(values @ values)
    if energy <= 1e-12:
        return []
    low = max(2, int(60000 / MAX_BPM / frame_ms))
    high = min(len(values) // 2, int(math.ceil(60000 / MIN_BPM / frame_ms)) + 2)
    if high <= low + 2:
        return []
    spectrum = np.fft.rfft(values, 2 * len(values))
    correlation = np.fft.irfft(spectrum * np.conj(spectrum))[: high + 1] / energy
    peaks, _ = find_peaks(correlation[low : high + 1])
    peaks = peaks + low
    peaks = peaks[np.argsort(-correlation[peaks])][:count]
    result = []
    for lag in peaks:
        # Sub-frame lag from a parabola through the peak.
        left, middle, right = correlation[lag - 1 : lag + 2]
        curvature = left - 2 * middle + right
        shift = 0.5 * (left - right) / curvature if curvature < 0 else 0.0
        result.append(60000 / ((lag + shift) * frame_ms))
    return result


def prior_score(bpm, prior_bpm):
    return -0.5 * (math.log2(bpm / prior_bpm) / PRIOR_OCTAVES) ** 2


def combined(envelope, beat_activation, weight=0.5):
    """Mix the audio onsets with the model's beat activation (both scaled to mean 1)."""
    envelope = np.asarray(envelope, np.float64)
    result = envelope / max(envelope.mean(), 1e-9)
    if beat_activation is None:
        return result
    beats = np.clip(np.asarray(beat_activation, np.float64)[: len(result)], 0, None)
    if len(beats) < len(result):
        beats = np.pad(beats, (0, len(result) - len(beats)))
    if beats.mean() <= 1e-9:
        return result
    return (1 - weight) * result + weight * beats / beats.mean()


@dataclass
class TempoEstimate:
    bpm: float
    phase: float  # ms, grid t = phase + k * beat
    coherence: float
    candidates: list = field(default_factory=list)  # (bpm, coherence, score)


def estimate_tempo(envelope, frame_ms, prior_bpm=None, hints=(), start=0, stop=None):
    """Constant tempo of `envelope[start:stop]`, octave chosen with a tempo prior."""
    prior = float(np.clip(prior_bpm or DEFAULT_PRIOR_BPM, 90, 280))
    segment = np.asarray(envelope[start:stop], np.float64)
    coarse = periodicity_candidates(segment, frame_ms)
    coarse += [float(value) for value in hints if value and MIN_BPM <= value <= MAX_BPM]
    seeds = set()
    for value in coarse:
        for factor in (0.5, 1.0, 2.0, 1.5, 2 / 3):
            candidate = value * factor
            if MIN_BPM <= candidate <= MAX_BPM:
                seeds.add(round(candidate, 1))
    if not seeds:
        return None
    seconds = len(segment) * frame_ms / 1000
    # Peak width ~ 1 / (beats in the song); sample it at least four times.
    steps = int(np.clip(0.03 * 4 * seconds * prior / 60, 121, 1601))
    found = []
    for seed in sorted(seeds):
        bpm, strength = refine(segment, frame_ms, seed, 0.015, steps)
        if MIN_BPM <= bpm <= MAX_BPM:
            found.append((bpm, strength))
    # Merge duplicates (same tempo found from several seeds).
    unique = []
    for bpm, strength in sorted(found, key=lambda item: -item[1]):
        if all(abs(bpm / other - 1) > 0.004 for other, _ in unique):
            unique.append((bpm, strength))
    if not unique:
        return None
    best_strength = max(strength for _, strength in unique)
    strongest = max(unique, key=lambda item: item[1])[0]
    scored = []
    for bpm, strength in unique:
        # Octave relatives of the strongest grid compete; unrelated weak tempos do not. Half
        # or double tempo locks less (off-beats cancel at half tempo), so it needs less.
        ratio = bpm / strongest
        octave = min(abs(ratio - 0.5) / 0.5, abs(ratio - 2) / 2) < 0.01
        eligible = strength >= (OCTAVE_RELATIVE_SHARE if octave else OCTAVE_SHARE) * best_strength
        score = math.log(max(strength, 1e-4)) + prior_score(bpm, prior)
        scored.append((bpm, strength, score if eligible else -math.inf))
    bpm, strength, _ = max(scored, key=lambda item: item[2])
    phase, _ = grid_phase(envelope, frame_ms, bpm, start, stop)
    return TempoEstimate(bpm, phase, strength, sorted(scored, key=lambda item: -item[2])[:6])


def round_tempo(envelope, frame_ms, bpm, start=0, stop=None):
    """The roundest tempo (1, 0.5, 0.25, 0.1 ... BPM steps) that locks the onsets about as
    well as the measured one. A wrong rounding (174.8 -> 175 over three minutes) drifts by
    a quarter beat and loses the coherence, so the coherence alone decides."""
    reference = coherence(envelope, frame_ms, [bpm], start, stop)[0]
    for step, share in ((1.0, 0.97), (0.5, 0.98), (0.25, 0.985), (0.1, 0.985), (0.05, 0.99)):
        rounded = round(bpm / step) * step
        if abs(rounded - bpm) > 0.006 * bpm:
            continue
        if coherence(envelope, frame_ms, [rounded], start, stop)[0] >= share * reference:
            return float(rounded)
    return float(round(bpm, 3))


def alternatives(envelope, frame_ms, bpm, start=0, stop=None):
    """Other tempos a mapper might choose: octaves and nearby whole numbers that still lock."""
    reference = coherence(envelope, frame_ms, [bpm], start, stop)[0]
    options = {round(bpm * factor, 3) for factor in (0.5, 2.0, 1.5, 2 / 3) if MIN_BPM <= bpm * factor <= MAX_BPM}
    options |= {float(value) for value in range(int(bpm) - 1, int(bpm) + 3) if value != bpm}
    result = []
    for option in sorted(options):
        strength = coherence(envelope, frame_ms, [option], start, stop)[0]
        if strength >= 0.5 * reference:
            result.append(dict(bpm=round(option, 3), coherence=round(float(strength), 4)))
    return result


# --------------------------------------------------------------------------------------
# Tempo sections


@dataclass
class TempoSection:
    start: int  # frames
    stop: int
    bpm: float
    phase: float  # ms
    coherence: float


def tempo_sections(envelope, frame_ms, estimate, active=None):
    """Constant tempo, unless runs of windows lock clearly better to another tempo."""
    frames = len(envelope)
    beat = 60000 / estimate.bpm
    window = max(8, int(WINDOW_BEATS * beat / frame_ms))
    hop = max(4, int(HOP_BEATS * beat / frame_ms))
    whole = [TempoSection(0, frames, estimate.bpm, estimate.phase, estimate.coherence)]
    if frames < 3 * window:
        return whole
    starts = list(range(0, frames - window + 1, hop))
    other = []
    for first in starts:
        last = first + window
        if active is not None and np.mean(active[first:last]) < 0.5:
            other.append(None)
            continue
        here = coherence(envelope, frame_ms, [estimate.bpm], first, last)[0]
        local, strength = refine(envelope, frame_ms, estimate.bpm, 0.5, 361, first, last)
        ratio = local / estimate.bpm
        simple = min(abs(ratio - value) / value for value in SIMPLE_RATIOS)
        # Notes that fall on the global grid's 1/4 or 1/3 ticks (a dotted or triplet
        # passage) need no new red line: the snap divisor already covers them.
        subdivided = False
        musical = min(abs(ratio - value) / value for value in SUBDIVISION_RATIOS) < LOCAL_DEVIATION
        for divisor in (4, 3) if musical else ():
            fine, fine_lock = grid_phase(envelope, frame_ms, estimate.bpm * divisor, first, last)
            tick = beat / divisor
            offset = abs((fine - estimate.phase + tick / 2) % tick - tick / 2)
            if fine_lock >= 0.5 * strength and offset < tick / 4:
                subdivided = True
        local_phase, _ = grid_phase(envelope, frame_ms, estimate.bpm, first, last)
        shift = abs((local_phase - estimate.phase + beat / 2) % beat - beat / 2)
        # The global grid still explains this window: same tempo, whatever the rhythm.
        explained = here >= 0.12 or shift < beat / 8
        if subdivided:
            different = False
        elif simple < LOCAL_DEVIATION:
            different = here < 0.15 and strength > 3 * max(here, 1e-6)
        else:
            different = (
                abs(ratio - 1) > LOCAL_DEVIATION
                and not explained
                and strength > LOCAL_RATIO * max(here, 1e-6)
                and strength > MIN_COHERENCE * 2
            )
        other.append((local, simple < LOCAL_DEVIATION) if different else 0.0)
    runs, current = [], []
    for index, value in enumerate(other):
        if value:
            if current and abs(value[0] / other[current[-1]][0] - 1) > 0.03:
                runs.append(current)
                current = []
            current.append(index)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)
    # Simple ratios (dotted, triplet, half time) need a longer run than other tempos.
    runs = [
        run
        for run in runs
        if len(run) >= MIN_SECTION_WINDOWS + int(any(other[i][1] for i in run))
    ]
    if not runs:
        return whole
    # Boundaries in frames: a run covers its windows; the rest keeps the global tempo.
    spans = []
    cursor = 0
    for run in runs:
        left = starts[run[0]] + (hop // 2 if run[0] else 0)
        right = starts[run[-1]] + window - (hop // 2 if run[-1] < len(starts) - 1 else 0)
        right = frames if run[-1] == len(starts) - 1 else right
        if left > cursor:
            spans.append((cursor, left, None))
        spans.append((left, right, float(np.median([other[i][0] for i in run]))))
        cursor = right
    if cursor < frames:
        spans.append((cursor, frames, None))
    sections = []
    for left, right, local in spans:
        if right - left < window // 2:
            continue
        if local is None and (right - left) * frame_ms < 30000:
            # Short stretches at the song's tempo keep the global grid instead of a refit.
            sections.append(
                TempoSection(left, right, estimate.bpm, estimate.phase, estimate.coherence)
            )
            continue
        seed = local if local else estimate.bpm
        bpm, strength = refine(envelope, frame_ms, seed, 0.01, 241, left, right)
        bpm = round_tempo(envelope, frame_ms, bpm, left, right)
        phase, strength = grid_phase(envelope, frame_ms, bpm, left, right)
        sections.append(TempoSection(left, right, bpm, phase, strength))
    merged = []
    for section in sections:
        if merged and abs(merged[-1].bpm - section.bpm) < 0.01:
            previous = merged[-1]
            difference = (section.phase - previous.phase) % (60000 / section.bpm)
            difference = min(difference, 60000 / section.bpm - difference)
            if difference < 5:
                merged[-1] = TempoSection(previous.start, section.stop, previous.bpm, previous.phase, previous.coherence)
                continue
        merged.append(section)
    return merged or whole


# --------------------------------------------------------------------------------------
# Bars


def beat_times(phase, beat, start_ms, stop_ms):
    first = phase + math.ceil((start_ms - phase) / beat) * beat
    return np.arange(first, stop_ms, beat)


def downbeat(mel, low, full, frame_ms, phase, beat, start_ms, stop_ms, meter=4):
    """(bar_origin_ms, confidence 0..1) from accents and harmonic change on the beats."""
    times = beat_times(phase, beat, start_ms, stop_ms)
    if len(times) < 4 * meter:
        return None, 0.0
    frames = np.clip(np.rint(times / frame_ms).astype(int), 0, len(low) - 1)
    reach = max(1, int(round(30 / frame_ms)))

    def around(values, frame):
        return float(values[max(0, frame - reach) : frame + reach + 1].max(initial=0.0))

    kick = np.array([around(low, frame) for frame in frames])
    hit = np.array([around(full, frame) for frame in frames])
    span = max(2, int(beat / frame_ms))
    mel = np.asarray(mel, np.float32)
    change = np.zeros(len(frames))
    for index, frame in enumerate(frames):
        before = mel[max(0, frame - span) : frame].mean(axis=0) if frame > 0 else None
        after = mel[frame : frame + span].mean(axis=0)
        if before is None or not len(after):
            continue
        a, b = before - before.mean(), after - after.mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        change[index] = 1 - float(a @ b / denominator) if denominator > 1e-9 else 0.0

    def z(values):
        return (values - values.mean()) / max(values.std(), 1e-6)

    accent = z(kick) + z(change) + 0.5 * z(hit)
    scores = np.array([accent[k::meter].mean() for k in range(meter)])
    order = np.argsort(-scores)
    best = int(order[0])
    spread = float(np.std(accent)) or 1.0
    margin = float(scores[order[0]] - scores[order[1]])
    confidence = float(np.clip(margin / (0.5 * spread), 0, 1))
    return float(times[best]), confidence


# --------------------------------------------------------------------------------------
# Public entry point


@dataclass
class TimingResult:
    points: list
    source: str
    info: dict


def detect_timing(mel, frame_ms, prediction=None, prior_bpm=None, convention_ms=None):
    """Red lines for a song from its mel features and, optionally, the model's rhythm
    prediction (channel 1: beat activation, channel 3: log beat
    length / 500 ms).

    The phase comes from the model's beat activation when it locks to the grid: it was
    trained on human red lines and therefore carries the training data's timing
    convention (where mappers put the beat relative to the decoded onsets). Without it the
    mel onsets plus `convention_ms` are used (the same detector measures the convention
    on the training maps). Returns a TimingResult with TimingPoints (the first at or before
    0, on a bar line when the bar phase is clear) and an `info` dict for the report.
    """
    mel = np.asarray(mel, np.float32)
    mel_full, mel_low, _ = onset_envelopes(mel)
    active = mel.max(axis=1) > -0.7
    beats = None
    hints = []
    if prediction is not None:
        prediction = np.asarray(prediction, np.float32)
        beats = np.clip(prediction[: len(mel), 1], 0, 1)
        values = prediction[: len(mel), 3]
        mask = active[: len(values)]
        values = values[mask] if np.any(mask) else values
        if len(values):
            model_bpm = 60000 / (500 * math.exp(float(np.median(np.clip(values, -2, 2)))))
            if MIN_BPM <= model_bpm <= MAX_BPM:
                hints.append(model_bpm)
                # The tempo head was trained on human red lines: it knows the octave mappers use.
                prior_bpm = model_bpm
    beat_lock = 0.0
    if beats is not None and beats.sum() > 0 and hints:
        beat_lock = float(refine(beats, frame_ms, hints[0], 0.04, 321)[1])
    use_beats = beat_lock >= 0.15
    # A linear-frequency flux from the waveform was tried and locked worse: hi-hats and
    # noise dominate its bins. The mel flux weights the bands like the model hears them.
    tempo_full, tempo_ms = mel_full, frame_ms
    tempo_active = active
    envelope = combined(tempo_full, beats if use_beats else None)
    estimate = estimate_tempo(envelope, tempo_ms, prior_bpm, hints)
    if estimate is None or estimate.coherence < MIN_COHERENCE:
        raise ValueError("No steady beat found in the audio. Enter BPM and offset manually.")
    estimate.bpm = round_tempo(envelope, tempo_ms, estimate.bpm)
    estimate.phase, estimate.coherence = grid_phase(envelope, tempo_ms, estimate.bpm)
    sections = tempo_sections(envelope, tempo_ms, estimate, tempo_active)
    points, details = [], []
    for index, section in enumerate(sections):
        beat = 60000 / section.bpm
        start_ms, stop_ms = section.start * tempo_ms, section.stop * tempo_ms
        first = int(start_ms / frame_ms)
        last = min(len(mel), int(math.ceil(stop_ms / frame_ms)))
        onset_phase, onset_lock = grid_phase(mel_full, frame_ms, section.bpm, first, last)
        phase, source = onset_phase, "audio"
        if use_beats:
            beat_phase, lock = grid_phase(beats, frame_ms, section.bpm, first, last)
            if lock >= 0.1:
                phase, source = beat_phase, "model-beats"
        if source == "audio" and convention_ms:
            phase = (phase + convention_ms) % beat
        origin, bar_confidence = downbeat(
            mel, mel_low, mel_full, frame_ms, phase, beat, start_ms, stop_ms
        )
        meter = 4
        anchor = origin if origin is not None and bar_confidence >= 0.25 else phase
        bar = beat * meter
        # Red line at the section start (the first one at or before 0), on a bar line.
        if index:
            time = anchor - math.ceil((anchor - start_ms) / bar) * bar
            if time < start_ms - 1e-6:
                time += bar
        else:
            time = anchor - math.ceil(max(0.0, anchor) / bar) * bar
        known_bar = origin is not None and bar_confidence >= 0.5
        points.append(
            TimingPoint(
                float(time),
                float(beat),
                meter=meter if known_bar else None,
                beat_confidence=1.0 if section.coherence >= 0.12 else 0.75,
                meter_confidence=0.75 if known_bar else 0.0,
                bar_origin=float(time) if known_bar else None,
            )
        )
        details.append(
            dict(
                start_ms=round(start_ms, 1),
                stop_ms=round(stop_ms, 1),
                bpm=round(section.bpm, 4),
                offset_ms=round(float(time), 2),
                coherence=round(section.coherence, 4),
                onset_coherence=round(onset_lock, 4),
                phase_source=source,
                bar_confidence=round(bar_confidence, 3),
                downbeat_ms=round(origin, 1) if origin is not None else None,
            )
        )
    cleaned = [points[0]]
    for point in points[1:]:
        if point.time > cleaned[-1].time + 1:
            cleaned.append(point)
    # How sure the grid is: sharp, strong lock and no close competitor.
    competitor = max(
        (strength for bpm, strength, _ in estimate.candidates[1:] if abs(bpm / estimate.bpm - 1) < 0.1),
        default=0.0,
    )
    confidence = float(
        np.clip((estimate.coherence - 0.08) / 0.3, 0, 1)
        * (1 - np.clip(competitor / max(estimate.coherence, 1e-6) - 0.5, 0, 0.5) * 2)
    )
    info = dict(
        bpm=round(estimate.bpm, 4),
        confidence=round(confidence, 3),
        alternatives=alternatives(envelope, tempo_ms, estimate.bpm),
        coherence=round(estimate.coherence, 4),
        beat_lock=round(beat_lock, 4),
        used_model_beats=use_beats,
        prior_bpm=round(prior_bpm, 2) if prior_bpm else None,
        candidates=[
            dict(bpm=round(bpm, 3), coherence=round(strength, 4))
            for bpm, strength, _ in estimate.candidates
        ],
        sections=details,
        constant=len(cleaned) == 1,
    )
    return TimingResult(cleaned, "detected-model" if use_beats else "detected-audio", info)


def grid_drift(envelope, frame_ms, timing, window_ms=8000, hop_ms=8000):
    """(time, phase_ms, lock, beat_ms) of the onsets relative to the red lines per window.

    A correct grid gives a small, constant value; a wrong tempo drifts or jumps.
    """
    from ..timing import red_points, timing_at

    points = red_points(timing)
    frames = len(envelope)
    result = []
    if not points:
        return result
    step = max(1, int(hop_ms / frame_ms))
    size = max(4, int(window_ms / frame_ms))
    for first in range(0, frames - size + 1, step):
        middle = (first + size / 2) * frame_ms
        point = timing_at(points, middle) if middle >= points[0].time else points[0]
        values = np.asarray(envelope[first : first + size], np.float64)
        if values.sum() <= 1e-9:
            continue
        times = (np.arange(first, first + size)) * frame_ms - point.time
        phasor = (values * np.exp(2j * np.pi * times / point.beat_length)).sum() / values.sum()
        phase = (np.angle(phasor) / (2 * np.pi)) * point.beat_length
        result.append((middle, float(phase), float(abs(phasor)), float(point.beat_length)))
    return result
