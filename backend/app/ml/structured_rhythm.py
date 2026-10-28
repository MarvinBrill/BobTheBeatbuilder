"""Bounded, deterministic rhythm selection with a soft, continuous density cost.

There is no meter inference here: pulse confidence permits local beat fractions,
never bar accents. Width one is the greedy ablation; eight keeps short alternatives.
"""

from dataclasses import dataclass
import math
import numpy as np
from scipy.signal import find_peaks
from .audio import FRAME_MS
from ..timing import timing_at as timing_values, beat_distance

MIN_EVIDENCE = 0.15
BEAM_WIDTH = 8
MIN_HEAD_GAP_MS = 35


@dataclass(frozen=True)
class Onset:
    frame: int
    time: float
    score: float
    density: float
    beat: float
    confidence: float


@dataclass
class RhythmState:
    score: float = 0.0
    onset: Onset | None = None
    mass: float = 0.0
    intervals: tuple = ()
    parent: "RhythmState | None" = None
    terms: dict | None = None


def pulse_confidence(prediction, frame, beat, settings, offset=0):
    if settings.get("bpm") or settings.get("timing_points"):
        return 1.0
    radius = max(1, round(beat * 2 / FRAME_MS))
    start = max(0, frame - radius)
    local = prediction[start : frame + radius + 1, 1]
    peaks, _ = find_peaks(local, height=0.55, prominence=0.15)
    if len(peaks) < 2:
        return 0.0
    intervals = np.diff(peaks) * FRAME_MS / beat
    phase = ((peaks + start) * FRAME_MS - offset) / beat
    consistent = ((intervals >= 0.75) & (abs(intervals - np.rint(intervals)) < 0.12)).mean()
    anchored = (abs(phase - np.rint(phase)) < 0.15).mean()
    return float(np.mean(local[peaks])) if consistent >= 0.75 and anchored >= 0.75 else 0.0


def snap_time(time, timing, divisor, confidence):
    if confidence < 0.6:
        return time
    point = timing_values(timing, time)
    offset, beat = point.time, point.beat_length
    # Compare straight and triplet grids locally. No common global grid is imposed.
    alternatives = []
    for division in sorted({divisor, 2, 3, 4, 6, 8, 12, 16}):
        grid = beat / division
        snapped = offset + round((time - offset) / grid) * grid
        error = abs(snapped - time)
        if error <= min(25, grid * 0.18):
            alternatives.append((error, division, float(snapped)))
    if not alternatives:
        return time
    # Mel frames cannot distinguish offsets inside half a frame. Prefer the
    # coarser supported fraction there, so e.g. 1/3 is not mistaken for 5/16.
    closest = min(error for error, _, _ in alternatives)
    supported = [
        (division, error, snapped)
        for error, division, snapped in alternatives
        if error <= closest + FRAME_MS / 2
    ]
    return min(supported)[2]


def onset_candidates(prediction, mel, settings, timing, trace=None):
    # Preserve low-level tails that the former relative -0.7 threshold discarded.
    # -1 is the existing audio frontend's absolute silence-floor sentinel.
    active = np.max(mel, axis=1) > -0.999
    profile = np.expm1(np.clip(prediction[:, 2], 0, 1.3) * np.log(11)) * active
    if profile.sum() > 0.01:
        profile *= settings["density"] / max(0.1, profile[active].mean())
    else:
        profile = active.astype(float) * settings["density"]
    peaks, _ = find_peaks(prediction[:, 0], prominence=0.005)
    candidates = []
    for frame in peaks:
        raw_time, score = float(frame * FRAME_MS), float(prediction[frame, 0])
        point = timing_values(timing, raw_time)
        offset, beat = point.time, point.beat_length
        confidence = (
            min(point.beat_confidence, pulse_confidence(prediction, frame, beat, settings, offset))
            if raw_time >= point.time
            else 0.0
        )
        time = (
            snap_time(raw_time, timing, settings["divisor"], confidence)
            if settings["snap"]
            else raw_time
        )
        snapped_frame = int(np.clip(round(time / FRAME_MS), 0, len(active) - 1))
        reason = (
            "silence_floor"
            if not active[frame]
            else (
                "weak_evidence"
                if score < MIN_EVIDENCE
                else (
                    "before_audio"
                    if time < 0
                    else (
                        "after_audio_margin"
                        if time > (len(mel) - 1) * FRAME_MS - 100
                        else "silence_after_snap" if not active[snapped_frame] else "eligible"
                    )
                )
            )
        )
        if trace is not None:
            trace.record(
                "onset_proposal",
                frame=int(frame),
                raw_time=raw_time,
                time=time,
                score=score,
                snap_shift_ms=time - raw_time,
                reason=reason,
                density_target=float(profile[frame]),
                timing_confidence=confidence,
            )
        if reason == "eligible":
            candidates.append(
                Onset(int(frame), time, score, float(profile[frame]), beat, confidence)
            )
    # Resolve actual post-snap collisions by evidence, not chronological accident.
    accepted, occupied = [], {}
    for onset in sorted(candidates, key=lambda value: (-value.score, value.time)):
        bucket = math.floor(onset.time / MIN_HEAD_GAP_MS)
        neighbors = [occupied[key] for key in (bucket - 1, bucket, bucket + 1) if key in occupied]
        collision = next(
            (other for other in neighbors if abs(onset.time - other.time) < MIN_HEAD_GAP_MS), None
        )
        if collision is None:
            accepted.append(onset)
            occupied[bucket] = onset
        elif trace is not None:
            trace.record(
                "onset_selection",
                frame=onset.frame,
                time=onset.time,
                selected=False,
                reason="collision_after_snap",
                competing_frame=collision.frame,
            )
    return sorted(accepted, key=lambda value: value.time)


def interval_family(interval):
    # 3/8 is straight subdivision; 1/3 or 2/3 is triplet. Neither implies meter.
    errors = [abs(interval * divisor - round(interval * divisor)) / divisor for divisor in (16, 12)]
    if min(errors) > 0.055 or abs(errors[0] - errors[1]) < 0.01:
        return 0
    return 1 if errors[0] < errors[1] else 2


def extension(state, onset, timing):
    tau = max(0.5, onset.beat * 4 / 1000)
    elapsed = (onset.time - state.onset.time) / 1000 if state.onset else math.inf
    mass = state.mass * math.exp(-elapsed / tau)
    target = max(0.25, onset.density * tau)
    overflow = (max(0, mass + 1 - target) ** 2 - max(0, mass - target) ** 2) / target
    # BCE scores are not calibrated probabilities. Bound the evidence contribution
    # so a confident peak cannot erase the user's sparse-density preference.
    evidence = (onset.score - MIN_EVIDENCE) / (1 - MIN_EVIDENCE)
    family_cost, motif = 0.0, 0.0
    intervals = state.intervals
    if state.onset:
        gap = beat_distance(timing, state.onset.time, onset.time)
        if gap > 4:
            intervals = ()  # A genuine rest ends local continuity; it incurs no penalty.
        elif min(onset.confidence, state.onset.confidence) >= 0.6:
            if intervals:
                previous_family, family = interval_family(intervals[-1]), interval_family(gap)
                if previous_family and family and previous_family != family:
                    family_cost = 0.25 * (1 - onset.score)
                # Match an ordered two-interval word, never just a bag of old gaps.
                if (
                    len(intervals) >= 3
                    and abs(intervals[-1] - intervals[-3]) < 0.055
                    and abs(gap - intervals[-2]) < 0.055
                ):
                    motif = 0.15
            intervals = (*intervals[-3:], round(gap, 4))
    # Even a near-zero density estimate may be wrong at a strong isolated accent.
    # Only accumulated recent activity incurs the full overflow cost.
    density_cost = 0.6 * mass / target + min(0.3, 0.3 / target) + 0.7 * overflow * min(mass, 1)
    terms = dict(
        evidence=evidence,
        density_cost=density_cost,
        family_cost=family_cost,
        interval_repetition=motif,
    )
    score = state.score + evidence - terms["density_cost"] - family_cost + motif
    return RhythmState(score, onset, mass + 1, intervals, state, terms)


def select_sequence(candidates, timing, width=BEAM_WIDTH, trace=None):
    if width not in (1, BEAM_WIDTH):
        raise ValueError("Rhythm search supports only greedy or bounded beam width")
    states = [RhythmState()]
    for onset in candidates:
        branches = states + [extension(state, onset, timing) for state in states]
        states = sorted(branches, key=lambda state: state.score, reverse=True)[:width]
    state, selected = states[0], []
    while state.onset is not None:
        selected.append(state)
        state = state.parent
    selected.reverse()
    chosen = {state.onset.frame: state for state in selected}
    if trace is not None:
        trace.record(
            "rhythm_search",
            method="greedy" if width == 1 else "beam",
            width=width,
            candidates=len(candidates),
            selected=len(selected),
            density="soft_continuous",
        )
        for onset in candidates:
            state = chosen.get(onset.frame)
            trace.record(
                "onset_selection",
                frame=onset.frame,
                raw_time=onset.frame * FRAME_MS,
                time=onset.time,
                score=onset.score,
                selected=state is not None,
                reason="sequence_selected" if state else "sequence_score",
                score_terms=state.terms if state else None,
            )
    return [state.onset.time for state in selected]
