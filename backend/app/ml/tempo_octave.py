"""Tempo octave correction for the audio-only tempo fallback.

Beat trackers with a 120 BPM prior regularly report half (or double) the tempo that osu!
mappers time a song with: a 199 BPM song becomes 99.4 BPM. The model is conditioned on the
tempo and every rhythm class is measured in beats, so a halved tempo turns 1/2 jumps into
"1/4 streams" with stream movement. Only when the onset envelope is nearly as periodic at
the other octave is the octave chosen by a tempo prior; a clearly periodic tempo is kept.
"""

import math
import numpy as np

DEFAULT_PRIOR_BPM = 175.0
PRIOR_OCTAVES = 0.6  # standard deviation of the log2-tempo prior
MIN_BPM, MAX_BPM = 70.0, 300.0
# The alternative octave needs at least this share of the reported tempo's periodicity.
MIN_PERIODICITY_RATIO = 0.8


def periodicity(envelope, fps, period_seconds, tolerance_frames=2):
    """Normalised autocorrelation of an onset envelope at one period (0 when unknown)."""
    values = np.asarray(envelope, np.float64)
    values = values - values.mean()
    energy = float(np.dot(values, values))
    lag = int(round(period_seconds * fps))
    if energy <= 1e-12 or lag < 1 or lag + tolerance_frames >= len(values):
        return 0.0
    best = 0.0
    for shift in range(max(1, lag - tolerance_frames), lag + tolerance_frames + 1):
        best = max(best, float(np.dot(values[:-shift], values[shift:])) / energy)
    return best


def grid_phase(envelope, fps, beat_ms, phases):
    """The phase (ms) whose beat grid collects the most onset strength."""
    values = np.asarray(envelope, np.float64)

    def strength(phase):
        frames = np.rint((phase % beat_ms + np.arange(0, len(values) / fps * 1000, beat_ms)) * fps / 1000)
        frames = frames[frames < len(values)].astype(int)
        return float(values[frames].sum())

    return max(phases, key=strength)


def model_tempo(prediction, active=None):
    """Median tempo of the model's own tempo head (channel 3 is log(beat / 500 ms))."""
    values = np.asarray(prediction, np.float64)[:, 3]
    if active is not None and np.any(active):
        values = values[np.asarray(active, bool)[: len(values)]]
    if not len(values):
        return None
    bpm = 60000 / (500 * math.exp(float(np.median(np.clip(values, -2, 2)))))
    return bpm if 60 <= bpm <= 320 else None


def prior_score(bpm, prior_bpm):
    return -0.5 * (math.log2(bpm / prior_bpm) / PRIOR_OCTAVES) ** 2


def octave_corrected(envelope, fps, bpm, prior_bpm=None):
    """`bpm`, `bpm * 2` or `bpm / 2`, whichever an osu! mapper would most likely use."""
    prior_bpm = float(prior_bpm) if prior_bpm and math.isfinite(prior_bpm) else DEFAULT_PRIOR_BPM
    prior_bpm = float(np.clip(prior_bpm, 100, 260))
    if not math.isfinite(bpm) or bpm <= 0:
        return bpm
    reference = periodicity(envelope, fps, 60 / bpm)
    best, best_score = bpm, math.log(max(reference, 1e-3)) + prior_score(bpm, prior_bpm)
    for candidate in (bpm * 2, bpm / 2):
        if not MIN_BPM <= candidate <= MAX_BPM:
            continue
        strength = periodicity(envelope, fps, 60 / candidate)
        if reference > 0 and strength < MIN_PERIODICITY_RATIO * reference:
            continue
        score = math.log(max(strength, 1e-3)) + prior_score(candidate, prior_bpm)
        if score > best_score:
            best, best_score = candidate, score
    return best
