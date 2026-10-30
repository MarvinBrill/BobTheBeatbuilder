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


REFINE_TOLERANCE = 0.03  # search +-3 % around the coarse tempo
REFINE_STEPS = 1201
# The refined tempo must lock the onsets clearly better than the coarse one.
MIN_COHERENCE_GAIN = 1.1
# ...and lock them at all: noise reaches ~0.01, a steady song 0.3-0.6.
MIN_COHERENCE = 0.1


def grid_coherence(envelope, fps, bpm):
    """How strongly the onsets of the whole song lock to one constant beat grid (0..1),
    and the grid phase in ms that collects them."""
    values = np.maximum(np.asarray(envelope, np.float64), 0)
    total = values.sum()
    if total <= 1e-12 or not math.isfinite(bpm) or bpm <= 0:
        return 0.0, 0.0
    beat_ms = 60000 / bpm
    times = np.arange(len(values)) * 1000 / fps
    phasor = (values * np.exp(2j * np.pi * times / beat_ms)).sum() / total
    phase = (np.angle(phasor) / (2 * np.pi)) % 1 * beat_ms
    return float(abs(phasor)), float(phase)


def refine_tempo(envelope, fps, bpm, tolerance=REFINE_TOLERANCE, steps=REFINE_STEPS):
    """Sub-frame tempo and phase of a constant-tempo song.

    Tempogram/autocorrelation estimates are quantised to whole onset frames: with a 256
    sample hop at 22.05 kHz a 200 BPM song is reported as 198.77 BPM (26 frames). That
    0.6 % error drifts by a whole beat every ~45 s, so objects slide in and out of the
    beat. Here the tempo is chosen by how coherently *all* onsets of the song fall onto
    one grid, which resolves the period to well below a frame. Returns (bpm, phase_ms);
    the coarse tempo is kept when no clearly better grid exists (e.g. tempo changes).
    """
    values = np.maximum(np.asarray(envelope, np.float64), 0)
    if not math.isfinite(bpm) or bpm <= 0 or values.sum() <= 1e-12 or len(values) < 4 * fps:
        return bpm, grid_coherence(values, fps, bpm)[1]
    total = values.sum()
    times = np.arange(len(values)) / fps  # seconds
    candidates = bpm * np.linspace(1 - tolerance, 1 + tolerance, steps)
    strength = np.empty(len(candidates))
    for start in range(0, len(candidates), 64):
        chunk = candidates[start : start + 64]
        phasors = np.exp(2j * np.pi * np.outer(chunk / 60, times)) @ values
        strength[start : start + len(chunk)] = np.abs(phasors) / total
    best = int(np.argmax(strength))
    refined = float(candidates[best])
    if 0 < best < len(candidates) - 1:
        left, middle, right = strength[best - 1 : best + 2]
        curvature = left - 2 * middle + right
        if curvature < 0:
            step = candidates[1] - candidates[0]
            refined += float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5)) * step
    coarse, _ = grid_coherence(values, fps, bpm)
    locked, phase = grid_coherence(values, fps, refined)
    if locked < max(MIN_COHERENCE, MIN_COHERENCE_GAIN * coarse):
        return bpm, grid_coherence(values, fps, bpm)[1]
    return refined, phase


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
