"""Onsets of the pitched layer of a song (vocals, lead melody), from the mel spectrogram.

The learned onset head is trained on where mappers put notes, and in most training maps
that is the drum layer: kicks, snares and claps. Vocals that start between the drums are
rarely strong enough in that head to win the rhythm choice, so generated maps ignored the
singer. This module adds an explicit, audio-only cue for new pitched notes:

1. harmonic/percussive separation by median filtering (Fitzgerald 2010): pitched sound is
   steady along time, drum hits are broad along frequency;
2. a step detector on the harmonic level in the vocal/lead range (~300-3000 Hz): mean
   level of the next STEP_FRAMES frames minus the previous ones, so only rises that last
   (a new note) count, not the brief dip and recovery around a drum hit;
3. an adaptive threshold (local median), peak picking and a song-level normalisation.

Measured on a 200 BPM vocal song: 85 % of the melodic onsets away from any drum hit lie
within 20 ms of the 1/4 grid (53 % by chance), most of them on 1/4 off-beats the drums leave empty.
It is a heuristic, not source separation: a pad or a guitar chord change counts as a
melodic onset too. It only adds evidence; the learned rhythm still decides.
"""

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from .audio import FRAME_MS, MELS, SAMPLE_RATE

LOW_HZ, HIGH_HZ = 300.0, 3000.0
HARMONIC_FRAMES = 17  # ~200 ms: pitched notes are steady at least this long
PERCUSSIVE_BANDS = 7
STEP_FRAMES = 4  # ~46 ms before/after the candidate onset
MIN_DISTANCE_FRAMES = 4
# The smoothed harmonic part reports note starts ~15 ms early; one frame compensates.
LATENCY_FRAMES = 1
THRESHOLD_MS = 400  # local median window of the adaptive threshold
NORMALISE_PERCENTILE = 98
DEFAULT_WEIGHT = 0.6  # melodic evidence relative to the learned onset probability


def _hz_to_mel(frequencies):
    """Slaney mel scale, as librosa's default (htk=False)."""
    frequencies = np.asarray(frequencies, np.float64)
    mels = frequencies / (200.0 / 3)
    log_region = frequencies >= 1000.0
    mels = np.where(
        log_region,
        15.0 + np.log(np.maximum(frequencies, 1e-9) / 1000.0) / (np.log(6.4) / 27.0),
        mels,
    )
    return mels


def _mel_to_hz(mels):
    mels = np.asarray(mels, np.float64)
    frequencies = (200.0 / 3) * mels
    log_region = mels >= 15.0
    return np.where(log_region, 1000.0 * np.exp((np.log(6.4) / 27.0) * (mels - 15.0)), frequencies)


def mel_band_centres(bands=MELS, sample_rate=SAMPLE_RATE):
    """Centre frequencies of librosa's default mel filterbank (fmin 0, fmax sr / 2)."""
    edges = np.linspace(_hz_to_mel(0.0), _hz_to_mel(sample_rate / 2), bands + 2)
    return _mel_to_hz(edges)[1:-1]


def melodic_onsets(mel, low_hz=LOW_HZ, high_hz=HIGH_HZ):
    """Per-frame strength (0..1) of new pitched notes in the vocal/lead range."""
    mel = np.asarray(mel, np.float32)
    frames = len(mel)
    if mel.ndim != 2 or frames < 2 * HARMONIC_FRAMES:
        return np.zeros(frames, np.float32)
    centres = mel_band_centres(mel.shape[1])
    band = (centres >= low_hz) & (centres <= high_hz)
    if not band.any():
        return np.zeros(frames, np.float32)
    silent = mel.max(axis=1) <= -0.99
    # The model features are (dB + 40) / 40 relative to the loudest frame; back to power.
    power = np.power(10.0, (mel.astype(np.float64) * 40 - 40) / 10)
    harmonic = median_filter(power, size=(HARMONIC_FRAMES, 1), mode="nearest")
    percussive = median_filter(power, size=(1, PERCUSSIVE_BANDS), mode="nearest")
    mask = harmonic**2 / np.maximum(harmonic**2 + percussive**2, 1e-30)
    level = 10 * np.log10(np.maximum(power * mask, 1e-10))[:, band]
    sums = np.vstack([np.zeros((1, level.shape[1])), np.cumsum(level, axis=0)])
    centre = np.arange(STEP_FRAMES, frames - STEP_FRAMES)
    after = (sums[centre + STEP_FRAMES] - sums[centre]) / STEP_FRAMES
    before = (sums[centre] - sums[centre - STEP_FRAMES]) / STEP_FRAMES
    step = np.zeros(frames)
    step[centre] = np.maximum(0, after - before).mean(axis=1)
    window = max(3, int(round(THRESHOLD_MS / FRAME_MS)) | 1)
    step = np.maximum(0, step - median_filter(step, size=window, mode="nearest"))
    step[silent] = 0
    peaks, _ = find_peaks(step, distance=MIN_DISTANCE_FRAMES)
    peaks = peaks[step[peaks] > 0]
    if not len(peaks):
        return np.zeros(frames, np.float32)
    reference = float(np.percentile(step[peaks], NORMALISE_PERCENTILE))
    if reference <= 1e-9:
        return np.zeros(frames, np.float32)
    strength = np.zeros(frames, np.float32)
    shifted = np.minimum(peaks + LATENCY_FRAMES, frames - 1)
    strength[shifted] = np.clip(step[peaks] / reference, 0, 1)
    strength[silent] = 0
    return strength


def combined_onsets(onset, mel, weight=DEFAULT_WEIGHT):
    """The learned onset probability, raised where a melodic note starts."""
    onset = np.asarray(onset, np.float32)
    weight = float(weight or 0)
    if weight <= 0:
        return onset
    melodic = melodic_onsets(mel)[: len(onset)]
    combined = onset.copy()
    combined[: len(melodic)] = np.maximum(onset[: len(melodic)], weight * melodic)
    return combined
