"""Song-level guidance for the learned rhythm sampler (style-v2 "patterns" mode).

The gap head predicts the next gap from the recent objects. Teacher forcing makes "repeat
the previous gap" its strongest cue, so in free generation one 1/4 run can continue for
the rest of the song. Two song-level signals keep the rhythm tied to the music instead:

* the model's own local density head (``prediction[:, 2]``), trained on every reference
  map's 4 s object rate including rests, which the pattern sampler did not use before;
* the relative loudness of the audio, so quiet intros, breakdowns and outros are mapped
  sparser than the loud sections ("dynamics").

Both only add soft costs to the gap choice: bursts and streams stay possible where the
music and the density envelope support them.
"""

from bisect import bisect_right
from dataclasses import dataclass, field
import math
import numpy as np
from scipy.ndimage import uniform_filter1d
from .audio import FRAME_MS
from .style import STREAM_GAP

DENSITY_WINDOW_MS = 4000  # the window of the density head's training target
LOUDNESS_WINDOW_MS = 1500
LOUD_PERCENTILE = 90
# A gap may be this many octaves shorter than the local mean gap without a cost (bursts).
BURST_TOLERANCE = 1.0
# The realised rate over the density window may exceed the target by 2 ** this much.
WINDOW_TOLERANCE = 0.35
# A gap may be this many octaves longer than the local mean gap without a cost (rests).
REST_TOLERANCE = 1.5
MAX_REST_COST = 2.0  # the learned decision to rest still wins when the model is sure
MIN_RATE, MAX_RATE = 0.05, 20.0
DYNAMICS_FLOOR = 0.25
DECIBELS_PER_HALVING = 6.0


def loudness_profile(mel):
    """Smoothed loudness per frame in dB relative to the loud parts of the song (<= ~0)."""
    mel = np.asarray(mel, np.float32)
    decibels = np.where(mel > -1, mel * 40 - 40, -120.0)
    power = np.power(10.0, decibels / 10).sum(axis=1)
    smooth = uniform_filter1d(power, max(1, round(LOUDNESS_WINDOW_MS / FRAME_MS)), mode="nearest")
    frame_db = 10 * np.log10(np.maximum(smooth, 1e-12))
    audible = mel.max(axis=1) > -0.7
    reference = np.percentile(frame_db[audible], LOUD_PERCENTILE) if audible.any() else 0.0
    return np.minimum(frame_db - reference, 6.0).astype(np.float32)


def dynamics_factor(loudness, strength=1.0):
    """Rate multiplier: every 6 dB below the loud sections halves the target density."""
    strength = max(0.0, float(strength))
    factor = np.power(2.0, strength * np.minimum(loudness, 0) / DECIBELS_PER_HALVING)
    return np.clip(factor, DYNAMICS_FLOOR, 1.0).astype(np.float32)


def density_rate(prediction, loudness=None, dynamics=1.0):
    """Target objects per second per frame from the learned density head and the loudness."""
    values = np.clip(np.asarray(prediction, np.float32)[:, 2], 0, 1.4)
    rate = np.power(11.0, values) - 1  # inverse of log1p(rate) / log(11) in frame_labels
    if loudness is not None and dynamics:
        rate = rate * dynamics_factor(loudness, dynamics)
    return np.clip(rate, MIN_RATE, MAX_RATE).astype(np.float32)


@dataclass
class RhythmGuide:
    rate: np.ndarray  # target objects per second per frame
    weight: float = 1.5  # cost per squared octave of a gap shorter than the burst tolerance
    window_weight: float = 3.0  # cost per squared octave of sustained overmapping
    rest_weight: float = 0.5  # cost per squared octave of a gap far longer than expected
    stream_soft_limit: int = 16  # objects per short-gap run before a soft cost starts
    stream_cost: float = 0.2  # per object beyond the soft limit
    history: list = field(default_factory=list)  # head times of the objects placed so far

    def rate_at(self, start, end=None):
        first = int(np.clip(round(start / FRAME_MS), 0, len(self.rate) - 1))
        last = first if end is None else int(np.clip(round(end / FRAME_MS), 0, len(self.rate) - 1))
        if last <= first:
            return float(self.rate[first])
        return float(self.rate[first : last + 1].mean())

    def cost(self, previous_time, time, run=0, short=False):
        """Soft cost of placing the next head at `time` after a head at `previous_time`."""
        gap = max(1.0, time - previous_time)
        expected = 1000 / self.rate_at(previous_time, time)
        ratio = math.log2(expected / gap)
        total = self.weight * max(0.0, ratio - BURST_TOLERANCE) ** 2
        # Weaker in the other direction: a long rest in a dense section is still possible.
        total += min(MAX_REST_COST, self.rest_weight * max(0.0, -ratio - REST_TOLERANCE) ** 2)
        window_start = time - DENSITY_WINDOW_MS
        first = bisect_right(self.history, window_start)  # history is in time order
        count = len(self.history) - first
        if count:
            span = max(1000.0, min(DENSITY_WINDOW_MS, time - self.history[first]))
            realised = (count + 1) * 1000 / span
            target = self.rate_at(max(0.0, window_start), time)
            overflow = math.log2(realised / target) - WINDOW_TOLERANCE
            total += self.window_weight * max(0.0, overflow) ** 2
        if short and self.stream_soft_limit and run + 2 > self.stream_soft_limit:
            total += self.stream_cost * (run + 2 - self.stream_soft_limit)
        return total


def build_guide(prediction, mel, settings):
    """The guide for one song, or None when the user disabled density guidance."""
    weight = float(settings.get("density_guidance", 1.0))
    if weight <= 0:
        return None
    loudness = loudness_profile(mel)
    rate = density_rate(prediction, loudness, settings.get("dynamics", 1.0))
    # An explicit maximum stream length replaces the soft limit.
    soft_limit = 0 if settings.get("max_stream") else int(settings.get("stream_soft_limit", 16))
    return RhythmGuide(
        rate=rate,
        weight=1.5 * weight,
        window_weight=3.0 * weight,
        rest_weight=0.5 * weight,
        stream_soft_limit=soft_limit,
    )


def is_short(gap_beats):
    return gap_beats <= STREAM_GAP
