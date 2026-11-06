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

With a song plan (see `song_plan.py`) the guide also knows the sections:

* the density target follows the section's style and intensity (calm intro, dense drop);
* accents: skipping one of a section's strongest onsets costs, landing on it is rewarded;
* monotony: the same gap for more than two bars costs, unless the music insists (strong
  onset evidence at the candidate);
* repeated sections reuse the rhythm of the section they repeat (with room for variation
  at the end of each 8-bar phrase);
* calm styles discourage 1/4 runs, the "streams" style encourages them where audible.

In October 2026 the onset head fired almost everywhere (precision 0.50 at recall 0.96),
so its evidence barely separated a strong beat from silence-adjacent off-beats.
`sharpen_onsets` multiplies it with the audio's own onset strength.
"""

from bisect import bisect_right
from dataclasses import dataclass, field
import math
import numpy as np
from scipy.ndimage import uniform_filter1d
from .audio import FRAME_MS
from .style import GAP_BEATS, REST, STREAM_GAP

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
ACCENT_SKIP_WEIGHT = 1.0  # per skipped accent of full strength
ACCENT_HIT_BONUS = 0.3
ACCENT_TOLERANCE_MS = 22.0
MONOTONY_BEATS = 8.0  # the same gap may run this long without a cost (two bars of 4/4)
MONOTONY_WEIGHT = 0.6
REPEAT_BONUS = 0.6
STREAM_STYLE_WEIGHT = 0.6
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


def density_rate(prediction, loudness=None, dynamics=1.0, plan=None, scale=1.0):
    """Target objects per second per frame from the learned density head and the loudness.

    With a song plan the section style replaces most of the loudness rule (the plan's
    intensity already follows the loudness); `scale` is the calibration's speed lever.
    """
    values = np.clip(np.asarray(prediction, np.float32)[:, 2], 0, 1.4)
    rate = np.power(11.0, values) - 1  # inverse of log1p(rate) / log(11) in frame_labels
    if loudness is not None and dynamics:
        strength = dynamics * (0.5 if plan is not None else 1.0)
        rate = rate * dynamics_factor(loudness, strength)
    if plan is not None and plan.sections:
        factor = np.ones(len(rate), np.float32)
        for section in plan.sections:
            first = int(np.clip(section.start_ms / FRAME_MS, 0, len(rate)))
            last = int(np.clip(section.end_ms / FRAME_MS, 0, len(rate)))
            factor[first:last] = plan.style_at(0.5 * (section.start_ms + section.end_ms))["density"]
        rate = rate * factor
    return np.clip(rate * float(scale), MIN_RATE, MAX_RATE).astype(np.float32)


def audio_onsets(mel):
    """The audio's own onset strength per frame (0..1), widened by +-2 frames."""
    from scipy.ndimage import maximum_filter1d
    from .tempo_grid import onset_envelopes

    full, low, _ = onset_envelopes(mel)
    strength = np.clip(np.maximum(full, 0.8 * low), 0, 1)
    return maximum_filter1d(strength, 5, mode="nearest").astype(np.float32)


def sharpen_onsets(onset, mel, weight=0.4):
    """Learned onset evidence times the audio's onset strength (geometric mix).

    `weight` 0 keeps the learned evidence; 1 uses the audio alone. A floor keeps quiet but
    musically sensible positions possible.
    """
    onset = np.asarray(onset, np.float32)
    weight = float(np.clip(weight, 0, 1))
    if weight <= 0 or mel is None:
        return onset
    audio = audio_onsets(mel)[: len(onset)]
    result = onset.copy()
    count = len(audio)
    result[:count] = np.power(np.clip(onset[:count], 1e-4, 1), 1 - weight) * np.power(
        0.08 + 0.92 * audio, weight
    )
    return result


@dataclass
class RhythmGuide:
    rate: np.ndarray  # target objects per second per frame
    weight: float = 1.5  # cost per squared octave of a gap shorter than the burst tolerance
    window_weight: float = 3.0  # cost per squared octave of sustained overmapping
    rest_weight: float = 0.5  # cost per squared octave of a gap far longer than expected
    stream_soft_limit: int = 16  # objects per short-gap run before a soft cost starts
    stream_cost: float = 0.2  # per object beyond the soft limit
    history: list = field(default_factory=list)  # head times of the objects placed so far
    classes: list = field(default_factory=list)  # gap classes chosen so far (REST included)
    plan: object = None  # SongPlan or None
    accent_weight: float = ACCENT_SKIP_WEIGHT
    monotony_weight: float = MONOTONY_WEIGHT
    repeat_weight: float = REPEAT_BONUS

    def rate_at(self, start, end=None):
        first = int(np.clip(round(start / FRAME_MS), 0, len(self.rate) - 1))
        last = first if end is None else int(np.clip(round(end / FRAME_MS), 0, len(self.rate) - 1))
        if last <= first:
            return float(self.rate[first])
        return float(self.rate[first : last + 1].mean())

    def accent_cost(self, release, time):
        """Skipped accents between the release and `time`, minus a bonus for hitting one."""
        if self.plan is None or not self.plan.accents:
            return 0.0
        total = 0.0
        for _, strength in self.plan.accents_between(release + ACCENT_TOLERANCE_MS, time - ACCENT_TOLERANCE_MS):
            total += self.accent_weight * strength
        for accent, strength in self.plan.accents_between(time - ACCENT_TOLERANCE_MS, time + ACCENT_TOLERANCE_MS):
            total -= ACCENT_HIT_BONUS * strength
        return total

    def monotony_cost(self, gap_class, evidence):
        """Cost of continuing the same gap for longer than two bars."""
        if gap_class is None or gap_class == REST or not self.classes:
            return 0.0
        span = 0.0
        for previous in reversed(self.classes):
            if previous != gap_class:
                break
            span += GAP_BEATS[previous]
        span += GAP_BEATS[gap_class]
        if span <= MONOTONY_BEATS:
            return 0.0
        excess = min(2.0, (span - MONOTONY_BEATS) / MONOTONY_BEATS + 0.5)
        return self.monotony_weight * excess * (1 - float(np.clip(evidence, 0, 1)))

    def repeat_bonus(self, time, beat):
        """Negative cost when `time` matches a head of the section this one repeats."""
        if self.plan is None or not self.history:
            return 0.0
        source = self.plan.repeat_source(time)
        if source is None:
            return 0.0
        section, shift = source
        current = self.plan.section_at(time)
        phrase = 8 * 4 * max(beat, 1.0)
        # Vary the last two bars of every 8-bar phrase: the repeat is a guide, not a copy.
        if ((time - current.start_ms) % phrase) / phrase > 0.75:
            return 0.0
        target = time + shift
        position = bisect_right(self.history, target + 15)
        if position and abs(self.history[position - 1] - target) <= 15:
            return -self.repeat_weight
        return 0.0

    def style_cost(self, time, short, evidence):
        if self.plan is None or not short:
            return 0.0
        streams = self.plan.style_at(time)["streams"]
        if streams < 1:
            return STREAM_STYLE_WEIGHT * (1 - streams)
        return -0.2 * (streams - 1) * float(np.clip(evidence, 0, 1))

    def cost(
        self, previous_time, time, run=0, short=False, gap_class=None, evidence=0.0,
        release=None, beat=500.0,
    ):  # fmt: skip
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
        if self.plan is not None:
            total += self.accent_cost(previous_time if release is None else release, time)
            total += self.monotony_cost(gap_class, evidence)
            total += self.repeat_bonus(time, beat)
            total += self.style_cost(time, short, evidence)
        return total


def build_guide(prediction, mel, settings, plan=None):
    """The guide for one song, or None when the user disabled density guidance."""
    weight = float(settings.get("density_guidance", 1.0))
    if weight <= 0:
        return None
    loudness = loudness_profile(mel)
    rate = density_rate(
        prediction, loudness, settings.get("dynamics", 1.0), plan,
        float(settings.get("density_scale", 1.0)),
    )  # fmt: skip
    # An explicit maximum stream length replaces the soft limit.
    soft_limit = 0 if settings.get("max_stream") else int(settings.get("stream_soft_limit", 16))
    guided = plan is not None and settings.get("plan_guidance", True)
    return RhythmGuide(
        rate=rate,
        weight=1.5 * weight,
        window_weight=3.0 * weight,
        rest_weight=0.5 * weight,
        stream_soft_limit=soft_limit,
        plan=plan if guided else None,
        accent_weight=ACCENT_SKIP_WEIGHT * float(settings.get("accent_weight", 1.0)),
        monotony_weight=MONOTONY_WEIGHT * float(settings.get("variation", 1.0)),
    )


def is_short(gap_beats):
    return gap_beats <= STREAM_GAP
