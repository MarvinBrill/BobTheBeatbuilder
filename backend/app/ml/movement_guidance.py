"""Song-level guidance for object placement and slider choice (style-v2 generation).

The decoder predicts each movement relative to the previous one (the last motion feature is
the log ratio to the previous spacing). Free generation feeds its own output back, so a
small bias towards shorter moves compounds: a 6-star map started with 130 px jumps and
ended with 12 px moves circling the playfield edge. Nothing tied the spacing back to the
style the user (or the star profile) asked for. The same held for sliders: 81 % sliders of
30 px (shorter than a circle) where the profile asked for 35 % sliders at 1.36x velocity.

The guide adds soft costs and a few extra proposals to the candidate ranking:

* spacing: a tolerance band around the requested spacing for the current gap
  (stream spacing for 1/4, jump spacing for 1/2 and longer), scaled down in quiet parts;
* jump proposals: model-shaped objects moved to the requested distance in directions drawn
  from the requested angle mix, so a jump exists even when every mixture mean is a stack;
* playfield use: a mild cost for hugging the border and for orbiting the centre;
* slider share: a cost once the recent slider share runs far from the requested ratio.

Everything is soft: a confident model can still place a stack, a stream or a slider run.
"""

from dataclasses import dataclass, field
import math

import numpy as np

from .style import HALF_GAP, STREAM_GAP

PLAYFIELD = np.array([512.0, 384.0])
CENTRE = PLAYFIELD / 2
STYLE_SOURCES = ("profile_style", "calibration_style", "style")  # later sources win
BASE_SLIDER_VELOCITY = 140.0  # px per beat at SV 1.0 with SliderMultiplier 1.4

# Spacing band: no cost from LOW to HIGH times the target (measured as (d+16)/(t+16)).
SPACING_LOW, SPACING_HIGH = 0.85, 1.45
SPACING_WEIGHT_LOW, SPACING_WEIGHT_HIGH = 3.0, 0.8
MAX_SPACING_COST = 4.0
DYNAMICS_DB_PER_HALVING = 12.0  # 12 dB quieter -> half the spacing
DYNAMICS_FLOOR = 0.6
JUMP_PROPOSALS = 3
EDGE_ZONE = 24.0  # px beyond the object radius in which the border cost rises
EDGE_WEIGHT = 0.4
ORBIT_HISTORY = 8
ORBIT_WEIGHT = 0.8
SLIDER_WINDOW = 16
SLIDER_TOLERANCE = 0.05
SLIDER_WEIGHT = 3.0
# Slider velocity band around the requested velocity (SV multiplier units).
VELOCITY_BAND = (0.55, 1.7)
UNGUIDED_VELOCITY = (0.1, 10.0)
MIN_SLIDER_RADII = 1.2  # a slider shorter than this many radii hides under its own head


def style_targets(settings):
    """Requested style values in UI units: star profile < calibration < user style."""
    merged = {}
    for source in STYLE_SOURCES:
        for key, value in ((settings or {}).get(source) or {}).items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                merged[key] = value
    return merged


def circle_radius(cs):
    return 54.4 - 4.48 * cs


def target_velocity(settings):
    """Requested slider velocity as an SV multiplier, None when unknown or unguided."""
    if not (settings or {}).get("slider_guidance", True):
        return None
    target = style_targets(settings).get("slider_velocity")
    if not target or target <= 0:
        return None
    return float(np.clip(target / BASE_SLIDER_VELOCITY, 0.1, 10.0))


def velocity_band(settings):
    """Allowed slider velocity (SV multiplier) for the requested style."""
    velocity = target_velocity(settings)
    if velocity is None:
        return UNGUIDED_VELOCITY
    low, high = VELOCITY_BAND
    return max(0.1, velocity * low), min(10.0, velocity * high)


def min_slider_length(settings):
    if not (settings or {}).get("slider_guidance", True):
        return 20.0
    return max(20.0, MIN_SLIDER_RADII * circle_radius(settings["cs"]))


def dynamics(loudness_db):
    """Spacing multiplier for the local loudness (dB relative to the loud parts, <= ~0)."""
    if loudness_db is None or not math.isfinite(loudness_db):
        return 1.0
    return float(np.clip(2 ** (min(loudness_db, 0) / DYNAMICS_DB_PER_HALVING), DYNAMICS_FLOOR, 1))


@dataclass
class MovementGuide:
    radius: float
    jump: float | None = None  # px after a 1/2 gap (and longer)
    stream: float | None = None  # px after a 1/4 gap
    slider_ratio: float | None = None
    sharp_angles: float = 0.35
    linear_angles: float = 0.25
    loudness: np.ndarray | None = None  # dB per frame relative to the loud parts
    frame_ms: float = 1.0
    weight: float = 1.0
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        # Own generator: the guide must not shift the model's sampling sequence.
        self.rng = np.random.default_rng(self.seed + 7919)

    def loudness_at(self, time):
        if self.loudness is None or not len(self.loudness):
            return None
        frame = int(np.clip(round(time / self.frame_ms), 0, len(self.loudness) - 1))
        return float(self.loudness[frame])

    def target_spacing(self, time, previous, beat):
        """Requested distance from the previous object's end to a head at `time` (px)."""
        if previous is None or self.jump is None:
            return None
        beat = max(float(beat), 1.0)
        start_gap = (time - previous.time) / beat
        if start_gap <= STREAM_GAP + 1e-6 and not previous.slider:
            base = self.stream if self.stream is not None else 0.25 * self.jump
        else:
            base = self.jump
            free = (time - previous.time - previous.duration) / beat
            if previous.slider and free <= HALF_GAP:
                # A slider end leaves less time to move than a circle at the same gap.
                base *= 0.8
        return base * dynamics(self.loudness_at(time))

    def spacing_cost(self, distance, target, stream=False):
        if target is None:
            return 0.0
        ratio = math.log2((distance + 16) / (target + 16))
        low = -math.inf if stream else math.log2(SPACING_LOW)
        cost = SPACING_WEIGHT_LOW * max(0.0, low - ratio) ** 2
        cost += SPACING_WEIGHT_HIGH * max(0.0, ratio - math.log2(SPACING_HIGH)) ** 2
        return min(MAX_SPACING_COST, cost)

    def edge_cost(self, x, y):
        border = min(x, PLAYFIELD[0] - x, y, PLAYFIELD[1] - y)
        return EDGE_WEIGHT * float(np.clip((self.radius + EDGE_ZONE - border) / EDGE_ZONE, 0, 1))

    def orbit_cost(self, x, y, history):
        """Cost of continuing a monotonic circle around the playfield centre."""
        heads = [(obj.x, obj.y) for obj in history[-ORBIT_HISTORY:]] + [(x, y)]
        if len(heads) < ORBIT_HISTORY + 1:
            return 0.0
        offsets = np.asarray(heads) - CENTRE
        if np.min(np.linalg.norm(offsets, axis=1)) < 100:
            return 0.0
        angles = np.unwrap(np.arctan2(offsets[:, 1], offsets[:, 0]))
        steps = np.diff(angles)
        if not (np.all(steps > 0) or np.all(steps < 0)):
            return 0.0
        turned = abs(angles[-1] - angles[0])
        return ORBIT_WEIGHT * float(np.clip((turned - 0.5 * math.pi) / (0.5 * math.pi), 0, 1))

    def slider_cost(self, obj, history):
        if self.slider_ratio is None or len(history) < SLIDER_WINDOW // 2:
            return 0.0
        recent = history[-SLIDER_WINDOW:]
        share = sum(item.slider for item in recent) / len(recent)
        if obj.slider and share > self.slider_ratio + SLIDER_TOLERANCE:
            return SLIDER_WEIGHT * (share - self.slider_ratio - SLIDER_TOLERANCE)
        if not obj.slider and share < self.slider_ratio - 2 * SLIDER_TOLERANCE:
            return 0.5 * SLIDER_WEIGHT * (self.slider_ratio - 2 * SLIDER_TOLERANCE - share)
        return 0.0

    def cost(self, obj, history, beat):
        """Total soft cost of a finished candidate object after `history`."""
        if not history:
            return 0.0
        previous = history[-1]
        distance = math.hypot(obj.x - previous.end_x, obj.y - previous.end_y)
        target = self.target_spacing(obj.time, previous, beat)
        stream = (obj.time - previous.time) / max(beat, 1.0) <= STREAM_GAP + 1e-6
        total = self.spacing_cost(distance, target, stream)
        total += self.edge_cost(obj.x, obj.y)
        total += self.orbit_cost(obj.x, obj.y, history)
        total += self.slider_cost(obj, history)
        return self.weight * total

    def jump_heads(self, time, history, beat, count=JUMP_PROPOSALS):
        """Head positions (px) at the requested distance in varied directions."""
        if not history or count <= 0:
            return []
        previous = history[-1]
        target = self.target_spacing(time, previous, beat)
        if target is None or target < self.radius:
            return []  # streams and stacks are the model's own business
        origin = np.array([previous.end_x, previous.end_y], np.float64)
        if len(history) > 1:
            older = history[-2]
            direction = np.array([previous.x - older.end_x, previous.y - older.end_y])
        else:
            direction = CENTRE - origin
        if np.linalg.norm(direction) < 1e-6:
            direction = self.rng.normal(size=2)
        direction = direction / np.linalg.norm(direction)
        margin = math.ceil(self.radius) + 1
        low, high = np.array([margin, margin]), PLAYFIELD - margin
        sharp = float(np.clip(self.sharp_angles, 0.05, 0.9))
        linear = float(np.clip(self.linear_angles, 0.05, 0.9))
        wide = max(0.05, 1 - sharp - linear)
        weights = np.array([linear, wide, sharp]) / (linear + wide + sharp)
        ranges = ((0, 45), (60, 120), (130, 180))  # degrees off the previous direction
        heads = []
        for _ in range(count):
            low_deg, high_deg = ranges[int(self.rng.choice(3, p=weights))]
            turn = math.radians(self.rng.uniform(low_deg, high_deg)) * self.rng.choice((-1, 1))
            distance = target * float(np.exp(np.clip(self.rng.normal() * 0.15, -0.3, 0.3)))
            for angle in (turn, -turn, None):
                if angle is None:  # neither side fits: jump towards the centre instead
                    vector = CENTRE - origin
                    vector = vector / max(np.linalg.norm(vector), 1e-6)
                else:
                    rotation = np.array(
                        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
                    )
                    vector = rotation @ direction
                head = origin + vector * distance
                if np.all(head >= low) and np.all(head <= high):
                    break
            heads.append(np.clip(head, low, high))
        return heads


def build_movement_guide(settings, loudness=None, frame_ms=1.0):
    """The guide for one song, or None when disabled or without a spacing target."""
    weight = float((settings or {}).get("movement_guidance", 1.0))
    if weight <= 0:
        return None
    targets = style_targets(settings)
    jump = targets.get("jump_spacing") or targets.get("spacing")
    slider_ratio = targets.get("slider_ratio") if settings.get("slider_guidance", True) else None
    if not jump and slider_ratio is None:
        return None
    return MovementGuide(
        radius=circle_radius(settings["cs"]),
        jump=float(jump) if jump else None,
        stream=targets.get("stream_spacing"),
        slider_ratio=slider_ratio,
        sharp_angles=targets.get("sharp_angles", 0.35),
        linear_angles=targets.get("linear_angles", 0.25),
        loudness=None if loudness is None else np.asarray(loudness, np.float32),
        frame_ms=frame_ms,
        weight=weight,
        seed=int(settings.get("seed", 0) or 0),
    )


def head_geometry(curve, head):
    """A normalised 8-value geometry: `curve` (normalised) translated to start at `head` (px)."""
    points = np.asarray(curve, np.float64).reshape(4, 2) * PLAYFIELD
    moved = points - points[0] + np.asarray(head, np.float64)
    return (moved / PLAYFIELD).reshape(-1)
