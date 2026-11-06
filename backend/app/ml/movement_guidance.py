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

October 2026: 88-90 % of all turns were sharper than 120 degrees and 47-69 % of the objects
sat in chains of same-direction ~144 degree turns: pentagram "star jumps". A constant
spacing band on a 512x384 field plus an autoregressive "same turn again" leaves a
regular star as the only way to keep going. Three changes break that loop:

* a pattern planner picks a shape per phrase (back-and-forth, zigzag, triangle, square,
  flowing curve, linear, rarely a star) from the section's style and proposes the next
  head along it;
* turn costs: chains of more than two same-direction sharp turns cost, and the recent
  share of sharp turns is pulled towards the requested angle mix;
* spacing depends on the rhythm (a 1/1 gap moves further than a 1/2 gap, accents a little
  further) and on the section instead of one band for every gap.
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
SPACING_LOW, SPACING_HIGH = 0.8, 1.5
SPACING_WEIGHT_LOW, SPACING_WEIGHT_HIGH = 2.0, 0.8
# Inside the band a mild pull to the target, so the rhythm-dependent targets are visible.
SPACING_CENTRE_WEIGHT = 0.5
# Distance relative to the 1/2 jump spacing per start-to-start gap (beats): readers see the
# rhythm in the spacing, so a longer gap moves further.
RHYTHM_SPACING = ((0.55, 1.0), (0.8, 1.1), (1.05, 1.2), (2.05, 1.1), (math.inf, 0.9))
ACCENT_SPACING = 1.15
SHARP_TURN = (115.0, 170.0)  # degrees: the turn range of star and triangle chains
CHAIN_FREE = 2  # same-direction sharp turns allowed in a row (one triangle)
CHAIN_WEIGHT = 1.2
ANGLE_WINDOW = 16
ANGLE_WEIGHT = 3.0
PATTERN_WEIGHT = 0.8
# Signed turn (degrees, positive = counter-clockwise in playfield coordinates) per step of
# each pattern; "alt" flips the sign every step, "keep" keeps one direction per phrase.
PATTERNS = {
    "back_forth": ((165, 180), "alt"),
    "zigzag": ((95, 130), "alt"),
    "triangle": ((115, 125), "keep"),
    "square": ((85, 95), "keep"),
    "flow": ((20, 60), "keep"),
    "linear": ((0, 20), "alt"),
    "star": ((140, 148), "keep"),
}
PATTERN_LENGTH = {"triangle": 3, "square": 4, "star": 5}
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
    plan: object = None  # SongPlan or None
    velocity: float | None = None  # requested slider velocity (SV multiplier)
    patterns: bool = True
    rng: np.random.Generator = field(init=False, repr=False)
    pattern: tuple = field(init=False, default=None, repr=False)  # (name, sign, start index, phrase key)

    def __post_init__(self):
        # Own generator: the guide must not shift the model's sampling sequence.
        self.rng = np.random.default_rng(self.seed + 7919)

    # --- section style --------------------------------------------------------------------
    def style_at(self, time):
        return self.plan.style_at(time) if self.plan is not None else None

    def angle_targets(self, time):
        style = self.style_at(time)
        if style is None:
            return self.sharp_angles, self.linear_angles
        # The requested mix (profile/user) and the section's style, averaged.
        return (
            0.5 * (self.sharp_angles + style["sharp"]),
            0.5 * (self.linear_angles + style["linear"]),
        )

    def accent_at(self, time, tolerance=22.0):
        if self.plan is None or not self.plan.accents:
            return 0.0
        hits = self.plan.accents_between(time - tolerance, time + tolerance)
        return max((strength for _, strength in hits), default=0.0)

    def velocity_at(self, time):
        """One slider velocity per section: the requested one, a little faster when intense."""
        if self.velocity is None:
            return None
        style = self.style_at(time)
        if style is None:
            return self.velocity
        factor = float(np.clip(0.9 + 0.2 * style["intensity"], 0.85, 1.15))
        return round(self.velocity * factor * 20) / 20

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
            factor = next(value for limit, value in RHYTHM_SPACING if start_gap <= limit)
            base = self.jump * factor
            free = (time - previous.time - previous.duration) / beat
            if previous.slider and free <= HALF_GAP:
                # A slider end leaves less time to move than a circle at the same gap.
                base *= 0.8
            if self.accent_at(time) > 0.5:
                base *= ACCENT_SPACING
        style = self.style_at(time)
        if style is not None:
            # The section's intensity already follows the loudness; only a little of the
            # frame loudness is kept for short dips inside a section.
            return base * style["spacing"] * dynamics(0.5 * (self.loudness_at(time) or 0.0))
        return base * dynamics(self.loudness_at(time))

    def spacing_cost(self, distance, target, stream=False):
        if target is None:
            return 0.0
        ratio = math.log2((distance + 16) / (target + 16))
        low = -math.inf if stream else math.log2(SPACING_LOW)
        cost = SPACING_WEIGHT_LOW * max(0.0, low - ratio) ** 2
        cost += SPACING_WEIGHT_HIGH * max(0.0, ratio - math.log2(SPACING_HIGH)) ** 2
        if not stream:
            cost += SPACING_CENTRE_WEIGHT * ratio**2
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

    # --- turns and patterns -----------------------------------------------------------------
    def _moves(self, history, count):
        """The last `count` movement vectors that are longer than a circle radius."""
        moves = []
        recent = history[-count - 1 :]
        for older, newer in zip(recent, recent[1:]):
            vector = np.array([newer.x - older.end_x, newer.y - older.end_y], np.float64)
            moves.append(vector)
        return moves

    @staticmethod
    def signed_turn(first, second):
        """Signed angle in degrees from movement `first` to movement `second`."""
        cross = first[0] * second[1] - first[1] * second[0]
        dot = first[0] * second[0] + first[1] * second[1]
        return math.degrees(math.atan2(cross, dot))

    def turn_history(self, history, count=ANGLE_WINDOW + 1):
        moves = self._moves(history, count)
        turns = []
        for first, second in zip(moves, moves[1:]):
            if np.linalg.norm(first) > self.radius and np.linalg.norm(second) > self.radius:
                turns.append(self.signed_turn(first, second))
            else:
                turns.append(None)
        return turns

    def candidate_turn(self, obj, history):
        if len(history) < 2:
            return None
        previous, older = history[-1], history[-2]
        first = np.array([previous.x - older.end_x, previous.y - older.end_y], np.float64)
        second = np.array([obj.x - previous.end_x, obj.y - previous.end_y], np.float64)
        if np.linalg.norm(first) <= self.radius or np.linalg.norm(second) <= self.radius:
            return None
        return self.signed_turn(first, second)

    def turn_cost(self, obj, history):
        """Same-direction sharp-turn chains and too many sharp turns overall."""
        turn = self.candidate_turn(obj, history)
        if turn is None:
            return 0.0
        low, high = SHARP_TURN
        sharp = low <= abs(turn) <= high
        cost = 0.0
        if sharp:
            chain = 0
            for previous in reversed(self.turn_history(history)):
                if previous is None or not low <= abs(previous) <= high:
                    break
                if np.sign(previous) != np.sign(turn):
                    break
                chain += 1
            if chain >= CHAIN_FREE:
                cost += CHAIN_WEIGHT * (chain + 1 - CHAIN_FREE)
        recent = [value for value in self.turn_history(history) if value is not None]
        if len(recent) >= 6:
            target_sharp, target_linear = self.angle_targets(obj.time)
            angles = np.abs(np.array(recent + [turn]))
            sharp_share = float(np.mean(angles > 120))
            linear_share = float(np.mean(angles < 45))
            if abs(turn) > 120:
                cost += ANGLE_WEIGHT * max(0.0, sharp_share - target_sharp - 0.05)
            if abs(turn) < 45:
                cost += 0.5 * ANGLE_WEIGHT * max(0.0, linear_share - target_linear - 0.15)
        return cost

    def phrase_key(self, time, beat):
        """A new pattern per bar (or after a rest / at a section start)."""
        section = self.plan.index_at(time) if self.plan is not None else 0
        return section, int(time // (4 * max(beat, 1.0)))

    def choose_pattern(self, time, beat, history):
        style = self.style_at(time)
        vocabulary = (style or {}).get("patterns") or {
            "back_forth": 0.25, "zigzag": 0.2, "triangle": 0.15, "square": 0.15, "flow": 0.2,
            "linear": 0.05,
        }  # fmt: skip
        names = list(vocabulary)
        weights = np.array([vocabulary[name] for name in names], np.float64)
        if self.pattern is not None and self.pattern[0] in names and len(names) > 1:
            # Vary: the previous phrase's shape is less likely to come again.
            weights[names.index(self.pattern[0])] *= 0.35
        weights /= weights.sum()
        name = names[int(self.rng.choice(len(names), p=weights))]
        sign = float(self.rng.choice((-1.0, 1.0)))
        self.pattern = (name, sign, len(history), self.phrase_key(time, beat))
        return self.pattern

    def planned_turn(self, time, beat, history):
        """The signed turn (degrees) the current pattern asks for at the next object."""
        if not self.patterns or len(history) < 2:
            return None
        key = self.phrase_key(time, beat)
        rest = (time - history[-1].time) / max(beat, 1.0) >= 2
        # Decided once per object: every candidate of this object sees the same pattern.
        if (
            self.pattern is None
            or self.pattern[3] != key
            or (rest and self.pattern[2] != len(history))
        ):
            self.choose_pattern(time, beat, history)
        name, sign, start, _ = self.pattern
        (low, high), mode = PATTERNS[name]
        step = len(history) - start
        length = PATTERN_LENGTH.get(name)
        if length and step >= length:
            # One closed shape, then the next one turns the other way.
            sign = -sign if (step // length) % 2 else sign
        if mode == "alt":
            sign = sign if step % 2 == 0 else -sign
        middle = 0.5 * (low + high)
        return sign * middle

    def pattern_cost(self, obj, history, beat):
        planned = self.planned_turn(obj.time, beat, history)
        turn = self.candidate_turn(obj, history)
        if planned is None or turn is None:
            return 0.0
        difference = abs((turn - planned + 180) % 360 - 180)
        return PATTERN_WEIGHT * difference / 180

    def slider_audio_cost(self, obj):
        """A slider body over strong accents swallows notes the music asks for."""
        if not obj.slider or self.plan is None or not self.plan.accents:
            return 0.0
        hits = self.plan.accents_between(obj.time + 25, obj.time + obj.duration - 25)
        return 0.7 * sum(strength for _, strength in hits)

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
        if not stream:
            total += self.turn_cost(obj, history)
            total += self.pattern_cost(obj, history, beat)
        total += self.slider_audio_cost(obj)
        return self.weight * total

    def jump_heads(self, time, history, beat, count=JUMP_PROPOSALS):
        """Head positions (px) at the requested distance: the planned turn first, then
        directions drawn from the requested angle mix."""
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
        target_sharp, target_linear = self.angle_targets(time)
        sharp = float(np.clip(target_sharp, 0.05, 0.9))
        linear = float(np.clip(target_linear, 0.05, 0.9))
        wide = max(0.05, 1 - sharp - linear)
        weights = np.array([linear, wide, sharp]) / (linear + wide + sharp)
        ranges = ((0, 45), (60, 120), (130, 180))  # degrees off the previous direction
        planned = self.planned_turn(time, beat, history)
        heads = []
        for index in range(count):
            if index == 0 and planned is not None:
                turn = math.radians(planned + self.rng.uniform(-8, 8))
            else:
                low_deg, high_deg = ranges[int(self.rng.choice(3, p=weights))]
                turn = math.radians(self.rng.uniform(low_deg, high_deg)) * self.rng.choice((-1, 1))
            distance = target * float(np.exp(np.clip(self.rng.normal() * 0.15, -0.3, 0.3)))
            head = None
            for angle in (turn, -turn):
                rotation = np.array(
                    [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
                )
                candidate = origin + (rotation @ direction) * distance
                if np.all(candidate >= low) and np.all(candidate <= high):
                    head = candidate
                    break
            if head is None:
                # Neither side fits: mirror the move off the border it would cross instead of
                # always jumping to the centre (which made every edge a sharp turn back).
                rotation = np.array(
                    [[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]]
                )
                vector = (rotation @ direction) * distance
                candidate = origin + vector
                for axis in (0, 1):
                    if candidate[axis] < low[axis] or candidate[axis] > high[axis]:
                        vector[axis] = -vector[axis]
                candidate = origin + vector
                head = candidate
            heads.append(np.clip(head, low, high))
        return heads


def build_movement_guide(settings, loudness=None, frame_ms=1.0, plan=None):
    """The guide for one song, or None when disabled or without a spacing target."""
    weight = float((settings or {}).get("movement_guidance", 1.0))
    if weight <= 0:
        return None
    targets = style_targets(settings)
    jump = targets.get("jump_spacing") or targets.get("spacing")
    slider_ratio = targets.get("slider_ratio") if settings.get("slider_guidance", True) else None
    if not jump and slider_ratio is None:
        return None
    guided = plan is not None and settings.get("plan_guidance", True)
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
        plan=plan if guided else None,
        velocity=target_velocity(settings),
        patterns=bool(settings.get("pattern_planner", True)),
    )


def head_geometry(curve, head):
    """A normalised 8-value geometry: `curve` (normalised) translated to start at `head` (px)."""
    points = np.asarray(curve, np.float64).reshape(4, 2) * PLAYFIELD
    moved = points - points[0] + np.asarray(head, np.float64)
    return (moved / PLAYFIELD).reshape(-1)
