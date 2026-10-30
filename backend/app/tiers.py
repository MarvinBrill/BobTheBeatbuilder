"""osu!standard difficulty tiers and the numbers the Ranking Criteria attach to them.

The tier follows the usual star-rating names (Easy < 2.0 ★ <= Normal < 2.7 ★ <= Hard < 4.0 ★
<= Insane < 5.3 ★ <= Expert). The values below are the per-difficulty guidelines of the
osu! Ranking Criteria (https://osu.ppy.sh/wiki/en/Ranking_criteria/osu%21). They are used as
soft targets for generation and as a checklist for the finished draft, never as a claim
that a generated map is rankable.
"""

from dataclasses import dataclass
import math

TIERS = ("Easy", "Normal", "Hard", "Insane", "Expert")
STAR_LIMITS = (2.0, 2.7, 4.0, 5.3)


@dataclass(frozen=True)
class TierRules:
    name: str
    # Inclusive guideline ranges for the difficulty settings; None = no limit.
    ar: tuple[float | None, float | None]
    od: tuple[float | None, float | None]
    hp: tuple[float | None, float | None]
    cs: tuple[float | None, float | None]
    spinner_recovery_beats: float  # free beats between a spinner's end and the next object
    spinner_min_beats: float  # shortest recommended spinner
    full_overlap_beats: float | None  # objects at most this many beats apart must not fully overlap
    overlap_is_rule: bool  # True: RC rule, False: guideline
    distance_snap: tuple[float, float] | None  # allowed ratio to the map's typical spacing
    distance_snap_max_beats: float  # pairs further apart are judged more loosely
    max_stream: int | None  # longest 1/4 run before a warning
    max_half_chain: int | None  # longest actively clicked 1/2 chain before a warning
    slow_rhythm: bool  # Easy: mostly 1/1 or slower


# name, AR, OD, HP, CS, spinner recovery/min beats, full-overlap beats (rule?),
# distance snap ratio and its beat range, longest 1/4 run, longest clicked 1/2 chain, slow.
# fmt: off
RULES = {
    "Easy":   TierRules("Easy",   (None, 5), (1, 3),    (1, 3),    (None, 4), 4,   4, 1.0,  True,  (0.8, 1.2), 2.0, 0,    None, True),
    "Normal": TierRules("Normal", (4, 6),    (3, 5),    (3, 5),    (None, 5), 2,   3, 1.0,  True,  (0.8, 1.3), 1.0, 0,    3,    False),
    "Hard":   TierRules("Hard",   (6, 8),    (5, 7),    (4, 6),    (None, 6), 1,   2, 0.5,  True,  None,       0,   5,    None, False),
    "Insane": TierRules("Insane", (7, 9.3),  (7, 9),    (5, 8),    (None, 7), 0.5, 0, 0.25, False, None,       0,   None, None, False),
    "Expert": TierRules("Expert", (8, None), (8, None), (5, None), (None, 7), 0.5, 0, None, False, None,       0,   None, None, False),
}
# fmt: on


def tier_name(stars):
    """Tier of a star rating; None when the rating is unknown."""
    if stars is None or not math.isfinite(float(stars)):
        return None
    for name, limit in zip(TIERS, STAR_LIMITS):
        if stars < limit:
            return name
    return TIERS[-1]


def rules_for(stars):
    name = tier_name(stars)
    return RULES[name] if name else None


def low_difficulty(stars):
    """Easy and Normal: maps for new players, where readability rules are strict."""
    return tier_name(stars) in ("Easy", "Normal")


def settings_stars(settings):
    """The star rating generation aims at (explicit target first)."""
    for key in ("target_stars", "condition_stars"):
        value = (settings or {}).get(key)
        if value is not None:
            return float(value)
    return None


def preempt_ms(ar):
    """Approach time of osu! for an approach rate."""
    ar = float(ar)
    return 1200 + 600 * (5 - ar) / 5 if ar < 5 else 1200 - 750 * (ar - 5) / 5
