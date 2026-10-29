"""Measurable mapping style for conditioning: difficulty, rhythm patterns, movement, sliders.

Every value is measured from the reference map itself, so the model learns what a requested
style means from real maps. Values are stored normalised (roughly 0..1); `decode` converts
back to the units shown in the UI. A feature can be unknown: its mask is then 0.
"""

import math
import re
import numpy as np

# Start-to-start gaps between consecutive objects, in beats of the current red line.
# They cover more than 99.8 % of gaps in a sample of 150 Insane/Extra maps.
GAP_BEATS = (
    1 / 8, 1 / 6, 1 / 4, 1 / 3, 3 / 8, 1 / 2, 2 / 3, 3 / 4, 5 / 6, 1, 7 / 6, 5 / 4, 4 / 3,
    3 / 2, 5 / 3, 7 / 4, 2, 5 / 2, 3, 7 / 2, 4, 9 / 2, 5, 6, 7, 8,
)  # fmt: skip
REST = len(GAP_BEATS)  # a pause longer than 8.5 beats: the next note follows the music
GAP_CLASSES = len(GAP_BEATS) + 1
TRIPLET_CLASSES = frozenset(
    index for index, value in enumerate(GAP_BEATS) if abs(value * 4 - round(value * 4)) > 1e-6
    and abs(value * 8 - round(value * 8)) > 1e-6
)
STREAM_GAP = 0.26  # at most a 1/4 beat (1/6 and 1/8 included)
HALF_GAP = 0.55

# key, group, encoder, decoder. Groups are dropped together during training so that the
# model also works when only some controls are set (for example just the target stars).
FEATURES = (
    ("stars", "difficulty"),
    ("aim_share", "difficulty"),
    ("ar", "settings"),
    ("cs", "settings"),
    ("od", "settings"),
    ("hp", "settings"),
    ("density", "density"),
    ("tempo", "tempo"),
    ("quarter_share", "rhythm"),
    ("half_share", "rhythm"),
    ("triplet_share", "rhythm"),
    ("burst_share", "rhythm"),
    ("stream_share", "rhythm"),
    ("stream_length", "rhythm"),
    ("spacing", "movement"),
    ("jump_spacing", "movement"),
    ("stream_spacing", "movement"),
    ("flow", "movement"),
    ("sharp_angles", "movement"),
    ("linear_angles", "movement"),
    ("stacks", "movement"),
    ("slider_ratio", "sliders"),
    ("slider_velocity", "sliders"),
    ("short_sliders", "sliders"),
    ("repeat_sliders", "sliders"),
    ("combo_length", "structure"),
)
KEYS = tuple(key for key, _ in FEATURES)
INDEX = {key: index for index, key in enumerate(KEYS)}
GROUPS = {}
for _key, _group in FEATURES:
    GROUPS.setdefault(_group, []).append(INDEX[_key])
FEATURE_COUNT = len(FEATURES)
CONDITION_SIZE = 2 * FEATURE_COUNT  # masked values followed by the mask itself
# Chance that a group is hidden from the model in one training example.
GROUP_DROPOUT = {
    "difficulty": 0.15,
    "settings": 0.0,
    "density": 0.2,
    "tempo": 0.1,  # generation first estimates the timing with the tempo still unknown
    "rhythm": 0.35,
    "movement": 0.35,
    "sliders": 0.35,
    "structure": 0.5,
}
MAPPER_DROPOUT = 0.3

_SCALE = {
    "stars": 10,
    "ar": 10,
    "cs": 10,
    "od": 10,
    "hp": 10,
    "density": 10,
    "spacing": 512,
    "jump_spacing": 512,
    "stream_spacing": 128,
    "slider_velocity": 560,
    "combo_length": 16,
}
LIMITS = {
    "stars": (0, 12),
    "ar": (0, 10),
    "cs": (0, 10),
    "od": (0, 10),
    "hp": (0, 10),
    "density": (0.1, 12),
    "tempo": (40, 400),
    "stream_length": (0, 256),
    "spacing": (0, 512),
    "jump_spacing": (0, 512),
    "stream_spacing": (0, 256),
    "slider_velocity": (10, 2000),
    "combo_length": (1, 64),
}


def encode(key, value):
    """UI units to the normalised model value."""
    low, high = LIMITS.get(key, (0, 1))
    value = float(np.clip(value, low, high))
    if key == "tempo":
        return math.log2(value / 120) / 2
    if key == "stream_length":
        return math.log2(value) / 6 if value >= 1 else 0.0
    return value / _SCALE.get(key, 1)


def decode(key, value):
    value = float(value)
    if key == "tempo":
        return 120 * 2 ** (value * 2)
    if key == "stream_length":
        return 2 ** (value * 6) if value > 0 else 0.0
    return value * _SCALE.get(key, 1)


def gap_class(gap_ms, beat_ms):
    """Nearest rhythm class of a start-to-start gap, -1 when it fits none."""
    if beat_ms <= 0 or gap_ms <= 0:
        return -1
    beats = gap_ms / beat_ms
    if beats > 8.5:
        return REST
    differences = np.abs(np.asarray(GAP_BEATS) - beats)
    index = int(differences.argmin())
    return index if differences[index] <= max(0.04, 12 / beat_ms) else -1


def allowed_gaps(divisor):
    """Gap classes that land on the chosen beat grid (1/12 mixes 1/4 and 1/3)."""
    return np.array(
        [abs(value * divisor - round(value * divisor)) < 1e-6 for value in GAP_BEATS], dtype=bool
    )


def dominant_beat(beatmap):
    reds = [
        (point.time, point.beat_length)
        for point in sorted(beatmap.timing, key=lambda point: point.time)
        if not point.inherited and point.beat_length > 0
    ]
    if not reds:
        return 500.0
    end = max((obj.time + obj.duration for obj in beatmap.objects), default=reds[-1][0])
    covered = {}
    for index, (time, length) in enumerate(reds):
        following = reds[index + 1][0] if index + 1 < len(reds) else end
        covered[round(length, 3)] = covered.get(round(length, 3), 0) + max(0.0, following - time)
    return float(max(covered, key=covered.get)) if any(covered.values()) else float(reds[0][1])


def circle_radius(cs):
    return 54.4 - 4.48 * cs


def _median(values):
    return float(np.median(values)) if len(values) else None


def _share(flags):
    return float(np.mean(flags)) if len(flags) else None


def measure(beatmap, attributes=None):
    """Raw style values (UI units) of one map. Unknown values are None."""
    objects, info, difficulty = beatmap.objects, beatmap.info(), beatmap.difficulty
    values = {key: None for key in KEYS}
    values.update(
        ar=difficulty.get("ApproachRate", difficulty.get("OverallDifficulty", 5)),
        cs=difficulty.get("CircleSize", 4),
        od=difficulty.get("OverallDifficulty", 5),
        hp=difficulty.get("HPDrainRate", 5),
        density=min(info["density"], 10),
        tempo=60000 / dominant_beat(beatmap),
    )
    if attributes:
        stars = attributes.get("star_rating")
        if stars is not None and math.isfinite(stars):
            values["stars"] = float(stars)
        aim, speed = attributes.get("aim_difficulty"), attributes.get("speed_difficulty")
        if aim is not None and speed is not None and aim + speed > 0:
            values["aim_share"] = float(aim / (aim + speed))
    if len(objects) < 2:
        return values
    radius = circle_radius(values["cs"])
    pairs = list(zip(objects, objects[1:]))
    gaps = np.array([(b.time - a.time) / a.beat_length for a, b in pairs])
    distances = np.array([math.hypot(b.x - a.end_x, b.y - a.end_y) for a, b in pairs])
    previous_circle = np.array([not a.slider for a, _ in pairs])
    quarter = (gaps > 0) & (gaps <= STREAM_GAP)
    half = (gaps > STREAM_GAP) & (gaps <= HALF_GAP)
    values["quarter_share"] = _share(quarter)
    values["half_share"] = _share(half)
    classes = [gap_class(b.time - a.time, a.beat_length) for a, b in pairs]
    values["triplet_share"] = _share([value in TRIPLET_CLASSES for value in classes])
    # Runs of consecutive <=1/4 gaps: 3-8 objects are bursts, 9 or more are streams.
    runs, length = [], 1
    for flag in quarter:
        if flag:
            length += 1
            continue
        if length > 1:
            runs.append(length)
        length = 1
    if length > 1:
        runs.append(length)
    total = len(objects)
    values["burst_share"] = sum(run for run in runs if 3 <= run <= 8) / total
    values["stream_share"] = sum(run for run in runs if run >= 9) / total
    patterned = [run for run in runs if run >= 3]
    values["stream_length"] = float(np.mean(patterned)) if patterned else 0.0
    values["spacing"] = float(np.median(distances))
    values["jump_spacing"] = _median(distances[half & previous_circle])
    values["stream_spacing"] = _median(distances[quarter & previous_circle])
    vectors = np.array([[b.x - a.end_x, b.y - a.end_y] for a, b in pairs])
    lengths = np.linalg.norm(vectors, axis=1)
    if len(vectors) > 1:
        dot = (vectors[1:] * vectors[:-1]).sum(-1)
        denominator = np.maximum(lengths[1:] * lengths[:-1], 1e-6)
        cosine = np.where(lengths[1:] * lengths[:-1] > 1e-6, dot / denominator, 0.0)
        values["flow"] = float(np.clip((cosine.mean() + 1) / 2, 0, 1))
        moving = (lengths[1:] > radius) & (lengths[:-1] > radius)
        values["sharp_angles"] = _share(cosine[moving] < -0.5)
        values["linear_angles"] = _share(cosine[moving] > 0.5)
    values["stacks"] = _share((distances < 3) & (gaps <= 1))
    sliders = [obj for obj in objects if obj.slider and obj.duration > 0]
    values["slider_ratio"] = len(sliders) / total
    if sliders:
        values["slider_velocity"] = _median(
            [obj.path_length * obj.repeats / (obj.duration / obj.beat_length) for obj in sliders]
        )
        values["short_sliders"] = _share([obj.duration / obj.beat_length <= HALF_GAP for obj in sliders])
        values["repeat_sliders"] = _share([obj.repeats > 1 for obj in sliders])
    combos = 1 + sum(obj.new_combo for obj in objects[1:])
    values["combo_length"] = total / combos
    return values


def vector(raw):
    """Normalised values and mask for a dict of raw values (missing or None means unknown)."""
    values = np.zeros(FEATURE_COUNT, np.float32)
    mask = np.zeros(FEATURE_COUNT, np.float32)
    for key, value in (raw or {}).items():
        if key in INDEX and value is not None and math.isfinite(float(value)):
            values[INDEX[key]] = encode(key, value)
            mask[INDEX[key]] = 1
    return values, mask


def condition_input(values, mask):
    return np.concatenate([values * mask, mask]).astype(np.float32)


def draw_style_dropout(rng):
    """Mask multipliers for one training example, drawn in a fixed order."""
    keep = np.ones(FEATURE_COUNT, np.float32)
    for group, indices in GROUPS.items():
        if rng.random() < GROUP_DROPOUT[group]:
            keep[indices] = 0
    mapper = rng.random() >= MAPPER_DROPOUT
    return keep, mapper


# Difficulty names that are not mapper names in "X's Insane" style versions.
_NOT_MAPPERS = {
    "easy", "normal", "hard", "insane", "extra", "expert", "extreme", "another", "hyper",
    "advanced", "beginner", "light", "lunatic", "master", "ultra", "special", "collab",
}  # fmt: skip
_POSSESSIVE = re.compile(r"^\s*(?P<name>[^\s'][^']{0,40}?)(?:'s|s'|')\s+\S")


def mapper_candidate(version):
    """Guest difficulty owner from names like "NiNo's Insane" or "deetz' Expert"."""
    match = _POSSESSIVE.match(version or "")
    if not match:
        return None
    name = match.group("name").strip()
    if not name or name.casefold() in _NOT_MAPPERS or len(name) < 2:
        return None
    return name


def resolve_mapper(creator, candidate, known):
    """The set creator owns a difficulty unless its name credits a guest mapper.

    `known` maps casefolded mapper names to their display form (creators of the dataset).
    An unverified possessive name ("Karen's Extra" may also be a character) is still not
    credited to the set host: rare names stay below the vocabulary threshold and become
    "unknown", which is safer than teaching the host's embedding a guest's style.
    """
    creator = (creator or "").strip() or "Unknown"
    if candidate:
        folded = candidate.casefold()
        return known.get(folded, candidate)
    return known.get(creator.casefold(), creator)


def known_mappers(records):
    """Casefolded name -> most common spelling among set creators."""
    spellings = {}
    for record in records:
        name = (record.get("creator") or "").strip()
        if name:
            spellings.setdefault(name.casefold(), {}).setdefault(name, 0)
            spellings[name.casefold()][name] += 1
    return {folded: max(names, key=names.get) for folded, names in spellings.items()}


def mapper_vocabulary(names, minimum):
    """Mappers with at least `minimum` training maps, most frequent first; index 0 = unknown."""
    counts = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    chosen = [name for name, count in counts.items() if count >= minimum]
    return sorted(chosen, key=lambda name: (-counts[name], name.casefold()))


def _summary(rows, masks):
    values, mask = np.stack(rows), np.stack(masks)
    result = {}
    for key in KEYS:
        known = values[mask[:, INDEX[key]] > 0, INDEX[key]]
        if len(known):
            result[key] = round(decode(key, float(np.median(known))), 4)
    return result


def mapper_profiles(items, vocabulary):
    """Median style of every mapper in the vocabulary, in UI units."""
    groups = {name: [] for name in vocabulary}
    for item in items:
        if "style" in item and item.get("mapper_name") in groups:
            groups[item["mapper_name"]].append(item)
    result = {}
    for name, members in groups.items():
        if not members:
            continue
        stars = [
            decode("stars", item["style"][INDEX["stars"]])
            for item in members
            if item["style_mask"][INDEX["stars"]] > 0
        ]
        result[name] = {
            "maps": len(members),
            "stars_min": round(min(stars), 2) if stars else None,
            "stars_max": round(max(stars), 2) if stars else None,
            "style": _summary([item["style"] for item in members], [item["style_mask"] for item in members]),
        }
    return result


def star_profiles(items, step=0.5):
    """Typical values per star bucket ("6.0" = 5.75 to 6.25 stars) from the training maps."""
    buckets = {}
    for item in items:
        if "style_mask" not in item or item["style_mask"][INDEX["stars"]] <= 0:
            continue
        stars = decode("stars", item["style"][INDEX["stars"]])
        bucket = f"{round(stars / step) * step:.1f}"
        buckets.setdefault(bucket, []).append(item)
    return {
        bucket: {
            "maps": len(members),
            "style": _summary([item["style"] for item in members], [item["style_mask"] for item in members]),
        }
        for bucket, members in sorted(buckets.items(), key=lambda entry: float(entry[0]))
        if len(members) >= 3
    }
