import hashlib
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
from scipy.ndimage import uniform_filter1d
from ..beatmaps import parse
from ..library import digest
from ..config import DATA
from .audio import features, FRAME_MS
from .architecture import GLOBAL_TOKENS, DURATION_BEATS
from .network import conditions
from .split import song_groups
from .movement import history_context, motion_numpy
from . import preparation_cache, style

# Packed event rows: time, geometry(8), kind, seconds, duration class, repeat class,
# combo, geometry mask, previous event(16), motion context(8), movement(8),
# beat length in seconds, rhythm class of the gap to the next object (-1: none).
EVENT_COLUMNS = 49
LEGACY_COLUMNS = 47


def cached_array(name, values):
    destination = DATA / "cache" / name
    if destination.exists():
        try:
            existing = np.load(destination, mmap_mode="r", allow_pickle=False)
            if (
                existing.shape == values.shape
                and existing.dtype == np.float32
                and np.isfinite(existing).all()
                and np.array_equal(existing, values)
            ):
                return existing
            del existing
        except (ValueError, OSError):
            pass
    temporary = destination.with_name(f"{destination.stem}-{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as output:
            np.save(output, np.asarray(values, dtype=np.float32), allow_pickle=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return np.load(destination, mmap_mode="r", allow_pickle=False)


def summary_features(mel):
    edges = np.linspace(0, len(mel), GLOBAL_TOKENS + 1).astype(int)
    return np.stack(
        [
            np.concatenate(
                [mel[left : max(left + 1, right)].mean(0), mel[left : max(left + 1, right)].max(0)]
            )
            for left, right in zip(edges[:-1], edges[1:])
        ]
    ).astype(np.float32)


def object_geometry(obj):
    if obj.curve:
        return (np.asarray(obj.curve) / [512, 384]).reshape(-1).astype(np.float32)
    return np.tile([obj.x / 512, obj.y / 384], 4).astype(np.float32)


def event_features(obj, previous=None):
    delta = (obj.time - previous.time) / 1000 if previous else obj.time / 1000
    distance = np.hypot(obj.x - previous.end_x, obj.y - previous.end_y) / 512 if previous else 0
    return np.array(
        [
            *object_geometry(obj),
            obj.end_x / 512,
            obj.end_y / 384,
            obj.duration / 4000,
            float(obj.slider),
            obj.repeats / 4,
            min(delta / 4, 4),
            float(obj.new_combo),
            distance,
        ],
        np.float32,
    )


def next_gap_classes(beatmap):
    """Rhythm target per object: the beat-relative gap to the following object."""
    objects, result = beatmap.objects, []
    for index, obj in enumerate(objects):
        if index + 1 == len(objects):
            result.append(-1)
            continue
        following = objects[index + 1]
        # A spinner between both objects is not a rhythm decision the decoder makes.
        interrupted = any(obj.time < start < following.time for start, _ in beatmap.ignored)
        result.append(-1 if interrupted else style.gap_class(following.time - obj.time, obj.beat_length))
    return result


def pack_events(beatmap, frame_count):
    rows, history, rejected_geometry = [], [], 0
    frames = set()
    gaps = next_gap_classes(beatmap)
    for index, obj in enumerate(beatmap.objects):
        frame = round(obj.time / FRAME_MS)
        if frame < 0 or frame >= frame_count or not 0 <= obj.x <= 512 or not 0 <= obj.y <= 384:
            raise ValueError("Object outside audio or playfield")
        if frame in frames:
            raise ValueError("Simultaneous objects below audio time resolution")
        frames.add(frame)
        geometry = object_geometry(obj)
        duration_beats = obj.duration / obj.beat_length
        duration_class = int(np.abs(np.array(DURATION_BEATS) - duration_beats).argmin())
        supported = (
            obj.geometry_valid
            and np.isfinite(geometry).all()
            and np.all((geometry >= -0.5) & (geometry <= 1.5))
            and duration_beats <= 16
            and obj.time + obj.duration < frame_count * FRAME_MS
        )
        if obj.slider and not supported:
            rejected_geometry += 1
        previous = (
            event_features(history[-1], history[-2] if len(history) > 1 else None)
            if history
            else np.zeros(16, np.float32)
        )
        context = history_context(history)
        motion = motion_numpy(
            geometry, obj.duration / 1000, obj.repeats, obj.slider, obj.time / 1000, context
        )
        rows.append(
            [
                obj.time / 1000,
                *geometry,
                float(obj.slider),
                obj.duration / 1000,
                duration_class,
                min(obj.repeats, 4) - 1,
                float(obj.new_combo),
                float(supported),
                *previous,
                *context,
                *motion,
                obj.beat_length / 1000,
                gaps[index],
            ]
        )
        history.append(obj)
    if not rows:
        raise ValueError("No trainable circles or sliders")
    return np.asarray(rows, dtype=np.float32), rejected_geometry


def frame_labels(beatmap, frame_count):
    target = np.zeros((frame_count, 6), np.float32)
    target[:, 4] = 1
    for start, end in beatmap.ignored:
        target[
            max(0, int(start / FRAME_MS)) : max(0, min(frame_count, int(end / FRAME_MS) + 1)), 4
        ] = 0
    for obj in beatmap.objects:
        target[round(obj.time / FRAME_MS), 0] = 1
    # Local density is a learned musical envelope, including rests.
    target[:, 2] = np.log1p(
        uniform_filter1d(target[:, 0], max(1, round(4000 / FRAME_MS)), mode="constant")
        * 1000
        / FRAME_MS
    ) / np.log(11)
    timing = [(time, beat) for time, beat, inherited in beatmap.timing if not inherited]
    for index, (offset, beat) in enumerate(timing):
        end = timing[index + 1][0] if index + 1 < len(timing) else frame_count * FRAME_MS
        left, right = max(0, int(np.ceil(offset / FRAME_MS))), min(
            frame_count, int(np.ceil(end / FRAME_MS))
        )
        if right <= left:
            continue
        times = np.arange(left, right) * FRAME_MS
        phase = ((times - offset + beat / 2) % beat) - beat / 2
        target[left:right, 1] = np.maximum(0, 1 - np.abs(phase) / 35)
        target[left:right, 3] = np.clip(np.log(beat / 500), -2, 2)
        target[left:right, 5] = 1
    return target


def attribute_signature(record):
    """Cached style values depend on the calculated stars, not only on the file contents."""
    attributes = record.get("difficulty_attributes") or {}
    return hashlib.sha256(json.dumps(attributes, sort_keys=True).encode()).hexdigest()[:16]


def red_lines(beatmap):
    return [[float(time), float(beat)] for time, beat, inherited in beatmap.timing if not inherited][:64]


def prepare_serial(records, context, issues=None):
    issues = issues if issues is not None else []
    prepared, groups = [], song_groups(records)
    audio_summaries = {}
    for index, record in enumerate(records):
        context.report(
            0.15 * index / max(1, len(records)), f"Preparing song/map pair {index+1}/{len(records)}"
        )
        try:
            audio_path, map_path = Path(record["audio_path"]), Path(record["map_path"])
            audio_hash = digest(audio_path)
            if audio_hash != record["audio_hash"]:
                raise ValueError("Audio changed after dataset creation")
            map_hash = digest(map_path)
            key = hashlib.sha256((map_hash + audio_hash).encode()).hexdigest()
            if audio_hash not in audio_summaries:
                mel = features(audio_path, audio_hash)
                if not np.any(mel > -0.7):
                    raise ValueError("Silent audio")
                audio_summaries[audio_hash] = (mel, summary_features(mel))
            mel, summary = audio_summaries[audio_hash]
            signature = attribute_signature(record)
            cached = preparation_cache.load(key, len(mel), signature)
            identity = {
                "mel": mel,
                "summary": summary,
                "song": groups[index],
                "audio_hash": audio_hash,
                "audio_path": str(audio_path),
                "map_hash": map_hash,
                "map_id": record["id"],
            }
            if cached is not None:
                prepared.append({**identity, **cached})
                continue
            beatmap = parse(map_path)
            events, masked = pack_events(beatmap, len(mel))
            labels = frame_labels(beatmap, len(mel))
            info, diff = beatmap.info(), beatmap.difficulty
            params = {
                "ar": diff.get("ApproachRate", diff.get("OverallDifficulty", 5)),
                "cs": diff.get("CircleSize", 4),
                "od": diff.get("OverallDifficulty", 5),
                "hp": diff.get("HPDrainRate", 5),
                "density": min(info["density"], 10),
            }
            if any(not 0 <= value <= 10 for value in params.values()):
                raise ValueError("Difficulty outside supported 0–10 range")
            params.update(
                spacing=float(np.clip(np.median(events[1:, 39]) if len(events) > 1 else 0, 0, 1)),
                flow=float(
                    np.clip((events[2:, 41].mean() + 1) / 2 if len(events) > 2 else 0.5, 0, 1)
                ),
                slider_bias=float(events[:, 9].mean()),
            )
            values, mask = style.vector(style.measure(beatmap, record.get("difficulty_attributes")))
            prepared.append(
                {
                    "mel": mel,
                    "events": cached_array(f"phrase-events-v4-{key}.npy", events),
                    "labels": cached_array(f"phrase-labels-v3-{key}.npy", labels),
                    "summary": summary,
                    "condition": np.array(conditions(params), np.float32),
                    "style": values,
                    "style_mask": mask,
                    "mapper_candidate": style.mapper_candidate(beatmap.metadata.get("Version")),
                    "creator": beatmap.metadata.get("Creator", ""),
                    "timing": red_lines(beatmap),
                    "attribute_signature": signature,
                    "song": groups[index],
                    "audio_hash": audio_hash,
                    "audio_path": str(audio_path),
                    "map_hash": map_hash,
                    "map_id": record["id"],
                    "masked_geometry": masked,
                }
            )
            preparation_cache.save(key, len(mel), prepared[-1])
        except (ValueError, OSError, IndexError) as error:
            issues.append({"map_id": record["id"], "reason": str(error)})
    if not prepared:
        raise ValueError(f"No usable training data: {issues[:3]}")
    return prepared


def prepare(records, context, issues=None):
    from .preparation_workers import prepare_parallel

    return prepare_parallel(records, context, issues, prepare_serial, processes=True)
