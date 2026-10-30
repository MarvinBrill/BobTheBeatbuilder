import json
import logging
import math
import zipfile
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from .. import difficulty, store
from ..beatmaps import write, parse
from ..artifacts import artifact_directory
from ..config import DATA
from ..timing import TimingPoint, red_points
from .audio import features, export_audio, FRAME_MS
from .network import conditions, device
from .phrase_model import create_model
from .rhythm_inference import predict_rhythm, timing_map
from .sequence_data import summary_features
from .sequence_generation import generate_objects, generate_objects_v2
from .architecture import ARCHITECTURE, FORMAT_VERSION, EVENT_SCHEMA, TARGET_SCHEMA
from .generation_trace import GenerationTrace, object_snapshot
from .combos import assign_combos
from .finishing import finish_map
from ..ranking_check import ranking_check
from . import style

STAR_TOLERANCE = 0.15
# Explicit style values the user can set; everything else is inferred by the model.
FIXED = {"ar", "cs", "od", "hp", "tempo", "stars", "aim_share"}


def mapper_index(metadata, name):
    if not name:
        return 0
    folded = name.strip().casefold()
    for index, known in enumerate(metadata.get("mapper_vocab") or []):
        if known.casefold() == folded:
            return index + 1
    return 0


# Style groups filled from the training maps at the target stars when the user left them
# open. Groups are dropped together in training, so they are filled together here too.
PROFILE_GROUPS = ("density", "rhythm", "movement", "sliders", "structure")


def profile_style(profile, user_style):
    """Typical values of the star bucket for every group the user did not touch.

    With only stars/AR/CS/OD/HP known the model rarely saw a condition this empty in
    training; free validation (full reference style) moved 100-150 px, real generation with
    an empty style collapsed to 5 px stacks. A typical 6-star style keeps it in range.
    """
    if not profile:
        return {}
    touched = {
        group for group, indices in style.GROUPS.items()
        if any(style.KEYS[index] in (user_style or {}) for index in indices)
    }  # fmt: skip
    chosen = {}
    for group in PROFILE_GROUPS:
        if group in touched:
            continue
        for index in style.GROUPS[group]:
            key = style.KEYS[index]
            if key in FIXED or profile.get(key) is None:
                continue
            chosen[key] = float(profile[key])
    return chosen


def styled_condition(settings, tempo):
    raw = {key: settings[key] for key in ("ar", "cs", "od", "hp")}
    raw["tempo"] = tempo
    if settings.get("condition_stars") is not None:
        raw["stars"] = settings["condition_stars"]
    if settings.get("aim_share") is not None:
        raw["aim_share"] = settings["aim_share"]
    chosen = {
        **(settings.get("profile_style") or {}),
        **(settings.get("calibration_style") or {}),
        **(settings.get("style") or {}),
    }
    for key, value in chosen.items():
        if key in style.INDEX and key not in FIXED and value is not None:
            raw[key] = value
    values, mask = style.vector(raw)
    return style.condition_input(values, mask)


def dominant_tempo(timing, duration_ms):
    points, covered = red_points(timing), {}
    for index, point in enumerate(points):
        following = points[index + 1].time if index + 1 < len(points) else duration_ms
        covered[point.beat_length] = covered.get(point.beat_length, 0) + max(
            0.0, following - max(0.0, point.time)
        )
    beat = max(covered, key=covered.get) if any(covered.values()) else points[0].beat_length
    return 60000 / beat


def star_profile(metadata, stars):
    """Typical style of the training maps closest to `stars` (None when unknown)."""
    profiles = metadata.get("star_profiles") or {}
    if stars is None or not profiles:
        return None
    bucket = min(profiles, key=lambda key: abs(float(key) - float(stars)))
    return profiles[bucket].get("style") or None


def trained_star_range(metadata):
    profiles = metadata.get("star_profiles") or {}
    if not profiles:
        return 0.5, 12.0
    values = [float(key) for key in profiles]
    return max(0.5, min(values) - 0.25), min(12.0, max(values) + 0.25)


def calibration_step(settings, target, stars, attributes, profile, star_range, patterns=None):
    """Next calibration settings from the measured stars and their aim/speed split.

    `patterns` (the map's own measured style, see `pattern_summary`) catches an aim deficit
    the aim/speed split hides: when the spacing collapses, aim *and* speed drop and the
    split can still look aim-heavy. A measured jump spacing far below the requested one
    therefore also pulls the jump lever instead of densifying the rhythm.

    Raising the condition stars and a uniform `tempo_bias` were the only levers before; both
    push every gap of the song towards 1/4 when the real deficit is aim (tiny spacing), which
    produced song-long streams. The stars condition now stays near the trained range and the
    target, `tempo_bias` only changes when speed is the weaker skill, and an aim deficit asks
    the model for wider jumps instead.
    """
    updated = dict(settings)
    error = target - stars
    low, high = star_range
    lower, upper = max(low, target - 1.5), min(high, target + 1.5)
    if lower > upper:  # a target outside the trained range: stay at the nearest edge
        lower = upper = float(np.clip(target, low, high))
    updated["condition_stars"] = float(
        np.clip(settings["condition_stars"] + 0.6 * error, lower, upper)
    )
    aim = (attributes or {}).get("aim_difficulty") or 0.0
    speed = (attributes or {}).get("speed_difficulty") or 0.0
    # The user's own aim/speed balance wins over the typical one of the star bucket.
    typical_share = settings.get("aim_share") or (profile or {}).get("aim_share") or 0.55
    share = aim / (aim + speed) if aim + speed > 0 else typical_share
    user_style = settings.get("style") or {}
    lever = "jump_spacing" not in user_style and bool(profile and profile.get("jump_spacing"))
    calibrated = settings.get("calibration_style") or {}
    measured_jump = (patterns or {}).get("jump_spacing")
    collapsed = bool(
        lever
        and measured_jump is not None
        and measured_jump < 0.75 * calibrated.get("jump_spacing", profile["jump_spacing"])
    )
    if error > 0 and lever and (share < typical_share - 0.05 or collapsed):
        # Aim is missing: wider jumps, never denser rhythm.
        updated["tempo_bias"] = float(np.clip(min(settings["tempo_bias"], 0.0) - 0.1, -1, 1))
        current = calibrated.get("jump_spacing", profile["jump_spacing"])
        updated["calibration_style"] = {
            **calibrated,  # the stored setting can be an explicit None
            "jump_spacing": float(np.clip(current * (1 + 0.15 * min(error, 2)), 40, 400)),
        }
    else:
        updated["tempo_bias"] = float(np.clip(settings["tempo_bias"] + 0.25 * error, -1, 1))
    return updated


def mark_spinner_combos(objects, spinners):
    """osu! starts a new combo after every spinner; keep the objects consistent with that."""
    spinners = [tuple(span) for span in spinners]
    for start, _ in spinners:
        following = next((obj for obj in objects if obj.time > start), None)
        if following is not None:
            following.new_combo = True
    return spinners


def verified_write(
    path, objects, settings, audio_name, timing, trace=None, spinners=(), finish=None
):
    """Write the map and check that osu! will read back exactly the generated objects."""
    bpm, offset = 60000 / timing[0].beat_length, timing[0].time
    exported = finish.export() if finish is not None else None
    write(path, objects, settings, audio_name, bpm, offset, timing, spinners, exported)
    verified = parse(path)
    if len(verified.objects) != len(objects):
        raise ValueError("Generated beatmap validation failed")
    if finish is not None:
        expected_breaks = [tuple(map(float, span)) for span in finish.breaks]
        if [tuple(span) for span in verified.breaks] != expected_breaks:
            raise ValueError("Break periods changed during export")
        if verified.preview_time != finish.preview_time:
            raise ValueError("Preview point changed during export")
        if [obj.hitsound for obj in verified.objects] != [int(obj.hitsound) for obj in objects]:
            raise ValueError("Hitsounds changed during export")
        if verified.sound_points and verified.sound_points[0][2]:
            raise ValueError("The first timing point must not toggle kiai")
    expected_spinners = [(round(start), round(end)) for start, end in sorted(spinners)]
    if [(round(start), round(end)) for start, end in verified.ignored] != expected_spinners:
        raise ValueError("Spinner timing changed during export")
    for start, end in expected_spinners:
        if any(start - 1 < obj.time + obj.duration and obj.time < end + 1 for obj in objects):
            raise ValueError("Generated spinner overlaps an object")
    for index, (actual, expected) in enumerate(zip(verified.objects, objects)):
        if trace is not None:
            trace.record(
                "export_roundtrip",
                index=index,
                proposed=object_snapshot(expected),
                reparsed=object_snapshot(actual),
            )
        if abs(actual.duration - expected.duration) > 2:
            raise ValueError("Slider timing changed during export")
        if actual.time != round(expected.time):
            raise ValueError("Head timing changed during export")
        if actual.slider != expected.slider or actual.repeats != expected.repeats:
            raise ValueError("Object type or repeats changed during export")
        if not np.allclose(
            [actual.end_x, actual.end_y], [expected.end_x, expected.end_y], atol=0.75
        ):
            raise ValueError("Final object endpoint changed during export")
        if (
            index + 1 < len(verified.objects)
            and actual.time + actual.duration >= verified.objects[index + 1].time
        ):
            raise ValueError("Generated objects overlap in time")
    return bpm, offset


def measured_stars(path):
    if not difficulty.available():
        return None
    try:
        return difficulty.calculate_one(path)
    except ValueError:
        logging.exception("Star rating of the generated map could not be calculated")
        return None


def pattern_summary(objects, settings, timing):
    """The generated map's own rhythm/movement statistics, in the units of the controls."""
    from ..beatmaps import Beatmap

    beatmap = Beatmap(
        objects=objects,
        timing=list(timing),
        difficulty={
            "ApproachRate": settings["ar"],
            "CircleSize": settings["cs"],
            "OverallDifficulty": settings["od"],
            "HPDrainRate": settings["hp"],
        },
    )
    raw = style.measure(beatmap)
    keys = (
        "density", "quarter_share", "burst_share", "stream_share", "stream_length",
        "jump_spacing", "stream_spacing", "flow", "sharp_angles", "slider_ratio",
    )  # fmt: skip
    return {
        key: round(float(raw[key]), 4)
        for key in keys
        if raw.get(key) is not None and math.isfinite(raw[key])
    }


def generate(settings, context, trace=None):
    settings = {
        "seed": 42,
        "temperature": 0.8,
        "candidates": 12,
        "style": {},
        "target_stars": None,
        "aim_share": None,
        "mapper": None,
        "calibrate": False,
        "calibration_rounds": 3,
        "max_stream": 0,
        "tempo_bias": 0.0,
        "rhythm_mode": "patterns",
        "hitsounds": "basic",
        "kiai": True,
        "breaks": True,
        **settings,
    }
    if trace is None and settings.get("trace", False):
        trace = GenerationTrace()
    if trace is not None:
        trace.record("generation_settings", settings=settings)
    metadata = store.get("model", settings["model_id"])
    for field, expected in (
        ("architecture", ARCHITECTURE),
        ("format_version", FORMAT_VERSION),
        ("event_schema", EVENT_SCHEMA),
        ("target_schema", TARGET_SCHEMA),
    ):
        if metadata.get(field) != expected:
            raise ValueError(
                f"Incompatible model {field}: expected {expected!r}, got {metadata.get(field)!r}. "
                "Train a new Phrase Mapper model."
            )
    selected_device = device(settings["device"])
    model = create_model(metadata["architecture"], metadata.get("model_config"))
    if metadata.get("trained_slider_geometry") == 0:
        settings = {**settings, "slider_bias": 0}
    audio_path = Path(store.get("audio", settings["audio_id"])["path"])
    context.report(0.02, "Audio analysieren")
    mel = features(audio_path)
    model.load_state_dict(load_file(str(DATA / "models" / metadata["id"] / "model.safetensors")))
    model.to(selected_device).eval()
    styled = model.styled
    duration_ms = len(mel) * FRAME_MS
    mapper, known_timing = None, {}
    if styled:
        settings["mapper_index"] = mapper_index(metadata, settings.get("mapper"))
        mapper = torch.tensor([float(settings["mapper_index"])], device=selected_device)
        chosen = settings["style"].get("slider_ratio")
        settings["slider_bias"] = 0 if chosen == 0 or settings.get("slider_bias") == 0 else 0.5
        star_range = trained_star_range(metadata)
        profile = star_profile(metadata, settings["target_stars"])
        settings["condition_stars"] = (
            float(np.clip(settings["target_stars"], *star_range))
            if settings["target_stars"] is not None
            else None
        )
        if settings.get("style_from_stars", True):
            settings["profile_style"] = profile_style(profile, settings["style"])
        if profile and profile.get("tempo") and settings.get("tempo_prior") is None:
            # Typical tempo of training maps at this difficulty: resolves 99 vs 199 BPM.
            settings["tempo_prior"] = float(profile["tempo"])
        if not settings.get("bpm") and not settings.get("timing_points"):
            # Tempo is part of the condition; find the timing once with tempo unknown.
            context.report(0.03, "Tempo bestimmen")
            values, mask = style.vector({key: settings[key] for key in ("ar", "cs", "od", "hp")})
            probe = torch.from_numpy(style.condition_input(values, mask))[None].to(selected_device)
            prediction = predict_rhythm(
                model, mel, summary_features(mel), probe, selected_device, context, mapper
            )
            timing, source = timing_map(prediction, audio_path, settings)
            known_timing = {"timing_override": timing, "timing_source": source}
    else:
        settings["rhythm_mode"] = "onsets"
    target = settings["target_stars"]
    rounds = (
        max(1, min(5, int(settings["calibration_rounds"])))
        if settings["calibrate"] and target is not None and difficulty.available()
        else 1
    )
    history, best = [], None
    with artifact_directory("outputs") as (identifier, directory):
        audio_name = "audio.ogg"
        for attempt in range(rounds):
            if trace is not None and rounds > 1:
                trace.record("calibration_round", round=attempt + 1, settings=settings)
            run_settings = {**settings, **known_timing}
            if styled:
                if settings.get("bpm"):
                    tempo = settings["bpm"]
                elif known_timing:
                    tempo = dominant_tempo(known_timing["timing_override"], duration_ms)
                else:
                    manual = [
                        TimingPoint(point["time"], point["beat_length"])
                        for point in settings["timing_points"]
                    ]
                    tempo = dominant_tempo(manual, duration_ms)
                condition = torch.from_numpy(styled_condition(settings, tempo))[None].to(
                    selected_device
                )
                patterns = settings["rhythm_mode"] == "patterns"
                generator = generate_objects_v2 if patterns else generate_objects
                objects, timing, report = generator(
                    model,
                    mel,
                    audio_path,
                    run_settings,
                    condition,
                    selected_device,
                    context,
                    mapper=mapper,
                    trace=trace,
                )
            else:
                condition = torch.tensor(
                    [conditions(settings)], dtype=torch.float32, device=selected_device
                )
                objects, timing, report = generate_objects(
                    model,
                    mel,
                    audio_path,
                    run_settings,
                    condition,
                    selected_device,
                    context,
                    trace=trace,
                )
            if settings.get("combos", True):
                assign_combos(objects, timing, report.get("spinners") or [], settings)
            spinners = mark_spinner_combos(objects, report.get("spinners") or [])
            candidate = directory / "candidate.osu"
            verified_write(candidate, objects, settings, audio_name, timing, spinners=spinners)
            attributes = measured_stars(candidate)
            stars = attributes["star_rating"] if attributes else None
            history.append(
                {
                    "round": attempt + 1,
                    "stars": stars,
                    "aim": attributes.get("aim_difficulty") if attributes else None,
                    "speed": attributes.get("speed_difficulty") if attributes else None,
                    "condition_stars": settings.get("condition_stars"),
                    "tempo_bias": settings.get("tempo_bias"),
                    "density": None if styled else settings.get("density"),
                    "objects": len(objects),
                }
            )
            error = abs(stars - target) if stars is not None and target is not None else 0
            if best is None or error < best[0]:
                best = (error, objects, timing, report, attributes, spinners)
            if stars is None or target is None or error <= STAR_TOLERANCE or attempt + 1 == rounds:
                break
            context.report(0.9, f"Sterne {stars:.2f} statt {target:.2f} · neuer Durchlauf")
            ratio = target / max(stars, 0.5)
            if styled:
                settings = calibration_step(
                    settings,
                    target,
                    stars,
                    attributes,
                    profile,
                    star_range,
                    pattern_summary(objects, settings, timing),
                )
            else:
                settings["density"] = float(np.clip(settings["density"] * ratio**0.8, 0.5, 10))
                settings["spacing"] = float(np.clip(settings["spacing"] * ratio**0.5, 0, 1))
        _, objects, timing, report, attributes, spinners = best
        candidate.unlink(missing_ok=True)
        export_audio(audio_path, directory / audio_name)
        # Editor layers the model does not learn: kiai, preview point, breaks, hitsounds.
        finish = finish_map(objects, timing, spinners, mel, settings)
        bpm, offset = verified_write(
            directory / "map.osu", objects, settings, audio_name, timing, trace, spinners, finish
        )
        if known_timing:
            report["timing_source"] = known_timing["timing_source"]
        report["difficulty"] = attributes
        report["calibration"] = history
        report["patterns"] = pattern_summary(objects, settings, timing)
        report["finishing"] = finish.summary
        report["ranking_check"] = ranking_check(
            objects,
            spinners,
            timing,
            settings,
            stars=attributes["star_rating"] if attributes else target,
            audio_ms=duration_ms,
            finish=finish,
        )
        with zipfile.ZipFile(directory / "beatmap.osz", "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(directory / "map.osu", "map.osu")
            archive.write(directory / audio_name, audio_name)
        preview = {
            "objects": [asdict(obj) for obj in objects],
            "spinners": [[float(start), float(end)] for start, end in spinners],
            "kiai": [[float(start), float(end)] for start, end in finish.kiai],
            "breaks": [[float(start), float(end)] for start, end in finish.breaks],
            "duration": len(mel) * FRAME_MS / 1000,
            "bpm": bpm,
            "offset": offset,
            "timing": [asdict(point) for point in timing],
            "quality": report,
            "cs": settings["cs"],
            "ar": settings["ar"],
            "waveform": np.max(mel[:: max(1, len(mel) // 180)], axis=1).tolist(),
        }
        (directory / "preview.json").write_text(
            json.dumps(preview, allow_nan=False), encoding="utf-8"
        )
        if trace is not None:
            (directory / "trace.json").write_text(
                json.dumps(trace.document(), allow_nan=False), encoding="utf-8"
            )
        context.check()
        store.save(
            "output",
            {
                "id": identifier,
                "name": f"{settings['title']}.osz",
                "path": str(directory / "beatmap.osz"),
                "audio_path": str(directory / audio_name),
                "preview_path": str(directory / "preview.json"),
                "objects": len(objects),
                "bpm": round(bpm, 2),
                "stars": round(attributes["star_rating"], 2) if attributes else None,
                "aim": round(attributes["aim_difficulty"], 2)
                if attributes and "aim_difficulty" in attributes
                else None,
                "speed": round(attributes["speed_difficulty"], 2)
                if attributes and "speed_difficulty" in attributes
                else None,
                "target_stars": target,
                "quality": report,
                "settings": settings,
                "created_at": store.now(),
            },
        )
        return {"output_id": identifier}
