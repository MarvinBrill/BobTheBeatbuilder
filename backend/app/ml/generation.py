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


def styled_condition(settings, tempo):
    raw = {key: settings[key] for key in ("ar", "cs", "od", "hp")}
    raw["tempo"] = tempo
    if settings.get("condition_stars") is not None:
        raw["stars"] = settings["condition_stars"]
    if settings.get("aim_share") is not None:
        raw["aim_share"] = settings["aim_share"]
    for key, value in (settings.get("style") or {}).items():
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


def verified_write(path, objects, settings, audio_name, timing, trace=None):
    """Write the map and check that osu! will read back exactly the generated objects."""
    bpm, offset = 60000 / timing[0].beat_length, timing[0].time
    write(path, objects, settings, audio_name, bpm, offset, timing)
    verified = parse(path)
    if len(verified.objects) != len(objects):
        raise ValueError("Generated beatmap validation failed")
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
        settings["condition_stars"] = settings["target_stars"]
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
            candidate = directory / "candidate.osu"
            verified_write(candidate, objects, settings, audio_name, timing)
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
                best = (error, objects, timing, report, attributes)
            if stars is None or target is None or error <= STAR_TOLERANCE or attempt + 1 == rounds:
                break
            context.report(0.9, f"Sterne {stars:.2f} statt {target:.2f} · neuer Durchlauf")
            ratio = target / max(stars, 0.5)
            if styled:
                settings["condition_stars"] = float(
                    np.clip(settings["condition_stars"] + 0.9 * (target - stars), 0.5, 12)
                )
                settings["tempo_bias"] = float(
                    np.clip(settings["tempo_bias"] + 0.35 * (target - stars), -2, 2)
                )
            else:
                settings["density"] = float(np.clip(settings["density"] * ratio**0.8, 0.5, 10))
                settings["spacing"] = float(np.clip(settings["spacing"] * ratio**0.5, 0, 1))
        _, objects, timing, report, attributes = best
        candidate.unlink(missing_ok=True)
        export_audio(audio_path, directory / audio_name)
        bpm, offset = verified_write(
            directory / "map.osu", objects, settings, audio_name, timing, trace
        )
        if known_timing:
            report["timing_source"] = known_timing["timing_source"]
        report["difficulty"] = attributes
        report["calibration"] = history
        report["patterns"] = pattern_summary(objects, settings, timing)
        with zipfile.ZipFile(directory / "beatmap.osz", "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(directory / "map.osu", "map.osu")
            archive.write(directory / audio_name, audio_name)
        preview = {
            "objects": [asdict(obj) for obj in objects],
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
