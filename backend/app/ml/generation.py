import json
import logging
import math
import tempfile
import zipfile
from uuid import uuid4
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from .. import difficulty, store
from ..beatmaps import HitObject, write, parse, unify_slider_velocities
from ..artifacts import artifact_directory
from ..config import DATA
from ..timing import red_points
from .audio import features, export_audio, FRAME_MS
from .network import conditions, device
from .phrase_model import create_model
from .rhythm_inference import detect, predict_rhythm, timing_map
from .sequence_data import summary_features
from .sequence_generation import generate_objects, generate_objects_v2
from .architecture import ARCHITECTURE, FORMAT_VERSION, EVENT_SCHEMA, TARGET_SCHEMA
from .generation_trace import GenerationTrace, object_snapshot
from .combos import assign_combos
from .finishing import finish_map
from .map_audit import audit
from .song_plan import SongPlan, build_plan, _plain as plain_json
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

    Levers: the stars condition (kept within +-0.75 stars of the target and the trained
    range), the jump spacing for aim, and `density_scale` (the rhythm guide's target
    density, section by section) for speed. `tempo_bias` used to be the speed lever: it
    shifted every gap of the song towards 1/4 or 1/1 at once and is now left to the user.
    """
    updated = dict(settings)
    error = target - stars
    low, high = star_range
    lower, upper = max(low, target - 0.75), min(high, target + 0.75)
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
    current = calibrated.get("jump_spacing", (profile or {}).get("jump_spacing"))
    collapsed = bool(lever and measured_jump is not None and measured_jump < 0.75 * current)
    density = float(settings.get("density_scale", 1.0))
    if error > 0 and lever and (share < typical_share - 0.05 or collapsed):
        # Aim is missing: wider jumps, never denser rhythm.
        updated["calibration_style"] = {
            **calibrated,  # the stored setting can be an explicit None
            "jump_spacing": float(np.clip(current * (1 + 0.15 * min(error, 2)), 40, 400)),
        }
    elif error < 0 and lever and share > typical_share + 0.05:
        # Too hard because of the jumps: narrower jumps first.
        updated["calibration_style"] = {
            **calibrated,
            "jump_spacing": float(np.clip(current * (1 + 0.1 * max(error, -2)), 40, 400)),
        }
    else:
        updated["density_scale"] = float(np.clip(density * 2 ** (0.25 * error), 0.6, 1.6))
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


DEFAULTS = {
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
    "density_scale": 1.0,
    "export_offset_ms": 0.0,
}
# Values that are runtime objects and never stored with an output.
TRANSIENT = ("song_plan_object", "timing_override")


def check_model(metadata):
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


def load_model(metadata, selected_device):
    check_model(metadata)
    model = create_model(metadata["architecture"], metadata.get("model_config"))
    model.load_state_dict(load_file(str(DATA / "models" / metadata["id"] / "model.safetensors")))
    return model.to(selected_device).eval()


def probe_prediction(model, mel, settings, selected_device, context, mapper):
    """The rhythm prediction with only the hit-window settings known (tempo unknown)."""
    values, mask = style.vector({key: settings[key] for key in ("ar", "cs", "od", "hp")})
    probe = torch.from_numpy(style.condition_input(values, mask))[None].to(selected_device)
    return predict_rhythm(model, mel, summary_features(mel), probe, selected_device, context, mapper)


def song_plan_for(settings, mel, prediction, audio_path):
    """(SongPlan, timing source, detection info) for a song.

    Order: an approved plan from the studio, manual timing (BPM/offset or sections), then
    detection from the audio and the model's beat activation.
    """
    if settings.get("song_plan"):
        plan = SongPlan.from_dict(settings["song_plan"])
        if plan.timing:
            return plan, "approved-plan", plan.timing_info
        # Sections without timing: detect the timing, keep the user's sections.
        result = detect(prediction, mel, settings)
        if result is None:
            raise ValueError("Kein stabiler Beat gefunden. Bitte BPM und Offset angeben.")
        plan.timing, plan.timing_source, plan.timing_info = result.points, result.source, result.info
        return plan, result.source, result.info
    if settings.get("bpm") or settings.get("timing_points"):
        timing, source = timing_map(prediction, audio_path, settings, mel)
        return build_plan(mel, FRAME_MS, timing, source), source, {}
    result = detect(prediction, mel, settings)
    if result is not None:
        plan = build_plan(mel, FRAME_MS, result.points, result.source, result.info)
        return plan, result.source, result.info
    timing, source = timing_map(prediction, audio_path, settings, mel)
    return build_plan(mel, FRAME_MS, timing, source), source, {}


def audit_reference(metadata, stars):
    """Typical audit values of the training maps at these stars (when measured)."""
    profiles = metadata.get("audit_profiles") or {}
    if stars is None or not profiles:
        return None
    bucket = min(profiles, key=lambda key: abs(float(key) - float(stars)))
    return profiles[bucket]


def storable(settings):
    return {key: value for key, value in settings.items() if key not in TRANSIENT}


def publish_output(
    objects, timing, spinners, report, attributes, settings, mel, audio_path, metadata,
    history=None, trace=None, parent=None,
):  # fmt: skip
    """Finish, export, measure and store one generated map; returns the output id."""
    duration_ms = len(mel) * FRAME_MS
    plan = report.pop("song_plan", None)
    with artifact_directory("outputs") as (identifier, directory):
        audio_name = "audio.ogg"
        unify_slider_velocities(objects)
        export_audio(audio_path, directory / audio_name, settings.get("export_offset_ms", 0.0))
        # Editor layers the model does not learn: kiai, preview point, breaks, hitsounds.
        finish = finish_map(objects, timing, spinners, mel, settings)
        bpm, offset = verified_write(
            directory / "map.osu", objects, settings, audio_name, timing, trace, spinners, finish
        )
        if attributes is None:
            attributes = measured_stars(directory / "map.osu")
        report["difficulty"] = attributes
        if history is not None:
            report["calibration"] = history
        report["patterns"] = pattern_summary(objects, settings, timing)
        report["finishing"] = finish.summary
        report["audit"] = plain_json(
            audit(
                objects, timing, mel, FRAME_MS, settings["cs"],
                audit_reference(
                    metadata, attributes["star_rating"] if attributes else settings.get("target_stars")
                ),
            )
        )  # fmt: skip
        if plan is not None:
            report["song_plan"] = plan.to_dict() if isinstance(plan, SongPlan) else plan
        report["ranking_check"] = ranking_check(
            objects,
            spinners,
            timing,
            settings,
            stars=attributes["star_rating"] if attributes else settings.get("target_stars"),
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
        stored = storable(settings)
        if plan is not None:
            stored["song_plan"] = report["song_plan"]
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
                "target_stars": settings.get("target_stars"),
                "quality": report,
                "settings": stored,
                "parent_id": parent,
                "created_at": store.now(),
            },
        )
        return identifier


def generate(settings, context, trace=None):
    settings = {**DEFAULTS, **settings}
    if trace is None and settings.get("trace", False):
        trace = GenerationTrace()
    if trace is not None:
        trace.record("generation_settings", settings=storable(settings))
    metadata = store.get("model", settings["model_id"])
    check_model(metadata)
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
    if metadata.get("timing_convention_ms") is not None:
        settings.setdefault("timing_convention_ms", metadata["timing_convention_ms"])
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
        # Song analysis first: timing and sections, decided once for every round.
        context.report(0.03, "Tempo und Songstruktur bestimmen")
        prediction = probe_prediction(model, mel, settings, selected_device, context, mapper)
        plan, source, timing_info = song_plan_for(settings, mel, prediction, audio_path)
        known_timing = {"timing_override": plan.timing, "timing_source": source}
        settings["song_plan_object"] = plan
        if trace is not None:
            trace.record("song_plan", plan=plan.to_dict())
    else:
        settings["rhythm_mode"] = "onsets"
        timing_info = {}
    target = settings["target_stars"]
    rounds = (
        max(1, min(5, int(settings["calibration_rounds"])))
        if settings["calibrate"] and target is not None and difficulty.available()
        else 1
    )
    history, best = [], None
    (DATA / "tmp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="generate-", dir=DATA / "tmp") as folder:
        scratch = Path(folder)
        audio_name = "audio.ogg"
        for attempt in range(rounds):
            if trace is not None and rounds > 1:
                trace.record("calibration_round", round=attempt + 1, settings=storable(settings))
            run_settings = {**settings, **known_timing}
            if styled:
                tempo = dominant_tempo(known_timing["timing_override"], duration_ms)
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
            unify_slider_velocities(objects)
            candidate = scratch / "candidate.osu"
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
                    "density_scale": settings.get("density_scale"),
                    "density": None if styled else settings.get("density"),
                    "objects": len(objects),
                }
            )
            error = abs(stars - target) if stars is not None and target is not None else 0
            if best is None or error < best[0]:
                best = (error, objects, timing, report, attributes, spinners, dict(settings))
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
        _, objects, timing, report, attributes, spinners, chosen_settings = best
    if known_timing:
        report["timing_source"] = known_timing["timing_source"]
    if timing_info:
        report["timing_detection"] = plain_json(timing_info)
    context.check()
    identifier = publish_output(
        objects, timing, spinners, report, attributes, chosen_settings, mel, audio_path,
        metadata, history=history, trace=trace,
    )  # fmt: skip
    return {"output_id": identifier}


def regenerate_section(request, context):
    """Generate one span of an existing output again and store the result as a new version.

    Objects before the span are the decoder history, objects after it stay as they are.
    """
    output = store.get("output", request["output_id"])
    settings = {**DEFAULTS, **output["settings"], "seed": int(request.get("seed", 1))}
    metadata = store.get("model", settings["model_id"])
    selected_device = device(request.get("device") or settings["device"])
    model = load_model(metadata, selected_device)
    if not model.styled:
        raise ValueError("Abschnitte neu generieren braucht ein Modell mit Stil-Steuerung.")
    audio_path = Path(store.get("audio", settings["audio_id"])["path"])
    context.report(0.05, "Audio laden")
    mel = features(audio_path)
    preview = json.loads(Path(output["preview_path"]).read_text(encoding="utf-8"))
    fields = HitObject.__dataclass_fields__
    objects = [HitObject(**{k: v for k, v in item.items() if k in fields}) for item in preview["objects"]]
    start, end = float(request["start_ms"]), float(request["end_ms"])
    if not 0 <= start < end:
        raise ValueError("Ungültiger Abschnitt")
    plan = SongPlan.from_dict(settings["song_plan"]) if settings.get("song_plan") else None
    if request.get("style") and plan is not None:
        for section in plan.sections:
            if section.start_ms < end and section.end_ms > start:
                section.style = request["style"]
        settings["song_plan"] = plan.to_dict()
    if plan is None or not plan.timing:
        raise ValueError("Diese Map hat keinen gespeicherten Song-Plan; bitte neu generieren.")
    before = [obj for obj in objects if obj.time + obj.duration < start - 1]
    after = [obj for obj in objects if obj.time >= end]
    spinners = [tuple(span) for span in preview.get("spinners") or [] if span[1] < start or span[0] >= end]
    hard_end = after[0].time if after else None
    for span in spinners:
        if span[0] >= start and (hard_end is None or span[0] < hard_end):
            hard_end = span[0]
    # A kept spinner just before the span: the new objects start after its recovery.
    previous_end = before[-1].time + before[-1].duration if before else 0.0
    for left, right in spinners:
        if previous_end <= right <= start:
            start = max(start, right + 300.0)
    if start >= end:
        raise ValueError("Der Abschnitt liegt vollständig in einem Spinner.")
    settings["song_plan_object"] = plan
    run_settings = {**settings, "timing_override": plan.timing, "timing_source": "approved-plan"}
    mapper = torch.tensor([float(settings.get("mapper_index", 0))], device=selected_device)
    tempo = dominant_tempo(plan.timing, len(mel) * FRAME_MS)
    condition = torch.from_numpy(styled_condition(settings, tempo))[None].to(selected_device)
    generated, timing, report = generate_objects_v2(
        model, mel, audio_path, run_settings, condition, selected_device, context,
        mapper=mapper, evaluation_range=(start, min(end, (len(mel) - 1) * FRAME_MS)),
        initial_objects=before, hard_end=hard_end, plan=plan,
    )  # fmt: skip
    merged = before + generated + after
    if settings.get("combos", True):
        assign_combos(merged, timing, spinners, settings)
    spinners = mark_spinner_combos(merged, spinners)
    full = quality_report_for(merged, settings, report)
    full["regenerated"] = {"start_ms": start, "end_ms": end, "objects": len(generated), "seed": settings["seed"]}
    full["song_plan"] = plan
    store.save(
        "feedback",
        {
            "id": uuid4().hex,
            "output_id": output["id"],
            "action": "regenerate",
            "start_ms": start,
            "end_ms": end,
            "objects_before": sum(start <= obj.time < end for obj in objects),
            "created_at": store.now(),
        },
    )
    context.check()
    identifier = publish_output(
        merged, timing, spinners, full, None, settings, mel, audio_path, metadata,
        parent=output["id"],
    )  # fmt: skip
    return {"output_id": identifier}


def quality_report_for(objects, settings, report):
    from .playability import quality_report

    result = quality_report(objects, settings)
    for key in ("timing_source", "timing_sections", "rhythm", "plan_guidance"):
        if key in report:
            result[key] = report[key]
    result["heavily_repaired_fraction"] = report.get("heavily_repaired_fraction", 0.0)
    return result
