import json
import zipfile
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from .. import store
from ..beatmaps import write, parse
from ..artifacts import artifact_directory
from ..config import DATA
from .audio import features, export_audio, FRAME_MS
from .network import conditions, device
from .phrase_model import create_model
from .sequence_generation import generate_objects
from .architecture import ARCHITECTURE, FORMAT_VERSION, EVENT_SCHEMA, TARGET_SCHEMA
from .generation_trace import GenerationTrace, object_snapshot


def generate(settings, context, trace=None):
    settings = {"seed": 42, "temperature": 0.8, "candidates": 12, **settings}
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
    condition = torch.tensor([conditions(settings)], dtype=torch.float32, device=selected_device)
    objects, timing, report = generate_objects(
        model, mel, audio_path, settings, condition, selected_device, context, trace=trace
    )
    with artifact_directory("outputs") as (identifier, directory):
        audio_name = "audio.ogg"
        export_audio(audio_path, directory / audio_name)
        bpm, offset = 60000 / timing[0].beat_length, timing[0].time
        write(directory / "map.osu", objects, settings, audio_name, bpm, offset, timing)
        verified = parse(directory / "map.osu")
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
                "quality": report,
                "settings": settings,
                "created_at": store.now(),
            },
        )
        return {"output_id": identifier}
