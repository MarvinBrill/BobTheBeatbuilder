"""Song analysis as its own step: timing and structure for review before generating.

The studio runs this first. The user can check the BPM and offset against a metronome,
move section boundaries, and set a style per section; generation then takes the approved
plan as it is (`GenerateRequest.song_plan`).
"""

from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from .. import store
from .audio import FRAME_MS, features
from .network import device
from .song_plan import SongPlan, _plain


def analyze(settings, context):
    from .generation import load_model, mapper_index, probe_prediction, song_plan_for, star_profile

    audio = store.get("audio", settings["audio_id"])
    audio_path = Path(audio["path"])
    context.report(0.05, "Audio laden")
    mel = features(audio_path)
    prediction, metadata = None, {}
    analysis_settings = {
        "ar": settings.get("ar", 9.0),
        "cs": settings.get("cs", 4.0),
        "od": settings.get("od", 8.0),
        "hp": settings.get("hp", 5.0),
        "bpm": settings.get("bpm"),
        "offset": settings.get("offset", 0.0),
        "timing_points": settings.get("timing_points") or [],
    }
    if settings.get("model_id"):
        metadata = store.get("model", settings["model_id"])
        selected_device = device(settings.get("device", "cpu"))
        model = load_model(metadata, selected_device)
        mapper = None
        if model.styled:
            index = mapper_index(metadata, settings.get("mapper"))
            mapper = torch.tensor([float(index)], device=selected_device)
        profile = star_profile(metadata, settings.get("target_stars"))
        if profile and profile.get("tempo"):
            analysis_settings["tempo_prior"] = float(profile["tempo"])
        if metadata.get("timing_convention_ms") is not None:
            analysis_settings["timing_convention_ms"] = metadata["timing_convention_ms"]
        if model.styled:  # legacy models take another condition; their beats are not used
            context.report(0.2, "Beats und Tempo erkennen")
            prediction = probe_prediction(
                model, mel, analysis_settings, selected_device, context, mapper
            )
    context.report(0.8, "Songstruktur bestimmen")
    plan, source, info = song_plan_for(analysis_settings, mel, prediction, audio_path)
    result = plan.to_dict()
    result["timing_source"] = source
    result["timing_info"] = _plain(info)
    result["waveform"] = np.max(mel[:: max(1, len(mel) // 600)], axis=1).round(3).tolist()
    identifier = uuid4().hex
    context.check()
    store.save(
        "analysis",
        {
            "id": identifier,
            "audio_id": settings["audio_id"],
            "model_id": settings.get("model_id"),
            "plan": result,
            "created_at": store.now(),
        },
    )
    return {"analysis_id": identifier}


def validated_plan(plan):
    """Round-trip a plan from the studio through SongPlan (normalises and checks it)."""
    parsed = SongPlan.from_dict(plan)
    if not parsed.sections:
        raise ValueError("Der Song-Plan enthält keine Abschnitte.")
    for left, right in zip(parsed.sections, parsed.sections[1:]):
        if right.start_ms < left.start_ms:
            raise ValueError("Abschnitte müssen zeitlich geordnet sein.")
    for left, right in zip(parsed.timing, parsed.timing[1:]):
        if right.time <= left.time:
            raise ValueError("Red Lines müssen zeitlich geordnet sein.")
    return parsed.to_dict()


__all__ = ["analyze", "validated_plan", "FRAME_MS"]
