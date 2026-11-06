import numpy as np
import torch
from .audio import FRAME_MS, tempo
from .tempo_octave import model_tempo
from .architecture import CENTRAL_FRAMES, AUDIO_MARGIN
from .sequence_batches import audio_window
from ..timing import TimingPoint, timing_at


def encode_window(model, mel, summary, condition, central, selected_device, mapper=None):
    values = audio_window(mel, summary, central)
    batch = {
        key: torch.from_numpy(value.copy()).unsqueeze(0).to(selected_device)
        for key, value in values.items()
    }
    with torch.no_grad():
        return model.encode(
            batch["audio"],
            batch["summary"],
            condition,
            batch["audio_times"],
            batch["global_times"],
            batch["audio_valid"],
            mapper,
        )


def predict_rhythm(model, mel, summary, condition, selected_device, context, mapper=None):
    output = np.empty((len(mel), 4), np.float32)
    for central in range(0, len(mel), CENTRAL_FRAMES):
        context.report(0.05 + 0.3 * central / len(mel), "Musikalischen Verlauf analysieren")
        _, _, rhythm, _ = encode_window(
            model, mel, summary, condition, central, selected_device, mapper
        )
        count = min(CENTRAL_FRAMES, len(mel) - central)
        output[central : central + count] = (
            rhythm[0, :, AUDIO_MARGIN : AUDIO_MARGIN + count].cpu().numpy().T
        )
    if not np.isfinite(output).all():
        raise ValueError("Model generated non-finite rhythm predictions")
    output[:, :2] = 1 / (1 + np.exp(-np.clip(output[:, :2], -60, 60)))
    return output


def detect(prediction, mel, settings):
    """Detected red lines for a song (see `tempo_grid.detect_timing`), or None."""
    from .tempo_grid import DEFAULT_CONVENTION_MS, detect_timing

    if mel is None:
        return None
    convention = settings.get("timing_convention_ms")
    try:
        return detect_timing(
            mel,
            FRAME_MS,
            prediction,
            prior_bpm=settings.get("tempo_prior"),
            convention_ms=DEFAULT_CONVENTION_MS if convention is None else float(convention),
        )
    except ValueError:
        return None


def timing_map(prediction, audio_path, settings, mel=None):
    """Red lines for generation: known timing first, then detection from the audio.

    Returns (points, source). Detection details are available via `detect`.
    """
    if settings.get("timing_override"):
        # Timing that is already known: the first calibration round's detection, the
        # user-approved song plan, or the reference map's red lines in free validation.
        points = list(settings["timing_override"])
        if not points or not all(isinstance(point, TimingPoint) for point in points):
            raise ValueError("Invalid timing override")
        return points, settings.get("timing_source", "reference")
    if settings.get("timing_points"):
        points = [
            TimingPoint(
                point["time"],
                point["beat_length"],
                meter=point.get("meter"),
                beat_confidence=1.0,
                meter_confidence=(
                    1.0 if point.get("meter") and point.get("bar_origin") is not None else 0.0
                ),
                bar_origin=point.get("bar_origin"),
            )
            for point in settings["timing_points"]
        ]
        if len(points) > 256 or any(
            right.time <= left.time for left, right in zip(points, points[1:])
        ):
            raise ValueError(
                "Manual timing points must be strictly ordered and contain at most 256 sections"
            )
        return points, "manual-sections"
    if settings["bpm"]:
        beat = 60000 / settings["bpm"]
        offset = settings["offset"] - np.ceil(max(0, settings["offset"]) / beat) * beat
        return [TimingPoint(float(offset), float(beat))], "manual"
    result = detect(prediction, mel, settings)
    if result is not None:
        return result.points, result.source
    # Octave prior: typical tempo at the target stars, else the model's own tempo head.
    prior = settings.get("tempo_prior") or model_tempo(prediction)
    bpm, offset = tempo(audio_path, prior)
    beat = 60000 / bpm
    return [
        TimingPoint(
            float(offset - np.ceil(max(0, offset) / beat) * beat), float(beat), beat_confidence=0.75
        )
    ], "audio-fallback"


def fit_beat_segment(times, estimate):
    # Frame-quantized intervals alone drift over long songs. Fit phase and slope to
    # all detected beats in a stable section, accounting for occasional missed beats.
    counts = np.concatenate([[0], np.cumsum(np.clip(np.rint(np.diff(times) / estimate), 1, 16))])
    design = np.column_stack([np.ones(len(counts)), counts])
    anchor, beat = np.linalg.lstsq(design, times, rcond=None)[0]
    residual = np.abs(times - (anchor + beat * counts))
    keep = residual <= max(35, float(np.median(residual)) * 3)
    if keep.sum() >= 4:
        anchor, beat = np.linalg.lstsq(design[keep], times[keep], rcond=None)[0]
    if not 150 <= beat <= 3000 or abs(beat - estimate) / estimate > 0.15:
        return float(times[0]), estimate
    return float(anchor), float(beat)


def select_times(prediction, mel, settings, timing, trace=None):
    from .structured_rhythm import onset_candidates, select_sequence

    candidates = onset_candidates(prediction, mel, settings, timing, trace)
    return select_sequence(candidates, timing, trace=trace)
