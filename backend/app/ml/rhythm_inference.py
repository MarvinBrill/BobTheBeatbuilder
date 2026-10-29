import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from .audio import FRAME_MS, tempo
from .architecture import CENTRAL_FRAMES, AUDIO_MARGIN
from .sequence_batches import audio_window


def encode_window(model, mel, summary, condition, central, selected_device, mapper=None):
    values = audio_window(mel, summary, central)
    batch = {
        key: torch.from_numpy(value.copy()).unsqueeze(0).to(selected_device) for key, value in values.items()
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
        _, _, rhythm = encode_window(
            model, mel, summary, condition, central, selected_device, mapper
        )
        count = min(CENTRAL_FRAMES, len(mel) - central)
        output[central : central + count] = rhythm[0, :, AUDIO_MARGIN : AUDIO_MARGIN + count].cpu().numpy().T
    if not np.isfinite(output).all():
        raise ValueError("Model generated non-finite rhythm predictions")
    output[:, :2] = 1 / (1 + np.exp(-np.clip(output[:, :2], -60, 60)))
    return output


def timing_map(prediction, audio_path, settings):
    if settings.get("timing"):
        # Red lines given explicitly, e.g. a reference map's timing in free validation.
        timing = sorted((float(offset), float(beat)) for offset, beat in settings["timing"])
        first, beat = timing[0]
        timing[0] = (first - np.ceil(max(0, first) / beat) * beat, beat)
        return timing, "reference"
    if settings["bpm"]:
        beat = 60000 / settings["bpm"]
        offset = settings["offset"] - np.ceil(max(0, settings["offset"]) / beat) * beat
        return [(float(offset), float(beat))], "manual"
    peaks, _ = find_peaks(
        prediction[:, 1], height=0.55, prominence=0.15, distance=max(1, round(150 / FRAME_MS))
    )
    times = peaks * FRAME_MS
    lengths = np.diff(times)
    expected = 500 * np.exp(np.clip(prediction[peaks[:-1], 3], -2, 2))
    good = (lengths >= 150) & (lengths <= 3000) & (np.abs(lengths - expected) / expected < 0.15)
    if good.sum() >= 8 and good.mean() >= 0.7:
        filtered = median_filter(lengths, size=5, mode="nearest")
        first_index = int(np.flatnonzero(good)[0])
        beat = float(filtered[first_index])
        boundaries = [first_index]
        for index in range(first_index + 1, len(filtered) - 3):
            if good[index : index + 3].all() and abs(filtered[index] - beat) / beat > 0.04:
                beat = float(filtered[index])
                boundaries.append(index)
        result = []
        for section, start in enumerate(boundaries):
            end = boundaries[section + 1] if section + 1 < len(boundaries) else len(times) - 1
            anchor, beat = fit_beat_segment(times[start : end + 1], float(filtered[start]))
            if not section:
                anchor -= np.ceil(max(0, anchor) / beat) * beat
            result.append((float(anchor), float(beat)))
        return result, "learned"
    bpm, offset = tempo(audio_path)
    beat = 60000 / bpm
    return [(float(offset - np.ceil(max(0, offset) / beat) * beat), float(beat))], "audio-fallback"


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


def timing_at(timing, time):
    return next(((offset, beat) for offset, beat in reversed(timing) if offset <= time), timing[0])


def select_times(prediction, mel, settings, timing):
    active = np.max(mel, axis=1) > -0.7
    profile = np.expm1(np.clip(prediction[:, 2], 0, 1.3) * np.log(11)) * active
    if profile.sum() > 0.01:
        profile *= settings["density"] / max(0.1, profile[active].mean())
    else:
        profile = active.astype(float) * settings["density"]
    peaks, _ = find_peaks(prediction[:, 0], distance=max(1, round(35 / FRAME_MS)), prominence=0.005)
    peaks = peaks[active[peaks]]
    window, chosen = round(4000 / FRAME_MS), []
    for start in range(0, len(mel), window):
        candidates = peaks[(peaks >= start) & (peaks < start + window)]
        budget = round(profile[start : start + window].sum() * FRAME_MS / 1000)
        chosen.extend(sorted(candidates, key=lambda index: prediction[index, 0], reverse=True)[:budget])
    result = []
    for frame in sorted(chosen):
        time = float(frame * FRAME_MS)
        if settings["snap"]:
            offset, beat = timing_at(timing, time)
            grid = beat / settings["divisor"]
            snapped = offset + round((time - offset) / grid) * grid
            if abs(snapped - time) <= min(30, grid * 0.3):
                time = float(snapped)
        index = max(0, min(len(active) - 1, round(time / FRAME_MS)))
        if (
            time < 0
            or time > (len(mel) - 1) * FRAME_MS - 100
            or not active[index]
            or (result and time - result[-1] < 35)
        ):
            continue
        result.append(time)
    return result
