"""Conservative evidence for retaining a held sound instead of one weak interior head."""

import numpy as np

from .audio import FRAME_MS


def onset_evidence(prediction, time):
    # Selected times may have snapped by up to 30 ms; preserve the original peak evidence.
    frame = int(round(time / FRAME_MS))
    radius = int(np.ceil(30 / FRAME_MS))
    values = prediction[max(0, frame - radius) : min(len(prediction), frame + radius + 1), 0]
    return float(np.max(values)) if len(values) else 0.0


def held_audio_evidence(mel, start, end):
    first, last = max(0, int(start / FRAME_MS)), min(len(mel), int(end / FRAME_MS) + 1)
    values = np.asarray(mel[first:last], dtype=np.float64)
    if len(values) < 2:
        return False, 0.0, 0.0
    audible = np.max(values, axis=1) > -0.85
    spectrum = values + 1
    norm = np.linalg.norm(spectrum, axis=1)
    similarity = (spectrum[1:] * spectrum[:-1]).sum(-1) / np.maximum(norm[1:] * norm[:-1], 1e-9)
    audible_fraction, continuity = float(audible.mean()), float(np.median(similarity))
    # This is only a sustained-texture proxy. An independent onset always takes precedence.
    return audible_fraction >= 0.9 and continuity >= 0.95, audible_fraction, continuity


def skip_option(times, index, beat, prediction, mel, trace=None):
    if index + 2 >= len(times) or times[index + 2] - times[index] > 2 * beat:
        return None
    head, inside, after = times[index : index + 3]
    score = onset_evidence(prediction, inside)
    boundaries = min(onset_evidence(prediction, head), onset_evidence(prediction, after))
    held, audible, continuity = held_audio_evidence(mel, head, after)
    eligible = score < 0.65 - 1e-6 and score + 0.15 < boundaries and held
    if trace is not None:
        trace.record(
            "joint_time_option",
            time=head,
            interior_time=inside,
            next_time=after,
            interior_score=score,
            boundary_score=boundaries,
            audible_fraction=audible,
            spectral_continuity=continuity,
            eligible=eligible,
            reason="weak_interior_held_audio" if eligible else "insufficient_skip_evidence",
        )
    if not eligible:
        return None
    alternative = times[: index + 1] + times[index + 2 :]
    # Removing onset evidence has a cost even when doing so is musically plausible.
    penalty = 0.5 * -np.log(max(1 - score, 1e-6))
    return alternative, inside, float(penalty)
