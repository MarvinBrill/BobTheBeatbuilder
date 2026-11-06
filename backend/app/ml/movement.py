import numpy as np
from functools import lru_cache

SCALE = (512.0, 384.0)
# Turn classes: twelve 30-degree bins of the signed turn from the previous movement, plus
# one class for "no turn" (either move shorter than TURN_MIN_PIXELS: stack, stream, start).
TURN_BINS = 12
TURN_CLASSES = TURN_BINS + 1
TURN_MIN_PIXELS = 30.0


def turn_class(first, second, minimum=TURN_MIN_PIXELS):
    """Turn class of movement `second` after movement `first` (pixel vectors)."""
    first, second = np.asarray(first, np.float64), np.asarray(second, np.float64)
    if np.linalg.norm(first) <= minimum or np.linalg.norm(second) <= minimum:
        return TURN_BINS
    cross = first[0] * second[1] - first[1] * second[0]
    angle = np.degrees(np.arctan2(cross, first @ second))
    return int(np.clip((angle + 180) // (360 / TURN_BINS), 0, TURN_BINS - 1))


def turn_targets(motion_target, motion_context):
    """Turn classes from prepared motion targets (torch): cos/sin are channels 2/3, the
    distance channel 0 (/512) and the previous distance channel 7 of the context (/512)."""
    import torch

    # Float class indices (like the gap targets): every backend, DirectML included.
    distance = motion_target[..., 0] * 512
    previous = motion_context[..., 7] * 512
    angle = torch.atan2(motion_target[..., 3], motion_target[..., 2]) * (180 / 3.141592653589793)
    bins = torch.clamp(torch.floor((angle + 180) / (360 / TURN_BINS)), 0, TURN_BINS - 1)
    none = (distance <= TURN_MIN_PIXELS) | (previous <= TURN_MIN_PIXELS)
    return torch.where(none, torch.full_like(bins, float(TURN_BINS)), bins)


def bezier_points(geometry, steps=17):
    points = np.asarray(geometry).reshape(4, 2)
    t = np.linspace(0, 1, steps)[:, None]
    return (
        (1 - t) ** 3 * points[0]
        + 3 * (1 - t) ** 2 * t * points[1]
        + 3 * (1 - t) * t * t * points[2]
        + t**3 * points[3]
    )


def motion_numpy(geometry, duration, repeats, slider, time, context):
    points = np.asarray(geometry).reshape(4, 2) * SCALE
    delta = points[0] - np.asarray(context[:2]) * SCALE
    previous = np.asarray(context[2:4]) * SCALE
    distance, previous_length = np.linalg.norm(delta), np.linalg.norm(previous)
    gap = max(0.025, time - context[5])
    speed = distance / gap
    denom = max(distance * previous_length, 1e-6)
    cosine = float(delta @ previous / denom)
    sine = float((previous[0] * delta[1] - previous[1] * delta[0]) / denom)
    path = bezier_points(geometry) * SCALE
    travel = np.linalg.norm(np.diff(path, axis=0), axis=1).sum() * repeats * slider
    return np.array(
        [
            distance / 512,
            np.log1p(speed) / 8,
            cosine,
            sine,
            (speed / 2000 - context[4]),
            travel / 512,
            np.log1p(travel / max(duration, 0.025)) / 8,
            np.log((distance + 8) / (context[7] * 512 + 8)) / 3,
        ],
        np.float32,
    )


@lru_cache(maxsize=16)
def motion_constants(device, dtype):
    import torch  # Lazy: batch worker processes import this module without PyTorch.

    # Reused constants avoid host-to-device copies in each loss and permit CUDA graph capture.
    return (
        torch.tensor(SCALE, dtype=dtype, device=device),
        torch.linspace(0, 1, 17, dtype=dtype, device=device).unsqueeze(-1),
    )


def motion_tensor(geometry, duration, repeats, slider, time, context):
    import torch

    scale, t = motion_constants(geometry.device, geometry.dtype)
    points = geometry.reshape(*geometry.shape[:-1], 4, 2)
    delta = (points[..., 0, :] - context[..., :2]) * scale
    previous = context[..., 2:4] * scale
    distance = torch.sqrt(delta.square().sum(-1) + 1e-6)
    previous_length = torch.sqrt(previous.square().sum(-1) + 1e-6)
    speed = distance / (time - context[..., 5]).clamp(min=0.025)
    denom = (distance * previous_length).clamp(min=1e-6)
    cosine = (delta * previous).sum(-1) / denom
    sine = (previous[..., 0] * delta[..., 1] - previous[..., 1] * delta[..., 0]) / denom
    path = (
        (1 - t) ** 3 * points[..., 0:1, :]
        + 3 * (1 - t) ** 2 * t * points[..., 1:2, :]
        + 3 * (1 - t) * t * t * points[..., 2:3, :]
        + t**3 * points[..., 3:4, :]
    ) * scale
    travel = (
        torch.sqrt((path[..., 1:, :] - path[..., :-1, :]).square().sum(-1) + 1e-6).sum(-1) * repeats * slider
    )
    return torch.stack(
        [
            distance / 512,
            torch.log1p(speed) / 8,
            cosine,
            sine,
            speed / 2000 - context[..., 4],
            travel / 512,
            torch.log1p(travel / duration.clamp(min=0.025)) / 8,
            torch.log((distance + 8) / (context[..., 7] * 512 + 8)) / 3,
        ],
        dim=-1,
    )


def motion_batch_numpy(geometry, duration, repeats, slider, time, context):
    """Vectorized preparation of all event targets, with the scalar reference's precision."""
    points = np.asarray(geometry, dtype=np.float64).reshape(-1, 4, 2)
    delta = (points[:, 0] - context[:, :2]) * SCALE
    previous = context[:, 2:4] * SCALE
    distance = np.linalg.norm(delta, axis=-1)
    previous_length = np.linalg.norm(previous, axis=-1)
    speed = distance / np.maximum(0.025, time - context[:, 5])
    denom = np.maximum(distance * previous_length, 1e-6)
    cosine = (delta * previous).sum(-1) / denom
    sine = (previous[:, 0] * delta[:, 1] - previous[:, 1] * delta[:, 0]) / denom
    t = np.linspace(0, 1, 17)[None, :, None]
    path = (
        (1 - t) ** 3 * points[:, 0:1]
        + 3 * (1 - t) ** 2 * t * points[:, 1:2]
        + 3 * (1 - t) * t * t * points[:, 2:3]
        + t**3 * points[:, 3:4]
    ) * SCALE
    travel = np.linalg.norm(np.diff(path, axis=1), axis=-1).sum(-1) * repeats * slider
    return np.stack(
        [
            distance / 512,
            np.log1p(speed) / 8,
            cosine,
            sine,
            speed / 2000 - context[:, 4],
            travel / 512,
            np.log1p(travel / np.maximum(duration, 0.025)) / 8,
            np.log((distance + 8) / (context[:, 7] * 512 + 8)) / 3,
        ],
        axis=-1,
    ).astype(np.float32)


def history_context(objects):
    if not objects:
        return np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0], np.float32)
    last = objects[-1]
    delta, speed, distance = np.zeros(2), 0, 0
    if len(objects) > 1:
        older = objects[-2]
        delta = np.array([last.x - older.end_x, last.y - older.end_y])
        distance = np.linalg.norm(delta)
        speed = distance / max(0.025, (last.time - older.time - older.duration) / 1000)
    return np.array(
        [
            last.end_x / 512,
            last.end_y / 384,
            delta[0] / 512,
            delta[1] / 384,
            speed / 2000,
            (last.time + last.duration) / 1000,
            last.time / 1000,
            distance / 512,
        ],
        np.float32,
    )
