"""Decode whole learned curves without independently jittering their control points."""

import numpy as np
from .movement import SCALE, bezier_points


def geometry_candidates(means, scales, weights, count, temperature, rng, slider):
    # Always consider unperturbed modes. Sampling alone can miss a well-learned curve.
    order = np.argsort(-weights)
    modes = min(len(order), max(1, count // 2))
    for index in range(count):
        if index < modes:
            yield means[order[index]].copy(), True
            continue
        component = rng.choice(len(weights), p=weights)
        if not slider:
            yield means[component] + rng.normal(size=8) * scales[component] * temperature, False
            continue
        curve = means[component].reshape(4, 2) * SCALE
        extent = max(32.0, np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())
        uncertainty = float(np.mean(scales[component].reshape(4, 2) * SCALE))
        # A shared translation/rotation/scale preserves the learned curve's topology.
        # Uncertainty cannot turn a short slider into a playfield-sized scribble.
        variation = min(0.12, uncertainty / extent) * temperature
        angle = np.clip(rng.normal() * variation, -0.3, 0.3)
        size = np.exp(np.clip(rng.normal() * variation, -0.2, 0.2))
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        shift_scale = np.minimum(scales[component, :2] * SCALE, min(24, extent * 0.1))
        shift = np.clip(rng.normal(size=2), -2, 2) * shift_scale * temperature
        sampled = (curve - curve[0]) @ rotation.T * size + curve[0] + shift
        yield (sampled / SCALE).reshape(-1), False


def fit_curve_bounds(curve, margin):
    """Move/scale the entire control polygon; pointwise clipping changes the curve."""
    lower = np.array([margin, margin], dtype=float)
    upper = np.asarray(SCALE) - lower
    span = np.ptp(curve, axis=0)
    factor = min(1.0, float(np.min((upper - lower) / np.maximum(span, 1e-6))))
    fitted = (curve - curve[0]) * factor + curve[0]
    shift = np.minimum(np.maximum(np.zeros(2), lower - fitted.min(0)), upper - fitted.max(0))
    return fitted + shift


def curve_diagnostics(curve, radius):
    """Conservative generation guard, not an osu! validity or difficulty rule.

    Smooth arcs and S curves are allowed. Crossings, sharp internal turns and
    extreme detours are excluded from this generator's ordinary slider style.
    """
    curve = np.asarray(curve, dtype=float)
    path = bezier_points(curve, 65)
    steps = np.diff(path, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    travel = float(lengths.sum())
    chord = float(np.linalg.norm(curve[-1] - curve[0]))
    detour = travel / max(chord, 1)

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    # Segment intersection, ignoring neighbours and zero-length segments.
    offset = path[None, :-1] - path[:-1, None]
    divisor = cross(steps[:, None], steps[None, :])
    valid = np.abs(divisor) > 1e-8
    safe = np.where(valid, divisor, 1)
    first = cross(offset, steps[None, :]) / safe
    second = cross(offset, steps[:, None]) / safe
    separated = np.abs(np.arange(64)[:, None] - np.arange(64)[None, :]) > 1
    crossing = bool(
        np.any(valid & separated & (first >= 0) & (first <= 1) & (second >= 0) & (second <= 1))
    )

    cosine = (steps[1:] * steps[:-1]).sum(-1) / np.maximum(lengths[1:] * lengths[:-1], 1e-9)
    turns = np.arccos(np.clip(cosine, -1, 1))
    local_radius = (lengths[1:] + lengths[:-1]) / np.maximum(2 * turns, 1e-9)
    moving = (lengths[1:] > 0.05) & (lengths[:-1] > 0.05)
    hook = bool(np.any(moving & (turns > np.pi / 6) & (local_radius < max(3, radius * 0.15))))
    # At a cusp one sample can be almost stationary; do not mask its direction reversal.
    reversal = (lengths[1:] > 1e-7) & (lengths[:-1] > 1e-7) & (turns > np.pi * 0.6)
    hook = hook or bool(np.any(reversal))
    excessive_detour = travel >= 20 and detour > 3.5
    # Repeated Bezier controls are segment separators in .osu, unlike SVG cubics.
    duplicate_controls = bool(np.any(np.linalg.norm(np.diff(curve, axis=0), axis=1) < 0.5))
    return {
        "unsafe": crossing or hook or excessive_detour or duplicate_controls,
        "crossing": crossing,
        "hook": hook,
        "detour": detour,
    }
