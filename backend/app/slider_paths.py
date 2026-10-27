"""Flatten osu! curves, then trim/extend to the declared pixel length."""

import math
import numpy as np


def bezier(points, depth=0):
    if len(points) <= 2:
        return points
    chord = points[-1] - points[0]
    length = np.linalg.norm(chord)
    if length > 0:
        distance = (
            np.abs(
                chord[0] * (points[:, 1] - points[0, 1]) - chord[1] * (points[:, 0] - points[0, 0])
            )
            / length
        )
        projections = (points - points[0]) @ chord / (length * length)
        flat = distance.max() < 0.1 and np.all(np.diff(projections) >= 0)
    else:
        flat = np.linalg.norm(points - points[0], axis=1).max() < 0.1
    if flat or depth >= 12:
        return np.array([points[0], points[-1]])
    left, right, level = [points[0]], [points[-1]], points
    while len(level) > 1:
        level = (level[:-1] + level[1:]) / 2
        left.append(level[0])
        right.append(level[-1])
    return np.concatenate(
        [bezier(np.array(left), depth + 1)[:-1], bezier(np.array(right[::-1]), depth + 1)]
    )


def arc(points):
    a, b, c = points
    matrix = 2 * np.array([b - a, c - a])
    if abs(np.linalg.det(matrix)) < 1e-6:
        return None
    centre = np.linalg.solve(matrix, np.array([b @ b - a @ a, c @ c - a @ a]))
    radius = np.linalg.norm(a - centre)
    angles = [math.atan2(*(point - centre)[::-1]) for point in points]
    sweep = (angles[2] - angles[0]) % math.tau
    if (angles[1] - angles[0]) % math.tau > sweep:
        sweep -= math.tau
    count = min(
        8192,
        max(2, math.ceil(abs(sweep) / max(0.001, 2 * math.acos(max(-1, 1 - 0.1 / radius)))) + 1),
    )
    angle = np.linspace(angles[0], angles[0] + sweep, count)
    return centre + radius * np.stack([np.cos(angle), np.sin(angle)], axis=1)


def flatten(kind, points):
    if kind == "L":
        return points
    if kind == "P" and len(points) == 3:
        circular = arc(points)
        if circular is not None:
            return circular
    if kind == "C":
        segments = []
        for index in range(len(points) - 1):
            p0, p1 = points[max(0, index - 1)], points[index]
            p2 = points[index + 1]
            p3 = points[index + 2] if index + 2 < len(points) else 2 * p2 - p1
            t = np.linspace(0, 1, 51)[:, None]
            segments.append(
                0.5
                * (
                    (2 * p1)
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t
                )
            )
        return np.concatenate(segments)
    if kind not in ("B", "P"):
        raise ValueError("Unsupported slider curve type")
    segments, start = [], 0
    for index in range(1, len(points)):
        if np.array_equal(points[index], points[index - 1]):
            segments.append(bezier(points[start:index]))
            start = index
    segments.append(bezier(points[start:]))
    return np.concatenate(segments)


def slider_path(x, y, curve):
    parts = curve.split("|")
    if len(parts) > 257:
        raise ValueError("Slider has too many control points")
    points = np.array(
        [[x, y]] + [[float(value) for value in part.split(":")] for part in parts[1:]], dtype=float
    )
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not np.isfinite(points).all()
        or np.abs(points).max() > 100000
    ):
        raise ValueError("Invalid slider control points")
    return flatten(parts[0], points) if len(points) > 1 else points


def endpoint(x, y, curve, pixels, *, path=None):
    path = slider_path(x, y, curve) if path is None else path
    remaining = pixels
    direction = None
    for start, end in zip(path[:-1], path[1:]):
        delta = end - start
        length = np.linalg.norm(delta)
        if length < 1e-9:
            continue
        direction = delta / length
        if remaining <= length:
            return tuple(start + direction * remaining)
        remaining -= length
    return tuple(path[-1] + direction * remaining) if direction is not None else tuple(path[-1])
