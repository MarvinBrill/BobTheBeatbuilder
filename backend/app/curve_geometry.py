import numpy as np
from .slider_paths import flatten


def fitted_curve(x, y, text, length, *, path=None):
    if path is None:
        parts = text.split("|")
        points = np.array([[x, y]] + [[float(v) for v in part.split(":")] for part in parts[1:]])
        path = flatten(parts[0], points) if len(points) > 1 else points
    distances = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    if distances[-1] < length and len(path) > 1:
        direction = path[-1] - path[-2]
        norm = np.linalg.norm(direction)
        if norm > 0:
            path = np.vstack([path, path[-1] + direction / norm * (length - distances[-1])])
            distances = np.append(distances, length)
    fraction = np.linspace(0, 1, 33)
    samples = np.stack(
        [np.interp(fraction * length, distances, path[:, axis]) for axis in (0, 1)], axis=1
    )
    t = fraction[:, None]
    basis = np.concatenate([3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t], axis=1)
    controls = np.linalg.lstsq(
        basis, samples - (1 - t) ** 3 * samples[0] - t**3 * samples[-1], rcond=None
    )[0]
    fitted = (1 - t) ** 3 * samples[0] + basis @ controls + t**3 * samples[-1]
    error = float(np.linalg.norm(fitted - samples, axis=1).max())
    return np.vstack([samples[0], controls, samples[-1]]).tolist(), error


def curve_length(curve):
    path = flatten("B", np.asarray(curve, dtype=float))
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
