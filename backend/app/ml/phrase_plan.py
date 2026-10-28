"""Small, explicitly weak mapping proxies on the existing song-summary timeline."""

import numpy as np
from ..timing import beat_distance
from .architecture import GLOBAL_TOKENS
from .audio import FRAME_MS

PLAN_KEYS = (
    "activity",
    "rest_coverage",
    "combo_boundary_rate",
    "slider_fraction",
    "jump_scale",
    "turn_continuity",
    "burst_regularity",
)
PLAN_FEATURES = len(PLAN_KEYS)


def plan_targets(events, labels, timing):
    """Describe the reference interpretation; never infer instruments or true skill labels.

    Bins are analysis spans, not asserted musical phrases. Unsupported geometry and
    spinner regions do not become trustworthy movement or phrase-style targets.
    """
    values = np.zeros((GLOBAL_TOKENS, PLAN_FEATURES), np.float32)
    masks = np.zeros_like(values)
    edges = np.linspace(0, len(labels), GLOBAL_TOKENS + 1).astype(int)
    frames = np.rint(events[:, 0] * 1000 / FRAME_MS).astype(int)
    coverage = np.zeros(len(labels), np.float32)
    for row, frame in zip(events, frames):
        # Activity occupancy is a mapping proxy: a head covers 100 ms, a slider its duration.
        end = min(len(labels), frame + max(1, round(max(0.1, row[10]) * 1000 / FRAME_MS)))
        coverage[max(0, frame) : end] = 1
    gaps = np.zeros(len(events), np.float32)
    if timing:
        for index in range(1, len(events)):
            gaps[index] = beat_distance(
                timing, events[index - 1, 0] * 1000, events[index, 0] * 1000
            )
    regular = np.zeros(len(events), np.float32)
    if len(events) > 2:
        regular[2:] = (gaps[2:] <= 0.55) & (gaps[2:] >= 0.08) & (abs(gaps[2:] - gaps[1:-1]) < 0.06)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if right <= left or labels[left:right, 4].mean() < 0.95:
            continue
        selected = (frames >= left) & (frames < right)
        rows = events[selected]
        seconds = (right - left) * FRAME_MS / 1000
        values[index, 0] = np.clip(np.log1p(len(rows) / seconds) / np.log(11), 0, 1)
        values[index, 1] = 1 - coverage[left:right].mean()
        masks[index, :2] = 1
        if not len(rows):
            continue
        values[index, 2:4] = [rows[:, 13].mean(), rows[:, 9].mean()]
        masks[index, 2:4] = 1
        supported = rows[rows[:, 14] > 0]
        if len(supported):
            values[index, 4:6] = [
                np.clip(np.median(supported[:, 39]), 0, 1),
                np.clip((supported[:, 41].mean() + 1) / 2, 0, 1),
            ]
            masks[index, 4:6] = 1
        if timing:
            values[index, 6] = regular[selected].mean()
            masks[index, 6] = 1
    return values, masks
