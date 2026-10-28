"""Fixed evaluation coverage, without pretending energy identifies a chorus."""

import numpy as np
from .audio import FRAME_MS
from .architecture import CENTRAL_FRAMES

SECTION_SECONDS = 24
MAX_SECTION_OBJECTS = 128
MAX_FULL_SECONDS = 300
MAX_FULL_OBJECTS = 2048


def section_plan(mel):
    duration = max(0.0, (len(mel) - 1) * FRAME_MS)
    width = min(duration, SECTION_SECONDS * 1000)
    starts = [
        ("opening", 0.0),
        ("middle", max(0, (duration - width) / 2)),
        ("ending", max(0, duration - width)),
    ]
    if duration > width:
        # Fixed audio-derived probes are annotations of activity, not musical labels.
        block = max(1, round(4000 / FRAME_MS))
        energy = np.max(mel, axis=1)
        blocks = np.array([energy[i : i + block].mean() for i in range(0, len(mel), block)])
        for label, index in (("low_activity", blocks.argmin()), ("high_activity", blocks.argmax())):
            center = (float(index) + 0.5) * block * FRAME_MS
            starts.append((label, float(np.clip(center - width / 2, 0, duration - width))))
        boundary = round(duration / (2 * CENTRAL_FRAMES * FRAME_MS)) * CENTRAL_FRAMES * FRAME_MS
        starts.append(
            ("window_transition", float(np.clip(boundary - width / 2, 0, duration - width)))
        )
    sections = []
    for label, start in starts:
        duplicate = next((s for s in sections if abs(s["start_ms"] - start) < FRAME_MS), None)
        if duplicate is not None:
            duplicate["probes"].append(label)
        else:
            sections.append(dict(probes=[label], start_ms=start, end_ms=start + width))
    return sections
