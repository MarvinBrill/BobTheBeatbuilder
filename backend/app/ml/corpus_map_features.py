"""Map-only inventory. Audio bounds and actual acoustic labels remain unknown."""

from pathlib import Path
import numpy as np
from ..beatmaps import parse


def inspect_map(path):
    beatmap = parse(Path(path))
    objects = beatmap.objects
    sliders = [obj for obj in objects if obj.slider]
    heads = np.array([[obj.x, obj.y] for obj in objects])
    ends = np.array([[obj.end_x, obj.end_y] for obj in objects])
    gaps = np.diff([obj.time for obj in objects]) / 1000
    jumps = np.linalg.norm(heads[1:] - ends[:-1], axis=1) if len(objects) > 1 else np.array([])
    unsupported = sum(
        not obj.geometry_valid or obj.duration / obj.beat_length > 16 for obj in sliders
    )
    meters, section = [], ""
    for raw in Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("["):
            section = line
        elif section == "[TimingPoints]" and line and not line.startswith("//"):
            parts = line.split(",")
            if len(parts) >= 3 and float(parts[1]) > 0:
                meters.append(int(parts[2]))
    return {
        "objects": len(objects),
        "sliders": len(sliders),
        "unsupported_geometry_or_duration_sliders": unsupported,
        "skipped_objects": beatmap.skipped,
        "spinner_intervals": len(beatmap.ignored),
        "red_timing_points": sum(not point.inherited for point in beatmap.timing),
        "timing_meters": sorted(set(meters)),
        "slider_fraction": len(sliders) / max(1, len(objects)),
        "short_interval_fraction": float(np.mean(gaps <= 0.18)) if len(gaps) else 0,
        "long_gap_fraction": float(np.mean(gaps >= 1)) if len(gaps) else 0,
        "jump_p95_pixels": float(np.percentile(jumps, 95)) if len(jumps) else 0,
        "note": "Map-only proxies. Audio bounds and true style/layer labels not inferred; unsupported count is not a full training mask audit.",
    }
