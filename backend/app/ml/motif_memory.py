"""Within-song motif hypotheses from ordered spectral evidence and relative motion.

These eight-beat analysis spans are not claimed to be detected musical phrases.
No coordinates or signatures are read from another song or from training maps.
"""

from dataclasses import dataclass
import numpy as np
from .audio import FRAME_MS
from ..timing import red_points


@dataclass(frozen=True)
class MotifLink:
    start: float
    end: float
    reference_start: float
    reference_end: float
    similarity: float


def spectral_signature(mel):
    if len(mel) < 16:
        return None
    edges = np.linspace(0, len(mel), 17).astype(int)
    spectrum = np.stack([mel[a:b].mean(0) for a, b in zip(edges[:-1], edges[1:])])
    # Remove broadband energy; an equal envelope with different notes is insufficient.
    spectrum = spectrum - spectrum.mean(1, keepdims=True)
    if spectrum.std() < 0.02 or np.diff(spectrum, axis=0).std() < 0.015:
        return None
    signature = np.concatenate([spectrum.reshape(-1), np.diff(spectrum, axis=0).reshape(-1)])
    return signature / np.linalg.norm(signature)


def link_spans(mel, spans):
    signatures, links = [], []
    for start, end in spans:
        first, last = max(0, round(start / FRAME_MS)), min(len(mel), round(end / FRAME_MS))
        signature = spectral_signature(mel[first:last])
        if signature is None:
            continue
        eligible = [
            (float(signature @ old), left, right)
            for left, right, old in signatures
            if right <= start and 0.85 <= (end - start) / (right - left) <= 1.15
        ]
        if eligible:
            similarity, left, right = max(eligible)
            if similarity >= 0.97:
                links.append(MotifLink(start, end, left, right, similarity))
        signatures.append((start, end, signature))
    return links


def analysis_spans(mel, timing):
    """Eight-beat spans, never eight-beat bars; uncertain pulse timing is excluded."""
    duration = (len(mel) - 1) * FRAME_MS
    points, spans = red_points(timing), []
    for index, point in enumerate(points):
        if point.beat_confidence < 0.6:
            continue
        end = points[index + 1].time if index + 1 < len(points) else duration
        start = point.time + np.ceil(max(0, -point.time) / point.beat_length) * point.beat_length
        width = 8 * point.beat_length
        while start + width <= end:
            spans.append((float(start), float(start + width)))
            start += width
    # Keep long-distance coverage with a strict analysis bound on long/fast songs.
    if len(spans) > 512:
        spans = [spans[index] for index in np.linspace(0, len(spans) - 1, 512).astype(int)]
    return spans


def relative_geometry(obj, anchor, direction=None):
    points = np.asarray(
        obj.curve if obj.slider and obj.curve else [[obj.x, obj.y]] * 4, dtype=float
    )
    points = points - np.asarray(anchor)
    if direction is not None:
        angle = np.arctan2(direction[1], direction[0])
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        points = points @ rotation
    return points


def transform_geometry(obj, source_anchor, target_anchor, angle=0, scale=1):
    """A single rigid/scale transform for head and every slider control, with no clipping."""
    points = relative_geometry(obj, source_anchor)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = points @ rotation.T * scale + np.asarray(target_anchor)
    tail = transformed[-1] if obj.slider and obj.repeats % 2 else transformed[0]
    return transformed, tail


class MotifMemory:
    def __init__(self, links):
        self.links = links

    def compatible_reference(self, time, history):
        for link in self.links:
            if not link.start <= time < link.end:
                continue
            reference = [
                obj for obj in history if link.reference_start <= obj.time < link.reference_end
            ]
            current = [obj for obj in history if link.start <= obj.time < link.end]
            if len(reference) < 3 or not current:
                continue
            scale_time = (link.reference_end - link.reference_start) / (link.end - link.start)
            if any(
                abs(
                    (obj.time - link.start) * scale_time
                    + link.reference_start
                    - reference[index].time
                )
                > 35
                for index, obj in enumerate(current[:2])
            ):
                continue
            relative_time = (time - link.start) / (link.end - link.start)
            expected_time = link.reference_start + relative_time * (
                link.reference_end - link.reference_start
            )
            expected = min(reference, key=lambda obj: abs(obj.time - expected_time))
            if abs(expected.time - expected_time) > 35:
                continue  # Different rhythm interpretation: do not force an unrelated geometry.
            return link, reference, current, expected
        return None

    def cost(self, obj, history):
        match = self.compatible_reference(obj.time, history)
        if match is None:
            return 0.0
        link, reference, current, expected = match
        if obj.slider != expected.slider or obj.repeats != expected.repeats:
            return 0.0  # A valid alternate structure is not automatically a bad motif.
        source, target = reference[0], current[0]
        source_direction = (
            [reference[1].x - source.end_x, reference[1].y - source.end_y]
            if len(reference) > 1
            else None
        )
        target_direction = (
            [current[1].x - target.end_x, current[1].y - target.end_y] if len(current) > 1 else None
        )
        # Rotation becomes identifiable only after two generated heads in this occurrence.
        if (
            target_direction is None
            or np.linalg.norm(target_direction) < 1
            or np.linalg.norm(source_direction) < 1
        ):
            return 0.0
        actual = relative_geometry(obj, [target.x, target.y], target_direction)
        desired = relative_geometry(expected, [source.x, source.y], source_direction)
        scale = np.clip(
            np.linalg.norm(target_direction) / np.linalg.norm(source_direction), 0.8, 1.2
        )
        desired *= scale
        distance = np.linalg.norm(actual - desired, axis=1).mean() / 512
        return float(min(1.0, distance) * link.similarity)

    def proposals(self, time, history):
        match = self.compatible_reference(time, history)
        if match is None:
            return []
        _, reference, current, expected = match
        if len(current) < 2:
            return []
        source, target = reference[0], current[0]
        before = np.array([reference[1].x - source.end_x, reference[1].y - source.end_y])
        after = np.array([current[1].x - target.end_x, current[1].y - target.end_y])
        if min(np.linalg.norm(before), np.linalg.norm(after)) < 1:
            return []
        angle = np.arctan2(after[1], after[0]) - np.arctan2(before[1], before[0])
        scale = float(np.clip(np.linalg.norm(after) / np.linalg.norm(before), 0.8, 1.2))
        points, _ = transform_geometry(
            expected, [source.x, source.y], [target.x, target.y], angle, scale
        )
        # Normal candidate guards validate bounds, duration, repeats and the actual end.
        return [(points / [512, 384]).reshape(-1)]
