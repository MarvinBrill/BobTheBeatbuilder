import numpy as np
from scipy.signal import find_peaks
from .audio import FRAME_MS


def onset_matches(predicted, reference, tolerance):
    left, right, matched = 0, 0, 0
    while left < len(predicted) and right < len(reference):
        delta = predicted[left] - reference[right]
        if abs(delta) <= tolerance:
            matched += 1
            left += 1
            right += 1
        elif delta < 0:
            left += 1
        else:
            right += 1
    return matched


class ValidationMetrics:
    def __init__(self):
        self.tp = self.predicted = self.actual = 0
        self.motion_error = self.motion_count = self.type_correct = self.event_count = 0

    def add(self, output, batch):
        probabilities = 1 / (1 + np.exp(-np.clip(output["rhythm"][:, 0].detach().cpu().numpy(), -60, 60)))
        labels = batch["frame_targets"].detach().cpu().numpy()
        for scores, target in zip(probabilities, labels):
            # Use only the central window once, even when it contains several decoder chunks.
            weight = float(target[4].max())
            expected = np.flatnonzero((target[0] > 0.5) & (target[4] > 0))
            peaks, _ = find_peaks(scores, height=0.5, distance=3)
            peaks = peaks[target[4, peaks] > 0]
            self.predicted += len(peaks) * weight
            self.actual += len(expected) * weight
            self.tp += onset_matches(peaks, expected, 35 / FRAME_MS) * weight
        valid = batch["event_mask"].detach().cpu().numpy()
        geometry_valid = valid * batch["geometry_valid"].detach().cpu().numpy()
        movement = output["motion"].detach().cpu().numpy()
        expected_motion = batch["motion_target"].detach().cpu().numpy()
        self.motion_error += float((np.abs(movement - expected_motion).mean(-1) * geometry_valid).sum())
        self.motion_count += float(geometry_valid.sum())
        kind = output["kind"].detach().cpu().numpy().argmax(-1)
        self.type_correct += float(((kind == batch["kind_target"].detach().cpu().numpy()) * valid).sum())
        self.event_count += float(valid.sum())

    def result(self):
        precision, recall = self.tp / max(1, self.predicted), self.tp / max(1, self.actual)
        return {
            "onset_precision": precision,
            "onset_recall": recall,
            "onset_f1": 2 * precision * recall / max(1e-9, precision + recall),
            "motion_mae": self.motion_error / max(1, self.motion_count),
            "object_type_accuracy": self.type_correct / max(1, self.event_count),
        }
