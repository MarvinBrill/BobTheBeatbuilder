"""Weak context-compatibility pairs, with matched structure and plausible geometry.

The preferences describe departure from a reference movement style. They are not
labels saying that another valid musical interpretation is wrong.
"""

import torch
from .movement import motion_tensor


def rolled(value, offset):
    offset %= value.shape[1]
    return torch.cat([value[:, -offset:], value[:, :-offset]], dim=1)


def counterfactuals(batch):
    geometry = batch["geometry_target"]
    valid = batch["event_mask"] * batch["geometry_valid"]
    # The positive and all alternatives must share the same bounds support.
    # Imported cubic controls can be outside the playfield while the visible path
    # remains valid; such positives must not be paired only with inside controls.
    valid = valid * ((geometry >= 0) & (geometry <= 1)).all(-1).to(valid.dtype)
    # Exclude BOS: without previous motion there is no connection to assess.
    valid = valid * (batch["previous"].abs().sum(-1) > 0).to(valid.dtype)
    best_geometry = geometry
    best_motion = batch["motion_target"]
    best_confidence = torch.zeros_like(valid)

    proposals = []
    for offset in (1, 5, 11):
        donor = rolled(geometry, offset)
        donor_valid = rolled(batch["event_valid"] * batch["geometry_valid"], offset)
        same_kind = batch["kind_target"] == rolled(batch["kind_target"], offset)
        same_repeat = batch["repeat_target"] == rolled(batch["repeat_target"], offset)
        duration_ratio = (batch["seconds_target"] + 0.05) / (
            rolled(batch["seconds_target"], offset) + 0.05
        )
        duration_match = (duration_ratio >= 0.85) & (duration_ratio <= 1.15)
        supported = ((donor >= 0) & (donor <= 1)).all(-1)
        # Donor curves come from valid real objects, not scrambled handles or padding.
        eligible = (
            valid
            * donor_valid
            * (same_kind & same_repeat & duration_match & supported).to(valid.dtype)
        )
        proposals.append((donor, eligible))

    # A second family changes a whole curve coherently. Reflections preserve its
    # length and shape, including repeated-slider parity; no individual handle
    # is scrambled. Reflecting an object is not intrinsically a bad mapping.
    points = geometry.reshape(*geometry.shape[:-1], 4, 2)
    for axes in ((-1, 1), (1, -1), (-1, -1)):
        x = 1 - points[..., 0] if axes[0] < 0 else points[..., 0]
        y = 1 - points[..., 1] if axes[1] < 0 else points[..., 1]
        transformed = torch.stack([x, y], dim=-1).reshape_as(geometry)
        supported = ((transformed >= 0) & (transformed <= 1)).all(-1)
        proposals.append((transformed, valid * supported.to(valid.dtype)))

    for donor, eligible in proposals:
        motion = motion_tensor(
            donor,
            batch["seconds_target"],
            batch["repeat_target"] + 1,
            batch["kind_target"],
            batch["event_times"],
            batch["motion_context"],
        )
        error = (motion - batch["motion_target"]).abs().clamp(max=2).mean(-1)
        # Near-equivalent continuations are never negative examples. Scores above
        # this tolerance express only weak compatibility evidence, capped at .75.
        confidence = ((error - 0.25) / 0.75).clamp(0, 1) * eligible
        take = confidence > best_confidence
        best_geometry = torch.where(take[..., None], donor, best_geometry)
        best_motion = torch.where(take[..., None], motion, best_motion)
        best_confidence = torch.maximum(best_confidence, confidence)

    return best_geometry, best_motion, best_confidence, 0.5 + 0.25 * best_confidence
