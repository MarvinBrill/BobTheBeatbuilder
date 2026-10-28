"""Bounded predicted-history auxiliary training; original supervised path stays intact."""

import torch
from .phrase_losses import mixture_parameters, categorical, binary_loss
from .movement import motion_constants

CONTEXT_WEIGHT = 0.1
MAX_SHIFT_PIXELS = 16.0


def predicted_history(batch, geometry, mixtures):
    previous = batch["previous"].detach().clone()
    weights, means, _ = mixture_parameters(geometry.detach().float(), mixtures)
    # Use the previous event's prediction, never the current/future target prediction.
    mode = weights.argmax(-1)
    predicted = (
        means * (torch.arange(mixtures, device=means.device) == mode[..., None])[..., None]
    ).sum(-2)
    delta = predicted[:, :-1, :2] - previous[:, 1:, :2]
    scale, _ = motion_constants(delta.device, delta.dtype)
    length = torch.sqrt((delta * scale).square().sum(-1, keepdim=True) + 1e-8)
    delta = delta * (MAX_SHIFT_PIXELS / length).clamp(max=1)
    points = previous[:, 1:, :10].reshape(*delta.shape[:-1], 5, 2)
    # A shared bounded translation cannot introduce an out-of-playfield cue.
    lower, upper = -points.min(-2).values, 1 - points.max(-2).values
    inside = ((points >= 0) & (points <= 1)).all(-1).all(-1)
    delta = torch.minimum(torch.maximum(delta, lower), upper)
    eligible = batch["event_valid"][:, :-1] * batch["event_valid"][:, 1:]
    selected = (
        (torch.rand_like(eligible) < batch["context_probability"]).to(delta.dtype)
        * eligible
        * inside.to(delta.dtype)
    )
    delta = delta * selected[..., None]
    # Rigidly translate start, both controls, tail and repeat-dependent end together.
    previous[:, 1:, :10] += delta.repeat(1, 1, 5)
    distance = (
        torch.sqrt(((previous[:, 1:, :2] - previous[:, :-1, 8:10]) * scale).square().sum(-1) + 1e-8)
        / 512
    )
    has_older = (previous[:, :-1].abs().sum(-1) > 0) & (eligible > 0)
    previous[:, 1:, 15] = torch.where(has_older, distance, previous[:, 1:, 15])
    changed = torch.cat([torch.zeros_like(selected[:, :1]), selected], dim=1)
    # An earlier changed key affects subsequent causal queries, not just the next one.
    affected = (changed.cumsum(1) > 0).to(delta.dtype) * batch["event_mask"]
    return previous, affected


def auxiliary_terms(model, output, batch):
    if "robust_geometry" not in output:
        return {}
    mask = output["robust_mask"]
    slider = mask * batch["kind_target"] * batch["geometry_valid"]

    def average(values, valid):
        return (values * valid).sum() / valid.sum().clamp(min=1)

    # An alternative history has no uniquely correct next absolute position.
    # Only unchanged structural targets supervise this auxiliary branch. Geometry,
    # motion and compatibility remain supervised on the coherent reference path.
    loss = 0.5 * average(categorical(output["robust_kind"], batch["kind_target"]), mask)
    loss += 0.3 * average(categorical(output["robust_duration"], batch["duration_target"]), slider)
    loss += 0.2 * average(categorical(output["robust_repeats"], batch["repeat_target"]), slider)
    loss += 0.1 * average(binary_loss(output["robust_combo"], batch["combo_target"]), mask)
    return {"context": (loss * CONTEXT_WEIGHT, loss.new_ones(()))}
