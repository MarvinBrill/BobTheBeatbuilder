import math
import torch
from .architecture import GEOMETRY_SIZE
from .movement import motion_tensor


def binary_loss(logits, target, positive_weight=1):
    softplus = torch.maximum(-logits, torch.zeros_like(logits)) + torch.log1p(
        torch.exp(-logits.abs())
    )
    return (1 - target) * logits + (1 + (positive_weight - 1) * target) * softplus


def categorical(logits, target):
    classes = torch.arange(logits.shape[-1], dtype=logits.dtype, device=logits.device)
    one_hot = (target.unsqueeze(-1) == classes).to(logits.dtype)
    shifted = logits - logits.max(dim=-1, keepdim=True).values.detach()
    return torch.log(torch.exp(shifted).sum(-1)) - (shifted * one_hot).sum(-1)


def mixture_parameters(raw, mixtures):
    values = raw.reshape(*raw.shape[:-1], mixtures, 1 + 2 * GEOMETRY_SIZE)
    weights = torch.log_softmax(values[..., 0], dim=-1)
    means = torch.sigmoid(values[..., 1 : 1 + GEOMETRY_SIZE]) * 2 - 0.5
    log_scales = values[..., 1 + GEOMETRY_SIZE :].clamp(-4.5, -0.5)
    return weights, means, log_scales


def mixture_loss(raw, target, dimensions, mixtures):
    log_weights, means, log_scales = mixture_parameters(raw, mixtures)
    error = (target.unsqueeze(-2) - means) * torch.exp(-log_scales)
    log_prob = -0.5 * error.square() - log_scales - 0.5 * math.log(2 * math.pi)
    joint = log_weights + (log_prob * dimensions.unsqueeze(-2)).sum(-1)
    maximum = joint.max(-1, keepdim=True).values.detach()
    log_total = maximum.squeeze(-1) + torch.log(torch.exp(joint - maximum).sum(-1))
    responsibilities = torch.softmax(joint, -1).detach()
    return -log_total / dimensions.sum(-1).clamp(min=1), responsibilities, means


def smooth_error(delta):
    absolute = delta.abs()
    return torch.where(absolute < 1, 0.5 * delta.square(), absolute - 0.5)


def parts(model, output, batch, onset_weight=20):
    frames = batch["frame_targets"]
    hard = frames[:, 0]
    soft = hard.clone()
    soft[:, 1:] = torch.maximum(soft[:, 1:], hard[:, :-1] * 0.5)
    soft[:, :-1] = torch.maximum(soft[:, :-1], hard[:, 1:] * 0.5)
    rhythm = output["rhythm"]
    mask, slider = (
        batch["event_mask"],
        batch["event_mask"] * batch["kind_target"] * batch["geometry_valid"],
    )
    dimensions = torch.ones_like(batch["geometry_target"])
    dimensions[..., 2:] *= (batch["kind_target"] * batch["geometry_valid"]).unsqueeze(-1)
    nll, responsibilities, means = mixture_loss(
        output["geometry"], batch["geometry_target"], dimensions, model.config.mixtures
    )
    motion = motion_tensor(
        means,
        batch["seconds_target"].unsqueeze(-1),
        batch["repeat_target"].unsqueeze(-1) + 1,
        batch["kind_target"].unsqueeze(-1),
        batch["event_times"].unsqueeze(-1),
        batch["motion_context"].unsqueeze(-2),
    )
    movement_error = (
        smooth_error(motion - batch["motion_target"].unsqueeze(-2)).mean(-1) * responsibilities
    ).sum(-1)
    positive = model.compatibility(
        output["hidden"], batch["geometry_target"], batch["motion_target"]
    )
    # Counterfactual geometry in the same audio/history: a compatibility objective, not human ratings.
    negative_geometry = torch.cat(
        [batch["geometry_target"][:, -7:], batch["geometry_target"][:, :-7]], dim=1
    )
    negative_motion = motion_tensor(
        negative_geometry,
        batch["seconds_target"],
        batch["repeat_target"] + 1,
        batch["kind_target"],
        batch["event_times"],
        batch["motion_context"],
    )
    negative = model.compatibility(output["hidden"], negative_geometry, negative_motion)
    distinct = ((negative_geometry - batch["geometry_target"]).square().sum(-1) > 0.02).to(
        mask.dtype
    )
    negative_valid = batch["event_valid"] * batch["geometry_valid"]
    negative_valid = torch.cat([negative_valid[:, -7:], negative_valid[:, :-7]], dim=1)
    pair_mask = mask * distinct * batch["geometry_valid"] * negative_valid
    ranking = binary_loss(positive - negative, torch.ones_like(positive))

    def term(value, valid, weight=1):
        return ((value * valid).sum() * weight, valid.sum())

    terms = {
        "onset": term(binary_loss(rhythm[:, 0], soft, onset_weight), frames[:, 4]),
        "beat": term(binary_loss(rhythm[:, 1], frames[:, 1], 5), frames[:, 4] * frames[:, 5], 0.3),
        "density": term(smooth_error(rhythm[:, 2] - frames[:, 2]), frames[:, 4], 0.5),
        "tempo": term(smooth_error(rhythm[:, 3] - frames[:, 3]), frames[:, 4] * frames[:, 5], 0.2),
        "geometry": term(nll, mask),
        "kind": term(categorical(output["kind"], batch["kind_target"]), mask, 0.5),
        "duration": term(categorical(output["duration"], batch["duration_target"]), slider, 0.3),
        "repeats": term(categorical(output["repeats"], batch["repeat_target"]), slider, 0.2),
        "combo": term(binary_loss(output["combo"], batch["combo_target"]), mask, 0.1),
        "motion": term(
            smooth_error(output["motion"] - batch["motion_target"]).mean(-1),
            mask * batch["geometry_valid"],
            0.3,
        ),
        "flow": term(movement_error, mask * batch["geometry_valid"], 0.15),
        "compatibility": term(ranking, pair_mask, 0.1),
    }
    if "robust_geometry" in output:
        from .context_training import auxiliary_terms

        terms.update(auxiliary_terms(model, output, batch))
    return terms


def objective(terms):
    return sum(numerator / denominator.clamp(min=1) for numerator, denominator in terms.values())


class LossMeter:
    def __init__(self):
        self.totals = {}
        self.cuda_totals = None
        self.keys = None

    def add(self, terms):
        if next(iter(terms.values()))[0].is_cuda:
            self.keys = list(terms)
            values = torch.stack(
                [
                    torch.stack([numerator.detach(), denominator.detach()])
                    for numerator, denominator in terms.values()
                ]
            )
            self.cuda_totals = values if self.cuda_totals is None else self.cuda_totals + values
            return
        for key, (numerator, denominator) in terms.items():
            current = self.totals.setdefault(key, [0.0, 0.0])
            current[0] += numerator.detach().cpu().item()
            current[1] += denominator.detach().cpu().item()

    def values(self):
        if self.cuda_totals is not None:
            values = self.cuda_totals.cpu().tolist()
            return {
                key: numerator / max(1, count) for key, (numerator, count) in zip(self.keys, values)
            }
        return {key: value / max(1, count) for key, (value, count) in self.totals.items()}

    def value(self):
        return sum(self.values().values())
