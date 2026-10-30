"""Bounded joint structure/geometry candidates scored after all object repairs."""

from dataclasses import dataclass
import math

import numpy as np
import torch

from .architecture import DURATION_BEATS
from .generation_trace import object_snapshot
from .movement_guidance import head_geometry
from .playability import candidate_object, low_difficulty_cost, readability_cost
from .slider_geometry import geometry_candidates


@dataclass
class Candidate:
    obj: object
    geometry: np.ndarray
    motion: np.ndarray
    score: float
    repair: float
    identifier: object


def movement_cost(motion, expected, spacing=True):
    """Deviation of a candidate's movement from the model's expected movement.

    `expected` is a regression output, i.e. a conditional mean. For direction (cosine and
    sine to the previous movement) an uncertain model predicts about 0, which is exactly
    the value of a stacked note (distance 0), so plain deviation rewarded stacks and the
    map collapsed to 0-5 px spacing. Direction now only counts as far as the model is sure
    of it, and the spacing itself is compared on a log scale, where the old /512 units made
    a 150 px jump versus a stack nearly free.

    Without a previous object (`spacing=False`) the distance feature is measured from the
    playfield centre, so the log-spacing comparison would only pull the song's first object
    towards a meaningless reference and outweigh its structure evidence.
    """
    motion, expected = np.asarray(motion, np.float64), np.asarray(expected, np.float64)
    deviation = np.minimum(np.abs(motion - expected), 5)
    certainty = float(np.clip(np.hypot(expected[2], expected[3]), 0, 1))
    weights = np.array([1, 1, certainty, certainty, 1, 1, 1, 1], np.float64)
    distance, wanted = max(0.0, motion[0] * 512), max(0.0, expected[0] * 512)
    gap = abs(math.log((distance + 16) / (wanted + 16))) if spacing else 0.0
    return float((deviation * weights).sum() / len(weights) + 0.5 * gap)


def probabilities(logits, temperature=1):
    scores = np.asarray(logits, dtype=np.float64) / temperature
    scores -= scores.max()
    weights = np.exp(scores)
    return weights / weights.sum()


def structure_candidates(kind, duration, repeats, available, beat, slider_bias):
    structures = []
    if slider_bias < 1:
        structures.append((False, 0, 0))
    if slider_bias > 0:
        fitting = [
            index for index, value in enumerate(DURATION_BEATS) if value * beat <= available + 1e-6
        ]
        fitting.sort(key=lambda index: -duration[index])
        for duration_index in fitting[:2]:
            for repeat_index in np.argsort(-repeats)[:2]:
                structures.append((True, duration_index, int(repeat_index)))
    # A candidate cap of one still makes a deterministic joint choice, not a random type draw.
    structures.sort(key=lambda item: -structure_score(item, kind, duration, repeats)[0])
    return structures


def structure_score(structure, kind, duration, repeats):
    slider, duration_index, repeat_index = structure
    kind_log = float(np.log(max(kind[int(slider)], 1e-12)))
    duration_log = float(np.log(max(duration[duration_index], 1e-12))) if slider else 0.0
    repeat_log = float(np.log(max(repeats[repeat_index], 1e-12))) if slider else 0.0
    duration_entropy = float(-(duration * np.log(np.maximum(duration, 1e-12))).sum())
    repeat_entropy = float(-(repeats * np.log(np.maximum(repeats, 1e-12))).sum())
    # Center conditional evidence on its own distribution: uninformative duration/
    # repeats must not create an extra negative type prior against every slider.
    conditional = ((duration_log + duration_entropy) + (repeat_log + repeat_entropy)) / 2
    score = 0.25 * kind_log + (0.15 * conditional if slider else 0)
    return score, kind_log, duration_log, repeat_log


def rank_candidates(
    model,
    output,
    time,
    next_time,
    beat,
    settings,
    objects,
    rng,
    selected_device,
    trace=None,
    branch="head",
    motif_memory=None,
    guide=None,
):
    """Candidates for the object at `time`, best first.

    `guide` (a MovementGuide) adds jump proposals at the requested spacing and soft costs
    for spacing collapse, border hugging, orbiting and a runaway slider share.
    """
    if not all(torch.isfinite(value).all() for value in output.values()):
        raise ValueError("Non-finite object prediction from model")
    raw = output["geometry"][0, -1].cpu().numpy().reshape(model.config.mixtures, 17)
    weights = probabilities(raw[:, 0], settings.get("temperature", 0.8))
    means = 1 / (1 + np.exp(-np.clip(raw[:, 1:9], -60, 60))) * 2 - 0.5
    scales = np.exp(np.clip(raw[:, 9:], -4.5, -0.5))
    kind = probabilities(output["kind"][0, -1].cpu().numpy())
    durations = probabilities(output["duration"][0, -1].cpu().numpy())
    repeats = probabilities(output["repeats"][0, -1].cpu().numpy())
    combo = float(output["combo"][0, -1].cpu()) > 0
    expected_motion = output["motion"][0, -1].cpu().numpy()
    structures = structure_candidates(
        kind, durations, repeats, next_time - time - 35, beat, settings["slider_bias"]
    )
    count = max(1, settings.get("candidates", 12))
    candidates = []
    geometry_count = math.ceil(count / max(1, len(structures)))
    # Every sampled shape is a whole-curve transform, shared by its structural alternatives.
    geometries = list(
        geometry_candidates(
            means, scales, weights, geometry_count, settings.get("temperature", 0.8), rng, True
        )
    )
    if motif_memory is not None and len(geometries) > 1:
        proposals = motif_memory.proposals(time, objects)
        if proposals:
            geometries[-1] = (np.asarray(proposals[0], dtype=np.float64), False)
            if trace is not None:
                trace.record(
                    "motif_proposal",
                    time=time,
                    branch=branch,
                    replaced_geometry_index=len(geometries) - 1,
                )
    if guide is not None and objects:
        # The most likely learned shape, moved to the requested jump distance: a jump
        # exists even when every mixture mean sits on top of the previous object.
        base = means[int(np.argmax(weights))]
        jumps = [
            (head_geometry(base, head), False) for head in guide.jump_heads(time, objects, beat)
        ]
        if jumps:
            geometries = geometries[:geometry_count] + jumps
            if trace is not None:
                trace.record("jump_proposals", time=time, branch=branch, count=len(jumps))
    for geometry_index, (proposal, _mode) in enumerate(geometries):
        for structure_index, structure in enumerate(structures):
            candidate_id = f"{branch}:{geometry_index * len(structures) + structure_index}"
            # The candidate cap applies to the learned proposals; jump proposals come on top.
            if geometry_index < geometry_count and (
                geometry_index * len(structures) + structure_index >= count
            ):
                break
            slider, duration_index, repeat_index = structure
            duration = DURATION_BEATS[duration_index] if slider else 0
            result = candidate_object(
                proposal,
                time,
                next_time,
                beat,
                slider,
                duration,
                repeat_index + 1,
                combo,
                settings,
                objects,
                trace=trace,
                candidate_id=candidate_id,
            )
            if result is None:
                continue
            obj, geometry, motion, repair = result
            dimensions = 8 if obj.slider else 2
            components = np.log(weights + 1e-12) - (
                0.5
                * ((geometry[:dimensions] - means[:, :dimensions]) / scales[:, :dimensions]) ** 2
                + np.log(scales[:, :dimensions])
            ).sum(-1)
            maximum = components.max()
            likelihood = (maximum + np.log(np.exp(components - maximum).sum())) / dimensions
            # Score the final type when a proposed short slider fell back to a circle.
            final_structure = (obj.slider, duration_index, repeat_index)
            structural, kind_log, duration_log, repeat_log = structure_score(
                final_structure, kind, durations, repeats
            )
            movement = movement_cost(motion, expected_motion, spacing=bool(objects))
            readability = readability_cost(obj, objects, settings["cs"], settings["ar"])
            beginner = low_difficulty_cost(obj, objects, beat, settings)
            motif_cost = motif_memory.cost(obj, objects) if motif_memory is not None else 0.0
            guidance = guide.cost(obj, objects, beat) if guide is not None else 0.0
            score = (
                0.15 * likelihood
                + structural
                - movement
                - 3 * repair
                - 0.3 * readability
                - 0.5 * beginner
                - 0.15 * motif_cost
                - guidance
            )
            candidates.append(Candidate(obj, geometry, motion, float(score), repair, candidate_id))
            if trace is not None:
                trace.record(
                    "candidate_score",
                    time=time,
                    candidate_id=candidate_id,
                    geometry_log_likelihood=likelihood,
                    geometry_dimensions=dimensions,
                    kind_log_probability=kind_log,
                    duration_log_probability=duration_log,
                    repeat_log_probability=repeat_log,
                    weighted_structure=structural,
                    weighted_geometry=0.15 * likelihood,
                    weighted_movement=-movement,
                    weighted_repair=-3 * repair,
                    weighted_readability=-0.3 * readability,
                    weighted_low_difficulty=-0.5 * beginner,
                    weighted_motif=-0.15 * motif_cost,
                    weighted_guidance=-guidance,
                    score_before_compatibility=score,
                    structure_probabilities_used_in_score=True,
                )
    if not candidates:
        result = candidate_object(
            means[int(weights.argmax())],
            time,
            next_time,
            beat,
            False,
            0,
            1,
            combo,
            settings,
            objects,
            trace=trace,
            candidate_id=f"{branch}:fallback",
        )
        obj, geometry, motion, repair = result
        candidates = [Candidate(obj, geometry, motion, -1.0, repair + 1, f"{branch}:fallback")]
        if trace is not None:
            trace.record(
                "slider_fallback",
                time=time,
                candidate_id=candidates[0].identifier,
                reason="no_suitable_candidate",
                score_before_compatibility=-1,
                repair=repair + 1,
            )
    geometries = torch.tensor(
        np.stack([item.geometry for item in candidates]),
        dtype=torch.float32,
        device=selected_device,
    )
    motions = torch.tensor(
        np.stack([item.motion for item in candidates]), dtype=torch.float32, device=selected_device
    )
    hidden = output["hidden"][0, -1].unsqueeze(0).expand(len(candidates), -1)
    critic = model.compatibility(hidden, geometries, motions).cpu().numpy()
    for index, candidate in enumerate(candidates):
        bonus = 0.2 * np.tanh(critic[index])
        candidate.score += bonus
        if trace is not None:
            trace.record(
                "candidate_rank",
                time=time,
                candidate_id=candidate.identifier,
                compatibility=critic[index],
                weighted_compatibility=bonus,
                total_score=candidate.score,
            )
    return sorted(candidates, key=lambda candidate: -candidate.score)


def choose_candidate(
    model, output, time, next_time, beat, settings, objects, rng, selected_device, trace=None
):
    chosen = rank_candidates(
        model, output, time, next_time, beat, settings, objects, rng, selected_device, trace
    )[0]
    if trace is not None:
        trace.record(
            "object_choice",
            time=time,
            candidate_id=chosen.identifier,
            object=object_snapshot(chosen.obj),
            repair=chosen.repair,
        )
    return chosen.obj, chosen.repair
