from types import SimpleNamespace

import torch

from app.ml.architecture import DURATION_BEATS
from app.ml.context_counterfactuals import counterfactuals, rolled
from app.ml.context_training import auxiliary_terms, predicted_history
from app.ml.movement import motion_tensor


def compatible_batch():
    curve = torch.tensor([0.10, 0.20, 0.18, 0.15, 0.26, 0.28, 0.35, 0.20])
    geometry = curve[None, None].repeat(1, 4, 1)
    context = torch.tensor([0.05, 0.20, 0.1, 0, 0.1, 0.95, 0.90, 0.1])[None, None].repeat(1, 4, 1)
    batch = dict(
        geometry_target=geometry,
        geometry_valid=torch.ones(1, 4),
        event_valid=torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        event_mask=torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        kind_target=torch.ones(1, 4),
        repeat_target=torch.zeros(1, 4),
        duration_target=torch.full((1, 4), DURATION_BEATS.index(1)),
        seconds_target=torch.full((1, 4), 0.5),
        combo_target=torch.zeros(1, 4),
        event_times=torch.ones(1, 4),
        motion_context=context,
        previous=torch.ones(1, 4, 16) * 0.2,
    )
    batch["previous"][:, 0] = 0
    batch["previous"][:, 3] = 0
    batch["motion_target"] = motion_tensor(
        geometry,
        batch["seconds_target"],
        batch["repeat_target"] + 1,
        batch["kind_target"],
        batch["event_times"],
        context,
    )
    return batch


def test_reflection_family_preserves_complete_shape_and_masks_bos_padding():
    batch = compatible_batch()
    geometry, motion, confidence, preference = counterfactuals(batch)
    assert confidence[0, 0] == 0 and confidence[0, 3] == 0
    assert confidence[0, 1:3].min() > 0
    assert preference.min() >= 0.5 and preference.max() <= 0.75
    assert ((geometry >= 0) & (geometry <= 1)).all()
    scale = torch.tensor([512, 384])
    original_points = batch["geometry_target"].reshape(1, 4, 4, 2) * scale
    transformed_points = geometry.reshape(1, 4, 4, 2) * scale
    original_distances = torch.cdist(original_points, original_points)
    transformed_distances = torch.cdist(transformed_points, transformed_points)
    assert torch.allclose(original_distances, transformed_distances, atol=0.0001)
    assert torch.isfinite(motion).all()


def test_equivalent_alternatives_receive_no_negative_supervision():
    batch = compatible_batch()
    batch["geometry_target"].fill_(0.5)
    batch["motion_target"] = motion_tensor(
        batch["geometry_target"],
        batch["seconds_target"],
        batch["repeat_target"] + 1,
        batch["kind_target"],
        batch["event_times"],
        batch["motion_context"],
    )
    _, _, confidence, preference = counterfactuals(batch)
    assert not confidence.any() and (preference == 0.5).all()
    assert torch.equal(rolled(torch.arange(4)[None], 5), torch.tensor([[3, 0, 1, 2]]))


def test_unsupported_geometry_and_padding_never_become_counterfactual_pairs():
    batch = compatible_batch()
    batch["geometry_valid"].zero_()
    _, _, confidence, _ = counterfactuals(batch)
    assert not confidence.any()


def test_positive_controls_outside_bounds_do_not_leak_a_counterfactual_label():
    batch = compatible_batch()
    batch["geometry_target"][0, 1, 2] = 1.2
    _, _, confidence, _ = counterfactuals(batch)
    assert confidence[0, 1] == 0


def test_auxiliary_loss_has_no_absolute_geometry_supervision_after_changed_history():
    batch = compatible_batch()
    output = dict(
        robust_geometry=torch.randn(1, 4, 34, requires_grad=True),
        robust_kind=torch.randn(1, 4, 2, requires_grad=True),
        robust_duration=torch.randn(1, 4, len(DURATION_BEATS), requires_grad=True),
        robust_repeats=torch.randn(1, 4, 4, requires_grad=True),
        robust_combo=torch.randn(1, 4, requires_grad=True),
        robust_mask=batch["event_mask"],
    )
    model = SimpleNamespace(config=SimpleNamespace(mixtures=2))
    loss = auxiliary_terms(model, output, batch)["context"][0]
    batch["geometry_target"] = 1 - batch["geometry_target"]
    alternate = auxiliary_terms(model, output, batch)["context"][0]
    assert torch.equal(loss, alternate)
    loss.backward()
    assert output["robust_geometry"].grad is None
    assert output["robust_kind"].grad is not None
    assert all(
        value.grad is None or torch.isfinite(value.grad).all()
        for key, value in output.items()
        if key != "robust_mask"
    )


def test_own_history_translation_stays_in_bounds_and_does_not_touch_bos():
    batch = compatible_batch()
    batch["context_probability"] = torch.ones(1)
    batch["previous"][0, 1:3, :10] = 0.995
    predicted = torch.ones(1, 4, 34) * 10
    previous, affected = predicted_history(batch, predicted, 2)
    assert (previous[..., :10] >= 0).all() and (previous[..., :10] <= 1).all()
    assert not previous[:, 0].any() and not previous[:, 3].any()
    assert not affected[:, 0].any() and not affected[:, 3].any()
