import math
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from app.ml.architecture import ModelConfig
from app.ml.phrase_model import PhraseMapper
from app.ml.compute_probe import probe_batch
from app.ml.phrase_losses import categorical, parts, objective, LossMeter, mixture_loss
from app.ml.movement import motion_numpy, motion_tensor
from app.ml.phrase_metrics import onset_matches


def tiny_model():
    torch.set_num_threads(2)
    torch.manual_seed(14)
    return PhraseMapper(
        ModelConfig(width=64, heads=2, audio_layers=1, global_layers=1, decoder_layers=2, dropout=0)
    )


def test_onset_matching_does_not_steal_the_next_valid_reference():
    assert onset_matches([130, 170], [100, 150], 35) == 2
    assert onset_matches([130, 140, 170], [100, 150], 35) == 2


def test_future_objects_cannot_leak_but_longer_history_and_song_audio_are_used():
    model, batch = tiny_model().eval(), probe_batch()
    with torch.no_grad():
        original = model(batch)
        changed = {key: value.clone() for key, value in batch.items()}
        changed["previous"][:, 8:] += 2
        changed["queries"][:, 8:] -= 3
        future = model(changed)
        assert torch.allclose(original["geometry"][:, :8], future["geometry"][:, :8], atol=1e-6)
        changed = {key: value.clone() for key, value in batch.items()}
        changed["previous"][:, 1] += 2
        history = model(changed)
        assert not torch.allclose(original["geometry"][:, 10], history["geometry"][:, 10], atol=1e-5)
        changed = {key: value.clone() for key, value in batch.items()}
        changed["summary"][:, 90:] += 0.5
        distant_audio = model(changed)
        assert not torch.allclose(original["geometry"][:, 5], distant_audio["geometry"][:, 5], atol=1e-5)


def test_joint_objectives_have_finite_gradients_and_can_fit_one_batch():
    model, batch = tiny_model(), probe_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
    losses = []
    for _ in range(14):
        optimizer.zero_grad()
        loss = objective(parts(model, model(batch), batch))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        losses.append(loss.item())
    assert min(losses[-3:]) < losses[0] - 0.5


def test_categorical_matches_reference_including_ties_and_extreme_logits():
    values = torch.tensor([[0.0, 0.0, 0.0], [1000, -1000, 0], [2, -3, 7]], requires_grad=True)
    labels = torch.tensor([2, 1, 0])
    actual = categorical(values, labels)
    reference = F.cross_entropy(values, labels, reduction="none")
    assert torch.allclose(actual, reference, atol=1e-6)
    assert torch.allclose(
        torch.autograd.grad(actual.sum(), values)[0],
        torch.autograd.grad(reference.sum(), values)[0],
        atol=1e-6,
    )


def test_joint_mixture_keeps_alternative_layouts_instead_of_averaging():
    raw = torch.zeros(1, 2, 17)
    raw[0, 0, 1:9] = math.log(0.375 / 0.625)  # normalized mean .25
    raw[0, 1, 1:9] = math.log(0.625 / 0.375)  # normalized mean .75
    raw[:, :, 9:] = -4
    dims = torch.tensor([[1.0, 1.0, 0, 0, 0, 0, 0, 0]])
    edge = mixture_loss(raw.reshape(1, -1), torch.full((1, 8), 0.25), dims, 2)[0]
    middle = mixture_loss(raw.reshape(1, -1), torch.full((1, 8), 0.5), dims, 2)[0]
    assert edge.item() < middle.item() - 10


def test_motion_training_matches_candidate_scoring():
    geometry = np.array([0.2, 0.3, 0.4, 0.1, 0.7, 0.6, 0.8, 0.4], np.float32)
    context = np.array([0.3, 0.5, 0.1, -0.2, 0.2, 1.0, 0.8, 0.2], np.float32)
    expected = motion_numpy(geometry, 0.5, 2, 1, 1.4, context)
    actual = motion_tensor(
        torch.tensor(geometry),
        torch.tensor(0.5),
        torch.tensor(2.0),
        torch.tensor(1.0),
        torch.tensor(1.4),
        torch.tensor(context),
    )
    assert np.allclose(actual.numpy(), expected, atol=1e-5)


def test_loss_aggregation_is_independent_of_batch_partition():
    model, batch = tiny_model().eval(), probe_batch()
    joined = {key: torch.cat([value, value], dim=0) for key, value in batch.items()}
    joined["event_mask"][1, 5:] = 0
    one, many = LossMeter(), LossMeter()
    with torch.no_grad():
        one.add(parts(model, model(joined), joined))
        for index in range(2):
            single = {key: value[index : index + 1] for key, value in joined.items()}
            many.add(parts(model, model(single), single))
    assert one.value() == pytest.approx(many.value(), abs=1e-5)


def styled_model(slots=4):
    torch.set_num_threads(2)
    torch.manual_seed(5)
    return PhraseMapper(
        ModelConfig(
            width=64, heads=2, audio_layers=1, global_layers=1, decoder_layers=2, dropout=0,
            style_version=2, mapper_slots=slots,
        )  # fmt: skip
    )


def test_styled_model_rhythm_head_is_causal_and_condition_and_mapper_matter():
    model, batch = styled_model().eval(), probe_batch(styled=True)
    assert batch["condition"].shape[-1] == 52 and "gap_target" in batch
    with torch.no_grad():
        original = model(batch)
        assert original["gap"].shape[-1] == 27
        changed = {key: value.clone() for key, value in batch.items()}
        changed["previous"][:, 8:] += 2
        changed["queries"][:, 8:] -= 3
        changed["kind_target"][:, 8:] = 1 - changed["kind_target"][:, 8:]
        future = model(changed)
        assert torch.allclose(original["gap"][:, :8], future["gap"][:, :8], atol=1e-6)
        other = {key: value.clone() for key, value in batch.items()}
        other["mapper"] = torch.full_like(batch["mapper"], 3)
        assert not torch.allclose(original["gap"], model(other)["gap"], atol=1e-6)
        stars = {key: value.clone() for key, value in batch.items()}
        stars["condition"][:, 0] += 0.5
        assert not torch.allclose(original["rhythm"], model(stars)["rhythm"], atol=1e-6)


def test_styled_objectives_include_rhythm_patterns_and_fit_one_batch():
    model, batch = styled_model(), probe_batch(styled=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
    losses = []
    for _ in range(14):
        optimizer.zero_grad()
        terms = parts(model, model(batch), batch)
        assert "gap" in terms and terms["gap"][1] > 0
        loss = objective(terms)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.mapper.weight.grad is not None and model.gap[0].weight.grad is not None
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        losses.append(loss.item())
    assert min(losses[-3:]) < losses[0] - 0.5


def test_warm_start_reuses_matching_weights_from_a_legacy_model(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from app.ml import checkpoints

    legacy = tiny_model()
    folder = tmp_path / "models" / "legacy"
    folder.mkdir(parents=True)
    save_file({key: value.contiguous() for key, value in legacy.state_dict().items()}, str(folder / "model.safetensors"))
    monkeypatch.setattr(checkpoints, "DATA", tmp_path)
    model = styled_model()
    result = checkpoints.warm_start(model, "legacy")
    assert result["copied_tensors"] > 20 and result["new_tensors"] >= 3
    assert torch.equal(model.audio_front[0].weight, legacy.audio_front[0].weight)
    assert not torch.equal(model.condition[0].weight[:, :8], legacy.condition[0].weight)
