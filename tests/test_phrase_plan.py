from dataclasses import replace
import numpy as np
import torch

from app.ml.architecture import ModelConfig
from app.ml.phrase_model import PhraseMapper
from app.ml.phrase_losses import parts, objective, LossMeter
from app.ml.compute_probe import probe_batch
from app.ml.phrase_plan import plan_targets, PLAN_FEATURES
from app.ml.sequence_batches import make_example, tasks_for
from test_phrase_data import example_item


def small_config(**kwargs):
    return ModelConfig(
        width=64, heads=2, audio_layers=1, global_layers=1, decoder_layers=1, dropout=0, **kwargs
    )


def test_plan_masks_rests_unsupported_geometry_and_unknown_timing():
    item = example_item(12)
    item["events"][:, 14] = 0
    values, masks = plan_targets(item["events"], item["labels"], [])
    assert values.shape == masks.shape == (128, PLAN_FEATURES)
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    assert not masks[:, 4:7].any()
    rests = (values[:, 0] == 0) & (masks[:, 0] > 0)
    assert rests.any() and np.all(values[rests, 1] == 1)
    assert not masks[rests, 2:].any()


def test_summary_plan_supervision_is_counted_once_across_overlapping_audio_chunks():
    item = example_item(150)
    total = np.zeros((128, PLAN_FEATURES), np.float32)
    for _, central, targets, weight in tasks_for([item]):
        total += make_example(item, central, targets, weight)["plan_mask"]
    np.testing.assert_allclose(total, item["plan_mask"], atol=1e-6)


def test_plan_ablation_retains_capacity_and_targets_never_feed_the_model():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    enabled = PhraseMapper(small_config()).eval()
    disabled = PhraseMapper(replace(enabled.config, enable_plan=False)).eval()
    disabled.load_state_dict(enabled.state_dict())
    batch = probe_batch()
    with torch.no_grad():
        output = enabled(batch)
        alternate = {**batch, "plan_targets": 1 - batch["plan_targets"]}
        other = enabled(alternate)
        baseline = disabled(batch)
    assert output["plan"].shape == (1, 128, PLAN_FEATURES)
    assert all(torch.equal(output[key], other[key]) for key in output)
    assert sum(p.numel() for p in enabled.parameters()) == sum(
        p.numel() for p in disabled.parameters()
    )
    assert not torch.allclose(output["rhythm"], baseline["rhythm"])
    assert not torch.allclose(output["geometry"], baseline["geometry"])


def test_plan_gradient_is_finite_but_never_selects_checkpoint():
    torch.set_num_threads(2)
    model, batch = PhraseMapper(small_config()), probe_batch()
    terms = parts(model, model(batch), batch)
    assert "plan" in terms
    objective(terms).backward()
    assert model.plan_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    meter = LossMeter()
    meter.add(terms)
    values = meter.values()
    assert (
        abs(
            meter.value()
            - sum(value for key, value in values.items() if key not in ("plan", "context"))
        )
        < 1e-8
    )


def test_musical_time_ablation_ignores_only_the_added_query_channels():
    model = PhraseMapper(small_config(enable_musical_time=False)).eval()
    batch = probe_batch()
    changed = {key: value.clone() for key, value in batch.items()}
    changed["queries"][..., 20:] += 10
    with torch.no_grad():
        before, after = model(batch), model(changed)
    assert torch.equal(before["geometry"], after["geometry"])
