import json
import math
import numpy as np
import pytest
import torch
from app.ml.training_control import TrainingControl, context_probability
from app.ml.quality_tracking import grouped_losses
from app.ml.context_training import predicted_history
from app.ml.compute_probe import probe_batch
from app.ml.architecture import ModelConfig
from app.ml.phrase_model import PhraseMapper
from app.ml.phrase_losses import parts, objective
from app.ml.training_runtime import TrainingRuntime


def small_model(device="cpu"):
    return PhraseMapper(
        ModelConfig(width=64, heads=2, audio_layers=1, global_layers=1, decoder_layers=1, dropout=0)
    ).to(device)


def test_plateau_reduces_before_stopping_and_restore_keeps_next_rate():
    control = TrainingControl(0.0003, 12, 2)
    assert control.step_rate() == 0.00015
    control.updates += 1
    assert control.step_rate() == 0.0003
    assert control.observe(-0.5) == (True, False, False)
    for epoch in range(1, 13):
        improved, reduced, stop = control.observe(-0.4)
        assert not improved
        assert reduced == (epoch in (3, 6, 9))
        assert stop == (epoch == 12)
        restored = TrainingControl(0.0003, 12, 200, state=json.loads(json.dumps(control.state())))
        assert restored.state() == control.state()
        assert restored.step_rate() == control.step_rate()
    assert control.lr == pytest.approx(0.0003 / 8)
    assert control.observe(-0.6) == (True, False, False)


def test_no_validation_does_not_early_stop_and_nonfinite_rejected():
    control = TrainingControl(0.0003, 1, 0)
    for _ in range(8):
        assert not control.observe(1, validation=False)[2]
    assert control.lr == 0.0003
    with pytest.raises(ValueError, match="Non-finite"):
        control.observe(math.nan)
    empty = TrainingControl(0.0003, 12, 2)
    assert json.loads(json.dumps(empty.state(), allow_nan=False))["best"] is None


def test_context_curriculum_and_loss_groups_keep_original_objective():
    assert [context_probability(e, True) for e in (1, 4, 5, 14, 100)] == [0, 0, 0.02, 0.2, 0.2]
    assert context_probability(100, False) == 0
    terms = dict(
        onset=2,
        beat=0.3,
        density=0.1,
        tempo=0.1,
        geometry=-1,
        kind=0.2,
        duration=0.1,
        repeats=0.1,
        combo=0.1,
        motion=0.1,
        flow=0.1,
        compatibility=0.1,
        context=99,
    )
    assert sum(grouped_losses(terms).values()) == pytest.approx(sum(terms.values()) - 99)


def test_predicted_history_is_causal_bounded_rigid_and_does_not_change_targets():
    batch = probe_batch()
    batch["context_probability"] = torch.ones(1)
    original = {key: value.clone() for key, value in batch.items()}
    geometry = torch.randn(1, 128, 8 * 17)
    actual, affected = predicted_history(batch, geometry, 8)
    delta = actual[..., :2] - batch["previous"][..., :2]
    distance = torch.linalg.vector_norm(delta * torch.tensor([512, 384]), dim=-1)
    assert distance.max() <= 16.001
    assert torch.equal(actual[:, 0], batch["previous"][:, 0])
    assert torch.equal(actual[:, 12:], batch["previous"][:, 12:])
    assert affected[0, 1:12].all() and not affected[0, 12:].any()
    for column in (2, 4, 6, 8):
        assert torch.allclose(
            actual[..., column : column + 2] - batch["previous"][..., column : column + 2],
            delta,
            atol=1e-6,
        )
    changed = geometry.clone()
    changed[:, 5:] += 10
    other, _ = predicted_history(batch, changed, 8)
    assert torch.equal(actual[:, :6], other[:, :6])
    assert all(torch.equal(batch[key], value) for key, value in original.items())


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
        ),
    ],
)
def test_auxiliary_path_preserves_primary_outputs_and_has_finite_gradients(device):
    torch.set_num_threads(2)
    model = small_model(device)
    batch = {key: value.to(device) for key, value in probe_batch().items()}
    runtime = TrainingRuntime(torch.device(device), "float32")
    baseline = runtime.forward(model, batch)
    base_loss = objective(parts(model, baseline, batch))
    base_gradient = torch.autograd.grad(base_loss, model.audio_front[0].weight)[0]
    batch["context_probability"] = torch.ones(1, device=device)
    augmented = runtime.forward(model, batch)
    assert all(torch.equal(augmented[key], value) for key, value in baseline.items())
    losses = parts(model, augmented, batch)
    assert "context" in losses
    objective(losses).backward()
    # Added decoder supervision must not introduce an audio-encoder gradient.
    assert torch.allclose(model.audio_front[0].weight.grad, base_gradient, atol=1e-6)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    assert "robust_geometry" not in model(batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_context_cuda_graph_accepts_changing_probability_and_bf16_updates():
    from app.ml.cuda_graph import CudaTrainingGraph

    model = small_model("cuda")
    runtime = TrainingRuntime(torch.device("cuda"), "bfloat16")
    batch = {key: value.cuda() for key, value in probe_batch().items()}
    batch["context_probability"] = torch.zeros(1, device="cuda")
    graph = CudaTrainingGraph(model, runtime, batch, 20)
    optimizer = runtime.optimizer(model, 0.0003, 0.01)
    loss, terms = graph.replay(batch)
    assert terms["context"][0].item() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    batch["context_probability"].fill_(1)
    loss, terms = graph.replay(batch)
    assert torch.isfinite(loss) and terms["context"][0].item() != 0
    runtime.clip_and_check(model, torch.isfinite(loss))
    optimizer.step()


def test_rollout_suite_uses_distinct_fixed_songs_and_reports_failures(monkeypatch):
    from app.ml import rollout_evaluation as rollout
    from app.jobs import Cancelled

    maps = [dict(map_id=str(i), song=str(i // 2), mel=np.zeros((100, 64))) for i in range(10)]
    chosen = rollout.rollout_songs(maps)
    assert len({item["song"] for item in chosen}) == 3
    assert chosen == rollout.rollout_songs(list(reversed(maps)))

    class Context:
        def check(self):
            pass

    def fail(*args, **kwargs):
        raise ValueError("No playable notes")

    monkeypatch.setattr(rollout, "evaluate_rollout", fail)
    suite = rollout.evaluate_suite(None, maps, None, Context())
    assert suite["attempted"] == 3 and suite["songs"] == 0 and len(suite["failures"]) == 3

    class Stop:
        def check(self):
            raise Cancelled()

    with pytest.raises(Cancelled):
        rollout.evaluate_suite(None, maps, None, Stop())


def test_training_resume_preserves_controller_and_separates_rollout_splits(monkeypatch):
    from app import store
    from app.config import DATA
    from app.ml import training
    from app.ml.training_setup import split_maps
    from dataclasses import replace
    from safetensors import safe_open
    from test_phrase_data import example_item
    from app.ml import style

    torch.set_num_threads(2)
    values, mask = style.vector({"ar": 8, "cs": 4, "od": 6, "hp": 5, "tempo": 120, "stars": 3})
    prepared = [
        dict(
            **example_item(12),
            style=values,
            style_mask=mask,
            creator="Tester",
            mapper_candidate=None,
            song=str(i),
            map_id=str(i),
            map_hash=str(i),
            audio_hash=str(i),
            masked_geometry=0
        )
        for i in range(8)
    ]
    train, validation, test = split_maps(prepared, 42)
    calls = []
    monkeypatch.setattr(training, "prepare", lambda *args: prepared)
    monkeypatch.setattr(
        training, "configuration", lambda _: replace(small_model().config, style_version=2)
    )
    monkeypatch.setattr(
        training, "evaluate", lambda *args, **kwargs: (1.0, {"loss_terms": {"onset": 1.0}})
    )

    def suite(model, maps, *args, **kwargs):
        calls.append({item["song"] for item in maps})
        return dict(songs=0, attempted=1, means={}, results=[], failures=[])

    monkeypatch.setattr(training, "evaluate_suite", suite)
    store.save("dataset", dict(id="policy-resume", maps=[]))
    settings = dict(
        dataset_id="policy-resume",
        name="Policy test",
        device="cpu",
        epochs=6,
        batch_size=4,
        gradient_accumulation=2,
        context_robustness=True,
        learning_rate=0.0003,
        seed=42,
    )

    class Context:
        def check(self):
            pass

        def report(self, *args, **kwargs):
            pass

    first = store.get("model", training.train(settings, Context())["model_id"])
    assert first["best_epoch"] == 1 and first["completed_epochs"] == 6
    assert first["metrics"][3]["learning_rate_reduced"]
    assert first["metrics"][4]["context_probability"] == 0.02
    assert calls[:-1] == [{item["song"] for item in validation}] * 3
    assert calls[-1] == {item["song"] for item in test}
    with safe_open(
        str(DATA / "models" / first["id"] / "training.safetensors"), framework="pt"
    ) as saved:
        control = json.loads(saved.metadata()["state"])["training_control"]
    assert control["stale"] == 5 and control["lr"] == 0.00015
    second = store.get(
        "model",
        training.train({**settings, "epochs": 1, "resume_model_id": first["id"]}, Context())[
            "model_id"
        ],
    )
    assert second["completed_epochs"] == 7 and second["best_epoch"] == 1
    assert second["metrics"][-1]["learning_rate"] == 0.00015
    assert second["training_summary"]["next_learning_rate"] == 0.000075
    assert second["training_summary"]["stale_epochs"] == 6
    # Downloaded inference weights still refer to the original best epoch.
    assert (DATA / "models" / first["id"] / "model.safetensors").read_bytes() == (
        DATA / "models" / second["id"] / "model.safetensors"
    ).read_bytes()


def test_legacy_models_cannot_be_resumed_but_rhythm_groups_are_reported():
    from app import store
    from app.ml import training

    store.save(
        "model",
        dict(id="legacy-resume", model_config=small_model().config.dictionary(), settings={"seed": 1}),
    )
    with pytest.raises(ValueError, match="Startpunkt"):
        training.train(dict(dataset_id="none", device="cpu", resume_model_id="legacy-resume"), None)
    assert grouped_losses({"onset": 1, "gap": 0.5})["pattern"] == 0.5
    assert "pattern" not in grouped_losses({"onset": 1})
