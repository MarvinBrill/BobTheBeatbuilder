import numpy as np
import pytest
import torch
from app.ml.movement import motion_numpy, motion_batch_numpy
from app.ml.training_runtime import TrainingRuntime
from app.ml.phrase_losses import LossMeter
from app.ml.batch_loading import device_batches
from app.ml.attention import Attention


def test_vectorized_motion_targets_match_scalar_reference():
    rng = np.random.default_rng(15)
    geometry = rng.uniform(0, 1, (128, 8)).astype(np.float32)
    context = rng.uniform(0, 1, (128, 8)).astype(np.float32)
    duration = rng.uniform(0.1, 2, 128).astype(np.float32)
    repeats = rng.integers(1, 5, 128)
    slider = rng.integers(0, 2, 128)
    time = np.arange(128, dtype=np.float32) * 0.1
    expected = np.stack(
        [
            motion_numpy(g, d, r, s, t, c)
            for g, d, r, s, t, c in zip(geometry, duration, repeats, slider, time, context)
        ]
    )
    actual = motion_batch_numpy(geometry, duration, repeats, slider, time, context)
    assert np.allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_cpu_runtime_keeps_float32_and_rejects_explicit_bf16():
    runtime = TrainingRuntime(torch.device("cpu"))
    assert not runtime.bf16 and runtime.precision == "float32"
    with pytest.raises(ValueError, match="requires a CUDA"):
        TrainingRuntime(torch.device("cpu"), "bfloat16")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
@pytest.mark.parametrize("causal", [False, True])
def test_cuda_attention_preserves_padding_causality_outputs_and_gradients(causal):
    torch.manual_seed(2)
    attention = Attention(64, 2, 0).cuda().eval()
    values = torch.randn(2, 12, 64, device="cuda", requires_grad=True)
    valid = torch.ones(2, 12, device="cuda")
    valid[0, 7:] = 0
    attention.cuda_sdpa = False
    reference = attention(values, values, valid, causal)
    reference_grad = torch.autograd.grad(reference.square().sum(), values)[0]
    attention.cuda_sdpa = True
    actual = attention(values, values, valid, causal)
    actual_grad = torch.autograd.grad(actual.square().sum(), values)[0]
    assert torch.allclose(actual, reference, rtol=2e-5, atol=2e-6)
    assert torch.allclose(actual_grad, reference_grad, rtol=2e-4, atol=2e-5)
    changed = values.detach().clone()
    changed[0, 7:] += 100
    if causal:
        changed[1, 7:] += 100
    assert torch.allclose(attention(changed, changed, valid, causal)[:, :7], actual[:, :7], atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_prefetch_preserves_order_and_propagates_producer_errors():
    def source():
        for index in range(5):
            yield {"x": np.full((2, 3), index, np.float32)}
        raise ValueError("producer failed")

    with device_batches(source(), torch.device("cuda")) as loaded:
        for index in range(5):
            assert torch.equal(next(loaded)["x"].cpu(), torch.full((2, 3), float(index)))
        with pytest.raises(ValueError, match="producer failed"):
            next(loaded)
    # Early cancellation closes the producer even when a future is already pending.
    with device_batches(source(), torch.device("cuda")) as loaded:
        next(loaded)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_gpu_loss_accumulation_is_detached_and_matches_cpu():
    gpu, cpu = LossMeter(), LossMeter()
    for index in range(5):
        value = torch.tensor(float(index), device="cuda", requires_grad=True)
        gpu.add({"test": (value, torch.tensor(2.0, device="cuda"))})
        cpu.add({"test": (value.detach().cpu(), torch.tensor(2.0))})
    assert not gpu.cuda_totals.requires_grad
    assert gpu.values() == cpu.values()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_nonfinite_accumulated_loss_refuses_optimizer_update():
    runtime = TrainingRuntime(torch.device("cuda"))
    model = torch.nn.Linear(2, 1).cuda()
    model(torch.ones(1, 2, device="cuda")).sum().backward()
    with pytest.raises(ValueError, match="update refused"):
        runtime.clip_and_check(model, torch.tensor(False, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_graph_replay_uses_new_inputs_preserves_gradients_and_does_not_freeze_dropout():
    from app.ml.compute_probe import probe_batch
    from app.ml.phrase_model import PhraseMapper
    from app.ml.architecture import ModelConfig
    from app.ml.cuda_graph import CudaTrainingGraph
    from app.ml.phrase_losses import objective, parts

    torch.set_num_threads(2)
    model = PhraseMapper(
        ModelConfig(width=64, heads=2, audio_layers=1, global_layers=1, decoder_layers=1, dropout=0)
    ).cuda()
    runtime = TrainingRuntime(torch.device("cuda"), "float32")
    batch = {key: value.cuda() for key, value in probe_batch().items()}
    random_state = torch.cuda.get_rng_state().clone()
    graph = CudaTrainingGraph(model, runtime, batch, 20)
    assert torch.equal(random_state, torch.cuda.get_rng_state())
    optimizer = runtime.optimizer(model, 0.0003, 0.01)
    for index in range(2):
        batch["audio"].add_(0.1)
        model.zero_grad(set_to_none=True)
        expected = objective(parts(model, runtime.forward(model, batch), batch))
        expected.backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        actual, _ = graph.replay(batch)
        assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-5)
        assert all(
            torch.allclose(p.grad, g, rtol=1e-3, atol=2e-5) for p, g in zip(model.parameters(), gradients)
        )
        # The next replay must read changed weights and overwrite, not accumulate, old gradients.
        optimizer.step()
    assert not graph.matches({**batch, "audio": batch["audio"].repeat(2, 1, 1)})
    model = PhraseMapper(
        ModelConfig(width=64, heads=2, audio_layers=1, global_layers=1, decoder_layers=1, dropout=0.1)
    ).cuda()
    graph = CudaTrainingGraph(model, runtime, batch, 20)
    first = graph.replay(batch)[0].item()
    second = graph.replay(batch)[0].item()
    assert first != second
