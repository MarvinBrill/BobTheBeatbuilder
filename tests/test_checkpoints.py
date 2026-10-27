import torch
import json
import numpy as np
import pytest
from app import store
from app.config import DATA
from app.ml.architecture import ARCHITECTURE
from app.ml.checkpoints import save_training, restore
from app.ml.architecture import EVENT_SCHEMA, ModelConfig
from app.ml.recovery import recover_training_runs
from app.jobs import Cancelled


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
        ),
    ],
)
def test_restored_optimizer_performs_identical_next_update(device):
    torch.manual_seed(5)
    first = torch.nn.Linear(3, 2).to(device)
    options = {"fused": True} if device == "cuda" else {"foreach": False}
    first_optimizer = torch.optim.AdamW(first.parameters(), lr=0.001, **options)
    values = torch.randn(4, 3, device=device)

    def step(model, optimizer):
        optimizer.zero_grad()
        model(values).square().mean().backward()
        optimizer.step()

    step(first, first_optimizer)
    identifier = f"resume-optimizer-test-{device}"
    directory = DATA / "models" / identifier
    directory.mkdir(exist_ok=True)
    store.save("model", dict(id=identifier, architecture=ARCHITECTURE, dataset_id="test"))
    save_training(first, first_optimizer, directory, dict(completed_epochs=2, optimizer_steps=1))
    second = torch.nn.Linear(3, 2).to(device)
    second_optimizer = torch.optim.AdamW(second.parameters(), lr=0.001, **options)
    _, state = restore(second, second_optimizer, identifier, "test")
    assert all(entry["step"].device.type == device for entry in second_optimizer.state.values())
    assert state == dict(completed_epochs=2, optimizer_steps=1)
    step(first, first_optimizer)
    step(second, second_optimizer)
    assert all(torch.equal(left, right) for left, right in zip(first.parameters(), second.parameters()))


def test_cancelled_training_is_preserved_and_can_continue(monkeypatch):
    from app.ml import training
    from app.ml.sequence_data import pack_events, frame_labels, summary_features
    from app.beatmaps import Beatmap, HitObject

    torch.set_num_threads(2)
    mel = np.zeros((1024, 64), np.float32)
    beatmap = Beatmap(
        objects=[
            HitObject(100 + i * 30, 100, 500 + i * 500, end_x=100 + i * 30, end_y=100) for i in range(8)
        ],
        timing=[(0, 500, False)],
    )
    events, _ = pack_events(beatmap, len(mel))
    item = dict(
        mel=mel,
        events=events,
        labels=frame_labels(beatmap, len(mel)),
        summary=summary_features(mel),
        condition=np.full(8, 0.5, np.float32),
        masked_geometry=0,
        audio_hash="cancel-test",
        map_hash="cancel-test",
        song="cancel-test",
        map_id="cancel-test",
    )
    monkeypatch.setattr(training, "prepare", lambda *args: [item])
    monkeypatch.setattr(
        training,
        "configuration",
        lambda _: ModelConfig(
            width=64, heads=2, audio_layers=1, global_layers=1, decoder_layers=1, dropout=0
        ),
    )
    store.save("dataset", dict(id="cancel-dataset", maps=[]))
    settings = dict(
        dataset_id="cancel-dataset",
        name="Cancellation test",
        device="cpu",
        epochs=1,
        batch_size=1,
        gradient_accumulation=1,
        learning_rate=0.0003,
        seed=42,
    )

    class Context:
        def check(self):
            pass

        def report(self, *args, **kwargs):
            pass

    class StopAfterUpdate(Context):
        def report(self, progress, message, **kwargs):
            if "Batch" in message:
                raise Cancelled()

    before = {entry["id"] for entry in store.all_records("model")}
    with pytest.raises(Cancelled):
        training.train(settings, StopAfterUpdate())
    interrupted = next(entry for entry in store.all_records("model") if entry["id"] not in before)
    assert interrupted["training_status"] == "interrupted" and interrupted["continuable"]
    assert interrupted["completed_epochs"] == 0
    resumed = training.train({**settings, "resume_model_id": interrupted["id"]}, Context())
    assert store.get("model", resumed["model_id"])["completed_epochs"] == 1
    assert not (DATA / "models" / resumed["model_id"] / "recovery.json").exists()


def test_crash_recovery_registers_saved_checkpoint_once():
    identifier = "crash-recovery-test"
    directory = DATA / "models" / identifier
    directory.mkdir(exist_ok=True)
    # Recovery moves no tensors and doesn't deserialize code; publication preserves saved bytes.
    (directory / "model.safetensors").write_bytes(b"already-saved-weights")
    (directory / "training.safetensors").write_bytes(b"already-saved-optimizer")
    metadata = dict(
        id=identifier, architecture=ARCHITECTURE, event_schema=EVENT_SCHEMA, metrics=[], warnings=[]
    )
    (directory / "recovery.json").write_text(json.dumps(metadata), encoding="utf-8")
    recover_training_runs()
    saved = store.get("model", identifier)
    assert saved["continuable"] and saved["training_status"] == "interrupted"
    assert (directory / "model.zip").is_file() and not (directory / "recovery.json").exists()
    recover_training_runs()
    assert store.get("model", identifier)["warnings"] == saved["warnings"]
