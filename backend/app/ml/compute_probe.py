import numpy as np
import torch
from ..beatmaps import Beatmap, HitObject
from .architecture import ModelConfig
from .phrase_model import PhraseMapper
from .phrase_losses import parts, objective
from .sequence_data import pack_events, frame_labels, summary_features
from .sequence_batches import make_example
from .training_runtime import TrainingRuntime
from .cuda_graph import CudaTrainingGraph
from .style import FEATURE_COUNT


def probe_batch(styled=False):
    mel = np.zeros((1024, 64), np.float32)
    objects = [
        HitObject(
            100 + (index % 4) * 80,
            100 + (index % 3) * 70,
            500 + index * 500,
            end_x=100 + (index % 4) * 80,
            end_y=100 + (index % 3) * 70,
        )
        for index in range(12)
    ]
    objects[3] = HitObject(
        200, 150, 2000, True, 200, 300, 180, True, [[200, 150], [220, 100], [270, 220], [300, 180]], 1, 500
    )
    beatmap = Beatmap(objects=objects, timing=[(0, 500, False)])
    events, _ = pack_events(beatmap, len(mel))
    item = {
        "mel": mel,
        "summary": summary_features(mel),
        "events": events,
        "labels": frame_labels(beatmap, len(mel)),
        "condition": np.full(8, 0.5, np.float32),
    }
    if styled:
        item.update(
            style=np.full(FEATURE_COUNT, 0.4, np.float32),
            style_mask=np.ones(FEATURE_COUNT, np.float32),
            mapper=1,
        )
    values = make_example(item, 0, np.arange(len(events)), 1)
    return {key: torch.from_numpy(value[None].copy()) for key, value in values.items()}


def check_compute(selected):
    model = PhraseMapper(
        ModelConfig(
            width=64, heads=2, audio_layers=1, global_layers=1, decoder_layers=1,
            style_version=2, mapper_slots=2,
        )  # fmt: skip
    ).to(selected)
    batch = {key: value.to(selected) for key, value in probe_batch(styled=True).items()}
    runtime = TrainingRuntime(selected)
    optimizer = runtime.optimizer(model, 0.0003, 0.01)
    if runtime.cuda:
        captured = CudaTrainingGraph(model, runtime, batch, 20)
        loss, _ = captured.replay(batch)
    else:
        loss = objective(parts(model, runtime.forward(model, batch), batch))
        loss.backward()
    runtime.clip_and_check(model, torch.isfinite(loss.detach()))
    optimizer.step()
