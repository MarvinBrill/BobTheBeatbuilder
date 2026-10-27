import numpy as np
import pytest
from app.beatmaps import Beatmap, HitObject
from app.ml.audio import FRAME_MS
from app.ml.sequence_data import pack_events, frame_labels, summary_features
from app.ml.sequence_batches import tasks_for, make_example
from app.ml.training_setup import dataset_signature, split_maps


def example_item(count=150):
    objects = [
        HitObject(
            100 + index % 4 * 80,
            100 + index % 3 * 60,
            100 + index * 60,
            end_x=100 + index % 4 * 80,
            end_y=100 + index % 3 * 60,
        )
        for index in range(count)
    ]
    beatmap = Beatmap(objects=objects, timing=[(0, 500, False)], ignored=[(9500, 10500)])
    mel = np.zeros((1200, 64), np.float32)
    events, _ = pack_events(beatmap, len(mel))
    return dict(
        events=events,
        mel=mel,
        summary=summary_features(mel),
        labels=frame_labels(beatmap, len(mel)),
        condition=np.full(8, 0.5, np.float32),
    )


def test_event_chunks_cover_every_target_once_and_do_not_duplicate_rhythm_loss():
    item = example_item()
    tasks = tasks_for([item])
    covered, weight = [], 0
    for _, central, indices, rhythm_weight in tasks:
        result = make_example(item, central, indices, rhythm_weight)
        covered.extend(result["event_times"][result["event_mask"] == 1].tolist())
        weight += result["frame_targets"][4].sum()
        assert result["event_valid"].sum() <= 128
    assert np.allclose(sorted(covered), item["events"][:, 0])
    assert weight == pytest.approx(item["labels"][:, 4].sum(), abs=0.001)
    assert item["labels"][round(10000 / FRAME_MS), 4] == 0


def test_teacher_forcing_is_shifted_and_mirroring_preserves_bos():
    item = example_item(12)
    assert np.all(item["events"][0, 15:31] == 0)
    assert np.allclose(item["events"][1, 15:23], item["events"][0, 1:9])
    for seed in range(8):
        batch = make_example(item, 0, np.arange(12), 1, np.random.default_rng(seed))
        assert not batch["previous"][0].any()
        assert np.isfinite(batch["motion_target"]).all()


def test_accelerated_batches_preserve_reference_data_order_augmentation_and_targets():
    import importlib.util
    from pathlib import Path
    from app.ml.sequence_batches import batches

    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmarks" / "reference_batches.py"
    spec = importlib.util.spec_from_file_location("reference_batches", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    items = [example_item(150), example_item(45)]
    expected = list(reference.batches(items, 2, np.random.default_rng(73), True))
    actual = list(batches(items, 8, np.random.default_rng(73), True))
    for key in expected[0]:
        left = np.concatenate([batch[key] for batch in expected])
        right = np.concatenate([batch[key] for batch in actual])
        assert left.shape == right.shape
        assert np.allclose(left, right, rtol=2e-6, atol=2e-6), key


def test_simultaneous_or_out_of_audio_objects_are_rejected():
    beatmap = Beatmap(objects=[HitObject(50, 50, 100), HitObject(50, 50, 100)])
    with pytest.raises(ValueError, match="Simultaneous"):
        pack_events(beatmap, 1000)
    beatmap.objects[1].time = 90000
    with pytest.raises(ValueError, match="outside audio"):
        pack_events(beatmap, 1000)


def test_resume_signature_detects_map_edits_and_splits_are_disjoint():
    maps = [dict(map_id=str(i), map_hash=str(i), audio_hash=str(i // 2), song=str(i // 2)) for i in range(30)]
    before = dataset_signature(maps)
    assert before == dataset_signature(list(reversed(maps)))
    maps[0]["map_hash"] = "changed coordinates with unchanged note count"
    assert before != dataset_signature(maps)
    train, val, test = split_maps(maps, 42)
    groups = [set(item["song"] for item in split) for split in (train, val, test)]
    assert (
        all(groups) and not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    )
