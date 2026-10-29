from dataclasses import asdict
import numpy as np
import pytest
import torch
from pydantic import ValidationError

from app.beatmaps import HitObject, parse, write
from app.timing import TimingPoint, beat_distance, timing_at
from app.schemas import GenerateRequest
from app.ml.musical_time import musical_time_features, disturbed_timing
from app.ml.rhythm_inference import timing_map
from app.ml.sequence_generation import decoder_input
from app.ml.sequence_batches import make_example
from app.ml.playability import candidate_object
from test_phrase_data import example_item


def test_meter_changes_survive_export_and_unknown_metrics_are_masked(tmp_path):
    timing = [
        TimingPoint(0, 500, meter=3, meter_confidence=1, bar_origin=0),
        TimingPoint(3000, 400, meter=5, meter_confidence=1, bar_origin=3000),
    ]
    path = tmp_path / "meters.osu"
    write(
        path,
        [HitObject(100, 100, 1000), HitObject(150, 100, 4000)],
        dict(title="Meter", artist="Test", version="Test", hp=5, cs=4, ar=8, od=6),
        "audio.ogg",
        120,
        0,
        timing,
    )
    parsed = parse(path)
    assert [point.meter for point in parsed.timing] == [3, 5]
    assert np.array_equal(
        musical_time_features(np.array([0, 1.5, 3, 5]), timing)[:, 9], [1, 1, 1, 1]
    )
    unknown = musical_time_features(np.array([0, 1.5]), [TimingPoint(0, 500)])
    assert not unknown[:, 5:10].any()
    assert (unknown[:, 4] == 1).all()


def test_continuous_beat_position_and_timing_change_do_not_drift():
    timing = [TimingPoint(500, 500), TimingPoint(2500, 400)]
    features = musical_time_features(np.array([0, 2.4999, 2.5, 182.5]), timing)
    assert not features[0].any()  # No extrapolated confident pickup before the first timing point.
    assert features[2, 0] * 256 == pytest.approx(4)
    assert features[3, 0] * 256 == pytest.approx(454)
    assert features[2, 0] - features[1, 0] < 0.00001
    assert features[2, 10] == 1 and features[3, 10] == 0
    assert beat_distance(timing, 2000, 3300) == pytest.approx(3)


def test_slider_before_first_positive_manual_timing_uses_unknown_technical_default(tmp_path):
    settings = GenerateRequest(
        model_id="m",
        audio_id="a",
        timing_points=[
            dict(time=1000, beat_length=400, meter=3, bar_origin=1000),
        ],
    ).dict()
    timing, _ = timing_map(np.zeros((100, 4), np.float32), "unused", settings)
    before = timing_at(timing, 500)
    assert before.beat_length == 500 and before.beat_confidence == 0
    assert before.meter is None and before.meter_confidence == 0
    assert not musical_time_features(np.array([0.5]), timing).any()
    assert timing_at(timing, 1000).beat_length == 400
    curve = (np.array([[100, 100], [150, 40], [220, 230], [280, 160]]) / [512, 384]).reshape(-1)
    for index, duration_beats in enumerate((0.8, 1.0)):
        obj = candidate_object(
            curve, 500, 1500, before.beat_length, True, duration_beats, 1, False, settings, []
        )[0]
        path = tmp_path / f"pickup-{index}.osu"
        write(path, [obj], settings, "audio.ogg", 150, 1000, timing)
        parsed = parse(path)
        assert parsed.objects[0].duration == pytest.approx(obj.duration, abs=0.01)
        assert parsed.objects[0].beat_length == 500
        assert [point.meter for point in parsed.timing if not point.inherited] == [3]


def test_manual_bpm_is_not_a_downbeat_but_complete_sections_can_be():
    prediction = np.zeros((100, 4), np.float32)
    bpm_only, _ = timing_map(prediction, "unused", dict(bpm=120, offset=100))
    assert bpm_only[0].meter is None and bpm_only[0].meter_confidence == 0
    settings = GenerateRequest(
        model_id="m",
        audio_id="a",
        timing_points=[
            dict(time=100, beat_length=500, meter=3, bar_origin=100),
            dict(time=3100, beat_length=400, meter=5, bar_origin=3100),
        ],
    ).dict()
    manual, source = timing_map(prediction, "unused", settings)
    assert source == "manual-sections" and manual[1].meter == 5
    assert musical_time_features(np.array([0.1, 3.1]), manual)[:, 9].tolist() == [1, 1]
    for points in (
        [dict(time=0, beat_length=float("nan"))],
        [dict(time=10, beat_length=500), dict(time=0, beat_length=500)],
        [dict(time=0, beat_length=500, meter=3, bar_origin=500)],
    ):
        with pytest.raises(ValidationError):
            GenerateRequest(model_id="m", audio_id="a", timing_points=points)


def test_train_and_inference_share_exact_temporal_queries():
    item = example_item(12)
    item["timing"] = [TimingPoint(0, 500, meter=3, meter_confidence=1, bar_origin=0)]
    batch = make_example(item, 0, np.arange(12), 1)
    times = (item["events"][:, 0] * 1000).tolist()
    objects = [
        HitObject(row[1] * 512, row[2] * 384, time) for row, time in zip(item["events"], times)
    ]
    duration = len(item["mel"]) * 256 / 22050
    _, queries, _ = decoder_input(objects, times, 11, duration, torch.device("cpu"), item["timing"])
    np.testing.assert_allclose(queries[0].numpy(), batch["queries"][:12], rtol=1e-5, atol=1e-6)


def test_styled_train_and_inference_share_exact_queries_without_the_next_gap():
    from test_style import styled_item

    item = styled_item()
    batch = make_example(item, 0, np.arange(12), 1)
    times = (item["events"][:, 0] * 1000).tolist()
    objects = [
        HitObject(row[1] * 512, row[2] * 384, time) for row, time in zip(item["events"], times)
    ]
    duration = len(item["mel"]) * 256 / 22050
    _, queries, _ = decoder_input(
        objects, times[:12], 11, duration, torch.device("cpu"), item["timing"], styled=True
    )
    np.testing.assert_allclose(queries[0].numpy(), batch["queries"][:12], rtol=1e-5, atol=1e-6)
    # Style-v2 queries never contain the time of the following object.
    _, later, _ = decoder_input(
        objects, times[:11] + [times[11] + 40], 10, duration, torch.device("cpu"), item["timing"],
        styled=True,
    )  # fmt: skip
    np.testing.assert_array_equal(queries[0, :11].numpy(), later[0].numpy())


def test_pre_drawn_timing_disturbance_matches_the_direct_draw():
    from app.ml.musical_time import draw_timing_disturbance, apply_timing_disturbance

    timing = [TimingPoint(0, 500, meter=3, meter_confidence=1, bar_origin=0)]
    for seed in range(30):
        direct = disturbed_timing(timing, np.random.default_rng(seed))
        drawn = draw_timing_disturbance(np.random.default_rng(seed))
        assert apply_timing_disturbance(timing, drawn) == direct
    assert apply_timing_disturbance(timing, None) == timing


def test_timing_disturbances_are_seeded_masked_and_leave_references_unchanged():
    timing = [TimingPoint(0, 500, meter=3, meter_confidence=1, bar_origin=0)]
    original = [asdict(point) for point in timing]
    outcomes = []
    for seed in range(20):
        left = disturbed_timing(timing, np.random.default_rng(seed))
        right = disturbed_timing(timing, np.random.default_rng(seed))
        assert left == right
        features = musical_time_features(np.array([0, 1, 2]), left)
        assert np.isfinite(features).all()
        if not left or left[0].meter_confidence == 0:
            assert not features[:, 5:10].any()
        outcomes.append((bool(left), bool(left and left[0].meter_confidence)))
    assert (False, False) in outcomes and (True, False) in outcomes and (True, True) in outcomes
    assert [asdict(point) for point in timing] == original
