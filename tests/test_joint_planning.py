from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import torch

from app.beatmaps import HitObject, parse, write
from app.ml.architecture import DURATION_BEATS
from app.ml.audio import FRAME_MS
from app.ml.candidate_selection import rank_candidates, structure_score
from app.ml.generation_trace import GenerationTrace
from app.ml.joint_planning import skip_option
from app.ml.playability import candidate_object
from app.ml.sequence_generation import generate_objects
from test_generation_trace import SETTINGS, CURVE, decoder_output, fake_model


def test_joint_candidates_compare_both_types_duration_and_repeat_with_fixed_budget():
    trace = GenerationTrace()
    ranked = rank_candidates(
        fake_model(),
        decoder_output(),
        1000,
        2500,
        500,
        SETTINGS,
        [],
        np.random.default_rng(42),
        torch.device("cpu"),
        trace,
    )
    assert len(ranked) <= SETTINGS["candidates"]
    assert {candidate.obj.slider for candidate in ranked} == {False, True}
    assert len({candidate.obj.repeats for candidate in ranked if candidate.obj.slider}) == 2
    scores = [event for event in trace.events if event["stage"] == "candidate_score"]
    assert all(event["structure_probabilities_used_in_score"] for event in scores)
    assert all(
        "kind_log_probability" in event and "repeat_log_probability" in event for event in scores
    )
    assert all(candidate.obj.duration <= 1465 for candidate in ranked)


def test_skip_evidence_protects_independent_accents_and_silence():
    times = [1000, 1250, 2000]
    prediction = np.zeros((300, 4), np.float32)
    prediction[[round(time / FRAME_MS) for time in times], 0] = [0.9, 0.2, 0.8]
    mel = np.zeros((300, 64), np.float32)
    assert skip_option(times, 0, 500, prediction, mel) is not None
    prediction[round(1250 / FRAME_MS), 0] = 0.65
    assert skip_option(times, 0, 500, prediction, mel) is None
    prediction[round(1250 / FRAME_MS), 0] = 0.2
    mel[100:160] = -1
    assert skip_option(times, 0, 500, prediction, mel) is None
    assert skip_option(times, 0, 400, prediction, np.zeros_like(mel)) is None


def test_joint_plan_keeps_supported_slider_and_removes_only_weak_interior_head(
    monkeypatch, tmp_path
):
    times = [1000, 1250, 2000, 3000]
    prediction = np.zeros((400, 4), np.float32)
    prediction[[round(time / FRAME_MS) for time in times], 0] = [0.9, 0.2, 0.8, 0.9]
    monkeypatch.setattr("app.ml.sequence_generation.predict_rhythm", lambda *args: prediction)
    monkeypatch.setattr(
        "app.ml.sequence_generation.select_times", lambda *args, **kwargs: times.copy()
    )
    monkeypatch.setattr(
        "app.ml.sequence_generation.encode_window",
        lambda *args: (None, None, None, torch.zeros((1, 128, 7))),
    )
    output = decoder_output()
    output["duration"].fill_(-20)
    output["duration"][..., DURATION_BEATS.index(1)] = 20
    output["repeats"] = torch.tensor([[[-20.0, 20.0, -20.0, -20.0]]])
    model = fake_model()
    observed_queries = []

    def decode(previous, queries, *args):
        observed_queries.append((previous.clone(), queries.clone()))
        return output

    model.decode = decode
    settings = {**SETTINGS, "slider_bias": 1}
    trace = GenerationTrace()
    objects, timing, report = generate_objects(
        model,
        np.zeros((400, 64), np.float32),
        "unused",
        settings,
        torch.zeros((1, 8)),
        torch.device("cpu"),
        SimpleNamespace(report=lambda *args: None),
        trace=trace,
    )
    assert [obj.time for obj in objects] == [1000, 2000, 3000]
    assert objects[0].duration == 500 and objects[0].repeats == 2
    assert report["skipped_interior_heads"] == 1
    assert not report["truncated"]
    assert objects[0].end_x == objects[0].x and objects[0].end_y == objects[0].y
    assert not torch.equal(observed_queries[0][1], observed_queries[1][1])
    path = tmp_path / "joint.osu"
    write(path, objects, settings, "audio.ogg", 120, 0, timing)
    parsed = parse(path).objects
    assert len(parsed) == len(objects)
    for actual, expected in zip(parsed, objects):
        assert actual.slider == expected.slider and actual.repeats == expected.repeats
        assert abs(actual.duration - expected.duration) < 0.1
        assert np.allclose(
            [actual.end_x, actual.end_y], [expected.end_x, expected.end_y], atol=0.75
        )


def test_movement_policy_is_independent_of_hit_window_od():
    history = [HitObject(50, 50, 900, end_x=50, end_y=50)]
    low = candidate_object(
        (CURVE / [512, 384]).reshape(-1),
        1000,
        2000,
        500,
        False,
        0,
        1,
        False,
        {**SETTINGS, "od": 0},
        history,
    )[0]
    high = candidate_object(
        (CURVE / [512, 384]).reshape(-1),
        1000,
        2000,
        500,
        False,
        0,
        1,
        False,
        {**SETTINGS, "od": 10},
        history,
    )[0]
    assert asdict(low) == asdict(high)


def test_uninformative_duration_and_repeats_do_not_introduce_circle_type_prior():
    kind = np.array([0.4, 0.6])
    duration = np.full(len(DURATION_BEATS), 1 / len(DURATION_BEATS))
    repeats = np.full(4, 0.25)
    circle = structure_score((False, 0, 0), kind, duration, repeats)[0]
    slider = structure_score((True, 0, 0), kind, duration, repeats)[0]
    assert slider > circle
    assert np.isclose(slider, 0.25 * np.log(0.6))
    assert np.isclose(circle, 0.25 * np.log(0.4))
