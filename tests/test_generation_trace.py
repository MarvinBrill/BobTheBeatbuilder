import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.beatmaps import HitObject, write
from app.config import DATA
from app.timing import TimingPoint
from app.ml.architecture import ARCHITECTURE, FORMAT_VERSION, EVENT_SCHEMA, TARGET_SCHEMA
from app.ml.architecture import DURATION_BEATS
from app.ml.generation_trace import GenerationTrace
from app.ml.playability import candidate_object
from app.ml.sequence_generation import choose_candidate, generate_objects


SETTINGS = dict(
    cs=4,
    ar=9,
    od=7,
    hp=5,
    slider_bias=0.7,
    temperature=0.8,
    candidates=12,
    title="Trace test",
    artist="Test",
    version="Test",
    seed=42,
    density=4,
    bpm=120,
    offset=0,
    snap=True,
    divisor=4,
)
CURVE = np.array([[100, 150], [150, 80], [230, 220], [300, 150]], dtype=float)


def decoder_output():
    raw = np.zeros((2, 17), np.float32)
    probability = ((CURVE / [512, 384]).reshape(-1) + 0.5) / 2
    raw[:, 1:9] = np.log(probability / (1 - probability))
    raw[:, 9:] = -2
    duration = torch.full((1, 1, len(DURATION_BEATS)), -10.0)
    duration[..., DURATION_BEATS.index(2)] = 10
    return {
        "geometry": torch.tensor(raw.reshape(1, 1, -1)),
        "kind": torch.tensor([[[0.0, 1.0]]]),
        "duration": duration,
        "repeats": torch.tensor([[[1.0, 2.0, 0.0, -1.0]]]),
        "combo": torch.zeros(1, 1),
        "motion": torch.zeros(1, 1, 8),
        "hidden": torch.zeros(1, 1, 4),
    }


def fake_model():
    return SimpleNamespace(
        config=SimpleNamespace(mixtures=2),
        compatibility=lambda hidden, geometry, motion: torch.zeros(len(geometry)),
        decode=lambda *args: decoder_output(),
    )


@pytest.mark.parametrize("seed", [42, 73])
def test_candidate_trace_preserves_objects_rng_and_export_bytes(tmp_path, seed):
    objects, states = [], []
    trace = GenerationTrace()
    for observer in (None, trace):
        history = []
        rng = np.random.default_rng(seed)
        for time in (1000, 1600, 2300, 3200):
            obj, _ = choose_candidate(
                fake_model(),
                decoder_output(),
                time,
                time + 600,
                500,
                SETTINGS,
                history,
                rng,
                torch.device("cpu"),
                trace=observer,
            )
            history.append(obj)
        objects.append(history)
        states.append(rng.bit_generator.state)
    assert [asdict(obj) for obj in objects[0]] == [asdict(obj) for obj in objects[1]]
    assert states[0] == states[1]
    for index, history in enumerate(objects):
        write(tmp_path / f"{index}.osu", history, SETTINGS, "audio.ogg", 120, 0)
    assert (tmp_path / "0.osu").read_bytes() == (tmp_path / "1.osu").read_bytes()
    assert {event["stage"] for event in trace.events} >= {
        "candidate_proposal",
        "candidate_final",
        "candidate_score",
        "candidate_rank",
        "object_choice",
    }
    json.dumps(trace.document(), allow_nan=False)


def test_trace_explains_duration_shortening_and_fallback_without_mutating_evidence():
    trace = GenerationTrace()
    geometry = (CURVE / [512, 384]).reshape(-1)
    candidate_object(geometry, 1000, 2000, 500, True, 2, 2, False, SETTINGS, [], trace=trace)
    shortened = next(
        event for event in trace.events if event.get("reason") == "next_head_duration_limit"
    )
    assert shortened["requested_ms"] == 1000 and shortened["final_ms"] == 750
    proposal = trace.events[0]["geometry_pixels"]
    geometry[:] = 0
    assert np.array_equal(proposal, CURVE)
    candidate_object(geometry, 3000, 3100, 500, True, 2, 1, False, SETTINGS, [], trace=trace)
    assert any(event.get("reason") == "short_path" for event in trace.events)


def test_generation_excerpt_keeps_real_next_head_and_reports_empty_rest(monkeypatch):
    monkeypatch.setattr(
        "app.ml.sequence_generation.predict_rhythm", lambda *args: np.zeros((500, 4))
    )
    monkeypatch.setattr(
        "app.ml.sequence_generation.select_times", lambda *args, **kwargs: [1000, 1500, 2000, 2500]
    )
    monkeypatch.setattr(
        "app.ml.sequence_generation.encode_window",
        lambda *args: (None, None, None, torch.zeros((1, 128, 7))),
    )
    context = SimpleNamespace(report=lambda *args: None)
    arguments = (
        fake_model(),
        np.zeros((500, 64), np.float32),
        "unused",
        SETTINGS,
        torch.zeros((1, 8)),
        torch.device("cpu"),
        context,
    )
    trace = GenerationTrace()
    objects, _, report = generate_objects(
        *arguments, max_objects=1, trace=trace, evaluation_range=(1000, 2200)
    )
    assert len(objects) == 1 and report["truncated"] and report["selected_count_before_limit"] == 3
    proposals = [event for event in trace.events if event["stage"] == "candidate_proposal"]
    assert all(event["next_time"] == 1500 for event in proposals)
    empty, _, report = generate_objects(*arguments, evaluation_range=(3000, 4000))
    assert empty == [] and report["objects"] == report["temporal_overlaps"] == 0
    assert report["mean_geometry_repair"] == 0 and not report["truncated"]


def test_trace_replaces_nonfinite_diagnostics_with_json_null():
    trace = GenerationTrace()
    trace.record("diagnostic", value=np.array([np.inf, np.nan, 1]))
    assert trace.events[0]["value"] == [None, None, 1]
    json.dumps(trace.document(), allow_nan=False)


def test_lookahead_trace_preserves_entire_generated_sequence(monkeypatch):
    monkeypatch.setattr(
        "app.ml.sequence_generation.predict_rhythm", lambda *args: np.zeros((500, 4))
    )
    monkeypatch.setattr(
        "app.ml.sequence_generation.select_times", lambda *args, **kwargs: [1000, 1500, 2000, 2500]
    )
    monkeypatch.setattr(
        "app.ml.sequence_generation.encode_window",
        lambda *args: (None, None, None, torch.zeros((1, 128, 7))),
    )
    arguments = (
        fake_model(),
        np.zeros((500, 64), np.float32),
        "unused",
        {**SETTINGS, "lookahead": True},
        torch.zeros((1, 8)),
        torch.device("cpu"),
        SimpleNamespace(report=lambda *args: None),
    )
    untraced, _, _ = generate_objects(*arguments)
    trace = GenerationTrace()
    traced, _, _ = generate_objects(*arguments, trace=trace)
    assert [asdict(obj) for obj in untraced] == [asdict(obj) for obj in traced]
    assert any(event["stage"] == "continuation_score" for event in trace.events)


def test_generation_persists_requested_trace_outside_export(monkeypatch):
    import zipfile
    from app.ml import generation

    model = SimpleNamespace(load_state_dict=lambda weights: None, eval=lambda: None, styled=False)
    model.to = lambda device: model
    monkeypatch.setattr(generation, "create_model", lambda *args: model)
    monkeypatch.setattr(generation, "load_file", lambda path: {})
    monkeypatch.setattr(generation, "features", lambda path: np.zeros((500, 64), np.float32))
    monkeypatch.setattr(
        generation, "export_audio", lambda source, target: target.write_bytes(b"audio")
    )
    monkeypatch.setattr(
        generation.store,
        "get",
        lambda kind, identifier: (
            {
                "id": identifier,
                "architecture": ARCHITECTURE,
                "format_version": FORMAT_VERSION,
                "event_schema": EVENT_SCHEMA,
                "target_schema": TARGET_SCHEMA,
            }
            if kind == "model"
            else {"path": "unused.wav"}
        ),
    )
    monkeypatch.setattr(generation.store, "save", lambda *args: None)
    monkeypatch.setattr(
        generation,
        "generate_objects",
        lambda *args, **kwargs: (
            [HitObject(100, 100, 1000, end_x=100, end_y=100)],
            [TimingPoint(0, 500)],
            {},
        ),
    )
    result = generation.generate(
        {**SETTINGS, "model_id": "model", "audio_id": "audio", "device": "cpu", "trace": True},
        SimpleNamespace(report=lambda *args: None, check=lambda: None),
    )
    directory = DATA / "outputs" / result["output_id"]
    saved = json.loads((directory / "trace.json").read_text(encoding="utf-8"))
    assert saved["schema"] == "generation-trace-v1"
    assert {event["stage"] for event in saved["events"]} == {
        "generation_settings",
        "export_roundtrip",
    }
    with zipfile.ZipFile(directory / "beatmap.osz") as archive:
        assert set(archive.namelist()) == {"audio.ogg", "map.osu"}
