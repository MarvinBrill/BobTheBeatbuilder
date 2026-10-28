import numpy as np
import pytest
from app.beatmaps import HitObject
from app.timing import TimingPoint
from app.ml import rollout_evaluation as rollout
from app.ml.audio import FRAME_MS
from app.ml.rollout_sections import section_plan


class Context:
    def check(self):
        pass


class Model:
    def eval(self):
        pass


def test_sections_cover_late_song_and_window_transition_without_claiming_chorus():
    mel = np.zeros((9000, 64), np.float32)
    mel[5000:6000] = -1
    first = section_plan(mel)
    assert first == section_plan(mel.copy())
    probes = {probe for section in first for probe in section["probes"]}
    assert probes == {
        "opening",
        "middle",
        "ending",
        "low_activity",
        "high_activity",
        "window_transition",
    }
    assert max(section["end_ms"] for section in first) == pytest.approx(8999 * FRAME_MS)
    assert all(0 <= section["start_ms"] < section["end_ms"] <= 8999 * FRAME_MS for section in first)
    assert len(section_plan(np.zeros((100, 64), np.float32))) == 1


def test_sparse_generation_does_not_shorten_reference_horizon(monkeypatch):
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs)
        return (
            [HitObject(100, 100, 1000, end_x=100, end_y=100)],
            [TimingPoint(0, 500)],
            {"truncated": True},
        )

    monkeypatch.setattr(rollout, "generate_objects", generate)
    events = np.zeros((3, 47), np.float32)
    events[:, 0] = [1, 5, 8]
    item = dict(
        condition=np.full(8, 0.5, np.float32),
        mel=np.zeros((1000, 64)),
        audio_path="unused",
        events=events,
        map_id="one",
        song="one",
    )
    result = rollout.evaluate_rollout(Model(), item, "cpu", Context(), max_objects=1)
    assert result["reference_objects"] == 3 and result["reference_onset_f1"] == 0.5
    assert result["completed_full_song"] is False
    assert result["scope"] == "song"
    assert calls[0]["evaluation_range"] == (0, 999 * FRAME_MS)
    assert result["roundtrip_objects"] == 1


def test_empty_rest_is_valid_and_reference_scope_is_shared(monkeypatch):
    monkeypatch.setattr(
        rollout,
        "generate_objects",
        lambda *args, **kwargs: ([], [TimingPoint(0, 500)], {"truncated": False}),
    )
    events = np.zeros((1, 47), np.float32)
    events[0, 0] = 1
    item = dict(
        condition=np.full(8, 0.5, np.float32),
        mel=np.zeros((1000, 64)),
        audio_path="unused",
        events=events,
        map_id="one",
        song="one",
    )
    result = rollout.evaluate_rollout(
        Model(),
        item,
        "cpu",
        Context(),
        section=dict(start_ms=5000, end_ms=8000, probes=["low_activity"]),
    )
    assert result["reference_objects"] == result["generated_objects"] == 0
    assert result["completed_full_song"] is False
    assert result["scope"] == "section"


def test_full_song_evaluation_is_bounded_and_explicit(monkeypatch):
    item = dict(map_id="one", song="one", mel=np.zeros((30000, 64)))
    monkeypatch.setattr(
        rollout,
        "evaluate_rollout",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("test failure")),
    )
    result = rollout.evaluate_suite(Model(), [item], "cpu", Context(), include_full_song=True)
    assert result["full_song_skip_reason"] == "No selected song within 300 seconds"
    assert result["attempted"] == len(section_plan(item["mel"]))
