import numpy as np
import pytest
from app.beatmaps import HitObject, parse, write
from app.ml.playability import candidate_object, circle_radius
from app.ml.rhythm_inference import timing_map, select_times, fit_beat_segment
from app.ml.audio import FRAME_MS

SETTINGS = dict(
    title="Curves",
    artist="Test",
    version="Hard",
    ar=9,
    cs=4,
    od=7,
    hp=5,
    density=4,
    snap=True,
    divisor=3,
    bpm=None,
    offset=0,
)
CURVE = np.array([[100, 100], [150, 40], [220, 230], [280, 160]]) / [512, 384]


@pytest.mark.parametrize("repeats", [1, 2, 3, 4])
def test_curved_repeated_sliders_roundtrip_across_tempo_changes(tmp_path, repeats):
    first = candidate_object(CURVE.reshape(-1), 1000, 2000, 500, True, 1, repeats, True, SETTINGS, [])[0]
    second = candidate_object(CURVE.reshape(-1), 3000, 4000, 400, True, 1, repeats, False, SETTINGS, [first])[
        0
    ]
    assert first.slider and second.slider
    path = tmp_path / "map.osu"
    write(path, [first, second], SETTINGS, "audio.ogg", 120, 0, [(0, 500), (2500, 400)])
    parsed = parse(path).objects
    for actual, expected in zip(parsed, [first, second]):
        assert actual.repeats == repeats and actual.new_combo == expected.new_combo
        assert abs(actual.duration - expected.duration) < 0.02
        assert np.allclose([actual.end_x, actual.end_y], [expected.end_x, expected.end_y], atol=0.5)
        assert actual.geometry_valid


def test_playability_preserves_triplets_bounds_and_time_to_next_object():
    obj, _, _, repair = candidate_object(
        CURVE.reshape(-1), 1000, 1200, 500, True, 1 / 3, 1, False, SETTINGS, []
    )
    assert obj.time + obj.duration <= 1200 - 35 + 1e-6
    triplet = candidate_object(CURVE.reshape(-1), 1000, 1300, 500, True, 1 / 3, 1, False, SETTINGS, [])[0]
    assert triplet.duration == pytest.approx(500 / 3)
    last = HitObject(40, 40, 900, end_x=40, end_y=40)
    circle, _, _, repair = candidate_object(
        np.ones(8) * 2, 1000, 1300, 500, False, 0, 1, False, SETTINGS, [last]
    )
    radius = circle_radius(SETTINGS["cs"])
    assert radius <= circle.x <= 512 - radius and radius <= circle.y <= 384 - radius
    assert repair > 0.1
    assert np.hypot(circle.x - last.end_x, circle.y - last.end_y) < 220


def test_learned_timing_recognizes_tempo_change_without_audio_fallback(monkeypatch):
    times = np.concatenate([np.arange(500, 10500, 500), np.arange(10400, 18400, 400)])
    prediction = np.zeros((1700, 4), np.float32)
    for time in times:
        frame = round(time / FRAME_MS)
        prediction[frame, 1] = 1
        prediction[frame, 3] = np.log((500 if time < 10000 else 400) / 500)

    def unexpected(_):
        raise AssertionError("Confident beat predictions should not use fallback")

    monkeypatch.setattr("app.ml.rhythm_inference.tempo", unexpected)
    timing, source = timing_map(prediction, "unused", SETTINGS)
    assert source == "learned" and len(timing) == 2
    assert timing[0][1] == pytest.approx(500, abs=15)
    assert timing[1][1] == pytest.approx(400, abs=15)


def test_manual_tempo_and_snapping_do_not_create_notes_in_silence():
    prediction = np.zeros((500, 4), np.float32)
    prediction[20::20, 0] = 1
    mel = np.zeros((500, 64), np.float32)
    mel[100:300] = -1
    timing, source = timing_map(prediction, "unused", {**SETTINGS, "bpm": 120, "offset": 5000})
    assert source == "manual" and timing[0][0] <= 0
    times = select_times(prediction, mel, SETTINGS, timing)
    assert times and all(not 100 <= round(time / FRAME_MS) < 300 for time in times)


def test_frame_quantization_does_not_accumulate_beat_drift():
    times = np.arange(500, 180001, 500)
    quantized = np.rint(times / FRAME_MS) * FRAME_MS
    anchor, beat = fit_beat_segment(quantized, np.median(np.diff(quantized)))
    assert abs(anchor + beat * (len(times) - 1) - times[-1]) < 2
    assert beat == pytest.approx(500, abs=0.02)
