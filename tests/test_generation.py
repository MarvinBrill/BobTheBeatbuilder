from app.timing import TimingPoint
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
    first = candidate_object(
        CURVE.reshape(-1), 1000, 2000, 500, True, 1, repeats, True, SETTINGS, []
    )[0]
    second = candidate_object(
        CURVE.reshape(-1), 3000, 4000, 400, True, 1, repeats, False, SETTINGS, [first]
    )[0]
    assert first.slider and second.slider
    path = tmp_path / "map.osu"
    write(
        path,
        [first, second],
        SETTINGS,
        "audio.ogg",
        120,
        0,
        [TimingPoint(0, 500), TimingPoint(2500, 400)],
    )
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
    triplet = candidate_object(
        CURVE.reshape(-1), 1000, 1300, 500, True, 1 / 3, 1, False, SETTINGS, []
    )[0]
    assert triplet.duration == pytest.approx(500 / 3)
    last = HitObject(40, 40, 900, end_x=40, end_y=40)
    circle, _, _, repair = candidate_object(
        np.ones(8) * 2, 1000, 1300, 500, False, 0, 1, False, SETTINGS, [last]
    )
    radius = circle_radius(SETTINGS["cs"])
    assert radius <= circle.x <= 512 - radius and radius <= circle.y <= 384 - radius
    assert repair > 0.1
    assert np.hypot(circle.x - last.end_x, circle.y - last.end_y) < 220


def clicks_mel(times, frames):
    """Mel frames with a broadband click at each time (ms), silence-free background."""
    mel = np.full((frames, 64), -0.6, np.float32)
    for time in times:
        frame = int(round(time / FRAME_MS))
        if frame < frames:
            mel[frame : frame + 3] = 0.6
    return mel


def test_detected_timing_finds_a_real_tempo_change_without_audio_fallback(monkeypatch):
    times = np.concatenate([np.arange(500, 60000, 500), np.arange(60000, 120000, 400)])
    frames = int(121000 / FRAME_MS)
    prediction = np.zeros((frames, 4), np.float32)
    for time in times:
        frame = round(time / FRAME_MS)
        prediction[frame, 1] = 1
        prediction[frame, 3] = np.log((500 if time < 60000 else 400) / 500)

    def unexpected(*_):
        raise AssertionError("Detected timing should not use the librosa fallback")

    monkeypatch.setattr("app.ml.rhythm_inference.tempo", unexpected)
    timing, source = timing_map(prediction, "unused", SETTINGS, clicks_mel(times, frames))
    assert source.startswith("detected") and len(timing) == 2
    assert timing[0].beat_length == pytest.approx(500, abs=0.5)
    assert timing[1].beat_length == pytest.approx(400, abs=0.5)
    assert timing[1].time == pytest.approx(60000, abs=400 * 4)


def test_constant_tempo_is_one_round_red_line_on_the_beat():
    from app.ml.tempo_grid import detect_timing

    beat = 60000 / 174.0
    times = np.arange(137, 150000, beat)
    frames = int(151000 / FRAME_MS)
    result = detect_timing(clicks_mel(times, frames), FRAME_MS, convention_ms=0)
    assert len(result.points) == 1 and 60000 / result.points[0].beat_length == pytest.approx(174.0)
    shift = ((result.points[0].time - 137) % beat + beat / 2) % beat - beat / 2
    assert abs(shift) < 20  # mel-flux latency is a few frames at most


def test_manual_tempo_and_snapping_do_not_create_notes_in_silence():
    prediction = np.zeros((500, 4), np.float32)
    prediction[20::20, 0] = 1
    mel = np.zeros((500, 64), np.float32)
    mel[100:300] = -1
    timing, source = timing_map(prediction, "unused", {**SETTINGS, "bpm": 120, "offset": 5000})
    assert source == "manual" and timing[0].time <= 0
    times = select_times(prediction, mel, SETTINGS, timing)
    assert times and all(not 100 <= round(time / FRAME_MS) < 300 for time in times)


def test_frame_quantization_does_not_accumulate_beat_drift():
    times = np.arange(500, 180001, 500)
    quantized = np.rint(times / FRAME_MS) * FRAME_MS
    anchor, beat = fit_beat_segment(quantized, np.median(np.diff(quantized)))
    assert abs(anchor + beat * (len(times) - 1) - times[-1]) < 2
    assert beat == pytest.approx(500, abs=0.02)


def test_styled_generation_places_patterns_on_the_beat_grid(monkeypatch):
    import torch
    from app.ml import style
    from app.ml.architecture import ModelConfig
    from app.ml.phrase_model import PhraseMapper
    from app.ml.sequence_generation import generate_objects_v2

    class Quiet:
        def check(self):
            pass

        def report(self, *args, **kwargs):
            pass

    torch.manual_seed(3)
    model = PhraseMapper(
        ModelConfig(
            width=64, heads=2, audio_layers=1, global_layers=1, decoder_layers=1,
            style_version=2, mapper_slots=2,
        )  # fmt: skip
    ).eval()
    mel = np.full((1400, 64), -0.2, np.float32)
    mel[600:700] = -1
    values, mask = style.vector({"ar": 9, "cs": 4, "od": 8, "hp": 5, "tempo": 120, "stars": 5})
    condition = torch.from_numpy(style.condition_input(values, mask))[None]
    settings = {
        **SETTINGS, "bpm": 120, "offset": 0, "divisor": 4, "seed": 4, "temperature": 0.8,
        "candidates": 4, "slider_bias": 0.5, "max_stream": 5,
    }  # fmt: skip
    objects, timing, report = generate_objects_v2(
        model, mel, "unused", settings, condition, torch.device("cpu"), Quiet(),
        mapper=torch.tensor([1.0]),
    )  # fmt: skip
    assert objects and report["rhythm"] == "learned-sequence" and timing[0].beat_length == 500
    offset = timing[0].time
    run = 0
    for index, obj in enumerate(objects):
        assert (obj.time - offset) / 125 == pytest.approx(round((obj.time - offset) / 125), abs=1e-6)
        assert not 600 <= round(obj.time / FRAME_MS) < 700
        if index + 1 < len(objects):
            gap = objects[index + 1].time - obj.time
            assert obj.time + obj.duration + 35 <= objects[index + 1].time + 1e-6
            run = run + 1 if gap <= 130 else 0
            assert run + 1 <= 5


def test_a_section_can_be_generated_again_between_kept_objects():
    import torch
    from app.beatmaps import HitObject
    from app.ml import style
    from app.ml.architecture import ModelConfig
    from app.ml.phrase_model import PhraseMapper
    from app.ml.sequence_generation import generate_objects_v2

    class Quiet:
        def check(self):
            pass

        def report(self, *args, **kwargs):
            pass

    torch.manual_seed(5)
    model = PhraseMapper(
        ModelConfig(
            width=64, heads=2, audio_layers=1, global_layers=1, decoder_layers=1,
            style_version=2, mapper_slots=2, turn_head=True,
        )  # fmt: skip
    ).eval()
    mel = np.full((1400, 64), -0.2, np.float32)
    values, mask = style.vector({"ar": 9, "cs": 4, "od": 8, "hp": 5, "tempo": 120, "stars": 5})
    condition = torch.from_numpy(style.condition_input(values, mask))[None]
    settings = {
        **SETTINGS, "bpm": 120, "offset": 0, "divisor": 4, "seed": 9, "temperature": 0.8,
        "candidates": 4, "slider_bias": 0.5,
    }  # fmt: skip
    before = [HitObject(100 + 60 * i, 150, 500.0 * (i + 1), end_x=100 + 60 * i, end_y=150, beat_length=500) for i in range(6)]
    after = [HitObject(300, 200, 12000.0, end_x=300, end_y=200, beat_length=500)]
    generated, timing, report = generate_objects_v2(
        model, mel, "unused", settings, condition, torch.device("cpu"), Quiet(),
        mapper=torch.tensor([1.0]), evaluation_range=(4000.0, 12000.0),
        initial_objects=before, hard_end=after[0].time,
    )  # fmt: skip
    assert generated and report["scope"] == "section"
    merged = before + generated + after
    assert all(4000 <= obj.time < 12000 for obj in generated)
    for left, right in zip(merged, merged[1:]):
        assert left.time + left.duration + 35 <= right.time + 1e-6
