import numpy as np
import pytest

from app.ml.audio import FRAME_MS
from app.ml.tempo_grid import detect_timing, grid_drift, onset_envelopes


def clicks(times, seconds, accents=()):
    """Mel frames with broadband clicks; `accents` (ms) get a stronger, bass-heavy click."""
    frames = int(seconds * 1000 / FRAME_MS)
    mel = np.full((frames, 64), -0.6, np.float32)
    accented = {int(round(time / FRAME_MS)) for time in accents}
    for time in times:
        frame = int(round(time / FRAME_MS))
        if frame + 3 >= frames:
            continue
        mel[frame : frame + 3] = 0.4
        if frame in accented:
            mel[frame : frame + 4, :6] = 0.9
    return mel


def wrapped(value, beat):
    return (value % beat + beat / 2) % beat - beat / 2


@pytest.mark.parametrize("bpm", [128.5, 174.3, 200.0, 245.0])
def test_constant_tempo_is_found_exactly_with_one_red_line(bpm):
    beat = 60000 / bpm
    times = np.arange(311, 150000, beat)
    result = detect_timing(clicks(times, 151), FRAME_MS, convention_ms=0)
    assert len(result.points) == 1
    assert 60000 / result.points[0].beat_length == pytest.approx(bpm, abs=1e-6)
    assert abs(wrapped(result.points[0].time - 311, beat)) < 20
    assert result.info["constant"] and result.info["confidence"] > 0.5


def test_a_dotted_passage_is_rhythm_not_a_tempo_change():
    beat = 300.0
    times = list(np.arange(100, 60000, beat))
    times += list(np.arange(100 + 200 * beat, 80000, 0.75 * beat))  # dotted, on the grid
    times += list(np.arange(100 + 267 * beat, 140000, beat))  # back on the same grid
    result = detect_timing(clicks(times, 141), FRAME_MS, convention_ms=0)
    assert len(result.points) == 1 and result.points[0].beat_length == pytest.approx(300, abs=0.01)


def test_the_model_tempo_head_decides_the_octave():
    beat = 60000 / 105.0
    times = np.arange(200, 120000, beat / 2)  # half-beat hats make 210 BPM plausible too
    mel = clicks(times, 121, accents=times[::2])  # the kick is on the beat
    prediction = np.zeros((len(mel), 4), np.float32)
    prediction[:, 3] = np.log(beat / 500)  # the model says 105 BPM...
    for time in times[::2]:
        prediction[int(round(time / FRAME_MS)), 1] = 0.9  # ...and where its beats are
    result = detect_timing(mel, FRAME_MS, prediction)
    assert 60000 / result.points[0].beat_length == pytest.approx(105, abs=1e-6)


def test_accented_bar_starts_become_the_bar_origin():
    beat = 60000 / 160
    times = np.arange(437, 120000, beat)
    accents = times[2::4]  # every bar starts on the third click
    result = detect_timing(clicks(times, 121, accents), FRAME_MS, convention_ms=0)
    point = result.points[0]
    assert point.meter == 4 and point.time <= 0
    assert abs(wrapped(point.time - accents[0], 4 * beat)) < 25


def test_grid_drift_shows_a_wrong_tempo():
    beat = 60000 / 175
    times = np.arange(0, 120000, beat)
    mel = clicks(times, 121)
    full, _, _ = onset_envelopes(mel)
    from app.timing import TimingPoint

    right = [phase for _, phase, lock, _ in grid_drift(full, FRAME_MS, [TimingPoint(0, beat)]) if lock > 0.1]
    wrong = [phase for _, phase, _, _ in grid_drift(full, FRAME_MS, [TimingPoint(0, 60000 / 177.5)])]
    assert np.ptp(right) < 10
    assert np.ptp(wrong) > 100


def test_silence_has_no_beat():
    with pytest.raises(ValueError):
        detect_timing(np.full((3000, 64), -1, np.float32), FRAME_MS)
