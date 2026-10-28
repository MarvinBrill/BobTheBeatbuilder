from app.timing import TimingPoint
import numpy as np
import pytest
from app.ml.audio import FRAME_MS
from app.ml.structured_rhythm import (
    onset_candidates,
    select_sequence,
    beat_distance,
    interval_family,
    Onset,
    pulse_confidence,
)


SETTINGS = dict(density=2, bpm=120, snap=True, divisor=4)


def select(peaks, density=2, width=8, mel_level=0, offset=0):
    length = max(8_000, max(time for time, _ in peaks) + 1000)
    prediction = np.zeros((round(length / FRAME_MS), 4), np.float32)
    prediction[:, 2] = np.log1p(density) / np.log(11)
    for time, score in peaks:
        prediction[round(time / FRAME_MS), 0] = score
    mel = np.full((len(prediction), 64), mel_level, np.float32)
    timing = [TimingPoint(offset, 500)]
    candidates = onset_candidates(prediction, mel, {**SETTINGS, "density": density}, timing)
    return select_sequence(candidates, timing, width)


@pytest.mark.parametrize("width", [1, 8])
def test_weak_peaks_do_not_fill_density_and_digital_silence_stays_empty(width):
    peaks = [(3500, 0.9), (3750, 0.8), (4100, 0.02), (4500, 0.03), (4700, 0.04)]
    assert select(peaks, density=5, width=width) == [3500, 3750]
    assert select(peaks, density=5, width=width, mel_level=-1) == []
    assert select(peaks, density=5, width=width, mel_level=-0.85) == [3500, 3750]


def test_interval_word_survives_old_window_boundary_and_time_translation():
    word = [3500, 3750, 3875, 4000, 4250, 4375, 4500]
    selected = select([(time, 0.9) for time in word], density=5)
    shifted = select([(time + 1000, 0.9) for time in word], density=5, offset=1000)
    assert selected == word
    assert np.allclose(np.array(shifted) - 1000, selected)


def test_triplets_and_syncopation_are_not_pulled_to_straight_grid():
    times = [1000, 1000 + 500 / 3, 1000 + 1000 / 3, 1625, 1875]
    assert np.allclose(select([(time, 0.95) for time in times], density=5), times)
    assert interval_family(1 / 3) == 2
    assert interval_family(1 / 12) == 2
    assert interval_family(3 / 8) == 1
    assert beat_distance(
        [TimingPoint(0, 500), TimingPoint(2000, 400)], 1500, 2800
    ) == pytest.approx(3)


def test_uncertain_beats_do_not_invent_snap_or_downbeat():
    prediction = np.zeros((300, 4), np.float32)
    prediction[87, 0] = 0.9
    mel = np.zeros((300, 64), np.float32)
    candidate = onset_candidates(prediction, mel, {**SETTINGS, "bpm": None}, [TimingPoint(0, 500)])[
        0
    ]
    assert candidate.time == 87 * FRAME_MS
    assert candidate.confidence == 0


def test_sparse_target_is_soft_and_rests_have_no_fill_requirement():
    peaks = [(time, 0.8) for time in range(1000, 7000, 125)]
    sparse = select(peaks, density=0.5)
    dense = select(peaks, density=6)
    assert 0 < len(sparse) < len(dense)
    assert select([(1000, 0.95), (6000, 0.95)], density=8) == [1000, 6000]


def test_post_snap_collision_keeps_stronger_accent():
    prediction = np.zeros((400, 4), np.float32)
    prediction[85, 0], prediction[88, 0] = 0.4, 0.95
    candidates = onset_candidates(
        prediction, np.zeros((400, 64), np.float32), SETTINGS, [TimingPoint(0, 500)]
    )
    assert len(candidates) == 1
    assert candidates[0].score == pytest.approx(0.95)


def test_nearly_zero_local_density_does_not_forbid_an_isolated_accent():
    for density in (0, 0.05, 0.1, 0.2):
        onset = Onset(86, 1000, 0.99, density, 500, 1)
        assert select_sequence([onset], [TimingPoint(0, 500)]) == [1000]


def test_two_inconsistent_pulses_do_not_establish_timing_confidence():
    prediction = np.zeros((300, 4), np.float32)
    prediction[[85, 94], 1] = 0.95
    assert pulse_confidence(prediction, 86, 500, {"bpm": None}) == 0
