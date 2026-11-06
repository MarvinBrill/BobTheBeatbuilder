import numpy as np

from app.ml.audio import FRAME_MS
from app.ml.rhythm_guidance import RhythmGuide, sharpen_onsets
from app.ml.song_plan import Section, SongPlan
from app.ml.style import GAP_BEATS
from app.timing import TimingPoint


def plan(accents=(), repeat=False):
    sections = [
        Section(0, 30000, "drop", 0.9, "auto", None, 20),
        Section(30000, 60000, "drop", 0.9, "auto", 0 if repeat else None, 20),
    ]
    return SongPlan(
        [TimingPoint(0, 375.0)], sections, sorted(accents), [1.0] * len(accents), duration_ms=60000
    )


def guide(**kwargs):
    rate = np.full(int(60000 / FRAME_MS), 4.0, np.float32)
    return RhythmGuide(rate=rate, **kwargs)


def test_skipping_an_accent_costs_and_hitting_it_is_rewarded():
    rhythm = guide(plan=plan(accents=[10375.0]))
    skip = rhythm.cost(10000, 10750, gap_class=GAP_BEATS.index(1), beat=375)
    land = rhythm.cost(10000, 10375, gap_class=GAP_BEATS.index(1), beat=375)
    plain = guide().cost(10000, 10750, beat=375)
    assert skip > plain + 0.9 and land < skip


def test_the_same_gap_for_more_than_two_bars_costs_unless_the_music_insists():
    rhythm = guide(plan=plan())
    half = GAP_BEATS.index(0.5)
    rhythm.classes = [half] * 20  # ten beats of 1/2 already
    weak = rhythm.monotony_cost(half, evidence=0.1)
    strong = rhythm.monotony_cost(half, evidence=0.95)
    assert weak > 0.3 > strong and rhythm.monotony_cost(GAP_BEATS.index(1), 0.1) == 0


def test_a_repeated_section_prefers_the_earlier_rhythm():
    rhythm = guide(plan=plan(repeat=True))
    rhythm.history = [1000.0, 1375.0, 1562.5]
    matching = rhythm.repeat_bonus(31375.0, 375)
    other = rhythm.repeat_bonus(31450.0, 375)
    assert matching < 0 == other


def test_audio_sharpening_keeps_strong_beats_and_lowers_silence():
    frames = 400
    mel = np.full((frames, 64), -0.6, np.float32)
    mel[100:103] = 0.6
    learned = np.full(frames, 0.8, np.float32)  # the onset head fires everywhere
    sharp = sharpen_onsets(learned, mel, 0.4)
    assert sharp[101] > 0.7 and sharp[300] < 0.5
    assert np.array_equal(sharpen_onsets(learned, mel, 0), learned)
