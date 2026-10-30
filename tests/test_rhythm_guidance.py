import numpy as np
import pytest
from app.beatmaps import HitObject, parse, write
from app.timing import TimingPoint
from app.ml import style
from app.ml.audio import FRAME_MS
from app.ml.rhythm_guidance import RhythmGuide, density_rate, dynamics_factor, loudness_profile
from app.ml.sequence_generation import choose_next_time, placed_spinners
from app.ml.spinners import drop_spinners, fill_spinners
from app.ml.tempo_octave import octave_corrected

FPS = 1000 / FRAME_MS
SETTINGS = dict(
    title="Guide", artist="Test", version="Insane", ar=9, cs=4, od=8, hp=5,
    divisor=4, seed=1, rhythm_temperature=0.05,
)  # fmt: skip


def pulse_train(period_s, seconds=60, offbeat=0.0):
    envelope = np.zeros(int(seconds * FPS))
    for time in np.arange(0, seconds, period_s):
        envelope[int(time * FPS)] += 1
        if offbeat:
            index = int((time + period_s / 2) * FPS)
            if index < len(envelope):
                envelope[index] += offbeat
    return envelope


def test_half_tempo_from_the_beat_tracker_is_doubled_only_when_equally_periodic():
    even = pulse_train(0.3)  # 200 BPM, every pulse equally strong
    assert octave_corrected(even, FPS, 100.0, 186) == pytest.approx(200)
    assert octave_corrected(even, FPS, 200.0, 186) == pytest.approx(200)
    # A clear 100 BPM pulse with weak off-beats stays at 100 BPM.
    assert octave_corrected(pulse_train(0.6, offbeat=0.3), FPS, 100.0, 186) == pytest.approx(100)


def loud_and_quiet_mel(frames=3000, quiet=slice(1500, 3000), drop_db=12):
    mel = np.full((frames, 64), 0.6, np.float32)
    mel[quiet] -= drop_db / 40
    return mel


def test_quiet_sections_get_a_lower_target_density():
    mel = loud_and_quiet_mel()
    loudness = loudness_profile(mel)
    assert loudness[:1000].mean() == pytest.approx(0, abs=0.5)
    assert loudness[2000:].mean() == pytest.approx(-12, abs=0.5)
    factor = dynamics_factor(loudness)
    assert factor[:1000].mean() == pytest.approx(1, abs=0.02)
    assert factor[2000:].mean() == pytest.approx(0.25, abs=0.02)
    prediction = np.zeros((len(mel), 4), np.float32)
    prediction[:, 2] = np.log1p(4) / np.log(11)  # the density head says 4 objects/s
    rate = density_rate(prediction, loudness)
    assert rate[:1000].mean() == pytest.approx(4, rel=0.02)
    assert rate[2000:].mean() == pytest.approx(1, rel=0.05)
    assert density_rate(prediction, loudness, 0)[2000:].mean() == pytest.approx(4, rel=0.02)


def test_guide_costs_dense_gaps_only_where_the_music_is_sparse():
    rate = np.full(4000, 8.0, np.float32)
    rate[2000:] = 1.0
    guide = RhythmGuide(rate)
    loud, quiet = 1000.0, 2500 * FRAME_MS
    assert guide.cost(loud, loud + 125) == 0  # a stream where 8 objects/s are expected
    assert guide.cost(quiet, quiet + 125) > 4  # the same stream in a 1 object/s section
    assert guide.cost(quiet, quiet + 1000) == 0
    # Sustained overmapping is costed even when each single gap looks like a burst.
    guide.history = list(np.arange(quiet - 3000, quiet, 250))
    assert guide.cost(quiet, quiet + 500) > guide.cost(quiet, quiet + 1000) > 0
    # Long rests in dense sections cost a little, never a lot.
    assert 0 < RhythmGuide(rate).cost(loud, loud + 4000) <= 2


def stream_logits():
    logits = np.full(style.GAP_CLASSES, -6.0)
    logits[style.GAP_BEATS.index(0.25)] = 6
    logits[style.GAP_BEATS.index(1)] = 0
    return logits


def test_a_stream_locked_rhythm_head_rests_in_quiet_music_but_streams_in_dense_music():
    frames = 4000
    onset = np.full(frames, 0.6, np.float32)
    active = np.ones(frames, bool)
    timing = [TimingPoint(0.0, 500.0)]
    rng = np.random.default_rng(1)
    dense = RhythmGuide(np.full(frames, 8.0, np.float32))
    sparse = RhythmGuide(np.full(frames, 1.0, np.float32))
    circle = HitObject(100, 100, 1000.0, beat_length=500)
    end = frames * FRAME_MS
    _, gap = choose_next_time(stream_logits(), circle, timing, onset, active, end, SETTINGS, 0, rng, dense)
    assert style.GAP_BEATS[gap] == 0.25
    _, gap = choose_next_time(stream_logits(), circle, timing, onset, active, end, SETTINGS, 0, rng, sparse)
    assert gap == style.REST or style.GAP_BEATS[gap] >= 0.5
    # The soft stream limit ends a run without a hard maximum.
    limited = RhythmGuide(np.full(frames, 8.0, np.float32), stream_soft_limit=8, stream_cost=5)
    _, gap = choose_next_time(stream_logits(), circle, timing, onset, active, end, SETTINGS, 12, rng, limited)
    assert gap == style.REST or style.GAP_BEATS[gap] > 0.26


def test_reserved_spinner_blocks_heads_and_resumes_at_the_drop():
    frames = 4000
    onset = np.full(frames, 0.6, np.float32)
    active = np.ones(frames, bool)
    timing = [TimingPoint(0.0, 500.0)]
    rng = np.random.default_rng(2)
    spinner = (3000.0, 5000.0, 5500.0)
    end = frames * FRAME_MS
    for time in (1000.0, 2500.0, 2800.0):
        circle = HitObject(100, 100, time, beat_length=500)
        following, _ = choose_next_time(
            stream_logits(), circle, timing, onset, active, end, SETTINGS, 0, rng, None, spinner
        )
        assert not 3000 - 250 < following < 5500
    late = HitObject(100, 100, 2750.0, beat_length=500)
    following, gap = choose_next_time(
        np.full(style.GAP_CLASSES, -6.0), late, timing, onset, active, end, SETTINGS, 0, rng, None, spinner
    )
    assert following == pytest.approx(5500) and gap == style.REST
    objects = [HitObject(100, 100, 2500.0, beat_length=500), HitObject(100, 100, 5500.0, beat_length=500)]
    assert placed_spinners(objects, [spinner], active, timing, end, SETTINGS) == [(3000.0, 5000.0)]
    crowded = [HitObject(100, 100, 2900.0, beat_length=500), *objects[1:]]
    assert placed_spinners(crowded, [spinner], active, timing, end, SETTINGS) == []


def test_spinners_before_a_loudness_jump_and_in_audible_gaps_but_not_in_silence():
    frames = int(60000 / FRAME_MS)
    mel = loud_and_quiet_mel(frames, slice(0, int(30000 / FRAME_MS)), drop_db=10)
    loudness = loudness_profile(mel)
    active = np.ones(frames, bool)
    timing = [TimingPoint(0.0, 500.0)]
    spans = drop_spinners(loudness, active, timing, 0, 60000, SETTINGS)
    assert len(spans) == 1
    start, end, drop, lead = spans[0]
    assert drop == pytest.approx(30000, abs=600) and end < drop and end - start >= 1000
    assert lead == pytest.approx(250)
    objects = [HitObject(100, 100, time, beat_length=500) for time in (1000, 9000, 20000, 50000)]
    gaps = fill_spinners(objects, active, timing, 60000, SETTINGS)
    assert gaps and all(1000 <= left and right < 50000 for left, right in gaps[:-1])
    silent = active.copy()
    silent[int(9500 / FRAME_MS) : int(19500 / FRAME_MS)] = False
    assert not any(9000 < left < 20000 for left, _ in fill_spinners(objects, silent, timing, 60000, SETTINGS))
    assert not drop_spinners(loudness, active, timing, 0, 60000, {**SETTINGS, "spinners": False})


def test_spinners_roundtrip_through_the_osu_file(tmp_path):
    objects = [
        HitObject(100, 100, 1000.0, beat_length=500),
        HitObject(200, 100, 6000.0, beat_length=500),
    ]
    path = tmp_path / "map.osu"
    write(path, objects, SETTINGS, "audio.ogg", 120, 0, [TimingPoint(0, 500)], [(2000.0, 4500.0)])
    parsed = parse(path)
    assert parsed.ignored == [(2000.0, 4500.0)] and len(parsed.objects) == 2
    assert parsed.objects[1].new_combo
    text = path.read_text()
    assert text.index("256,192,2000,12,0,4500") < text.index("200,100,6000")


def test_calibration_widens_jumps_for_an_aim_deficit_instead_of_streaming():
    from app.ml.generation import calibration_step, trained_star_range

    profile = {"jump_spacing": 170.0, "aim_share": 0.55, "tempo": 186.0}
    settings = {"condition_stars": 6.2, "tempo_bias": 0.0, "style": {}}
    attributes = {"aim_difficulty": 1.0, "speed_difficulty": 1.8}
    star_range = trained_star_range({"star_profiles": {"1.0": {}, "8.0": {}}})
    step = calibration_step(settings, 6.2, 3.0, attributes, profile, star_range)
    assert step["tempo_bias"] <= 0
    assert step["calibration_style"]["jump_spacing"] > 170
    assert step["condition_stars"] <= 6.2 + 1.5
    speed = calibration_step(
        settings, 6.2, 5.0, {"aim_difficulty": 2.5, "speed_difficulty": 1.5}, profile, star_range
    )
    assert 0 < speed["tempo_bias"] <= 1 and "calibration_style" not in speed
    harder = calibration_step(settings, 4.0, 6.0, attributes, profile, star_range)
    assert harder["tempo_bias"] < 0 and harder["condition_stars"] < 6.2


def test_free_validation_reports_stream_lock_in():
    from app.ml.rollout_evaluation import longest_short_run

    stream = [HitObject(0, 0, 1000 + 125 * index, beat_length=500) for index in range(20)]
    jumps = [HitObject(0, 0, 1000 + 250 * index, beat_length=500) for index in range(20)]
    assert longest_short_run(stream) == 20 and longest_short_run(jumps) == 1


def test_calibration_respects_requested_aim_share_and_trained_range():
    from app.ml.generation import calibration_step

    profile = {"jump_spacing": 170.0, "aim_share": 0.55}
    attributes = {"aim_difficulty": 1.0, "speed_difficulty": 1.6}  # aim share 0.38
    speed_map = {"condition_stars": 5.0, "tempo_bias": 0.0, "style": {}, "aim_share": 0.4}
    step = calibration_step(speed_map, 5.0, 4.0, attributes, profile, (0.5, 12))
    assert step["tempo_bias"] > 0 and "calibration_style" not in step
    low = {"condition_stars": 3.25, "tempo_bias": 0.0, "style": {}}
    assert calibration_step(low, 1.0, 2.0, attributes, profile, (3.25, 6.75))["condition_stars"] == 3.25
    # Without a star profile there is no jump lever; the rhythm lever stays bounded.
    plain = calibration_step(low, 6.0, 3.0, attributes, None, (0.5, 12))
    assert 0 < plain["tempo_bias"] <= 1


def test_truncated_free_validation_compares_only_the_covered_span():
    from app.ml.rollout_evaluation import covered_ratio

    predicted = np.arange(0, 128) * 75.0  # a stream cut off by the object limit
    reference = np.arange(0, 24000, 150.0)
    assert covered_ratio(predicted, reference, True) == pytest.approx(2, rel=0.03)
    assert covered_ratio(predicted, reference, False) < 1


def test_model_tempo_head_gives_the_octave_prior():
    from app.ml.tempo_octave import model_tempo

    prediction = np.zeros((100, 4), np.float32)
    prediction[:, 3] = np.log(300 / 500)
    assert model_tempo(prediction) == pytest.approx(200)


def test_uncertain_direction_no_longer_rewards_stacked_notes():
    from app.ml.candidate_selection import movement_cost

    expected = np.array([150 / 512, 0.6, 0.05, 0.0, 0, 0, 0, 0])  # 150 px, direction unknown
    stack = np.zeros(8)
    jump = np.array([150 / 512, 0.6, -0.9, 0.4, 0, 0, 0, 0])
    assert movement_cost(jump, expected) < 0.1 < 1 < movement_cost(stack, expected)
    # An intended stack (the model expects ~0 px) stays cheap.
    assert movement_cost(stack, np.zeros(8)) == pytest.approx(0)


def test_unset_style_groups_are_filled_from_the_star_profile():
    from app.ml.generation import profile_style, styled_condition
    from app.ml import style as style_module

    profile = {"jump_spacing": 171.0, "spacing": 134.0, "slider_ratio": 0.36, "stream_share": 0.05,
               "burst_share": 0.18, "density": 4.5, "tempo": 186.0, "stars": 6.0}  # fmt: skip
    filled = profile_style(profile, {})
    assert filled["jump_spacing"] == 171 and filled["slider_ratio"] == 0.36 and "tempo" not in filled
    # A group the user touched stays entirely the user's.
    partial = profile_style(profile, {"stream_share": 0.3})
    assert "burst_share" not in partial and partial["jump_spacing"] == 171
    settings = dict(ar=9, cs=4, od=7, hp=5, condition_stars=6.0, style={"jump_spacing": 90.0},
                    profile_style=filled)  # fmt: skip
    values = styled_condition(settings, 186.0)
    index = style_module.INDEX["jump_spacing"]
    assert values[index] == pytest.approx(style_module.encode("jump_spacing", 90.0))
    assert values[style_module.FEATURE_COUNT + style_module.INDEX["slider_ratio"]] == 1


def test_combos_start_on_bars_after_rests_and_spinners():
    from app.ml.combos import assign_combos

    timing = [TimingPoint(0.0, 500.0)]
    objects = [HitObject(100, 100, float(time), beat_length=500) for time in range(0, 8000, 250)]
    objects += [HitObject(100, 100, 12000.0 + time, beat_length=500) for time in range(0, 2000, 500)]
    count = assign_combos(objects, timing, [(9000.0, 11000.0)])
    starts = [obj.time for obj in objects if obj.new_combo]
    assert starts[0] == 0 and 12000 in starts
    assert all(start % 2000 == 0 for start in starts)  # dense: one combo per 4-beat bar
    assert count == len(starts) >= 5
