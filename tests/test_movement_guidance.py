import math

import numpy as np
import pytest

from app.beatmaps import HitObject
from app.ml.melodic_onsets import combined_onsets, mel_band_centres, melodic_onsets
from app.ml.movement_guidance import (
    MovementGuide,
    build_movement_guide,
    min_slider_length,
    style_targets,
    velocity_band,
)
from app.ml.playability import candidate_object

SCALE = np.array([512, 384])
# Typical 6.5-star profile of the training maps (UI units).
PROFILE = {
    "spacing": 127.7, "jump_spacing": 170.0, "stream_spacing": 29.1, "sharp_angles": 0.47,
    "linear_angles": 0.23, "slider_ratio": 0.35, "slider_velocity": 190.0,
}  # fmt: skip
SETTINGS = dict(cs=4, ar=9, od=8, hp=5, seed=3, profile_style=PROFILE)


def circle(x, y, time, beat=300):
    return HitObject(x, y, time, end_x=x, end_y=y, beat_length=beat)


def test_user_style_overrides_calibration_and_profile():
    settings = {
        "profile_style": {"jump_spacing": 100, "slider_ratio": 0.3},
        "calibration_style": {"jump_spacing": 140},
        "style": {"jump_spacing": 180},
    }
    assert style_targets(settings) == {"jump_spacing": 180, "slider_ratio": 0.3}
    assert style_targets({"calibration_style": None}) == {}


def test_spacing_collapse_costs_more_than_a_requested_jump():
    guide = build_movement_guide(SETTINGS)
    history = [circle(100, 100, 1000), circle(250, 100, 1150)]
    stacked = circle(262, 104, 1300)
    jump = circle(256 + 150, 180, 1300)
    assert guide.cost(stacked, history, 300) > 2 > 0.5 > guide.cost(jump, history, 300)
    # 1/4 gaps follow the stream spacing, and stacks there stay possible.
    stream = circle(270, 100, 1225)
    assert guide.cost(stream, history, 300) < 0.5


def test_quiet_sections_ask_for_smaller_spacing():
    loudness = np.zeros(1000, np.float32)
    loudness[500:] = -12
    guide = build_movement_guide(SETTINGS, loudness, 10.0)
    previous = circle(100, 100, 1000)
    loud = guide.target_spacing(1150, previous, 300)
    quiet = guide.target_spacing(6150, circle(100, 100, 6000), 300)
    assert loud == pytest.approx(170) and quiet == pytest.approx(0.6 * 170, rel=0.2)


def test_jump_proposals_keep_the_distance_and_stay_on_the_playfield():
    guide = build_movement_guide(SETTINGS)
    history = [circle(300, 192, 1000), circle(470, 192, 1150)]  # moving right, at the edge
    radius = 54.4 - 4.48 * 4
    for _ in range(20):
        for head in guide.jump_heads(1300, history, 300):
            assert np.all(head >= radius) and np.all(head <= SCALE - radius)
            assert 90 < math.dist(head, (470, 192)) < 260
    # The same seed gives the same proposals; the model's own RNG is not touched.
    first = build_movement_guide(SETTINGS).jump_heads(1300, history, 300)
    second = build_movement_guide(SETTINGS).jump_heads(1300, history, 300)
    assert np.allclose(first, second)


def test_orbiting_the_playfield_edge_is_discouraged():
    guide = MovementGuide(radius=36.5)
    angles = np.linspace(0, 1.6 * math.pi, 10)
    ring = [circle(256 + 170 * math.cos(a), 192 + 150 * math.sin(a), 1000 + 300 * i)
            for i, a in enumerate(angles)]  # fmt: skip
    assert guide.orbit_cost(ring[-1].x, ring[-1].y, ring[:-1]) > 0.5
    zigzag = [circle(150 + 200 * (i % 2), 192 + 20 * i - 90, 1000 + 300 * i) for i in range(10)]
    assert guide.orbit_cost(zigzag[-1].x, zigzag[-1].y, zigzag[:-1]) == 0
    assert guide.edge_cost(40, 192) > guide.edge_cost(256, 192) == 0


def test_runaway_slider_share_is_pushed_back_towards_the_requested_ratio():
    guide = build_movement_guide(SETTINGS)
    sliders = [HitObject(100, 100, 1000 + 300 * i, True, 150, beat_length=300) for i in range(16)]
    assert guide.slider_cost(sliders[0], sliders) > 0.5
    assert guide.slider_cost(circle(100, 100, 9000), sliders) == 0


def test_crawling_tiny_slider_is_scaled_to_the_requested_velocity():
    tiny = np.array([[200, 200], [210, 204], [220, 208], [230, 212]], float)
    geometry = (tiny / SCALE).reshape(-1)
    obj = candidate_object(geometry, 1000, 2000, 300, True, 1, 1, False, SETTINGS, [])[0]
    assert obj.slider and (obj.x, obj.y) == (200, 200)
    velocity = obj.path_length * obj.repeats * obj.beat_length / (140 * obj.duration)
    assert velocity == pytest.approx(190 / 140, rel=0.05)
    low, high = velocity_band(SETTINGS)
    assert low < 190 / 140 < high
    # Without a requested velocity the old range applies, and a 30 px slider is a circle.
    plain = {"cs": 4, "ar": 9, "od": 8}
    assert velocity_band(plain) == (0.1, 10.0)
    assert min_slider_length(plain) > 40
    assert not candidate_object(geometry, 1000, 2000, 300, True, 1, 1, False, plain, [])[0].slider


def test_melodic_onsets_find_pitched_notes_but_not_drum_hits():
    frames = 1200
    mel = np.full((frames, 64), -0.5, np.float32)
    centres = mel_band_centres()
    voice = np.flatnonzero((centres > 400) & (centres < 2000))
    for start in (200, 500, 800):  # sustained sung notes of ~1 s with two harmonics
        mel[start : start + 80, voice[::4]] = 0.6
    for hit in range(100, frames, 43):  # broadband drum clicks of one frame
        mel[hit, :] = 0.8
    strength = melodic_onsets(mel)
    found = np.flatnonzero(strength > 0.3)
    for start in (200, 500, 800):
        assert np.any(np.abs(found - start) <= 3)
    notes = (200, 500, 800)
    clicks = [hit for hit in range(100, frames, 43) if all(abs(hit - s) > 5 for s in notes)]
    assert all(strength[max(0, hit - 1) : hit + 2].max() < 0.3 for hit in clicks)
    onset = np.zeros(frames, np.float32)
    combined = combined_onsets(onset, mel, 0.6)
    assert combined.max() == pytest.approx(0.6) and np.all(combined >= onset)
    assert np.array_equal(combined_onsets(onset, mel, 0), onset)


def test_calibration_widens_jumps_when_the_spacing_collapsed_despite_an_aim_heavy_split():
    from app.ml.generation import calibration_step

    profile = {"jump_spacing": 170.0, "aim_share": 0.55}
    # Measured on the failing map: aim 2.20, speed 1.34 (share 0.62), 19 px jumps.
    attributes = {"aim_difficulty": 2.2, "speed_difficulty": 1.34}
    settings = {"condition_stars": 6.4, "tempo_bias": 0.0, "style": {}, "calibration_style": None}
    step = calibration_step(
        settings, 6.4, 3.83, attributes, profile, (0.5, 12), {"jump_spacing": 19.4}
    )
    assert step["tempo_bias"] <= 0 and step["calibration_style"]["jump_spacing"] > 170
    fine = calibration_step(
        settings, 6.4, 3.83, attributes, profile, (0.5, 12), {"jump_spacing": 160.0}
    )
    assert fine.get("calibration_style") is None and fine["tempo_bias"] > 0
