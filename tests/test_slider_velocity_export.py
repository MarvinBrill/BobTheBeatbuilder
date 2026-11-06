import numpy as np
import pytest

from app.beatmaps import HitObject, parse, unify_slider_velocities, write
from app.curve_geometry import curve_length
from app.timing import TimingPoint

SETTINGS = dict(title="SV", artist="Test", version="Test", ar=9, cs=4, od=8, hp=5)


def slider(x, time, length, beat=300.0, duration_beats=0.5):
    curve = [[x, 100], [x + length / 3, 100], [x + 2 * length / 3, 100], [x + length, 100]]
    duration = duration_beats * beat
    return HitObject(
        x, 100, time, True, duration, x + length, 100, True, curve, 1, beat, False,
        curve_length(curve),
    )  # fmt: skip


def velocities(path):
    text = path.read_text()
    lines = text.split("[TimingPoints]")[1].split("[")[0].strip().splitlines()
    return [line for line in lines if line.split(",")[6] == "0"]


def test_near_equal_velocities_share_one_green_line_and_roundtrip(tmp_path):
    lengths = [90, 91, 89, 92, 90, 140, 141, 139]
    objects = [slider(60 + (i % 3) * 120, 1000 + 600 * i, length) for i, length in enumerate(lengths)]
    unify_slider_velocities(objects)
    path = tmp_path / "map.osu"
    write(path, objects, SETTINGS, "audio.ogg", 200, 0, [TimingPoint(0, 300.0)])
    greens = velocities(path)
    assert len(greens) == 2  # one per velocity group, not one per slider
    parsed = parse(path).objects
    for original, read in zip(objects, parsed):
        assert read.duration == pytest.approx(original.duration, abs=0.5)
        assert np.hypot(read.end_x - original.end_x, read.end_y - original.end_y) < 0.75


def test_a_volume_change_does_not_reset_the_slider_velocity(tmp_path):
    objects = [slider(100, 1000 + 600 * i, 120) for i in range(6)]
    unify_slider_velocities(objects)
    path = tmp_path / "map.osu"
    finish = {"sound_states": [(0, 65, False), (2500, 80, True)]}
    write(path, objects, SETTINGS, "audio.ogg", 200, 0, [TimingPoint(0, 300.0)], finish=finish)
    parsed = parse(path).objects
    assert [round(obj.duration) for obj in parsed] == [round(obj.duration) for obj in objects]
