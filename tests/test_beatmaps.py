import math
import pytest
from app.beatmaps import HitObject, parse, write
from app.config import inside


def test_slider_timing_inherited_velocity_and_repeats(tmp_path):
    path = tmp_path / "map.osu"
    path.write_text(
        """osu file format v14
[General]
AudioFilename: audio.mp3
Mode: 0
[Difficulty]
SliderMultiplier:1.4
[TimingPoints]
0,500,4,2,0,70,1,0
1000,-50,4,2,0,70,0,0
2000,400,4,2,0,70,1,0
[HitObjects]
100,100,500,2,0,L|240:100,1,140
100,100,1000,6,0,L|240:100,2,140
100,100,2000,2,0,L|240:100,1,140
256,192,3000,8,0,3500
"""
    )
    beatmap = parse(path)
    assert [obj.duration for obj in beatmap.objects] == [500, 500, 400]
    assert beatmap.objects[1].end_x == 100
    assert beatmap.skipped == 1


@pytest.mark.parametrize(
    "filename", ["../escape", "/absolute", "D:/other", "..\\escape", "folder/../../escape"]
)
def test_reject_paths_outside_root(tmp_path, filename):
    with pytest.raises(ValueError):
        inside(tmp_path, filename)


def test_writer_roundtrip_and_metadata_injection(tmp_path):
    settings = {
        "title": "Title\n[HitObjects]\u2028100,100,100,1,0\x00",
        "artist": "Artist",
        "version": "Hard",
        "ar": 9,
        "cs": 4,
        "hp": 5,
        "od": 7,
    }
    objects = [HitObject(100, 120, 1000), HitObject(200, 200, 1500, True, 500, 340, 200)]
    path = tmp_path / "map.osu"
    write(path, objects, settings, "audio.mp3", 120, 0)
    beatmap = parse(path)
    assert len(beatmap.objects) == 2
    assert beatmap.objects[1].duration == 500
    assert beatmap.audio == "audio.mp3"
    assert beatmap.difficulty["ApproachRate"] == 9
    assert math.isfinite(beatmap.objects[1].duration)
