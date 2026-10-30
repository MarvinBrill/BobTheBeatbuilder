import numpy as np
from app.beatmaps import HitObject, parse
from app.ml.audio import FRAME_MS
from app.ml.finishing import finish_map
from app.ml.generation import verified_write
from app.preview_sounds import sound_events
from app.ranking_check import ranking_check
from app.timing import TimingPoint

SETTINGS = {
    "title": "Export",
    "artist": "Test",
    "version": "Hard",
    "ar": 7.5,
    "cs": 4,
    "od": 6,
    "hp": 5,
    "hitsounds": "full",
}


def test_verified_write_keeps_finishing_layers(tmp_path):
    timing = [TimingPoint(0, 500, meter=4, meter_confidence=1.0, bar_origin=0)]
    objects = []
    for index in range(80):
        x, y = 80 + (index % 6) * 60, 120 + (index % 2) * 100
        objects.append(HitObject(x, y, 1000.0 + index * 500, end_x=x, end_y=y))
    curve = [[100, 300], [160, 300], [220, 300], [280, 300]]
    objects[40] = HitObject(100, 300, 21000.0, True, 1000, 100, 300, True, curve, 2, 500)
    objects = [obj for obj in objects if not 21000 < obj.time <= 22000]
    objects = [obj for obj in objects if not 30000 < obj.time < 38000]
    spinners = [(31000.0, 34000.0)]
    frames = int(42000 / FRAME_MS)
    mel = np.full((frames, 64), 0.2, np.float32)
    mel[int(14000 / FRAME_MS) : int(29000 / FRAME_MS)] = 0.6
    finish = finish_map(objects, timing, spinners, mel, SETTINGS)
    path = tmp_path / "map.osu"
    verified_write(path, objects, SETTINGS, "audio.ogg", timing, spinners=spinners, finish=finish)
    beatmap = parse(path)
    assert beatmap.preview_time == finish.preview_time >= 0
    assert beatmap.breaks == [tuple(map(float, span)) for span in finish.breaks]
    assert finish.kiai and any(kiai for _, _, kiai in beatmap.sound_points)
    assert not beatmap.sound_points[0][2]
    events = sound_events(path)
    assert {"hitclap", "hitwhistle", "hitfinish"} <= {event["sample"] for event in events}
    report = ranking_check(
        objects, spinners, timing, SETTINGS, stars=3.0, audio_ms=42000, finish=finish
    )
    assert report["tier"] == "Hard"
    assert "hitsounds" not in {item["id"] for item in report["findings"]}
    assert "preview_point" not in {item["id"] for item in report["findings"]}
