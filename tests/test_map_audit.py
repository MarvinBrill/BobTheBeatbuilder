import math

import numpy as np

from app.beatmaps import HitObject
from app.ml.audio import FRAME_MS
from app.ml.map_audit import audit, movement_audit, rhythm_audit
from app.timing import TimingPoint


def circle(x, y, time, beat=300):
    return HitObject(x, y, time, end_x=x, end_y=y, beat_length=beat)


def walk(turns, step=170, start=(180.0, 150.0)):
    position, heading, objects = np.array(start), 0.0, []
    for index, turn in enumerate(turns):
        objects.append(circle(*position, 1000 + 150 * index))
        position = position + step * np.array([math.cos(heading), math.sin(heading)])
        heading += math.radians(turn)
    return objects


def test_pentagram_chains_are_counted_and_back_and_forth_is_not():
    star = movement_audit(walk([144] * 40))
    assert star["star_chain_share"] > 0.9 and star["longest_star_chain"] > 30
    assert star["sharp_share"] > 0.9
    zigzag = movement_audit(walk([150, -150] * 20))
    assert zigzag["star_chain_share"] == 0 and zigzag["longest_star_chain"] == 0
    flow = movement_audit(walk([30] * 40))
    assert flow["sharp_share"] == 0


def test_onsets_on_and_between_the_clicks():
    beat = 400.0
    frames = int(60000 / FRAME_MS)
    mel = np.full((frames, 64), -0.6, np.float32)
    for time in np.arange(500, 59000, beat):
        frame = int(round(time / FRAME_MS))
        mel[frame : frame + 3] = 0.5
    timing = [TimingPoint(500, beat)]
    on = [circle(100 + (i % 2) * 200, 200, 500 + beat * i, beat) for i in range(100)]
    off = [circle(100 + (i % 2) * 200, 200, 500 + beat * i + beat * 0.37, beat) for i in range(100)]
    good, bad = rhythm_audit(on, mel, FRAME_MS, timing), rhythm_audit(off, mel, FRAME_MS, timing)
    assert good["onset_hit_share"] > 0.9 > 0.3 > bad["onset_hit_share"] or bad["onset_offset_ms"] != 0
    assert good["strong_onset_coverage"] > 0.9
    report = audit(walk([144] * 40), timing, mel, FRAME_MS)
    assert any("Stern" in warning for warning in report["warnings"])
