import numpy as np
import pytest
from app.beatmaps import HitObject, parse, write
from app.ml.audio import FRAME_MS
from app.ml.finishing import (
    CLAP,
    FINISH,
    VOLUME_KIAI,
    WHISTLE,
    break_periods,
    finish_map,
)
from app.ml.spinners import min_spinner_ms, recovery_ms
from app.ml.playability import low_difficulty_cost
from app.preview_sounds import sound_events
from app.timing import TimingPoint

SETTINGS = {
    "title": "Finishing",
    "artist": "Test",
    "version": "Hard",
    "ar": 8,
    "cs": 4,
    "od": 6,
    "hp": 5,
}
TIMING = [TimingPoint(0, 500)]  # 120 BPM, 4/4 from 0 ms


def mel_with_sections(seconds, loud):
    """64-band mel: quiet (-32 dB) everywhere except the loud (-16 dB) spans in seconds."""
    frames = int(seconds * 1000 / FRAME_MS)
    mel = np.full((frames, 64), 0.2, np.float32)
    for start, end in loud:
        mel[int(start * 1000 / FRAME_MS) : int(end * 1000 / FRAME_MS)] = 0.6
    return mel


def beat_objects(start_ms, end_ms, step=500):
    return [
        HitObject(100 + (index % 5) * 60, 150, float(time))
        for index, time in enumerate(range(start_ms, end_ms, step))
    ]


def test_kiai_preview_hitsounds_and_volume_follow_the_loud_section(tmp_path):
    objects = beat_objects(2000, 60000)
    mel = mel_with_sections(62, [(20, 40)])
    finish = finish_map(objects, TIMING, [], mel, {**SETTINGS, "hitsounds": "basic"})

    assert len(finish.kiai) == 1
    start, end = finish.kiai[0]
    # Starts on a note at a bar line near the loud section, ends near its end.
    assert abs(start - 20000) <= 2000 and start % 2000 == 0
    assert abs(end - 40000) <= 4000
    assert finish.preview_time == round(start)
    assert finish.summary["kiai_sections"] == 1

    by_time = {obj.time: obj.hitsound for obj in objects}
    # Backbeat claps (beats 2 and 4) in the loud part, a finish on the kiai start.
    assert by_time[start] & FINISH
    assert by_time[start + 500] & CLAP and by_time[start + 1500] & CLAP
    assert not by_time[start + 1000] & CLAP
    # Calm intro: no claps, whistles on the downbeats instead.
    assert not any(by_time[time] & CLAP for time in range(4000, 12000, 500))
    assert by_time[4000] & WHISTLE

    volumes = {volume for _, volume, _ in finish.sound_states}
    assert VOLUME_KIAI in volumes and len(volumes) >= 2

    path = tmp_path / "map.osu"
    write(path, objects, SETTINGS, "audio.ogg", 120, 0, TIMING, (), finish.export())
    beatmap = parse(path)
    assert beatmap.preview_time == finish.preview_time
    assert [obj.hitsound for obj in beatmap.objects] == [obj.hitsound for obj in objects]
    kiai_points = [(time, kiai) for time, _, kiai in beatmap.sound_points]
    assert kiai_points[0] == (0, False)  # the first red line never toggles kiai
    assert (round(start), True) in kiai_points and (round(end), False) in kiai_points

    events = sound_events(path)
    claps = [event for event in events if event["sample"] == "hitclap"]
    assert claps and all(
        event["volume"] == VOLUME_KIAI / 100
        for event in claps
        if start / 1000 <= event["time"] < end / 1000
    )


def test_flat_loud_song_gets_no_kiai_but_a_preview_point():
    objects = beat_objects(1000, 40000)
    mel = np.full((int(42000 / FRAME_MS), 64), 0.6, np.float32)
    finish = finish_map(objects, TIMING, [], mel, SETTINGS)
    assert finish.kiai == []
    assert 1000 <= finish.preview_time <= 40000


def test_break_periods_follow_lazer_gaps():
    objects = [HitObject(100, 100, 1000), HitObject(200, 100, 1500), HitObject(300, 100, 12000)]
    breaks = break_periods(objects, [], ar=8)
    # Preempt at AR 8 is 750 ms.
    assert breaks == [(1700, 11250)]
    # A short rest is no break; spinners count as objects.
    assert break_periods(objects, [(2000, 11000)], ar=8) == []


def test_kiai_at_the_first_red_line_uses_a_green_line(tmp_path):
    objects = beat_objects(0, 4000)
    finish = {"sound_states": [(0, 80, True), (2000, 65, False)], "preview_time": 0}
    path = tmp_path / "map.osu"
    write(path, objects, SETTINGS, "audio.ogg", 120, 0, TIMING, (), finish)
    lines = path.read_text().split("[TimingPoints]\n")[1].split("\n\n")[0].splitlines()
    red, green = lines[0].split(","), lines[1].split(",")
    assert red[6] == "1" and red[7] == "0" and red[5] == "80"
    assert green[0] == "0" and green[6] == "0" and green[7] == "1"
    assert lines[2].startswith("2000,") and lines[2].endswith(",65,0,0")


def test_slider_edges_get_their_own_hitsounds(tmp_path):
    slider = HitObject(
        100,
        100,
        2500,
        True,
        1000,
        100,
        100,
        True,
        [[100, 100], [150, 100], [200, 100], [240, 100]],
        2,
        500,
    )
    objects = beat_objects(0, 2500) + [slider] + beat_objects(4000, 30000)
    mel = mel_with_sections(32, [(0, 32)])
    finish_map(objects, TIMING, [], mel, {**SETTINGS, "kiai": False})
    # Head on beat 2 (clap), repeat on beat 3, tail on beat 4 (clap).
    assert slider.hitsound == 0
    assert slider.edge_sounds[0] & CLAP and slider.edge_sounds[2] & CLAP
    assert not slider.edge_sounds[1] & CLAP
    path = tmp_path / "map.osu"
    write(path, objects, SETTINGS, "audio.ogg", 120, 0, TIMING)
    parsed = next(obj for obj in parse(path).objects if obj.slider)
    assert parsed.edge_sounds == slider.edge_sounds


def test_hitsounds_off_keeps_the_flat_volume():
    objects = beat_objects(1000, 30000)
    mel = mel_with_sections(32, [(10, 25)])
    finish = finish_map(objects, TIMING, [], mel, {**SETTINGS, "hitsounds": "off"})
    assert not any(obj.hitsound for obj in objects)
    assert {volume for _, volume, _ in finish.sound_states} == {70}
    with pytest.raises(ValueError):
        finish_map(objects, TIMING, [], mel, {**SETTINGS, "hitsounds": "loud"})


def test_spinner_rules_follow_the_target_tier():
    beat = 500
    assert recovery_ms(beat, {"ar": 5, "target_stars": 1.5}) == 4 * beat
    assert recovery_ms(beat, {"ar": 5, "target_stars": 2.3}) == 2 * beat
    assert recovery_ms(beat, {"ar": 9}) == max(beat, 0.5 * 600, 300)
    assert min_spinner_ms(beat, {"target_stars": 1.5}) == 4 * beat
    assert min_spinner_ms(beat, {"target_stars": 5}) == 1000


def test_low_difficulty_cost_flags_misleading_reading_order():
    settings = {"cs": 4, "ar": 5, "target_stars": 1.8}
    history = [HitObject(100, 200, 0), HitObject(300, 200, 1000)]
    misleading = HitObject(130, 200, 2000)  # 30 px from the first note, 170 px from the last
    clear = HitObject(470, 200, 2000)
    assert low_difficulty_cost(misleading, history, 500, settings) > 0.5
    assert low_difficulty_cost(clear, history, 500, settings) == 0
    assert low_difficulty_cost(misleading, history, 500, {**settings, "target_stars": 4.5}) == 0
