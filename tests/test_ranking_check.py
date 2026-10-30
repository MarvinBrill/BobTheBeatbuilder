from app.beatmaps import HitObject
from app.ml.finishing import Finish
from app.ranking_check import ranking_check, timestamp
from app.tiers import tier_name
from app.timing import TimingPoint

TIMING = [TimingPoint(0, 500)]


def ids(result):
    return {item["id"] for item in result["findings"]}


def test_tiers_and_timestamps():
    assert [tier_name(value) for value in (1.5, 2.3, 3.5, 4.5, 6.1)] == [
        "Easy", "Normal", "Hard", "Insane", "Expert"
    ]  # fmt: skip
    assert tier_name(None) is None
    assert timestamp(83456) == "01:23:456"


def test_easy_draft_reports_settings_density_and_reading_order():
    objects = []
    for bar in range(40):
        base = 2000 + bar * 2000
        # 1/2 rhythm in an Easy, and every third note next to the one before last.
        objects += [
            HitObject(100, 200, base),
            HitObject(300, 200, base + 500),
            HitObject(120, 210, base + 750),
        ]
    settings = {"ar": 9, "od": 7, "hp": 5, "cs": 4}
    result = ranking_check(objects, [], TIMING, settings, stars=1.6, audio_ms=90000)
    assert result["tier"] == "Easy"
    found = ids(result)
    assert {"settings_ar", "settings_od", "settings_hp", "easy_density", "reading_order"} <= found
    assert "ai_policy" in found and "hitsounds" in found
    reading = next(item for item in result["findings"] if item["id"] == "reading_order")
    assert reading["times"][0] == "00:02:750"


def test_hard_stream_snapping_drain_and_outro():
    objects = [HitObject(100 + index * 20, 200, 1000 + index * 125) for index in range(8)]
    objects.append(HitObject(400, 200, 3310))  # 2.5 ms off a 1/16 tick
    result = ranking_check(
        objects,
        [],
        TIMING,
        {"ar": 7, "od": 6, "hp": 5, "cs": 4},
        stars=3.2,
        audio_ms=60000,
        finish=Finish(preview_time=-1),
    )
    found = ids(result)
    assert {"stream_length", "snapping", "drain_time", "outro", "preview_point"} <= found
    assert result["problems"] >= 2
    assert result["findings"][0]["severity"] == "problem"


def test_spinner_recovery_for_normal():
    objects = [HitObject(100, 100, time) for time in range(1000, 40000, 1000)]
    objects = [obj for obj in objects if not 10000 <= obj.time <= 13000]
    spinners = [(10000, 13500)]  # next object 14000: one beat of recovery, Normal wants two
    result = ranking_check(
        objects, spinners, TIMING, {"ar": 5, "od": 4, "hp": 4, "cs": 4}, stars=2.3
    )
    assert "spinner_recovery" in ids(result)
    assert "spinner_length" not in ids(result)


def test_clean_hard_map_has_only_the_policy_note():
    objects, x = [], 100
    for index in range(80):
        objects.append(HitObject(x, 150 + (index % 2) * 120, 1000 + index * 500))
        x = 100 if x > 380 else x + 70
    for obj in objects:
        obj.hitsound = 8
    result = ranking_check(
        objects,
        [],
        TIMING,
        {"ar": 7.5, "od": 6, "hp": 5, "cs": 4},
        stars=3.0,
        audio_ms=45000,
        finish=Finish(preview_time=1000),
    )
    assert ids(result) == {"ai_policy"}, result["findings"]
