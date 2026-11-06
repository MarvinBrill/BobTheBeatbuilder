import numpy as np

from app.ml.audio import FRAME_MS
from app.ml.song_plan import SongPlan, build_plan
from app.timing import TimingPoint

BEAT = 375.0  # 160 BPM
BAR = 4 * BEAT


def song():
    """16 calm bars, 16 dense loud bars (A), 8 quiet bars, 16 loud bars (A again)."""
    rng = np.random.default_rng(0)
    plan = [("calm", 16), ("drop", 16), ("break", 8), ("drop", 16)]
    frames = int(sum(bars for _, bars in plan) * BAR / FRAME_MS) + 10
    mel = np.full((frames, 64), -0.9, np.float32)
    timbre_drop = rng.uniform(-0.2, 0.5, 64).astype(np.float32)
    timbre_calm = rng.uniform(-0.6, -0.2, 64).astype(np.float32)
    start = 0.0
    for kind, bars in plan:
        first, last = int(start / FRAME_MS), int((start + bars * BAR) / FRAME_MS)
        mel[first:last] = timbre_drop if kind == "drop" else timbre_calm
        step = BEAT / 2 if kind == "drop" else BEAT * 2
        for time in np.arange(start, start + bars * BAR, step):
            frame = int(time / FRAME_MS)
            mel[frame : frame + 2] += 0.4 if kind == "drop" else 0.2
        start += bars * BAR
    return np.clip(mel, -1, 1)


def test_sections_follow_the_structure_and_the_repeat_is_found():
    plan = build_plan(song(), FRAME_MS, [TimingPoint(0, BEAT, meter=4, meter_confidence=0.75, bar_origin=0)])
    starts = [round(section.start_ms / BAR) for section in plan.sections]
    assert {16, 32, 40} <= set(starts)
    first_drop = next(s for s in plan.sections if round(s.start_ms / BAR) == 16)
    last_drop = next(s for s in plan.sections if round(s.start_ms / BAR) == 40)
    quiet = plan.sections[0]
    assert first_drop.intensity > quiet.intensity + 0.2
    assert first_drop.kind == "drop" and last_drop.kind == "drop"
    assert last_drop.repeat_of == plan.sections.index(first_drop)
    assert plan.accents and all(plan.accents[i] <= plan.accents[i + 1] for i in range(len(plan.accents) - 1))


def test_plan_roundtrip_and_lookups():
    plan = build_plan(song(), FRAME_MS, [TimingPoint(0, BEAT)])
    again = SongPlan.from_dict(plan.to_dict())
    assert again.to_dict() == plan.to_dict()
    drop = plan.style_at(20 * BAR)
    calm = plan.style_at(2 * BAR)
    assert drop["spacing"] > calm["spacing"] and drop["density"] > calm["density"]
    middle = plan.sections[1]
    hits = plan.accents_between(middle.start_ms, middle.end_ms)
    assert all(middle.start_ms < time < middle.end_ms for time, _ in hits)
    assert plan.section_at(-5) is plan.sections[0]


def test_user_edits_survive_from_dict():
    plan = build_plan(song(), FRAME_MS, [TimingPoint(0, BEAT)]).to_dict()
    plan["sections"][0]["style"] = "streams"
    plan["sections"][0]["kind"] = "nonsense"
    parsed = SongPlan.from_dict(plan)
    assert parsed.sections[0].style == "streams" and parsed.sections[0].kind == "verse"
    assert parsed.style_at(10)["streams"] > 1
