import numpy as np
import pytest
from app.beatmaps import Beatmap, HitObject
from app.ml import style
from app.ml.sequence_data import pack_events, frame_labels, summary_features
from app.ml.sequence_batches import make_example, draw_augmentation
from app.ml.sequence_generation import choose_next_time, next_onset, snap_to_grid
from app.ml.audio import FRAME_MS


def pattern_map(bursts=True):
    """Five-note 1/4 bursts followed by 1/2 jumps at 120 BPM."""
    objects, time = [], 500.0
    for index in range(96):
        objects.append(HitObject(100 + index % 4 * 90, 120 + index % 3 * 70, time, beat_length=500))
        objects[-1].end_x, objects[-1].end_y = objects[-1].x, objects[-1].y
        time += 125 if bursts and index % 8 < 4 else 250
    return Beatmap(
        objects=objects,
        timing=[(0, 500, False)],
        difficulty={"ApproachRate": 9, "CircleSize": 4, "OverallDifficulty": 8, "HPDrainRate": 6},
        metadata={"Version": "NiNo's Insane", "Creator": "Sotarks"},
    )


def styled_item(beatmap=None, frames=4000):
    beatmap = beatmap or pattern_map()
    mel = np.zeros((frames, 64), np.float32)
    events, _ = pack_events(beatmap, len(mel))
    values, mask = style.vector(style.measure(beatmap, {"star_rating": 5.1, "aim_difficulty": 2.4, "speed_difficulty": 2.2}))
    return dict(
        events=events,
        mel=mel,
        summary=summary_features(mel),
        labels=frame_labels(beatmap, len(mel)),
        condition=np.full(8, 0.5, np.float32),
        style=values,
        style_mask=mask,
        mapper=2,
    )


def test_gap_classes_and_grids():
    assert style.GAP_BEATS[style.gap_class(125, 500)] == 0.25
    assert style.GAP_BEATS[style.gap_class(500 / 3, 500)] == pytest.approx(1 / 3)
    assert style.gap_class(5000, 500) == style.REST
    assert style.gap_class(215, 500) == -1
    quarter, mixed = style.allowed_gaps(4), style.allowed_gaps(12)
    third = style.GAP_BEATS.index(1 / 3)
    assert not quarter[third] and mixed[third] and quarter[style.GAP_BEATS.index(0.25)]


def test_measure_detects_bursts_streams_and_jumps():
    raw = style.measure(pattern_map(), {"star_rating": 5.1, "aim_difficulty": 3, "speed_difficulty": 1})
    assert raw["stars"] == 5.1 and raw["aim_share"] == pytest.approx(0.75)
    assert raw["tempo"] == pytest.approx(120)
    assert raw["burst_share"] == pytest.approx(5 / 8, abs=0.03) and raw["stream_share"] == 0
    assert raw["quarter_share"] == pytest.approx(0.5, abs=0.02)
    assert raw["stream_length"] == pytest.approx(5)
    assert raw["slider_ratio"] == 0 and raw["slider_velocity"] is None
    streams = style.measure(pattern_map(bursts=False))
    assert streams["burst_share"] == 0 and streams["quarter_share"] == 0 and streams["stars"] is None
    values, mask = style.vector(streams)
    assert mask[style.INDEX["stars"]] == 0 and mask[style.INDEX["slider_velocity"]] == 0
    condition = style.condition_input(values, mask)
    assert condition.shape == (style.CONDITION_SIZE,) and not condition[: style.FEATURE_COUNT][mask == 0].any()
    for key in style.KEYS:
        assert style.decode(key, style.encode(key, 3.0)) == pytest.approx(
            np.clip(3.0, *style.LIMITS.get(key, (0, 1)))
        )


def test_guest_difficulties_are_credited_to_the_guest_mapper():
    known = style.known_mappers([{"creator": "Sotarks"}, {"creator": "sotarks"}, {"creator": "Sotarks"}, {"creator": "Kagetsu"}])
    assert style.resolve_mapper("Sotarks", style.mapper_candidate("NiNo's Insane"), known) == "NiNo"
    assert style.resolve_mapper("Rizia", style.mapper_candidate("kagetsu's Extra"), known) == "Kagetsu"
    assert style.resolve_mapper("sotarks", style.mapper_candidate("Extra"), known) == "Sotarks"
    assert style.mapper_candidate("deetz' Expert") == "deetz"
    assert style.mapper_candidate("Extra: Kise") is None
    assert style.mapper_candidate("Hard's Insane") is None
    assert style.mapper_vocabulary(["a", "b", "a", "c", "a", "b"], 2) == ["a", "b"]


def test_rhythm_targets_skip_spinners_and_queries_never_see_the_next_object():
    beatmap = pattern_map()
    beatmap.ignored = [(beatmap.objects[10].time + 10, beatmap.objects[11].time - 10)]
    events, _ = pack_events(beatmap, 4000)
    assert events.shape[1] == 49 and events[10, 48] == -1 and events[0, 48] == style.GAP_BEATS.index(0.25)
    assert events[-1, 48] == -1 and np.allclose(events[:, 47], 0.5)
    item = styled_item()
    first = make_example(item, 0, np.arange(30), 1)
    moved = {**item, "events": item["events"].copy()}
    moved["events"][20, 0] += 0.05
    second = make_example(moved, 0, np.arange(30), 1)
    assert np.array_equal(first["queries"][:20], second["queries"][:20])
    assert first["condition"].shape == (style.CONDITION_SIZE,) and first["mapper"] == 2
    assert first["gap_target"][0] == style.GAP_BEATS.index(0.25) and first["gap_target"][100] == -1


def test_style_augmentation_keeps_the_legacy_random_stream_and_hides_groups():
    legacy = draw_augmentation(np.random.default_rng(3))
    styled = draw_augmentation(np.random.default_rng(3), True)
    assert legacy[0] == styled[0] and np.array_equal(legacy[1], styled[1]) and legacy[2] == styled[2]
    hidden = [draw_augmentation(np.random.default_rng(seed), True)[3] for seed in range(200)]
    assert all(keep[style.INDEX["ar"]] == 1 for keep in hidden)
    assert 0.2 < np.mean([keep[style.INDEX["stream_share"]] == 0 for keep in hidden]) < 0.5
    example = make_example(styled_item(), 0, np.arange(20), 1, np.random.default_rng(9))
    assert np.isfinite(example["audio"]).all() and example["audio"].max() <= 1


def test_profiles_summarise_mappers_and_star_buckets():
    items = []
    for index in range(6):
        values, mask = style.vector({"stars": 5.0 + index * 0.05, "jump_spacing": 150 + index, "ar": 9})
        items.append({"style": values, "style_mask": mask, "mapper_name": "NiNo" if index < 4 else "Other"})
    profiles = style.mapper_profiles(items, ["NiNo"])
    assert profiles["NiNo"]["maps"] == 4 and profiles["NiNo"]["style"]["jump_spacing"] == pytest.approx(151.5)
    buckets = style.star_profiles(items)
    assert list(buckets) == ["5.0", "5.5"] or list(buckets) == ["5.0"]
    assert buckets["5.0"]["style"]["ar"] == pytest.approx(9)


def test_rhythm_choice_stays_on_the_grid_respects_sliders_streams_and_silence():
    frames = 3000
    onset = np.full(frames, 0.05, np.float32)
    beat_frames = (np.arange(0, frames * FRAME_MS, 125) / FRAME_MS).astype(int)
    onset[beat_frames[beat_frames < frames]] = 0.9
    active = np.ones(frames, bool)
    active[1500:1800] = False
    timing = [(0.0, 500.0)]
    settings = {"divisor": 4, "seed": 1, "rhythm_temperature": 0.05}
    rng = np.random.default_rng(1)
    logits = np.full(style.GAP_CLASSES, -5.0)
    logits[style.GAP_BEATS.index(0.25)] = 5
    circle = HitObject(100, 100, 1000.0, beat_length=500)
    time, gap = choose_next_time(logits, circle, timing, onset, active, frames * FRAME_MS, settings, 0, rng)
    assert gap == style.GAP_BEATS.index(0.25) and time == pytest.approx(1125)
    capped = choose_next_time(logits, circle, timing, onset, active, frames * FRAME_MS, {**settings, "max_stream": 4}, 3, rng)
    assert capped[1] == style.REST or style.GAP_BEATS[capped[1]] > 0.26
    slider = HitObject(100, 100, 1000.0, True, 500, 200, 100, beat_length=500)
    time, _ = choose_next_time(logits, slider, timing, onset, active, frames * FRAME_MS, settings, 0, rng)
    assert time >= 1500 + 0.24 * 500 - 1e-6 and (time / 125) == pytest.approx(round(time / 125))
    before_break = HitObject(100, 100, 1480 * FRAME_MS, beat_length=500)
    time, _ = choose_next_time(logits, before_break, timing, onset, active, frames * FRAME_MS, settings, 0, rng)
    assert not 1500 <= round(time / FRAME_MS) < 1800
    assert snap_to_grid(1130, timing, 4) == 1125
    assert next_onset(np.zeros(frames), active, 0, frames * FRAME_MS, timing, 4) is None
    assert next_onset(np.zeros(frames), active, 0, frames * FRAME_MS, timing, 4, anywhere=True) == 0
