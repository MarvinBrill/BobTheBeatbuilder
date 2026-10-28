import importlib.util
from pathlib import Path
from app.beatmaps import parse

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmarks/blind_playtest.py"
SPEC = importlib.util.spec_from_file_location("blind_playtest", SCRIPT)
playtest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(playtest)


def test_blinding_preserves_objects_and_timing_but_removes_condition_metadata(tmp_path):
    source = tmp_path / "source.osu"
    source.write_text(
        "osu file format v14\n[General]\nAudioFilename: original.mp3\n[Metadata]\nTitle:Condition X\nCreator:Model name\nVersion:Architecture B\nTags:baseline\n[Difficulty]\nSliderMultiplier:1.4\n[TimingPoints]\n0,500,4,2,0,70,1,0\n[HitObjects]\n100,100,1000,1,0,0:0:0:0:\n"
    )
    anonymous = playtest.anonymous_map(source, "abc", "A")
    target = tmp_path / "A.osu"
    target.write_text(anonymous)
    assert parse(source).objects == parse(target).objects
    assert parse(source).timing == parse(target).timing
    assert (
        "Condition X" not in anonymous
        and "Model name" not in anonymous
        and "Architecture B" not in anonymous
    )
    assert "AudioFilename: audio.ogg" in anonymous


def test_public_playtest_has_no_conditions_or_fabricated_responses():
    source = dict(map_path="x.osu", audio_path="x.ogg", condition="secret")
    pairs = [
        dict(
            song_id=str(i),
            seed=42,
            section=dict(start_ms=0, end_ms=24000),
            left=source,
            right=source,
        )
        for i in range(8)
    ]
    public, keys = playtest.plan_pairs(pairs, 73)
    assert (public, keys) == playtest.plan_pairs(list(reversed(pairs)), 73)
    assert all(set(row) == {"pair_id", "section", "play_order"} for row in public)
    assert keys[0]["assignments"]["A"]["condition"] == "secret"
    form = playtest.questionnaire(public)
    assert all(
        row["overall_preference"] is None
        and row["confidence"] is None
        and row["observations"] == []
        for row in form["responses"]
    )
    assert set(form["responses"][0]["dimensions"]) == set(playtest.DIMENSIONS)
