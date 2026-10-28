import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from app import preview_skin
from app.preview_sounds import sound_events


def test_exported_timing_ticks_repeats_and_tail(tmp_path):
    path = tmp_path / "map.osu"
    path.write_text(
        """osu file format v14
[General]
SampleSet:Soft
[Difficulty]
SliderMultiplier:1.4
SliderTickRate:1
[TimingPoints]
0,500,4,2,0,70,1,0
1000,-50,4,2,0,70,0,0
[HitObjects]
100,100,500,1,0,0:0:0:0:
100,100,1000,2,0,L|450:100,2,350,0|0|0,0:0|0:0|0:0,0:0:0:0:
"""
    )
    # SV 2: 350 px at 560 px/s -> 625 ms per span, 1250 ms total.
    events = sound_events(path)
    assert [e["time"] for e in events if e["sample"] == "hitnormal"] == [
        0.5,
        1,
        1.625,
        2.25,
    ]
    assert [e["time"] for e in events if e["sample"] == "slidertick"] == [1.5, 1.75]
    loops = [e for e in events if e["sample"] == "sliderslide"]
    assert loops == [{"time": 1, "end": 2.25, "sample": "sliderslide", "volume": 0.7}]


def test_ticks_near_tail_are_suppressed(tmp_path):
    path = tmp_path / "map.osu"
    path.write_text(
        """osu file format v14
[Difficulty]
SliderMultiplier:1.4
[TimingPoints]
0,500,4,2,0,70,1,0
[HitObjects]
100,100,1000,2,0,L|242:100,1,142,0|0,0:0|0:0,0:0:0:0:
"""
    )
    assert not [e for e in sound_events(path) if e["sample"] == "slidertick"]


def test_skin_selection_prefers_matching_active_skin_but_not_other_variants():
    skins = [
        {"id": "1", "name": "[Anime] [ssM4in]"},
        {"id": "2", "name": "ssM4in Skin"},
        {"id": "3", "name": "ssM4in2"},
    ]
    assert preview_skin.select_skin(skins, "ssM4in", "1") == skins[0]
    assert preview_skin.select_skin(skins, "ssM4in Skin", "1") == skins[1]
    with pytest.raises(ValueError):
        preview_skin.select_skin(skins, "ssM4in", "3")


def test_reader_snapshot_cleanup_soft_fallback_and_intentional_silence(
    tmp_path, monkeypatch
):
    (tmp_path / "client.realm").write_bytes(b"test snapshot")
    reader = tmp_path / "reader.dll"
    reader.touch()
    monkeypatch.setattr(preview_skin, "READER", reader)
    files = []
    for index, (name, payload) in enumerate(
        [
            ("soft-hitnormal.wav", b"soft"),
            ("normal-hitnormal.wav", b"normal"),
            ("normal-slidertick.ogg", b"tick"),
            ("soft-sliderslide.wav", b""),
            ("normal-sliderslide.wav", b"must not replace silence"),
        ]
    ):
        digest = str(index) * 64
        path = preview_skin.hash_path(tmp_path, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files.append({"name": name, "hash": digest})
    snapshots = []

    def read(args, **kwargs):
        assert args[-1] == "--skins"
        snapshots.append(Path(args[-2]))
        assert snapshots[-1].read_bytes() == b"test snapshot"
        return SimpleNamespace(
            returncode=0, stdout=json.dumps([{"name": "ssM4in", "files": files}])
        )

    monkeypatch.setattr(preview_skin.subprocess, "run", read)
    skin = preview_skin.read_skin(tmp_path, "ssM4in")
    assert skin["samples"]["hitnormal"]["path"].read_bytes() == b"soft"
    assert skin["samples"]["slidertick"]["media_type"] == "audio/ogg"
    assert "sliderslide" not in skin["samples"]
    assert not snapshots[0].parent.exists()
    assert (tmp_path / "client.realm").read_bytes() == b"test snapshot"


def test_only_allowlisted_samples_can_be_served(monkeypatch):
    monkeypatch.setattr(
        preview_skin, "current_skin", lambda: pytest.fail("Must reject before reading")
    )
    with pytest.raises(preview_skin.HTTPException) as error:
        preview_skin.sample_file("client.realm")
    assert error.value.status_code == 404
