from app.timing import TimingPoint
import math
import numpy as np
import pytest
import soundfile as sf
from app.beatmaps import HitObject, parse, write
from app.config import DATA
from app.ml.audio import features
from app.ml.rhythm_inference import select_times
from app.ml.split import song_groups
from app.slider_paths import endpoint

SETTINGS = dict(
    density=4,
    snap=True,
    divisor=4,
    od=7,
    slider_bias=0.5,
    title="Test",
    artist="Test",
    version="Hard",
    ar=9,
    cs=4,
    hp=5,
)


def test_digital_silence_and_broken_cache_do_not_generate_notes(tmp_path):
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(44100, np.float32), 22050)
    cache = DATA / "cache/mel-v2-silence-test.npy"
    cache.write_bytes(b"interrupted cache write")
    mel = features(path, "silence-test")
    assert np.all(mel == -1)
    scores = np.random.default_rng(1).random((len(mel), 4)).astype(np.float32)
    assert select_times(scores, mel, SETTINGS, [TimingPoint(0, 500)]) == []


def test_invalid_cache_shape_is_rebuilt_on_windows(tmp_path):
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(44100, np.float32), 22050)
    cache = DATA / "cache/mel-v2-shape-test.npy"
    np.save(cache, np.ones((10, 3), np.float32))
    mel = features(path, "shape-test")
    assert mel.shape[1] == 64 and np.all(mel == -1)


def test_split_unions_metadata_and_audio_transitively():
    records = [
        {"audio_hash": "a", "artist": "Artist", "title": "Song!"},
        {"audio_hash": "b", "artist": "ARTIST", "title": "song"},
        {"audio_hash": "b", "artist": "Alias", "title": "Alternate title"},
        {"audio_hash": "c", "artist": "Other", "title": "Different"},
        {"audio_hash": "d", "artist": "Unknown", "title": "Untitled"},
        {"audio_hash": "e", "artist": "Unknown", "title": "Untitled"},
    ]
    groups = song_groups(records)
    assert groups[0] == groups[1] == groups[2]
    assert len(set(groups)) == 4


@pytest.mark.parametrize(
    "curve,length,expected",
    [
        ("L|200:100", 50, (150, 100)),
        ("L|200:100", 150, (250, 100)),
        ("L|200:100|200:200", 150, (200, 150)),
        ("B|150:100|200:100|200:100|200:200", 150, (200, 150)),
        ("P|150:50|200:100", math.pi * 50 / 2, (150, 50)),
    ],
)
def test_slider_tail_follows_declared_path_length(curve, length, expected):
    assert np.allclose(endpoint(100, 100, curve, length), expected, atol=0.25)


def test_positive_timing_offset_preserves_slider_duration_before_offset(tmp_path):
    path = tmp_path / "map.osu"
    write(path, [HitObject(100, 100, 200, True, 500, 310, 100)], SETTINGS, "audio.ogg", 180, 5000)
    assert abs(parse(path).objects[0].duration - 500) < 0.01
