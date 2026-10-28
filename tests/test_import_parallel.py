"""Parallel scan/import must register exactly what the sequential path registers."""

import hashlib
import shutil
import zipfile
from uuid import uuid4
import pytest
from app import import_workers, library, store
from app.beatmaps import HitObject, write
from app.config import DATA
from app.jobs import Cancelled


class Context:
    def __init__(self, cancel_after=None):
        self.reports, self.cancel_after = 0, cancel_after

    def check(self):
        if self.cancel_after is not None and self.reports >= self.cancel_after:
            raise Cancelled()

    def report(self, *args, **kwargs):
        self.check()
        self.reports += 1


SETTINGS = {"artist": "Test", "ar": 9, "cs": 4, "od": 7, "hp": 5}


def song(folder, title, difficulties=2, audio=b"audio"):
    folder.mkdir(parents=True)
    (folder / "audio.wav").write_bytes(audio + title.encode())
    for index in range(difficulties):
        objects = [
            HitObject(100 + index, 100, 1000),
            HitObject(200, 200, 1500, True, 500, 340, 200),
        ]
        write(
            folder / f"{index}.osu",
            objects,
            {**SETTINGS, "title": title, "version": f"Diff {index}"},
            "audio.wav",
            120,
            0,
        )


def clear_maps():
    with store.connect() as connection:
        connection.execute("DELETE FROM records WHERE kind='map'")


@pytest.fixture
def library_root():
    clear_maps()
    root = DATA / "scan-tests" / uuid4().hex
    yield root
    clear_maps()
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(params=["inline", "processes"])
def workers(request, monkeypatch):
    if request.param == "processes":
        monkeypatch.setenv("BOB_IMPORT_WORKERS", "2")
        monkeypatch.setattr(import_workers, "PROCESS_THRESHOLD", 1)
    else:
        monkeypatch.setenv("BOB_IMPORT_WORKERS", "1")
    return request.param


def test_parallel_scan_matches_sequential_registration(library_root, workers):
    for index in range(4):
        song(library_root / f"song {index}", f"Song {index}")
    (library_root / "song 0" / "broken.osu").write_text("not a beatmap")
    (library_root / "song 1" / "taiko.osu").write_text(
        (library_root / "song 1" / "0.osu").read_text().replace("Mode: 0", "Mode: 1")
    )
    (library_root / "song 2" / "lost.osu").write_text(
        (library_root / "song 2" / "0.osu").read_text().replace("audio.wav", "missing.wav")
    )
    (library_root / "song 3" / "escape.osu").write_text(
        (library_root / "song 3" / "0.osu").read_text().replace("audio.wav", "../../x.wav")
    )
    result = library.scan_songs(library_root, Context())
    assert result["imported"] == 8
    assert result["error_count"] == 3
    assert any(error.startswith("broken.osu: Invalid or damaged") for error in result["errors"])
    assert "lost.osu: Audio missing: missing.wav" in result["errors"]
    assert any(error.startswith("escape.osu: Path escapes") for error in result["errors"])

    parallel = {item["id"]: item for item in store.all_records("map")}
    clear_maps()
    for path in sorted(library_root.rglob("[0-9].osu")):
        library.register(path, path.parent / "audio.wav", "songs")
    sequential = {item["id"]: item for item in store.all_records("map")}
    assert parallel == sequential


def test_rescan_keeps_known_lazer_ratings(library_root, workers):
    song(library_root / "rated", "Rated", difficulties=1)
    path = library_root / "rated" / "0.osu"
    original = library.register(path, path.parent / "audio.wav", "lazer", stars=6.4)
    assert library.scan_songs(library_root, Context())["imported"] == 1
    rescanned = store.get("map", original["id"])
    assert rescanned["source"] == "songs"
    assert (rescanned["star_rating"], rescanned["star_source"]) == (6.4, "lazer")


def test_lazer_candidates_count_other_modes_and_mapping_errors(library_root, workers):
    song(library_root / "set", "Lazer", difficulties=2)
    files = library_root / "files"
    entries = []
    for name in ("0.osu", "1.osu", "audio.wav"):
        content = (library_root / "set" / name).read_bytes()
        if name == "1.osu":
            content = content.replace(b"Mode: 0", b"Mode: 3")
        digest = hashlib.sha256(content).hexdigest()
        target = files / digest[0] / digest[:2] / digest
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        entries.append({"name": name, "hash": digest, "star_rating": 3.21})
    mapping = {item["name"].casefold(): item for item in entries}
    candidates = [
        *(library.lazer_candidate(library_root, "set-1", mapping, item) for item in entries[:2]),
        # A difficulty whose set lists no audio file, and a malformed content hash.
        library.lazer_candidate(library_root, "set-1", {}, dict(entries[0], name="orphan.osu")),
        library.lazer_candidate(library_root, "set-1", mapping, {"name": "bad.osu", "hash": "x"}),
    ]
    count, skipped, errors = library.register_many(candidates, Context())
    assert (count, skipped) == (1, 1)
    assert errors == ["Missing audio mapping for orphan.osu", "Invalid lazer content hash"]
    (item,) = store.all_records("map")
    assert (item["star_rating"], item["source_id"]) == (3.21, "set-1")


def test_cancelled_parallel_scan_registers_nothing(library_root, monkeypatch):
    monkeypatch.setenv("BOB_IMPORT_WORKERS", "2")
    monkeypatch.setattr(import_workers, "PROCESS_THRESHOLD", 1)
    for index in range(3):
        song(library_root / f"song {index}", f"Song {index}")
    with pytest.raises(Cancelled):
        library.scan_songs(library_root, Context(cancel_after=0))
    assert store.count("map") == 0


@pytest.mark.filterwarnings("ignore:Duplicate name")
def test_parallel_archive_extraction_keeps_last_duplicate_and_rejects_slip(workers):
    archive = DATA / "uploads" / f"{uuid4().hex}.osz"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for index in range(12):
            output.writestr(f"set/{index}.wav", f"audio {index}" * 1000)
        output.writestr("set/notes.txt", "ignored")
        output.writestr("set/0.wav", "replacement")
    destination = DATA / "library" / uuid4().hex
    destination.mkdir()
    try:
        library.extract_archive(archive, destination, Context())
        files = sorted(path.name for path in destination.rglob("*") if path.is_file())
        assert files == sorted(f"{index}.wav" for index in range(12))
        assert (destination / "set" / "0.wav").read_text() == "replacement"
        assert (destination / "set" / "11.wav").read_text() == "audio 11" * 1000
    finally:
        shutil.rmtree(destination)

    unsafe = DATA / "uploads" / f"{uuid4().hex}.osz"
    with zipfile.ZipFile(unsafe, "w") as output:
        output.writestr("first.wav", "written before the bad member in the old code")
        output.writestr("../escape.osu", "bad")
    destination.mkdir()
    try:
        with pytest.raises(ValueError, match="escapes"):
            library.extract_archive(unsafe, destination, Context())
        assert not any(destination.iterdir())
    finally:
        shutil.rmtree(destination)
        archive.unlink()
        unsafe.unlink()


def test_store_batch_helpers_round_trip():
    items = [{"id": f"batch-{index}", "path": str(DATA / "x" / str(index))} for index in range(3)]
    try:
        store.save_many("output", items)
        found = store.get_many("output", ["batch-2", "missing", "batch-0"])
        assert set(found) == {"batch-0", "batch-2"}
        assert found["batch-2"]["path"] == items[2]["path"]
    finally:
        with store.connect() as connection:
            connection.execute("DELETE FROM records WHERE kind='output' AND id LIKE 'batch-%'")
