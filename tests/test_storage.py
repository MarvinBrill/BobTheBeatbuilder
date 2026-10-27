import io
import os
import threading
import zipfile
from pathlib import Path
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from app import api, artifacts, artifact_deletion, library, store
from app import storage_files, storage_inventory, storage_management
from app.jobs import Jobs, Cancelled
from app.main import app


@pytest.fixture
def storage(tmp_path, monkeypatch):
    root = tmp_path / "managed"
    for name in (
        "models",
        "outputs",
        "datasets",
        "uploads",
        "cache",
        "library",
        "tmp",
        "snapshots",
    ):
        (root / name).mkdir(parents=True)
    for module in (
        api,
        artifacts,
        artifact_deletion,
        library,
        store,
        storage_files,
        storage_inventory,
        storage_management,
    ):
        monkeypatch.setattr(module, "DATA", root)
    manager = Jobs()
    monkeypatch.setattr(api, "jobs", manager)
    yield root, TestClient(app), manager
    manager.executor.shutdown(wait=True)


def record(root, kind, **extra):
    identifier = uuid4().hex
    folder = root / storage_inventory.FOLDERS[kind]
    if kind == "audio":
        path = folder / f"{identifier}.wav"
        path.write_bytes(b"audio")
    else:
        path = folder / identifier
        path.mkdir()
        (path / "owned.bin").write_bytes(b"owned")
    store.save(kind, {"id": identifier, "name": kind, "path": str(path), **extra})
    return identifier, path


def test_inventory_accounts_every_byte_including_hidden_models_and_orphans(storage):
    root, client, _ = storage
    record(root, "model", architecture="obsolete")
    record(root, "dataset")
    (root / "outputs" / "unfinished.osz").write_bytes(b"orphan")
    (root / "tests").mkdir()
    (root / "tests" / "checkpoint").write_bytes(b"test data")
    (root / "venv").mkdir()
    (root / "venv" / "runtime").write_bytes(b"keep")
    report = client.get("/api/storage").json()
    assert {"model", "dataset", "orphan", "development", "runtime"} <= {
        row["kind"] for row in report["entries"]
    }
    assert report["total_bytes"] == sum(storage_files.measure(p) for p in root.iterdir())
    assert not next(row for row in report["entries"] if row["id"] == "venv")["removable"]
    assert client.delete("/api/storage/runtime/venv").status_code == 400
    orphan = next(row for row in report["entries"] if row["kind"] == "orphan")
    assert client.delete(f"/api/storage/orphan/{orphan['id']}").status_code == 200
    assert not (root / "outputs" / "unfinished.osz").exists()


def test_dataset_audio_deletion_preserves_other_assets_and_clears_analysis(storage, tmp_path):
    root, client, _ = storage
    dataset, folder = record(root, "dataset")
    model, model_folder = record(root, "model", dataset_id=dataset)
    audio, upload = record(root, "audio")
    output, output_folder = record(root, "output", settings={"audio_id": audio})
    original = tmp_path / "osu-original.mp3"
    original.write_bytes(b"original")
    cache = root / "cache" / "mel-v2-example.npy"
    cache.write_bytes(b"cached")
    job = {"id": uuid4().hex, "status": "completed", "result": {"dataset_id": dataset}}
    store.save("job", job)
    assert client.delete(f"/api/storage/dataset/{dataset}").status_code == 200
    assert not folder.exists() and not cache.exists()
    assert store.get("model", model) and model_folder.exists()
    with pytest.raises(ValueError):
        store.get("job", job["id"])
    assert client.delete(f"/api/storage/audio/{audio}").status_code == 200
    assert not upload.exists() and output_folder.exists() and store.get("output", output)
    assert original.read_bytes() == b"original"


def test_cache_limit_is_applied_after_job_and_zero_disables_persistence(storage):
    root, client, manager = storage
    assert client.post("/api/storage/settings", json={"cache_limit_gib": 0}).status_code == 200

    def operation(context):
        (root / "cache" / "mel-v2-job.npy").write_bytes(b"computed")
        (root / "cache" / "unrelated").write_bytes(b"keep")

    manager.submit("test", "cache lifecycle", operation)
    manager.executor.shutdown(wait=True)
    assert not manager.events
    assert not (root / "cache" / "mel-v2-job.npy").exists()
    assert (root / "cache" / "unrelated").exists()
    assert client.post("/api/storage/settings", json={"cache_limit_gib": -1}).status_code == 422


def test_all_cleanup_actions_block_during_jobs_and_uploads(storage):
    root, client, manager = storage
    dataset, _ = record(root, "dataset")
    for kind in ("uploads", "events"):
        if kind == "uploads":
            manager.uploads = 1
        else:
            manager.events["pending"] = threading.Event()
        assert client.get("/api/storage").json()["blocked"]
        assert client.delete(f"/api/storage/dataset/{dataset}").status_code == 409
        assert client.delete("/api/storage/cleanup/cache").status_code == 409
        assert client.post("/api/storage/settings", json={"cache_limit_gib": 0}).status_code == 409
        manager.uploads = 0
        manager.events.clear()


def test_linked_tree_refused_without_deleting_any_files(storage, tmp_path):
    root, client, _ = storage
    outside = tmp_path / "original"
    outside.mkdir()
    (outside / "keep").write_bytes(b"keep")
    link = root / "cache" / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    (root / "cache" / "keep-until-validation").write_bytes(b"cached")
    try:
        assert client.delete("/api/storage/cleanup/cache").status_code == 400
        assert (outside / "keep").exists()
        assert (root / "cache" / "keep-until-validation").exists()
    finally:
        link.unlink()


def test_catalog_reset_does_not_delete_external_originals_or_training_copies(storage, tmp_path):
    root, client, _ = storage
    original = tmp_path / "original.osu"
    original.write_bytes(b"keep")
    imported = root / "library" / uuid4().hex
    imported.mkdir()
    (imported / "map.osu").write_bytes(b"map")
    store.save("map", {"id": "external", "map_path": str(original)})
    store.save("map", {"id": "imported", "map_path": str(imported / "map.osu")})
    _, dataset = record(root, "dataset")
    assert client.delete("/api/storage/catalog/all").status_code == 200
    assert store.count("map") == 0 and not imported.exists()
    assert original.exists() and dataset.exists()


class Context:
    def check(self):
        pass

    def report(self, *args, **kwargs):
        pass


def test_cancelled_export_has_no_archive_or_record(storage):
    root, _, _ = storage

    class Cancel(Context):
        def report(self, *args, **kwargs):
            raise Cancelled()

    with pytest.raises(Cancelled):
        library.export_selection(["any"], Cancel())
    assert not list((root / "outputs").iterdir()) and store.count("output") == 0


def test_failed_import_rolls_back_files_and_partial_registration(storage, monkeypatch):
    root, _, _ = storage
    archive = root / "uploads" / "test.osz"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("map.osu", "map")

    def fail(folder, context):
        store.save("map", {"id": "partial", "map_path": str(folder / "map.osu")})
        raise Cancelled()

    monkeypatch.setattr(library, "scan_songs", fail)
    with pytest.raises(Cancelled):
        library.import_archive(archive, Context())
    assert not list((root / "library").iterdir())
    assert not archive.exists() and store.count("map") == 0


def test_duplicate_import_does_not_accumulate_copies(storage, monkeypatch):
    root, _, _ = storage

    def scan(folder, context):
        store.save(
            "map",
            {
                "id": "same",
                "map_path": str(folder / "map.osu"),
                "audio_path": str(folder / "audio.mp3"),
            },
        )
        return {"imported": 1}

    monkeypatch.setattr(library, "scan_songs", scan)
    for _ in range(2):
        archive = root / "uploads" / f"{uuid4().hex}.osz"
        with zipfile.ZipFile(archive, "w") as out:
            out.writestr("map.osu", "map")
            out.writestr("audio.mp3", "audio")
            out.writestr("unused.mp3", "unused")
        library.import_archive(archive, Context())
    assert store.count("map") == 1
    assert len(list((root / "library").iterdir())) == 1
    assert sorted(p.name for p in (root / "library").rglob("*") if p.is_file()) == [
        "audio.mp3",
        "map.osu",
    ]


def test_real_upload_is_registered_as_busy_until_record_is_saved(storage, monkeypatch):
    root, client, manager = storage
    save = store.save

    def guarded(kind, item):
        if kind == "audio":
            assert manager.uploads == 1
            assert client.delete("/api/storage/cleanup/tmp").status_code == 409
        return save(kind, item)

    monkeypatch.setattr(store, "save", guarded)
    response = client.post("/api/audio", files={"file": ("sound.wav", b"audio", "audio/wav")})
    assert response.status_code == 200 and manager.uploads == 0
    assert client.delete(f"/api/storage/audio/{response.json()['id']}").status_code == 200
    assert not list((root / "uploads").iterdir())


def test_history_retention_and_manual_clear_preserve_models(storage):
    root, client, _ = storage
    model, _ = record(root, "model")
    for index in range(105):
        store.save("job", {"id": str(index), "status": "completed"})
    storage_files.prune_job_history()
    assert store.count("job") == 100
    assert client.delete("/api/storage/history/all").status_code == 200
    assert store.count("job") == 0 and store.get("model", model)


def test_cache_limit_evicts_oldest_analysis_files_only(storage):
    root, client, _ = storage
    old = root / "cache" / "labels-v2-old.npy"
    recent = root / "cache" / "mel-v2-recent.npy"
    old.write_bytes(bytes(1024))
    recent.write_bytes(bytes(1024))
    os.utime(old, (1, 1))
    (root / "cache" / "dependency").write_bytes(b"keep")
    assert (
        client.post("/api/storage/settings", json={"cache_limit_gib": 1024 / 1024**3}).status_code
        == 200
    )
    assert not old.exists() and recent.exists()
    assert (root / "cache" / "dependency").exists()


def test_failed_cache_cleanup_does_not_discard_model_record(storage, monkeypatch):
    root, client, _ = storage
    model, folder = record(root, "model")
    cache = root / "cache" / "mel-v2-locked.npy"
    cache.write_bytes(b"cached")
    original = Path.unlink

    def fail(path, *args, **kwargs):
        if path == cache:
            raise PermissionError("locked")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail)
        assert client.delete(f"/api/storage/model/{model}").status_code == 409
    assert folder.exists() and store.get("model", model)
    assert client.delete(f"/api/storage/model/{model}").status_code == 200
