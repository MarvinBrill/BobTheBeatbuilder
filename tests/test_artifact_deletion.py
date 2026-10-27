import shutil
import threading
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from app import api as endpoints, store
from app import artifact_deletion
from app.config import DATA
from app.main import app
from app.jobs import Jobs
from app.ml.architecture import ARCHITECTURE, EVENT_SCHEMA
from app.ml.recovery import recover_training_runs


@pytest.fixture
def artifacts():
    records, paths = [], []

    def create(record_kind, **extra):
        kind = record_kind
        identifier = uuid4().hex
        item = dict(id=identifier, name=f"Deletion fixture {identifier}", **extra)
        if kind in ("model", "output"):
            folder = DATA / ("models" if kind == "model" else "outputs") / identifier
            folder.mkdir()
            paths.append(folder)
            names = (
                ("model.safetensors", "training.safetensors", "model.zip", "recovery.json")
                if kind == "model"
                else ("map.osu", "beatmap.osz", "audio.ogg", "preview.json")
            )
            for name in names:
                (folder / name).write_text("{}")
            if kind == "model":
                item.update(architecture=ARCHITECTURE, event_schema=EVENT_SCHEMA)
            else:
                item.update(
                    path=str(folder / "beatmap.osz"),
                    audio_path=str(folder / "audio.ogg"),
                    preview_path=str(folder / "preview.json"),
                )
        store.save(kind, item)
        records.append((kind, identifier))
        return item

    yield create
    with store.connect() as connection:
        connection.executemany("DELETE FROM records WHERE kind=? AND id=?", records)
    for path in paths:
        if path.exists():
            shutil.rmtree(path)


def test_delete_model_removes_checkpoints_and_its_history_only(artifacts):
    cancelled = artifacts("job", kind="training", status="cancelled")
    model = artifacts("model", job_id=cancelled["id"])
    completed = artifacts("job", status="completed", result={"model_id": model["id"]})
    other = artifacts("model")
    dataset = artifacts("dataset")
    output = artifacts("output", settings={"model_id": model["id"]})
    client = TestClient(app)
    response = client.delete(f"/api/models/{model['id']}")
    assert response.status_code == 200, response.text
    assert response.json()["removed_jobs"] == 2
    assert not (DATA / "models" / model["id"]).exists()
    recover_training_runs()
    with pytest.raises(ValueError):
        store.get("model", model["id"])
    for job in (cancelled, completed):
        with pytest.raises(ValueError):
            store.get("job", job["id"])
    assert store.get("model", other["id"])
    assert store.get("dataset", dataset["id"])
    assert store.get("output", output["id"])
    assert client.delete(f"/api/models/{model['id']}").status_code == 404


def test_delete_generated_map_removes_audio_preview_export_and_history(artifacts, tmp_path):
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"shared original")
    upload = artifacts("audio", path=str(source_audio))
    model = artifacts("model")
    output = artifacts("output", settings={"model_id": model["id"], "audio_id": upload["id"]})
    job = artifacts("job", status="completed", result={"output_id": output["id"]})
    other = artifacts("output")
    client = TestClient(app)
    assert client.delete(f"/api/outputs/{output['id']}").status_code == 200
    assert not (DATA / "outputs" / output["id"]).exists()
    assert source_audio.read_bytes() == b"shared original"
    assert store.get("audio", upload["id"]) and store.get("model", model["id"])
    assert (DATA / "outputs" / other["id"] / "beatmap.osz").exists()
    with pytest.raises(ValueError):
        store.get("job", job["id"])
    assert output["id"] not in [item["id"] for item in client.get("/api/state").json()["outputs"]]


def test_missing_files_and_standalone_archive_can_be_removed(artifacts):
    output = artifacts("output")
    shutil.rmtree(DATA / "outputs" / output["id"])
    archive = DATA / "outputs" / f"{output['id']}.osz"
    archive.write_bytes(b"archive")
    client = TestClient(app)
    assert client.delete(f"/api/outputs/{output['id']}").status_code == 200
    assert not archive.exists()
    model = artifacts("model")
    shutil.rmtree(DATA / "models" / model["id"])
    assert client.delete(f"/api/models/{model['id']}").status_code == 200


def test_delete_does_not_follow_paths_from_metadata(artifacts, tmp_path):
    source = tmp_path / "original.osz"
    source.write_bytes(b"keep")
    output = artifacts("output")
    store.save("output", {**output, "path": str(source)})
    client = TestClient(app)
    assert client.delete(f"/api/outputs/{output['id']}").status_code == 200
    assert source.read_bytes() == b"keep"
    for identifier in ("..", "not-an-id", "a" * 31, "C:drive"):
        assert client.delete(f"/api/models/{identifier}").status_code in (400, 404, 405)
        with pytest.raises(ValueError):
            artifact_deletion.delete_artifact("model", identifier)


def test_file_error_preserves_record_and_allows_retry(artifacts, monkeypatch):
    model = artifacts("model")
    job = artifacts("job", status="completed", result={"model_id": model["id"]})
    client = TestClient(app)
    with monkeypatch.context() as patch:

        def fail(_):
            raise PermissionError("File in use")

        patch.setattr(artifact_deletion.shutil, "rmtree", fail)
        assert client.delete(f"/api/models/{model['id']}").status_code == 409
    assert store.get("model", model["id"]) and store.get("job", job["id"])
    assert client.delete(f"/api/models/{model['id']}").status_code == 200


def test_linked_child_is_rejected_before_any_files_are_removed(artifacts, tmp_path):
    model = artifacts("model")
    target = tmp_path / "original"
    target.mkdir()
    (target / "keep").write_bytes(b"original")
    link = DATA / "models" / model["id"] / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation unavailable on this host")
    try:
        assert TestClient(app).delete(f"/api/models/{model['id']}").status_code == 400
        assert (target / "keep").read_bytes() == b"original"
        assert (link.parent / "model.safetensors").exists()
    finally:
        link.unlink()


def test_active_and_cancelled_but_not_finished_jobs_block_deletion(artifacts, monkeypatch):
    manager = Jobs()
    monkeypatch.setattr(endpoints, "jobs", manager)
    gate, entered = threading.Event(), threading.Event()

    def operation(context):
        entered.set()
        assert gate.wait(10)

    model = artifacts("model")
    job = manager.submit("test", "Active deletion guard", operation)
    try:
        assert entered.wait(10)
        queued = manager.submit("test", "Queued deletion guard", lambda context: None)
        manager.cancel(queued["id"])
        client = TestClient(app)
        assert client.delete(f"/api/models/{model['id']}").status_code == 409
        manager.cancel(job["id"])
        assert client.delete(f"/api/models/{model['id']}").status_code == 409
        assert store.get("model", model["id"])
    finally:
        gate.set()
        manager.executor.shutdown(wait=True)
    assert client.delete(f"/api/models/{model['id']}").status_code == 200


def test_generation_cannot_enqueue_a_model_deleted_during_request(artifacts, monkeypatch):
    model, audio = artifacts("model"), artifacts("audio")
    entered, release = threading.Event(), threading.Event()
    original = artifact_deletion.shutil.rmtree

    def slow_remove(path):
        entered.set()
        assert release.wait(10)
        original(path)

    monkeypatch.setattr(artifact_deletion.shutil, "rmtree", slow_remove)
    replies = {}

    def remove():
        replies["delete"] = TestClient(app).delete(f"/api/models/{model['id']}")

    def generate():
        replies["generate"] = TestClient(app).post(
            "/api/generate", json={"model_id": model["id"], "audio_id": audio["id"]}
        )

    deletion = threading.Thread(target=remove)
    generation = threading.Thread(target=generate)
    deletion.start()
    try:
        assert entered.wait(10)
        generation.start()
    finally:
        release.set()
        deletion.join(10)
        generation.join(10)
    assert replies["delete"].status_code == 200
    assert replies["generate"].status_code == 400
    assert not endpoints.jobs.events


def test_deletion_waits_for_cleanup_after_job_is_marked_completed(artifacts, monkeypatch):
    manager = Jobs()
    monkeypatch.setattr(endpoints, "jobs", manager)
    entered, release = threading.Event(), threading.Event()
    model = artifacts("model")

    def cleanup():
        entered.set()
        assert release.wait(10)

    job = manager.submit("test", "Cleanup deletion guard", lambda context: None, cleanup)
    try:
        assert entered.wait(10)
        assert store.get("job", job["id"])["status"] == "completed"
        assert TestClient(app).get("/api/state").json()["deletion_blocked"]
        assert TestClient(app).delete(f"/api/models/{model['id']}").status_code == 409
    finally:
        release.set()
        manager.executor.shutdown(wait=True)
    assert TestClient(app).delete(f"/api/models/{model['id']}").status_code == 200
    assert not TestClient(app).get("/api/state").json()["deletion_blocked"]
