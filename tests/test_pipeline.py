import io
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from fastapi.testclient import TestClient
from app.main import app
from app import store
from app.beatmaps import HitObject, write, parse
from app.config import DATA
from app.library import import_archive


class Context:
    def check(self):
        pass

    def report(self, *args, **kwargs):
        pass


def await_job(client, identifier, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = next(
            job for job in client.get("/api/state").json()["jobs"] if job["id"] == identifier
        )
        if job["status"] not in ("queued", "running"):
            assert job["status"] == "completed", job
            return job["result"]
        time.sleep(0.1)
    raise AssertionError("Job did not complete")


def test_full_api_training_generation_and_download(tmp_path):
    torch.set_num_threads(2)
    settings = {
        "title": "Synthetic rhythm",
        "artist": "Test",
        "version": "Normal",
        "ar": 7,
        "cs": 4,
        "od": 5,
        "hp": 5,
    }
    archives = []
    for track in range(2):
        folder = tmp_path / str(track)
        folder.mkdir()
        samples = np.zeros(22050 * 4, np.float32)
        for start in np.arange(0.25, 3.6, 0.5):
            position = int(start * 22050)
            phase = np.arange(2205) / 22050
            samples[position : position + 2205] = (
                np.sin(2 * np.pi * (440 + track * 80) * phase) * np.exp(-phase * 20)
            ).astype(np.float32)
        sf.write(folder / "audio.wav", samples, 22050)
        objects = [
            HitObject(80 + index * 40, 100 + index % 2 * 100, 250 + index * 500)
            for index in range(7)
        ]
        write(
            folder / "map.osu",
            objects,
            {**settings, "title": f"Synthetic rhythm {track}"},
            "audio.wav",
            120,
            250,
        )
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as output:
            output.write(folder / "map.osu", "map.osu")
            output.write(folder / "audio.wav", "audio.wav")
        archives.append(archive.getvalue())

    with TestClient(app) as client:
        for archive in archives:
            response = client.post("/api/import", files={"file": ("test.osz", archive)})
            assert response.status_code == 200, response.text
            assert await_job(client, response.json()["id"])["imported"] == 1
        maps = client.get("/api/maps").json()["items"]
        assert len(maps) == 2
        dataset_job = client.post(
            "/api/datasets", json={"ids": [item["id"] for item in maps], "name": "Test pairs"}
        ).json()
        dataset_id = await_job(client, dataset_job["id"])["dataset_id"]
        training_job = client.post(
            "/api/train",
            json={
                "dataset_id": dataset_id,
                "name": "Test model",
                "epochs": 1,
                "batch_size": 2,
                "device": "cpu",
                "model_size": "compact",
                "gradient_accumulation": 1,
            },
        ).json()
        model_id = await_job(client, training_job["id"])["model_id"]
        model = store.get("model", model_id)
        visible_model = next(
            item for item in client.get("/api/state").json()["models"] if item["id"] == model_id
        )
        assert visible_model["maps"] == 2
        assert "excluded_maps" not in visible_model
        assert model["trained_slider_geometry"] == 0
        assert model["best_epoch"] == 1
        assert model["training_songs"] == model["validation_songs"] == 1
        assert model["metrics"][0]["validation_loss"] is not None
        bundle = client.get(f"/api/models/{model_id}/download")
        with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
            assert set(archive.namelist()) == {"model.safetensors", "metadata.json"}
        imported_job = client.post(
            "/api/models/import", files={"file": ("model.zip", bundle.content)}
        ).json()
        imported_id = await_job(client, imported_job["id"])["model_id"]
        assert imported_id != model_id
        assert store.get("model", imported_id)["architecture"] == "phrase-mapper-v5"
        audio = client.post(
            "/api/audio", files={"file": ("new.wav", (tmp_path / "0/audio.wav").read_bytes())}
        ).json()
        job = client.post(
            "/api/generate",
            json={
                "model_id": model_id,
                "audio_id": audio["id"],
                **settings,
                "bpm": 120,
                "density": 4,
            },
        ).json()
        output_id = await_job(client, job["id"])["output_id"]
        preview = client.get(f"/api/outputs/{output_id}/preview").json()
        assert preview["objects"]
        assert not any(obj["slider"] for obj in preview["objects"])
        for index, obj in enumerate(preview["objects"]):
            assert 0 <= obj["x"] <= 512 and 0 <= obj["y"] <= 384
            if index + 1 < len(preview["objects"]):
                assert obj["time"] + obj["duration"] < preview["objects"][index + 1]["time"]
        result = client.get(f"/api/outputs/{output_id}/download")
        with zipfile.ZipFile(io.BytesIO(result.content)) as archive:
            assert set(archive.namelist()) == {"map.osu", "audio.ogg"}
            saved = tmp_path / "generated.osu"
            saved.write_bytes(archive.read("map.osu"))
            assert len(parse(saved).objects) == len(preview["objects"])
        export_job = client.post("/api/export", json={"ids": [item["id"] for item in maps]}).json()
        export_id = await_job(client, export_job["id"])["output_id"]
        archive = client.get(f"/api/outputs/{export_id}/download")
        reimport = client.post("/api/import", files={"file": ("again.osz", archive.content)}).json()
        assert await_job(client, reimport["id"])["imported"] == 2


def test_api_rejects_cross_origin_and_bad_parameters():
    with TestClient(app) as client:
        response = client.post(
            "/api/scan", json={"path": "."}, headers={"origin": "https://malicious.example"}
        )
        assert response.status_code == 403
        assert client.post("/api/generate", json={"ar": 99}).status_code == 422
        assert client.get("/api/state", headers={"host": "malicious.example"}).status_code == 400


def test_zip_slip_is_rejected():
    archive = DATA / "uploads/unsafe.osz"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.osu", "bad")
    import pytest

    with pytest.raises(ValueError, match="escapes"):
        import_archive(archive, Context())
    assert not (DATA / "escape.osu").exists()


def test_storage_paths_are_portable():
    item = {"id": "portable", "path": str(DATA / "outputs" / "portable.osz")}
    store.save("output", item)
    with store.connect() as connection:
        body = connection.execute("SELECT body FROM records WHERE id='portable'").fetchone()[0]
    assert "data://outputs/portable.osz" in body
    assert store.get("output", "portable")["path"] == item["path"]
