import hashlib
import numpy as np
import pytest
from app import library, store, api
from app.beatmaps import HitObject, write
from app.ml import sequence_data, preparation_cache
from app.main import app
from fastapi.testclient import TestClient


class Context:
    def check(self):
        pass

    def report(self, *args, **kwargs):
        pass


def test_preparation_reuses_verified_targets_and_shared_audio(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio fixture")
    maps = []
    for index in range(2):
        path = tmp_path / f"{index}.osu"
        write(
            path,
            [HitObject(100 + index, 100, 500), HitObject(200, 100, 1000)],
            dict(title="test", artist="test", version=str(index), ar=8, cs=4, od=6, hp=5),
            "audio.wav",
            120,
            0,
        )
        maps.append(
            dict(
                id=str(index),
                audio_path=str(audio),
                map_path=str(path),
                audio_hash=library.digest(audio),
                title="test",
                artist="test",
            )
        )
    calls = []
    monkeypatch.setattr(
        sequence_data,
        "features",
        lambda *args: calls.append(args) or np.zeros((300, 64), np.float32),
    )
    first = sequence_data.prepare(maps, Context())
    assert len(calls) == 1
    parse = sequence_data.parse
    monkeypatch.setattr(
        sequence_data, "parse", lambda *args: pytest.fail("Warm cache should skip parsing")
    )
    warm = sequence_data.prepare(maps, Context())
    for left, right in zip(first, warm):
        for key in ("mel", "events", "labels", "condition", "summary"):
            np.testing.assert_array_equal(left[key], right[key])
        assert left["song"] == right["song"]
    # Valid-shaped but corrupted data must not become training targets.
    key = hashlib.sha256((first[0]["map_hash"] + first[0]["audio_hash"]).encode()).hexdigest()
    manifest, files = preparation_cache.paths(key)
    del first, warm
    corrupted = np.load(files["events"], mmap_mode="r+")
    corrupted[0, 1] += 0.1
    corrupted.flush()
    del corrupted
    assert preparation_cache.load(key, 300) is None
    monkeypatch.setattr(sequence_data, "parse", parse)
    fixed = sequence_data.prepare(maps, Context())
    assert fixed[0]["events"][0, 1] == np.float32(100 / 512)
    manifest.unlink()
    assert preparation_cache.load(key, 300) is None


def test_state_etag_skips_full_state_and_invalidates_on_all_mutations(monkeypatch):
    client = TestClient(app)
    response = client.get("/api/state")
    assert response.status_code == 200
    tag = response.headers["etag"]
    original = api.state
    monkeypatch.setattr(
        api, "state", lambda: pytest.fail("Unchanged requests must skip state assembly")
    )
    response = client.get("/api/state", headers={"If-None-Match": tag})
    assert response.status_code == 304 and response.content == b""
    monkeypatch.setattr(api, "state", original)
    for action in ["insert", "update", "delete"]:
        with store.connect() as connection:
            if action == "insert":
                connection.execute("INSERT INTO records VALUES ('settings','etag-test','{}')")
            elif action == "update":
                connection.execute("UPDATE records SET body='{}' WHERE id='etag-test'")
            else:
                connection.execute("DELETE FROM records WHERE id='etag-test'")
        response = client.get("/api/state", headers={"If-None-Match": tag})
        assert response.status_code == 200 and response.headers["etag"] != tag
        tag = response.headers["etag"]
    api.jobs.uploads += 1
    try:
        response = client.get("/api/state", headers={"If-None-Match": tag})
        assert response.status_code == 200 and response.json()["deletion_blocked"]
    finally:
        api.jobs.uploads -= 1


def test_parallel_preparation_preserves_order_groups_and_exclusion_order(monkeypatch):
    from app.ml.preparation_workers import prepare_parallel
    from app.ml.split import song_groups

    monkeypatch.setenv("BOB_PREPARE_WORKERS", "2")
    records = [
        dict(id=str(i), audio_hash=str(i % 3), title=f"song {i % 3}", artist="test")
        for i in range(9)
    ]
    groups = song_groups(records)

    def operation(items, context, issues):
        context.check()
        result = []
        for item in items:
            if item["id"] in ("2", "3"):
                issues.append({"map_id": item["id"], "reason": "fixture"})
            else:
                result.append({"map_id": item["id"], "song": "local-group"})
        return result

    issues = []
    result = prepare_parallel(records, Context(), issues, operation)
    assert [item["map_id"] for item in result] == ["0", "1", "4", "5", "6", "7", "8"]
    assert [item["song"] for item in result] == [groups[int(item["map_id"])] for item in result]
    assert [item["map_id"] for item in issues] == ["2", "3"]


def test_parallel_preparation_honors_cancellation(monkeypatch):
    from app.ml.preparation_workers import prepare_parallel
    from app.jobs import Cancelled

    monkeypatch.setenv("BOB_PREPARE_WORKERS", "2")

    class Cancel(Context):
        def check(self):
            raise Cancelled()

    def operation(items, context, issues):
        context.check()

    records = [dict(id=str(i), audio_hash=str(i), title=str(i), artist="test") for i in range(3)]
    with pytest.raises(Cancelled):
        prepare_parallel(records, Cancel(), [], operation)
