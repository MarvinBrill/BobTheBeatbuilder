from app import store


def test_all_records_drops_large_fields_before_parsing():
    store.save(
        "dataset",
        {"id": "payload-set", "name": "Set", "count": 2, "songs": 1, "maps": ["a", "b"]},
    )
    [record] = [item for item in store.all_records("dataset", drop=("maps",)) if item["id"] == "payload-set"]
    assert record == {"id": "payload-set", "name": "Set", "count": 2, "songs": 1}
    # Without `drop`, records stay complete for callers that need the maps.
    assert store.get("dataset", "payload-set")["maps"] == ["a", "b"]


def test_all_records_drop_ignores_missing_keys():
    store.save("model", {"id": "payload-model", "name": "M", "maps": 3})
    [record] = [
        item
        for item in store.all_records("model", drop=("training_song_hashes",))
        if item["id"] == "payload-model"
    ]
    assert record == {"id": "payload-model", "name": "M", "maps": 3}
