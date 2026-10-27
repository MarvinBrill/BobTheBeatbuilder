import math
import pytest
from fastapi.testclient import TestClient
from app import store
from app.main import app
from app.map_catalog import star_rating


@pytest.fixture
def catalog():
    identifiers = []
    ratings = [4.0, 7.0, 3.99, 7.01, None, -1, float("nan"), 0.0] + [5.5] * 260
    for index, value in enumerate(ratings):
        identifier = f"star-filter-test-{index}"
        identifiers.append(identifier)
        store.save(
            "map",
            dict(
                id=identifier,
                title="Star Filter Fixture",
                artist="Artist",
                version="Insane",
                creator="Mapper",
                star_rating=value,
                map_path="private-map-path",
                audio_path="private-audio-path",
            ),
        )
    yield identifiers
    with store.connect() as connection:
        connection.executemany(
            "DELETE FROM records WHERE kind='map' AND id=?", [(i,) for i in identifiers]
        )


def test_inclusive_filter_and_select_all_match_every_page(catalog):
    client = TestClient(app)
    query = "q=STAR%20FILTER%20FIXTURE&stars_min=4&stars_max=7"
    selection = client.get(f"/api/maps/ids?{query}").json()
    assert selection["total"] == 262
    assert set(selection["ids"]) == {catalog[0], catalog[1], *catalog[8:]}
    pages = [
        client.get(f"/api/maps?{query}&offset={offset}&limit=100").json()
        for offset in (0, 100, 200)
    ]
    assert all(page["total"] == 262 for page in pages)
    assert [item["id"] for page in pages for item in page["items"]] == selection["ids"]
    assert all(4 <= item["star_rating"] <= 7 for page in pages for item in page["items"])
    assert all(
        "map_path" not in item and "audio_path" not in item
        for page in pages
        for item in page["items"]
    )
    assert client.get(f"/api/maps?{query}&offset=1000").json()["items"] == []


def test_unknown_ratings_stay_unknown_and_one_sided_bounds_work(catalog):
    client = TestClient(app)
    unfiltered = client.get("/api/maps?q=Star%20Filter%20Fixture&limit=250").json()
    # Unknown values appear without a filter but must never be interpreted as zero stars.
    low = client.get("/api/maps?q=Star%20Filter%20Fixture&stars_max=4").json()
    assert {item["id"] for item in low["items"]} == {catalog[0], catalog[2], catalog[7]}
    high = client.get("/api/maps/ids?q=Star%20Filter%20Fixture&stars_min=7").json()
    assert set(high["ids"]) == {catalog[1], catalog[3]}
    assert unfiltered["total"] == len(catalog)
    unknown_page = client.get("/api/maps?q=Star%20Filter%20Fixture&offset=250").json()
    for item in unknown_page["items"]:
        if item["id"] in catalog[4:7]:
            assert item["star_rating"] is None


@pytest.mark.parametrize(
    "query", ["stars_min=7&stars_max=4", "stars_min=-1", "stars_max=nan", "stars_min=inf"]
)
def test_invalid_star_bounds_rejected_for_both_endpoints(query):
    client = TestClient(app)
    for endpoint in ("maps", "maps/ids"):
        assert client.get(f"/api/{endpoint}?{query}").status_code == 400


def test_rating_normalization_uses_display_precision_and_never_estimates():
    assert star_rating(4.123456) == 4.12
    assert star_rating(0) == 0
    for value in (None, True, "5", -1, math.nan, math.inf):
        assert star_rating(value) is None


def test_reimport_preserves_known_rating_for_identical_content(tmp_path):
    from app.beatmaps import HitObject, write
    from app.library import register

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"same audio content")
    beatmap = tmp_path / "map.osu"
    write(
        beatmap,
        [HitObject(256, 192, 1000)],
        {
            "title": "Rating retention",
            "artist": "Test",
            "version": "Normal",
            "ar": 7,
            "cs": 4,
            "od": 5,
            "hp": 5,
        },
        "audio.wav",
        120,
        0,
    )
    original = register(beatmap, audio, "lazer", stars=5.67)
    try:
        imported = register(beatmap, audio, "archive")
        assert imported["id"] == original["id"]
        assert imported["star_rating"] == 5.67
        assert imported["star_source"] == "lazer"
        # A fresh lazer scan is authoritative, including a reset to unknown.
        assert register(beatmap, audio, "lazer", stars=-1)["star_rating"] is None
    finally:
        with store.connect() as connection:
            connection.execute("DELETE FROM records WHERE kind='map' AND id=?", (original["id"],))


@pytest.mark.parametrize("policy,rating", [("easiest", 2), ("middle", 5), ("hardest", 8)])
def test_one_difficulty_per_song_matches_all_pages_and_selection(monkeypatch, policy, rating):
    records = [
        dict(
            id=f"song-{song}-{stars}",
            title=f"Song {song}",
            artist="Artist",
            audio_hash=f"audio-{song}",
            star_rating=stars,
        )
        for stars in (2, 5, 8)
        for song in range(107)
    ]
    monkeypatch.setattr(store, "all_records", lambda kind: records)
    client = TestClient(app)
    query = f"per_song={policy}"
    selection = client.get(f"/api/maps/ids?{query}").json()
    assert selection["total"] == 107
    pages = [
        client.get(f"/api/maps?{query}&offset={offset}&limit=100").json() for offset in (0, 100)
    ]
    assert [item["id"] for page in pages for item in page["items"]] == selection["ids"]
    assert all(item["star_rating"] == rating for page in pages for item in page["items"])
    assert client.get("/api/maps?per_song=all").json()["total"] == 321


def test_song_choice_applies_after_bounds_and_search_with_lower_middle(monkeypatch):
    from app.map_catalog import matching_maps

    records = [
        dict(
            id=str(i),
            title="Song",
            artist="Artist",
            audio_hash="same",
            star_rating=rating,
            creator="Mapper" if i else "Other",
        )
        for i, rating in enumerate((2, 4, 6, 8, None))
    ]
    monkeypatch.setattr(store, "all_records", lambda _: records)
    assert [m["star_rating"] for m in matching_maps(per_song="middle")] == [4]
    assert [m["star_rating"] for m in matching_maps("Mapper", per_song="middle")] == [6]
    assert [m["star_rating"] for m in matching_maps(minimum=4, maximum=7, per_song="hardest")] == [
        6
    ]
    assert matching_maps(minimum=9, per_song="hardest") == []
    records[0]["star_rating"] = None
    assert len(matching_maps(per_song="all")) == 5


def test_grouping_keeps_filtered_bridge_and_stable_ties(monkeypatch):
    from app.map_catalog import matching_maps

    records = [
        dict(id="a", title="Title A", artist="Artist", audio_hash="one", star_rating=5),
        dict(id="bridge", title="Title A", artist="Artist", audio_hash="two", star_rating=1),
        dict(id="b", title="Title B", artist="Artist", audio_hash="two", star_rating=6),
        dict(id="c", title="Title B", artist="Artist", audio_hash="two", star_rating=6),
        dict(id="unknown", title="Unknown stars", artist="Artist", star_rating=None),
        dict(id="missing1", title="Untitled", artist="Unknown", star_rating=4),
        dict(id="missing2", title="Untitled", artist="Unknown", star_rating=4),
    ]
    monkeypatch.setattr(store, "all_records", lambda _: records)
    first = {m["id"] for m in matching_maps(minimum=4, per_song="hardest")}
    records.reverse()
    assert first == {m["id"] for m in matching_maps(minimum=4, per_song="hardest")}
    assert first == {"c", "missing1", "missing2"}
    assert matching_maps("Unknown stars", per_song="hardest") == []


def test_invalid_song_policy_rejected_by_both_endpoints():
    client = TestClient(app)
    for endpoint in ("maps", "maps/ids"):
        assert client.get(f"/api/{endpoint}?per_song=invalid").status_code == 400
