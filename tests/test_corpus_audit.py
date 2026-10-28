import numpy as np
import pytest
from app.ml.corpus_audit import inventory, assert_disjoint_groups, acoustic_candidates
from app.ml.split import song_groups


def records():
    return [
        dict(id=str(i), audio_hash=str(i), artist="Artist", title=f"Song {i}") for i in range(20)
    ]


def test_group_ids_and_reservation_are_invariant_to_record_order():
    maps = records()
    maps.append(dict(id="alternate", audio_hash="encoding2", artist="artist", title="SONG 1"))
    normal = dict(zip([m["id"] for m in maps], song_groups(maps)))
    reverse = dict(zip([m["id"] for m in reversed(maps)], song_groups(list(reversed(maps)))))
    assert normal == reverse and normal["1"] == normal["alternate"]
    left = inventory(maps)
    right = inventory(list(reversed(maps)))
    assert left["signature"] == right["signature"]
    assert left["reservation"] == right["reservation"]


def test_exposure_union_includes_previous_tests_and_transitive_alt_audio():
    maps = records()
    maps.append(dict(id="alternate", audio_hash="other", artist="Artist", title="Song 1"))
    models = [
        dict(
            id="old",
            training_song_hashes=["0"],
            validation_song_hashes=["other"],
            test_song_hashes=["2"],
        )
    ]
    report = inventory(maps, models)
    assert report["prior_exposed_groups"] == 3
    reserved = report["reservation"]["final_test_map_ids"]
    assert not {"0", "1", "2", "alternate"} & set(reserved)
    assert report["reservation"]["status"] == "reserved_unrun"
    incomplete = inventory(maps, [dict(id="missing")])
    assert incomplete["reservation"]["status"] == "reserved_unrun_history_incomplete"


def test_split_audit_rejects_hash_and_metadata_leakage():
    maps = [
        dict(id="one", audio_hash="a", artist="artist", title="song"),
        dict(id="two", audio_hash="b", artist="artist", title="song"),
    ]
    with pytest.raises(ValueError, match="duplicates"):
        assert_disjoint_groups(maps, {"train": ["one"], "test": ["two"]})
    with pytest.raises(ValueError, match="more than one"):
        assert_disjoint_groups(maps, {"train": ["one"], "test": ["one"]})
    assert_disjoint_groups(maps, {"train": ["one", "two"], "test": []})


def test_acoustic_hints_do_not_merge_or_claim_reencoded_audio_is_certain():
    mel = np.random.default_rng(42).normal(size=(320, 64)).astype(np.float32)
    prepared = [
        dict(map_id="one", audio_hash="a", mel=mel),
        dict(map_id="two", audio_hash="b", mel=mel + 0.001),
    ]
    candidates = acoustic_candidates(prepared)
    assert len(candidates) == 1 and candidates[0]["status"] == "manual_review_required"
    prepared[1]["mel"] = np.zeros_like(mel)
    assert acoustic_candidates(prepared) == []
