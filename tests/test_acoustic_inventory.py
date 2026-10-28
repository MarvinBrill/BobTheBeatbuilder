import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from app.ml.corpus_audit import acoustic_candidates, acoustic_signature, compare_acoustic_signatures
from app.ml.split import song_groups

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmarks/acoustic_inventory.py"
SPEC = importlib.util.spec_from_file_location("acoustic_inventory", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def reservation_fixture():
    records = [
        dict(
            id="dev", audio_hash="a", artist="Artist", title="Development", audio_path="missing-dev"
        ),
        dict(
            id="held", audio_hash="b", artist="Artist", title="Reserved", audio_path="must-not-open"
        ),
        dict(
            id="alternate",
            audio_hash="c",
            artist="Artist",
            title="Reserved",
            audio_path="must-not-open-either",
        ),
    ]
    groups = song_groups(records)
    rows = [
        {**record, "map_id": record["id"], "song_group": group}
        for record, group in zip(records, groups)
    ]
    reservation = dict(
        schema="song-reservation-v1",
        final_test_map_ids=["held"],
        final_test_groups=[groups[1]],
        development_map_ids=["dev", "alternate"],
    )
    return records, dict(rows=rows, reservation=reservation), reservation


def test_development_selection_excludes_entire_reserved_group_before_any_file_read():
    records, inventory, reservation = reservation_fixture()
    chosen, excluded = audit.development_audio(records, inventory, reservation)
    assert [item["audio_hash"] for item in chosen] == ["a"]
    assert excluded["excluded_holdout_audio_hashes"] == ["b", "c"]
    # A catalog rename that connects development to a reserved group must also be excluded.
    records[0]["title"] = "Reserved"
    chosen, excluded = audit.development_audio(records, inventory, reservation)
    assert chosen == [] and excluded["excluded_holdout_audio_hashes"] == ["a", "b", "c"]


def test_new_and_changed_audio_are_not_silently_added_to_development():
    records, inventory, reservation = reservation_fixture()
    records.append(
        dict(id="new", audio_hash="new", artist="Other", title="New", audio_path="missing")
    )
    records[0]["audio_hash"] = "changed"
    chosen, excluded = audit.development_audio(records, inventory, reservation)
    assert chosen == []
    assert excluded["changed_catalog_map_ids_not_analysed"] == ["dev"]
    with pytest.raises(ValueError, match="differ"):
        audit.development_audio(records, inventory, {**reservation, "final_test_groups": []})


def test_observation_uses_readonly_cache_and_retains_only_compact_signature(tmp_path):
    audio = tmp_path / "audio.bin"
    audio.write_bytes(b"verified source")
    audio_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
    cache = tmp_path / "cache"
    cache.mkdir()
    mel = np.random.default_rng(42).uniform(-1, 1, (320, 64)).astype(np.float32)
    target = cache / f"mel-v2-{audio_hash}.npy"
    np.save(target, mel, allow_pickle=False)
    before = target.read_bytes()

    def forbidden(*args):
        raise AssertionError("A valid existing cache must not be recomputed")

    result = audit.observe_audio(
        dict(map_id="one", audio_hash=audio_hash, audio_path=str(audio)),
        tmp_path,
        [cache],
        forbidden,
    )
    assert result["status"] == "analysed" and len(result["signature"]) == 256
    assert result["feature_source"] == "existing_cache" and "mel" not in result
    assert target.read_bytes() == before
    assert np.allclose(result["signature"], acoustic_signature(mel))


def test_invalid_cache_recomputes_without_overwriting_production_cache(tmp_path):
    audio = tmp_path / "audio.bin"
    audio.write_bytes(b"source")
    audio_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
    target = tmp_path / f"mel-v2-{audio_hash}.npy"
    target.write_bytes(b"damaged")
    mel = np.random.default_rng(3).uniform(-1, 1, (320, 64)).astype(np.float32)
    result = audit.observe_audio(
        dict(map_id="one", audio_hash=audio_hash, audio_path=str(audio)),
        tmp_path,
        [tmp_path],
        lambda path, key: mel,
    )
    assert result["feature_source"] == "isolated_recompute"
    assert result["invalid_caches_ignored"] == [str(target)]
    assert target.read_bytes() == b"damaged"
    changed = audit.observe_audio(
        dict(map_id="one", audio_hash="different", audio_path=str(audio)),
        tmp_path,
        [tmp_path],
        lambda *args: pytest.fail("Changed audio must not be decoded"),
    )
    assert changed["status"] == "error" and "no longer matches" in changed["error"]


def test_compact_pairing_matches_existing_audit_and_rejects_duration_mismatch():
    mel = np.random.default_rng(4).uniform(-1, 1, (320, 64)).astype(np.float32)
    prepared = [
        dict(map_id="one", audio_hash="a", mel=mel),
        dict(map_id="two", audio_hash="b", mel=mel.copy()),
    ]
    signatures = [(item, len(item["mel"]), acoustic_signature(item["mel"])) for item in prepared]
    assert compare_acoustic_signatures(signatures) == acoustic_candidates(prepared)
    assert (
        compare_acoustic_signatures(
            [(prepared[0], 320, signatures[0][2]), (prepared[1], 330, signatures[1][2])]
        )
        == []
    )
    assert acoustic_signature(np.zeros((320, 64), np.float32)) is None
