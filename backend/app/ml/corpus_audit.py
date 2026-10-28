"""Read-only corpus evidence and conservative split reservations, never quality labels."""

import hashlib
import json
import numpy as np
from .split import song_groups, identity

SCHEMA = "corpus-audit-v1"
SPLIT_SCHEMA = "song-reservation-v1"


def historical_exposure(models, datasets=()):
    exposed, unknown = set(), []
    datasets = {item["id"]: item for item in datasets}
    for model in models:
        hashes = set().union(
            *(
                set(model.get(f"{part}_song_hashes", []))
                for part in ("training", "validation", "test")
            )
        )
        dataset = datasets.get(model.get("dataset_id"))
        if dataset:
            hashes.update(item["audio_hash"] for item in dataset["maps"])
        if not hashes:
            unknown.append(model["id"])
        exposed.update(hashes)
    return exposed, unknown


def style_proxies(events):
    if not len(events):
        return {}
    gaps = np.diff(events[:, 0])
    jumps = events[1:, 39] * 512
    return {
        "slider_fraction": float(events[:, 9].mean()),
        "short_interval_fraction": float(np.mean(gaps <= 0.18)) if len(gaps) else 0,
        "long_gap_fraction": float(np.mean(gaps >= 1)) if len(gaps) else 0,
        "jump_p95_pixels": float(np.percentile(jumps, 95)) if len(jumps) else 0,
        "turn_cosine_mean": float(events[2:, 41].mean()) if len(events) > 2 else None,
        "note": "Uncertain measured proxies; no instrument, gameplay tag or true difficulty labels.",
    }


def acoustic_signature(mel):
    """Compact 32-by-8 whole-song summary; no signature for uninformative audio."""
    mel = np.asarray(mel)
    if len(mel) < 32 or float(mel.std()) < 0.01:
        return None
    edges = np.linspace(0, len(mel), 33).astype(int)
    pooled = np.stack(
        [mel[a:b].mean(0).reshape(8, -1).mean(1) for a, b in zip(edges[:-1], edges[1:])]
    )
    vector = (pooled - pooled.mean()).reshape(-1)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-6 else None


def compare_acoustic_signatures(fingerprints):
    """Similarity >= .995 and duration difference below 1%; hints only, never merges."""
    candidates = []
    for index, (left, frames, vector) in enumerate(fingerprints):
        for right, other_frames, other_vector in fingerprints[index + 1 :]:
            if left["audio_hash"] == right["audio_hash"] or abs(frames / other_frames - 1) >= 0.01:
                continue
            similarity = float(vector @ other_vector)
            if similarity >= 0.995:
                candidates.append(
                    dict(
                        left=left["map_id"],
                        right=right["map_id"],
                        similarity=similarity,
                        status="manual_review_required",
                    )
                )
    return candidates


def acoustic_candidates(prepared):
    """Conservative whole-song spectral similarity hints; never automatically merge."""
    fingerprints = []
    for item in prepared:
        vector = acoustic_signature(item["mel"])
        if vector is not None:
            fingerprints.append((item, len(item["mel"]), vector))
    return compare_acoustic_signatures(fingerprints)


def inventory(records, models=(), datasets=(), prepared=(), issues=(), seed=42):
    groups = song_groups(records)
    exposed, unknown = historical_exposure(models, datasets)
    prepared_by_id = {item["map_id"]: item for item in prepared}
    rejected = {item["map_id"]: item["reason"] for item in issues}
    rows, grouped = [], {}
    for record, group in zip(records, groups):
        identifier = record.get("map_id", record.get("id"))
        item = prepared_by_id.get(identifier)
        row = dict(
            map_id=identifier,
            song_group=group,
            audio_hash=record["audio_hash"],
            artist=record.get("artist"),
            title=record.get("title"),
            metadata_identity=identity(record),
            difficulty=record.get("difficulty", {}),
            star_rating=record.get("star_rating"),
            duration_seconds=record.get("duration"),
            prior_exposure=record["audio_hash"] in exposed,
            masked_sliders=int(item["masked_geometry"]) if item else None,
            excluded_reason=rejected.get(identifier),
            style_proxies=style_proxies(item["events"]) if item else {},
            map_observations=record.get("map_observations"),
        )
        rows.append(row)
        grouped.setdefault(group, []).append(row)
    # Metadata-connected alternate encodings inherit the entire group's exposure.
    fresh = sorted(
        group for group, members in grouped.items() if not any(m["prior_exposure"] for m in members)
    )
    ordered = sorted(
        fresh, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
    )
    count = max(1, round(len(ordered) * 0.1)) if len(ordered) >= 5 else 0
    reserved = set(ordered[:count])
    reservation = {
        "schema": SPLIT_SCHEMA,
        "seed": seed,
        "final_test_groups": sorted(reserved),
        "final_test_map_ids": sorted(
            row["map_id"] for row in rows if row["song_group"] in reserved
        ),
        "development_map_ids": sorted(
            row["map_id"] for row in rows if row["song_group"] not in reserved
        ),
        "historical_exposure_complete": not unknown,
        "unresolved_historical_models": unknown,
        "status": (
            "unavailable_all_groups_previously_exposed"
            if not fresh
            else (
                "unavailable_too_few_unexposed_groups"
                if not reserved
                else "reserved_unrun" if not unknown else "reserved_unrun_history_incomplete"
            )
        ),
    }
    signature = hashlib.sha256(
        json.dumps(
            sorted((row["map_id"], row["audio_hash"], row["song_group"]) for row in rows)
        ).encode()
    ).hexdigest()
    return {
        "schema": SCHEMA,
        "signature": signature,
        "maps": len(rows),
        "song_groups": len(grouped),
        "unique_audio_hashes": len({row["audio_hash"] for row in rows}),
        "metadata_missing_maps": sum(row["metadata_identity"] is None for row in rows),
        "prior_exposed_groups": sum(
            any(row["prior_exposure"] for row in members) for members in grouped.values()
        ),
        "prepared_maps": len(prepared_by_id),
        "rows": rows,
        "acoustic_duplicate_candidates": acoustic_candidates(prepared),
        "acoustic_audit_coverage": len(prepared),
        "reservation": reservation,
        "limitations": [
            "No acoustic certainty for unanalysed audio or partial duplicates.",
            "Ranked status, popularity and difficulty names are not style labels.",
            "Song groups do not measure genre or musical diversity; those labels need annotation.",
        ],
    }


def assert_disjoint_groups(records, partitions):
    assignment = {identifier: name for name, ids in partitions.items() for identifier in ids}
    if sum(map(len, partitions.values())) != len(assignment):
        raise ValueError("A map occurs in more than one split")
    groups = {}
    for record, group in zip(records, song_groups(records)):
        name = assignment.get(record.get("map_id", record.get("id")))
        if name is not None:
            groups.setdefault(group, set()).add(name)
    if any(len(names) > 1 for names in groups.values()):
        raise ValueError("Audio or metadata duplicates cross split boundaries")
