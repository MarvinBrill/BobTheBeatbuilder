import hashlib
import json
import zipfile
import numpy as np
from .. import store
from .architecture import ModelConfig, ARCHITECTURE, FORMAT_VERSION, EVENT_SCHEMA


def configuration(size):
    """New models always use the style-v2 conditioning (stars, patterns, mapper)."""
    if size == "compact":
        return ModelConfig(width=128, heads=4, audio_layers=4, decoder_layers=3, style_version=2)
    if size == "large":
        return ModelConfig(width=384, heads=8, audio_layers=8, decoder_layers=6, style_version=2)
    return ModelConfig(style_version=2)


def split_maps(prepared, seed):
    songs = sorted({item["song"] for item in prepared})
    np.random.default_rng(seed).shuffle(songs)
    validation_count = max(1, round(len(songs) * 0.15)) if len(songs) > 1 else 0
    test_count = max(1, round(len(songs) * 0.1)) if len(songs) >= 5 else 0
    validation = set(songs[:validation_count])
    test = set(songs[validation_count : validation_count + test_count])
    return (
        [item for item in prepared if item["song"] not in validation | test],
        [item for item in prepared if item["song"] in validation],
        [item for item in prepared if item["song"] in test],
    )


def dataset_signature(prepared):
    return hashlib.sha256(
        json.dumps(
            sorted((item["map_id"], item["map_hash"], item["audio_hash"], item["song"]) for item in prepared),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def metadata_for(identifier, settings, cfg, prepared, split, issues):
    training, validation, test = split
    warnings = []
    if len({item["song"] for item in prepared}) < 20:
        warnings.append("Weniger als 20 Song-Gruppen: nur ein Funktionstest, hohes Overfitting-Risiko.")
    if not validation:
        warnings.append("Keine unabhängigen Prüfsongs verfügbar.")
    if not test:
        warnings.append("Zu wenige Song-Gruppen für einen separaten Abschlusstest.")
    if issues:
        warnings.append(
            f"{len(issues)} ungeeignete Maps ausgeschlossen; Gründe stehen in den Modellmetadaten."
        )
    masked = sum(item["masked_geometry"] for item in prepared)
    if masked:
        warnings.append(
            f"{masked} Slider außerhalb der unterstützten Kurven-, Wiederholungs- oder Zeitgrenzen: nur Startposition und Objekttyp werden gelernt."
        )
    slider_count = int(sum((item["events"][:, 9] * item["events"][:, 14]).sum() for item in training))
    if not slider_count:
        warnings.append("Keine geeigneten Trainings-Slider: das Modell wird nur Circles erzeugen.")
    return {
        "id": identifier,
        "name": settings["name"],
        "architecture": ARCHITECTURE,
        "format_version": FORMAT_VERSION,
        "event_schema": EVENT_SCHEMA,
        "model_config": cfg.dictionary(),
        "created_at": store.now(),
        "settings": settings,
        "dataset_id": settings["dataset_id"],
        "dataset_signature": dataset_signature(prepared),
        "maps": len(prepared),
        "training_songs": len({item["song"] for item in training}),
        "validation_songs": len({item["song"] for item in validation}),
        "test_songs": len({item["song"] for item in test}),
        "training_song_hashes": sorted({item["audio_hash"] for item in training}),
        "validation_song_hashes": sorted({item["audio_hash"] for item in validation}),
        "test_song_hashes": sorted({item["audio_hash"] for item in test}),
        "trained_slider_geometry": slider_count,
        "warnings": warnings,
        "excluded_maps": issues,
        "metrics": [],
        "best_epoch": 0,
        "best_loss": None,
        "training_status": "running",
    }


def publish(directory, metadata):
    metadata["continuable"] = (directory / "training.safetensors").is_file()
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary = directory / "bundle.tmp"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("metadata.json", "model.safetensors"):
            archive.write(directory / name, name)
    temporary.replace(directory / "model.zip")
    store.save("model", metadata)
    (directory / "recovery.json").unlink(missing_ok=True)
