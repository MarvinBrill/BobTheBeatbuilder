import json
import shutil
import zipfile
from uuid import uuid4
import torch
from safetensors.torch import load_file
from . import store
from .config import DATA
from .ml.phrase_model import create_model
from .ml.architecture import ARCHITECTURE, FORMAT_VERSION, EVENT_SCHEMA, TARGET_SCHEMA, ModelConfig


def import_model(path, context):
    identifier = uuid4().hex
    directory = DATA / "models" / identifier
    directory.mkdir()
    completed = False
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) != 2 or {item.filename for item in entries} != {
                "metadata.json",
                "model.safetensors",
            }:
                raise ValueError(
                    "Expected a Bob model ZIP containing metadata.json and model.safetensors"
                )
            if sum(item.file_size for item in entries) > 256 * 1024 * 1024:
                raise ValueError("Model bundle exceeds 256 MiB unpacked")
            metadata = json.loads(archive.read("metadata.json"))
            if (
                not isinstance(metadata, dict)
                or metadata.get("architecture") != ARCHITECTURE
                or metadata.get("format_version") != FORMAT_VERSION
                or metadata.get("event_schema") != EVENT_SCHEMA
                or metadata.get("target_schema") != TARGET_SCHEMA
            ):
                raise ValueError("Unsupported model architecture or format version")
            weights_path = directory / "model.safetensors"
            weights_path.write_bytes(archive.read("model.safetensors"))
        context.report(0.5, "Validating model tensor shapes and values")
        weights = load_file(str(weights_path))
        if any(
            tensor.dtype != torch.float32 or not torch.isfinite(tensor).all()
            for tensor in weights.values()
        ):
            raise ValueError("Model contains invalid tensor types or values")
        config = ModelConfig.from_dict(metadata.get("model_config"))
        create_model(metadata["architecture"], config.dictionary()).load_state_dict(
            weights, strict=True
        )
        model = {
            "id": identifier,
            "name": str(metadata.get("name", "Imported model"))[:100],
            "architecture": metadata["architecture"],
            "format_version": FORMAT_VERSION,
            "model_config": config.dictionary(),
            "event_schema": EVENT_SCHEMA,
            "target_schema": TARGET_SCHEMA,
            "created_at": store.now(),
            "maps": max(0, int(metadata.get("maps", 0))),
            "training_songs": max(0, int(metadata.get("training_songs", 0))),
            "validation_songs": max(0, int(metadata.get("validation_songs", 0))),
            "metrics": [],
            "imported": True,
            "continuable": False,
        }
        if config.style_version >= 2:
            vocabulary = metadata.get("mapper_vocab") or []
            if (
                not isinstance(vocabulary, list)
                or not all(isinstance(name, str) and len(name) <= 100 for name in vocabulary)
                or len(vocabulary) + 1 != config.mapper_slots
            ):
                raise ValueError("Model mapper list does not match its weights")
            model["mapper_vocab"] = vocabulary
            model["style_version"] = config.style_version
            for key in ("mapper_profiles", "star_profiles"):
                if isinstance(metadata.get(key), dict):
                    model[key] = metadata[key]
            if isinstance(metadata.get("style_features"), list):
                model["style_features"] = [str(key) for key in metadata["style_features"]]
        if "trained_slider_geometry" in metadata:
            model["trained_slider_geometry"] = max(0, int(metadata["trained_slider_geometry"]))
            if not model["trained_slider_geometry"]:
                model["warnings"] = [
                    "No supported slider geometry in training songs: this model generates circles only."
                ]
        (directory / "metadata.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
        with zipfile.ZipFile(directory / "model.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for filename in ("metadata.json", "model.safetensors"):
                archive.write(directory / filename, filename)
        context.check()
        store.save("model", model)
        completed = True
        return {"model_id": identifier}
    finally:
        path.unlink(missing_ok=True)
        if not completed:
            shutil.rmtree(directory)
