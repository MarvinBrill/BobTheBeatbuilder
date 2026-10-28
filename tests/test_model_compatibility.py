import json
import zipfile
import pytest
import torch
from safetensors.torch import save_file
from app import store
from app.config import DATA
from app.ml.architecture import (
    ARCHITECTURE,
    FORMAT_VERSION,
    EVENT_SCHEMA,
    TARGET_SCHEMA,
    ModelConfig,
)
from app.ml.phrase_model import PhraseMapper
from app.models import import_model


class Context:
    def check(self):
        pass

    def report(self, *args, **kwargs):
        pass


def bundle_for(tmp_path, invalid=False, architecture=ARCHITECTURE):
    cfg = ModelConfig(width=64, heads=2, audio_layers=1, global_layers=1, decoder_layers=1)
    weights = PhraseMapper(cfg).state_dict()
    if invalid:
        weights["kind.bias"][0] = float("nan")
    path = tmp_path / "weights.safetensors"
    save_file(weights, str(path))
    bundle = tmp_path / "model.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(
            "metadata.json",
            json.dumps(
                dict(
                    architecture=architecture,
                    format_version=FORMAT_VERSION,
                    event_schema=EVENT_SCHEMA,
                    target_schema=TARGET_SCHEMA,
                    model_config=cfg.dictionary(),
                    trained_slider_geometry=0,
                )
            ),
        )
        archive.write(path, "model.safetensors")
    return path, bundle


def test_model_bundle_roundtrip_preserves_weights(tmp_path):
    torch.set_num_threads(2)
    path, bundle = bundle_for(tmp_path)
    result = import_model(bundle, Context())
    stored = store.get("model", result["model_id"])
    assert stored["architecture"] == ARCHITECTURE
    assert stored["trained_slider_geometry"] == 0
    assert not stored["continuable"]
    assert (DATA / "models" / stored["id"] / "model.safetensors").read_bytes() == path.read_bytes()
    assert not bundle.exists()


@pytest.mark.parametrize(
    "invalid,architecture,message",
    [(True, ARCHITECTURE, "invalid tensor"), (False, "beatnet-spatial-v2", "Unsupported")],
)
def test_import_rejects_invalid_or_obsolete_model_and_cleans_up(
    tmp_path, invalid, architecture, message
):
    _, bundle = bundle_for(tmp_path, invalid, architecture)
    before = set((DATA / "models").iterdir())
    with pytest.raises(ValueError, match=message):
        import_model(bundle, Context())
    assert set((DATA / "models").iterdir()) == before


@pytest.mark.parametrize(
    "field", ["architecture", "format_version", "event_schema", "target_schema"]
)
@pytest.mark.parametrize("missing", [False, True])
def test_generation_rejects_incompatible_stored_artifact_before_loading(
    monkeypatch, field, missing
):
    from app.ml import generation

    metadata = dict(
        id=f"generation-incompatible-{field}-{missing}",
        architecture=ARCHITECTURE,
        format_version=FORMAT_VERSION,
        event_schema=EVENT_SCHEMA,
        target_schema=TARGET_SCHEMA,
    )
    if missing:
        metadata.pop(field)
    else:
        metadata[field] = FORMAT_VERSION - 1 if field == "format_version" else "obsolete"
    store.save("model", metadata)
    before = set((DATA / "outputs").iterdir())

    def unexpected(*args, **kwargs):
        raise AssertionError(
            "An incompatible artifact must be rejected before model or audio loading"
        )

    monkeypatch.setattr(generation, "create_model", unexpected)
    monkeypatch.setattr(generation, "features", unexpected)
    monkeypatch.setattr(generation, "load_file", unexpected)
    try:
        with pytest.raises(ValueError, match=f"Incompatible model {field}.*Train a new"):
            generation.generate({"model_id": metadata["id"], "device": "cpu"}, Context())
        assert set((DATA / "outputs").iterdir()) == before
    finally:
        with store.connect() as connection:
            connection.execute("DELETE FROM records WHERE kind='model' AND id=?", (metadata["id"],))


def test_model_listing_ignores_missing_architecture_without_hiding_current_models(monkeypatch):
    from app import api

    current = dict(id="current", architecture=ARCHITECTURE, event_schema=EVENT_SCHEMA)
    monkeypatch.setattr(
        store,
        "all_records",
        lambda kind: [{"id": "incomplete"}, current] if kind == "model" else [],
    )
    assert [model["id"] for model in api.state()["models"]] == ["current"]
