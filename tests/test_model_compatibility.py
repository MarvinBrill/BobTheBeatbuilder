import json
import zipfile
import pytest
import torch
from safetensors.torch import save_file
from app import store
from app.config import DATA
from app.ml.architecture import ARCHITECTURE, FORMAT_VERSION, EVENT_SCHEMA, ModelConfig
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
def test_import_rejects_invalid_or_obsolete_model_and_cleans_up(tmp_path, invalid, architecture, message):
    _, bundle = bundle_for(tmp_path, invalid, architecture)
    before = set((DATA / "models").iterdir())
    with pytest.raises(ValueError, match=message):
        import_model(bundle, Context())
    assert set((DATA / "models").iterdir()) == before
