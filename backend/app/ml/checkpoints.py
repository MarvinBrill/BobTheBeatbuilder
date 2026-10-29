import json
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file
from ..config import DATA
from .. import store
from .architecture import ARCHITECTURE


def save_weights(model, path):
    weights = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    if not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Non-finite model weights; checkpoint refused")
    temporary = path.with_suffix(".tmp")
    save_file(weights, str(temporary))
    temporary.replace(path)


def save_training(model, optimizer, directory, state):
    tensors = {f"model.{key}": value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    for index, parameter in enumerate(model.parameters()):
        for key, value in optimizer.state.get(parameter, {}).items():
            if torch.is_tensor(value):
                tensors[f"optimizer.{index}.{key}"] = value.detach().cpu().contiguous()
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("Non-finite training checkpoint refused")
    temporary = directory / "training.tmp"
    # State and tensors are a single atomic file; no pickle is ever loaded.
    save_file(tensors, str(temporary), metadata={"state": json.dumps(state)})
    temporary.replace(directory / "training.safetensors")


def warm_start(model, identifier):
    """Copy every weight whose shape matches from another model (any style version).

    New layers (style conditioning, mapper embedding, rhythm head) keep their initialisation;
    a larger mapper table keeps the base model's rows for the mappers it already knew.
    """
    path = DATA / "models" / identifier / "model.safetensors"
    if not path.is_file():
        raise ValueError("Das Startmodell hat keine Gewichte mehr")
    source = load_file(str(path))
    target = model.state_dict()
    copied, skipped = 0, []
    with torch.no_grad():
        for key, value in target.items():
            weights = source.get(key)
            if weights is None:
                skipped.append(key)
            elif weights.shape == value.shape:
                value.copy_(weights.to(value.dtype))
                copied += 1
            elif key == "mapper.weight" and weights.shape[1:] == value.shape[1:]:
                rows = min(weights.shape[0], value.shape[0])
                value[:rows].copy_(weights[:rows].to(value.dtype))
                copied += 1
            else:
                skipped.append(key)
    return {"copied_tensors": copied, "new_tensors": len(skipped)}


def restore(model, optimizer, identifier, dataset_id):
    from safetensors import safe_open

    metadata = store.get("model", identifier)
    if metadata["architecture"] != ARCHITECTURE or metadata.get("dataset_id") != dataset_id:
        raise ValueError("Continue training requires the same dataset and Phrase Mapper architecture")
    path = DATA / "models" / identifier / "training.safetensors"
    if not path.is_file():
        raise ValueError(
            "This model has no local training checkpoint (downloaded bundles contain inference weights only)"
        )
    values = load_file(str(path))
    model.load_state_dict({key[6:]: value for key, value in values.items() if key.startswith("model.")})
    for index, parameter in enumerate(model.parameters()):
        prefix = f"optimizer.{index}."
        for key, value in values.items():
            if key.startswith(prefix):
                field = key[len(prefix) :]
                gpu_step = optimizer.defaults.get("fused", False) or optimizer.defaults.get(
                    "capturable", False
                )
                target = "cpu" if field == "step" and not gpu_step else parameter.device
                optimizer.state[parameter][field] = value.to(target)
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        state = json.loads(checkpoint.metadata()["state"])
    return metadata, state
