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
