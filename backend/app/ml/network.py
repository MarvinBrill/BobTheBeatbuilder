import torch
from ..config import BACKEND

CONDITION_KEYS = ["ar", "cs", "od", "hp", "density"]


def device(name=None):
    selected = name or BACKEND
    if selected == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was selected but no CUDA GPU is available")
        return torch.device("cuda")
    if selected == "directml":
        try:
            import torch_directml

            return torch_directml.device()
        except (ImportError, RuntimeError) as error:
            raise ValueError(f"DirectML is unavailable: {error}") from error
    if selected != "cpu":
        raise ValueError("Unknown compute backend")
    return torch.device("cpu")


def conditions(settings):
    return [float(settings[key]) / 10 for key in CONDITION_KEYS] + [
        float(settings.get("spacing", 0.3)),
        float(settings.get("flow", 0.5)),
        float(settings.get("slider_bias", 0.5)),
    ]
