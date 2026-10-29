import uvicorn
from .config import BACKEND


def check_device():
    # Imported lazily: spawned import workers re-import this module and must not load torch.
    from .ml.network import device
    from .ml.compute_probe import check_compute

    selected = device()
    check_compute(selected)
    print(f"Compute backend ready: {BACKEND} ({selected})", flush=True)


if __name__ == "__main__":
    check_device()
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=1)
