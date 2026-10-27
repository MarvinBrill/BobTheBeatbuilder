import uvicorn
from .config import BACKEND
from .ml.network import device
from .ml.compute_probe import check_compute


def check_device():
    selected = device()
    check_compute(selected)
    print(f"Compute backend ready: {BACKEND} ({selected})", flush=True)


if __name__ == "__main__":
    check_device()
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=1)
