import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.getenv("BOB_DATA", ROOT / "data")).resolve()
for name in ("library", "datasets", "models", "outputs", "uploads", "cache", "tmp", "snapshots"):
    (DATA / name).mkdir(parents=True, exist_ok=True)

os.environ.setdefault("NUMBA_CACHE_DIR", str(DATA / "cache" / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(DATA / "cache" / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(DATA / "cache" / "torch"))
os.environ.setdefault("CUDA_CACHE_PATH", str(DATA / "cache" / "cuda"))
os.environ.setdefault("TMPDIR", str(DATA / "tmp"))
BACKEND = os.getenv("BOB_DEVICE", "cpu")
READER = Path(os.getenv("BOB_READER", DATA / "tools" / "exporter" / "LazerReader.dll"))
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus"}
MAX_UPLOAD = 512 * 1024 * 1024


def inside(root: Path, value: str) -> Path:
    normalized = value.replace("\\", "/")
    if ":" in normalized or normalized.startswith("/"):
        raise ValueError("Absolute paths are not allowed inside a beatmap.")
    target = (root / normalized).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Path escapes the selected directory.")
    return target
