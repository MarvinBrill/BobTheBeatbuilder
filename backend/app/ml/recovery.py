import json
import logging
from ..config import DATA
from .architecture import ARCHITECTURE, EVENT_SCHEMA
from .training_setup import publish


def recover_training_runs():
    """Register atomically saved runs left behind by a process or machine crash."""
    for path in (DATA / "models").glob("*/recovery.json"):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("architecture") != ARCHITECTURE or metadata.get("event_schema") != EVENT_SCHEMA:
                continue
            if metadata.get("id") != path.parent.name:
                raise ValueError("Checkpoint directory does not match model ID")
            if not all(
                (path.parent / name).is_file() for name in ("training.safetensors", "model.safetensors")
            ):
                continue
            metadata["training_status"] = "interrupted"
            metadata.setdefault("warnings", []).append(
                "Nach einem Neustart wiederhergestellt. Fortsetzen ab dem letzten gespeicherten Update; die angebrochene Epoche beginnt erneut."
            )
            publish(path.parent, metadata)
        except (OSError, ValueError, KeyError):
            logging.exception("Could not recover training checkpoint %s", path)
