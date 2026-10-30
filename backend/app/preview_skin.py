"""Read only the selected lazer skin's preview samples, without persistent copies."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .config import DATA, READER
from .library import hash_path

router = APIRouter(prefix="/api/preview-skin")
SAMPLES = (
    "hitnormal",
    "hitwhistle",
    "hitfinish",
    "hitclap",
    "slidertick",
    "sliderslide",
)
FORMATS = {".wav": "audio/wav", ".ogg": "audio/ogg", ".mp3": "audio/mpeg"}
lock = Lock()


def select_skin(skins, query, active_id=None):
    exact = [skin for skin in skins if skin["name"].casefold() == query.casefold()]
    matches = exact or [
        skin
        for skin in skins
        if re.search(r"(?<!\w)" + re.escape(query) + r"(?!\w)", skin["name"], re.I)
    ]
    if len(matches) > 1:
        active = [skin for skin in matches if skin.get("id") == active_id]
        if active:
            matches = active
    if len(matches) != 1:
        raise ValueError(
            f"Skin '{query}' nicht eindeutig gefunden. BOB_PREVIEW_SKIN auf den vollständigen Skin-Namen setzen."
        )
    return matches[0]


def read_skin(root, query):
    source = root / "client.realm"
    if not source.is_file() or not READER.is_file():
        raise ValueError(
            "Skin nicht verfügbar: osu!-Datenordner und Lazer-Reader prüfen."
        )
    with tempfile.TemporaryDirectory(prefix="skin-", dir=DATA / "snapshots") as folder:
        target = Path(folder) / "client.realm"
        before = source.stat()
        shutil.copyfile(source, target)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(
                "osu! hat seine Datenbank geändert. Vorschau später erneut öffnen."
            )
        result = subprocess.run(
            ["dotnet", str(READER), str(target), "--skins"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
        )
        if result.returncode:
            raise ValueError(
                "Skin konnte nicht gelesen werden. Lazer-Reader mit dem Setup-Skript aktualisieren."
            )
        active_id = None
        configuration = root / "game.ini"
        if configuration.is_file():
            for line in configuration.read_text(encoding="utf-8-sig").splitlines():
                if line.partition("=")[0].strip() == "Skin":
                    active_id = line.partition("=")[2].strip()
                    break
        selected = select_skin(json.loads(result.stdout), query, active_id)

    files = {entry["name"].casefold(): entry for entry in selected["files"]}
    samples = {}
    for sample in SAMPLES:
        # A missing Soft sample may inherit Normal; an explicitly empty sample
        # is intentional silence and must not be replaced by another sound.
        entry = next(
            (
                files[name]
                for bank in ("soft", "normal")
                for suffix in FORMATS
                if (name := f"{bank}-{sample}{suffix}") in files
            ),
            None,
        )
        if entry is None:
            continue
        path = hash_path(root, entry["hash"])
        size = path.stat().st_size
        if size > 8 * 1024 * 1024:
            raise ValueError("Skin-Sample überschreitet 8 MiB.")
        if size:
            samples[sample] = {
                "path": path,
                "filename": entry["name"],
                "media_type": FORMATS[Path(entry["name"]).suffix.lower()],
            }
    return {"name": selected["name"], "samples": samples}


@lru_cache(maxsize=4)
def cached_skin(root, query):
    return read_skin(Path(root), query)


def current_skin():
    root = os.getenv("BOB_OSU_PATH", "/osu" if os.name != "nt" else "D:/osu! lazer")
    query = os.getenv("BOB_PREVIEW_SKIN", "ssM4in")
    # Concurrent sample requests must not create concurrent Realm snapshots.
    with lock:
        return cached_skin(root, query)


@router.get("")
def manifest():
    try:
        skin = current_skin()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        return {"name": None, "samples": {}, "warning": str(error)}
    return {
        "name": skin["name"],
        "samples": {
            name: f"/api/preview-skin/samples/{name}" for name in skin["samples"]
        },
        "warning": (
            None
            if len(skin["samples"]) == len(SAMPLES)
            else "Einzelne Skin-Samples fehlen oder sind bewusst stumm."
        ),
    }


@router.get("/samples/{name}")
def sample_file(name: str):
    if name not in SAMPLES:
        raise HTTPException(404, "Unbekanntes Skin-Sample")
    try:
        sample = current_skin()["samples"].get(name)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        raise HTTPException(404, "Skin nicht verfügbar") from error
    if sample is None or not sample["path"].is_file():
        raise HTTPException(404, "Skin-Sample fehlt")
    return FileResponse(sample["path"], media_type=sample["media_type"])
