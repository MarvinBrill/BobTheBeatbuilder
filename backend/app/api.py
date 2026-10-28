import json
import os
from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, Header, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from . import library, store
from .config import AUDIO_EXTENSIONS, BACKEND, DATA, MAX_UPLOAD, READER
from .jobs import Jobs
from .schemas import GenerateRequest, ScanRequest, Selection, TrainRequest
from .ml.architecture import ARCHITECTURE, EVENT_SCHEMA
from .ml.recovery import recover_training_runs
from .map_catalog import matching_maps
from .artifact_deletion import delete_artifact

router = APIRouter(prefix="/api")
recover_training_runs()
jobs = Jobs()
STATE_SESSION = uuid4().hex


def public(item):
    return {
        key: value
        for key, value in item.items()
        if key not in ("map_path", "audio_path", "path", "preview_path", "excluded_maps")
        and not (key == "maps" and isinstance(value, list))
    }


@router.get("/state")
def conditional_state(response: Response, if_none_match: str | None = Header(default=None)):
    with jobs.lock:
        blocked = bool(jobs.events or jobs.uploads)
    tag = f'"{STATE_SESSION}-{store.revision()}-{int(blocked)}"'
    headers = {"ETag": tag, "Cache-Control": "no-store"}
    if if_none_match == tag:
        return Response(status_code=304, headers=headers)
    response.headers.update(headers)
    return state()


def state():
    with jobs.lock:
        deletion_blocked = bool(jobs.events or jobs.uploads)
    datasets = store.all_records("dataset")
    dataset_ids = {item["id"] for item in datasets}
    return {
        "deletion_blocked": deletion_blocked,
        "datasets": [public(item) for item in datasets],
        "models": [
            public(
                {
                    **item,
                    "continuable": bool(
                        item.get("continuable") and item.get("dataset_id") in dataset_ids
                    ),
                }
            )
            for item in store.all_records("model")
            if item["architecture"] == ARCHITECTURE and item.get("event_schema") == EVENT_SCHEMA
        ],
        "jobs": store.all_records("job")[:100],
        "outputs": [public(item) for item in store.all_records("output")],
        "map_count": store.count("map"),
        "device": BACKEND,
        "reader_ready": READER.is_file(),
        "default_path": os.getenv("BOB_OSU_PATH", "/osu" if os.name != "nt" else "D:/osu! lazer"),
    }


@router.get("/hardware")
def hardware():
    import torch

    result = {"cpu": True, "cuda": torch.cuda.is_available(), "directml": False, "name": "CPU"}
    if result["cuda"]:
        result["name"] = torch.cuda.get_device_name(0)
    try:
        import torch_directml

        result["directml"] = torch_directml.device_count() > 0
        if result["directml"] and BACKEND == "directml":
            result["name"] = torch_directml.device_name(0)
    except (ImportError, RuntimeError):
        pass
    return result


@router.get("/browse")
def browse(path: str):
    folder = Path(path).resolve()
    if not folder.is_dir():
        raise ValueError("Folder not found. Docker only sees mounted folders such as /osu.")
    children = sorted(
        [
            {"name": child.name, "path": str(child)}
            for child in folder.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        ],
        key=lambda child: child["name"].casefold(),
    )
    return {"path": str(folder), "parent": str(folder.parent), "children": children[:300]}


@router.get("/maps")
def maps(
    q: str = "",
    offset: int = 0,
    limit: int = 100,
    stars_min: float | None = None,
    stars_max: float | None = None,
    per_song: str = "all",
):
    filtered = matching_maps(q, stars_min, stars_max, per_song)
    return {
        "items": [
            public(item)
            for item in filtered[max(0, offset) : max(0, offset) + min(250, max(1, limit))]
        ],
        "total": len(filtered),
    }


@router.get("/maps/ids")
def map_ids(
    q: str = "",
    stars_min: float | None = None,
    stars_max: float | None = None,
    per_song: str = "all",
):
    items = matching_maps(q, stars_min, stars_max, per_song)
    if len(items) > 50000:
        raise ValueError("Bitte den Filter auf höchstens 50.000 Maps eingrenzen.")
    return {"ids": [item["id"] for item in items], "total": len(items)}


@router.post("/scan")
def scan(request: ScanRequest):
    return jobs.submit("scan", "Scan library", lambda ctx: library.scan(request.path, ctx))


async def upload(file: UploadFile, extensions):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in extensions:
        raise ValueError("Unsupported file type")
    identifier = uuid4().hex
    destination = DATA / "uploads" / f"{identifier}{extension}"
    try:
        total = 0
        with destination.open("wb") as output:
            while block := await file.read(1024 * 1024):
                total += len(block)
                if total > MAX_UPLOAD:
                    raise ValueError("Upload exceeds 512 MiB")
                output.write(block)
        if not total:
            raise ValueError("Empty upload")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return identifier, destination


@router.post("/import")
async def import_maps(file: UploadFile):
    _, path = await upload(file, {".osz", ".zip"})
    return jobs.submit(
        "import",
        "Import beatmaps",
        lambda ctx: library.import_archive(path, ctx),
        cleanup=lambda: path.unlink(missing_ok=True),
    )


@router.post("/audio")
async def upload_audio(file: UploadFile):
    identifier, path = await upload(file, AUDIO_EXTENSIONS)
    try:
        store.save("audio", {"id": identifier, "path": str(path), "name": file.filename})
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return {"id": identifier, "name": file.filename}


@router.post("/datasets")
def dataset(request: Selection):
    with jobs.lock:
        for identifier in request.ids:
            store.get("map", identifier)
        return jobs.submit(
            "dataset",
            request.name,
            lambda ctx: library.create_dataset(list(dict.fromkeys(request.ids)), request.name, ctx),
        )


@router.post("/models/import")
async def upload_model(file: UploadFile):
    from .models import import_model

    _, path = await upload(file, {".zip"})
    return jobs.submit(
        "model-import",
        "Import model",
        lambda ctx: import_model(path, ctx),
        cleanup=lambda: path.unlink(missing_ok=True),
    )


@router.post("/export")
def export(request: Selection):
    return jobs.submit(
        "export",
        "Export selected maps",
        lambda ctx: library.export_selection(list(dict.fromkeys(request.ids)), ctx),
    )


@router.post("/train")
def train(request: TrainRequest):
    from .ml.training import train as operation

    with jobs.lock:
        store.get("dataset", request.dataset_id)
        if request.resume_model_id:
            store.get("model", request.resume_model_id)
        return jobs.submit("training", request.name, lambda ctx: operation(request.dict(), ctx))


@router.post("/generate")
def generate(request: GenerateRequest):
    from .ml.generation import generate as operation

    with jobs.lock:
        store.get("model", request.model_id)
        store.get("audio", request.audio_id)
        return jobs.submit("generation", request.title, lambda ctx: operation(request.dict(), ctx))


def remove_artifact(kind, identifier):
    with jobs.lock:
        if jobs.events or jobs.uploads:
            raise HTTPException(
                409, "Bitte laufende und wartende Aufgaben zuerst beenden oder abbrechen."
            )
        try:
            return delete_artifact(kind, identifier)
        except ValueError as error:
            if str(error).startswith(f"Unknown {kind}:"):
                raise HTTPException(404, "Dieser Eintrag wurde bereits entfernt.") from error
            raise
        except OSError as error:
            raise HTTPException(
                409,
                "Dateien konnten nicht vollständig gelöscht werden. Schließe offene Downloads oder Dateien und versuche es erneut.",
            ) from error


@router.delete("/models/{identifier}")
def delete_model(identifier: str):
    return remove_artifact("model", identifier)


@router.delete("/outputs/{identifier}")
def delete_output(identifier: str):
    return remove_artifact("output", identifier)


@router.post("/jobs/{identifier}/cancel")
def cancel(identifier: str):
    return jobs.cancel(identifier)


@router.get("/models/{identifier}/download")
def download_model(identifier: str):
    store.get("model", identifier)
    return FileResponse(DATA / "models" / identifier / "model.zip", filename="bob-model.zip")


@router.get("/outputs/{identifier}/download")
def download_output(identifier: str):
    item = store.get("output", identifier)
    return FileResponse(item["path"], filename=item["name"])


@router.get("/outputs/{identifier}/preview")
def preview(identifier: str):
    from .preview_sounds import sound_events

    item = store.get("output", identifier)
    if "preview_path" not in item:
        raise HTTPException(404, "No preview for library exports")
    path = Path(item["preview_path"])
    result = json.loads(path.read_text(encoding="utf-8"))
    result["sound_events"] = sound_events(path.parent / "map.osu")
    return result


@router.get("/outputs/{identifier}/audio")
def audio(identifier: str):
    item = store.get("output", identifier)
    if "audio_path" not in item:
        raise HTTPException(404, "No preview audio")
    return FileResponse(item["audio_path"])
