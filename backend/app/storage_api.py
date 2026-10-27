from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from . import store
from .storage_inventory import inventory
from .storage_management import remove_entry
from .storage_files import trim_analysis_cache

router = APIRouter(prefix="/api/storage")


class CacheSettings(BaseModel):
    cache_limit_gib: float = Field(ge=0, le=64, allow_inf_nan=False)


def manager():
    # Resolve at request time; tests and the app share the exact same admission lock.
    from .api import jobs

    return jobs


def require_idle(jobs):
    if jobs.events or jobs.uploads:
        raise HTTPException(409, "Zuerst laufende Aufgaben und Uploads beenden oder abbrechen.")


@router.get("")
def storage():
    jobs = manager()
    with jobs.lock:
        result = inventory()
        result["blocked"] = bool(jobs.events or jobs.uploads)
        return result


@router.post("/settings")
def settings(request: CacheSettings):
    jobs = manager()
    with jobs.lock:
        require_idle(jobs)
        store.save("settings", {"id": "storage", "cache_limit_gib": request.cache_limit_gib})
        try:
            trim_analysis_cache()
        except OSError as error:
            raise HTTPException(
                409,
                "Cache-Limit gespeichert; offene Cache-Dateien konnten noch nicht entfernt werden.",
            ) from error
    return {"saved": True}


@router.delete("/{kind}/{identifier}")
def remove(kind: str, identifier: str):
    jobs = manager()
    with jobs.lock:
        require_idle(jobs)
        try:
            return remove_entry(kind, identifier)
        except OSError as error:
            raise HTTPException(
                409,
                "Dateien sind noch geöffnet oder nicht zugänglich. Eintrag bleibt zum erneuten Löschen sichtbar.",
            ) from error
