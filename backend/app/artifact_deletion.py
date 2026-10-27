"""Remove one owned artifact and its history without touching shared source data."""

import json
import re
import shutil
import stat
from . import store
from .config import DATA


def validate_path(path, folder):
    # Junctions on Windows need the reparse flag; is_symlink alone is insufficient.
    for parent in (folder, path):
        if not parent.exists() and not parent.is_symlink():
            continue
        attributes = parent.lstat()
        if (
            parent.is_symlink()
            or getattr(attributes, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ValueError("Verknüpfte Dateien oder Ordner werden nicht gelöscht.")
    if folder.resolve() != DATA / folder.name or not path.resolve().is_relative_to(folder):
        raise ValueError("Der Löschpfad liegt außerhalb des vorgesehenen Datenordners.")
    if path.is_dir():
        for child in path.iterdir():
            validate_path(child, folder)


def delete_artifact(kind, identifier):
    from .storage_inventory import artifact_paths
    from .storage_files import trim_analysis_cache

    if kind not in ("model", "output", "dataset", "audio") or not re.fullmatch(
        r"[a-f0-9]{32}", identifier
    ):
        raise ValueError("Ungültige Artefakt-ID.")
    item = store.get(kind, identifier)
    targets = artifact_paths(kind, identifier)
    for target in targets:
        validate_path(target, target.parent)

    # Recomputable arrays have no durable ownership and can be shared by many runs.
    # Clear them before deleting records, so a filesystem failure remains retryable.
    trim_analysis_cache(clear=True)

    # Keep the catalogue entry on a filesystem failure so deletion can be retried.
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)

    with store.connect() as connection:
        jobs = connection.execute("SELECT id, body FROM records WHERE kind='job'").fetchall()
        related = []
        for job_id, body in jobs:
            job = json.loads(body)
            if (job.get("result") or {}).get(f"{kind}_id") == identifier or job_id == item.get(
                "job_id"
            ):
                related.append((job_id,))
        connection.executemany("DELETE FROM records WHERE kind='job' AND id=?", related)
        connection.execute("DELETE FROM records WHERE kind=? AND id=?", (kind, identifier))
    return {"deleted": identifier, "removed_jobs": len(related)}
