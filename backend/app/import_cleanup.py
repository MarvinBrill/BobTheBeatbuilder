"""Rollback interrupted imports and avoid retaining duplicate archive contents."""

from pathlib import Path
from . import store
from .storage_inventory import import_maps
from .storage_files import remove_paths


def finish_import(folder, previous):
    retained = set()
    for item in import_maps(folder):
        old = previous.get(item["id"])
        if old and all(Path(old[key]).is_file() for key in ("map_path", "audio_path")):
            # Identical content already lives elsewhere; keep that copy, drop this one.
            store.save("map", old)
            continue
        retained.update(Path(item[key]) for key in ("map_path", "audio_path"))
    if not retained:
        remove_paths([folder])
        return
    remove_paths(path for path in folder.rglob("*") if path.is_file() and path not in retained)


def rollback_import(folder, previous):
    affected = import_maps(folder)
    remove_paths([folder])
    with store.connect() as connection:
        for item in affected:
            connection.execute("DELETE FROM records WHERE kind='map' AND id=?", (item["id"],))
    for item in affected:
        if item["id"] in previous:
            store.save("map", previous[item["id"]])
