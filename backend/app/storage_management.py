"""Explicit storage actions, called while job/upload admission is locked."""

from . import store
from .config import DATA
from .artifact_deletion import delete_artifact
from .storage_inventory import inventory, import_maps
from .storage_files import remove_paths, trim_analysis_cache


def remove_entry(kind, identifier):
    if kind in ("model", "output", "dataset", "audio"):
        return delete_artifact(kind, identifier)

    report = inventory()
    entry = next(
        (row for row in report["entries"] if row["kind"] == kind and row["id"] == identifier), None
    )
    if not entry or not entry["removable"]:
        raise ValueError(
            "Dieser Speichereintrag darf nicht gelöscht werden oder existiert nicht mehr."
        )

    paths = [DATA / relative for relative in entry["paths"]]
    if kind == "history":
        with store.connect() as connection:
            connection.execute("DELETE FROM records WHERE kind='job'")
        with store.connect() as connection:
            connection.execute("VACUUM")
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return {"deleted": identifier}
    maps = []
    if kind == "import":
        maps = import_maps(DATA / "library" / identifier)
    elif kind == "catalog":
        maps = store.all_records("map")
        paths = list((DATA / "library").iterdir())

    cache_freed = trim_analysis_cache(clear=True) if kind in ("catalog", "import") else 0
    freed = remove_paths(paths) + cache_freed
    # Keep catalogue records until filesystem deletion succeeds so retries are possible.
    with store.connect() as connection:
        connection.executemany(
            "DELETE FROM records WHERE kind='map' AND id=?", [(item["id"],) for item in maps]
        )
    return {"deleted": identifier, "freed_bytes": freed}
