"""Inventory uses server-owned paths; clients never submit deletion paths."""

import hashlib
from pathlib import Path
from . import store
from .config import DATA
from .storage_files import cache_limit, linked, measure

FOLDERS = {"model": "models", "output": "outputs", "dataset": "datasets", "audio": "uploads"}


def artifact_paths(kind, identifier):
    folder = DATA / FOLDERS[kind]
    if kind == "audio":
        return list(folder.glob(f"{identifier}.*"))
    paths = [folder / identifier]
    if kind == "output":
        paths.append(folder / f"{identifier}.osz")
    return paths


def import_maps(folder):
    return [
        item
        for item in store.all_records("map")
        if any(Path(item.get(key, "")).is_relative_to(folder) for key in ("map_path", "audio_path"))
    ]


def inventory():
    entries, owned = [], set()

    def add(kind, identifier, name, paths, detail, removable=True):
        present = [path for path in paths if path.exists() or path.is_symlink()]
        owned.update(present)
        entries.append(
            {
                "kind": kind,
                "id": identifier,
                "name": name,
                "bytes": sum(measure(path) for path in present),
                "detail": detail,
                "removable": removable and not any(linked(path) for path in present),
                "paths": [str(path.relative_to(DATA)) for path in present],
            }
        )

    for kind in FOLDERS:
        for item in store.all_records(kind):
            detail = {
                "model": "Gewichte, Trainingsstände, ZIP und Modellverlauf. Trainingsdatensätze bleiben separat erhalten.",
                "output": "Export, Vorschau, Audiokopie und Verlauf. Ursprüngliche Audio-Uploads bleiben separat erhalten.",
                "dataset": "Alle kopierten Songs und Maps dieses Datensatzes. Modelle bleiben spielbereit; Weitertraining mit diesem Datensatz entfällt.",
                "audio": "Ursprünglicher Audio-Upload. Bestehende Beatmaps besitzen ihre eigene Audiokopie und bleiben erhalten.",
            }[kind]
            add(
                kind,
                item["id"],
                item.get("name", item["id"]),
                artifact_paths(kind, item["id"]),
                detail,
            )

    for folder in (DATA / "library").iterdir():
        maps = import_maps(folder)
        if maps:
            add(
                "import",
                folder.name,
                f"Import: {maps[0].get('title', folder.name)}",
                [folder],
                f"{len(maps)} Bibliothekseinträge mit lokal importierten Originaldateien. Trainingskopien bleiben erhalten.",
            )

    add(
        "catalog",
        "all",
        "Bibliothekskatalog zurücksetzen",
        [],
        f"{store.count('map')} Einträge und alle lokal importierten Bibliotheksdateien entfernen. Externe osu!-Originale und Trainingsdatensätze bleiben erhalten.",
    )
    add(
        "history",
        "all",
        "Aufgabenverlauf leeren",
        [],
        "Abgeschlossene, fehlgeschlagene und abgebrochene Aufgaben entfernen. Modelle und ihre eigenen Lernkurven bleiben erhalten.",
    )

    for name in ("cache", "tmp", "snapshots"):
        detail = (
            "Wiederherstellbare Analyse- und Download-Caches. Nächste Verarbeitung/Installation kann länger dauern."
            if name == "cache"
            else "Temporäre Dateien und Datenbankkopien abgebrochener Vorgänge."
        )
        add(
            "cleanup",
            name,
            {"cache": "Caches leeren", "tmp": "Temporäre Dateien", "snapshots": "Scan-Kopien"}[
                name
            ],
            list((DATA / name).iterdir()),
            detail,
        )

    for name in (*FOLDERS.values(), "library"):
        for path in (DATA / name).iterdir():
            if path not in owned:
                token = hashlib.sha256(str(path.relative_to(DATA)).encode()).hexdigest()
                add(
                    "orphan",
                    token,
                    f"Nicht zugeordnet: {path.relative_to(DATA)}",
                    [path],
                    "Datei/Ordner ohne Katalogeintrag, z. B. von einem abgebrochenen Vorgang. Vor dem Löschen prüfen.",
                )

    managed = set(FOLDERS.values()) | {"library", "cache", "tmp", "snapshots"}
    for path in DATA.iterdir():
        if path.name in managed:
            continue
        disposable = path.name in ("tests", "benchmarks") or path.name.startswith("pytest-")
        add(
            "development" if disposable else "runtime",
            path.name,
            path.name,
            [path],
            (
                "Entwicklungs- und Testdateien; keine regulären Nutzerartefakte."
                if disposable
                else "Laufzeit, Werkzeuge, Datenbank oder sonstige Datei. Nur Anzeige; benötigte Abhängigkeiten werden nicht automatisch gelöscht."
            ),
            disposable,
        )

    return {
        "entries": entries,
        "total_bytes": sum(item["bytes"] for item in entries),
        "cache_limit_gib": cache_limit(),
    }
