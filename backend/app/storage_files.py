"""File accounting and bounded, disposable analysis caches."""

import gc
import re
import shutil
import stat
from .config import DATA
from . import store


def linked(path):
    attributes = path.lstat()
    return path.is_symlink() or bool(
        getattr(attributes, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def measure(path):
    if not path.exists() and not path.is_symlink():
        return 0
    if linked(path):
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(measure(child) for child in path.iterdir())


def remove_paths(paths):
    from .artifact_deletion import validate_path

    paths = list(paths)
    # Validate the whole plan before touching any file, including junction children.
    for path in paths:
        relative = path.relative_to(DATA)
        if not relative.parts:
            raise ValueError("Der Datenordner selbst darf nicht gelöscht werden.")
        validate_path(path, DATA / relative.parts[0])
    freed = sum(measure(path) for path in paths)
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    return freed


def analysis_files():
    pattern = r"(?:mel-|phrase-|labels-).+\.(?:npy|npz|json|tmp)"
    return [path for path in (DATA / "cache").iterdir() if re.fullmatch(pattern, path.name)]


def cache_limit():
    try:
        return store.get("settings", "storage")["cache_limit_gib"]
    except (ValueError, KeyError):
        return 1.0


def trim_analysis_cache(clear=False):
    # Called only when no operation is using the memory-mapped arrays anymore.
    gc.collect()
    paths = sorted(analysis_files(), key=lambda path: path.lstat().st_mtime_ns)
    remaining = sum(measure(path) for path in paths)
    limit = 0 if clear else cache_limit() * 1024**3
    removed = []
    for path in paths:
        if not clear and limit > 0 and remaining <= limit:
            break
        remaining -= measure(path)
        removed.append(path)
    return remove_paths(removed)


def prune_job_history():
    terminal = [
        item for item in store.all_records("job") if item.get("status") not in ("queued", "running")
    ]
    with store.connect() as connection:
        connection.executemany(
            "DELETE FROM records WHERE kind='job' AND id=?",
            [(item["id"],) for item in terminal[100:]],
        )
