import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from uuid import uuid4
from . import store
from .artifacts import artifact_directory
from .ml.split import song_groups
from .beatmaps import parse
from .config import AUDIO_EXTENSIONS, DATA, READER, inside
from .import_workers import Hasher, Progress, analyze_all
from .map_catalog import star_rating


def digest(path):
    stat = path.stat()
    return cached_digest(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=20000)
def cached_digest(filename, size, modified):
    result = hashlib.sha256()
    with Path(filename).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def register(map_path, audio_path, source, source_id="", stars=None):
    beatmap = parse(map_path)
    if beatmap.mode != 0 or not beatmap.objects:
        return None
    if not audio_path.is_file():
        raise ValueError(f"Audio missing: {beatmap.audio}")
    item = map_record(
        beatmap.info(),
        digest(map_path),
        map_path,
        audio_path,
        digest(audio_path),
        beatmap.audio,
        source,
        source_id,
        star_rating(stars),
    )
    if item["star_rating"] is None and source != "lazer":
        try:
            known = store.get("map", item["id"])
            item = with_rating(item, star_rating(known.get("star_rating")))
        except ValueError:
            pass
    return store.save("map", item)


def map_record(
    info, map_hash, map_path, audio_path, audio_hash, audio_name, source, source_id, rating
):
    identifier = hashlib.sha256((map_hash + audio_hash).encode()).hexdigest()[:24]
    return with_rating(
        {
            "id": identifier,
            **info,
            "map_path": str(map_path),
            "audio_path": str(audio_path),
            "audio_hash": audio_hash,
            "audio_name": audio_name,
            "source": source,
            "source_id": source_id,
        },
        rating,
    )


def with_rating(item, rating):
    return {**item, "star_rating": rating, "star_source": "lazer" if rating is not None else None}


def unrated(item):
    return item["star_rating"] is None and item["source"] != "lazer"


class Candidate:
    """One .osu file to register; resolve maps its AudioFilename to a file path."""

    def __init__(self, map_path, resolve, label, source, source_id="", stars=None, error=None):
        self.map_path, self.resolve, self.label = map_path, resolve, label
        self.source, self.source_id, self.stars, self.error = source, source_id, stars, error


def register_many(candidates, context, start=0.0, end=1.0, audio_first=False):
    """Parse in parallel, hash each audio once, then save all maps in one transaction.

    Returns (registered, skipped_other_modes, errors) with errors in candidate order.
    """
    errors, skipped, accepted = [], 0, []
    runnable = []
    for index, candidate in enumerate(candidates):
        if candidate.error is not None:
            errors.append((index, candidate.label(candidate.error)))
        else:
            runnable.append(index)
    hasher = Hasher(digest)
    progress = Progress(context, start, start + 0.9 * (end - start))

    def collect(position, result):
        nonlocal skipped
        index = runnable[position]
        candidate = candidates[index]
        if "error" in result:
            errors.append((index, candidate.label(result["error"])))
            return
        try:
            audio = candidate.resolve(result["audio"]) if audio_first else None
            if result["mode"] != 0:
                skipped += 1
                return
            if audio is None:
                audio = candidate.resolve(result["audio"])
            if not result["has_objects"]:
                return
            if not audio.is_file():
                raise ValueError(f"Audio missing: {result['audio']}")
        except (ValueError, OSError, IndexError) as error:
            errors.append((index, candidate.label(error)))
            return
        accepted.append((index, candidate, result, audio, hasher.submit(audio)))

    try:
        analyze_all([candidates[index].map_path for index in runnable], context, collect, progress)
        context.report(start + 0.9 * (end - start), f"Hashing {len(hasher.futures)} audio files")
        hasher.wait(context)
        items = []
        for index, candidate, result, audio, key in accepted:
            try:
                audio_hash = hasher.result(key)
            except OSError as error:
                errors.append((index, candidate.label(error)))
                continue
            items.append(
                map_record(
                    result["info"],
                    result["map_hash"],
                    candidate.map_path,
                    audio,
                    audio_hash,
                    result["audio"],
                    candidate.source,
                    candidate.source_id,
                    star_rating(candidate.stars),
                )
            )
    finally:
        hasher.close()
    # Content identical to an earlier import keeps its known lazer rating.
    known = store.get_many("map", [item["id"] for item in items if unrated(item)])
    items = [
        with_rating(item, star_rating(known[item["id"]].get("star_rating")))
        if unrated(item) and item["id"] in known
        else item
        for item in items
    ]
    context.report(start + 0.95 * (end - start), f"Saving {len(items)} maps")
    store.save_many("map", items)
    errors.sort(key=lambda entry: entry[0])
    return len(items), skipped, [message for _, message in errors]


def hash_path(root, value):
    if not re.fullmatch(r"[a-fA-F0-9]{64}", value):
        raise ValueError("Invalid lazer content hash")
    return inside(root / "files", f"{value[0]}/{value[:2]}/{value}")


def scan_lazer(root, context):
    if not READER.is_file():
        raise ValueError("Lazer reader missing. Run the setup script or build the Docker image.")
    source = root / "client.realm"
    if not source.is_file():
        raise ValueError("Select the lazer data folder containing client.realm and files.")
    snapshot = DATA / "snapshots" / uuid4().hex
    snapshot.mkdir()
    try:
        before = source.stat()
        shutil.copyfile(source, snapshot / "client.realm")
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(
                "lazer changed its database during copying. Close osu! and scan again."
            )
        context.report(0.05, "Reading a private copy of the lazer database")
        process = subprocess.run(
            ["dotnet", str(READER), str(snapshot / "client.realm")],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
        )
        if process.returncode:
            raise ValueError(process.stderr[-2000:])
        sets = json.loads(process.stdout)
    finally:
        shutil.rmtree(snapshot)

    candidates = []
    for entry in sets:
        files = {item["name"].replace("\\", "/").casefold(): item for item in entry["files"]}
        for item in entry["files"]:
            if not item["name"].lower().endswith(".osu"):
                continue
            candidates.append(lazer_candidate(root, entry["id"], files, item))
    context.report(0.1, f"Reading {len(candidates)} beatmaps from {len(sets)} sets")
    count, skipped, errors = register_many(candidates, context, 0.1, 0.95)
    return {
        "imported": count,
        "skipped_other_modes": skipped,
        "error_count": len(errors),
        "errors": errors[:30],
    }


def lazer_candidate(root, set_id, files, item):
    def resolve(audio_name):
        audio = files.get(audio_name.replace("\\", "/").casefold())
        if audio is None:
            raise ValueError(f"Missing audio mapping for {item['name']}")
        return hash_path(root, audio["hash"])

    try:
        map_path, error = hash_path(root, item["hash"]), None
    except ValueError as invalid:
        map_path, error = None, invalid
    return Candidate(map_path, resolve, str, "lazer", set_id, item.get("star_rating"), error)


def scan_songs(root, context):
    maps = list(root.rglob("*.osu"))
    candidates = [
        Candidate(
            path,
            lambda audio_name, folder=path.parent: inside(folder, audio_name),
            lambda error, name=path.name: f"{name}: {error}",
            "songs",
        )
        for path in maps
    ]
    count, _, errors = register_many(candidates, context, audio_first=True)
    return {"imported": count, "error_count": len(errors), "errors": errors[:30]}


def scan(root, context):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Directory not found. Inside Docker use the mounted path /osu.")
    return (
        scan_lazer(root, context) if (root / "client.realm").exists() else scan_songs(root, context)
    )


def extract_archive(path, destination, context):
    """Validate every member first, then extract in parallel (zlib releases the GIL)."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 10000 or sum(item.file_size for item in entries) > 2 * 1024**3:
            raise ValueError("Archive exceeds 10,000 files or 2 GiB unpacked")
        planned = {}
        for item in entries:
            context.check()
            target = inside(destination, item.filename)
            if target.suffix.lower() not in AUDIO_EXTENSIONS | {".osu"} or item.is_dir():
                continue
            if item.file_size > 512 * 1024**2:
                raise ValueError("Archive member exceeds 512 MiB")
            # Later duplicates win, as with sequential extraction; normcase merges
            # names that collide on case-insensitive file systems.
            key = os.path.normcase(str(target))
            planned.pop(key, None)
            planned[key] = (item, target)
    for parent in {target.parent for _, target in planned.values()}:
        parent.mkdir(parents=True, exist_ok=True)
    if not planned:
        return
    local = threading.local()
    opened, lock = [], threading.Lock()

    def extract(job):
        context.check()
        item, target = job
        archive = getattr(local, "archive", None)
        if archive is None:
            archive = local.archive = zipfile.ZipFile(path)
            with lock:
                opened.append(archive)
        with archive.open(item) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output, 1024 * 1024)

    workers = max(1, min(4, os.cpu_count() or 1, len(planned)))
    if workers == 1:
        with zipfile.ZipFile(path) as archive:
            local.archive = archive
            for job in planned.values():
                extract(job)
        return
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bob-extract")
    try:
        for _ in pool.map(extract, planned.values()):
            pass
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for archive in opened:
            archive.close()


def import_archive(path, context):
    from .import_cleanup import finish_import, rollback_import

    previous = {item["id"]: item for item in store.all_records("map")}
    destination = DATA / "library" / uuid4().hex
    destination.mkdir()
    try:
        context.report(0, "Extracting archive")
        extract_archive(path, destination, context)
        result = scan_songs(destination, context)
        context.check()
        finish_import(destination, previous)
        return result
    except BaseException:
        rollback_import(destination, previous)
        raise
    finally:
        path.unlink(missing_ok=True)


def export_selection(identifiers, context):
    with artifact_directory("outputs") as (output_id, folder):
        destination = folder / "selected-beatmaps.osz"
        written_audio = set()
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for index, identifier in enumerate(identifiers):
                context.report(
                    index / len(identifiers), f"Exporting {index + 1}/{len(identifiers)}"
                )
                item = store.get("map", identifier)
                map_path, audio_path = Path(item["map_path"]), Path(item["audio_path"])
                audio_name = item["audio_hash"] + Path(item["audio_name"]).suffix.lower()
                text = map_path.read_text(encoding="utf-8-sig")
                text = re.sub(r"(?mi)^AudioFilename\s*:.*$", f"AudioFilename: {audio_name}", text)
                # Export training essentials only; omit references to uncopied artwork/storyboards.
                text = re.sub(r"(?ms)^\[Events\].*?(?=^\[|\Z)", "[Events]\n\n", text)
                archive.writestr(f"{identifier}.osu", text)
                if audio_name not in written_audio:
                    archive.write(audio_path, audio_name)
                    written_audio.add(audio_name)
        context.check()
        store.save(
            "output",
            {
                "id": output_id,
                "path": str(destination),
                "name": "selected-beatmaps.osz",
                "created_at": store.now(),
            },
        )
        return {"output_id": output_id}


def create_dataset(identifiers, name, context):
    with artifact_directory("datasets") as (identifier, folder):
        records = []
        for index, map_id in enumerate(identifiers):
            context.report(
                index / len(identifiers), f"Copying training pair {index + 1}/{len(identifiers)}"
            )
            item = store.get("map", map_id)
            audio_name = item["audio_hash"] + Path(item["audio_name"]).suffix.lower()
            audio_path, map_path = folder / audio_name, folder / f"{map_id}.osu"
            if not audio_path.exists():
                shutil.copyfile(item["audio_path"], audio_path)
            shutil.copyfile(item["map_path"], map_path)
            audio_hash = digest(audio_path)
            copied_id = hashlib.sha256((digest(map_path) + audio_hash).encode()).hexdigest()[:24]
            if copied_id != map_id or audio_hash != item["audio_hash"]:
                raise ValueError("Source files changed after scanning. Scan the library again.")
            records.append({**item, "audio_path": str(audio_path), "map_path": str(map_path)})
        dataset = {
            "id": identifier,
            "name": name,
            "count": len(records),
            "songs": len(set(song_groups(records))),
            "created_at": store.now(),
            "maps": records,
        }
        context.check()
        store.save("dataset", dataset)
        return {"dataset_id": identifier}
