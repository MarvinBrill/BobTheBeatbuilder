import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from functools import lru_cache
from pathlib import Path
from uuid import uuid4
from . import store
from .artifacts import artifact_directory
from .ml.split import song_groups
from .beatmaps import parse
from .config import AUDIO_EXTENSIONS, DATA, READER, inside
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
    map_hash, audio_hash = digest(map_path), digest(audio_path)
    identifier = hashlib.sha256((map_hash + audio_hash).encode()).hexdigest()[:24]
    rating = star_rating(stars)
    if rating is None and source != "lazer":
        try:
            rating = star_rating(store.get("map", identifier).get("star_rating"))
        except ValueError:
            pass
    item = {
        "id": identifier,
        **beatmap.info(),
        "map_path": str(map_path),
        "audio_path": str(audio_path),
        "audio_hash": audio_hash,
        "audio_name": beatmap.audio,
        "source": source,
        "source_id": source_id,
        "star_rating": rating,
        "star_source": "lazer" if rating is not None else None,
    }
    return store.save("map", item)


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

    count, skipped, errors = 0, 0, []
    for index, entry in enumerate(sets):
        context.report(
            0.1 + 0.85 * index / max(1, len(sets)),
            f"Reading set {index + 1}/{len(sets)} · {count} maps",
        )
        files = {item["name"].replace("\\", "/").casefold(): item for item in entry["files"]}
        for item in entry["files"]:
            if not item["name"].lower().endswith(".osu"):
                continue
            try:
                map_path = hash_path(root, item["hash"])
                beatmap = parse(map_path)
                if beatmap.mode != 0:
                    skipped += 1
                    continue
                audio = files.get(beatmap.audio.replace("\\", "/").casefold())
                if audio is None:
                    raise ValueError(f"Missing audio mapping for {item['name']}")
                if register(
                    map_path,
                    hash_path(root, audio["hash"]),
                    "lazer",
                    entry["id"],
                    item.get("star_rating"),
                ):
                    count += 1
            except (ValueError, OSError, IndexError) as error:
                errors.append(str(error))
    return {
        "imported": count,
        "skipped_other_modes": skipped,
        "error_count": len(errors),
        "errors": errors[:30],
    }


def scan_songs(root, context):
    maps = list(root.rglob("*.osu"))
    count, errors = 0, []
    for index, path in enumerate(maps):
        context.report(index / max(1, len(maps)), f"Reading beatmap {index + 1}/{len(maps)}")
        try:
            beatmap = parse(path)
            if register(path, inside(path.parent, beatmap.audio), "songs"):
                count += 1
        except (ValueError, OSError, IndexError) as error:
            errors.append(f"{path.name}: {error}")
    return {"imported": count, "error_count": len(errors), "errors": errors[:30]}


def scan(root, context):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Directory not found. Inside Docker use the mounted path /osu.")
    return (
        scan_lazer(root, context) if (root / "client.realm").exists() else scan_songs(root, context)
    )


def import_archive(path, context):
    from .import_cleanup import finish_import, rollback_import

    previous = {item["id"]: item for item in store.all_records("map")}
    destination = DATA / "library" / uuid4().hex
    destination.mkdir()
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > 10000 or sum(item.file_size for item in entries) > 2 * 1024**3:
                raise ValueError("Archive exceeds 10,000 files or 2 GiB unpacked")
            for item in entries:
                context.check()
                target = inside(destination, item.filename)
                if target.suffix.lower() not in AUDIO_EXTENSIONS | {".osu"} or item.is_dir():
                    continue
                if item.file_size > 512 * 1024**2:
                    raise ValueError("Archive member exceeds 512 MiB")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
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
