"""Parallel beatmap analysis for library scans and archive imports.

Parsing (slider flattening and curve fitting) is CPU-bound Python, so it runs in
spawned worker processes. Audio hashing is I/O-bound and releases the GIL, so it
runs on threads in the calling process where the digest cache lives. Results are
always returned in input order so registration stays deterministic.
"""

import hashlib
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from .beatmaps import parse

# Below this many maps, starting worker processes costs more than it saves.
PROCESS_THRESHOLD = 24
EXPECTED_ERRORS = (ValueError, OSError, IndexError)


def analyze(filename):
    """Parse one .osu file in a worker; never raises the expected per-map errors."""
    path = Path(filename)
    try:
        beatmap = parse(path)
        result = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                result.update(block)
        return {
            "mode": beatmap.mode,
            "has_objects": bool(beatmap.objects),
            "audio": beatmap.audio,
            "info": beatmap.info(),
            "map_hash": result.hexdigest(),
        }
    except EXPECTED_ERRORS as error:
        return {"error": str(error)}


def worker_count(total):
    configured = os.getenv("BOB_IMPORT_WORKERS")
    if configured:
        # Windows caps process pools at 61 handles.
        workers = max(1, min(32, int(configured)))
    else:
        cpus = (getattr(os, "process_cpu_count", None) or os.cpu_count)() or 2
        # Leave a core for the web server; beyond ~8 workers scans become disk-bound and
        # every worker costs its own numpy import (~30 MB).
        workers = min(8, max(min(2, cpus), cpus - 1))
    return max(1, min(workers, total))


class Progress:
    """Throttle job updates: every report is a database write and an ETag change."""

    def __init__(self, context, start, end, interval=0.25):
        self.context, self.start, self.end, self.interval = context, start, end, interval
        self.last = 0.0

    def __call__(self, done, total, message):
        now = time.monotonic()
        if now - self.last >= self.interval:
            self.last = now
            fraction = done / max(1, total)
            self.context.report(self.start + (self.end - self.start) * fraction, message)
        else:
            self.context.check()


def analyze_all(filenames, context, on_result, progress):
    """Run analyze() over filenames, calling on_result(index, result) in order."""
    filenames = [str(name) for name in filenames]
    total = len(filenames)
    workers = worker_count(total) if total >= PROCESS_THRESHOLD else 1
    done = 0
    if workers > 1:
        try:
            done = _in_processes(filenames, workers, context, on_result, progress)
        except BrokenProcessPool:
            # Restricted environments may forbid subprocesses; finish the rest inline.
            logging.exception("Import worker processes failed; continuing in-process")
    for index in range(done, total):
        on_result(index, analyze(filenames[index]))
        progress(index + 1, total, f"Reading beatmap {index + 1}/{total}")
    return workers


def _in_processes(filenames, workers, context, on_result, progress):
    total, done = len(filenames), 0
    # spawn behaves identically on Windows and in Docker and never forks server threads.
    spawn = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=spawn)
    try:
        chunk = max(1, min(32, total // (workers * 8)))
        for result in pool.map(analyze, filenames, chunksize=chunk):
            on_result(done, result)
            done += 1
            progress(done, total, f"Reading beatmap {done}/{total} · {workers} workers")
    except BrokenProcessPool:
        if done == 0:
            raise
        # Partial results were consumed; let the caller redo only the remainder inline.
        logging.exception("Import worker processes failed after %s maps", done)
        return done
    finally:
        # Wait for in-flight chunks so no worker still holds files that a rollback deletes.
        pool.shutdown(wait=True, cancel_futures=True)
    return done


class Hasher:
    """Hash each distinct audio file once on a small thread pool while parsing continues."""

    def __init__(self, digest, workers=None):
        self.digest = digest
        self.pool = ThreadPoolExecutor(
            max_workers=workers or max(2, min(8, os.cpu_count() or 2)),
            thread_name_prefix="bob-hash",
        )
        self.futures = {}

    def submit(self, path):
        key = os.path.normcase(str(path))
        if key not in self.futures:
            self.futures[key] = self.pool.submit(self.digest, path)
        return key

    def result(self, key):
        return self.futures[key].result()

    def wait(self, context):
        pending = list(self.futures.values())
        for index, future in enumerate(pending):
            if index % 16 == 0:
                context.check()
            future.exception()

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
