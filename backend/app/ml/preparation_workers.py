"""Bounded CPU preparation; preserve record order, split identity and exclusions."""

import logging
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import numpy as np
from .split import song_groups


class WorkerContext:
    def __init__(self, context):
        self.context = context

    def check(self):
        self.context.check()

    def report(self, *args, **kwargs):
        self.check()


class _ProcessContext:
    """Cancellation is checked by the parent between song groups."""

    def check(self):
        pass

    def report(self, *args, **kwargs):
        pass


def preparation_workers():
    """Beatmap parsing and event packing are pure Python and hold the GIL, so the
    full pipeline runs in worker processes. Leave headroom for the app."""
    configured = os.getenv("BOB_PREPARE_WORKERS")
    if configured:
        return max(1, min(16, int(configured)))
    cpus = (getattr(os, "process_cpu_count", None) or os.cpu_count)() or 2
    return max(2, min(8, cpus - 2))


def _portable(value):
    # Cached arrays are memory-mapped files: send the file reference, not the data.
    if isinstance(value, np.memmap) and value.filename and value.flags.c_contiguous:
        return ("npy", str(value.filename), tuple(value.shape), str(value.dtype))
    return value


def _resolve(value, opened):
    if not isinstance(value, tuple) or len(value) != 4 or value[0] != "npy":
        return value
    _, filename, shape, dtype = value
    array = opened.get(filename)
    if array is None:
        array = opened[filename] = np.load(filename, mmap_mode="r", allow_pickle=False)
    if tuple(array.shape) != shape or str(array.dtype) != dtype:
        raise ValueError(f"Prepared cache changed on disk: {filename}")
    return array


def _run_group(operation, records, context):
    excluded = []
    try:
        prepared = operation(records, context, excluded)
    except ValueError:
        if len(excluded) != len(records):
            raise
        prepared = []
    return prepared, excluded


def _prepare_in_process(records):
    from .sequence_data import prepare_serial

    prepared, excluded = _run_group(prepare_serial, records, _ProcessContext())
    return [{key: _portable(value) for key, value in item.items()} for item in prepared], excluded


def prepare_parallel(records, context, issues, operation, processes=False):
    """`processes=True` requires `operation` to be sequence_data.prepare_serial."""
    issues = issues if issues is not None else []
    workers = preparation_workers()
    if workers == 1 or len(records) < 2:
        return operation(records, context, issues)
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(record["audio_hash"], []).append((index, record))
    if len(groups) == 1:
        return operation(records, context, issues)
    songs = song_groups(records)
    results, failures, done = {}, {}, set()
    opened = {}

    def run(position):
        group = ordered[position]
        prepared, excluded = _run_group(
            operation, [record for _, record in group], WorkerContext(context)
        )
        return position, prepared, excluded

    def collect(result, mode):
        position, prepared, excluded = result
        done.add(position)
        by_id = {item["map_id"]: item for item in prepared}
        excluded_by_id = {item["map_id"]: item for item in excluded}
        for index, record in ordered[position]:
            if record["id"] in by_id:
                item = {key: _resolve(value, opened) for key, value in by_id[record["id"]].items()}
                results[index] = {**item, "song": songs[index]}
            else:
                failures[index] = excluded_by_id[record["id"]]
        completed = len(results) + len(failures)
        context.report(
            0.15 * completed / len(records),
            f"Prepared song/map pairs {completed}/{len(records)} · {workers} {mode}",
        )

    ordered = list(groups.values())
    # Resolve lazy audio imports/JIT once before concurrent decoding.
    context.report(0, f"Preparing {len(records)} song/map pairs · audio initialization")
    collect(run(0), "workers")
    if processes:
        try:
            _in_processes(ordered, workers, context, collect)
        except BrokenProcessPool:
            # Restricted environments may forbid subprocesses; finish the rest on threads.
            logging.exception("Preparation worker processes failed; continuing on threads")
    remaining = [position for position in range(len(ordered)) if position not in done]
    if remaining:
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bob-prepare")
        try:
            futures = [pool.submit(run, position) for position in remaining]
            for future in as_completed(futures):
                context.check()
                collect(future.result(), "workers")
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    issues.extend(failures[index] for index in sorted(failures))
    if not results:
        raise ValueError(f"No usable training data: {issues[:3]}")
    return [results[index] for index in sorted(results)]


def _in_processes(ordered, workers, context, collect):
    # spawn behaves identically on Windows and in Docker and never forks server threads.
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    try:
        futures = {
            pool.submit(_prepare_in_process, [record for _, record in ordered[position]]): position
            for position in range(1, len(ordered))
        }
        for future in as_completed(futures):
            context.check()
            prepared, excluded = future.result()
            collect((futures[future], prepared, excluded), "processes")
    finally:
        # Cancellation waits only for the song groups currently being decoded.
        pool.shutdown(wait=True, cancel_futures=True)
