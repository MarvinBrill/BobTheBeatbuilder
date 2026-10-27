"""Bounded CPU preparation; preserve record order, split identity and exclusions."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from .split import song_groups


class WorkerContext:
    def __init__(self, context):
        self.context = context

    def check(self):
        self.context.check()

    def report(self, *args, **kwargs):
        self.check()


def prepare_parallel(records, context, issues, operation):
    issues = issues if issues is not None else []
    workers = max(1, min(4, int(os.getenv("BOB_PREPARE_WORKERS", "2"))))
    if workers == 1 or len(records) < 2:
        return operation(records, context, issues)
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(record["audio_hash"], []).append((index, record))
    if len(groups) == 1:
        return operation(records, context, issues)
    songs = song_groups(records)
    results, failures = {}, {}

    def run(group):
        excluded = []
        try:
            prepared = operation([record for _, record in group], WorkerContext(context), excluded)
        except ValueError:
            if len(excluded) != len(group):
                raise
            prepared = []
        return group, prepared, excluded

    def collect(result):
        group, prepared, excluded = result
        by_id = {item["map_id"]: item for item in prepared}
        excluded_by_id = {item["map_id"]: item for item in excluded}
        for index, record in group:
            if record["id"] in by_id:
                results[index] = {**by_id[record["id"]], "song": songs[index]}
            else:
                failures[index] = excluded_by_id[record["id"]]
        completed = len(results) + len(failures)
        context.report(
            0.15 * completed / len(records),
            f"Prepared song/map pairs {completed}/{len(records)} · {workers} workers",
        )

    ordered = list(groups.values())
    # Resolve lazy audio imports/JIT on one thread before concurrent decoding.
    context.report(0, f"Preparing {len(records)} song/map pairs · audio initialization")
    collect(run(ordered[0]))
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bob-prepare")
    try:
        futures = [pool.submit(run, group) for group in ordered[1:]]
        for future in as_completed(futures):
            context.check()
            collect(future.result())
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    issues.extend(failures[index] for index in sorted(failures))
    if not results:
        raise ValueError(f"No usable training data: {issues[:3]}")
    return [results[index] for index in sorted(results)]
