import logging
import multiprocessing
import os
import sys
from collections import OrderedDict, deque
from multiprocessing import shared_memory
from uuid import uuid4
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import numpy as np
from .architecture import AUDIO_WINDOW, CENTRAL_FRAMES, AUDIO_MARGIN, EVENT_CONTEXT, EVENT_TARGETS
from .audio import FRAME_MS
from .movement import motion_batch_numpy


def clock_features(seconds, duration):
    seconds = np.asarray(seconds, np.float32)
    periods = np.array([0.125, 0.25, 0.5, 1, 2, 4, 8, 16], np.float32)
    phase = seconds[..., None] * (2 * np.pi) / periods
    return np.concatenate(
        [np.sin(phase), np.cos(phase), (seconds / max(duration, 1))[..., None]], axis=-1
    ).astype(np.float32)


def event_queries(times, previous_times, next_times, duration):
    before, after = np.maximum(0, times - previous_times), np.maximum(0, next_times - times)
    return np.concatenate(
        [
            clock_features(times, duration),
            np.log1p(before)[..., None] / 3,
            np.log1p(after)[..., None] / 3,
            np.log((before + 0.025) / (after + 0.025))[..., None] / 4,
        ],
        axis=-1,
    ).astype(np.float32)


def tasks_for(maps):
    tasks = []
    for map_index, item in enumerate(maps):
        frames = item["events"][:, 0] * 1000 / FRAME_MS
        for central in range(0, len(item["mel"]), CENTRAL_FRAMES):
            indices = np.flatnonzero((frames >= central) & (frames < central + CENTRAL_FRAMES))
            chunks = [
                indices[start : start + EVENT_TARGETS] for start in range(0, len(indices), EVENT_TARGETS)
            ] or [indices]
            for chunk in chunks:
                tasks.append((map_index, central, chunk, 1 / len(chunks)))
    return tasks


def audio_window(mel, summary, central):
    start = central - AUDIO_MARGIN
    left, right = max(0, start), min(len(mel), start + AUDIO_WINDOW)
    dest = left - start
    audio, valid = np.full((AUDIO_WINDOW, 64), -1, np.float32), np.zeros(AUDIO_WINDOW, np.float32)
    audio[dest : dest + right - left], valid[dest : dest + right - left] = mel[left:right], 1
    duration = len(mel) * FRAME_MS / 1000
    times = (start + np.arange(AUDIO_WINDOW // 4) * 4) * FRAME_MS / 1000
    global_times = (np.arange(len(summary)) + 0.5) / len(summary) * duration
    return {
        "audio": audio.T.copy(),
        "summary": np.array(summary, copy=True),
        "audio_valid": valid,
        "audio_times": clock_features(times, duration),
        "global_times": clock_features(global_times, duration),
    }


def draw_augmentation(rng):
    """Draw one example's augmentation in exactly the order make_example always used.

    Drawing ahead of time lets examples be built on worker threads while the shared
    training RNG stream stays identical to the sequential implementation.
    """
    flips = (rng.random() < 0.5, rng.random() < 0.5)
    noise = rng.normal(0, 0.006, (EVENT_CONTEXT, 10))
    gain = rng.uniform(0.9, 1.1)
    return flips, noise, gain


def make_example(item, central, target_indices, rhythm_weight, rng=None, augmentation=None):
    result = audio_window(item["mel"], item["summary"], central)
    result["condition"] = item["condition"].copy()
    start, duration = central - AUDIO_MARGIN, len(item["mel"]) * FRAME_MS / 1000
    labels = np.zeros((AUDIO_WINDOW, 6), np.float32)
    left, right = max(0, start), min(len(item["mel"]), start + AUDIO_WINDOW)
    labels[left - start : right - start] = item["labels"][left:right]
    central_mask = np.zeros(AUDIO_WINDOW, np.float32)
    central_mask[AUDIO_MARGIN : AUDIO_MARGIN + CENTRAL_FRAMES] = 1
    labels[:, 4] *= central_mask * rhythm_weight
    result["frame_targets"] = labels.T

    rows = np.zeros((EVENT_CONTEXT, 47), np.float32)
    queries = np.zeros((EVENT_CONTEXT, 20), np.float32)
    valid, mask = np.zeros(EVENT_CONTEXT, np.float32), np.zeros(EVENT_CONTEXT, np.float32)
    if len(target_indices):
        end = int(target_indices[-1]) + 1
        beginning = max(0, end - EVENT_CONTEXT)
        selected = np.arange(beginning, end)
        count = len(selected)
        rows[:count] = item["events"][selected]
        valid[:count] = 1
        mask[:count] = np.isin(selected, target_indices)
        all_times = item["events"][:, 0]
        before = np.where(selected > 0, all_times[np.maximum(0, selected - 1)], 0)
        after = np.where(
            selected + 1 < len(all_times), all_times[np.minimum(len(all_times) - 1, selected + 1)], duration
        )
        queries[:count] = event_queries(rows[:count, 0], before, after, duration)
    else:
        valid[0] = 1  # Valid BOS prevents all-masked attention in rests.
    if augmentation is None and rng is not None:
        augmentation = draw_augmentation(rng)
    if augmentation is not None:
        flips, noise, gain = augmentation
        has_previous = np.any(rows[:, 15:31] != 0, axis=1)
        for axis in (0, 1):
            if flips[axis]:
                rows[:, np.arange(1 + axis, 9, 2)] = 1 - rows[:, np.arange(1 + axis, 9, 2)]
                rows[:, np.arange(15 + axis, 25, 2)] = 1 - rows[:, np.arange(15 + axis, 25, 2)]
                rows[:, 31 + axis] = 1 - rows[:, 31 + axis]
                rows[:, 33 + axis] *= -1
        # Mild context noise reduces sensitivity to small generated-position errors.
        rows[:, 15:25] += noise.astype(np.float32) * valid[:, None]
        rows[~has_previous, 15:31] = 0
        # In place: same values as np.clip((audio + 1) * gain - 1, -1, 1), fewer temporaries.
        audio = result["audio"]
        audio += 1
        audio *= gain
        audio -= 1
        np.clip(audio, -1, 1, out=audio)
        audio[:, result["audio_valid"] == 0] = -1
    indices = np.flatnonzero(valid)
    rows[indices, 39:47] = motion_batch_numpy(
        rows[indices, 1:9],
        rows[indices, 10],
        rows[indices, 12] + 1,
        rows[indices, 9],
        rows[indices, 0],
        rows[indices, 31:39],
    )
    result.update(
        {
            "previous": rows[:, 15:31].copy(),
            "queries": queries,
            "event_valid": valid,
            "event_mask": mask,
            "geometry_target": rows[:, 1:9].copy(),
            "kind_target": rows[:, 9].copy(),
            "seconds_target": rows[:, 10].copy(),
            "duration_target": rows[:, 11].copy(),
            "repeat_target": rows[:, 12].copy(),
            "combo_target": rows[:, 13].copy(),
            "geometry_valid": rows[:, 14].copy(),
            "motion_context": rows[:, 31:39].copy(),
            "motion_target": rows[:, 39:47].copy(),
            "event_times": rows[:, 0].copy(),
        }
    )
    return result


def batch_workers():
    """Processes that build training examples. The per-example NumPy work is many small
    operations that hold the GIL, so threads do not scale; processes do."""
    cpus = (getattr(os, "process_cpu_count", None) or os.cpu_count)() or 2
    default = max(1, min(6, cpus // 2))
    try:
        value = int(os.getenv("BOB_BATCH_WORKERS", default))
    except ValueError:
        value = default
    # Windows caps process pools at 61 handles.
    return max(0, min(32, value))


def _portable(array):
    """Memory-mapped cache arrays travel as file references, never as copied data."""
    if isinstance(array, np.memmap) and array.filename and array.flags.c_contiguous:
        return ("npy", str(array.filename), tuple(array.shape), str(array.dtype))
    return np.asarray(array)


_OPEN = OrderedDict()


def _resolve(value):
    if not isinstance(value, tuple):
        return value
    _, filename, shape, dtype = value
    array = _OPEN.get(filename)
    if array is None:
        array = np.load(filename, mmap_mode="r", allow_pickle=False)
        _OPEN[filename] = array
        while len(_OPEN) > 512:
            _OPEN.popitem(last=False)
    else:
        _OPEN.move_to_end(filename)
    if tuple(array.shape) != shape or str(array.dtype) != dtype:
        raise ValueError(f"Cached training array changed on disk: {filename}")
    return array


def _build(examples):
    built = [
        make_example(item, central, indices, weight, augmentation=draw)
        for item, central, indices, weight, draw in examples
    ]
    return {key: np.stack([example[key] for example in built]) for key in built[0]}


def _resolved(examples):
    items, result = {}, []
    for source, central, indices, weight, draw in examples:
        key = id(source)  # Pickle keeps one object per map inside a task.
        if key not in items:
            items[key] = {name: _resolve(value) for name, value in source.items()}
        result.append((items[key], central, indices, weight, draw))
    return result


_SHARED, _SHARED_CALL = {}, [None]


def _attach(name, call):
    if _SHARED_CALL[0] != call:
        # Slots of earlier batch streams are released; only one stream runs at a time.
        for old in _SHARED.values():
            old.close()
        _SHARED.clear()
        _SHARED_CALL[0] = call
    block = _SHARED.get(name)
    if block is None:
        if sys.version_info >= (3, 13):
            block = shared_memory.SharedMemory(name=name, track=False)
        else:
            block = shared_memory.SharedMemory(name=name)
            if os.name == "posix":
                from multiprocessing import resource_tracker

                # The creating process owns and unlinks the block, not this worker.
                resource_tracker.unregister(block._name, "shared_memory")
        _SHARED[name] = block
    return block


def _build_shared(examples, name, layout, call):
    """Write a stacked batch straight into the caller's shared-memory slot."""
    built = [make_example(*example[:4], augmentation=example[4]) for example in _resolved(examples)]
    block = _attach(name, call)
    count = len(built)
    for key, shape, dtype, offset in layout:
        view = np.ndarray((count, *shape), dtype, buffer=block.buf, offset=offset)
        np.stack([example[key] for example in built], out=view)
        del view
    return count


def _layout(example, batch_size):
    layout, offset = [], 0
    for key, value in example.items():
        layout.append((key, tuple(value.shape), value.dtype.str, offset))
        size = batch_size * value.nbytes
        offset += (size + 63) // 64 * 64
    return layout, max(offset, 64)


class ExampleWorkers:
    """Worker processes shared by all epochs and validation passes of one training run."""

    def __init__(self, workers=None):
        self.workers = batch_workers() if workers is None else max(0, int(workers))
        self.pool, self.failed = None, self.workers < 1

    def executor(self):
        if self.failed:
            return None
        if self.pool is None:
            try:
                # spawn behaves identically on Windows and in Docker and never forks server threads.
                self.pool = ProcessPoolExecutor(
                    max_workers=self.workers, mp_context=multiprocessing.get_context("spawn")
                )
            except (OSError, ValueError, NotImplementedError):
                logging.exception("Batch worker processes unavailable; building batches inline")
                self.failed = True
        return self.pool

    def broken(self):
        logging.exception("Batch worker processes failed; building remaining batches inline")
        self.failed = True
        self.close()

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def batches(maps, batch_size, rng, shuffle, tasks=None, workers=None):
    """Yield stacked batches. `workers` is an optional ExampleWorkers pool.

    Augmentation is drawn sequentially from `rng` before any example is built, so the
    data is identical with or without worker processes and for any worker count.
    """
    tasks = list(tasks if tasks is not None else tasks_for(maps))
    if shuffle:
        rng.shuffle(tasks)

    def plan(start):
        chunk = tasks[start : start + batch_size]
        return [
            (index, central, indices, weight, draw_augmentation(rng) if shuffle else None)
            for index, central, indices, weight in chunk
        ]

    def local(planned):
        return _build([(maps[index], *rest) for index, *rest in planned])

    starts = range(0, len(tasks), batch_size)
    pool = workers.executor() if workers is not None and len(starts) > 1 else None
    if pool is None:
        for start in starts:
            yield local(plan(start))
        return
    sources = {}

    def portable(planned):
        args = []
        for index, *rest in planned:
            if index not in sources:
                sources[index] = {
                    name: _portable(maps[index][name])
                    for name in ("mel", "summary", "events", "labels", "condition")
                }
            args.append((sources[index], *rest))
        return args

    # Shapes and dtypes are fixed; one unaugmented example (no RNG use) gives the layout.
    index, central, indices, weight = tasks[0]
    layout, size = _layout(make_example(maps[index], central, indices, weight), batch_size)
    lookahead = 2 * workers.workers
    call = uuid4().hex
    slots = []
    try:
        slots = [shared_memory.SharedMemory(create=True, size=size) for _ in range(lookahead + 1)]
    except OSError:
        logging.exception("Shared memory unavailable; building batches inline")
        for slot in slots:
            slot.close()
            slot.unlink()
        slots, pool = [], None
    free, pending = deque(slots), deque()

    def collect(slot, count):
        batch = {
            key: np.ndarray((count, *shape), dtype, buffer=slot.buf, offset=offset).copy()
            for key, shape, dtype, offset in layout
        }
        free.append(slot)
        return batch

    def fallback():
        nonlocal pool
        workers.broken()
        pool = None
        for _, item, slot in pending:
            free.append(slot)
            yield local(item)
        pending.clear()

    try:
        for position, start in enumerate(starts):
            planned = plan(start)
            if pool is None:
                yield local(planned)
                continue
            slot = free.popleft()
            try:
                future = pool.submit(_build_shared, portable(planned), slot.name, layout, call)
            except (BrokenProcessPool, RuntimeError):
                free.append(slot)
                yield from fallback()
                yield local(planned)
                continue
            pending.append((future, planned, slot))
            while len(pending) > lookahead or (pending and position + 1 == len(starts)):
                future, item, slot = pending.popleft()
                try:
                    count = future.result()
                except BrokenProcessPool:
                    free.append(slot)
                    yield local(item)
                    yield from fallback()
                    break
                yield collect(slot, count)
    finally:
        for future, _, _ in pending:
            future.cancel()
        for future, _, _ in pending:
            # A running task may still write into its slot; wait before releasing it.
            if not future.cancelled():
                try:
                    future.exception()
                except Exception:
                    pass
        for slot in slots:
            slot.close()
            slot.unlink()
