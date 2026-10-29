"""Frozen pre-CUDA-optimization batch builder, for reproducible comparisons only."""

import numpy as np
from app.ml.architecture import AUDIO_WINDOW, CENTRAL_FRAMES, AUDIO_MARGIN, EVENT_CONTEXT, EVENT_TARGETS
from app.ml.audio import FRAME_MS
from app.ml.movement import motion_numpy


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


def make_example(item, central, target_indices, rhythm_weight, rng=None):
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
        rows[:count] = item["events"][selected][:, :47]  # later columns are newer targets
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
    if rng is not None:
        has_previous = np.any(rows[:, 15:31] != 0, axis=1)
        for axis in (0, 1):
            if rng.random() < 0.5:
                rows[:, np.arange(1 + axis, 9, 2)] = 1 - rows[:, np.arange(1 + axis, 9, 2)]
                rows[:, np.arange(15 + axis, 25, 2)] = 1 - rows[:, np.arange(15 + axis, 25, 2)]
                rows[:, 31 + axis] = 1 - rows[:, 31 + axis]
                rows[:, 33 + axis] *= -1
        # Mild context noise reduces sensitivity to small generated-position errors.
        rows[:, 15:25] += rng.normal(0, 0.006, rows[:, 15:25].shape).astype(np.float32) * valid[:, None]
        rows[~has_previous, 15:31] = 0
        gain = rng.uniform(0.9, 1.1)
        result["audio"] = np.clip((result["audio"] + 1) * gain - 1, -1, 1)
        result["audio"][:, result["audio_valid"] == 0] = -1
    for index in np.flatnonzero(valid):
        rows[index, 39:47] = motion_numpy(
            rows[index, 1:9],
            rows[index, 10],
            rows[index, 12] + 1,
            rows[index, 9],
            rows[index, 0],
            rows[index, 31:39],
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


def batches(maps, batch_size, rng, shuffle, tasks=None):
    tasks = list(tasks if tasks is not None else tasks_for(maps))
    if shuffle:
        rng.shuffle(tasks)
    for start in range(0, len(tasks), batch_size):
        examples = [
            make_example(maps[index], central, indices, weight, rng if shuffle else None)
            for index, central, indices, weight in tasks[start : start + batch_size]
        ]
        yield {key: np.stack([example[key] for example in examples]) for key in examples[0]}
