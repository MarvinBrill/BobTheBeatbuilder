import numpy as np
import torch
from scipy.signal import find_peaks
from .architecture import EVENT_CONTEXT, CENTRAL_FRAMES, DURATION_BEATS
from .audio import FRAME_MS
from .sequence_data import event_features, summary_features
from .sequence_batches import event_queries, event_queries_v2
from .rhythm_inference import encode_window, predict_rhythm, timing_map, timing_at, select_times
from .playability import candidate_object, readability_cost, quality_report
from .slider_geometry import geometry_candidates
from .style import GAP_BEATS, REST, STREAM_GAP, allowed_gaps


def probabilities(logits, temperature=1):
    scores = np.asarray(logits, dtype=np.float64) / temperature
    scores -= scores.max()
    weights = np.exp(scores)
    return weights / weights.sum()


def decoder_input(objects, times, index, duration, selected_device):
    beginning = max(0, index - EVENT_CONTEXT + 1)
    indices = list(range(beginning, index + 1))
    previous = np.stack(
        [
            (
                event_features(objects[i - 1], objects[i - 2] if i > 1 else None)
                if i
                else np.zeros(16, np.float32)
            )
            for i in indices
        ]
    )
    current = np.asarray([times[i] / 1000 for i in indices])
    before = np.asarray([times[i - 1] / 1000 if i else 0 for i in indices])
    after = np.asarray([times[i + 1] / 1000 if i + 1 < len(times) else duration for i in indices])
    query = event_queries(current, before, after, duration)
    return (
        torch.from_numpy(previous).unsqueeze(0).to(selected_device),
        torch.from_numpy(query).unsqueeze(0).to(selected_device),
        torch.ones((1, len(indices)), dtype=torch.float32, device=selected_device),
    )


def choose_candidate(model, output, time, next_time, beat, settings, objects, rng, selected_device):
    raw = output["geometry"][0, -1].cpu().numpy().reshape(model.config.mixtures, 17)
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite geometry from model")
    weights = probabilities(raw[:, 0], settings.get("temperature", 0.8))
    means = 1 / (1 + np.exp(-np.clip(raw[:, 1:9], -60, 60))) * 2 - 0.5
    scales = np.exp(np.clip(raw[:, 9:], -4.5, -0.5))
    slider_probability = probabilities(output["kind"][0, -1].cpu().numpy())[1]
    slider = settings["slider_bias"] > 0 and (
        settings["slider_bias"] >= 1 or rng.random() < slider_probability
    )
    duration_probabilities = probabilities(output["duration"][0, -1].cpu().numpy())
    repeat_probabilities = probabilities(output["repeats"][0, -1].cpu().numpy())
    combo = float(output["combo"][0, -1].cpu()) > 0
    expected_motion = output["motion"][0, -1].cpu().numpy()
    candidates = []
    for geometry, mode in geometry_candidates(
        means,
        scales,
        weights,
        settings.get("candidates", 12),
        settings.get("temperature", 0.8),
        rng,
        slider,
    ):
        duration_index = (
            duration_probabilities.argmax()
            if mode
            else rng.choice(len(DURATION_BEATS), p=duration_probabilities)
        )
        repeat_index = (
            repeat_probabilities.argmax() if mode else rng.choice(4, p=repeat_probabilities)
        )
        duration = DURATION_BEATS[duration_index]
        repeats = int(repeat_index) + 1
        candidate = candidate_object(
            geometry, time, next_time, beat, slider, duration, repeats, combo, settings, objects
        )
        if candidate is None:
            continue
        obj, geometry, motion, repair = candidate
        dimensions = 8 if obj.slider else 2
        log_components = np.log(weights + 1e-12) - (
            0.5 * ((geometry[:dimensions] - means[:, :dimensions]) / scales[:, :dimensions]) ** 2
            + np.log(scales[:, :dimensions])
        ).sum(-1)
        maximum = log_components.max()
        likelihood = (maximum + np.log(np.exp(log_components - maximum).sum())) / dimensions
        movement_cost = np.minimum(np.abs(motion - expected_motion), 5).mean()
        score = (
            0.15 * likelihood
            - movement_cost
            - 3 * repair
            - 0.3 * readability_cost(obj, objects, settings["cs"], settings["ar"])
        )
        candidates.append((obj, geometry, motion, score, repair))
    if not candidates:
        # No suitable shape/duration/velocity: a circle is safer than inventing a curve.
        candidate = candidate_object(
            means[int(weights.argmax())],
            time,
            next_time,
            beat,
            False,
            0,
            1,
            combo,
            settings,
            objects,
        )
        obj, geometry, motion, repair = candidate
        candidates = [(obj, geometry, motion, -1, repair + 1)]
    geometries = torch.tensor(
        np.stack([item[1] for item in candidates]), dtype=torch.float32, device=selected_device
    )
    motions = torch.tensor(
        np.stack([item[2] for item in candidates]), dtype=torch.float32, device=selected_device
    )
    hidden = output["hidden"][0, -1].unsqueeze(0).expand(len(candidates), -1)
    critic = model.compatibility(hidden, geometries, motions).cpu().numpy()
    choice = int(
        np.argmax([item[3] + 0.2 * np.tanh(critic[index]) for index, item in enumerate(candidates)])
    )
    return candidates[choice][0], candidates[choice][4]


def generate_objects(
    model, mel, audio_path, settings, condition, selected_device, context, max_objects=None
):
    summary = summary_features(mel)
    prediction = predict_rhythm(model, mel, summary, condition, selected_device, context)
    timing, timing_source = timing_map(prediction, audio_path, settings)
    times = select_times(prediction, mel, settings, timing)
    if max_objects is not None:
        times = times[:max_objects]
    if not times:
        raise ValueError(
            "No playable notes predicted. Check the audio, model training and density."
        )
    rng = np.random.default_rng(settings["seed"])
    objects, repairs, central_cache = [], [], None
    duration = (len(mel) - 1) * FRAME_MS / 1000
    with torch.no_grad():
        for index, time in enumerate(times):
            context.report(
                0.35 + 0.55 * index / len(times),
                f"Bewegungsfolge erzeugen · {index+1}/{len(times)}",
            )
            central = int(time / FRAME_MS) // CENTRAL_FRAMES * CENTRAL_FRAMES
            if central_cache != central:
                memory, memory_valid, _ = encode_window(
                    model, mel, summary, condition, central, selected_device
                )
                central_cache = central
            previous, queries, valid = decoder_input(
                objects, times, index, duration, selected_device
            )
            output = model.decode(previous, queries, valid, memory, memory_valid)
            if not all(torch.isfinite(value).all() for value in output.values()):
                raise ValueError("Model generated non-finite object predictions")
            next_time = times[index + 1] if index + 1 < len(times) else duration * 1000
            _, beat = timing_at(timing, time)
            obj, repair = choose_candidate(
                model, output, time, next_time, beat, settings, objects, rng, selected_device
            )
            objects.append(obj)
            repairs.append(repair)
    report = quality_report(objects, settings)
    report.update(
        timing_source=timing_source,
        timing_sections=len(timing),
        mean_geometry_repair=float(np.mean(repairs)),
        heavily_repaired_fraction=float(np.mean(np.asarray(repairs) > 0.1)),
    )
    return objects, timing, report


# ---------------------------------------------------------------------------------------
# Style v2: the decoder chooses the rhythm itself, one beat-grid gap at a time, so bursts,
# streams and 1/2 jump sections are learned as sequences instead of independent onsets.


def snap_to_grid(time, timing, divisor):
    offset, beat = timing_at(timing, time)
    grid = beat / divisor
    return float(offset + round((time - offset) / grid) * grid)


def onset_evidence(onset, time):
    frame = int(round(time / FRAME_MS))
    return float(onset[max(0, frame - 2) : frame + 3].max(initial=0.0))


def audible(active, time):
    frame = int(round(time / FRAME_MS))
    return 0 <= frame < len(active) and bool(active[frame])


def next_onset(onset, active, start, end, timing, divisor, threshold=0.25, anywhere=False):
    """First clear onset at or after `start` in audible audio, on the beat grid.

    With `anywhere`, an audible grid position is used when the model predicts no onset
    at all, so a song with sound is never left unmapped after a break.
    """
    first = max(0, int(np.ceil(start / FRAME_MS)))
    last = min(len(onset), int(end / FRAME_MS) + 1)
    if last - first < 3:
        return None
    for height in (threshold, 0.1):
        peaks, _ = find_peaks(onset[first:last], height=height, distance=3)
        for peak in peaks + first:
            if not active[peak]:
                continue
            time = snap_to_grid(peak * FRAME_MS, timing, divisor)
            if start - 1e-6 <= time <= end and audible(active, time):
                return time
    if anywhere:
        for frame in np.flatnonzero(active[first:last]) + first:
            time = snap_to_grid(frame * FRAME_MS, timing, divisor)
            if start - 1e-6 <= time <= end and audible(active, time):
                return time
    return None


def silence_after(active, time):
    frame = int(round(time / FRAME_MS))
    quiet = np.flatnonzero(~active[frame:])
    return (frame + quiet[0]) * FRAME_MS if len(quiet) else len(active) * FRAME_MS


def decoder_input_v2(objects, times, beats, index, duration, selected_device):
    beginning = max(0, index - EVENT_CONTEXT + 1)
    indices = list(range(beginning, index + 1))
    previous = np.stack(
        [
            (
                event_features(objects[i - 1], objects[i - 2] if i > 1 else None)
                if i
                else np.zeros(16, np.float32)
            )
            for i in indices
        ]
    )
    current = np.asarray([times[i] / 1000 for i in indices])
    before = np.asarray([times[i - 1] / 1000 if i else 0 for i in indices])
    beat_seconds = np.asarray([beats[i] / 1000 for i in indices])
    query = event_queries_v2(current, before, beat_seconds, duration)
    return (
        torch.from_numpy(previous).unsqueeze(0).to(selected_device),
        torch.from_numpy(query).unsqueeze(0).to(selected_device),
        torch.ones((1, len(indices)), dtype=torch.float32, device=selected_device),
    )


def choose_next_time(logits, obj, timing, onset, active, end, settings, run, rng):
    """Combine the learned rhythm distribution with audible onsets on the beat grid."""
    divisor = int(settings.get("divisor", 4))
    allowed = allowed_gaps(divisor)
    beat = obj.beat_length
    shifted = logits - logits.max()
    log_probs = shifted - np.log(np.exp(shifted).sum())
    release = obj.time + obj.duration + (max(35.0, 0.24 * beat) if obj.slider else 35.0)
    weight = float(settings.get("onset_weight", 0.7))
    bias = float(settings.get("tempo_bias", 0.0))
    longest = int(settings.get("max_stream") or 0)
    options = []
    for index, gap in enumerate(GAP_BEATS):
        if not allowed[index]:
            continue
        # `run` counts consecutive short gaps; a run of n gaps spans n + 1 objects.
        if gap <= STREAM_GAP and longest and run + 2 > longest:
            continue
        time = snap_to_grid(obj.time + gap * beat, timing, divisor)
        if time < release - 1e-6 or time > end or not audible(active, time):
            continue
        evidence = onset_evidence(onset, time)
        options.append(
            (log_probs[index] + weight * np.log(evidence + 1e-3) - bias * np.log2(gap), time, index)
        )
    pause = next_onset(onset, active, max(release, obj.time + 8.5 * beat), end, timing, divisor)
    if pause is not None:
        options.append(
            (log_probs[REST] + weight * np.log(onset_evidence(onset, pause) + 1e-3), pause, REST)
        )
    if not options:
        # Silence or a break ahead: continue with the next audible onset, if there is one.
        resume = next_onset(onset, active, release, end, timing, divisor, anywhere=True)
        return (resume, REST) if resume is not None else None
    scores = np.array([option[0] for option in options], np.float64)
    temperature = max(0.05, float(settings.get("rhythm_temperature", 0.6)))
    scores = (scores - scores.max()) / temperature
    chances = np.exp(scores) / np.exp(scores).sum()
    _, time, index = options[int(rng.choice(len(options), p=chances))]
    return time, index


def generate_objects_v2(
    model, mel, audio_path, settings, condition, selected_device, context, max_objects=None, mapper=None
):
    summary = summary_features(mel)
    prediction = predict_rhythm(model, mel, summary, condition, selected_device, context, mapper)
    timing, timing_source = timing_map(prediction, audio_path, settings)
    active = np.max(mel, axis=1) > -0.7
    onset = prediction[:, 0]
    duration = (len(mel) - 1) * FRAME_MS / 1000
    end = duration * 1000 - 100
    divisor = int(settings.get("divisor", 4))
    first = next_onset(onset, active, 0.0, end, timing, divisor, threshold=0.3, anywhere=True)
    if first is None:
        raise ValueError(
            "No playable notes predicted. Check the audio, model training and density."
        )
    rng = np.random.default_rng(settings["seed"])
    objects, times, beats, repairs, classes = [], [first], [timing_at(timing, first)[1]], [], []
    run, central_cache = 0, None
    with torch.no_grad():
        while True:
            index, time = len(objects), times[-1]
            context.report(
                0.35 + 0.55 * min(1.0, time / max(1.0, end)),
                f"Rhythmus und Bewegung erzeugen · {index + 1} Objekte",
            )
            central = int(time / FRAME_MS) // CENTRAL_FRAMES * CENTRAL_FRAMES
            if central_cache != central:
                memory, memory_valid, _ = encode_window(
                    model, mel, summary, condition, central, selected_device, mapper
                )
                central_cache = central
            previous, queries, valid = decoder_input_v2(
                objects, times, beats, index, duration, selected_device
            )
            output = model.decode(previous, queries, valid, memory, memory_valid)
            if not all(torch.isfinite(value).all() for value in output.values()):
                raise ValueError("Model generated non-finite object predictions")
            beat = beats[-1]
            # Sliders may not run into silence, the end of the audio or past eight beats.
            horizon = min(time + 8 * beat, silence_after(active, time), end + 100) + 35
            obj, repair = choose_candidate(
                model, output, time, horizon, beat, settings, objects, rng, selected_device
            )
            objects.append(obj)
            repairs.append(repair)
            if max_objects is not None and len(objects) >= max_objects:
                break
            logits = (
                model.gap_logits(
                    output["hidden"][:, -1],
                    torch.tensor([float(obj.slider)], device=selected_device),
                    torch.tensor([obj.duration / beat], device=selected_device),
                    torch.tensor([float(obj.repeats)], device=selected_device),
                )[0]
                .float()
                .cpu()
                .numpy()
            )
            if not np.isfinite(logits).all():
                raise ValueError("Model generated non-finite rhythm predictions")
            choice = choose_next_time(logits, obj, timing, onset, active, end, settings, run, rng)
            if choice is None:
                break
            following, gap = choice
            run = run + 1 if gap != REST and GAP_BEATS[gap] <= STREAM_GAP else 0
            times.append(following)
            beats.append(timing_at(timing, following)[1])
            classes.append(gap)
    report = quality_report(objects, settings)
    report.update(
        timing_source=timing_source,
        timing_sections=len(timing),
        mean_geometry_repair=float(np.mean(repairs)),
        heavily_repaired_fraction=float(np.mean(np.asarray(repairs) > 0.1)),
        rhythm="learned-sequence",
    )
    return objects, timing, report
