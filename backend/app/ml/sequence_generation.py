import numpy as np
import torch
from .architecture import EVENT_CONTEXT, CENTRAL_FRAMES, DURATION_BEATS
from .audio import FRAME_MS
from .sequence_data import event_features, summary_features
from .sequence_batches import event_queries
from .rhythm_inference import encode_window, predict_rhythm, timing_map, timing_at, select_times
from .playability import candidate_object, readability_cost, quality_report
from .slider_geometry import geometry_candidates


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
