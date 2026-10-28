import numpy as np
import torch
from copy import deepcopy
from .architecture import EVENT_CONTEXT, CENTRAL_FRAMES
from .audio import FRAME_MS
from .sequence_data import event_features, summary_features
from .sequence_batches import event_queries
from .rhythm_inference import encode_window, predict_rhythm, timing_map, timing_at, select_times
from .playability import quality_report
from .candidate_selection import choose_candidate, rank_candidates
from .generation_trace import object_snapshot
from .joint_planning import skip_option
from .continuation_search import choose_continuation
from .motif_memory import MotifMemory, analysis_spans, link_spans


def decoder_input(objects, times, index, duration, selected_device, timing):
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
    query = event_queries(current, before, after, duration, timing)
    return (
        torch.from_numpy(previous).unsqueeze(0).to(selected_device),
        torch.from_numpy(query).unsqueeze(0).to(selected_device),
        torch.ones((1, len(indices)), dtype=torch.float32, device=selected_device),
    )


def generate_objects(
    model,
    mel,
    audio_path,
    settings,
    condition,
    selected_device,
    context,
    max_objects=None,
    trace=None,
    evaluation_range=None,
    motif_memory=None,
):
    summary = summary_features(mel)
    prediction = predict_rhythm(model, mel, summary, condition, selected_device, context)
    timing, timing_source = timing_map(prediction, audio_path, settings)
    if motif_memory is None and settings.get("motifs", False):
        motif_memory = MotifMemory(link_spans(mel, analysis_spans(mel, timing)))
        if trace is not None:
            from dataclasses import asdict

            trace.record(
                "motif_hypotheses",
                links=[asdict(link) for link in motif_memory.links],
                source="within_song_ordered_spectrum",
                musical_phrase_labels=False,
            )
    times = select_times(prediction, mel, settings, timing, trace=trace)
    duration = (len(mel) - 1) * FRAME_MS / 1000
    start, end = evaluation_range if evaluation_range is not None else (0, duration * 1000)
    if not 0 <= start < end <= duration * 1000 + FRAME_MS:
        raise ValueError("Invalid evaluation range")
    # Preserve the real next head outside an excerpt/object cap for query and duration planning.
    times = [time for time in times if time >= start]
    selected_count = sum(time < end for time in times)
    object_count = min(selected_count, max_objects) if max_objects is not None else selected_count
    if not object_count and evaluation_range is None:
        raise ValueError(
            "No playable notes predicted. Check the audio, model training and density."
        )
    rng = np.random.default_rng(settings["seed"])
    objects, repairs, skipped_heads, memories = [], [], 0, {}
    generated_plan = None

    def decode_history(history, planned_times, query_index):
        nonlocal generated_plan
        central = int(planned_times[query_index] / FRAME_MS) // CENTRAL_FRAMES * CENTRAL_FRAMES
        if central not in memories:
            memory, memory_valid, _, plan = encode_window(
                model, mel, summary, condition, central, selected_device
            )
            if generated_plan is None:
                generated_plan = plan[0].detach().float().cpu().numpy()
                if trace is not None:
                    trace.record(
                        "phrase_plan",
                        proxies=generated_plan,
                        musical_layer="unknown",
                        semantic_style_labels=False,
                    )
            memories[central] = (memory, memory_valid)
            if len(memories) > 2:
                del memories[next(iter(memories))]
        previous, queries, valid = decoder_input(
            history, planned_times, query_index, duration, selected_device, timing
        )
        return model.decode(previous, queries, valid, *memories[central])

    with torch.no_grad():
        while len(objects) < object_count and len(objects) < len(times):
            index = len(objects)
            time = times[index]
            if time >= end:
                break
            context.report(
                0.35 + 0.55 * index / max(1, object_count),
                f"Bewegungsfolge erzeugen · {index+1}/{object_count}",
            )
            beat = timing_at(timing, time).beat_length
            plans = [(times, None, 0.0)]
            alternative = skip_option(times, index, beat, prediction, mel, trace)
            if alternative is not None:
                plans.append(alternative)
            choices = []
            for branch, (planned_times, skipped_time, penalty) in enumerate(plans):
                output = decode_history(objects, planned_times, index)
                next_time = (
                    planned_times[index + 1] if index + 1 < len(planned_times) else duration * 1000
                )
                ranked = rank_candidates(
                    model,
                    output,
                    time,
                    next_time,
                    beat,
                    settings,
                    objects,
                    rng,
                    selected_device,
                    trace,
                    branch=f"time-{branch}",
                    motif_memory=motif_memory,
                )
                for candidate in ranked:
                    if skipped_time is not None and (
                        not candidate.obj.slider
                        or candidate.obj.time + candidate.obj.duration < skipped_time + 35
                    ):
                        continue
                    choices.append(
                        (candidate.score - penalty, candidate, planned_times, skipped_time)
                    )
            if settings.get("lookahead", False) and index + 1 < object_count:
                # Equal random proposals across branches; lookahead does not consume the main RNG.
                branch_rng_state = deepcopy(rng.bit_generator.state)

                def evaluate_next(history, planned_times, branch):
                    next_index = len(history)
                    if next_index >= len(planned_times) or planned_times[next_index] >= end:
                        return []
                    next_time = planned_times[next_index]
                    following_time = (
                        planned_times[next_index + 1]
                        if next_index + 1 < len(planned_times)
                        else duration * 1000
                    )
                    following_beat = timing_at(timing, next_time).beat_length
                    branch_rng = np.random.default_rng(0)
                    branch_rng.bit_generator.state = deepcopy(branch_rng_state)
                    return rank_candidates(
                        model,
                        decode_history(history, planned_times, next_index),
                        next_time,
                        following_time,
                        following_beat,
                        settings,
                        history,
                        branch_rng,
                        selected_device,
                        trace,
                        branch=f"continuation-{branch}",
                        motif_memory=motif_memory,
                    )

                choice = choose_continuation(choices, objects, evaluate_next, trace)
            else:
                choice = max(choices, key=lambda choice: choice[0])
            plan_score, chosen, times, skipped_time = choice
            objects.append(chosen.obj)
            repairs.append(chosen.repair)
            skipped_heads += int(skipped_time is not None)
            if trace is not None:
                trace.record(
                    "object_choice",
                    time=time,
                    candidate_id=chosen.identifier,
                    object=object_snapshot(chosen.obj),
                    repair=chosen.repair,
                    skipped_time=skipped_time,
                    plan_score=plan_score,
                )
    report = quality_report(objects, settings)
    report.update(
        timing_source=timing_source,
        timing_sections=len(timing),
        unknown_meter_sections=sum(point.meter_confidence == 0 for point in timing),
        export_meter_default=4,
        mean_geometry_repair=float(np.mean(repairs)) if repairs else 0.0,
        heavily_repaired_fraction=float(np.mean(np.asarray(repairs) > 0.1)) if repairs else 0.0,
        evaluation_start_ms=float(start),
        evaluation_end_ms=float(end),
        selected_count_before_limit=selected_count,
        truncated=bool(
            max_objects is not None
            and len(objects) >= max_objects
            and len(objects) < len(times)
            and times[len(objects)] < end
        ),
        skipped_interior_heads=skipped_heads,
        scope="section" if evaluation_range is not None else "song",
        motif_links=len(motif_memory.links) if motif_memory is not None else 0,
        phrase_plan=generated_plan.tolist() if generated_plan is not None else [],
    )
    return objects, timing, report
