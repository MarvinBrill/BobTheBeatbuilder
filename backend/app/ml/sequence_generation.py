import numpy as np
import torch
from copy import deepcopy
from dataclasses import replace
from scipy.signal import find_peaks
from .architecture import EVENT_CONTEXT, CENTRAL_FRAMES
from .audio import FRAME_MS
from .sequence_data import event_features, summary_features
from .sequence_batches import event_queries, event_queries_v2
from .rhythm_inference import encode_window, predict_rhythm, timing_map, timing_at, select_times
from .playability import quality_report
from .candidate_selection import choose_candidate, rank_candidates  # noqa: F401 (re-exported)
from .generation_trace import object_snapshot
from .joint_planning import skip_option
from .continuation_search import choose_continuation
from .motif_memory import MotifMemory, analysis_spans, link_spans
from .style import GAP_BEATS, REST, STREAM_GAP, allowed_gaps
from .rhythm_guidance import build_guide, loudness_profile
from .spinners import drop_spinners, fill_spinners
from ..timing import red_points


def decoder_input(objects, times, index, duration, selected_device, timing, styled=False):
    """Decoder history and queries; style-v2 models never see the gap to the next head."""
    if styled:
        beats = [grid_point(timing, time).beat_length for time in times[: index + 1]]
        return decoder_input_v2(objects, times, beats, index, duration, selected_device, timing)
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
    mapper=None,
):
    """Onset-first generation: select all heads from the rhythm map, then decode objects."""
    styled = getattr(model, "styled", False)
    summary = summary_features(mel)
    prediction = predict_rhythm(model, mel, summary, condition, selected_device, context, mapper)
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
                model, mel, summary, condition, central, selected_device, mapper
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
            history, planned_times, query_index, duration, selected_device, timing, styled
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
        rhythm="onsets",
    )
    return objects, timing, report


# ---------------------------------------------------------------------------------------
# Style v2: the decoder chooses the rhythm itself, one beat-grid gap at a time, so bursts,
# streams and 1/2 jump sections are learned as sequences instead of independent onsets.


def grid_point(timing, time):
    """Red line whose grid covers `time`; the first one also extends backwards."""
    points = red_points(timing)
    if points and time < points[0].time:
        return points[0]
    return timing_at(points, time)


def snap_to_grid(time, timing, divisor):
    point = grid_point(timing, time)
    grid = point.beat_length / divisor
    return float(point.time + round((time - point.time) / grid) * grid)


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


def spinner_lead(span, beat):
    """The lead time stored with a planned spinner (older 3-tuples: half a beat)."""
    return span[3] if len(span) > 3 else max(0.5 * beat, 150.0)


def placed_spinners(objects, reserved, active, timing, end, settings, truncated=False):
    """Reserved spans that stayed free of objects, plus long audible gaps and the outro."""
    spinners = []
    for span in reserved:
        left, right = span[0], span[1]
        lead = spinner_lead(span, grid_point(timing, left).beat_length)
        if not objects or objects[-1].time < right:
            continue  # the rhythm never resumed after it: the map ended before the drop
        if all(
            obj.time + obj.duration <= left - lead + 1e-6 or obj.time >= right + 1e-6
            for obj in objects
        ):
            spinners.append((left, right))
    if not truncated:
        spinners += fill_spinners(objects, active, timing, end, settings, spinners)
    return sorted(spinners)


def decoder_input_v2(objects, times, beats, index, duration, selected_device, timing):
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
    query = event_queries_v2(current, before, beat_seconds, duration, timing)
    return (
        torch.from_numpy(previous).unsqueeze(0).to(selected_device),
        torch.from_numpy(query).unsqueeze(0).to(selected_device),
        torch.ones((1, len(indices)), dtype=torch.float32, device=selected_device),
    )


def choose_next_time(
    logits, obj, timing, onset, active, end, settings, run, rng, guide=None, spinner=None
):
    """Combine the learned rhythm distribution with audible onsets on the beat grid.

    `guide` (a RhythmGuide) adds soft costs for gaps far denser than the song's local
    density envelope. `spinner` is the next reserved (start, end, resume_at) span: no head
    may start inside it or too close before it, and the rhythm resumes at `resume_at`.
    """
    divisor = int(settings.get("divisor", 4))
    allowed = allowed_gaps(divisor)
    beat = obj.beat_length
    shifted = logits - logits.max()
    log_probs = shifted - np.log(np.exp(shifted).sum())
    release = obj.time + obj.duration + (max(35.0, 0.24 * beat) if obj.slider else 35.0)
    weight = float(settings.get("onset_weight", 0.7))
    bias = float(settings.get("tempo_bias", 0.0))
    longest = int(settings.get("max_stream") or 0)
    blocked_from = blocked_until = None
    if spinner is not None:
        blocked_from = spinner[0] - spinner_lead(spinner, beat)
        blocked_until = spinner[2]

    def free(time):
        return blocked_from is None or not blocked_from < time < blocked_until

    def cost(time, short=False):
        return guide.cost(obj.time, time, run, short) if guide is not None else 0.0

    options = []
    for index, gap in enumerate(GAP_BEATS):
        if not allowed[index]:
            continue
        short = gap <= STREAM_GAP
        # `run` counts consecutive short gaps; a run of n gaps spans n + 1 objects.
        if short and longest and run + 2 > longest:
            continue
        time = snap_to_grid(obj.time + gap * beat, timing, divisor)
        if time < release - 1e-6 or time > end or not audible(active, time) or not free(time):
            continue
        evidence = onset_evidence(onset, time)
        options.append(
            (
                log_probs[index]
                + weight * np.log(evidence + 1e-3)
                - bias * np.log2(gap)
                - cost(time, short),
                time,
                index,
            )
        )
    near = blocked_from is not None and release < blocked_until
    near = near and blocked_from - obj.time <= 8.5 * beat
    if near:
        # The spinner follows: the next head is the first onset after its recovery time.
        resume = next_onset(
            onset, active, max(release, blocked_until), end, timing, divisor, anywhere=True
        )
        if resume is not None:
            options.append((log_probs[REST] - cost(resume), resume, REST))
    else:
        pause = next_onset(onset, active, max(release, obj.time + 8.5 * beat), end, timing, divisor)
        if pause is not None and free(pause):
            options.append(
                (
                    log_probs[REST] + weight * np.log(onset_evidence(onset, pause) + 1e-3)
                    - cost(pause),
                    pause,
                    REST,
                )
            )
    if not options:
        # Silence or a break ahead: continue with the next audible onset, if there is one.
        resume = next_onset(onset, active, release, end, timing, divisor, anywhere=True)
        if resume is not None and not free(resume):
            resume = next_onset(onset, active, blocked_until, end, timing, divisor, anywhere=True)
        return (resume, REST) if resume is not None else None
    scores = np.array([option[0] for option in options], np.float64)
    temperature = max(0.05, float(settings.get("rhythm_temperature", 0.6)))
    scores = (scores - scores.max()) / temperature
    chances = np.exp(scores) / np.exp(scores).sum()
    _, time, index = options[int(rng.choice(len(options), p=chances))]
    return time, index


def generate_objects_v2(
    model,
    mel,
    audio_path,
    settings,
    condition,
    selected_device,
    context,
    max_objects=None,
    mapper=None,
    trace=None,
    evaluation_range=None,
    motif_memory=None,
):
    """Pattern-first generation for style-v2 models: place an object, then let the rhythm
    head choose the beat-grid gap to the next one, combined with the audible onsets."""
    summary = summary_features(mel)
    prediction = predict_rhythm(model, mel, summary, condition, selected_device, context, mapper)
    timing, timing_source = timing_map(prediction, audio_path, settings)
    # Training only ever saw beat confidence 1, 0.75 (slightly disturbed timing) or no timing;
    # the audio fallback's 0.35 put every musical-time query out of distribution.
    timing = [
        replace(point, beat_confidence=max(point.beat_confidence, 0.75))
        if not point.inherited and 0 < point.beat_confidence < 0.75
        else point
        for point in timing
    ]
    if motif_memory is None and settings.get("motifs", False):
        motif_memory = MotifMemory(link_spans(mel, analysis_spans(mel, timing)))
    active = np.max(mel, axis=1) > -0.7
    onset = prediction[:, 0]
    duration = (len(mel) - 1) * FRAME_MS / 1000
    start, stop = evaluation_range if evaluation_range is not None else (0, duration * 1000)
    if not 0 <= start < stop <= duration * 1000 + FRAME_MS:
        raise ValueError("Invalid evaluation range")
    end = min(duration * 1000 - 100, stop)
    divisor = int(settings.get("divisor", 4))
    first = next_onset(onset, active, start, end, timing, divisor, threshold=0.3, anywhere=True)
    if first is None and evaluation_range is None:
        raise ValueError(
            "No playable notes predicted. Check the audio, model training and density."
        )
    rng = np.random.default_rng(settings["seed"])
    objects, repairs, classes = [], [], []
    times = [first] if first is not None else []
    beats = [grid_point(timing, first).beat_length] if first is not None else []
    run, central_cache, generated_plan, truncated = 0, None, None, False
    guide = build_guide(prediction, mel, settings)
    loudness = loudness_profile(mel)
    spans = []
    if first is not None:
        spans = [
            span
            for span in drop_spinners(loudness, active, timing, start, end, settings)
            if span[0] - span[3] > first
        ]
    reserved = list(spans)
    if trace is not None and spans:
        trace.record("spinner_plan", spans=[list(span) for span in spans], source="loudness_jump")
    with torch.no_grad():
        while times:
            index, time = len(objects), times[-1]
            while spans and spans[0][2] <= time - 1e-6:
                spans.pop(0)  # passed; `reserved` keeps every planned span
            upcoming = spans[0] if spans else None
            context.report(
                0.35 + 0.55 * min(1.0, (time - start) / max(1.0, end - start)),
                f"Rhythmus und Bewegung erzeugen · {index + 1} Objekte",
            )
            central = int(time / FRAME_MS) // CENTRAL_FRAMES * CENTRAL_FRAMES
            if central_cache != central:
                memory, memory_valid, _, plan = encode_window(
                    model, mel, summary, condition, central, selected_device, mapper
                )
                if generated_plan is None:
                    generated_plan = plan[0].detach().float().cpu().numpy()
                central_cache = central
            previous, queries, valid = decoder_input_v2(
                objects, times, beats, index, duration, selected_device, timing
            )
            output = model.decode(previous, queries, valid, memory, memory_valid)
            if not all(torch.isfinite(value).all() for value in output.values()):
                raise ValueError("Model generated non-finite object predictions")
            beat = beats[-1]
            # Sliders may not run into silence, the end of the audio or past eight beats.
            horizon = min(time + 8 * beat, silence_after(active, time), end + 100) + 35
            if upcoming is not None and time < upcoming[0]:
                # ...nor into a reserved spinner (keeping the lead time before it).
                horizon = max(time + 36, min(horizon, upcoming[0] - spinner_lead(upcoming, beat) + 35))
            chosen = rank_candidates(
                model,
                output,
                time,
                horizon,
                beat,
                settings,
                objects,
                rng,
                selected_device,
                trace,
                motif_memory=motif_memory,
            )[0]
            objects.append(chosen.obj)
            repairs.append(chosen.repair)
            if trace is not None:
                trace.record(
                    "object_choice",
                    time=time,
                    candidate_id=chosen.identifier,
                    object=object_snapshot(chosen.obj),
                    repair=chosen.repair,
                )
            obj = chosen.obj
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
            if guide is not None:
                guide.history.append(obj.time)
            choice = choose_next_time(
                logits, obj, timing, onset, active, end, settings, run, rng, guide, upcoming
            )
            if choice is None:
                break
            if max_objects is not None and len(objects) >= max_objects:
                truncated = True
                break
            following, gap = choice
            if trace is not None:
                trace.record("rhythm_choice", time=time, next_time=following, gap_class=gap)
            run = run + 1 if gap != REST and GAP_BEATS[gap] <= STREAM_GAP else 0
            times.append(following)
            beats.append(grid_point(timing, following).beat_length)
            classes.append(gap)
    spinners = placed_spinners(objects, reserved, active, timing, end, settings, truncated)
    report = quality_report(objects, settings)
    report.update(
        timing_source=timing_source,
        timing_sections=len(timing),
        unknown_meter_sections=sum(point.meter_confidence == 0 for point in timing),
        export_meter_default=4,
        mean_geometry_repair=float(np.mean(repairs)) if repairs else 0.0,
        heavily_repaired_fraction=float(np.mean(np.asarray(repairs) > 0.1)) if repairs else 0.0,
        evaluation_start_ms=float(start),
        evaluation_end_ms=float(stop),
        spinners=[[float(left), float(right)] for left, right in spinners],
        density_guidance=guide is not None,
        selected_count_before_limit=len(objects) + int(truncated),
        truncated=truncated,
        skipped_interior_heads=0,
        scope="section" if evaluation_range is not None else "song",
        motif_links=len(motif_memory.links) if motif_memory is not None else 0,
        phrase_plan=generated_plan.tolist() if generated_plan is not None else [],
        rhythm="learned-sequence",
    )
    return objects, timing, report
