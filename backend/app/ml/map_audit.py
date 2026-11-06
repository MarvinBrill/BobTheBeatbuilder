"""Measurements a mapper would check first: is it on the music, and do the patterns vary?

The quality report already counted overlaps and repairs, but none of the failures of the
October 2026 maps showed up in it: a grid half a beat off the music (Dealer), 90 % sharp
turns in same-direction chains (pentagram "star jumps"), every slider half a beat long
with its own slider velocity. These metrics measure exactly that, from the objects, the
red lines and the mel features, and compare them with typical human values.

Reference values (median of 14 human 5.6-6.6 star training maps, same measurement):
sharp turns 44 %, objects in star chains 7 % (longest chain 4), objects within 20 ms of a
mel onset 70 %, strongest onsets covered 74 %, half-beat sliders 47 %, 4 slider velocities.
"""

import math

import numpy as np
from scipy.signal import find_peaks

from .tempo_grid import grid_drift, onset_envelopes

HUMAN_REFERENCE = {
    "sharp_share": 0.45,
    "same_rotation_share": 0.54,
    "star_chain_share": 0.06,
    "longest_star_chain": 4,
    "onset_hit_share": 0.70,
    "strong_onset_coverage": 0.74,
    "slider_half_beat_share": 0.47,
    "slider_velocities": 4,
}
# A warning starts well outside the human range (p10-p90 of the reference maps).
LIMITS = {
    "star_chain_share": 0.25,
    "sharp_share": 0.65,
    "onset_hit_share": 0.45,
    "strong_onset_coverage": 0.55,
    "slider_half_beat_share": 0.85,
    "timing_drift_ms": 30.0,
    "longest_same_gap_bars": 8.0,
    "slider_velocities": 12,
}


def circle_radius(cs):
    return 54.4 - 4.48 * cs


def _turns(objects, radius):
    """(turn angle in degrees, rotation sign, index) for consecutive moving pairs."""
    vectors, lengths = [], []
    for a, b in zip(objects, objects[1:]):
        vector = np.array([b.x - a.end_x, b.y - a.end_y], np.float64)
        vectors.append(vector)
        lengths.append(float(np.linalg.norm(vector)))
    turns = []
    for k in range(1, len(vectors)):
        if lengths[k] > radius and lengths[k - 1] > radius:
            cosine = float(vectors[k] @ vectors[k - 1] / (lengths[k] * lengths[k - 1]))
            cross = vectors[k - 1][0] * vectors[k][1] - vectors[k - 1][1] * vectors[k][0]
            turns.append((math.degrees(math.acos(np.clip(cosine, -1, 1))), np.sign(cross), k))
    return turns, np.asarray(lengths)


def star_chains(turns, low=115.0, high=170.0, members=False):
    """Lengths (in turns) of runs of sharp turns that keep rotating the same way; with
    `members`, also the set of object indices that take part in such a run."""
    chains, inside, current, sign, previous, first = [], set(), 0, 0, -5, None

    def close():
        if current >= 3:
            chains.append(current)
            # Turn k is at object k; the run involves the objects first-1 .. last+1.
            inside.update(range(first - 1, previous + 2))

    for angle, rotation, index in turns:
        sharp = low <= angle <= high
        if sharp and current and rotation == sign and index == previous + 1:
            current += 1
        else:
            close()
            current, first = (1, index) if sharp else (0, None)
        sign, previous = rotation, index
    close()
    return (chains, inside) if members else chains


def movement_audit(objects, cs=4.0):
    if len(objects) < 4:
        return {}
    radius = circle_radius(cs)
    turns, lengths = _turns(objects, radius)
    angles = np.array([turn[0] for turn in turns])
    same = [
        turns[j][1] == turns[j - 1][1]
        for j in range(1, len(turns))
        if turns[j][2] == turns[j - 1][2] + 1 and turns[j][0] > 115 and turns[j - 1][0] > 115
    ]
    chains, members = star_chains(turns, members=True)
    gaps = np.array([(b.time - a.time) / max(a.beat_length, 1) for a, b in zip(objects, objects[1:])])

    def median(mask):
        return float(np.median(lengths[mask])) if np.any(mask) else None

    half = median((gaps > 0.27) & (gaps <= 0.55))
    whole = median((gaps > 0.55) & (gaps <= 1.05))
    sliders = [obj for obj in objects if obj.slider and obj.duration > 0]
    spans = [obj.duration / max(obj.beat_length, 1) / max(1, obj.repeats) for obj in sliders]
    velocities = [
        round(obj.path_length * obj.repeats * obj.beat_length / (140 * obj.duration), 2)
        for obj in sliders
        if obj.path_length
    ]
    cosine = []
    for k in range(1, len(objects) - 1):
        a, b, c = objects[k - 1], objects[k], objects[k + 1]
        first = np.array([b.x - a.end_x, b.y - a.end_y])
        second = np.array([c.x - b.end_x, c.y - b.end_y])
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator > 1e-6:
            cosine.append(float(first @ second / denominator))
    result = {
        "sharp_share": float((angles > 120).mean()) if len(angles) else None,
        "same_rotation_share": float(np.mean(same)) if same else None,
        "star_chain_share": len(members) / len(objects),
        "longest_star_chain": max(chains, default=0),
        "flow": float((np.mean(cosine) + 1) / 2) if cosine else None,
        "spacing_half": half,
        "spacing_beat": whole,
        "spacing_rhythm_ratio": whole / half if half and whole else None,
        "slider_half_beat_share": (
            float(np.mean([abs(span - 0.5) < 0.05 for span in spans])) if spans else None
        ),
        "slider_lengths": len({round(span, 3) for span in spans}) if spans else 0,
        "slider_velocities": len(set(velocities)),
    }
    return result


def rhythm_audit(objects, mel, frame_ms, timing=None):
    """Onset agreement (after removing one global offset) and rhythm monotony."""
    if len(objects) < 4 or mel is None or not len(mel):
        return {}
    full, _, _ = onset_envelopes(mel)
    peaks, _ = find_peaks(full, height=0.15, distance=max(1, int(40 / frame_ms)))
    if not len(peaks):
        return {}
    peak_times, strength = peaks * frame_ms, full[peaks]
    times = np.array([obj.time for obj in objects])
    nearest = np.array([peak_times[np.argmin(np.abs(peak_times - time))] - time for time in times])
    close = np.abs(nearest) < 40
    offset = float(np.median(nearest[close])) if np.any(close) else 0.0
    hit = float(np.mean(np.abs(nearest - offset) <= 20))
    strong = peak_times[strength >= np.percentile(strength, 80)]
    strong = strong[(strong > times.min()) & (strong < times.max())]
    covered = (
        float(np.mean([np.min(np.abs(times + offset - time)) <= 25 for time in strong]))
        if len(strong)
        else None
    )
    gaps = [round((b.time - a.time) / max(a.beat_length, 1) * 12) for a, b in zip(objects, objects[1:])]
    counts = np.bincount(np.clip(gaps, 0, 200))
    share = counts / max(1, counts.sum())
    entropy = float(-(share[share > 0] * np.log2(share[share > 0])).sum())
    longest, run = 0.0, 0.0
    for k in range(1, len(gaps)):
        if gaps[k] == gaps[k - 1] and gaps[k] <= 12:
            run += gaps[k] / 12
            longest = max(longest, run)
        else:
            run = 0.0
    result = {
        "onset_hit_share": hit,
        "strong_onset_coverage": covered,
        "onset_offset_ms": offset,
        "gap_entropy_bits": entropy,
        "top3_gap_share": float(np.sort(share)[-3:].sum()),
        "longest_same_gap_bars": longest / 4,
    }
    if timing:
        windows = [item for item in grid_drift(full, frame_ms, timing) if item[2] >= 0.15]
        if len(windows) >= 3:
            phases = np.array([item[1] for item in windows])
            beats = np.array([item[3] for item in windows])
            centred = phases - np.median(phases)
            # Windows dominated by off-beat sounds sit half a beat away without any drift;
            # fold them onto the beat so only a wandering grid counts.
            half = beats / 2
            folded = (centred + half / 2) % half - half / 2
            result["timing_drift_ms"] = float(np.percentile(np.abs(folded), 75))
    return result


def audit(objects, timing, mel, frame_ms, cs=4.0, reference=None):
    """All metrics plus German warnings for the studio."""
    values = {**movement_audit(objects, cs), **rhythm_audit(objects, mel, frame_ms, timing)}
    reference = {**HUMAN_REFERENCE, **(reference or {})}
    warnings = []

    def above(key, text):
        value = values.get(key)
        if value is not None and value > LIMITS[key]:
            warnings.append(text.format(value=value))

    def below(key, text):
        value = values.get(key)
        if value is not None and value < LIMITS[key]:
            warnings.append(text.format(value=value))

    above("timing_drift_ms", "Raster wandert gegen die Musik (um ~{value:.0f} ms) – Timing prüfen.")
    below("onset_hit_share", "Nur {value:.0%} der Objekte liegen auf einem hörbaren Onset.")
    below("strong_onset_coverage", "Nur {value:.0%} der stärksten Akzente sind gemappt.")
    above("star_chain_share", "{value:.0%} der Objekte stecken in Stern-Ketten (gleichgerichtete scharfe Drehungen).")
    above("sharp_share", "{value:.0%} scharfe Richtungswechsel (Menschen ~45 %).")
    above("slider_half_beat_share", "{value:.0%} der Slider sind genau 1/2 Beat lang.")
    above("longest_same_gap_bars", "Gleicher Rhythmus über {value:.0f} Takte am Stück.")
    above("slider_velocities", "{value} verschiedene Slider-Geschwindigkeiten (Menschen ~4).")
    ratio = values.get("spacing_rhythm_ratio")
    if ratio is not None and ratio < 0.8:
        warnings.append("1/1-Abstände sind kleiner als 1/2-Abstände: Rhythmus schwer lesbar.")
    return {
        "metrics": {key: (round(value, 4) if isinstance(value, float) else value) for key, value in values.items()},
        "reference": reference,
        "warnings": warnings,
    }
