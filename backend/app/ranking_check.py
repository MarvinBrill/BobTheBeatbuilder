"""A Ranking-Criteria checklist for generated drafts.

Modders in the osu! forums ("Mapping Discussion") review maps against the Ranking Criteria
per difficulty tier, and point to exact editor timestamps. This module runs the parts of
that review a program can measure on our own export:

* general rules: snapping, drain time, hitsound feedback, preview point, unmapped outro;
* tier guidelines: AR/OD/HP/CS ranges, spinner length and recovery, overlaps, stream and
  1/2-chain limits, Easy/Normal distance snapping, Normal rest spacing;
* readability for new players: a third object that sits visually closer to the first one
  than to its real predecessor is easily read in the wrong order (forum poll "Visually
  similar spacings between objects in lower difficulties", t/2247525, 2026: 75 % of 164
  votes see this as an issue in Easy, 48 % also in Normal), and the Hard guideline against
  similar visual spacing for different rhythms.

Findings carry editor timestamps (mm:ss:mmm) so they can be pasted into osu!. The list is
a review aid. It never claims rankability: the Ranking Criteria require hit objects,
hitsounds and timing to come from direct human input.
"""

import math
from statistics import median
from .tiers import RULES, tier_name
from .timing import red_points, timing_at

MAX_TIMES = 6
TICK_TOLERANCE_MS = 2.0
MIN_DRAIN_MS = 30000
OUTRO_SHARE = 0.2


def circle_radius(cs):
    return 54.4 - 4.48 * float(cs)


def timestamp(ms):
    ms = max(0, int(round(ms)))
    return f"{ms // 60000:02d}:{ms // 1000 % 60:02d}:{ms % 1000:03d}"


def grid_point(timing, time):
    points = red_points(timing)
    if points and time < points[0].time:
        return points[0]
    return timing_at(points, time)


def beat_at(timing, time):
    beat = grid_point(timing, time).beat_length
    return beat if beat > 0 else 500.0


def end_time(obj):
    return obj.time + obj.duration


def distance(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


class Report:
    def __init__(self):
        self.findings = []

    def add(self, identifier, severity, title, message, times=()):
        times = sorted(times)
        self.findings.append(
            {
                "id": identifier,
                "severity": severity,
                "title": title,
                "message": message,
                "count": len(times),
                "times": [timestamp(time) for time in times[:MAX_TIMES]],
            }
        )


def check_settings(report, rules, settings):
    for key, label in (("ar", "AR"), ("od", "OD"), ("hp", "HP"), ("cs", "CS")):
        low, high = getattr(rules, key)
        value = settings.get(key)
        if value is None:
            continue
        if (low is not None and value < low - 1e-6) or (high is not None and value > high + 1e-6):
            wanted = f"{low if low is not None else '–'}–{high if high is not None else '–'}"
            if low is None:
                wanted = f"höchstens {high}"
            elif high is None:
                wanted = f"mindestens {low}"
            report.add(
                f"settings_{key}",
                "warning",
                f"{label} außerhalb der {rules.name}-Richtlinie",
                f"{label} {value:g} – für {rules.name} empfiehlt das RC {wanted}.",
            )


def check_spinners(report, rules, objects, spinners, timing):
    short, rushed = [], []
    for start, end in sorted(spinners):
        beat = beat_at(timing, start)
        if rules.spinner_min_beats and (end - start) / beat < rules.spinner_min_beats - 0.05:
            short.append(start)
        following = next((obj.time for obj in objects if obj.time > end), None)
        if following is not None and (following - end) / beat < rules.spinner_recovery_beats - 0.05:
            rushed.append(end)
    if short:
        report.add(
            "spinner_length",
            "warning",
            "Spinner zu kurz",
            f"{rules.name}: Spinner sollten mindestens {rules.spinner_min_beats:g} Beats lang sein.",
            short,
        )
    if rushed:
        report.add(
            "spinner_recovery",
            "warning",
            "Zu wenig Erholung nach dem Spinner",
            f"{rules.name}: nach einem Spinner mindestens {rules.spinner_recovery_beats:g} Beat(s) "
            "Pause bis zum nächsten Objekt.",
            rushed,
        )


def check_overlaps(report, rules, objects, timing, radius):
    if rules.full_overlap_beats is None:
        return
    overlaps, touching = [], []
    for previous, obj in zip(objects, objects[1:]):
        beat = beat_at(timing, obj.time)
        gap = (obj.time - end_time(previous)) / beat
        head_gap = (obj.time - previous.time) / beat
        spacing = distance(previous.end_x, previous.end_y, obj.x, obj.y)
        if spacing < 3:
            continue  # osu! offsets exact stacks; the RC allows stacks in Normal and above
        if rules.name in ("Easy", "Normal"):
            # Easy: avoid overlaps; Normal: only 1/2 overlaps caused by distance snap are fine.
            limit = 1.0 if rules.name == "Easy" else 0.75
            if spacing < 2 * radius and (rules.name == "Easy" or head_gap >= limit) and gap <= 1.05:
                touching.append(obj.time)
        elif spacing < 0.25 * radius and gap <= rules.full_overlap_beats + 0.01:
            overlaps.append(obj.time)
    if touching:
        report.add(
            "overlap",
            "warning",
            "Überlappende Objekte",
            f"{rules.name}: Kreise, Köpfe und Enden sollten sich nicht überlappen – mehr Abstand "
            "oder höhere Slider-Geschwindigkeit.",
            touching,
        )
    if overlaps:
        report.add(
            "full_overlap",
            "problem" if rules.overlap_is_rule else "warning",
            "Fast vollständige Überlappung",
            f"{rules.name}: Objekte im Abstand von höchstens {rules.full_overlap_beats:g} Beat "
            "dürfen sich nicht vollständig überdecken.",
            overlaps,
        )


def short_runs(objects, timing, limit_beats):
    """Runs of consecutive heads at most `limit_beats` apart: [(first time, objects)]."""
    found, start, count = [], None, 1
    for previous, obj in zip(objects, objects[1:]):
        if (obj.time - previous.time) / beat_at(timing, obj.time) <= limit_beats + 0.02:
            if start is None:
                start, count = previous.time, 1
            count += 1
        else:
            if start is not None:
                found.append((start, count))
            start = None
    if start is not None:
        found.append((start, count))
    return found


def check_rhythm(report, rules, objects, timing):
    if rules.max_stream is not None:
        long_runs = [
            start for start, count in short_runs(objects, timing, 0.25) if count > rules.max_stream
        ]
        if long_runs:
            message = (
                f"{rules.name}: Streams mit mehr als {rules.max_stream} Noten vermeiden; kurze "
                "Repeat-Slider können sie ersetzen."
                if rules.max_stream
                else f"{rules.name}: 1/4-Rhythmen sind für diese Stufe zu dicht."
            )
            report.add("stream_length", "warning", "Zu dichte Rhythmen", message, long_runs)
    if rules.max_half_chain is not None:
        chains, chain, chain_start = [], 0, None
        for previous, obj in zip(objects, objects[1:]):
            gap = (obj.time - previous.time) / beat_at(timing, obj.time)
            clicked = not previous.slider and 0.4 <= gap <= 0.6
            if clicked:
                chain_start = previous.time if chain == 0 else chain_start
                chain += 1
                if chain == rules.max_half_chain + 1:
                    chains.append(chain_start)
            else:
                chain = 0
        if chains:
            report.add(
                "half_chain",
                "warning",
                "Lange 1/2-Ketten",
                f"{rules.name}: höchstens {rules.max_half_chain} aktiv geklickte 1/2-Rhythmen "
                "hintereinander.",
                chains,
            )
    if rules.name == "Normal":
        gaps, last_rest = [], objects[0].time if objects else 0
        for previous, obj in zip(objects, objects[1:]):
            beat = beat_at(timing, obj.time)
            meter = grid_point(timing, obj.time).meter or 4
            rest = (
                obj.time - end_time(previous)
            ) / beat >= 0.95 or previous.duration > 2 * meter * beat
            if rest:
                last_rest = obj.time
            elif obj.time - last_rest > 2 * meter * beat + 1:
                gaps.append(obj.time)
                last_rest = obj.time
        if gaps:
            report.add(
                "normal_rest",
                "warning",
                "Keine 1/1-Pause in zwei Takten",
                "Normal (als niedrigste Stufe): mindestens alle zwei Takte eine Lücke von 1/1 oder mehr.",
                gaps,
            )
    if rules.slow_rhythm and len(objects) > 1:
        fast = [
            obj.time
            for previous, obj in zip(objects, objects[1:])
            if not previous.slider and (obj.time - previous.time) / beat_at(timing, obj.time) < 0.9
        ]
        if len(fast) > 0.15 * (len(objects) - 1):
            report.add(
                "easy_density",
                "warning",
                "Zu viele schnelle Noten für Easy",
                f"Easy: überwiegend 1/1 oder langsamer – {round(100 * len(fast) / (len(objects) - 1))} % "
                "der geklickten Abstände sind kürzer.",
                fast,
            )


def check_distance_snap(report, rules, objects, timing):
    if rules.distance_snap is None:
        return
    pairs = []
    for previous, obj in zip(objects, objects[1:]):
        beats = (obj.time - end_time(previous)) / beat_at(timing, obj.time)
        spacing = distance(previous.end_x, previous.end_y, obj.x, obj.y)
        if 0.25 <= beats <= rules.distance_snap_max_beats + 0.01 and spacing >= 3:
            pairs.append((obj.time, spacing / beats))
    if len(pairs) < 8:
        return
    typical = median(ratio for _, ratio in pairs)
    low, high = rules.distance_snap
    outside = [time for time, ratio in pairs if not low * typical <= ratio <= high * typical]
    if len(outside) > 0.1 * len(pairs):
        report.add(
            "distance_snap",
            "warning",
            "Uneinheitliche Abstände (Distance Snap)",
            f"{rules.name}: gleiche Zeit = gleicher Abstand. {round(100 * len(outside) / len(pairs))} % "
            f"der Abstände liegen außerhalb {low:g}–{high:g}× des typischen Abstands.",
            outside,
        )


def check_visual_spacing(report, rules, objects, timing, radius):
    if rules.name not in ("Easy", "Normal", "Hard"):
        return
    misleading, similar = [], []
    for first, second, third in zip(objects, objects[1:], objects[2:]):
        beat = beat_at(timing, third.time)
        if (third.time - end_time(second)) / beat > 2.05:
            continue
        real = distance(second.end_x, second.end_y, third.x, third.y)
        decoy = min(
            distance(first.x, first.y, third.x, third.y),
            distance(first.end_x, first.end_y, third.x, third.y),
        )
        # The third object looks like it follows the first: new players read the order wrong.
        if rules.name != "Hard" and decoy >= 3 and real > 2 * radius and decoy < 0.8 * real:
            misleading.append(third.time)
        if rules.name == "Hard":
            gaps = (
                (second.time - end_time(first)) / beat,
                (third.time - end_time(second)) / beat,
            )
            spacings = (distance(first.end_x, first.end_y, second.x, second.y), real)
            if (
                min(gaps) > 0.2
                and max(gaps) / min(gaps) >= 1.9
                and min(spacings) > radius
                and max(spacings) / min(spacings) <= 1.15
            ):
                similar.append(third.time)
    if misleading:
        report.add(
            "reading_order",
            "warning",
            "Missverständliche Lesereihenfolge",
            f"{rules.name}: das Objekt liegt näher am vorletzten als am direkt vorherigen Objekt. "
            "Neue Spieler folgen dem nächstgelegenen Kreis statt den Nummern (Forum-Umfrage 2026).",
            misleading,
        )
    if similar:
        report.add(
            "similar_spacing",
            "warning",
            "Ähnlicher Abstand für unterschiedliche Rhythmen",
            "Hard: 1/2 und 1/1 sollten optisch unterscheidbar sein; Sprünge auf Betonungen sind okay, "
            "wenn sie klar größer sind.",
            similar,
        )


def unsnapped(timing, time):
    point = grid_point(timing, time)
    beats = (time - point.time) / point.beat_length
    errors = []
    for divisor in (16, 12):
        ticks = beats * divisor
        errors.append(abs(ticks - round(ticks)) * point.beat_length / divisor)
    return min(errors) > TICK_TOLERANCE_MS


def check_general(report, objects, spinners, timing, audio_ms, finish):
    times = []
    for obj in objects:
        times.append(obj.time)
        if obj.slider and obj.duration > 0:
            times.append(end_time(obj))
    times += [value for span in spinners for value in span]
    off_grid = [time for time in times if unsnapped(timing, time)]
    if off_grid:
        report.add(
            "snapping",
            "problem",
            "Nicht auf dem Raster",
            "Objekte müssen weniger als 2 ms neben einem 1/16- oder 1/12-Tick liegen.",
            off_grid,
        )
    spans = sorted([(obj.time, end_time(obj)) for obj in objects] + list(spinners))
    if spans:
        last = max(end for _, end in spans)
        breaks = sum(end - start for start, end in (finish.breaks if finish else ()))
        drain = last - spans[0][0] - breaks
        if drain < MIN_DRAIN_MS:
            report.add(
                "drain_time",
                "problem",
                "Drain-Zeit unter 30 Sekunden",
                f"Jede Difficulty braucht mindestens 30 s Drain-Zeit (aktuell {drain / 1000:.1f} s).",
            )
        if audio_ms and audio_ms - last > OUTRO_SHARE * audio_ms:
            report.add(
                "outro",
                "warning",
                "Ungemapptes Outro",
                f"Die letzten {(audio_ms - last) / 1000:.0f} s sind leer. Das RC empfiehlt, mehr "
                "als 20 % ungemapptes Audio am Ende abzuschneiden.",
                [last],
            )
    if objects and not any(obj.hitsound or any(obj.edge_sounds or ()) for obj in objects):
        report.add(
            "hitsounds",
            "warning",
            "Keine Hitsound-Additions",
            "Das RC verlangt eine Hitsound-Ausarbeitung (Whistle/Finish/Clap auf wichtigen Stellen). "
            "Option „Hitsounds“ aktivieren oder im Editor ergänzen.",
        )
    if finish is not None and finish.preview_time < 0:
        report.add(
            "preview_point",
            "warning",
            "Kein Preview-Point",
            "Preview-Points müssen gesetzt und in allen Difficulties gleich sein.",
        )


def ranking_check(objects, spinners, timing, settings, stars=None, audio_ms=None, finish=None):
    """Findings for one generated difficulty; `stars` selects the tier."""
    objects = sorted(objects, key=lambda obj: obj.time)
    spinners = [tuple(span) for span in spinners]
    report = Report()
    report.add(
        "ai_policy",
        "info",
        "Entwurf, nicht rankbar",
        "Laut Ranking Criteria müssen Objekte, Hitsounds und Timing ausschließlich von Menschen "
        "stammen. Nutze die Map als Ideenquelle und Übungsentwurf; für Ranked/Loved neu mappen.",
    )
    name = tier_name(stars)
    radius = circle_radius(settings.get("cs", 4))
    check_general(report, objects, spinners, timing, audio_ms, finish)
    if name and objects:
        rules = RULES[name]
        check_settings(report, rules, settings)
        check_spinners(report, rules, objects, spinners, timing)
        check_overlaps(report, rules, objects, timing, radius)
        check_rhythm(report, rules, objects, timing)
        check_distance_snap(report, rules, objects, timing)
        check_visual_spacing(report, rules, objects, timing, radius)
    order = {"problem": 0, "warning": 1, "info": 2}
    findings = sorted(report.findings, key=lambda item: (order[item["severity"]], item["id"]))
    return {
        "tier": name,
        "stars": round(float(stars), 2) if stars is not None else None,
        "problems": sum(item["severity"] == "problem" for item in findings),
        "warnings": sum(item["severity"] == "warning" for item in findings),
        "findings": findings,
        "note": "Automatische Prüfung eines Teils der Ranking Criteria; ersetzt keinen Mod.",
    }
