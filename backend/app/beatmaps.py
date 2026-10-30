import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from .slider_paths import endpoint, slider_path
from .curve_geometry import fitted_curve, curve_length
from .timing import TimingPoint, red_points, timing_at


@dataclass
class HitObject:
    x: float
    y: float
    time: float
    slider: bool = False
    duration: float = 0
    end_x: float = 256
    end_y: float = 192
    geometry_valid: bool = True
    curve: list | None = None
    repeats: int = 1
    beat_length: float = 500
    new_combo: bool = False
    path_length: float = 0
    # osu! additions bitmask (2 whistle, 4 finish, 8 clap): the circle's sound, or the
    # slider body's; `edge_sounds` holds one value per slider head, repeat and tail.
    hitsound: int = 0
    edge_sounds: list | None = None


@dataclass
class Beatmap:
    metadata: dict = field(default_factory=dict)
    difficulty: dict = field(default_factory=dict)
    audio: str = ""
    mode: int = 0
    timing: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    skipped: int = 0
    ignored: list = field(default_factory=list)
    spinner_sounds: list = field(default_factory=list)  # additions per entry of `ignored`
    # Every timing point's (time, volume, kiai), red and green, in file order by time.
    sound_points: list = field(default_factory=list)
    breaks: list = field(default_factory=list)
    preview_time: int = -1

    def timing_at(self, time):
        beat, velocity = 500.0, 1.0
        for point in self.timing:
            if point.time > time:
                break
            if not point.inherited:
                beat, velocity = point.beat_length, 1.0
            else:
                velocity = min(10.0, max(0.1, -100 / point.beat_length))
        return beat, velocity

    def info(self):
        length = max((o.time + o.duration for o in self.objects), default=0) / 1000
        first = min((o.time for o in self.objects), default=0) / 1000
        return {
            "title": self.metadata.get("Title", "Untitled"),
            "artist": self.metadata.get("Artist", "Unknown"),
            "creator": self.metadata.get("Creator", "Unknown"),
            "version": self.metadata.get("Version", "Normal"),
            "mode": self.mode,
            "objects": len(self.objects),
            "duration": round(length, 2),
            "density": round(len(self.objects) / max(length - first, 1), 2),
            "difficulty": self.difficulty,
            "skipped_objects": self.skipped,
        }


def parse(path: Path) -> Beatmap:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Beatmap text exceeds 16 MiB")
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if not text.lstrip().startswith("osu file format v") or "\x00" in text:
        raise ValueError(f"Invalid or damaged .osu file: {path.name}")
    result, section, lines = Beatmap(), "", []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif section == "Events":
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3 and parts[0] in ("2", "Break"):
                start, end = float(parts[1]), float(parts[2])
                if math.isfinite(start + end) and end > start:
                    result.breaks.append((start, end))
        elif section in ("Metadata", "Difficulty", "General") and ":" in line:
            key, value = (part.strip() for part in line.split(":", 1))
            if section == "Metadata":
                result.metadata[key] = value
            elif section == "Difficulty":
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("Non-finite difficulty")
                result.difficulty[key] = number
            elif key == "AudioFilename":
                result.audio = value.strip('"')
            elif key == "Mode":
                result.mode = int(value)
            elif key == "PreviewTime":
                result.preview_time = int(float(value))
        elif section == "TimingPoints":
            parts = line.split(",")
            time, beat = float(parts[0]), float(parts[1])
            if not math.isfinite(time + beat) or beat == 0:
                raise ValueError("Invalid timing point")
            inherited = parts[6] == "0" if len(parts) > 6 else beat < 0
            if inherited != (beat < 0):
                raise ValueError("Timing point sign disagrees with inherited flag")
            meter = int(parts[2]) if len(parts) > 2 and parts[2] else None
            known_meter = meter is not None and not inherited
            volume = int(parts[5]) if len(parts) > 5 and parts[5].strip() else 100
            kiai = bool(int(parts[7]) & 1) if len(parts) > 7 and parts[7].strip() else False
            result.sound_points.append((time, max(0, min(100, volume)), kiai))
            result.timing.append(
                TimingPoint(
                    time,
                    beat,
                    inherited,
                    meter,
                    1.0,
                    1.0 if known_meter else 0.0,
                    time if known_meter else None,
                )
            )
        elif section == "HitObjects":
            lines.append(line)

    result.timing.sort(key=lambda point: point.time)
    result.sound_points.sort(key=lambda point: point[0])
    for line in lines:
        parts = line.split(",")
        x, y, time = map(float, parts[:3])
        kind = int(parts[3])
        if not math.isfinite(x + y + time):
            raise ValueError("Invalid hit object")
        sound = int(parts[4]) if len(parts) > 4 and parts[4].strip() else 0
        if not (kind & 1 or kind & 2):
            result.skipped += 1
            if kind & 8:
                end = float(parts[5])
                if not math.isfinite(end) or end < time:
                    raise ValueError("Invalid spinner end time")
                result.ignored.append((time, end))
                result.spinner_sounds.append(sound)
            continue
        obj = HitObject(x, y, time, end_x=x, end_y=y, new_combo=bool(kind & 4), hitsound=sound)
        obj.beat_length = result.timing_at(time)[0]
        if kind & 2:
            repeats, pixels = int(parts[6]), float(parts[7])
            if not math.isfinite(pixels) or not 0 <= pixels <= 100000 or not 1 <= repeats <= 1000:
                raise ValueError("Invalid slider length/repeats")
            beat, velocity = result.timing_at(time)
            multiplier = max(0.1, result.difficulty.get("SliderMultiplier", 1.4))
            obj.slider = True
            obj.duration = pixels * repeats * beat / (100 * multiplier * velocity)
            path = slider_path(x, y, parts[5])
            tail = endpoint(x, y, parts[5], pixels, path=path)
            if repeats % 2:
                obj.end_x, obj.end_y = tail
            obj.curve, error = fitted_curve(x, y, parts[5], pixels, path=path)
            obj.repeats, obj.path_length = repeats, pixels
            if len(parts) > 8 and parts[8].strip():
                edges = [int(value) if value.strip() else 0 for value in parts[8].split("|")]
                obj.edge_sounds = (edges + [0] * (repeats + 1))[: repeats + 1]
            obj.geometry_valid = error <= 8 and repeats <= 4 and not parts[5].startswith("C")
            if not math.isfinite(obj.end_x + obj.end_y + obj.duration):
                raise ValueError("Invalid slider geometry")
        result.objects.append(obj)
    result.objects.sort(key=lambda obj: obj.time)
    return result


def clean(value):
    return "".join(char if char.isprintable() else " " for char in str(value))[:180]


DEFAULT_VOLUME = 70


def edge_field(obj):
    """`edgeSounds` and `edgeSets` of a slider: one entry per head, repeat and tail."""
    edges = list(obj.edge_sounds or [])
    edges = (edges + [0] * (obj.repeats + 1))[: obj.repeats + 1]
    return "|".join(str(int(value)) for value in edges), "|".join(["0:0"] * len(edges))


def write(
    path, objects, settings, audio_name, bpm, offset, timing=None, spinners=(), finish=None
):
    """Write circles, sliders and optional spinners, given as (start, end) in ms.

    `finish` (see `ml.finishing.Finish.export`) adds the editor layers: preview point,
    break periods, per-section volume and kiai, and spinner hitsounds. Object hitsounds
    come from `HitObject.hitsound` / `edge_sounds`.
    """
    finish = finish or {}
    beat = 60000 / bpm
    # Keep the same grid phase while establishing timing before the first object.
    first = min((obj.time for obj in objects), default=0)
    offset -= math.ceil(max(0, offset - first) / beat) * beat
    preview = int(finish.get("preview_time", -1))
    breaks = [
        f"2,{round(start)},{round(end)}"
        for start, end in sorted(finish.get("breaks") or ())
        if round(end) > round(start)
    ]
    lines = [
        "osu file format v14",
        "",
        "[General]",
        f"AudioFilename: {audio_name}",
        "AudioLeadIn: 0",
        f"PreviewTime: {preview if preview >= 0 else -1}",
        "Countdown: 0",
        "SampleSet: Soft",
        "StackLeniency: 0.7",
        "Mode: 0",
        "",
        "[Metadata]",
        f"Title:{clean(settings['title'])}",
        f"Artist:{clean(settings['artist'])}",
        "Creator:BobTheBeatbuilder",
        f"Version:{clean(settings['version'])}",
        "Tags:AI assisted generated bobthebeatbuilder",
        "BeatmapID:0",
        "BeatmapSetID:-1",
        "",
        "[Difficulty]",
        f"HPDrainRate:{settings['hp']}",
        f"CircleSize:{settings['cs']}",
        f"OverallDifficulty:{settings['od']}",
        f"ApproachRate:{settings['ar']}",
        "SliderMultiplier:1.4",
        "SliderTickRate:1",
        "",
        "[Events]",
        "//Break Periods",
        *breaks,
        "",
        "[TimingPoints]",
        *timing_lines(
            objects, timing or [TimingPoint(offset, beat)], finish.get("sound_states") or ()
        ),
        "",
        "[Colours]",
        "Combo1:255,108,176",
        "Combo2:123,196,255",
        "Combo3:176,155,255",
        "",
        "[HitObjects]",
    ]
    ordered = sorted(spinners)
    sounds = list(finish.get("spinner_sounds") or ())
    spinner_lines = [
        (
            round(start),
            f"256,192,{round(start)},{8 | 4},{int(sounds[index]) if index < len(sounds) else 0},"
            f"{round(end)},0:0:0:0:",
        )
        for index, (start, end) in enumerate(ordered)
        if round(end) > round(start)
    ]
    after_spinner = {
        next((index for index, obj in enumerate(objects) if obj.time > start), None)
        for start, _ in spinner_lines
    }
    for index, obj in enumerate(objects):
        while spinner_lines and spinner_lines[0][0] < round(obj.time):
            lines.append(spinner_lines.pop(0)[1])
        x, y, time = round(obj.x), round(obj.y), round(obj.time)
        # The object after a spinner always starts a new combo in osu!.
        combo = 4 if obj.new_combo or index == 0 or index in after_spinner else 0
        sound = int(obj.hitsound or 0)
        if obj.slider and obj.duration > 0:
            edges, sets = edge_field(obj)
            if obj.curve:
                controls = "|".join(f"{round(x)}:{round(y)}" for x, y in obj.curve[1:])
                pixels = curve_length([[round(x), round(y)] for x, y in obj.curve])
                lines.append(
                    f"{x},{y},{time},{2 | combo},{sound},B|{controls},{obj.repeats},{pixels:.4f},{edges},{sets},0:0:0:0:"
                )
                continue
            pixels = obj.duration / beat * 140
            edges, sets = edge_field(HitObject(0, 0, 0, edge_sounds=obj.edge_sounds))
            lines.append(
                f"{x},{y},{time},{2 | combo},{sound},L|{round(obj.end_x)}:{round(obj.end_y)},1,{pixels:.4f},{edges},{sets},0:0:0:0:"
            )
        else:
            lines.append(f"{x},{y},{time},{1 | combo},{sound},0:0:0:0:")
    lines.extend(line for _, line in spinner_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def first(item):
    return item[0]


def state_at(states, time):
    """(volume, kiai) of the sound state active at `time`."""
    volume, kiai = (states[0][1], False) if states else (DEFAULT_VOLUME, False)
    for start, state_volume, state_kiai in states:
        if start > time + 1e-6:
            break
        volume, kiai = state_volume, state_kiai
    return int(volume), bool(kiai)


def timing_lines(objects, timing, states=()):
    """Red lines, slider velocity lines and volume/kiai lines, one per kind and time.

    osu! allows one red and one green line at the same time. A green line carries the
    slider velocity of a slider starting there plus the active volume and kiai; a state
    change without a slider gets a neutral green line (sliders always set their own
    velocity at their head). The first red line never toggles kiai (Ranking Criteria):
    a kiai starting there is set by a green line at the same time.
    """
    timing = [replace(point, time=round(point.time)) for point in red_points(timing)]
    if not timing:
        raise ValueError("Export requires beat timing")
    states = sorted(((round(time), volume, kiai) for time, volume, kiai in states), key=first)
    first_red = round(timing[0].time)
    # Nothing may precede the first red line: earlier states become its initial state.
    states = [(max(time, first_red), volume, kiai) for time, volume, kiai in states]
    # osu! requires a positive meter. Four is only an export default when unknown;
    # it must not be fed back as evidence for generation or timing features.
    entries = []
    for index, point in enumerate(timing):
        time = round(point.time)
        volume, kiai = state_at(states, time)
        flag = int(kiai and index > 0)
        line = f"{time:.3f},{point.beat_length:.8f},{point.meter or 4},2,0,{volume},1,{flag}"
        entries.append((time, 0, line))
    greens = {}
    for obj in objects:
        if not obj.slider or not obj.curve or obj.duration <= 0:
            continue
        length = curve_length([[round(x), round(y)] for x, y in obj.curve])
        point = timing_at(timing, round(obj.time))
        beat = point.beat_length
        velocity = length * obj.repeats * beat / (140 * obj.duration)
        if not 0.1 <= velocity <= 10:
            raise ValueError("Generated slider velocity is outside osu! limits")
        greens[round(obj.time)] = (velocity, point.meter or 4)
    reds = {round(point.time) for point in timing}
    previous = None
    for time, volume, kiai in states:
        state = (volume, bool(kiai))
        # A red line already carries the state, except kiai on the very first red line.
        needs_green = time not in reds or (time == first_red and kiai)
        if state != previous and needs_green and time not in greens:
            greens[time] = (None, timing_at(timing, time).meter or 4)
        previous = state
    for time, (velocity, meter) in greens.items():
        volume, kiai = state_at(states, time)
        value = -100.0 if velocity is None else -100 / velocity
        entries.append((time, 1, f"{time},{value:.8f},{meter},2,0,{volume},0,{int(kiai)}"))
    return [line for _, _, line in sorted(entries, key=lambda entry: entry[:2])]
