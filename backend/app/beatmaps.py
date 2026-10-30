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
    for line in lines:
        parts = line.split(",")
        x, y, time = map(float, parts[:3])
        kind = int(parts[3])
        if not math.isfinite(x + y + time):
            raise ValueError("Invalid hit object")
        if not (kind & 1 or kind & 2):
            result.skipped += 1
            if kind & 8:
                end = float(parts[5])
                if not math.isfinite(end) or end < time:
                    raise ValueError("Invalid spinner end time")
                result.ignored.append((time, end))
            continue
        obj = HitObject(x, y, time, end_x=x, end_y=y, new_combo=bool(kind & 4))
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
            obj.geometry_valid = error <= 8 and repeats <= 4 and not parts[5].startswith("C")
            if not math.isfinite(obj.end_x + obj.end_y + obj.duration):
                raise ValueError("Invalid slider geometry")
        result.objects.append(obj)
    result.objects.sort(key=lambda obj: obj.time)
    return result


def clean(value):
    return "".join(char if char.isprintable() else " " for char in str(value))[:180]


def write(path, objects, settings, audio_name, bpm, offset, timing=None, spinners=()):
    """Write circles, sliders and optional spinners, given as (start, end) in ms."""
    beat = 60000 / bpm
    # Keep the same grid phase while establishing timing before the first object.
    first = min((obj.time for obj in objects), default=0)
    offset -= math.ceil(max(0, offset - first) / beat) * beat
    lines = [
        "osu file format v14",
        "",
        "[General]",
        f"AudioFilename: {audio_name}",
        "AudioLeadIn: 0",
        "PreviewTime: -1",
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
        "Tags:AI assisted bobthebeatbuilder",
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
        "",
        "[TimingPoints]",
        *timing_lines(objects, timing or [TimingPoint(offset, beat)]),
        "",
        "[Colours]",
        "Combo1:255,108,176",
        "Combo2:123,196,255",
        "Combo3:176,155,255",
        "",
        "[HitObjects]",
    ]
    spinner_lines = [
        (round(start), f"256,192,{round(start)},{8 | 4},0,{round(end)},0:0:0:0:")
        for start, end in sorted(spinners)
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
        if obj.slider and obj.duration > 0:
            if obj.curve:
                controls = "|".join(f"{round(x)}:{round(y)}" for x, y in obj.curve[1:])
                pixels = curve_length([[round(x), round(y)] for x, y in obj.curve])
                lines.append(
                    f"{x},{y},{time},{2 | combo},0,B|{controls},{obj.repeats},{pixels:.4f},0|0,0:0|0:0,0:0:0:0:"
                )
                continue
            pixels = obj.duration / beat * 140
            lines.append(
                f"{x},{y},{time},{2 | combo},0,L|{round(obj.end_x)}:{round(obj.end_y)},1,{pixels:.4f},0|0,0:0|0:0,0:0:0:0:"
            )
        else:
            lines.append(f"{x},{y},{time},{1 | combo},0,0:0:0:0:")
    lines.extend(line for _, line in spinner_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def timing_lines(objects, timing):
    timing = [replace(point, time=round(point.time)) for point in red_points(timing)]
    if not timing:
        raise ValueError("Export requires beat timing")
    # osu! requires a positive meter. Four is only an export default when unknown;
    # it must not be fed back as evidence for generation or timing features.
    entries = [
        (
            round(point.time),
            0,
            f"{round(point.time):.3f},{point.beat_length:.8f},{point.meter or 4},2,0,70,1,0",
        )
        for point in timing
    ]
    for obj in objects:
        if not obj.slider or not obj.curve or obj.duration <= 0:
            continue
        length = curve_length([[round(x), round(y)] for x, y in obj.curve])
        point = timing_at(timing, round(obj.time))
        beat = point.beat_length
        velocity = length * obj.repeats * beat / (140 * obj.duration)
        if not 0.1 <= velocity <= 10:
            raise ValueError("Generated slider velocity is outside osu! limits")
        entries.append(
            (
                round(obj.time),
                1,
                f"{round(obj.time)},{-100 / velocity:.8f},{point.meter or 4},2,0,70,0,0",
            )
        )
    return [line for _, _, line in sorted(entries, key=lambda entry: entry[:2])]
