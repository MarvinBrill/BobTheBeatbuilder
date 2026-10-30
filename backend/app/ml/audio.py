import hashlib
import subprocess
from uuid import uuid4
from pathlib import Path
import numpy as np
from ..config import DATA

SAMPLE_RATE = 22050
HOP = 256
MELS = 64
FRAME_MS = HOP / SAMPLE_RATE * 1000
MAX_SECONDS = 20 * 60


def waveform(path: Path):
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        str(path),
        "-t",
        str(MAX_SECONDS + 1),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=180)
    except FileNotFoundError as error:
        raise ValueError(
            "FFmpeg is missing. Install dependencies with the provided startup script."
        ) from error
    if result.returncode:
        raise ValueError("Audio decode failed: " + result.stderr.decode(errors="replace")[-500:])
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if len(samples) < SAMPLE_RATE or len(samples) > SAMPLE_RATE * MAX_SECONDS:
        raise ValueError("Audio must be between 1 second and 20 minutes long")
    if not np.isfinite(samples).all():
        raise ValueError("Audio contains non-finite samples")
    return np.clip(samples, -1, 1)


def features(path, key=None):
    import librosa

    if key is None:
        stat = path.stat()
        key = hashlib.sha256(f"{path}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()
    cache = DATA / "cache" / f"mel-v2-{key}.npy"
    if cache.exists():
        try:
            cached = np.load(cache, mmap_mode="r", allow_pickle=False)
            if (
                cached.ndim == 2
                and 0 < len(cached) <= MAX_SECONDS * SAMPLE_RATE // HOP + 1
                and cached.shape[1] == MELS
                and cached.dtype == np.float32
                and np.isfinite(cached).all()
                and np.all((cached >= -1) & (cached <= 1))
            ):
                return cached
            del cached
        except (ValueError, OSError):
            pass
    samples = waveform(path)
    mel = librosa.feature.melspectrogram(y=samples, sr=SAMPLE_RATE, n_fft=2048, hop_length=HOP, n_mels=MELS)
    values = (librosa.power_to_db(mel, ref=np.max, top_db=80).T + 40) / 40
    # A relative dB reference maps all-zero input to 0 dB. Preserve absolute silence.
    rms = librosa.feature.rms(y=samples, frame_length=2048, hop_length=HOP)[0]
    values[rms < 0.0001] = -1
    temporary = cache.with_name(f"{cache.stem}-{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as output:
            np.save(output, np.clip(values, -1, 1).astype(np.float32), allow_pickle=False)
        temporary.replace(cache)
    finally:
        temporary.unlink(missing_ok=True)
    return np.load(cache, mmap_mode="r", allow_pickle=False)


def tempo(path, prior_bpm=None):
    """Audio-only tempo estimate, octave-corrected towards the tempo mappers would use."""
    import librosa
    from .tempo_octave import grid_phase, octave_corrected, refine_tempo

    samples = waveform(path)
    # Same envelope beat_track computes from `y` itself (median aggregation).
    envelope = librosa.onset.onset_strength(
        y=samples, sr=SAMPLE_RATE, hop_length=HOP, aggregate=np.median
    )
    bpm, beats = librosa.beat.beat_track(
        onset_envelope=envelope, sr=SAMPLE_RATE, hop_length=HOP
    )
    value = reported = float(np.asarray(bpm).reshape(-1)[0])
    if not np.isfinite(value) or value <= 0:
        value = reported = 120.0
    else:
        # librosa's 120 BPM prior often reports half the tempo (99 instead of 199 BPM).
        value = float(octave_corrected(envelope, 1000 / FRAME_MS, value, prior_bpm))
    first = float(beats[0] * FRAME_MS) if len(beats) else 0.0
    halved = value < reported * 0.75
    # beat_track's tempo is quantised to whole frames (200 BPM -> 198.77 BPM) and drifts by
    # a beat every ~45 s; fit tempo and phase to all onsets of the song instead.
    refined, phase = refine_tempo(envelope, 1000 / FRAME_MS, value)
    if refined != value:
        value, first = refined, phase
    if halved:
        # Halved: every second tracked beat is a downbeat; keep the stronger phase.
        first = grid_phase(envelope, 1000 / FRAME_MS, 60000 / value, (first, first + 30000 / value))
    return value, first % (60000 / value)


def export_audio(source, destination):
    """Use a portable osu!/browser audio format even for AAC/M4A inputs."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(source),
            "-vn",
            "-map_metadata",
            "-1",
            "-c:a",
            "libvorbis",
            "-q:a",
            "5",
            str(destination),
        ],
        capture_output=True,
        timeout=180,
    )
    if result.returncode:
        raise ValueError("Audio export failed: " + result.stderr.decode(errors="replace")[-500:])
