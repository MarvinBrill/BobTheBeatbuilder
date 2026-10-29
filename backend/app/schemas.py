from typing import Literal, Optional
import math
from pydantic import BaseModel, Field, root_validator

Device = Literal["cpu", "cuda", "directml"]


class ScanRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1000)


class Selection(BaseModel):
    ids: list[str] = Field(min_items=1, max_items=50000)
    name: str = Field(default="My training collection", min_length=1, max_length=100)


class TrainRequest(BaseModel):
    context_robustness: bool = False
    rollout_validation: bool = True
    precision: Literal["auto", "float32", "bfloat16"] = "auto"
    cuda_graphs: bool = True
    model_size: Literal["compact", "standard", "large"] = "standard"
    gradient_accumulation: int = Field(default=4, ge=1, le=32)
    weight_decay: float = Field(default=0.01, ge=0, le=0.2)
    patience: int = Field(default=12, ge=1, le=1000)
    resume_model_id: Optional[str] = None
    # Start from another model's weights (also older models); new layers start fresh.
    init_model_id: Optional[str] = None
    mapper_min_maps: int = Field(default=10, ge=2, le=1000)
    dataset_id: str
    name: str = Field(default="My first mapper", min_length=1, max_length=100)
    epochs: int = Field(default=100, ge=1, le=10000)
    batch_size: int = Field(default=2, ge=1, le=64)
    learning_rate: float = Field(default=0.0003, ge=0.000001, le=0.01)
    seed: int = Field(default=42, ge=0, le=2147483647)
    device: Device = "cpu"


class ManualTimingPoint(BaseModel):
    time: float = Field(ge=-60000, le=1200000, allow_inf_nan=False)
    beat_length: float = Field(ge=150, le=3000, allow_inf_nan=False)
    meter: Optional[int] = Field(default=None, ge=1, le=32)
    bar_origin: Optional[float] = Field(default=None, ge=-60000, le=1200000, allow_inf_nan=False)

    @root_validator(skip_on_failure=True)
    def metric_evidence(cls, values):
        origin, meter = values["bar_origin"], values["meter"]
        if origin is not None and meter is None:
            raise ValueError("A bar origin requires an explicit meter")
        if origin is not None:
            bars = (origin - values["time"]) / (values["beat_length"] * meter)
            if not math.isclose(bars, round(bars), abs_tol=1e-7):
                raise ValueError(
                    "Bar origin must agree with the red timing point modulo whole measures"
                )
        return values


class GenerateRequest(BaseModel):
    motifs: bool = False
    timing_points: list[ManualTimingPoint] = Field(default_factory=list, max_items=256)
    trace: bool = False
    lookahead: bool = False
    spacing: float = Field(default=0.3, ge=0, le=1)
    flow: float = Field(default=0.5, ge=0, le=1)
    temperature: float = Field(default=0.8, ge=0.1, le=1.5)
    candidates: int = Field(default=12, ge=1, le=48)
    seed: int = Field(default=42, ge=0, le=2147483647)
    model_id: str
    audio_id: str
    title: str = Field(default="Untitled", min_length=1, max_length=100)
    artist: str = Field(default="Unknown", min_length=1, max_length=100)
    version: str = Field(default="AI draft", min_length=1, max_length=100)
    ar: float = Field(default=9, ge=0, le=10)
    cs: float = Field(default=4, ge=0, le=10)
    od: float = Field(default=7, ge=0, le=10)
    hp: float = Field(default=5, ge=0, le=10)
    density: float = Field(default=3, ge=0.5, le=10)
    slider_bias: float = Field(default=0.5, ge=0, le=1)
    bpm: Optional[float] = Field(default=None, ge=20, le=400)
    offset: float = Field(default=0, ge=-10000, le=60000)
    snap: bool = True
    divisor: Literal[1, 2, 3, 4, 6, 8, 12] = 4
    device: Device = "cpu"
    # Style-v2 models: target difficulty, mapper and explicit pattern values (UI units).
    # Values left out are inferred by the model from the audio and the other settings.
    target_stars: Optional[float] = Field(default=None, ge=0.5, le=12)
    aim_share: Optional[float] = Field(default=None, ge=0, le=1)
    mapper: Optional[str] = Field(default=None, max_length=100)
    style: dict[str, float] = Field(default_factory=dict)
    max_stream: int = Field(default=0, ge=0, le=256)
    rhythm_temperature: float = Field(default=0.6, ge=0.05, le=1.5)
    calibrate: bool = False
    calibration_rounds: int = Field(default=3, ge=1, le=5)
    # "patterns": the model chooses each next note on the beat grid (style-v2 models).
    # "onsets": all heads are selected from the onset map first; legacy models always do.
    rhythm_mode: Literal["patterns", "onsets"] = "patterns"

    @root_validator(skip_on_failure=True)
    def ordered_timing(cls, values):
        points = values["timing_points"]
        if any(right.time <= left.time for left, right in zip(points, points[1:])):
            raise ValueError("Manual timing points must be strictly ordered")
        if points and values["bpm"] is not None:
            raise ValueError("Supply either complete timing points or a constant BPM")
        return values
