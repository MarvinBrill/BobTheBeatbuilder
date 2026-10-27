from typing import Literal, Optional
from pydantic import BaseModel, Field

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
    dataset_id: str
    name: str = Field(default="My first mapper", min_length=1, max_length=100)
    epochs: int = Field(default=100, ge=1, le=10000)
    batch_size: int = Field(default=2, ge=1, le=64)
    learning_rate: float = Field(default=0.0003, ge=0.000001, le=0.01)
    seed: int = Field(default=42, ge=0, le=2147483647)
    device: Device = "cpu"


class GenerateRequest(BaseModel):
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
    divisor: Literal[1, 2, 3, 4, 6, 8] = 4
    device: Device = "cpu"
