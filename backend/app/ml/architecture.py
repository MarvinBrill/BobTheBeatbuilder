from dataclasses import asdict, dataclass

ARCHITECTURE = "phrase-mapper-v5"
FORMAT_VERSION = 5
EVENT_SCHEMA = "cubic-phrase-plan-v3"
TARGET_SCHEMA = "musical-plan-targets-v3"
QUERY_FEATURES = 31
AUDIO_WINDOW = 2048
CENTRAL_FRAMES = 1024
AUDIO_MARGIN = (AUDIO_WINDOW - CENTRAL_FRAMES) // 2
GLOBAL_TOKENS = 128
EVENT_CONTEXT = 128
EVENT_TARGETS = 64
GEOMETRY_SIZE = 8  # cubic Bezier: start, two controls, end
EVENT_FEATURES = 16
MOTION_FEATURES = 8
DURATION_BEATS = (
    1 / 12,
    0.125,
    1 / 6,
    0.25,
    1 / 3,
    0.375,
    0.5,
    2 / 3,
    0.75,
    1,
    4 / 3,
    1.5,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
)


@dataclass(frozen=True)
class ModelConfig:
    width: int = 256
    heads: int = 8
    audio_layers: int = 6
    global_layers: int = 2
    decoder_layers: int = 4
    mixtures: int = 8
    dropout: float = 0.1
    enable_musical_time: bool = True
    enable_plan: bool = True
    # 1: legacy eight-value condition. 2: measured style with masks, star rating, mapper
    # embedding and a rhythm-pattern head that chooses the next note on the beat grid.
    style_version: int = 1
    mapper_slots: int = 1

    def dictionary(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values=None):
        result = cls(**(values or {}))
        if result.width not in (64, 128, 192, 256, 384) or result.heads not in (2, 4, 8):
            raise ValueError("Unsupported model width/heads")
        if result.width % result.heads or not all(
            1 <= value <= 8
            for value in (result.audio_layers, result.global_layers, result.decoder_layers)
        ):
            raise ValueError("Invalid model depth")
        if not 2 <= result.mixtures <= 16 or not 0 <= result.dropout <= 0.3:
            raise ValueError("Invalid mixture/dropout configuration")
        if type(result.enable_musical_time) is not bool or type(result.enable_plan) is not bool:
            raise ValueError("Invalid musical-time or plan ablation setting")
        if result.style_version not in (1, 2) or not 1 <= result.mapper_slots <= 4096:
            raise ValueError("Invalid style configuration")
        if result.style_version == 1 and result.mapper_slots != 1:
            raise ValueError("Legacy models have no mapper embedding")
        return result
