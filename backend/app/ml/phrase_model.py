import torch
from torch import nn
from torch.nn import functional as F
from .architecture import (
    ModelConfig,
    EVENT_FEATURES,
    GEOMETRY_SIZE,
    MOTION_FEATURES,
    DURATION_BEATS,
)
from .attention import EncoderBlock, DecoderBlock
from .style import CONDITION_SIZE, GAP_CLASSES

LEGACY_CONDITION = 8


class PhraseMapper(nn.Module):
    """Hierarchical audio, causal object sequences, joint geometry and a compatibility critic."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg, width = self.config, self.config.width
        self.audio_front = nn.Sequential(
            nn.Conv1d(64, width, 7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2),
            nn.GELU(),
        )
        self.audio_position = nn.Linear(17, width)
        self.styled = cfg.style_version >= 2
        conditions = CONDITION_SIZE if self.styled else LEGACY_CONDITION
        self.condition = nn.Sequential(
            nn.Linear(conditions, width), nn.GELU(), nn.Linear(width, width)
        )
        if self.styled:
            # Index 0 is "no particular mapper"; it is also used for rare mappers.
            self.mapper = nn.Embedding(cfg.mapper_slots, width)
            nn.init.normal_(self.mapper.weight, std=0.02)
        self.global_input = nn.Linear(128, width)
        self.global_blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.global_layers)])
        self.audio_blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.audio_layers)])
        self.audio_norm = nn.LayerNorm(width)
        # onset, beat pulse, local log-density, log beat length
        self.rhythm_up = nn.ConvTranspose1d(width, 64, 4, stride=4)
        self.rhythm_local = nn.Conv1d(64, 64, 5, padding=2)
        self.rhythm = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=1), nn.GELU(), nn.Conv1d(64, 4, 1)
        )
        self.event_input = nn.Linear(EVENT_FEATURES, width)
        self.query_input = nn.Linear(20, width)
        self.decoder = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.decoder_layers)])
        self.event_norm = nn.LayerNorm(width)
        self.geometry = nn.Linear(width, cfg.mixtures * (1 + 2 * GEOMETRY_SIZE))
        self.kind = nn.Linear(width, 2)
        self.duration = nn.Linear(width, len(DURATION_BEATS))
        self.repeats = nn.Linear(width, 4)
        self.combo = nn.Linear(width, 1)
        self.motion = nn.Linear(width, MOTION_FEATURES)
        self.critic = nn.Sequential(
            nn.Linear(width + GEOMETRY_SIZE + MOTION_FEATURES, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        if self.styled:
            # Rhythm patterns: the beat-relative gap to the next object, given the history,
            # the audio and the object just placed (circle or slider and its length).
            self.gap = nn.Sequential(nn.Linear(width + 4, width), nn.GELU(), nn.Linear(width, GAP_CLASSES))

    def conditioning(self, condition, mapper=None):
        value = self.condition(condition)
        if self.styled:
            if mapper is None:
                mapper = torch.zeros(condition.shape[0], device=condition.device)
            slots = self.config.mapper_slots
            # Float32 keeps indices above 256 exact under BF16 autocast. A one-hot product
            # instead of an embedding lookup works on every backend, DirectML included.
            index = mapper.reshape(-1, 1).float().round().clamp(0, slots - 1)
            choice = (index == torch.arange(slots, device=value.device, dtype=torch.float32)).to(
                value.dtype
            )
            value = value + choice @ self.mapper.weight.to(value.dtype)
        return value

    def encode(self, audio, summary, condition, audio_times, global_times, audio_valid, mapper=None):
        conditioning = self.conditioning(condition, mapper).unsqueeze(1)
        global_memory = (
            self.global_input(summary) + self.audio_position(global_times) + conditioning
        )
        for block in self.global_blocks:
            global_memory = block(global_memory)
        local = (
            self.audio_front(audio).transpose(1, 2)
            + self.audio_position(audio_times)
            + conditioning
        )
        local_valid = F.max_pool1d(audio_valid.unsqueeze(1), 4, 4).squeeze(1)
        # Local detailed audio can attend to the entire song's coarse representation.
        memory = torch.cat([local, global_memory], dim=1)
        valid = torch.cat([local_valid, torch.ones_like(global_times[:, :, 0])], dim=1)
        for block in self.audio_blocks:
            memory = block(memory, valid)
        memory = self.audio_norm(memory)
        source = memory[:, : local.shape[1]]
        local_detail = self.rhythm_local(audio)
        if self.styled and self.training:
            # The onset branch overfit within a few epochs on large datasets; regularise it.
            source = F.dropout(source, 0.1, True)
            local_detail = F.dropout(local_detail, 0.1, True)
        detailed = self.rhythm_up(source.transpose(1, 2))
        rhythm = self.rhythm(torch.cat([detailed, local_detail], dim=1))
        return memory, valid, rhythm

    def decode(self, previous, queries, event_valid, memory, memory_valid):
        value = self.event_input(previous) + self.query_input(queries)
        for block in self.decoder:
            value = block(value, memory, event_valid, memory_valid)
        hidden = self.event_norm(value)
        return {
            "geometry": self.geometry(hidden),
            "kind": self.kind(hidden),
            "duration": self.duration(hidden),
            "repeats": self.repeats(hidden),
            "combo": self.combo(hidden).squeeze(-1),
            "motion": self.motion(hidden),
            "hidden": hidden,
        }

    def compatibility(self, hidden, geometry, motion):
        return self.critic(torch.cat([hidden, geometry, motion], dim=-1)).squeeze(-1)

    def gap_logits(self, hidden, kind, duration_beats, repeats):
        slider = kind.to(hidden.dtype)
        extra = torch.stack(
            [
                1 - slider,
                slider,
                torch.log2(duration_beats.to(hidden.dtype).clamp(min=0) + 1 / 16) / 4 * slider,
                repeats.to(hidden.dtype) / 4 * slider,
            ],
            dim=-1,
        )
        return self.gap(torch.cat([hidden, extra], dim=-1))

    def forward(self, batch):
        memory, valid, rhythm = self.encode(
            batch["audio"],
            batch["summary"],
            batch["condition"],
            batch["audio_times"],
            batch["global_times"],
            batch["audio_valid"],
            batch.get("mapper"),
        )
        result = self.decode(
            batch["previous"], batch["queries"], batch["event_valid"], memory, valid
        )
        result["rhythm"] = rhythm
        if self.styled:
            result["gap"] = self.gap_logits(
                result["hidden"],
                batch["kind_target"],
                batch["seconds_target"] / batch["beat_seconds"].clamp(min=1e-3),
                batch["repeat_target"] + 1,
            )
        if self.training and "context_probability" in batch:
            from .context_training import predicted_history

            previous, affected = predicted_history(batch, result["geometry"], self.config.mixtures)
            # This auxiliary path trains the object decoder without adding an audio-encoder loss.
            robust = self.decode(
                previous, batch["queries"], batch["event_valid"], memory.detach(), valid
            )
            result.update({f"robust_{key}": value for key, value in robust.items()})
            result["robust_mask"] = affected
        return result


def create_model(architecture, config=None):
    from .architecture import ARCHITECTURE

    if architecture != ARCHITECTURE:
        raise ValueError(
            "This model uses an obsolete architecture. Train a new Phrase Mapper model."
        )
    return PhraseMapper(ModelConfig.from_dict(config))
