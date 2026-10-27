import math
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    """CUDA uses PyTorch's fused attention; DirectML keeps the explicit equivalent."""

    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.depth = heads, width // heads
        self.query, self.key, self.value = (nn.Linear(width, width) for _ in range(3))
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.cuda_sdpa = True

    def forward(self, query, memory, valid=None, causal=False):
        batch, length, width = query.shape

        def split(value):
            return value.reshape(batch, -1, self.heads, self.depth).transpose(1, 2)

        q, k, v = split(self.query(query)), split(self.key(memory)), split(self.value(memory))
        if query.is_cuda and self.cuda_sdpa:
            bias = (1 - valid[:, None, None, :]) * -10000 if valid is not None else None
            if causal:
                indices = torch.arange(length, device=query.device)
                future = (indices[None, :] > indices[:, None]).to(q.dtype) * -10000
                bias = future[None, None] if bias is None else bias + future[None, None]
            attended = (
                F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=bias,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                .transpose(1, 2)
                .reshape(batch, length, width)
            )
            return self.output(attended)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.depth)
        if valid is not None:
            scores = scores + (1 - valid[:, None, None, :]) * -10000
        if causal:
            indices = torch.arange(length, device=query.device, dtype=query.dtype)
            future = (indices[None, :] > indices[:, None]).to(query.dtype)
            scores = scores + future[None, None] * -10000
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.matmul(self.dropout(probabilities), v).transpose(1, 2).reshape(batch, length, width)
        return self.output(attended)


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.width
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = Attention(width, config.heads, config.dropout)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(width * 4, width)
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, value, valid=None):
        normal = self.norm1(value)
        value = value + self.dropout(self.attention(normal, normal, valid))
        return value + self.dropout(self.ff(self.norm2(value)))


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.width
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.self_attention = Attention(width, config.heads, config.dropout)
        self.cross_attention = Attention(width, config.heads, config.dropout)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(width * 4, width)
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, value, memory, event_valid, memory_valid):
        normal = self.norms[0](value)
        value = value + self.dropout(self.self_attention(normal, normal, event_valid, causal=True))
        value = value + self.dropout(self.cross_attention(self.norms[1](value), memory, memory_valid))
        return value + self.dropout(self.ff(self.norms[2](value)))
