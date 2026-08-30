"""Rotary position embeddings, in the partial and multimodal forms Qwen3.5 uses.

Ordinary RoPE rotates every pair of channels in a head by an angle that grows
linearly with position. Two things make this model's variant different:

*Partial rotary.* `partial_rotary_factor` is 0.25, so only the first quarter of
each 256-wide head carries position information; the remaining 192 channels are
passed through untouched and are free to encode content. You still have to
allocate and copy them, so the cache cost is unchanged.

*mRoPE.* The rotary channels are split into sections, here [11, 11, 10] pairs,
and each section reads its angle from a different coordinate of the position:
time, height, width. For text-only input all three coordinates are the same
number and mRoPE collapses to plain RoPE, which is why the text path can ignore
the distinction until images arrive.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def build_rope_cache(
    rotary_dim: int,
    max_position: int,
    theta: float = 10_000.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Precompute cos and sin tables of shape (max_position, rotary_dim // 2).

    Building these once and indexing them is the whole trick: the angles depend
    only on position, never on the activations, so they are the same for every
    layer and every request.
    """
    half = rotary_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    positions = torch.arange(max_position, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    """Map (x1, x2) -> (-x2, x1) over the last dimension, split in halves."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_partial(
    x: Tensor, cos: Tensor, sin: Tensor, rotary_dim: int
) -> Tensor:
    """Rotate the first `rotary_dim` channels of x and leave the rest alone.

    Args:
        x: (batch, heads, seq, head_dim)
        cos, sin: (seq, rotary_dim // 2), already gathered for these positions.
        rotary_dim: How many leading channels to rotate.
    """
    rot, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    # (seq, rotary_dim//2) -> (1, 1, seq, rotary_dim), duplicated for both halves
    cos_full = torch.cat((cos, cos), dim=-1)[None, None, :, :].to(x.dtype)
    sin_full = torch.cat((sin, sin), dim=-1)[None, None, :, :].to(x.dtype)
    rotated = rot * cos_full + _rotate_half(rot) * sin_full
    if passthrough.numel() == 0:
        return rotated
    return torch.cat((rotated, passthrough), dim=-1)


def apply_mrope_sections(
    cos: Tensor, sin: Tensor, sections: list[int]
) -> tuple[Tensor, Tensor]:
    """Select each section's angles from its own position coordinate.

    Args:
        cos, sin: (3, seq, rotary_dim // 2) — one plane per coordinate.
        sections: Pair counts per coordinate, e.g. [11, 11, 10].
    """
    if cos.dim() != 3 or cos.shape[0] != len(sections):
        raise ValueError(
            f"Expected cos of shape ({len(sections)}, seq, half), got {tuple(cos.shape)}"
        )
    cos_parts, sin_parts, start = [], [], 0
    for axis, width in enumerate(sections):
        cos_parts.append(cos[axis, :, start : start + width])
        sin_parts.append(sin[axis, :, start : start + width])
        start += width
    return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)


class RotaryEmbedding(nn.Module):
    """Holds the cos/sin tables and grows them on demand."""

    def __init__(
        self, rotary_dim: int, max_position: int, theta: float = 10_000.0
    ) -> None:
        super().__init__()
        self.rotary_dim = rotary_dim
        self.theta = theta
        self.max_position = max_position
        cos, sin = build_rope_cache(rotary_dim, max_position, theta)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _ensure(self, needed: int, device: torch.device) -> None:
        if needed <= self.cos_cached.shape[0] and self.cos_cached.device == device:
            return
        size = max(needed, self.cos_cached.shape[0] * 2)
        cos, sin = build_rope_cache(self.rotary_dim, size, self.theta, device=device)
        self.cos_cached, self.sin_cached = cos, sin
        self.max_position = size

    def forward(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Gather the angles for a 1-D tensor of absolute positions."""
        self._ensure(int(positions.max().item()) + 1, positions.device)
        return self.cos_cached[positions], self.sin_cached[positions]
