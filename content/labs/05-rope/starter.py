"""Lab 05 — rotary position embeddings."""

import torch


def build_rope_cache(rotary_dim: int, max_position: int, theta: float = 10_000.0,
                     device="cpu", dtype=torch.float32):
    """Return cos and sin tables of shape (max_position, rotary_dim // 2).

    Frequency j uses the angle position * theta ** (-2j / rotary_dim), for j
    from 0 to rotary_dim // 2 - 1.
    """
    # TODO
    raise NotImplementedError


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map (x1, x2) to (-x2, x1), splitting the last dimension in halves."""
    # TODO
    raise NotImplementedError


def apply_rotary_partial(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                         rotary_dim: int) -> torch.Tensor:
    """Rotate the first `rotary_dim` channels and pass the rest through.

    Args:
        x: (batch, heads, seq, head_dim)
        cos, sin: (seq, rotary_dim // 2), already gathered for these positions.
        rotary_dim: How many leading channels to rotate.
    """
    # TODO
    raise NotImplementedError
