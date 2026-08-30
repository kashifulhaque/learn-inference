"""Lab 07 — grouped-query attention."""

import math

import torch


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    """Expand KV heads so each query head has one.

    Args:
        x: (batch, kv_heads, seq, head_dim)
        repeats: How many query heads share each KV head.

    Returns:
        (batch, kv_heads * repeats, seq, head_dim), with head h of the output
        reading KV head h // repeats.
    """
    # TODO
    raise NotImplementedError


def causal_mask(q_len: int, kv_len: int, device) -> torch.Tensor:
    """Return a boolean mask of shape (q_len, kv_len), True where a score is masked.

    Query row i is at absolute position i + (kv_len - q_len), so it may attend
    to key positions up to and including that.
    """
    # TODO
    raise NotImplementedError


def attention(q, k, v, causal: bool = True, scale: float | None = None):
    """Softmax attention over grouped heads.

    Args:
        q: (batch, heads, q_len, head_dim)
        k, v: (batch, kv_heads, kv_len, head_dim); kv_heads divides heads.

    Compute in float32 and return the value dtype.
    """
    # TODO
    raise NotImplementedError


def apply_output_gate(attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Scale the attention output by a sigmoid gate, elementwise."""
    # TODO
    raise NotImplementedError


def kv_bytes_per_token(num_kv_heads: int, head_dim: int, num_full_layers: int,
                       bytes_per_element: int = 2) -> int:
    """Bytes the KV cache grows by per token across the full-attention layers."""
    # TODO
    raise NotImplementedError
