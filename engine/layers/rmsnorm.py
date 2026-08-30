"""Root-mean-square normalisation.

    y = x / sqrt(mean(x^2) + eps) * w

LayerNorm subtracts the mean and divides by the standard deviation. RMSNorm
drops the mean subtraction, which costs nothing in quality and saves a pass
over the data. It is memory bound: the arithmetic is trivial and the time goes
entirely into reading x and writing y, which is why fusing it with the residual
add (chapter 10) pays off.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Reference RMSNorm.

    The reduction runs in float32 even when x is bfloat16. Summing thousands of
    squared bfloat16 values accumulates visible error otherwise: bfloat16 has
    8 bits of mantissa, so it stops being able to represent the difference
    between a running sum and its next addend well before the reduction ends.
    """
    dtype = x.dtype
    x32 = x.float()
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    normed = x32 * torch.rsqrt(variance + eps)
    return (normed * weight.float()).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class HeadRMSNorm(nn.Module):
    """RMSNorm applied per attention head, over the head dimension.

    Qwen3 normalises queries and keys this way before RoPE. It keeps the
    logit scale stable across heads, which matters once you train at long
    context.
    """

    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., heads, head_dim)
        return rms_norm(x, self.weight, self.eps)
