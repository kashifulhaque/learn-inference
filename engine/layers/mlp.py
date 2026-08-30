"""The gated feed-forward block.

    SwiGLU(x) = down( silu(gate(x)) * up(x) )

Two projections up, one down, so the block holds 3 * hidden * intermediate
parameters — for this model 3 * 5120 * 17408, about 267M per layer, which is
where most of the 27B lives. Fusing silu with the elementwise multiply saves a
full read and write of a (tokens, 17408) tensor; see chapter 10.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def swiglu_reference(
    x: Tensor, gate_w: Tensor, up_w: Tensor, down_w: Tensor
) -> Tensor:
    """Functional form, for labs that build the block from raw weight tensors."""
    gate = F.silu(x @ gate_w.T)
    up = x @ up_w.T
    return (gate * up) @ down_w.T
