"""Grouped-query attention with an output gate — the full-attention layers.

Only 16 of this model's 64 layers land here; the other 48 use linear attention.
These are the layers that own a KV cache, so they are the ones that make long
context expensive.

Three things to notice:

*Grouped queries.* 24 query heads share 4 KV heads, so each KV head serves 6
queries. The cache shrinks by 6x against multi-head attention, and since decode
is bound by how fast you can read the cache, that is close to a 6x speedup.

*Query and key normalisation.* Qwen3 applies RMSNorm per head to q and k before
the rotation.

*The output gate.* The query projection emits twice the width it needs; the
second half becomes a sigmoid gate on the attention output. It costs one extra
hidden x heads*head_dim matrix but lets a head suppress its own contribution.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rmsnorm import HeadRMSNorm
from .rope import apply_rotary_partial


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: float | None = None,
    causal: bool = True,
) -> Tensor:
    """The textbook formula, materialising the full score matrix.

    Args:
        q: (batch, heads, q_len, head_dim)
        k, v: (batch, heads, kv_len, head_dim)

    This allocates a (batch, heads, q_len, kv_len) tensor. At 8k context with 24
    heads that is 12 GB in bfloat16 for a single sequence, which is the entire
    reason FlashAttention exists. Use it to check the kernels you write in
    chapter 11; do not use it to serve.
    """
    scale = scale or 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if causal:
        q_len, kv_len = scores.shape[-2], scores.shape[-1]
        # Row i of the query block attends to key positions up to
        # i + (kv_len - q_len): the offset accounts for cached prefix tokens.
        offset = kv_len - q_len
        idx_q = torch.arange(q_len, device=scores.device).unsqueeze(-1)
        idx_k = torch.arange(kv_len, device=scores.device).unsqueeze(0)
        scores = scores.masked_fill(idx_k > idx_q + offset, float("-inf"))
    weights = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.matmul(weights, v)


def repeat_kv(x: Tensor, repeats: int) -> Tensor:
    """Expand KV heads to match query heads.

    This is the naive way to do GQA and it materialises `repeats` copies of the
    cache. A real kernel indexes the shared KV head instead; chapter 12's paged
    attention does exactly that.
    """
    if repeats == 1:
        return x
    batch, heads, seq, dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, heads, repeats, seq, dim)
        .reshape(batch, heads * repeats, seq, dim)
    )


class GatedGroupedQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_dim: int,
        output_gate: bool = True,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.output_gate = output_gate
        self.group_size = num_heads // num_kv_heads
        self.scale = head_dim**-0.5

        q_out = num_heads * head_dim * (2 if output_gate else 1)
        self.q_proj = nn.Linear(hidden_size, q_out, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = HeadRMSNorm(head_dim, eps) if qk_norm else None
        self.k_norm = HeadRMSNorm(head_dim, eps) if qk_norm else None

    def project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Split the projections into q, k, v and the optional gate."""
        batch, seq, _ = x.shape
        q = self.q_proj(x)
        gate = None
        if self.output_gate:
            q, gate = q.chunk(2, dim=-1)

        q = q.view(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # (batch, heads, seq, head_dim)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), gate

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        kv_cache=None,
        layer_idx: int = 0,
        attn_impl: str = "sdpa",
    ) -> Tensor:
        batch, seq, _ = x.shape
        q, k, v, gate = self.project(x)

        q = apply_rotary_partial(q, cos, sin, self.rotary_dim)
        k = apply_rotary_partial(k, cos, sin, self.rotary_dim)

        if kv_cache is not None:
            k, v = kv_cache.append(layer_idx, k, v)

        if attn_impl == "sdpa":
            # PyTorch dispatches this to a fused kernel and never builds the
            # full score matrix.
            out = F.scaled_dot_product_attention(
                q,
                repeat_kv(k, self.group_size),
                repeat_kv(v, self.group_size),
                is_causal=k.shape[-2] > 1 and seq > 1,
                scale=self.scale,
            )
        elif attn_impl == "naive":
            out = scaled_dot_product_attention(
                q,
                repeat_kv(k, self.group_size),
                repeat_kv(v, self.group_size),
                scale=self.scale,
                causal=seq > 1,
            )
        else:
            raise ValueError(f"Unknown attention implementation: {attn_impl}")

        out = out.transpose(1, 2).reshape(batch, seq, self.num_heads * self.head_dim)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return self.o_proj(out)
