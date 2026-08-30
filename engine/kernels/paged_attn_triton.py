"""Paged attention for decode.

Decode attends one query token against the whole cached prefix. There is no
score matrix worth tiling and no reuse of Q, so the kernel is bound entirely by
how fast it can read K and V. The paged part means those live in scattered
fixed-size blocks rather than one contiguous range, so the kernel reads a block
table and follows it.

One program handles one (sequence, head) pair and streams the blocks with the
same online-softmax running max and denominator as FlashAttention.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_fwd(
    q_ptr, k_cache_ptr, v_cache_ptr, out_ptr, block_table_ptr, context_len_ptr,
    stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_tb, stride_ts,
    stride_os, stride_oh, stride_od,
    n_heads,
    scale,
    kv_group: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // kv_group

    context_len = tl.load(context_len_ptr + seq)

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + seq * stride_qs + head * stride_qh + offs_d * stride_qd)
    q = q.to(tl.float32) * scale

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    offs_s = tl.arange(0, BLOCK_SIZE)
    num_blocks = tl.cdiv(context_len, BLOCK_SIZE)

    for b in range(0, MAX_BLOCKS):
        if b < num_blocks:
            physical = tl.load(block_table_ptr + seq * stride_tb + b * stride_ts)
            token_pos = b * BLOCK_SIZE + offs_s
            valid = token_pos < context_len

            k_ptrs = (
                k_cache_ptr + physical * stride_kb
                + offs_s[:, None] * stride_ks + kv_head * stride_kh
                + offs_d[None, :] * stride_kd
            )
            k = tl.load(k_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
            v = tl.load(
                v_cache_ptr + physical * stride_kb
                + offs_s[:, None] * stride_ks + kv_head * stride_kh
                + offs_d[None, :] * stride_kd,
                mask=valid[:, None], other=0.0,
            ).to(tl.float32)

            scores = tl.sum(k * q[None, :], axis=1)
            scores = tl.where(valid, scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            correction = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            p = tl.where(valid, p, 0.0)

            l_i = l_i * correction + tl.sum(p, axis=0)
            acc = acc * correction + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

    acc = acc / tl.where(l_i == 0.0, 1.0, l_i)
    tl.store(out_ptr + seq * stride_os + head * stride_oh + offs_d * stride_od,
             acc.to(out_ptr.dtype.element_ty))


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """One decode step for a batch of sequences.

    Args:
        q: (num_seqs, heads, head_dim)
        k_cache, v_cache: (num_blocks, block_size, kv_heads, head_dim)
        block_table: (num_seqs, max_blocks) of physical block ids.
        context_lens: (num_seqs,) tokens cached per sequence.
    """
    num_seqs, heads, head_dim = q.shape
    _, block_size, kv_heads, _ = k_cache.shape
    max_blocks = block_table.shape[1]

    out = torch.empty_like(q)
    scale = scale or head_dim**-0.5

    _paged_decode_fwd[(num_seqs, heads)](
        q, k_cache, v_cache, out, block_table, context_lens,
        *q.stride(), *k_cache.stride(), *block_table.stride(), *out.stride(),
        heads,
        scale,
        kv_group=heads // kv_heads,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        MAX_BLOCKS=max_blocks,
        num_warps=4,
    )
    return out
