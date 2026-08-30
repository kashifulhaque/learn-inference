"""FlashAttention forward pass in Triton.

The naive attention path builds the full (q_len, kv_len) score matrix, writes it
to HBM, reads it back for softmax, writes it again, and reads it once more for
the value multiply. At 8k context with 24 heads that matrix is 12 GB in
bfloat16 for one sequence, and every byte of it crosses the memory bus three
times.

FlashAttention never materialises it. It walks the keys in tiles, keeps a
running max `m` and a running denominator `l` for each query row, and rescales
the accumulator whenever the max moves:

    m_new = max(m, max(s_tile))
    l_new = l * exp(m - m_new) + sum(exp(s_tile - m_new))
    acc   = acc * exp(m - m_new) + exp(s_tile - m_new) @ v_tile

That is the online softmax. The scores live in registers and shared memory and
never reach HBM, so the whole thing reads Q, K, V once and writes O once. The
arithmetic is identical to the naive version, which is why the labs check them
against each other to a tight tolerance.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    n_heads, q_len, kv_len,
    kv_group: tl.constexpr,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // n_heads
    head = bh % n_heads
    # Grouped-query attention: several query heads read one KV head, so the
    # kernel indexes the shared head instead of the caller duplicating it.
    kv_head = head // kv_group

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = (
        q_ptr + batch * stride_qb + head * stride_qh
        + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=offs_m[:, None] < q_len, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Causal masking: query row i (absolute position i + kv_len - q_len) may see
    # key j only when j <= that. Anything past the diagonal block is skipped
    # outright, which is where roughly half the work goes when q_len == kv_len.
    offset = kv_len - q_len
    hi = tl.minimum(kv_len, (start_m + 1) * BLOCK_M + offset) if IS_CAUSAL else kv_len

    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + offs_n
        k_ptrs = (
            k_ptr + batch * stride_kb + kv_head * stride_kh
            + cols[:, None] * stride_kn + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            v_ptr + batch * stride_vb + kv_head * stride_vh
            + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd
        )
        k = tl.load(k_ptrs, mask=cols[:, None] < kv_len, other=0.0)
        v = tl.load(v_ptrs, mask=cols[:, None] < kv_len, other=0.0)

        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(cols[None, :] < kv_len, scores, float("-inf"))
        if IS_CAUSAL:
            scores = tl.where(
                cols[None, :] <= offs_m[:, None] + offset, scores, float("-inf")
            )

        # --- online softmax update ---
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        correction = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        l_i = l_i * correction + tl.sum(p, axis=1)
        acc = acc * correction[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
    o_ptrs = (
        o_ptr + batch * stride_ob + head * stride_oh
        + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_m[:, None] < q_len)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    scale: float | None = None,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """FlashAttention forward.

    Args:
        q: (batch, heads, q_len, head_dim)
        k, v: (batch, kv_heads, kv_len, head_dim); kv_heads must divide heads.
    """
    batch, heads, q_len, head_dim = q.shape
    kv_heads, kv_len = k.shape[1], k.shape[2]
    if heads % kv_heads:
        raise ValueError(f"{heads} query heads do not divide into {kv_heads} KV heads")

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    scale = scale or head_dim**-0.5

    grid = (triton.cdiv(q_len, block_m), batch * heads)
    _flash_fwd[grid](
        q, k, v, out,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        heads, q_len, kv_len,
        kv_group=heads // kv_heads,
        scale=scale,
        IS_CAUSAL=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=8 if head_dim >= 128 else 4,
        num_stages=2,
    )
    return out
