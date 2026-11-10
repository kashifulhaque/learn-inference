"""Reference solution for lab 14, matching engine/kernels/flash_attn_triton.py.

The launch-grid helpers at the bottom run on any machine. The Triton kernel is
defined only when Triton is importable, so loading this file on a CPU-only
machine does not fail.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # CPU-only machine: the grid helpers still work
    HAS_TRITON = False

SHARED_MEMORY_BYTES = 160 * 1024
A100_SMS = 108


if HAS_TRITON:

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
    block_m: int = 128,
    block_n: int = 64,
) -> torch.Tensor:
    """FlashAttention forward.

    Args:
        q: (batch, heads, q_len, head_dim)
        k, v: (batch, kv_heads, kv_len, head_dim); kv_heads must divide heads.

    The default tiling is 128 queries by 64 keys. A wider query tile amortises
    the Q load and the online-softmax rescaling over more rows, and it matters a
    lot: measured on an A100 at 8192 tokens with 8 heads of width 128, 128x64
    runs in 1.64 ms against 3.35 ms for 64x64.

    On float32 inputs `tl.dot` uses the TF32 tensor cores, which carry 10
    mantissa bits, so results differ from a full float32 matmul by around 1e-3
    relative. That is the intended trade and it is why the labs compare against
    a tolerance rather than expecting agreement to float32 precision.
    """
    if not HAS_TRITON:
        raise RuntimeError("flash_attention needs Triton and a CUDA device")
    batch, heads, q_len, head_dim = q.shape
    kv_heads, kv_len = k.shape[1], k.shape[2]
    if heads % kv_heads:
        raise ValueError(f"{heads} query heads do not divide into {kv_heads} KV heads")

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    scale = scale or head_dim**-0.5

    # Triton stages the K and V tiles through shared memory, one buffer per
    # pipeline stage. An A100 SM has 164 KB of it, and 3 stages of float32 tiles
    # need 192 KB, so the stage count has to follow the dtype:
    #
    #     bytes = num_stages * BLOCK_N * HEAD_DIM * element_size * 2
    #
    # Exceeding the limit raises OutOfResources at launch rather than failing
    # quietly, so this is safe to compute rather than guess.
    element_size = q.element_size()
    num_stages = 4
    while num_stages > 1 and (
        num_stages * block_n * head_dim * element_size * 2 > SHARED_MEMORY_BYTES
    ):
        num_stages -= 1

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
        num_stages=num_stages,
    )
    return out


def count_programs(batch: int, heads: int, q_len: int, kv_len: int, block_m: int) -> int:
    """Thread blocks (Triton programs) the forward kernel launches.

    The grid is `(ceil(q_len / block_m), batch * heads)`. Each program owns one
    tile of query rows for one query head, so the count depends on the number
    of query tiles and query heads. It does not depend on `kv_len`: the key
    loop runs inside a program. It does not depend on the number of KV heads
    either, because grouped-query attention shares K and V, not programs.
    """
    query_tiles = -(-q_len // block_m)
    return query_tiles * batch * heads


def sm_fill(programs: int, num_sms: int = A100_SMS) -> float:
    """Fraction of the SMs that hold at least one program in the first wave."""
    return min(programs, num_sms) / num_sms


def rescale_elements(
    q_len: int,
    kv_len: int,
    block_m: int,
    block_n: int,
    head_dim: int,
    version: int,
    causal: bool = True,
) -> int:
    """Accumulator elements multiplied or divided by a row factor, per head.

    These are non-matmul FLOPs on the (rows, head_dim) output tile.

    FlashAttention-1 rescales both terms of the output update on every KV tile
    (the old output and the new `P @ V`), so it does 2 scalings per tile and no
    final divide. FlashAttention-2 keeps the output unnormalized: 1 scaling per
    tile (the old output, by exp(m_old - m_new)), then one divide by the row sum
    when the loop ends.
    """
    if version not in (1, 2):
        raise ValueError("version must be 1 or 2")
    total = 0
    offset = kv_len - q_len
    for tile in range(-(-q_len // block_m)):
        rows = min(block_m, q_len - tile * block_m)
        hi = min(kv_len, (tile + 1) * block_m + offset) if causal else kv_len
        steps = -(-hi // block_n)
        scalings = 2 * steps if version == 1 else steps + 1
        total += scalings * rows * head_dim
    return total
