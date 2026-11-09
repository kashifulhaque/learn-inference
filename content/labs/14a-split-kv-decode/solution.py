"""Reference solution for the split-KV decode lab.

`choose_num_splits` is the same rule as
`scripts/figures/14a-split-kv-decode-attention.py`. The Triton kernel is
loaded only when a CUDA device is visible, so importing this file on a CPU
does not import Triton.
"""

from __future__ import annotations

import torch

BLOCK_N = 64
MAX_SPLITS = 128


def merge_partials(outs, lses):
    """Merge normalized partial outputs with their log-sum-exp values.

    `outs` is (splits, ..., dim) and `lses` is (splits, ...). A split whose
    log-sum-exp is not finite contributes nothing. The result is the
    log-sum-exp weighted average, which matches a softmax over the union of
    the splits in exact arithmetic.
    """
    outs = torch.as_tensor(outs, dtype=torch.float32)
    lses = torch.as_tensor(lses, dtype=torch.float32)
    if outs.shape[:-1] != lses.shape:
        raise ValueError(
            f"lses shape {tuple(lses.shape)} does not match outs {tuple(outs.shape)}"
        )
    neg = torch.full_like(lses, float("-inf"))
    finite = torch.isfinite(lses)
    safe = torch.where(finite, lses, neg)
    peak = safe.amax(dim=0)
    shifted = torch.where(finite, lses - peak, neg)
    weights = torch.exp(shifted)
    denom = weights.sum(dim=0)
    lse = torch.where(
        denom > 0,
        peak + torch.log(denom),
        torch.full_like(denom, float("-inf")),
    )
    numer = (weights.unsqueeze(-1) * outs).sum(dim=0)
    out = torch.zeros_like(numer)
    ok = (denom > 0).unsqueeze(-1)
    out = torch.where(ok, numer / denom.unsqueeze(-1).clamp(min=1), out)
    return out, lse


def choose_num_splits(batch: int, heads: int, kv_len: int, num_sms: int) -> int:
    """Split count for a query of length 1.

    Return 1 when `batch * heads` already covers 80% of `num_sms`, which
    includes the case `batch * heads >= num_sms`. Otherwise take the smallest
    split count whose wave efficiency is at least 85% of the best count up to
    `min(128, num_sms)`, then cap by the number of 64-token KV blocks.
    """
    if batch < 1 or heads < 1 or kv_len < 1 or num_sms < 1:
        return 1
    tasks = batch * heads
    if tasks * 5 >= num_sms * 4:
        return 1
    limit = min(MAX_SPLITS, num_sms)
    best = 0.0
    effs: list[float] = []
    for splits in range(1, limit + 1):
        waves = (tasks * splits + num_sms - 1) // num_sms
        eff = (tasks * splits) / (num_sms * waves)
        effs.append(eff)
        if eff > best:
            best = eff
    cutoff = 0.85 * best
    ideal = 1
    for splits, eff in enumerate(effs, start=1):
        if eff >= cutoff:
            ideal = splits
            break
    n_blocks = (kv_len + BLOCK_N - 1) // BLOCK_N
    return max(1, min(ideal, n_blocks))


def split_kv_decode(q, k, v, num_splits: int):
    """Attention for one query token, splitting the KV length.

    `q` is (batch, heads, dim). `k` and `v` are (batch, heads, kv_len, dim),
    one KV head per query head. The partial kernel writes a normalized output
    and a log-sum-exp per split. `merge_partials` is the reduction.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("split_kv_decode needs a CUDA device")
    import triton
    import triton.language as tl

    if q.ndim != 3 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("expected q (B, H, D) and k, v (B, H, L, D)")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[2] != k.shape[3]:
        raise ValueError("q and k disagree on batch, heads, or dim")
    splits = max(int(num_splits), 1)
    batch, heads, dim = q.shape
    kv_len = k.shape[2]
    scale = dim ** -0.5
    block_d = triton.next_power_of_2(dim)
    max_blocks = max(triton.cdiv(kv_len, BLOCK_N), 1)

    partial_out = torch.empty(
        splits, batch, heads, dim, device=q.device, dtype=torch.float32
    )
    partial_lse = torch.empty(
        splits, batch, heads, device=q.device, dtype=torch.float32
    )

    @triton.jit
    def _partial(
        q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kk, stride_kd,
        stride_vb, stride_vh, stride_vk, stride_vd,
        stride_ob, stride_oh, stride_od,
        stride_lb, stride_lh,
        kv_len, num_splits, head_dim, scale,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        head = tl.program_id(1)
        split = tl.program_id(2)
        offs_d = tl.arange(0, HEAD_DIM)
        dim_mask = offs_d < head_dim

        q = tl.load(
            q_ptr + batch_id * stride_qb + head * stride_qh + offs_d * stride_qd,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)

        n_blocks = tl.cdiv(kv_len, BLOCK_N)
        base = n_blocks // num_splits
        rem = n_blocks - base * num_splits
        count = base + tl.where(split < rem, 1, 0)
        start = split * base + tl.where(split < rem, split, rem)

        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        offs_n = tl.arange(0, BLOCK_N)

        for block in range(MAX_BLOCKS):
            if (block >= start) and (block < start + count):
                cols = block * BLOCK_N + offs_n
                valid = cols < kv_len
                k_ptrs = (
                    k_ptr + batch_id * stride_kb + head * stride_kh
                    + cols[:, None] * stride_kk + offs_d[None, :] * stride_kd
                )
                v_ptrs = (
                    v_ptr + batch_id * stride_vb + head * stride_vh
                    + cols[:, None] * stride_vk + offs_d[None, :] * stride_vd
                )
                key = tl.load(k_ptrs, mask=valid[:, None] & dim_mask[None, :], other=0.0)
                value = tl.load(v_ptrs, mask=valid[:, None] & dim_mask[None, :], other=0.0)
                key = key.to(tl.float32)
                value = value.to(tl.float32)
                scores = tl.sum(key * q[None, :], axis=1) * scale
                scores = tl.where(valid, scores, float("-inf"))

                m_new = tl.maximum(m_i, tl.max(scores, axis=0))
                correction = tl.exp(m_i - m_new)
                prob = tl.exp(scores - m_new)
                prob = tl.where(valid, prob, 0.0)
                l_i = l_i * correction + tl.sum(prob, axis=0)
                acc = acc * correction + tl.sum(prob[:, None] * value, axis=0)
                m_i = m_new

        acc = acc / tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc * tl.where(l_i == 0.0, 0.0, 1.0)
        lse = m_i + tl.log(tl.where(l_i == 0.0, 1.0, l_i))
        lse = tl.where(l_i == 0.0, float("-inf"), lse)
        tl.store(
            out_ptr + split * stride_ob + batch_id * stride_oh + head * stride_od
            + offs_d,
            acc,
            mask=dim_mask,
        )
        tl.store(lse_ptr + split * stride_lb + batch_id * stride_lh + head, lse)

    _partial[(batch, heads, splits)](
        q, k, v, partial_out, partial_lse,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        partial_out.stride(0), partial_out.stride(1), partial_out.stride(2),
        partial_lse.stride(0), partial_lse.stride(1),
        kv_len, splits, dim, scale,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=block_d,
        MAX_BLOCKS=max_blocks,
    )
    merged, _lse = merge_partials(
        partial_out.reshape(splits, batch * heads, dim),
        partial_lse.reshape(splits, batch * heads),
    )
    return merged.view(batch, heads, dim).to(dtype=q.dtype)
