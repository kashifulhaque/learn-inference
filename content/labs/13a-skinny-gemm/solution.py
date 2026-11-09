"""Reference counts for a skinny GEMM, and a W4A16 unpack.

Peaks are chapter 10's rated pair: 312 TFLOP/s and 1935 GB/s.
The Triton row kernel is imported only when `skinny_gemm` runs, so this
file loads on a machine with no CUDA toolkit.
"""

from __future__ import annotations

import torch

GROUP = 128
PEAK_FLOPS = 312 * 10**12
PEAK_BYTES = 1935 * 10**9


def gemm_flops(m: int, k: int, n: int) -> int:
    """FLOPs in C = A @ B, with A shaped (m, k) and B shaped (k, n)."""
    return 2 * m * k * n


def _weight_bytes(k: int, n: int, storage: str) -> int:
    if storage == "bf16":
        return 2 * k * n
    if storage == "w4a16":
        payload = (k * n + 1) // 2
        scales = ((k + GROUP - 1) // GROUP) * n * 2
        return payload + scales
    raise ValueError(f"storage must be 'bf16' or 'w4a16', got {storage!r}")


def gemm_bytes(m: int, k: int, n: int, storage: str = "bf16") -> int:
    """Bytes for A, the weight, and C. The output term is 2*m*n."""
    return 2 * m * k + _weight_bytes(k, n, storage) + 2 * m * n


def gemm_roofline(m: int, k: int, n: int, storage: str = "bf16") -> dict:
    """Intensity and the rated-ridge bound. Times are seconds."""
    flops = gemm_flops(m, k, n)
    nbytes = gemm_bytes(m, k, n, storage)
    left = flops * 1935
    right = nbytes * 312_000
    if left < right:
        kind = "memory"
    elif left > right:
        kind = "compute"
    else:
        kind = "ridge"
    return {
        "flops": flops,
        "bytes": nbytes,
        "intensity": flops / nbytes,
        "bound": kind,
        "memory_s": nbytes / PEAK_BYTES,
        "compute_s": flops / PEAK_FLOPS,
    }


def w4a16_gemv(x, packed, scales, group_size: int = 128):
    """Unpack signed int4 nibbles, scale by group, and multiply."""
    k, _nbytes = packed.shape
    n = scales.shape[-1]
    low = (packed & 0x0F).to(torch.int16) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    q = torch.empty(k, n, dtype=torch.float32, device=packed.device)
    even_cols = torch.arange(0, n, 2, device=packed.device)
    odd_cols = torch.arange(1, n, 2, device=packed.device)
    q[:, even_cols] = low[:, : even_cols.numel()].to(torch.float32)
    if odd_cols.numel():
        q[:, odd_cols] = high[:, : odd_cols.numel()].to(torch.float32)
    groups = torch.arange(k, device=packed.device) // group_size
    weight = q * scales.to(dtype=torch.float32, device=packed.device)[groups]
    return x.to(dtype=torch.float32, device=packed.device) @ weight


def skinny_gemm(a, b):
    """Row GEMM: one program per row of A and a tile of columns of B."""
    import triton
    import triton.language as tl

    @triton.jit
    def _row_gemm(
        a_ptr, b_ptr, c_ptr, n, k,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.program_id(1)
        offs_n = col * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, k, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < k
            a_vec = tl.load(
                a_ptr + row * stride_am + offs_k * stride_ak,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            b_tile = tl.load(
                b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(a_vec[:, None] * b_tile, axis=0)
        tl.store(
            c_ptr + row * stride_cm + offs_n * stride_cn,
            acc,
            mask=mask_n,
        )

    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)
    block_n = 128
    block_k = 64
    grid = (m, triton.cdiv(n, block_n))
    _row_gemm[grid](
        a, b, c, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return c
