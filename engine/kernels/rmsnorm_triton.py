"""Fused RMSNorm, with and without the residual add.

RMSNorm moves 2 bytes in and 2 bytes out per element and does about 4 FLOPs.
That is an arithmetic intensity near 1, far under the A100's ridge point of
about 161, so it is purely bandwidth bound and the only way to make it faster is
to move less data. Fusing the residual add is exactly that: the unfused pair
reads x, reads the residual, writes the sum, reads it back, and writes the norm
— five trips. The fused version does two reads and two writes.

One block handles one row, and the row stays in registers between the reduction
and the scaling, so each element crosses the memory bus once in each direction.
That caps the row width at what fits in a block; past `MAX_SINGLE_BLOCK` the
kernel falls back to walking the row in tiles, which costs a second pass. The
second pass usually hits L2, but on a working set larger than 40 MB it does not,
so the single-block path is the one to stay on when the row allows it.

Measured on an A100 80GB PCIe at 4096 rows of 5120 columns in bfloat16:

    plain copy, the bandwidth floor      0.066 ms
    single block, num_warps=8            0.089 ms   945 GB/s
    tiled, BLOCK=2048                    0.090 ms   929 GB/s

num_warps=8 beat both 16 and 32 on that shape. More warps per row buys
parallelism within the row and costs occupancy across rows, and for a row this
narrow the trade lands early.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Widest row a single block still handles well. Above this the register file
# spills and the tiled path is faster.
MAX_SINGLE_BLOCK = 16384
TILE = 2048


@triton.jit
def _rms_norm_fwd(
    x_ptr, w_ptr, out_ptr,
    stride_row,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * stride_row
    out_ptr += row * stride_row

    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    # Accumulate in float32 whatever the input dtype is. Summing 5120 squared
    # bfloat16 values loses the tail of the reduction.
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    scale = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + cols, (x * scale * w).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rms_norm_tiled_fwd(
    x_ptr, w_ptr, out_ptr,
    stride_row,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    """Two-pass fallback for rows too wide to hold in registers."""
    row = tl.program_id(0)
    x_ptr += row * stride_row
    out_ptr += row * stride_row

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        acc += x * x
    scale = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / n_cols + eps)

    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + cols, (x * scale * w).to(out_ptr.dtype.element_ty),
                 mask=mask)


@triton.jit
def _rms_norm_residual_fwd(
    x_ptr, res_ptr, w_ptr, out_ptr, new_res_ptr,
    stride_row,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offset = row * stride_row
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(res_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    total = x + res

    # The next layer's residual connection needs the sum, so write it here. It
    # is never read back: `total` stays in registers for the normalisation.
    tl.store(new_res_ptr + offset + cols,
             total.to(new_res_ptr.dtype.element_ty), mask=mask)

    var = tl.sum(total * total, axis=0) / n_cols
    scale = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offset + cols,
             (total * scale * w).to(out_ptr.dtype.element_ty), mask=mask)


def _config(n_cols: int) -> tuple[int, int, bool]:
    """Return (block, num_warps, use_tiled) for a row of `n_cols` elements."""
    if n_cols <= MAX_SINGLE_BLOCK:
        return triton.next_power_of_2(n_cols), 8, False
    return TILE, 8, True


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in replacement for engine.layers.rmsnorm.rms_norm.

    Pass `out` to write into an existing buffer. Allocating a 42 MB output on
    every call costs about as much as the kernel does at this size, so the
    engine's hot paths hand in a buffer they already own.
    """
    shape = x.shape
    x = x.reshape(-1, shape[-1]).contiguous()
    out = torch.empty_like(x) if out is None else out.reshape(-1, shape[-1])
    n_cols = x.shape[-1]
    block, num_warps, tiled = _config(n_cols)
    kernel = _rms_norm_tiled_fwd if tiled else _rms_norm_fwd
    kernel[(x.shape[0],)](
        x, weight, out, x.stride(0), n_cols, eps, BLOCK=block, num_warps=num_warps,
    )
    return out.reshape(shape)


def rms_norm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
    new_residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (normed, x + residual), reading each input once.

    Rows wider than MAX_SINGLE_BLOCK are not supported here: the whole point is
    to hold the sum in registers, and a tiled version would have to read it back
    and give up the saving.

    Pass `out` and `new_residual` to write into buffers you already own. This
    version writes two 42 MB tensors, so on a shape like this the allocations
    cost more than the traffic the fusion saves. Measured at 4096 rows of 5120
    columns: with fresh allocations the fused path is 0.91x the unfused one;
    with buffers reused it is 1.29x.
    """
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    res2 = residual.reshape(-1, shape[-1]).contiguous()
    n_cols = x2.shape[-1]
    if n_cols > MAX_SINGLE_BLOCK:
        raise ValueError(
            f"rms_norm_residual handles rows up to {MAX_SINGLE_BLOCK} wide; "
            f"got {n_cols}. Add the residual separately and call rms_norm."
        )

    out = torch.empty_like(x2) if out is None else out.reshape(-1, n_cols)
    new_res = (
        torch.empty_like(x2) if new_residual is None
        else new_residual.reshape(-1, n_cols)
    )
    block, num_warps, _ = _config(n_cols)
    _rms_norm_residual_fwd[(x2.shape[0],)](
        x2, res2, weight, out, new_res, x2.stride(0), n_cols, eps,
        BLOCK=block, num_warps=num_warps,
    )
    return out.reshape(shape), new_res.reshape(shape)
