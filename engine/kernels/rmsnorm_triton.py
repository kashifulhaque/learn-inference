"""Fused RMSNorm, with and without the residual add.

RMSNorm moves 2 bytes in and 2 bytes out per element and does about 4 FLOPs.
That is an arithmetic intensity of 1, far under the A100's ridge point of ~150,
so it is purely bandwidth bound and the only way to make it faster is to move
less data. Fusing the residual add is exactly that: the unfused pair reads x and
residual, writes the sum, reads it back, and writes the norm — five trips. The
fused version does two reads and two writes.

One block handles one row, so the whole row stays in registers between the
reduction and the scaling. That caps the row width at what fits, which is fine
here: 5120 elements.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


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

    # Accumulate in float32 whatever the input dtype is.
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    scale = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + cols, (x * scale * w).to(out_ptr.dtype.element_ty), mask=mask)


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
    # The updated residual is what the next layer needs, so write it here
    # rather than making the caller add again.
    tl.store(
        new_res_ptr + offset + cols,
        total.to(new_res_ptr.dtype.element_ty),
        mask=mask,
    )

    var = tl.sum(total * total, axis=0) / n_cols
    scale = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + offset + cols,
        (total * scale * w).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Drop-in replacement for engine.layers.rmsnorm.rms_norm."""
    shape = x.shape
    x = x.reshape(-1, shape[-1]).contiguous()
    out = torch.empty_like(x)
    n_cols = x.shape[-1]
    block = triton.next_power_of_2(n_cols)
    _rms_norm_fwd[(x.shape[0],)](
        x, weight, out, x.stride(0), n_cols, eps,
        BLOCK=block,
        num_warps=max(1, min(16, block // 256)),
    )
    return out.reshape(shape)


def rms_norm_residual(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (normed, x + residual) in one pass."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    res2 = residual.reshape(-1, shape[-1]).contiguous()
    out = torch.empty_like(x2)
    new_res = torch.empty_like(x2)
    n_cols = x2.shape[-1]
    block = triton.next_power_of_2(n_cols)
    _rms_norm_residual_fwd[(x2.shape[0],)](
        x2, res2, weight, out, new_res, x2.stride(0), n_cols, eps,
        BLOCK=block,
        num_warps=max(1, min(16, block // 256)),
    )
    return out.reshape(shape), new_res.reshape(shape)
