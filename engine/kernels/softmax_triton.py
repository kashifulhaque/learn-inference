"""Row-wise softmax, the single-kernel version.

PyTorch's softmax makes several passes over the row: one for the max, one for
the exponentials, one for the sum, one to divide. Each pass is a separate kernel
and each reads the row from HBM again. Holding the row in registers does all
four in one trip, which is roughly a 4x saving on a bandwidth-bound operation.

The same trick, applied when the row does not fit in registers, is the online
softmax that FlashAttention runs on.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_fwd(x_ptr, out_ptr, stride_row, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=float("-inf"))
    x = x.to(tl.float32)

    # Subtract the max before exponentiating: exp overflows float32 above ~88,
    # and attention logits comfortably exceed that at long context.
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    out = numerator / tl.sum(numerator, axis=0)

    tl.store(out_ptr + row * stride_row + cols,
             out.to(out_ptr.dtype.element_ty), mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x = x.reshape(-1, shape[-1]).contiguous()
    out = torch.empty_like(x)
    n_cols = x.shape[-1]
    block = triton.next_power_of_2(n_cols)
    _softmax_fwd[(x.shape[0],)](
        x, out, x.stride(0), n_cols,
        BLOCK=block,
        num_warps=max(1, min(16, block // 256)),
    )
    return out.reshape(shape)
