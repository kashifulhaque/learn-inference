"""Fused SwiGLU activation: silu(gate) * up in one pass.

The two matmuls dominate the FLOPs, but the elementwise step between them is
pure bandwidth: unfused it reads gate, writes silu(gate), reads it back, reads
up, and writes the product. For a batch of 4096 tokens and an intermediate size
of 17408 each of those tensors is 143 MB in bfloat16. Fusing turns five trips
into three.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_fwd(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # silu(x) = x * sigmoid(x)
    silu = gate * tl.sigmoid(gate)
    tl.store(out_ptr + offsets, (silu * up).to(out_ptr.dtype.element_ty), mask=mask)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate, up = gate.contiguous(), up.contiguous()
    out = torch.empty_like(gate)
    n = gate.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _swiglu_fwd[grid](gate, up, out, n, BLOCK=1024, num_warps=4)
    return out
