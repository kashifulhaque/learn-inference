"""Lab 13 — fused kernels in Triton."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_fwd(x_ptr, w_ptr, out_ptr, stride_row, n_cols, eps,
                  BLOCK: tl.constexpr):
    """One block per row: load, reduce, scale, store."""
    # TODO
    pass


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
             out: torch.Tensor | None = None):
    """Normalize over the last dimension. Accept any leading shape.

    When `out` is given, write into it instead of allocating. The harness uses
    that path to time the kernel without the allocator in the way.
    """
    # TODO: reshape to 2-D, launch one program per row, reshape back.
    raise NotImplementedError


@triton.jit
def _rms_norm_residual_fwd(x_ptr, res_ptr, w_ptr, out_ptr, new_res_ptr,
                           stride_row, n_cols, eps, BLOCK: tl.constexpr):
    """Add the residual and normalize the sum, in one pass over the row."""
    # TODO
    pass


def rms_norm_residual(x, residual, weight, eps: float = 1e-6,
                      out: torch.Tensor | None = None,
                      new_residual: torch.Tensor | None = None):
    """Return (normalized, x + residual).

    The updated residual is written because the next layer needs it, but it is
    never read back, so this saves one full trip over the tensor.

    As with `rms_norm`, `out` and `new_residual` let the caller supply buffers.
    """
    # TODO
    raise NotImplementedError


@triton.jit
def _swiglu_fwd(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    """out = silu(gate) * up, where silu(v) = v * sigmoid(v)."""
    # TODO
    pass


def swiglu(gate: torch.Tensor, up: torch.Tensor, out: torch.Tensor | None = None):
    """Fused activation for the MLP's middle step."""
    # TODO
    raise NotImplementedError
