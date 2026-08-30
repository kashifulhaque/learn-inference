"""Lab 04 — RMSNorm.

    y = x / sqrt(mean(x^2) + eps) * w
"""

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize over the last dimension, accumulating in float32.

    The input may be bfloat16. The output must have the same dtype as the input.
    """
    # TODO
    raise NotImplementedError


def rms_norm_low_precision(x: torch.Tensor, weight: torch.Tensor,
                           eps: float = 1e-6) -> torch.Tensor:
    """The same operation with the reduction left in the input's dtype.

    Write this one deliberately wrong, so the harness can measure how much the
    float32 accumulation is worth.
    """
    # TODO
    raise NotImplementedError


def bytes_moved(x: torch.Tensor) -> int:
    """Bytes an ideal RMSNorm kernel must read and write for this input.

    Count the read of x and the write of y. The weight vector is negligible and
    stays in cache.
    """
    # TODO
    raise NotImplementedError
