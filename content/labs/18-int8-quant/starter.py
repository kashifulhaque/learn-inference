"""Lab 18 — weight-only int8 quantization.

Decode is memory bound, so halving the bytes read roughly doubles it. The
question is what that costs in accuracy, and the answer depends almost entirely
on how many weights share a scale.
"""

import torch


def quantize_per_tensor(weight: torch.Tensor):
    """One scale for the whole tensor.

    Returns (quantized_int8, scale) where scale is a scalar tensor.
    """
    # TODO
    raise NotImplementedError


def quantize_per_channel(weight: torch.Tensor):
    """One scale per output row.

    Args:
        weight: (out_features, in_features)

    Returns (quantized_int8, scale) with scale of shape (out_features, 1).
    """
    # TODO
    raise NotImplementedError


def quantize_per_group(weight: torch.Tensor, group_size: int = 128):
    """One scale per `group_size` contiguous weights along the input dimension.

    Returns (quantized_int8, scale) with scale of shape
    (out_features, in_features // group_size). `in_features` is a multiple of
    `group_size`.
    """
    # TODO
    raise NotImplementedError


def dequantize(quantized: torch.Tensor, scale: torch.Tensor,
               group_size: int | None = None) -> torch.Tensor:
    """Reconstruct float weights.

    When `group_size` is given, `scale` has one entry per group and must be
    broadcast back across the group.
    """
    # TODO
    raise NotImplementedError


def relative_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Frobenius norm of the difference over the norm of the original."""
    # TODO
    raise NotImplementedError


def storage_bytes(shape, group_size: int | None = None,
                  scale_bytes: int = 2) -> int:
    """Bytes needed to store the quantized weights plus their scales.

    With no group size, assume one scale per output row.
    """
    # TODO
    raise NotImplementedError
