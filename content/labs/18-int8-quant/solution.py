import torch


def _quantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (weight / scale).round().clamp(-128, 127).to(torch.int8)


def quantize_per_tensor(weight: torch.Tensor):
    scale = weight.abs().max() / 127.0
    scale = torch.clamp(scale, min=1e-12)
    return _quantize(weight, scale), scale


def quantize_per_channel(weight: torch.Tensor):
    scale = weight.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = torch.clamp(scale, min=1e-12)
    return _quantize(weight, scale), scale


def quantize_per_group(weight: torch.Tensor, group_size: int = 128):
    out_features, in_features = weight.shape
    grouped = weight.reshape(out_features, in_features // group_size, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = torch.clamp(scale, min=1e-12)
    quantized = _quantize(grouped, scale).reshape(out_features, in_features)
    return quantized, scale.squeeze(-1)


def dequantize(quantized: torch.Tensor, scale: torch.Tensor,
               group_size: int | None = None) -> torch.Tensor:
    if group_size is None:
        return quantized.float() * scale.float()
    out_features, in_features = quantized.shape
    grouped = quantized.reshape(out_features, in_features // group_size, group_size)
    return (grouped.float() * scale.float().unsqueeze(-1)).reshape(
        out_features, in_features
    )


def relative_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    original = original.float()
    reconstructed = reconstructed.float()
    return ((original - reconstructed).norm() / original.norm()).item()


def storage_bytes(shape, group_size: int | None = None,
                  scale_bytes: int = 2) -> int:
    out_features, in_features = shape
    weights = out_features * in_features
    scales = out_features if group_size is None else out_features * (
        in_features // group_size
    )
    return weights + scales * scale_bytes
