import torch


def build_rope_cache(rotary_dim: int, max_position: int, theta: float = 10_000.0,
                     device="cpu", dtype=torch.float32):
    half = rotary_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    positions = torch.arange(max_position, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_partial(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                         rotary_dim: int) -> torch.Tensor:
    rot, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    cos_full = torch.cat((cos, cos), dim=-1)[None, None, :, :].to(x.dtype)
    sin_full = torch.cat((sin, sin), dim=-1)[None, None, :, :].to(x.dtype)
    rotated = rot * cos_full + rotate_half(rot) * sin_full
    if passthrough.numel() == 0:
        return rotated
    return torch.cat((rotated, passthrough), dim=-1)
