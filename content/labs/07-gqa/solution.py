import math

import torch


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    batch, heads, seq, dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, heads, repeats, seq, dim)
        .reshape(batch, heads * repeats, seq, dim)
    )


def causal_mask(q_len: int, kv_len: int, device) -> torch.Tensor:
    offset = kv_len - q_len
    idx_q = torch.arange(q_len, device=device).unsqueeze(-1)
    idx_k = torch.arange(kv_len, device=device).unsqueeze(0)
    return idx_k > idx_q + offset


def attention(q, k, v, causal: bool = True, scale: float | None = None):
    scale = scale or 1.0 / math.sqrt(q.shape[-1])
    repeats = q.shape[1] // k.shape[1]
    k = repeat_kv(k, repeats)
    v = repeat_kv(v, repeats)

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if causal:
        scores = scores.masked_fill(
            causal_mask(q.shape[2], k.shape[2], scores.device), float("-inf")
        )
    weights = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.matmul(weights, v)


def apply_output_gate(attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return attn_out * torch.sigmoid(gate)


def kv_bytes_per_token(num_kv_heads: int, head_dim: int, num_full_layers: int,
                       bytes_per_element: int = 2) -> int:
    return 2 * num_kv_heads * head_dim * bytes_per_element * num_full_layers
