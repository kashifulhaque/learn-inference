"""Numbers for the int4 and KV chapter.

Nothing here is an A100 kernel time. The GPTQ and AWQ errors are the
lab's synthetic layers. The KV errors are one seeded head with a planted
outlier channel. The FP8 rows are the cast in the lab, checked against
torch.float8_e4m3fn and torch.float8_e5m2 when those dtypes exist.

Run from the repository root:

    python3 scripts/figures/18a-int4-and-kv-quantization.py
"""

from __future__ import annotations

import math

import torch


def quantize_groupwise_int4(weight: torch.Tensor, group: int):
    """Symmetric int4 along the last dimension. Scale is amax / 7."""
    values = weight.to(torch.float64)
    *lead, n = values.shape
    n_groups = (n + group - 1) // group
    pad = n_groups * group - n
    if pad:
        values = torch.nn.functional.pad(values, (0, pad))
    grouped = values.reshape(*lead, n_groups, group)
    peak = grouped.abs().amax(dim=-1, keepdim=True)
    scale = peak / 7
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = torch.clamp(torch.round(grouped / safe), -7, 7)
    codes = torch.where(scale == 0, torch.zeros_like(codes), codes)
    dequant = (codes * scale).reshape(*lead, n_groups * group)[..., :n]
    return dequant


def gptq_quantize(weight: torch.Tensor, hessian: torch.Tensor, group: int):
    """Algorithm 1 from GPTQ, with a symmetric int4 group grid.

    The Hessian is X^T X. Damping is 1% of the mean diagonal. The update
    uses the upper Cholesky factor of the inverse, which is the factor
    Algorithm 1 stores.
    """
    nmax = 7
    work = weight.to(torch.float64).clone()
    gram = hessian.to(torch.float64).clone()
    gram.diagonal().add_(0.01 * gram.diag().mean())
    factor = torch.linalg.cholesky(gram)
    inverse = torch.linalg.cholesky(torch.cholesky_inverse(factor), upper=True)
    out = torch.zeros_like(work)
    columns = work.shape[1]
    for start in range(0, columns, group):
        end = min(start + group, columns)
        block = work[:, start:end].clone()
        scale = block.abs().amax(dim=1, keepdim=True) / nmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        local = inverse[start:end, start:end]
        residual = torch.zeros_like(block)
        for offset in range(end - start):
            column = block[:, offset]
            code = torch.clamp(torch.round(column / scale[:, 0]), -nmax, nmax)
            dequant = code * scale[:, 0]
            out[:, start + offset] = dequant
            error = (column - dequant) / local[offset, offset]
            residual[:, offset] = error
            block[:, offset:] -= error.unsqueeze(1) * local[offset, offset:]
        if end < columns:
            work[:, end:] -= residual @ inverse[start:end, end:]
    return out


def output_error(inputs, weight, quantized):
    target = inputs.to(torch.float64) @ weight.to(torch.float64).T
    actual = inputs.to(torch.float64) @ quantized.to(torch.float64).T
    return (target - actual).norm().item() / target.norm().item()


def correlated_layer(rho: float, seed: int = 18):
    """128 rows, 32 inputs, 4 outputs. Float32 draws, seed 18."""
    generator = torch.Generator().manual_seed(seed)
    shared = torch.randn(128, 1, generator=generator)
    noise = torch.randn(128, 32, generator=generator)
    inputs = math.sqrt(rho) * shared + math.sqrt(1 - rho) * noise
    inputs = inputs / inputs.std()
    weight = torch.randn(4, 32, generator=generator)
    hessian = inputs.double().T @ inputs.double()
    return inputs, weight, hessian


def toy_layer():
    """Three weights and 64 correlated rows. Seed 0, float64 draws."""
    generator = torch.Generator().manual_seed(0)
    shared = torch.randn(64, 1, generator=generator, dtype=torch.float64)
    noise = torch.randn(64, 3, generator=generator, dtype=torch.float64)
    inputs = 0.9 * shared + 0.1 * noise
    weight = torch.tensor([[0.20, 0.10, 0.40]], dtype=torch.float64)
    return inputs, weight, inputs.T @ inputs


def codes_of(quantized, scale):
    return [int(torch.round(value / scale).item()) for value in quantized.view(-1)]


def attention(query, key, value):
    scores = (query @ key.T) / math.sqrt(query.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    return (weights @ value).squeeze(0)


def symmetric_int4(tensor, dim):
    peak = tensor.abs().amax(dim=dim, keepdim=True)
    scale = peak / 7
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = torch.clamp(torch.round(tensor / safe), -7, 7)
    codes = torch.where(scale == 0, torch.zeros_like(codes), codes)
    return codes * scale


def kv_head():
    generator = torch.Generator().manual_seed(18)
    key = torch.randn(32, 16, generator=generator) * 0.4
    key[:, 0] = 12
    key[7, 1:] = key[7, 1:] + 2.5
    value = torch.randn(32, 16, generator=generator)
    query = torch.zeros(1, 16)
    query[0, 1] = 4
    return query, key, value


def relative(actual, expected):
    return (actual - expected).norm().item() / expected.norm().item()


def awq_table(inputs, weight):
    magnitude = inputs.abs().mean(dim=0).double()
    base = magnitude / magnitude.mean()
    target = inputs.double() @ weight.double().T
    norm = target.norm()
    rows = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        scale = base.pow(alpha)
        scaled = weight.double() * scale.unsqueeze(0)
        restored = quantize_groupwise_int4(scaled, weight.shape[1]) / scale.unsqueeze(0)
        error = (inputs.double() @ restored.T - target).norm().item() / norm.item()
        rows.append((alpha, error))
    return rows


def fp8_samples():
    e4 = torch.tensor(
        [1.0625, 1.1875, 448.0, 464.0, 480.0, 2.0**-9, 2.0**-10],
        dtype=torch.float32,
    )
    e5 = torch.tensor(
        [1.125, 57344.0, 60000.0, 61440.0, 2.0**-16, 2.0**-17],
        dtype=torch.float32,
    )
    print("e4m3 torch", e4.to(torch.float8_e4m3fn).float().tolist())
    print("e5m2 torch", e5.to(torch.float8_e5m2).float().tolist())


def main() -> None:
    inputs, weight, hessian = toy_layer()
    nearest = quantize_groupwise_int4(weight, 3)
    compensated = gptq_quantize(weight, hessian, 3)
    scale = 0.40 / 7
    print(
        "toy rtn codes",
        codes_of(nearest, scale),
        f"rel {output_error(inputs, weight, nearest):.4f}",
    )
    print(
        "toy gptq codes",
        codes_of(compensated, scale),
        f"rel {output_error(inputs, weight, compensated):.4f}",
    )

    print("rho rtn gptq margin")
    for rho in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
        layer_x, layer_w, layer_h = correlated_layer(rho)
        rtn = quantize_groupwise_int4(layer_w, 8)
        gptq = gptq_quantize(layer_w, layer_h, 8)
        rtn_error = output_error(layer_x, layer_w, rtn)
        gptq_error = output_error(layer_x, layer_w, gptq)
        print(f"{rho:.2f} {rtn_error:.6f} {gptq_error:.6f} {rtn_error - gptq_error:.6f}")

    torch.manual_seed(4)
    channel = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2, 12.0])
    awq_x = torch.randn(48, 6) * channel
    awq_w = torch.randn(2, 6)
    awq_w[:, -1] *= 0.2
    print("awq alpha error")
    for alpha, error in awq_table(awq_x, awq_w):
        print(f"{alpha:.2f} {error:.6f}")

    query, key, value = kv_head()
    base = attention(query, key, value)
    key_token = symmetric_int4(key, dim=1)
    key_channel = symmetric_int4(key, dim=0)
    value_token = symmetric_int4(value, dim=1)
    value_channel = symmetric_int4(value, dim=0)
    print(f"kv per-token key {relative(attention(query, key_token, value), base):.6f}")
    print(f"kv per-channel key {relative(attention(query, key_channel, value), base):.6f}")
    print(f"kv per-token value {relative(attention(query, key, value_token), base):.6f}")
    print(f"kv per-channel value {relative(attention(query, key, value_channel), base):.6f}")
    print(f"bytes g128 {0.5 + 2 / 128:.6f}")
    print(f"bytes n10 g4 {0.5 + 2 * math.ceil(10 / 4) / 10:.6f}")
    fp8_samples()


if __name__ == "__main__":
    main()
