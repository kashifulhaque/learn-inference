"""Reference solution for int4 packing, GPTQ, AWQ, KV axes, and FP8."""

from __future__ import annotations

import torch


def pack_int4(codes):
    values = codes.to(torch.int64)
    *lead, n = values.shape
    width = (n + 7) // 8
    pad = width * 8 - n
    if pad:
        # A pad of 0 would become nibble 8. Unused nibbles stay 0, which is code -8.
        values = torch.nn.functional.pad(values, (0, pad), value=-8)
    nibbles = ((values + 8) & 0xF).reshape(*lead, width, 8)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    words = (nibbles << shifts).sum(dim=-1)
    return words.to(torch.int32)


def unpack_int4(packed, n: int):
    shifts = torch.arange(8, dtype=torch.int64) * 4
    words = packed.to(torch.int64) & 0xFFFFFFFF
    nibbles = (words.unsqueeze(-1) >> shifts) & 0xF
    codes = nibbles.to(torch.int32) - 8
    flat = codes.reshape(*packed.shape[:-1], -1)
    return flat[..., :n]


def quantize_groupwise_int4(weight, group_size: int):
    values = weight.to(torch.float64)
    *lead, n = values.shape
    n_groups = (n + group_size - 1) // group_size
    codes = torch.empty(*lead, n, dtype=torch.int64)
    scales = torch.empty(*lead, n_groups, dtype=torch.float64)
    flat_v = values.reshape(-1, n)
    flat_q = codes.reshape(-1, n)
    flat_s = scales.reshape(-1, n_groups)
    for group in range(n_groups):
        start = group * group_size
        end = min(start + group_size, n)
        block = flat_v[:, start:end]
        peak = block.abs().amax(dim=1, keepdim=True)
        scale = peak / 7
        safe = torch.where(peak == 0, torch.ones_like(scale), scale)
        rounded = torch.round(block / safe).clamp(-7, 7)
        rounded = torch.where(peak == 0, torch.zeros_like(rounded), rounded)
        flat_q[:, start:end] = rounded.to(torch.int64)
        flat_s[:, group] = scale.squeeze(1)
    return codes.to(torch.int32), scales.to(torch.float32)


def _dequant(codes, scales, group_size: int) -> torch.Tensor:
    n = codes.shape[-1]
    index = torch.arange(n) // group_size
    return codes.to(torch.float64) * scales.to(torch.float64)[..., index]


def gptq_quantize(weight, hessian, bits: int, group: int):
    nmax = 2 ** (bits - 1) - 1
    work = weight.to(torch.float64).clone()
    gram = hessian.to(torch.float64).clone()
    gram.diagonal().add_(0.01 * gram.diag().mean())
    factor = torch.linalg.cholesky(gram)
    inverse = torch.linalg.cholesky(torch.cholesky_inverse(factor), upper=True)
    rows, columns = work.shape
    dequant = torch.zeros(rows, columns, dtype=torch.float64)
    for start in range(0, columns, group):
        end = min(start + group, columns)
        block = work[:, start:end].clone()
        peak = block.abs().amax(dim=1, keepdim=True)
        scale = peak / nmax
        scale = torch.where(peak == 0, torch.ones_like(scale), scale)
        local = inverse[start:end, start:end]
        residual = torch.zeros_like(block)
        for offset in range(end - start):
            column = block[:, offset]
            code = torch.round(column / scale[:, 0]).clamp(-nmax, nmax)
            restored = code * scale[:, 0]
            dequant[:, start + offset] = restored
            error = (column - restored) / local[offset, offset]
            residual[:, offset] = error
            block[:, offset:] -= error.unsqueeze(1) * local[offset, offset:]
        if end < columns:
            work[:, end:] -= residual @ inverse[start:end, end:]
    return dequant.to(weight.dtype)


def awq_scale_search(inputs, weight, group_size: int, alphas):
    magnitude = inputs.detach().abs().to(torch.float64).mean(dim=0).clamp(min=1e-12)
    base = magnitude / magnitude.mean()
    target = inputs.to(torch.float64) @ weight.to(torch.float64).T
    norm = target.norm().clamp(min=1e-12)
    best_error = None
    best_alpha = None
    best_scale = None
    for alpha in alphas:
        scale = base.pow(float(alpha))
        scaled = weight.to(torch.float64) * scale.unsqueeze(0)
        codes, scales = quantize_groupwise_int4(scaled, group_size)
        restored = _dequant(codes, scales, group_size) / scale.unsqueeze(0)
        error = (inputs.to(torch.float64) @ restored.T - target).norm() / norm
        alpha_value = float(alpha)
        if (
            best_error is None
            or error < best_error
            or (error == best_error and alpha_value < best_alpha)
        ):
            best_error = error
            best_alpha = alpha_value
            best_scale = scale
    return best_scale.to(torch.float32)


def _symmetric(tensor, dim: int) -> torch.Tensor:
    peak = tensor.abs().amax(dim=dim, keepdim=True)
    scale = peak / 7
    safe = torch.where(peak == 0, torch.ones_like(scale), scale)
    codes = torch.round(tensor / safe).clamp(-7, 7)
    codes = torch.where(peak == 0, torch.zeros_like(codes), codes)
    return codes * scale


def quantize_kv(key, value):
    return _symmetric(key, dim=0), _symmetric(value, dim=1)


def fp8_cast(values, fmt: str):
    """Round ties to even. Matches torch.float8_* on the values the lab checks."""
    if fmt == "e4m3":
        min_normal = 2.0**-6
        sub_ulp = 2.0**-9
        max_exp = 8
        max_finite = 448.0
        mantissa_bits = 3
        overflow_nan = True
    elif fmt == "e5m2":
        min_normal = 2.0**-14
        sub_ulp = 2.0**-16
        max_exp = 15
        max_finite = 57344.0
        mantissa_bits = 2
        overflow_nan = False
    else:
        raise ValueError(fmt)

    original = torch.as_tensor(values, dtype=torch.float32)
    nan = torch.isnan(original)
    positive_inf = original == float("inf")
    negative_inf = original == float("-inf")
    magnitude = torch.where(nan | positive_inf | negative_inf, torch.zeros_like(original), original.abs())
    subnormal = magnitude < min_normal
    _mantissa, exponent = torch.frexp(torch.clamp(magnitude, min=sub_ulp))
    binade = torch.clamp(exponent - 1, max=max_exp)
    normal_ulp = torch.pow(torch.tensor(2.0, dtype=torch.float32), (binade - mantissa_bits).float())
    ulp = torch.where(subnormal, torch.full_like(magnitude, sub_ulp), normal_ulp)
    rounded = torch.round(magnitude / ulp) * ulp
    overflow = rounded > max_finite
    if overflow_nan:
        result = torch.where(overflow | positive_inf | negative_inf | nan, torch.full_like(original, float("nan")), rounded)
    else:
        result = torch.where(overflow | positive_inf, torch.full_like(original, float("inf")), rounded)
        result = torch.where(negative_inf, torch.full_like(result, float("-inf")), result)
        result = torch.where(nan, torch.full_like(result, float("nan")), result)
    signed = torch.copysign(result, original)
    signed = torch.where(nan, torch.full_like(signed, float("nan")), signed)
    if overflow_nan:
        signed = torch.where(positive_inf | negative_inf | overflow, torch.full_like(signed, float("nan")), signed)
    return signed


def w4a16_matmul(inputs, packed, scales, group_size: int = 128):
    """Triton matmul. Not timed, and not called when CUDA is absent."""
    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(
        x_ptr, packed_ptr, scale_ptr, out_ptr,
        n, k,
        group: tl.constexpr,
        stride_xm, stride_xk, stride_pk, stride_pw,
        stride_sg, stride_sn, stride_cm, stride_cn,
    ):
        pid = tl.program_id(0)
        row = pid // n
        col = pid % n
        word = col // 8
        shift = (col % 8) * 4
        acc = 0.0
        for t in range(k):
            activation = tl.load(x_ptr + row * stride_xm + t * stride_xk).to(tl.float32)
            packed_word = tl.load(packed_ptr + t * stride_pk + word * stride_pw)
            nibble = (packed_word >> shift) & 0xF
            scale = tl.load(scale_ptr + (t // group) * stride_sg + col * stride_sn).to(tl.float32)
            acc += activation * (nibble.to(tl.float32) - 8.0) * scale
        tl.store(out_ptr + row * stride_cm + col * stride_cn, acc)

    rows, k = inputs.shape
    n = scales.shape[-1]
    output = torch.empty((rows, n), device=inputs.device, dtype=torch.float32)
    _kernel[(rows * n,)](
        inputs, packed, scales, output,
        n, k, group_size,
        inputs.stride(0), inputs.stride(1),
        packed.stride(0), packed.stride(1),
        scales.stride(0), scales.stride(1),
        output.stride(0), output.stride(1),
    )
    return output
