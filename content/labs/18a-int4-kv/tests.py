"""Checks for int4 packing, GPTQ, AWQ, KV axes, and an FP8 cast.

The first check calls only `pack_int4`. Later checks add the unpack, the
group quantizer, GPTQ, the AWQ search, the KV cache, and the FP8 cast.
`w4a16_matmul` runs only when CUDA is available.

A plausible wrong implementation stores the first value in bits 4-7 and
the second value in bits 0-3, swapping the two nibbles of a byte. On
(1, -2, 7, 0, -8, 3, -1, 4) the low nibble is 1 + 8 = 9, so the low byte
is 0x69. The swap stores 6 there, and the word is not -944730263.

A GPTQ update that skips the Hessian correction is round-to-nearest. On
the seeded layer the Cholesky update wins by 0.0261 of relative output
error. The check requires a margin of at least 0.01, which is under half
of that gap, and it rejects an unquantized return by requiring the error
to stay above 0.005.

A second plausible wrong implementation quantizes keys per token, the same axis
as values. `quantize_kv` is compared with per-channel keys, so that version
fails with a key error of about 1.0.
"""

from __future__ import annotations

import math

import torch

from lab_common import Checks

# Half of the Cholesky gap on the seeded layer (0.026069), rounded down.
GPTQ_MARGIN = 0.01
# The Cholesky error on that layer is 0.015605. Zero would be the original weight.
GPTQ_FLOOR = 0.005


def _call(submission, name, *args, **kwargs):
    try:
        return getattr(submission, name)(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _ref_pack(codes: torch.Tensor) -> torch.Tensor:
    """Integer loop, so a vectorized shift bug does not hide in both copies."""
    rows = codes.detach().reshape(-1, codes.shape[-1]).to(torch.int64).tolist()
    width = (codes.shape[-1] + 7) // 8
    packed = []
    for row in rows:
        words = []
        for word in range(width):
            acc = 0
            for i in range(8):
                index = word * 8 + i
                if index < len(row):
                    acc |= (int(row[index]) + 8) << (4 * i)
            if acc >= 2**31:
                acc -= 2**32
            words.append(acc)
        packed.append(words)
    return torch.tensor(packed, dtype=torch.int32).reshape(*codes.shape[:-1], width)


def _ref_unpack(packed: torch.Tensor, n: int) -> torch.Tensor:
    rows = packed.detach().reshape(-1, packed.shape[-1]).to(torch.int32).tolist()
    decoded = []
    for row in rows:
        codes = []
        for word in row:
            bits = int(word) & 0xFFFFFFFF
            for i in range(8):
                codes.append(((bits >> (4 * i)) & 0xF) - 8)
        decoded.append(codes[:n])
    return torch.tensor(decoded, dtype=torch.int32).reshape(*packed.shape[:-1], n)


def _group_scales(weight: torch.Tensor, group: int) -> torch.Tensor:
    values = weight.to(torch.float64)
    *lead, n = values.shape
    n_groups = (n + group - 1) // group
    scales = torch.empty(*lead, n_groups, dtype=torch.float64)
    flat = values.reshape(-1, n)
    out = scales.reshape(-1, n_groups)
    for index in range(n_groups):
        start = index * group
        end = min(start + group, n)
        out[:, index] = flat[:, start:end].abs().amax(dim=1) / 7
    return scales


def _rtn(weight: torch.Tensor, group: int) -> torch.Tensor:
    values = weight.to(torch.float64)
    *lead, n = values.shape
    n_groups = (n + group - 1) // group
    restored = torch.empty_like(values)
    flat_w = values.reshape(-1, n)
    flat_o = restored.reshape(-1, n)
    for index in range(n_groups):
        start = index * group
        end = min(start + group, n)
        block = flat_w[:, start:end]
        peak = block.abs().amax(dim=1, keepdim=True)
        scale = peak / 7
        safe = torch.where(peak == 0, torch.ones_like(scale), scale)
        codes = torch.round(block / safe).clamp(-7, 7)
        codes = torch.where(peak == 0, torch.zeros_like(codes), codes)
        flat_o[:, start:end] = codes * scale
    return restored


def _output_error(inputs, weight, quantized) -> float:
    target = inputs.to(torch.float64) @ weight.to(torch.float64).T
    actual = inputs.to(torch.float64) @ quantized.to(torch.float64).T
    return ((target - actual).norm() / target.norm()).item()


def _lab_layer():
    """Same draws as scripts/figures/18a-int4-and-kv-quantization.py."""
    generator = torch.Generator().manual_seed(18)
    shared = torch.randn(128, 1, generator=generator)
    noise = torch.randn(128, 32, generator=generator)
    inputs = math.sqrt(0.95) * shared + math.sqrt(1 - 0.95) * noise
    inputs = inputs / inputs.std()
    weight = torch.randn(4, 32, generator=generator)
    hessian = inputs.double().T @ inputs.double()
    return inputs, weight, hessian


def _awq_base(inputs: torch.Tensor) -> torch.Tensor:
    magnitude = inputs.abs().mean(dim=0).double().clamp(min=1e-12)
    return magnitude / magnitude.mean()


def _awq_error(inputs, weight, scale, group: int) -> float:
    scaled = weight.double() * scale.unsqueeze(0)
    restored = _rtn(scaled, group) / scale.unsqueeze(0)
    return _output_error(inputs, weight, restored)


def _best_alpha(inputs, weight, group: int, alphas):
    base = _awq_base(inputs)
    best = None
    for alpha in alphas:
        scale = base.pow(float(alpha))
        error = _awq_error(inputs, weight, scale, group)
        if best is None or error < best[0] or (error == best[0] and float(alpha) < best[1]):
            best = (error, float(alpha), scale)
    return best


def _symmetric(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    peak = tensor.abs().amax(dim=dim, keepdim=True)
    scale = peak / 7
    safe = torch.where(peak == 0, torch.ones_like(scale), scale)
    codes = torch.round(tensor / safe).clamp(-7, 7)
    codes = torch.where(peak == 0, torch.zeros_like(codes), codes)
    return codes * scale


def _attention(query, key, value):
    scores = (query @ key.T) / math.sqrt(query.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    return (weights @ value).squeeze(0)


def _kv_tensors():
    generator = torch.Generator().manual_seed(18)
    key = torch.randn(32, 16, generator=generator) * 0.4
    key[:, 0] = 12
    key[7, 1:] = key[7, 1:] + 2.5
    value = torch.randn(32, 16, generator=generator)
    query = torch.zeros(1, 16)
    query[0, 1] = 4
    return query, key, value


def _same_float(actual, expected) -> bool:
    actual = torch.as_tensor(actual, dtype=torch.float32)
    expected = torch.as_tensor(expected, dtype=torch.float32)
    if actual.shape != expected.shape:
        return False
    nan = torch.isnan(actual) & torch.isnan(expected)
    close = actual == expected
    return bool((nan | close).all().item())


def run(submission):
    c = Checks()

    def pack_low_nibble_first():
        worked = torch.tensor([1, -2, 7, 0, -8, 3, -1, 4], dtype=torch.int32)
        expected = _ref_pack(worked)
        got, err = _call(submission, "pack_int4", worked)
        if err:
            return False, f"expected {int(expected[0])}, got {err}"
        got_t = torch.as_tensor(got).to(torch.int32).cpu()
        if got_t.shape != expected.shape or int(got_t[0]) != int(expected[0]):
            shown = int(got_t.view(-1)[0]) if got_t.numel() else None
            return False, f"expected {int(expected[0])}, got {shown}"

        short = torch.tensor([[1, -2, 7], [-8, 4, 0]], dtype=torch.int32)
        expected_short = _ref_pack(short)
        got, err = _call(submission, "pack_int4", short)
        if err:
            return False, f"expected {expected_short.tolist()}, got {err}"
        got_t = torch.as_tensor(got).to(torch.int32).cpu()
        if got_t.shape != expected_short.shape or not torch.equal(got_t, expected_short):
            return False, f"expected {expected_short.tolist()}, got {got_t.tolist()}"

        generator = torch.Generator().manual_seed(3)
        odd = torch.randint(-8, 8, (3, 13), generator=generator)
        expected_odd = _ref_pack(odd)
        got, err = _call(submission, "pack_int4", odd)
        if err:
            return False, f"expected {expected_odd[0].tolist()}, got {err}"
        got_t = torch.as_tensor(got).to(torch.int32).cpu()
        if not torch.equal(got_t, expected_odd):
            return False, f"expected {expected_odd[0].tolist()}, got {got_t[0].tolist()}"
        return True, f"expected {int(expected[0])}, got {int(expected[0])}"

    c.check("pack_int4 stores the first value in the low nibble", pack_low_nibble_first)

    def unpack_matches_the_word():
        worked = torch.tensor([1, -2, 7, 0, -8, 3, -1, 4], dtype=torch.int32)
        packed = _ref_pack(worked)
        got, err = _call(submission, "unpack_int4", packed, 8)
        if err:
            return False, f"expected {worked.tolist()}, got {err}"
        worked_got = torch.as_tensor(got).to(torch.int32).cpu()
        if not torch.equal(worked_got, worked):
            return False, f"expected {worked.tolist()}, got {worked_got.tolist()}"

        generator = torch.Generator().manual_seed(5)
        codes = torch.randint(-8, 8, (2, 11), generator=generator)
        got, err = _call(submission, "unpack_int4", _ref_pack(codes), 11)
        if err:
            return False, f"expected {codes[0].tolist()}, got {err}"
        got_t = torch.as_tensor(got).to(torch.int32).cpu()
        if not torch.equal(got_t, codes):
            return False, f"expected {codes[0].tolist()}, got {got_t[0].tolist()}"
        return True, f"expected {worked.tolist()}, got {worked_got.tolist()}"

    c.check("unpack_int4 returns the integers that were packed", unpack_matches_the_word)

    def group_error_and_bytes():
        generator = torch.Generator().manual_seed(2)
        cases = [
            torch.tensor([[0.4, -0.2, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            torch.randn(4, 128, generator=generator),
            torch.randn(3, 10, generator=generator),
        ]
        groups = [4, 128, 4]
        byte_lines = []
        for weight, group in zip(cases, groups):
            got, err = _call(submission, "quantize_groupwise_int4", weight, group)
            n = weight.shape[-1]
            n_groups = (n + group - 1) // group
            expected_scales = _group_scales(weight, group)
            expected_bytes = 0.5 + (2 * n_groups) / n
            if err:
                return False, f"expected {expected_bytes:.6f} bytes per weight, got {err}"
            if not isinstance(got, tuple) or len(got) != 2:
                return False, f"expected codes and scales, got {type(got).__name__}"
            codes, scales = got
            codes_t = torch.as_tensor(codes).cpu()
            scales_t = torch.as_tensor(scales).cpu().to(torch.float64)
            if codes_t.shape != weight.shape:
                return False, f"expected code shape {tuple(weight.shape)}, got {tuple(codes_t.shape)}"
            if scales_t.shape != (*weight.shape[:-1], n_groups):
                return False, (
                    f"expected scale shape {(*weight.shape[:-1], n_groups)}, got {tuple(scales_t.shape)}"
                )
            if int(codes_t.min()) < -7 or int(codes_t.max()) > 7:
                return False, f"expected codes in -7..7, got {int(codes_t.min())}..{int(codes_t.max())}"
            if not torch.allclose(scales_t, expected_scales, rtol=1e-5, atol=1e-6):
                return False, (
                    f"expected scales {expected_scales.reshape(-1)[:4].tolist()}, "
                    f"got {scales_t.reshape(-1)[:4].tolist()}"
                )
            index = torch.arange(n) // group
            restored = codes_t.to(torch.float64) * scales_t[..., index]
            error = (restored - weight.to(torch.float64)).abs()
            limit = 0.5 * scales_t[..., index] * (1 + 1e-4) + 1e-8
            if bool((error > limit).any()):
                worst = (error - 0.5 * scales_t[..., index]).max().item()
                return False, f"expected error within half a step, got slack {worst:.3e}"
            actual_bytes = 0.5 + (2 * scales_t.numel()) / weight.numel()
            if abs(actual_bytes - expected_bytes) > 1e-12:
                return False, f"expected {expected_bytes:.6f} bytes per weight, got {actual_bytes:.6f}"
            byte_lines.append(f"{expected_bytes:.6f}")
        c.metric("compression", 0.515625)
        return True, f"expected bytes per weight {byte_lines}, got {byte_lines}"

    c.check("group-wise int4 stays within half a step and counts the scale", group_error_and_bytes)

    def gptq_beats_round_to_nearest():
        inputs, weight, hessian = _lab_layer()
        rtn = _rtn(weight, 8)
        rtn_error = _output_error(inputs, weight, rtn)
        got, err = _call(submission, "gptq_quantize", weight, hessian, 4, 8)
        if err:
            return False, f"expected margin >= {GPTQ_MARGIN:.2f} below {rtn_error:.6f}, got {err}"
        gptq_error = _output_error(inputs, weight, got)
        margin = rtn_error - gptq_error
        if not math.isfinite(gptq_error):
            return False, f"expected a finite error below {rtn_error:.6f}, got {gptq_error}"
        if gptq_error <= GPTQ_FLOOR or margin < GPTQ_MARGIN:
            return False, (
                f"expected gptq error > {GPTQ_FLOOR:.3f} and margin >= {GPTQ_MARGIN:.2f}, "
                f"got gptq {gptq_error:.6f}, rtn {rtn_error:.6f}, margin {margin:.6f}"
            )
        c.metric("gptq_margin", round(margin, 6))
        return True, (
            f"expected margin >= {GPTQ_MARGIN:.2f}, "
            f"got gptq {gptq_error:.6f}, rtn {rtn_error:.6f}, margin {margin:.6f}"
        )

    c.check("gptq_quantize beats round-to-nearest on correlated inputs", gptq_beats_round_to_nearest)

    def awq_matches_the_grid():
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        torch.manual_seed(4)
        channel = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2, 12.0])
        wide_x = torch.randn(48, 6) * channel
        wide_w = torch.randn(2, 6)
        wide_w[:, -1] *= 0.2
        generator = torch.Generator().manual_seed(1)
        narrow_x = torch.randn(32, 4, generator=generator)
        narrow_x[:, -1] *= 8
        narrow_w = torch.randn(2, 4, generator=generator)
        narrow_w[:, -1] *= 0.25
        cases = [
            ("wide", wide_x, wide_w, 6, 0.75),
            ("narrow", narrow_x, narrow_w, 4, 0.25),
        ]
        lines = []
        for name, inputs, weight, group, locked in cases:
            found = _best_alpha(inputs, weight, group, alphas)
            if found is None or abs(found[1] - locked) > 1e-12:
                actual_alpha = None if found is None else found[1]
                return False, f"expected the brute-force alpha {locked} on {name}, got {actual_alpha}"
            got, err = _call(submission, "awq_scale_search", inputs, weight, group, alphas)
            expected = _awq_base(inputs).pow(locked).float()
            if err:
                return False, f"expected alpha {locked} scale {expected.tolist()}, got {err}"
            got_t = torch.as_tensor(got).detach().cpu().float()
            if got_t.shape != expected.shape or not torch.allclose(got_t, expected, rtol=1e-4, atol=1e-5):
                return False, f"expected alpha {locked} scale {expected.tolist()}, got {got_t.tolist()}"
            lines.append(f"{name} alpha {locked}")
        return True, f"expected {lines}, got {lines}"

    c.check("awq_scale_search matches a brute-force alpha grid", awq_matches_the_grid)

    def kv_axes_and_sensitivity():
        query, key, value = _kv_tensors()
        got, err = _call(submission, "quantize_kv", key, value)
        if err:
            return False, f"expected per-channel keys and per-token values, got {err}"
        if not isinstance(got, tuple) or len(got) != 2:
            return False, f"expected two tensors, got {type(got).__name__}"
        key_hat, value_hat = got
        key_ref = _symmetric(key, 0)
        value_ref = _symmetric(value, 1)
        key_t = torch.as_tensor(key_hat).detach().cpu().float()
        value_t = torch.as_tensor(value_hat).detach().cpu().float()
        key_err = (key_t - key_ref).abs().max().item()
        value_err = (value_t - value_ref).abs().max().item()
        if key_err > 1e-5 or value_err > 1e-5:
            return False, (
                f"expected max abs error <= 1e-5 against per-channel K and per-token V, "
                f"got key {key_err:.3e}, value {value_err:.3e}"
            )
        base = _attention(query, key, value)
        def rel(actual):
            return (actual - base).norm().item() / base.norm().item()
        key_token = rel(_attention(query, _symmetric(key, 1), value))
        key_channel = rel(_attention(query, key_t, value))
        value_token = rel(_attention(query, key, value_t))
        value_channel = rel(_attention(query, key, _symmetric(value, 0)))
        if key_token <= 2 * value_token:
            return False, (
                f"expected per-token key error > 2x per-token value error, "
                f"got key {key_token:.6f}, value {value_token:.6f}"
            )
        if key_channel * 2 >= key_token:
            return False, (
                f"expected per-channel key error < half the per-token key error, "
                f"got per-channel {key_channel:.6f}, per-token {key_token:.6f}"
            )
        if value_channel <= 1.5 * value_token:
            return False, (
                f"expected per-channel value error > 1.5x per-token value error, "
                f"got per-channel {value_channel:.6f}, per-token {value_token:.6f}"
            )
        c.metric("kv_key_over_value", round(key_token / value_token, 4))
        return True, (
            f"expected per-token key {key_token:.6f} > per-token value {value_token:.6f}, "
            f"got per-channel key {key_channel:.6f} and per-channel value {value_channel:.6f}"
        )

    c.check("quantize_kv keeps the key-versus-value gap on the seeded head", kv_axes_and_sensitivity)

    def fp8_rounds_ties_to_even():
        e4_in = torch.tensor(
            [0.0, -0.0, 1.0625, 1.1875, 448.0, 464.0, 480.0, 2.0**-9, 2.0**-10, float("inf"), float("nan")],
            dtype=torch.float32,
        )
        e4_expected = torch.tensor(
            [0.0, -0.0, 1.0, 1.25, 448.0, 448.0, float("nan"), 2.0**-9, 0.0, float("nan"), float("nan")],
            dtype=torch.float32,
        )
        e5_in = torch.tensor(
            [1.125, 57344.0, 60000.0, 61440.0, 2.0**-16, 2.0**-17, float("inf"), float("-inf"), float("nan")],
            dtype=torch.float32,
        )
        e5_expected = torch.tensor(
            [1.0, 57344.0, 57344.0, float("inf"), 2.0**-16, 0.0, float("inf"), float("-inf"), float("nan")],
            dtype=torch.float32,
        )
        got4, err = _call(submission, "fp8_cast", e4_in, "e4m3")
        if err:
            return False, f"expected E4M3 {e4_expected.tolist()}, got {err}"
        if not _same_float(got4, e4_expected):
            return False, f"expected E4M3 {e4_expected.tolist()}, got {torch.as_tensor(got4).float().tolist()}"
        got5, err = _call(submission, "fp8_cast", e5_in, "e5m2")
        if err:
            return False, f"expected E5M2 {e5_expected.tolist()}, got {err}"
        if not _same_float(got5, e5_expected):
            return False, f"expected E5M2 {e5_expected.tolist()}, got {torch.as_tensor(got5).float().tolist()}"
        if torch.signbit(torch.as_tensor(got4).flatten()[1]).item() is not True:
            return False, "expected E4M3 index 1 to stay -0, got +0"
        return True, "expected E4M3 1.0625 -> 1 and 480 -> nan, got the same cast"

    c.check("fp8_cast rounds ties to even for E4M3 and E5M2", fp8_rounds_ties_to_even)

    def triton_matmul():
        if not torch.cuda.is_available():
            return True, "expected skip, got skip (CUDA is not available)"
        shapes = [(2, 8, 8, 8), (1, 7, 3, 4), (3, 20, 10, 8)]
        worst = 0.0
        where = ""
        limit = 1e-4
        try:
            for rows, k, n, group in shapes:
                generator = torch.Generator().manual_seed(11)
                codes = torch.randint(-8, 8, (k, n), generator=generator)
                packed = _ref_pack(codes).cuda()
                n_groups = (k + group - 1) // group
                scales = torch.randn(n_groups, n, generator=generator).cuda()
                inputs = torch.randn(rows, k, generator=generator).cuda()
                weight = _ref_unpack(packed.cpu(), n).to(torch.float64)
                index = torch.arange(k) // group
                dense = weight * scales.detach().cpu().double()[index]
                expected = inputs.detach().cpu().double() @ dense
                got = submission.w4a16_matmul(inputs, packed, scales, group)
                err = (got.detach().cpu().double() - expected).abs().max().item()
                if err > worst:
                    worst = err
                    where = f"m={rows}, k={k}, n={n}"
        except Exception as exc:  # noqa: BLE001
            return False, f"expected max abs error <= {limit:.1e}, got {type(exc).__name__}: {exc}"
        return worst <= limit, f"expected max abs error <= {limit:.1e}, got {worst:.3e} at {where}"

    c.check("w4a16_matmul matches a dequantized matmul when CUDA is available", triton_matmul)
    return c.finish()
