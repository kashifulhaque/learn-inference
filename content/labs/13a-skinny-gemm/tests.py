"""Checks for decode-GEMM traffic and a W4A16 unpack.

The first check calls only `gemm_flops`. Later checks call `gemm_bytes`,
`gemm_roofline`, and `w4a16_gemv`. `skinny_gemm` runs only when CUDA is
available.

A plausible wrong implementation drops the output matrix from the byte
count: 2*(m*k + k*n) instead of 2*(m*k + k*n + m*n). At m=1, k=5120,
n=17408 that omits 34,816 bytes and the bound stays "memory", so a
bound check does not catch it. At m=4096 it leaves 220,200,960 bytes
instead of 362,807,296. The byte check compares those integers.

Classifying m=1 as compute bound fails the roofline check. The
intensity there is 178257920/178302976, about 0.9997, and the rated
ridge is 312e12/1935e9, about 161.2.
"""

from __future__ import annotations

import torch

from lab_common import Checks

# Chapter 10 rated peaks. The ridge comparison is integer:
# flops * 1935 ? bytes * 312_000.
K = 5120
N = 17408
GROUP = 128
PEAK_FLOPS = 312 * 10**12
PEAK_BYTES = 1935 * 10**9


def _weight(k: int, n: int, storage: str) -> int:
    if storage == "bf16":
        return 2 * k * n
    payload = (k * n + 1) // 2
    scales = ((k + GROUP - 1) // GROUP) * n * 2
    return payload + scales


def _flops(m: int, k: int, n: int) -> int:
    return 2 * m * k * n


def _bytes(m: int, k: int, n: int, storage: str) -> int:
    return 2 * m * k + _weight(k, n, storage) + 2 * m * n


def _bound(flops: int, nbytes: int) -> str:
    left = flops * 1935
    right = nbytes * 312_000
    if left < right:
        return "memory"
    if left > right:
        return "compute"
    return "ridge"


def _call(submission, name, *args, **kwargs):
    """Call `submission.name`. A missing function is a failed check, not a crash."""
    try:
        fn = getattr(submission, name)
        return fn(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _pack(q: torch.Tensor) -> torch.Tensor:
    k, n = q.shape
    packed = torch.zeros(k, (n + 1) // 2, dtype=torch.uint8)
    shifted = (q + 8).to(torch.uint8)
    for col in range(n):
        nib = shifted[:, col]
        if col % 2 == 0:
            packed[:, col // 2] |= nib
        else:
            packed[:, col // 2] |= nib << 4
    return packed


def _ref_gemv(x, packed, scales, group: int) -> torch.Tensor:
    """Independent unpack, in float64, so it does not share a vectorized bug."""
    x64 = x.detach().to(torch.float64).cpu()
    packed = packed.detach().cpu()
    scales = scales.detach().to(torch.float64).cpu()
    m, k = x64.shape
    n = scales.shape[1]
    out = torch.zeros(m, n, dtype=torch.float64)
    for row in range(m):
        for col in range(n):
            acc = 0.0
            for t in range(k):
                byte = int(packed[t, col // 2])
                nib = byte & 0x0F if col % 2 == 0 else (byte >> 4) & 0x0F
                acc += float(x64[row, t]) * (nib - 8) * float(scales[t // group, col])
            out[row, col] = acc
    return out


def run(submission):
    c = Checks()

    def flops_one_function():
        expected = 2 * 2 * 3 * 5
        got, err = _call(submission, "gemm_flops", 2, 3, 5)
        if err:
            return False, f"expected {expected}, got {err}"
        return got == expected, f"expected {expected}, got {got}"

    c.check("gemm_flops counts two FLOPs per product", flops_one_function)

    def bytes_keep_output():
        # Omitting m*n*2 is the bug named in the module docstring.
        shapes = [(1, K, N), (4096, K, N), (1, 8, 4)]
        for m, k, n in shapes:
            expected = _bytes(m, k, n, "bf16")
            dropped = 2 * (m * k + k * n)
            if dropped == expected:
                return False, (
                    f"expected dropping C to change the byte count at "
                    f"m={m}, k={k}, n={n}, and both were {expected}"
                )
            got, err = _call(submission, "gemm_bytes", m, k, n, "bf16")
            if err:
                return False, f"expected {expected} bytes, got {err}"
            if got != expected:
                return False, (
                    f"expected {expected} bytes, got {got} "
                    f"(dropping the output gives {dropped})"
                )
        shown = _bytes(4096, K, N, "bf16")
        return True, f"expected {shown} bytes at m=4096, got {shown}"

    c.check("gemm_bytes counts the output matrix", bytes_keep_output)

    def bytes_w4_scales():
        cases = [(1, K, N, _bytes(1, K, N, "w4a16")), (1, 100, 16, _bytes(1, 100, 16, "w4a16"))]
        for m, k, n, expected in cases:
            got, err = _call(submission, "gemm_bytes", m, k, n, "w4a16")
            if err:
                return False, f"expected {expected} bytes, got {err}"
            if got != expected:
                return False, f"expected {expected} bytes at k={k}, n={n}, got {got}"
        no_scale = 2 * 1 * 100 + (100 * 16) // 2 + 2 * 1 * 16
        if cases[1][3] == no_scale:
            return False, f"expected {cases[1][3]} bytes, got a count with no scale bytes ({no_scale})"
        return True, f"expected {cases[0][3]} W4 bytes at m=1, got {cases[0][3]}"

    c.check("w4a16 bytes include a scale on a short last group", bytes_w4_scales)

    def bounds_bf16():
        rows = []
        for m in (1, 8, 64, 4096):
            flops = _flops(m, K, N)
            nbytes = _bytes(m, K, N, "bf16")
            expected_bound = _bound(flops, nbytes)
            expected_i = flops / nbytes
            got, err = _call(submission, "gemm_roofline", m, K, N, "bf16")
            if err:
                return False, f"expected bound {expected_bound} at m={m}, got {err}"
            if not isinstance(got, dict):
                return False, f"expected a dict at m={m}, got {type(got).__name__}"
            actual_bound = got.get("bound")
            actual_i = got.get("intensity")
            actual_b = got.get("bytes")
            if actual_b != nbytes or actual_bound != expected_bound:
                return False, (
                    f"expected bound {expected_bound} and {nbytes} bytes at m={m}, "
                    f"got bound {actual_bound} and {actual_b} bytes"
                )
            if actual_i is None or abs(float(actual_i) - expected_i) > 1e-9:
                return False, f"expected intensity {expected_i:.6f} at m={m}, got {actual_i}"
            rows.append(f"m={m} {expected_bound} {expected_i:.4f}")
        intensity_m1 = _flops(1, K, N) / _bytes(1, K, N, "bf16")
        c.metric("intensity_m1", round(intensity_m1, 6))
        c.metric("bound_m1", _bound(_flops(1, K, N), _bytes(1, K, N, "bf16")))
        return True, f"expected {rows}, got {rows}"

    c.check("bfloat16 bound at m = 1, 8, 64, and 4096", bounds_bf16)

    def bounds_w4_and_crossing():
        # m=64 is memory bound in bf16 and compute bound in w4 at the rated ridge.
        # m=168/169 and m=43/44 are the integers on either side of the crossing.
        cases = [
            (1, "w4a16"),
            (8, "w4a16"),
            (64, "w4a16"),
            (4096, "w4a16"),
            (43, "w4a16"),
            (44, "w4a16"),
            (168, "bf16"),
            (169, "bf16"),
        ]
        lines = []
        for m, storage in cases:
            flops = _flops(m, K, N)
            nbytes = _bytes(m, K, N, storage)
            expected_bound = _bound(flops, nbytes)
            got, err = _call(submission, "gemm_roofline", m, K, N, storage)
            if err:
                return False, f"expected {expected_bound} at m={m} {storage}, got {err}"
            actual = got.get("bound") if isinstance(got, dict) else None
            if actual != expected_bound:
                return False, (
                    f"expected bound {expected_bound} at m={m} {storage}, got {actual}"
                )
            lines.append(f"m={m} {storage} {expected_bound}")
        bf = _bytes(1, K, N, "bf16") / PEAK_BYTES
        w4 = _bytes(1, K, N, "w4a16") / PEAK_BYTES
        got, err = _call(submission, "gemm_roofline", 1, K, N, "w4a16")
        if err or not isinstance(got, dict):
            return False, f"expected memory_s {w4:.6e}, got {err or type(got).__name__}"
        actual_s = float(got.get("memory_s"))
        if abs(actual_s - w4) > 1e-12 or not (actual_s < bf):
            return False, f"expected memory_s {w4:.6e} and below {bf:.6e}, got {actual_s:.6e}"
        c.metric("w4_intensity_m1", round(_flops(1, K, N) / _bytes(1, K, N, "w4a16"), 6))
        return True, f"expected {lines}, and memory_s {w4:.6e}, got {lines} and {actual_s:.6e}"

    c.check("w4a16 stays memory bound at m=1 and crosses earlier than bf16", bounds_w4_and_crossing)

    def unpack_matches():
        q = torch.tensor(
            [
                [1, -2, 7],
                [0, -8, 3],
                [-1, 4, -4],
                [2, 5, -7],
            ],
            dtype=torch.int8,
        )
        scales = torch.tensor(
            [[0.5, 0.25, 1.0], [0.5, 2.0, 0.25]],
            dtype=torch.float32,
        )
        x = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
        packed = _pack(q)
        hand = torch.tensor([[1.0, 15.5, 7.25], [0.5, -0.5, 7.0]], dtype=torch.float64)
        ref = _ref_gemv(x, packed, scales, 2)
        if not torch.allclose(ref, hand, atol=1e-9):
            return False, f"expected the worked product {hand.tolist()}, reference got {ref.tolist()}"
        got, err = _call(submission, "w4a16_gemv", x, packed, scales, group_size=2)
        if err:
            return False, f"expected {hand.tolist()}, got {err}"
        got_t = torch.as_tensor(got).to(torch.float64).cpu()
        if got_t.shape != hand.shape:
            return False, f"expected shape {tuple(hand.shape)}, got {tuple(got_t.shape)}"
        err_v = (got_t - hand).abs().max().item()
        if err_v > 1e-5:
            return False, f"expected {hand.tolist()}, got {got_t.tolist()}"

        torch.manual_seed(13)
        m, k, n, group = 3, 7, 5, 4
        q2 = torch.randint(-8, 8, (k, n), dtype=torch.int8)
        scales2 = torch.randn((k + group - 1) // group, n)
        x2 = torch.randn(m, k)
        packed2 = _pack(q2)
        ref2 = _ref_gemv(x2, packed2, scales2, group)
        got2, err = _call(submission, "w4a16_gemv", x2, packed2, scales2, group_size=group)
        if err:
            return False, f"expected {ref2[0].tolist()}, got {err}"
        got2_t = torch.as_tensor(got2).to(torch.float64).cpu()
        if got2_t.shape != ref2.shape:
            return False, f"expected shape {tuple(ref2.shape)}, got {tuple(got2_t.shape)}"
        err2 = (got2_t - ref2).abs().max().item()
        if err2 > 1e-5:
            return False, (
                f"expected {ref2[0].tolist()}, got {got2_t[0].tolist()} "
                f"(max abs error {err2:.3e})"
            )
        return True, f"expected {hand.tolist()}, got {got_t.tolist()}"

    c.check("w4a16_gemv unpacks nibbles and applies the group scale", unpack_matches)

    def triton_kernel():
        if not torch.cuda.is_available():
            return True, "expected skip, got skip (CUDA is not available)"
        torch.manual_seed(13)
        shapes = [(1, 100, 13), (7, 17, 64), (8, 1, 3), (3, 128, 5)]
        worst = 0.0
        where = ""
        limit = 1e-4
        try:
            for m, k, n in shapes:
                a = torch.randn(m, k, device="cuda")
                b = torch.randn(k, n, device="cuda")
                expected = a @ b
                got = submission.skinny_gemm(a, b)
                if got is None:
                    return False, f"expected a tensor at m={m}, k={k}, n={n}, got None"
                err = (got.float() - expected.float()).abs().max().item()
                if err > worst:
                    worst = err
                    where = f"m={m}, k={k}, n={n}"
        except Exception as exc:  # noqa: BLE001
            return False, f"expected max abs error <= {limit:.1e}, got {type(exc).__name__}: {exc}"
        return worst <= limit, f"expected max abs error <= {limit:.1e}, got {worst:.3e} at {where}"

    c.check("skinny_gemm matches a matmul when CUDA is available", triton_kernel)
    return c.finish()
