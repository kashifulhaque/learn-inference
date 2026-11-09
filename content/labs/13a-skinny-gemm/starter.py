"""Lab 13a — a skinny GEMM for decode, and a W4A16 unpack.

A is (m, k), B is (k, n), C is (m, n). During decode, m is the batch.
The byte count includes C. The roofline uses the rated peaks from
chapter 10: 312 TFLOP/s and 1935 GB/s.

`storage` is "bf16" or "w4a16". W4A16 keeps activations in bfloat16
and packs weights as signed int4 with one bfloat16 scale per 128 rows
of K. There is no zero point.
"""

from __future__ import annotations


def gemm_flops(m: int, k: int, n: int) -> int:
    """FLOPs in C = A @ B. Count one multiply and one add per product."""
    raise NotImplementedError("gemm_flops")


def gemm_bytes(m: int, k: int, n: int, storage: str = "bf16") -> int:
    """Bytes moved for A, the weight, and C.

    Bfloat16 elements are 2 bytes. For "w4a16", the weight is a 4-bit
    payload, two values per byte, plus 2 bytes of scale for every group
    of 128 rows of K. Round the group count up.
    """
    raise NotImplementedError("gemm_bytes")


def gemm_roofline(m: int, k: int, n: int, storage: str = "bf16") -> dict:
    """Return flops, bytes, intensity, bound, memory_s, and compute_s.

    `intensity` is flops divided by bytes. `bound` is "memory" when that
    intensity is under 312e12/1935e9, "compute" when it is over, and
    "ridge" when the two sides match. Times are seconds at those peaks.
    """
    raise NotImplementedError("gemm_roofline")


def w4a16_gemv(x, packed, scales, group_size: int = 128):
    """Return x @ W for a packed int4 weight.

    `packed` is uint8 with shape (k, ceil(n / 2)). The low nibble of
    column j is output column 2j, and the high nibble is column 2j+1.
    A nibble stores `value + 8`, so the signed weight is `nibble - 8`
    and lies in -8..7. An odd n leaves the last high nibble unused.

    `scales` has shape (ceil(k / group_size), n). Row t of the weight
    uses scale row `t // group_size`. Multiply, then the matmul.
    """
    raise NotImplementedError("w4a16_gemv")


def skinny_gemm(a, b):
    """Return a @ b for a small m. This runs on CUDA."""
    raise NotImplementedError("skinny_gemm")
