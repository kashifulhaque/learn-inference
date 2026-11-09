"""Arithmetic intensity of one decode GEMM, bfloat16 and W4A16.

The layer is the up projection from chapter 10: K = 5120, N = 17408.
Peaks are that chapter's rated pair, 312 TFLOP/s and 1935 GB/s. The
ridge is 312/1.935. Nothing here is an A100 kernel measurement.

A bfloat16 product moves 2 bytes for A, for B, and for C. A W4A16
weight is a 4-bit payload plus one bfloat16 scale per group of 128
along K, which is 33/64 bytes per weight when K is divisible by 128.
Activations stay bfloat16. FLOPs are 2*M*K*N. Dequantization is not
in that count.
"""

from __future__ import annotations

K = 5120
N = 17408
GROUP = 128
# Chapter 10, rated A100 80GB PCIe. Not a new peak.
PEAK_FLOPS = 312 * 10**12
PEAK_BYTES = 1935 * 10**9
RIDGE = PEAK_FLOPS / PEAK_BYTES

BATCHES = (1, 8, 64, 4096)


def weight_bytes(k: int, n: int, storage: str) -> int:
    if storage == "bf16":
        return 2 * k * n
    if storage == "w4a16":
        payload = (k * n + 1) // 2
        scales = ((k + GROUP - 1) // GROUP) * n * 2
        return payload + scales
    raise ValueError(storage)


def gemm_flops(m: int, k: int, n: int) -> int:
    return 2 * m * k * n


def gemm_bytes(m: int, k: int, n: int, storage: str) -> int:
    return 2 * m * k + weight_bytes(k, n, storage) + 2 * m * n


def intensity(m: int, storage: str, k: int = K, n: int = N) -> float:
    return gemm_flops(m, k, n) / gemm_bytes(m, k, n, storage)


def bound(m: int, storage: str, k: int = K, n: int = N) -> str:
    flops = gemm_flops(m, k, n)
    nbytes = gemm_bytes(m, k, n, storage)
    # flops / nbytes ? 312e12 / 1935e9, compared in integers.
    left = flops * 1935
    right = nbytes * 312_000
    if left < right:
        return "memory"
    if left > right:
        return "compute"
    return "ridge"


def main() -> None:
    print(f"ridge {RIDGE:.12f}")
    print(f"weight bf16 {weight_bytes(K, N, 'bf16')}")
    print(f"weight w4 {weight_bytes(K, N, 'w4a16')}")
    print(
        "M storage bytes flops intensity bound "
        "memory_us compute_us"
    )
    for storage in ("bf16", "w4a16"):
        for m in BATCHES:
            f = gemm_flops(m, K, N)
            b = gemm_bytes(m, K, N, storage)
            print(
                f"{m} {storage} {b} {f} {intensity(m, storage):.6f} "
                f"{bound(m, storage)} {b / PEAK_BYTES * 1e6:.4f} "
                f"{f / PEAK_FLOPS * 1e6:.4f}"
            )
    for storage in ("bf16", "w4a16"):
        previous = bound(1, storage)
        for m in range(2, 4097):
            kind = bound(m, storage)
            if kind != previous:
                print(
                    f"cross {storage} {m - 1} {previous} "
                    f"{intensity(m - 1, storage):.6f} -> {m} {kind} "
                    f"{intensity(m, storage):.6f}"
                )
                previous = kind
                break
    # Dropping C. The M=1 gap is small and the bound does not flip.
    full = gemm_bytes(4096, K, N, "bf16")
    dropped = 2 * (4096 * K + K * N)
    print(f"drop_output m4096 full {full} dropped {dropped}")
    full1 = gemm_bytes(1, K, N, "bf16")
    dropped1 = 2 * (1 * K + K * N)
    print(f"drop_output m1 full {full1} dropped {dropped1}")
    # K not divisible by the group.
    odd = gemm_bytes(1, 100, 16, "w4a16")
    print(f"w4 k100 n16 m1 {odd}")
    # Column tiles of 128 on this N, against 108 SMs.
    print(f"column_tiles {N // 128}")


if __name__ == "__main__":
    main()
