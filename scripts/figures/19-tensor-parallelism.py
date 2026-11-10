"""Numbers for the latency-bound all-reduce section of chapter 19.

Every time is the alpha-beta model in
content/labs/19-tensor-parallel/solution.py (`all_reduce_time`,
`step_all_reduce_time`). Nothing here is an A100 measurement. The hop latency
is an assumption, and the bandwidth is NVIDIA's NVLink figure for one
direction.

Run from the repository root:

    python3 scripts/figures/19-tensor-parallelism.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOL = ROOT / "content" / "labs" / "19-tensor-parallel" / "solution.py"

HIDDEN = 5120
LAYERS = 64
BYTES = 2
ALPHA = 4e-6  # assumed latency of one ring hop, seconds
NVLINK_ONE_WAY = 300e9  # 12 links x 25 GB/s per direction
PCIE_ONE_WAY = 32e9  # PCIe Gen4 x16, 64 GB/s both directions combined
PREFILL_TOKENS = 4096
WEIGHT_READ_MS = 53.8e9 / 1275e9 * 1e3  # one GPU, from chapter 14a


def load():
    spec = importlib.util.spec_from_file_location("tp_solution", SOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    m = load()
    decode_msg = HIDDEN * BYTES
    prefill_msg = PREFILL_TOKENS * HIDDEN * BYTES
    print(f"decode message {decode_msg} B, prefill message {prefill_msg} B")

    print("\nOne all-reduce, NVLink one way, alpha 4 us")
    for n in (2, 4, 8):
        for name, size in (("decode", decode_msg), ("prefill", prefill_msg)):
            lat = 2 * (n - 1) * ALPHA
            bw = 2 * (n - 1) / n * size / NVLINK_ONE_WAY
            total = m.all_reduce_time(size, n, ALPHA, NVLINK_ONE_WAY)
            print(f"  N={n} {name:7s} latency {lat*1e6:8.2f} us  "
                  f"bandwidth {bw*1e6:9.3f} us  total {total*1e6:9.2f} us  "
                  f"latency share {lat/total:6.1%}")

    print("\nCrossover message size S* = N * alpha * bandwidth")
    for n in (2, 4, 8):
        s_star = n * ALPHA * NVLINK_ONE_WAY
        print(f"  N={n}: {s_star/1e6:.2f} MB = {s_star/decode_msg:.0f} tokens")

    print("\nPer forward pass, 128 all-reduces, decode message")
    for n in (2, 4, 8):
        t = m.step_all_reduce_time(LAYERS, decode_msg, n, ALPHA, NVLINK_ONE_WAY)
        per_layer = 2 * m.all_reduce_time(decode_msg, n, ALPHA, NVLINK_ONE_WAY)
        step = WEIGHT_READ_MS / n + t * 1e3
        print(f"  N={n}: per layer {per_layer*1e6:6.1f} us, "
              f"per token {t*1e3:6.3f} ms, step {step:6.2f} ms, "
              f"speedup {WEIGHT_READ_MS/step:5.2f}")

    print("\nPrefill, 4096 tokens, N=2, whole forward pass")
    tot = m.step_all_reduce_time(LAYERS, prefill_msg, 2, ALPHA, NVLINK_ONE_WAY)
    pcie = m.step_all_reduce_time(LAYERS, prefill_msg, 2, ALPHA, PCIE_ONE_WAY)
    comp = 2 * 26.9e9 * PREFILL_TOKENS / (2 * 312e12)
    print(f"  NVLink {tot*1e3:.2f} ms, PCIe {pcie*1e3:.2f} ms, "
          f"compute {comp*1e3:.1f} ms")
    print(f"  NVLink share {tot/comp:.1%}, PCIe share {pcie/comp:.1%}")


if __name__ == "__main__":
    main()
