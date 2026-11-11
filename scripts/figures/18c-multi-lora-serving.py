"""Numbers for chapter 18c.

Every figure here is arithmetic on the model's config or the output of
content/labs/18c-multi-lora/solution.py on the seeded trace in tests.py.
Nothing is an A100 measurement. PCIe is 32 GB/s one way (chapter 17b).

Run from the repository root:

    python3 scripts/figures/18c-multi-lora-serving.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "content" / "labs" / "18c-multi-lora"
sys.path.insert(0, str(ROOT / "content" / "labs"))

GIB = 2**30
MLP_WEIGHT_BYTES = 2 * 64 * 3 * 5120 * 17408  # chapter 02: 17,112,760,320 values, 2 bytes each
DECODE_FLOOR_MS = 42.0  # chapter 00: 53.8 GB at the measured 1275 GB/s copy


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    sol = load("lora_solution", LAB / "solution.py")
    tests = load("lora_tests", LAB / "tests.py")

    print("Up projection (5120 x 17408), one token")
    for r in (8, 16, 64):
        cost = sol.lora_cost(5120, 17408, r)
        print(
            f"  rank {r:>2}: base {cost['base_flops']:,} FLOPs, LoRA {cost['lora_flops']:,} FLOPs "
            f"({100 * cost['flops_ratio']:.4f}%), LoRA weights {cost['lora_bytes']:,} bytes "
            f"({100 * cost['bytes_ratio']:.4f}% of {cost['base_bytes']:,})"
        )

    print("\nMLP adapter (gate, up, down in 64 layers)")
    for r in (8, 16, 64):
        n = sol.adapter_bytes(r)
        ms = sol.adapter_load_ms(n)
        fit = (4 * GIB) // n
        breakeven = MLP_WEIGHT_BYTES / n
        print(
            f"  rank {r:>2}: {n:,} bytes ({n / 2**20:.1f} MiB), {ms:.3f} ms over PCIe "
            f"({ms / DECODE_FLOOR_MS * 100:.1f}% of a {DECODE_FLOOR_MS:.0f} ms step), "
            f"{fit} fit in 4 GiB, {breakeven:.1f} distinct adapters read as many bytes as the MLP"
        )
    print(f"  MLP weight bytes (bfloat16): {MLP_WEIGHT_BYTES:,}")
    n16 = sol.adapter_bytes(16)
    for m in (1, 8, 64):
        print(f"  {m:>2} distinct rank-16 adapters read {m * n16:,} bytes = {100 * m * n16 / MLP_WEIGHT_BYTES:.1f}% of the MLP bytes")

    trace, sizes, ranks = tests.make_trace()
    print(f"\nSeeded trace: {len(trace)} batches, 48 adapters, ranks 8/16/64 counts "
          f"{ranks.count(8)}/{ranks.count(16)}/{ranks.count(64)}")
    for cap in (2, 3, 4, 6, 8):
        row = []
        for policy in ("fifo", "lru", "freq"):
            out = sol.simulate_adapter_pool(trace, sizes, cap * GIB, policy)
            row.append((policy, out))
        text = ", ".join(f"{p} {o['misses']} misses {o['load_ms']:.1f} ms" for p, o in row)
        print(f"  capacity {cap} GiB: {text}")
    out = {p: sol.simulate_adapter_pool(trace, sizes, 4 * GIB, p) for p in ("fifo", "lru", "freq")}
    best = min(out["fifo"]["load_ms"], out["lru"]["load_ms"])
    print(f"  at 4 GiB freq / best plain = {out['freq']['load_ms'] / best:.3f}; hits fifo/lru/freq "
          f"{out['fifo']['hits']}/{out['lru']['hits']}/{out['freq']['hits']}")


if __name__ == "__main__":
    main()
