"""Numbers for chapter 19b.

Every value comes from content/labs/19b-pipeline-and-ring/solution.py. Nothing
here is a multi-GPU measurement. The A100 figures are the course's: 1275 GB/s
copy, 300 GB/s NVLink one way, 32 GB/s PCIe one way, 312 TFLOP/s peak.

Run from the repository root:

    python3 scripts/figures/19b-pipeline-and-context-parallelism.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SOL = ROOT / "content" / "labs" / "19b-pipeline-and-ring" / "solution.py"


def load():
    spec = importlib.util.spec_from_file_location("ring_solution", SOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    m = load()

    print("== Pipeline Gantt tables (stage rows, -1 is idle) ==")
    for p, mb in [(4, 4)]:
        for row in m.pipeline_schedule(p, mb):
            print(row)
    for p, mb in [(2, 1), (2, 8), (4, 1), (4, 4), (4, 16), (8, 32)]:
        got = m.bubble_fraction(m.pipeline_schedule(p, mb))
        print(f"p={p} m={mb}: bubble {got:.4f} formula {m.bubble_formula(p, mb):.4f}")
    for v in (1, 2, 4):
        print(f"p=4 m=16 interleaved v={v}: {m.bubble_formula(4, 16, v):.4f}")

    print("\n== Decode through a pipeline ==")
    for p in (2, 4):
        stage = m.WEIGHT_BYTES / p / m.COPY_BPS
        hop = m.ALPHA + m.HIDDEN * 2 / m.NVLINK_BPS
        print(f"p={p}: stage {stage * 1e3:.3f} ms, hop {hop * 1e6:.3f} us")
        for k in (1, 2, 4):
            d = m.decode_pipeline(p, k, stage, hop)
            print(
                f"  in flight {k}: utilization {d['utilization']:.2f}, inter-token "
                f"{d['latency'] * 1e3:.3f} ms, trips/s {d['rate']:.2f}"
            )
    print("Fixed batch split into micro-batches (each stage re-reads its weights per micro-batch):")
    for p in (4,):
        stage = m.WEIGHT_BYTES / p / m.COPY_BPS
        for k in (1, 2, 4, 8):
            slots = len(m.pipeline_schedule(p, k)[0])
            print(f"  p={p} m={k}: {slots} slots, step {slots * stage * 1e3:.2f} ms")

    print("\n== Ring attention, 64 tokens, 4 ranks, 2 heads, dim 16, seed 19 ==")
    g = torch.Generator().manual_seed(19)
    q, k, v = [torch.randn(2, 64, 16, generator=g) for _ in range(3)]
    pos = torch.arange(64)
    full, _ = m.block_attention(q, k, v, pos, pos)
    for kind in ("contiguous", "striped", "zigzag"):
        out, work = m.ring_attention(q, k, v, 4, kind)
        b = m.work_balance(work)
        print(
            f"{kind}: error {(out - full).abs().max().item():.2e}, totals {b['totals']}, "
            f"imbalance {b['imbalance']:.4f}, critical {b['critical']}, efficiency {b['efficiency']:.4f}"
        )
        print(f"  work {work}")
    for world in (2, 4, 8):
        _, work = m.ring_attention(q, k, v, world, "striped")
        b = m.work_balance(work)
        _, wc = m.ring_attention(q, k, v, world, "contiguous")
        bc = m.work_balance(wc)
        print(
            f"world {world}: contiguous imbalance {bc['imbalance']:.4f} efficiency {bc['efficiency']:.4f}; "
            f"striped imbalance {b['imbalance']:.4f} efficiency {b['efficiency']:.4f}"
        )

    print("\n== Memory per rank, one layer, 1,048,576 tokens ==")
    for world in (1, 2, 4, 8):
        r = m.memory_per_rank(2**20, world)
        print(f"world {world}: cache {r['cache'] / 2**30:.3f} GiB, peak {r['peak'] / 2**30:.3f} GiB, "
              f"cache ratio {r['cache_ratio']}, peak ratio {r['peak_ratio']}")
    print(f"KV bytes per token per layer {m.KV_BYTES_PER_TOKEN_LAYER}, per token {m.KV_BYTES_PER_TOKEN}")
    print(f"KV at 2^20 tokens: {2**20 * m.KV_BYTES_PER_TOKEN / 2**30:.1f} GiB; usable budget {m.USABLE_GIB:.2f} GiB")
    print(f"min shard tokens: NVLink {m.min_shard_tokens(m.NVLINK_BPS):.2f}, PCIe {m.min_shard_tokens(m.PCIE_BPS):.0f}")

    print("\n== One request at 1,048,576 tokens on 4 GPUs ==")
    print(f"prefill FLOPs {m.prefill_flops(2**20):.4e}; dense {2 * m.PARAMS * 2**20:.4e}")
    table = m.compare_strategies(2**20, 4)
    for name, r in table.items():
        print(
            f"{name}: weights {r['weights_gib']:.2f} GiB, cache {r['kv_gib']:.2f} GiB, fits {r['fits']}, "
            f"decode {r['decode_ms']:.2f} ms, comm {r['prefill_comm_gb']:.2f} GB "
            f"({r['prefill_comm_s']:.3f} s NVLink, {r['prefill_comm_gb'] * 1e9 / m.PCIE_BPS:.3f} s PCIe), "
            f"compute {r['prefill_compute_s']:.1f} s"
        )
    print(f"single-GPU decode floor {table['dp']['decode_ms']:.2f} ms; TP all-reduce part "
          f"{2 * m.LAYERS * (2 * 3 * m.ALPHA + 1.5 * m.HIDDEN * 2 / m.NVLINK_BPS) * 1e3:.3f} ms")
    # Ulysses: four all-to-alls per layer (Q, K, V in, O out); each rank sends
    # (P - 1) / P of its local shard. Needs KV heads divisible by P.
    world, tokens = 4, 2**20
    local = tokens // world
    q_b = local * m.Q_HEADS * m.HEAD_DIM * 2
    kv_b = local * m.KV_BYTES_PER_TOKEN_LAYER
    ulysses = (2 * q_b + kv_b) * (world - 1) / world * m.ATTN_LAYERS
    ring = (world - 1) * m.ATTN_LAYERS * local * m.KV_BYTES_PER_TOKEN_LAYER
    print(f"Ulysses per rank {ulysses / 1e9:.2f} GB, ring {ring / 1e9:.2f} GB (Q and O {q_b / 1e9:.2f} GB each, K and V {kv_b / 1e9:.2f} GB)")
    for world in (2, 4, 8, 16):
        print(f"  CP shard at {world} ranks for 2^20 tokens: {2**20 // world} tokens, above NVLink minimum: "
              f"{2**20 // world > m.min_shard_tokens(m.NVLINK_BPS)}")


if __name__ == "__main__":
    main()
