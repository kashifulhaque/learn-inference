"""Numbers for chapter 17b.

Every number is computed by content/labs/17b-disaggregation/solution.py on a
simulated clock, with chapter 17a's step-time floor. Nothing here is an A100 or
a network measurement. Run from the repository root:

    python3 scripts/figures/17b-disaggregated-serving.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOL = ROOT / "content" / "labs" / "17b-disaggregation" / "solution.py"


def load():
    spec = importlib.util.spec_from_file_location("disagg_solution", SOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# One-way rates in GB/s. NVLink and PCIe are chapter 19's. The rest are line
# rates divided by 8: InfiniBand HDR 200 Gbit/s, NDR 400 Gbit/s (Wikipedia's
# InfiniBand table), and Ethernet at 100 and 25 Gbit/s.
LINKS = [
    ("NVLink, one way", 300.0),
    ("InfiniBand NDR 400 Gbit/s", 50.0),
    ("InfiniBand HDR 200 Gbit/s", 25.0),
    ("100 Gbit/s Ethernet", 12.5),
    ("PCIe 4.0 x16, one way", 32.0),
    ("25 Gbit/s Ethernet", 3.125),
]


def main() -> None:
    m = load()
    MiB = 1024 * 1024
    print("== KV bytes, hybrid and dense")
    for tokens in (128, 512, 2048, 8192, 32768, 131072):
        h = m.kv_transfer_bytes("hybrid", tokens)
        d = m.kv_transfer_bytes("dense", tokens)
        print(f"  {tokens:>7} tokens: hybrid {h:>14,} B = {h / MiB:9.2f} MiB; "
              f"dense {d / MiB:10.2f} MiB; state share {m.STATE_BYTES / h:6.1%}")
    print(f"  state alone: {m.STATE_BYTES:,} B = {m.STATE_BYTES / MiB:.4f} MiB; "
          f"per-token KV {m.KV_BYTES_PER_LAYER_TOKEN * m.N_FULL:,} B")
    cross = m.STATE_BYTES / (m.KV_BYTES_PER_LAYER_TOKEN * m.N_FULL)
    print(f"  tokens at which KV equals the state: {cross:.1f}")

    print("== Bulk copy time, ms, and its share of the prefill")
    for tokens in (128, 512, 2048, 8192):
        P = m.prefill_seconds(tokens)
        row = [f"{tokens} tok prefill {P * 1e3:8.2f} ms"]
        for name, g in LINKS:
            X = m.transfer_seconds(m.kv_transfer_bytes("hybrid", tokens), g)
            row.append(f"{g:g}: {X * 1e3:7.2f} ({X / P:6.1%})")
        print("  " + " | ".join(row))

    print("== Least link rate (GB/s)")
    for tokens in (128, 2048, 8192, 32768):
        P = m.prefill_seconds(tokens)
        bulk10 = m.min_link_gbs("hybrid", tokens, 0.1 * P)
        bulk10ms = m.min_link_gbs("hybrid", tokens, 0.010)
        lw5 = m.min_link_gbs("hybrid", tokens, 0.005, P)
        lw0p5 = m.min_link_gbs("hybrid", tokens, 0.0005, P)
        print(f"  {tokens:>6} tok, prefill {P * 1e3:9.2f} ms: bulk 10% of prefill {bulk10:8.3f}; "
              f"bulk 10 ms {bulk10ms:8.3f}; streamed 5 ms {lw5:8.3f}; streamed 0.5 ms {lw0p5:8.3f}")
    print("== Exposed wait with streaming, 8192 tokens, ms")
    P = m.prefill_seconds(8192)
    for name, g in LINKS:
        bulk = m.transfer_seconds(m.kv_transfer_bytes("hybrid", 8192), g)
        lw = m.layerwise_exposed_seconds("hybrid", 8192, P, g)
        print(f"  {name:28} bulk {bulk * 1e3:8.2f}  streamed {lw * 1e3:8.3f}")
    print("== Streaming at 128 tokens, ms")
    P = m.prefill_seconds(128)
    for name, g in LINKS:
        bulk = m.transfer_seconds(m.kv_transfer_bytes("hybrid", 128), g)
        lw = m.layerwise_exposed_seconds("hybrid", 128, P, g)
        print(f"  {name:28} bulk {bulk * 1e3:8.2f}  streamed {lw * 1e3:8.3f}")

    print("== Planted interference trace: 8 decoders plus one 8192-token prompt at t = 2 s")
    decoders = [{"arrival": 0.0, "prompt": 128, "output": 300} for _ in range(8)]
    trace = decoders + [{"arrival": 2.0, "prompt": 8192, "output": 2}]
    print(f"  8192-token prefill alone: {m.prefill_seconds(8192) * 1e3:.3f} ms")
    for chunk in (1_000_000, 2048, 512, 192, 160, 128):
        r = m.simulate(trace, {"mode": "colocated", "gpus": 1, "chunk": chunk, "max_batch": 64})
        print(f"  colocated chunk {chunk:>7}: worst gap {r['p99_worst_gap'] * 1e3:8.2f} ms, "
              f"slowest decoder's mean TPOT {r['p99_tpot'] * 1e3:6.2f} ms, long prompt TTFT {r['p99_ttft']:.3f} s")
    r = m.simulate(trace, {"mode": "disagg", "prefill_gpus": 1, "decode_gpus": 1,
                           "link_gbs": 300, "max_batch": 64})
    print(f"  disaggregated 1+1       : worst gap {r['p99_worst_gap'] * 1e3:8.2f} ms, "
          f"long prompt TTFT {r['p99_ttft']:.3f} s")

    print("== Step time with 8 decodes at context 2048 plus one chunk (ms)")
    base = m.step_time(8, 8 * 2048)
    print(f"  decode only: {base * 1e3:.3f} ms")
    for n in (64, 128, 160, 192, 256, 512, 1024):
        s = m.step_time(8, 8 * 2048, [(2048, n)])
        print(f"  chunk {n:>5}: {s * 1e3:7.2f} ms ({s / base:5.2f}x)")
    ridge = next(n for n in range(1, 4096)
                 if 2 * m.HYBRID_PARAMS * (8 + n) / (m.A100['bf16_tflops'] * 1e12)
                 > m.step_time(8, 8 * 2048) - 2e-3)
    print(f"  compute alone passes the decode step's byte floor at about {ridge} chunk tokens")

    print("== Prefill-heavy load: 8 GPUs, 8192 in, 32 out, 3 req/s, SLO TTFT 1.6 s, worst gap 36 ms")
    slo = {"slo_ttft": 1.6, "slo_tpot": 0.036, "tpot_stat": "max", "max_batch": 32, "link_gbs": 300.0}
    heavy = m.poisson_trace(100, 3.0, 8192, 32, 7)
    for p in range(1, 8):
        r = m.simulate(heavy, dict(slo, mode="disagg", prefill_gpus=p, decode_gpus=8 - p))
        print(f"  {p}:{8 - p} split: goodput {r['goodput']:.3f}/s, met {r['met']}/100, "
              f"p99 TTFT {r['p99_ttft']:.2f} s")
    for chunk in (128, 160, 192, 256, 512, 4096):
        r = m.simulate(heavy, dict(slo, mode="colocated", gpus=8, chunk=chunk))
        print(f"  colocated chunk {chunk:>4}: goodput {r['goodput']:.3f}/s, met {r['met']}/100, "
              f"p99 TTFT {r['p99_ttft']:.2f} s, p99 worst gap {r['p99_worst_gap'] * 1e3:.1f} ms")
    print("  best_split:", m.best_split(8, heavy, slo), " capacity_split:",
          m.capacity_split(8, heavy, 0.036, 32))
    print("  layer-wise streaming at the best split:")
    r = m.simulate(heavy, dict(slo, mode="disagg", prefill_gpus=7, decode_gpus=1, layerwise=True))
    print(f"    goodput {r['goodput']:.3f}/s")

    print("== Decode-heavy load: 8 GPUs, 128 in, 512 out, 8 req/s, SLO TTFT 1.0 s, worst gap 40 ms")
    slo2 = {"slo_ttft": 1.0, "slo_tpot": 0.04, "tpot_stat": "max", "max_batch": 32, "link_gbs": 300.0}
    light = m.poisson_trace(100, 8.0, 128, 512, 11)
    for p in range(1, 8):
        r = m.simulate(light, dict(slo2, mode="disagg", prefill_gpus=p, decode_gpus=8 - p))
        print(f"  {p}:{8 - p} split: goodput {r['goodput']:.3f}/s, met {r['met']}/100")
    for chunk in (128, 160, 192, 256, 4096):
        r = m.simulate(light, dict(slo2, mode="colocated", gpus=8, chunk=chunk))
        print(f"  colocated chunk {chunk:>4}: goodput {r['goodput']:.3f}/s, met {r['met']}/100")
    print("  best_split:", m.best_split(8, light, slo2))

    print("== Short prompts over a slow link: 128 in, 16 out, 2 req/s, seed 5")
    short = m.poisson_trace(40, 2.0, 128, 16, 5)
    for name, g in LINKS:
        r = m.simulate(short, {"mode": "disagg", "prefill_gpus": 1, "decode_gpus": 1,
                               "link_gbs": g, "max_batch": 32})
        print(f"  {name:28} median TTFT {r['p50_ttft'] * 1e3:7.2f} ms")
    r = m.simulate(short, {"mode": "colocated", "gpus": 2, "chunk": 160, "max_batch": 32})
    print(f"  colocated 2 GPUs, chunk 160    median TTFT {r['p50_ttft'] * 1e3:7.2f} ms")
    print(f"  128-token prefill alone: {m.prefill_seconds(128) * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
