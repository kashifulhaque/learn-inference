"""Numbers for chapter 17a.

Every printed time is the floor from content/labs/17a-performance-model/solution.py.
Nothing here is an A100 measurement. The fit rows are a synthetic trace with
planted constants and seed-17 noise; a device trace would replace them.

Run from the repository root:

    python3 scripts/figures/17a-performance-model.py
"""

from __future__ import annotations

import importlib.util
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# One-way rates in GB/s (chapter 19): 300 is half of the 600 GB/s NVLink rating, 32 is half of 64 GB/s PCIe.
LINKS = ((300.0, "nvlink"), (32.0, "pcie"))
SOL = ROOT / "content" / "labs" / "17a-performance-model" / "solution.py"


def load():
    spec = importlib.util.spec_from_file_location("perf_solution", SOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def a100(m):
    return {
        "bf16_tflops": 312.0,
        "hbm_gbs": 1935.0,
        "launch_us": 5.0,
        "launches": 400,
        "allreduce_us": 8.0,
        "link_gbs": 300.0,
        "memory_bytes": 80 * 1024**3,
        "tp": 1,
    }


def gauss(rng):
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def crossing(m, kind, ctx):
    """Real batch where compute time equals memory time, at degree 1.

    Returns None when the asymptotic intensity is at or below the ridge, so
    no batch crosses.
    """
    hw = a100(m)
    flops_1 = m.decode_flops(kind, 1, ctx)
    # Traffic = fixed weights + per-sequence * batch.
    traffic_1 = m.decode_traffic_bytes(kind, 1, ctx, 1)
    traffic_2 = m.decode_traffic_bytes(kind, 2, ctx, 1)
    weight = m._weight_bytes(kind)
    per = traffic_2 - traffic_1
    # traffic_1 = weight + per, so per = traffic_1 - weight. Check.
    c = hw["bf16_tflops"] * 1e12
    beta = hw["hbm_gbs"] * 1e9
    denom = flops_1 * beta - per * c
    if denom <= 0:
        return None, flops_1 / per
    return (weight * c) / denom, flops_1 / per


def max_fit(m, kind, ctx, tp):
    hw = a100(m)
    weight = m._weight_bytes(kind)
    per = m.KV_BYTES_PER_LAYER * m._full_layers(kind) * ctx
    if kind == "hybrid":
        per += m.STATE_BYTES
    room = hw["memory_bytes"] * tp - weight
    if room < per:
        return 0
    return int(room // per)


def main() -> None:
    m = load()
    assert m.HYBRID_PARAMS == 26_895_979_520, m.HYBRID_PARAMS
    assert m.DENSE_PARAMS == 26_367_104_000, m.DENSE_PARAMS
    assert m.STATE_BYTES == 154_927_104, m.STATE_BYTES
    assert m.AR_BYTES_PER_LAYER == 40_960, m.AR_BYTES_PER_LAYER
    assert m.KV_BYTES_PER_LAYER == 4096

    hw = a100(m)
    hybrid = {"kind": "hybrid"}
    dense = {"kind": "dense"}
    beta = hw["hbm_gbs"] * 1e9
    peak = hw["bf16_tflops"] * 1e12
    ridge = peak / beta
    print(f"ridge_flops_per_byte {ridge:.6f}")
    print(f"hybrid_params {m.HYBRID_PARAMS}")
    print(f"dense_params {m.DENSE_PARAMS}")
    print(f"hybrid_weight_bytes {m.HYBRID_PARAMS * 2}")
    print(f"dense_weight_bytes {m.DENSE_PARAMS * 2}")
    print(f"copy_fraction {1275 / 1935:.6f}")

    for kind, cfg in (("hybrid", hybrid), ("dense", dense)):
        flops = m.decode_flops(kind, 1, 1)
        nbytes = m.decode_traffic_bytes(kind, 1, 1, 1)
        compute = flops / peak
        memory = nbytes / beta
        launch = m.launch_seconds(hw)
        total = m.decode_step_time(cfg, hw, 1, 1)
        print(
            f"decode {kind} b1 c1 flops {flops} bytes {nbytes:.0f} "
            f"compute_ms {compute * 1e3:.6f} memory_ms {memory * 1e3:.6f} "
            f"launch_ms {launch * 1e3:.3f} total_ms {total * 1e3:.6f}"
        )

    tokens = 2048
    pf = m.prefill_flops("hybrid", tokens)
    pb = m.prefill_traffic_bytes("hybrid", tokens, 1)
    print(
        f"prefill 2048 flops {pf} bytes {pb:.0f} "
        f"compute_ms {pf / peak * 1e3:.6f} memory_ms {pb / beta * 1e3:.6f} "
        f"total_ms {m.prefill_time(hybrid, hw, tokens) * 1e3:.6f}"
    )

    # Chapter 19's prefill communication example, bandwidth term only.
    comm_bytes = 64 * 4096 * 40960 * 0.5
    print(f"ch19_prefill_comm_bytes {comm_bytes} gb {comm_bytes / 1e9:.4f}")
    print(f"ch19_nvlink_one_way_ms {comm_bytes / 300e9 * 1e3:.4f}")
    print(f"model_prefill_comm_ms {m.comm_seconds(hw, 4096, 2) * 1e3:.4f}")

    print("crossing and fit")
    for ctx in (1, 128, 256, 512, 1024, 4096, 32768):
        cross, asym = crossing(m, "hybrid", ctx)
        fits = {tp: max_fit(m, "hybrid", ctx, tp) for tp in (1, 2, 4)}
        cross_s = "none" if cross is None else f"{cross:.2f}"
        print(
            f"  ctx {ctx} asym_intensity {asym:.3f} cross_batch {cross_s} "
            f"fit_p1 {fits[1]} fit_p2 {fits[2]} fit_p4 {fits[4]}"
        )

    print("throughput hybrid ctx=1024 tp=1")
    for batch in (1, 2, 4, 8, 16, 32, 48, 64):
        if m.resident_bytes("hybrid", batch, 1024, 1) > hw["memory_bytes"]:
            print(f"  b {batch} does not fit")
            continue
        step = m.decode_step_time(hybrid, hw, batch, 1024)
        flops = m.decode_flops("hybrid", batch, 1024) / peak
        mem = m.decode_traffic_bytes("hybrid", batch, 1024, 1) / beta
        print(
            f"  b {batch} step_ms {step * 1e3:.4f} tok_s {batch / step:.2f} "
            f"compute_ms {flops * 1e3:.4f} memory_ms {mem * 1e3:.4f} "
            f"bound {'compute' if flops > mem else 'memory'}"
        )

    workload = {"prompt": 128, "ctx": 1024, "max_batch": 64}
    ttft = m.prefill_time(hybrid, hw, 128)
    print(f"ttft_prompt_128_ms {ttft * 1e3:.4f}")
    for slo_tpot in (0.02, 0.03, 0.04, 0.05, 0.08, 0.2, 1.0):
        g = m.max_goodput(hybrid, hw, 1.0, slo_tpot, workload)
        print(f"  goodput slo_tpot {slo_tpot} tok_s {g:.3f}")

    print("layouts n_gpus=4 ctx=1024 max_batch=64 prompt=128")
    for name, slo_ttft, slo_tpot in (
        ("loose", 1.0, 1.0),
        ("tight", ttft * 2, m.decode_step_time(hybrid, hw, 1, 1024) * 0.7),
    ):
        choice = m.choose_layout(hybrid, hw, 4, slo_ttft, slo_tpot, workload)
        print(f"  {name} slo_tpot {slo_tpot:.6f} -> {choice}")
        for tp in (1, 2, 4):
            hw_p = dict(hw, tp=tp)
            per = m.max_goodput(hybrid, hw_p, slo_ttft, slo_tpot, workload)
            print(f"    tp {tp} replicas {4 // tp} per {per:.3f} total {per * (4 // tp):.3f}")

    # Synthetic trace, same seed and formula as the lab.
    alpha, tau, sigma = 0.7, 0.002, 5e-6
    rng = random.Random(17)
    rows = []
    print("synthetic trace (not measured)")
    for batch in (1, 2, 4, 8, 16):
        flops = m.decode_flops("hybrid", batch, 128)
        nbytes = m.decode_traffic_bytes("hybrid", batch, 128, 1)
        noise = sigma * gauss(rng)
        seconds = nbytes / (alpha * beta) + tau + noise
        rows.append({"bytes": nbytes, "flops": flops, "seconds": seconds})
        print(f"  b {batch} bytes {nbytes:.0f} seconds {seconds:.8f} noise_s {noise:.3e}")
    fit = m.fit_efficiency(rows, hw["hbm_gbs"])
    print(f"recovered alpha {fit['alpha']:.8f} overhead_s {fit['overhead_s']:.8e}")
    xs = [row["bytes"] / beta for row in rows]
    n = len(xs)
    mean_x = sum(xs) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    se_slope = sigma / math.sqrt(sxx)
    se_tau = sigma * math.sqrt(1 / n + mean_x**2 / sxx)
    se_alpha = se_slope / ((1 / alpha) ** 2)
    print(f"se_alpha {se_alpha:.6e} se_tau {se_tau:.6e}")
    print(f"tol_alpha_6se {6 * se_alpha:.6e} tol_tau_6se {6 * se_tau:.6e}")
    print(f"err_alpha {fit['alpha'] - alpha:.6e} err_tau {fit['overhead_s'] - tau:.6e}")

    # Dollar example with a caller-supplied rate, not a fetched A100 price.
    g = m.max_goodput(hybrid, hw, 1.0, 1.0, {"prompt": 128, "ctx": 1024, "max_batch": 4096})
    print(f"single_gpu_cap_tok_s {g:.6f}")
    print(f"tokens_per_dollar_over_price {g * 3600:.6f}")

    print("layout table, 4 GPUs, ctx 1024, prompt 128, max_batch 4096, TTFT SLO 1 s")
    wl = {"prompt": 128, "ctx": 1024, "max_batch": 4096}
    for link, name in (LINKS):
        hw_l = dict(hw, link_gbs=link)
        for slo_tpot in (0.025, 0.040, 0.080, 1.0):
            row = []
            for tp in (1, 2, 4):
                per = m.max_goodput(hybrid, dict(hw_l, tp=tp), 1.0, slo_tpot, wl)
                row.append(f"{per * (4 // tp):.0f}")
            print(f"  {name} tpot {slo_tpot} replicas4 / tp2 / tp4: " + " / ".join(row))

    print("full-cap layouts, 4 GPUs, prompt 128, max_batch 4096")
    for ctx in (1024, 4096, 32768):
        wl = {"prompt": 128, "ctx": ctx, "max_batch": 4096}
        for link, name in LINKS:
            hw_l = dict(hw, link_gbs=link)
            choice = m.choose_layout(hybrid, hw_l, 4, 1.0, 1.0, wl)
            print(f"  ctx {ctx} {name} pick tp {choice['tp']} replicas {choice['replicas']} tok_s {choice['tokens_per_s']:.2f}")
            alts = [m.max_goodput(hybrid, dict(hw_l, tp=tp), 1.0, 1.0, wl) * (4 // tp) for tp in (1, 2, 4)]
            print(f"    replicas4 / tp2 / tp4: " + " / ".join(f"{a:.0f}" for a in alts))

    # Chapter 19 weight-read at 1275, for the comparison sentence.
    print(f"weight_only_1935_ms {53_791_959_040 / 1935e9 * 1e3:.4f}")
    print(f"weight_only_1275_ms {53_791_959_040 / 1275e9 * 1e3:.4f}")


if __name__ == "__main__":
    main()
