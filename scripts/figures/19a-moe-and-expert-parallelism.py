"""Numbers for chapter 19a.

Every printed number is arithmetic on the Qwen3-30B-A3B config or output of
content/labs/19a-moe/solution.py. Routing statistics come from seeded
simulations (seed 19) with synthetic routers. Nothing here is a measurement of a
trained router or of an A100.

Qwen3-30B-A3B config (huggingface.co/Qwen/Qwen3-30B-A3B, config.json): hidden
2048, 48 layers, 128 experts, top-8, expert width 768, 32 query heads, 4 KV
heads, head dim 128, vocabulary 151,936, untied embeddings, `norm_topk_prob`
true, bfloat16.

Run from the repository root:

    python3 scripts/figures/19a-moe-and-expert-parallelism.py
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SOL = ROOT / "content" / "labs" / "19a-moe" / "solution.py"
SEED = 19

# Qwen3-30B-A3B.
D, FF, L, E, K = 2048, 768, 48, 128, 8
HEADS, KV_HEADS, HEAD_DIM, VOCAB = 32, 4, 128, 151_936
BPP = 2

# The course's hardware figures (chapters 10, 17a, 19).
RIDGE = 312e12 / 1935e9
COPY_BW = 1275e9
NVLINK_ONE_WAY = 300e9
ALPHA = 4e-6
DENSE_BYTES = 53_791_959_040  # chapter 2: Qwen3.8-27B in bfloat16


def load():
    spec = importlib.util.spec_from_file_location("moe_solution", SOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def zipf_probs(n, k, s):
    w = [(i + 1) ** -s for i in range(n)]
    p = [0.0] * n
    free = set(range(n))
    budget = float(k)
    while True:
        total = sum(w[i] for i in free)
        over = [i for i in free if budget * w[i] / total > 1.0]
        if not over:
            for i in free:
                p[i] = budget * w[i] / total
            return p
        for i in over:
            p[i] = 1.0
            free.discard(i)
            budget -= 1.0


def sample_experts(probs, tokens, gen):
    """k distinct experts per token with marginals `probs`.

    Systematic sampling on a fresh random order of the experts for each token,
    so neighbouring experts are not picked together.
    """
    p = torch.tensor(probs, dtype=torch.float64)
    n = len(probs)
    k = int(round(p.sum().item()))
    perm = torch.argsort(torch.rand(tokens, n, generator=gen, dtype=torch.float64), dim=1)
    cum = torch.cumsum(p[perm], dim=1)
    u = torch.rand(tokens, 1, generator=gen, dtype=torch.float64)
    pts = u + torch.arange(k, dtype=torch.float64).unsqueeze(0)
    pos = torch.searchsorted(cum, pts, right=True).clamp(max=n - 1)
    return perm.gather(1, pos)


def main():
    m = load()

    print("== Geometry (config arithmetic) ==")
    expert_params = 3 * D * FF
    attn = D * HEADS * HEAD_DIM * 2 + D * KV_HEADS * HEAD_DIM * 2
    router = D * E
    norms_layer = 2 * D + 2 * HEAD_DIM
    embed = VOCAB * D
    non_expert = L * (attn + router + norms_layer) + 2 * embed + D
    experts_all = L * E * expert_params
    total = non_expert + experts_all
    active = non_expert + L * K * expert_params
    print(f"expert params {expert_params:,}, bytes {expert_params * BPP:,} "
          f"({expert_params * BPP / 2**20:.1f} MiB)")
    print(f"experts per layer {E * expert_params:,} params; all layers {experts_all:,}")
    print(f"attention per layer {attn:,}; router per layer {router:,}; embeddings 2 x {embed:,}")
    print(f"non-expert params {non_expert:,} ({non_expert * BPP / 1e9:.3f} GB)")
    print(f"total params {total:,} ({total / 1e9:.2f} B); bytes {total * BPP:,} "
          f"= {total * BPP / 1e9:.2f} GB = {total * BPP / 2**30:.2f} GiB")
    print(f"active params at top-{K}: {active:,} ({active / 1e9:.2f} B); "
          f"fraction of total {active / total:.3f}")
    left = 80 * 2**30 - total * BPP
    print(f"left on an 80 GiB A100 after weights: {left / 2**30:.2f} GiB")
    kv_tok = 2 * KV_HEADS * HEAD_DIM * BPP * L
    print(f"KV per token {kv_tok:,} B = {kv_tok / 1024:.0f} KiB; tokens in that room "
          f"{left // kv_tok:,}")

    print("\n== Distinct experts and the decode step (uniform routing) ==")
    print("batch distinct  frac   expert_MLP_intensity  weight_GB  step_ms  tokens/s  "
          "GB/token  dense_ms dense_tok/s dense_GB/token")
    expert_bytes_layer = expert_params * BPP
    for b in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        d = m.expected_distinct_experts(b, E, K)
        r = m.moe_decode_intensity(b, E, K, D, FF)
        wbytes = non_expert * BPP + L * d * expert_bytes_layer
        t = wbytes / COPY_BW
        td = DENSE_BYTES / COPY_BW
        print(f"{b:5d} {d:8.2f} {d / E:6.3f} {r['intensity']:10.3f} {wbytes / 1e9:14.2f} "
              f"{t * 1e3:8.2f} {b / t:9.0f} {wbytes / b / 1e9:8.3f} {td * 1e3:8.2f} "
              f"{b / td:10.0f} {DENSE_BYTES / b / 1e9:10.3f}")
    print(f"ridge {RIDGE:.3f} FLOP/B")
    rb = m.batch_to_ridge(E, K, RIDGE)
    print(f"batch to ridge: uniform {rb}; dense (intensity = batch) {math.ceil(RIDGE)}; "
          f"ratio {rb / math.ceil(RIDGE):.2f}; E/k = {E / K:.0f}")
    for s in (0.5, 0.8, 1.0):
        p = zipf_probs(E, K, s)
        print(f"skew s={s}: distinct at batch 32 = {m.expected_distinct_experts(32, E, K, p):.2f}, "
              f"at batch 128 = {m.expected_distinct_experts(128, E, K, p):.2f}, "
              f"batch to ridge {m.batch_to_ridge(E, K, RIDGE, probs=p)}, max p {max(p):.3f}")
    t1 = m.moe_decode_intensity(1, E, K, D, FF)
    base1 = non_expert * BPP + L * K * expert_bytes_layer
    print(f"batch-1 expert read per layer {t1['bytes']:,.0f} B; per step {base1:,} B")
    b32 = m.expected_distinct_experts(32, E, K)
    print(f"batch 32: bytes vs batch 1 = "
          f"{(non_expert * BPP + L * b32 * expert_bytes_layer) / base1:.2f}x")

    print("\n== Capacity factor, uniform routing, seed 19 ==")
    gen = torch.Generator().manual_seed(SEED)
    for tokens in (256, 1024, 4096):
        ex = torch.stack([torch.randperm(E, generator=gen)[:K] for _ in range(tokens)])
        loads = m.expert_loads(ex, E)
        line = f"tokens {tokens}: mean load {tokens * K / E:.1f}, max load {max(loads)}"
        for cf in (1.0, 1.25, 1.5, 2.0):
            cap = m.expert_capacity(tokens, K, E, cf)
            keep = m.apply_capacity(ex, E, cap)
            line += f"; cf {cf}: cap {cap}, dropped {(~keep).float().mean().item() * 100:.2f}%"
        print(line)

    print("\n== Expert parallelism, EP 8, 16 experts per rank ==")
    world = 8
    per_rank_experts = E // world
    print(f"expert bytes per rank {L * per_rank_experts * expert_bytes_layer / 1e9:.2f} GB; "
          f"total per rank with replicated non-expert weights "
          f"{(L * per_rank_experts * expert_bytes_layer + non_expert * BPP) / 1e9:.2f} GB")
    print("tokens_per_rank  dispatch_bytes  MiB   t_bw_us(300GB/s)  t_alpha_us  t_one_a2a_us  "
          "per_step_ms(2*48)")
    for tpr in (1, 16, 128, 1024, 4096):
        vol = m.all_to_all_volume(tpr, K, D, world)
        t_bw = vol / NVLINK_ONE_WAY
        t_lat = (world - 1) * ALPHA
        t = t_lat + t_bw
        print(f"{tpr:8d} {vol:14,.0f} {vol / 2**20:8.3f} {t_bw * 1e6:10.2f} {t_lat * 1e6:10.1f} "
              f"{t * 1e6:10.2f} {2 * L * t * 1e3:10.3f}")
    print(f"PCIe 32 GB/s, 128 tokens per rank: bandwidth term "
          f"{m.all_to_all_volume(128, K, D, world) / 32e9 * 1e6:.1f} us per all-to-all")
    print(f"copies per token that leave the rank: {K * (world - 1) / world:.2f} of {K}")

    print("\n== Load imbalance and EPLB, synthetic skew, seed 19 ==")
    tokens = 1024
    for s in (0.0, 0.5, 0.8):
        probs = [K / E] * E
        if s > 0:
            # Popularity is Zipf-shaped, and the popular experts get random ids.
            ranked = zipf_probs(E, K, s)
            order = torch.randperm(E, generator=torch.Generator().manual_seed(SEED + 1)).tolist()
            probs = [0.0] * E
            for rank, expert in enumerate(order):
                probs[expert] = ranked[rank]
        gen = torch.Generator().manual_seed(SEED)
        ex = sample_experts(probs, tokens, gen)
        loads = m.expert_loads(ex, E)
        rl = m.rank_loads(loads, E // world)
        c = m.imbalance_cost(rl)
        line = (f"s={s}: rank loads {rl}, factor {c['factor']:.3f}, "
                f"efficiency {c['efficiency']:.3f}")
        for extra in (0, 8, 16, 32):
            plan = m.eplb_plan(loads, E + extra, world)
            cc = m.imbalance_cost(plan["rank_loads"])
            line += f"; +{extra} slots: factor {cc['factor']:.3f}"
        print(line)

    print("\n== Other published geometries (routed experts only) ==")
    # Mixtral 8x7B: 8 experts, top-2 (arXiv:2401.04088); the sizes cancel in the ratios. DeepSeek-V3: 256 routed
    # experts, top-8, hidden 7168, expert width 2048 (config.json, arXiv:2412.19437).
    for name, n_exp, k_, d_, f_ in (("Qwen3-30B-A3B", 128, 8, 2048, 768),
                                    ("DeepSeek-V3", 256, 8, 7168, 2048),
                                    ("Mixtral 8x7B", 8, 2, 0, 0)):
        rb_ = m.batch_to_ridge(n_exp, k_, RIDGE)
        print(f"{name}: E={n_exp}, k={k_}, E/k={n_exp / k_:.0f}, "
              f"distinct at batch 32 {m.expected_distinct_experts(32, n_exp, k_):.2f} "
              f"({m.expected_distinct_experts(32, n_exp, k_) / n_exp:.3f} of E), "
              f"batch to ridge {rb_}")

    print("\n== Chapter 19a worked values ==")
    print(f"E*(1-(1-k/E)^B) at B=1, 8, 32, 64: "
          f"{[round(m.expected_distinct_experts(b, E, K), 2) for b in (1, 8, 32, 64)]}")
    print(f"naive slot-draw formula at B=1: {E * (1 - (1 - 1 / E) ** K):.4f}")


if __name__ == "__main__":
    main()
