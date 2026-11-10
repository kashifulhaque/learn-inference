"""Checks for routing, dispatch and combine, expert counts, and expert parallelism.

The first check calls only `route`. Later checks add the capacity rule, dispatch
and combine, the whole layer against a per-token loop, the grouped matmul, the
expected number of experts a batch touches, decode intensity, the all-to-all,
and load balancing. Randomness is seeded (seed 19). There is no gradient in
this lab, so there is no finite-difference check.

Plausible wrong implementations, each of which the checks reject:

1. `route` that takes the top k of the softmax and forgets to renormalize. The
   weights then sum to less than 1, and "route weights" fails. Qwen3-30B-A3B
   sets `norm_topk_prob` to true for this reason.
2. `expected_distinct_experts` that treats the batch * k slots as independent
   draws, E * (1 - (1 - 1/E)^(batch * k)). A token's k experts are distinct, so
   at batch 1 this gives 7.78 for E=128, k=8, not 8. "distinct experts" fails.
3. `all_to_all_volume` that counts every copy as sent, without the
   (world - 1) / world share that stays on the rank. "all-to-all" fails.
"""

import math

import torch
import torch.nn.functional as F

from lab_common import Checks, close

SEED = 19
E, K, D, FF = 8, 2, 16, 24


def _setup(tokens: int):
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(tokens, D, generator=g)
    router_w = torch.randn(E, D, generator=g)
    gate = torch.randn(E, FF, D, generator=g) * 0.3
    up = torch.randn(E, FF, D, generator=g) * 0.3
    down = torch.randn(E, D, FF, generator=g) * 0.3
    return x, router_w, gate, up, down


def _ref_keep(experts, capacity):
    """Per-expert counters in row-major order. No sorting, so it is independent."""
    seen = {}
    keep = torch.ones_like(experts, dtype=torch.bool)
    if capacity is None:
        return keep
    for t in range(experts.shape[0]):
        for j in range(experts.shape[1]):
            e = int(experts[t, j])
            seen[e] = seen.get(e, 0) + 1
            keep[t, j] = seen[e] <= capacity
    return keep


def ref_moe(x, router_w, gate, up, down, k, capacity=None):
    """One token at a time, one expert at a time."""
    probs = torch.softmax(x @ router_w.T, dim=-1)
    out = torch.zeros_like(x)
    top = [torch.topk(probs[t], k) for t in range(x.shape[0])]
    experts = torch.stack([t.indices for t in top])
    keep = _ref_keep(experts, capacity)
    for t in range(x.shape[0]):
        w = top[t].values / top[t].values.sum()
        for j in range(k):
            if not keep[t, j]:
                continue
            e = int(experts[t, j])
            h = F.silu(x[t] @ gate[e].T) * (x[t] @ up[e].T)
            out[t] += w[j] * (h @ down[e].T)
    return out


def inclusion_probs(n, k, s):
    """Zipf-shaped per-expert inclusion probabilities, capped at 1, summing to k."""
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


def sample_distinct(probs, batch, trials, seed):
    """Mean and standard error of distinct experts. Systematic sampling picks k
    distinct experts per token with exactly the marginals `probs`."""
    g = torch.Generator().manual_seed(seed)
    p = torch.tensor(probs, dtype=torch.float64)
    k = int(round(p.sum().item()))
    cum = torch.cumsum(p, 0)
    counts = []
    for _ in range(trials):
        u = torch.rand(batch, 1, generator=g, dtype=torch.float64)
        pts = u + torch.arange(k, dtype=torch.float64).unsqueeze(0)
        idx = torch.searchsorted(cum, pts.reshape(-1), right=True).clamp(max=len(probs) - 1)
        counts.append(len(torch.unique(idx)))
    c = torch.tensor(counts, dtype=torch.float64)
    return c.mean().item(), (c.std() / math.sqrt(trials)).item()


def run(sub) -> dict:
    c = Checks()
    names = ["route", "expert_capacity", "apply_capacity", "dispatch", "combine",
             "grouped_matmul_loop", "grouped_matmul_padded", "padded_rows", "moe_layer",
             "expected_distinct_experts", "moe_decode_intensity", "batch_to_ridge",
             "all_to_all_matrix", "all_to_all_bytes", "all_to_all_volume",
             "expert_loads", "rank_loads", "imbalance_cost", "eplb_plan"]
    if not c.require(sub, *names):
        return c.finish()

    # 1. Routing.
    def route_check():
        logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [0.0, 3.0, 1.0, 2.0]])
        w, e = sub.route(logits, 2)
        sig = 1.0 / (1.0 + math.exp(-1.0))
        want_w = torch.tensor([[sig, 1 - sig], [sig, 1 - sig]])
        want_e = torch.tensor([[0, 1], [1, 3]])
        ok_e = torch.equal(e, want_e)
        ok_w, msg = close(w, want_w, 1e-6)
        sums = w.sum(-1)
        ok_s = bool(torch.allclose(sums, torch.ones(2), atol=1e-6))
        return (ok_e and ok_w and ok_s,
                f"experts {e.tolist()} (expected {want_e.tolist()}); weights {msg}; "
                f"row sums {[round(v, 6) for v in sums.tolist()]} (expected [1.0, 1.0])")
    c.check("route weights", route_check)

    # 2. Capacity.
    def capacity_check():
        n = [sub.expert_capacity(4, 2, 4, 1.0), sub.expert_capacity(16, 2, 4, 1.25),
             sub.expert_capacity(10, 2, 8, 1.0)]
        want_n = [2, 10, 3]
        experts = torch.tensor([[0, 1], [0, 2], [0, 1], [3, 0]])
        keep = sub.apply_capacity(experts, 4, 2)
        want = torch.tensor([[True, True], [True, True], [False, True], [True, False]])
        return (n == want_n and torch.equal(keep, want),
                f"capacities {n} (expected {want_n}); keep {keep.tolist()} (expected {want.tolist()})")
    c.check("capacity and token dropping", capacity_check)

    # 3. Dispatch and combine.
    def dispatch_check():
        experts = torch.tensor([[2, 0], [1, 2], [0, 1]])
        x = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        keep = torch.tensor([[True, True], [True, True], [True, False]])
        xs, ti, si, counts = sub.dispatch(x, experts, keep, 3)
        want_t, want_s, want_c = [0, 2, 1, 0, 1], [1, 0, 0, 0, 1], [2, 1, 2]
        ok = ti.tolist() == want_t and si.tolist() == want_s and counts.tolist() == want_c
        ok = ok and torch.equal(xs, x[torch.tensor(want_t)])
        w = torch.tensor([[0.75, 0.25], [0.5, 0.5], [1.0, 0.0]])
        out = sub.combine(torch.ones(5, 2), ti, si, w, 3)
        want_out = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        # token 2 keeps only slot 0 (weight 1.0); the others sum to 1.
        ok_c, msg = close(out, want_out, 1e-6)
        return (ok and ok_c,
                f"tokens {ti.tolist()} (expected {want_t}), slots {si.tolist()} (expected {want_s}), "
                f"counts {counts.tolist()} (expected {want_c}); combine {msg}")
    c.check("dispatch and combine", dispatch_check)

    # 4. The layer equals the per-token reference, with and without dropping.
    def layer_check():
        x, rw, g, u, d = _setup(37)
        want = ref_moe(x, rw, g, u, d, K)
        got = sub.moe_layer(x, rw, g, u, d, K)
        ok1, m1 = close(got, want, 1e-4)
        cap = sub.expert_capacity(37, K, E, 1.0)
        want2 = ref_moe(x, rw, g, u, d, K, capacity=cap)
        got2 = sub.moe_layer(x, rw, g, u, d, K, capacity=cap)
        ok2, m2 = close(got2, want2, 1e-4)
        dropped = int((~_ref_keep(torch.topk(torch.softmax(x @ rw.T, -1), K).indices, cap)).sum())
        ok3 = dropped > 0 and not torch.allclose(want, want2)
        return (ok1 and ok2 and ok3,
                f"no drop: {m1}; capacity {cap}: {m2}; {dropped} dropped (expected > 0)")
    c.check("layer equals per-token reference", layer_check)

    # 5. Permutation invariance.
    def perm_check():
        x, rw, g, u, d = _setup(29)
        perm = torch.randperm(29, generator=torch.Generator().manual_seed(SEED + 1))
        a = sub.moe_layer(x, rw, g, u, d, K)[perm]
        b = sub.moe_layer(x[perm], rw, g, u, d, K)
        return close(a, b, 1e-5)
    c.check("permuting the tokens permutes the output", perm_check)

    # 6. Grouped matmul against the loop.
    def grouped_check():
        gen = torch.Generator().manual_seed(SEED + 2)
        counts = torch.tensor([5, 0, 17, 2, 0, 9, 1, 4])
        xs = torch.randn(int(counts.sum()), D, generator=gen)
        w = torch.randn(E, FF, D, generator=gen)
        loop = sub.grouped_matmul_loop(xs, counts, w)
        pad = sub.grouped_matmul_padded(xs, counts, w, 8)
        want = torch.cat([xs[s:s + n] @ w[e].T for e, (s, n) in enumerate(
            zip((torch.cumsum(counts, 0) - counts).tolist(), counts.tolist()))])
        ok1, m1 = close(loop, want, 1e-5)
        ok2, m2 = close(pad, loop, 1e-5)
        rows = sub.padded_rows(counts, 8)
        return (ok1 and ok2 and rows == 8 * 24,
                f"loop {m1}; padded {m2}; padded rows {rows} (expected 192 for 38 real rows)")
    c.check("grouped matmul equals the loop", grouped_check)

    # 7. Distinct experts and decode intensity.
    def distinct_check():
        ok = []
        got1 = sub.expected_distinct_experts(1, 128, 8)
        ok.append(abs(got1 - 8.0) < 1e-9)
        got32 = sub.expected_distinct_experts(32, 128, 8)
        want32 = 128 * (1 - (15 / 16) ** 32)
        ok.append(abs(got32 - want32) < 1e-9)
        c.metric("distinct_experts_batch_32", round(got32, 3))
        ok.append(abs(sub.expected_distinct_experts(5000, 128, 8) - 128) < 1e-6)
        # Seeded Monte Carlo, tolerance five standard errors.
        p = [8 / 128] * 128
        mean, se = sample_distinct(p, 16, 3000, SEED)
        theory = sub.expected_distinct_experts(16, 128, 8)
        ok.append(abs(mean - theory) <= 5 * se)
        # Skew touches fewer experts, and the closed form still matches.
        ps = inclusion_probs(128, 8, 0.8)
        meanS, seS = sample_distinct(ps, 16, 3000, SEED + 3)
        theoryS = sub.expected_distinct_experts(16, 128, 8, probs=ps)
        ok.append(abs(meanS - theoryS) <= 5 * seS and theoryS < theory)
        return (all(ok),
                f"B=1: {got1:.4f} (expected 8); B=32: {got32:.4f} (expected {want32:.4f}); "
                f"B=16 uniform theory {theory:.3f} vs simulated {mean:.3f}+-{se:.3f}; "
                f"skew theory {theoryS:.3f} vs simulated {meanS:.3f}+-{seS:.3f}; checks {ok}")
    c.check("distinct experts: closed form and simulation", distinct_check)

    def intensity_check():
        r1 = sub.moe_decode_intensity(1, 128, 8, 2048, 768)
        r64 = sub.moe_decode_intensity(64, 128, 8, 2048, 768)
        per_expert = 3 * 2048 * 768 * 2
        want_bytes1 = 8 * per_expert
        want_flops64 = 2 * 64 * 8 * 3 * 2048 * 768
        d64 = 128 * (1 - (15 / 16) ** 64)
        ok = (abs(r1["bytes"] - want_bytes1) < 1 and abs(r1["intensity"] - 1.0) < 1e-9
              and abs(r64["flops"] - want_flops64) < 1
              and abs(r64["bytes"] - d64 * per_expert) < 1e-3 * per_expert)
        ridge = 312e12 / 1935e9
        b_uniform = sub.batch_to_ridge(128, 8, ridge)
        # Dense with the same active width reaches the ridge at ceil(ridge) tokens.
        below = sub.moe_decode_intensity(b_uniform - 1, 128, 8, 1, 1)["intensity"]
        above = sub.moe_decode_intensity(b_uniform, 128, 8, 1, 1)["intensity"]
        ok = ok and below < ridge <= above
        c.metric("ridge_batch_uniform", b_uniform)
        return (ok,
                f"B=1 bytes {r1['bytes']:.0f} (expected {want_bytes1}), intensity {r1['intensity']:.4f} "
                f"(expected 1); B=64 intensity {r64['intensity']:.4f}; ridge {ridge:.2f} reached at "
                f"batch {b_uniform} ({below:.3f} just below, {above:.3f} at)")
    c.check("decode intensity and the batch that reaches the ridge", intensity_check)

    # 8. All-to-all.
    def a2a_check():
        # 4 ranks, 2 experts per rank (8 experts), 2 tokens per rank, k = 2.
        experts = torch.tensor([[0, 1], [2, 7], [3, 4], [5, 6],
                                [0, 2], [1, 3], [4, 6], [7, 5]])
        token_rank = torch.arange(8) // 2
        m = sub.all_to_all_matrix(experts, token_rank, 2, 4)
        want = torch.tensor([[2, 1, 0, 1], [0, 1, 2, 1], [2, 2, 0, 0], [0, 0, 2, 2]])
        # By hand: rank 0 holds tokens 0 and 1. Experts 0 and 1 live on rank 0, expert 2
        # on rank 1, and expert 7 on rank 3, so its row is [2, 1, 0, 1].
        ok_m = torch.equal(m, want)
        per_rank = sub.all_to_all_bytes(m, 2048, 2)
        want_bytes = [n * 2048 * 2 for n in (2, 3, 4, 2)]
        # Closed form against a seeded uniform draw, averaged over many batches.
        gen = torch.Generator().manual_seed(SEED + 4)
        world, tpr, kk, n_exp = 8, 64, 8, 128
        sent = []
        for _ in range(40):
            ex = torch.stack([torch.randperm(n_exp, generator=gen)[:kk] for _ in range(world * tpr)])
            mm = sub.all_to_all_matrix(ex, torch.arange(world * tpr) // tpr, n_exp // world, world)
            sent.append(sum(sub.all_to_all_bytes(mm, 1, 1)) / world)
        mc = sum(sent) / len(sent)
        closed = sub.all_to_all_volume(tpr, kk, 1, world, 1)
        exact = sub.all_to_all_volume(64, 8, 2048, 8, 2)
        ok = (ok_m and per_rank == want_bytes and abs(mc - closed) / closed < 0.01
              and abs(exact - 64 * 8 * 2048 * 2 * 7 / 8) < 1e-6)
        c.metric("all_to_all_bytes_per_rank", int(exact))
        return (ok,
                f"matrix {m.tolist()} (expected {want.tolist()}); bytes {per_rank} (expected "
                f"{want_bytes}); closed form {closed:.1f} vs simulated {mc:.1f} off-rank copies per rank; "
                f"64 tokens, k=8, d=2048, 8 ranks: {exact:.0f} bytes (expected {64*8*2048*2*7//8})")
    c.check("all-to-all matrix and volume", a2a_check)

    # 9. Imbalance and EPLB.
    def balance_check():
        r = sub.imbalance_cost([10, 10, 10, 2])
        ok = (r["max"] == 10 and abs(r["mean"] - 8.0) < 1e-12 and abs(r["factor"] - 1.25) < 1e-12
              and abs(r["efficiency"] - 0.8) < 1e-12 and abs(r["idle_fraction"] - 0.2) < 1e-12)
        loads = [9, 4, 2, 1]
        base = sub.imbalance_cost(sub.rank_loads(loads, 2))
        ok = ok and base["max"] == 13 and abs(base["factor"] - 13 / 8) < 1e-12
        counted = sub.expert_loads(torch.tensor([[0, 1], [0, 2], [0, 1], [3, 0]]), 4)
        ok = ok and counted == [4, 2, 1, 1]
        plan = sub.eplb_plan(loads, 6, 2)
        after = sub.imbalance_cost(plan["rank_loads"])
        ok = (ok and plan["replicas"] == [2, 2, 1, 1] and plan["ranks"] == [[0, 1, 2], [0, 1, 3]]
              and plan["rank_loads"] == [8.5, 7.5] and abs(after["factor"] - 8.5 / 8) < 1e-12)
        c.metric("eplb_imbalance_factor", round(after["factor"], 4))
        return (ok,
                f"[10,10,10,2]: factor {r['factor']} (expected 1.25); loads [9,4,2,1] on 2 ranks: "
                f"max {base['max']} (expected 13); plan replicas {plan['replicas']} (expected [2, 2, 1, 1]), "
                f"ranks {plan['ranks']} (expected [[0, 1, 2], [0, 1, 3]]), loads {plan['rank_loads']} "
                f"(expected [8.5, 7.5]), factor {after['factor']:.4f} (expected 1.0625)")
    c.check("imbalance cost and EPLB plan", balance_check)

    return c.finish()
