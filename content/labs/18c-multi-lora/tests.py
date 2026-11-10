"""Checks for multi-LoRA serving: the update, bgmv, sgmv, cost, and an adapter pool.

The first check calls only `lora_forward`. Later checks add `merge_lora`,
`bgmv`, `segment_by_adapter`, `sgmv`, the cost functions, and the pool.

Plausible wrong implementation 1: scale the update by alpha instead of
alpha / r. With alpha 32 and rank 8 the update is 8 times too large. The
worked example (alpha 4, rank 2) gives [35, 22] instead of [21, 16], and the
`bgmv` checks use alpha 32 with rank 8.

Plausible wrong implementation 2: index the adapter stack by position in the
batch, `A_stack[arange(T)]`, instead of by request id. It passes when request
i happens to use adapter i. The `bgmv` check with permuted ids and the check
with more tokens than adapters catch it.

A pool that picks its victim by recency alone is flushed by a scan of cold
adapters. The seeded trace plants a scan every 30 batches. The check requires
the frequency policy to spend at most 90% of the PCIe time of the better of
FIFO and LRU. On this trace the real ratio is about 0.78.
"""

from __future__ import annotations

import random

import torch

from lab_common import Checks

GIB = 2**30
CAPACITY = 4 * GIB


def _call(submission, name, *args, **kwargs):
    try:
        return getattr(submission, name)(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def mlp_adapter_bytes(rank: int) -> int:
    """Adapter bytes for the MLP of all 64 layers, written out by hand."""
    per_layer = 2 * rank * ((5120 + 17408) + (5120 + 17408) + (17408 + 5120))
    return 64 * per_layer


def make_trace(seed: int = 17, n_adapters: int = 48, batches: int = 300, per_batch: int = 4):
    """Zipf traffic over 48 adapters, plus a scan of cold adapters every 30 batches."""
    rng = random.Random(seed)
    ranks = [rng.choices([8, 16, 64], [0.4, 0.45, 0.15])[0] for _ in range(n_adapters)]
    sizes = {i: mlp_adapter_bytes(r) for i, r in enumerate(ranks)}
    weights = [1 / (i + 1) ** 1.2 for i in range(n_adapters)]
    trace = []
    scan = 0
    for t in range(batches):
        if t % 30 >= 20:
            trace.append([(scan + j) % n_adapters for j in range(per_batch)])
            scan += per_batch
        else:
            trace.append(rng.choices(range(n_adapters), weights, k=per_batch))
    return trace, sizes, ranks


def _stacks(n, d_in, d_out, r, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, d_in, r, generator=g, dtype=torch.float64)
    B = torch.randn(n, r, d_out, generator=g, dtype=torch.float64)
    return A, B


def _loop_delta(x, A, B, ids, alpha):
    """Per-request loop, the ground truth for bgmv and sgmv."""
    r = A.shape[2]
    rows = []
    for t in range(x.shape[0]):
        a = int(ids[t])
        if a < 0:
            rows.append(torch.zeros(B.shape[2], dtype=x.dtype))
        else:
            rows.append((x[t] @ A[a]) @ B[a] * (alpha / r))
    return torch.stack(rows)


def _ref_pool(trace, sizes, capacity, policy):
    """Independent pool simulator. Returns (misses, bytes_loaded)."""
    held, last, born, count = {}, {}, {}, {}
    used = misses = loaded = clock = 0
    for step, batch in enumerate(trace, start=1):
        distinct = list(dict.fromkeys(batch))
        for a in distinct:
            count[a] = count.get(a, 0) + 1
        for a in distinct:
            if a in held:
                last[a] = step
                continue
            misses += 1
            while used + sizes[a] > capacity:
                pool = [k for k in held if k not in distinct]
                if policy == "fifo":
                    v = min(pool, key=lambda k: born[k])
                elif policy == "lru":
                    v = min(pool, key=lambda k: (last[k], born[k]))
                else:
                    v = min(pool, key=lambda k: (count[k], last[k], born[k]))
                used -= sizes[v]
                del held[v]
            clock += 1
            held[a], last[a], born[a] = True, step, clock
            used += sizes[a]
            loaded += sizes[a]
    return misses, loaded


def run(submission):
    c = Checks()
    if not c.require(
        submission, "lora_forward", "merge_lora", "bgmv", "segment_by_adapter", "sgmv",
        "lora_cost", "adapter_bytes", "adapter_load_ms", "simulate_adapter_pool",
    ):
        return c.finish()

    def worked_example():
        x = torch.tensor([[1.0, 2.0]])
        W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        A = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
        B = torch.tensor([[1.0, 0.0], [2.0, 1.0]])
        got, err = _call(submission, "lora_forward", x, W, A, B, 4.0)
        want = [[21.0, 16.0]]
        if err:
            return False, f"expected {want}, got {err}"
        ok = torch.allclose(torch.as_tensor(got).float(), torch.tensor(want), atol=1e-5)
        return ok, f"expected {want} (scale alpha/r = 2), got {torch.as_tensor(got).tolist()}"

    c.check("lora_forward matches the worked example (alpha 4, rank 2)", worked_example)

    def merge_equals_unmerged():
        g = torch.Generator().manual_seed(1)
        x = torch.randn(5, 12, generator=g, dtype=torch.float64)
        W = torch.randn(12, 9, generator=g, dtype=torch.float64)
        A = torch.randn(12, 4, generator=g, dtype=torch.float64)
        B = torch.randn(4, 9, generator=g, dtype=torch.float64)
        want = x @ W + (x @ A) @ B * (16.0 / 4)
        merged, err = _call(submission, "merge_lora", W, A, B, 16.0)
        if err:
            return False, f"expected x @ merged == unmerged, got {err}"
        gap = (x @ merged - want).abs().max().item()
        if gap > 1e-9:
            return False, f"expected max abs gap <= 1e-9, got {gap:.3e}"
        # A weight merged for adapter 0 serves adapter 1's rows wrongly.
        A2 = torch.randn(12, 4, generator=g, dtype=torch.float64)
        B2 = torch.randn(4, 9, generator=g, dtype=torch.float64)
        right = x @ W + (x @ A2) @ B2 * 4.0
        wrong_gap = (x @ merged - right).abs().max().item()
        return wrong_gap > 1.0, f"expected merged-for-adapter-0 to miss adapter 1 by > 1, got gap {gap:.1e}, miss {wrong_gap:.2f}"

    c.check("merge_lora matches the unmerged path and cannot serve a second adapter", merge_equals_unmerged)

    def bgmv_matches_loop():
        n, d_in, d_out, r = 5, 16, 10, 8
        A, B = _stacks(n, d_in, d_out, r, 2)
        g = torch.Generator().manual_seed(3)
        x = torch.randn(11, d_in, generator=g, dtype=torch.float64)
        ids = torch.tensor([3, 3, 0, 4, -1, 1, 3, 0, 4, 2, 1])
        want = _loop_delta(x, A, B, ids, 32.0)
        got, err = _call(submission, "bgmv", x, A, B, ids, 32.0)
        if err:
            return False, f"expected the per-request loop result, got {err}"
        got = torch.as_tensor(got).double()
        if got.shape != want.shape:
            return False, f"expected shape {tuple(want.shape)}, got {tuple(got.shape)}"
        gap = (got - want).abs().max().item()
        return gap <= 1e-8, f"expected max abs gap <= 1e-8 (alpha 32, rank 8, scale 4), got {gap:.3e}"

    c.check("bgmv equals the per-request loop, with repeats and a base-only row", bgmv_matches_loop)

    def bgmv_uses_request_ids():
        d_in, d_out, r = 8, 6, 4
        A, B = _stacks(4, d_in, d_out, r, 4)
        g = torch.Generator().manual_seed(5)
        x = torch.randn(4, d_in, generator=g, dtype=torch.float64)
        ids = torch.tensor([2, 0, 3, 1])  # a permutation, so position i is not adapter i
        want = _loop_delta(x, A, B, ids, 8.0)
        got, err = _call(submission, "bgmv", x, A, B, ids, 8.0)
        if err:
            return False, f"expected permuted ids to work, got {err}"
        gap = (torch.as_tensor(got).double() - want).abs().max().item()
        if gap > 1e-8:
            return False, f"expected max abs gap <= 1e-8 for ids [2, 0, 3, 1], got {gap:.3e}"
        x2 = torch.randn(9, d_in, generator=g, dtype=torch.float64)
        ids2 = torch.tensor([1, 1, 1, 0, 0, 3, 3, 3, 3])  # 9 tokens, 4 adapters
        want2 = _loop_delta(x2, A, B, ids2, 8.0)
        got2, err = _call(submission, "bgmv", x2, A, B, ids2, 8.0)
        if err:
            return False, f"expected 9 tokens over 4 adapters to work, got {err}"
        gap2 = (torch.as_tensor(got2).double() - want2).abs().max().item()
        return gap2 <= 1e-8, f"expected max abs gap <= 1e-8 with more tokens than adapters, got {gap2:.3e}"

    c.check("bgmv indexes by request adapter id, not batch position", bgmv_uses_request_ids)

    def segments():
        ids = torch.tensor([2, 0, -1, 2, 1, 0, 2])
        got, err = _call(submission, "segment_by_adapter", ids)
        if err:
            return False, f"expected order [2, 1, 5, 4, 0, 3, 6], got {err}"
        order, segs = got
        want_order = [2, 1, 5, 4, 0, 3, 6]
        want_segs = [(0, 1, 3), (1, 3, 4), (2, 4, 7)]
        ok = torch.as_tensor(order).tolist() == want_order and [tuple(s) for s in segs] == want_segs
        return ok, (
            f"expected order {want_order} and segments {want_segs}, "
            f"got order {torch.as_tensor(order).tolist()} and segments {[tuple(s) for s in segs]}"
        )

    c.check("segment_by_adapter sorts stably and skips the base-only rows", segments)

    def sgmv_matches():
        n, d_in, d_out, r = 6, 20, 14, 8
        A, B = _stacks(n, d_in, d_out, r, 6)
        g = torch.Generator().manual_seed(7)
        x = torch.randn(23, d_in, generator=g, dtype=torch.float64)
        ids = torch.randint(-1, n, (23,), generator=g)
        want = _loop_delta(x, A, B, ids, 32.0)
        got, err = _call(submission, "sgmv", x, A, B, ids, 32.0)
        if err:
            return False, f"expected the per-request loop result, got {err}"
        got = torch.as_tensor(got).double()
        gap = (got - want).abs().max().item()
        return gap <= 1e-8, f"expected rows back in request order, max abs gap <= 1e-8, got {gap:.3e}"

    c.check("sgmv equals bgmv and returns rows in request order", sgmv_matches)

    def cost_numbers():
        up = {r: submission.lora_cost(5120, 17408, r) for r in (8, 16, 64)}
        want_flops = {8: 360448, 16: 720896, 64: 2883584}
        for r in (8, 16, 64):
            if up[r]["lora_flops"] != want_flops[r]:
                return False, f"expected lora_flops {want_flops[r]} at rank {r}, got {up[r]['lora_flops']}"
        if up[16]["base_flops"] != 178257920 or up[16]["base_bytes"] != 178257920:
            return False, f"expected base 178257920 FLOPs and bytes, got {up[16]['base_flops']}, {up[16]['base_bytes']}"
        if up[16]["lora_bytes"] != 720896:
            return False, f"expected lora_bytes 720896 at rank 16, got {up[16]['lora_bytes']}"
        if abs(up[16]["flops_ratio"] - 720896 / 178257920) > 1e-12:
            return False, f"expected ratio {720896 / 178257920:.6f}, got {up[16]['flops_ratio']:.6f}"
        sizes = {r: submission.adapter_bytes(r) for r in (8, 16, 64)}
        want = {r: mlp_adapter_bytes(r) for r in (8, 16, 64)}
        if sizes != want:
            return False, f"expected adapter bytes {want}, got {sizes}"
        ms = submission.adapter_load_ms(want[16])
        if abs(ms - 138412032 / 32e9 * 1e3) > 1e-9:
            return False, f"expected {138412032 / 32e9 * 1e3:.4f} ms for 138412032 bytes at 32 GB/s, got {ms}"
        return True, "expected rank-16 MLP adapter 138,412,032 bytes, 4.3254 ms over PCIe, got the same"

    c.check("lora_cost, adapter_bytes, and adapter_load_ms match the hand arithmetic", cost_numbers)

    def small_pool():
        sizes = {"a": 2, "b": 2, "c": 2, "d": 4}
        trace = [["a", "b"], ["a"], ["c"], ["a", "d"], ["b"]]
        # Capacity 6. The independent simulator gives the expected counts.
        for policy in ("fifo", "lru", "freq"):
            got, err = _call(submission, "simulate_adapter_pool", trace, sizes, 6, policy)
            if err:
                return False, f"expected a result for {policy}, got {err}"
            wm, wb = _ref_pool(trace, sizes, 6, policy)
            if (got["misses"], got["bytes_loaded"]) != (wm, wb):
                return False, f"expected {policy} misses {wm} and bytes {wb}, got {got['misses']} and {got['bytes_loaded']}"
            if got["hits"] + got["misses"] != sum(len(set(b)) for b in trace):
                return False, f"expected hits + misses = 7 distinct references, got {got['hits']} + {got['misses']}"
        return True, "expected fifo, lru, and freq to match the reference on a 5-batch trace, got the same"

    c.check("simulate_adapter_pool matches a reference on a small hand trace", small_pool)

    def pool_policies():
        trace, sizes, _ = make_trace()
        results = {}
        for policy in ("fifo", "lru", "freq"):
            got, err = _call(submission, "simulate_adapter_pool", trace, sizes, CAPACITY, policy)
            if err:
                return False, f"expected a result for {policy}, got {err}"
            wm, wb = _ref_pool(trace, sizes, CAPACITY, policy)
            if (got["misses"], got["bytes_loaded"]) != (wm, wb):
                return False, f"expected {policy} misses {wm} and bytes {wb}, got {got['misses']} and {got['bytes_loaded']}"
            results[policy] = got
        best_plain = min(results["fifo"]["load_ms"], results["lru"]["load_ms"])
        ratio = results["freq"]["load_ms"] / best_plain
        c.metric("freq_over_best_plain_load_time", round(ratio, 4))
        c.metric("freq_load_ms", round(results["freq"]["load_ms"], 1))
        return ratio <= 0.9, (
            f"expected freq load time <= 0.9 of the better of fifo and lru "
            f"({best_plain:.1f} ms), got {results['freq']['load_ms']:.1f} ms (ratio {ratio:.3f})"
        )

    c.check("the frequency policy beats FIFO and LRU on the seeded trace with a scan", pool_policies)
    return c.finish()
