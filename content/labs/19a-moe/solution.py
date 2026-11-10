"""Lab 19a: mixture-of-experts routing, dispatch, and expert parallelism.

Reference solution. Everything runs on the CPU in float32. Shapes:

- `x`: (T, d) tokens. `router_w`: (E, d). `logits`: (T, E).
- `gate`, `up`: (E, f, d). `down`: (E, d, f). Each expert is a SwiGLU block.
- `experts`, `weights`, `keep`: (T, k). A row lists one token's k experts.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def route(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k softmax routing with renormalization.

    Softmax over all E experts, keep the k largest, divide by their sum.
    Returns `(weights, experts)`, both (T, k), largest weight first.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    top, experts = torch.topk(probs, k, dim=-1)
    return top / top.sum(dim=-1, keepdim=True), experts


def expert_capacity(tokens: int, k: int, n_experts: int, capacity_factor: float) -> int:
    """Slots per expert: ceil(capacity_factor * tokens * k / n_experts)."""
    return math.ceil(capacity_factor * tokens * k / n_experts)


def apply_capacity(experts: torch.Tensor, n_experts: int, capacity: int) -> torch.Tensor:
    """Boolean (T, k) mask of the assignments that fit.

    Walk the assignments in row-major order, token first and slot second. Each
    expert keeps the first `capacity` it sees and drops the rest.
    """
    flat = experts.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=n_experts)
    starts = torch.cumsum(counts, 0) - counts
    sorted_e = flat[order]
    rank = torch.arange(flat.numel()) - starts[sorted_e]
    keep_sorted = rank < capacity
    keep = torch.zeros(flat.numel(), dtype=torch.bool)
    keep[order] = keep_sorted
    return keep.reshape(experts.shape)


def dispatch(x: torch.Tensor, experts: torch.Tensor, keep: torch.Tensor,
             n_experts: int):
    """Sort the kept (token, slot) pairs by expert, stably.

    Returns `(x_sorted, token_idx, slot_idx, counts)`: one row of `x` per pair,
    the pair's token and slot, and the number of pairs per expert (length E).
    """
    T, k = experts.shape
    flat_keep = keep.reshape(-1)
    tok = torch.arange(T).repeat_interleave(k)[flat_keep]
    slot = torch.arange(k).repeat(T)[flat_keep]
    e = experts.reshape(-1)[flat_keep]
    order = torch.argsort(e, stable=True)
    token_idx, slot_idx = tok[order], slot[order]
    counts = torch.bincount(e, minlength=n_experts)
    return x[token_idx], token_idx, slot_idx, counts


def combine(y_sorted: torch.Tensor, token_idx: torch.Tensor, slot_idx: torch.Tensor,
            weights: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Scale each expert output by its routing weight and add it to its token."""
    out = torch.zeros(n_tokens, y_sorted.shape[1], dtype=y_sorted.dtype)
    scale = weights[token_idx, slot_idx].unsqueeze(1)
    return out.index_add(0, token_idx, y_sorted * scale)


def grouped_matmul_loop(x_sorted: torch.Tensor, counts: torch.Tensor,
                        w: torch.Tensor) -> torch.Tensor:
    """Row segment e of `x_sorted` times `w[e].T`, one matmul per expert.

    `w` is (E, n_out, d_in). An expert with zero rows is skipped.
    """
    out = x_sorted.new_zeros(x_sorted.shape[0], w.shape[1])
    start = 0
    for e, c in enumerate(counts.tolist()):
        if c:
            out[start:start + c] = x_sorted[start:start + c] @ w[e].T
        start += c
    return out


def _round_up(n: int, block: int) -> int:
    return -(-n // block) * block


def padded_rows(counts: torch.Tensor, block: int) -> int:
    """Rows a padded batched matmul computes: E * round_up(max count, block)."""
    return len(counts) * _round_up(int(counts.max()), block)


def grouped_matmul_padded(x_sorted: torch.Tensor, counts: torch.Tensor,
                          w: torch.Tensor, block: int = 8) -> torch.Tensor:
    """The same product as the loop, as one `torch.bmm` over padded segments."""
    E, n_out, d_in = w.shape
    m = _round_up(int(counts.max()), block)
    buf = x_sorted.new_zeros(E, m, d_in)
    start = 0
    spans = []
    for e, c in enumerate(counts.tolist()):
        buf[e, :c] = x_sorted[start:start + c]
        spans.append((e, start, c))
        start += c
    prod = torch.bmm(buf, w.transpose(1, 2))
    out = x_sorted.new_zeros(x_sorted.shape[0], n_out)
    for e, s, c in spans:
        out[s:s + c] = prod[e, :c]
    return out


def moe_layer(x, router_w, gate, up, down, k, capacity=None,
              grouped=grouped_matmul_loop):
    """The whole layer: route, drop, dispatch, three grouped matmuls, combine."""
    n_experts = router_w.shape[0]
    weights, experts = route(x @ router_w.T, k)
    if capacity is None:
        keep = torch.ones_like(experts, dtype=torch.bool)
    else:
        keep = apply_capacity(experts, n_experts, capacity)
    xs, token_idx, slot_idx, counts = dispatch(x, experts, keep, n_experts)
    h = F.silu(grouped(xs, counts, gate)) * grouped(xs, counts, up)
    y = grouped(h, counts, down)
    return combine(y, token_idx, slot_idx, weights, x.shape[0])


def expected_distinct_experts(batch: int, n_experts: int, k: int, probs=None) -> float:
    """Mean number of distinct experts a batch of `batch` tokens touches.

    Each token picks k distinct experts. Expert e is in a token's set with
    probability p_e (k/E for every expert when `probs` is None), and tokens are
    independent, so expert e is missed by the whole batch with probability
    (1 - p_e)^batch.
    """
    if probs is None:
        return n_experts * (1.0 - (1.0 - k / n_experts) ** batch)
    return float(sum(1.0 - (1.0 - float(p)) ** batch for p in probs))


def moe_decode_intensity(batch: int, n_experts: int, k: int, d_model: int,
                         d_ff: int, bytes_per_param: int = 2, probs=None) -> dict:
    """Bytes, FLOPs, and FLOPs per byte of one layer's expert MLPs in a decode step.

    Each touched expert's three matrices are read once. Each token does
    2 * 3 * d_model * d_ff FLOPs in each of its k experts.
    """
    distinct = expected_distinct_experts(batch, n_experts, k, probs)
    nbytes = distinct * 3 * d_model * d_ff * bytes_per_param
    flops = 2.0 * batch * k * 3 * d_model * d_ff
    return {"distinct": distinct, "bytes": nbytes, "flops": flops,
            "intensity": flops / nbytes if nbytes else 0.0}


def batch_to_ridge(n_experts: int, k: int, ridge: float, bytes_per_param: int = 2,
                   probs=None) -> int:
    """Smallest integer batch whose expert-MLP intensity reaches `ridge`."""
    def reaches(b: int) -> bool:
        return moe_decode_intensity(b, n_experts, k, 1, 1, bytes_per_param,
                                    probs)["intensity"] >= ridge

    hi = 1
    while not reaches(hi):
        hi *= 2
    lo = hi // 2 + 1 if hi > 1 else 1
    while lo < hi:
        mid = (lo + hi) // 2
        if reaches(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def all_to_all_matrix(experts: torch.Tensor, token_rank: torch.Tensor,
                      experts_per_rank: int, world: int) -> torch.Tensor:
    """(world, world) counts: entry [s, d] is the (token, slot) pairs that go
    from a token on rank s to an expert on rank d. Expert e lives on rank
    e // experts_per_rank."""
    dest = experts // experts_per_rank
    src = token_rank.unsqueeze(1).expand_as(dest)
    flat = src.reshape(-1) * world + dest.reshape(-1)
    return torch.bincount(flat, minlength=world * world).reshape(world, world)


def all_to_all_bytes(matrix: torch.Tensor, hidden: int, bytes_per_el: int = 2) -> list[int]:
    """Bytes each rank sends to other ranks in the dispatch: off-diagonal row sums."""
    off = matrix.sum(dim=1) - torch.diagonal(matrix)
    return [int(v) * hidden * bytes_per_el for v in off]


def all_to_all_volume(tokens_per_rank: int, k: int, hidden: int, world: int,
                      bytes_per_el: int = 2) -> float:
    """Expected dispatch bytes per rank when routing is uniform.

    Each of the tokens_per_rank * k copies is off-rank with probability
    (world - 1) / world. The combine moves the same amount back.
    """
    return tokens_per_rank * k * hidden * bytes_per_el * (world - 1) / world


def expert_loads(experts: torch.Tensor, n_experts: int) -> list[int]:
    """Assignments per expert."""
    return torch.bincount(experts.reshape(-1), minlength=n_experts).tolist()


def rank_loads(loads: list, experts_per_rank: int) -> list:
    """Sum the per-expert loads over contiguous blocks of experts."""
    return [sum(loads[i:i + experts_per_rank])
            for i in range(0, len(loads), experts_per_rank)]


def imbalance_cost(loads: list) -> dict:
    """A step waits on its busiest rank.

    `factor` is max / mean (1.0 is balanced), `efficiency` is mean / max, and
    `idle_fraction` is 1 - efficiency.
    """
    mx, mean = max(loads), sum(loads) / len(loads)
    eff = mean / mx
    return {"max": mx, "mean": mean, "factor": mx / mean, "efficiency": eff,
            "idle_fraction": 1.0 - eff}


def eplb_plan(loads: list, total_slots: int, world: int) -> dict:
    """Replicate hot experts, then pack the replicas onto ranks.

    Replication: start with one replica per expert. For each spare slot, give a
    replica to the expert with the largest load / replicas, skipping experts that
    already have `world` replicas. Ties go to the lowest expert id.

    Packing: every replica carries load / replicas. Take replicas in order of
    decreasing load, ties by expert id. Put each on the rank with the least load
    that has a free slot (total_slots / world per rank) and does not already hold
    that expert. Ties go to the lowest rank.

    Returns `{"replicas", "ranks", "rank_loads"}`: replicas per expert, the
    expert ids on each rank, and each rank's load.
    """
    n = len(loads)
    slots = total_slots // world
    replicas = [1] * n
    for _ in range(total_slots - n):
        best = None
        for e in range(n):
            if replicas[e] >= world:
                continue
            ratio = loads[e] / replicas[e]
            if best is None or ratio > best[0]:
                best = (ratio, e)
        replicas[best[1]] += 1
    items = []
    for e in range(n):
        items += [(loads[e] / replicas[e], e)] * replicas[e]
    items.sort(key=lambda it: (-it[0], it[1]))
    ranks = [[] for _ in range(world)]
    load = [0.0] * world
    for item_load, e in items:
        cands = [r for r in range(world) if len(ranks[r]) < slots and e not in ranks[r]]
        r = min(cands, key=lambda r: (load[r], r))
        ranks[r].append(e)
        load[r] += item_load
    return {"replicas": replicas, "ranks": ranks, "rank_loads": load}
