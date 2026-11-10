"""Reference solution for lab 19b: pipeline schedules and ring attention.

Everything runs on one device. A "rank" is an index, and a ring step is a loop
iteration. The model constants are the course's Qwen3.8-27B geometry and the
A100 80GB numbers from chapters 2, 9, 17a, and 19. Nothing here is a measured
multi-GPU time.
"""

from __future__ import annotations

import math

import torch

# Model geometry (chapters 2 and 9) and chapter 19's hardware figures.
LAYERS = 64
ATTN_LAYERS = 16
HIDDEN = 5120
Q_HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
WEIGHT_BYTES = 53.8e9
PARAMS = 26.9e9
COPY_BPS = 1275e9  # measured device-to-device copy
NVLINK_BPS = 300e9  # one way, per A100
PCIE_BPS = 32e9  # one way, PCIe 4.0 x16
ALPHA = 4e-6  # per-hop latency, the assumption chapter 19 uses
PEAK_FLOPS = 312e12  # bfloat16 tensor-core peak
GIB = 2**30
# Bytes of K and V for one token in one full-attention layer.
KV_BYTES_PER_TOKEN_LAYER = 2 * KV_HEADS * HEAD_DIM * 2
KV_BYTES_PER_TOKEN = ATTN_LAYERS * KV_BYTES_PER_TOKEN_LAYER  # 65,536
# Chapter 2's budget: the weights plus about 19 GiB of cache per GPU.
USABLE_GIB = WEIGHT_BYTES / GIB + 19


def ring_all_reduce(shards: list):
    """Given. Lab 19's simulated ring. Returns (results, bytes_sent) per rank."""
    n = len(shards)
    shape = shards[0].shape
    bufs = [s.detach().clone().reshape(-1) for s in shards]
    base, extra = divmod(bufs[0].numel(), n)
    bounds, start = [], 0
    for i in range(n):
        size = base + (1 if i < extra else 0)
        bounds.append((start, start + size))
        start += size
    sent = [0] * n
    size_of = bufs[0].element_size()
    for s in range(n - 1):
        msgs = [(r, (r - s) % n, bufs[r][bounds[(r - s) % n][0]:bounds[(r - s) % n][1]].clone()) for r in range(n)]
        for r, idx, payload in msgs:
            lo, hi = bounds[idx]
            bufs[(r + 1) % n][lo:hi] += payload
            sent[r] += payload.numel() * size_of
    for s in range(n - 1):
        msgs = [(r, (r + 1 - s) % n, bufs[r][bounds[(r + 1 - s) % n][0]:bounds[(r + 1 - s) % n][1]].clone()) for r in range(n)]
        for r, idx, payload in msgs:
            lo, hi = bounds[idx]
            bufs[(r + 1) % n][lo:hi] = payload
            sent[r] += payload.numel() * size_of
    return [b.reshape(shape) for b in bufs], sent



# --- Attention on one block --------------------------------------------------


def block_attention(q, k, v, q_pos, k_pos, causal: bool = True):
    """Attention of `q` over one block of keys.

    `q` is (heads, tq, dim), `k` and `v` are (heads, tk, dim), and `q_pos`
    and `k_pos` hold each token's position in the full sequence. A key at
    position j is visible to a query at position i when j <= i. Returns the
    normalized output (heads, tq, dim) and the log-sum-exp (heads, tq). A row
    with no visible key returns zeros and a log-sum-exp of -inf.
    """
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        visible = k_pos[None, :] <= q_pos[:, None]
        scores = scores.masked_fill(~visible[None], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    safe = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
    weights = torch.exp(scores - safe[..., None])
    return weights @ v, lse


def merge_partials(outs, lses):
    """Chapter 14a's log-sum-exp merge. `outs` is (n, ..., dim), `lses` is (n, ...)."""
    peak = lses.amax(dim=0)
    safe_peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
    weights = torch.exp(lses - safe_peak)
    denom = weights.sum(dim=0)
    out = (weights[..., None] * outs).sum(dim=0) / denom.clamp(min=1e-30)[..., None]
    lse = torch.where(denom > 0, safe_peak + torch.log(denom.clamp(min=1e-30)), peak)
    return out, lse


# --- Ring attention ----------------------------------------------------------


def shard_indices(total: int, world: int, kind: str = "contiguous") -> list:
    """Token positions each rank owns.

    contiguous: rank r owns one run of total / world tokens.
    striped: rank r owns r, r + world, r + 2 * world, ... (Striped Attention).
    zigzag: cut into 2 * world chunks, rank r owns chunk r and chunk
    2 * world - 1 - r (the Llama 3 context-parallel layout).
    """
    if kind == "contiguous":
        if total % world:
            raise ValueError(f"{total} tokens do not divide by {world} ranks")
        size = total // world
        return [torch.arange(r * size, (r + 1) * size) for r in range(world)]
    if kind == "striped":
        if total % world:
            raise ValueError(f"{total} tokens do not divide by {world} ranks")
        return [torch.arange(r, total, world) for r in range(world)]
    if kind == "zigzag":
        if total % (2 * world):
            raise ValueError(f"{total} tokens do not divide by {2 * world} chunks")
        size = total // (2 * world)
        out = []
        for r in range(world):
            a = torch.arange(r * size, (r + 1) * size)
            b = torch.arange((2 * world - 1 - r) * size, (2 * world - r) * size)
            out.append(torch.cat([a, b]))
        return out
    raise ValueError(f"unknown assignment {kind!r}")


def ring_attention(q, k, v, world: int, kind: str = "contiguous", causal: bool = True):
    """Exact attention with the sequence sharded over `world` simulated ranks.

    `q`, `k`, `v` are (heads, tokens, dim) in token order. Rank r keeps its
    query shard and starts with its own K and V shard. At ring step s it holds
    the shard that began on rank (r - s) % world, attends to it, and merges
    the partial result with the log-sum-exp merge. Returns the output in
    token order and `work`, a list over steps of a list over ranks of the
    number of (query, key) pairs that rank computed. A block with no visible
    pair is skipped and counts 0.
    """
    total = q.shape[1]
    owners = shard_indices(total, world, kind)
    q_sh = [q[:, idx] for idx in owners]
    k_sh = [k[:, idx] for idx in owners]
    v_sh = [v[:, idx] for idx in owners]
    run_out = [None] * world
    run_lse = [None] * world
    work = []
    for step in range(world):
        row = []
        for r in range(world):
            src = (r - step) % world
            q_pos, k_pos = owners[r], owners[src]
            if causal:
                visible = k_pos[None, :] <= q_pos[:, None]
                pairs = int(visible.sum())
            else:
                pairs = len(q_pos) * len(k_pos)
            row.append(pairs)
            if pairs == 0:
                continue
            out, lse = block_attention(q_sh[r], k_sh[src], v_sh[src], q_pos, k_pos, causal)
            if run_out[r] is None:
                run_out[r], run_lse[r] = out, lse
            else:
                run_out[r], run_lse[r] = merge_partials(
                    torch.stack([run_out[r], out]), torch.stack([run_lse[r], lse])
                )
        work.append(row)
    result = torch.zeros_like(q)
    for r in range(world):
        result[:, owners[r]] = run_out[r]
    return result, work


def work_balance(work) -> dict:
    """Summarize a work matrix from `ring_attention`.

    `imbalance` is the busiest rank's total pairs over the mean. `critical`
    is the sum over steps of the busiest rank's pairs, because a ring step
    waits for its slowest rank. `efficiency` is the ideal time, the mean
    rank's total, over `critical`.
    """
    world = len(work[0])
    totals = [sum(work[s][r] for s in range(len(work))) for r in range(world)]
    mean = sum(totals) / world
    critical = sum(max(row) for row in work)
    return {
        "totals": totals,
        "imbalance": max(totals) / mean,
        "critical": critical,
        "efficiency": mean / critical,
    }


def memory_per_rank(tokens: int, world: int) -> dict:
    """KV bytes one rank holds for one full-attention layer's K and V.

    `cache` is the shard that stays resident. `peak` adds the receive buffer
    for the next ring step, so with one rank there is no buffer. Ratios are
    against one rank holding every token.
    """
    full = tokens * KV_BYTES_PER_TOKEN_LAYER
    cache = full // world
    peak = cache * (2 if world > 1 else 1)
    return {"cache": cache, "peak": peak, "cache_ratio": cache / full, "peak_ratio": peak / full}


def min_shard_tokens(link_bps: float, flops: float = PEAK_FLOPS) -> float:
    """Shard length at which one ring step's causal compute equals its transfer.

    Compute is 2 * b^2 * heads * dim FLOPs for a balanced causal block of b
    queries and b keys. The transfer is b tokens of K and V for one layer.
    """
    return KV_BYTES_PER_TOKEN_LAYER * flops / (2 * Q_HEADS * HEAD_DIM * link_bps)


# --- Pipeline schedules ------------------------------------------------------


def pipeline_schedule(stages: int, microbatches: int) -> list:
    """Fill-and-drain schedule as a Gantt table.

    Returns `stages` rows of `microbatches + stages - 1` slots. Slot t of row
    s holds the micro-batch stage s runs at time t, or -1 when idle. Stage s
    runs micro-batch t - s.
    """
    slots = microbatches + stages - 1
    return [
        [t - s if 0 <= t - s < microbatches else -1 for t in range(slots)]
        for s in range(stages)
    ]


def bubble_fraction(schedule) -> float:
    """Idle slots divided by all slots in a Gantt table."""
    cells = [c for row in schedule for c in row]
    return sum(1 for c in cells if c == -1) / len(cells)


def bubble_formula(stages: int, microbatches: int, chunks: int = 1) -> float:
    """(p - 1) / (v * m + p - 1) for `chunks` = v model chunks per stage.

    With v = 1 this is GPipe's (p - 1) / (m + p - 1). Megatron's interleaved
    schedule gives each stage v smaller chunks and cuts the bubble time by v.
    """
    return (stages - 1) / (chunks * microbatches + stages - 1)


def decode_pipeline(stages: int, in_flight: int, stage_time: float, hop_time: float) -> dict:
    """Steady-state decode through a pipeline.

    `in_flight` independent micro-batches circulate, one token each per trip.
    One trip takes stages * (stage_time + hop_time), so a sequence's
    inter-token latency never shrinks. At most `stages` micro-batches can be
    working at once. Returns utilization, inter-token latency, and trips
    finished per second.
    """
    trip = stages * (stage_time + hop_time)
    busy = min(in_flight, stages)
    return {"utilization": busy / stages, "latency": trip, "rate": busy / trip}


# --- Long-context comparison -------------------------------------------------


def _allreduce_time(num_bytes, world):
    steps = 2 * (world - 1)
    return steps * ALPHA + steps / world * num_bytes / NVLINK_BPS


def prefill_flops(tokens: int) -> float:
    """Dense FLOPs (2 per parameter per token) plus causal attention FLOPs."""
    dense = 2 * PARAMS * tokens
    attn = 2 * tokens**2 * Q_HEADS * HEAD_DIM * ATTN_LAYERS
    return dense + attn


def compare_strategies(tokens: int, world: int, chunks: int = 16) -> dict:
    """One long request on `world` A100s, per strategy.

    decode_ms is a bandwidth floor at batch 1: bytes each GPU reads at
    1275 GB/s plus the collective time at 300 GB/s and 4 us per hop. The
    prefill numbers are per rank: bytes sent, and the compute time at the
    312 TFLOP/s peak, stretched by the pipeline bubble for pipeline
    parallelism with `chunks` prompt chunks. fits compares weights plus cache
    against chapter 2's budget.
    """
    if KV_HEADS % world:
        raise ValueError(f"{KV_HEADS} KV heads do not divide by {world} ranks")
    kv = tokens * KV_BYTES_PER_TOKEN
    flops = prefill_flops(tokens)
    hidden_bytes = HIDDEN * 2
    ring_bytes = (
        (world - 1) * ATTN_LAYERS * (tokens // world) * KV_BYTES_PER_TOKEN_LAYER
    )
    allgather = (world - 1) * (ALPHA + Q_HEADS * (HEAD_DIM * 4 + 4) / NVLINK_BPS)
    bubble = bubble_formula(world, chunks)
    rows = {
        "dp": dict(w=WEIGHT_BYTES, kv=kv,
                   decode=(WEIGHT_BYTES + kv) / COPY_BPS,
                   comm=0.0, compute=flops / PEAK_FLOPS),
        "tp": dict(w=WEIGHT_BYTES / world, kv=kv / world,
                   decode=(WEIGHT_BYTES + kv) / world / COPY_BPS
                   + 2 * LAYERS * _allreduce_time(hidden_bytes, world),
                   comm=LAYERS * tokens * 4 * (world - 1) / world * hidden_bytes,
                   compute=flops / world / PEAK_FLOPS),
        "pp": dict(w=WEIGHT_BYTES / world, kv=kv / world,
                   decode=(WEIGHT_BYTES + kv) / COPY_BPS
                   + (world - 1) * (ALPHA + hidden_bytes / NVLINK_BPS),
                   comm=tokens * hidden_bytes,
                   compute=flops / world / PEAK_FLOPS / (1 - bubble)),
        "cp": dict(w=WEIGHT_BYTES, kv=kv / world,
                   decode=(WEIGHT_BYTES + kv / world) / COPY_BPS + ATTN_LAYERS * allgather,
                   comm=ring_bytes,
                   compute=flops / world / PEAK_FLOPS),
    }
    out = {}
    for name, r in rows.items():
        out[name] = {
            "weights_gib": r["w"] / GIB,
            "kv_gib": r["kv"] / GIB,
            "fits": (r["w"] + r["kv"]) / GIB <= USABLE_GIB,
            "decode_ms": r["decode"] * 1e3,
            "prefill_comm_gb": r["comm"] / 1e9,
            "prefill_comm_s": r["comm"] / NVLINK_BPS,
            "prefill_compute_s": r["compute"],
        }
    return out
