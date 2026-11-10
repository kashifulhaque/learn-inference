"""Lab 19b: pipeline schedules and ring attention.

Everything runs on one device. A rank is an index and a ring step is a loop
iteration. `merge_partials` and `ring_all_reduce` are given: they are your
chapter 14a and lab 19 work.
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
KV_BYTES_PER_TOKEN_LAYER = 2 * KV_HEADS * HEAD_DIM * 2  # K and V, bfloat16
KV_BYTES_PER_TOKEN = ATTN_LAYERS * KV_BYTES_PER_TOKEN_LAYER  # 65,536
USABLE_GIB = WEIGHT_BYTES / GIB + 19  # chapter 2: weights plus about 19 GiB


def merge_partials(outs, lses):
    """Given. Chapter 14a's merge. `outs` is (n, ..., dim), `lses` is (n, ...)."""
    peak = lses.amax(dim=0)
    safe_peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
    weights = torch.exp(lses - safe_peak)
    denom = weights.sum(dim=0)
    out = (weights[..., None] * outs).sum(dim=0) / denom.clamp(min=1e-30)[..., None]
    lse = torch.where(denom > 0, safe_peak + torch.log(denom.clamp(min=1e-30)), peak)
    return out, lse


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


def block_attention(q, k, v, q_pos, k_pos, causal: bool = True):
    """Attention of `q` over one block of keys.

    `q` is (heads, tq, dim); `k` and `v` are (heads, tk, dim). `q_pos` and
    `k_pos` hold each token's position in the full sequence. A key at
    position j is visible to a query at position i when j <= i. Return the
    normalized output (heads, tq, dim) and the log-sum-exp (heads, tq). A row
    with no visible key returns zeros and a log-sum-exp of -inf.
    """
    raise NotImplementedError("block_attention")


def shard_indices(total: int, world: int, kind: str = "contiguous") -> list:
    """Token positions each rank owns, as a list of index tensors.

    contiguous: rank r owns one run of total / world tokens.
    striped: rank r owns r, r + world, r + 2 * world, ...
    zigzag: cut into 2 * world chunks; rank r owns chunk r and chunk
    2 * world - 1 - r.
    """
    raise NotImplementedError("shard_indices")


def ring_attention(q, k, v, world: int, kind: str = "contiguous", causal: bool = True):
    """Exact attention with the sequence sharded over `world` simulated ranks.

    `q`, `k`, `v` are (heads, tokens, dim) in token order. At ring step s rank
    r attends to the K and V shard that began on rank (r - s) % world and
    merges the partial result into its running one. Return the output in
    token order and `work`: a list over steps of a list over ranks of the
    (query, key) pairs that rank computed. Skip a block with no visible pair
    and count 0.
    """
    raise NotImplementedError("ring_attention")


def work_balance(work) -> dict:
    """Summarize `work` with `totals` (pairs per rank), `imbalance` (max total
    over mean total), `critical` (sum over steps of the busiest rank's pairs),
    and `efficiency` (mean total over `critical`)."""
    raise NotImplementedError("work_balance")


def memory_per_rank(tokens: int, world: int) -> dict:
    """KV bytes one rank holds for one full-attention layer.

    Return `cache` (the resident shard), `peak` (the shard plus a receive
    buffer when world > 1), and `cache_ratio` and `peak_ratio` against one
    rank holding every token.
    """
    raise NotImplementedError("memory_per_rank")


def min_shard_tokens(link_bps: float, flops: float = PEAK_FLOPS) -> float:
    """Shard length b where a balanced causal ring step's compute equals its
    transfer. Compute is 2 * b**2 * Q_HEADS * HEAD_DIM FLOPs. The transfer is
    b tokens of K and V for one layer."""
    raise NotImplementedError("min_shard_tokens")


def pipeline_schedule(stages: int, microbatches: int) -> list:
    """Fill-and-drain Gantt table: `stages` rows of `microbatches + stages - 1`
    slots. Slot t of row s holds the micro-batch stage s runs then, or -1 when
    idle."""
    raise NotImplementedError("pipeline_schedule")


def bubble_fraction(schedule) -> float:
    """Idle slots divided by all slots in a Gantt table."""
    raise NotImplementedError("bubble_fraction")


def bubble_formula(stages: int, microbatches: int, chunks: int = 1) -> float:
    """(p - 1) / (v * m + p - 1) with v = `chunks` model chunks per stage."""
    raise NotImplementedError("bubble_formula")


def decode_pipeline(stages: int, in_flight: int, stage_time: float, hop_time: float) -> dict:
    """Steady-state decode with `in_flight` micro-batches circulating.

    One trip takes stages * (stage_time + hop_time). At most `stages`
    micro-batches work at once. Return `utilization`, `latency` (one trip),
    and `rate` (trips finished per second).
    """
    raise NotImplementedError("decode_pipeline")


def prefill_flops(tokens: int) -> float:
    """Dense FLOPs (2 per parameter per token) plus causal attention FLOPs in
    the 16 full-attention layers (2 * tokens**2 * Q_HEADS * HEAD_DIM each)."""
    raise NotImplementedError("prefill_flops")


def compare_strategies(tokens: int, world: int, chunks: int = 16) -> dict:
    """One long request on `world` A100s for "dp", "tp", "pp", and "cp".

    Each entry has weights_gib, kv_gib, fits, decode_ms (a batch-1 bandwidth
    floor at 1275 GB/s plus collectives at 300 GB/s and 4 us per hop),
    prefill_comm_gb and prefill_comm_s (bytes one rank sends), and
    prefill_compute_s (at the 312 TFLOP/s peak, stretched by the pipeline
    bubble for "pp" with `chunks` prompt chunks). Raise ValueError when the
    KV heads do not divide by `world`.

    The formulas, with S = HIDDEN * 2 bytes and kv = tokens * KV_BYTES_PER_TOKEN:
    - dp: weights WEIGHT_BYTES, cache kv, decode (W + kv) / COPY_BPS, no comm,
      prefill compute on one GPU.
    - tp: weights and cache divide by `world`. Decode is (W + kv) / world /
      COPY_BPS plus 2 * LAYERS ring all-reduces of S (chapter 19's alpha-beta
      model). Prefill sends LAYERS * tokens * 4 * (world - 1) / world * S.
    - pp: weights and cache divide by `world`. Decode is (W + kv) / COPY_BPS
      plus (world - 1) hops of ALPHA + S / NVLINK_BPS. Prefill sends tokens * S.
    - cp: weights stay whole, cache divides by `world`. Decode is (W + kv /
      world) / COPY_BPS plus ATTN_LAYERS all-gathers: each is (world - 1) hops
      of ALPHA + Q_HEADS * (HEAD_DIM * 4 + 4) / NVLINK_BPS, the float32 partial
      output and log-sum-exp per head. Prefill sends (world - 1) * ATTN_LAYERS *
      (tokens // world) * KV_BYTES_PER_TOKEN_LAYER.
    - Prefill compute is prefill_flops / world / PEAK_FLOPS. For "pp" divide it
      by 1 - bubble_formula(world, chunks).
    - fits: weights plus cache, in GiB, is at most USABLE_GIB.
    """
    raise NotImplementedError("compare_strategies")
