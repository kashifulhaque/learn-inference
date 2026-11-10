"""Lab 17b: a discrete-event simulator for disaggregated serving.

You write six functions: `kv_transfer_bytes`, `layerwise_exposed_seconds`,
`min_link_gbs`, `simulate_colocated_gpu`, `simulate_disagg_prefill`, and
`best_split`. The cost model, the decode pool, the metrics, and the glue
(`simulate`) are given.


Everything here is a model on a simulated clock. There is no GPU and no
network. Times are seconds, sizes are bytes, link rates are GB/s in one
direction (10^9 bytes per second).

The cost of one forward pass is chapter 17a's floor,

    t = max(F / C, B / beta) + t_launch

with the same counts. `step_time` generalizes it from "a batch of decodes" or
"one whole prompt" to a mixed step: some decodes plus some prefill chunks.
Chapter 17a's batch-1, context-1 decode is 29.962 ms and its 2048-token
prefill is 360.094 ms. `step_time` reproduces both.

The model is chapter 2's: 64 layers, 16 full attention and 48 gated delta.
Full attention sits on every fourth layer, counting from index 3 (chapter 1).
"""

from __future__ import annotations

import math
import random

# Chapter 2's geometry and parameter counts, as in lab 17a.
H = 5120
E = 2
N_LAYERS = 64
N_FULL = 16
N_LINEAR = 48
N_HEADS = 24
N_KV = 4
D_HEAD = 256
N_V_HEADS = 48
D_STATE = 128
N_NORMS = 64 * 2 + 1
CHUNK = 64  # chapter 6's recurrence chunk. Not the scheduler's prefill chunk.

FULL_ATTN_PARAMS = 104_857_600
LINEAR_ATTN_PARAMS = 115_875_840
MLP_PARAMS = 267_386_880
EMBED_PARAMS = 2_542_796_800
NORM_PARAMS = 660_480
HYBRID_PARAMS = (
    N_FULL * (FULL_ATTN_PARAMS + MLP_PARAMS)
    + N_LINEAR * (LINEAR_ATTN_PARAMS + MLP_PARAMS)
    + EMBED_PARAMS
    + NORM_PARAMS
)
DENSE_PARAMS = N_LAYERS * (FULL_ATTN_PARAMS + MLP_PARAMS) + EMBED_PARAMS + NORM_PARAMS

KV_BYTES_PER_LAYER_TOKEN = 2 * N_KV * D_HEAD * E  # 4,096: chapter 2's 4 KiB
LINEAR_LAYER_STATE_BYTES = 3_227_648  # chapter 2: 3,145,728 state + 81,920 conv
STATE_BYTES = LINEAR_LAYER_STATE_BYTES * N_LINEAR  # 154,927,104 = 147.8 MiB

# 17a's hardware, as a dict a caller can change.
A100 = {
    "bf16_tflops": 312.0,
    "hbm_gbs": 1935.0,
    "launches": 400,
    "launch_us": 5.0,
}


# --- Piece 1: what has to cross the link ----------------------------------


def _check_kind(kind: str) -> None:
    if kind not in ("hybrid", "dense"):
        raise ValueError("kind must be 'hybrid' or 'dense'")


def kv_transfer_bytes(kind: str, tokens: int) -> int:
    """Bytes a prefill GPU ships to a decode GPU for one finished prompt.

    Each full-attention layer ships its K and V for every prompt token:
    2 * 4 * 256 * 2 = 4,096 bytes per layer per token. A hybrid has 16 such
    layers, so 65,536 bytes per token. Each of the 48 linear layers ships its
    recurrent state and convolution window, 3,227,648 bytes, however long the
    prompt is. A dense stack (all 64 layers full attention) ships 262,144 bytes
    per token and no state.
    """
    # TODO
    raise NotImplementedError


def layer_is_full(index: int) -> bool:
    """True for the full-attention layers: every fourth, from index 3."""
    return index % 4 == 3


def layer_bytes(kind: str, tokens: int) -> list[int]:
    """Bytes each of the 64 layers ships, in layer order."""
    _check_kind(kind)
    out = []
    for i in range(N_LAYERS):
        if kind == "dense" or layer_is_full(i):
            out.append(KV_BYTES_PER_LAYER_TOKEN * tokens)
        else:
            out.append(LINEAR_LAYER_STATE_BYTES)
    return out


def transfer_seconds(nbytes: float, link_gbs: float, latency_s: float = 0.0) -> float:
    """One bulk copy: a fixed setup latency plus bytes over the link rate."""
    return latency_s + nbytes / (link_gbs * 1e9)


def layerwise_exposed_seconds(
    kind: str,
    tokens: int,
    prefill_s: float,
    link_gbs: float,
    latency_s: float = 0.0,
    link_free_at: float = 0.0,
) -> float:
    """Seconds the transfer keeps the request waiting after the prefill ends.

    Layer i finishes at (i + 1) * prefill_s / 64, taking the prefill as equal
    time per layer. A layer's bytes go onto the link when the layer is done and
    the link is free of earlier layers. Each layer's copy pays `latency_s`.
    The result is the time the last layer lands, minus `prefill_s`, floored at
    0. `link_free_at` is when the link frees, measured from the prefill start.
    """
    # TODO
    raise NotImplementedError


def min_link_gbs(
    kind: str,
    tokens: int,
    budget_s: float,
    prefill_s: float | None = None,
    latency_s: float = 0.0,
) -> float:
    """Smallest one-way link rate, in GB/s, that keeps the wait within `budget_s`.

    With `prefill_s` None the copy is one bulk transfer after the prefill, so
    the answer is closed form: bytes / (budget - latency). With `prefill_s`
    set, layers stream as they finish and the answer is the smallest rate whose
    `layerwise_exposed_seconds` is at most `budget_s`. That function falls as
    the rate rises, so the answer is found by bisection.
    """
    # TODO
    raise NotImplementedError


# --- Piece 2: the cost of one step ----------------------------------------


def step_time(
    decode_batch: int,
    decode_ctx_sum: int,
    chunks: list[tuple[int, int]] | None = None,
    hw: dict | None = None,
) -> float:
    """Seconds for one forward pass with `decode_batch` decodes and prefill chunks.

    `decode_ctx_sum` is the sum of the decodes' context lengths. Each chunk is
    (prefix, n): n new prompt tokens of one prompt that already has `prefix`
    tokens in cache. Weights are read once for the whole step, so decodes and
    chunks share that read. With no chunks this is chapter 17a's decode floor,
    and with one chunk (0, T) and no decodes it is 17a's prefill floor.
    """
    hw = hw or A100
    chunks = chunks or []
    new_tokens = sum(n for _, n in chunks)
    tokens = decode_batch + new_tokens
    if tokens == 0:
        return 0.0
    flops = 2 * HYBRID_PARAMS * tokens
    # Decode attention: chapter 10's 4 * heads * ctx * d per full layer.
    flops += N_FULL * 4 * N_HEADS * D_HEAD * decode_ctx_sum
    flops += N_LINEAR * N_V_HEADS * 7 * D_STATE * D_STATE * decode_batch
    nbytes = HYBRID_PARAMS * E
    nbytes += KV_BYTES_PER_LAYER_TOKEN * N_FULL * decode_ctx_sum
    nbytes += 2 * STATE_BYTES * decode_batch
    per_head_chunked = 7 * CHUNK * D_STATE + 6 * D_STATE * D_STATE
    for prefix, n in chunks:
        # Chunk tokens attend to the cached prefix and, causally, to each other.
        pairs = n * prefix + n * (n + 1) // 2
        flops += N_FULL * 4 * N_HEADS * D_HEAD * pairs
        flops += N_LINEAR * N_V_HEADS * per_head_chunked * n
        nbytes += KV_BYTES_PER_LAYER_TOKEN * N_FULL * (n + prefix)
        nbytes += 2 * STATE_BYTES * ((n + CHUNK - 1) // CHUNK)
    nbytes += (N_LAYERS + N_NORMS) * 2 * tokens * H * E
    launch = hw["launches"] * hw["launch_us"] * 1e-6
    compute = flops / (hw["bf16_tflops"] * 1e12)
    memory = nbytes / (hw["hbm_gbs"] * 1e9)
    return max(compute, memory) + launch


def prefill_seconds(tokens: int, hw: dict | None = None) -> float:
    """A whole prompt in one step, alone on the GPU."""
    return step_time(0, 0, [(0, tokens)], hw)


# --- Piece 3: the simulator -----------------------------------------------


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value with at least q of the data at or below it."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def simulate_decode_pool(arrivals: list[tuple[float, int]], trace: list[dict], max_batch: int,
                          hw: dict, record: dict) -> None:
    """One decode GPU. `arrivals` is (ready_time, request_index), sorted.

    A request is ready with its first token already out. It joins at the next
    step boundary when there is room. Each step is a pure decode.
    """
    running: list[list] = []  # [index, ctx, remaining]
    waiting = list(arrivals)
    t = 0.0
    while waiting or running:
        if not running and waiting and waiting[0][0] > t:
            t = waiting[0][0]
        while waiting and waiting[0][0] <= t and len(running) < max_batch:
            _, idx = waiting.pop(0)
            req = trace[idx]
            if req["output"] > 1:
                running.append([idx, req["prompt"] + 1, req["output"] - 1])
        if not running:
            continue
        dt = step_time(len(running), sum(r[1] for r in running), None, hw)
        t += dt
        still = []
        for r in running:
            record["token_times"][r[0]].append(t)
            r[1] += 1
            r[2] -= 1
            if r[2] > 0:
                still.append(r)
        running = still


def simulate_colocated_gpu(indices: list[int], trace: list[dict], chunk: int, max_batch: int,
                            hw: dict, record: dict) -> None:
    """One colocated GPU: every step carries all decodes plus up to `chunk` prompt tokens.

    `indices` are this GPU's requests. The clock `t` starts at 0 and jumps to
    the next arrival when the GPU is idle. Each step:

    1. Move every request with arrival <= t into the waiting queue, in order.
    2. Spend a budget of `chunk` tokens on waiting prompts, first come first
       served. A prompt takes min(budget, tokens left). A prompt that has not
       started may start only while running + started < `max_batch`.
    3. The step is `step_time(len(running), sum of running contexts, chunks)`,
       where a chunk is (tokens already prefilled, tokens in this chunk). Add
       it to `t`.
    4. Every running decode gets one token at the new `t`: append `t` to
       `record["token_times"][i]` and grow its context by 1. A request leaves
       when it has emitted `output` tokens.
    5. A prompt whose last chunk ran emits its first token at the new `t`, then
       joins the running decodes with context prompt + 1 and output - 1 tokens
       left (a request with output 1 does not join).

    A request's decode context is its prompt plus the tokens emitted so far.
    """
    # TODO
    raise NotImplementedError


def simulate_disagg_prefill(trace: list[dict], config: dict, hw: dict,
                            record: dict) -> list[tuple[float, int]]:
    """The prefill side of a disaggregated run.

    Requests are taken in arrival order. Each runs alone, whole, on the prefill
    GPU that frees first (the lowest index on a tie). Its KV then crosses that
    GPU's own link, and a later transfer from the same GPU queues behind it.
    With `layerwise` false the copy starts after the prefill ends. With it true
    the layers stream, using `layerwise_exposed_seconds` with the link's
    busy-until time measured from the prefill start. The first token is out
    when the copy lands. Append that time to `record["token_times"][i]` and
    return (ready_time, i) for every request.
    """
    # TODO
    raise NotImplementedError


def summarize(trace: list[dict], record: dict, config: dict) -> dict:
    """Turn per-request token times into the metrics `simulate` returns."""
    ttfts, tpots, worst, gaps = [], [], [], []
    for i, req in enumerate(trace):
        times = record["token_times"][i]
        ttfts.append(times[0] - req["arrival"])
        if len(times) > 1:
            g = [b - a for a, b in zip(times, times[1:])]
            gaps.extend(g)
            tpots.append((times[-1] - times[0]) / (len(times) - 1))
            # The first gap holds the handoff, so the worst gap starts after it.
            worst.append(max(g[1:]) if len(g) > 1 else 0.0)
        else:
            tpots.append(0.0)
            worst.append(0.0)
    finish = max(times[-1] for times in record["token_times"])
    span = finish - min(req["arrival"] for req in trace)
    out = {
        "p50_ttft": percentile(ttfts, 0.50),
        "p99_ttft": percentile(ttfts, 0.99),
        "p99_tpot": percentile(tpots, 0.99),
        "p99_worst_gap": percentile(worst, 0.99),
        "p99_gap": percentile(gaps, 0.99),
        "max_gap": max(gaps) if gaps else 0.0,
        "span": span,
    }
    if "slo_ttft" in config and "slo_tpot" in config:
        per_request = worst if config.get("tpot_stat", "mean") == "max" else tpots
        ok = sum(1 for a, b in zip(ttfts, per_request)
                 if a <= config["slo_ttft"] and b <= config["slo_tpot"])
        out["met"] = ok
        out["goodput"] = ok / span
    return out


def simulate(trace: list[dict], config: dict, hw: dict | None = None) -> dict:
    """Replay `trace` on a simulated clock and return latency and goodput numbers.

    A trace entry is {"arrival": s, "prompt": tokens, "output": tokens}. The
    first output token comes out of the prompt's last prefill step.

    Colocated: {"mode": "colocated", "gpus": n, "chunk": tokens, "max_batch": b}.
    Requests go to the n GPUs in arrival order, round robin. Each GPU runs
    steps of all its decodes plus up to `chunk` prompt tokens.

    Disaggregated: {"mode": "disagg", "prefill_gpus": p, "decode_gpus": d,
    "link_gbs": g, "max_batch": b, "layerwise": bool, "link_latency_s": s}.
    Ready requests go to the d decode GPUs in ready order, round robin.

    Optional `slo_ttft`, `slo_tpot`, and `tpot_stat` ("mean", the default, or
    "max") add `met` and `goodput`. TTFT includes the wait for the KV copy.
    """
    hw = hw or A100
    max_batch = int(config.get("max_batch", 64))
    n = len(trace)
    record: dict = {"token_times": [[] for _ in range(n)]}
    if config["mode"] == "colocated":
        gpus = int(config["gpus"])
        order = sorted(range(n), key=lambda i: (trace[i]["arrival"], i))
        groups: list[list[int]] = [[] for _ in range(gpus)]
        for k, i in enumerate(order):
            groups[k % gpus].append(i)
        for g in groups:
            simulate_colocated_gpu(g, trace, int(config["chunk"]), max_batch, hw, record)
    elif config["mode"] == "disagg":
        d = int(config["decode_gpus"])
        ready = sorted(simulate_disagg_prefill(trace, config, hw, record))
        pools: list[list[tuple[float, int]]] = [[] for _ in range(d)]
        for k, item in enumerate(ready):
            pools[k % d].append(item)
        for pool in pools:
            simulate_decode_pool(pool, trace, max_batch, hw, record)
    else:
        raise ValueError("mode must be 'colocated' or 'disagg'")
    return summarize(trace, record, config)


def poisson_trace(n: int, rate: float, prompt: int, output: int, seed: int) -> list[dict]:
    """n requests with exponential gaps of mean 1/rate, all the same shape."""
    rng = random.Random(seed)
    t = 0.0
    out = []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append({"arrival": t, "prompt": prompt, "output": output})
    return out


# --- Piece 4: choosing the split ------------------------------------------


def best_split(total_gpus: int, trace: list[dict], config: dict, hw: dict | None = None) -> dict:
    """The (prefill_gpus, decode_gpus) split with the highest goodput.

    `config` is a disaggregated config without the two GPU counts, and it
    carries `slo_ttft` and `slo_tpot`. The search tries every split with at
    least one GPU on each side. A tie goes to the split with fewer prefill
    GPUs. Returns {"prefill_gpus", "decode_gpus", "goodput", "met"}.
    """
    # TODO
    raise NotImplementedError


def capacity_split(total_gpus: int, trace: list[dict], slo_tpot: float, max_batch: int = 64,
                   hw: dict | None = None) -> int:
    """Prefill GPUs that match the two stages' work, without running the simulator.

    A prefill GPU spends prefill_seconds(prompt) per request. A decode GPU runs
    the largest batch b whose step meets `slo_tpot` (at least 1, at most
    `max_batch`), so it spends (output - 1) * step(b) / b per request. Split
    the GPUs in proportion to the two totals, round to the nearest integer, and
    keep at least one on each side.
    """
    hw = hw or A100
    n = len(trace)
    pre = sum(prefill_seconds(r["prompt"], hw) for r in trace)
    mean_ctx = sum(r["prompt"] + r["output"] / 2 for r in trace) / n
    b = 1
    for cand in range(1, max_batch + 1):
        if step_time(cand, int(cand * mean_ctx), None, hw) <= slo_tpot:
            b = cand
    step = step_time(b, int(b * mean_ctx), None, hw)
    dec = sum((r["output"] - 1) * step / b for r in trace)
    p = round(total_gpus * pre / (pre + dec))
    return min(max(p, 1), total_gpus - 1)
