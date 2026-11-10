"""Lab 17a — a step-time model for capacity planning.

The time this file returns is a floor built from chapters 2, 6, 10, and 19.
It is not a measurement. No term uses a peak, a price, or a latency that those
chapters do not already state.

Decode and prefill:

    t = max(F / C, B / beta) + t_comm + t_launch

C is the rated bfloat16 tensor-core peak. beta is the bandwidth you pass in
(the rated 1935 GB/s for the floor in chapter 10). t_launch is chapter 10's
worked example, 400 launches at 5 µs. t_comm is 0 at tensor-parallel degree 1.
Otherwise it is chapter 19's 128 all-reduces at 8 µs each, plus the ring
all-reduce bytes at the link rate you pass in (300 GB/s is A100 NVLink one way in that
chapter; 600 GB/s is the datasheet total of both directions).

`kind` is "hybrid" (16 full-attention layers and 48 linear layers) or "dense"
(chapter 2's hypothetical stack in which all 64 layers are full attention).
"""

from __future__ import annotations

# Chapter 2's geometry and parameter counts.
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
N_NORMS = 64 * 2 + 1  # chapter 10: two norms per layer, plus the final norm
CHUNK = 64  # chapter 6's prefill chunk

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
DENSE_PARAMS = (
    N_LAYERS * (FULL_ATTN_PARAMS + MLP_PARAMS)
    + EMBED_PARAMS
    + NORM_PARAMS
)

# One full-attention layer, K and V, bfloat16: chapter 2's 4 KiB.
KV_BYTES_PER_LAYER = 2 * N_KV * D_HEAD * E
# Chapter 2: 3,227,648 bytes per linear layer, 48 layers, 154,927,104 bytes.
STATE_BYTES = 3_227_648 * N_LINEAR

# Chapter 19: two all-reduces of a 10,240-byte hidden state, before the ring
# factor (P-1)/P. 2 * 2 * 10240 = 40960 bytes per token per layer at P infinite.
AR_BYTES_PER_LAYER = 2 * 2 * 10_240


def _kind(cfg: dict) -> str:
    kind = cfg["kind"]
    if kind not in ("hybrid", "dense"):
        raise ValueError("kind must be 'hybrid' or 'dense'")
    return kind


def _tp(hw: dict) -> int:
    return int(hw.get("tp", 1))


def _params(kind: str) -> int:
    return HYBRID_PARAMS if kind == "hybrid" else DENSE_PARAMS


def _full_layers(kind: str) -> int:
    return N_FULL if kind == "hybrid" else N_LAYERS


def _weight_bytes(kind: str) -> int:
    return _params(kind) * E


def _state_stored(kind: str, sequences: int) -> int:
    if kind != "hybrid":
        return 0
    return STATE_BYTES * sequences


def _activation_bytes(tokens: int) -> int:
    """Hidden-state traffic that stays replicated across tensor-parallel ranks.

    Chapter 10 counts an MLP's input and output, and an RMSNorm's read and
    write. Chapter 19's all-reduce reassembles that hidden state on every
    rank, so this traffic is not divided by the tensor-parallel degree.
    """
    per_tensor = 2 * tokens * H * E
    return (N_LAYERS + N_NORMS) * per_tensor


def decode_flops(kind: str, batch: int, ctx: int) -> int:
    """FLOPs for one decode step of `batch` sequences at context `ctx`.

    Matmul FLOPs are 2 per parameter per token, the count chapter 19 uses.
    Full-attention scores and values add chapter 10's 4 * heads * ctx * d
    per layer. A linear layer adds chapter 6's sequential 7 d^2 per value head.
    """
    flops = 2 * _params(kind) * batch
    flops += _full_layers(kind) * (4 * N_HEADS * ctx * D_HEAD) * batch
    if kind == "hybrid":
        flops += N_LINEAR * N_V_HEADS * (7 * D_STATE * D_STATE) * batch
    return flops


def decode_traffic_bytes(kind: str, batch: int, ctx: int, tp: int) -> float:
    """HBM bytes moved on one rank during a decode step.

    Weights and the KV read are sharded by `tp`. The recurrent state is read
    and written, then sharded the same way. Activations stay replicated.
    """
    sharded = (
        _weight_bytes(kind)
        + KV_BYTES_PER_LAYER * _full_layers(kind) * ctx * batch
        + 2 * _state_stored(kind, batch)
    )
    return sharded / tp + _activation_bytes(batch)


def resident_bytes(kind: str, batch: int, ctx: int, tp: int) -> float:
    """Bytes that have to sit in one rank's memory: weights, KV, and state.

    The state is stored once. The extra write in the traffic count is bus
    traffic, not a second copy. Activations are transient and are not reserved,
    which is chapter 2's budget.
    """
    stored = (
        _weight_bytes(kind)
        + KV_BYTES_PER_LAYER * _full_layers(kind) * ctx * batch
        + _state_stored(kind, batch)
    )
    return stored / tp


def prefill_flops(kind: str, tokens: int) -> int:
    """FLOPs for one sequence of `tokens` prompt tokens.

    Causal attention extends chapter 10's one-query count over prefixes
    1..tokens. Linear layers use chapter 6's chunked per-token count, because
    prefill runs the chunked form.
    """
    flops = 2 * _params(kind) * tokens
    triangle = tokens * (tokens + 1) // 2
    flops += _full_layers(kind) * (4 * N_HEADS * D_HEAD * triangle)
    if kind == "hybrid":
        per_head = 7 * CHUNK * D_STATE + 6 * D_STATE * D_STATE
        flops += N_LINEAR * N_V_HEADS * per_head * tokens
    return flops


def prefill_traffic_bytes(kind: str, tokens: int, tp: int) -> float:
    """HBM bytes for one sequence's prefill.

    The KV cache is written once. A tiled attention kernel does not reread the
    whole prefix per query, so this count does not add an O(tokens^2) read.
    The recurrent state is read and written once per chunk of 64.
    """
    kv_write = KV_BYTES_PER_LAYER * _full_layers(kind) * tokens
    if kind == "hybrid":
        chunks = (tokens + CHUNK - 1) // CHUNK
        state = 2 * STATE_BYTES * chunks
    else:
        state = 0
    sharded = _weight_bytes(kind) + kv_write + state
    return sharded / tp + _activation_bytes(tokens)


def comm_seconds(hw: dict, tokens: int, tp: int) -> float:
    """Chapter 19's all-reduce time. Zero when the degree is 1."""
    if tp <= 1:
        return 0.0
    comm_bytes = N_LAYERS * tokens * AR_BYTES_PER_LAYER * (tp - 1) / tp
    latency = (N_LAYERS * 2) * hw["allreduce_us"] * 1e-6
    return latency + comm_bytes / (hw["link_gbs"] * 1e9)


def launch_seconds(hw: dict) -> float:
    return hw["launches"] * hw["launch_us"] * 1e-6


def _step(flops: float, nbytes: float, hw: dict, comm_tokens: int, tp: int) -> float:
    compute = flops / (hw["bf16_tflops"] * 1e12)
    memory = nbytes / (hw["hbm_gbs"] * 1e9)
    return max(compute, memory) + comm_seconds(hw, comm_tokens, tp) + launch_seconds(hw)


def decode_step_time(cfg: dict, hw: dict, batch: int, ctx: int) -> float:
    """Seconds for one decode step. A floor, not a device measurement."""
    kind = _kind(cfg)
    tp = _tp(hw)
    flops = decode_flops(kind, batch, ctx) / tp
    nbytes = decode_traffic_bytes(kind, batch, ctx, tp)
    return _step(flops, nbytes, hw, batch, tp)


def prefill_time(cfg: dict, hw: dict, tokens: int) -> float:
    """Seconds to prefill one sequence of `tokens` tokens."""
    kind = _kind(cfg)
    tp = _tp(hw)
    flops = prefill_flops(kind, tokens) / tp
    nbytes = prefill_traffic_bytes(kind, tokens, tp)
    return _step(flops, nbytes, hw, tokens, tp)


def max_goodput(cfg: dict, hw: dict, slo_ttft: float, slo_tpot: float, workload: dict) -> float:
    """Largest decode throughput, in output tokens per second, that meets both SLOs.

    Time to first token is the prefill of one prompt. Time per output token is
    the decode step at the workload's context length. A batch that does not
    fit in memory, or whose step is slower than the TPOT SLO, contributes
    nothing. Returns 0 when the prefill misses the TTFT SLO.
    """
    if prefill_time(cfg, hw, workload["prompt"]) > slo_ttft:
        return 0.0
    kind = _kind(cfg)
    tp = _tp(hw)
    ctx = int(workload["ctx"])
    best = 0.0
    for batch in range(1, int(workload["max_batch"]) + 1):
        if resident_bytes(kind, batch, ctx, tp) > hw["memory_bytes"]:
            continue
        step = decode_step_time(cfg, hw, batch, ctx)
        if step <= slo_tpot:
            best = max(best, batch / step)
    return best


def tokens_per_dollar(tokens_per_s: float, usd_per_gpu_hour: float, gpus: int) -> float:
    """Output tokens per dollar at a rate you supply.

    `usd_per_gpu_hour` is your price. This course has no citable A100 hourly
    rate, so the function does not contain one.
    """
    return tokens_per_s * 3600.0 / (usd_per_gpu_hour * gpus)


def fit_efficiency(rows: list[dict], hbm_gbs: float) -> dict:
    """Recover a bandwidth fraction and a fixed overhead from a memory-bound trace.

    Each row has "bytes", "flops", and "seconds". The fit uses bytes only:

        seconds = bytes / (alpha * beta) + overhead

    which is a straight line in bytes/beta. "flops" is present so a fit that
    prices the step at the arithmetic peak has the number it would misuse.
    The line is ordinary least squares. Call this on a trace whose rows are
    memory-bound; a compute-bound row is a different model.
    """
    beta = hbm_gbs * 1e9
    xs = [row["bytes"] / beta for row in rows]
    ys = [row["seconds"] for row in rows]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    overhead = mean_y - slope * mean_x
    return {"alpha": 1.0 / slope, "overhead_s": overhead}


def choose_layout(
    cfg: dict, hw: dict, n_gpus: int, slo_ttft: float, slo_tpot: float, workload: dict
) -> dict:
    """Pick a tensor-parallel degree and a replica count for `n_gpus` GPUs.

    The degree is 1, 2, or 4, because this model has 4 KV heads (chapter 19).
    Replicas are independent copies: n_gpus / degree. The winner is the layout
    with the higher aggregate tokens per second. An exact tie prefers more
    replicas.
    """
    best: tuple[float, int, int] | None = None
    for tp in (1, 2, 4):
        if n_gpus % tp != 0:
            continue
        replicas = n_gpus // tp
        hw_p = dict(hw)
        hw_p["tp"] = tp
        per = max_goodput(cfg, hw_p, slo_ttft, slo_tpot, workload)
        total = per * replicas
        cand = (total, replicas, tp)
        if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
            best = cand
    assert best is not None
    return {"tp": best[2], "replicas": best[1], "tokens_per_s": best[0]}
