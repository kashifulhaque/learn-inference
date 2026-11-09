"""Checks for the step-time model, the SLO search, and the planted fit.

The first check calls only `decode_step_time`. Later checks add prefill, the
dollar conversion, the goodput search, the fit, and the layout search.
Randomness is seed 17, and it is used only to build the synthetic trace.

A plausible wrong implementation prices a memory-bound decode step at the
arithmetic peak, F/C, and ignores the byte term. At batch 1 that peak is a
fraction of a millisecond, and the memory floor is about 28 ms before the
launch term, so the two are far outside the tolerance below.

The tolerance on a step time is 1 ns. Chapter 10's batch-1 RMSNorm moves
20,480 bytes, which is 10.6 ns at 1935 GB/s, and the launch term is 5 µs.
One nanosecond is below both, and below the KV read of a single token
(65,536 / 1.935e12 seconds, about 34 ns). A dropped term misses by more than
1 ns. This is not a percentage of a measured step time: the course has no
bundled table of A100 step times.

Another plausible wrong implementation fits the fixed overhead with the wrong
sign, or regresses on FLOPs instead of bytes. The planted overhead is
positive. The tolerance around it is six standard errors of the least-squares
estimator at the planted noise, derived below.
"""

import math
import random

from lab_common import Checks

# Chapter 2, chapter 6, and chapter 10. Same counts as the solution.
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
N_NORMS = 129
CHUNK = 64
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
KV_BYTES_PER_LAYER = 2 * N_KV * D_HEAD * E
STATE_BYTES = 3_227_648 * N_LINEAR
AR_BYTES_PER_LAYER = 2 * 2 * 10_240

# 1 ns. See the module docstring for why this is the tolerance.
STEP_TOL = 1e-9

PLANTED_ALPHA = 0.7
PLANTED_TAU = 0.002
NOISE_SIGMA = 5e-6
FIT_SEED = 17


def a100():
    return {
        "bf16_tflops": 312.0,
        "hbm_gbs": 1935.0,
        "launch_us": 5.0,
        "launches": 400,
        "allreduce_us": 8.0,
        "link_gbs": 600.0,
        "memory_bytes": 80 * 1024**3,
        "tp": 1,
    }


def _params(kind):
    return HYBRID_PARAMS if kind == "hybrid" else DENSE_PARAMS


def _full_layers(kind):
    return N_FULL if kind == "hybrid" else N_LAYERS


def _activation_bytes(tokens):
    return (N_LAYERS + N_NORMS) * (2 * tokens * H * E)


def _state_stored(kind, sequences):
    return STATE_BYTES * sequences if kind == "hybrid" else 0


def ref_decode_flops(kind, batch, ctx):
    flops = 2 * _params(kind) * batch
    flops += _full_layers(kind) * (4 * N_HEADS * ctx * D_HEAD) * batch
    if kind == "hybrid":
        flops += N_LINEAR * N_V_HEADS * (7 * D_STATE * D_STATE) * batch
    return flops


def ref_decode_traffic(kind, batch, ctx, tp):
    sharded = (
        _params(kind) * E
        + KV_BYTES_PER_LAYER * _full_layers(kind) * ctx * batch
        + 2 * _state_stored(kind, batch)
    )
    return sharded / tp + _activation_bytes(batch)


def ref_resident(kind, batch, ctx, tp):
    stored = (
        _params(kind) * E
        + KV_BYTES_PER_LAYER * _full_layers(kind) * ctx * batch
        + _state_stored(kind, batch)
    )
    return stored / tp


def ref_prefill_flops(kind, tokens):
    flops = 2 * _params(kind) * tokens
    triangle = tokens * (tokens + 1) // 2
    flops += _full_layers(kind) * (4 * N_HEADS * D_HEAD * triangle)
    if kind == "hybrid":
        per_head = 7 * CHUNK * D_STATE + 6 * D_STATE * D_STATE
        flops += N_LINEAR * N_V_HEADS * per_head * tokens
    return flops


def ref_prefill_traffic(kind, tokens, tp):
    kv_write = KV_BYTES_PER_LAYER * _full_layers(kind) * tokens
    if kind == "hybrid":
        chunks = (tokens + CHUNK - 1) // CHUNK
        state = 2 * STATE_BYTES * chunks
    else:
        state = 0
    sharded = _params(kind) * E + kv_write + state
    return sharded / tp + _activation_bytes(tokens)


def ref_comm(hw, tokens, tp):
    if tp <= 1:
        return 0.0
    comm_bytes = N_LAYERS * tokens * AR_BYTES_PER_LAYER * (tp - 1) / tp
    latency = (N_LAYERS * 2) * hw["allreduce_us"] * 1e-6
    return latency + comm_bytes / (hw["link_gbs"] * 1e9)


def ref_launch(hw):
    return hw["launches"] * hw["launch_us"] * 1e-6


def ref_step(flops, nbytes, hw, comm_tokens, tp):
    compute = flops / (hw["bf16_tflops"] * 1e12)
    memory = nbytes / (hw["hbm_gbs"] * 1e9)
    return max(compute, memory) + ref_comm(hw, comm_tokens, tp) + ref_launch(hw)


def ref_decode(cfg, hw, batch, ctx):
    tp = int(hw.get("tp", 1))
    kind = cfg["kind"]
    return ref_step(ref_decode_flops(kind, batch, ctx) / tp,
                    ref_decode_traffic(kind, batch, ctx, tp), hw, batch, tp)


def ref_prefill(cfg, hw, tokens):
    tp = int(hw.get("tp", 1))
    kind = cfg["kind"]
    return ref_step(ref_prefill_flops(kind, tokens) / tp,
                    ref_prefill_traffic(kind, tokens, tp), hw, tokens, tp)


def ref_goodput(cfg, hw, slo_ttft, slo_tpot, workload):
    """Brute-force search over batch. The submission's search must match it."""
    if ref_prefill(cfg, hw, workload["prompt"]) > slo_ttft:
        return 0.0
    kind = cfg["kind"]
    tp = int(hw.get("tp", 1))
    ctx = int(workload["ctx"])
    best = 0.0
    for batch in range(1, int(workload["max_batch"]) + 1):
        if ref_resident(kind, batch, ctx, tp) > hw["memory_bytes"]:
            continue
        step = ref_decode(cfg, hw, batch, ctx)
        if step <= slo_tpot:
            best = max(best, batch / step)
    return best


def ref_layout(cfg, hw, n_gpus, slo_ttft, slo_tpot, workload):
    best = None
    for tp in (1, 2, 4):
        if n_gpus % tp != 0:
            continue
        replicas = n_gpus // tp
        hw_p = dict(hw)
        hw_p["tp"] = tp
        total = ref_goodput(cfg, hw_p, slo_ttft, slo_tpot, workload) * replicas
        cand = (total, replicas, tp)
        if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
            best = cand
    return {"tp": best[2], "replicas": best[1], "tokens_per_s": best[0]}


def _gauss(rng):
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def synthetic_rows():
    """Memory-bound rows from the model, with planted alpha, overhead, and noise.

    The seconds are not a device measurement. A real trace would replace them.
    """
    rng = random.Random(FIT_SEED)
    beta = a100()["hbm_gbs"] * 1e9
    rows = []
    for batch in (1, 2, 4, 8, 16):
        flops = ref_decode_flops("hybrid", batch, 128)
        nbytes = ref_decode_traffic("hybrid", batch, 128, 1)
        seconds = nbytes / (PLANTED_ALPHA * beta) + PLANTED_TAU + NOISE_SIGMA * _gauss(rng)
        rows.append({"bytes": nbytes, "flops": flops, "seconds": seconds})
    return rows


def fit_tolerance(rows):
    """Six standard errors of the planted least-squares estimator.

    The model is y = x / alpha + tau + noise, x = bytes / beta, noise has
    standard deviation NOISE_SIGMA. For ordinary least squares the slope
    variance is sigma^2 / Sxx and the intercept variance is
    sigma^2 (1/n + mean(x)^2 / Sxx). Alpha is 1/slope, so the delta method
    divides the slope's standard error by slope^2. Six standard errors is
    the acceptance window around the planted constants.
    """
    beta = a100()["hbm_gbs"] * 1e9
    xs = [row["bytes"] / beta for row in rows]
    n = len(xs)
    mean_x = sum(xs) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    slope = 1.0 / PLANTED_ALPHA
    se_slope = NOISE_SIGMA / math.sqrt(sxx)
    se_tau = NOISE_SIGMA * math.sqrt(1.0 / n + mean_x**2 / sxx)
    se_alpha = se_slope / (slope ** 2)
    return 6.0 * se_alpha, 6.0 * se_tau


def _call(fn, expected, tol=None):
    try:
        actual = fn()
    except Exception as exc:  # noqa: BLE001
        return False, f"expected {expected}, got {type(exc).__name__}: {exc}"
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            return False, f"expected {expected}, got {actual}"
        for key, want in expected.items():
            got = actual[key]
            limit = tol if isinstance(want, float) else 0
            if isinstance(want, float):
                if abs(got - want) > limit:
                    return False, f"expected {expected}, got {actual}"
            elif got != want:
                return False, f"expected {expected}, got {actual}"
        return True, f"expected {expected}, got {actual}"
    if tol is None:
        ok = actual == expected
    else:
        ok = abs(actual - expected) <= tol
    return ok, f"expected {expected}, got {actual}"


def run(submission):
    c = Checks()
    hw = a100()
    hybrid = {"kind": "hybrid"}
    dense = {"kind": "dense"}

    # Expanded from the chapters, for batch 1 and context 1, degree 1.
    # Weights: chapter 2's 26,895,979,520 parameters at 2 bytes.
    # KV: chapter 2's 64 KiB per token, one token.
    # State: read and write of chapter 2's 154,927,104 bytes.
    # Activations: chapter 10's MLP input and output on 64 layers, plus 129 norms.
    # FLOPs: 2 per parameter, plus chapter 10's attention at L=1, plus chapter 6's
    # 7 d^2 on 48 heads of 48 layers.
    weight = HYBRID_PARAMS * E
    kv = KV_BYTES_PER_LAYER * N_FULL * 1 * 1
    state_rw = 2 * STATE_BYTES
    act = (N_LAYERS + N_NORMS) * (2 * 1 * H * E)
    flops = (
        2 * HYBRID_PARAMS
        + N_FULL * (4 * N_HEADS * 1 * D_HEAD)
        + N_LINEAR * N_V_HEADS * (7 * D_STATE * D_STATE)
    )
    beta = hw["hbm_gbs"] * 1e9
    peak = hw["bf16_tflops"] * 1e12
    launch = hw["launches"] * hw["launch_us"] * 1e-6
    memory_s = (weight + kv + state_rw + act) / beta
    compute_s = flops / peak
    expected_decode = max(compute_s, memory_s) + launch

    def check_decode_batch1():
        # The arithmetic-peak implementation returns compute_s + launch and
        # drops the memory term. It must fall outside STEP_TOL.
        wrong = compute_s + launch
        if abs(wrong - expected_decode) <= STEP_TOL:
            return False, (
                f"expected the memory floor {expected_decode} to differ from the "
                f"arithmetic peak {wrong} by more than {STEP_TOL}"
            )
        return _call(lambda: submission.decode_step_time(hybrid, hw, 1, 1),
                     expected_decode, STEP_TOL)

    c.check(
        "decode at batch 1 matches the memory floor plus the launch",
        check_decode_batch1,
    )

    dense_weight = DENSE_PARAMS * E
    dense_kv = KV_BYTES_PER_LAYER * N_LAYERS * 1
    dense_flops = 2 * DENSE_PARAMS + N_LAYERS * (4 * N_HEADS * 1 * D_HEAD)
    dense_act = act
    dense_expected = max(dense_flops / peak, (dense_weight + dense_kv + dense_act) / beta) + launch

    def check_dense():
        got_h = None
        try:
            got_d = submission.decode_step_time(dense, hw, 1, 1)
            got_h = submission.decode_step_time(hybrid, hw, 1, 1)
        except Exception as exc:  # noqa: BLE001
            return False, f"expected {dense_expected}, got {type(exc).__name__}: {exc}"
        if abs(got_d - got_h) <= STEP_TOL:
            return False, (
                f"expected the dense step {dense_expected} to differ from the hybrid "
                f"step {got_h}"
            )
        if abs(got_d - dense_expected) > STEP_TOL:
            return False, f"expected {dense_expected}, got {got_d}"
        return True, f"expected {dense_expected}, got {got_d}"

    c.check("a dense stack at batch 1 uses 64 KV layers and no recurrent state", check_dense)

    # Prefill of 2048 tokens. The quadratic attention term is chapter 10's
    # one-query count summed over prefixes 1..2048. The linear layers use
    # chapter 6's chunked per-token FLOPs.
    tokens = 2048
    triangle = tokens * (tokens + 1) // 2
    prefill_flops = (
        2 * HYBRID_PARAMS * tokens
        + N_FULL * (4 * N_HEADS * D_HEAD * triangle)
        + N_LINEAR * N_V_HEADS * (7 * CHUNK * D_STATE + 6 * D_STATE * D_STATE) * tokens
    )
    chunks = (tokens + CHUNK - 1) // CHUNK
    prefill_bytes = (
        HYBRID_PARAMS * E
        + KV_BYTES_PER_LAYER * N_FULL * tokens
        + 2 * STATE_BYTES * chunks
        + (N_LAYERS + N_NORMS) * (2 * tokens * H * E)
    )
    prefill_compute = prefill_flops / peak
    prefill_memory = prefill_bytes / beta
    prefill_expected = max(prefill_compute, prefill_memory) + launch

    def check_prefill():
        try:
            got = submission.prefill_time(hybrid, hw, tokens)
        except Exception as exc:  # noqa: BLE001
            return False, f"expected {prefill_expected}, got {type(exc).__name__}: {exc}"
        if prefill_compute <= prefill_memory:
            return False, (
                f"expected prefill compute {prefill_compute} to exceed memory "
                f"{prefill_memory} at {tokens} tokens"
            )
        if abs(got - prefill_expected) > STEP_TOL:
            return False, f"expected {prefill_expected}, got {got}"
        return True, f"expected {prefill_expected}, got {got}"

    c.check("a 2048-token prefill matches the floor and is compute-bound", check_prefill)

    def check_dollars():
        # 1000 tokens/s, $2.50 per GPU-hour, 4 GPUs: 1000 * 3600 / 10.
        expected = 1000 * 3600 / (2.5 * 4)
        return _call(lambda: submission.tokens_per_dollar(1000, 2.5, 4), expected, 1e-9)

    c.check("tokens per dollar uses the price you pass in", check_dollars)

    workload = {"prompt": 128, "ctx": 1024, "max_batch": 64}
    loose = (1.0, 1.0)
    # A TPOT SLO between the batch-1 step and a large-batch step, so the search
    # has to stop before the memory cap. Computed from the reference, not copied
    # from a measurement.
    step_1 = ref_decode(hybrid, hw, 1, workload["ctx"])
    step_8 = ref_decode(hybrid, hw, 8, workload["ctx"])
    tight_tpot = (step_1 + step_8) / 2
    ttft = ref_prefill(hybrid, hw, workload["prompt"])

    def check_goodput():
        cases = (
            ("loose", loose[0], loose[1]),
            ("tight TPOT", ttft * 2, tight_tpot),
            ("TTFT miss", ttft / 2, loose[1]),
        )
        for name, slo_ttft, slo_tpot in cases:
            expected = ref_goodput(hybrid, hw, slo_ttft, slo_tpot, workload)
            try:
                got = submission.max_goodput(hybrid, hw, slo_ttft, slo_tpot, workload)
            except Exception as exc:  # noqa: BLE001
                return False, f"expected {expected} ({name}), got {type(exc).__name__}: {exc}"
            if abs(got - expected) > 1e-6:
                return False, f"expected {expected} ({name}), got {got}"
            if name == "TTFT miss" and expected != 0.0:
                return False, f"expected 0 for a missed TTFT, got {expected}"
        expected = ref_goodput(hybrid, hw, loose[0], loose[1], workload)
        return True, f"expected {expected} on the loose SLO, got {expected}"

    c.check("max goodput matches a brute-force batch search", check_goodput)

    def check_monotonic():
        prev_tpot = -1.0
        last = None
        for slo_tpot in (step_1, tight_tpot, step_8 * 2, 1.0):
            expected = ref_goodput(hybrid, hw, 1.0, slo_tpot, workload)
            try:
                got = submission.max_goodput(hybrid, hw, 1.0, slo_tpot, workload)
            except Exception as exc:  # noqa: BLE001
                return False, f"expected {expected}, got {type(exc).__name__}: {exc}"
            if abs(got - expected) > 1e-6:
                return False, f"expected {expected} at TPOT SLO {slo_tpot}, got {got}"
            if got + 1e-9 < prev_tpot:
                return False, (
                    f"expected goodput to be non-decreasing in the TPOT SLO, "
                    f"got {got} after {prev_tpot}"
                )
            prev_tpot = got
            last = got
        prev_ttft = -1.0
        for slo_ttft in (ttft / 2, ttft, ttft * 2):
            expected = ref_goodput(hybrid, hw, slo_ttft, 1.0, workload)
            try:
                got = submission.max_goodput(hybrid, hw, slo_ttft, 1.0, workload)
            except Exception as exc:  # noqa: BLE001
                return False, f"expected {expected}, got {type(exc).__name__}: {exc}"
            if abs(got - expected) > 1e-6:
                return False, f"expected {expected} at TTFT SLO {slo_ttft}, got {got}"
            if got + 1e-9 < prev_ttft:
                return False, (
                    f"expected goodput to be non-decreasing in the TTFT SLO, "
                    f"got {got} after {prev_ttft}"
                )
            prev_ttft = got
        return True, (
            f"expected non-decreasing goodput, got {last} at the loosest TPOT SLO "
            f"and {prev_ttft} at the loosest TTFT SLO"
        )

    c.check("max goodput is non-decreasing in each SLO", check_monotonic)

    rows = synthetic_rows()
    tol_alpha, tol_tau = fit_tolerance(rows)

    def check_fit():
        try:
            got = submission.fit_efficiency(rows, hw["hbm_gbs"])
        except Exception as exc:  # noqa: BLE001
            return False, (
                f"expected alpha {PLANTED_ALPHA} and overhead {PLANTED_TAU}, "
                f"got {type(exc).__name__}: {exc}"
            )
        alpha = got.get("alpha")
        overhead = got.get("overhead_s")
        if alpha is None or overhead is None:
            return False, (
                f"expected alpha {PLANTED_ALPHA} and overhead {PLANTED_TAU}, got {got}"
            )
        # A wrong-sign overhead is negative. The planted value is +0.002.
        if abs(alpha - PLANTED_ALPHA) > tol_alpha or abs(overhead - PLANTED_TAU) > tol_tau:
            return False, (
                f"expected alpha {PLANTED_ALPHA} ± {tol_alpha} and overhead "
                f"{PLANTED_TAU} ± {tol_tau}, got alpha {alpha} and overhead {overhead}"
            )
        return True, (
            f"expected alpha {PLANTED_ALPHA} ± {tol_alpha} and overhead "
            f"{PLANTED_TAU} ± {tol_tau}, got alpha {alpha} and overhead {overhead}"
        )

    c.check("the fit recovers the planted bandwidth fraction and overhead", check_fit)

    def check_layout():
        cases = (
            ("loose", 1.0, 1.0),
            ("tight TPOT", ttft * 2, step_1 * 0.7),
        )
        lines = []
        for name, slo_ttft, slo_tpot in cases:
            expected = ref_layout(hybrid, hw, 4, slo_ttft, slo_tpot, workload)
            try:
                got = submission.choose_layout(hybrid, hw, 4, slo_ttft, slo_tpot, workload)
            except Exception as exc:  # noqa: BLE001
                return False, f"expected {expected} ({name}), got {type(exc).__name__}: {exc}"
            if got.get("tp") != expected["tp"] or got.get("replicas") != expected["replicas"]:
                return False, f"expected {expected} ({name}), got {got}"
            if abs(got.get("tokens_per_s", -1) - expected["tokens_per_s"]) > 1e-4:
                return False, f"expected {expected} ({name}), got {got}"
            lines.append(f"{name}: tp {expected['tp']}, replicas {expected['replicas']}")
        return True, f"expected {lines[0]}; {lines[1]}, got the same"

    c.check("the layout search matches brute force over degree and replicas", check_layout)

    c.metric("decode_batch1_ms", round(expected_decode * 1e3, 6))
    c.metric("prefill_2048_ms", round(prefill_expected * 1e3, 6))
    loose_goodput = ref_goodput(hybrid, hw, 1.0, 1.0, workload)
    c.metric("goodput_tokens_per_s", round(loose_goodput, 3))
    return c.finish()
