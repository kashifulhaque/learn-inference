"""Lab 02a: scaling laws and the shape of a model that is cheap to serve.

Two halves. The first half is arithmetic on the fitted loss
L(N, D) = E + A / N^alpha + B / D^beta from Hoffmann et al. (2022), extended
with a lifetime inference term. The second half prices one decoded token in
bytes of memory traffic from a config, reusing chapter 2's formulas.
"""

from __future__ import annotations

import math

# Constants of the fitted loss. Hoffmann et al. print the exponents of the
# frontier in the paper and keep the digits of the fit in their TeX source.
# Besiroglu et al. (2024) report those digits and their own refit.
HOFFMANN = {"E": 1.6934, "A": 406.4, "B": 410.7, "alpha": 0.3392, "beta": 0.2849}
BESIROGLU = {"E": 1.8172, "A": 482.01, "B": 2085.43, "alpha": 0.3478, "beta": 0.3658}

HBM_BYTES_PER_S = 1935e9  # A100 80GB rated, chapter 10
BF16_FLOPS = 312e12  # A100 bfloat16 tensor-core peak, chapter 10


# ---- Scaling laws -----------------------------------------------------------


def chinchilla_loss(n: float, d: float, c: dict = HOFFMANN) -> float:
    """Return L(N, D) = E + A / N^alpha + B / D^beta."""
    return c["E"] + c["A"] / n ** c["alpha"] + c["B"] / d ** c["beta"]


def compute_optimal(flops: float, c: dict = HOFFMANN) -> tuple[float, float]:
    """Return the (N, D) that minimizes loss under flops = 6 N D."""
    a, b = c["alpha"], c["beta"]
    g = (a * c["A"] / (b * c["B"])) ** (1.0 / (a + b))
    n = g * (flops / 6.0) ** (b / (a + b))
    return n, flops / (6.0 * n)


def tokens_for_loss(n: float, target: float, c: dict = HOFFMANN) -> float:
    """Return the D at which an N-parameter model reaches `target` loss.

    Return infinity when the target is at or below the floor E + A / N^alpha.
    """
    room = target - c["E"] - c["A"] / n ** c["alpha"]
    if room <= 0:
        return math.inf
    return (c["B"] / room) ** (1.0 / c["beta"])


def lifetime_flops(n: float, d_train: float, d_inf: float, inference_weight: float = 1.0) -> float:
    """Return training plus inference cost: 6 N per trained token, 2 N per served token.

    `inference_weight` scales the served-token term. It is the cost of one inference
    FLOP relative to one training FLOP. At 1.0 the result is a plain FLOP count.
    """
    return 6.0 * n * d_train + 2.0 * n * d_inf * inference_weight


def optimal_with_inference(
    target: float, d_inf: float, c: dict = HOFFMANN, inference_weight: float = 1.0
) -> tuple[float, float]:
    """Return the (N, D_train) that reaches `target` loss for the fewest lifetime FLOPs.

    `d_inf` is the number of tokens served over the model's life. Search over N:
    each N fixes D_train through `tokens_for_loss`.
    """
    n_floor = (c["A"] / (target - c["E"])) ** (1.0 / c["alpha"])

    def cost(log_n: float) -> float:
        n = math.exp(log_n)
        return lifetime_flops(n, tokens_for_loss(n, target, c), d_inf, inference_weight)

    lo = math.log(n_floor * (1 + 1e-6))
    hi = math.log(n_floor * 1e3)
    phi = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
    f1, f2 = cost(x1), cost(x2)
    for _ in range(200):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - phi * (hi - lo)
            f1 = cost(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + phi * (hi - lo)
            f2 = cost(x2)
    n = math.exp((lo + hi) / 2)
    return n, tokens_for_loss(n, target, c)


# ---- Shapes and decode cost -------------------------------------------------


def num_full_attention_layers(cfg: dict) -> int:
    interval = cfg["full_attention_interval"]
    return sum(1 for i in range(cfg["num_hidden_layers"]) if (i + 1) % interval == 0)


def param_count(cfg: dict) -> int:
    """Count parameters the way chapter 2 does: projections, MLPs, embeddings, norms."""
    d = cfg["hidden_size"]
    layers = cfg["num_hidden_layers"]
    full = num_full_attention_layers(cfg)
    q_width = cfg["num_attention_heads"] * cfg["head_dim"]
    kv_width = cfg["num_key_value_heads"] * cfg["head_dim"]
    q_out = 2 * q_width if cfg["attn_output_gate"] else q_width
    attn_full = d * q_out + 2 * d * kv_width + q_width * d
    nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    dk, dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    qkvz = 2 * nk * dk + 2 * nv * dv
    conv = (2 * nk * dk + nv * dv) * cfg["linear_conv_kernel_dim"]
    attn_linear = d * qkvz + conv + d * 2 * nv + nv * dv * d
    mlp = 3 * d * cfg["intermediate_size"]
    embed = 2 * cfg["vocab_size"] * d
    norms = layers * 2 * d + d
    return full * attn_full + (layers - full) * attn_linear + layers * mlp + embed + norms


def kv_bytes_per_token(cfg: dict) -> int:
    per_layer = 2 * cfg["num_key_value_heads"] * cfg["head_dim"] * cfg["bytes_per_element"]
    return per_layer * num_full_attention_layers(cfg)


def recurrent_state_bytes(cfg: dict, state_dtype_bytes: int = 4) -> int:
    layers = cfg["num_hidden_layers"] - num_full_attention_layers(cfg)
    nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    dk, dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    recurrent = nv * dk * dv * state_dtype_bytes
    channels = 2 * nk * dk + nv * dv
    conv = channels * cfg["linear_conv_kernel_dim"] * cfg["bytes_per_element"]
    return (recurrent + conv) * layers


def inference_cost_per_token(cfg: dict, batch: int, context: int) -> float:
    """Bytes of HBM traffic per generated token in a decode step.

    One step reads every weight once, shared by the `batch` sequences. Each
    sequence reads its own KV cache at `context` tokens, and reads and writes
    its recurrent state.
    """
    weights = param_count(cfg) * cfg["bytes_per_element"]
    kv = kv_bytes_per_token(cfg) * context
    state = 2 * recurrent_state_bytes(cfg)
    return weights / batch + kv + state


def decode_floor_ms(cfg: dict, batch: int, context: int) -> float:
    """Return the roofline floor of one decode step in milliseconds.

    The step takes the longer of the byte time and the matmul time, 2 N FLOPs
    per token. Launch overhead, activation traffic, and attention FLOPs are left out.
    """
    n = param_count(cfg)
    step_bytes = inference_cost_per_token(cfg, batch, context) * batch
    step_flops = 2.0 * n * batch
    return 1e3 * max(step_bytes / HBM_BYTES_PER_S, step_flops / BF16_FLOPS)


def rank_shapes(shapes: dict, batch: int, context: int) -> list[str]:
    """Return shape names ordered from cheapest to costliest decode token."""
    return sorted(shapes, key=lambda k: (inference_cost_per_token(shapes[k], batch, context), k))


def kv_equals_weights_context(cfg: dict, batch: int) -> float:
    """Return the context at which one sequence's KV read equals its share of the weights."""
    kv = kv_bytes_per_token(cfg)
    if kv == 0:
        return math.inf
    return param_count(cfg) * cfg["bytes_per_element"] / batch / kv


def match_params(cfg: dict, layers: int, target: int) -> dict:
    """Return a copy of `cfg` with `layers` layers and the hidden size nearest `target` parameters.

    The hidden size moves in steps of 256. The MLP keeps the base width ratio.
    """
    ratio = cfg["intermediate_size"] / cfg["hidden_size"]

    def make(k: int) -> dict:
        d = 256 * k
        return {**cfg, "num_hidden_layers": layers, "hidden_size": d,
                "intermediate_size": 256 * round(ratio * d / 256)}

    lo, hi = 1, 400
    while lo < hi:
        mid = (lo + hi) // 2
        if param_count(make(mid)) < target:
            lo = mid + 1
        else:
            hi = mid
    best = min((lo - 1, lo), key=lambda k: abs(param_count(make(k)) - target))
    return make(best)
