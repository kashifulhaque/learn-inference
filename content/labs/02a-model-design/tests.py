"""Checks lab 02a: the fitted loss, the lifetime-FLOP optimum, and decode cost from a config.

Every pinned number is arithmetic on the constants in this file and on chapter 2's
geometry. None is a measurement.

Plausible wrong implementations these checks catch:

* Counting a served token as 6 N FLOPs instead of 2 N, the training rate applied to
  inference. The "lifetime FLOPs" check fails on the exact sum, and the
  "optimum with inference" check fails on the pinned N*.
* Leaving the embedding and output projection out of the weight read in
  `inference_cost_per_token`. The "bytes per token" check fails on the exact 54,101,813,248
  at batch 1 and context 0.
"""

import math

from lab_common import Checks

HOFFMANN = {"E": 1.6934, "A": 406.4, "B": 410.7, "alpha": 0.3392, "beta": 0.2849}
BESIROGLU = {"E": 1.8172, "A": 482.01, "B": 2085.43, "alpha": 0.3478, "beta": 0.3658}

CONFIG = {
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "intermediate_size": 17408,
    "vocab_size": 248320,
    "full_attention_interval": 4,
    "attn_output_gate": True,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "bytes_per_element": 2,
}

# Loss of a compute-optimal model with this model's 26,895,979,520 parameters under HOFFMANN.
TARGET = 1.9514866130473323


def rel(a, b):
    return abs(a - b) / abs(b)


def run(submission):
    c = Checks()
    needed = (
        "chinchilla_loss", "compute_optimal", "tokens_for_loss", "lifetime_flops",
        "optimal_with_inference", "param_count", "inference_cost_per_token",
        "rank_shapes", "match_params",
    )
    if not c.require(submission, *needed):
        return c.finish()

    def loss_ok():
        got = submission.chinchilla_loss(1e9, 2e10, HOFFMANN)
        want = 2.5305435106183145
        return abs(got - want) < 1e-6, f"expected {want:.6f}, got {got:.6f}"

    c.check("the fitted loss at 1e9 parameters and 2e10 tokens is 2.53054", loss_ok)

    def optimal_ok():
        n, d = submission.compute_optimal(1e24, HOFFMANN)
        want_n, want_d = 51854284861.26662, 3214134899605.2007
        ok_exact = rel(n, want_n) < 1e-6 and rel(d, want_d) < 1e-6
        ok_budget = rel(6 * n * d, 1e24) < 1e-9
        h = 1e-4 * n
        f = lambda x: submission.chinchilla_loss(x, 1e24 / (6 * x), HOFFMANN)
        slope = (f(n + h) - f(n - h)) / (2 * h) * n  # dL/dlogN by central difference
        ok_grad = abs(slope) < 1e-6 and f(0.9 * n) > f(n) and f(1.1 * n) > f(n)
        return ok_exact and ok_budget and ok_grad, (
            f"expected N={want_n:.4e}, D={want_d:.4e}; got N={n:.4e}, D={d:.4e}; "
            f"6ND/C={6 * n * d / 1e24:.9f}; dL/dlogN={slope:.2e}"
        )

    c.check("the compute-optimal split at 1e24 FLOPs is 51.85B parameters on 3.214T tokens", optimal_ok)

    def tokens_ok():
        d1 = submission.tokens_for_loss(1e9, 2.5, HOFFMANN)
        back = submission.chinchilla_loss(1e9, d1, HOFFMANN)
        floor = HOFFMANN["E"] + HOFFMANN["A"] / 1e9 ** HOFFMANN["alpha"]
        below = submission.tokens_for_loss(1e9, floor - 0.01, HOFFMANN)
        ok = rel(d1, 25225891342.832882) < 1e-6 and abs(back - 2.5) < 1e-9 and math.isinf(below)
        return ok, (
            f"expected 2.5226e10 tokens, loss 2.5 on the way back, and inf below the {floor:.4f} floor; "
            f"got {d1:.4e}, loss {back:.9f}, {below}"
        )

    c.check("tokens_for_loss inverts the fit and returns inf under the floor", tokens_ok)

    def flops_ok():
        a = submission.lifetime_flops(1e9, 2e10, 1e11)
        b = submission.lifetime_flops(1e9, 2e10, 1e11, 5.0)
        return a == 3.2e20 and b == 1.12e21, (
            f"expected 6*1e9*2e10 + 2*1e9*1e11 = 3.2e20 and 1.12e21 at weight 5; got {a:.4e} and {b:.4e}"
        )

    c.check("lifetime FLOPs charge 6 N per trained token and 2 N per served token", flops_ok)

    def with_inference_ok():
        n0, d0 = submission.optimal_with_inference(TARGET, 0.0, HOFFMANN)
        zero_ok = rel(n0, 26_895_979_520) < 1e-4
        rows = [submission.optimal_with_inference(TARGET, v, HOFFMANN) for v in (1e12, 1e13, 1e14)]
        want = [(21012435309.76, 1922077928536.0), (12070140505.34, 4279416118202.0),
                (6920434051.45, 15762385935743.0)]
        pinned = all(rel(r[0], w[0]) < 1e-3 and rel(r[1], w[1]) < 1e-3 for r, w in zip(rows, want))
        n13, d13 = rows[1]
        f = lambda x: submission.lifetime_flops(
            x, submission.tokens_for_loss(x, TARGET, HOFFMANN), 1e13)
        slope = (f(n13 * 1.0001) - f(n13 * 0.9999)) / (2e-4 * f(n13))  # dlog(cost)/dlogN
        grad_ok = abs(slope) < 1e-4 and f(0.9 * n13) > f(n13) and f(1.1 * n13) > f(n13)
        shrinks = rows[0][0] > rows[1][0] > rows[2][0] < n0 and rows[0][1] / rows[0][0] < rows[2][1] / rows[2][0]
        return zero_ok and pinned and grad_ok and shrinks, (
            "expected N*=26.896B with no inference, then 21.012B, 12.070B, 6.920B at 1e12, 1e13, 1e14 served tokens, "
            f"a flat cost at the optimum; got {n0 / 1e9:.3f}B, "
            + ", ".join(f"{r[0] / 1e9:.3f}B" for r in rows) + f"; slope {slope:.2e}"
        )

    c.check("the optimum with inference shrinks the model and over-trains it as served tokens grow", with_inference_ok)

    def weight_ok():
        light, _ = submission.optimal_with_inference(TARGET, 1e13, HOFFMANN, 1.0)
        heavy, dh = submission.optimal_with_inference(TARGET, 1e13, HOFFMANN, 5.0)
        return rel(heavy, 8037562269.03) < 1e-3 and heavy < light, (
            f"expected 8.038B at weight 5 against 12.070B at weight 1; got {heavy / 1e9:.3f}B and {light / 1e9:.3f}B"
        )

    c.check("a costlier served token (weight 5) pushes the optimum smaller still", weight_ok)

    def params_ok():
        base = submission.param_count(CONFIG)
        mha = submission.param_count({**CONFIG, "num_key_value_heads": 24})
        vocab = submission.param_count({**CONFIG, "vocab_size": 128000})
        dense = submission.param_count({**CONFIG, "full_attention_interval": 1})
        ok = (base, mha, vocab, dense) == (26_895_979_520, 27_734_840_320, 25_663_902_720, 26_367_104_000)
        return ok, (
            "expected 26,895,979,520 / 27,734,840,320 / 25,663,902,720 / 26,367,104,000 "
            f"for the base, 24 KV heads, a 128,000 vocabulary, and no linear layers; got {base:,} / {mha:,} / {vocab:,} / {dense:,}"
        )

    c.check("param_count reproduces chapter 2's 26.9B and moves with the shape", params_ok)

    def bytes_ok():
        empty = submission.inference_cost_per_token(CONFIG, 1, 0)
        busy = submission.inference_cost_per_token(CONFIG, 32, 32768)
        dense = submission.inference_cost_per_token({**CONFIG, "full_attention_interval": 1}, 32, 32768)
        mha = submission.inference_cost_per_token(
            {**CONFIG, "num_key_value_heads": 24, "full_attention_interval": 1}, 32, 32768)
        ok = (empty, busy, dense, mha) == (54_101_813_248, 4_138_336_576, 10_237_878_592, 53_397_266_752)
        return ok, (
            "expected 54,101,813,248 at batch 1 and context 0 (every weight, plus the state read and written), "
            "then 4,138,336,576 at batch 32 and 32,768 tokens, 10,237,878,592 dense, 53,397,266,752 dense MHA; "
            f"got {empty:,.0f}, {busy:,.0f}, {dense:,.0f}, {mha:,.0f}"
        )

    c.check("decode bytes per token: weights shared over the batch, KV and state per sequence", bytes_ok)

    shapes = {
        "base": CONFIG,
        "mha_24": {**CONFIG, "num_key_value_heads": 24},
        "kv_2": {**CONFIG, "num_key_value_heads": 2},
        "dense": {**CONFIG, "full_attention_interval": 1},
        "vocab_128k": {**CONFIG, "vocab_size": 128000},
        "vocab_32k": {**CONFIG, "vocab_size": 32000},
    }

    def rank_ok():
        long = submission.rank_shapes(shapes, 32, 32768)
        short = submission.rank_shapes(shapes, 1, 1024)
        want_long = ["kv_2", "vocab_32k", "vocab_128k", "base", "dense", "mha_24"]
        want_short = ["vocab_32k", "vocab_128k", "dense", "kv_2", "base", "mha_24"]
        return long == want_long and short == want_short, (
            f"expected {want_long} at batch 32, 32,768 tokens and {want_short} at batch 1, 1,024 tokens; "
            f"got {long} and {short}"
        )

    c.check("the ranking of shapes flips between a weight-bound and a KV-bound decode", rank_ok)

    def match_ok():
        target = submission.param_count(CONFIG)
        m32 = submission.match_params(CONFIG, 32, target)
        p = submission.param_count(m32)
        ok = m32["num_hidden_layers"] == 32 and m32["hidden_size"] == 7424 and rel(p, target) < 0.01
        return ok, (
            f"expected 32 layers, hidden size 7424, within 1% of {target:,} parameters; "
            f"got {m32['num_hidden_layers']} layers, hidden size {m32['hidden_size']}, {p:,}"
        )

    c.check("match_params trades depth for width at equal parameters", match_ok)

    try:
        c.metric("optimal_N_billions_at_1e13_served",
                 round(submission.optimal_with_inference(TARGET, 1e13, HOFFMANN)[0] / 1e9, 3))
        c.metric("decode_gb_per_token_b32_32k",
                 round(submission.inference_cost_per_token(CONFIG, 32, 32768) / 1e9, 3))
    except Exception:  # noqa: BLE001
        pass  # the checks above already report the failure
    return c.finish()
