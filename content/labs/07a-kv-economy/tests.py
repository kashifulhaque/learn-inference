"""Checks for MLA, a rolling window with sinks, and H2O eviction.

Everything runs on the CPU in float64 with seeded draws. The first check
calls only `mla_kv_bytes`.

Plausible wrong implementations this file is built to catch:

* MLA bytes that count only the latent and leave out the decoupled RoPE key
  report 512 * 60 * 2 = 61,440 bytes per token for DeepSeek-V2, not 69,120.
  The first check fails.
* An absorbed attention that scales its scores by 1 / sqrt(d_c + d_r), the
  width of the latent it now reads, instead of 1 / sqrt(d_h + d_r). It
  matches the naive form only when those widths are equal. The absorbed check
  uses d_h = 8, d_r = 4, d_c = 6, so it fails.
* A rolling buffer that overwrites slot 0 like any other slot, so the sink
  leaves with the window. The slot check and the masked-attention check fail,
  and the planted-sink error is above 1.
* An H2O that keeps only the newest `budget` tokens, which is a plain sliding
  window. It retains about 4% of the oracle's attention mass on the planted
  heavy hitters, against 0.9 required, and it drops every planted token.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from lab_common import Checks

# DeepSeek-V2 (arXiv 2405.04434, section 3.1.2): 60 layers, 128 heads of 128
# dimensions, d_c 512, decoupled RoPE key of 64 dimensions.
V2 = {"layers": 60, "n_heads": 128, "n_kv_heads": 128, "d_head": 128, "d_c": 512, "d_rope": 64}
# This course's model: 16 full-attention layers, 24 query heads, 4 KV heads of 256.
THIS = {"layers": 16, "n_heads": 24, "n_kv_heads": 4, "d_head": 256, "d_c": 512, "d_rope": 64}

# Measured by the reference solution on the two planted layers (0.976 and 0.975). A sliding
# window gets 0.04. The bound sits between them, well under the correct value.
H2O_MIN_RATIO = 0.9


def _call(submission, name, *args, **kwargs):
    try:
        return getattr(submission, name)(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _mla_inputs(seed, heads, d_head, d_rope, d_c, d_model, t):
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64)

    return (
        r(heads, d_head),
        r(heads, d_rope),
        r(t, d_c),
        r(t, d_rope),
        r(heads, d_head, d_c) / math.sqrt(d_c),
        r(heads, d_head, d_c) / math.sqrt(d_c),
        r(d_model, heads * d_head) / math.sqrt(heads * d_head),
    )


def _ref_mla(q_nope, q_rope, c_kv, k_rope, w_uk, w_uv, w_o):
    """Expand keys and values, then call PyTorch's own attention."""
    heads, d_head = q_nope.shape
    d_rope = q_rope.shape[1]
    outs = []
    for h in range(heads):
        key = torch.cat([c_kv @ w_uk[h].T, k_rope], dim=1)
        value = c_kv @ w_uv[h].T
        query = torch.cat([q_nope[h], q_rope[h]])
        out = F.scaled_dot_product_attention(
            query[None, None, None, :],
            key[None, None],
            value[None, None],
            scale=1.0 / math.sqrt(d_head + d_rope),
        )
        outs.append(out.reshape(-1))
    return w_o @ torch.cat(outs)


def _ref_mask(n, window, sinks):
    allowed = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        for j in range(i + 1):
            allowed[i, j] = (i - j < window) or (j < sinks)
    return allowed


def _full_attention(query, keys, values, allowed_row):
    scores = keys @ query / math.sqrt(keys.shape[1])
    scores = scores.masked_fill(~allowed_row, float("-inf"))
    return torch.softmax(scores, dim=0) @ values


def _planted_logits(seed, heavy, n=128, bias=4.0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, n, generator=g, dtype=torch.float64) * 0.5
    for k in heavy:
        logits[:, k] += bias
    t = torch.arange(n).unsqueeze(1)
    k = torch.arange(n).unsqueeze(0)
    logits = logits - 0.04 * (t - k).clamp(min=0)
    return logits.masked_fill(k > t, float("-inf"))


def _mass(attn, keep):
    return attn[:, list(keep)].sum().item()


def run(submission):
    c = Checks()
    if not c.require(
        submission,
        "gqa_kv_bytes",
        "mla_kv_bytes",
        "window_kv_bytes",
        "mla_attention_naive",
        "mla_attention_absorbed",
        "gqa_decode_cost",
        "mla_decode_cost",
        "sliding_window_mask",
        "RollingKVCache",
        "evict_h2o",
    ):
        return c.finish()

    # 1. MLA bytes per token. Calls only `mla_kv_bytes`.
    def mla_bytes():
        got, err = _call(submission, "mla_kv_bytes", V2)
        if err:
            return False, err
        return got == 69120, f"expected 69120 bytes per token for DeepSeek-V2 ((512 + 64) * 60 * 2), got {got}"

    c.check("MLA bytes per token: 69,120 for DeepSeek-V2", mla_bytes)

    # 2. GQA and window bytes.
    def gqa_bytes():
        got_gqa, err = _call(submission, "gqa_kv_bytes", THIS)
        if err:
            return False, err
        got_mha, err = _call(submission, "gqa_kv_bytes", V2)
        if err:
            return False, err
        short, err = _call(submission, "window_kv_bytes", THIS, 1000, 4096, 4)
        if err:
            return False, err
        long, err = _call(submission, "window_kv_bytes", THIS, 32768, 4096, 4)
        if err:
            return False, err
        want = (65536, 3932160, 1000 * 65536, 4100 * 65536)
        got = (got_gqa, got_mha, short, long)
        return got == want, (
            f"expected {want} (GQA this model, MHA V2, window at 1,000 and 32,768 tokens), got {got}"
        )

    c.check("GQA bytes 65,536 per token; window bytes cap at window plus sinks", gqa_bytes)

    # 3. The naive MLA form against PyTorch attention on explicit keys and values.
    def naive_form():
        worst = 0.0
        for seed, (heads, dh, dr, dc, dm, t) in enumerate(
            [(4, 8, 4, 6, 16, 11), (3, 5, 2, 9, 7, 1)]
        ):
            args = _mla_inputs(seed, heads, dh, dr, dc, dm, t)
            got, err = _call(submission, "mla_attention_naive", *args)
            if err:
                return False, err
            want = _ref_mla(*args)
            if got.shape != want.shape:
                return False, f"expected shape {tuple(want.shape)}, got {tuple(got.shape)}"
            worst = max(worst, (got - want).abs().max().item())
        return worst <= 1e-9, f"expected max abs error <= 1e-9, got {worst:.3e}"

    c.check("naive MLA matches SDPA on expanded keys and values", naive_form)

    # 4. Absorbed equals naive.
    def absorbed_form():
        worst = 0.0
        for seed, (heads, dh, dr, dc, dm, t) in enumerate(
            [(4, 8, 4, 6, 16, 11), (3, 5, 2, 9, 7, 1), (2, 4, 4, 4, 5, 20)]
        ):
            args = _mla_inputs(100 + seed, heads, dh, dr, dc, dm, t)
            got, err = _call(submission, "mla_attention_absorbed", *args)
            if err:
                return False, err
            want = _ref_mla(*args)
            worst = max(worst, (got - want).abs().max().item())
        return worst <= 1e-9, f"expected max abs error <= 1e-9 against the naive form, got {worst:.3e}"

    c.check("absorbed MLA equals the naive form", absorbed_form)

    # 5. Decode cost: V2 at 4096 tokens, and the GQA intensity from chapter 7.
    def decode_cost():
        t = 4096
        h, dh, dc, dr, layers = 128, 128, 512, 64, 60
        want_abs = layers * (2 * h * (dc + dr) * t + 2 * h * dc * t + 2 * h * dh * dc + 2 * h * dh * dc)
        want_naive = layers * (
            2 * h * (dh + dr) * t + 2 * h * dh * t + 2 * t * dc * h * dh + 2 * t * dc * h * dh
        )
        want_bytes = (dc + dr) * 2 * layers * t
        absd, err = _call(submission, "mla_decode_cost", V2, t, "absorbed")
        if err:
            return False, err
        naive, err = _call(submission, "mla_decode_cost", V2, t, "naive")
        if err:
            return False, err
        gqa, err = _call(submission, "gqa_decode_cost", THIS, t)
        if err:
            return False, err
        c.metric("intensity_absorbed", round(absd["intensity"], 3))
        if (absd["flops"], absd["bytes"]) != (want_abs, want_bytes):
            return False, (
                f"expected absorbed (flops, bytes) = ({want_abs}, {want_bytes}), "
                f"got ({absd['flops']}, {absd['bytes']})"
            )
        if (naive["flops"], naive["bytes"]) != (want_naive, want_bytes):
            return False, (
                f"expected naive (flops, bytes) = ({want_naive}, {want_bytes}), "
                f"got ({naive['flops']}, {naive['bytes']})"
            )
        if abs(gqa["intensity"] - 6.0) > 1e-9:
            return False, f"expected GQA intensity 6.0 (the group size), got {gqa['intensity']}"
        return abs(absd["intensity"] - 248.889) < 0.01, (
            f"expected absorbed intensity 248.889, got {absd['intensity']:.3f}"
        )

    c.check("decode cost: exact FLOPs and bytes, absorbed intensity 248.89, GQA 6", decode_cost)

    # 6. The mask.
    def mask():
        for n, w, s in [(8, 3, 0), (8, 3, 2), (6, 10, 1), (9, 1, 1)]:
            got, err = _call(submission, "sliding_window_mask", n, w, s)
            if err:
                return False, err
            want = _ref_mask(n, w, s)
            if got.shape != want.shape or not torch.equal(got.bool(), want):
                bad = (got.bool() != want).nonzero()[0].tolist() if got.shape == want.shape else "shape"
                return False, f"expected the mask for n={n}, window={w}, sinks={s}; first difference at {bad}"
        return True, ""

    c.check("sliding_window_mask counts the current token in the window", mask)

    # 7. Which tokens the buffer holds.
    def held_positions():
        window, sinks, d = 5, 2, 4
        cache = submission.RollingKVCache(window, sinks, d)
        g = torch.Generator().manual_seed(6)
        for p in range(40):
            k = torch.randn(d, generator=g, dtype=torch.float64)
            slot = cache.append(k, k)
            if not 0 <= slot < sinks + window:
                return False, f"expected a slot in 0..{sinks + window - 1}, got {slot} at token {p}"
            _, _, pos = cache.gather()
            want = sorted(set(range(min(sinks, p + 1))) | set(range(max(0, p - window + 1), p + 1)))
            if [int(x) for x in pos] != want:
                return False, f"after token {p} expected positions {want}, got {[int(x) for x in pos]}"
        return True, ""

    c.check("rolling cache holds the sinks and the last window tokens, nothing else", held_positions)

    # 8. Attention against masked full attention at every step.
    def masked_equivalence():
        worst = 0.0
        for window, sinks in [(8, 4), (8, 0), (3, 1)]:
            n, d = 60, 8
            g = torch.Generator().manual_seed(7 + window + sinks)
            keys = torch.randn(n, d, generator=g, dtype=torch.float64)
            values = torch.randn(n, d, generator=g, dtype=torch.float64)
            queries = torch.randn(n, d, generator=g, dtype=torch.float64)
            allowed = _ref_mask(n, window, sinks)
            cache = submission.RollingKVCache(window, sinks, d)
            for t in range(n):
                cache.append(keys[t], values[t])
                got = cache.attend(queries[t])
                want = _full_attention(queries[t], keys, values, allowed[t])
                worst = max(worst, (got - want).abs().max().item())
        return worst <= 1e-9, f"expected max abs error <= 1e-9 against masked full attention, got {worst:.3e}"

    c.check("rolling attention equals masked full attention at every step", masked_equivalence)

    # 9. Planted sink: one early key every query likes.
    def planted_sink():
        n, d, window = 200, 16, 32
        g = torch.Generator().manual_seed(1)
        u = torch.randn(d, generator=g, dtype=torch.float64)
        u = u / u.norm()
        keys = torch.randn(n, d, generator=g, dtype=torch.float64)
        values = torch.randn(n, d, generator=g, dtype=torch.float64)
        keys[0] = 6 * math.sqrt(d) * u
        queries = 2 * u + 0.3 * torch.randn(n, d, generator=g, dtype=torch.float64)
        full = _full_attention(queries[-1], keys, values, torch.ones(n, dtype=torch.bool))
        errors = {}
        for sinks in (0, 4):
            cache = submission.RollingKVCache(window, sinks, d)
            for t in range(n):
                cache.append(keys[t], values[t])
            got = cache.attend(queries[-1])
            errors[sinks] = ((got - full).norm() / full.norm()).item()
        c.metric("sink_error_window_only", round(errors[0], 4))
        c.metric("sink_error_with_sinks", round(errors[4], 6))
        ok = errors[0] > 0.5 and errors[4] < 0.01
        return ok, (
            "expected relative error > 0.5 without sinks and < 0.01 with 4 sinks, "
            f"got {errors[0]:.4f} and {errors[4]:.6f}"
        )

    c.check("four sinks recover full attention that a bare window loses", planted_sink)

    # 10. H2O against an oracle.
    def h2o_planted():
        budget, recent = 24, 8
        ratios = []
        for seed, heavy in [(0, (3, 17, 40, 41, 77, 90)), (5, (1, 9, 22, 58, 63, 100))]:
            logits = _planted_logits(seed, heavy)
            n = logits.shape[0]
            attn = torch.softmax(logits, dim=1)
            oracle = torch.topk(attn.sum(0), budget).indices.tolist()
            kept, err = _call(submission, "evict_h2o", logits, budget, recent)
            if err:
                return False, err
            kept = [int(x) for x in kept]
            if kept != sorted(set(kept)) or len(kept) != budget:
                return False, f"expected {budget} sorted distinct positions, got {len(kept)}: {kept}"
            missing = [k for k in heavy if k not in kept]
            if missing:
                return False, f"seed {seed}: expected planted heavy hitters {list(heavy)} kept, missing {missing}"
            if not set(range(n - recent, n)) <= set(kept):
                return False, f"seed {seed}: expected the newest {recent} positions kept"
            ratios.append(_mass(attn, kept) / _mass(attn, oracle))
        small = _planted_logits(0, (3, 17), n=20)
        everything, err = _call(submission, "evict_h2o", small, 50, 8)
        if err:
            return False, err
        if [int(x) for x in everything] != list(range(20)):
            return False, f"expected all 20 positions kept under a budget of 50, got {everything}"
        c.metric("h2o_mass_ratio", round(min(ratios), 4))
        return min(ratios) >= H2O_MIN_RATIO, (
            f"expected retained attention mass >= {H2O_MIN_RATIO} of the oracle's, got {ratios}"
        )

    c.check("H2O keeps the planted heavy hitters and nearly the oracle's attention mass", h2o_planted)

    return c.finish()
