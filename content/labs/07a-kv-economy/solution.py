"""Lab 07a — KV economy: MLA, a rolling window with sinks, and H2O eviction.

Reference solution. Everything runs on the CPU in float64.
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Bytes per token
# ---------------------------------------------------------------------------


def gqa_kv_bytes(cfg: dict, dtype_bytes: int = 2) -> int:
    """Bytes of K and V per token over all layers: 2 * n_kv * d_head * layers * b."""
    return 2 * cfg["n_kv_heads"] * cfg["d_head"] * cfg["layers"] * dtype_bytes


def mla_kv_bytes(cfg: dict, dtype_bytes: int = 2) -> int:
    """Bytes of the latent and the decoupled RoPE key per token over all layers."""
    return (cfg["d_c"] + cfg["d_rope"]) * cfg["layers"] * dtype_bytes


def window_kv_bytes(
    cfg: dict, context: int, window: int, sinks: int, dtype_bytes: int = 2
) -> int:
    """GQA bytes held for a sequence of `context` tokens in a window-plus-sinks cache."""
    return min(context, window + sinks) * gqa_kv_bytes(cfg, dtype_bytes)


# ---------------------------------------------------------------------------
# MLA decode attention, one query token
# ---------------------------------------------------------------------------


def mla_attention_naive(q_nope, q_rope, c_kv, k_rope, w_uk, w_uv, w_o):
    """Expand the latent into per-head keys and values, then attend.

    Shapes: q_nope (H, dh), q_rope (H, dr), c_kv (T, dc), k_rope (T, dr),
    w_uk (H, dh, dc), w_uv (H, dh, dc), w_o (D, H * dh). Returns (D,).
    The score scale is 1 / sqrt(dh + dr).
    """
    heads, d_head = q_nope.shape
    d_rope = q_rope.shape[1]
    scale = 1.0 / math.sqrt(d_head + d_rope)
    outs = []
    for h in range(heads):
        k_nope = c_kv @ w_uk[h].T  # (T, dh)
        value = c_kv @ w_uv[h].T  # (T, dh)
        scores = (k_nope @ q_nope[h] + k_rope @ q_rope[h]) * scale
        probs = torch.softmax(scores, dim=0)
        outs.append(probs @ value)
    return w_o @ torch.cat(outs)


def mla_attention_absorbed(q_nope, q_rope, c_kv, k_rope, w_uk, w_uv, w_o):
    """Attend in the latent space. Same arguments and result as the naive form.

    The keys' up-projection moves onto the query, so the scores read `c_kv`
    directly. The values' up-projection moves after the weighted sum.
    """
    heads, d_head = q_nope.shape
    d_rope = q_rope.shape[1]
    scale = 1.0 / math.sqrt(d_head + d_rope)
    q_lat = torch.einsum("hd,hdc->hc", q_nope, w_uk)  # (H, dc)
    scores = (q_lat @ c_kv.T + q_rope @ k_rope.T) * scale  # (H, T)
    probs = torch.softmax(scores, dim=-1)
    ctx = probs @ c_kv  # (H, dc)
    per_head = torch.einsum("hc,hdc->hd", ctx, w_uv)  # (H, dh)
    return w_o @ per_head.reshape(-1)


def gqa_decode_cost(cfg: dict, context: int, dtype_bytes: int = 2) -> dict:
    """FLOPs and cache bytes for one decode token, attention core only, all layers."""
    flops = 4 * cfg["n_heads"] * cfg["d_head"] * context * cfg["layers"]
    nbytes = gqa_kv_bytes(cfg, dtype_bytes) * context
    return {"flops": flops, "bytes": nbytes, "intensity": flops / nbytes}


def mla_decode_cost(cfg: dict, context: int, form: str, dtype_bytes: int = 2) -> dict:
    """FLOPs and cache bytes for one decode token, attention core only, all layers.

    `form` is "naive" or "absorbed". The projections from the hidden state to
    the query and the latent, and the output projection W_O, are the same in
    both forms and are left out.
    """
    h, dh = cfg["n_heads"], cfg["d_head"]
    dc, dr, t = cfg["d_c"], cfg["d_rope"], context
    if form == "naive":
        scores = 2 * h * (dh + dr) * t
        mix = 2 * h * dh * t
        expand = 4 * t * dc * h * dh
        flops = scores + mix + expand
    elif form == "absorbed":
        scores = 2 * h * (dc + dr) * t
        mix = 2 * h * dc * t
        fixed = 4 * h * dh * dc
        flops = scores + mix + fixed
    else:
        raise ValueError(f"form must be 'naive' or 'absorbed', got {form!r}")
    flops *= cfg["layers"]
    nbytes = mla_kv_bytes(cfg, dtype_bytes) * context
    return {"flops": flops, "bytes": nbytes, "intensity": flops / nbytes}


# ---------------------------------------------------------------------------
# Sliding window with sinks
# ---------------------------------------------------------------------------


def sliding_window_mask(seq_len: int, window: int, sinks: int = 0) -> torch.Tensor:
    """Boolean (seq_len, seq_len) matrix, True where query i may read key j.

    Key j is readable when j <= i and either i - j < window or j < sinks.
    The window counts the current token.
    """
    i = torch.arange(seq_len).unsqueeze(1)
    j = torch.arange(seq_len).unsqueeze(0)
    return (j <= i) & (((i - j) < window) | (j < sinks))


class RollingKVCache:
    """A fixed buffer of `sinks + window` slots for one head.

    The first `sinks` tokens live in slots 0..sinks-1 forever. Token p >= sinks
    goes to slot `sinks + (p - sinks) % window`, overwriting the token that
    left the window.
    """

    def __init__(self, window: int, sinks: int, head_dim: int):
        self.window = window
        self.sinks = sinks
        self.head_dim = head_dim
        slots = sinks + window
        self.keys = torch.zeros(slots, head_dim, dtype=torch.float64)
        self.values = torch.zeros(slots, head_dim, dtype=torch.float64)
        self.pos = torch.full((slots,), -1, dtype=torch.int64)
        self.count = 0

    def append(self, key: torch.Tensor, value: torch.Tensor) -> int:
        """Store the next token's key and value. Return the slot it used."""
        p = self.count
        slot = p if p < self.sinks else self.sinks + (p - self.sinks) % self.window
        self.keys[slot] = key.to(torch.float64)
        self.values[slot] = value.to(torch.float64)
        self.pos[slot] = p
        self.count += 1
        return slot

    def gather(self):
        """Return (keys, values, positions) of the held tokens, oldest first."""
        filled = (self.pos >= 0).nonzero().squeeze(-1)
        order = filled[torch.argsort(self.pos[filled])]
        return self.keys[order], self.values[order], self.pos[order]

    def attend(self, query: torch.Tensor) -> torch.Tensor:
        """Softmax attention of one query over the held tokens."""
        keys, values, _ = self.gather()
        scores = keys @ query.to(torch.float64) / math.sqrt(self.head_dim)
        return torch.softmax(scores, dim=0) @ values


# ---------------------------------------------------------------------------
# H2O eviction
# ---------------------------------------------------------------------------


def evict_h2o(logits: torch.Tensor, budget: int, recent: int) -> list[int]:
    """Simulate decoding with an H2O cache. Return the kept positions, sorted.

    `logits` is (T, T) pre-softmax scores, row t holding query t against keys
    0..t. At step t the new token enters the cache, the query softmaxes over
    the cached keys only, and each cached key's accumulated score grows by its
    probability. When the cache then exceeds `budget`, drop the cached key
    with the lowest accumulated score that is not one of the `recent` newest.
    Ties drop the oldest.
    """
    n = logits.shape[0]
    recent = min(recent, budget)
    score = torch.zeros(n, dtype=torch.float64)
    kept: list[int] = []
    for t in range(n):
        kept.append(t)
        idx = torch.tensor(kept)
        probs = torch.softmax(logits[t, idx].to(torch.float64), dim=0)
        score[idx] += probs
        if len(kept) > budget:
            pool = kept[: len(kept) - recent] if recent else kept
            victim = min(pool, key=lambda p: (score[p].item(), p))
            kept.remove(victim)
    return sorted(kept)
