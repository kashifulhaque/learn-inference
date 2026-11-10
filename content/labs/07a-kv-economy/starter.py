"""Lab 07a — KV economy: MLA, a rolling window with sinks, and H2O eviction.

Everything runs on the CPU. Use float64 so the naive and absorbed forms of
MLA agree to 1e-9.

`cfg` is a dict with `layers`, `n_heads`, `n_kv_heads`, `d_head`, `d_c`
(the MLA latent width), and `d_rope` (the decoupled RoPE key width).
"""

from __future__ import annotations

import math

import torch


def gqa_kv_bytes(cfg: dict, dtype_bytes: int = 2) -> int:
    """Bytes of K and V per token over all layers.

    `2 * n_kv_heads * d_head * layers * dtype_bytes`. With `n_kv_heads` equal
    to `n_heads` this is multi-head attention.
    """
    raise NotImplementedError("gqa_kv_bytes")


def mla_kv_bytes(cfg: dict, dtype_bytes: int = 2) -> int:
    """Bytes per token over all layers for MLA's cache.

    Each layer stores the latent `c_kv` (`d_c` values) and one shared RoPE
    key (`d_rope` values).
    """
    raise NotImplementedError("mla_kv_bytes")


def window_kv_bytes(cfg: dict, context: int, window: int, sinks: int, dtype_bytes: int = 2) -> int:
    """GQA bytes held for a sequence of `context` tokens in a window-plus-sinks cache.

    The cache holds at most `window + sinks` tokens.
    """
    raise NotImplementedError("window_kv_bytes")


def mla_attention_naive(q_nope, q_rope, c_kv, k_rope, w_uk, w_uv, w_o):
    """One decode token of MLA, expanding the latent into keys and values.

    Shapes: q_nope (H, dh), q_rope (H, dr), c_kv (T, dc), k_rope (T, dr),
    w_uk (H, dh, dc), w_uv (H, dh, dc), w_o (D, H * dh). Return shape (D,).

    For head h, the key is `[c_kv @ w_uk[h].T, k_rope]` and the query is
    `[q_nope[h], q_rope[h]]`. The score scale is 1 / sqrt(dh + dr). The
    value is `c_kv @ w_uv[h].T`. Concatenate the heads, then apply `w_o`.
    """
    raise NotImplementedError("mla_attention_naive")


def mla_attention_absorbed(q_nope, q_rope, c_kv, k_rope, w_uk, w_uv, w_o):
    """The same result as `mla_attention_naive`, without building any key or value.

    Move `w_uk` onto the query, so scores are `q_lat @ c_kv.T + q_rope @
    k_rope.T`. Take the weighted sum of `c_kv` itself, then apply `w_uv`.
    Keep the scale 1 / sqrt(dh + dr).
    """
    raise NotImplementedError("mla_attention_absorbed")


def gqa_decode_cost(cfg: dict, context: int, dtype_bytes: int = 2) -> dict:
    """FLOPs and cache bytes for one decode token, attention core, all layers.

    The scores and the weighted sum are each `2 * n_heads * d_head * context`
    FLOPs per layer. Return `{"flops", "bytes", "intensity"}` where bytes is
    `gqa_kv_bytes * context` and intensity is flops over bytes.
    """
    raise NotImplementedError("gqa_decode_cost")


def mla_decode_cost(cfg: dict, context: int, form: str, dtype_bytes: int = 2) -> dict:
    """FLOPs and cache bytes for one decode token, attention core, all layers.

    `form` is "naive" or "absorbed". Count 2 FLOPs per multiply-add and leave
    out the projections from the hidden state and `W_O`, which are the same
    in both forms. Naive pays for scores over `dh + dr`, the weighted sum
    over `dh`, and expanding keys and values for every cached token. Absorbed
    pays for scores over `dc + dr`, the weighted sum over `dc`, and two
    per-token matrix-vector products (query and output) that do not grow
    with the context. Bytes are `mla_kv_bytes * context` for both.
    """
    raise NotImplementedError("mla_decode_cost")


def sliding_window_mask(seq_len: int, window: int, sinks: int = 0):
    """Boolean (seq_len, seq_len) matrix, True where query i may read key j.

    Key j is readable when `j <= i` and either `i - j < window` or
    `j < sinks`. The window counts the current token.
    """
    raise NotImplementedError("sliding_window_mask")


class RollingKVCache:
    """A fixed buffer of `sinks + window` slots for one head.

    The first `sinks` tokens stay in slots `0..sinks-1`. Token `p >= sinks`
    goes to slot `sinks + (p - sinks) % window`.
    """

    def __init__(self, window: int, sinks: int, head_dim: int):
        raise NotImplementedError("RollingKVCache.__init__")

    def append(self, key, value) -> int:
        """Store the next token's key and value (shape `(head_dim,)`). Return its slot."""
        raise NotImplementedError("RollingKVCache.append")

    def gather(self):
        """Return `(keys, values, positions)` of the held tokens, oldest first."""
        raise NotImplementedError("RollingKVCache.gather")

    def attend(self, query):
        """Softmax attention of one query over the held tokens. Shape `(head_dim,)`."""
        raise NotImplementedError("RollingKVCache.attend")


def evict_h2o(logits, budget: int, recent: int) -> list:
    """Simulate decoding with an H2O cache and return the kept positions, sorted.

    `logits` is (T, T), row t holding query t against keys 0..t. At step t the
    new token joins the cache and query t softmaxes over the cached keys only.
    Each cached key's accumulated score grows by its probability. If the cache
    now exceeds `budget`, drop the key with the lowest accumulated score among
    those older than the newest `recent`. Ties drop the oldest.
    """
    raise NotImplementedError("evict_h2o")
