"""Lab 07b — three Qwen4-style blocks, in small illustrative form.

Qwen published no block size, top-k, indexer equations, gate equations, or
hashing scheme for Qwen4-Max. Everything here is this course's own design,
built to show what each idea costs at inference time:

1. A block-sparse attention step. A cheap indexer scores micro-blocks of keys,
   and the step attends to the top-k blocks only.
2. A gated residual with n streams: a read gate mixes the streams into the
   layer input, and a write gate spreads the layer output back.
3. A hashed n-gram embedding that adds a table row to the hidden state at
   layer 2.

Use float64 and the CPU. Shapes are in each docstring.
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# 1. Block-sparse attention
# ---------------------------------------------------------------------------


def pool_blocks(k_idx, block: int):
    """Mean-pool index keys into micro-blocks.

    `k_idx` is `(T, d_idx)`. Block `b` covers tokens `b*block` to
    `(b+1)*block - 1`, and the last block can be shorter. Return
    `(ceil(T / block), d_idx)`, the mean of the rows in each block.
    """
    raise NotImplementedError("pool_blocks")


def select_blocks(q_idx, k_idx, block: int, top_k: int):
    """Indices of the `top_k` blocks with the highest indexer score, ascending.

    The score of a block is `q_idx` dotted with the block's pooled index key.
    `q_idx` is `(d_idx,)`. A tie goes to the lower block index. When
    `top_k` is at least the number of blocks, return every block.
    """
    raise NotImplementedError("select_blocks")


def block_token_indices(block_ids, block: int, n_tokens: int):
    """Token positions covered by the chosen blocks, ascending.

    The last block is clipped at `n_tokens`.
    """
    raise NotImplementedError("block_token_indices")


def sparse_attention_step(q, K, V, block_ids, block: int):
    """One decode step that attends to the chosen blocks only.

    `q` is `(H, d)`, and `K` and `V` are `(T, d)`, shared by all H heads. The
    scale is `1 / sqrt(d)`. Softmax runs over the kept tokens only. Return
    `(H, d)`.
    """
    raise NotImplementedError("sparse_attention_step")


def sparse_decode_cost(
    n_tokens: int,
    block: int,
    top_k: int,
    n_heads: int,
    n_kv_heads: int,
    d_head: int,
    d_idx: int,
    kv_bytes: int = 2,
    idx_bytes: int = 2,
) -> dict:
    """Bytes and FLOPs one layer reads and does for one decode token.

    Keys: `n_blocks`, `kept_tokens`, `indexer_bytes`, `kv_bytes`,
    `total_bytes`, `dense_bytes`, `byte_ratio` (dense over total),
    `extra_cache_bytes`, `indexer_flops`, `attention_flops`, `dense_flops`.
    """
    raise NotImplementedError("sparse_decode_cost")


# ---------------------------------------------------------------------------
# 2. Gated residual with n streams
# ---------------------------------------------------------------------------


def gated_residual(X, f, read, write):
    """One layer of a gated residual with n streams.

    `X` is `(n, d)`. The layer input is `h = read @ X`, with `read` of shape
    `(n,)`. The output is `y = f(h)`, shape `(d,)`. Stream `i` becomes
    `X[i] + write[i] * y`. Return the new `(n, d)`.
    """
    raise NotImplementedError("gated_residual")


def dynamic_gates(X, w_read, w_write):
    """Gates computed from the streams. `w_read` and `w_write` are `(d,)`.

    `read = softmax(X @ w_read)` and `write = sigmoid(X @ w_write)`, both
    `(n,)`. This is the lab's design, not Qwen's.
    """
    raise NotImplementedError("dynamic_gates")


def gated_residual_cost(n_streams: int, d: int, dtype_bytes: int = 2) -> dict:
    """Per-token cost of one gated-residual layer, beyond the layer itself.

    Keys: `stream_bytes` (`n * d * b`), `plain_bytes` (`d * b`),
    `stream_ratio`, `rw_bytes` (read plus write of all streams),
    `plain_rw_bytes`, and `gate_flops` (`8 * n * d`: two gate dot products,
    the read mix, and the write-back).
    """
    raise NotImplementedError("gated_residual_cost")


# ---------------------------------------------------------------------------
# 3. Hashed n-gram embedding
# ---------------------------------------------------------------------------

_MASK = (1 << 64) - 1
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
PAD = -1


def ngram_hash(window, table_size: int, seed: int = 0) -> int:
    """Hash a tuple of token ids to a row in `[0, table_size)`.

    FNV-1a over `id + 1`, then a 64-bit finalizer. Deterministic, and it
    depends on the order of the ids. For each id, XOR `id + 1` into `h`, then
    multiply by `_FNV_PRIME` mod 2**64, starting from `_FNV_OFFSET ^ seed`.
    Finish with `h ^= h >> 33; h *= 0xFF51AFD7ED558CCD; h ^= h >> 33;
    h *= 0xC4CEB9FE1A85EC53; h ^= h >> 33`, each multiply mod 2**64. Any
    hash that is deterministic, uniform, and order-sensitive passes the checks.
    Don't use `sum(ids)`, which ignores order.
    """
    raise NotImplementedError("ngram_hash")


def ngram_indices(ids, n: int, table_size: int, seed: int = 0):
    """Table row for each position, from the last `n` ids ending there.

    Positions before the start are padded with `PAD`, so position `t` always
    hashes a window of exactly `n` ids. Return a `(T,)` int64 tensor.
    """
    raise NotImplementedError("ngram_indices")


def ngram_embedding(ids, table, n: int, seed: int = 0):
    """Look up one row per position. `table` is `(table_size, d)`."""
    raise NotImplementedError("ngram_embedding")


def inject_ngram(hidden, emb, layer: int, at_layer: int = 2):
    """Add `emb` to `hidden` when `layer` (1-based) equals `at_layer`."""
    raise NotImplementedError("inject_ngram")


def ngram_accounting(table_size: int, d: int, n_tables: int = 1, dtype_bytes: int = 2) -> dict:
    """Parameters, bytes, and per-token work of the n-gram tables.

    Keys: `params` (`n_tables * table_size * d`), `stored_bytes`,
    `bytes_read_per_token` (one row per table), `flops_per_token` (the add,
    `d` per table; the lookup does no arithmetic), `dense_flops_per_token`
    (a dense layer with the same parameters, `2 * params`), and `flop_ratio`
    (dense over lookup).
    """
    raise NotImplementedError("ngram_accounting")
