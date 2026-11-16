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
    n_tokens, d_idx = k_idx.shape
    n_blocks = math.ceil(n_tokens / block)
    out = torch.zeros(n_blocks, d_idx, dtype=k_idx.dtype)
    for b in range(n_blocks):
        out[b] = k_idx[b * block : (b + 1) * block].mean(dim=0)
    return out


def select_blocks(q_idx, k_idx, block: int, top_k: int):
    """Indices of the `top_k` blocks with the highest indexer score, ascending.

    The score of a block is `q_idx` dotted with the block's pooled index key.
    `q_idx` is `(d_idx,)`. A tie goes to the lower block index. When
    `top_k` is at least the number of blocks, return every block.
    """
    pooled = pool_blocks(k_idx, block)
    scores = pooled @ q_idx
    order = torch.argsort(-scores, stable=True)
    keep = order[: min(top_k, pooled.shape[0])]
    return torch.sort(keep).values


def block_token_indices(block_ids, block: int, n_tokens: int):
    """Token positions covered by the chosen blocks, ascending.

    The last block is clipped at `n_tokens`.
    """
    parts = [torch.arange(int(b) * block, min((int(b) + 1) * block, n_tokens)) for b in block_ids]
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.long)


def sparse_attention_step(q, K, V, block_ids, block: int):
    """One decode step that attends to the chosen blocks only.

    `q` is `(H, d)`, and `K` and `V` are `(T, d)`, shared by all H heads. The
    scale is `1 / sqrt(d)`. Softmax runs over the kept tokens only. Return
    `(H, d)`.
    """
    idx = block_token_indices(block_ids, block, K.shape[0])
    scores = (q @ K[idx].T) / math.sqrt(q.shape[-1])
    return torch.softmax(scores, dim=-1) @ V[idx]


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
    n_blocks = math.ceil(n_tokens / block)
    kept = min(top_k * block, n_tokens)
    per_token = 2 * n_kv_heads * d_head * kv_bytes
    indexer_bytes = n_blocks * d_idx * idx_bytes
    kv = kept * per_token
    dense = n_tokens * per_token
    return {
        "n_blocks": n_blocks,
        "kept_tokens": kept,
        "indexer_bytes": indexer_bytes,
        "kv_bytes": kv,
        "total_bytes": indexer_bytes + kv,
        "dense_bytes": dense,
        "byte_ratio": dense / (indexer_bytes + kv),
        "extra_cache_bytes": indexer_bytes,
        "indexer_flops": 2 * n_blocks * d_idx,
        "attention_flops": 4 * n_heads * d_head * kept,
        "dense_flops": 4 * n_heads * d_head * n_tokens,
    }


# ---------------------------------------------------------------------------
# 2. Gated residual with n streams
# ---------------------------------------------------------------------------


def gated_residual(X, f, read, write):
    """One layer of a gated residual with n streams.

    `X` is `(n, d)`. The layer input is `h = read @ X`, with `read` of shape
    `(n,)`. The output is `y = f(h)`, shape `(d,)`. Stream `i` becomes
    `X[i] + write[i] * y`. Return the new `(n, d)`.
    """
    h = read @ X
    y = f(h)
    return X + write[:, None] * y[None, :]


def dynamic_gates(X, w_read, w_write):
    """Gates computed from the streams. `w_read` and `w_write` are `(d,)`.

    `read = softmax(X @ w_read)` and `write = sigmoid(X @ w_write)`, both
    `(n,)`. This is the lab's design, not Qwen's.
    """
    return torch.softmax(X @ w_read, dim=0), torch.sigmoid(X @ w_write)


def gated_residual_cost(n_streams: int, d: int, dtype_bytes: int = 2) -> dict:
    """Per-token cost of one gated-residual layer, beyond the layer itself.

    Keys: `stream_bytes` (`n * d * b`), `plain_bytes` (`d * b`),
    `stream_ratio`, `rw_bytes` (read plus write of all streams),
    `plain_rw_bytes`, and `gate_flops` (`8 * n * d`: two gate dot products,
    the read mix, and the write-back).
    """
    return {
        "stream_bytes": n_streams * d * dtype_bytes,
        "plain_bytes": d * dtype_bytes,
        "stream_ratio": n_streams,
        "rw_bytes": 2 * n_streams * d * dtype_bytes,
        "plain_rw_bytes": 2 * d * dtype_bytes,
        "gate_flops": 8 * n_streams * d,
    }


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
    h = (_FNV_OFFSET ^ seed) & _MASK
    for token in window:
        h ^= (int(token) + 1) & _MASK
        h = (h * _FNV_PRIME) & _MASK
    h ^= h >> 33
    h = (h * 0xFF51AFD7ED558CCD) & _MASK
    h ^= h >> 33
    h = (h * 0xC4CEB9FE1A85EC53) & _MASK
    h ^= h >> 33
    return h % table_size


def ngram_indices(ids, n: int, table_size: int, seed: int = 0):
    """Table row for each position, from the last `n` ids ending there.

    Positions before the start are padded with `PAD`, so position `t` always
    hashes a window of exactly `n` ids. Return a `(T,)` int64 tensor.
    """
    ids = [int(t) for t in ids]
    out = []
    for t in range(len(ids)):
        window = ids[max(0, t - n + 1) : t + 1]
        window = [PAD] * (n - len(window)) + window
        out.append(ngram_hash(window, table_size, seed))
    return torch.tensor(out, dtype=torch.long)


def ngram_embedding(ids, table, n: int, seed: int = 0):
    """Look up one row per position. `table` is `(table_size, d)`."""
    return table[ngram_indices(ids, n, table.shape[0], seed)]


def inject_ngram(hidden, emb, layer: int, at_layer: int = 2):
    """Add `emb` to `hidden` when `layer` (1-based) equals `at_layer`."""
    return hidden + emb if layer == at_layer else hidden


def ngram_accounting(table_size: int, d: int, n_tables: int = 1, dtype_bytes: int = 2) -> dict:
    """Parameters, bytes, and per-token work of the n-gram tables.

    Keys: `params` (`n_tables * table_size * d`), `stored_bytes`,
    `bytes_read_per_token` (one row per table), `flops_per_token` (the add,
    `d` per table; the lookup does no arithmetic), `dense_flops_per_token`
    (a dense layer with the same parameters, `2 * params`), and `flop_ratio`
    (dense over lookup).
    """
    params = n_tables * table_size * d
    flops = n_tables * d
    return {
        "params": params,
        "stored_bytes": params * dtype_bytes,
        "bytes_read_per_token": n_tables * d * dtype_bytes,
        "flops_per_token": flops,
        "dense_flops_per_token": 2 * params,
        "flop_ratio": 2 * params / flops,
    }
