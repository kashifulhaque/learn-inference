"""Lab 14a — split-KV decode attention.

Merge partial softmax results, choose how many KV splits fill the SMs, and,
on a CUDA device, run the split decode kernel.
"""

from __future__ import annotations


def merge_partials(outs, lses):
    """Return `(out, lse)` for partial outputs and their log-sum-exp values.

    `outs` has shape (splits, ..., dim) and `lses` has shape (splits, ...).
    An empty split has a log-sum-exp of negative infinity.
    """
    raise NotImplementedError("merge_partials")


def choose_num_splits(batch: int, heads: int, kv_len: int, num_sms: int) -> int:
    """Return the split count for a query of length 1."""
    raise NotImplementedError("choose_num_splits")


def split_kv_decode(q, k, v, num_splits: int):
    """Decode attention that splits `k` and `v` into `num_splits` chunks."""
    raise NotImplementedError("split_kv_decode")
