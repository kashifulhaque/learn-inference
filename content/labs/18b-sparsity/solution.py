"""Lab 18b — magnitude and Wanda pruning, 2:4 sparsity, and a distillation loss.

Scores rank the weights of one linear layer `W` of shape `(out, in)`, and
every pruning function keeps the highest scores. `wanda_scores` multiplies each weight's magnitude by the L2 norm of
its input channel, taken over the calibration tokens `X` of shape
`(tokens, in)`.

`prune_rowwise` removes a fraction of each output row. `prune_2_4` keeps
two of every four consecutive input weights. `compress_2_4` stores the two
kept values and their 2-bit positions. `kd_loss` is the temperature-scaled
KL divergence from Hinton et al. `sparse_semi_structured_check` is a CUDA
only check that this file has never run: it skips on a CPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def magnitude_scores(weight):
    """Return `abs(weight)`."""
    return weight.abs()


def wanda_scores(weight, inputs):
    """Wanda score: `abs(W[o, j]) * ||X[:, j]||_2`.

    `weight` is `(out, in)` and `inputs` is `(tokens, in)`. The norm is
    over the token axis, so each input channel has one norm shared by all
    output rows. Return a tensor shaped like `weight`.
    """
    norms = inputs.to(weight.dtype).norm(dim=0)
    return weight.abs() * norms.unsqueeze(0)


def prune_rowwise(weight, scores, sparsity: float):
    """Zero the lowest-scoring weights of every output row.

    Each row drops `round(sparsity * in)` weights, so every row has the same
    count. Ties drop the lower index first. Return `(pruned, mask)`, where
    `mask` is a bool tensor that is True for a kept weight.
    """
    out, n = weight.shape
    drop = int(round(sparsity * n))
    # A stable sort ascending puts the lowest scores first, ties by index.
    order = torch.sort(scores, dim=1, stable=True).indices
    mask = torch.ones(out, n, dtype=torch.bool)
    mask.scatter_(1, order[:, :drop], False)
    return weight * mask.to(weight.dtype), mask


def prune_2_4(weight, scores=None):
    """Keep the two highest-scoring weights of every group of four.

    Groups are four consecutive input columns, `in` is a multiple of 4, and
    `scores` defaults to the magnitude. Ties keep the lower index. Return
    `(pruned, mask)` with a bool mask.
    """
    if scores is None:
        scores = weight.abs()
    out, n = weight.shape
    grouped = scores.reshape(out, n // 4, 4)
    # Descending stable sort: the first two positions are kept.
    order = torch.sort(grouped, dim=2, descending=True, stable=True).indices
    mask = torch.zeros(out, n // 4, 4, dtype=torch.bool)
    mask.scatter_(2, order[:, :, :2], True)
    mask = mask.reshape(out, n)
    return weight * mask.to(weight.dtype), mask


def is_valid_2_4(mask) -> bool:
    """True when every group of four consecutive columns has exactly 2 kept."""
    out, n = mask.shape
    if n % 4 != 0:
        return False
    counts = mask.to(torch.int64).reshape(out, n // 4, 4).sum(dim=2)
    return bool((counts == 2).all())


def compress_2_4(pruned, mask):
    """Store a 2:4 matrix as values plus 2-bit positions.

    Return `(values, indices)`. Both have shape `(out, in // 2)`. For each
    group of four, in ascending column order, `values` holds the two kept
    weights and `indices` holds their positions 0 to 3 inside the group as
    int64. Use the mask, not the value, to decide what is kept, because a
    kept weight can be exactly 0.
    """
    out, n = pruned.shape
    groups = n // 4
    m = mask.reshape(out, groups, 4).to(torch.int64)
    # Positions of the kept columns in ascending order: sort so kept first.
    pos = torch.arange(4).expand(out, groups, 4)
    key = pos + 4 * (1 - m)
    order = torch.sort(key, dim=2).indices[:, :, :2]
    values = torch.gather(pruned.reshape(out, groups, 4), 2, order)
    return values.reshape(out, groups * 2), order.reshape(out, groups * 2)


def decompress_2_4(values, indices, n: int):
    """Inverse of `compress_2_4`. Return a dense `(out, n)` tensor."""
    out = values.shape[0]
    groups = n // 4
    dense = torch.zeros(out, groups, 4, dtype=values.dtype)
    dense.scatter_(2, indices.reshape(out, groups, 2), values.reshape(out, groups, 2))
    return dense.reshape(out, n)


def compressed_bytes(out: int, n: int, value_bytes: int):
    """Return `(value_bytes_total, metadata_bytes)` for a 2:4 matrix.

    There are `out * n / 2` stored values of `value_bytes` each, and each
    stored value has a 2-bit index. Metadata bytes are the bit count
    divided by 8.
    """
    kept = out * n // 2
    return kept * value_bytes, kept * 2 // 8


def output_error(weight, pruned, inputs) -> float:
    """Relative layer-output error `||XW^T - XW_p^T||_F / ||XW^T||_F`."""
    ref = inputs @ weight.T
    got = inputs @ pruned.T
    return float((ref - got).norm() / ref.norm())


def kd_loss(student_logits, teacher_logits, temperature: float):
    """Hinton distillation loss, averaged over the batch.

    `T^2 * mean_b KL(softmax(t_b / T) || softmax(s_b / T))`. The `T^2`
    keeps the gradient size roughly independent of `T`. Logits are
    `(batch, vocab)`. The teacher gets no gradient. Differentiable in
    `student_logits`.
    """
    t = teacher_logits.detach()
    log_p_t = F.log_softmax(t / temperature, dim=-1)
    log_p_s = F.log_softmax(student_logits / temperature, dim=-1)
    kl = (log_p_t.exp() * (log_p_t - log_p_s)).sum(dim=-1)
    return temperature**2 * kl.mean()


def sparse_semi_structured_check(weight, inputs):
    """CUDA only. Compare a dense matmul with PyTorch's 2:4 sparse matmul.

    `weight` is a 2:4 pruned `(out, in)` tensor and `inputs` is
    `(tokens, in)`. On a CPU return None. On an Ampere or newer GPU, move
    both to half precision, convert the weight with
    `torch.sparse.to_sparse_semi_structured`, and return the largest absolute
    difference from the dense product. This lab was never run on a GPU.
    """
    if not torch.cuda.is_available():
        return None
    from torch.sparse import to_sparse_semi_structured

    w = weight.half().cuda()
    x = inputs.half().cuda()
    dense = F.linear(x, w)
    sparse_w = to_sparse_semi_structured(w)
    sparse = F.linear(x, sparse_w)
    return float((dense - sparse).abs().max())
