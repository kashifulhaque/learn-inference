"""Lab 19 — tensor parallelism.

You do not need two GPUs to check the partitioning. Simulate every rank on one
device and confirm the combined result matches the unsharded layer.

Weights follow the PyTorch convention: an nn.Linear with `in` inputs and `out`
outputs stores a weight of shape (out, in), and computes x @ weight.T.
"""

import torch


def split_column_parallel(weight: torch.Tensor, world_size: int) -> list:
    """Split a weight by output features, one shard per rank.

    Args:
        weight: (out_features, in_features)

    Returns:
        `world_size` tensors of shape (out_features // world_size, in_features).
    """
    # TODO
    raise NotImplementedError


def split_row_parallel(weight: torch.Tensor, world_size: int) -> list:
    """Split a weight by input features, one shard per rank.

    Returns:
        `world_size` tensors of shape (out_features, in_features // world_size).
    """
    # TODO
    raise NotImplementedError


def column_parallel_forward(x: torch.Tensor, shards: list) -> list:
    """Run a column-parallel layer. Input is replicated, output is sharded."""
    # TODO
    raise NotImplementedError


def row_parallel_forward(x_shards: list, shards: list) -> torch.Tensor:
    """Run a row-parallel layer and all-reduce the partial sums.

    Args:
        x_shards: One input shard per rank, split along the feature dimension.
    """
    # TODO
    raise NotImplementedError


def parallel_mlp(x, gate_w, up_w, down_w, world_size: int) -> torch.Tensor:
    """A tensor-parallel SwiGLU block.

    gate and up are column parallel, down is row parallel, and the activation
    runs independently on each shard. One all-reduce, at the end.
    """
    # TODO
    raise NotImplementedError


def split_attention_heads(num_heads: int, num_kv_heads: int, world_size: int):
    """Return (heads_per_rank, kv_heads_per_rank).

    Raise ValueError when the KV heads do not divide evenly, because that would
    force replication.
    """
    # TODO
    raise NotImplementedError


def all_reduce_bytes(tokens: int, hidden: int, world_size: int,
                     bytes_per_element: int = 2) -> int:
    """Bytes one rank sends and receives for a ring all-reduce.

    A ring all-reduce moves 2 * (world_size - 1) / world_size times the tensor
    size per rank.
    """
    # TODO
    raise NotImplementedError


def ring_all_reduce(shards: list):
    """Simulate a ring all-reduce over len(shards) ranks.

    Args:
        shards: N tensors of the same shape, one per rank.

    Returns:
        (results, bytes_sent): the per-rank results, each equal to the sum of
        the shards, and the bytes each rank sent.

    Reduce-scatter first: for N - 1 steps, rank r sends chunk (r - step) % N to
    rank r + 1, which adds it to its own copy. Then all-gather: for N - 1
    steps, rank r forwards the finished chunk (r + 1 - step) % N, and the
    neighbour overwrites its copy. Cut the flattened tensor into N chunks that
    cover every element even when the length is not divisible by N. Do not
    modify the inputs.
    """
    # TODO
    raise NotImplementedError


def all_reduce_time(num_bytes: float, world_size: int, alpha: float,
                    bandwidth: float) -> float:
    """Alpha-beta time, in seconds, of one ring all-reduce.

    time = 2 (N - 1) alpha + 2 (N - 1) / N * num_bytes / bandwidth

    `alpha` is the latency of one hop in seconds and `bandwidth` is bytes per
    second in one direction. Return 0.0 for a single rank.
    """
    # TODO
    raise NotImplementedError


def step_all_reduce_time(num_layers: int, message_bytes: float, world_size: int,
                         alpha: float, bandwidth: float) -> float:
    """All-reduce time for one forward pass: two all-reduces per layer."""
    # TODO
    raise NotImplementedError
