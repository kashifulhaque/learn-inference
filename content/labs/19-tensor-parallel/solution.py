import torch
import torch.nn.functional as F


def split_column_parallel(weight: torch.Tensor, world_size: int) -> list:
    return list(weight.chunk(world_size, dim=0))


def split_row_parallel(weight: torch.Tensor, world_size: int) -> list:
    return list(weight.chunk(world_size, dim=1))


def column_parallel_forward(x: torch.Tensor, shards: list) -> list:
    return [x @ shard.T for shard in shards]


def row_parallel_forward(x_shards: list, shards: list) -> torch.Tensor:
    partials = [x @ shard.T for x, shard in zip(x_shards, shards)]
    total = partials[0]
    for partial in partials[1:]:
        total = total + partial
    return total


def parallel_mlp(x, gate_w, up_w, down_w, world_size: int) -> torch.Tensor:
    gate_shards = split_column_parallel(gate_w, world_size)
    up_shards = split_column_parallel(up_w, world_size)
    down_shards = split_row_parallel(down_w, world_size)

    gate_out = column_parallel_forward(x, gate_shards)
    up_out = column_parallel_forward(x, up_shards)
    # The activation is elementwise, so each rank runs it on its own shard with
    # no communication.
    hidden = [F.silu(g) * u for g, u in zip(gate_out, up_out)]
    return row_parallel_forward(hidden, down_shards)


def split_attention_heads(num_heads: int, num_kv_heads: int, world_size: int):
    if num_heads % world_size:
        raise ValueError(
            f"{num_heads} query heads do not divide across {world_size} ranks"
        )
    if num_kv_heads % world_size:
        raise ValueError(
            f"{num_kv_heads} KV heads do not divide across {world_size} ranks; "
            "they would have to be replicated, which wastes cache"
        )
    return num_heads // world_size, num_kv_heads // world_size


def all_reduce_bytes(tokens: int, hidden: int, world_size: int,
                     bytes_per_element: int = 2) -> int:
    tensor_bytes = tokens * hidden * bytes_per_element
    return int(2 * (world_size - 1) / world_size * tensor_bytes)


def _chunk_bounds(length: int, world_size: int) -> list:
    """Split `length` elements into `world_size` chunks; the first
    `length % world_size` chunks get one extra element, so none is dropped."""
    base, extra = divmod(length, world_size)
    bounds, start = [], 0
    for i in range(world_size):
        size = base + (1 if i < extra else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def ring_all_reduce(shards: list):
    """Simulate a ring all-reduce over `len(shards)` ranks.

    Returns (results, bytes_sent), one entry per rank.
    """
    n = len(shards)
    shape = shards[0].shape
    bufs = [s.detach().clone().reshape(-1) for s in shards]
    bounds = _chunk_bounds(bufs[0].numel(), n)
    sent = [0] * n
    itemsize = bufs[0].element_size()

    # Reduce-scatter: at step s, rank r sends chunk (r - s) % n to rank r + 1,
    # which adds it into its own copy.
    for s in range(n - 1):
        msgs = []
        for r in range(n):
            lo, hi = bounds[(r - s) % n]
            msgs.append((r, (r - s) % n, bufs[r][lo:hi].clone()))
        for r, idx, payload in msgs:
            lo, hi = bounds[idx]
            bufs[(r + 1) % n][lo:hi] += payload
            sent[r] += payload.numel() * itemsize

    # All-gather: rank r now owns the finished chunk (r + 1) % n. At step s it
    # forwards chunk (r + 1 - s) % n, and the neighbour overwrites its copy.
    for s in range(n - 1):
        msgs = []
        for r in range(n):
            lo, hi = bounds[(r + 1 - s) % n]
            msgs.append((r, (r + 1 - s) % n, bufs[r][lo:hi].clone()))
        for r, idx, payload in msgs:
            lo, hi = bounds[idx]
            bufs[(r + 1) % n][lo:hi] = payload
            sent[r] += payload.numel() * itemsize

    return [b.reshape(shape) for b in bufs], sent


def all_reduce_time(num_bytes: float, world_size: int, alpha: float,
                    bandwidth: float) -> float:
    """Alpha-beta time of one ring all-reduce, in seconds."""
    if world_size <= 1:
        return 0.0
    steps = 2 * (world_size - 1)
    return steps * alpha + (steps / world_size) * num_bytes / bandwidth


def step_all_reduce_time(num_layers: int, message_bytes: float, world_size: int,
                         alpha: float, bandwidth: float) -> float:
    """All-reduce time for one forward pass: two per layer."""
    return 2 * num_layers * all_reduce_time(
        message_bytes, world_size, alpha, bandwidth)
