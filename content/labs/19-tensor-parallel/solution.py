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
