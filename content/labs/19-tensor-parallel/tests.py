import math

import torch
import torch.nn.functional as F
from lab_common import Checks, close, pick_device

HIDDEN, INTERMEDIATE = 256, 704
HEADS, KV_HEADS, HEAD_DIM = 24, 4, 64


def run(submission):
    c = Checks()
    needed = ("split_column_parallel", "split_row_parallel",
              "column_parallel_forward", "row_parallel_forward", "parallel_mlp",
              "split_attention_heads", "all_reduce_bytes")
    if not c.require(submission, *needed):
        return c.finish()

    device = pick_device()
    torch.manual_seed(0)

    weight = torch.randn(64, 32, device=device)
    col = submission.split_column_parallel(weight, 4)
    c.check("column parallel splits the output dimension",
            lambda: all(tuple(s.shape) == (16, 32) for s in col),
            f"got {[tuple(s.shape) for s in col]}")
    c.check("concatenating column shards rebuilds the weight",
            lambda: close(torch.cat(col, dim=0), weight, 0))

    row = submission.split_row_parallel(weight, 4)
    c.check("row parallel splits the input dimension",
            lambda: all(tuple(s.shape) == (64, 8) for s in row),
            f"got {[tuple(s.shape) for s in row]}")
    c.check("concatenating row shards rebuilds the weight",
            lambda: close(torch.cat(row, dim=1), weight, 0))

    x = torch.randn(8, 32, device=device)
    outputs = submission.column_parallel_forward(x, col)
    c.check("column-parallel output concatenates to the unsharded result",
            lambda: close(torch.cat(outputs, dim=-1), x @ weight.T, 1e-4))

    x_full = torch.randn(8, 32, device=device)
    x_shards = list(x_full.chunk(4, dim=-1))
    reduced = submission.row_parallel_forward(x_shards, row)
    c.check("row-parallel partial sums add to the unsharded result",
            lambda: close(reduced, x_full @ weight.T, 1e-4))

    gate_w = torch.randn(INTERMEDIATE, HIDDEN, device=device) * 0.05
    up_w = torch.randn(INTERMEDIATE, HIDDEN, device=device) * 0.05
    down_w = torch.randn(HIDDEN, INTERMEDIATE, device=device) * 0.05
    x = torch.randn(16, HIDDEN, device=device)
    reference = (F.silu(x @ gate_w.T) * (x @ up_w.T)) @ down_w.T

    worst = 0.0
    for world_size in (1, 2, 4):
        mine = submission.parallel_mlp(x, gate_w, up_w, down_w, world_size)
        err = (mine - reference).abs().max().item()
        worst = max(worst, err)
        c.check(f"the {world_size}-way parallel MLP matches the unsharded block",
                lambda e=err: (e < 1e-3, f"max abs error {e:.2e}"))

    c.check("24 heads and 4 KV heads split evenly across 2 ranks",
            lambda: submission.split_attention_heads(HEADS, KV_HEADS, 2) == (12, 2),
            f"got {submission.split_attention_heads(HEADS, KV_HEADS, 2)}")
    c.check("they split across 4 ranks too",
            lambda: submission.split_attention_heads(HEADS, KV_HEADS, 4) == (6, 1))
    c.check("8-way parallelism is rejected: 4 KV heads cannot divide by 8",
            lambda: _raises(
                lambda: submission.split_attention_heads(HEADS, KV_HEADS, 8)))

    # Attention itself: each rank owns whole heads and needs no communication
    # until the output projection.
    q = torch.randn(1, HEADS, 16, HEAD_DIM, device=device)
    k = torch.randn(1, KV_HEADS, 16, HEAD_DIM, device=device)
    v = torch.randn(1, KV_HEADS, 16, HEAD_DIM, device=device)
    group = HEADS // KV_HEADS

    def attend(q_, k_, v_):
        k_ = k_.repeat_interleave(q_.shape[1] // k_.shape[1], dim=1)
        v_ = v_.repeat_interleave(q_.shape[1] // v_.shape[1], dim=1)
        scores = (q_ @ k_.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
        mask = torch.triu(torch.ones(16, 16, dtype=torch.bool, device=device), 1)
        return torch.softmax(scores.masked_fill(mask, float("-inf")), -1) @ v_

    full = attend(q, k, v)
    world_size = 2
    heads_per, kv_per = submission.split_attention_heads(HEADS, KV_HEADS, world_size)
    shards = [
        attend(
            q[:, rank * heads_per : (rank + 1) * heads_per],
            k[:, rank * kv_per : (rank + 1) * kv_per],
            v[:, rank * kv_per : (rank + 1) * kv_per],
        )
        for rank in range(world_size)
    ]
    attn_err = (torch.cat(shards, dim=1) - full).abs().max().item()
    c.check("splitting attention by head reproduces the unsharded output",
            lambda: (attn_err < 1e-4, f"max abs error {attn_err:.2e}"))

    single = submission.all_reduce_bytes(4096, 5120, 1)
    two = submission.all_reduce_bytes(4096, 5120, 2)
    four = submission.all_reduce_bytes(4096, 5120, 4)
    c.check("a single rank communicates nothing", lambda: single == 0,
            f"got {single}")
    c.check("2-way moves one tensor's worth per rank",
            lambda: two == 4096 * 5120 * 2, f"got {two:,}")
    c.check("more ranks move more, approaching twice the tensor size",
            lambda: two < four < 2 * 4096 * 5120 * 2)

    c.metric("mlp_error", float(f"{worst:.3e}"))
    c.metric("attention_error", float(f"{attn_err:.3e}"))
    c.metric("comm_bytes_per_layer", 2 * two)
    return c.finish()


def _raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False
