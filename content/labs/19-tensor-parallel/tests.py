import math

import torch
import torch.nn.functional as F
from lab_common import Checks, close, pick_device

HIDDEN, INTERMEDIATE = 256, 704
HEADS, KV_HEADS, HEAD_DIM = 24, 4, 64


def _partition_checks(c, submission):
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

# Constants from the chapter. ALPHA is an assumption: the latency of one ring
# hop. NVLINK_ONE_WAY is 12 links x 25 GB/s per direction, which is half of the
# 600 GB/s NVIDIA quotes for the total of both directions.
HIDDEN_FULL, LAYERS, BYTES = 5120, 64, 2
ALPHA, NVLINK_ONE_WAY = 4e-6, 300e9
DECODE_MSG = HIDDEN_FULL * BYTES
PREFILL_MSG = 4096 * HIDDEN_FULL * BYTES


def _collective_checks(c, submission):
    torch.manual_seed(19)

    def ring(n, length, dtype=torch.float64):
        shards = [torch.randn(length, dtype=dtype) for _ in range(n)]
        results, sent = submission.ring_all_reduce(shards)
        return shards, results, sent

    def worst_error(shards, results):
        ref = torch.stack(shards).sum(0)
        return max((r - ref).abs().max().item() for r in results)

    shards, results, sent = ring(4, 12)
    err = worst_error(shards, results)
    c.check("ring all-reduce: every rank holds the sum of the shards",
            lambda: (len(results) == 4 and err < 1e-12,
                     f"{len(results)} results, max abs error {err:.2e}"))
    c.check("ring all-reduce: all ranks end with identical values",
            lambda: all(torch.equal(results[0], r) for r in results[1:]))

    # Catches a chunking bug that drops the remainder: with length 10 on 4
    # ranks, floor(10 / 4) = 2 elements per chunk would leave two elements
    # unreduced.
    for n, length in ((4, 10), (3, 7), (8, 5)):
        shards, results, _ = ring(n, length)
        err = worst_error(shards, results)
        c.check(f"ring all-reduce handles length {length} on {n} ranks "
                "(not divisible)",
                lambda e=err: (e < 1e-12, f"max abs error {e:.2e}, expected 0"))

    inputs = [torch.randn(12, dtype=torch.float64) for _ in range(4)]
    before = [s.clone() for s in inputs]
    submission.ring_all_reduce(inputs)
    c.check("ring all-reduce does not modify its inputs",
            lambda: all(torch.equal(a, b) for a, b in zip(inputs, before)))

    shards, results, sent = ring(3, 6)
    c.check("ring all-reduce keeps the input shape",
            lambda: submission.ring_all_reduce(
                [torch.ones(2, 3), torch.ones(2, 3)])[0][0].shape == (2, 3))
    _, one_sent = submission.ring_all_reduce([torch.ones(5)])
    c.check("one rank sends nothing", lambda: list(one_sent) == [0],
            f"expected [0], got {one_sent}")

    # Catches naive all-to-one accounting, (N - 1) * message bytes per rank,
    # instead of the ring's 2 * (N - 1) / N * message bytes.
    for n in (2, 4, 8):
        length = 1024
        message = length * 4
        expected = int(2 * (n - 1) / n * message)
        _, _, got = ring(n, length, torch.float32)
        c.check(f"bytes sent per rank on {n} ranks equal 2(N-1)/N of the message",
                lambda g=got, e=expected: (list(g) == [e] * len(g),
                                           f"expected {[e] * len(g)}, got {list(g)}"))
    n, length = 4, 10
    _, _, got = ring(n, length, torch.float32)
    expected_total = 2 * (n - 1) * length * 4
    c.check("a non-divisible length still moves 2(N-1) messages in total",
            lambda: (sum(got) == expected_total,
                     f"expected {expected_total} bytes in total, got {sum(got)}"))

    def one_way(num_bytes, n):
        return submission.all_reduce_time(num_bytes, n, ALPHA, NVLINK_ONE_WAY)

    expected = 2 * 1 * ALPHA + 2 * 1 / 2 * 1e6 / NVLINK_ONE_WAY
    got = one_way(1e6, 2)
    c.check("all_reduce_time matches the alpha-beta formula at N = 2",
            lambda: (math.isclose(got, expected, rel_tol=1e-9),
                     f"expected {expected:.6e} s, got {got:.6e} s"))
    expected = 2 * 3 * ALPHA + 2 * 3 / 4 * 1e6 / NVLINK_ONE_WAY
    got = one_way(1e6, 4)
    c.check("all_reduce_time matches the alpha-beta formula at N = 4",
            lambda: (math.isclose(got, expected, rel_tol=1e-9),
                     f"expected {expected:.6e} s, got {got:.6e} s"))
    c.check("all_reduce_time is zero on one rank", lambda: one_way(1e6, 1) == 0.0)

    for n in (2, 4, 8):
        total = one_way(DECODE_MSG, n)
        latency = 2 * (n - 1) * ALPHA
        share = latency / total
        c.check(f"a {DECODE_MSG}-byte decode message on {n} ranks is latency bound",
                lambda s=share: (s > 0.99,
                                 f"expected latency share above 0.99, got {s:.4f}"))
    total = one_way(PREFILL_MSG, 2)
    latency = 2 * ALPHA
    c.check("a 4096-token prefill message is bandwidth bound",
            lambda: (latency / total < 0.1,
                     f"expected latency share below 0.1, got {latency / total:.4f}"))
    n = 4
    crossover = n * ALPHA * NVLINK_ONE_WAY
    ratio = one_way(crossover, n) / (2 * (2 * (n - 1) * ALPHA))
    c.check("at S = N * alpha * bandwidth the two terms are equal",
            lambda: (math.isclose(ratio, 1.0, rel_tol=1e-9),
                     f"expected total / (2 x latency) = 1, got {ratio:.6f}"))

    expected_ms = {2: 1.028, 4: 3.079, 8: 7.176}
    got_ms = {n: submission.step_all_reduce_time(
        LAYERS, DECODE_MSG, n, ALPHA, NVLINK_ONE_WAY) * 1e3 for n in (2, 4, 8)}
    for n, e in expected_ms.items():
        c.check(f"128 decode all-reduces per token on {n} ranks",
                lambda n=n, e=e: (abs(got_ms[n] - e) < 0.005,
                                  f"expected {e} ms, got {got_ms[n]:.4f} ms"))
    c.metric("decode_all_reduce_ms_tp2", round(got_ms[2], 3))
    c.metric("decode_all_reduce_ms_tp8", round(got_ms[8], 3))


def run(submission):
    c = Checks()
    needed = ("split_column_parallel", "split_row_parallel",
              "column_parallel_forward", "row_parallel_forward", "parallel_mlp",
              "split_attention_heads", "all_reduce_bytes", "ring_all_reduce",
              "all_reduce_time", "step_all_reduce_time")
    if not c.require(submission, *needed):
        return c.finish()
    for name, fn in (("collective checks", _collective_checks),
                     ("partition checks", _partition_checks)):
        try:
            fn(c, submission)
        except Exception as exc:  # noqa: BLE001
            c.add(f"{name} ran to completion", False,
                  f"{type(exc).__name__}: {exc}")
    return c.finish()


def _raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False
