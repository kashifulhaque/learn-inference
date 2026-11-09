"""Occupancy, split counts, and a log-sum-exp merge for chapter 14a.

The SM count is 108 and the copy bandwidth is 1275 GB/s, the same pair
chapter 10 uses. Nothing here is an A100 kernel measurement. The split
count is the heuristic in the lab solution: fill 80% of the SMs before
splitting, then take the smallest split count whose wave efficiency is
within 85% of the best, and cap that count by the number of 64-token
KV blocks.

FlashAttention's launcher (flash_api.cpp, num_splits_heuristic) also
drops split counts that do not shorten the blocks. That filter can step
the count down when the cache grows by one block. This chapter omits it,
so a longer cache never asks for fewer splits.
"""

from __future__ import annotations

import math

SMS = 108
COPY_GBPS = 1275  # measured device-to-device copy, chapter 10
BLOCK_N = 64  # FlashAttention split-KV tile at head_dim 256
QUERY_HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
MAX_SPLITS = 128


def choose_num_splits(batch: int, heads: int, kv_len: int, num_sms: int) -> int:
    """Same rule as content/labs/14a-split-kv-decode/solution.py."""
    if batch < 1 or heads < 1 or kv_len < 1 or num_sms < 1:
        return 1
    tasks = batch * heads
    # 80% of the SMs, in integers: tasks / num_sms >= 4/5.
    if tasks * 5 >= num_sms * 4:
        return 1
    limit = min(MAX_SPLITS, num_sms)
    best = 0.0
    effs: list[float] = []
    for splits in range(1, limit + 1):
        waves = (tasks * splits + num_sms - 1) // num_sms
        eff = (tasks * splits) / (num_sms * waves)
        effs.append(eff)
        if eff > best:
            best = eff
    cutoff = 0.85 * best
    ideal = 1
    for splits, eff in enumerate(effs, start=1):
        if eff >= cutoff:
            ideal = splits
            break
    n_blocks = (kv_len + BLOCK_N - 1) // BLOCK_N
    return max(1, min(ideal, n_blocks))


def full_softmax(scores: list[float], values: list[list[float]]):
    m = max(scores)
    weights = [math.exp(s - m) for s in scores]
    denom = sum(weights)
    dim = len(values[0])
    out = [0.0] * dim
    for w, row in zip(weights, values):
        for i, item in enumerate(row):
            out[i] += w * item
    out = [item / denom for item in out]
    return out, m + math.log(denom)


def partial(scores: list[float], values: list[list[float]]):
    return full_softmax(scores, values)


def merge(parts: list[tuple[list[float], float]]):
    finite = [(out, lse) for out, lse in parts if math.isfinite(lse)]
    if not finite:
        return [0.0] * len(parts[0][0]), float("-inf")
    m = max(lse for _out, lse in finite)
    weights = [math.exp(lse - m) for _out, lse in finite]
    denom = sum(weights)
    dim = len(finite[0][0])
    out = [0.0] * dim
    for w, (row, _lse) in zip(weights, finite):
        for i, item in enumerate(row):
            out[i] += w * item
    out = [item / denom for item in out]
    return out, m + math.log(denom)


def main() -> None:
    blocks = QUERY_HEADS  # batch 1, query length 1
    util = blocks / SMS
    implied = COPY_GBPS * util
    filled_blocks = 96  # 24 heads * 4 splits
    filled_util = filled_blocks / SMS
    filled_gbs = COPY_GBPS * filled_util
    kv_tokens = 32768
    kv_bytes = 2 * KV_HEADS * HEAD_DIM * 2 * kv_tokens
    print(f"unsplit_blocks {blocks}")
    print(f"unsplit_util {util:.6f}")
    print(f"unsplit_gbs {implied:.6f}")
    print(f"split4_blocks {filled_blocks}")
    print(f"split4_util {filled_util:.6f}")
    print(f"split4_gbs {filled_gbs:.6f}")
    print(f"kv_bytes_32k {kv_bytes}")
    print(f"us_at_1275 {(kv_bytes / (COPY_GBPS * 1e9)) * 1e6:.4f}")
    print(f"us_at_unsplit {(kv_bytes / (implied * 1e9)) * 1e6:.4f}")
    print(f"us_at_split4 {(kv_bytes / (filled_gbs * 1e9)) * 1e6:.4f}")
    partial_bytes = QUERY_HEADS * 4 * (HEAD_DIM * 4 + 4)
    print(f"reduction_bytes_4 {partial_bytes}")
    print(f"reduction_over_kv {partial_bytes / kv_bytes:.6e}")

    lengths = [1, 17, 64, 65, 128, 192, 256, 4096, 32768, 65536]
    print("kv splits blocks")
    prev = 0
    for length in lengths:
        splits = choose_num_splits(1, QUERY_HEADS, length, SMS)
        if splits < prev:
            raise SystemExit(f"not monotone at {length}")
        prev = splits
        print(f"{length} {splits} {(length + BLOCK_N - 1) // BLOCK_N}")

    print("full grid", choose_num_splits(1, 108, 65536, 108))
    print("eighty", choose_num_splits(1, 87, 65536, 108))
    print("just under", choose_num_splits(1, 86, 65536, 108))
    print("batch8", choose_num_splits(8, 24, 65536, 108))
    print("one head", choose_num_splits(1, 1, 65536, 108))

    # Monotone on a dense sweep, three grids.
    for batch, heads in ((1, 24), (1, 1), (2, 8)):
        last = 0
        for length in range(1, 8193):
            splits = choose_num_splits(batch, heads, length, SMS)
            if splits < last:
                raise SystemExit(f"dip at batch {batch} heads {heads} len {length}")
            last = splits
        print(f"monotone batch {batch} heads {heads} final {last}")

    scores = [0.0, 2.0, 1.0]
    values = [[4.0], [6.0], [10.0]]
    out, lse = full_softmax(scores, values)
    a = partial([0.0, 2.0], [[4.0], [6.0]])
    b = partial([1.0], [[10.0]])
    empty = ([0.0], float("-inf"))
    merged, mlse = merge([a, b, empty])
    print(f"chunk_a out {a[0][0]:.6f} lse {a[1]:.6f}")
    print(f"chunk_b out {b[0][0]:.6f} lse {b[1]:.6f}")
    print(f"full out {out[0]:.6f} lse {lse:.6f}")
    print(f"merged out {merged[0]:.6f} lse {mlse:.6f}")
    # Unscaled add, the wrong merge the lab rejects.
    wrong = (a[0][0] + b[0][0]) / 2
    print(f"unscaled_mean {wrong:.6f}")

    # Pairwise associativity on three unequal chunks.
    c0 = partial([0.0], [[4.0]])
    c1 = partial([2.0], [[6.0]])
    c2 = partial([1.0], [[10.0]])
    left, _ = merge([merge([c0, c1]), c2])
    right, _ = merge([c0, merge([c1, c2])])
    print(f"assoc_left {left[0]:.8f} assoc_right {right[0]:.8f}")
    _check_solution()


def _check_solution() -> None:
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "content"
        / "labs"
        / "14a-split-kv-decode"
        / "solution.py"
    )
    spec = importlib.util.spec_from_file_location("sol_14a", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    samples = [
        (1, 24, 64, 108),
        (1, 24, 65, 108),
        (1, 24, 256, 108),
        (1, 24, 65536, 108),
        (1, 108, 65536, 108),
        (1, 87, 65536, 108),
        (1, 86, 65536, 108),
        (1, 1, 65536, 108),
        (2, 8, 8192, 108),
        (8, 24, 65536, 108),
    ]
    for args in samples:
        got = module.choose_num_splits(*args)
        expected = choose_num_splits(*args)
        if got != expected:
            raise SystemExit(f"solution {got} != script {expected} for {args}")
    print("solution matches script")


if __name__ == "__main__":
    main()
