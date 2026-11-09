"""Launch floor and padding waste for chapter 10a.

The launch cost is chapter 10's 5 µs, the same value chapter 17a records in
docs/FACTS.md. This script does not measure a device. It multiplies that
cost by a kernel count, and it sums padding waste on a fixed trace.

Run from the repository root:

    python3 scripts/figures/10a-cuda-graphs-and-launch-overhead.py
"""

from __future__ import annotations

# Chapter 10's lower end of the 5 to 10 µs launch range. Chapter 17a uses it.
LAUNCH_US = 5
KERNELS = 400

# One graph replay is one launch of the captured step, at the same 5 µs.
GRAPH_LAUNCHES = 1

POW2 = [1, 2, 4, 8, 16, 32]
CHOSEN = [1, 2, 4, 6, 8, 12, 16, 24, 32]

# 200 decode steps. The chapter's table is this map, in ascending batch size.
COUNTS = {
    1: 40,
    2: 20,
    3: 20,
    5: 30,
    6: 24,
    7: 12,
    8: 10,
    9: 16,
    10: 12,
    12: 8,
    13: 4,
    15: 4,
}


def pick_bucket(batch: int, buckets: list[int]) -> int:
    fits = [size for size in buckets if size >= batch]
    if not fits:
        raise ValueError(f"batch {batch} exceeds every bucket in {buckets}")
    return min(fits)


def trace() -> list[int]:
    steps: list[int] = []
    for batch, count in COUNTS.items():
        steps.extend([batch] * count)
    return steps


def totals(steps: list[int], buckets: list[int]) -> tuple[int, int, int]:
    """Return (real slots, padded slots, idle slots)."""
    real = 0
    padded = 0
    for batch in steps:
        real += batch
        padded += pick_bucket(batch, buckets)
    return real, padded, padded - real


def main() -> None:
    eager_us = KERNELS * LAUNCH_US
    graph_us = GRAPH_LAUNCHES * LAUNCH_US
    removed_us = eager_us - graph_us
    print(f"eager launches: {KERNELS} x {LAUNCH_US} µs = {eager_us} µs = {eager_us / 1000:.3f} ms")
    print(f"graph launch: {GRAPH_LAUNCHES} x {LAUNCH_US} µs = {graph_us} µs = {graph_us / 1000:.3f} ms")
    print(f"removed: {KERNELS - GRAPH_LAUNCHES} x {LAUNCH_US} µs = {removed_us} µs = {removed_us / 1000:.3f} ms")

    steps = trace()
    print(f"steps: {len(steps)}")
    real, pow_padded, pow_idle = totals(steps, POW2)
    _, chosen_padded, chosen_idle = totals(steps, CHOSEN)
    print(f"real slots: {real}")
    print(f"powers of two: idle {pow_idle} / padded {pow_padded} = {pow_idle}/{pow_padded}")
    print(f"chosen buckets: idle {chosen_idle} / padded {chosen_padded} = {chosen_idle}/{chosen_padded}")
    print("batch  steps  pow2  chosen  idle_pow2  idle_chosen")
    for batch, count in COUNTS.items():
        pow_b = pick_bucket(batch, POW2)
        chosen_b = pick_bucket(batch, CHOSEN)
        print(
            f"{batch:5d}  {count:5d}  {pow_b:4d}  {chosen_b:6d}  "
            f"{pow_b - batch:9d}  {chosen_b - batch:11d}"
        )

    hand = [1, 3, 5]
    _, hand_pow_padded, hand_pow_idle = totals(hand, [1, 2, 4, 8])
    _, hand_ch_padded, hand_ch_idle = totals(hand, [1, 2, 4, 6, 8])
    print(
        f"hand powers of two: idle {hand_pow_idle} / padded {hand_pow_padded}"
    )
    print(
        f"hand chosen: idle {hand_ch_idle} / padded {hand_ch_padded}"
    )


if __name__ == "__main__":
    main()
