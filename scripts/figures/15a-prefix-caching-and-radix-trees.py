"""Checkpoint placement and the prefill time a hit rate skips.

The placement is not a measured hit rate. One hundred prompts stand in for
a workload: 40 of length 256, 30 of length 512, 20 of length 1,024, and 10
of length 2,048. A checkpoint is one recurrent state, 154,927,104 bytes,
the count chapter 2 derives (3,227,648 bytes in one linear layer, times 48).

A prompt of length L uses the rightmost checkpoint at or before L, and that
position is the number of prefill tokens the snapshot saves. The search
picks the lexicographically smallest list among placements with the same
total.

The prefill term is chapter 15's 2NT count, with the exact parameter count
from chapter 2 and the 312 TFLOP/s bfloat16 peak from chapter 10. It is not
chapter 17a's prefill floor, and it is not an A100 measurement.
"""

from __future__ import annotations

from itertools import combinations

STATE_BYTES = 154_927_104
LENGTHS = [256] * 40 + [512] * 30 + [1024] * 20 + [2048] * 10
CANDIDATES = (256, 512, 1024, 2048)
PARAMS = 26_895_979_520
PEAK = 312 * 10**12


def saved(positions: tuple[int, ...]) -> int:
    total = 0
    for length in LENGTHS:
        best = 0
        for position in positions:
            if 0 < position <= length and position > best:
                best = position
        total += best
    return total


def place(slots: int) -> tuple[list[int], int]:
    best: list[int] = []
    best_score = -1
    for combo in combinations(CANDIDATES, slots):
        score = saved(combo)
        chosen = list(combo)
        if score > best_score or (score == best_score and chosen < best):
            best_score = score
            best = chosen
    return best, best_score


def main() -> None:
    print(f"prompts {len(LENGTHS)}")
    print(f"state_bytes {STATE_BYTES}")
    print(f"state_mib {STATE_BYTES / 2**20:.2f}")
    every_block = (2048 // 16) * STATE_BYTES
    print(f"every_16_tokens_of_2048_bytes {every_block}")
    print(f"every_16_tokens_of_2048_gib {every_block / 2**30:.5f}")
    print("k positions total mean bytes mib")
    for slots in range(0, 5):
        positions, total = place(slots)
        bytes_ = slots * STATE_BYTES
        shown = ",".join(str(p) for p in positions) if positions else "-"
        print(
            f"{slots} {shown} {total} {total / len(LENGTHS):.1f} "
            f"{bytes_} {bytes_ / 2**20:.2f}"
        )
    flops = 2 * PARAMS * 500
    seconds = flops / PEAK
    print(f"prefill_500_flops {flops}")
    print(f"prefill_500_ms {seconds * 1000:.3f}")
    print(f"hit_0.75_saved_tokens {0.75 * 500:.0f}")
    print(f"hit_0.75_saved_ms {0.75 * seconds * 1000:.3f}")
    print(f"hit_0.75_remain_ms {0.25 * seconds * 1000:.3f}")
    print(f"shared_prompt_hit {99 * 32}/{100 * 48} = {99 * 32 / (100 * 48):.2f}")


if __name__ == "__main__":
    main()
