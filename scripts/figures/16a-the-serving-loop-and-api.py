"""Idle-fraction arithmetic for chapter 16a.

The clock is simulated. A step is a CPU schedule of S ticks, a GPU forward
of F ticks, and a CPU post stage (sample, detokenize, stream) of P ticks.
The GPU is busy only during the forward.

Serial: the three stages run one after another.
    wall = n * (S + F + P)
    idle = n * (S + P)

Overlap: the next step's schedule runs during the current forward. The first
schedule cannot hide, and the post stage stays after the forward because
sampling needs the logits. A schedule longer than the forward leaves a
leftover of max(0, S - F) on each of the n - 1 overlapped steps.
    idle = S + (n - 1) * max(0, S - F) + n * P
    wall = idle + n * F

The chapter's worked point is S=6, F=10, P=2, n=8. This machine has no GPU.
The numbers are tick counts, not A100 timings. The lab measures the same
ratio under the name gpu_idle_fraction.
"""

from __future__ import annotations


def idle_and_wall(n: int, schedule: int, forward: int, post: int, overlap: bool):
    busy = n * forward
    if not overlap:
        wall = n * (schedule + forward + post)
    else:
        leftover = max(0, schedule - forward)
        idle = schedule + (n - 1) * leftover + n * post
        wall = idle + busy
    idle = wall - busy
    return idle, wall, busy


def main() -> None:
    n, schedule, forward, post = 8, 6, 10, 2
    print(f"n={n} S={schedule} F={forward} P={post}")
    for overlap in (False, True):
        idle, wall, busy = idle_and_wall(n, schedule, forward, post, overlap)
        name = "overlap" if overlap else "serial"
        print(
            f"{name}: idle={idle} wall={wall} busy={busy} "
            f"fraction={idle / wall:.6f} = {idle}/{wall}"
        )


if __name__ == "__main__":
    main()
