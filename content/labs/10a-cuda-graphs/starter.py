"""Lab 10a — batch buckets, padding waste, and a captured decode step.

`pick_bucket` and `padding_waste` run on the CPU. `failing_captures` is graded
on the CPU against the capture rules in the chapter. `capture_decode` runs
only when a CUDA device is visible. This machine's harness skips that check
when it is not, and it does not invent a GPU result.

The four snippets the harness passes to `failing_captures` are the ones in
the chapter:

- static_copy: `return x * scale`, after the caller copies new values in.
- host_item: `gate = x.sum().item()`.
- dynamic_rows: `return x[:n] * 2`, with a Python `n` that changes.
- allocator_cache: `torch.cuda.empty_cache()` before the arithmetic.
"""

from __future__ import annotations


# TODO: replace this power-of-two list. The harness trace wastes fewer
# padded slots on a tighter set, and this list is the set it has to beat.
BUCKETS = [1, 2, 4, 8, 16, 32]


def pick_bucket(batch: int, buckets: list[int]) -> int | None:
    """Smallest bucket that is at least `batch`.

    Return None when `batch` is larger than every bucket. `buckets` may be
    unsorted.
    """
    # TODO
    raise NotImplementedError


def padding_waste(trace: list[int], buckets: list[int]) -> float:
    """Idle padded slots divided by padded slots, for one bucket set.

    Each step of `trace` is a real batch size. Pad it with `pick_bucket`.
    Raise ValueError if any batch has no bucket.
    """
    # TODO
    raise NotImplementedError


def failing_captures(snippets: dict[str, str]) -> list[str]:
    """Sorted names of the snippets that fail the capture rules.

    This classification is the chapter's rule table. It is not a capture
    you ran on this CPU.
    """
    # TODO
    raise NotImplementedError


def capture_decode(model_step, static_inputs):
    """Capture `model_step(*static_inputs)` and return a replayable graph.

    Warm up on a side stream, then capture with `torch.cuda.graph`. The
    returned object has `replay()`, which runs the graph and returns the
    output tensor from capture. Addresses and shapes stay fixed. The caller
    changes contents with `copy_` before each replay.

    Needs a CUDA device. The harness skips this check when none is visible.
    """
    # TODO
    raise NotImplementedError
