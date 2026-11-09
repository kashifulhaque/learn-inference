"""Reference solution for batch buckets and a captured decode step.

`pick_bucket` chooses the smallest supplied bucket that can hold the batch.
It does not round the batch up to a power of two. `padding_waste` is the
share of padded slots that do no work. `failing_captures` applies the
chapter's capture rules to source text. `capture_decode` follows the
warmup-then-capture sequence in the PyTorch CUDA graphs note.
"""

from __future__ import annotations


# Tighter than powers of two. On the harness trace this leaves 150 idle
# slots out of 1220, against 402 out of 1472 for [1, 2, 4, 8, 16, 32].
BUCKETS = [1, 2, 4, 6, 8, 12, 16, 24, 32]


def pick_bucket(batch: int, buckets: list[int]) -> int | None:
    """Smallest bucket that is at least `batch`, or None if none fits."""
    fits = [size for size in buckets if size >= batch]
    if not fits:
        return None
    return min(fits)


def padding_waste(trace: list[int], buckets: list[int]) -> float:
    """Idle padded slots divided by padded slots."""
    real = 0
    padded = 0
    for batch in trace:
        bucket = pick_bucket(batch, buckets)
        if bucket is None:
            raise ValueError(f"batch {batch} exceeds every bucket")
        real += batch
        padded += bucket
    if padded == 0:
        return 0.0
    return (padded - real) / padded


def failing_captures(snippets: dict[str, str]) -> list[str]:
    """Names whose source breaks a capture rule, in sorted order.

    `.item()` is a host sync. `x[:n]` is a shape that follows a Python int.
    `torch.cuda.empty_cache()` returns cached blocks to the driver and
    synchronizes. A multiply of two fixed tensors does none of these.
    """
    failing = []
    for name, source in snippets.items():
        if _breaks_capture(source):
            failing.append(name)
    return sorted(failing)


def _breaks_capture(source: str) -> bool:
    if ".item(" in source:
        return True
    if "torch.cuda.empty_cache(" in source:
        return True
    if "x[:n]" in source:
        return True
    return False


class CapturedStep:
    """A graph and the output tensor capture wrote."""

    def __init__(self, graph, output) -> None:
        self.graph = graph
        self.output = output

    def replay(self):
        self.graph.replay()
        return self.output


def capture_decode(model_step, static_inputs):
    """Warm up on a side stream, capture, and return a `CapturedStep`."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("capture_decode needs a CUDA device")
    if not isinstance(static_inputs, (tuple, list)):
        static_inputs = (static_inputs,)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            model_step(*static_inputs)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = model_step(*static_inputs)
    # Keep `output` alive for the life of the graph. Replay writes it.
    return CapturedStep(graph, output)
