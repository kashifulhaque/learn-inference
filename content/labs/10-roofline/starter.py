"""Lab 10 — roofline analysis.

Decide what limits each operation before trying to make it faster.
"""

A100_80GB = {"bf16_tflops": 312.0, "hbm_bandwidth_gbs": 1935.0}


def matmul_flops(m: int, k: int, n: int) -> int:
    """FLOPs for an (m, k) by (k, n) matrix multiply."""
    # TODO
    raise NotImplementedError


def mlp_cost(tokens: int, hidden: int, intermediate: int,
             bytes_per_element: int = 2) -> tuple[int, int]:
    """Return (flops, bytes) for one SwiGLU block over `tokens` tokens.

    Count the three matmuls. For bytes, count the weights read once, plus the
    input read and the output written. Ignore the transient intermediate.
    """
    # TODO
    raise NotImplementedError


def decode_attention_cost(context_len: int, num_heads: int, num_kv_heads: int,
                          head_dim: int, bytes_per_element: int = 2
                          ) -> tuple[int, int]:
    """Return (flops, bytes) for one decode step's attention in one layer.

    The query is a single token. Scores are one matmul against the cached keys,
    and the output is one matmul against the cached values. Bytes are the K and
    V reads.
    """
    # TODO
    raise NotImplementedError


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOPs per byte."""
    # TODO
    raise NotImplementedError


def ridge_point(spec: dict = A100_80GB) -> float:
    """Intensity at which the compute and bandwidth limits meet."""
    # TODO
    raise NotImplementedError


def roofline(flops: float, bytes_moved: float, spec: dict = A100_80GB) -> dict:
    """Return a lower bound on runtime and what limits it.

    Returns a dict with "compute_ms", "memory_ms", "floor_ms", "intensity", and
    "bound_by", which is either "compute" or "memory".
    """
    # TODO
    raise NotImplementedError
