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


# Chapter 10 prices a launch at 5 µs, the low end of its 5 to 10 µs range.
LAUNCH_US = 5.0


def launch_floor_ms(num_kernels: int, launch_us: float = LAUNCH_US) -> float:
    """Time the launches alone take if the GPU waits on every one."""
    # TODO
    raise NotImplementedError


def kernel_time_ms(flops: float, bytes_moved: float, launch_us: float = LAUNCH_US,
                   spec: dict = A100_80GB) -> float:
    """One kernel's step time in milliseconds.

    A kernel that runs longer than a launch takes hides the next launch behind
    it. A shorter kernel leaves the GPU waiting for the next launch. Decide per
    kernel which one sets the time.
    """
    # TODO
    raise NotImplementedError


def step_time_ms(kernels: list[tuple[float, float]], launch_us: float = LAUNCH_US,
                 graph: bool = False, spec: dict = A100_80GB) -> float:
    """Time for a step made of (flops, bytes) kernels run one after another.

    Eager (graph=False): every kernel pays its own launch. Graph (graph=True):
    the whole step pays one launch, then every kernel runs at its roofline floor.
    """
    # TODO
    raise NotImplementedError


def classify_step(kernels: list[tuple[float, float]], launch_us: float = LAUNCH_US,
                  spec: dict = A100_80GB) -> str:
    """Return "launch", "memory", or "compute" for an eager step.

    Charge each kernel's time to what set it, then return the label with the
    most time.
    """
    # TODO
    raise NotImplementedError
