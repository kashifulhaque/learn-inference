A100_80GB = {"bf16_tflops": 312.0, "hbm_bandwidth_gbs": 1935.0}


def matmul_flops(m: int, k: int, n: int) -> int:
    return 2 * m * k * n


def mlp_cost(tokens: int, hidden: int, intermediate: int,
             bytes_per_element: int = 2) -> tuple[int, int]:
    flops = (
        2 * matmul_flops(tokens, hidden, intermediate)   # gate and up
        + matmul_flops(tokens, intermediate, hidden)     # down
    )
    weight_bytes = 3 * hidden * intermediate * bytes_per_element
    activation_bytes = 2 * tokens * hidden * bytes_per_element
    return flops, weight_bytes + activation_bytes


def decode_attention_cost(context_len: int, num_heads: int, num_kv_heads: int,
                          head_dim: int, bytes_per_element: int = 2
                          ) -> tuple[int, int]:
    flops = 2 * (2 * num_heads * context_len * head_dim)
    bytes_moved = 2 * num_kv_heads * context_len * head_dim * bytes_per_element
    return flops, bytes_moved


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    return flops / max(bytes_moved, 1.0)


def ridge_point(spec: dict = A100_80GB) -> float:
    return (spec["bf16_tflops"] * 1e12) / (spec["hbm_bandwidth_gbs"] * 1e9)


def roofline(flops: float, bytes_moved: float, spec: dict = A100_80GB) -> dict:
    compute_ms = flops / (spec["bf16_tflops"] * 1e12) * 1e3
    memory_ms = bytes_moved / (spec["hbm_bandwidth_gbs"] * 1e9) * 1e3
    return {
        "compute_ms": compute_ms,
        "memory_ms": memory_ms,
        "floor_ms": max(compute_ms, memory_ms),
        "intensity": arithmetic_intensity(flops, bytes_moved),
        "bound_by": "compute" if compute_ms > memory_ms else "memory",
    }


# Chapter 10 prices a launch at 5 µs, the low end of its 5 to 10 µs range.
LAUNCH_US = 5.0


def launch_floor_ms(num_kernels: int, launch_us: float = LAUNCH_US) -> float:
    """Time the launches alone take if the GPU waits on every one."""
    return num_kernels * launch_us / 1e3


def kernel_time_ms(flops: float, bytes_moved: float, launch_us: float = LAUNCH_US,
                   spec: dict = A100_80GB) -> float:
    """One kernel's step time: the larger of its launch floor and its roofline floor.

    A kernel that runs longer than a launch takes hides the next launch behind
    it, so the launch costs nothing. A shorter kernel leaves the GPU waiting
    for the next launch, so the launch sets the time.
    """
    return max(launch_floor_ms(1, launch_us), roofline(flops, bytes_moved, spec)["floor_ms"])


def step_time_ms(kernels: list[tuple[float, float]], launch_us: float = LAUNCH_US,
                 graph: bool = False, spec: dict = A100_80GB) -> float:
    """Time for a step made of (flops, bytes) kernels run one after another.

    Eager: take the max per kernel, then sum. Graph: pay one launch for the
    whole step, then every kernel runs at its roofline floor.
    """
    if graph:
        floors = sum(roofline(f, b, spec)["floor_ms"] for f, b in kernels)
        return launch_floor_ms(1, launch_us) + floors
    return sum(kernel_time_ms(f, b, launch_us, spec) for f, b in kernels)


def classify_step(kernels: list[tuple[float, float]], launch_us: float = LAUNCH_US,
                  spec: dict = A100_80GB) -> str:
    """Return "launch", "memory", or "compute" for an eager step.

    Charge each kernel's time to what set it: the launch when the launch floor
    wins, otherwise the roofline's bound. The step takes the label with the
    most time.
    """
    launch_ms = launch_floor_ms(1, launch_us)
    totals = {"launch": 0.0, "memory": 0.0, "compute": 0.0}
    for flops, bytes_moved in kernels:
        r = roofline(flops, bytes_moved, spec)
        if launch_ms > r["floor_ms"]:
            totals["launch"] += launch_ms
        else:
            totals[r["bound_by"]] += r["floor_ms"]
    return max(totals, key=totals.get)
