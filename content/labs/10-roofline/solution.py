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
