"""Measurement helpers.

Timing GPU code with time.time() gives you the time it took to *queue* the
work, not to do it. CUDA launches are asynchronous. Every function here
synchronises before it stops the clock, and every one runs warmup iterations
first so you are not timing kernel compilation and autotuning.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed(label: str, out: dict[str, float] | None = None):
    sync()
    start = time.perf_counter()
    yield
    sync()
    elapsed = time.perf_counter() - start
    if out is not None:
        out[label] = elapsed
    else:
        print(f"{label}: {elapsed * 1000:.2f} ms")


@dataclass
class Timing:
    label: str
    runs: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    min_ms: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark(
    fn: Callable[[], Any],
    label: str = "fn",
    warmup: int = 5,
    runs: int = 20,
) -> Timing:
    for _ in range(warmup):
        fn()
    sync()

    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1000)

    samples.sort()
    return Timing(
        label=label,
        runs=runs,
        mean_ms=round(statistics.fmean(samples), 4),
        median_ms=round(statistics.median(samples), 4),
        p90_ms=round(samples[int(0.9 * (runs - 1))], 4),
        min_ms=round(samples[0], 4),
    )


def gpu_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "memory_gb": round(props.total_memory / 1024**3, 1),
        "sms": props.multi_processor_count,
    }


# Published A100 numbers, used for roofline work in chapters 8 and 9.
A100_80GB = {
    "hbm_bandwidth_gbs": 2039.0,
    "bf16_tflops": 312.0,
    "fp32_tflops": 19.5,
    "tf32_tflops": 156.0,
    "l2_cache_mb": 40,
    "sms": 108,
}
A100_40GB = {**A100_80GB, "hbm_bandwidth_gbs": 1555.0, "l2_cache_mb": 40}


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOPs per byte. Compare against peak_flops / peak_bandwidth."""
    return flops / max(bytes_moved, 1.0)


def roofline_ms(flops: float, bytes_moved: float, spec: dict[str, float] | None = None) -> dict[str, float]:
    """Lower bound on runtime from compute and from bandwidth, whichever binds."""
    spec = spec or A100_80GB
    compute_ms = flops / (spec["bf16_tflops"] * 1e12) * 1e3
    memory_ms = bytes_moved / (spec["hbm_bandwidth_gbs"] * 1e9) * 1e3
    return {
        "compute_ms": compute_ms,
        "memory_ms": memory_ms,
        "bound_by": "compute" if compute_ms > memory_ms else "memory",
        "floor_ms": max(compute_ms, memory_ms),
        "intensity": arithmetic_intensity(flops, bytes_moved),
        "ridge_point": spec["bf16_tflops"] * 1e12 / (spec["hbm_bandwidth_gbs"] * 1e9),
    }


def memory_report() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }
