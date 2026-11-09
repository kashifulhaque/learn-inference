"""Lab 17a — a step-time model for capacity planning.

You predict a floor, not a measurement. There is no GPU and no weight file.

`cfg["kind"]` is "hybrid" or "dense". `hw` carries the peaks and the
tensor-parallel degree:

- `bf16_tflops`, `hbm_gbs`: the arithmetic peak and the bandwidth.
- `launches`, `launch_us`: the launch floor, a count times a per-launch time.
- `allreduce_us`, `link_gbs`: the collective latency and the link rate.
- `memory_bytes`: one GPU's memory.
- `tp`: the tensor-parallel degree. Missing means 1.

Times are seconds. A degree of 1 has no communication.
"""


def decode_step_time(cfg: dict, hw: dict, batch: int, ctx: int) -> float:
    """Seconds for one decode step of `batch` sequences at context `ctx`.

    max(compute, memory) plus communication plus the launch floor.
    """
    # TODO
    raise NotImplementedError


def prefill_time(cfg: dict, hw: dict, tokens: int) -> float:
    """Seconds to prefill one sequence of `tokens` tokens."""
    # TODO
    raise NotImplementedError


def tokens_per_dollar(tokens_per_s: float, usd_per_gpu_hour: float, gpus: int) -> float:
    """Output tokens per dollar. The price is the caller's argument."""
    # TODO
    raise NotImplementedError


def max_goodput(cfg: dict, hw: dict, slo_ttft: float, slo_tpot: float,
                workload: dict) -> float:
    """Largest decode throughput, in tokens per second, that meets both SLOs.

    `workload` has `prompt`, `ctx`, and `max_batch`. Return 0 when no batch
    meets the SLOs and fits in memory.
    """
    # TODO
    raise NotImplementedError


def fit_efficiency(rows: list[dict], hbm_gbs: float) -> dict:
    """Fit `alpha` and `overhead_s` on a memory-bound trace.

    Each row has `bytes`, `flops`, and `seconds`. The model is

        seconds = bytes / (alpha * bandwidth) + overhead

    Return `{"alpha": ..., "overhead_s": ...}`.
    """
    # TODO
    raise NotImplementedError


def choose_layout(cfg: dict, hw: dict, n_gpus: int, slo_ttft: float, slo_tpot: float,
                  workload: dict) -> dict:
    """Choose a tensor-parallel degree in {1, 2, 4} and the matching replica count.

    Return `{"tp", "replicas", "tokens_per_s"}` for the layout with the highest
    aggregate tokens per second. On an exact tie, prefer more replicas.
    """
    # TODO
    raise NotImplementedError
