"""Lab 10b: read a trace, diagnose a kernel from counters, gate a regression.

Times in a Chrome trace are microseconds. A trace has `traceEvents`. Complete
events have `ph == "X"`, `ts`, `dur`, `cat`, `name`.
GPU work is `cat` in ("kernel", "gpu_memcpy", "gpu_memset"). Ignore
`gpu_user_annotation`, `cpu_op`, `cuda_runtime`, and non-"X" events when you
measure GPU work.
"""

from __future__ import annotations

LAUNCH_US = 5.0  # chapter 10's launch cost

# Nsight Compute metric names the diagnosis tests pass in.
OCC = "sm__warps_active.avg.pct_of_peak_sustained_active"
DRAM = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
SM = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
CONFLICTS = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"
SHARED_WAVEFRONTS = "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"
SPILL_BYTES = "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum"
SECTORS = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
REQUESTS = "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"


def parse_trace(trace, top_n: int = 3, min_gap_us: float = 1.0) -> dict:
    """Summarize GPU activity. `trace` is a dict or a JSON string.

    Return a dict with:
    - gpu_busy_fraction: time covered by at least one GPU event, divided by the
      window from the first GPU start to the last GPU end. Overlapping events
      on different streams count once. 0.0 when there is no GPU event.
    - top_kernels: up to top_n dicts {"name", "total_us", "count"} for events
      with cat "kernel", sorted by total_us descending, then name.
    - idle_gaps: dicts {"start_us", "end_us", "dur_us", "cause"} for each
      stretch with no GPU work that is at least min_gap_us long, in time order.
      `cause` is "host_sync" when a sync event (cudaStreamSynchronize,
      cudaDeviceSynchronize, cudaEventSynchronize, aten::item,
      aten::_local_scalar_dense) ends from 10 µs before the gap starts up to
      the gap's end. Otherwise "idle".
    - window_us, kernel_count, median_kernel_us.
    """
    raise NotImplementedError


def classify_trace(parsed: dict) -> str:
    """"sync_bound" if any gap has cause "host_sync". Else "launch_bound" if
    there are at least 10 kernels, busy fraction is under 0.5, and the median
    kernel is under 2 * LAUNCH_US. Else "healthy"."""
    raise NotImplementedError


def many_small_ops(trace, min_count: int = 50) -> dict:
    """{name: count} for cat "cpu_op" complete events that occur at least
    min_count times, sorted by name."""
    raise NotImplementedError


def diagnose(m: dict) -> dict:
    """Return {"bound": ..., "fix": ...} from a metric dict.

    Check in this order and return the first that matches:
    1. register_spill: SPILL_BYTES > 4096.
    2. shared_bank_conflict: CONFLICTS / SHARED_WAVEFRONTS >= 0.25.
    3. uncoalesced_global: SECTORS / REQUESTS > 8.
    4. memory_bandwidth: DRAM >= 60 and DRAM > SM.
    5. compute: SM >= 60 and SM >= DRAM.
    6. low_occupancy: OCC < 30.
    7. latency_bound: anything else.
    Write a one-sentence `fix` for each. Missing metrics count as 0 (OCC as 100).
    """
    raise NotImplementedError


def compare_to_baseline(baseline, current, rel_tol: float = 0.05,
                        higher_is_better: bool = False) -> dict:
    """Compare two lists of repeated measurements by their medians.

    Return {"status", "baseline", "current", "change"} where change is
    (median(current) - median(baseline)) / median(baseline). status is
    "regression" when the result is worse by more than rel_tol, "improvement"
    when better by more than rel_tol, else "ok". Lower is worse only when
    higher_is_better is True.
    """
    raise NotImplementedError
