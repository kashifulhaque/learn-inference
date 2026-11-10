"""Lab 10b solution: parse a Chrome trace, diagnose from counters, gate a regression."""

from __future__ import annotations

import json
import statistics

LAUNCH_US = 5.0  # chapter 10's launch cost
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
SYNC_NAMES = ("cudaStreamSynchronize", "cudaDeviceSynchronize",
              "cudaEventSynchronize", "aten::item", "aten::_local_scalar_dense")
SYNC_SLACK_US = 10.0

# Nsight Compute metric names.
OCC = "sm__warps_active.avg.pct_of_peak_sustained_active"
DRAM = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
SM = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
CONFLICTS = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"
SHARED_WAVEFRONTS = "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"
SPILL_BYTES = "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum"
SECTORS = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
REQUESTS = "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"

SPILL_MIN_BYTES = 4096
CONFLICT_RATIO = 0.25
SECTORS_PER_REQUEST_MAX = 8.0
HIGH_PCT = 60.0
LOW_OCC_PCT = 30.0


def _merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def parse_trace(trace, top_n: int = 3, min_gap_us: float = 1.0) -> dict:
    """Summarize GPU activity in a Chrome trace. Times are microseconds."""
    if isinstance(trace, (str, bytes)):
        trace = json.loads(trace)
    events = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X"]
    gpu = [e for e in events if e.get("cat") in GPU_CATS]
    syncs = [(e["ts"], e["ts"] + e["dur"]) for e in events
             if e.get("name") in SYNC_NAMES]
    kernels = [e for e in gpu if e["cat"] == "kernel"]

    totals: dict[str, list[float]] = {}
    for e in kernels:
        t = totals.setdefault(e["name"], [0.0, 0])
        t[0] += e["dur"]
        t[1] += 1
    top = sorted(totals.items(), key=lambda kv: (-kv[1][0], kv[0]))[:top_n]
    top_kernels = [{"name": n, "total_us": v[0], "count": v[1]} for n, v in top]
    median = statistics.median(e["dur"] for e in kernels) if kernels else 0.0

    if not gpu:
        return {"gpu_busy_fraction": 0.0, "top_kernels": [], "idle_gaps": [],
                "window_us": 0.0, "kernel_count": 0, "median_kernel_us": 0.0}

    merged = _merge([(e["ts"], e["ts"] + e["dur"]) for e in gpu])
    start, end = merged[0][0], merged[-1][1]
    busy = sum(e - s for s, e in merged)
    gaps = []
    for (_, prev_end), (next_start, _) in zip(merged, merged[1:]):
        dur = next_start - prev_end
        if dur < min_gap_us:
            continue
        hit = any(prev_end - SYNC_SLACK_US <= se <= next_start for _, se in syncs)
        gaps.append({"start_us": prev_end, "end_us": next_start, "dur_us": dur,
                     "cause": "host_sync" if hit else "idle"})
    return {"gpu_busy_fraction": busy / (end - start), "top_kernels": top_kernels,
            "idle_gaps": gaps, "window_us": end - start,
            "kernel_count": len(kernels), "median_kernel_us": median}


def classify_trace(parsed: dict) -> str:
    """Return "sync_bound", "launch_bound", or "healthy"."""
    if any(g["cause"] == "host_sync" for g in parsed["idle_gaps"]):
        return "sync_bound"
    if (parsed["kernel_count"] >= 10 and parsed["gpu_busy_fraction"] < 0.5
            and parsed["median_kernel_us"] < 2 * LAUNCH_US):
        return "launch_bound"
    return "healthy"


def many_small_ops(trace, min_count: int = 50) -> dict:
    """Names of CPU ops that run at least `min_count` times, with their counts."""
    if isinstance(trace, (str, bytes)):
        trace = json.loads(trace)
    counts: dict[str, int] = {}
    for e in trace.get("traceEvents", []):
        if e.get("ph") == "X" and e.get("cat") == "cpu_op":
            counts[e["name"]] = counts.get(e["name"], 0) + 1
    return {n: c for n, c in sorted(counts.items()) if c >= min_count}


def diagnose(m: dict) -> dict:
    """Map Nsight Compute metrics to {"bound": ..., "fix": ...}."""
    if m.get(SPILL_BYTES, 0) > SPILL_MIN_BYTES:
        return {"bound": "register_spill",
                "fix": "Cut register use: smaller tile, __launch_bounds__, or a lower maxrregcount."}
    wave = m.get(SHARED_WAVEFRONTS, 0)
    if wave and m.get(CONFLICTS, 0) / wave >= CONFLICT_RATIO:
        return {"bound": "shared_bank_conflict",
                "fix": "Pad the shared tile by one column so a column read hits 32 different banks."}
    req = m.get(REQUESTS, 0)
    if req and m.get(SECTORS, 0) / req > SECTORS_PER_REQUEST_MAX:
        return {"bound": "uncoalesced_global",
                "fix": "Coalesce: make adjacent threads read adjacent addresses, or stage a transpose through shared memory."}
    dram, sm = m.get(DRAM, 0.0), m.get(SM, 0.0)
    if dram >= HIGH_PCT and dram > sm:
        return {"bound": "memory_bandwidth",
                "fix": "Move fewer bytes: fuse the neighbouring ops, or use a narrower dtype."}
    if sm >= HIGH_PCT and sm >= dram:
        return {"bound": "compute",
                "fix": "Use tensor cores and a narrower dtype, or cut FLOPs."}
    if m.get(OCC, 100.0) < LOW_OCC_PCT:
        return {"bound": "low_occupancy",
                "fix": "Launch more resident warps: lower registers or shared memory per block, or a bigger grid."}
    return {"bound": "latency_bound",
            "fix": "Add independent work per thread (ILP), more bytes in flight, or software pipelining."}


def compare_to_baseline(baseline, current, rel_tol: float = 0.05,
                        higher_is_better: bool = False) -> dict:
    """Compare repeated samples by median. A change within `rel_tol` is noise."""
    b, c = statistics.median(baseline), statistics.median(current)
    change = (c - b) / b
    worse = -change if higher_is_better else change
    if worse > rel_tol:
        status = "regression"
    elif worse < -rel_tol:
        status = "improvement"
    else:
        status = "ok"
    return {"status": status, "baseline": b, "current": c, "change": change}
