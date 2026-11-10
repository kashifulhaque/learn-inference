"""Numbers for chapter 10b.

Every figure is computed from the hand-built traces in
content/labs/10b-profiling/fixture_traces.py and the lab solution. Nothing here
is a GPU measurement. The occupancy and bandwidth lines are arithmetic on the
A100 limits and on chapter 10's rated and measured bandwidth.

Run from the repository root:

    python3 scripts/figures/10b-profiling.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "content" / "labs" / "10b-profiling"
sys.path.insert(0, str(LAB))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def viz_data() -> dict:
    """Interval data for frontend/src/components/viz/performance/10b-trace-timeline.tsx."""
    sol = load("prof_solution", LAB / "solution.py")
    import fixture_traces as fx

    out = {}
    for name in ("launch_bound", "sync_bound", "multistream"):
        trace = getattr(fx, name)()
        p = sol.parse_trace(trace)
        t0 = min(e["ts"] for e in trace["traceEvents"]
                 if e.get("ph") == "X" and e.get("cat") in sol.GPU_CATS)
        spans = [[e["ts"] - t0, e["ts"] - t0 + e["dur"], e["tid"]]
                 for e in trace["traceEvents"]
                 if e.get("ph") == "X" and e.get("cat") in sol.GPU_CATS]
        out[name] = {
            "window": p["window_us"],
            "busy": round(p["gpu_busy_fraction"], 4),
            "class": sol.classify_trace(p),
            "spans": sorted(spans),
            "gaps": [[g["start_us"] - t0, g["end_us"] - t0, g["cause"]]
                     for g in p["idle_gaps"]],
        }
    return out


def main() -> None:
    if "--viz" in sys.argv:
        import json
        print(json.dumps(viz_data()))
        return
    sol = load("prof_solution", LAB / "solution.py")
    import fixture_traces as fx

    print("Trace summaries (microseconds)")
    for name in ("healthy", "launch_bound", "sync_bound", "multistream"):
        p = sol.parse_trace(getattr(fx, name)())
        print(f"  {name:13s} busy {p['gpu_busy_fraction']:.4f}  window {p['window_us']:.0f}"
              f"  gaps {len(p['idle_gaps'])}  idle {sum(g['dur_us'] for g in p['idle_gaps']):.0f}"
              f"  class {sol.classify_trace(p)}")
    s = sol.parse_trace(fx.sync_bound())["idle_gaps"][0]
    print(f"  sync gap: {s['dur_us']:.0f} us, cause {s['cause']}")

    print("\nLaunch floor (chapter 10 and 10a)")
    print(f"  400 kernels x {sol.LAUNCH_US:.0f} us = {400 * sol.LAUNCH_US / 1000:.3f} ms")

    print("\nCoalescing (32 threads, 32 B sectors)")
    for stride in (1, 2, 4, 8, 32):
        # Each thread reads 4 B at address 4*stride*t. Count distinct 32 B sectors.
        sectors = len({(4 * stride * t) // 32 for t in range(32)})
        print(f"  stride {stride:2d} floats: {sectors:2d} sectors per request,"
              f" {sectors * 32:4d} B moved for 128 B used")

    print("\nOccupancy on an A100 (64 warps, 65,536 registers per SM)")
    for regs in (32, 64, 128, 255):
        warps = min(64, 65536 // (regs * 32))
        print(f"  {regs:3d} registers per thread: {warps:2d} warps, {warps / 64:.1%} theoretical")

    print("\nBandwidth ceiling for a copy")
    print(f"  1275 / 1935 = {1275 / 1935:.4f}")

    print("\nBank conflicts: a 32x32 float tile read down a column")
    for pitch in (32, 33):
        banks = len({(r * pitch) % 32 for r in range(32)})
        print(f"  row pitch {pitch}: {banks} distinct banks, {32 // banks}-way conflict")

    print("\nRegression band")
    base = [10.0, 10.1, 9.9, 10.0, 10.2]
    out = [10.0, 10.1, 9.9, 10.0, 30.0]
    mean = sum(out) / 5
    print(f"  mean of outlier run {mean:.2f} vs baseline mean {sum(base) / 5:.2f}"
          f" ({mean / (sum(base) / 5) - 1:+.1%}); medians 10.00 vs 10.00")
    print(f"  gate: {sol.compare_to_baseline(base, out)['status']}")


if __name__ == "__main__":
    main()
