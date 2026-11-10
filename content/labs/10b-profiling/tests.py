"""Checks for lab 10b: trace parsing, counter diagnosis, regression gate.

The traces are hand-built in fixture_traces.py, so every expected number is
exact arithmetic. Randomness is not used. The CPU profiler check runs
`torch.profiler` on a fixed loop, and its assertions use op counts, not times.

Plausible wrong implementations these checks catch:

* `parse_trace` that adds up kernel durations instead of merging intervals.
  Two streams that overlap are counted twice. Caught by "overlapping streams
  merge", where the sum gives 1.0 instead of 21/26.
* `parse_trace` that counts every `ph == "X"` event, or the
  `gpu_user_annotation` range. It reports a busy fraction near 1.0 on the
  launch-bound trace. Caught by "launch-bound trace".
* `diagnose` that tests DRAM throughput before sectors per request. A strided
  read can reach 78% of peak DRAM throughput while moving 8x the bytes it
  uses, and that version calls it memory_bandwidth. Caught by "diagnosis,
  planted kernels" (the strided case).
* `compare_to_baseline` that compares means. One outlier run flags a
  regression. Caught by "regression gate ignores one outlier".
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lab_common import Checks  # noqa: E402

import fixture_traces as fx  # noqa: E402

T0 = fx.T0


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def _ran(fn, expected):
    try:
        actual = fn()
    except Exception as exc:  # noqa: BLE001
        return False, f"expected {expected}, got {type(exc).__name__}: {exc}"
    return actual == expected, f"expected {expected}, got {actual}"


def _cases(sub):
    """Planted kernels. Each is (name, metrics, bound, fix keyword)."""
    return [
        ("copy-like", {sub.DRAM: 88.0, sub.SM: 12.0, sub.OCC: 71.0,
                       sub.SECTORS: 4000, sub.REQUESTS: 1000},
         "memory_bandwidth", "fuse"),
        ("gemm-like", {sub.DRAM: 30.0, sub.SM: 85.0, sub.OCC: 50.0,
                       sub.SECTORS: 4000, sub.REQUESTS: 1000},
         "compute", "tensor"),
        ("strided read, high DRAM", {sub.DRAM: 78.0, sub.SM: 15.0, sub.OCC: 60.0,
                                     sub.SECTORS: 32000, sub.REQUESTS: 1000},
         "uncoalesced_global", "coalesc"),
        ("transposed shared tile", {sub.DRAM: 20.0, sub.SM: 30.0, sub.OCC: 55.0,
                                    sub.CONFLICTS: 7000,
                                    sub.SHARED_WAVEFRONTS: 10000,
                                    sub.SECTORS: 4000, sub.REQUESTS: 1000},
         "shared_bank_conflict", "pad"),
        ("128 registers, spilling", {sub.DRAM: 35.0, sub.SM: 40.0, sub.OCC: 25.0,
                                     sub.SPILL_BYTES: 262144,
                                     sub.SECTORS: 4000, sub.REQUESTS: 1000},
         "register_spill", "register"),
        ("tiny grid", {sub.DRAM: 20.0, sub.SM: 15.0, sub.OCC: 12.0,
                       sub.SECTORS: 4000, sub.REQUESTS: 1000},
         "low_occupancy", "resident"),
        ("dependent loads", {sub.DRAM: 25.0, sub.SM: 30.0, sub.OCC: 68.0,
                             sub.SECTORS: 4000, sub.REQUESTS: 1000},
         "latency_bound", "independent"),
    ]


def run(sub):
    c = Checks()
    if not c.require(sub, "parse_trace", "classify_trace", "many_small_ops",
                     "diagnose", "compare_to_baseline"):
        return c.finish()

    # 1. Smallest piece: one back-to-back stream.
    def healthy():
        p = sub.parse_trace(fx.healthy())
        ok = (_close(p["gpu_busy_fraction"], 1.0) and p["idle_gaps"] == []
              and p["top_kernels"][0]["count"] == 10
              and _close(p["top_kernels"][0]["total_us"], 1000))
        return ok, (f"expected busy 1.0, no gaps, one kernel x10 totalling "
                    f"1000 us; got busy {p['gpu_busy_fraction']}, "
                    f"{len(p['idle_gaps'])} gaps, top {p['top_kernels']}")
    c.check("parse_trace on a back-to-back stream", healthy)

    # 2. Overlap and memcpy.
    def multi():
        p = sub.parse_trace(fx.to_json(fx.multistream()))
        g = p["idle_gaps"]
        ok = (_close(p["gpu_busy_fraction"], 210 / 260) and len(g) == 1
              and _close(g[0]["start_us"], T0 + 150)
              and _close(g[0]["dur_us"], 50)
              and [k["name"] for k in p["top_kernels"]] == ["kernel_A", "kernel_B"])
        return ok, (f"expected busy 21/26 = {210/260:.6f}, one gap of 50 us at "
                    f"{T0 + 150}, kernels A then B; got busy "
                    f"{p['gpu_busy_fraction']:.6f}, gaps {g}, "
                    f"top {[k['name'] for k in p['top_kernels']]}")
    c.check("overlapping streams merge, memcpy counts", multi)

    # 3. Launch-bound.
    def launch():
        p = sub.parse_trace(fx.launch_bound())
        gaps = sorted(g["dur_us"] for g in p["idle_gaps"])
        ok = (_close(p["gpu_busy_fraction"], 70 / 193)
              and gaps == [6] * 10 + [7] * 9 and p["kernel_count"] == 20
              and _close(p["median_kernel_us"], 3.5)
              and [(k["name"][-3:], k["total_us"], k["count"])
                   for k in p["top_kernels"]] == [("mul", 40, 10), ("add", 30, 10)]
              and all(g["cause"] == "idle" for g in p["idle_gaps"]))
        return ok, (f"expected busy 70/193 = {70/193:.6f}, 10 gaps of 6 us and "
                    f"9 of 7 us, kernels mul 40 us x10 then add 30 us x10; got "
                    f"busy {p['gpu_busy_fraction']:.6f}, gaps {gaps}, "
                    f"top {p['top_kernels']}")
    c.check("launch-bound trace", launch)
    try:
        c.metric("launch_bound_busy_fraction",
                 round(sub.parse_trace(fx.launch_bound())["gpu_busy_fraction"], 6))
    except Exception:  # noqa: BLE001
        pass

    # 4. Sync-bound.
    def sync():
        p = sub.parse_trace(fx.sync_bound())
        g = p["idle_gaps"]
        ok = (_close(p["gpu_busy_fraction"], 810 / 930) and len(g) == 1
              and _close(g[0]["start_us"], T0 + 410)
              and _close(g[0]["end_us"], T0 + 530)
              and _close(g[0]["dur_us"], 120) and g[0]["cause"] == "host_sync"
              and p["top_kernels"][0] == {"name": "ampere_bf16_s16816gemm_bf16_128x128_nn",
                                          "total_us": 800, "count": 16})
        return ok, (f"expected busy 810/930 = {810/930:.6f}, one 120 us gap "
                    f"from {T0 + 410} to {T0 + 530} caused by host_sync, GEMM "
                    f"800 us x16; got busy {p['gpu_busy_fraction']:.6f}, "
                    f"gaps {g}, top {p['top_kernels'][:1]}")
    c.check("sync-bound trace: gap and cause", sync)
    try:
        c.metric("sync_gap_us", sub.parse_trace(fx.sync_bound())["idle_gaps"][0]["dur_us"])
    except Exception:  # noqa: BLE001
        pass

    # 5. Classification.
    def cls():
        got = {n: sub.classify_trace(sub.parse_trace(getattr(fx, n)()))
               for n in ("healthy", "launch_bound", "sync_bound", "multistream")}
        want = {"healthy": "healthy", "launch_bound": "launch_bound",
                "sync_bound": "sync_bound", "multistream": "healthy"}
        return got == want, f"expected {want}, got {got}"
    c.check("classify_trace names the planted pattern", cls)

    # 6. min_gap_us and an empty GPU side.
    def edge():
        none = sub.parse_trace({"traceEvents": [
            {"ph": "X", "cat": "cpu_op", "name": "aten::mul", "ts": 1, "dur": 2,
             "pid": 1, "tid": 1}]})
        wide = sub.parse_trace(fx.launch_bound(), min_gap_us=6.5)
        ok = (none["gpu_busy_fraction"] == 0.0 and none["idle_gaps"] == []
              and none["top_kernels"] == []
              and len(wide["idle_gaps"]) == 9)
        return ok, (f"expected busy 0.0 with no GPU events, and 9 gaps of 7 us "
                    f"at min_gap_us=6.5; got {none['gpu_busy_fraction']}, "
                    f"{len(wide['idle_gaps'])} gaps")
    c.check("no GPU events, and the min_gap_us filter", edge)

    # 7. Diagnosis.
    def diag():
        bad = []
        for name, m, bound, kw in _cases(sub):
            d = sub.diagnose(m)
            if d.get("bound") != bound or kw not in str(d.get("fix", "")).lower():
                bad.append(f"{name}: expected bound {bound!r} with fix "
                           f"mentioning {kw!r}, got {d}")
        return (not bad), "; ".join(bad) or "7 planted kernels diagnosed"
    c.check("diagnosis, planted kernels", diag)

    # 8. Regression gate.
    def gate():
        base = [10.0, 10.1, 9.9, 10.0, 10.2]
        cases = [
            ("one 30 ms outlier is noise", base, [10.0, 10.1, 9.9, 10.0, 30.0], {}, "ok"),
            ("a steady 15% slowdown", base, [11.5, 11.6, 11.4, 11.5, 11.7], {}, "regression"),
            ("exactly 5% slower is inside the band", [100.0] * 3, [105.0] * 3, {}, "ok"),
            ("a 6% drop in tokens/s", [100.0] * 3, [94.0] * 3,
             {"higher_is_better": True}, "regression"),
            ("a 20% faster run", base, [8.0, 8.1, 7.9, 8.0, 8.0], {}, "improvement"),
            ("a 20% rise in tokens/s", [100.0] * 3, [120.0] * 3,
             {"higher_is_better": True}, "improvement"),
        ]
        bad = []
        for name, b, cur, kw, want in cases:
            got = sub.compare_to_baseline(b, cur, **kw)["status"]
            if got != want:
                bad.append(f"{name}: expected {want}, got {got}")
        return (not bad), "; ".join(bad) or "6 gate cases"
    c.check("regression gate ignores one outlier", gate)

    # 9. A real torch.profiler run on the CPU.
    def real():
        import json
        import tempfile

        import torch
        from torch.profiler import ProfilerActivity, profile

        torch.manual_seed(10)
        x = torch.randn(32, 32)
        w = torch.randn(32, 32)
        n = 60
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            for _ in range(n):
                x = x * 1.01 + 0.1
            x = x @ w
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.json")
            prof.export_chrome_trace(path)
            with open(path) as f:
                trace = json.load(f)
        small = sub.many_small_ops(trace, min_count=50)
        parsed = sub.parse_trace(trace)
        ok = (small.get("aten::mul", 0) >= n and small.get("aten::add", 0) >= n
              and "aten::mm" not in small
              and parsed["gpu_busy_fraction"] == 0.0)
        c.metric("cpu_profile_mul_count", small.get("aten::mul", 0))
        return ok, (f"expected aten::mul and aten::add each >= {n} and no "
                    f"aten::mm in {sorted(small)}, and busy 0.0 on a CPU-only "
                    f"trace; got {small}, busy {parsed['gpu_busy_fraction']}")
    c.check("torch.profiler on CPU shows the many-small-ops pattern", real)

    # 10. GPU run. Skips when no device is visible.
    import torch
    if not torch.cuda.is_available():
        c.add("torch.profiler on a GPU (skipped: no CUDA device)", True,
              "skipped; run on a device to check parse_trace on real kernels")
    else:
        def gpu():
            import json
            import tempfile

            from torch.profiler import ProfilerActivity, profile
            x = torch.randn(1024, 1024, device="cuda")
            for _ in range(3):
                x = x @ x
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU,
                                     ProfilerActivity.CUDA]) as prof:
                for _ in range(20):
                    x = (x @ x).tanh()
                torch.cuda.synchronize()
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "trace.json")
                prof.export_chrome_trace(path)
                with open(path) as f:
                    p = sub.parse_trace(json.load(f))
            ok = p["kernel_count"] >= 40 and 0.0 < p["gpu_busy_fraction"] <= 1.0
            return ok, (f"expected at least 40 kernels and a busy fraction in "
                        f"(0, 1]; got {p['kernel_count']} kernels, busy "
                        f"{p['gpu_busy_fraction']}")
        c.check("torch.profiler on a GPU", gpu)

    return c.finish()
