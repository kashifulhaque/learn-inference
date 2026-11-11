"""Numbers for chapter 17c.

Every number comes from content/labs/17c-observability/solution.py, run on its
seeded synthetic event log and gauge series. Nothing here is an A100, a
registry, or a disk measurement. The bandwidths and the graph-capture time in
the cold-start tables are inputs chosen for the chapter, not measured values.

Run from the repository root:

    python3 scripts/figures/17c-observability-and-cold-start.py
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOL = ROOT / "content" / "labs" / "17c-observability" / "solution.py"

LAYOUTS = {
    "coarse": [0.05, 0.1, 0.5, 1.0, 5.0],
    "lab": None,  # filled from LAT_BOUNDS
    "fine": [0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25,
             0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
}


def load():
    spec = importlib.util.spec_from_file_location("obs_solution", SOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    m = load()
    LAYOUTS["lab"] = m.LAT_BOUNDS
    events, bad = m.parse_events(m.synth_event_log())
    metrics = m.request_metrics(events)
    s = m.latency_samples(metrics)
    print(f"log: {len(events)} events, {bad} bad lines, {len(metrics)} requests")
    print(f"preemptions: {sum(r['preemptions'] for r in metrics.values())}")
    for name in ("queue", "ttft", "itl"):
        print(f"{name}: {len(s[name])} samples, min {min(s[name]):.5f}, max {max(s[name]):.5f}")

    print("\nquantile from buckets against exact (chapter 17's percentile)")
    print("series layout q estimate exact error")
    for name in ("queue", "ttft", "itl"):
        for layout, bounds in LAYOUTS.items():
            h = m.build_histogram(s[name], bounds)
            for q in (0.5, 0.99):
                est = m.histogram_quantile(q, h)
                ex = m.exact_quantile(s[name], q)
                print(f"{name:5s} {layout:6s} p{q * 100:g} {est:.5f} {ex:.5f} {est - ex:+.5f}")

    h = m.build_histogram(s["itl"], m.LAT_BOUNDS)
    print("\nitl cumulative buckets (lab bounds):", dict(zip(map(str, h["le"]), h["cum"])))
    h = m.build_histogram(s["ttft"], m.LAT_BOUNDS)
    print("ttft cumulative buckets (lab bounds):", dict(zip(map(str, h["le"]), h["cum"])))
    above = sum(1 for x in s["ttft"] if x > m.LAT_BOUNDS[-1])
    print("ttft samples above the top bound:", above)

    print("\nhand example: 100 samples at 0.22, bounds 0.1 0.2 0.3 0.5")
    hh = m.build_histogram([0.22] * 100, [0.1, 0.2, 0.3, 0.5])
    print("p95 estimate", m.histogram_quantile(0.95, hh), "exact 0.22")

    a = m.build_histogram([0.05] * 1000, m.LAT_BOUNDS)
    b = m.build_histogram([2.0] * 10, m.LAT_BOUNDS)
    avg = (m.histogram_quantile(0.99, a) + m.histogram_quantile(0.99, b)) / 2
    print("replica p99s", m.histogram_quantile(0.99, a), m.histogram_quantile(0.99, b),
          "average", avg, "merged", m.histogram_quantile(0.99, m.merge_histograms(a, b)))

    g = m.synth_gauges()
    print("\nKV alert (>= 0.95), for 30, keep 15:", m.evaluate_alert(g["kv"], 0.95, 30, 15))
    print("same, keep 0:", m.evaluate_alert(g["kv"], 0.95, 30, 0))
    print("same, for 0, keep 0: intervals", len(m.evaluate_alert(g["kv"], 0.95, 0, 0)))
    rates = m.rate_series(g["preemptions"], 30)
    print("preemption rate (>= 1/s over 30 s), for 30, keep 15:", m.evaluate_alert(rates, 1.0, 30, 15))
    print("peak rate:", max(r for _, r in rates))

    print("\ncold start, 53.8 GB over 32 GB/s PCIe, 0.5 GB chunks")
    print("source GB/s | read | copy | mmap | stream | saved")
    for src in (1.25, 2.0, 7.0, 25.0, 64.0):
        read, copy = 53.8 / src, 53.8 / 32.0
        mm = m.weight_load_seconds(53.8, src, 32.0, "mmap")
        st = m.weight_load_seconds(53.8, src, 32.0, "stream")
        print(f"{src:6.2f} | {read:.3f} | {copy:.3f} | {mm:.3f} | {st:.3f} | {mm - st:.3f} ({(mm - st) / mm:.1%})")
    for loader in ("mmap", "stream"):
        t = m.time_to_ready(8.0, 1.0, 2.0, loader, 9, 2.0)
        print(loader, {k: round(v, 6) for k, v in t.items()})
    t = m.time_to_ready(8.0, 1.0, 2.0, "stream", 9, 2.0)
    for rate in (0.05, 0.1, 0.5):
        print(f"warm pool at {rate}/s over {t['total_s']:.3f} s:", m.warm_pool_size(rate, t["total_s"]))
    print("failureThreshold at 10 s for load + warmup",
          m.min_failure_threshold(t["weight_load_s"] + t["warmup_s"], 10.0),
          "for the whole start", m.min_failure_threshold(t["total_s"], 10.0))
    print("PCIe copy alone:", 53.8 / 32.0, "s; read at 2 GB/s:", 53.8 / 2.0, "s")
    assert math.isclose(t["total_s"], 52.915625)


if __name__ == "__main__":
    main()
