"""Numbers for chapter 16b.

Every number comes from content/labs/16b-policy-and-routing/solution.py, run
on the seeded traces in that lab's tests.py. The clock is simulated. Nothing
here is an A100 measurement.

Run from the repository root:

    python3 scripts/figures/16b-scheduling-policy-and-routing.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "content" / "labs" / "16b-policy-and-routing"
WEIGHT_BYTES = 53_791_959_040  # chapter 2: the hybrid's bf16 weights


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    sys.path.insert(0, str(ROOT / "content" / "labs"))
    sol = load("sol16b", LAB / "solution.py")
    tests = load("tests16b", LAB / "tests.py")

    print("== SLO to token budget (step = 2 ms + 0.125 ms per token)")
    for slo in (10.0, 20.0, 40.0):
        print(f"slo {slo:5.1f} ms -> budget {sol.token_budget_for_slo(slo)} tokens")

    print("\n== aging: one priority-3 request per ms, one priority-0 request at 0")
    for aging in (None, 10.0, 20.0):
        print(f"aging {aging}: low starts at {tests.serve_priority(sol, aging)}")

    print("\n== VTC, seed 18, A and B backlogged, C joins at admission 200")
    worst, biggest, served = tests.run_vtc(sol)
    print(f"with lift: worst gap {worst}, largest request cost {biggest}, bound {2 * biggest}, service {served}")

    class NoLift(sol.VTCScheduler):
        def arrive(self, client, request):
            self.counters.setdefault(client, 0.0)
            self.queues.setdefault(client, []).append(request)

    class Shim:
        VTCScheduler = NoLift

    worst, _, served = tests.run_vtc(Shim)
    print(f"no lift:   worst gap {worst}, service {served}")

    print("\n== routing, seed 18, 4 replicas, 600 requests, 40 req/s, 96-block caches")
    trace = tests.make_trace()
    rows = [
        ("round-robin", sol.RoundRobin()),
        ("least-loaded", sol.LeastLoaded()),
        ("affinity", sol.PrefixAware(None)),
    ] + [(f"bounded {e}", sol.PrefixAware(e)) for e in (0.1, 0.25, 0.5, 1.0)]
    for name, router in rows:
        r = tests.simulate(router, trace)
        print(
            f"{name:13s} hit {r['hit_rate']:.4f}  p99 TTFT {r['p99_ttft']:7.2f} ms"
            f"  busiest share {r['busiest_share']:.4f}"
        )
    print("warm request: prefill 256 cached of 320 tokens ->",
          round(tests.BASE_MS + tests.PREFILL_MS * 64, 3), "ms; cold ->",
          round(tests.BASE_MS + tests.PREFILL_MS * 320, 3), "ms")
    print("decode of 24 tokens:", round(tests.DECODE_MS * 24, 3), "ms")

    print("\n== cold start (weights 53,791,959,040 bytes, fixed setup 20 s assumed)")
    for gbs in (1.0, 2.0, 5.0, 12.5):
        print(f"read {gbs:5.1f} GB/s -> {sol.cold_start_s(WEIGHT_BYTES, gbs, 20.0):.3f} s "
              f"(load part {WEIGHT_BYTES / (gbs * 1e9):.3f} s)")
    print("\n== replicas needed at service 0.07 s per request, target utilisation 0.7")
    for rate in (20.0, 40.0, 60.0, 100.0):
        print(f"{rate:5.1f} req/s -> {sol.replicas_needed(rate, 0.07, 0.7)} replicas")
    print("backlog built while a replica starts: 60 req/s against 4 replicas' capacity",
          f"{4 / 0.07:.1f} req/s ->", round((60 - 4 / 0.07) , 2), "req/s excess")


if __name__ == "__main__":
    main()
