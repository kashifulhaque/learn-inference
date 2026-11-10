import importlib.util
import math
import random
import statistics
from pathlib import Path

from lab_common import Checks

NEW = ("closed_loop_latencies", "open_loop_latencies", "drive_engine", "sweep",
       "find_knee")


def record(arrival, ttft_s, tokens, itl_s):
    first = arrival + ttft_s
    return {
        "arrival": arrival,
        "first_token": first,
        "completion": first + itl_s * (tokens - 1),
        "prompt_tokens": 100,
        "generated_tokens": tokens,
    }


# --- Engine for the client checks -------------------------------------------
#
# Chapter 16a's EngineCore is the engine the client drives. When that lab's
# reference solution sits next to this one, the checks load it. Otherwise they
# use a stand-in with the same contract: add(request), step() returning one
# event per running request, and an integer clock that advances by
# forward_ticks per step. Both give the same numbers.


class _Request:
    def __init__(self, id, max_new_tokens):
        self.id = id
        self.max_new_tokens = max_new_tokens


class _StandInEngine:
    def __init__(self, forward_ticks):
        self.forward_ticks = forward_ticks
        self.clock = 0
        self._waiting, self._running = [], []

    def add(self, request):
        request.generated = 0
        self._waiting.append(request)

    def step(self):
        self._running.extend(self._waiting)
        self._waiting = []
        if not self._running:
            return []
        self.clock += self.forward_ticks
        events, still = [], []
        for req in self._running:
            req.generated += 1
            last = req.generated >= req.max_new_tokens
            events.append({"id": req.id, "text": "a",
                           "finish_reason": "length" if last else None})
            if not last:
                still.append(req)
        self._running = still
        return events


class _ConstantModel:
    def step(self, ids):
        return [[1.0] for _ in ids]


def _engine_kit(forward_ticks):
    """Return (engine, make_request(i, tokens), source)."""
    try:
        path = Path(__file__).resolve().parents[1] / "16a-serving-loop" / "solution.py"
        spec = importlib.util.spec_from_file_location("_lab16a_reference", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pool = module.BlockPool(64, 64)
        engine = module.EngineCore(_ConstantModel(), pool, ["a"],
                                   forward_ticks=forward_ticks)
        return engine, (lambda i, n: module.Request(f"r{i}", n)), "16a EngineCore"
    except Exception:  # noqa: BLE001
        engine = _StandInEngine(forward_ticks)
        return engine, (lambda i, n: _Request(f"r{i}", n)), "stand-in engine"


def _reference_arrivals(rate, count, seed):
    rng = random.Random(seed)
    times, now = [], 0.0
    for _ in range(count):
        times.append(now)
        now += -math.log(1.0 - rng.random()) / rate
    return times


def _planted(n, base, stall_index, stall_s):
    times = [base] * n
    times[stall_index] = stall_s
    return times


def _safe(fn):
    """Build a failure message without letting an unfinished stub raise."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def _safe_value(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def _pct(values, q):
    ordered = sorted(values)
    pos = q / 100.0 * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def run(submission):
    c = Checks()
    needed = ("ttft", "mean_inter_token_ms", "percentile", "summarise", "goodput",
              "poisson_arrivals") + NEW
    if not c.require(submission, *needed):
        return c.finish()

    one = record(10.0, 0.25, 11, 0.03)
    c.check("ttft is arrival to first token",
            lambda: abs(submission.ttft(one) - 0.25) < 1e-9,
            _safe(lambda: f"got {submission.ttft(one)}"))
    c.check("inter-token latency uses n - 1 gaps",
            lambda: abs(submission.mean_inter_token_ms(one) - 30.0) < 1e-6,
            _safe(lambda: f"got {submission.mean_inter_token_ms(one)}"))
    c.check("a single-token response has no inter-token latency",
            lambda: submission.mean_inter_token_ms(record(0, 0.1, 1, 0)) is None)

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    c.check("the 50th percentile of five values is the middle one",
            lambda: abs(submission.percentile(values, 50) - 3.0) < 1e-9,
            _safe(lambda: f"got {submission.percentile(values, 50)}"))
    c.check("the 0th percentile is the minimum",
            lambda: abs(submission.percentile(values, 0) - 1.0) < 1e-9)
    c.check("the 100th percentile is the maximum",
            lambda: abs(submission.percentile(values, 100) - 5.0) < 1e-9)
    c.check("percentiles interpolate between samples",
            lambda: abs(submission.percentile([0.0, 10.0], 25) - 2.5) < 1e-9,
            _safe(lambda: f"got {submission.percentile([0.0, 10.0], 25)}"))
    c.check("an unsorted input is handled",
            lambda: abs(submission.percentile([5.0, 1.0, 3.0], 50) - 3.0) < 1e-9)

    # A run whose arithmetic is easy to verify by hand.
    records = [record(i * 1.0, 0.2, 11, 0.05) for i in range(10)]
    summary = _safe_value(lambda: submission.summarise(records))
    c.check("the summary counts every request",
            lambda: summary["requests"] == 10)
    c.check("output tokens are summed",
            lambda: summary["total_output_tokens"] == 110,
            _safe(lambda: f"got {summary['total_output_tokens']}"))
    c.check("median TTFT is right",
            lambda: abs(summary["ttft_p50"] - 0.2) < 1e-6,
            _safe(lambda: f"got {summary['ttft_p50']}"))
    c.check("median inter-token latency is right",
            lambda: abs(summary["itl_p50_ms"] - 50.0) < 1e-6,
            _safe(lambda: f"got {summary['itl_p50_ms']}"))
    # Span runs from arrival 0 to the last completion: 9 + 0.2 + 0.5 = 9.7 s.
    c.check("throughput divides tokens by the wall-clock span",
            lambda: abs(summary["throughput_tok_s"] - 110 / 9.7) < 1e-3,
            _safe(lambda: f"got {summary['throughput_tok_s']:.2f}, expected {110 / 9.7:.2f}"))

    # A tail: one slow request must move p99 but not p50.
    tailed = records + [record(10.0, 5.0, 11, 0.05)]
    tail_summary = _safe_value(lambda: submission.summarise(tailed))
    c.check("a slow request raises p99 TTFT",
            lambda: tail_summary["ttft_p99"] > 1.0,
            _safe(lambda: f"got {tail_summary['ttft_p99']:.2f}"))
    c.check("a single slow request leaves p50 alone",
            lambda: abs(tail_summary["ttft_p50"] - 0.2) < 1e-6)

    c.check("goodput counts only requests that met both targets",
            lambda: abs(submission.goodput(tailed, 1.0, 100.0) - 10 / 11) < 1e-6,
            _safe(lambda: f"got {submission.goodput(tailed, 1.0, 100.0):.4f}"))
    c.check("a strict inter-token target excludes everything",
            lambda: submission.goodput(tailed, 10.0, 1.0) == 0.0)
    c.check("loose targets admit everything",
            lambda: submission.goodput(tailed, 100.0, 10000.0) == 1.0)

    rng = random.Random(0)
    arrivals = _safe_value(lambda: submission.poisson_arrivals(20.0, 5000, rng)) or [float("nan")] * 2
    c.check("arrivals start at zero", lambda: abs(arrivals[0]) < 1e-9)
    c.check("arrivals are non-decreasing",
            lambda: all(b >= a for a, b in zip(arrivals, arrivals[1:])))
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    mean_gap = statistics.fmean(gaps)
    c.check("the mean gap matches 1 / rate",
            lambda: abs(mean_gap - 0.05) < 0.005,
            _safe(lambda: f"got {mean_gap:.4f}, expected 0.0500"))
    # For an exponential distribution the standard deviation equals the mean.
    c.check("the gaps are exponential, not uniform",
            lambda: abs(statistics.stdev(gaps) / mean_gap - 1.0) < 0.1,
            _safe(lambda: f"stdev/mean = {statistics.stdev(gaps) / mean_gap:.3f}"))

    # --- Coordinated omission on a planted stall ---------------------------
    # 1000 requests, 5 ms each, except request 100 which stalls the server for
    # 5 s. The open-loop client sends every 10 ms.
    n, base, interval, stall_s, at = 1000, 0.005, 0.010, 5.0, 100
    planted = _planted(n, base, at, stall_s)
    closed = _safe_value(lambda: submission.closed_loop_latencies(planted))
    opened = _safe_value(lambda: submission.open_loop_latencies(planted, interval))

    # Request 100 is intended at 1.00 s and finishes at 6.00 s. Request 100 + k
    # is intended at 1.00 + 0.01k and finishes at 6.00 + 0.005k, so its latency
    # is 5 - 0.005k seconds, until the queue drains.
    expected_open = [base] * n
    for k in range(n - at):
        expected_open[at + k] = stall_s - 0.005 * k
    closed_p99 = _pct([base] * (n - 1) + [stall_s], 99)
    open_p99 = _pct(expected_open, 99)

    def check_closed():
        got = submission.percentile(closed, 99)
        return (abs(got - closed_p99) < 1e-9,
                f"expected closed-loop p99 {closed_p99:.6f} s, got {got:.6f} s")

    c.check("a closed-loop client records one slow sample, so its p99 is the base time",
            check_closed)

    # Plausible wrong implementation: the open-loop client measures latency from
    # the time it actually sent, which is when the previous reply came back.
    # That is the closed-loop number again, and it reintroduces coordinated
    # omission. The expected p99 is 4.95005 s, the wrong one is 0.005 s.
    def check_open():
        got = submission.percentile(opened, 99)
        return (abs(got - open_p99) < 1e-6,
                f"expected open-loop p99 {open_p99:.5f} s, got {got:.5f} s "
                "(measure from the intended send time)")

    c.check("an open-loop client measured from the intended send time sees the stall in p99",
            check_open)

    def check_each():
        bad = [i for i in range(n) if abs(opened[i] - expected_open[i]) > 1e-6]
        return (not bad,
                f"expected {len(expected_open)} latencies matching 5 - 0.005k after the stall; "
                f"first mismatch at index {bad[0] if bad else None}: expected "
                f"{expected_open[bad[0]] if bad else 0:.5f}, got "
                f"{opened[bad[0]] if bad else 0:.5f}")

    c.check("every request queued behind the stall is charged its wait",
            check_each)
    c.check("closed-loop p99 is lower than open-loop p99 on the same stall",
            lambda: (submission.percentile(closed, 99) < submission.percentile(opened, 99),
                     f"expected closed {closed_p99:.5f} < open {open_p99:.5f}, got closed "
                     f"{submission.percentile(closed, 99):.5f} and open "
                     f"{submission.percentile(opened, 99):.5f}"))
    quiet = [base] * n
    c.check("with no stall the two clients agree",
            lambda: (max(abs(a - b) for a, b in zip(
                submission.closed_loop_latencies(quiet),
                submission.open_loop_latencies(quiet, interval))) < 1e-9,
                "expected identical latencies of 0.005 s when the server keeps up"))

    # --- A client that drives the engine in process ------------------------
    # forward_ticks = 10 ticks of 1 ms. A: arrives 0.000 s, 3 tokens.
    # B: arrives 0.005 s, 2 tokens. Steps end at ticks 10, 20, 30.
    # A: first 0.010, done 0.030. B joins at the tick-10 step boundary and gets
    # its first token at 0.020: TTFT 0.015 s from its intended arrival.
    engine, make, source = _engine_kit(10)
    tokens_of = {0: 3, 1: 2}
    records = _safe_value(lambda: submission.drive_engine(
        engine, [0.0, 0.005], lambda i: make(i, tokens_of[i]), tick_s=0.001))

    def check_records():
        want = [(0.0, 0.010, 0.030, 3), (0.005, 0.020, 0.030, 2)]
        got = [(r["arrival"], r["first_token"], r["completion"], r["generated_tokens"])
               for r in records]
        ok = len(got) == 2 and all(
            abs(a - b) < 1e-9 for g, w in zip(got, want) for a, b in zip(g[:3], w[:3])
        ) and [g[3] for g in got] == [3, 2]
        return ok, f"expected (arrival, first, done, tokens) {want}, got {got} [{source}]"

    c.check("drive_engine records the intended arrival and the engine's clock", check_records)
    # Plausible wrong implementation: stamping the arrival when add() is called
    # (tick 10) turns B's 0.015 s TTFT into 0.010 s, hiding the queueing.
    c.check("TTFT counts the wait before the client could add the request",
            lambda: (abs(submission.ttft(records[1]) - 0.015) < 1e-9,
                     f"expected TTFT 0.015 s for B, got {submission.ttft(records[1]):.6f} s"))
    c.check("inter-token latency of an engine run is the step time",
            lambda: (abs(submission.mean_inter_token_ms(records[0]) - 10.0) < 1e-6,
                     f"expected 10.0 ms, got {submission.mean_inter_token_ms(records[0])}"))

    def idle_gap():
        eng, mk, _ = _engine_kit(10)
        recs = submission.drive_engine(eng, [0.0, 1.0], lambda i: mk(i, 2), tick_s=0.001)
        got = [(r["first_token"], r["completion"]) for r in recs]
        want = [(0.010, 0.020), (1.010, 1.020)]
        ok = all(abs(a - b) < 1e-9 for g, w in zip(got, want) for a, b in zip(g, w))
        return ok, f"expected (first, done) {want} after an idle gap, got {got}"

    c.check("the client skips ahead when the engine is idle", idle_gap)

    def cap():
        eng, mk, _ = _engine_kit(10)
        recs = submission.drive_engine(
            eng, [0.0] * 4, lambda i: mk(i, 2), tick_s=0.001, max_running=2)
        firsts = sorted(round(r["first_token"], 6) for r in recs)
        return (firsts == [0.010, 0.010, 0.030, 0.030],
                f"expected first-token times [0.01, 0.01, 0.03, 0.03] with 2 slots, got {firsts}")

    c.check("max_running holds extra requests in the client queue", cap)

    # --- Goodput versus rate, and the knee ---------------------------------
    c.check("find_knee stops at the first rate below the threshold",
            lambda: (submission.find_knee([1, 2, 3, 4, 5, 6],
                                          [1.0, 0.98, 0.6, 0.95, 0.5, 0.2], 0.9) == 2,
                     "expected knee 2 (the dip at 3 ends it, the recovery at 4 does not "
                     "count), got "
                     f"{submission.find_knee([1, 2, 3, 4, 5, 6], [1.0, 0.98, 0.6, 0.95, 0.5, 0.2], 0.9)}"))
    c.check("find_knee sorts the rates first",
            lambda: submission.find_knee([3, 1, 2], [0.5, 1.0, 0.95], 0.9) == 2,
            "expected knee 2 after sorting the rates")
    c.check("find_knee returns None when the lowest rate already misses",
            lambda: submission.find_knee([1, 2], [0.5, 0.4], 0.9) is None)
    c.check("a rate that exactly meets the threshold counts",
            lambda: submission.find_knee([1, 2], [1.0, 0.9], 0.9) == 2)

    # Each request needs 10 tokens at 10 ms a step, so it holds a slot for 100 ms.
    # Four slots serve 40 requests per second. The SLOs: TTFT 100 ms, ITL 15 ms.
    rates = [10, 20, 30, 40, 50, 60, 80]

    def run_at_rate(rate):
        eng, mk, _ = _engine_kit(10)
        arrivals = _reference_arrivals(rate, 300, seed=7)
        return submission.drive_engine(
            eng, arrivals, lambda i: mk(i, 10), tick_s=0.001, max_running=4)

    rows = _safe_value(lambda: submission.sweep(rates, run_at_rate, 0.100, 15.0)) or []
    goods = _safe_value(lambda: [r["goodput"] for r in rows]) or [float("nan")] * len(rates)
    knee = _safe_value(lambda: submission.find_knee(rates, goods, 0.9))

    def check_rows():
        keys = {"rate", "goodput", "goodput_req_s", "throughput_tok_s", "ttft_p99"}
        ok = bool(rows) and len(rows) == len(rates) and all(keys <= set(r) for r in rows)
        ok = ok and [r["rate"] for r in rows] == rates
        return ok, f"expected {len(rates)} rows with keys {sorted(keys)}, got {[sorted(r) for r in rows][:1]}"

    c.check("sweep returns one row per rate with the documented keys", check_rows)
    c.check("goodput in requests per second is the fraction times the rate",
            lambda: bool(rows) and all(abs(r["goodput_req_s"] - r["goodput"] * r["rate"]) < 1e-9 for r in rows),
            "expected goodput_req_s == goodput * rate in every row")
    c.check("a lightly loaded engine meets every SLO",
            lambda: (goods[0] >= 0.99,
                     f"expected goodput at 10 req/s of at least 0.99, got {goods[0]:.3f}"))
    c.check("an overloaded engine misses most SLOs",
            lambda: (goods[-1] < 0.5,
                     f"expected goodput at 80 req/s below 0.5, got {goods[-1]:.3f}"))
    c.check("p99 TTFT grows with the rate",
            lambda: (rows[-1]["ttft_p99"] > 10 * rows[0]["ttft_p99"],
                     f"expected p99 at 80 req/s above ten times the p99 at 10 req/s "
                     f"({rows[0]['ttft_p99']:.3f} s), got {rows[-1]['ttft_p99']:.3f} s"))
    c.check("the knee sits below the 40 req/s capacity of four slots",
            lambda: (knee is not None and 10 <= knee < 40,
                     f"expected a knee in [10, 40), got {knee}; goodput {[round(g, 3) for g in goods]}"))

    if opened:
        c.metric("open_loop_p99_s", round(submission.percentile(opened, 99), 5))
    c.metric("knee_rate", knee)
    if summary and tail_summary:
        c.metric("p99_ttft", round(tail_summary["ttft_p99"], 4))
        c.metric("throughput", round(summary["throughput_tok_s"], 2))
        c.metric("goodput_ratio", round(submission.goodput(tailed, 1.0, 100.0), 4))
    return c.finish()
