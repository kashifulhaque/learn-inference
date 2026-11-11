"""Checks for aging priority, VTC fair sharing, and replica routing.

The first check calls only `token_budget_for_slo`. Every random draw uses
seed 18. The cluster is simulated: a request costs 2 ms plus 0.125 ms per
uncached prompt token to prefill, then 2.125 ms per output token (chapter 16's
simulated clock). The numbers are not an A100 measurement.

A plausible wrong implementation for VTC never lifts the counter of a client
that arrives with an empty queue. The newcomer's counter is 0 while the others
hold thousands, so the newcomer takes every slot until it catches up, and the
clients that were already waiting get nothing. The invariant check fails it.

Another wrong implementation ignores aging, so a steady stream of
high-priority requests starves a low-priority one forever. A third compares
`in_flight <= cap` instead of `< cap`, which lets a replica go over the
bounded-load cap.
"""

import math
import random
from collections import OrderedDict

from lab_common import Checks

BLOCK = 16
BASE_MS = 2.0
PREFILL_MS = 0.125
DECODE_MS = 2.125

# Cluster trace, seed 18.
REPLICAS = 4
REQUESTS = 600
RATE_PER_S = 40.0
GROUPS = 12
PREFIX_BLOCKS = 16
SUFFIX_BLOCKS = 4
OUT_TOKENS = 24
CACHE_BLOCKS = 96


def _ran(fn, expected):
    """Run fn and report expected and actual, including on an exception."""
    try:
        actual = fn()
    except Exception as exc:  # noqa: BLE001
        return False, f"expected {expected}, got {type(exc).__name__}: {exc}"
    return actual == expected, f"expected {expected}, got {actual}"


def percentile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


# --- priority with aging -------------------------------------------------

def serve_priority(submission, aging_ms, horizon=100):
    """One server, 1 ms per request. One priority-3 request arrives each ms.

    A priority-0 request arrives at 0. Return the time it starts, or None.
    """
    waiting = [{"id": "low", "priority": 0, "arrival": 0.0}]
    now = 0.0
    while now < horizon:
        waiting.append({"id": f"hi{int(now)}", "priority": 3, "arrival": now})
        index = submission.pick_next(waiting, now, aging_ms)
        item = waiting.pop(index)
        if item["id"] == "low":
            return now
        now += 1.0
    return None


# --- VTC trace -----------------------------------------------------------

def vtc_trace():
    """Requests for clients A (small), B (large), and C (late), seed 18."""
    rng = random.Random(18)
    reqs = {"A": [], "B": [], "C": []}
    for client, (lo, hi), n in (("A", (8, 48), 400), ("B", (128, 256), 200), ("C", (32, 96), 400)):
        for i in range(n):
            reqs[client].append((f"{client}{i}", rng.randint(lo, hi), rng.randint(16, 64)))
    return reqs


def run_vtc(submission, join_at=200, admissions=400):
    """A and B are backlogged from the start. C joins after `join_at` admissions.

    Return (max pairwise service gap after C joins, largest request cost,
    service by client since C joined).
    """
    reqs = vtc_trace()
    sched = submission.VTCScheduler()
    for client in ("A", "B"):
        for request in reqs[client]:
            sched.arrive(client, request)
    cost = lambda r: 1.0 * r[1] + 2.0 * r[2]  # noqa: E731
    biggest = max(cost(r) for client in reqs for r in reqs[client])
    served = {"A": 0.0, "B": 0.0, "C": 0.0}
    worst = 0.0
    for step in range(admissions):
        if step == join_at:
            for request in reqs["C"]:
                sched.arrive("C", request)
            served = {"A": 0.0, "B": 0.0, "C": 0.0}
        got = sched.next_request()
        if got is None:
            break
        client, request = got
        served[client] += cost(request)
        if step >= join_at:
            gap = max(served.values()) - min(served.values())
            worst = max(worst, gap)
    return worst, biggest, served


# --- cluster simulation --------------------------------------------------

class Req:
    def __init__(self, id, arrival, blocks, out_tokens):
        self.id = id
        self.arrival = arrival
        self.blocks = blocks
        self.out_tokens = out_tokens


def make_trace(seed=18, n=REQUESTS, rate=RATE_PER_S):
    """Poisson arrivals. Group k has weight 1/(k+1), so group 0 is hot."""
    rng = random.Random(seed)
    weights = [1.0 / (k + 1) for k in range(GROUPS)]
    t = 0.0
    trace = []
    for i in range(n):
        t += rng.expovariate(rate) * 1000.0
        group = rng.choices(range(GROUPS), weights)[0]
        blocks = [group * 1000 + j for j in range(PREFIX_BLOCKS)]
        blocks += [10_000_000 + i * 10 + j for j in range(SUFFIX_BLOCKS)]
        trace.append(Req(i, t, blocks, OUT_TOKENS))
    return trace


class Replica:
    """One FCFS server with an LRU cache of KV blocks, named by parent hash."""

    def __init__(self, id, capacity=CACHE_BLOCKS):
        self.id = id
        self.capacity = capacity
        self.cache = OrderedDict()
        self.free_at = 0.0
        self.done = []

    @staticmethod
    def chain(blocks):
        parent = 0
        for block in blocks:
            parent = hash((parent, block))
            yield parent

    def match(self, blocks):
        count = 0
        for key in self.chain(blocks):
            if key not in self.cache:
                break
            count += 1
        return count

    def in_flight(self, now):
        return sum(1 for t in self.done if t > now)

    def queued_ms(self, now):
        return max(0.0, self.free_at - now)

    def admit(self, request, now):
        """Serve `request`. Return (ttft_ms, cached_tokens)."""
        hit = self.match(request.blocks)
        keys = list(self.chain(request.blocks))
        # Touch the deepest block first and the root last. The root is then the
        # most recent, so the LRU end evicts leaves before their parents.
        for key in reversed(keys):
            self.cache[key] = True
            self.cache.move_to_end(key)
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        start = max(now, self.free_at)
        uncached = (len(request.blocks) - hit) * BLOCK
        prefill = BASE_MS + PREFILL_MS * uncached
        self.free_at = start + prefill + DECODE_MS * request.out_tokens
        self.done.append(self.free_at)
        return start - now + prefill, hit * BLOCK


def simulate(router, trace, n=REPLICAS, check_cap=None):
    """Route every request. Return hit rate, p99 TTFT, and the busiest share.

    `check_cap` is an epsilon. When set, record whether any routing put a
    request on a replica whose in-flight count was already at the bounded-load
    cap.
    """
    replicas = [Replica(i) for i in range(n)]
    ttfts, cached, prompt = [], 0, 0
    counts = [0] * n
    violations = 0
    for req in trace:
        now = req.arrival
        if check_cap is not None:
            total = sum(r.in_flight(now) for r in replicas)
            cap = math.ceil((1 + check_cap) * (total + 1) / n)
        choice = router.route(req, replicas, now)
        if check_cap is not None and replicas[choice].in_flight(now) >= cap:
            violations += 1
        ttft, hit_tokens = replicas[choice].admit(req, now)
        ttfts.append(ttft)
        cached += hit_tokens
        prompt += len(req.blocks) * BLOCK
        counts[choice] += 1
    return {
        "hit_rate": cached / prompt,
        "p99_ttft": percentile(ttfts, 99),
        "busiest_share": max(counts) / len(trace),
        "violations": violations,
    }


class FakeReplica:
    def __init__(self, id, queued, flight=0, match=0):
        self.id, self._q, self._f, self._m = id, queued, flight, match

    def queued_ms(self, now):
        return self._q

    def in_flight(self, now):
        return self._f

    def match(self, blocks):
        return self._m


def run(submission):
    c = Checks()
    if not c.require(
        submission, "token_budget_for_slo", "effective_priority", "pick_next", "VTCScheduler",
        "RoundRobin", "LeastLoaded", "PrefixAware", "cold_start_s", "replicas_needed",
    ):
        return c.finish()

    # 1. The smallest piece.
    c.check(
        "token_budget_for_slo inverts the step-time line",
        lambda: _ran(
            lambda: (
                submission.token_budget_for_slo(20.0),
                submission.token_budget_for_slo(10.0),
                submission.token_budget_for_slo(2.0),
                submission.token_budget_for_slo(1.0),
            ),
            (144, 64, 0, 0),
        ),
    )

    # 2. Aging arithmetic and ordering.
    def aging_unit():
        sp = submission.effective_priority
        got = (sp(1, 50.0, 10.0), sp(2, 50.0, None), sp(2, 50.0, 0))
        if got != (6.0, 2.0, 2.0):
            return False, f"expected effective priorities (6.0, 2.0, 2.0), got {got}"
        waiting = [
            {"id": "a", "priority": 0, "arrival": 0.0},
            {"id": "b", "priority": 2, "arrival": 90.0},
            {"id": "c", "priority": 2, "arrival": 94.0},
        ]
        order = (
            submission.pick_next(waiting, 100.0, None),
            submission.pick_next(waiting, 100.0, 10.0),
            submission.pick_next([], 0.0, 10.0),
        )
        if order != (1, 0, None):
            return False, f"expected picks (1, 0, None), got {order}"
        return starvation()

    # 3. Starvation, called at the end of the aging check.
    def starvation():
        no_aging = serve_priority(submission, None)
        aged = serve_priority(submission, 10.0)
        if no_aging is not None or aged != 30.0:
            return False, f"expected start None without aging and 30.0 with aging of 10 ms, got {no_aging} and {aged}"
        return True, f"expected start None and 30.0, got {no_aging} and {aged}"

    c.check("aging orders a queue and lets a priority-0 request through a priority-3 stream", aging_unit)

    # 4. VTC charge and choice.
    def vtc_basic():
        sched = submission.VTCScheduler()
        sched.arrive("A", ("a0", 10, 5))
        sched.arrive("B", ("b0", 100, 50))
        sched.arrive("A", ("a1", 10, 5))
        order = []
        for _ in range(3):
            client, request = sched.next_request()
            order.append((client, request[0]))
        counters = (sched.counter("A"), sched.counter("B"))
        want_order = [("A", "a0"), ("B", "b0"), ("A", "a1")]
        want_counters = (40.0, 200.0)
        ok = order == want_order and counters == want_counters
        return ok, f"expected {want_order} and counters {want_counters}, got {order} and {counters}"

    c.check("VTC admits the smallest counter and charges in + 2 * out", vtc_basic)

    # 5. The lift.
    def vtc_lift():
        sched = submission.VTCScheduler()
        for i in range(3):
            sched.arrive("A", (f"a{i}", 100, 0))
            sched.arrive("B", (f"b{i}", 100, 0))
        for _ in range(4):
            sched.next_request()  # A, B, A, B: both counters 200
        sched.next_request()  # A at 300, with a2 gone
        sched.arrive("C", ("c0", 10, 0))
        lifted = sched.counter("C")
        sched2 = submission.VTCScheduler()
        sched2.arrive("A", ("a0", 50, 25))
        sched2.next_request()
        sched2.arrive("B", ("b0", 10, 0))
        after_empty = sched2.counter("B")
        ok = lifted == 200.0 and after_empty == 100.0
        return ok, f"expected lifted counters 200.0 and 100.0, got {lifted} and {after_empty}"

    c.check("a newly backlogged client is lifted to the smallest backlogged counter", vtc_lift)

    # 6. The invariant on the seeded trace.
    def vtc_invariant():
        worst, biggest, served = run_vtc(submission)
        bound = 2 * biggest
        c.metric("max_vtc_gap", round(worst, 1))
        ok = worst <= bound and min(served.values()) > 0
        return ok, (
            f"expected gap <= {bound} (2 x the largest request cost) and every client served, "
            f"got gap {worst} and service {served}"
        )

    c.check("backlogged clients stay within 2U of each other, including a late joiner", vtc_invariant)

    # 7. Simple routers.
    def simple_routers():
        reps = [FakeReplica(0, 30.0), FakeReplica(1, 5.0), FakeReplica(2, 5.0), FakeReplica(3, 50.0)]
        rr = submission.RoundRobin()
        seq = [rr.route(None, reps, 0.0) for _ in range(6)]
        ll = submission.LeastLoaded().route(None, reps, 0.0)
        return _ran(lambda: (seq, ll), ([0, 1, 2, 3, 0, 1], 1))

    c.check("round-robin cycles and least-loaded picks the lowest queue", simple_routers)

    # 8. Routing on the seeded trace.
    trace = make_trace()
    results = {}

    def hit_rate():
        results["rr"] = simulate(submission.RoundRobin(), trace)
        results["ll"] = simulate(submission.LeastLoaded(), trace)
        results["aff"] = simulate(submission.PrefixAware(None), trace)
        results["bnd"] = simulate(submission.PrefixAware(0.25), trace)
        rr, bnd = results["rr"]["hit_rate"], results["bnd"]["hit_rate"]
        c.metric("hit_rate_rr", round(rr, 4))
        c.metric("hit_rate_bounded", round(bnd, 4))
        ok = bnd >= rr + 0.15
        return ok, f"expected prefix-aware hit rate >= round-robin {rr:.4f} + 0.15, got {bnd:.4f}"

    c.check("prefix-aware routing beats round-robin on prefix hit rate", hit_rate)

    def tail():
        if len(results) < 4:
            return False, "expected the routing runs from the previous check, got none"
        rr, ll, aff, bnd = (results[k]["p99_ttft"] for k in ("rr", "ll", "aff", "bnd"))
        c.metric("p99_ttft_bounded_ms", round(bnd, 1))
        ok = bnd < rr and bnd < ll and bnd < aff
        return ok, (
            f"expected bounded p99 below round-robin, least-loaded, and pure affinity, "
            f"got bounded {bnd:.1f}, rr {rr:.1f}, ll {ll:.1f}, affinity {aff:.1f} ms"
        )

    c.check("bounded-load prefix routing has the lowest p99 TTFT", tail)

    def cap_respected():
        out = simulate(submission.PrefixAware(0.25), trace, check_cap=0.25)
        share = results.get("aff", {}).get("busiest_share", 1.0)
        ok = out["violations"] == 0
        return ok, (
            f"expected 0 routings at or over the cap ceil(1.25 x (total + 1) / 4), "
            f"got {out['violations']} (pure affinity sends {share:.3f} of requests to its busiest replica)"
        )

    c.check("the bounded-load cap holds at every routing", cap_respected)

    # 9. Autoscaling arithmetic.
    def autoscale():
        weights = 53_791_959_040
        got = (
            round(submission.cold_start_s(weights, 2.0, 20.0), 3),
            round(submission.cold_start_s(weights, 12.5, 20.0), 3),
            submission.replicas_needed(40.0, 0.07, 0.7),
            submission.replicas_needed(0.0, 0.07, 0.7),
            submission.replicas_needed(100.0, 0.07, 0.7),
        )
        return _ran(lambda: got, (46.896, 24.303, 4, 1, 10))

    c.check("cold_start_s and replicas_needed follow the arithmetic", autoscale)
    return c.finish()
