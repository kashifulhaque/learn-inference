"""Lab 17c: observability, reliability, and cold start.

You write twelve small functions:

* `parse_events` and `request_metrics` turn an engine's event log into queue
  time, time to first token (TTFT), and inter-token latency (ITL).
* `build_histogram`, `histogram_quantile`, and `merge_histograms` build
  Prometheus-style cumulative histograms and read a quantile back out.
* `counter_rate` and `evaluate_alert` turn a counter and a gauge into alerts
  that do not fire on one sample and do not flap.
* `probe_states` and `min_failure_threshold` model liveness and readiness
  while the weights load.
* `weight_load_seconds`, `time_to_ready`, and `warm_pool_size` model a cold
  start.

Everything runs on the CPU, on a seeded synthetic log and seeded gauge series.
Times are seconds, sizes are decimal gigabytes (10^9 bytes), and rates are
GB/s. Nothing here is an A100 or network measurement.
"""

from __future__ import annotations

import bisect
import math
import random
import re

INF = math.inf

# Chapter 0: the weights are 53.8 GB in bfloat16. Chapter 19: PCIe 4.0 x16 is
# 32 GB/s in each direction.
WEIGHT_GB = 53.8
PCIE_GBS = 32.0

# Bucket upper bounds in seconds. These are this lab's choice, not vLLM's.
LAT_BOUNDS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

_LINE = re.compile(r"^(\d+(?:\.\d+)?) req=(\d+) (arrived|scheduled|token|preempted|finished)$")


# ---------------------------------------------------------------------------
# Given: a seeded event log, a flattening helper, an exact percentile, a gauge
# scenario, and a rate series.
# ---------------------------------------------------------------------------


def synth_event_log(seed: int = 31, n_requests: int = 400, rate: float = 8.0) -> list[str]:
    """A seeded engine event log as text lines, sorted by time.

    Each line is `<seconds> req=<id> <event>`. Events: `arrived`, `scheduled`
    (the first one ends the queue wait), `token` (the first one is the first
    token), `preempted`, and `finished`. About 5% of requests hit a congested
    queue and about 0.4% of tokens follow a preemption that stalls the request.
    Three malformed lines are mixed in, as a real log has.
    """
    rng = random.Random(seed)
    t = 0.0
    events: list[tuple[float, int, str]] = []
    for rid in range(n_requests):
        t += rng.expovariate(rate)
        queue = rng.expovariate(1 / 0.04)
        if rng.random() < 0.05:
            queue += rng.uniform(0.5, 2.0)
        sched = t + queue
        first = sched + 0.12 + rng.expovariate(1 / 0.06)
        n_out = rng.randint(16, 64)
        events += [(t, rid, "arrived"), (sched, rid, "scheduled"), (first, rid, "token")]
        cur = first
        for _ in range(1, n_out):
            gap = 0.025 + rng.expovariate(1 / 0.004)
            if rng.random() < 0.004:
                stall = rng.uniform(0.4, 1.2)
                events.append((cur + 0.0001, rid, "preempted"))
                events.append((cur + gap + stall - 0.03, rid, "scheduled"))
                gap += stall
            cur += gap
            events.append((cur, rid, "token"))
        events.append((cur, rid, "finished"))
    events.sort(key=lambda e: e[0])
    lines = [f"{ts:.6f} req={rid} {ev}" for ts, rid, ev in events]
    lines.insert(0, "# engine event log")
    lines.insert(5, "12.500000 req= scheduled")
    lines.insert(40, "garbage line from another process")
    return lines


def latency_samples(metrics: dict) -> dict[str, list[float]]:
    """Flatten per-request metrics into three sample lists, skipping None."""
    out: dict[str, list[float]] = {"queue": [], "ttft": [], "itl": []}
    for m in metrics.values():
        if m["queue"] is not None:
            out["queue"].append(m["queue"])
        if m["ttft"] is not None:
            out["ttft"].append(m["ttft"])
        out["itl"].extend(m["itl"])
    return out


def exact_quantile(samples: list[float], q: float) -> float:
    """The exact quantile from raw samples, as chapter 17 defines a percentile.

    Linear interpolation between order statistics at rank q * (n - 1), for q
    in [0, 1].
    """
    xs = sorted(samples)
    r = q * (len(xs) - 1)
    i = int(math.floor(r))
    if i + 1 >= len(xs):
        return xs[-1]
    return xs[i] + (r - i) * (xs[i + 1] - xs[i])


def synth_gauges(seed: int = 31) -> dict[str, list[tuple[float, float]]]:
    """Two seeded series sampled once a second for 600 s.

    `kv`: the KV-cache usage fraction. It sits near 0.62, then
    * t = 100: one sample at 1.0, a spike;
    * t = 200 to 319: a storm near 0.975, with a three-sample dip to 0.90 at
      t = 260 to 262;
    * t = 400 to 439: it alternates 0.96 and 0.90 every second.

    `preemptions`: a cumulative counter. About 0.05 a second normally, 2 to 5 a
    second during the storm, and a reset to 0 at t = 500 (an engine restart).
    """
    rng = random.Random(seed)
    kv, pre = [], []
    total = 0
    for t in range(600):
        base = 0.62 + 0.04 * math.sin(t / 30) + rng.uniform(-0.02, 0.02)
        storm = 200 <= t < 320
        if t == 100:
            v = 1.0
        elif storm:
            v = 0.90 if 260 <= t <= 262 else 0.975 + rng.uniform(-0.015, 0.015)
        elif 400 <= t < 440:
            v = 0.96 if t % 2 == 0 else 0.90
        else:
            v = base
        kv.append((float(t), v))
        if t == 500:
            total = 0
        total += rng.randint(2, 5) if storm else (1 if rng.random() < 0.05 else 0)
        pre.append((float(t), float(total)))
    return {"kv": kv, "preemptions": pre}


def rate_series(samples: list[tuple[float, float]], window_s: float) -> list[tuple[float, float]]:
    """`counter_rate` at every sample time where it is defined."""
    out = []
    for t, _ in samples:
        r = counter_rate(samples, t, window_s)
        if r is not None:
            out.append((t, r))
    return out


# ---------------------------------------------------------------------------
# Your code.
# ---------------------------------------------------------------------------


def parse_events(lines: list[str]) -> tuple[list[dict], int]:
    """Parse log lines into events sorted by time, and count the bad lines.

    A good line is `<seconds> req=<id> <event>`. Return `(events, n_bad)`.
    Each event is `{"t": float, "req": int, "event": str}`. Blank lines and
    lines that start with `#` are skipped and are not bad. Any other line that
    does not match is bad. The lines may arrive in any order, so sort by `t`
    (a stable sort, so equal times keep their input order).
    """
    raise NotImplementedError


def request_metrics(events: list[dict]) -> dict[int, dict]:
    """Per-request queue time, TTFT, ITL gaps, and preemption count.

    `events` are sorted by time. For each request:
    * `queue`: the first `scheduled` minus `arrived`, or None if either is missing;
    * `ttft`: the first `token` minus `arrived`, or None if either is missing;
    * `itl`: the gaps between consecutive `token` events (a stall from a
      preemption stays inside the gap it delays);
    * `preemptions`: the number of `preempted` events.
    """
    raise NotImplementedError


def build_histogram(samples: list[float], bounds: list[float]) -> dict:
    """A Prometheus-style cumulative histogram.

    `bounds` are finite, ascending upper bounds. Return
    `{"le": bounds + [inf], "cum": [...], "sum": float, "count": int}` where
    `cum[i]` counts the samples that are less than or equal to `le[i]`. The
    last bucket, `+Inf`, holds every sample, so `cum[-1] == count`.
    """
    raise NotImplementedError


def histogram_quantile(q: float, h: dict) -> float:
    """Estimate the q-quantile (q in [0, 1]) from a cumulative histogram.

    This is the rule Prometheus's `histogram_quantile` uses. Let
    `rank = q * count`. Find the first bucket whose cumulative count is at
    least `rank`, and interpolate linearly inside it, as if its samples were
    spread evenly:

        lower + (upper - lower) * (rank - cum_before) / (cum_here - cum_before)

    The first bucket's lower bound is 0. If the bucket found is `+Inf`, return
    the highest finite bound, because nothing says how far above it the
    samples go. An empty histogram gives nan, q < 0 gives -inf, and q > 1
    gives +inf. If the bucket has no samples of its own, return its lower bound.
    """
    raise NotImplementedError


def merge_histograms(a: dict, b: dict) -> dict:
    """Add two histograms with the same bounds, as Prometheus adds replicas.

    Raise ValueError if the `le` lists differ.
    """
    raise NotImplementedError


def counter_rate(samples: list[tuple[float, float]], t: float, window_s: float) -> float | None:
    """Per-second increase of a counter over the samples in [t - window_s, t].

    Use the samples whose time is in that closed window. With fewer than two,
    return None. Add up the increases between consecutive samples. A drop means
    the counter reset to 0 (a restart), so count the new value as that step's
    increase. Divide by the time between the first and last sample used. No
    extrapolation to the window edges.
    """
    raise NotImplementedError


def evaluate_alert(
    samples: list[tuple[float, float]], threshold: float, for_s: float, keep_firing_for_s: float
) -> list[tuple[float, float | None]]:
    """Run a Prometheus-style alert over `(t, value)` samples. Return the firing intervals.

    The condition is `value >= threshold`. Rules, like `for` and
    `keep_firing_for` in a Prometheus alerting rule:
    * The alert is pending from the first sample where the condition holds.
      Any sample where it does not hold, while pending, clears the pending state.
    * It starts firing at the first sample whose time is at least `for_s` after
      the pending start. With `for_s = 0` that is the first true sample.
    * Once firing, it keeps firing through samples where the condition is
      false until the condition has been false for `keep_firing_for_s`. It
      resolves at the first sample whose time is at least that long after the
      first false sample. A true sample resets that clock.
    Return `[(start, end), ...]`. `end` is None if still firing at the last sample.
    """
    raise NotImplementedError


def probe_states(t: float, image_s: float, load_s: float, warmup_s: float) -> dict:
    """What a liveness and a readiness probe report `t` seconds after the pod is scheduled.

    The image pull comes first and no container runs. Then the process starts,
    loads weights for `load_s`, then warms up for `warmup_s`. Return
    `{"phase": ..., "live": bool, "ready": bool}`, where `phase` is one of
    "pulling", "loading", "warmup", "ready". Liveness is true from the moment the
    process starts, because it only says the process answers. Readiness is
    true only when the weights are loaded and warmup is done. A phase ends at
    its end time, so `t` exactly at the end belongs to the next phase.
    """
    raise NotImplementedError


def min_failure_threshold(startup_s: float, period_s: float) -> int:
    """The least `failureThreshold` for a startup probe to cover `startup_s`.

    The startup window is `failureThreshold * periodSeconds`, so return the
    smallest integer whose window is at least `startup_s`. Allow 1e-9 of
    float slack so an exact multiple does not round up.
    """
    raise NotImplementedError


def weight_load_seconds(
    weights_gb: float,
    source_gbs: float,
    pcie_gbs: float = PCIE_GBS,
    loader: str = "mmap",
    chunk_gb: float = 0.5,
) -> float:
    """Seconds to get `weights_gb` from storage into GPU memory.

    Two stages: read from the source at `source_gbs`, then copy host to GPU at
    `pcie_gbs`.
    * "mmap": the read happens as the copy touches pages, one stage after the
      other, so the time is `weights/source + weights/pcie`.
    * "stream": a streaming loader reads chunk k+1 while it copies chunk k, in
      chunks of `chunk_gb`. With n = weights/chunk chunks, a two-stage pipeline
      takes `n * max(a, b) + min(a, b)` where `a = chunk/source` and
      `b = chunk/pcie`.
    Any other loader raises ValueError.
    """
    raise NotImplementedError


def time_to_ready(
    image_gb: float,
    registry_gbs: float,
    source_gbs: float,
    loader: str = "stream",
    n_graphs: int = 9,
    capture_s: float = 2.0,
    weights_gb: float = WEIGHT_GB,
    pcie_gbs: float = PCIE_GBS,
    chunk_gb: float = 0.5,
) -> dict:
    """Break a cold start into image pull, weight load, and warmup.

    * `image_pull_s`: `image_gb / registry_gbs`;
    * `weight_load_s`: `weight_load_seconds` with the same arguments;
    * `warmup_s`: `n_graphs * capture_s`, one capture per CUDA graph bucket;
    * `total_s`: the sum.
    """
    raise NotImplementedError


def warm_pool_size(arrival_rate_per_s: float, time_to_ready_s: float) -> int:
    """Replicas to keep ready so a burst does not wait for a cold start.

    While one new replica starts, `time_to_ready_s` passes, and new replicas
    are wanted at `arrival_rate_per_s`. By Little's law, that is
    `rate * time_to_ready` replicas wanted during one start. Return that,
    rounded up (allow 1e-9 of float slack).
    """
    raise NotImplementedError
