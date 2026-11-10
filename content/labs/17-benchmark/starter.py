"""Lab 17 — benchmarks that mean something.

A record is a dict with these keys, all in seconds:

    arrival, first_token, completion, prompt_tokens, generated_tokens
"""

import math


def ttft(record: dict) -> float:
    """Time to first token: arrival to the first streamed token."""
    # TODO
    raise NotImplementedError


def mean_inter_token_ms(record: dict) -> float | None:
    """Mean gap between consecutive tokens, in milliseconds.

    A run of n tokens has n - 1 gaps, spanning first_token to completion.
    Return None when fewer than two tokens were generated.
    """
    # TODO
    raise NotImplementedError


def percentile(values: list[float], q: float) -> float:
    """Return the q-th percentile, with q in [0, 100], by linear interpolation."""
    # TODO
    raise NotImplementedError


def summarise(records: list[dict]) -> dict:
    """Aggregate a run.

    Returns a dict with:
        requests, total_output_tokens,
        ttft_p50, ttft_p99, itl_p50_ms, itl_p99_ms,
        throughput_tok_s: output tokens divided by the wall-clock span from the
            first arrival to the last completion.
    """
    # TODO
    raise NotImplementedError


def goodput(records: list[dict], ttft_target_s: float,
            itl_target_ms: float) -> float:
    """Fraction of requests that met both targets."""
    # TODO
    raise NotImplementedError


def poisson_arrivals(rate_per_s: float, count: int, rng) -> list[float]:
    """Return `count` arrival times from a Poisson process, starting at 0.

    Gaps are exponential with the given rate. `rng` is a random.Random.
    """
    # TODO
    raise NotImplementedError


def closed_loop_latencies(service_times: list[float]) -> list[float]:
    """A closed-loop client sends request i only after request i - 1 returns.

    The server is a single FIFO queue and service_times[i] is how long request i
    takes once it starts. Latency is measured from the actual send time.
    """
    # TODO
    raise NotImplementedError


def open_loop_latencies(service_times: list[float], interval: float) -> list[float]:
    """An open-loop client sends request i at the fixed time i * interval,
    whether or not earlier requests have returned.

    Measure latency from the intended send time, so the wait behind a stall
    counts against the requests that queued.
    """
    # TODO
    raise NotImplementedError


def drive_engine(engine, arrivals: list[float], make_request, tick_s: float = 0.001,
                 max_running: int | None = None) -> list[dict]:
    """Drive an engine in process on a fixed trace and return one record per request.

    `engine` has `add(request)`, `step()` returning events shaped
    {"id", "text", "finish_reason"}, and an integer `clock` in ticks, as
    chapter 16a's EngineCore does. `arrivals` are sorted intended arrival times
    in seconds. `make_request(i)` builds request i, which has an `id`.

    Loop: add every request whose arrival tick has passed (at most `max_running`
    inside the engine, the rest wait in your own queue), jump `engine.clock`
    forward to the next arrival when nothing is running, then call `step()`.
    A token's time is `engine.clock * tick_s` after the step that emitted it.
    Assume one event per generated token. Each record's `arrival` is the
    intended arrival time, not the tick at which you called `add`. The
    `prompt_tokens` field can be 0.
    """
    # TODO
    raise NotImplementedError


def sweep(rates: list[float], run_at_rate, ttft_target_s: float,
          itl_target_ms: float) -> list[dict]:
    """Run the trace at each rate and report goodput against both targets.

    `run_at_rate(rate)` returns the records of one run. Return one dict per rate
    with the keys rate, goodput (a fraction), goodput_req_s (the fraction times
    the rate), throughput_tok_s, and ttft_p99.
    """
    # TODO
    raise NotImplementedError


def find_knee(rates: list[float], goodputs: list[float], threshold: float):
    """The largest rate at which goodput stays at or above `threshold`.

    Walk the rates from low to high and stop at the first one below the
    threshold. A later rate that recovers does not count. Return None when
    even the lowest rate misses.
    """
    # TODO
    raise NotImplementedError
