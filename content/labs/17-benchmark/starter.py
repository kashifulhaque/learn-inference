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
