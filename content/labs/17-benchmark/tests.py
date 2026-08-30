import random
import statistics

from lab_common import Checks


def record(arrival, ttft_s, tokens, itl_s):
    first = arrival + ttft_s
    return {
        "arrival": arrival,
        "first_token": first,
        "completion": first + itl_s * (tokens - 1),
        "prompt_tokens": 100,
        "generated_tokens": tokens,
    }


def run(submission):
    c = Checks()
    needed = ("ttft", "mean_inter_token_ms", "percentile", "summarise", "goodput",
              "poisson_arrivals")
    if not c.require(submission, *needed):
        return c.finish()

    one = record(10.0, 0.25, 11, 0.03)
    c.check("ttft is arrival to first token",
            lambda: abs(submission.ttft(one) - 0.25) < 1e-9,
            f"got {submission.ttft(one)}")
    c.check("inter-token latency uses n - 1 gaps",
            lambda: abs(submission.mean_inter_token_ms(one) - 30.0) < 1e-6,
            f"got {submission.mean_inter_token_ms(one)}")
    c.check("a single-token response has no inter-token latency",
            lambda: submission.mean_inter_token_ms(record(0, 0.1, 1, 0)) is None)

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    c.check("the 50th percentile of five values is the middle one",
            lambda: abs(submission.percentile(values, 50) - 3.0) < 1e-9,
            f"got {submission.percentile(values, 50)}")
    c.check("the 0th percentile is the minimum",
            lambda: abs(submission.percentile(values, 0) - 1.0) < 1e-9)
    c.check("the 100th percentile is the maximum",
            lambda: abs(submission.percentile(values, 100) - 5.0) < 1e-9)
    c.check("percentiles interpolate between samples",
            lambda: abs(submission.percentile([0.0, 10.0], 25) - 2.5) < 1e-9,
            f"got {submission.percentile([0.0, 10.0], 25)}")
    c.check("an unsorted input is handled",
            lambda: abs(submission.percentile([5.0, 1.0, 3.0], 50) - 3.0) < 1e-9)

    # A run whose arithmetic is easy to verify by hand.
    records = [record(i * 1.0, 0.2, 11, 0.05) for i in range(10)]
    summary = submission.summarise(records)
    c.check("the summary counts every request",
            lambda: summary["requests"] == 10)
    c.check("output tokens are summed",
            lambda: summary["total_output_tokens"] == 110,
            f"got {summary['total_output_tokens']}")
    c.check("median TTFT is right",
            lambda: abs(summary["ttft_p50"] - 0.2) < 1e-6,
            f"got {summary['ttft_p50']}")
    c.check("median inter-token latency is right",
            lambda: abs(summary["itl_p50_ms"] - 50.0) < 1e-6,
            f"got {summary['itl_p50_ms']}")
    # Span runs from arrival 0 to the last completion: 9 + 0.2 + 0.5 = 9.7 s.
    c.check("throughput divides tokens by the wall-clock span",
            lambda: abs(summary["throughput_tok_s"] - 110 / 9.7) < 1e-3,
            f"got {summary['throughput_tok_s']:.2f}, expected {110 / 9.7:.2f}")

    # A tail: one slow request must move p99 but not p50.
    tailed = records + [record(10.0, 5.0, 11, 0.05)]
    tail_summary = submission.summarise(tailed)
    c.check("a slow request raises p99 TTFT",
            lambda: tail_summary["ttft_p99"] > 1.0,
            f"got {tail_summary['ttft_p99']:.2f}")
    c.check("a single slow request leaves p50 alone",
            lambda: abs(tail_summary["ttft_p50"] - 0.2) < 1e-6)

    c.check("goodput counts only requests that met both targets",
            lambda: abs(submission.goodput(tailed, 1.0, 100.0) - 10 / 11) < 1e-6,
            f"got {submission.goodput(tailed, 1.0, 100.0):.4f}")
    c.check("a strict inter-token target excludes everything",
            lambda: submission.goodput(tailed, 10.0, 1.0) == 0.0)
    c.check("loose targets admit everything",
            lambda: submission.goodput(tailed, 100.0, 10000.0) == 1.0)

    rng = random.Random(0)
    arrivals = submission.poisson_arrivals(20.0, 5000, rng)
    c.check("arrivals start at zero", lambda: abs(arrivals[0]) < 1e-9)
    c.check("arrivals are non-decreasing",
            lambda: all(b >= a for a, b in zip(arrivals, arrivals[1:])))
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    mean_gap = statistics.fmean(gaps)
    c.check("the mean gap matches 1 / rate",
            lambda: abs(mean_gap - 0.05) < 0.005,
            f"got {mean_gap:.4f}, expected 0.0500")
    # For an exponential distribution the standard deviation equals the mean.
    c.check("the gaps are exponential, not uniform",
            lambda: abs(statistics.stdev(gaps) / mean_gap - 1.0) < 0.1,
            f"stdev/mean = {statistics.stdev(gaps) / mean_gap:.3f}")

    c.metric("p99_ttft", round(tail_summary["ttft_p99"], 4))
    c.metric("throughput", round(summary["throughput_tok_s"], 2))
    c.metric("goodput_ratio", round(submission.goodput(tailed, 1.0, 100.0), 4))
    return c.finish()
