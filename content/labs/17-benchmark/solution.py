import math


def ttft(record: dict) -> float:
    return record["first_token"] - record["arrival"]


def mean_inter_token_ms(record: dict) -> float | None:
    generated = record["generated_tokens"]
    if generated < 2:
        return None
    span = record["completion"] - record["first_token"]
    return span / (generated - 1) * 1000.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (q / 100.0) * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarise(records: list[dict]) -> dict:
    ttfts = [ttft(r) for r in records]
    itls = [v for r in records if (v := mean_inter_token_ms(r)) is not None]
    output_tokens = sum(r["generated_tokens"] for r in records)
    span = max(r["completion"] for r in records) - min(r["arrival"] for r in records)
    return {
        "requests": len(records),
        "total_output_tokens": output_tokens,
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p99": percentile(ttfts, 99),
        "itl_p50_ms": percentile(itls, 50) if itls else float("nan"),
        "itl_p99_ms": percentile(itls, 99) if itls else float("nan"),
        "throughput_tok_s": output_tokens / span if span > 0 else float("inf"),
    }


def goodput(records: list[dict], ttft_target_s: float,
            itl_target_ms: float) -> float:
    if not records:
        return 0.0
    met = 0
    for record in records:
        itl = mean_inter_token_ms(record)
        if ttft(record) <= ttft_target_s and (itl is None or itl <= itl_target_ms):
            met += 1
    return met / len(records)


def poisson_arrivals(rate_per_s: float, count: int, rng) -> list[float]:
    times, now = [], 0.0
    for _ in range(count):
        times.append(now)
        now += -math.log(1.0 - rng.random()) / rate_per_s
    return times
