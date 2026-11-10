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


def _fifo_finish_times(send_times: list[float], service_times: list[float]) -> list[float]:
    """Finish times of a single-server FIFO queue: each request starts when it
    has been sent and the server is free."""
    free = 0.0
    finishes = []
    for sent, service in zip(send_times, service_times):
        free = max(sent, free) + service
        finishes.append(free)
    return finishes


def closed_loop_latencies(service_times: list[float]) -> list[float]:
    """A closed-loop client sends request i only after request i - 1 returns.

    Latency is measured from the actual send time.
    """
    now = 0.0
    latencies = []
    for service in service_times:
        finish = _fifo_finish_times([now], [service])[0]
        latencies.append(finish - now)
        now = finish
    return latencies


def open_loop_latencies(service_times: list[float], interval: float) -> list[float]:
    """An open-loop client sends request i at the fixed time i * interval,
    whether or not earlier requests have returned.

    Latency is measured from the intended send time, so the wait behind a stall
    counts against the requests that queued.
    """
    intended = [i * interval for i in range(len(service_times))]
    finishes = _fifo_finish_times(intended, service_times)
    return [done - sent for done, sent in zip(finishes, intended)]


def drive_engine(engine, arrivals: list[float], make_request, tick_s: float = 0.001,
                 max_running: int | None = None) -> list[dict]:
    """Drive an engine in process on a fixed trace and return one record per request.

    `engine` has `add(request)`, `step()` returning events shaped
    {"id", "text", "finish_reason"}, and an integer `clock` in ticks, as
    chapter 16a's EngineCore does. `arrivals` are sorted intended arrival times
    in seconds. `make_request(i)` builds request i, which has an `id`.
    The client holds at most `max_running` requests inside the engine and keeps
    the rest in its own queue. It assumes one event per generated token.

    Each record's `arrival` is the intended arrival time, not the tick at which
    the client got to call `add`.
    """
    count = len(arrivals)
    arrival_ticks = [a / tick_s for a in arrivals]
    queue: list[int] = []
    running: set[int] = set()
    index_of: dict = {}
    first: dict[int, float] = {}
    done: dict[int, float] = {}
    tokens = [0] * count
    nxt = 0
    while len(done) < count:
        while nxt < count and arrival_ticks[nxt] <= engine.clock:
            queue.append(nxt)
            nxt += 1
        while queue and (max_running is None or len(running) < max_running):
            i = queue.pop(0)
            request = make_request(i)
            index_of[request.id] = i
            engine.add(request)
            running.add(i)
        if not running:
            engine.clock = max(engine.clock, math.ceil(arrival_ticks[nxt]))
            continue
        events = engine.step()
        now = engine.clock * tick_s
        for event in events:
            i = index_of[event["id"]]
            tokens[i] += 1
            first.setdefault(i, now)
            if event.get("finish_reason"):
                done[i] = now
                running.discard(i)
    return [
        {
            "arrival": arrivals[i],
            "first_token": first[i],
            "completion": done[i],
            "prompt_tokens": 0,
            "generated_tokens": tokens[i],
        }
        for i in range(count)
    ]


def sweep(rates: list[float], run_at_rate, ttft_target_s: float,
          itl_target_ms: float) -> list[dict]:
    """Run the trace at each rate and report goodput against both targets.

    `run_at_rate(rate)` returns the records of one run. Each row has
    rate, goodput (a fraction), goodput_req_s, throughput_tok_s, ttft_p99.
    """
    rows = []
    for rate in rates:
        records = run_at_rate(rate)
        summary = summarise(records)
        fraction = goodput(records, ttft_target_s, itl_target_ms)
        rows.append({
            "rate": rate,
            "goodput": fraction,
            "goodput_req_s": fraction * rate,
            "throughput_tok_s": summary["throughput_tok_s"],
            "ttft_p99": summary["ttft_p99"],
        })
    return rows


def find_knee(rates: list[float], goodputs: list[float], threshold: float):
    """The largest rate at which goodput stays at or above `threshold`.

    Walk the rates from low to high and stop at the first one below the
    threshold. A later rate that recovers does not count. Return None when
    even the lowest rate misses.
    """
    knee = None
    for rate, value in sorted(zip(rates, goodputs)):
        if value < threshold:
            break
        knee = rate
    return knee
