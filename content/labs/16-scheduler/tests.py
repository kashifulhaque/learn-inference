"""Checks lab 16 by draining randomized workloads and watching invariants."""

import random

from lab_common import Checks


def drive(submission, requests, num_blocks=400, max_batched_tokens=1024,
          chunk_size=256, max_batch_size=8, max_steps=20000):
    """Run the scheduler to completion, returning statistics and any violation."""
    pool = submission.BlockPool(num_blocks, block_size=16)
    scheduler = submission.Scheduler(
        pool, max_batch_size=max_batch_size,
        max_batched_tokens=max_batched_tokens, chunk_size=chunk_size,
    )
    for request in requests:
        scheduler.add(request)

    steps = 0
    tokens = 0
    violation = None
    while scheduler.has_work and steps < max_steps:
        try:
            batch = scheduler.step()
        except Exception as exc:  # noqa: BLE001
            violation = f"step raised {type(exc).__name__}: {exc}"
            break
        if batch.empty:
            # The final step only retires the last finished request, so an empty
            # batch is expected once nothing is left to do.
            if scheduler.has_work:
                violation = "step returned an empty batch while work remained"
            break
        if batch.token_count > max_batched_tokens:
            violation = (
                f"batch of {batch.token_count} tokens exceeds the "
                f"{max_batched_tokens} budget"
            )
            break
        if len(batch.decode) > max_batch_size:
            violation = f"{len(batch.decode)} decodes exceeds the batch size"
            break

        steps += 1
        tokens += batch.token_count
        for request, chunk in batch.prefill:
            scheduler.commit_prefill(request, chunk)
        for request in batch.decode:
            scheduler.commit_token(request)

    return {
        "steps": steps,
        "tokens": tokens,
        "scheduler": scheduler,
        "pool": pool,
        "violation": violation,
    }


# The simulated clock charges overhead + per_token * tokens milliseconds a step.
# Both constants are powers of two, so every time below is exact in floating
# point and the checks can compare with a tiny tolerance.
OVERHEAD = 2.0
PER_TOKEN = 0.125


def safe_replay(submission, requests, **config):
    """Replay a trace, returning (result, scheduler, error message)."""
    pool = submission.BlockPool(400, block_size=16)
    scheduler = submission.Scheduler(pool, max_batch_size=8, **config)
    try:
        result = submission.replay(scheduler, requests, OVERHEAD, PER_TOKEN)
    except Exception as exc:  # noqa: BLE001
        return None, scheduler, f"replay raised {type(exc).__name__}: {exc}"
    return result, scheduler, ""


def stall_trace(submission, chunk_size, max_batched_tokens):
    """Four decoders, plus one 960-token prompt that arrives at 10 ms."""
    Request = submission.Request
    requests = [Request(f"D{i}", 8, 40) for i in range(4)]
    requests.append(Request("L", 960, 4, arrival=10.0))
    return safe_replay(submission, requests, max_batched_tokens=max_batched_tokens,
                       chunk_size=chunk_size)


def priority_trace(submission, priority_of_c):
    """A and B wait from 0 ms. C arrives at 1 ms, with the given priority."""
    Request = submission.Request
    requests = [Request("A", 32, 2), Request("B", 32, 2),
                Request("C", 32, 2, arrival=1.0, priority=priority_of_c)]
    return safe_replay(submission, requests, max_batched_tokens=32, chunk_size=32)


def near(actual, expected, tol=1e-9):
    return actual is not None and abs(actual - expected) <= tol


def run(submission):
    c = Checks()
    if not c.require(submission, "Scheduler", "BlockPool", "Request",
                     "percentile", "replay"):
        return c.finish()

    Request = submission.Request

    # A single request must complete.
    result = drive(submission, [Request("solo", 100, 10)])
    c.check("a single request completes", lambda: result["violation"] is None
            and len(result["scheduler"].finished) == 1,
            result["violation"] or "")

    # Mixed lengths, comfortable memory.
    random.seed(0)
    requests = [
        Request(f"r{i}", random.choice([40, 300, 700]), random.choice([5, 20, 50]))
        for i in range(12)
    ]
    result_mixed = drive(submission, requests)
    scheduler, pool = result_mixed["scheduler"], result_mixed["pool"]

    c.check("no invariant was violated", lambda: result_mixed["violation"] is None,
            result_mixed["violation"] or "")
    c.check("every request finished",
            lambda: len(scheduler.finished) == len(requests),
            f"{len(scheduler.finished)} of {len(requests)}")
    c.check("every request generated what it asked for",
            lambda: all(r.generated == r.max_new_tokens for r in requests))
    c.check("every request was fully prefilled",
            lambda: all(r.prefilled == r.prompt_len for r in requests))
    c.check("all blocks were returned", lambda: pool.free == pool.num_blocks,
            f"{pool.free} of {pool.num_blocks} free")
    c.check("the queues are empty",
            lambda: not scheduler.waiting and not scheduler.running)

    # Chunked prefill: a prompt longer than chunk_size must span several steps.
    long_result = drive(submission, [Request("long", 2000, 5)], chunk_size=256)
    c.check("a long prompt is prefilled in chunks",
            lambda: long_result["steps"] >= 8,
            f"{long_result['steps']} steps for a 2000-token prompt")

    # Tight memory: preemption must still drain the queue.
    random.seed(7)
    tight_requests = [
        Request(f"t{i}", random.choice([500, 900]), random.choice([20, 60]))
        for i in range(10)
    ]
    tight = drive(submission, tight_requests, num_blocks=90, max_batch_size=4)
    c.check("a memory-constrained workload still drains",
            lambda: tight["violation"] is None
            and len(tight["scheduler"].finished) == len(tight_requests),
            tight["violation"] or
            f"{len(tight['scheduler'].finished)} of {len(tight_requests)}")
    c.check("blocks are conserved under preemption",
            lambda: tight["pool"].free == tight["pool"].num_blocks,
            f"{tight['pool'].free} of {tight['pool'].num_blocks}")

    # Decode should be prioritised: with a running decode and a waiting prompt,
    # the decode must appear in the batch.
    pool = submission.BlockPool(400)
    scheduler = submission.Scheduler(pool, max_batch_size=8,
                                     max_batched_tokens=64, chunk_size=64)
    running = Request("running", 32, 10)
    scheduler.add(running)
    try:
        scheduler.step()
        scheduler.commit_prefill(running, 32)
        scheduler.add(Request("newcomer", 512, 10))
        batch = scheduler.step()
    except Exception as exc:  # noqa: BLE001
        batch = submission.Batch()
        batch_error = f"step raised {type(exc).__name__}: {exc}"
    else:
        batch_error = ""
    c.check("a running decode is scheduled ahead of a new prompt",
            lambda: running in batch.decode,
            batch_error or f"decodes {[r.id for r in batch.decode]}, "
            f"prefills {[r.id for r, _ in batch.prefill]}")

    # Priority: a later, higher-priority request is admitted before an earlier,
    # lower-priority one when the token budget admits one prompt per step.
    pool = submission.BlockPool(400)
    scheduler = submission.Scheduler(pool, max_batch_size=8,
                                     max_batched_tokens=32, chunk_size=32)
    for request in (Request("lo1", 32, 5), Request("lo2", 32, 5),
                    Request("hi", 32, 5, priority=3)):
        scheduler.add(request)
    try:
        first = [r.id for r, _ in scheduler.step().prefill]
    except Exception as exc:  # noqa: BLE001
        first = f"step raised {type(exc).__name__}: {exc}"
    c.check("the highest-priority waiting request is admitted first",
            lambda: first == ["hi"], f"expected ['hi'], got {first}")

    result, _, error = priority_trace(submission, 5)
    c.check("a higher-priority arrival overtakes an earlier lower-priority one",
            lambda: result is not None and result["order"] == ["A", "C", "B"],
            error or f"expected ['A', 'C', 'B'], got {result['order']}")
    result, _, error = priority_trace(submission, 0)
    c.check("equal priorities stay first come, first served",
            lambda: result is not None and result["order"] == ["A", "B", "C"],
            error or f"expected ['A', 'B', 'C'], got {result['order']}")

    pool = submission.BlockPool(400)
    scheduler = submission.Scheduler(pool)
    old_high = Request("old_high", 16, 5, priority=5)
    old_low = Request("old_low", 16, 5)
    new_low = Request("new_low", 16, 5)
    new_high = Request("new_high", 16, 5, priority=5)
    scheduler.running = [old_high, old_low, new_low, new_high]
    victims = []
    try:
        for _ in range(3):
            scheduler.preempt()
            victims.append(scheduler.waiting[0].id)
    except Exception as exc:  # noqa: BLE001
        victims.append(f"preempt raised {type(exc).__name__}: {exc}")
    c.check("preemption evicts the lowest priority, then the newest",
            lambda: victims == ["new_low", "old_low", "new_high"],
            f"expected ['new_low', 'old_low', 'new_high'], got {victims}")

    # Percentiles use linear interpolation, as in chapter 17.
    def percentile_cases():
        eleven = [0.2] * 10 + [5.0]
        got = {
            "p25 of [0, 10]": (submission.percentile([0, 10], 25), 2.5),
            "p50 of eleven": (submission.percentile(eleven, 50), 0.2),
            "p99 of eleven": (submission.percentile(eleven, 99), 4.52),
            "p100 of eleven": (submission.percentile(eleven, 100), 5.0),
            "p99 of one value": (submission.percentile([7.0], 99), 7.0),
        }
        bad = [f"{name}: expected {want}, got {have}"
               for name, (have, want) in got.items() if not near(have, want)]
        return not bad, "; ".join(bad)
    c.check("percentile interpolates linearly between order statistics",
            percentile_cases)

    # Arrival-trace replay. The clock jumps over idle time, and TTFT counts from
    # arrival. Request B arrives at 100 ms, long after A has finished.
    idle, _, error = safe_replay(
        submission,
        [Request("A", 8, 2), Request("B", 8, 2, arrival=100.0)],
        max_batched_tokens=64, chunk_size=64)
    c.check("the clock jumps over idle time to the next arrival",
            lambda: idle is not None and near(idle["ttft"]["B"], 3.0)
            and near(idle["makespan"], 105.125),
            error or f"expected B TTFT 3.0 and makespan 105.125, got "
            f"{idle['ttft']['B']} and {idle['makespan']}")

    # PLAUSIBLE WRONG IMPLEMENTATION 1: measuring TTFT from admission instead of
    # arrival. Queueing then disappears from the number. In this trace C is
    # admitted at 6 ms and B at 12 ms, so the wrong clock reports 12.0 for C and
    # 8.25 for B instead of 17.0 and 20.25.
    #
    # PLAUSIBLE WRONG IMPLEMENTATION 2: dividing by n tokens instead of n - 1
    # gaps. A emits 2 tokens 6 ms apart, so the wrong formula reports 3.0 for A
    # and 1.0625 for B instead of 6.0 and 2.125.
    result, _, error = priority_trace(submission, 5)
    c.check("TTFT counts from arrival, so queueing is included",
            lambda: result is not None and near(result["ttft"]["A"], 6.0)
            and near(result["ttft"]["B"], 20.25) and near(result["ttft"]["C"], 17.0),
            error or f"expected TTFT A 6.0, B 20.25, C 17.0, got {result['ttft']}")
    c.check("ITL divides by n - 1 gaps, not by n tokens",
            lambda: result is not None and near(result["itl"]["A"], 6.0)
            and near(result["itl"]["B"], 2.125) and near(result["itl"]["C"], 2.25),
            error or f"expected ITL A 6.0, B 2.125, C 2.25, got {result['itl']}")

    # Chunked prefill against whole-prompt prefill, on a planted trace. Four
    # requests decode while a 960-token prompt arrives at 10 ms. Whole-prompt
    # prefill puts all 960 tokens in one step of 2 + 964/8 = 122.5 ms. A budget
    # of 64 tokens gives each step 4 decodes and a 60-token chunk, which costs
    # 2 + 64/8 = 10.0 ms. A decode-only step costs 2.5 ms.
    whole, whole_sched, error_w = stall_trace(submission, 2048, 2048)
    chunked, chunked_sched, error_c = stall_trace(submission, 60, 64)
    c.check("whole-prompt prefill stalls the decoders for the full prefill step",
            lambda: whole is not None and near(whole["max_gap"], 122.5)
            and near(whole["p99_itl"], 122.5)
            and whole_sched.preemptions == 0,
            error_w or f"expected max gap 122.5 and p99_itl 122.5, got "
            f"{whole['max_gap']} and {whole['p99_itl']}")
    c.check("chunked prefill bounds every decoder gap by one step",
            lambda: chunked is not None and near(chunked["max_gap"], 10.0)
            and near(chunked["p99_itl"], 10.0)
            and chunked_sched.preemptions == 0,
            error_c or f"expected max gap 10.0 and p99_itl 10.0, got "
            f"{chunked['max_gap']} and {chunked['p99_itl']}")
    c.check("chunking cuts the worst decode stall by more than 12x",
            lambda: whole is not None and chunked is not None
            and whole["max_gap"] / chunked["max_gap"] > 12,
            error_w or error_c or f"expected a ratio above 12, got "
            f"{whole['max_gap']} / {chunked['max_gap']}")
    c.check("chunking costs the long prompt TTFT: 123.5 ms against 161.0 ms",
            lambda: whole is not None and chunked is not None
            and near(whole["ttft"]["L"], 123.5)
            and near(chunked["ttft"]["L"], 161.0),
            error_w or error_c or f"expected 123.5 and 161.0, got "
            f"{whole['ttft']['L']} and {chunked['ttft']['L']}")
    c.check("p99 TTFT interpolates over per-request TTFTs",
            lambda: whole is not None and chunked is not None
            and near(whole["p99_ttft"], 118.8, 1e-6)
            and near(chunked["p99_ttft"], 154.8, 1e-6),
            error_w or error_c or f"expected 118.8 and 154.8, got "
            f"{whole['p99_ttft']} and {chunked['p99_ttft']}")
    c.check("both policies finish the trace at the same time, 223.875 ms",
            lambda: whole is not None and chunked is not None
            and near(whole["makespan"], 223.875)
            and near(chunked["makespan"], 223.875),
            error_w or error_c or f"expected 223.875 for both, got "
            f"{whole['makespan']} and {chunked['makespan']}")

    utilisation = result_mixed["tokens"] / max(1, result_mixed["steps"] * 1024)
    c.metric("steps", result_mixed["steps"])
    c.metric("tokens", result_mixed["tokens"])
    c.metric("budget_utilisation", round(utilisation, 3))
    c.metric("preemptions", tight["scheduler"].preemptions)
    if chunked is not None:
        c.metric("p99_ttft", round(chunked["p99_ttft"], 3))
        c.metric("p99_itl", round(chunked["p99_itl"], 3))
    return c.finish()
