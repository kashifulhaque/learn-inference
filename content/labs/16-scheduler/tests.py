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
        batch = scheduler.step()
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


def run(submission):
    c = Checks()
    if not c.require(submission, "Scheduler", "BlockPool", "Request"):
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
    result = drive(submission, requests)
    scheduler, pool = result["scheduler"], result["pool"]

    c.check("no invariant was violated", lambda: result["violation"] is None,
            result["violation"] or "")
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
    scheduler.step()
    scheduler.commit_prefill(running, 32)
    scheduler.add(Request("newcomer", 512, 10))
    batch = scheduler.step()
    c.check("a running decode is scheduled ahead of a new prompt",
            lambda: running in batch.decode,
            f"decodes {[r.id for r in batch.decode]}, "
            f"prefills {[r.id for r, _ in batch.prefill]}")

    utilisation = result["tokens"] / (result["steps"] * 1024)
    c.metric("steps", result["steps"])
    c.metric("tokens", result["tokens"])
    c.metric("budget_utilisation", round(utilisation, 3))
    c.metric("preemptions", tight["scheduler"].preemptions)
    return c.finish()
