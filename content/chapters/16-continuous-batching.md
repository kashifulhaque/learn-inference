---
title: Continuous batching
slug: 16-continuous-batching
part: "Part 5 — Serving"
summary: Scheduling per iteration instead of per batch, with chunked prefill and preemption.
minutes: 70
gpu: false
objectives:
  - Explain why static batching wastes most of a GPU under realistic traffic.
  - Implement a scheduler with waiting and running queues under a token budget.
  - Describe what chunked prefill fixes and what it costs.
lab: 16-scheduler
---

# Continuous batching

The kernels are fast. Now decide what to run. Under realistic traffic the
scheduler matters more than the kernels, because it decides whether the GPU has
enough work to be worth its bandwidth.

## What static batching wastes

Collect 16 requests, run them together, wait for all 16, start the next batch.

Generation lengths vary by two orders of magnitude in real traffic. One request
generating 2000 tokens holds the batch while the other 15 finished at 20. For
those 1980 steps the GPU processes one sequence in a slot sized for 16.

The waste is proportional to the variance in output length, and that variance is
large. Measured utilization for static batching on chat traffic is commonly under
30%.

## Scheduling per iteration

Run the scheduler before every forward pass. Finished sequences leave immediately;
waiting ones take their place. A request that arrives while a batch is running
joins the next step rather than waiting for the batch to drain.

The scheduler holds three collections:

```python
self.waiting: deque[Request]    # admitted, no cache blocks yet
self.running: list[Request]     # in the batch, holding blocks
self.finished: list[Request]
```

Each step answers one question: which sequences run now, under a token budget and
a block budget.

## Decode before prefill

`Scheduler.step` schedules decodes first, then prefills. The order matters.

A running sequence has already paid for its prefill and is holding cache blocks.
Finishing it frees those blocks. Admitting a new sequence ahead of it does the
opposite — it takes more blocks and pushes the running sequence further from
completion. Prioritizing decode keeps cache pressure falling rather than rising,
and it improves tail latency because sequences near completion get to complete.

## Chunked prefill

A 4000-token prefill in one forward pass takes maybe 200 ms. Every decode in
flight stalls for that long, so every user watching a stream sees a 200 ms hitch
whenever anyone submits a long prompt.

Chunked prefill splits it. Process 512 tokens per step alongside the decodes:

```python
chunk = min(self.chunk_size, budget, request.prompt_len - request.prefilled)
if chunk > 0 and self._reserve(request, chunk):
    batch.prefill.append((request, chunk))
    budget -= chunk
```

Now each step carries a few hundred prefill tokens plus the decodes. Two things
improve at once. Decode latency stops spiking. And decode, which at batch 16
sits at an arithmetic intensity of 16, gets to ride along with prefill tokens that
push the combined batch above the ridge point — the GPU does the decode work
almost for free.

The cost is a slightly longer time to first token for the long prompt, since it
now spans several steps. That trade is nearly always right: one user's TTFT
degrades a little, and every other user's inter-token latency stops jittering.

`max_batched_tokens` sets the budget. Too low and prefill crawls. Too high and
you're back to stalling decodes. Between 2048 and 8192 is the usual range, and
tuning it against your own traffic is worth an afternoon.

## Preemption

The cache is finite. When a running sequence needs a block and none is free, the
scheduler must free one.

`_preempt` evicts the most recently admitted sequence and returns it to the front
of the waiting queue. Newest-first is deliberate: the oldest sequences are closest
to finishing, and finishing them frees blocks. Evicting them instead would push
every sequence away from completion at once.

The evicted request restarts from scratch — its prefill is recomputed when it's
readmitted. Chapter 15 covers why recomputation beats swapping.

Preemption also needs a watermark. Allowing allocation down to the last block
means the next decode step has nowhere to write and immediately preempts, which
thrashes. Holding back 2% gives the scheduler room to make a decision.

## Admission

Admitting a request needs both a token budget and enough blocks for its first
chunk. When either fails, stop admitting — don't skip ahead to a smaller request:

```python
while self.waiting and budget > 0 and len(self.running) < self.max_batch_size:
    candidate = self.waiting[0]
    if not self._reserve(candidate, chunk):
        break
```

Skipping ahead is tempting, and it starves large requests indefinitely under load.
First-come-first-served on the head of the queue is fair and predictable, and it's
what the reference does.

## Verifying a scheduler

A scheduler is easy to test without a GPU, and worth testing hard. The properties
that matter:

- **Every request finishes.** No deadlock, no starvation.
- **Blocks are conserved.** After every request completes, the free list is back
  to its original size. A leak here surfaces days later as capacity that
  mysteriously shrinks.
- **Nothing exceeds its budget.** `batch.token_count` never exceeds
  `max_batched_tokens`.
- **Preemption terminates.** A preempted request eventually completes rather than
  being evicted forever.

The reference implementation drains 12 requests with prompts from 40 to 700 tokens
and outputs from 5 to 50 tokens in 73 steps, returning every block:

```text
steps: 73  tokens processed: 5059
stats: {'waiting': 0, 'running': 0, 'finished': 12, 'cache_utilisation': 0.0}
```

## Lab

Implement `Scheduler.step` with decode priority, chunked prefill, admission
control, and preemption. The harness runs a randomized workload and checks all
four properties, then reports the token budget's utilization across steps.

## Further reading

- [Orca: a distributed serving system for transformer-based generative models](https://www.usenix.org/conference/osdi22/presentation/yu) — the paper that introduced iteration-level scheduling.
- [SARATHI: efficient LLM inference by piggybacking decodes with chunked prefills](https://arxiv.org/abs/2308.16369)
