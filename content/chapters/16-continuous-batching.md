---
title: Continuous batching
slug: 16-continuous-batching
part: "Part 5 — Serving"
summary: Scheduling per iteration instead of per batch, with the throughput-latency arithmetic, chunked prefill, preemption, and admission control.
minutes: 120
gpu: false
objectives:
  - Quantify what head-of-line blocking costs a static batch under realistic traffic.
  - Write the admit, run, retire loop as pseudocode with the tensor shapes it produces.
  - Derive how batch size moves arithmetic intensity toward the ridge point, and what it does to inter-token latency.
  - Size a prefill chunk from the decode step's memory floor.
  - Explain preemption, admission watermarks, and what a fairness policy does to tail latency.
lab: 16-scheduler
---

# Continuous batching

> [!TLDR]
> - Static batching wastes the GPU on length variance: one long request in a
>   batch of 16 drops utilization to 7.2%.
> - Continuous batching reschedules before every forward pass: retire, decode,
>   continue prefills, then admit from the head of the queue.
> - Batch 1 to batch 64 multiplies throughput by 48 and inter-token latency by
>   only 1.34, because every sequence shares the 53.8 GB weight read.
> - Decode stays memory bound at every batch size the cache can hold. Chunked
>   prefill puts the idle tensor cores to work, and about 250 prefill tokens per
>   step are free.
> - Preempt newest-first, requeue at the front, keep a watermark of free
>   blocks, and never skip the head of the queue.

The kernels are as fast as you're going to make them. Now decide what to run.

Under realistic traffic, the scheduler matters more than the kernels, because
it decides whether the GPU has enough work in flight to be worth its bandwidth.
==A perfect kernel running one sequence at a time wastes 97% of an A100.== A
mediocre kernel running sixty-four wastes almost none. This chapter is about the
loop that picks the sixty-four.

## Before you start

**The roofline from [chapter 10](/c/10-roofline).** An A100 does 312 TFLOP/s
of bfloat16 arithmetic and moves 1275 GB/s in a measured copy against a rating
of 1935. The ridge point quoted throughout the course, 161 FLOPs per byte, is
the ratio against the rating. Below it, an operation is memory bound; above it,
compute bound.

**The memory numbers from [chapter 2](/c/02-memory-arithmetic) and
[chapter 15](/c/15-paged-attention).** Weights 53.8 GB. About 19 GB, or
18,120 MiB, left for cache. KV 64 KiB per token. Recurrent state 147.8 MiB per
sequence, fixed. A page of 16 tokens costs 1 MiB.

**Two latency metrics, kept separate.** *Time to first token* (TTFT) runs from
arrival to the first streamed token and is dominated by queueing and prefill.
*Inter-token latency* (ITL) is the gap between consecutive tokens after that,
and it equals the duration of one scheduler step, whatever else is in the batch.
Users feel TTFT as sluggishness and ITL as stuttering, and almost every
scheduling decision trades one against the other, or both against throughput.

**The request lifecycle.** A request arrives, waits, is admitted, prefills
(possibly across several steps), then decodes one token per step until it emits
an end token or hits its limit. Then it retires and returns its blocks.

## What static batching wastes

Static batching wastes the GPU in proportion to the variance in output length,
and no batch size fixes it. The design is the obvious one: collect $N$
requests, run them together, wait for all $N$ to finish, and start the next
batch. It fails on two independent counts.

**Head-of-line blocking inside the batch.** Generation lengths in chat traffic
span two orders of magnitude. Take a batch of 16 where fifteen requests stop at
20 tokens and one runs to 2000. Every step processes 16 slots, so the useful
token-steps are as follows:

$$
15 \times 20 + 2000 = 2300,
$$

against $16 \times 2000 = 32{,}000$ slot-steps paid for. That's 7.2%
utilization. For 1980 of those 2000 steps, the GPU is doing batch-1 decode in a
batch-16 shaped kernel.

**Queueing outside the batch.** A request that arrives one step after a batch
starts waits for the whole batch to drain. At the decode step time derived
later, about 45 ms, a 2000-step batch takes 91 seconds. The user sees 91 seconds
of nothing before their prompt is even read.

> [!KEY] Tuning the batch size can't fix static batching
> Measured utilization for static batching on chat traffic is commonly under
> 30%. A larger $N$ increases both the variance you're exposed to and the
> queueing delay.

## Scheduling per iteration

Continuous batching, also called *iteration-level scheduling*, runs the
scheduler before *every* forward pass rather than before every batch. A finished
sequence leaves the moment it finishes, and ==a waiting request joins the next
step, not the next batch==.

The state is three collections:

```python
self.waiting: deque[Request]    # arrived, no cache blocks yet
self.running: list[Request]     # in the batch, holding blocks
self.finished: list[Request]
```

Each step retires, decodes, continues prefills, and admits, in that order:

```text
step():
    for each request in running:
        if it finished last step:        retire it, release its blocks
        else if it still needs prefill:  leave it for the prefill pass
        else:                            reserve 1 token; add to batch.decode

    for each running request that still needs prefill:
        chunk = min(chunk_size, remaining_budget, prompt_len - prefilled)
        if blocks can be reserved:       add (request, chunk) to batch.prefill

    while waiting and budget > 0 and len(running) < max_batch_size:
        candidate = waiting[0]           # head of the queue, no skipping
        chunk = min(chunk_size, budget, candidate.prompt_len)
        if blocks cannot be reserved:    break
        move candidate to running; add (candidate, chunk) to batch.prefill

    if batch is empty and running is not empty:
        preempt the newest running sequence and retry
```

Three limits bound every pass:

- **`max_batched_tokens`** caps the tokens in one forward pass, because prefill
  is compute bound and an unbounded prefill starves decode.
- **`max_batch_size`** caps concurrent sequences.
- **The block pool** caps everything underneath both, and `_reserve` is the only
  place that touches it.

### What the batch becomes

The scheduler returns a plan, not tensors, but the shapes it implies are worth
writing down once. Let $P$ be the total prefill tokens in the step,
$D = |\text{batch.decode}|$, and $T = P + D$:

| Tensor | Shape | Note |
|---|---|---|
| `input_ids` | $(T,)$ | prefill chunks and decode tokens, flattened |
| `positions` | $(T,)$ | absolute position per token, per sequence |
| `slot_mapping` | $(T,)$ | flat KV slot per token, from chapter 15 |
| `cu_seqlens` | $(P_{\text{seqs}}+1,)$ | prefix sums bounding each prefill chunk |
| `block_tables` | $(D, \text{max blocks})$ | physical blocks per decoding sequence |
| `context_lens` | $(D,)$ | cached tokens per decoding sequence |
| hidden states | $(T, 5120)$ | one row per token, no padding |
| $Q$ in a full-attention layer | $(T, 24, 256)$ | |
| $K$, $V$ written this step | $(T, 4, 256)$ | scattered by `slot_mapping` |
| logits | $(S, 248320)$ | $S$ = sequences that need a sample |

There's ==no batch dimension anywhere except the block tables==. Tokens from
every sequence are concatenated into one flat run, and attention is told where
the boundaries are. That's what makes ragged batches free.

> [!WARNING] Slice before the LM head
> The logits row is the one people get wrong. Logits are needed only for the
> final token of each decode sequence, and for the final token of a prefill only
> when that chunk completes the prompt. Computing them for every token costs
> $2 \times 5120 \times 248320 = 2.54$ GFLOP per token. At $T = 4096$ that's
> 10.4 TFLOP, or 33 ms at peak, plus 2.03 GB of logits to hold. Slicing isn't an
> optimization; it's a requirement.

## Decode before prefill

`Scheduler.step` schedules decodes first, then continues prefills, then admits
new work. The order is deliberate, and it's the highest-leverage line in the
file.

A running sequence has already paid for its prefill and is holding blocks, so
finishing it *frees* memory. Admitting a new sequence ahead of it does the
opposite: it takes blocks and pushes the running sequence further from
completion.

> [!KEY] Decode priority makes cache pressure fall
> Under memory pressure, prioritizing decode makes cache pressure fall.
> Prioritizing admission makes it rise, which triggers preemption, which
> triggers recomputation, which consumes the compute that would have finished
> the sequences you already had.

Decode priority also improves tail latency, for the same reason
shortest-remaining-processing scheduling does: sequences near completion get to
complete, and a completed request stops accruing latency.

## Throughput against latency

Batching buys throughput almost for free until the per-sequence reads catch up
with the weight read. Every number in this section is arithmetic over the byte
and FLOP counts from chapters 2 and 10, not a measurement.

One decode step reads all the weights, plus, per sequence, its KV cache and its
recurrent state. At context length $L$ and batch $B$:

$$
\text{bytes}(B, L) = \hla{53.8\ \text{GB}} + B \left( \hlb{L \times 64\ \text{KiB}} + \hlc{147.8\ \text{MiB}} \right).
$$

> [!INTUITION]
> The $\hla{\text{weight read}}$ is paid once per step, however many sequences
> ride along. Only the $\hlb{\text{KV cache}}$ and the
> $\hlc{\text{recurrent state}}$ grow with $B$.

FLOPs are about $2 N B$ for $N = 26.9$ billion parameters, so
$53.8\ \text{GFLOP} \times B$. Take $L = 2048$, so each sequence contributes
$\hlb{134.2} + \hlc{155.0} = 289.2$ MB. Divide bytes by the measured 1275 GB/s
and FLOPs by 312 TFLOP/s, and take the larger:

| Batch $B$ | Bytes/step | Step time (ITL) | Throughput | Intensity |
|---|---|---|---|---|
| 1 | 54.1 GB | 42.4 ms | 24 tok/s | 1.0 |
| 8 | 56.1 GB | 44.0 ms | 182 tok/s | 7.7 |
| 16 | 58.4 GB | 45.8 ms | 349 tok/s | 14.7 |
| 32 | 63.1 GB | 49.5 ms | 647 tok/s | 27.3 |
| 64 | 72.3 GB | 56.7 ms | 1129 tok/s | 47.6 |

**The trade is extremely favorable at the bottom.** Going from batch 1 to batch
64 multiplies throughput by 48 and multiplies each user's inter-token latency by
1.34. Every sequence added rides along on weight bytes that were being moved
anyway.

**It stops being favorable when the per-sequence term catches up.** At
$B = 64$, the per-sequence reads are 18.5 GB against 53.8 GB of weights. Past
that, each new sequence costs almost as much as it contributes, the curve
flattens, and ITL rises roughly linearly.

**You can't batch your way to the ridge point.** Intensity at $B = 64$ is 47.6
FLOPs per byte, against a ridge point of 161 for the rated bandwidth, or 245
against the bandwidth you can measure:

$$
\frac{312 \times 10^{12}\ \text{FLOP/s}}{1275 \times 10^{9}\ \text{B/s}}
= 245\ \text{FLOPs per byte}.
$$

The batch can't grow much further, either. At $L = 2048$, each sequence needs
$128 + 147.8 = 275.8$ MiB, so the 18,120 MiB pool holds 65 of them.

> [!KEY] Decode is memory bound at every reachable batch size
> The only way to put compute-bound work on the tensor cores is to mix prefill
> tokens into the decode step. That's the argument for chunked prefill.

## Chunked prefill

Chunked prefill splits a long prompt across steps, so that it fills idle tensor
cores instead of stalling every decode in flight.

Prefill is the opposite of decode: it's compute bound and it's bursty. A
4000-token prompt costs the following:

$$
4000 \times 53.8\ \text{GFLOP} = 215\ \text{TFLOP},
$$

or 690 ms at the A100's peak, and more in practice. Run it as one forward pass
and every decode in flight stalls for that long. ==Every user watching a stream
sees a 690 ms hitch== because a stranger submitted a long prompt.

The chunking itself is four lines:

```python
chunk = min(self.chunk_size, budget, request.prompt_len - request.prefilled)
if chunk > 0 and self._reserve(request, chunk):
    batch.prefill.append((request, chunk))
    budget -= chunk
```

Each step now carries a few hundred prefill tokens alongside the decodes: decode
latency stops spiking, and the otherwise idle tensor cores get work.

### Sizing the chunk

The chunk size follows from the throughput table. A decode step at $B = 16$ is
memory bound with a floor of 45.8 ms. Adding $T_p$ prefill tokens adds
$53.8\ \text{GFLOP} \times T_p$ of arithmetic and almost no new bytes, because
the weights are already being read. Step time becomes roughly the larger of the
memory floor and the compute time:

$$
\max\left(\hla{45.8\ \text{ms}},\ \hlb{0.172\ \text{ms} \times (T_p + B)}\right),
$$

where 0.172 ms is one token's arithmetic at peak. The
$\hla{\text{memory floor}}$ and the $\hlb{\text{compute time}}$ meet at
$T_p + B = 266$:

| Prefill tokens in the step | Added compute at peak | Step time |
|---|---|---|
| 128 | 22 ms | 45.8 ms — free |
| 256 | 44 ms | 45.8 ms — free |
| 512 | 88 ms | 88 ms |
| 1024 | 176 ms | 176 ms |
| 4096 | 705 ms | 705 ms |

> [!INTUITION]
> Up to about 250 tokens per step, prefill hides entirely under the memory time
> the decode step was already paying. Past that, every prefill token is charged
> to every decoding user's inter-token latency.

No real kernel reaches peak, so the free chunk on real hardware is smaller than
250. Measure it rather than trusting the arithmetic.

This is why the lab drives the scheduler with `chunk_size=256`. The engine
defaults to 1024 with a 4096-token budget, which favors throughput over
smoothness. Those two constants are exactly the knob, and tuning them against
your own traffic is worth an afternoon.

### Why mixing prefill and decode needs care

A step that carries both kinds of token is more than a bigger step, for five
reasons.

**Two attention kernels, one batch.** Prefill tokens need a causal,
variable-length attention over their own chunk plus whatever prefix is already
cached. Decode tokens need the paged single-query kernel from chapter 15. Split
the batch by kind before attention and rejoin it after; the linear layers and
the MLP see all $T$ tokens as one flat run.

**A chunk attends to its own predecessors.** Chunk $c$ of a prompt must attend
to every token in chunks $1 \dots c-1$, which are already in the cache, *and*
causally within itself. Getting the offset wrong here is
[chapter 14](/c/14-flash-attention)'s decode offset bug in a new costume: it
passes single-chunk tests and corrupts every prompt longer than `chunk_size`.

**Ordering within the step.** This step's K and V must be written to the cache
before attention reads them for the prefill tokens, and the decode tokens must
see their own new key too. One ordering bug here produces a model that's subtly
worse rather than obviously broken.

**Only the last chunk samples.** A prompt that's 40% prefilled produces no
token. `commit_prefill` advances `prefilled` and nothing else; sampling happens
on the step where `prefilled` reaches `prompt_len`.

**TTFT gets worse for the chunked request.** That's the cost, and it's the right
trade: one user's first token arrives a little later, and every other user's
stream stops stuttering.

## Preemption

When a running sequence needs a block and none is free, the scheduler takes one
from the newest sequence and sends it back to the front of the queue:

```python
def _preempt(self, running: list[Request]) -> None:
    if not running:
        return
    victim = running.pop()          # most recently admitted
    if self.cache is not None:
        self.cache.release(victim.id)
    victim.state = State.PREEMPTED
    victim.prefilled = 0
    victim.output_ids.clear()
    self.waiting.appendleft(victim)
```

Both halves of that policy matter:

- **Newest-first.** The oldest sequences are closest to finishing, and finishing
  them frees blocks. Evicting them would push every sequence away from
  completion at once.
- **Back to the front.** Requeueing at the front rather than the back keeps a
  preempted request from being overtaken by every arrival behind it, which is
  what turns preemption into starvation.

The victim restarts from scratch: its prefill is recomputed on readmission.
Chapter 15 prices that choice. For a 2000-token sequence, recompute costs
roughly 345 ms of prefill arithmetic, against about 13 ms each way to swap
273 MiB over PCIe. The reference still recomputes, despite the transfer looking
cheaper: scattered blocks transfer poorly, a swap-in sits on the critical path,
pinned host memory is a scarce global resource, and recompute is a few lines
instead of a subsystem.

Preemption must also terminate, so `step` bounds its retries:

```python
for _ in range(self.max_batch_size + 1):
    batch = self._schedule()
    if not batch.empty or not self.running:
        return batch
    self._preempt(self.running)
```

At worst, it empties the running list and returns whatever the final attempt
produced. Without that bound, a pathological workload spins forever, preempting
and readmitting the same sequence.

## Admission control

Admission is where overcommitment is prevented, and it has three guards.

**The token budget.** A candidate's first chunk must fit in what remains of
`max_batched_tokens`.

**The block budget with a watermark.** `_reserve` refuses to allocate down to
the last block:

```python
def _reserve(self, request: Request, tokens: int) -> bool:
    needed = self.cache.blocks_needed(request.id, tokens)
    reserve = int(self.cache.num_blocks * self.watermark)
    if needed > max(0, self.cache.free_blocks - reserve):
        return False
    self.cache.allocate(request.id, tokens)
    return True
```

At a 2% watermark on an 18,120-page pool, 362 pages stay unallocated. Without
that reserve, the scheduler allocates the last block, the next decode step has
nowhere to write, and it preempts immediately, then readmits, then preempts
again. ==The watermark buys the scheduler room to decide rather than react.==

> [!WARNING] Admitting to the edge of capacity guarantees preemption later
> A naive admission check forgets two costs. The recurrent state is 147.8 MiB
> per sequence, whatever the length. And an admitted sequence keeps growing: one
> admitted with room for its prompt asks for another block every 16 steps for
> as long as it generates.

**No skipping ahead.** The admission loop inspects only the head of the queue
and breaks when it doesn't fit:

```python
while self.waiting and budget > 0 and len(self.running) < self.max_batch_size:
    candidate = self.waiting[0]
    chunk = min(self.chunk_size, budget, candidate.prompt_len)
    if not self._reserve(candidate, chunk):
        break
```

It's `break`, not `continue`. Skipping to a smaller request that does fit is
tempting: it raises utilization on this step, and it's what a bin-packing
heuristic would do. The next section explains why the reference refuses.

## Fairness and the tail

Skipping ahead is shortest-job-first, and under sustained load, where the queue
never empties, ==shortest-job-first can pass over a large request
indefinitely==. Its mean wait improves; its p99 goes to infinity.

The reference does first-come-first-served (FCFS) on the head of the queue. It's
predictable, it bounds your wait by the work ahead of you, and it's easy to
reason about when a customer asks why their request took 40 seconds.

If you need differentiated service, the policies and their costs are as follows:

| Policy | Mean latency | Tail latency | Notes |
|---|---|---|---|
| FCFS | Worst | Best, and bounded | The reference |
| Shortest-job-first | Best | Unbounded under load | Starves long prompts |
| Priority classes | Depends | Unbounded for low class | Needs aging to be safe |
| FCFS with aging | Near FCFS | Bounded | Priority rises with wait time |

Aging is the usual compromise: order by priority, but raise a request's priority
with the time it has waited, so that nothing can be overtaken forever.

One more fairness effect is invisible in the queue and visible to users.
Continuous batching makes each user's ITL depend on what everyone else is doing,
because ITL is the step time and the step time depends on the batch. A user
whose stream was arriving at 45 ms per token sees 176 ms per token the moment
someone else's 1024-token prefill chunk joins the step. Smaller chunks and a
capped `max_batched_tokens` keep that jitter within a range users don't notice.

## Verifying a scheduler

A scheduler is easy to test without a GPU, and worth testing hard, because its
bugs appear after hours of load rather than on the first request. The properties
that matter are as follows:

- **Every request finishes.** No deadlock, no starvation, and no request left in
  `waiting` when the loop ends.
- **Blocks are conserved.** After every request completes, the free list is back
  to its original size. A leak here surfaces days later as capacity that
  mysteriously shrinks.
- **Nothing exceeds its budget.** `batch.token_count` never exceeds
  `max_batched_tokens`, and `len(batch.decode)` never exceeds `max_batch_size`.
- **Preemption terminates.** A preempted request eventually completes rather
  than being evicted forever.
- **An empty batch means an empty system.** Returning an empty batch while
  sequences are resident deadlocks the serving loop, because nothing will ever
  change to unblock it.

The reference implementation drains 12 requests, with prompts from 40 to 700
tokens and outputs from 5 to 50 tokens, in 73 steps, returning every block:

```text
steps: 73  tokens processed: 5059
stats: {'waiting': 0, 'running': 0, 'finished': 12, 'cache_utilisation': 0.0}
```

Token budget utilization is a useful secondary metric but a poor target. A run
that stays well under the budget isn't necessarily broken: it may have had only
decodes to run, which is one token per sequence per step by definition.

## What goes wrong

**Mutating `self.running` while iterating it.** The retire-and-decode pass
rebuilds the list into `still_running` instead of removing from it in place.
Removing during iteration skips the element after each removal, so a sequence
silently stops being scheduled while still holding its blocks.

**Preempting the sequence you're currently scheduling.** `_preempt` pops from
the running list, and if the request being examined is still in that list, the
scheduler can evict it and then add it to the batch. The reference preempts from
`still_running`, the list of sequences already processed this pass, to avoid
that.

**Charging decode zero tokens.** Each decode step consumes one token of budget
and, one step in sixteen, a new block. A scheduler that only budgets prefill
overcommits gradually and starts thrashing once the batch is large.

**Admitting without room to grow.** The symptom is a preemption rate that climbs
with uptime while the request mix is unchanged.

**Losing generated tokens on preemption.** The reference clears `output_ids`, so
a preempted request restarts generation. In a course scheduler with no model
behind it, the tests still pass. In a real engine, a user watching a stream
would see their output rewind, so production code keeps the generated token ids
and recomputes only their KV.

**Tuning `max_batched_tokens` in isolation.** Too low, and prefill crawls and
TTFT climbs. Too high, and you're back to stalling decodes. It interacts with
`chunk_size`, with the batch size, and with your traffic's prompt length
distribution; change one, and measure all four of throughput, TTFT, ITL, and
preemption rate.

> [!RECAP]
> - Continuous batching reschedules every step, so finished sequences leave
>   and waiting ones join without draining a batch.
> - Schedule decodes before prefills and admissions: finishing running work
>   frees blocks.
> - The shared 53.8 GB weight read makes batching nearly free in latency, but
>   decode never reaches the ridge point, and 65 sequences at 2k context fill
>   the pool.
> - Chunked prefill fills the idle tensor cores. About 250 tokens per step hide
>   under the decode step's memory floor; beyond that, every user's ITL pays.
> - Preempt newest-first to the front of the queue, keep a watermark, admit only
>   from the head, and bound the retry loop.

## Check your understanding

> [!QUESTION] Why does adding sequences to a decode batch raise throughput so much more than it raises latency?
> Because the dominant cost, the 53.8 GB weight read, is shared across the whole
> batch. Adding a sequence adds only its own KV and recurrent state: 289 MB at
> 2k context against 53,800 MB of weights. Throughput scales with $B$ while step
> time grows by well under 1% per sequence, until the per-sequence term becomes
> comparable to the weight read.

> [!QUESTION] If decode never reaches the ridge point, why bother batching at all?
> Because the roofline bounds the *rate*, not the *work*. At batch 1, the GPU
> moves 54 GB to produce one token; at batch 64, it moves 72 GB to produce 64.
> Both are memory bound, and the second is 48 times more efficient per token.
> Reaching the ridge point would mean saturating the tensor cores, which only
> prefill does.

> [!QUESTION] A 2000-token prompt arrives while 16 sequences are decoding. What does the scheduler do, step by step?
> It admits the request only when it's at the head of the queue and enough
> blocks for the first chunk are free. Then it prefills `chunk_size` tokens per
> step alongside the 16 decodes, so at 256 tokens per chunk the prompt takes 8
> steps. The decodes continue throughout, each step a little slower than a pure
> decode step. On the eighth step, the prompt completes and samples its first
> token; from step nine, the request is an ordinary decode.

> [!QUESTION] Why requeue a preempted request at the front rather than the back?
> Because the back is unbounded. A request preempted once is likely to be
> preempted again, since it's newest and newest is the eviction target. Sending
> it to the back of a busy queue means it can be overtaken repeatedly.
> Front-of-queue requeueing bounds the number of times it loses its place to
> one.

## Lab

> [!TRY]
> Implement `BlockPool`, `Request`, and `Scheduler.step` with decode priority,
> chunked prefill, admission control under a watermark, and newest-first
> preemption. You pass when every workload drains with blocks conserved and no
> invariant breaks at any step.

The harness drives the scheduler to completion on several workloads and checks
invariants at every step:

- **A single request** must complete.
- **A mixed workload** of 12 requests, with prompts of 40, 300, or 700 tokens
  and outputs of 5, 20, or 50 tokens, must finish entirely: every request fully
  prefilled, every request generating exactly what it asked for, all blocks
  returned, and both queues empty.
- **A 2000-token prompt** at `chunk_size=256` must take at least 8 steps, which
  proves the prefill is chunked.
- **A memory-constrained run** of 10 requests of 500 to 900 tokens, against a
  90-block pool with `max_batch_size=4`, must still drain, with blocks conserved
  across preemptions.
- **One sequence decoding with a 512-token prompt waiting** must put the decode
  in the batch, which proves decode is scheduled ahead of admission.

Throughout, `batch.token_count` must never exceed `max_batched_tokens`,
`len(batch.decode)` must never exceed `max_batch_size`, and an empty batch must
never be returned while work remains.

## Further reading

- [Orca: a distributed serving system for transformer-based generative models](https://www.usenix.org/conference/osdi22/presentation/yu) — the paper that introduced iteration-level scheduling.
- [SARATHI: efficient LLM inference by piggybacking decodes with chunked prefills](https://arxiv.org/abs/2308.16369)
- [Taming throughput-latency tradeoff in LLM inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310)
- [Efficient memory management for large language model serving with PagedAttention](https://arxiv.org/abs/2309.06180)
