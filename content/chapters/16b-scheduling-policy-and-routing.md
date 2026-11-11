---
title: Scheduling policy, routing, and the cluster
slug: 16b-scheduling-policy-and-routing
part: "Part 5 — Serving"
summary: Priority with aging, fair sharing with a Virtual Token Counter, routing across replicas by prefix hit with a bounded load, and the arithmetic of autoscaling and cold start.
minutes: 75
gpu: false
objectives:
  - Turn an inter-token latency target into a per-step token budget, and add aging so a low-priority request cannot starve.
  - State the Virtual Token Counter fairness invariant, and explain why a newly backlogged client's counter must be lifted.
  - Compare round-robin, least-loaded, and prefix-aware routing by prefix hit rate and p99 TTFT, and explain what a bounded load fixes.
  - Estimate how many replicas a load needs, and how long a new replica takes to serve its first request.
lab: 16b-policy-and-routing
---

# Scheduling policy, routing, and the cluster

> [!TLDR]
> - A latency target sets the step's token budget. On chapter 16's simulated
>   clock, a 20 ms step holds 144 tokens. Priority needs aging, or a steady
>   high-priority stream starves a low-priority request forever.
> - Virtual Token Counter (VTC) scheduling serves the backlogged client with
>   the least weighted service. Two backlogged clients never drift apart by more
>   than a constant. A client that arrives with an empty queue must have its
>   counter lifted, or it takes every slot until it catches up.
> - Routing across replicas trades cache hits against balance. On the lab's
>   seeded trace, prefix-aware routing lifts the hit rate from 0.44 to 0.76, and
>   pure affinity also raises p99 TTFT from 176 ms to 273 ms. A bounded load
>   keeps most of the hits and cuts p99 to 136 ms.
> - A new replica takes tens of seconds to serve. Size the fleet for the load
>   that arrives before the replica is ready, not the load it sees now.

[Chapter 16](/c/16-continuous-batching) schedules one replica: first come,
first served, with a priority number and preemption. A production service adds
three questions on top. Whose latency target does the next step protect? Who
gets the slots when two clients compete? Which replica gets the request?

This chapter answers each with a rule you can test. The lab runs all of them on
a simulated clock, so every number here is computed or simulated. None is an
A100 measurement.

## Before you start

**The scheduler, from [chapter 16](/c/16-continuous-batching).** A step runs a
batch under a token budget. The lab clock charges 2 ms plus 0.125 ms per token.
Decodes go first, and the lab's `priority` knob orders the waiting queue. That
knob has no aging, which this chapter adds.

**Prefix caching, from [chapter 15a](/c/15a-prefix-caching-and-radix-trees).**
A request whose leading blocks are already cached skips their prefill. The
hit lives on one replica, so which replica gets the request decides whether
there is a hit.

**Percentiles, from [chapter 17](/c/17-benchmarks).** p99 TTFT is the time the
99th-slowest of 100 requests waits for its first token. Queueing drives it.

**Weights, from [chapter 1](/c/01-the-model-on-disk).** The model's 53,791,959,040
bytes of bfloat16 weights sit in 18 shards. A new replica has to read them
before it serves anything.

## Let the target set the budget, and let age raise the priority

This section answers how a latency target and a priority number turn into a
schedule.

Sarathi-Serve splits a long prefill into chunks and adds them to a batch
without pausing the decodes already running (arXiv 2403.02310). The size of
the chunk is the knob. A bigger token budget raises throughput and lengthens
every step, so the inter-token latency (ITL) target caps it.

On chapter 16's clock a step costs $\hla{2} + \hlb{0.125}\,n$ ms for $n$
tokens. Solve for the most tokens that fit under a target $T$:

$$
n_{\max} = \left\lfloor \frac{T - \hla{2}}{\hlb{0.125}} \right\rfloor
$$

Read it as "the budget is what is left after the fixed cost, divided by the
cost per token."

> [!EXAMPLE] Three targets
> A 10 ms target gives $(10 - 2)/0.125 = 64$ tokens. A 20 ms target gives
> $(20 - 2)/0.125 = 144$. A 40 ms target gives 304. A target of 2 ms or less
> leaves no room for a token, so the budget is 0. A tighter target buys
> smoother streaming and costs prefill throughput.

A priority number then orders the waiting queue. Strict priority has a flaw
that chapter 16 names: a steady stream of high-priority work can pass over a
low-priority request indefinitely. *Aging* fixes it by adding one level for
every $a$ ms a request has waited:

$$
p_{\text{eff}} = p + \frac{\text{waited}}{a}
$$

The winner has the highest effective priority. Break a tie by the earliest
arrival, so two requests that age at the same rate keep their order.

> [!KEY] Aging puts a ceiling on the wait
> Suppose a priority-0 request meets a stream of priority-3 requests. After
> $3a$ ms, its effective priority equals theirs, and it arrived first, so it
> wins the tie. With $a = 10$ ms the lab's low request starts at 30 ms. With
> $a = 20$ ms it starts at 60 ms. Without aging it never starts while the
> stream lasts.

A deadline is the SLO version of the same idea. Order by *slack*, the time to
the deadline minus the estimated remaining work, so the request closest to
missing its target goes first. Priority classes with aging are the
approximation you can explain to a customer. Llumnix goes a step further
and migrates running requests between instances to isolate priorities and
balance load (arXiv 2406.03243). This chapter stays with admission order.

## Share fairly with a Virtual Token Counter

This section answers who gets the slots when two clients both want more than
the replica can give.

Fairness by request count is easy to game. A client that sends 4,000-token
prompts costs more than one that sends 40-token prompts. Sheng et al. count
weighted tokens instead (arXiv 2401.00588). Each client $u$ has a counter
$c_u$. Admitting a request with $L_{\text{in}}$ input and $L_{\text{out}}$
output tokens charges

$$
c_u \mathrel{+}= \hla{w_{\text{in}}} L_{\text{in}} + \hlb{w_{\text{out}}} L_{\text{out}}
$$

with $w_{\text{in}} = 1$ and $w_{\text{out}} = 2$ in the lab, because an output
token costs more than an input token. The next request always comes from the
backlogged client with the smallest counter. A real engine charges the output
tokens as it decodes them. The lab charges the whole request at admission,
which keeps the invariant easy to test.

The paper's guarantee is a bound on the service difference between two
backlogged clients, which it calls a 2x tight upper bound. The lab checks
the form $|W_f - W_g| \le 2U$, where $W$ is the service a client has received
since both were backlogged and $U$ is the largest charge of one request.

> [!EXAMPLE] Small requests against large ones
> Client A sends requests of 8 to 48 input tokens. Client B sends 128 to 256.
> Counting requests would hand each client every other slot, and B would use
> five times the compute. VTC lets A run several requests for each one of B's,
> until their counters meet. On the lab's seeded trace, after 200 admissions
> A holds 15,583 weighted tokens and B holds 15,738.

### The lift

A client that has been idle has a low counter. If it returns and keeps that
counter, it would hold the smallest counter for a long time and starve the
others. The fix is a *lift*. When a request arrives and the client's queue is
empty, set the client's counter to the larger of its own and the smallest
counter among the backlogged clients. If nobody is backlogged, use the counter
of the client whose queue emptied last.

> [!WARNING] A newcomer at counter 0
> Without the lift, client C joins at admission 200, when A and B hold about
> 15,600 each. C's counter is 0, so C wins every admission until it has caught
> up. In the lab's trace, the next 200 admissions give C 19,993 weighted
> tokens against 4,321 and 4,444 for A and B, a gap of 15,931. The invariant
> allows 756. With the lift, the three get 9,990, 9,795, and 10,033, and the
> worst gap is 360. Every request still gets served,
> which is why a latency dashboard shows nothing until A and B's users
> complain.

The lift gives a returning client no credit for its idle time. That is the
design. Fairness covers the time you are in the queue, not the time you were
away.

## Route requests across replicas

This section answers which replica gets a request, when each one has its own
KV cache.

With $n$ replicas, a router sees a request and picks one. Three policies cover
the usual ground:

| Policy | Rule | Strength | Weakness |
|---|---|---|---|
| Round-robin | Request $k$ goes to replica $k \bmod n$ | Even counts | Ignores both load and cache |
| Least-loaded | Fewest queued ms of work | Balances the queue | Ignores the cache |
| Prefix-aware | Longest cached prefix, then least load | Reuses prefill | A hot prefix piles onto one replica |

Prefix-aware routing is what Preble studies with a distributed prompt
scheduler that weighs KV reuse against load (arXiv 2407.00023), and chapter 15a
explains why a hit saves prefill.

### The lab's trace

The lab draws 600 requests with seed 18 at 40 per second onto four replicas.
Each request is a 16-block shared prefix (256 tokens) plus a 4-block unique
suffix, and asks for 24 tokens. Twelve prefixes have weights $1/(k+1)$, so the
first is the hot one. Each replica has a 96-block LRU cache, which holds about
four prefixes after the unique suffixes crowd it. A request costs
$2 + 0.125 \times (\text{uncached tokens})$ ms to prefill and 2.125 ms per
output token. A warm request prefills in 10.0 ms and a cold one in 42.0 ms.

Round-robin spreads every prefix over all four caches, so the caches thrash.
Prefix-aware routing gives each prefix a home.

| Policy | Prefix hit rate | p99 TTFT | Share on the busiest replica |
|---|---|---|---|
| Round-robin | 0.4397 | 175.6 ms | 25.0% |
| Least-loaded | 0.4310 | 165.9 ms | 27.5% |
| Prefix-aware, pure affinity | 0.7577 | 273.3 ms | 32.7% |
| Prefix-aware, bounded, $\epsilon = 0.25$ | 0.6630 | 135.9 ms | 27.5% |

```viz
16b-routing-tradeoff
```

Pure affinity wins on hits and loses on the tail. The hot prefix carries about
a third of the traffic, and every request for it goes to the replica that holds
it. That replica queues while the others idle. The hit rate is the right
metric for the cache and the wrong one for the user, who feels p99 TTFT.

### A bounded load

Mirrokni, Thorup, and Zadimoghaddam show how to keep affinity and cap the
damage. With $m$ items over $n$ bins and a parameter $c = 1 + \epsilon$, no bin
holds more than $\lceil c\,m/n \rceil$ (arXiv 1608.01350). Their paper does this
for consistent hashing. The lab borrows the cap and replaces the hash ring
with the cache ranking.

Count the requests still in flight. Before routing a new request, with
$\text{total}$ requests in flight across $n$ replicas, set

$$
\text{cap} = \left\lceil \frac{(1 + \hla{\epsilon})\,(\hlb{\text{total} + 1})}{n} \right\rceil
$$

Rank replicas by cached blocks, then by queued work. Take the first one whose
in-flight count is below the cap. The $+1$ counts the request you are placing.
Some replica is always under the cap, because the in-flight counts sum to
$\text{total}$ and $n \cdot \text{cap} \ge \text{total} + 1$.

> [!INTUITION] A cap, not a ban
> Affinity is a preference. The cap says "prefer the replica with your prefix,
> unless it already holds more than its share." A small $\epsilon$ is close to
> least-loaded. A large $\epsilon$ is close to pure affinity.

The lab's single seed does not give a smooth curve. The p99 values for
$\epsilon$ of 0.1, 0.25, 0.5, and 1.0 are 118.9, 135.9, 118.8, and 177.2 ms,
and the hit rates are 0.666, 0.663, 0.709, and 0.718. Trust the direction, a
better tail than all three simple policies for the first three, and do not
read a trend into the noise. Tune $\epsilon$ on your own trace.

## Plan for autoscaling and cold start

This section answers how many replicas you need, and how long a new one takes.

The fleet size follows from utilisation. With request rate $\lambda$ per second,
mean service time $s$ seconds, and a target utilisation $\rho$, you need

$$
n = \left\lceil \frac{\lambda\, s}{\rho} \right\rceil
$$

The lab's mean service time is near 0.07 s: at the 0.66 hit rate, a request
prefills about 108 uncached tokens (13.5 ms plus 2 ms) and decodes 24 tokens
(51.0 ms), which is 66.5 ms. At a target of 0.7, 40 requests per second needs 4
replicas, 60 needs 6, and 100 needs 10.

A new replica then has to start. The wait is the weight load plus a fixed setup
(the runtime start and any graph capture). The load part is the weight bytes
over the read rate. The 53,791,959,040 bytes from
[chapter 1](/c/01-the-model-on-disk) take 26.9 s at an assumed 2 GB/s and 4.3 s
at 12.5 GB/s, which is the line rate of 100 Gbit/s Ethernet from
[chapter 17b](/c/17b-disaggregated-serving). With an assumed 20 s of fixed
setup, the totals are 46.9 s and 24.3 s. The read rates and the setup time are
assumptions, not measurements.

> [!EXAMPLE] Load that grows faster than the fleet
> Four replicas serve about $4 / 0.07 = 57$ requests per second. If traffic
> jumps from 40 to 60, the excess is about 3 requests per second. Over a
> 46.9 s cold start, about 130 requests queue up with nowhere to go. The
> autoscaler reacted correctly and the users still waited. Scale on the
> trend, keep headroom in the target $\rho$, or keep a warm replica.

Another chapter covers loading and warm pools in depth. Here, treat the cold
start as a delay you must add to the scaling rule.

## What goes wrong

**Strict priority with no aging.** The high-priority class is fine and the low
class never runs. Add the aging term, and test it with a stream that never
pauses.

**Skipping the VTC lift.** A client that returns after an idle spell, or joins
late, starts at 0 and takes every slot. The service gap passes the bound
to more than 20 times the bound in the lab. Lift the counter on arrival to an empty queue.

**Fairness by request count.** One client sends long prompts and gets five
times the compute. Charge tokens, with the output weighted higher.

**Reading the hit rate alone.** Pure affinity gives the best hit rate and the
worst p99 on the lab's trace. Grade a router on both.

**A bounded load that is off by one.** Compare in-flight against the cap with
`<`, and count the request you are placing in $\text{total} + 1$. With `<=`,
a replica runs over the cap on 97 of the 600 routings in the lab.

**Sizing for the load you see.** A replica started now serves in tens of
seconds. Size for the load that arrives by then.

> [!RECAP]
> - The ITL target sets the token budget: $\lfloor (T - 2)/0.125 \rfloor$ on the
>   lab clock, 144 tokens at 20 ms.
> - Priority needs aging. A request that waits $3a$ ms overtakes a priority
>   gap of 3, and wins ties by arrival.
> - VTC charges $w_{\text{in}} L_{\text{in}} + w_{\text{out}} L_{\text{out}}$ and
>   serves the smallest counter. Lift a counter when a client arrives with an
>   empty queue.
> - Prefix-aware routing raises the hit rate and, unbounded, the tail. Cap the
>   in-flight count at $\lceil (1 + \epsilon)(\text{total} + 1)/n \rceil$.
> - Fleet size is $\lceil \lambda s / \rho \rceil$, and a new replica adds a
>   cold start of tens of seconds.

## Check your understanding

> [!QUESTION] A 25 ms ITL target, on a step that costs 2 ms plus 0.125 ms per token, allows how many tokens?
> $\lfloor (25 - 2)/0.125 \rfloor = 184$ tokens. A step of 184 tokens costs
> exactly $2 + 23 = 25$ ms.

> [!QUESTION] A priority-1 request has waited 40 ms with $a = 10$ ms, and a new priority-4 request has waited 0. Which runs first?
> The old one. Its effective priority is $1 + 40/10 = 5$, against 4. Both
> age at the same rate, so the older request stays ahead.

> [!QUESTION] Client C joins with counter 0 while A and B hold about 15,600. Why is that a bug, when C has used nothing?
> C's counter says it is owed about 15,600 tokens of service for a period when it
> was not in the queue. VTC serves the smallest counter, so C takes every slot
> until it has been charged that much, and A and B get nothing. The lift sets C
> to the smallest backlogged counter, so C starts level with the others.

> [!QUESTION] Why does pure affinity raise p99 TTFT even as the hit rate goes up?
> A hit saves prefill. The hot prefix puts about a third of the requests on one
> replica, which queues them. The wait in that queue is larger than the prefill
> it saves, and p99 is set by the longest queue.

> [!QUESTION] Four replicas hold 12 requests in flight, and $\epsilon = 0.25$. What is the cap, and can a replica with 5 in flight take the next request?
> $\lceil 1.25 \times 13 / 4 \rceil = \lceil 4.06 \rceil = 5$. A replica takes a
> request only when its in-flight count is below 5, so a replica with 5 does
> not.

## Lab

> [!TRY]
> Implement `token_budget_for_slo`, `effective_priority`, `pick_next`,
> `VTCScheduler`, `LeastLoaded`, `PrefixAware`, `cold_start_s`, and
> `replicas_needed` on the CPU. You pass when the budget inverts the step-time
> line, aging lets a priority-0 request through, VTC keeps the service gap
> within $2U$ for a late joiner, prefix-aware routing beats round-robin on hit
> rate, and the bounded version has the lowest p99 TTFT.

The harness defines the replicas, the trace, and the simulator. You write
the policies. Every draw uses seed 18, and the lab runs in well under a
second.

`token_budget_for_slo` is the first check: 144, 64, 0, and 0 tokens for
targets of 20, 10, 2, and 1 ms. Next come the aging arithmetic and a stream of
priority-3 requests. The low request has to start at 30.0 ms with $a = 10$
and never start without aging.

The VTC checks cover the charge, the lift, and the invariant. A late joiner
that is not lifted fails the invariant check with a gap near 15,931 against
a bound of 756.

The routing checks replay the 600-request trace through round-robin,
least-loaded, pure affinity, and the bounded router at $\epsilon = 0.25$. The
bounded router must beat round-robin's hit rate by at least 0.15 and have
the lowest p99 of the four. A separate check recomputes the cap at every
routing and counts the requests that land on a replica at or over it. The last
check covers `cold_start_s` and `replicas_needed`.

## Further reading

- [Taming throughput-latency tradeoff in LLM inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) — chunked prefills and stall-free scheduling.
- [Fairness in serving large language models](https://arxiv.org/abs/2401.00588) — the Virtual Token Counter and its bound on the service difference.
- [Llumnix: dynamic scheduling for large language model serving](https://arxiv.org/abs/2406.03243) — live migration of requests across instances.
- [Preble: efficient distributed prompt scheduling for LLM serving](https://arxiv.org/abs/2407.00023) — prefix sharing and load balancing across GPUs.
- [Consistent hashing with bounded loads](https://arxiv.org/abs/1608.01350) — the load cap of $\lceil c\,m/n \rceil$.
- [Chapter 16](/c/16-continuous-batching) is the single-replica scheduler this chapter extends, and [chapter 15a](/c/15a-prefix-caching-and-radix-trees) explains the prefix hits that drive the router.
- [Chapter 17a](/c/17a-performance-model) models step time, and [chapter 17b](/c/17b-disaggregated-serving) splits prefill and decode across GPUs.
