---
title: Benchmarks that mean something
slug: 17-benchmarks
part: "Part 5 — Serving"
summary: TTFT, inter-token latency, throughput, and goodput, each defined from per-request timestamps, with what each one hides and how to measure it honestly.
minutes: 105
gpu: true
objectives:
  - Define TTFT, ITL, end-to-end latency, throughput, and goodput as formulas over five per-request values.
  - Explain why p50 and p99 diverge under batching, and compute the gap from a mixture of step times.
  - Apply Little's law to relate concurrency, throughput, and latency.
  - Time GPU code correctly, and say how many repeats a claim needs.
  - Build a load generator that models arrivals and length distributions rather than sending everything at once.
  - Read a latency-throughput curve and find the knee.
lab: 17-benchmark
---

# Benchmarks that mean something

> [!TLDR]
> - Every serving metric comes from five values you record per request: when it
>   arrived, when its first and last tokens left, and how many tokens it read
>   and wrote. Each metric hides something, so report throughput with latency,
>   and goodput with a tail percentile.
> - Users feel the slow steps, not the average. Under batching, one slow step in
>   20 barely moves the mean but triples the 99th-percentile latency.
> - Little's law ties together how many requests are in flight, how fast they
>   arrive, and how long each one takes, so it exposes numbers that can't all be
>   true.
> - A GPU timing is right only after warmup, with a wait for the GPU to finish,
>   and on one card variant. Twenty runs resolve about a 2% difference.

"1000 tokens per second" isn't a number. Under what load, at what batch size,
with what prompt lengths, and at what latency? Every chapter after this one
claims a speedup, and a speedup is a comparison between two measurements. If the
measurements are sloppy, the rest of the course is decoration.

This chapter treats measurement as its own discipline. First, you define each
metric as a formula over timestamps you can record, and see what each formula
throws away. Then you see what makes a GPU measurement wrong: missing
synchronization, missing warmup, too few repeats, and a workload easier than
the real one.

## Before you start

You need four things, none of them deep.

**Five values per request.** Everything here derives from arrival, first token,
completion, prompt token count, and generated token count.
`Request.metrics` in `engine/scheduler.py` returns exactly those.

**Order statistics.** Sort $n$ samples from smallest to largest, and call them
$x_0 \le \dots \le x_{n-1}$. Those sorted values are the *order statistics*. A
percentile is an interpolation between two of them.

**Two distributions.** The exponential distribution models the gap between
arrivals, and the log-normal models generation lengths. This chapter derives
what you need of both.

**CUDA's asynchrony.** A kernel launch returns to the host, the CPU side of your
program, almost immediately. The work runs later on a *stream*, the GPU's queue
of pending work. It's like posting a letter: the mailbox accepts it at once,
and delivery happens later. This is the most common source of a wrong GPU
timing, and [the notation chapter](/c/00a-notation-and-prerequisites) covers the
vocabulary.

You don't need queueing theory. Little's law, the one result you need from it,
has a four-line proof in this chapter.

## The five timestamps

A benchmark is only as good as what it records, so start with the raw data.
Record five values per request $i$, and nothing else:

| Symbol | Meaning | Kind |
|---|---|---|
| $\hla{a_i}$ | Arrival: when the request entered the system | Timestamp |
| $\hlb{f_i}$ | First token: when the first token was streamed out | Timestamp |
| $\hlc{c_i}$ | Completion: when the last token was streamed out | Timestamp |
| $m_i$ | Prompt tokens | Count |
| $n_i$ | Generated tokens | Count |

==Every metric derives from these five==, and aggregates aren't recoverable in
the other direction. Keep the raw records, not only the summary.

The lab's first record is a good one to hold in your head for the rest of the
chapter:

- It arrives at $\hla{a} = 10.00$ s, with a 100-token prompt.
- Its first token streams out at $\hlb{f} = 10.25$ s.
- It generates 11 tokens, 30 ms apart, so it completes at
  $\hlc{c} = 10.25 + 10 \times 0.03 = 10.55$ s.

The next section turns those numbers into latency metrics.

## Latency: TTFT, ITL, and end to end

Latency answers "how long did the user wait?", and there's more than one kind of
waiting. Three metrics split a request's life at its first token, and each one
hides something different.

### Time to first token

*Time to first token* (TTFT) is the silence between pressing enter and seeing
anything. Subtract the arrival from the first token:

$$
\operatorname{TTFT}_i = \hlb{f_i} - \hla{a_i}
$$

For the lab's record, that's $10.25 - 10.00 = 0.25$ s. TTFT covers queue time
plus *prefill*, the pass that processes the prompt before the first token.

**What it hides.** It sums two terms with different causes and reports neither.
A TTFT of 800 ms could be 780 ms of queueing with 20 ms of prefill, or 50 ms of
queueing with 750 ms of prefill on a long prompt. The first is a capacity
problem, and the second is a kernel problem. Record the moment the scheduler
first admitted the request, and split TTFT in two.

### Inter-token latency

*Inter-token latency* (ITL) is the gap between consecutive tokens once the
stream has started: how smoothly the text flows.

Generating $n_i$ tokens produces $n_i - 1$ gaps, not $n_i$, because TTFT already
accounts for the first token. Think of fence posts: 11 posts have 10 gaps
between them. Divide the streaming time by the gap count:

$$
\operatorname{ITL}_i = \frac{\hlc{c_i} - \hlb{f_i}}{n_i - 1}, \qquad n_i \ge 2
$$

For the lab's record, that's $(10.55 - 10.25)/10 = 0.03$ s, or 30 ms.

For $n_i = 1$ there's no gap, and the metric is undefined. ==Return nothing
rather than zero==: a zero silently drags every aggregate down, and the lab
checks for it.

**What it hides.** This is a mean over one request's own gaps, so it hides that
request's jitter. A stream that emits 127 tokens 20 ms apart and then stalls for
1.2 seconds has a mean ITL of $(127 \times 0.02 + 1.2)/128 \approx 29$ ms, which
looks healthy and isn't. To see stalls, take percentiles within a request, not
only across requests.

### End-to-end latency

End-to-end latency (E2E) is the whole span, and it equals TTFT plus every gap:

$$
\operatorname{E2E}_i = \hlc{c_i} - \hla{a_i}
= \operatorname{TTFT}_i + (n_i - 1)\operatorname{ITL}_i
$$

For the lab's record, that's $0.25 + 10 \times 0.03 = 0.55$ s.

> [!TIP] Assert the identity in your harness
> The example record from `Request.metrics` satisfies it. With `ttft_s` 0.214,
> `latency_s` 3.891, and 128 generated tokens, $(3.891 - 0.214)/127 = 0.02895$
> s, which is the reported `inter_token_ms` of 28.9. If the identity doesn't
> close, a timestamp is taken in the wrong place.

**What it hides.** E2E confounds speed with length. A request generating 2000
tokens has a larger E2E than one generating 20 on an engine performing
identically for both. Compare TTFT and ITL instead, or normalize by $n_i$.

## Throughput and goodput

Latency describes one user's experience; throughput and goodput describe the
whole service. Throughput counts the work a run finishes per second, and goodput
counts only the requests that met their deadlines.

### Throughput

Two quantities share the name. Both divide by the wall-clock span of the run,
from the first arrival to the last completion:

$$
X_{\text{tok}} = \frac{\sum_{i=1}^{N} n_i}{\max_i \hlc{c_i} - \min_i \hla{a_i}},
\qquad
X_{\text{req}} = \frac{N}{\max_i \hlc{c_i} - \min_i \hla{a_i}}
$$

$X_{\text{tok}}$ is output tokens per second over $N$ requests, and
$X_{\text{req}}$ is requests per second. The denominator is ==the wall-clock
span==. It isn't the sum of per-request durations, which double-counts overlap,
and it isn't the time the GPU was busy, which flatters an engine that idles.

> [!EXAMPLE] The lab's ten-request run
> Ten requests arrive one second apart at $a_i = 0, 1, \dots, 9$, each with
> $\operatorname{TTFT} = 0.2$ s, 11 generated tokens, and
> $\operatorname{ITL} = 0.05$ s. The last one completes at
> $c_9 = 9 + 0.2 + 10 \times 0.05 = 9.7$ s. The span is 9.7 s and the token
> count is $10 \times 11 = 110$, so $X_{\text{tok}} = 110/9.7 = 11.34$ tokens
> per second and $X_{\text{req}} = 10/9.7 = 1.03$ requests per second.

**What it hides.** All of the latency. An engine that batches 512 requests and
answers every one 40 seconds later has excellent throughput and is unusable.
Throughput is only meaningful paired with the latency it was achieved at, which
is why the standard presentation is a curve.

It also ignores prompt tokens. Prefill does real work, often most of the FLOPs,
and $X_{\text{tok}}$ counts only output. Report $\sum_i (m_i + n_i)$ separately
to compare total work.

### Goodput

*Goodput* asks a sharper question than throughput: how many requests were served
well enough? It's the fraction of requests that met their *service level
objective* (SLO), a target for TTFT and one for ITL:

$$
G = \frac{1}{N} \sum_{i=1}^{N}
\mathbf{1}\!\left[\operatorname{TTFT}_i \le T_{\text{ttft}}
\;\wedge\; \operatorname{ITL}_i \le T_{\text{itl}}\right]
$$

The indicator $\mathbf{1}[\cdot]$ is 1 when the condition holds and 0 otherwise,
so the sum counts the requests that met both targets. Multiply by
$X_{\text{req}}$ for goodput in requests per second rather than as a fraction.

Goodput catches a configuration that looks excellent on throughput while missing
every deadline. Raise the batch size far enough and throughput keeps climbing
while goodput falls off a cliff. A benchmark that reports only throughput
recommends that configuration.

Add one slow request with a 5-second TTFT to the ten-request run. The following
table shows goodput under two sets of targets:

| $T_{\text{ttft}}$ | $T_{\text{itl}}$ | Requests passing | $G$ |
|---|---|---|---|
| 1 s | 100 ms | 10 of 11 | $10/11 = 0.909$ |
| 1 s | 1 ms | 0 of 11 | 0 |

Throughput is the same in both rows.

**What it hides.** Goodput is binary per request, so it can't tell a request
that missed by 1 ms from one that missed by 10 seconds. Report it alongside a
tail percentile.

## Percentiles

An average blends the bad requests in with the good ones, and users complain
about the bad ones. A *percentile* answers "how slow is the slow end?": p99, the
99th percentile, is the latency that 99% of requests come in at or under. p50,
the median, is the middle.

A percentile needs a precise definition, because several exist and they
disagree on small samples. Use linear interpolation between order statistics,
which is what NumPy and the lab both do.

For sorted samples $x_0 \le \dots \le x_{n-1}$ and $q \in [0, 100]$, compute the
fractional rank $r$, split it into an index $i$ and a fraction $\phi$, and
interpolate:

$$
\begin{gathered}
r = \frac{q}{100}(n-1), \qquad i = \lfloor r \rfloor, \qquad \phi = r - i \\
\boxed{P_q = x_i + \phi\,(x_{i+1} - x_i)}
\end{gathered}
$$

with $P_{100} = x_{n-1}$. The $(n-1)$ rather than $n$ is what makes $P_0$ the
minimum and $P_{100}$ the maximum exactly. For example, for $x = [0, 10]$ and
$q = 25$, $r = 0.25$ and $P_{25} = 0 + 0.25 \times 10 = 2.5$.

> [!EXAMPLE] One slow request in eleven
> Take the eleven-request run, with ten TTFTs of 0.2 s and one of 5.0 s:
>
> $$
> P_{50}: \; r = 5 \Rightarrow x_5 = 0.2 \text{ s},
> \qquad
> P_{99}: \; r = 9.9 \Rightarrow 0.2 + 0.9\,(5.0 - 0.2) = 4.52 \text{ s}
> $$
>
> One bad request moved p99 by a factor of 23 and left p50 exactly where it
> was.

That's the property you want from a tail metric. It's also why the mean is
useless here: the mean of those eleven values is 0.636 s, which ==describes no
request in the run==.

Percentiles need enough samples to exist. A p99 from 50 requests interpolates
between the two largest samples and swings wildly between runs. Budget enough
requests that about ten samples land beyond the percentile, which is
$10 \times 100/(100-q)$ requests:

- For a usable p99, that's about 1000 requests.
- For a p90, it's about 100.

## Why p50 and p99 diverge under batching

Why does the tail of a batched engine look so much worse than its average? The
short answer: batching turns a narrow step-time distribution into a mixture of
fast and slow steps, and a mixture moves the tail far more than the mean.

At batch 1 the ITL distribution is narrow, because every decode step does the
same work. Under continuous batching with chunked prefill, most steps carry
decodes only and take $\hla{t_d}$. Occasionally a step also carries a prefill
chunk for a newly admitted request and takes a much longer $\hlb{t_m}$. With a
fraction $\hlc{\rho}$ of steps mixed, the mean is a weighted average:

$$
\mathbb{E}[t] = (1-\hlc{\rho})\,\hla{t_d} + \hlc{\rho}\,\hlb{t_m}
$$

The percentiles behave differently. While $\hlc{\rho} < 0.01$, the 99th
percentile is still $\hla{t_d}$, and as soon as $\hlc{\rho} > 0.01$ it jumps to
$\hlb{t_m}$.

> [!EXAMPLE] A prefill chunk every twentieth step
> Take $t_d = 30$ ms, $t_m = 90$ ms for a step carrying a 1024-token prefill
> chunk, and $\rho = 0.05$:
>
> $$
> \mathbb{E}[t] = 0.95 \times 30 + 0.05 \times 90 = 33 \text{ ms},
> \qquad P_{50} = 30, \qquad P_{99} = 90
> $$

```viz
17-tail-vs-mean
```

> [!KEY] The mean moved 10% and the tail moved 200%
> A user watching a stream sees the tail: a visible hitch every twentieth token.
> That's why chapter 16 tunes `max_batched_tokens`. It sets $\hlb{t_m}$, and
> therefore p99.

Queueing adds a second mechanism. Past the *knee* of the latency-throughput
curve, the load at which latency starts to climb, the queue grows. Queue time
then enters TTFT for later arrivals but not earlier ones, so the TTFT
distribution becomes long-tailed even though every request does identical work.

## Little's law

Little's law gives you a free consistency check on any benchmark. Picture a
coffee shop where 10 customers arrive per minute and each one stays 3 minutes:
on average, 30 customers are inside. The law says that this arithmetic holds
for any stable system.

Formally, over a long enough window, the mean number of requests in the system
$\hla{L}$ equals the arrival rate $\hlb{\lambda}$ times the mean time a request
spends in the system $\hlc{W}$:

$$
\boxed{\hla{L} = \hlb{\lambda}\,\hlc{W}}
$$

No distributional assumption enters, so ==it holds for any arrival process and
any service discipline==.

> [!DEEPDIVE] The proof is an area argument
> Let $n(t)$ be the number of requests in the system over a window $[0, T]$
> that starts and ends empty. The area under $n(t)$ counts request-seconds, and
> each request contributes exactly $W_i$ of them:
>
> $$
> \int_0^T n(t)\,dt = \sum_{i=1}^{N} W_i
> $$
>
> Divide by $T$. The left side is the time-average concurrency $L$, and the
> right side is
> $\frac{N}{T} \cdot \frac{1}{N}\sum_{i=1}^{N} W_i = \lambda\,W$.

**What it buys you.** Two of the three quantities determine the third, so a
benchmark reporting all three is either consistent or wrong. Suppose your engine
sustains $\hlb{\lambda} = 5.3$ requests per second at a mean latency of
$\hlc{W} = 6.0$ s. Then $\hla{L} = 31.8$ requests are resident on average. The
scheduler's default `max_batch_size` is 32, so you're at the batch-size limit.
The fix for latency is a larger batch or a second GPU, not a faster kernel.

Applied to tokens, the same law gives the decode ceiling. Each running sequence
emits one token every $\operatorname{ITL}$ seconds, so:

$$
X_{\text{tok}} = \frac{\hla{L}}{\operatorname{ITL}}
$$

With $L = 32$ and $\operatorname{ITL} = 30$ ms, that's 1067 tokens per second.
If you measure 400 at batch 32, Little's law says your ITL must be 80 ms, and
you know which number to check.

## Timing GPU code

So far you've defined what to measure. The rest of the chapter is about
measuring it without fooling yourself, starting with the clock. A GPU timing is
right only after warmup, with a synchronize, and with the right clock for the
question.

**Warm up first.** Four things happen on the first call and never again:

- Kernels are compiled or loaded.
- Triton and cuBLAS *autotune*, trying candidate configurations to find the
  fastest.
- The caching allocator requests memory from the driver.
- The caches are cold.

A first Triton call can take hundreds of milliseconds against a steady state
under one. `engine/bench.py` discards five iterations and then times twenty:

```python
def benchmark(fn, label="fn", warmup=5, runs=20):
    for _ in range(warmup):
        fn()
    sync()

    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1000)
```

Warm up with the exact call you're about to time. A different shape re-triggers
autotuning on the first timed iteration.

**Wall clock with a synchronize** measures everything the host experiences:
launch overhead, gaps between kernels, Python time, and the kernels themselves.
That's right for a serving metric, because a user waits for all of it. It's what
`sync()` in the preceding listing provides, wrapping `torch.cuda.synchronize()`,
which blocks the host until the GPU has finished all queued work.

**CUDA events** are markers recorded into a stream. The device timestamps them,
so the elapsed time between two events is device time, excluding host-side gaps:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
fn()
end.record()
end.synchronize()          # the trap: you still have to wait
ms = start.elapsed_time(end)
```

Use events for kernel-level work, where you want device time without the Python
overhead. Use wall clock with a synchronize for anything a user experiences.

> [!WARNING] The synchronization trap has two halves
> **No synchronize at all.** `fn()` returns as soon as the launches are queued,
> so the clock measures how long it took to queue the work. On a decode step of
> several hundred kernels that reports a few microseconds, which looks
> spectacular and is a bug. The symptom is a time below chapter 10's roofline
> floor, the fastest the hardware could possibly move the bytes. If a kernel
> appears to move 53.8 GB in 2 ms, that's 27 TB/s, and the measurement is wrong
> rather than the kernel fast.
>
> **`elapsed_time` before the end event is recorded on the device.** It needs
> both events complete. Without `end.synchronize()`, it raises or reads a
> partially completed stream. Events don't remove the wait, they change what the
> wait measures. Events are also per stream: record on the default stream while
> the work runs on another, and you time nothing.

## How many repeats a claim needs

Timings wobble from run to run, so a single run can't tell a real 2% speedup
from noise. Twenty runs resolve about a 2% difference, and this section shows
where that number comes from.

Let $\bar{t}$ be the sample mean of $n$ timings and $s$ their standard
deviation. Their *coefficient of variation* $c = s/\bar{t}$ is the spread as a
fraction of the mean. The 95% confidence interval for the mean has this relative
half-width:

$$
\frac{1.96\,s}{\bar{t}\sqrt{n}} = \frac{1.96\,c}{\sqrt{n}}
$$

GPU timings in steady state typically have $c$ around 0.03. The half-width then
shrinks with the square root of the number of runs:

- At $n = 5$, it's 2.6%.
- At $n = 20$, it's 1.3%.
- At $n = 100$, it's 0.6%.

Comparing two configurations is harder, because both estimates carry error. For
two independent samples of size $n$ with the same $c$, the difference of means
has relative half-width $1.96\,c\sqrt{2/n}$. So resolving a difference $d$
needs this many runs:

$$
\boxed{n \ge 2\left(\frac{1.96\,c}{d}\right)^{2}}
$$

With $c = 0.03$ and $d = 0.02$, that's $n \ge 17.3$, so 18 runs, and `benchmark`
does 20. A 2% speedup is at the edge of what 20 runs supports, and a 1% speedup
would need $n \ge 69$.

This bounds variance ==within one process on one allocation==. Across processes
you also get a different memory allocation, a different clock history, and
possibly different silicon. No number of in-process repeats measures that.

## Comparing across GPU variants invalidates the comparison

Even a perfectly repeated measurement is wrong if the two runs landed on
different hardware. The A100 80GB ships in two forms with different memory
bandwidth: the SXM4 module rated at 2039 GB/s and the PCIe card at 1935. A
cloud provider hands you whichever is free.

For a memory-bound kernel, which is every decode kernel in this engine, time is
bytes over bandwidth. So the ratio of times across the two cards is the inverse
ratio of bandwidths:

$$
\frac{t_{\text{PCIe}}}{t_{\text{SXM4}}} = \frac{2039}{1935} = 1.054
$$

That's a 5.4% difference from silicon alone, against the 2% that 20 runs can
resolve. A measured 5% "improvement" between runs on different cards is
indistinguishable from landing on the faster card, and a real 5% regression can
hide the same way.

> [!WARNING] Record the environment with every measurement
> Record `gpu_info()` with every measurement, and refuse to compare runs whose
> `name` differs. The same applies to driver version, PyTorch version, and host
> tenancy. A benchmark result without its environment is a rumor.

Clocks are the other environmental axis. An A100 boosts when cool and drops when
hot, so a 10-second benchmark runs at boost clocks, and a 10-minute one runs 10
to 15% lower. Both are real, and they answer different questions:

- Burst clocks say what one request sees on an idle server.
- Sustained clocks say what a loaded server delivers all day.

Five warmup iterations warm the caches and the autotuner but leave the die
cold, so `benchmark` reports boost-clock numbers by construction. That's right
for kernel work and wrong for a capacity estimate.

## What a load generator must model

Sending 100 requests at once measures a burst, not a service. Queueing, which
dominates TTFT under load, only appears when arrivals are spread over time, so a
load generator models when requests arrive and how long they are.

**The arrival process.** Real users don't coordinate. Each one clicks
occasionally and independently, so arrivals come in random clumps: mostly short
gaps, with the occasional long one. The *Poisson process* with rate $\lambda$
is the model for exactly that, and the superposition of many independent,
low-rate users converges to it. Its gaps are exponential, with this cumulative
distribution function (CDF), the probability that a gap is at most $t$:

$$
F(t) = 1 - e^{-\lambda t}, \qquad t \ge 0
$$

Sample the gaps by inverse transform. Set $U = F(\Delta)$ for $U$ uniform on
$(0,1]$ and solve for the gap $\Delta$:

$$
e^{-\lambda \Delta} = 1 - U
\quad\Longrightarrow\quad
\boxed{\Delta = -\frac{\ln(1-U)}{\lambda}}
$$

Since $1-U$ is uniform on $[0,1)$ whenever $U$ is uniform on $(0,1]$, you can
use $\Delta = -\ln(U)/\lambda$ directly, which is what the lab asks for. Arrival
times are the running sum of the gaps, starting at zero.

An exponential has mean $1/\lambda$ and standard deviation also $1/\lambda$, so
==its coefficient of variation is exactly 1==. That's a useful self-check.
Uniform gaps have a coefficient of variation of $1/\sqrt{3} = 0.577$, and the
lab's test separates the two on exactly this statistic.

**The length distributions.** Each length drives a different cost:

- Prompt length drives prefill FLOPs and the chunk the scheduler admits.
- Output length drives how long a sequence holds a batch slot and how many
  cache blocks it accumulates.

Output length matters most, and log-normal is a reasonable default: most
outputs are short, and a few are very long. If $\ln n \sim \mathcal{N}(\mu,
\sigma^2)$, then:

$$
\text{median} = e^{\mu}, \qquad
\mathbb{E}[n] = e^{\mu + \sigma^2/2}, \qquad
P_{99} = e^{\mu + 2.326\,\sigma}
$$

> [!EXAMPLE] A median of 128 tokens
> Take a median of 128 tokens and $\sigma = 1$. Then
> $\mathbb{E}[n] = 128\,e^{0.5} = 211$ and $P_{99} = 128\,e^{2.326} = 1311$
> tokens. One request in a hundred generates ten times the median, and that
> ratio is the whole reason continuous batching exists.

Long prompts also tend to produce long outputs, so sampling the two
independently understates the variance. Replaying a trace of real traffic
avoids the question.

## Why a fixed-length benchmark flatters an engine

The previous section says what realistic load looks like. This one shows what
you lose by skipping it. Fix every prompt at 512 tokens and every output at
128, and four hard problems disappear from view:

- **Static batching stops looking bad.** A static batch runs until its longest
  member finishes, and with identical lengths every member finishes on the same
  step. Chapter 16's entire argument disappears, not because the engine improved
  but because the workload removed the problem.
- **Cache fragmentation disappears.** Every sequence claims the same blocks and
  releases them together, so the free list never fragments, and chapter 15's
  allocator is never tested.
- **The scheduler never preempts.** Cache pressure is constant, so the hardest
  path to get right is never exercised.
- **The tail collapses.** With no variance in the work, p99 converges on p50, and
  the benchmark reports a tail the engine will never reproduce.

A fixed-length benchmark measures kernel speed at one shape. That's legitimate,
and it's what `benchmark` in `engine/bench.py` is for. It ==isn't a serving
benchmark==, and reporting it as one overstates the engine by a large and
unpredictable factor.

## The latency-throughput curve

One curve brings throughput and latency together and tells you how much load to
accept. Sweep the request rate $\lambda$, record throughput and p99 TTFT at each
rate, and plot them against each other. Three regions appear:

| Region | Throughput | Latency | What's happening |
|---|---|---|---|
| Below the knee | Rises linearly with $\lambda$ | Flat | Every request finds the GPU nearly idle. $W$ is roughly constant, so concurrency rises in step with the rate. |
| At the knee | Approaches its ceiling | Starts to climb | The service rate can't absorb fluctuations in arrivals, so a queue forms during bursts and drains between them. This is the operating point you want. |
| Past the knee | Plateaus | Grows without bound | The queue grows faster than it drains, so $W$ grows with time and never settles. Little's law still holds instantaneously, but no steady-state $W$ exists. |

Past the knee, adding load makes everything worse and improves nothing.

> [!KEY] Admit requests below the knee
> A service that accepts requests past the knee is a service that times out.
> Goodput locates the knee cleanly: plotted against $\lambda$, the knee becomes
> a maximum rather than a bend, and the rate at which goodput peaks is the rate
> to admit at.

## Comparing against a production engine

Comparing your engine against a production one is the honest test, and it's
sobering the first time. Match the conditions: same model, dtype, context
length, batch size, sampling parameters, and GPU variant. `enforce_eager=True`
in vLLM turns off CUDA graphs, which replay a recorded sequence of kernel
launches with almost no launch overhead. That makes the comparison fairer if you
haven't implemented them.

Expect to be 2 to 5 times slower at first. The gap comes from CUDA graphs, a
fused GEMM stack, and a great deal of tuning. Decompose the gap rather than
report it: time prefill and decode separately, then compare each against its
roofline floor from chapter 10. A decode step at 40% of its floor with prefill
at 90% is a completely different problem from the reverse.

## What goes wrong

Each of these mistakes produces a number that looks reasonable:

- **Timing without a synchronize.** The measured time falls below the roofline
  floor. Check every result against chapter 10's floor before believing it.
- **Counting $n_i$ gaps instead of $n_i - 1$.** ITL is understated by
  $n_i/(n_i-1)$, which is 0.8% at 128 tokens and 100% at 2. The bug hides in
  long runs and screams in short ones.
- **Using the sum of per-request durations as the throughput denominator.**
  Overlapping requests are counted several times, so throughput is understated by
  roughly the concurrency. The symptom is throughput that falls as batch size
  rises.
- **Reporting the mean of a right-skewed distribution.** GPU timings have long
  tails from allocator activity, scheduling, and clock throttling. The mean sits
  between p50 and p90 and describes neither.
- **Comparing runs from different cards, drivers, or tenancy.** This failure
  produces confident, wrong conclusions, because nothing in the numbers looks
  anomalous.

> [!RECAP]
> - Derive every metric from five per-request values. ITL divides by $n_i - 1$
>   gaps, and throughput by the wall-clock span.
> - Tail percentiles expose the mixed steps that batching creates.
> - Little's law, $L = \lambda W$, holds for any stable system.
> - Warm up, synchronize, stay on one card, and run $n \ge 2(1.96\,c/d)^2$
>   times to resolve a difference $d$.
> - Load with Poisson arrivals and variable lengths, and admit below the knee.

## Check your understanding

> [!QUESTION] Your engine reports 1200 output tokens per second at batch 48 with a p50 ITL of 30 ms. Is that consistent?
> Little's law for tokens gives $48/0.030 = 1600$ tokens per second if all 48
> slots are always full. The measured 1200 is 75% of that, so on average only 36
> slots are occupied. Either the scheduler isn't filling the batch, or requests
> finish and leave gaps before replacements are admitted. Neither is visible in
> the throughput number alone.

> [!QUESTION] Throughput is unchanged when you double `max_batched_tokens`, but p99 ITL doubles. What happened?
> Bigger prefill chunks. The mixed steps got longer, so $t_m$ roughly doubled,
> while $t_d$ and the fraction $\rho$ of mixed steps barely moved. Throughput
> depends on the mean step time, which $t_d$ dominates, so it didn't change. The
> tail depends on $t_m$, so it did. Goodput would have caught this.

> [!QUESTION] A change makes your decode kernel 3% faster over 20 runs. Do you ship it?
> Not on that evidence alone. With $c = 0.03$, resolving 3% needs
> $n \ge 2(1.96)^2 = 7.7$ runs, so 20 is enough within one process. If the two
> measurements came from different processes, the A100 variant alone accounts
> for 5.4%. Re-run both configurations interleaved in one process on one card.

> [!QUESTION] Why does a fixed-length benchmark make continuous batching look pointless?
> Continuous batching wins by replacing a finished sequence mid-batch rather than
> waiting for the slowest member. With identical output lengths every sequence
> finishes on the same step, so there's nothing to replace. The benchmark removed
> the variance the technique exists to exploit.

## Lab

> [!TRY]
> Implement the metric functions over the record format in this chapter, plus a
> Poisson arrival generator. You pass when each formula matches its
> hand-computable case and your arrival gaps look exponential.

Implement the following functions:

- `ttft`.
- `mean_inter_token_ms`, with the $n-1$ gap rule and a `None` for single-token
  responses.
- `percentile`, with linear interpolation between order statistics.
- `summarise`, returning request count, total output tokens, p50 and p99 for
  both TTFT and ITL, and throughput over the wall-clock span.
- `goodput`, against a TTFT target and an ITL target.
- `poisson_arrivals`, returning arrival times whose gaps are exponential.

The harness checks each formula against hand-computable cases: the 9.7-second
span, the 110 tokens, the $10/11$ goodput, and the p99 that moves while p50
doesn't. It also checks that your arrival gaps have a mean of $1/\lambda$ and a
coefficient of variation near 1, which distinguishes exponential gaps from
uniform ones.

Your results are saved, so you can compare runs as you optimize in later
chapters.

## Further reading

- [The vLLM benchmarking suite](https://github.com/vllm-project/vllm/tree/main/benchmarks)
- [DistServe: disaggregating prefill and decoding for goodput-optimized LLM serving](https://arxiv.org/abs/2401.09670)
- [Little's law as viewed on its 50th anniversary](https://pubsonline.informs.org/doi/10.1287/opre.1110.0940)
- [How NOT to measure latency](https://www.youtube.com/watch?v=lJ8ydIuPFeU) — Gil Tene on the mean, the tail, and coordinated omission.
