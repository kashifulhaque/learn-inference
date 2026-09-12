---
title: Benchmarks that mean something
slug: 17-benchmarks
part: "Part 5 — Serving"
summary: TTFT, inter-token latency, throughput, and goodput, each with a formula and with what it hides.
minutes: 90
gpu: true
objectives:
  - Define TTFT, ITL, end-to-end latency, throughput, and goodput as formulas over five timestamps.
  - Explain why p50 and p99 diverge under batching, and compute the gap from a mixture of step times.
  - Apply Little's law to relate concurrency, throughput, and latency.
  - Time GPU code correctly, and say how many repeats a claim needs.
  - Build a load generator that models arrivals and length distributions rather than sending everything at once.
  - Read a latency-throughput curve and find the knee.
lab: 17-benchmark
---

# Benchmarks that mean something

"1000 tokens per second" is not a number. Under what load, at what batch size,
with what prompt lengths, and at what latency? Every chapter after this one
claims a speedup, and a speedup is a comparison between two measurements. If the
measurements are sloppy, the rest of the course is decoration.

This chapter treats measurement as its own discipline. It defines each metric as
a formula over timestamps you can record, says what each formula throws away,
and then covers the mechanics that make a GPU measurement wrong: missing
synchronization, missing warmup, too few repeats, and a workload easier than the
real one.

## Before you start

You need four things, none of them deep.

**Five timestamps per request.** Everything here derives from arrival, first
token, completion, prompt token count, and generated token count.
`Request.metrics` in `engine/scheduler.py` returns exactly those.

**Order statistics.** Given $n$ samples, the sorted values
$x_0 \le \dots \le x_{n-1}$ are the order statistics. A percentile is an
interpolation between two of them.

**Two distributions.** The exponential, which models the gap between arrivals,
and the log-normal, which models generation lengths. Both are derived below.

**CUDA's asynchrony.** A kernel launch returns to the host almost immediately;
the work runs later on a stream. This is the most common source of a wrong GPU
timing. Chapter 0a covers the vocabulary.

You do not need queueing theory. Little's law, the one result you need from it,
is proved below in four lines.

## The five timestamps

Fix the notation for request $i$:

| Symbol | Meaning |
|---|---|
| $a_i$ | Arrival: when the request entered the system |
| $f_i$ | First token: when the first token was streamed out |
| $c_i$ | Completion: when the last token was streamed out |
| $m_i$ | Prompt tokens |
| $n_i$ | Generated tokens |

Record those five per request and nothing else. Everything below is derivable
from them, and aggregates are not recoverable in the other direction.

## Time to first token

$$
\operatorname{TTFT}_i = f_i - a_i
$$

TTFT covers queue time plus prefill. It is what an interactive user feels first:
the silence between pressing enter and seeing anything.

**What it hides.** It sums two terms with different causes and reports neither.
A TTFT of 800 ms could be 780 ms of queueing with 20 ms of prefill, or 50 ms of
queueing with 750 ms of prefill on a long prompt. The first is a capacity
problem and the second is a kernel problem. Record the moment the scheduler
first admitted the request and split TTFT in two.

## Inter-token latency

Generating $n_i$ tokens produces $n_i - 1$ gaps, not $n_i$. The first token is
already accounted for by TTFT.

$$
\operatorname{ITL}_i = \frac{c_i - f_i}{n_i - 1}, \qquad n_i \ge 2
$$

For $n_i = 1$ there is no gap and the metric is undefined. Return nothing rather
than zero; a zero silently drags every aggregate down, and the lab checks for
it.

**What it hides.** This is a mean over one request's own gaps, so it hides that
request's jitter. A stream that emits 127 tokens 20 ms apart and then stalls for
1.2 seconds has a mean ITL of $(127 \times 0.02 + 1.2)/128 \approx 29$ ms, which
looks healthy and is not. To see stalls, take percentiles within a request, not
just across requests.

## End-to-end latency

$$
\operatorname{E2E}_i = c_i - a_i
= \operatorname{TTFT}_i + (n_i - 1)\operatorname{ITL}_i
$$

Use that identity as an assertion in your harness. The example record from
`Request.metrics` satisfies it: with `ttft_s` 0.214, `latency_s` 3.891 and 128
generated tokens, $(3.891 - 0.214)/127 = 0.02895$ s, which is the reported
`inter_token_ms` of 28.9. If that does not close, a timestamp is taken in the
wrong place.

**What it hides.** E2E confounds speed with length. A request generating 2000
tokens has a larger E2E than one generating 20 on an engine performing
identically for both. Compare TTFT and ITL instead, or normalize by $n_i$.

## Throughput

Two quantities share the name. **Output tokens per second** over $N$ requests:

$$
X_{\text{tok}} = \frac{\sum_{i=1}^{N} n_i}{\max_i c_i - \min_i a_i}
$$

and **requests per second**:

$$
X_{\text{req}} = \frac{N}{\max_i c_i - \min_i a_i}
$$

The denominator is the wall-clock span of the run. Not the sum of per-request
durations, which double-counts overlap, and not the time the GPU was busy, which
flatters an engine that idles.

Work through the lab's example, which is small enough to check by hand. Ten
requests arrive one second apart at $a_i = 0, 1, \dots, 9$, each with
$\operatorname{TTFT} = 0.2$ s, 11 generated tokens, and
$\operatorname{ITL} = 0.05$ s. The last one completes at

$$
c_9 = 9 + 0.2 + 10 \times 0.05 = 9.7 \text{ s}
$$

so the span is 9.7 s, the token count is $10 \times 11 = 110$, and
$X_{\text{tok}} = 110/9.7 = 11.34$ tokens per second while
$X_{\text{req}} = 10/9.7 = 1.03$ requests per second.

**What it hides.** All of the latency. An engine that batches 512 requests and
answers every one 40 seconds later has excellent throughput and is unusable.
Throughput is only meaningful paired with the latency it was achieved at, which
is why the standard presentation is a curve. It also ignores prompt tokens:
prefill does real work, often most of the FLOPs, and $X_{\text{tok}}$ counts
only output. Report $\sum_i (m_i + n_i)$ separately to compare total work.

## Goodput

Throughput counts every token. Goodput counts only requests that met their
service level objective:

$$
G = \frac{1}{N} \sum_{i=1}^{N}
\mathbf{1}\!\left[\operatorname{TTFT}_i \le T_{\text{ttft}}
\;\wedge\; \operatorname{ITL}_i \le T_{\text{itl}}\right]
$$

where $\mathbf{1}[\cdot]$ is 1 when the condition holds and 0 otherwise.
Multiply by $X_{\text{req}}$ for goodput in requests per second rather than as a
fraction.

Goodput catches a configuration that looks excellent on throughput while missing
every deadline. Raise the batch size far enough and throughput keeps climbing
while goodput falls off a cliff; a benchmark reporting only throughput will
recommend that configuration.

Add one slow request with a 5-second TTFT to the run above. With
$T_{\text{ttft}} = 1$ s and $T_{\text{itl}} = 100$ ms, ten of eleven pass, so
$G = 10/11 = 0.909$. Tighten the inter-token target to 1 ms and $G = 0$, with
throughput unchanged.

**What it hides.** Goodput is binary per request, so it cannot tell a request
that missed by 1 ms from one that missed by 10 seconds. Report it alongside a
tail percentile.

## Percentiles

A percentile needs a definition, because several exist and they disagree on
small samples. Use linear interpolation between order statistics, which is what
NumPy and the lab both do. For sorted samples $x_0 \le \dots \le x_{n-1}$ and
$q \in [0, 100]$, compute the fractional rank

$$
r = \frac{q}{100}(n-1), \qquad i = \lfloor r \rfloor, \qquad \phi = r - i
$$

and interpolate $P_q = x_i + \phi\,(x_{i+1} - x_i)$, with $P_{100} = x_{n-1}$.
The $(n-1)$ rather than $n$ is what makes $P_0$ the minimum and $P_{100}$ the
maximum exactly. Check: for $x = [0, 10]$ and $q = 25$, $r = 0.25$ and
$P_{25} = 0 + 0.25 \times 10 = 2.5$.

Now the eleven-request run, with ten TTFTs of 0.2 s and one of 5.0 s:

$$
P_{50}: \; r = 5 \Rightarrow x_5 = 0.2 \text{ s}
$$

$$
P_{99}: \; r = 9.9 \Rightarrow 0.2 + 0.9\,(5.0 - 0.2) = 4.52 \text{ s}
$$

One bad request moved p99 by a factor of 23 and left p50 exactly where it was.
That is the property you want from a tail metric, and it is why the mean is
useless: the mean of those eleven values is 0.636 s, which describes no request
in the run.

Percentiles need enough samples to exist. A p99 from 50 requests is
interpolating between the two largest samples and swings wildly between runs.
Budget roughly ten samples beyond the $100/(100-q)$ point: about 1000 requests
for a usable p99, 100 for a p90.

## Why p50 and p99 diverge under batching

At batch 1 the ITL distribution is narrow, because every decode step does the
same work. Batching couples requests, and coupling creates a mixture.

Under continuous batching with chunked prefill, a step is one of two kinds. Most
carry decodes only. Occasionally one also carries a prefill chunk for a newly
admitted request, and that step takes much longer. With decode-only steps at
$t_d$, mixed steps at $t_m$, and a fraction $\rho$ of steps mixed, the mean is

$$
\mathbb{E}[t] = (1-\rho)\,t_d + \rho\,t_m
$$

but the percentiles behave differently: while $\rho < 0.01$ the 99th percentile
is still $t_d$, and as soon as $\rho > 0.01$ it jumps to $t_m$. Put numbers on
it, with $t_d = 30$ ms, $t_m = 90$ ms for a step carrying a 1024-token prefill
chunk, and $\rho = 0.05$:

$$
\mathbb{E}[t] = 0.95 \times 30 + 0.05 \times 90 = 33 \text{ ms},
\qquad P_{50} = 30, \qquad P_{99} = 90
$$

The mean moved 10% and the tail moved 200%. A user watching a stream sees the
tail: a visible hitch every twentieth token. This is the concrete reason chapter
16 tunes `max_batched_tokens` — it sets $t_m$, and therefore p99.

Queueing adds a second mechanism. Past the knee of the latency-throughput curve
the queue grows, so queue time enters TTFT for later arrivals but not earlier
ones, and the TTFT distribution becomes long-tailed even though every request
does identical work.

## Little's law

For any stable system, over a long enough window,

$$
L = \lambda\,W
$$

where $L$ is the mean number of requests in the system, $\lambda$ the arrival
rate, and $W$ the mean time a request spends in the system.

The proof is an area argument. Let $n(t)$ be the number of requests in the
system over a window $[0, T]$ that starts and ends empty. The area under $n(t)$
counts request-seconds, and each request contributes exactly $W_i$ of them:

$$
\int_0^T n(t)\,dt = \sum_{i=1}^{N} W_i
$$

Divide by $T$. The left side is the time-average concurrency $L$; the right side
is

$$
\frac{N}{T} \cdot \frac{1}{N}\sum_{i=1}^{N} W_i = \lambda\,W
$$

No distributional assumption enters, so it holds for any arrival process and any
service discipline.

**What it buys you.** Two of the three quantities determine the third, so a
benchmark reporting all three is either consistent or wrong. If your engine
sustains $\lambda = 5.3$ requests per second at a mean latency of $W = 6.0$ s,
then $L = 31.8$ requests are resident on average. The scheduler's default
`max_batch_size` is 32, so you are at the batch-size limit and the fix for
latency is a larger batch or a second GPU, not a faster kernel.

Applied to tokens, the same law gives the decode ceiling. Each running sequence
emits one token every $\operatorname{ITL}$ seconds, so

$$
X_{\text{tok}} = \frac{L}{\operatorname{ITL}}
$$

With $L = 32$ and $\operatorname{ITL} = 30$ ms that is 1067 tokens per second. If
you measure 400 at batch 32, Little's law says your ITL must be 80 ms, and you
know which number to check.

## Timing GPU code

**Warm up first.** Four things happen on the first call and never again: kernels
are compiled or loaded, Triton and cuBLAS autotune over candidate
configurations, the caching allocator requests memory from the driver, and the
caches are cold. A first Triton call can take hundreds of milliseconds against a
steady state under one. `engine/bench.py` discards five iterations and then
times twenty:

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

Warm up with the exact call you are about to time. A different shape re-triggers
autotuning on the first timed iteration.

**Wall clock with a synchronize** measures everything the host experiences:
launch overhead, gaps between kernels, Python time, and the kernels themselves.
That is right for a serving metric, because a user waits for all of it. It is
what `sync()` above provides, wrapping `torch.cuda.synchronize()`.

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

**The synchronization trap** has two halves. The first is timing with no
synchronize at all: `fn()` returns as soon as the launches are queued, so the
clock measures how long it took to *queue* the work. On a decode step of several
hundred kernels that reports a few microseconds, which looks spectacular and is
a bug. The symptom is a measured time below the roofline floor from chapter 10 —
if a kernel appears to move 53.8 GB in 2 ms, that is 27 TB/s, and the
measurement is wrong rather than the kernel being fast.

The second half is calling `elapsed_time` before the end event has been recorded
on the device. It requires both events complete; without `end.synchronize()` it
raises or reads a partially completed stream. Events do not remove the wait,
they change what the wait measures. Events are also per stream: record on the
default stream while the work runs on another and you time nothing.

## How many repeats a claim needs

Twenty runs is not arbitrary. Let $\bar{t}$ be the sample mean of $n$ timings,
$s$ their standard deviation, and $c = s/\bar{t}$ the coefficient of variation.
The 95% confidence interval for the mean has relative half-width

$$
\frac{1.96\,s}{\bar{t}\sqrt{n}} = \frac{1.96\,c}{\sqrt{n}}
$$

GPU timings in steady state typically have $c$ around 0.03, giving 2.6% at
$n = 5$, 1.3% at $n = 20$, and 0.6% at $n = 100$.

Comparing two configurations is harder, because both estimates carry error. For
two independent samples of size $n$ with the same $c$, the difference of means
has relative half-width $1.96\,c\sqrt{2/n}$, so resolving a difference $d$ needs

$$
n \ge 2\left(\frac{1.96\,c}{d}\right)^{2}
$$

With $c = 0.03$ and $d = 0.02$ that is $n \ge 17.3$, so 18 runs. `benchmark`
does 20. A 2% speedup is at the edge of what 20 runs supports; a 1% speedup
would need $n \ge 69$.

The caveat: this bounds variance *within one process on one allocation*. Across
processes you also get a different memory allocation, a different clock history,
and possibly different silicon. No number of in-process repeats measures that.

## Comparing across GPU variants invalidates the comparison

The A100 80GB ships in two forms with different memory bandwidth: the SXM4
module rated at 2039 GB/s and the PCIe card at 1935. A cloud provider hands you
whichever is free.

For a memory-bound kernel — every decode kernel in this engine — time is bytes
over bandwidth, so the ratio of times across the two cards is

$$
\frac{t_{\text{PCIe}}}{t_{\text{SXM4}}} = \frac{2039}{1935} = 1.054
$$

A 5.4% difference from silicon alone, against the 2% that 20 runs can resolve.
A measured 5% "improvement" between runs on different cards is
indistinguishable from having landed on the faster card, and a real 5%
regression can be masked the same way. Record `gpu_info()` with every
measurement and refuse to compare runs whose `name` differs. The same applies to
driver version, PyTorch version, and host tenancy. A benchmark result without
its environment is a rumor.

Clocks are the other environmental axis. An A100 boosts when cool and drops when
hot, so a 10-second benchmark runs at boost clocks and a 10-minute one runs 10
to 15% lower. Both are real and they answer different questions: burst clocks
say what one request sees on an idle server, sustained clocks say what a loaded
server delivers all day. Five warmup iterations warm the caches and the
autotuner but leave the die cold, so `benchmark` reports boost-clock numbers by
construction — right for kernel work, wrong for a capacity estimate.

## What a load generator must model

Sending 100 requests at once measures a burst, not a service. Queueing — the
thing that dominates TTFT under load — only appears when arrivals are spread
over time.

**The arrival process.** Use a Poisson process with rate $\lambda$. The
justification is not convenience: the superposition of many independent,
low-rate users converges to a Poisson process, which is close to what a real
service sees. Its gaps are exponential, with CDF

$$
F(t) = 1 - e^{-\lambda t}, \qquad t \ge 0
$$

Sample by inverse transform. Setting $U = F(\Delta)$ for $U$ uniform on $(0,1]$
and solving,

$$
e^{-\lambda \Delta} = 1 - U
\quad\Longrightarrow\quad
\Delta = -\frac{\ln(1-U)}{\lambda}
$$

and since $1-U$ is uniform on $[0,1)$ whenever $U$ is uniform on $(0,1]$, you
can use $\Delta = -\ln(U)/\lambda$ directly, which is what the lab asks for.
Arrival times are the running sum of the gaps, starting at zero.

An exponential has mean $1/\lambda$ and standard deviation also $1/\lambda$, so
its coefficient of variation is exactly 1. That is a useful self-check: uniform
gaps have a coefficient of variation of $1/\sqrt{3} = 0.577$, and the lab's test
separates the two on exactly this statistic.

**The length distributions.** Prompt length drives prefill FLOPs and the chunk
the scheduler admits; output length drives how long a sequence holds a batch
slot and how many cache blocks it accumulates. Output length matters most, and
log-normal is a reasonable default. If $\ln n \sim \mathcal{N}(\mu, \sigma^2)$,

$$
\text{median} = e^{\mu}, \qquad
\mathbb{E}[n] = e^{\mu + \sigma^2/2}, \qquad
P_{99} = e^{\mu + 2.326\,\sigma}
$$

Take a median of 128 tokens and $\sigma = 1$. Then $\mathbb{E}[n] = 128\,e^{0.5}
= 211$ and $P_{99} = 128\,e^{2.326} = 1311$ tokens. One request in a hundred
generates ten times the median, and that ratio is the whole reason continuous
batching exists. Long prompts also tend to produce long outputs, so sampling the
two independently understates the variance; replaying a trace avoids the
question.

## Why a fixed-length benchmark flatters an engine

Fix every prompt at 512 tokens and every output at 128 and four things become
artificially easy.

- **Static batching stops looking bad.** A static batch runs until its longest
  member finishes; with identical lengths every member finishes on the same
  step. Chapter 16's entire argument disappears, not because the engine
  improved but because the workload removed the problem.
- **Cache fragmentation disappears.** Every sequence claims the same blocks and
  releases them together, so the free list never fragments and chapter 15's
  allocator is never tested.
- **The scheduler never preempts.** Cache pressure is constant, so the hardest
  path to get right is never exercised.
- **The tail collapses.** With no variance in the work, p99 converges on p50 and
  the benchmark reports a tail the engine will never reproduce.

A fixed-length benchmark measures kernel speed at one shape. That is legitimate,
and it is what `benchmark` in `engine/bench.py` is for. It is not a serving
benchmark, and reporting it as one overstates the engine by a large and
unpredictable factor.

## The latency-throughput curve

Sweep the request rate $\lambda$, record throughput and p99 TTFT at each rate,
and plot them against each other. Three regions appear.

**Below the knee**, throughput rises linearly with $\lambda$ and latency stays
flat. Every request finds the GPU nearly idle. By Little's law, $W$ is roughly
constant, so concurrency rises in step with the rate.

**At the knee**, throughput approaches its ceiling and latency starts to climb.
The service rate can no longer absorb fluctuations in the arrival rate, so a
queue forms during bursts and drains between them. This is the operating point
you want.

**Past the knee**, throughput plateaus and latency grows without bound. The
queue grows faster than it drains, so $W$ grows with time and never settles —
Little's law still holds instantaneously, but no steady-state $W$ exists. Adding
load here makes everything worse and improves nothing.

Find the knee and set admission control below it. A service that accepts
requests past the knee is a service that times out. Goodput is the clean way to
locate it: plotted against $\lambda$, the knee becomes a maximum rather than a
bend, and the rate at which goodput peaks is the rate you should admit at.

## Comparing against a production engine

Comparing your engine against a production one is the honest test, and it is
sobering the first time. Match the conditions: same model, dtype, context
length, batch size, sampling parameters, and GPU variant. `enforce_eager=True`
in vLLM turns off CUDA graphs, which makes the comparison fairer if you have not
implemented them.

Expect to be 2 to 5 times slower at first. The gap comes from CUDA graphs, a
fused GEMM stack, and a great deal of tuning. Decompose it rather than report
it: time prefill and decode separately, then compare each against its roofline
floor from chapter 10. A decode step at 40% of its floor with prefill at 90% is
a completely different problem from the reverse.

## What goes wrong

**Timing without a synchronize.** The measured time falls below the roofline
floor. Check every result against chapter 10's floor before believing it.

**Counting $n_i$ gaps instead of $n_i - 1$.** ITL is understated by
$n_i/(n_i-1)$, which is 0.8% at 128 tokens and 100% at 2. The bug hides in long
runs and screams in short ones.

**Using the sum of per-request durations as the throughput denominator.**
Overlapping requests are counted several times, so throughput is understated by
roughly the concurrency. The symptom is throughput that falls as batch size
rises.

**Reporting the mean of a right-skewed distribution.** GPU timings have long
tails from allocator activity, scheduling, and clock throttling. The mean sits
between p50 and p90 and describes neither.

**Comparing runs from different cards, drivers, or tenancy.** This is the
failure that produces confident, wrong conclusions, because nothing in the
numbers looks anomalous.

## Check your understanding

**Your engine reports 1200 output tokens per second at batch 48 with a p50 ITL
of 30 ms. Is that consistent?**

Little's law for tokens gives $48/0.030 = 1600$ tokens per second if all 48
slots are always full. The measured 1200 is 75% of that, so on average only 36
slots are occupied. Either the scheduler is not filling the batch, or requests
finish and leave gaps before replacements are admitted. Neither is visible in
the throughput number alone.

**Throughput is unchanged when you double `max_batched_tokens`, but p99 ITL
doubles. What happened?**

Bigger prefill chunks. The mixed steps got longer, so $t_m$ roughly doubled,
while $t_d$ and the fraction $\rho$ of mixed steps barely moved. Throughput
depends on the mean step time, which $t_d$ dominates, so it did not change. The
tail depends on $t_m$, so it did. Goodput would have caught this.

**A change makes your decode kernel 3% faster over 20 runs. Do you ship it?**

Not on that evidence alone. With $c = 0.03$, resolving 3% needs
$n \ge 2(1.96)^2 = 7.7$ runs, so 20 is enough *within one process*. If the two
measurements came from different processes, the A100 variant alone accounts for
5.4%. Re-run both configurations interleaved in one process on one card.

**Why does a fixed-length benchmark make continuous batching look pointless?**

Continuous batching wins by replacing a finished sequence mid-batch rather than
waiting for the slowest member. With identical output lengths every sequence
finishes on the same step, so there is nothing to replace. The benchmark removed
the variance the technique exists to exploit.

## Lab

Implement the metric functions over the record format above: `ttft`,
`mean_inter_token_ms` with the $n-1$ gap rule and a `None` for single-token
responses, `percentile` with linear interpolation between order statistics,
`summarise` returning request count, total output tokens, p50 and p99 for both
TTFT and ITL, and throughput over the wall-clock span, `goodput` against a TTFT
target and an ITL target, and `poisson_arrivals` returning arrival times whose
gaps are exponential.

The harness checks each formula against hand-computable cases: the 9.7-second
span above, the 110 tokens, the $10/11$ goodput, and the p99 that moves while
p50 does not. It also checks that your arrival gaps have a mean of $1/\lambda$
and a coefficient of variation near 1, which distinguishes exponential gaps from
uniform ones.

Your results are saved, so you can compare runs as you optimize in later
chapters.

## Further reading

- [The vLLM benchmarking suite](https://github.com/vllm-project/vllm/tree/main/benchmarks)
- [DistServe: disaggregating prefill and decoding for goodput-optimized LLM serving](https://arxiv.org/abs/2401.09670)
- [Little's law as viewed on its 50th anniversary](https://pubsonline.informs.org/doi/10.1287/opre.1110.0940)
- [How NOT to measure latency](https://www.youtube.com/watch?v=lJ8ydIuPFeU) — Gil Tene on the mean, the tail, and coordinated omission.
