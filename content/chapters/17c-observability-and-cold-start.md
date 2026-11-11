---
title: Observability, reliability, and cold start
slug: 17c-observability-and-cold-start
part: "Part 5 — Serving"
summary: How to turn an engine's event log into queue, TTFT, and ITL histograms, what a quantile read back from buckets can and cannot tell you, how to alert on a preemption storm without flapping, and how long a replica takes to become ready.
minutes: 70
gpu: false
objectives:
  - Parse an engine event log into queue time, time to first token, and inter-token latency, and say where a preemption stall shows up.
  - Build Prometheus-style cumulative histograms with a +Inf bucket, and read a quantile back with linear interpolation.
  - Bound the error of a bucketed quantile, and say why you add histograms from replicas and never average their percentiles.
  - Write an alert with a pending window and a keep-firing window, so a spike pages no one and a dip does not page twice.
  - Split a cold start into image pull, weight load, and warmup, and size a warm pool from it.
lab: 17c-observability
---

# Observability, reliability, and cold start

> [!TLDR]
> - An engine tells you what it did through a log of events and a page of Prometheus metrics. From the events you can rebuild queue time, TTFT, and ITL. On the lab's seeded log, 78 preemptions each leave an inter-token gap over 0.25 s, among 15,603 gaps.
> - A histogram stores cumulative counts per upper bound (`le`), ending in `+Inf`. You can add histograms from replicas. You can't recover a quantile exactly: the estimate is a linear guess inside one bucket, so its error is at most that bucket's width. One p99 comes out at 2.184 s against an exact 1.871 s.
> - Liveness says the process answers. Readiness says it can serve a request. While 53.8 GB of weights load, a replica must be live and not ready.
> - Alert on a windowed condition, not a sample. A pending window of 30 s ignores a one-sample spike, and a keep-firing window of 15 s turns a storm with a 3-second dip into one alert instead of two.
> - On assumed rates, a cold start takes 52.9 s: 8.0 s of image pull, 26.9 s of weight load, and 18.0 s of warmup. A streaming loader saves 1.7 s of the load. The read dominates. A warm pool of 6 replicas covers a burst of 0.1 replicas a second.
> - Every number here comes from the lab's seeded logs and its model. None is an A100, disk, or network measurement.

[Chapter 17](/c/17-benchmarks) measured an engine from outside, on a trace you chose. A production engine is measured the other way round: it runs all day on traffic you didn't choose, and you learn what happened from what it emits. This chapter covers the three things you need to run one: numbers that describe latency, alerts that fire when something is wrong, and an honest figure for how long a new replica takes to help.

The lab runs on the CPU. It parses a seeded event log, builds histograms from it, runs alert rules over seeded gauge series, and models a cold start with the rates you pass in.

## Before you start

**TTFT, ITL, and percentiles, from [chapter 17](/c/17-benchmarks).** Time to first token (TTFT) runs from a request's arrival to its first output token. Inter-token latency (ITL) is the gap between two consecutive tokens of one request. A percentile uses linear interpolation between order statistics at rank $q(n-1)$. A p99 needs on the order of 1,000 samples to mean anything.

**The engine loop, from [chapter 16a](/c/16a-the-serving-loop-and-api).** The core schedules, runs a step, and returns an event for each request that made progress. Those events carry the timestamps this chapter uses.

**Safetensors and mmap, from [chapter 1](/c/01-the-model-on-disk).** A safetensors file is a header plus raw bytes, and `mmap` lets you touch a tensor without reading the whole file: the kernel fetches a 4 KB page when you first touch it.

**CUDA graphs, from [chapter 10a](/c/10a-cuda-graphs-and-launch-overhead).** An engine captures one graph per batch-size bucket before it serves, and a captured set of 9 sizes means 9 captures at startup.

## What an engine tells you

This section lists where the numbers come from, and the real names vLLM gives them.

An engine reports in two ways. It writes a log of what happened to each request, and it exposes a page of counters, gauges, and histograms at `/metrics` for Prometheus to scrape. A *counter* only goes up, a *gauge* goes up and down, and a *histogram* counts observations into buckets.

vLLM documents these names. The type column is from its metrics design page and its metrics usage page.

| What it measures | vLLM name | Type |
|---|---|---|
| Time to first token | `vllm:time_to_first_token_seconds` | Histogram |
| Gap between tokens | `vllm:inter_token_latency_seconds` | Histogram |
| Wait before first schedule | `vllm:request_queue_time_seconds` | Histogram |
| Whole request | `vllm:e2e_request_latency_seconds` | Histogram |
| Requests running, waiting | `vllm:num_requests_running`, `vllm:num_requests_waiting` | Gauges |
| Fraction of KV blocks in use | `vllm:kv_cache_usage_perc` | Gauge, 0 to 1 |
| Preemptions, all requests | `vllm:num_preemptions` | Counter |
| Preemptions per request | `vllm:request_num_preemptions` | Histogram |

The pages list `vllm:num_preemptions` without a suffix, and Prometheus client libraries add `_total` to a counter's exposed name. Read your server's `/metrics` output for the exact spelling before you write a query. The design page says it hasn't fixed bucket boundaries, and its example `time_to_first_token_seconds` output shows bounds from 0.001 s to 0.1 s. The lab picks its own.

Google's SRE book names four signals for any service: latency, traffic, errors, and saturation. The metrics above map onto them. TTFT, ITL, and queue time are latency. Running and waiting requests are traffic and saturation. The KV-cache fraction and preemptions are saturation, the early warning that a GPU is full. The book also recommends recording latency in buckets, not only as an average, which is the next two sections.

## From an event log to three samples

This section answers how to get queue time, TTFT, and ITL out of a log of timestamps.

A request's life leaves five kinds of event: `arrived`, `scheduled`, `token`, `preempted`, and `finished`. The lab's log has one per line, in the form `12.340000 req=7 token`. Lines from different processes arrive out of order, and a real log has junk in it, so the parser sorts by time and counts the lines it can't read.

Three numbers come out for each request:

- **Queue time** is the first `scheduled` minus `arrived`.
- **TTFT** is the first `token` minus `arrived`.
- **ITL** is each `token` minus the one before it.

> [!EXAMPLE] A request that gets preempted
> Request 1 arrives at 0.000 s, is scheduled at 0.020 s, and emits tokens at 0.100 s and 0.130 s. It is preempted at 0.500 s, scheduled again at 0.900 s, and emits its next token at 0.950 s. Its queue time is 0.020 s, its TTFT is 0.100 s, and its ITL gaps are 0.030 s and 0.820 s. The second `scheduled` doesn't count toward queue time, and the stall stays in the gap it delays.

==A preemption doesn't appear in TTFT or queue time. It appears as one long ITL gap.== On the lab's seeded log, with 400 requests and 8 requests a second, the 15,603 ITL gaps include 78 over 0.25 s, one for each preemption. The rest are near the 25 ms floor. A mean ITL over the log hides those 78. A histogram keeps them, because they land in the upper buckets.

## Histograms

This section answers what a histogram stores, and why it stores it cumulatively.

A histogram picks upper bounds in advance and counts how many observations fell at or under each one. Prometheus labels each bucket `le`, for "less than or equal," and every bucket counts all the smaller ones too. The last bucket has `le="+Inf"` and counts everything, so it equals the total. Two more series, `_sum` and `_count`, hold the sum and the number of observations.

> [!EXAMPLE] Five samples, three bounds
> Take the samples 0.05, 0.1, 0.1, 0.3, and 9.0 s, and the bounds 0.1, 0.5, and 1.0 s. A sample equal to a bound counts in that bound's bucket. The cumulative counts are 3, 4, 4, and 5 for `le` of 0.1, 0.5, 1.0, and `+Inf`. The sum is 9.55 and the count is 5.

The cumulative form earns its keep in two ways. It is self-checking: counts can only grow as `le` grows, so a histogram that breaks that rule is corrupt. And two histograms with the same bounds add: add their buckets, sums, and counts, and the result is the histogram of the combined samples. That is how Prometheus combines replicas, with `sum by (le)`.

On the seeded log, the lab's 11 bounds (0.005 s to 10 s) give these ITL counts:

| `le` | 0.025 | 0.05 | 0.1 | 0.25 | 0.5 | 1.0 | 2.5 | `+Inf` |
|---|---|---|---|---|---|---|---|---|
| Cumulative ITL count | 0 | 15,494 | 15,525 | 15,525 | 15,536 | 15,588 | 15,603 | 15,603 |

All of the typical gaps are in one bucket, the one from 0.025 s to 0.05 s. The next section shows what that costs.

## What a quantile from buckets can and cannot tell you

This section answers how close a quantile read from buckets comes to the true one.

The histogram forgot where in a bucket each sample fell. To produce a quantile, Prometheus's `histogram_quantile` assumes the samples in the bucket are spread evenly. For a quantile $q$ and a total count $N$, take the rank $q N$, find the first bucket whose cumulative count reaches it, and interpolate:

$$
\boxed{\hla{\text{lower}} + (\hlb{\text{upper}} - \hla{\text{lower}})\,\frac{qN - \hlc{c_{\text{before}}}}{c_{\text{here}} - \hlc{c_{\text{before}}}}}
$$

Here $\hla{\text{lower}}$ and $\hlb{\text{upper}}$ are the bucket's bounds, $\hlc{c_{\text{before}}}$ is the cumulative count of the bucket below it, and $c_{\text{here}}$ is this bucket's cumulative count. The first bucket's lower bound is 0. The estimate moves from the lower bound to the upper bound as the rank moves through the bucket's own samples.

> [!EXAMPLE] A p95 that is 34% too high
> Put 100 samples, all at exactly 0.22 s, into buckets with bounds 0.1, 0.2, 0.3, and 0.5 s. All of them land in the bucket from 0.2 s to 0.3 s. The p95 has rank 95, and nothing is below the bucket, so the estimate is $0.2 + 0.1 \times 95/100 = 0.295$ s. The exact p95 is 0.22 s. The estimate is 0.075 s high, and it would give the same answer if the samples were spread from 0.2 s to 0.3 s.

Prometheus's histogram guide says the same thing: the error is limited by the width of the bucket that holds the quantile. The estimate always lands inside that bucket, and so does the exact value, so ==the error can never exceed the bucket's width==. The width, not the number of buckets, sets the error. Three layouts on the seeded log show it:

| Quantile | Exact | Coarse (5 bounds) | Lab (11 bounds) | Fine (22 bounds) |
|---|---|---|---|---|
| TTFT p99 | 1.871 s | 4.158 s (+2.287) | 2.184 s (+0.313) | 1.875 s (+0.004) |
| ITL p50 | 27.8 ms | 25.2 ms (-2.6) | 37.6 ms (+9.8) | 28.5 ms (+0.7) |

The lab layout puts the TTFT p99 in the bucket from 1 s to 2.5 s, which is 1.5 s wide. The ITL p50 sits in the bucket from 25 ms to 50 ms, and the estimate is 35% high. The coarse layout's ITL p50 looks good only by luck: it divides the whole range from 0 to 50 ms evenly, and this median happens to sit near the middle. ==Put bucket bounds where you want the answer to be accurate==, at the SLO targets you alert on, not at round numbers.

```viz
17c-bucket-quantile
```

Two more limits matter. If the quantile lands in the `+Inf` bucket, nothing says how far above the top bound the samples go, so `histogram_quantile` returns the highest finite bound. On the hand example with 50 samples at 0.05 s and 50 at 7.0 s, bounds 0.1 and 1.0 s, the p99 comes back as 1.0 s. The truth is 7.0 s. And you can't average percentiles across replicas. Replica A serves 1,000 requests at 0.05 s and has a p99 of 0.04975 s. Replica B serves 10 requests at 2.0 s and has a p99 of 2.485 s. The average of the two is 1.267 s, which describes no request. Add the buckets first, then take the quantile: the merged p99 is 0.0500 s, because 1,000 of the 1,010 requests were fast.

> [!KEY] Add buckets, then take the quantile
> Buckets, sums, and counts add across replicas and across time. Percentiles don't. Compute the quantile as late as you can, on the merged buckets, and read it as a value inside one bucket.

## Liveness, readiness, and startup

This section answers what a health check should say while the weights load.

Kubernetes has three probes, and an inference server needs all three because it takes about a minute to be useful. A *liveness* probe asks whether the container is still working, and after `failureThreshold` failures in a row the kubelet kills it and restarts it. A *readiness* probe asks whether the container can accept traffic. On failure the pod leaves the Service's endpoints and isn't restarted. A *startup* probe protects a slow start: while it runs, the other two are disabled, and its window is `failureThreshold` times `periodSeconds`.

The lab models the timeline of one pod, with an 8 s image pull, a 26.9 s load, and an 18 s warmup:

| Time | Phase | Live | Ready |
|---|---|---|---|
| 0 to 8 s | Pulling the image. No container runs. | No | No |
| 8 to 34.9 s | Loading weights. | Yes | No |
| 34.9 to 52.9 s | Warmup, with graph capture. | Yes | No |
| 52.9 s on | Serving. | Yes | Yes |

The first wrong answer is to make liveness wait for readiness. The process is up while it loads, and a liveness probe that fails then gets the pod killed and restarted, and it never finishes loading. The second wrong answer is to report ready when the weights arrive. The pod joins the Service, and its first requests wait for graph capture, which is TTFT you can avoid.

Size the startup probe from the time after the container starts. Load plus warmup is 44.9 s, so a period of 10 s needs a `failureThreshold` of 5. The pull happens before the container exists and doesn't count. Which path your engine serves for each probe is engine-specific. Check its docs and your own server's behavior, and don't assume one path covers both.

## Alerts that don't flap

This section answers how to turn a gauge into an alert that pages once, for an incident.

Two conditions say a GPU is in trouble. The KV-cache fraction, `vllm:kv_cache_usage_perc`, near 1 means the pool is full. A rising preemption rate means the engine is evicting running requests to make room, recomputing or swapping them. Both come from a series sampled every second or every 15 seconds, and a raw series is noisy. An alert on a single sample pages on every spike.

Prometheus's alerting rules solve this with two clauses. `for` holds an alert *pending* until the condition has been true for that long. `keep_firing_for` keeps an alert *firing* for that long after the condition goes false. The lab implements both in `evaluate_alert`.

For a counter you first turn it into a rate. `counter_rate` adds the increases between consecutive samples in a window, counts a drop as a reset to 0 (a restart), and divides by the time covered.

The lab's seeded gauges run for 600 s with four planted events: a one-sample spike to 1.0 at $t = 100$, a storm from 200 s to 319 s with a three-second dip at 260 to 262, an alternating 0.96 and 0.90 from 400 s to 439 s, and a counter reset at 500 s. The rule is `kv >= 0.95`.

| Rule | Intervals |
|---|---|
| `for` 0 s, `keep` 0 s | 23: the spike, the storm in two halves, and 20 one-second flaps |
| `for` 30 s, `keep` 0 s | (230, 260) and (293, 320) |
| `for` 30 s, `keep` 15 s | (230, 335) |

A pending window of 30 s drops the spike and the flap, because neither holds for 30 samples in a row. The dip splits the storm in two unless the rule keeps firing through it, and 15 s of keep-firing bridges a 3 s dip. The cost is delay: the alert fires 30 s late, and resolves 15 s late. That is the trade the SRE book makes: a page should be urgent and actionable, and a noisy one costs more than a late one.

The preemption rule uses the same machinery on the rate. A rate of 1 preemption a second or more over a 30 s window, with `for` 30 s and `keep` 15 s, fires once on the seeded counter, from 238 s to 357 s. The peak windowed rate is 3.73 a second. The reset at 500 s doesn't fire it. A rate that ignores resets reads that restart as a large negative increase.

> [!NOTE] Page on the symptom
> A full KV cache is a cause. The symptom is a TTFT or ITL SLO breach, which is what a user sees. The SRE book suggests paging on symptoms and using causes, like cache fullness or preemptions, as dashboards and debugging aids. Use these rules to explain a page, or to page when the cache fills faster than the autoscaler can respond.

## How long until a new replica helps

This section answers how long a new replica takes to serve, and what you can do about it.

A cold start has three costs in series. The image pull moves the container image, $\text{image}/\text{registry rate}$. The weight load gets 53.8 GB from storage into GPU memory. The warmup captures a CUDA graph for each batch-size bucket, as [chapter 10a](/c/10a-cuda-graphs-and-launch-overhead) describes. The sum is the time to ready:

$$
t_{\text{ready}} = \hla{t_{\text{pull}}} + \hlb{t_{\text{load}}} + \hlc{n_{\text{graphs}}\, t_{\text{capture}}}
$$

The rates below are assumptions chosen for the chapter. Nobody measured them here. The image is 8 GB at a registry rate of 1 GB/s, the storage rate is 2 GB/s, and a graph capture takes 2.0 s for each of the 9 buckets chapter 10a captures. The one number the course owns is the one-way PCIe rate of 32 GB/s from [chapter 19](/c/19-tensor-parallelism).

The weights move in two stages: a read from storage at the source rate, and a copy from host memory to the GPU. The copy is 53.8 GB over 32 GB/s, which is 1.681 s. The read is 53.8 GB over the source rate, which is 26.9 s at 2 GB/s.

**An mmap loader runs the stages in series.** It maps the file and lets the copy touch the pages, so each page is read as it is first used. In the lab's model the time is the read plus the copy. The kernel's readahead can overlap some of this in practice, and the model doesn't say how much.

**A streaming loader runs them as a pipeline.** It reads chunk $k+1$ while it copies chunk $k$. With $n$ equal chunks, where $a$ is a chunk's read time and $b$ is its copy time, a two-stage pipeline takes $n \max(a, b) + \min(a, b)$. That is the slower stage's total plus one chunk of the faster stage:

$$
\boxed{t_{\text{load}} = \max\!\left(\frac{W}{r_{\text{src}}},\ \frac{W}{r_{\text{pcie}}}\right) + \min\!\left(\frac{c}{r_{\text{src}}},\ \frac{c}{r_{\text{pcie}}}\right)}
$$

Here $W$ is the weight size, $c$ the chunk size, and $r$ a rate. CoreWeave's Tensorizer and the Run:ai Model Streamer are built on this idea. The streamer reads a safetensors checkpoint with several threads into CPU buffers while the application copies tensors already read to the GPU. Tensorizer serializes tensors into one file that it can stream from HTTP, S3, or disk. The Tensorizer README claims wire speed, about 5 GB/s for GPT-J on a 40 GbE network. The Run:ai page I read gives no speed figure, so none is quoted here.

With 0.5 GB chunks, the model gives:

| Source rate | Read | Copy | mmap | Streaming | Saved |
|---|---|---|---|---|---|
| 1.25 GB/s (10 GbE) | 43.040 s | 1.681 s | 44.721 s | 43.056 s | 1.666 s (3.7%) |
| 2 GB/s | 26.900 s | 1.681 s | 28.581 s | 26.916 s | 1.666 s (5.8%) |
| 7 GB/s | 7.686 s | 1.681 s | 9.367 s | 7.701 s | 1.666 s (17.8%) |
| 25 GB/s | 2.152 s | 1.681 s | 3.833 s | 2.168 s | 1.666 s (43.5%) |
| 64 GB/s | 0.841 s | 1.681 s | 2.522 s | 1.689 s | 0.833 s (33.0%) |

==Streaming hides the faster stage, so it saves at most the smaller of the read and the copy.== The copy is 1.681 s at every source rate, so the loader never saves more than that. At 2 GB/s that is 5.8% of the load. The read is the cost, so the way to cut the cold start is to read less or read faster: keep the weights on a local NVMe disk or in a page cache, and don't pull 53.8 GB over a network on each start. At 64 GB/s the source outruns PCIe, and the copy becomes the bottleneck.

The totals at 2 GB/s are 8.0 + 26.916 + 18.0 = 52.916 s with a streaming loader and 54.581 s with mmap. The warmup is the second-largest piece, and fewer buckets or a faster capture cuts it. The model leaves out tensor-parallel loading, decompression, and pageable-memory copy rates below the PCIe figure.

> [!KEY] A warm pool pays the cold start in advance
> If a burst wants new replicas at a rate of $\lambda$ a second, then during one cold start of 52.9 s it wants $\lambda \times 52.9$ of them, and none is ready in time. By Little's law, that count is the pool to keep ready. At 0.1 replicas a second it is $\lceil 5.29 \rceil = 6$, and at 0.5 a second it is 27.

A warm pool trades money for latency: idle GPUs wait for a burst. Run it only when the cold start is longer than your autoscaler's reaction budget. Scaling up before the burst, on queue depth or on the KV-cache fraction, is cheaper than a pool.

## What goes wrong

- **Non-cumulative buckets.** A bucket that counts only the samples between its bound and the one below it breaks `histogram_quantile` and sums across replicas. The lab's check expects cumulative counts of 3, 4, 4, 5 and fails a version that returns 3, 1, 0, 1.
- **A quantile with no interpolation.** Returning the bucket's upper bound gives 0.3 s for the 0.22 s sample set, instead of 0.295 s. It is the same wrong by a different amount, and it hides how much the bucket width matters.
- **Reading the `+Inf` bucket as a number.** A p99 in that bucket is the top finite bound, and it is a lower bound on the truth.
- **Averaging p99s across replicas.** The mean of 0.04975 s and 2.485 s is 1.267 s.
- **An alert on one sample.** It pages on the spike at 100 s and on every flap. The check expects 23 pages without a pending window and 1 with it.
- **No keep-firing window.** A 3 s dip splits the storm into two pages. The check expects (230, 335).
- **Liveness tied to readiness.** The kubelet kills the pod during the load. The probe check expects a pod that is live and not ready while loading.
- **Reporting ready before warmup.** The first requests pay for graph capture.
- **Adding the read and the copy for a streaming loader.** That is the mmap time. The check expects 26.915625 s for streaming and 28.58125 s for mmap at 2 GB/s.

> [!RECAP]
> - A request's queue time, TTFT, and ITL come from timestamps in the event log. A preemption appears as one long ITL gap, not in TTFT.
> - A histogram stores cumulative counts per `le` bound, ending in `+Inf`. Histograms with the same bounds add.
> - A quantile from buckets interpolates linearly inside one bucket, so its error is at most that bucket's width. A quantile in `+Inf` returns the top finite bound. Never average percentiles.
> - Liveness means the process answers, and readiness means it can serve. A startup probe covers a slow load.
> - An alert needs a pending window to ignore a spike and a keep-firing window to survive a dip.
> - Time to ready is pull plus load plus warmup, 52.9 s on the chapter's assumed rates. Streaming saves at most the smaller of the read and the copy, and a warm pool of $\lceil \lambda\, t_{\text{ready}} \rceil$ covers a burst.

## Check your understanding

> [!QUESTION] Why does a preemption raise p99 ITL and leave TTFT alone?
> A preempted request has already emitted its first token. Its stall sits between two later tokens, in one ITL gap. Its queue time ends at the first `scheduled` event, and a later one doesn't count.

> [!QUESTION] A histogram has 100 samples in a bucket from 0.2 s to 0.3 s. What is the largest error its p95 can have?
> The bucket's width, 0.1 s. The estimate is 0.295 s, and the true value can be anywhere from 0.2 s to 0.3 s, so the error ranges from 0 to 0.095 s here, and it is bounded by 0.1 s for any quantile in the bucket.

> [!QUESTION] Why is the average of two replicas' p99 values wrong, and what should you do?
> The average weights a replica with 10 requests the same as one with 1,000. Add their bucket counts and take the quantile of the sum. Here the merged p99 is 0.0500 s and the average is 1.267 s.

> [!QUESTION] A pod's liveness probe fails while it loads weights for 27 s. What happens, and what is the fix?
> After `failureThreshold` failures, the kubelet kills the container and restarts it, and the load starts over. Make liveness true once the process answers. Use readiness for "weights loaded and warm," and a startup probe whose window of `failureThreshold` times `periodSeconds` covers the load and warmup.

> [!QUESTION] Why does a streaming loader save at most 1.666 s at every source rate in the table?
> Streaming overlaps the read and the copy, so it hides the shorter one. The copy is 1.681 s at 32 GB/s, and it is shorter than the read at every source rate up to 32 GB/s. The saving is the copy less one chunk's worth of fill, 1.681 - 0.016 = 1.666 s.

## Lab

> [!TRY] Build the aggregator, the alert, and the cold-start model
> Implement `parse_events`, `request_metrics`, `build_histogram`, `histogram_quantile`, `merge_histograms`, `counter_rate`, `evaluate_alert`, `probe_states`, `min_failure_threshold`, `weight_load_seconds`, `time_to_ready`, and `warm_pool_size`. You pass when the hand-made log gives the exact queue, TTFT, and ITL values, the cumulative counts and the interpolated quantile match the hand numbers, the planted storm gives one alert at (230, 335), and the cold start totals 52.915625 s.

The starter gives you the seeded event log and gauge series, `latency_samples`, `exact_quantile`, and `rate_series`. Checks run from the smallest piece to the whole: the parser, the per-request metrics, the cumulative histogram, the quantile, a merge of two replicas, a counter's rate, the alert state machine on a hand-made series, the planted storm, the probes, and the cold start. Every message shows the expected and actual value.

A histogram whose buckets aren't cumulative fails the histogram check. A quantile that returns the bucket's upper bound fails the quantile check, with 0.3 s where it expects 0.295 s. An alert that ignores its pending window pages on the spike, and an alert with no keep-firing window pages twice for the storm. A readiness probe that goes true before warmup fails the probe check. A streaming loader that adds the read and the copy fails the cold-start check.

`scripts/figures/17c-observability-and-cold-start.py` prints every number in this chapter from the lab solution.

## Further reading

- [vLLM metrics design](https://docs.vllm.ai/en/latest/design/metrics/). The metric names, their types, and the open questions about bucket boundaries.
- [vLLM production metrics](https://docs.vllm.ai/en/latest/usage/metrics/). The `/metrics` endpoint and the list of exposed metrics, including `vllm:num_preemptions`.
- [Prometheus: histograms and summaries](https://prometheus.io/docs/practices/histograms/). Cumulative `le` buckets, and why the error of a quantile is limited by a bucket's width.
- [Prometheus: query functions, `histogram_quantile`](https://prometheus.io/docs/prometheus/latest/querying/functions/). Linear interpolation, the `+Inf` rule, and aggregating with `sum by (le)`.
- [Prometheus: alerting rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/). `for`, `keep_firing_for`, and the pending and firing states.
- [Kubernetes: liveness, readiness, and startup probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/). What each probe does on failure.
- [Google SRE book: monitoring distributed systems](https://sre.google/sre-book/monitoring-distributed-systems/). The four golden signals, distributions over averages, and alerts on symptoms.
- [CoreWeave Tensorizer](https://github.com/coreweave/tensorizer). Serialized tensors that stream from HTTP, S3, or disk.
- [Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer). Concurrent reads of safetensors into CPU buffers while tensors move to the GPU.
- [The safetensors repository](https://github.com/huggingface/safetensors). The format [chapter 1](/c/01-the-model-on-disk) parses.
