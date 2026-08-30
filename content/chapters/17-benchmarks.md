---
title: Benchmarks that mean something
slug: 17-benchmarks
part: "Part 5 — Serving"
summary: TTFT, inter-token latency, throughput, and the ways each one hides a problem.
minutes: 60
gpu: true
objectives:
  - Define TTFT, ITL, and throughput, and explain what each one misses.
  - Build a load generator that models arrivals rather than sending everything at once.
  - Read a latency-throughput curve and find the knee.
lab: 17-benchmark
---

# Benchmarks that mean something

"1000 tokens per second" is not a number. Under what load, at what batch size,
with what prompt lengths, and at what latency? This chapter covers the metrics
that survive contact with a real workload.

## The metrics

**Time to first token (TTFT)** is arrival to first streamed token. It covers queue
time plus prefill, so it degrades under load in a way that pure prefill time
doesn't. For interactive use this is the number users feel first.

**Inter-token latency (ITL)** is the gap between consecutive tokens, once
generation starts. It's decode speed plus scheduling jitter. Report the
distribution, not the mean: a p99 of 200 ms with a mean of 30 ms means visible
stuttering, and the mean hides it completely.

**Throughput** is tokens per second across all requests. It's what determines cost
per token, and it trades directly against latency.

**Goodput** is throughput counting only requests that met a latency target. It's
the honest metric for a service with an SLO, and it's the one that catches a
configuration that looks great on throughput while missing every deadline.

## Load generation

Sending 100 requests at once measures a burst, not a service. Real arrivals are
spread over time, and queueing behavior — the thing that dominates TTFT under
load — only appears when you model them.

Use a Poisson process: sample inter-arrival gaps from an exponential distribution
with the target rate. Then sweep the rate and watch what happens.

Prompt and output lengths matter as much as the rate. A benchmark with fixed
512-token prompts and fixed 128-token outputs makes continuous batching look
pointless, because there's no variance for it to exploit. Sample from a realistic
distribution — log-normal is a reasonable default — or replay a trace.

## The latency-throughput curve

Sweep request rate, plot throughput against p99 TTFT, and you get a curve with
three regions.

At low rate, throughput rises with load and latency stays flat. The GPU is
underused and every request runs almost alone.

At the *knee*, throughput approaches its ceiling and latency starts climbing.
This is the operating point you want: near-maximum throughput, latency still
acceptable.

Past the knee, throughput plateaus and latency grows without bound. The queue
grows faster than it drains. Adding load here makes everything worse and improves
nothing.

Find the knee for your configuration and set admission control below it. A service
that accepts requests past the knee is a service that times out.

## Measuring on a GPU

Three ways to get a wrong number, all common.

**Not synchronizing.** CUDA is asynchronous, so `time.perf_counter()` around a
launch measures queueing. Call `torch.cuda.synchronize()` before stopping the
clock. `engine/bench.py` does this in every path.

**Not warming up.** The first call compiles kernels, allocates, and populates
caches. Discard at least five iterations.

**Reporting the mean.** GPU timings are right-skewed — occasional long tails from
allocator activity, scheduling, or clock throttling. Report the median and p90.
The mean sits between them and describes neither.

## Sustained versus burst

An A100 boosts its clocks when cool and drops them when hot. A 10-second benchmark
runs at boost clocks; a 10-minute one runs at sustained clocks, which can be 10 to
15% lower. Both numbers are real and they measure different things. Say which one
you're reporting.

## Comparing against vLLM

Comparing your engine against a production one is the honest test, and it's
sobering the first time. Match the conditions carefully: same model, same dtype,
same context length, same batch size, same sampling parameters. `enforce_eager=True`
in vLLM turns off CUDA graphs, which makes the comparison fairer if you haven't
implemented them.

Expect to be 2 to 5 times slower at first. The gap comes from CUDA graphs, a fused
GEMM stack, and a great deal of tuning. Closing it is the rest of the work, and
knowing exactly where the gap is worth more than closing it blindly.

## What to record

Per request: arrival, first token, completion, prompt tokens, generated tokens.
Everything else derives from those five numbers. `Request.metrics` returns them:

```python
{
    "prompt_tokens": 300,
    "generated_tokens": 128,
    "ttft_s": 0.214,
    "latency_s": 3.891,
    "inter_token_ms": 28.9,
}
```

Store per-request records rather than aggregates. Aggregates can be recomputed
from records; records can't be recovered from aggregates, and you always end up
wanting a percentile you didn't think to compute.

## Lab

Build a load generator with Poisson arrivals and log-normal length distributions,
run it against the engine at several request rates, and produce a
latency-throughput curve. The harness checks that your metrics are computed
correctly and asks you to identify the knee.

Your results are saved, so you can compare runs as you optimize in later chapters.

## Further reading

- [The vLLM benchmarking suite](https://github.com/vllm-project/vllm/tree/main/benchmarks)
- [DistServe: disaggregating prefill and decoding for goodput-optimized LLM serving](https://arxiv.org/abs/2401.09670)
