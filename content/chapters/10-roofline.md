---
title: The roofline
slug: 10-roofline
part: "Part 3 — Making it fast"
summary: Deciding whether an operation is limited by arithmetic or by memory, and what to do about each.
minutes: 55
gpu: true
objectives:
  - Compute arithmetic intensity and compare it against a GPU's ridge point.
  - Predict whether prefill or decode limits a given workload.
  - Use measured bandwidth to tell a slow kernel from a memory-bound one.
lab: 10-roofline
---

# The roofline

Before optimizing anything, decide what's limiting it. The roofline model answers
that with one number per operation, and it's usually right.

## The model

A GPU has two peak rates: arithmetic and memory bandwidth. For the A100 80GB
PCIe, which is what the labs run on:

| Quantity | Value |
|---|---|
| bfloat16 tensor core throughput | 312 TFLOP/s |
| HBM2e bandwidth, rated | 1935 GB/s |
| HBM2e bandwidth, measured by a plain copy | 1275 GB/s |
| Ridge point | 161 FLOPs/byte |
| Streaming multiprocessors | 108 |
| L2 cache | 40 MB |

Check which card you have before trusting any of these. The 80GB A100 ships in
two forms: the SXM4 module is rated at 2039 GB/s and the PCIe card at 1935.
Chapter 0's lab prints the name.

The *ridge point* is the ratio. An operation that does more than 161 FLOPs per
byte it moves can, in principle, saturate the tensor cores. An operation below
that can't — it finishes its arithmetic before the next bytes arrive, and adding
compute does nothing.

The rated bandwidth is not reachable. A device-to-device copy, which is the
simplest bandwidth-bound kernel there is, measures 1275 GB/s on this card — 66%
of rated. Compare your kernels against that number, not against 1935, or you
will chase a ceiling that does not exist.

For any operation, count FLOPs and bytes moved, divide, and compare:

```python
def roofline_ms(flops, bytes_moved, spec):
    compute_ms = flops / (spec["bf16_tflops"] * 1e12) * 1e3
    memory_ms  = bytes_moved / (spec["hbm_bandwidth_gbs"] * 1e9) * 1e3
    return max(compute_ms, memory_ms)
```

The larger of the two is a lower bound on runtime. It's not a prediction — real
kernels reach 70 to 90% of it at best — but it tells you which direction to
optimize, and it tells you when to stop.

## Where the model's operations land

Take one MLP layer: hidden 5120, intermediate 17408, at *T* tokens.

The three matrices hold `3 x 5120 x 17408` parameters — 267M, or 535 MB in
bfloat16. FLOPs are `2 x T x 3 x 5120 x 17408 = 535e6 x T`.

For *T* tokens, arithmetic intensity is about:

```text
535e6 x T / (535e6 + activation bytes) ≈ T
```

So intensity is roughly *T*, the number of tokens in the batch. Compare against
161:

| Tokens in batch | Intensity | Bound by |
|---|---|---|
| 1 (decode, batch 1) | ~1 | Memory, by 160x |
| 16 | ~16 | Memory, by 10x |
| 161 | ~161 | The ridge point |
| 2048 (prefill) | ~2000 | Compute |

This one table explains most of the engine's design. Decode at batch 1 wastes
99.4% of the A100's arithmetic. Getting 161 tokens into a forward pass is what
makes the GPU work, and that's what continuous batching and chunked prefill exist
to do.

## Attention is different

The MLP reads weights that are shared across the batch. Attention reads a KV cache
that is *per sequence*, so batching doesn't raise its intensity the same way.

A decode step's attention, per sequence at context *L*:

```text
FLOPs = 2 x 2 x heads x L x head_dim   (scores, then values)
Bytes = 2 x kv_heads x L x head_dim x 2
```

With 24 query heads and 4 KV heads, intensity works out to exactly 6 — the GQA
group size. The context length cancels, so it doesn't depend on *L* at all, and
it doesn't improve with batch size either. Attention during decode is permanently memory bound, which is why the fix is
to read fewer bytes: grouped queries, paged storage, quantized caches.

## Measuring what you actually got

Convert a measured time into achieved bandwidth and compare against peak:

```python
achieved_gbs = bytes_moved / (elapsed_s * 1e9)
efficiency = achieved_gbs / 1275   # against the measured copy, not the rating
```

A memory-bound kernel at 85% of the copy ceiling is finished. Rewriting it is
wasted effort; the only remaining move is to make it touch less data, usually by
fusing it with a neighbor. A memory-bound kernel at 30% has a real problem —
uncoalesced access, too few blocks to fill the SMs, or a launch overhead that
dominates a tiny kernel.

For reference, the fused RMSNorm in chapter 13 reaches 945 GB/s on 4096 rows of
5120 columns, which is 74% of the copy ceiling and 49% of the rating. Quoting
the second number would make a good kernel look broken.

## Two things the roofline doesn't cover

**Kernel launch overhead** is 5 to 10 microseconds. A decode step runs several
hundred kernels, so 400 launches cost 2 to 4 ms — comparable to the arithmetic
itself. CUDA graphs capture the whole step and replay it as one launch, which is
why every production engine uses them for decode.

**Occupancy** is whether you have enough parallel work to fill 108 SMs. A kernel
that launches 8 blocks uses 7% of the GPU regardless of how efficient each block
is. Decode at batch 1 struggles here: there simply isn't much work.

## Lab

Compute the roofline for the MLP, for attention, and for RMSNorm, at batch sizes
from 1 to 2048. Then measure each on the GPU and report achieved bandwidth as a
fraction of peak. The harness checks your intensity calculations against the
reference and asks you to identify which operations are memory bound.

## Further reading

- [Roofline: an insightful visual performance model for multicore architectures](https://dl.acm.org/doi/10.1145/1498765.1498785)
- [Making deep learning go brrrr from first principles](https://horace.io/brrr_intro.html)
