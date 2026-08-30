---
title: Fusion in Triton
slug: 13-fusion-in-triton
part: "Part 4 — Kernels"
summary: Writing kernels at block granularity, and fusing away the memory traffic between operations.
minutes: 70
gpu: true
objectives:
  - Write a Triton kernel and explain how its programming model differs from CUDA's.
  - Fuse RMSNorm with the residual add and measure the bandwidth saved.
  - Decide when fusion is worth it and when it is not.
lab: 13-fused-rmsnorm
---

# Fusion in Triton

CUDA makes you think about individual threads. Triton makes you think about
blocks: you write code that operates on tensors of a compile-time size, and the
compiler handles the thread mapping, the vectorization, and the shared-memory
staging.

For the kinds of kernels an inference engine needs, that trade is almost always
worth taking. Triton kernels are a quarter the length and usually within 10% of
hand-tuned CUDA.

## The programming model

```python
@triton.jit
def _softmax_fwd(x_ptr, out_ptr, stride_row, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=float("-inf"))
    x = x.to(tl.float32)

    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    out = numerator / tl.sum(numerator, axis=0)

    tl.store(out_ptr + row * stride_row + cols, out, mask=mask)
```

Four things to notice.

`tl.program_id(0)` is the block index, equivalent to `blockIdx.x`. There's no
`threadIdx` — you never address a single thread.

`tl.arange(0, BLOCK)` creates a vector of offsets. `BLOCK` is `tl.constexpr`, so
it's known at compile time and the compiler specializes the kernel for it.

`mask` handles the tail. Loads outside the mask return `other`, and stores outside
it don't happen. This replaces the `if (i < n)` guard.

`tl.max` and `tl.sum` are block-level reductions. Triton generates the warp
shuffles and shared-memory staging.

## Why fusion pays, and when it doesn't

Chapter 10 established that most non-GEMM operations are memory bound. For
those, time is proportional to bytes moved, and fusion moves fewer bytes.

Take RMSNorm applied to a residual sum. Unfused:

1. Read `x`, read `residual`, write `x + residual`.
2. Read `x + residual`, write the normalized result.

That's three reads and two writes — five trips over a `tokens x 5120` tensor.
Fused:

1. Read `x`, read `residual`, write the sum, write the normalized result.

Two reads and two writes. The sum still has to be written because the next
layer's residual connection needs it, but it is never read back. Four trips
instead of five, which predicts a 20% saving.

The kernel keeps the row in registers between the two uses:

```python
total = x + res
tl.store(new_res_ptr + offset + cols, total, mask=mask)

var = tl.sum(total * total, axis=0) / n_cols
scale = 1.0 / tl.sqrt(var + eps)
tl.store(out_ptr + offset + cols, total * scale * w, mask=mask)
```

Now measure it. Across tensor sizes, on an A100 80GB PCIe with 5120-wide rows:

| Rows | Tensor size | Fused | Unfused | Speedup |
|---|---|---|---|---|
| 256 | 2.6 MB | 0.072 ms | 0.074 ms | 1.02x |
| 1024 | 10.5 MB | 0.081 ms | 0.074 ms | 0.91x |
| 2048 | 21.0 MB | 0.107 ms | 0.090 ms | 0.84x |
| 4096 | 41.9 MB | 0.155 ms | 0.148 ms | 0.95x |
| 8192 | 83.9 MB | 0.252 ms | 0.263 ms | 1.05x |
| 16384 | 167.8 MB | 0.444 ms | 0.495 ms | 1.12x |
| 32768 | 335.5 MB | 0.823 ms | 0.958 ms | 1.16x |

Fusion *loses* below about 4096 rows and wins above it. The prediction was 20%
either way, so the model is missing something.

## What the byte count misses: L2

The A100 has 40 MB of L2, shared by the whole GPU. When the unfused pair writes
its intermediate and immediately reads it back, that read hits L2 as long as the
intermediate fits. The traffic the fusion was supposed to save was never going
to HBM in the first place.

At 4096 rows the intermediate is 41.9 MB — just over L2, mostly resident. At
16384 rows it is 167.8 MB, four times L2, so the read really does go to HBM and
the fusion collects the saving the byte count promised.

Two things follow, and both generalize well past this kernel.

**Count bytes that reach HBM, not bytes the kernel touches.** A producer and
consumer that run back to back on a working set smaller than L2 are already
effectively fused, by the cache. The roofline model in chapter 10 assumes every
byte comes from HBM, which is why it over-predicts the benefit here.

**Fusion still wins on the shapes that matter.** Prefill runs thousands of
tokens at a time: 4096 tokens times 5120 is 42 MB for the residual stream, and
the MLP's intermediate at 17408 wide is 143 MB. Those are past L2, which is why
the fused SwiGLU measures 1.55x while the fused RMSNorm at the same token count
measures 0.95x. Decode at small batch is under L2 for everything, and there
fusion buys nothing — but decode at small batch is bound by reading the weights
anyway.

The lesson is not that fusion is unreliable. It is that a kernel-level
prediction has to account for the cache, and the only way to know is to measure
across sizes rather than at one.

## One block per row

Both kernels assign one block to one row and load the whole row at once. That
caps the row width at what fits in a block's registers — with `BLOCK = 8192` and
float32 accumulation you're near the limit, and 5120 fits comfortably.

Keeping the row in registers is what makes the fusion pay. A tiled version that
walks the row in chunks has to read it a second time to scale it, and on a
working set larger than L2 that second read goes to HBM. Measured on 4096 rows
of 5120 columns in bfloat16, the single-block kernel runs at 945 GB/s and the
tiled one at 929 — but the tiled one moves half again as many bytes to get
there, so it takes longer in wall-clock terms for the same work.

The `num_warps` parameter tunes how many warps cooperate on a block. More warps
give more parallelism within a row and less occupancy across rows, and for rows
this narrow the trade lands early. On 5120-wide rows, 8 warps beat both 16 and
32:

| num_warps | Time | Effective bandwidth |
|---|---|---|
| 8 | 0.0888 ms | 945 GB/s |
| 16 | 0.0908 ms | 924 GB/s |
| 32 | 0.0955 ms | 878 GB/s |

Don't guess this. Triton has `@triton.autotune`, which benchmarks a list of
configurations on the first call and caches the winner; for a fixed hidden size,
measuring once and hardcoding the answer is enough.

## Preallocate the output

Even where fusion wins, a kernel that allocates its own output can give the
saving back. Two 42 MB tensors per call cost about as much as these kernels do.
The engine's `rms_norm` and `rms_norm_residual` both take an `out` argument, and
the decode loop hands in buffers it already owns.

When a microbenchmark reports no speedup for a fusion whose byte count clearly
went down, check two things before concluding the fusion does not work: what it
allocates, and whether the intermediate fits in L2.

## Measure against what is reachable

A plain device-to-device copy on this A100 moves 42 MB in and 42 MB out in
0.066 ms, which is 1275 GB/s. The card is rated at 1935. No kernel beats the
copy, so the copy is the ceiling worth comparing against — 945 GB/s is 74% of
what is reachable, and quoting it as 49% of the rating would make a good kernel
look broken.

## When not to fuse

Fusion isn't free.

**Compute-bound operations don't benefit.** Fusing an activation into a large GEMM
saves a small fraction of a kernel that's limited by arithmetic.

**Fused kernels are harder to check.** Every fusion is a new kernel with its own
bugs. Always keep the unfused path and test against it.

**Register pressure.** A kernel holding too much per thread spills to local memory,
which lives in HBM, and the fused version becomes slower than the unfused one. The
symptom is a fusion that's inexplicably slow; the diagnosis is a smaller block
size.

## Correctness first

Every lab from here on checks correctness before speed, and you should too. A
kernel that's twice as fast and slightly wrong is worse than no kernel, because
the error compounds over 64 layers and hundreds of tokens and surfaces as "the
model got dumber", which is nearly impossible to trace.

Compare in float32 against the PyTorch reference, on shapes that exercise the
tails: a row width that isn't a power of two, a batch of 1, a batch that isn't a
multiple of the block size.

## Lab

Write three Triton kernels: RMSNorm, fused RMSNorm with residual add, and fused
SwiGLU. Verify each against PyTorch to 1e-5 in float32, then measure achieved
bandwidth and report the speedup from fusion.

## Further reading

- [Triton: an intermediate language and compiler for tiled neural network computations](https://dl.acm.org/doi/10.1145/3315508.3329973)
- [The Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
