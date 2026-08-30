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

## Why fusion pays

Chapter 10 established that most non-GEMM operations are memory bound. For those,
time is proportional to bytes moved, and fusion moves fewer bytes.

Take RMSNorm applied to a residual sum. Unfused:

1. Read `x`, read `residual`, write `x + residual`.
2. Read `x + residual`, write the normalized result.

That's three reads and two writes — five trips over a `tokens x 5120` tensor.
Fused:

1. Read `x`, read `residual`, write the sum, write the normalized result.

Two reads and two writes. The sum has to be written because the next layer's
residual connection needs it, but it's never read back. Four trips instead of
five, for a 20% saving.

The kernel keeps the row in registers between the two uses:

```python
total = x + res
tl.store(new_res_ptr + offset + cols, total, mask=mask)

var = tl.sum(total * total, axis=0) / n_cols
scale = 1.0 / tl.sqrt(var + eps)
tl.store(out_ptr + offset + cols, total * scale * w, mask=mask)
```

SwiGLU's activation is a bigger win. Unfused it reads `gate`, writes `silu(gate)`,
reads it back, reads `up`, and writes the product: five trips over a
`tokens x 17408` tensor, which at 4096 tokens is 143 MB each. Fused, it's three.

## One block per row

Both kernels assign one block to one row and load the whole row at once. That caps
the row width at what fits in a block's registers — with `BLOCK = 8192` and
float32 accumulation you're near the limit, and 5120 fits comfortably.

The `num_warps` parameter tunes how many warps cooperate on a block. More warps
give more parallelism per row but less occupancy across rows. A reasonable
heuristic:

```python
num_warps = max(1, min(16, block // 256))
```

Triton also has `@triton.autotune`, which benchmarks a list of configurations on
the first call and caches the winner. It's worth using for kernels whose shapes
vary; for a fixed hidden size the heuristic is enough.

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
