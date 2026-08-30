---
title: Your first CUDA kernel
slug: 12-your-first-cuda-kernel
part: "Part 4 — Kernels"
summary: Threads, blocks, warps, and coalescing, learned by writing C++ that PyTorch compiles at run time.
minutes: 80
gpu: true
objectives:
  - Write and compile a CUDA kernel from Python and call it on a tensor.
  - Explain the thread hierarchy and map it onto an A100's hardware.
  - Demonstrate the cost of uncoalesced memory access.
lab: 12-cuda-vector-add
---

# Your first CUDA kernel

Everything so far has been PyTorch calling someone else's kernels. This chapter
writes one by hand. You won't ship hand-written CUDA — chapter 13 moves to Triton,
which is more productive — but understanding what the hardware wants makes every
later decision obvious rather than mysterious.

## Compiling CUDA from Python

`torch.utils.cpp_extension.load_inline` compiles a string of CUDA C++ at run time
and returns a callable module:

```python
from torch.utils.cpp_extension import load_inline

source = r"""
__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(),
        out.data_ptr<float>(), n);
    return out;
}
"""

module = load_inline(
    name="vector_add",
    cpp_sources="torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);",
    cuda_sources=source,
    functions=["vector_add"],
)
```

The first call compiles and takes 30 to 60 seconds. Results are cached, so
subsequent calls are instant.

## The thread hierarchy

A kernel launches a *grid* of *blocks*, each holding up to 1024 *threads*. Threads
within a block run on one streaming multiprocessor and can share memory and
synchronize. Threads in different blocks cannot.

```text
grid  →  blocks  →  warps (32 threads)  →  threads
```

The *warp* is where the hardware actually lives. Thirty-two threads execute in
lockstep, one instruction at a time. That single fact drives two rules.

**Avoid divergence.** When threads in a warp take different branches, the hardware
runs both paths and masks off the inactive threads. A two-way branch inside a warp
costs twice the time. A branch on `i < n` at the array's tail is fine, because it
only affects the last warp.

**Coalesce.** When the 32 threads of a warp read 32 consecutive 4-byte addresses,
the memory system serves them in one 128-byte transaction. When they read
addresses 128 bytes apart, it takes 32 transactions and you get 1/32 of the
bandwidth.

```cuda
// Coalesced: thread i reads element i.
out[i] = a[i];

// Strided: consecutive threads read addresses 32 floats apart.
out[i] = a[i * 32];
```

The lab measures both. On an A100 the coalesced kernel reaches 1304 GB/s and the
strided one 189 GB/s — a factor of 6.9. A warp's 32 lanes fall into at most 32
separate transactions when strided, but consecutive warps still reuse cache
lines, so the loss is nearer 7x than the 32x the worst case suggests. It is
still the single most important performance property of any memory-bound kernel,
and as chapter 10 showed, most of them are.

That 1304 GB/s is worth noting on its own: a hand-written vector add matches a
`torch.Tensor.clone`, because there is nothing to beat. Both saturate the memory
bus, and the bus tops out around 1300 GB/s on a card rated at 1935.

## Grid sizing

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;
```

The ceiling division is the standard idiom; plain division leaves the tail
unprocessed. 256 threads per block is a good default: it's 8 warps, divides evenly
into the SM's scheduling resources, and leaves room for several blocks per SM.

An A100 has 108 SMs and can hold multiple blocks each. A kernel launching fewer
than about 216 blocks leaves SMs idle. For a tensor of 1M elements at 256 threads
you get 4096 blocks, which is plenty.

## The memory hierarchy

| Level | Size | Latency | Scope |
|---|---|---|---|
| Registers | 256 KB per SM | ~1 cycle | One thread |
| Shared memory | 164 KB per SM | ~30 cycles | One block |
| L2 cache | 40 MB | ~200 cycles | Whole GPU |
| HBM | 80 GB | ~400 cycles | Whole GPU |

Every optimization in the rest of this course is a variation on one theme: move
data up this hierarchy once and use it many times. FlashAttention keeps attention
scores in registers and shared memory instead of writing them to HBM. Fusion keeps
an intermediate in registers instead of round-tripping it. Tiling loads a block of
a matrix into shared memory and reuses it across a whole tile of output.

## Checking your work

Two kinds of error, and both bite:

**Wrong answers.** Compare against PyTorch with `torch.allclose`. Do this before
timing anything.

**Silent failures.** CUDA errors are asynchronous. A kernel that reads out of
bounds may not report until several operations later, at a line that's fine. Set
`CUDA_LAUNCH_BLOCKING=1` while debugging, and call
`torch.cuda.synchronize()` before checking results — the lab harness does both.

## Timing

`time.perf_counter()` around a kernel launch measures how long it took to *queue*
the work. CUDA is asynchronous. Synchronize first:

```python
torch.cuda.synchronize()
start = time.perf_counter()
fn()
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

Warm up first, too. The first call compiles, allocates, and populates caches.
`engine/bench.py` handles both.

## Lab

Write three kernels: a coalesced vector add, a deliberately strided version of the
same, and an RMSNorm where each block handles one row. Verify each against
PyTorch, then measure achieved bandwidth. The harness asks you to report the
coalesced-to-strided ratio and checks that your RMSNorm accumulates in float32.

Compilation is part of the run, so expect the first attempt to take about a
minute.

## Further reading

- [CUDA C++ programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [How to access global memory efficiently in CUDA C/C++ kernels](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
