---
title: Your first CUDA kernel
slug: 12-your-first-cuda-kernel
part: "Part 4 — Kernels"
summary: The host/device split, kernel launches, grid geometry, warps, and coalescing — derived rather than quoted, and measured on an A100.
minutes: 120
gpu: true
objectives:
  - Write and compile a CUDA kernel from Python and call it on a tensor.
  - Derive the global thread index and explain why the digits are in that order.
  - Choose a grid and block geometry and say what happens at the array tail.
  - Explain coalescing at the level of the 32-byte sectors a warp's request becomes.
  - Derive why a stride of 32 floats costs the factor it does, and check it against 6.9x.
  - Time a kernel with CUDA events and check it for errors that are asynchronous.
lab: 12-cuda-vector-add
---

# Your first CUDA kernel

> [!TLDR]
> - Every thread runs the kernel and finds its element as
>   `blockIdx.x * blockDim.x + threadIdx.x`, so adjacent threads touch adjacent
>   memory.
> - Launch $\lceil n / B \rceil$ blocks and guard with `if (i < n)`; the tail
>   costs one diverging warp.
> - Memory moves in 32-byte sectors. The lab's 6.9x strided slowdown is 4.5x
>   from wasted bytes times 1.53x from scattered requests.
> - A coalesced vector add reaches 1304 GB/s, level with the 1275 GB/s copy
>   ceiling, so it's finished.
> - CUDA errors are asynchronous and sticky. Synchronize before you read results
>   or timings.

Every chapter so far has been PyTorch calling someone else's kernels. This one
writes a kernel by hand, in C++, compiled at run time from a Python string.

You won't ship hand-written CUDA. [Chapter 13](/c/13-fusion-in-triton) moves to
Triton, which is shorter, safer, and usually within 10% of hand-tuned CUDA for
the kernels an inference engine needs. The reason to spend a chapter here is
that Triton hides the execution model, and every performance decision in the
rest of the course is a decision about that model. Once you've written a kernel
by hand, "memory bound" and "uncoalesced" stop being phrases and become
properties you can compute.

This chapter assumes you've never written CUDA. If you have, skip to
"Coalescing, sector by sector", which derives the
lab's measured result.

## Before you start

**C++, minimally.** You need pointers (`float*` is the address of the first of
several floats), `const` on a pointer argument meaning the kernel won't write
through it, and integer division truncating toward zero. Nothing else.

**GPU vocabulary.** [Chapter 0a](/c/00a-notation-and-prerequisites) introduces
streaming multiprocessor (SM), warp, thread block, HBM, L2, shared memory,
coalescing, and occupancy. This chapter uses all of them and re-derives the ones
that carry the performance argument, so you can read it either way round.

**The A100 this course targets.** Compute capability 8.0, 108 SMs, 40 MB of L2,
and 80 GB of HBM2e rated at 1935 GB/s on the PCIe card. [Chapter 10](/c/10-roofline)
measured a plain device-to-device copy at 1275 GB/s, and that number, not the
rating, is what you compare a kernel against.

**Chapter 10's conclusion.** Almost everything an inference engine does outside
the matrix multiplies is memory bound. That's why this chapter spends most of
its length on how bytes move and almost none on arithmetic.

## The host and the device

A CUDA program runs on two machines at once, and each can do things the other
can't.

- The *host* is the CPU and its memory. It runs your Python, your PyTorch, and
  any C++ you compile normally. It can allocate device memory, copy to and from
  it, and launch kernels, but it can't dereference a device pointer.
- The *device* is the GPU and its memory. It runs kernels. A kernel can't call
  into Python, can't allocate host memory, and can't print to your terminal
  except through a buffered `printf` that arrives at the next synchronization.

Three function qualifiers mark the split:

| Qualifier | Runs on | Callable from |
|---|---|---|
| `__host__` (the default) | Host | Host |
| `__global__` | Device | Host — this is a *kernel* |
| `__device__` | Device | Device |

A `torch.Tensor` on `cuda` is a device allocation with some host-side
bookkeeping around it, and `tensor.data_ptr<float>()` hands you the raw device
pointer. ==That pointer is the only thing the kernel sees==. It knows nothing
about shapes, strides, or dtypes, which is why every kernel in this chapter
takes the element count as a separate argument.

## What a kernel launch is

A launch is a queued command, not a function call. This line isn't C++:

```cuda
add_kernel<<<blocks, threads>>>(a_ptr, b_ptr, out_ptr, n);
```

`nvcc` rewrites the triple-angle-bracket syntax into a call to the CUDA runtime
that packages the arguments and pushes a launch command onto a *stream*. The
call returns to the host almost immediately, usually in 5 to 10 microseconds,
long before the GPU has started. That's where chapter 10's launch overhead comes
from.

The full form takes four parameters:

```cuda
kernel<<<grid, block, shared_bytes, stream>>>(args...);
```

- `grid`: how many thread blocks to run, as a `dim3` of `(x, y, z)`. An integer
  is shorthand for `(x, 1, 1)`.
- `block`: how many threads per block, also a `dim3`. At most 1024 threads
  total.
- `shared_bytes`: dynamic shared memory per block, in bytes. Defaults to 0.
- `stream`: which queue to enqueue on. Defaults to the current stream, which is
  what PyTorch is already using.

The launch asks for `grid.x * grid.y * grid.z` blocks, each of
`block.x * block.y * block.z` threads, all running the same function body. Every
thread gets four built-in variables that say who it is: `blockIdx`, `blockDim`,
`threadIdx`, and `gridDim`. That's the entire interface.

Two rules follow from how the hardware schedules blocks:

- **A block is assigned to one SM and stays there.** Threads in a block can
  share memory and synchronize with each other because they're co-resident.
  Threads in different blocks can't; there's no block-level barrier inside a
  kernel.
- **Blocks run in no particular order.** The scheduler hands blocks to whichever
  SMs have room. Nothing guarantees that block 0 starts before block 5000, or
  that they overlap, or that they don't.

> [!WARNING] Block order is never guaranteed
> A kernel whose correctness depends on the order blocks run in is wrong even
> when it passes. The next GPU, driver, or grid size can reorder it.

## Compile CUDA from Python

`torch.utils.cpp_extension.load_inline` compiles a string of CUDA C++ at run time
and returns an importable module:

```python
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>

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

The two functions live on different machines. `add_kernel` is `__global__` and
runs on the device, once per thread. `vector_add` is an ordinary host function:
it allocates the output, computes the geometry, launches, and returns. Python
only ever calls the second one.

### What nvcc produces

`load_inline` writes your sources into a build directory under
`~/.cache/torch_extensions`, generates a small pybind11 wrapper that exposes the
names in `functions`, runs `ninja`, and imports the resulting shared object.
Inside that, the pipeline runs in four steps:

1. `nvcc` splits the translation unit. Host code goes to the system C++
   compiler. Device code goes on.
2. Device code compiles to **PTX**, a virtual instruction set versioned by
   *virtual architecture* (`compute_80` for this A100). PTX is portable forward:
   a driver can compile it just in time for a newer GPU.
3. `ptxas` assembles PTX into **SASS**, the machine code for a *real
   architecture* (`sm_80`). SASS is what the SMs execute, and it's where
   register allocation and instruction scheduling happen.
4. Both are packed into a *fatbinary* embedded in the host object, so one binary
   can carry code for several architectures.

> [!TIP] Read the SASS once
> Run `cuobjdump -sass` on the built `.so`. It's worth doing once, to see that
> your `if (i < n)` became a predicated instruction rather than a branch.

The first call takes 30 to 60 seconds, and almost all of it is the C++ compiler
working through the PyTorch headers, not your kernel. Results are cached by a
hash of the source, so later calls are instant, and ==editing one character
costs you the full minute again==.

## The thread index

Every thread runs the same body, so the body must compute which element this
thread owns. With $G$ blocks of $B$ threads, a thread knows two coordinates:
`blockIdx.x` in $[0, G)$ and `threadIdx.x` in $[0, B)$. You need a one-to-one map
from those pairs onto $[0, GB)$.

Write the block index as $\hla{b}$ and the thread index as $\hlb{t}$, and treat
them as the digits of a mixed-radix number:

$$
\boxed{i = \hla{b} \cdot B + \hlb{t}}
$$

This map is a bijection. Given $i$, you recover $\hla{b} = \lfloor i / B \rfloor$
and $\hlb{t} = i \bmod B$ uniquely, because $0 \le \hlb{t} < B$. In code it's the
following line:

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

The other mixed-radix map, $i = \hlb{t} \cdot G + \hla{b}$, is also a bijection
and is also wrong. Both cover $[0, GB)$ exactly once; they differ in which digit
varies fastest:

- Under $i = \hla{b} \cdot B + \hlb{t}$, consecutive `threadIdx.x` gives
  consecutive $i$, so adjacent threads touch adjacent memory.
- Under $i = \hlb{t} \cdot G + \hla{b}$, adjacent threads land $G$ elements
  apart. For a grid of 65,536 blocks, that's 256 KB apart.

> [!KEY] `threadIdx.x` is the least significant digit
> That's the only reason the formula is written in that order. "Coalescing,
> sector by sector" shows what the other order costs.

For a 2-D block, the same construction nests:

```cuda
int t = threadIdx.y * blockDim.x + threadIdx.x;   // threadIdx.x still fastest
int i = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + t;
```

The rule is unchanged: `x` varies fastest at every level, because that's the
order in which the hardware packs threads into warps.

## Grid geometry and the tail

To cover $n$ elements with blocks of $B$ threads, you need
$\lceil n / B \rceil$ blocks. Integer division truncates, so write the ceiling as
follows:

```cuda
int blocks = (n + threads - 1) / threads;
```

Plain `n / threads` rounds down, which ==silently leaves the last partial block
unprocessed==. That bug passes every test whose size happens to be a multiple
of the block size.

> [!DEEPDIVE] Why `(n + B - 1) / B` is the ceiling
> Write $n = qB + r$ with $0 \le r < B$. Then
>
> $$
> \left\lfloor \frac{n + B - 1}{B} \right\rfloor = q + \left\lfloor \frac{r + B - 1}{B} \right\rfloor = \begin{cases} q & r = 0 \\ q + 1 & 1 \le r < B, \end{cases}
> $$
>
> because $r + B - 1$ lies in $[B - 1, 2B - 2]$, which is below $B$ only when
> $r = 0$. Plain `n / threads` gives $q$ and drops the last $r$ elements.

Ceiling division launches more threads than there are elements, so the kernel
needs a guard:

```cuda
if (i < n) out[i] = a[i] + b[i];
```

### What the tail costs

The guard is a branch, so the question is how many warps it makes diverge. The
answer is one.

> [!EXAMPLE] The lab's awkward size, $n = 1{,}000{,}003$ with $B = 256$
> Count the blocks and the threads they launch:
>
> $$
> \text{blocks} = \left\lceil \frac{1{,}000{,}003}{256} \right\rceil = 3907, \qquad 3907 \times 256 = 1{,}000{,}192.
> $$
>
> So 189 threads launch with nothing to do. They all sit in the last block,
> which covers indices $999{,}936$ through $1{,}000{,}191$. The first
> $1{,}000{,}003 - 999{,}936 = 67$ of its lanes are live.
>
> A block of 256 threads is 8 warps of 32. In the last block:
>
> - Warps 0 and 1 (lanes 0–63) are entirely live and take the branch uniformly.
> - Warp 2 holds lanes 64–95, of which only 64, 65, and 66 are live. It's the
>   one warp in the entire launch where threads disagree about the branch.
> - Warps 3 through 7 are entirely dead and also take the branch uniformly, so
>   they cost a scheduling slot and no divergence.
>
> That's one diverging warp out of $3907 \times 8 = 31{,}256$.

The tail guard is free. Advice to avoid branches in CUDA is about branches
inside the main loop, not about this one.

### Choose the block size

Start with 256 threads per block. It's 8 warps, it divides the SM's scheduling
resources evenly, and it's small enough that several blocks fit on one SM at
once, which lets the scheduler hide memory latency by switching between them.

The grid size then follows from $n$. Run two quick checks on it:

- **Enough blocks to fill the machine.** An A100 has 108 SMs and holds several
  blocks each. A launch of fewer than about 216 blocks, two per SM, leaves SMs
  idle no matter how good the kernel is. At $n = 2^{24}$ and $B = 256$ you get
  65,536 blocks, or 607 per SM; at $n = 2^{20}$ you get 4096 blocks, or 38 per
  SM. Both are ample.
- **Not so many threads that each does nothing.** With one element per thread,
  each thread executes an index computation, a bounds check, two loads, an add,
  and a store. The index arithmetic is a real fraction of that. For very large
  $n$, amortize it.

## The grid-stride loop

The alternative to one thread per element is to size the grid to the *machine*
and let each thread walk the array:

```cuda
__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int stride = blockDim.x * gridDim.x;       // total threads in the grid
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}
```

The pattern exists for five reasons, and they compound:

- **The grid size stops depending on $n$.** Launch 216 blocks, or however many
  fill the GPU, and the same kernel handles any array. You tune the geometry
  once instead of per call.
- **The tail guard disappears.** The loop condition is the guard. There's no
  separate `if`.
- **Index setup is amortized.** Each thread computes its starting index once and
  then adds a constant stride, so the per-element cost is one add.
- **Coalescing is preserved.** The stride is the size of the whole grid, so
  within any single iteration the threads of a warp are still on 32 consecutive
  elements. That's why the stride is `blockDim.x * gridDim.x` and not something
  smaller.
- **It's debuggable.** Launch it with `<<<1, 1>>>` and it becomes a plain serial
  loop over the array. If the serial version is wrong, the bug is in your
  arithmetic and not in your indexing.

The one-element-per-thread form in the lab is simpler to read and is what the
harness expects. Use the grid-stride form when you're writing something you'll
keep.

## Warps

The block is a programming abstraction; the *warp* is the hardware. An SM
schedules threads in groups of 32, in order, so threads 0–31 of a block are
warp 0, threads 32–63 are warp 1, and so on. ==All 32 lanes of a warp issue the
same instruction at the same time==, and that single fact produces two rules:
divergence and coalescing.

### Divergence

When lanes of a warp disagree about a branch, the hardware runs both sides and
masks off the lanes that shouldn't be executing. A two-way branch inside a warp
costs the sum of both paths, not the maximum.

Divergence is measured per warp, not per block:

| Branch condition | Warps that diverge | Cost |
|---|---|---|
| `i < n` (the tail guard) | 1 of 31,256 | Nothing |
| `i % 2 == 0` | Every warp | Doubles the kernel |
| `blockIdx.x % 2 == 0` | None | Nothing: every lane of a warp shares a `blockIdx` |

> [!NOTE] Lanes aren't in lockstep since Volta
> Since Volta, lanes of a warp have independent program counters and can be at
> different instructions. You can't assume implicit warp-lockstep
> synchronization. If you need the lanes of a warp in step, say so with
> `__syncwarp()`, or use the shuffle intrinsics, which carry their own mask.

## Coalescing, sector by sector

This section answers how many bytes a warp's load really moves. When a warp
issues a load, the memory system doesn't fetch 32 separate values. It looks at
the 32 addresses and works out which *sectors* they fall in. A sector is 32
bytes, a cache line is four of them (128 bytes), and ==the unit of traffic
between the caches and HBM is the sector==.

**Coalesced.** Thread $\hlb{t}$ reads element $i_0 + \hlb{t}$ of a `float32`
array. The warp's 32 addresses span $32 \times 4 = 128$ contiguous bytes, which
is exactly 4 sectors. Every byte fetched is used, so efficiency is 100%.

```cuda
out[i] = a[i];                 // thread i reads element i
```

**Strided by 32 floats.** Thread $\hlb{t}$ reads element $32\hlb{t}$, so the
addresses are $128\hlb{t}$ bytes apart. Because 128 is a multiple of 32, each
thread's 4 bytes land at the start of its own sector, and no two threads share
one. To deliver 128 useful bytes, the warp now needs
$32 \text{ sectors} \times 32 \text{ bytes} = 1024 \text{ bytes}$. Efficiency is
$128/1024 = 12.5\%$, an eightfold read amplification.

```cuda
out[i] = a[i * 32];            // consecutive threads land 128 bytes apart
```

The size of the unit decides the answer, and only the sector model gets it
right:

| Model of the memory system | Unit of traffic | Predicted read amplification |
|---|---|---|
| Each thread gets its own transaction | One per thread | 32x |
| The unit is the cache line | 128-byte line | 32x (32 lines for 128 useful bytes) |
| The unit is the sector | 32-byte sector | 8x |

> [!KEY] A stride of 32 floats costs 8x on the read, not 32x
> The memory system fetches whole 32-byte sectors. A scattered warp fetches 32
> of them for 128 useful bytes, where a coalesced warp fetches 4.

## Derive the lab's 6.9x

The sector model predicts 8x on the read; the lab measures 6.9x end to end. This
section closes the gap. The lab runs two kernels on $n = 2^{24}$ float32
elements and reports achieved bandwidth from the bytes each one *logically*
moves:

| Kernel | Counted bytes | Measured |
|---|---|---|
| Coalesced vector add (2 reads, 1 write) | $3n \times 4$ | 1304 GB/s |
| Strided copy (1 read, 1 write) | $2n \times 4$ | 189 GB/s |

The ratio is $1304 / 189 = 6.9$, which is the figure in the README. Three steps
account for it.

**Step 1: count the bytes that really move.** Per element, the strided copy
reads 4 useful bytes and writes 4, so it counts 8. The write is coalesced and
moves 4. The read pulls a whole 32-byte sector for its 4 bytes. So the kernel
moves the following per element:

$$
32 \ (\text{read}) + 4 \ (\text{write}) = 36 \ \text{bytes per element},
$$

an amplification of $\hlc{36 / 8 = 4.5}$.

**Step 2: predict the bandwidth.** If the memory system moved real bytes at the
same rate either way, the strided kernel's counted bandwidth would be
$1304 / 4.5 = 290$ GB/s.

**Step 3: measure the gap.** The kernel measures 189 GB/s, so the prediction is
high by $290 / 189 = \hld{1.53}$. The measured slowdown therefore splits into
two factors:

$$
\boxed{6.9 = \hlc{4.5} \times \hld{1.53}}
$$

> [!INTUITION] Wasted bytes, then wasted requests
> The factor $\hlc{4.5}$ is bytes fetched and thrown away. The factor
> $\hld{1.53}$ is the memory system running slower when the sectors are
> scattered. That second factor is about *requests*, not bytes: a coalesced warp
> issues 4 sector requests and a strided one issues 32. Each request occupies a
> slot in the SM's miss-handling structures and in the queues to L2, and having
> eight times as many in flight costs queueing delay that no amount of bandwidth
> fixes.

Coalescing is the most important performance property of any memory-bound
kernel, and chapter 10 established that almost all of them are. It's also the
easiest to get wrong by accident. Transposing a loop, indexing a 2-D array on
the wrong axis, or writing $i = \hlb{t} \cdot G + \hla{b}$ instead of
$i = \hla{b} \cdot B + \hlb{t}$ all produce ==a correct kernel that runs at a
seventh of the speed==.

> [!NOTE] The strided read fits in L2, which flatters it
> The lab's strided kernel reads `a[(i * 32) % n]`. With $n = 2^{24}$ and a
> stride of 32, that expression only produces multiples of 32, so it touches
> $2^{24}/32 = 524{,}288$ distinct elements. At one 32-byte sector each, the
> read footprint is $524{,}288 \times 32 = 16.8$ MB, comfortably inside the
> A100's 40 MB L2. After the benchmark's warm-up,
> the reads come from cache, which *helps* the strided kernel. A strided pattern
> whose footprint exceeded L2 would be worse than 6.9x. Chapter 13 makes the
> same observation the centre of its argument.

### What the coalesced number means

The coalesced 1304 GB/s means there was nothing left to optimize. Chapter 10
measured a plain device-to-device copy on this card at 1275 GB/s. A
hand-written vector add that matches a `torch.Tensor.clone` isn't a triumph of
optimization: both kernels read and write at the bus's rate, and the bus tops
out near 1300 GB/s on a card rated at 1935.

That's the shape of every memory-bound kernel you'll write. Getting to the
ceiling is a matter of not making mistakes, and going past it requires moving
fewer bytes, which is [chapter 13](/c/13-fusion-in-triton).

## The memory hierarchy

The fast storage on an A100 is small, and every kernel decides what to put in
it:

| Level | Size | Latency | Scope |
|---|---|---|---|
| Registers | 256 KB per SM | ~1 cycle | One thread |
| Shared memory | 164 KB per SM | ~30 cycles | One block |
| L2 cache | 40 MB | ~200 cycles | Whole GPU |
| HBM | 80 GB | ~400 cycles | Whole GPU |

Across the whole GPU, the register file is $108 \times 256 \text{ KB} = 27.6$ MB
and the shared memory is $108 \times 164 \text{ KB} = 17.7$ MB. Both are
*smaller* than the 40 MB L2.

You don't hide latency by making any one load faster. You hide it by keeping
enough warps resident that the scheduler always has one ready to issue while
the others wait. That's what occupancy means, and it's why 256 threads per block
with many blocks per SM is a better default than 1024 threads per block with
few.

Every optimization in the rest of this course is a variation on one theme:
==move data up this hierarchy once and use it many times==.

- FlashAttention ([chapter 14](/c/14-flash-attention)) keeps attention scores in
  registers and shared memory instead of writing a score matrix to HBM.
- Fusion (chapter 13) keeps an intermediate in registers instead of
  round-tripping it.
- Tiling loads a block of a matrix into shared memory and reuses it across a
  whole tile of output.

## Shared memory and synchronization

The lab's third kernel, an RMSNorm with one block per row, is the smallest
interesting use of shared memory. Here's the reduction, annotated:

```cuda
__global__ void rms_norm_kernel(const float* x, const float* w, float* out,
                                int cols, float eps) {
    extern __shared__ float partial[];            // size set at launch
    int row = blockIdx.x;                          // one block owns one row
    const float* x_row = x + (long long)row * cols;
    float* out_row = out + (long long)row * cols;

    // Each thread sums a strided subset, so every load stays coalesced.
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = x_row[i];
        sum += v * v;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    // Tree reduction over the 256 partial sums: 8 halving steps.
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            partial[threadIdx.x] += partial[threadIdx.x + offset];
        }
        __syncthreads();
    }

    float scale = rsqrtf(partial[0] / cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = x_row[i] * scale * w[i];
    }
}
```

At 4096 rows of 1024 columns with 256 threads, that's a grid of 4096 blocks, 4
loop iterations per thread, and $\log_2 256 = 8$ reduction steps. The dynamic
shared memory is `threads * sizeof(float)` = 1024 bytes, against the 164 KB an
SM has.

Five things in that kernel are load-bearing:

- **`extern __shared__` takes its size from the launch.** The third launch
  parameter, `rms_norm_kernel<<<rows, threads, threads * sizeof(float)>>>`, is
  where 1024 comes from. Get it wrong and threads write past the allocation into
  whatever the SM put next.
- **The accumulation loop strides by `blockDim.x`, not by a per-thread chunk.**
  Within one iteration, the 32 lanes of a warp read 32 consecutive floats: 128
  bytes, 4 sectors, fully coalesced. The obvious alternative, giving thread
  $\hlb{t}$ the contiguous columns $[4\hlb{t}, 4\hlb{t} + 4)$, spreads each load
  instruction's warp over 512 bytes and needs 16 sectors to deliver the same 128
  useful bytes. It recovers only if the neighbouring sectors survive in L1 until
  the next iteration. The striding form needs no such luck.
- **`__syncthreads()` is outside the `if`.** See the following warning.
- **The tree assumes `blockDim.x` is a power of two.** Halving from 256 reaches 1
  exactly. From 200 it would reach 100, 50, 25, 12, 6, 3, 1 and lose element 24
  at the step from 25. Either pick powers of two or pad.
- **The reduction is better conditioned than a serial sum.** The tree adds
  numbers of similar magnitude at each level, so rounding error grows like
  $\log_2 B$ rather than like $B$. The lab checks this by normalizing
  activations scaled by 100, where a poorly ordered float32 sum starts to show.

> [!WARNING] A barrier inside a branch stops the block
> `__syncthreads()` is a barrier that *every* thread in the block must reach. Put
> it inside `if (threadIdx.x < offset)` and the threads above `offset` never
> arrive, so the block stops responding or, worse, produces undefined results
> that look plausible. This is the most common shared-memory bug.

## Check your work

Kernels fail in two ways, and the second is the one that costs you an afternoon.

### Wrong answers

Compare against PyTorch before you time anything:

```python
reference = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
torch.cuda.synchronize()
assert torch.allclose(mine, reference, atol=1e-4)
```

Exercise the shapes that break indexing: a size that isn't a multiple of the
block size, a size of 1, and a size of exactly one block. The lab tests
$n = 1024$, $n = 1{,}000{,}003$, and $n = 1$ for this reason.

### Silent failures

CUDA errors are asynchronous. A kernel that reads out of bounds doesn't raise
at the launch; it raises at the next call that synchronizes, which might be
several operations later ==at a line that's perfectly fine==.

There are two error channels, and you need both:

```cuda
add_kernel<<<blocks, threads>>>(...);
// Launch-time errors: bad geometry, too much shared memory, invalid stream.
C10_CUDA_CHECK(cudaGetLastError());
// Execution-time errors: illegal address, misaligned access, assertion.
C10_CUDA_CHECK(cudaDeviceSynchronize());
```

`cudaGetLastError()` immediately after the launch catches configuration errors
before any work runs. Only a synchronizing call can report what the kernel did.

Three habits make this bearable:

- Set `CUDA_LAUNCH_BLOCKING=1` while debugging. Every launch becomes
  synchronous, so the error surfaces at the launch that caused it. It also
  destroys performance, so never leave it on for a measurement. The lab harness
  sets it.
- Call `torch.cuda.synchronize()` before reading results. The lab harness does
  this too.
- Run `compute-sanitizer python your_script.py` when you suspect an
  out-of-bounds access. It reports the offending thread and address.

> [!WARNING] Most CUDA errors are sticky
> Once a kernel faults, the context is poisoned, and every later CUDA call
> returns the same error, no matter what the call is. The first error in your
> log is the real one; everything after it is noise.

## Time a kernel

`time.perf_counter()` around a launch measures how long it took to *queue* the
work, which on an A100 is about 5 microseconds regardless of what the kernel
does. You have to synchronize.

The precise tool is a pair of CUDA events. The GPU records and timestamps them
in the stream, so they measure device time and exclude whatever the host was
doing:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for _ in range(10):        # warm up: compile, allocate, populate caches
    fn()

start.record()
for _ in range(runs):
    fn()
end.record()
torch.cuda.synchronize()   # wait for `end` to be reached before reading it

ms = start.elapsed_time(end) / runs
```

The wall-clock form, with a synchronize on each side, measures the same interval
when the GPU is running nothing else. It's what `engine/bench.py` uses:

```python
torch.cuda.synchronize()
start = time.perf_counter()
fn()
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

Either is fine. Events are better when you want to time one kernel inside a
larger stream, or when the host is doing work you don't want counted.

Whichever you use, get three things right:

- **Warm up.** The first call compiles or autotunes, allocates, and leaves the
  caches cold. `engine/bench.py` defaults to 5 warm-up iterations and 20 timed
  ones.
- **Report the median, not the mean.** One preempted run, one clock-throttle
  event, or one page fault skews a mean and doesn't move a median.
  `engine/bench.py` returns mean, median, p90, and min; quote the median.
- **Convert to bandwidth and compare against something reachable.** Divide the
  bytes the kernel logically moves by the elapsed time, and compare against the
  1275 GB/s copy, not the 1935 GB/s rating. Chapter 10 spells out why.

## What goes wrong

**Half the output is zeros or garbage.** You used plain division instead of
ceiling division for the block count, so the last partial block never launched.

**The kernel is 7x slower than expected.** The access is uncoalesced. Check
whether `threadIdx.x` is the fastest-varying part of your index.

**An error appears at an unrelated line.** CUDA reports errors asynchronously.
Set `CUDA_LAUNCH_BLOCKING=1` and rerun.

**The kernel stops responding.** A `__syncthreads()` that some threads never
reach, almost always because it's inside a divergent branch.

**It works at 1024 columns and fails at 2048.** A shared-memory array is sized
for one and indexed for the other, or a block-size assumption is baked into a
reduction.

**Results change between runs.** The kernel reads uninitialized memory, or two
threads race to write the same location without a barrier.

**It's correct but slow, and the profiler blames memory.** You're probably at
the roofline. Compute the achieved bandwidth before you rewrite anything. If
it's above 80% of 1275 GB/s, the kernel is finished, and the only remaining move
is to touch less data.

> [!RECAP]
> - The global index is $i = \hla{b} \cdot B + \hlb{t}$, with `threadIdx.x` as
>   the fastest digit so a warp reads 32 consecutive elements.
> - Launch `(n + threads - 1) / threads` blocks and guard with `if (i < n)`. The
>   tail diverges in one warp and costs nothing.
> - Memory moves in 32-byte sectors. The lab's 6.9x is $\hlc{4.5}$ from wasted
>   bytes times $\hld{1.53}$ from scattered requests.
> - A coalesced memory-bound kernel tops out near the 1275 GB/s copy ceiling.
>   Past that, the only move is fewer bytes.
> - Errors are asynchronous and sticky. Time with events or synchronize on both
>   sides, warm up, and quote the median.

## Check your understanding

> [!QUESTION] Why is the global index `blockIdx.x * blockDim.x + threadIdx.x` and not `threadIdx.x * gridDim.x + blockIdx.x`?
> Both are bijections onto $[0, GB)$, so both are correct. Only the first makes
> consecutive threads touch consecutive addresses. The second would put adjacent
> lanes $G$ elements apart, which for a 65,536-block grid is 256 KB, and every
> load would be as uncoalesced as the lab's deliberately strided kernel.

> [!QUESTION] At $n = 1{,}000{,}003$ with 256 threads per block, how many warps diverge?
> One. The launch is 3907 blocks of 8 warps, 31,256 warps in total. The last
> block has 67 live lanes, so its warps 0 and 1 are entirely live and its warps 3
> through 7 are entirely dead, all uniform. Only warp 2, holding lanes 64 through
> 95 with three live, has lanes that disagree.

> [!QUESTION] A stride of 32 floats amplifies the read eightfold, yet the measured penalty is 6.9x, not 8x. Where does the difference go?
> The 8x applies to the read only. The strided kernel's write is coalesced, so of
> the 8 bytes per element it counts, it really moves $32 + 4 = 36$, an
> amplification of 4.5, not 8. That predicts $1304 / 4.5 = 290$ GB/s. The
> measured 189 is a further 1.53x slower, which is the cost of issuing 32 sector
> requests per warp instead of 4.

> [!QUESTION] Why does the lab's RMSNorm accumulate in float32 even though the input is already float32?
> Because "float32 input" doesn't determine the accumulator. A kernel that summed
> into a `__half`, or that let the compiler contract the sum differently, would
> lose the tail of a 1024-term reduction. The lab checks this by normalizing
> activations scaled by 100, where the squares reach $10^4$ and a poorly
> conditioned sum drifts past the 1e-3 tolerance.

## Lab

> [!TRY]
> Write three CUDA kernels and return them as a source string that the harness
> compiles with `load_inline`:
>
> - `vector_add(a, b)`: elementwise sum, one thread per element, consecutive
>   threads on consecutive addresses.
> - `strided_copy(a, stride)`: `out[i] = a[(i * stride) % n]`, deliberately
>   uncoalesced so the harness can measure what scattered access costs.
> - `rms_norm(x, w, eps)`: `x` is `(rows, cols)` with `cols` at most 1024. One
>   block per row, a shared-memory reduction of the sum of squares in float32,
>   then a scale by `w`.

The harness checks that the source compiles, that `vector_add` is correct at
$n = 1024$, $n = 1{,}000{,}003$, and $n = 1$, that `strided_copy` reads the
elements it should, and that `rms_norm` matches the PyTorch reference to 1e-4.
It must still match to 1e-3 on activations scaled by 100, which is the float32
accumulation check.

It then benchmarks both memory kernels at $n = 2^{24}$ and requires the
coalesced-to-strided bandwidth ratio to exceed 3.0, and the coalesced kernel to
reach at least 60% of the 1275 GB/s copy ceiling. It reports `coalesced_gbs`,
`strided_gbs`, `coalescing_ratio`, and `vs_copy_ceiling`.

Compilation is part of the run, so expect the first attempt to take about a
minute.

## Further reading

- [CUDA C++ programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [How to access global memory efficiently in CUDA C/C++ kernels](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [CUDA pro tip: write flexible kernels with grid-stride loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)
- [NVIDIA A100 tensor core GPU architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf)
- [Nsight Compute kernel profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
