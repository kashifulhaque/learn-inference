---
title: Fusion in Triton
slug: 13-fusion-in-triton
part: "Part 4 — Kernels"
summary: Writing kernels at block granularity, fusing away the memory traffic between operations, and finding out why the byte count over-promises inside a 40 MB cache.
minutes: 130
gpu: true
objectives:
  - Write a Triton kernel and explain how its block-level model differs from CUDA's thread-level one.
  - Explain masks, tl.constexpr, and what autotuning does.
  - Derive the byte saving a fusion predicts, and the speedup that implies.
  - Explain why the prediction fails inside a 40 MB L2, and where the crossover is.
  - Explain why the SwiGLU fusion pays 1.52x where the RMSNorm fusion pays 1.10x.
  - Decide when fusion is worth it and when it is not.
lab: 13-fused-rmsnorm
---

# Fusion in Triton

> [!TLDR]
> - Triton has you write what one *block* does. The compiler handles lanes,
>   shared memory, and barriers, and you handle masks.
> - Fusing a memory-bound pair removes the intermediate's trips to memory. For
>   the RMSNorm residual fusion, the byte count predicts 1.25x.
> - The A100 delivers 0.92x at 4096 rows and 1.10x at 16384 rows, because a
>   40 MB L2 already absorbs the round trip that fusion removes.
> - SwiGLU pays 1.52x because its intermediate is dead: fusion removes both of
>   its trips, and its tensors are far past L2.
> - Count the bytes that reach HBM, not the bytes a kernel touches, and measure
>   across sizes.

[Chapter 12](/c/12-your-first-cuda-kernel) established the ceiling: a memory-bound
kernel that reaches the bus's rate is finished, and the only way past it is to
move fewer bytes. This chapter does that.

It also contains the course's most interesting measurement. The obvious byte
count predicts a 1.25x speedup, and the hardware delivers 1.10x on one shape and
0.92x on another.

That disagreement isn't an experimental error. It's the roofline model's
assumption, that every byte a kernel touches comes from HBM, failing on a chip
with 40 MB of last-level cache. Working out exactly where it fails is worth more
than the fusion is.

## Before you start

**Chapter 12's execution model.** Grid, block, warp, SM, coalescing, the memory
hierarchy, and the fact that a warp's request becomes 32-byte sectors. Triton
hides the thread level but not the hardware, so all of it still applies.

**Chapter 10's roofline.** Arithmetic intensity, the ridge point at 161 FLOPs per
byte against rated bandwidth, and the measured copy ceiling of 1275 GB/s.

**Chapter 4's RMSNorm.** The operation, and why the sum of squares accumulates in
float32 whatever the input dtype is.

**The A100's cache.** 40 MB of L2, shared by all 108 SMs, write-back. That number
is the pivot of the whole chapter.

**Python decorators and keyword-only arguments.** That's how you declare and
launch a Triton kernel.

## The Triton programming model

The two languages differ in the unit you write. CUDA makes you describe what one
thread does. The block exists only in your index arithmetic, and you're
responsible for the mapping, the vectorization, the shared-memory staging, and
the barriers.

Triton makes you describe what ==one *block* does==. You write code that operates
on vectors of a compile-time size, and the compiler assigns lanes, picks vector
widths, allocates registers, stages through shared memory, and inserts barriers.

The following is a complete softmax kernel, the one in
`engine/kernels/softmax_triton.py`:

```python
@triton.jit
def _softmax_fwd(x_ptr, out_ptr, stride_row, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)                          # (BLOCK,) int32
    mask = cols < n_cols                                # (BLOCK,) bool

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=float("-inf"))
    x = x.to(tl.float32)                                # (BLOCK,) float32

    x = x - tl.max(x, axis=0)                           # scalar max, broadcast
    numerator = tl.exp(x)
    out = numerator / tl.sum(numerator, axis=0)         # scalar sum, broadcast

    tl.store(out_ptr + row * stride_row + cols, out, mask=mask)
```

It's twelve lines with no `threadIdx`, no `__shared__`, and no
`__syncthreads()`, and it handles any row width. The following subsections take
it apart.

### `tl.program_id`

```python
row = tl.program_id(0)
```

A *program* is Triton's name for what CUDA calls a block. `tl.program_id(axis)`
is `blockIdx` along that axis, and there's no `threadIdx` at all. You give the
grid at launch:

```python
_softmax_fwd[(n_rows,)](x, out, x.stride(0), n_cols, BLOCK=block, num_warps=w)
```

The square brackets are the grid. `(n_rows,)` launches one program per row, so
`tl.program_id(0)` runs from 0 to `n_rows - 1`.

The grid can also be a callable of the kernel's compile-time parameters. That's
how a kernel whose block size is being autotuned sizes its own grid:

```python
grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
_swiglu_fwd[grid](gate, up, out, n, BLOCK=1024, num_warps=4)
```

### `tl.arange` and vector pointers

```python
cols = tl.arange(0, BLOCK)
```

This is a vector of `BLOCK` int32 values, `[0, 1, ..., BLOCK-1]`, and it's the
central object in every Triton kernel. `BLOCK` must be a power of two and must be
known at compile time.

Adding a vector to a pointer gives a vector of pointers, so
`x_ptr + row * stride_row + cols` addresses the whole row at once. That's why you
pass strides in explicitly: the kernel does its own address arithmetic, and
`x.stride(0)` is the number of elements between the start of one row and the
next.

Every value inside a Triton kernel is either a scalar or a vector of length
`BLOCK`. Reductions like `tl.max` and `tl.sum` turn the second into the first,
and arithmetic between them broadcasts.

### Masks, and why every load needs one

```python
mask = cols < n_cols
x = tl.load(ptr + cols, mask=mask, other=float("-inf"))
```

`BLOCK` is a power of two, and row widths aren't. This model's hidden size is
5120, so `triton.next_power_of_2(5120)` gives `BLOCK = 8192`, and the last 3072
lanes address memory that belongs to the next row, or past the end of the tensor
entirely.

The mask prevents that. Masked-off lanes of a `tl.load` don't access memory at
all, and take the value of `other`. Masked-off lanes of a `tl.store` don't write.

> [!WARNING] A missing mask never faults
> Without a mask on the load, you read whatever is next in memory, which is
> usually the next row and never an error, so the kernel is silently wrong.
> Without a mask on the store, you *corrupt* the next row. Neither faults,
> neither raises, and both survive a test whose row width happens to be a power
> of two. Mask every load and every store unless you can prove the whole block
> is in bounds.

The choice of `other` matters too, and it depends on the reduction. ==Pick
`other` to be the identity element of whatever reduction follows==:

| Kernel | Reduction | `other` | If you swap them |
|---|---|---|---|
| Softmax | Maximum | `-inf`, which never wins | The softmax returns `NaN` |
| RMSNorm | Sum of squares | `0.0`, which adds nothing | The variance is infinite |

### `tl.load` and `tl.store`

```python
x = tl.load(ptr_vector, mask=mask, other=0.0)
tl.store(ptr_vector, value, mask=mask)
```

These are the only memory operations. There's no shared-memory declaration and
no explicit staging. If the compiler decides a value belongs in shared memory
rather than registers, it puts it there.

`tl.load` returns the pointee's dtype. The idiom everywhere in `engine/` is to
cast immediately:

```python
x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
```

On the way out, cast back to the output pointer's own element type, so the
kernel works for float32 and bfloat16 without being written twice:

```python
tl.store(out_ptr + cols, value.to(out_ptr.dtype.element_ty), mask=mask)
```

### `BLOCK` as `tl.constexpr`

```python
def _softmax_fwd(..., BLOCK: tl.constexpr):
```

`tl.constexpr` marks a parameter as known at compile time. Triton compiles a
separate kernel for every distinct value it sees, caching by the tuple of
constexpr arguments and a few properties of the runtime ones. Inside that
compilation the compiler knows `BLOCK` exactly, so it can unroll loops, choose
vector widths, size the register allocation, and lay out the reduction tree.

The cost is that a new `BLOCK` value triggers a compile, which takes a fraction
of a second. A kernel called with a hundred different row widths compiles a
hundred times. That's why the engine uses `triton.next_power_of_2` rather than
the exact width: it collapses the whole range 4097 to 8192 onto one compilation.

### `num_warps`

`num_warps` isn't a `tl.constexpr` parameter you declare. It's a launch argument
the compiler consumes, and it says how many warps the block's work is spread
across. Divide the block by the number of lanes to get each lane's share:

$$
\text{elements per lane} = \frac{\texttt{BLOCK}}{32 \times \texttt{num\_warps}}.
$$

With `BLOCK = 8192`, the share works out as follows:

| `num_warps` | Lanes | Elements per lane |
|---|---|---|
| 8 | 256 | 32 |
| 16 | 512 | 16 |
| 32 | 1024 | 8 |

More warps means less register pressure per lane and more parallelism inside one
row, at the cost of fewer rows resident on an SM at once. The measurement in
the "Tuning `num_warps`" section later in this chapter settles it for 5120-wide
rows.

### Autotuning

You don't have to guess. `@triton.autotune` takes a list of configurations and a
`key` of runtime arguments:

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=8),
        triton.Config({"BLOCK": 4096}, num_warps=8),
    ],
    key=["n_cols"],
)
@triton.jit
def _kernel(..., BLOCK: tl.constexpr): ...
```

On the first call with a new value of `n_cols`, Triton benchmarks every
configuration and caches the winner for that key. Later calls with the same
`n_cols` use it directly.

The trade is startup cost against generality. `engine/kernels/rmsnorm_triton.py`
doesn't autotune. It hardcodes `num_warps=8` from a measurement, because the
engine runs one hidden size, and paying a benchmarking pass on the first token of
every server start is worse than measuring once. ==Autotune when the shapes
vary==; measure and hardcode when they don't.

### What Triton gives up

The chapter's argument depends on Triton not being magic, so the following table
states the comparison plainly:

| | CUDA | Triton |
|---|---|---|
| Unit you write | One thread | One block |
| Tail handling | `if (i < n)` | `mask=` on load and store |
| Reductions | Shared memory, tree, barriers | `tl.sum`, `tl.max` |
| Shared memory | You declare and index it | Compiler decides |
| Barriers | You place `__syncthreads()` | Compiler inserts them |
| Register allocation | `ptxas` | `ptxas`, but you cannot hint it |
| Warp-level primitives | Shuffles, ballots, `__syncwarp` | Not exposed |
| Inspecting output | `cuobjdump -sass` | `kernel.asm["ptx"]`, `kernel.asm["sass"]` |

The RMSNorm in chapter 12's lab is about 35 lines of CUDA with a hand-written
tree reduction, a shared-memory allocation, and a power-of-two block size
assumption. The Triton version in this chapter is twelve lines, handles any
width, and works for float32 and bfloat16 without a second code path. For the
kernels an inference engine needs, that trade is almost always worth taking.

What you lose is the last 10% and the ability to diagnose it. When a Triton
kernel is slower than it ought to be, your levers are `BLOCK`, `num_warps`,
`num_stages`, and restructuring the algorithm. You can't fix the register
allocation by hand.

## Why fusion pays

[Chapter 10](/c/10-roofline) established that most non-GEMM operations are
memory bound. For those, runtime is proportional to bytes moved, so the way to
make them faster is to move fewer bytes. ==The easiest bytes to remove== are the
ones a kernel writes only so that the next kernel can read them back.

Take RMSNorm applied to a residual sum, which each of this model's 64 layers
does twice. Write $N$ for the number of rows, $H = 5120$ for the hidden size, and
$b = 2$ bytes for bfloat16. One activation tensor is then this many bytes:

$$
S = N H b
$$

At $N = 4096$, that's $4096 \times 5120 \times 2 = 41.9$ MB.

Unfused, PyTorch runs two kernels:

| Kernel | Reads | Writes | Traffic |
|---|---|---|---|
| `torch.add(x, residual, out=total)` | `x`, `residual` | `total` | $3S$ |
| `rms_norm(total, w, out=out)` | `total` | `out` | $2S$ |
| | | | $\mathbf{5S}$ |

Fused, one kernel does the same work:

| Reads | Writes | Traffic |
|---|---|---|
| `x`, `residual` | `new_residual`, `out` | $\mathbf{4S}$ |

The sum still has to be written, because the next layer's residual connection
needs it. What the fusion removes is the *read back*: the kernel writes `total`
once, and then `total` stays in registers for the normalization instead of making
a second trip.

The weight vector is $H b = 10.2$ KB, read by every program. It's three orders of
magnitude below $S$ and lives in L2 after the first few rows, so drop it from the
count.

Divide the unfused traffic by the fused traffic to get the prediction:

$$
\boxed{\frac{5S}{4S} = 1.25}
$$

That's a 20% saving in bytes, and a predicted speedup of 1.25x at every size.

## The fused kernel, line by line

The following code is `_rms_norm_residual_fwd` from
`engine/kernels/rmsnorm_triton.py`, with the shape of every value. `BLOCK = 8192`
and `n_cols = 5120`.

```python
@triton.jit
def _rms_norm_residual_fwd(x_ptr, res_ptr, w_ptr, out_ptr, new_res_ptr,
                           stride_row, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)                  # scalar, 0 .. n_rows-1
    offset = row * stride_row               # scalar, elements to this row's start
    cols = tl.arange(0, BLOCK)              # (8192,) int32
    mask = cols < n_cols                    # (8192,) bool: 5120 true, 3072 false

    x = tl.load(x_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(res_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    total = x + res                         # (8192,) float32, in registers

    tl.store(new_res_ptr + offset + cols,
             total.to(new_res_ptr.dtype.element_ty), mask=mask)

    var = tl.sum(total * total, axis=0) / n_cols    # scalar float32
    scale = 1.0 / tl.sqrt(var + eps)                # scalar float32
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offset + cols,
             (total * scale * w).to(out_ptr.dtype.element_ty), mask=mask)
```

The kernel breaks down as follows:

- **`row` and `offset`.** The grid is `(n_rows,)`, so each program owns exactly
  one row. `stride_row` is `x.stride(0)`, which for a contiguous 2-D tensor is
  `n_cols`. The wrapper calls `.contiguous()` before launching, so that identity
  holds.
- **`cols` and `mask`.** 8192 lanes, of which 5120 are live. The 3072 dead lanes
  exist because `BLOCK` must be a power of two, and they cost register space and
  issue slots but no memory traffic.
- **The two loads.** `other=0.0` is the identity for the sum of squares that
  follows. The `.to(tl.float32)` is chapter 4's point: summing 5120 squared
  bfloat16 values in bfloat16 loses the tail of the reduction, because a running
  sum in an 8-bit significand can't represent its next addend once it's grown
  past about 256 times that addend. The cast costs no bandwidth: the data crosses
  the bus in bfloat16 either way, and float32 registers are free of traffic.
- **`total = x + res`.** A vector of 8192 float32 values in the block's
  registers. ==This line is the fusion==: everything after it uses `total`
  without touching memory again.
- **The first store.** The updated residual, written in the input's dtype. This
  kernel never reads it back.
- **`var`.** `tl.sum` reduces the vector to a scalar, and the compiler generates
  the warp shuffles and shared-memory staging that chapter 12 made you write by
  hand. The masked lanes contribute exactly zero because `other=0.0`, so the sum
  is over exactly `n_cols` terms. That's why the divisor is `n_cols` and not
  `BLOCK`.
- **`w`.** Loaded with `cols` but not `offset`, because the weight is one vector
  shared by every row.
- **The second store.** `out_ptr.dtype.element_ty` makes the kernel
  dtype-agnostic: bfloat16 in, bfloat16 out; float32 in, float32 out. The lab
  checks exactly this.

> [!WARNING] Dividing by `BLOCK` gives a 26.5% error
> Dividing the sum of squares by `BLOCK` instead of `n_cols` scales the variance
> by $5120/8192 = 0.625$ and the output by $1/\sqrt{0.625} = 1.265$. No amount of
> staring at the formula reveals it.

### The wrapper

The wrapper flattens the leading dimensions, launches one program per row, and
reshapes back:

```python
def rms_norm_residual(x, residual, weight, eps=1e-6, out=None, new_residual=None):
    shape = x.shape                                   # (..., 5120)
    x2 = x.reshape(-1, shape[-1]).contiguous()        # (N, 5120)
    res2 = residual.reshape(-1, shape[-1]).contiguous()
    n_cols = x2.shape[-1]
    if n_cols > MAX_SINGLE_BLOCK:
        raise ValueError(...)
    out = torch.empty_like(x2) if out is None else out.reshape(-1, n_cols)
    new_res = torch.empty_like(x2) if new_residual is None else new_residual.reshape(-1, n_cols)
    block, num_warps, _ = _config(n_cols)             # (8192, 8, False)
    _rms_norm_residual_fwd[(x2.shape[0],)](
        x2, res2, weight, out, new_res, x2.stride(0), n_cols, eps,
        BLOCK=block, num_warps=num_warps)
    return out.reshape(shape), new_res.reshape(shape)
```

The `out` and `new_residual` arguments matter for a reason that the "Preallocate
the output" section gives.

## The measurement

On an A100 80GB PCIe
with 5120-wide rows in bfloat16, reusing output buffers so the allocator isn't in
the way, the lab reports two points:

| Rows | Tensor size | Against L2 | Speedup |
|---|---|---|---|
| 4096 | 41.9 MB | about 1x | **0.92x** |
| 16384 | 167.8 MB | 4x | **1.10x** |

==Fusion *loses* at the smaller size.== A finer sweep across sizes, from the same
card, shows the shape of it:

| Rows | Tensor size | Fused | Unfused | Speedup |
|---|---|---|---|---|
| 256 | 2.6 MB | 0.072 ms | 0.074 ms | 1.02x |
| 1024 | 10.5 MB | 0.081 ms | 0.074 ms | 0.91x |
| 2048 | 21.0 MB | 0.107 ms | 0.090 ms | 0.84x |
| 4096 | 41.9 MB | 0.155 ms | 0.148 ms | 0.95x |
| 8192 | 83.9 MB | 0.252 ms | 0.263 ms | 1.05x |
| 16384 | 167.8 MB | 0.444 ms | 0.495 ms | 1.12x |
| 32768 | 335.5 MB | 0.823 ms | 0.958 ms | 1.16x |

The trend isn't noise: the speedup rises monotonically from 2048 rows onward and
crosses 1.0 between 4096 and 8192.

> [!NOTE] Why the two runs disagree slightly
> The runs differ at the few-percent level: 0.95x against 0.92x at 4096 rows, and
> 1.12x against 1.10x at 16384. Chapter 10 explains why: the provider hands you
> whichever 80GB A100 is free, and the SXM4 module and the PCIe card don't have
> the same bandwidth. Treat differences under 5% as noise.

The byte count predicted 1.25x at every size. Something is missing.

## What the byte count misses: L2

The prediction fails because of the cache. The A100 has 40 MB of L2, shared by
all 108 SMs and write-back. When the unfused pair writes its intermediate and
the next kernel immediately reads it back, that read hits L2 as long as the
intermediate is still resident. The traffic the fusion was supposed to save was
never going to HBM in the first place.

> [!KEY] Count the bytes that reach HBM
> The fusion's benefit isn't the bytes it stops touching. It's the bytes it stops
> sending to HBM, and those are two different quantities whenever the working set
> is near the size of the cache.

### Put a number on it

Two parameters explain every measurement in the sweep:

- **The HBM fraction $\hla{\varphi} \in [0, 1]$** is the fraction of the
  intermediate that genuinely round-trips through HBM in the unfused path.
  $\hla{\varphi} = 0$ means the cache absorbed all of it, and
  $\hla{\varphi} = 1$ means none of it.
- **The efficiency $\hlb{e}$** is the fused kernel's efficiency per byte relative
  to the pair it replaces. The fused kernel isn't the same kernel: it does two
  stores from one pass, holds a whole 5120-wide row in float32 registers, and so
  fits fewer rows on an SM than the trivial elementwise `add` it replaces.

The unfused path's HBM traffic is $4S + \hla{\varphi} S$ rather than $5S$. Scale
the traffic ratio by the efficiency:

$$
\text{speedup} = \hlb{e} \cdot \frac{4 + \hla{\varphi}}{4} .
$$

> [!INTUITION]
> The fraction on the right is what the cache leaves for fusion to save: 1.00
> when L2 absorbs the round trip, 1.25 when none of it does. The factor
> $\hlb{e}$ is the tax the fused kernel pays for being a heavier kernel.

Now invert the measurements to find $\hlb{e}$ at each size.

> [!EXAMPLE] Three sizes, one efficiency
> | Rows | Intermediate | $\hla{\varphi}$ | Traffic term | Measured | $\hlb{e}$ |
> |---|---|---|---|---|---|
> | 4096 | 41.9 MB, about L2 | $\approx 0$ | 1.00 | 0.92x | 0.92 |
> | 16384 | 167.8 MB, 4x L2 | $\approx 1$ | 1.25 | 1.10x | 0.88 |
> | 32768 | 335.5 MB | $\approx 1$ | 1.25 | 1.16x | 0.93 |
>
> At 4096 rows, the unfused path's first kernel streams $3S = 126$ MB through
> the cache, and the cache absorbs essentially all of the round trip.

Three independent measurements give three values of $\hlb{e}$ between 0.88 and
0.93. The model fits with a per-byte efficiency of about 0.90 in every case,
which is a real and stable property of the fused kernel rather than a free
parameter doing the work. The honest statement of what fusion buys here is the
following:

$$
\boxed{\text{speedup} \approx 0.90 \times \frac{4 + \hla{\varphi}}{4}}
$$

That's 0.90x when the cache absorbs the intermediate and 1.13x when it doesn't.
The measured 0.92x and 1.10x sit on either side of those.

### Where the crossover is

Break-even needs the traffic term to cover the 10% efficiency loss:

$$
\frac{4 + \hla{\varphi}}{4} \ge \frac{1}{0.90} = 1.111 \implies \hla{\varphi} \ge 0.44 .
$$

Nearly half the intermediate has to reach HBM before the fusion is worth running
at all. From the sweep, that happens ==between one and two times L2==: between
41.9 MB and 83.9 MB, or between about 4000 and 8000 rows of 5120.

It's one to two times rather than exactly one because the intermediate doesn't
get the cache to itself. During the unfused pair's first kernel, L2 also holds
`x` and `residual`, and during the second, it holds `out`. A byte of the
intermediate survives from its write to its read only if less than 40 MB of
*all* traffic passes in between, and at $S = 42$ MB the first kernel alone pushes
126 MB through.

Residency is partial and position-dependent: bytes written late in the first
kernel survive, and bytes written early don't. That's why the curve in the sweep
is a smooth ramp rather than a step.

Two lessons follow, and both generalize well past this kernel:

- **Count bytes that reach HBM, not bytes the kernel touches.** A producer and a
  consumer that run back to back on a working set smaller than L2 are already
  effectively fused, by the cache. The roofline model in chapter 10 assumes every
  byte comes from HBM, which is why it over-predicts here.
- **Measure across sizes, not at one.** A single measurement at 4096 rows says
  fusion doesn't work. A single measurement at 32768 says it gives 1.16x. Only
  the sweep says what's going on.

## Why the SwiGLU fusion pays more

The same lab fuses the MLP's activation and measures **1.52x**, far better than
the RMSNorm fusion's 1.10x. There are three reasons, and the first is the big one.

**It removes two trips out of five, not one out of five.** Unfused,
`F.silu(gate) * up` is two PyTorch kernels:

| Kernel | Reads | Writes | Traffic |
|---|---|---|---|
| `silu(gate)` | `gate` | `t` | $2S$ |
| `t * up` | `t`, `up` | `out` | $3S$ |
| | | | $\mathbf{5S}$ |

Fused, the kernel reads `gate`, reads `up`, and writes `out`, which is $3S$. The
prediction is the following:

$$
\frac{5S}{3S} = 1.667 .
$$

The intermediate `t` is written *and* read purely for the benefit of the second
kernel, so both trips disappear. The RMSNorm fusion could only ever remove one,
because the next layer's residual connection genuinely needs the sum written out.

Check the efficiency factor: $1.52 / 1.667 = 0.91$, the same 0.90 that the
RMSNorm fusion showed. The model holds across two different kernels.

**Its tensors are far past L2.** The MLP's intermediate size is 17408. At 4096
tokens in bfloat16, each of `gate`, `up`, and `out` is this large:

$$
4096 \times 17408 \times 2 = 142.6 \text{ MB}.
$$

That's 3.6 times the 40 MB L2. No cache absorbs the round trip, so
$\hla{\varphi} \approx 1$ and the full $5/3$ is available.

**It's a flat elementwise kernel.** There's no reduction, so there's no
cross-lane communication and no wide row held in registers. `BLOCK = 1024` with
`num_warps = 4` gives eight elements per lane. Occupancy is high, and the kernel
is about as close to a pure stream as a Triton kernel gets:

```python
@triton.jit
def _swiglu_fwd(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)        # (1024,) int32
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    silu = gate * tl.sigmoid(gate)                     # silu(v) = v * sigmoid(v)
    tl.store(out_ptr + offsets, (silu * up).to(out_ptr.dtype.element_ty), mask=mask)
```

Compare it with the RMSNorm kernel. The grid is over *elements*, not rows, so
the kernel treats the tensor as one flat array and the leading shape is
irrelevant. `pid * BLOCK + tl.arange(0, BLOCK)` is chapter 12's global index,
written at block granularity. At 4096 by 17408 that's 71.3M elements and
$\lceil 71.3\text{M} / 1024 \rceil = 69{,}632$ programs, or 645 per SM.

> [!KEY] Fuse where the intermediate is dead
> The most valuable fusion is one whose intermediate no consumer outside the
> fused region ever reads, because then both of its trips disappear. A fusion
> that still has to publish its intermediate can only ever halve the saving.

## One block per row

Both RMSNorm kernels assign one program to one row and load the whole row at
once. That keeps the row in registers between the reduction and the scaling, and
it caps the row width at what a block's registers hold. The engine sets the cap
at `MAX_SINGLE_BLOCK = 16384` and falls back to a tiled kernel above it:

```python
def _config(n_cols):
    if n_cols <= MAX_SINGLE_BLOCK:
        return triton.next_power_of_2(n_cols), 8, False
    return TILE, 8, True          # TILE = 2048
```

The tiled kernel walks the row in 2048-element chunks to accumulate the sum of
squares, and then walks it again to scale. That reads the row twice, $3S$
instead of $2S$ for a plain RMSNorm. The following was measured at 4096 rows of
5120 columns in bfloat16:

| Kernel | Time | Effective bandwidth |
|---|---|---|
| Plain copy, the bandwidth floor | 0.066 ms | — |
| Single block, `num_warps=8` | 0.089 ms | 945 GB/s |
| Tiled, `BLOCK=2048` | 0.090 ms | 929 GB/s |

The two are ==within 1%, not the 50%== the extra read suggests. The reason is the
chapter's own argument turned around: the re-read is a 10 KB per-row working set
that the block touched moments earlier, so it's still in cache. The second
pass costs real traffic only when rows are wide enough that the blocks in flight
together exceed L2.

So `rms_norm` keeps the tiled fallback for correctness on wide rows, while
`rms_norm_residual` refuses them outright and raises. The fusion's entire value is
holding the sum in registers, and a tiled version would have to read the sum back
and give the saving away.

### Tuning `num_warps`

On 5120-wide rows, 8 warps beat both 16 and 32:

| `num_warps` | Elements per lane | Time | Effective bandwidth |
|---|---|---|---|
| 8 | 32 | 0.0888 ms | 945 GB/s |
| 16 | 16 | 0.0908 ms | 924 GB/s |
| 32 | 8 | 0.0955 ms | 878 GB/s |

More warps per row buys parallelism inside the row and costs occupancy across
rows, and for a row this narrow the trade lands early. Two effects push the same
way:

- With 32 warps, the reduction tree is two levels deeper.
- A block that claims 1024 lanes leaves room for fewer blocks per SM, so fewer
  rows are in flight to hide latency with.

Don't guess `num_warps`. Either autotune it or measure it once and hardcode the
answer, as the engine does.

## Preallocate the output

Even where fusion wins, a kernel that allocates its own output can give the
saving back. At 4096 rows, the fused kernel writes two 41.9 MB tensors, and
allocating them on every call costs about as much as the kernels do. The lab's
harness says so in a comment, and it reuses buffers for exactly that reason.

Both engine entry points take output buffers:

```python
def rms_norm(x, weight, eps=1e-6, out=None): ...
def rms_norm_residual(x, residual, weight, eps=1e-6, out=None, new_residual=None): ...
```

The decode loop hands in buffers it already owns.

> [!TIP] Before you conclude a fusion doesn't work
> When a microbenchmark reports no speedup for a fusion whose byte count clearly
> went down, check three things:
>
> 1. What it allocates, and whether the comparison is timing the allocator.
> 2. Whether the intermediate fits in L2.
> 3. Whether the kernel is spilling registers.

## Measure against what is reachable

Judge a kernel against the copy, not the rated peak. A plain
device-to-device copy on this A100 moves 42 MB in and 42 MB out in 0.066 ms,
which is 1275 GB/s. The card is rated at 1935. No kernel beats the copy, so ==the
copy is the ceiling worth comparing against==:

$$
\frac{945}{1275} = 74\%, \qquad \frac{945}{1935} = 49\% .
$$

Quoting the second number would make a good kernel look broken. The lab's own
check is against the copy ceiling, and it requires 60%.

## When not to fuse

Fusion isn't free, and it loses in the following cases:

- **Compute-bound operations don't benefit.** Fusing an activation into a large
  GEMM saves a small fraction of a kernel limited by arithmetic. Chapter 10's
  table says which side of the ridge point an operation is on, so check before
  you start.
- **The intermediate might not be dead.** If something outside the fused region
  reads it, you still have to write it, and the saving halves. That's the entire
  difference between 1.52x and 1.10x in this chapter.
- **Fused kernels are harder to check.** Every fusion is a new kernel with its
  own bugs, and it replaces two kernels that were each tested on their own.
  Always keep the unfused path and test against it.
- **Register pressure.** A kernel holding too much per lane spills to local
  memory, which lives in HBM, and the fused version becomes slower than the
  unfused one. The symptom is a fusion that is inexplicably slow and gets
  *faster* when you reduce `BLOCK`. The fix is a smaller block size or more
  warps.
- **The launch geometries might not match.** Fusing a row-wise reduction with a
  column-wise one means one of them ends up with the wrong access pattern, and
  chapter 12 priced that at up to 6.9x.
- **The working set might fit in L2.** That's this chapter's whole point. Decode
  at small batch is under L2 for everything, and there fusion buys nothing,
  though decode at small batch is bound by reading the 53.8 GB of weights anyway,
  so nothing else helps either.

## Correctness first

Every lab from here on checks
correctness before speed, and you ought to as well. A kernel that's twice as fast
and slightly wrong is worse than no kernel, because the error compounds over 64
layers and hundreds of tokens and surfaces as "the model got dumber," which is
nearly impossible to trace.

Compare in float32 against the PyTorch reference, on shapes that exercise the
tails. The lab picks four for the RMSNorm, and each one tests something:

| Shape | What it catches |
|---|---|
| `(64, 5120)` | The ordinary case |
| `(1, 5120)` | A grid of one program — an off-by-one in the grid size |
| `(4096, 5120)` | Enough rows to fill 108 SMs several times over |
| `(13, 4097)` | A width that is not a power of two, and fewer rows than SMs |

==`(13, 4097)` is the important one.== `triton.next_power_of_2(4097)` is 8192, so
4095 of the 8192 lanes are masked off, nearly half the block. A kernel with a
missing or wrong mask passes the first three shapes and fails this one.

## What goes wrong

- **The output is scaled by a constant factor, uniformly.** The reduction divided
  by `BLOCK` instead of `n_cols`. At `BLOCK = 8192` and `n_cols = 5120`, the
  factor is 1.265.
- **Rows after the first are corrupted.** A missing mask on `tl.store`. The tail
  lanes wrote into the next row.
- **The result is wrong only for widths that aren't powers of two.** A missing
  mask on `tl.load`, with the tail reading the next row's data into the
  reduction.
- **`NaN` everywhere in a softmax-shaped kernel.** `other=0.0` on a load feeding
  a maximum. Use `-inf`.
- **The error is small but larger than 1e-4 in float32.** The reduction ran in
  the input dtype. Cast to `tl.float32` on load.
- **It works in float32 and fails in bfloat16.** The store cast is hardcoded
  rather than using `out_ptr.dtype.element_ty`.
- **A fusion whose byte count dropped measures slower.** Allocation inside the
  call, or a working set inside L2, or register spilling, in that order.
- **The first call takes a second and later ones are instant.** That's the JIT
  compiling for a new set of `tl.constexpr` values. If it happens on *every*
  call, your `BLOCK` is varying, so round it with `triton.next_power_of_2`.

> [!RECAP]
> - A Triton kernel describes one block. Mask every load and store, and set
>   `other` to the identity of the reduction that follows.
> - The RMSNorm residual fusion cuts traffic from $5S$ to $4S$ (a 1.25x
>   prediction). SwiGLU cuts it from $5S$ to $3S$ (a 1.667x prediction).
> - Measured speedup is about $0.90 \times (4 + \hla{\varphi})/4$: the cache
>   decides $\hla{\varphi}$, and the heavier fused kernel costs about 10% per
>   byte.
> - The RMSNorm fusion breaks even between one and two times L2, and loses
>   inside it.
> - Preallocate outputs, divide by `n_cols` not `BLOCK`, and compare against the
>   1275 GB/s copy ceiling, not the rated 1935.

## Check your understanding

> [!QUESTION] The unfused RMSNorm pair moves 5S bytes and the fused kernel moves 4S. Why is the measured speedup ever less than 1?
> Because the 5S count assumes all five trips go to HBM. When the intermediate
> fits in L2, the trip the fusion removes was being served by cache and cost
> almost nothing, so the traffic term is 1.00 rather than 1.25. And the fused
> kernel is about 10% less efficient per byte than the two kernels it replaces,
> because it holds a whole 5120-wide row in float32 registers and fits fewer
> rows on an SM. 0.90 times 1.00 is 0.90.

> [!QUESTION] Why does the SwiGLU fusion reach 1.52x when the RMSNorm fusion reaches 1.10x?
> Its intermediate is dead. `silu(gate)` is written and read purely so the second
> kernel can consume it, so fusing removes both trips: 5S becomes 3S, a
> prediction of 1.667x. The RMSNorm fusion still has to publish the residual sum
> for the next layer, so it can only remove the read: 5S becomes 4S, a prediction
> of 1.25x. Both land at about 0.91 of their prediction.

> [!QUESTION] Why does every `tl.load` need a mask when the row width is 5120?
> Because `BLOCK` has to be a power of two, so it's 8192, and lanes 5120 through
> 8191 address the next row. Without the mask, the load succeeds, returns the
> neighbouring row's values, and feeds them into the sum of squares. Nothing
> faults and nothing raises; the numbers are wrong.

> [!QUESTION] You fuse two kernels, the byte count drops by a third, and the benchmark shows no change. What do you check first?
> Whether the benchmark is timing the allocator. A fused kernel that calls
> `torch.empty_like` allocates its output on every iteration, and at 42 MB that
> costs about as much as the kernel. Pass in a buffer, then re-measure. After
> that, check whether the working set fits in L2, and then whether the kernel is
> spilling.

## Lab

> [!TRY]
> Write three Triton kernels: `rms_norm`, `rms_norm_residual`, and `swiglu`. You
> pass when all three match their references, the fusion wins past L2 and does
> worse inside it, and `rms_norm` reaches 60% of the copy ceiling.

All three kernels take an optional `out` buffer so the harness can time them
without the allocator in the way, and `rms_norm_residual` also takes
`new_residual`.

The harness checks correctness first:

- `rms_norm` matches the PyTorch reference to 1e-4 at `(64, 5120)`, `(1, 5120)`,
  `(4096, 5120)`, and `(13, 4097)`.
- It accepts a 3-D input `(2, 8, 5120)` and returns the same shape.
- It returns bfloat16 for a bfloat16 input.
- `rms_norm_residual` returns `(normalized, x + residual)`, and both halves are
  checked.
- `swiglu` matches `silu(gate) * up` at `(1024, 17408)` in bfloat16 to 5e-2
  absolute.

Then it measures, with buffers reused:

- The fusion at 4096 rows (41.9 MB, about L2) and at 16384 rows (167.8 MB, four
  times L2). It requires the larger size to exceed 1.05x, and the smaller size to
  come in *below* the larger. The point of the exercise is the gap, not the
  speedup.
- `swiglu` against `F.silu(gate) * up` at `(4096, 17408)`, requiring better than
  1.1x.
- `rms_norm` at `(4096, 5120)` in bfloat16, requiring at least 60% of the 1275
  GB/s copy ceiling.

It reports `fusion_speedup_past_l2`, `fusion_speedup_within_l2`, `achieved_gbs`,
`vs_copy_ceiling`, `vs_rated_peak`, and `swiglu_speedup`.

## Further reading

- [Triton: an intermediate language and compiler for tiled neural network computations](https://dl.acm.org/doi/10.1145/3315508.3329973)
- [The Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [Root mean square layer normalization](https://arxiv.org/abs/1910.07467)
- [GLU variants improve transformer](https://arxiv.org/abs/2002.05202) — where SwiGLU comes from.
- [Making deep learning go brrrr from first principles](https://horace.io/brrr_intro.html) — the case for fusion, from the bandwidth side.
