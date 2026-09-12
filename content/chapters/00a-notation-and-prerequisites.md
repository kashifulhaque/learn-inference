---
title: Notation and prerequisites
slug: 00a-notation-and-prerequisites
part: "Part 1 — Ground truth"
summary: The shapes, einsum strings, floating-point facts, and GPU vocabulary the rest of the course assumes.
minutes: 35
gpu: false
objectives:
  - Read any tensor shape or einsum string in this course without guessing.
  - Explain why softmax implementations subtract the maximum before exponentiating.
  - Predict when a bfloat16 running sum stops absorbing small addends.
  - Use the GPU vocabulary — SM, warp, HBM, coalescing, occupancy — precisely.
---

# Notation and prerequisites

This is a reference page, not a lesson. It collects the background the rest of
the course leans on: the shape conventions, the einsum notation, the four or
five linear-algebra facts that recur, what floating-point formats actually
store, and the GPU words that later chapters use without stopping to define.

Skim it once. Come back to it when a chapter uses a symbol you do not
recognize. Nothing here is proved at length; each item exists so that chapter 6
can write $S_t = a_t S_{t-1} + \ldots$ and chapter 14 can say "subtract the
running max" without a detour.

The course assumes you are comfortable with matrix multiplication, transposes,
eigenvalues in passing, and the idea of a derivative. It assumes nothing about
GPUs, transformers, or serving.

## Shapes and index conventions

Every tensor in this course is written as a tuple of named dimensions, in the
order they appear in memory, outermost first:

```text
(batch, heads, seq, head_dim)
```

The rightmost dimension is contiguous: consecutive `head_dim` values sit in
consecutive addresses. This matters constantly. A kernel that reads along the
last dimension reads consecutive bytes; a kernel that reads along `seq` jumps
`head_dim` elements per step.

The standard names, used consistently from here on:

- `batch` or `B` — independent sequences processed together.
- `seq`, `q_len`, `kv_len` — token positions. `q_len` is how many queries this
  call computes, `kv_len` how many keys they attend to. In decode `q_len` is 1
  and `kv_len` is the whole context.
- `heads`, `kv_heads` — attention heads. They differ in this model.
- `head_dim` — width of one head, 256 here.
- `hidden` — width of the residual stream, 5120 here.

In maths, a bracketed subscript indexes a tensor and a plain subscript names a
position. So $Q[b,h,i,c]$ is one element, and $q_t$ is the query vector at
position $t$: a vector of length `head_dim`. Vectors are columns, so $q_t^\top
k_j$ is a scalar and $k_j v_j^\top$ is a matrix.

A capital letter without indices is the whole tensor. $K$ is the key matrix of
shape `(kv_len, head_dim)`, one key per row.

When a listing shows a shape in a comment, the shape is exact, not schematic:

```python
q = q.view(batch, q_len, heads, head_dim).transpose(1, 2)
# q: (batch, heads, q_len, head_dim)
```

Shape confusion is the most common way to get stuck in this course, so the
listings say the shape after every reshape.

## Einstein summation

`torch.einsum` writes a contraction by naming the indices instead of arranging
transposes and `reshape` calls until the dimensions line up. The rule is
mechanical:

1. Each input gets a group of letters, one per dimension, in order.
2. A letter that appears in the inputs but not in the output is summed over.
3. A letter that appears in the output is kept, in the order the output lists.

That is the entire specification. Three examples, each written out as an
explicit sum.

**A matrix multiply.** `torch.einsum("td,dk->tk", x, W)` with `x` of shape
`(T, d)` and `W` of shape `(d, k)`. The letter `d` is absent from the output, so
it is summed:

$$
Y[t,k] = \sum_{c=0}^{d-1} x[t,c]\, W[c,k]
$$

**Attention scores.** `torch.einsum("bhid,bhjd->bhij", q, k)` with both inputs
of shape `(batch, heads, seq, head_dim)`. Here `b` and `h` appear in every term,
so they are carried along untouched — batch dimensions. Only `d` is summed:

$$
\mathrm{scores}[b,h,i,j] = \sum_{c=0}^{d_h-1} Q[b,h,i,c]\, K[b,h,j,c]
$$

That is a dot product between query $i$ and key $j$, computed for every pair.

**An outer product, per head.** `torch.einsum("hk,hv->hkv", k, v)` with `k` of
shape `(heads, key_dim)` and `v` of shape `(heads, value_dim)`. No letter is
missing from the output, so nothing is summed and the result is larger than
either input:

$$
\mathrm{KV}[h,i,j] = k[h,i]\, v[h,j]
$$

This one builds the linear-attention state in chapter 6. Reading the state back
is the reverse contraction, `torch.einsum("hkv,hk->hv", S, q)`:

$$
o[h,j] = \sum_{i} S[h,i,j]\, q[h,i]
$$

which is $o_h = S_h^\top q_h$ written index by index.

Two practical notes. `einsum` does not promise a fast kernel; for a plain
matrix multiply, `torch.matmul` dispatches to a tuned library and `einsum` may
not. Use `einsum` when it makes the contraction readable, and `matmul` in the
inner loop. And `einsum` silently accepts a string that contracts the wrong
axis, as long as the sizes happen to match — which is exactly the bug that
produces plausible-looking garbage.

## The linear algebra that keeps coming back

**Outer products build state.** $k v^\top$ is a matrix of rank 1: every column
is a multiple of $k$. Summing outer products, $S = \sum_j k_j v_j^\top$, packs
many key-value pairs into one fixed-size matrix. Reading it back with a query,
$S^\top q$, returns a blend of the stored values, weighted by how much $q$
resembles each $k_j$:

$$
S^\top q = \sum_j (q^\top k_j)\, v_j
$$

That identity is the whole of linear attention. Chapter 6 derives it from
softmax attention.

**$q^\top K^\top$ is a similarity vector.** If $K$ has shape
`(kv_len, head_dim)` with one key per row, then $K q$ is a vector of length
`kv_len` whose $j$-th entry is $q^\top k_j$. Written as a row vector that is
$q^\top K^\top$. Each entry measures alignment between the query and one key:
large and positive when they point the same way, negative when opposed. Softmax
then turns those numbers into weights. Attention is a similarity search
expressed as a matrix multiply.

**Matrix-vector and matrix-matrix cost the same per weight, but not per byte.**
Multiplying an $n \times m$ matrix by one vector costs $2nm$ FLOPs — one
multiply and one add per entry — and reads $nm$ matrix entries. Multiplying the
same matrix by $T$ vectors at once costs $2nmT$ FLOPs and still reads $nm$
entries. The arithmetic per byte read grows with $T$:

$$
\text{intensity} \approx \frac{2nmT}{b \cdot nm} = \frac{2T}{b}
$$

where $b$ is bytes per matrix entry. This single ratio, with $b = 2$ for
bfloat16, is why decoding one token at a time wastes a GPU and why batching
fixes it. Chapters 0 and 10 use it.

**Low-rank updates are cheap.** Adding $k v^\top$ to an $n \times m$ matrix
costs $nm$ multiply-adds and no factorization. Chapter 6's delta rule is a
rank-1 update applied once per token; the chunked form batches many of them and
turns the sequence of rank-1 updates into one matrix multiply.

**Triangular solves avoid inverses.** A system $Tx = b$ with $T$ lower
triangular is solved by forward substitution in $O(n^2)$ operations: read off
$x_1$, substitute, read off $x_2$, and so on. Forming $T^{-1}$ explicitly costs
$O(n^3)$ and loses accuracy. Chapter 6's chunked delta rule inverts a unit lower
triangular matrix of size `chunk x chunk`; it is a triangular solve, and that is
why the chunk size can stay small.

## Softmax, and why every implementation subtracts the max

Softmax turns a vector of scores into a probability distribution:

$$
\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

The entries are positive and sum to 1. The function is invariant to adding a
constant to every entry. Let $c$ be any scalar:

$$
\mathrm{softmax}(x + c)_i
= \frac{e^{x_i + c}}{\sum_j e^{x_j + c}}
= \frac{e^{c} e^{x_i}}{e^{c} \sum_j e^{x_j}}
= \mathrm{softmax}(x)_i
$$

The $e^c$ cancels. Mathematically the shift changes nothing.

Numerically it changes everything. `float32` overflows at about $3.4 \times
10^{38}$, and $e^{89}$ already exceeds that. Attention scores at long context
routinely reach values where $e^{x}$ is `inf`, and `inf / inf` is `nan`. So every
implementation picks $c = -\max_i x_i$ first:

$$
\mathrm{softmax}(x)_i = \frac{e^{x_i - \max_j x_j}}{\sum_j e^{x_j - \max_j x_j}}
$$

Now the largest exponent is $e^0 = 1$, nothing overflows, and the smallest terms
underflow to zero — which is harmless, because they were going to contribute
almost nothing anyway.

Two later chapters depend on this. Chapter 11 applies a temperature and a
top-*p* cut to logits before sampling, and the shift is what keeps that stable.
Chapter 14 goes further: FlashAttention never has the whole score row in memory
at once, so it carries a *running* maximum and rescales the partial sum every
time the maximum grows. Shift invariance is what makes that rescaling exact
rather than approximate.

## Floating point

Three formats appear in this course. Each stores a sign bit, an exponent field,
and a mantissa field, and the value is roughly $(-1)^s \times 1.m \times
2^{e - \text{bias}}$.

| Format | Sign | Exponent | Mantissa | Significand bits | Max finite |
|---|---|---|---|---|---|
| float32 | 1 | 8 | 23 | 24 | $\approx 3.4 \times 10^{38}$ |
| float16 | 1 | 5 | 10 | 11 | $65504$ |
| bfloat16 | 1 | 8 | 7 | 8 | $\approx 3.4 \times 10^{38}$ |

The significand column counts the stored mantissa bits plus the implicit leading
1.

bfloat16 is float32 with 16 mantissa bits deleted. It keeps the 8-bit exponent,
so it has float32's range: any float32 value that is not subnormal converts to
bfloat16 without overflowing or flushing to zero. What it gives up is precision
— 8 significand bits, about 2 decimal digits. float16 makes the opposite trade:
11 significand bits, but an exponent that runs out at 65504, which is why
float16 training needs loss scaling and why this course keeps weights in
bfloat16.

**ULP and relative error.** A *unit in the last place* is the gap between one
representable number and the next. For a value in $[2^e, 2^{e+1})$ with $p$
significand bits, that gap is

$$
\mathrm{ULP} = 2^{e - p + 1}
$$

Rounding to nearest puts the answer within half a ULP, so the *relative* error
of a single stored value is at most $2^{-p}$. That constant is the unit roundoff:

| Format | Unit roundoff $2^{-p}$ | Decimal |
|---|---|---|
| float32 | $2^{-24}$ | $6.0 \times 10^{-8}$ |
| float16 | $2^{-11}$ | $4.9 \times 10^{-4}$ |
| bfloat16 | $2^{-8}$ | $3.9 \times 10^{-3}$ |

A bfloat16 number carries about four significant bits of fraction. That is why
this course checks kernels against a float32 reference and accepts agreement to
around $10^{-2}$ in bfloat16, while a float32 kernel is expected to agree to
$10^{-6}$.

**A running sum stops absorbing small addends.** Take a sum $s$ and an addend
$x$. The exact result $s + x$ is rounded back into the format, and if $x$ is
smaller than half a ULP of $s$, the rounded result is $s$ again. The addend
vanishes.

Work it through in bfloat16. At $s = 256 = 2^8$ with $p = 8$ significand bits,

$$
\mathrm{ULP}(256) = 2^{8 - 8 + 1} = 2
$$

so the representable neighbours of 256 are 254, 256, 258. Now add 1:

$$
256 + 1 = 257 = \tfrac{256 + 258}{2}
$$

which is exactly halfway. Round-to-nearest-even picks 256. Adding 1 to 256 in
bfloat16 returns 256. Add 1 a thousand more times and the sum is still 256.

The general rule follows from the ULP formula: a running sum in a format with
$p$ significand bits stops moving once the addends fall below about $s \cdot
2^{-p}$. In bfloat16 that is a ratio of $1/256$; in float32, $1/16{,}777{,}216$.
This is why the linear-attention state in chapter 6 is kept in float32 even
though the weights are bfloat16 — the state accumulates over thousands of
tokens, and a bfloat16 accumulator would quietly stop learning from the tail of
the sequence. It is also why reductions inside kernels accumulate in float32 and
cast down only at the end.

**What goes wrong, and how it looks.** Overflow in float16 shows up as `inf`
then `nan` and is loud. Precision loss in bfloat16 is silent: the output is
finite, plausible, and wrong in the third digit, which a correctness test with a
loose tolerance will pass. The symptom is usually a model that works at short
context and degrades at long context, because the error accumulates with the
number of terms summed.

## GPU vocabulary

Enough to read the kernel chapters. The numbers are for the A100 80GB this
course targets.

**Streaming multiprocessor (SM).** The GPU's unit of independent execution. An
A100 has 108 of them. Each has its own registers, scheduler, and shared memory.
A kernel that does not produce enough work to occupy all 108 leaves most of the
chip idle no matter how good the inner loop is.

**Warp.** 32 threads that execute the same instruction at the same time. The
warp, not the thread, is the real unit of scheduling. If threads in a warp take
different branches, the warp executes both paths in turn and masks off the
inactive threads — *divergence*, and it costs exactly what it sounds like.

**Thread block.** A group of threads, up to 1024, that runs on one SM and can
cooperate through shared memory and barriers. Blocks within a kernel launch
cannot synchronize with each other. You choose the block size; it is usually a
multiple of 32.

**The memory hierarchy.** Four levels, each an order of magnitude faster and
smaller than the one before:

| Level | Size on an A100 | Scope | Cost to reach |
|---|---|---|---|
| Registers | 256 KB per SM | One thread | Free, part of the instruction |
| Shared memory | up to 164 KB per SM | One block | Tens of cycles |
| L2 cache | 40 MB | Whole GPU | Hundreds of cycles |
| HBM (global memory) | 80 GB | Whole GPU | Hundreds of cycles, 1275 GB/s measured |

HBM is high-bandwidth memory: the DRAM stacked next to the die. It is where the
weights and the KV cache live, and it is the bottleneck in almost everything
this course measures. Shared memory is a software-managed scratchpad, not a
cache — you copy into it explicitly. FlashAttention is, in one sentence, an
attention kernel that keeps its working set in shared memory instead of
round-tripping through HBM.

**Kernel launch.** Handing one GPU function to the driver to run across many
blocks. Each launch costs 5 to 10 microseconds of overhead, which is nothing for
a big kernel and a serious cost for a decode step that issues several hundred
small ones. That overhead is why CUDA graphs exist.

**Memory coalescing.** When the 32 threads of a warp read 32 consecutive
addresses, the hardware merges them into a few wide transactions. When they read
addresses scattered across memory, it issues many narrow ones and most of each
transaction is thrown away. On this hardware the measured gap between coalesced
and strided access is 6.9x. Coalescing is the main reason kernel code cares
which tensor dimension is contiguous.

**Occupancy.** The fraction of each SM's thread slots that are actually
resident. Registers and shared memory are the limits: a kernel that uses many
registers per thread fits fewer warps per SM, and with fewer warps the scheduler
has less work to hide memory latency with. High occupancy is a means, not a
goal; a kernel at 25% occupancy that saturates bandwidth is finished.

## Units

Storage has two conventions and this course uses both, deliberately:

$$
1\ \mathrm{GB} = 10^{9}\ \text{bytes}, \qquad
1\ \mathrm{GiB} = 2^{30} = 1{,}073{,}741{,}824\ \text{bytes}
$$

The convention here: **decimal units (GB) for anything a vendor prints**, and
**binary units (KiB, MiB, GiB) for anything computed from a power-of-two shape.**
So the weights are 53.8 GB, matching the way model sizes are quoted, and the KV
cache is 64 KiB per token, because $64 \times 1024$ is what the shape arithmetic
produces. The same 53.8 GB is 50.1 GiB; both numbers describe the same bytes.

Bandwidth is always decimal: 1935 GB/s means $1.935 \times 10^{12}$ bytes per
second.

The A100's "80GB" is the awkward case. Chapter 2's lab treats the card as
$80 \times 2^{30}$ bytes and subtracts a fixed 6 GiB for the driver, the CUDA
context, and an activation workspace. The real usable figure depends on the
driver version, which is why the budget carries explicit headroom rather than
pretending to be exact.

## Symbols used across the course

| Symbol | Meaning | This model |
|---|---|---|
| $T$ | tokens in one forward pass | 1 in decode, up to thousands in prefill |
| $L$ | context length, tokens already in the cache | 4k to 128k in the examples |
| $B$ | batch size, concurrent sequences | set by memory |
| $N$ | number of decoder layers | 64 |
| $d$ | hidden size, width of the residual stream | 5120 |
| $d_{ff}$ | MLP intermediate width | 17408 |
| $h$ | query heads in a full-attention layer | 24 |
| $h_{kv}$ | key-value heads | 4 |
| $g$ | GQA group size, $h / h_{kv}$ | 6 |
| $d_h$ | head dimension | 256 |
| $d_r$ | rotary dimension, the rotated channels per head | 64 |
| $V$ | vocabulary size | 248,320 |
| $b$ | bytes per element | 2, bfloat16 |
| $q_t, k_t, v_t$ | query, key, value vectors at position $t$ | length $d_h$ |
| $S_t$ | linear-attention recurrent state after token $t$ | a matrix per head |
| $a_t$ | forget gate of the gated delta rule, in $(0,1)$ | per value head |
| $\beta_t$ | write strength of the delta rule | per value head |
| $o_t$ | layer output at position $t$ | length $d_h$ per head |
| $I$ | arithmetic intensity, FLOPs per byte moved | ridge point 161 |

Where a chapter needs a symbol not in this table, it defines it on first use.

## Check your understanding

**A tensor has shape `(batch, heads, seq, head_dim)`. Which reads are
coalesced?** Reads along `head_dim`, the last and contiguous dimension. A read
that walks `seq` with the other indices fixed strides by `head_dim` elements —
512 bytes in bfloat16 — and every warp transaction wastes most of its payload.
This is why attention kernels tile over `seq` but keep `head_dim` whole.

**Why does subtracting the maximum not change the softmax output?** Because the
factor $e^c$ appears in every numerator and in the denominator, so it cancels.
The subtraction is free mathematically and mandatory numerically.

**In bfloat16, at what running-sum magnitude does adding 1.0 stop having any
effect?** Once the sum exceeds roughly $2^8 = 256$, since the ULP there is 2 and
1.0 is at most half a ULP. The exact crossover depends on rounding mode and the
sum's exponent, but the order of magnitude — a ratio of 256 between sum and
addend — is the number to remember.

**`torch.einsum("bhid,bhjd->bhij", q, k)` runs without error on tensors where
`q_len` and `head_dim` happen to be equal. What could still be wrong?** Nothing
in the shapes, but everything in the meaning: if `q` was left in
`(batch, seq, heads, head_dim)` order and never transposed, the string contracts
over heads instead of head_dim and the sizes still line up. The result is a
finite, wrong tensor. Assert the shape, do not infer it.

## Further reading

- [What every computer scientist should know about floating-point arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
- [BFloat16: the secret to high performance on cloud TPUs](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
- [The CUDA C++ programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA A100 tensor core GPU architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf)
- [Einsum is all you need](https://rockt.github.io/2018/04/30/einsum)
