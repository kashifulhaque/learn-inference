---
title: FlashAttention
slug: 14-flash-attention
part: "Part 4 — Kernels"
summary: Deriving the online softmax one step at a time, then building the tiled kernel that never writes a score matrix to memory.
minutes: 150
gpu: true
objectives:
  - Derive the online softmax rescaling and prove it returns the same numbers as a one-pass softmax.
  - Say what lives in registers and what lives in shared memory at every point in the tiled loop.
  - Write a tiled attention kernel in Triton with correct causal masking and grouped-query indexing.
  - Explain the IO complexity result and why a kernel that does more arithmetic runs faster.
  - Read a benchmark honestly when a teaching kernel loses to a vendor kernel.
lab: 14-flash-attention
---

# FlashAttention

Attention is the one operation in the model whose intermediate result is larger
than the model. Everything else — the projections, the MLP, the norms — produces
activations proportional to the number of tokens. Attention produces a score
matrix proportional to the *square* of the number of tokens, and the naive
implementation writes that matrix to memory, reads it back, writes it again, and
reads it once more.

FlashAttention never writes it. The scores are produced in tiles, consumed
immediately, and thrown away before the next tile arrives. The arithmetic is
identical to the naive version — identical, not approximate — and the algorithm
runs several times faster because it moves far fewer bytes.

This chapter derives the trick that makes that possible. The obstacle is the
softmax: it needs a normalizer over an entire row, and you do not have the whole
row until you have seen every key. The fix, the *online softmax*, is four lines
of algebra that most treatments compress into one. Here it gets four sections.

## Before you start

You need these, and nothing more:

**Attention as three matrices.** For one head, queries $Q$ of shape
$(L_q, d)$, keys $K$ of shape $(L_k, d)$, values $V$ of shape $(L_k, d)$. The
output is

$$
O = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V,
$$

with the softmax taken along each row. $L_q$ is the number of query positions in
this call, $L_k$ the number of cached key positions, and $d$ the head dimension —
256 in this model, 128 in the lab.

**Softmax and its shift invariance.** For a vector $s$,

$$
\operatorname{softmax}(s)_j = \frac{e^{s_j}}{\sum_k e^{s_k}}
= \frac{e^{s_j - c}}{\sum_k e^{s_k - c}}
$$

for any constant $c$, because $e^{-c}$ cancels between numerator and
denominator. Chapter 0a proves it. Implementations take $c = \max_k s_k$ so that
every exponent is at most zero and $e^{s_j-c} \in (0, 1]$, which makes overflow
impossible. That choice of $c$ is the only reason attention needs a maximum at
all, and it is the thing the online softmax has to work around.

**Registers, shared memory, HBM.** Three levels, each roughly an order of
magnitude smaller and faster than the one below it. HBM is the 80 GB of device
memory, reaching 1275 GB/s on a device-to-device copy. Shared memory is 164 KB
per streaming multiprocessor, addressable by every thread in a block. Registers
are private to a thread; an A100 SM has 256 KB of register file. Chapter 0a has
the vocabulary. The entire design of this kernel is a statement about which
tensor belongs at which level.

**Triton basics from chapter 13.** `tl.load` and `tl.store` with masks,
`tl.dot` for a tile matmul, `tl.max` and `tl.sum` with an `axis`, and
`tl.program_id` to find out which tile this program owns.

## What the naive version costs

Take the model's real geometry: 24 query heads, head dimension 256, and a
sequence of 8192 tokens. One head's score matrix holds

$$
8192 \times 8192 = 67{,}108{,}864 \text{ entries},
$$

and in bfloat16 that is 134.2 MB. Across 24 heads, 3.22 GB for a single
sequence. The reference path in `engine/layers/attention.py` computes in
float32 and holds the scores and the probabilities at the same time, which
doubles the element size and doubles the count: about 12.9 GB for one sequence,
against 53.8 GB of weights.

Now count traffic rather than capacity. Write the scores, read them for the
softmax, write the probabilities, read them for the value multiply: four passes
over an $L^2$ matrix, where the inputs $Q$, $K$, $V$ together are only $3Ld$
elements. At $L = 8192$ and $d = 256$, the inputs are 12.6 MB per head and the
score traffic is 537 MB per head. The operation spends 97% of its memory traffic
on a matrix that exists only to be consumed immediately.

This is why long-context prefill runs out of memory before it runs out of time.

## Why softmax resists tiling

Tiling the two matrix multiplies is easy. $QK^\top$ is a matmul, $PV$ is a
matmul, and matmuls tile: you can compute a block of the output from blocks of
the inputs, accumulating partial sums.

The softmax in between does not tile, at least not obviously. To normalize row
$i$ you need

$$
\ell_i = \sum_{j=1}^{L_k} e^{s_{ij} - m_i},
\qquad m_i = \max_j s_{ij},
$$

and both $m_i$ and $\ell_i$ are reductions over the whole row. A tile gives you
64 columns of that row. You cannot divide by a denominator you have not finished
computing.

Two escapes exist. One: make two passes over the keys, computing $m_i$ and
$\ell_i$ in the first and the output in the second. That doubles the reads of
$K$. Two: compute everything in one pass and *repair* the running values each
time the maximum moves. The second is the online softmax, and the repair costs a
multiply.

## Stage 1: the streaming maximum

Process the keys in blocks. After the first $t$ blocks, define the running
maximum over the scores seen so far. Write $S_t$ for the set of key indices
covered by the first $t$ blocks, and $s_j$ for the scaled score of key $j$
against the query row you are tracking:

$$
m^{(t)} = \max_{j \in S_t} s_j .
$$

Updating it is trivial. If the new block $t+1$ contains scores
$s_j, j \in \Delta_{t+1}$, then

$$
m^{(t+1)} = \max\!\left(m^{(t)},\ \max_{j \in \Delta_{t+1}} s_j\right).
$$

Start at $m^{(0)} = -\infty$, so the first block sets the maximum outright. The
sequence $m^{(0)} \le m^{(1)} \le \dots$ never decreases. That monotonicity is
what makes the rest safe, as you will see.

## Stage 2: the rescaling identity

Everything turns on one line of algebra. For any score $s_j$ and any two shifts
$m_{\text{old}}$ and $m_{\text{new}}$:

$$
e^{s_j - m_{\text{new}}}
= e^{s_j - m_{\text{old}} + m_{\text{old}} - m_{\text{new}}}
= e^{s_j - m_{\text{old}}} \cdot e^{m_{\text{old}} - m_{\text{new}}}.
$$

Add and subtract $m_{\text{old}}$ in the exponent, then split the exponential of
a sum into a product. That is all it is.

The factor

$$
\alpha = e^{m_{\text{old}} - m_{\text{new}}}
$$

does not depend on $j$. So a whole vector of exponentials taken against the old
shift converts to the new shift by multiplying by one scalar. Anything linear in
those exponentials — a sum, a weighted sum of value vectors — converts the same
way.

Because $m_{\text{new}} \ge m_{\text{old}}$, the exponent
$m_{\text{old}} - m_{\text{new}}$ is at most zero, so $\alpha \in (0, 1]$. The
correction only ever shrinks the accumulators. It cannot overflow. If the
maximum jumps a long way, $\alpha$ underflows to zero, which is exactly right:
the values accumulated so far were negligible against the new maximum.

## Stage 3: correcting the running sum

Define the running denominator, shifted by the *current* running maximum:

$$
\ell^{(t)} = \sum_{j \in S_t} e^{s_j - m^{(t)}}.
$$

Read that carefully. The shift inside the sum is $m^{(t)}$, which changes from
step to step. That is what makes the update non-obvious, and it is why you need
stage 2.

Split the new sum into old terms and new terms:

$$
\ell^{(t+1)}
= \sum_{j \in S_t} e^{s_j - m^{(t+1)}}
+ \sum_{j \in \Delta_{t+1}} e^{s_j - m^{(t+1)}}.
$$

Apply the rescaling identity to every term in the first sum, with
$m_{\text{old}} = m^{(t)}$ and $m_{\text{new}} = m^{(t+1)}$:

$$
\sum_{j \in S_t} e^{s_j - m^{(t+1)}}
= e^{m^{(t)} - m^{(t+1)}} \sum_{j \in S_t} e^{s_j - m^{(t)}}
= \alpha_{t+1}\, \ell^{(t)} .
$$

The old sum is already stored. Multiply it by one scalar and it is correct under
the new shift. So

$$
\ell^{(t+1)} = \alpha_{t+1}\, \ell^{(t)}
+ \sum_{j \in \Delta_{t+1}} e^{s_j - m^{(t+1)}},
\qquad \alpha_{t+1} = e^{m^{(t)} - m^{(t+1)}}.
$$

with $\ell^{(0)} = 0$.

## Stage 4: correcting the running output

Do the same for the numerator. Define the unnormalized running output, a vector
of length $d$:

$$
u^{(t)} = \sum_{j \in S_t} e^{s_j - m^{(t)}}\, v_j ,
$$

where $v_j$ is the value row for key $j$. The same split applies, and the same
identity, because each term is linear in its exponential:

$$
u^{(t+1)} = \alpha_{t+1}\, u^{(t)}
+ \sum_{j \in \Delta_{t+1}} e^{s_j - m^{(t+1)}}\, v_j .
$$

The second term is a small matmul: a row of $B_c$ probabilities against a
$(B_c, d)$ block of values. The first term is a scalar times a $d$-vector. Start
at $u^{(0)} = 0$.

That is the whole algorithm. Three running quantities per query row —
$m^{(t)}$, $\ell^{(t)}$, $u^{(t)}$ — and one correction factor shared between
the last two.

## Why the result is exact

People expect a streaming algorithm to approximate something. This one does not,
and the proof is two lines.

Claim: after the final block $T$, with $S_T = \{1, \dots, L_k\}$,

$$
\frac{u^{(T)}}{\ell^{(T)}}
= \frac{\sum_j e^{s_j - m^{(T)}} v_j}{\sum_j e^{s_j - m^{(T)}}}
= \sum_j \operatorname{softmax}(s)_j\, v_j .
$$

The first equality is the definition of $u^{(T)}$ and $\ell^{(T)}$, which the
induction in stages 3 and 4 maintains at every step: the invariant is that
$\ell^{(t)}$ and $u^{(t)}$ are the exact sums over $S_t$ shifted by $m^{(t)}$,
and the update preserves it. The second equality is softmax's shift invariance
with $c = m^{(T)}$, which is the true row maximum because $m$ tracked the maximum
over every block.

No term is dropped, no series is truncated, no tolerance is chosen. The only
difference from a one-pass softmax is the *order* in which floating-point
additions happen, and floating-point addition is not associative, so the last
bits can differ. In float32 the labs see agreement to about $10^{-5}$; in
bfloat16, to about $10^{-2}$. Both are rounding, not approximation.

There is a small irony here. The online version does *more* arithmetic than the
naive one — every block pays for a maximum, an exponential of the correction,
and two rescales it would not otherwise need. It is faster anyway, because the
arithmetic was never the constraint.

## The tiled algorithm

Now assemble it. Choose two block sizes: $B_r$ query rows per tile and $B_c$ key
columns per tile. The kernel in `engine/kernels/flash_attn_triton.py` defaults to
$B_r = 128$ and $B_c = 64$, named `BLOCK_M` and `BLOCK_N`.

The outer loop runs over query tiles and is not a loop at all — it is the launch
grid, one program per tile, all running at once:

```python
grid = (triton.cdiv(q_len, block_m), batch * heads)
```

The inner loop, inside each program, runs over key blocks. In pseudocode, for one
query tile $Q_i$ of shape $(B_r, d)$:

1. Load $Q_i$ from HBM into registers. It stays there for the whole pass.
2. Set $m \leftarrow -\infty$ (shape $(B_r,)$), $\ell \leftarrow 0$ (shape
   $(B_r,)$), $u \leftarrow 0$ (shape $(B_r, d)$).
3. For each key block $j$:
   - Load $K_j$ and $V_j$, each $(B_c, d)$, into shared memory.
   - $S \leftarrow Q_i K_j^\top / \sqrt{d}$, shape $(B_r, B_c)$, in registers.
   - Apply the mask: set masked entries of $S$ to $-\infty$.
   - $m_{\text{new}} \leftarrow \max(m, \operatorname{rowmax} S)$.
   - $\alpha \leftarrow \exp(m - m_{\text{new}})$, shape $(B_r,)$.
   - $P \leftarrow \exp(S - m_{\text{new}})$, shape $(B_r, B_c)$.
   - $\ell \leftarrow \alpha \odot \ell + \operatorname{rowsum} P$.
   - $u \leftarrow \alpha \odot u + P V_j$.
   - $m \leftarrow m_{\text{new}}$.
4. $O_i \leftarrow u / \ell$, and store to HBM.

Two details are worth naming. The division by $\ell$ happens once, at the end,
not once per block — dividing inside the loop would be correct but would cost
$B_r \times d$ divisions per block instead of per tile. And the accumulator $u$
never leaves registers, so the output is written to HBM exactly once.

The original FlashAttention paper runs the loops the other way round: outer over
key blocks, inner over query blocks. That order forces the partial output to be
re-read and re-written from HBM on every outer iteration. FlashAttention-2 swaps
them, which is the order above, and is why this kernel keeps one accumulator in
registers for the whole pass.

## What lives where

Work out the residency for the lab's geometry: $B_r = 128$, $B_c = 64$,
$d = 128$, bfloat16 inputs.

**In registers, per program:**

| Tensor | Shape | Type | Bytes |
|---|---|---|---|
| $Q_i$ | $(128, 128)$ | bfloat16 | 32 KiB |
| $u$ (`acc`) | $(128, 128)$ | float32 | 64 KiB |
| $m$, $\ell$ | $(128,)$ each | float32 | 1 KiB |
| $S$, $P$ | $(128, 64)$ | float32 | 32 KiB each, transient |

An A100 SM has 256 KB of register file shared by every resident program. The
accumulator alone is 64 KiB, which is why the kernel asks for `num_warps=8` when
`head_dim >= 128`: more warps means more threads to spread those registers over.

Push $d$ to 256, this model's real head dimension, and the accumulator doubles to
128 KiB. That is most of an SM's register file for one program, and it is the
reason a kernel tuned at $d = 128$ may need $B_r = 64$ at $d = 256$. Register
pressure is a function of the block sizes and the head dimension, and nothing
else.

**In shared memory, per program:** the $K_j$ and $V_j$ tiles, staged so that the
loads for block $j+1$ overlap the arithmetic for block $j$. Triton calls the
number of overlapping copies `num_stages`, and the engine computes it rather
than guessing:

```python
element_size = q.element_size()
num_stages = 4
while num_stages > 1 and (
    num_stages * block_n * head_dim * element_size * 2 > SHARED_MEMORY_BYTES
):
    num_stages -= 1
```

`SHARED_MEMORY_BYTES` is 160 KB, a little under the 164 KB Ampere exposes. The
factor of 2 is for $K$ and $V$. Run the arithmetic for the two dtypes at
$B_c = 64$, $d = 128$:

- bfloat16: $4 \times 64 \times 128 \times 2 \times 2 = 131{,}072$ bytes, or
  128 KiB. Four stages fit.
- float32: $4 \times 64 \times 128 \times 4 \times 2 = 262{,}144$ bytes. Too
  big. Three stages need 192 KiB, also too big. Two stages need 128 KiB and fit.

Exceeding the limit raises `OutOfResources` at launch rather than failing
quietly, so this is safe to compute rather than guess.

**In HBM:** $Q$, $K$, $V$, and $O$. Nothing else. The score matrix has no
address.

## The kernel, line by line

Here is the inner loop from `engine/kernels/flash_attn_triton.py`, with the
surrounding setup.

```python
start_m = tl.program_id(0)          # which query tile
bh = tl.program_id(1)               # which (batch, head) pair
batch = bh // n_heads
head = bh % n_heads
kv_head = head // kv_group          # grouped-query indexing

offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
offs_n = tl.arange(0, BLOCK_N)                       # (BLOCK_N,)
offs_d = tl.arange(0, HEAD_DIM)                      # (HEAD_DIM,)
```

`offs_m` holds this tile's absolute query positions, `offs_d` the head
dimension. Triton works in tiles of indices, not scalars; every load below is a
broadcast of two of these index vectors into a 2D tile.

```python
q_ptrs = (
    q_ptr + batch * stride_qb + head * stride_qh
    + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
)
q = tl.load(q_ptrs, mask=offs_m[:, None] < q_len, other=0.0)   # (BLOCK_M, HEAD_DIM)
```

One load, outside the loop. The mask handles the last tile when `q_len` is not a
multiple of `BLOCK_M`; padded rows load zeros and their outputs are discarded by
the store mask at the end.

```python
m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)   # (BLOCK_M,)
l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # (BLOCK_M,)
acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)       # (BLOCK_M, HEAD_DIM)
```

The three running quantities, initialized as the derivation requires. All three
are float32 even when the inputs are bfloat16. This is not optional: `l_i`
accumulates thousands of exponentials, and bfloat16 carries 8 mantissa bits, so
once the running sum passes 256 a new term of 1.0 stops changing it at all.
Chapter 0a has the format details.

```python
for start_n in range(0, hi, BLOCK_N):
    cols = start_n + offs_n                                  # (BLOCK_N,)
    k = tl.load(k_ptrs, mask=cols[:, None] < kv_len, other=0.0)  # (BLOCK_N, HEAD_DIM)
    v = tl.load(v_ptrs, mask=cols[:, None] < kv_len, other=0.0)  # (BLOCK_N, HEAD_DIM)

    scores = tl.dot(q, tl.trans(k)) * scale                  # (BLOCK_M, BLOCK_N)
    scores = tl.where(cols[None, :] < kv_len, scores, float("-inf"))
    if IS_CAUSAL:
        scores = tl.where(
            cols[None, :] <= offs_m[:, None] + offset, scores, float("-inf")
        )
```

`tl.dot` runs on the tensor cores. `scale` is $1/\sqrt{d}$. The first `tl.where`
kills columns past the end of the sequence — they loaded as zeros, which would
otherwise produce a score of zero and a probability of $e^{-m}$, quietly wrong.
The second applies causal masking, covered below.

```python
    m_new = tl.maximum(m_i, tl.max(scores, axis=1))          # (BLOCK_M,)
    correction = tl.exp(m_i - m_new)                         # (BLOCK_M,)
    p = tl.exp(scores - m_new[:, None])                      # (BLOCK_M, BLOCK_N)

    l_i = l_i * correction + tl.sum(p, axis=1)
    acc = acc * correction[:, None] + tl.dot(p.to(v.dtype), v)
    m_i = m_new
```

Stages 1 through 4, in order: the streaming maximum, the correction factor
$\alpha$, the probabilities under the new shift, then the two corrected
accumulators. `axis=1` reduces along the key dimension. Getting that axis wrong
is the single most common bug in a first attempt, and it does not crash — it
normalizes across queries instead of across keys and returns plausible-looking
garbage.

`correction[:, None]` broadcasts the per-row scalar across the head dimension.
`p.to(v.dtype)` casts the probabilities down so the second `tl.dot` also runs on
the tensor cores; the accumulation it feeds is still float32.

```python
acc = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_m[:, None] < q_len)
```

One division, one store. The `tl.where` guards a row whose scores were all
masked, where $\ell = 0$; dividing by zero there would put NaN into an output
that is about to be discarded anyway, and NaNs propagate through everything
downstream.

## Causal masking and the blocks you can skip

Causal attention lets query position $i$ see key position $j$ only when
$j \le i$. Masking the score matrix after computing it saves nothing: you paid
for the matmul. The saving comes from never launching the blocks that are
entirely above the diagonal.

```python
offset = kv_len - q_len
hi = tl.minimum(kv_len, (start_m + 1) * BLOCK_M + offset) if IS_CAUSAL else kv_len
```

`offset` aligns the query tile's local row indices with absolute positions. The
last query in tile `start_m` sits at absolute position
`(start_m + 1) * BLOCK_M - 1 + offset`, so no key beyond that can contribute, and
`hi` cuts the loop there.

Count what that saves for a square causal problem at $L = 8192$ with
$B_r = 128$, $B_c = 64$. There are $8192/128 = 64$ query tiles and
$8192/64 = 128$ key blocks, so a full pass is $64 \times 128 = 8192$ block pairs.
Under the causal bound, query tile $i$ (zero-indexed) runs to
$(i+1) \times 128$ keys, which is $2(i+1)$ key blocks. The total is

$$
\sum_{i=0}^{63} 2(i+1) = 2 \cdot \frac{64 \cdot 65}{2} = 4160 ,
$$

or 50.8% of the full count. Half the work disappears, and the extra 0.8% is the
diagonal blocks — the ones that straddle the boundary and still need the
elementwise `tl.where`. Every block strictly below the diagonal needs no mask at
all, and a more aggressive kernel specializes those into a separate loop with no
`tl.where` in it.

The `offset` is the same one from chapter 7, and it is where decode goes wrong.
During prefill, `q_len == kv_len` and the offset is zero. During decode,
`q_len` is 1 and `kv_len` is the whole cached prefix, so the offset is large and
a kernel that assumes zero will mask away the entire context. That bug passes
every prefill test. The lab checks a decode step against the last row of a full
prefill precisely to catch it.

## Grouped queries in the kernel

The model has 24 query heads and 4 KV heads, so six query heads share each KV
head. The caller could materialize the duplication with `repeat_interleave`,
which is what the naive reference does, and that would write six copies of $K$
and $V$ to HBM. The kernel indexes the shared head instead:

```python
kv_head = head // kv_group      # kv_group = heads // kv_heads = 6
```

Six programs now read the same $K$ and $V$ bytes. Those programs are scheduled
close together, so the reads mostly hit L2 rather than HBM, and the effective
bandwidth requirement for the KV read drops by a factor of six. That is the real
benefit of grouped-query attention, and it exists only if the kernel does the
sharing. A kernel that duplicates the heads gets the memory savings in the cache
and none of the bandwidth savings at read time.

## The IO complexity result

Now make the byte counting precise, because the asymptotics are the actual
theorem behind FlashAttention.

**Naive.** The score matrix is written and read a constant number of times, so
HBM traffic is

$$
\Theta(L^2 + L d)
$$

elements, and the $L^2$ term dominates once $L > d$. At $L = 8192$ and
$d = 128$, per head, four passes over the scores move 537 MB while the inputs
are 6.3 MB.

**Tiled.** Let $M$ be the number of key rows that fit in fast on-chip memory, so
a $K$ tile and a $V$ tile together occupy $\Theta(M d)$ elements. Each of the
$L/B_r$ query tiles streams all of $K$ and $V$: that is $2Ld$ elements per query
tile. With $B_r$ bounded by the same on-chip budget, $B_r = \Theta(M)$, the
total is

$$
\frac{L}{B_r} \cdot 2Ld = \Theta\!\left(\frac{L^2 d}{M}\right)
$$

elements. Written in elements of SRAM rather than rows — call that
$M_{\text{elem}} = M d$ — the same bound is
$\Theta(L^2 d^2 / M_{\text{elem}})$, which is how the paper states it.

Two things follow.

First, the tiled version is still quadratic in $L$. Anyone who tells you
FlashAttention makes attention linear in memory traffic is talking about *peak
allocation*, which is genuinely linear, not about traffic. The keys are re-read
once per query tile.

Second, the improvement factor is $M/d$: the bigger the on-chip memory relative
to the head dimension, the fewer times you re-read $K$. Put A100 numbers in. Of
the 164 KB of shared memory per SM, the kernel budgets 160 KB, and at $d = 128$
in bfloat16 a key row is 256 bytes and a key-plus-value row pair is 512 bytes, so
roughly $M = 320$ row pairs fit. Against $d = 128$ that predicts a factor of
about 2.5 in HBM traffic, which is far less than the speedups people report.

The gap is L2. At $L = 8192$, $d = 128$, one head's $K$ and $V$ are 4.2 MB
together, and the lab's eight heads are 33.6 MB — under the A100's 40 MB L2. The
re-reads are quadratic in count but they mostly never reach HBM. This is the
same effect chapter 13 measured for fused RMSNorm, where a saving predicted by
the byte count vanished because the data never left cache; here it works in your
favor instead.

## Speed: 0.63x, and why

The README's measurement on the target hardware:

| Measurement | Result |
|---|---|
| Triton FlashAttention against the vendor kernel | 0.63x at 8k tokens |
| FlashAttention peak memory against naive | 27.5x less |

The kernel is about a third slower than what
`F.scaled_dot_product_attention` dispatches to. That deserves an honest
explanation rather than a shrug.

Start with the arithmetic floor. At $L = 8192$, 8 heads, $d = 128$, causal, the
two matmuls cost

$$
2 \times 2 L^2 d = 4 \times 8192^2 \times 128 = 34.4 \text{ GFLOP}
$$

per head, halved by causality to 17.2 GFLOP, and $\times 8$ heads gives
137 GFLOP. At the A100's 312 TFLOP/s for bfloat16 that is 0.44 ms.

The engine's docstring records 1.64 ms for this kernel at that geometry with the
$128 \times 64$ tiling, against 3.35 ms for a $64 \times 64$ tiling. So the
kernel reaches about 27% of peak, and the vendor kernel, at 0.63x of 1.64 ms,
lands near 1 ms, or about 43%. Neither is close to peak, which is normal for
attention; what needs explaining is the gap between them.

Four things account for most of it:

**Non-matmul work on the wrong units.** Every block computes $B_r \times B_c$
exponentials. Over a causal pass that is $L^2/2$ exponentials per head, or 268
million across 8 heads for one call. Exponentials run on the special function
units, not the tensor cores, and they do not overlap the matmuls for free.
FlashAttention-2's main contribution was reducing exactly this class of work.

**The rescale in the inner loop.** `acc = acc * correction[:, None] + ...`
touches $B_r \times d$ float32 registers on every iteration. A tuned kernel
defers more of that work, keeping the correction as a scalar applied at tile
boundaries rather than a full accumulator multiply each time.

**Scheduling.** cuDNN and CUTLASS kernels use warp specialization — some warps do
nothing but issue asynchronous copies while others do nothing but matmul — plus
hand-tuned pipelining and, on newer hardware, TMA descriptors. Triton generates a
good generic schedule. Good generic loses to hand-tuned by tens of percent.

**Autotuning.** The vendor kernel picks tile sizes per shape, per dtype, per
head dimension, from a table built by measurement. This kernel has two defaults.
The measured $128 \times 64$ against $64 \times 64$ gap — 1.64 ms against
3.35 ms, a factor of two from one parameter — shows how much is on the table.

Landing within 2x of a vendor kernel with 40 lines you can read in one sitting is
a good result. The lab's threshold is 0.33x for that reason.

## Memory: 27.5x

The memory result is not a near miss, and it is the one that changes what you can
serve.

The lab measures peak allocation at $L = 4096$, 8 heads, $d = 128$. Check the
number by hand. The naive path materializes the scores in float32:

$$
8 \times 4096 \times 4096 \times 4 = 536{,}870{,}912 \text{ bytes} = 512 \text{ MiB}.
$$

FlashAttention's live set is $Q$, $K$, $V$, and $O$: 8 MiB for $Q$, 2 MiB each
for $K$ and $V$ with two KV heads, 8 MiB for $O$, so 20 MiB. The ratio of
$512 + \epsilon$ to 20 is 27.5 when $\epsilon$ is the roughly 38 MiB of float32
copies the reference path makes of its inputs. The measurement and the arithmetic
agree, which is a good sign that you understand what the measurement measured.

Peak memory, not clock time, is why every serving stack uses FlashAttention. A
27.5x reduction in the attention working set is the difference between 8k context
and 200k context on the same card.

## What goes wrong

**The reduction axis.** `tl.sum(p, axis=1)` sums over keys. `axis=0` sums over
queries. Both run. The second gives a result that is smooth, finite, and wrong,
and it survives a shape check because the tile is square when `BLOCK_M == BLOCK_N`.
Test with a non-square tiling and the shape error appears immediately.

**Forgetting the out-of-range mask on scores.** Columns past `kv_len` load as
zeros, and $e^{0 - m}$ is not zero. Without
`tl.where(cols[None, :] < kv_len, ...)` those phantom keys take a share of the
probability mass. The symptom is an error that grows as the sequence length moves
further from a multiple of `BLOCK_N`, which is why the lab includes a length of
130.

**bfloat16 accumulators.** Keeping `l_i` or `acc` in bfloat16 saves registers and
destroys the result at long context. The sum stops growing once the running total
is 256 times larger than the new term. The symptom is an error that grows with
`kv_len` and looks like a masking bug.

**A fully masked row.** If every score in a row is $-\infty$, then
$m^{(t)} = -\infty$ and the correction computes $\exp(-\infty + \infty)$, which
is NaN. Causal masking as written never produces one — query row $i$ always sees
key 0, which is in the first block — but a custom mask can, and the NaN spreads
to the whole output. The `l_i == 0.0` guard catches the division but not this.

**The decode offset.** Covered above, and worth repeating: it is the bug that
passes prefill tests and breaks in production.

**TF32.** On float32 inputs `tl.dot` uses the TF32 tensor cores, which keep 10
mantissa bits. Results differ from a true float32 matmul by around $10^{-3}$
relative. That is why the lab's float32 tolerance is $5 \times 10^{-3}$ and not
$10^{-6}$.

## Check your understanding

**If the correction factor is $e^{m_{\text{old}} - m_{\text{new}}}$ and the
maximum only rises, why is there no matching correction when the maximum does not
move?**

There is; it is just invisible. When $m^{(t+1)} = m^{(t)}$ the factor is
$e^0 = 1$, and multiplying by it changes nothing. The kernel computes it anyway
rather than branching, because a branch would diverge across the rows of a tile
and cost more than the multiply.

**Why keep a running maximum at all? The algorithm would be simpler without it.**

Correctness would survive; range would not. float32 overflows above $e^{88}$,
and trained models produce score outliers far larger than a random-normal
analysis predicts. Without the shift, one large score makes the numerator and
the denominator both infinite and the output NaN. The maximum guarantees every
exponent is at most zero, whatever the scores turn out to be.

**The tiled version re-reads $K$ and $V$ once per query tile, so it moves more
key bytes than the naive version. Why is it still faster?**

Because the naive version's $L^2$ score traffic dwarfs both. At $L = 8192$ and
$d = 128$, one head's $K$ and $V$ are 4.2 MB and its score matrix is 134 MB
touched four times. Re-reading 4.2 MB sixty-four times is 268 MB of requests,
most of which hit a 40 MB L2, against 537 MB of score traffic that cannot hit
anything because it is streamed once and never reused.

**Does the block size change the answer?**

No. It changes speed, register pressure, and how many blocks causality lets you
skip, but the online softmax is exact for any $B_r$ and $B_c$, including
$B_c = 1$. The lab checks a sequence length of 130 against a block size of 128
for exactly this reason: the tail block has 2 valid columns and the answer must
still be right.

## Lab

Write the FlashAttention forward pass in Triton, matching the signature in
`starter.py`: `flash_attention(q, k, v, causal=True, scale=None)` with `q` of
shape `(batch, heads, q_len, head_dim)` and `k`, `v` of shape
`(batch, kv_heads, kv_len, head_dim)`, where `kv_heads` divides `heads`.

The harness checks correctness against a float32 naive implementation across six
cases: non-causal square, causal square, a single decode query behind a
1024-token prefix, a 64-token chunk behind a 512-token prefix, a length of 130
that is not a multiple of the block size, and a 1024-token causal pass. Each must
land within $5 \times 10^{-3}$ absolute. A bfloat16 run must stay within
$5 \times 10^{-2}$. One more check compares a single decode query against the
last row of a full prefill, which is where an incorrect causal offset shows up.

Then it benchmarks against `F.scaled_dot_product_attention` at 512, 2048, and
8192 tokens, requiring at least 0.33x at 8192, and compares peak memory against
the naive path at 4096 tokens, requiring at least 3x. The reference lands at
0.63x and 27.5x.

## Further reading

- [FlashAttention: fast and memory-efficient exact attention with IO-awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: faster attention with better parallelism and work partitioning](https://arxiv.org/abs/2307.08691)
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
- [Self-attention does not need $O(n^2)$ memory](https://arxiv.org/abs/2112.05682)
- [Triton tutorial: fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
