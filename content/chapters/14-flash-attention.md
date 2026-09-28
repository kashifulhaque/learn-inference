---
title: FlashAttention
slug: 14-flash-attention
part: "Part 4 — Kernels"
summary: Building the online softmax from a four-score example and then deriving it one step at a time, then writing the tiled kernel that never writes a score matrix to memory.
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

> [!TLDR]
> - Ordinary attention builds a table of scores, one for every pair of tokens,
>   and pushes it through GPU memory four times. At 8192 tokens, that table is
>   97% of the memory traffic.
> - The *online softmax* avoids the table. Each query keeps three running
>   values, a maximum, a sum, and an output, and fixes the last two with one
>   multiply whenever the maximum rises.
> - The result is exact, not approximate. Only the order of floating-point
>   additions changes.
> - The tiled kernel keeps its running output on the chip, streams the keys and
>   values past it, and never gives the score table a place in memory.
> - The Triton kernel runs at 0.63x the speed of the vendor kernel and uses 27.5x
>   less peak memory than the naive path.

Attention is the one operation in the model whose intermediate result is larger
than the model. Everything else, including the projections, the MLP, and the
norms, produces activations proportional to the number of tokens. Attention
produces a score matrix proportional to the *square* of the number of tokens.
The naive implementation writes that matrix to memory, reads it back, writes it
again, and reads it once more.

FlashAttention never writes it. The kernel produces the scores in tiles,
consumes them immediately, and throws them away before the next tile arrives.
The arithmetic is ==identical to the naive version, not approximate==, and the
algorithm runs several times faster because it moves far fewer bytes.

This chapter builds the trick that makes that possible. The obstacle is the
softmax: it needs a normalizer over an entire row, and you don't have the whole
row until you've seen every key. The fix, the online softmax, is four lines of
algebra that most treatments compress into one. Here it gets a worked example
you can check by hand, and then four sections.

## Before you start

**Attention as three matrices.** For one head, you have queries $Q$ of shape
$(L_q, d)$, keys $K$ of shape $(L_k, d)$, and values $V$ of shape $(L_k, d)$.
The output takes the softmax along each row:

$$
O = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V.
$$

$L_q$ is the number of query positions in this call, and $L_k$ is the number of
cached key positions. $d$ is the head dimension, `head_dim` in code: 256 in this
model, 128 in the lab. Each entry of $QK^\top / \sqrt{d}$ is a *score*, one for
every query-key pair.

**Softmax and its shift invariance.** For a vector $s$ and any constant $c$, the
$e^{-c}$ cancels between numerator and denominator:

$$
\operatorname{softmax}(s)_j = \frac{e^{s_j}}{\sum_k e^{s_k}}
= \frac{e^{s_j - c}}{\sum_k e^{s_k - c}}.
$$

Chapter 0a proves it. Implementations take $c = \max_k s_k$, so that every
exponent is at most zero and $e^{s_j-c} \in (0, 1]$, which makes overflow
impossible. That choice of $c$ is the only reason attention needs a maximum at
all, and it's what the online softmax has to work around.

**Registers, shared memory, HBM.** A GPU has three levels of memory that
matter here. Each is roughly an order of magnitude smaller and faster than the
one below it. An SM, or streaming multiprocessor, is one of the A100's 108
processor cores:

| Level | Size on an A100 | Who can address it |
|---|---|---|
| HBM | 80 GB, 1275 GB/s on a device-to-device copy | Every thread |
| Shared memory | 164 KB per streaming multiprocessor | Every thread in a block |
| Registers | 256 KB of register file per SM | One thread each |

Chapter 0a has the vocabulary. The entire design of this kernel is a statement
about which tensor belongs at which level.

**Tensor cores.** The units on each SM that do small matrix multiplies far
faster than ordinary arithmetic. A kernel reaches them through a matmul call,
`tl.dot` in Triton.

**Triton basics from [chapter 13](/c/13-fusion-in-triton).** `tl.load` and
`tl.store` with masks, `tl.dot` for a tile matmul, `tl.max` and `tl.sum` with an
`axis`, and `tl.program_id` to find out which tile this program owns.

## What the naive version costs

This section measures the problem that FlashAttention solves. The naive version
spends almost all of its memory, and almost all of its memory traffic, on the
score matrix. Take the model's real geometry: 24 query heads, head dimension
256, and a sequence of 8192 tokens. One head's score matrix holds this many
entries:

$$
8192 \times 8192 = 67{,}108{,}864 \text{ entries}.
$$

Count capacity first:

- In bfloat16, one head's scores are 134.2 MB.
- Across 24 heads, that's 3.22 GB for a single sequence.
- The reference path in `engine/layers/attention.py` computes in float32 and
  holds the scores and the probabilities at the same time. That doubles the
  element size and doubles the count: about 12.9 GB for one sequence, against
  53.8 GB of weights.

Now count traffic. The naive path makes four passes over an $L^2$ matrix:

1. Write the scores.
2. Read them for the softmax.
3. Write the probabilities.
4. Read them for the value multiply.

The inputs $Q$, $K$, and $V$ together are only $3Ld$ elements. At $L = 8192$
and $d = 256$, the inputs are 12.6 MB per head, and the score traffic is 537 MB
per head.

The operation spends ==97% of its memory traffic== on a matrix that exists only
to be consumed immediately. That's why long-context *prefill*, the phase that
processes the whole prompt at once, runs out of memory before it runs out of
time.

## Why softmax resists tiling

This section explains why you can't cut the score matrix into tiles and handle
them one at a time, which is the obvious fix. The two matmuls tile, and the
softmax between them doesn't. $QK^\top$ is a matmul, $PV$ is a matmul, and
matmuls tile: you can compute a block of the output from blocks of the inputs,
accumulating partial sums.

To normalize row $i$, the softmax needs a maximum and a sum over the whole row:

$$
\ell_i = \sum_{j=1}^{L_k} e^{s_{ij} - m_i},
\qquad m_i = \max_j s_{ij}.
$$

Both $m_i$ and $\ell_i$ are reductions over the whole row, and a tile gives you
64 columns of that row. You can't divide by a denominator you haven't finished
computing.

Two escapes exist:

1. **Two passes.** Compute $m_i$ and $\ell_i$ over the keys in the first pass,
   and the output in the second. That doubles the reads of $K$.
2. **One pass with repair.** Compute everything in one pass, and *repair* the
   running values each time the maximum moves. This is the online softmax, and
   ==the repair costs a multiply==.

Next, you watch the repair work on four numbers before seeing it in general.

## The online softmax on four scores

This section runs the whole algorithm by hand on a row small enough to check.
One query has four scores, split into two blocks of two. Each key carries a
value, a single number here instead of a vector:

| | Key 1 | Key 2 | Key 3 | Key 4 |
|---|---|---|---|---|
| Block | 1 | 1 | 2 | 2 |
| Score $s_j$ | 1 | 2 | 4 | 3 |
| Value $v_j$ | 10 | 20 | 30 | 40 |

The goal is the softmax-weighted average of the values, computed while you see
only one block at a time. You keep three running numbers, each with its own
colour for the rest of the chapter:

- The running maximum $\hla{m}$: the largest score so far.
- The running sum $\hlb{\ell}$: the sum of $e^{s_j - \hla{m}}$ over the scores
  so far, measured against the current maximum.
- The running output $\hlc{u}$: the sum of $e^{s_j - \hla{m}} v_j$, measured
  against the same maximum.

**Block 1.** The larger score is 2, so $\hla{m} = 2$. Measure each score
against it:

- $e^{1-2} = 0.368$ and $e^{2-2} = 1$.
- $\hlb{\ell} = 0.368 + 1 = 1.368$.
- $\hlc{u} = 0.368 \times 10 + 1 \times 20 = 23.68$.

**Block 2, and the maximum moves.** This block's larger score is 4, so the new
maximum is 4. The new terms, measured against 4, are $e^{4-4} = 1$ and
$e^{3-4} = 0.368$. They add $1.368$ to the sum and
$1 \times 30 + 0.368 \times 40 = 44.72$ to the output.

You can't add those to the old totals yet. The old $\hlb{\ell}$ and $\hlc{u}$
were measured against a maximum of 2, and the new terms against 4. Multiplying
any old term $e^{s_j - 2}$ by $e^{2-4}$ turns it into $e^{s_j - 4}$, so one
multiply converts each whole total. That factor is the correction
$\hld{\alpha} = e^{2-4} = 0.135$:

- $\hlb{\ell} = 0.135 \times 1.368 + 1.368 = 0.185 + 1.368 = 1.553$.
- $\hlc{u} = 0.135 \times 23.68 + 44.72 = 3.20 + 44.72 = 47.92$.

**Finish.** Divide once: $47.92 / 1.553 = 30.86$.

**Check against the ordinary softmax.** With all four scores at once, the
maximum is 4. The exponentials are $e^{-3}$, $e^{-2}$, $e^{0}$, and $e^{-1}$,
or 0.050, 0.135, 1, and 0.368, which sum to 1.553. The weighted sum is
$0.050 \times 10 + 0.135 \times 20 + 1 \times 30 + 0.368 \times 40 = 47.92$,
and $47.92 / 1.553 = 30.86$. It's the same answer.

> [!INTUITION]
> The running totals are recorded in units of "relative to the current
> maximum." When the maximum rises, $\hld{\alpha}$ is the exchange rate that
> converts an old total into the new units. It costs one multiply per total, no
> matter how many terms the total already holds.

```viz
14-online-softmax
```

Three facts made that work, and the next four sections turn each one into
general algebra:

- The maximum only ever moves up (stage 1).
- One scalar converts every old term to the new maximum (stage 2).
- The sum and the output both convert with that same scalar (stages 3 and 4).

## Stage 1: the streaming maximum

The first question is how to track a row's maximum when you see the row one
block at a time, and it's the gentlest of the four stages. Write $S_t$ for the set of key indices
covered by the first $t$ blocks, and $s_j$ for the scaled score of key $j$
against the query row you're tracking. The running maximum after $t$ blocks is
the following:

$$
\hla{m^{(t)}} = \max_{j \in S_t} s_j .
$$

If the new block $t+1$ contains scores $s_j, j \in \Delta_{t+1}$, the update
takes the larger of the old maximum and the new block's maximum:

$$
\hla{m^{(t+1)}} = \max\!\left(\hla{m^{(t)}},\ \max_{j \in \Delta_{t+1}} s_j\right).
$$

Start at $\hla{m^{(0)}} = -\infty$, so the first block sets the maximum outright.
The sequence $\hla{m^{(0)}} \le \hla{m^{(1)}} \le \dots$ ==never decreases==, and
that monotonicity is what makes the rest safe. In the example, $\hla{m}$ went
from $-\infty$ to 2 to 4.

## Stage 2: the rescaling identity

This stage generalizes the exchange rate from the example, and everything turns
on one line of algebra. For any score $s_j$ and any two shifts
$m_{\text{old}}$ and $m_{\text{new}}$, add and subtract $m_{\text{old}}$ in the
exponent, then split the exponential of a sum into a product:

$$
e^{s_j - m_{\text{new}}}
= e^{s_j - m_{\text{old}} + m_{\text{old}} - m_{\text{new}}}
= e^{s_j - m_{\text{old}}} \cdot \hld{e^{m_{\text{old}} - m_{\text{new}}}}.
$$

Name the second factor the correction factor:

$$
\boxed{\hld{\alpha} = e^{m_{\text{old}} - m_{\text{new}}}}
$$

> [!KEY] One scalar moves a whole row to the new shift
> $\hld{\alpha}$ doesn't depend on $j$. So a whole vector of exponentials taken
> against the old shift converts to the new shift by multiplying by one scalar.
> Anything linear in those exponentials, like a sum or a weighted sum of value
> vectors, converts the same way.

The correction is also safe. Because $m_{\text{new}} \ge m_{\text{old}}$, the
exponent $m_{\text{old}} - m_{\text{new}}$ is at most zero, so
$\hld{\alpha} \in (0, 1]$:

- The correction only ever shrinks the accumulators, so it can't overflow.
- If the maximum jumps a long way, $\hld{\alpha}$ underflows to zero, which is
  exactly right: the values accumulated so far were negligible against the new
  maximum.

## Stage 3: correcting the running sum

This stage shows that the running sum updates with one multiply by
$\hld{\alpha}$. Define the sum shifted by the *current* running maximum:

$$
\hlb{\ell^{(t)}} = \sum_{j \in S_t} e^{s_j - \hla{m^{(t)}}}.
$$

Read that carefully. The shift inside the sum is $\hla{m^{(t)}}$, which changes
from step to step. That's what makes the update non-obvious, and it's why you
need stage 2. Split the sum into old and new terms, and rescale the old ones, to
get the update:

$$
\boxed{\hlb{\ell^{(t+1)}} = \hld{\alpha_{t+1}}\, \hlb{\ell^{(t)}}
+ \sum_{j \in \Delta_{t+1}} e^{s_j - m^{(t+1)}},
\qquad \hld{\alpha_{t+1}} = e^{m^{(t)} - m^{(t+1)}}}
$$

with $\hlb{\ell^{(0)}} = 0$.

> [!INTUITION]
> The old sum is already stored, but under the old shift. Multiply it by
> $\hld{\alpha}$ and it's correct under the new shift. Then add the new block's
> terms, which you compute directly under the new shift. In the example, that
> was $0.135 \times 1.368 + 1.368$.

> [!DEEPDIVE] Derive the running-sum update
> Split the new sum into old terms and new terms:
>
> $$
> \ell^{(t+1)}
> = \sum_{j \in S_t} e^{s_j - m^{(t+1)}}
> + \sum_{j \in \Delta_{t+1}} e^{s_j - m^{(t+1)}}.
> $$
>
> Apply the rescaling identity to every term in the first sum, with
> $m_{\text{old}} = m^{(t)}$ and $m_{\text{new}} = m^{(t+1)}$:
>
> $$
> \sum_{j \in S_t} e^{s_j - m^{(t+1)}}
> = e^{m^{(t)} - m^{(t+1)}} \sum_{j \in S_t} e^{s_j - m^{(t)}}
> = \alpha_{t+1}\, \ell^{(t)} .
> $$
>
> Substitute that back into the split, and you have the boxed update.

## Stage 4: correcting the running output

This stage shows that the running output updates the same way as the sum.
Define the unnormalized running output, a vector of length $d$, where $v_j$ is
the value row for key $j$:

$$
\hlc{u^{(t)}} = \sum_{j \in S_t} e^{s_j - \hla{m^{(t)}}}\, v_j .
$$

The same split and the same identity apply, because each term is linear in its
exponential:

$$
\boxed{\hlc{u^{(t+1)}} = \hld{\alpha_{t+1}}\, \hlc{u^{(t)}}
+ \sum_{j \in \Delta_{t+1}} e^{s_j - m^{(t+1)}}\, v_j}
$$

Start at $\hlc{u^{(0)}} = 0$. The second term is a small matmul: a row of $B_c$
probabilities, where $B_c$ is the number of keys per block, against a
$(B_c, d)$ block of values. The first term is a scalar times a $d$-vector.

That's the whole algorithm: ==three running quantities per query row==,
$\hla{m^{(t)}}$, $\hlb{\ell^{(t)}}$, and $\hlc{u^{(t)}}$, and one correction
factor $\hld{\alpha}$ shared between the last two.

## Why the result is exact

This section proves that the streaming result equals the ordinary softmax, not
an approximation of it. People expect a streaming algorithm to approximate
something. This one doesn't, and the proof is two lines. After the final block
$T$, with $S_T = \{1, \dots, L_k\}$, divide the output by the sum:

$$
\boxed{\frac{\hlc{u^{(T)}}}{\hlb{\ell^{(T)}}}
= \frac{\sum_j e^{s_j - m^{(T)}} v_j}{\sum_j e^{s_j - m^{(T)}}}
= \sum_j \operatorname{softmax}(s)_j\, v_j}
$$

Each equality has one justification:

- **The first** is the definition of $\hlc{u^{(T)}}$ and $\hlb{\ell^{(T)}}$,
  which the induction in stages 3 and 4 maintains at every step. The invariant is
  that $\hlb{\ell^{(t)}}$ and $\hlc{u^{(t)}}$ are the exact sums over $S_t$
  shifted by $\hla{m^{(t)}}$, and the update preserves it.
- **The second** is softmax's shift invariance with $c = \hla{m^{(T)}}$, which is
  the true row maximum because $\hla{m}$ tracked the maximum over every block.

No term is dropped, no series is truncated, and no tolerance is chosen. The only
difference from a one-pass softmax is the *order* in which floating-point
additions happen. Floating-point addition isn't associative, so the last bits
can differ: in float32 the labs see agreement to about $10^{-5}$, and in
bfloat16 to about $10^{-2}$. Both are ==rounding, not approximation==.

There's a small irony here. The online version does *more* arithmetic than the
naive one: every block pays for a maximum, an exponential of the correction, and
two rescales it wouldn't otherwise need. It's faster anyway, because the
arithmetic was never the constraint.

## The tiled algorithm

This section turns the four stages into a loop over tiles, which is the shape
the kernel takes. Choose two block sizes: $B_r$ query rows per tile and $B_c$
key columns per tile. The kernel in `engine/kernels/flash_attn_triton.py`
defaults to $B_r = 128$ and $B_c = 64$, named `BLOCK_M` and `BLOCK_N`.

The outer loop runs over query tiles, and it isn't a loop at all. It's the
launch grid, one program per tile, all running at once:

```python
grid = (triton.cdiv(q_len, block_m), batch * heads)
```

The inner loop, inside each program, runs over key blocks. In pseudocode, for one
query tile $Q_i$ of shape $(B_r, d)$:

1. Load $Q_i$ from HBM into registers. It stays there for the whole pass.
2. Set $\hla{m} \leftarrow -\infty$ (shape $(B_r,)$), $\hlb{\ell} \leftarrow 0$
   (shape $(B_r,)$), $\hlc{u} \leftarrow 0$ (shape $(B_r, d)$).
3. For each key block $j$:
   - Load $K_j$ and $V_j$, each $(B_c, d)$, into shared memory.
   - $S \leftarrow Q_i K_j^\top / \sqrt{d}$, shape $(B_r, B_c)$, in registers.
   - Apply the mask: set masked entries of $S$ to $-\infty$.
   - $m_{\text{new}} \leftarrow \max(\hla{m}, \operatorname{rowmax} S)$.
   - $\hld{\alpha} \leftarrow \exp(\hla{m} - m_{\text{new}})$, shape $(B_r,)$.
   - $P \leftarrow \exp(S - m_{\text{new}})$, shape $(B_r, B_c)$.
   - $\hlb{\ell} \leftarrow \hld{\alpha} \odot \hlb{\ell} + \operatorname{rowsum} P$.
   - $\hlc{u} \leftarrow \hld{\alpha} \odot \hlc{u} + P V_j$.
   - $\hla{m} \leftarrow m_{\text{new}}$.
4. $O_i \leftarrow \hlc{u} / \hlb{\ell}$, and store to HBM.

Each pass of step 3 is the "Block 2" step of the worked example, done for 128
query rows at once. Two details are worth naming:

- **The division by $\hlb{\ell}$ happens once, at the end.** Dividing inside the
  loop would be correct, but it would cost $B_r \times d$ divisions per block
  instead of per tile.
- **The accumulator $\hlc{u}$ never leaves registers.** So the kernel writes the
  output to HBM exactly once.

> [!NOTE] Why the loops run this way round
> The original FlashAttention paper runs the loops the other way round: outer
> over key blocks, inner over query blocks. That order forces the partial output
> to be re-read and re-written from HBM on every outer iteration.
> FlashAttention-2 swaps them, which is the order in this chapter, and is why
> this kernel keeps one accumulator in registers for the whole pass.

## What lives where

This section places every tensor at a memory level, because that placement is
the whole design. It uses the lab's geometry: $B_r = 128$, $B_c = 64$,
$d = 128$, and bfloat16 inputs.

**In registers, per program,** each program holds the following:

| Tensor | Shape | Type | Bytes |
|---|---|---|---|
| $Q_i$ | $(128, 128)$ | bfloat16 | 32 KiB |
| $\hlc{u}$ (`acc`) | $(128, 128)$ | float32 | 64 KiB |
| $\hla{m}$, $\hlb{\ell}$ | $(128,)$ each | float32 | 1 KiB |
| $S$, $P$ | $(128, 64)$ | float32 | 32 KiB each, transient |

An A100 SM has 256 KB of register file, shared by every resident program. The
accumulator alone is 64 KiB. That's why the kernel asks for `num_warps=8` when
`head_dim >= 128`: more warps means more threads to spread those registers over.

Push $d$ to 256, this model's real head dimension, and the accumulator doubles
to 128 KiB. That's most of an SM's register file for one program, and it's the
reason a kernel tuned at $d = 128$ might need $B_r = 64$ at $d = 256$.
==Register pressure is a function of the block sizes and the head dimension==,
and nothing else.

**In shared memory, per program,** the kernel stages the $K_j$ and $V_j$ tiles,
so that the loads for block $j+1$ overlap the arithmetic for block $j$. Triton
calls the number of overlapping copies `num_stages`, and the engine computes it
rather than guessing:

```python
element_size = q.element_size()
num_stages = 4
while num_stages > 1 and (
    num_stages * block_n * head_dim * element_size * 2 > SHARED_MEMORY_BYTES
):
    num_stages -= 1
```

`SHARED_MEMORY_BYTES` is 160 KB, a little under the 164 KB that Ampere exposes.
The factor of 2 is for $K$ and $V$.

> [!EXAMPLE] Stages that fit at $B_c = 64$, $d = 128$
> - bfloat16: $4 \times 64 \times 128 \times 2 \times 2 = 131{,}072$ bytes, or
>   128 KiB. Four stages fit.
> - float32: $4 \times 64 \times 128 \times 4 \times 2 = 262{,}144$ bytes. Too
>   big. Three stages need 192 KiB, also too big. Two stages need 128 KiB and
>   fit.

Exceeding the limit raises `OutOfResources` at launch rather than failing
quietly, so it's safe to compute this rather than guess.

**In HBM,** the kernel keeps $Q$, $K$, $V$, and $O$. Nothing else. ==The score
matrix has no address.==

## The kernel, line by line

This section walks through the kernel in
`engine/kernels/flash_attn_triton.py`, so you can map each line to a step of
the algorithm. The first lines find this program's tile and head:

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

`offs_m` holds this tile's absolute query positions, and `offs_d` the head
dimension. Triton works in tiles of indices, not scalars, and every load that
follows broadcasts two of these index vectors into a 2D tile. Next, the program
loads its query tile:

```python
q_ptrs = (
    q_ptr + batch * stride_qb + head * stride_qh
    + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
)
q = tl.load(q_ptrs, mask=offs_m[:, None] < q_len, other=0.0)   # (BLOCK_M, HEAD_DIM)
```

There's one load, outside the loop. The mask handles the last tile when `q_len`
isn't a multiple of `BLOCK_M`. Padded rows load zeros, and the store mask at the
end discards their outputs. Then the program initializes the running
quantities:

```python
m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)   # (BLOCK_M,)
l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # (BLOCK_M,)
acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)       # (BLOCK_M, HEAD_DIM)
```

These are $\hla{m}$, $\hlb{\ell}$, and $\hlc{u}$, initialized as the derivation
requires. All three are float32 even when the inputs are bfloat16, and that
isn't optional: `l_i` accumulates thousands of exponentials, and bfloat16
carries 8 mantissa bits, so ==once the running sum passes 256, a new term of 1.0
stops changing it== at all. Chapter 0a has the format details. The loop body
first scores one key block:

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

`tl.dot` runs on the tensor cores, and `scale` is $1/\sqrt{d}$. The two
`tl.where` calls do different jobs:

- The first kills columns past the end of the sequence. They loaded as zeros,
  which would otherwise produce a score of zero and a probability of $e^{-m}$,
  quietly wrong.
- The second applies causal masking, which the next section covers.

The rest of the loop body updates the running quantities:

```python
    m_new = tl.maximum(m_i, tl.max(scores, axis=1))          # (BLOCK_M,)
    correction = tl.exp(m_i - m_new)                         # (BLOCK_M,)
    p = tl.exp(scores - m_new[:, None])                      # (BLOCK_M, BLOCK_N)

    l_i = l_i * correction + tl.sum(p, axis=1)
    acc = acc * correction[:, None] + tl.dot(p.to(v.dtype), v)
    m_i = m_new
```

These are stages 1 through 4, in order: the streaming maximum, the correction
factor $\hld{\alpha}$, the probabilities under the new shift, and then the two
corrected accumulators. `axis=1` reduces along the key dimension.

> [!WARNING] The wrong axis doesn't crash
> Getting the reduction axis wrong is the single most common bug in a first
> attempt. It normalizes across queries instead of across keys and returns
> plausible-looking garbage.

`correction[:, None]` broadcasts the per-row scalar across the head dimension.
`p.to(v.dtype)` casts the probabilities down so the second `tl.dot` also runs on
the tensor cores, and the accumulation it feeds is still float32. After the
loop, the program normalizes and stores:

```python
acc = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_m[:, None] < q_len)
```

That's one division and one store. The `tl.where` guards a row whose scores were
all masked, where $\hlb{\ell} = 0$. Dividing by zero there would put NaN into an
output that's about to be discarded anyway, and NaNs propagate through
everything downstream.

## Causal masking and the blocks you can skip

This section shows how causal attention halves the work, and where the saving
really comes from. Causal attention lets query position $i$ see key position $j$
only when $j \le i$. Masking the score matrix after computing it saves nothing,
because you already paid for the matmul. ==The saving comes from never
launching the blocks== that are entirely above the diagonal:

```python
offset = kv_len - q_len
hi = tl.minimum(kv_len, (start_m + 1) * BLOCK_M + offset) if IS_CAUSAL else kv_len
```

`offset` aligns the query tile's local row indices with absolute positions. The
last query in tile `start_m` sits at absolute position
`(start_m + 1) * BLOCK_M - 1 + offset`, so no key beyond that can contribute,
and `hi` cuts the loop there.

> [!EXAMPLE] Blocks a square causal pass runs at $L = 8192$
> With $B_r = 128$ and $B_c = 64$, there are $8192/128 = 64$ query tiles and
> $8192/64 = 128$ key blocks, so a full pass is $64 \times 128 = 8192$ block
> pairs. Under the causal bound, query tile $i$ (zero-indexed) runs to
> $(i+1) \times 128$ keys, which is $2(i+1)$ key blocks. The total is the
> following:
>
> $$
> \sum_{i=0}^{63} 2(i+1) = 2 \cdot \frac{64 \cdot 65}{2} = 4160 ,
> $$
>
> or 50.8% of the full count.

Half the work disappears. The extra 0.8% is the diagonal blocks, the ones that
straddle the boundary and still need the elementwise `tl.where`. Every block
strictly below the diagonal needs no mask at all, and a more aggressive kernel
specializes those into a separate loop with no `tl.where` in it.

```viz
14-causal-blocks
```

> [!WARNING] The decode offset passes every prefill test
> The `offset` is the same one from chapter 7, and it's where *decode*, the
> phase that generates one token at a time, goes wrong. During prefill,
> `q_len == kv_len` and the offset is zero. During decode, `q_len` is 1 and
> `kv_len` is the whole cached prefix, so the offset is large, and a kernel that
> assumes zero masks away the entire context. The lab checks a decode step
> against the last row of a full prefill precisely to catch it.

## Grouped queries in the kernel

This section shows how the kernel handles *grouped-query attention* (GQA),
where several query heads share one key-value head. The kernel shares KV heads
by indexing, not by copying. The model has 24 query heads and 4 KV heads, so six
query heads share each KV head.

The caller could materialize the duplication with `repeat_interleave`, which is
what the naive reference does, and that would write six copies of $K$ and $V$ to
HBM. The kernel indexes the shared head instead:

```python
kv_head = head // kv_group      # kv_group = heads // kv_heads = 6
```

Six programs now read the same $K$ and $V$ bytes. Those programs are scheduled
close together, so the reads mostly hit L2, the GPU's 40 MB on-chip cache,
rather than HBM. The effective bandwidth requirement for the KV read drops by a
factor of six.

> [!KEY] The bandwidth saving exists only if the kernel shares
> That's the real benefit of grouped-query attention. A kernel that duplicates
> the heads gets the memory savings in the cache and none of the bandwidth
> savings at read time.

## The IO complexity result

This section makes the byte counting precise, because the asymptotics are the
actual theorem behind FlashAttention. "IO" here means traffic between HBM and
the chip.

**Naive.** The score matrix is written and read a constant number of times, so
HBM traffic, in elements, is the following:

$$
\Theta(L^2 + L d)
$$

The $L^2$ term dominates once $L > d$. At $L = 8192$ and $d = 128$, per head,
four passes over the scores move 537 MB while the inputs are 6.3 MB.

**Tiled.** Let $M$ be the number of key rows that fit in fast on-chip memory, so
a $K$ tile and a $V$ tile together occupy $\Theta(M d)$ elements. Each of the
$L/B_r$ query tiles streams all of $K$ and $V$, which is $2Ld$ elements per
query tile. With $B_r$ bounded by the same on-chip budget, $B_r = \Theta(M)$,
and the total in elements is the following:

$$
\boxed{\frac{L}{B_r} \cdot 2Ld = \Theta\!\left(\frac{L^2 d}{M}\right)}
$$

Written in elements of SRAM, the on-chip memory, rather than rows, call that
$M_{\text{elem}} = M d$. The same bound is then
$\Theta(L^2 d^2 / M_{\text{elem}})$, which is how the paper states it.

Two things follow:

- **The tiled version is still quadratic in $L$.** Anyone who tells you
  FlashAttention makes attention linear in memory traffic is talking about *peak
  allocation*, which is genuinely linear, not about traffic. The keys are re-read
  once per query tile.
- **The improvement factor is $M/d$.** The bigger the on-chip memory relative to
  the head dimension, the fewer times you re-read $K$.

> [!EXAMPLE] The improvement factor on an A100
> Of the 164 KB of shared memory per SM, the kernel budgets 160 KB. At $d = 128$
> in bfloat16, a key row is 256 bytes and a key-plus-value row pair is 512 bytes,
> so roughly $M = 320$ row pairs fit. Against $d = 128$, that predicts a factor
> of about 2.5 in HBM traffic, which is far less than the speedups people report.

The gap is L2. At $L = 8192$ and $d = 128$, one head's $K$ and $V$ are 4.2 MB
together, and the lab's two KV heads are 8.4 MB, well under the A100's 40 MB L2. The
re-reads are quadratic in count, but most of them never reach HBM.

This is the same effect chapter 13 measured for fused RMSNorm, where a saving
predicted by the byte count vanished because the data never left cache. Here it
works in your favor instead.

## Speed: 0.63x, and why

This section explains why the teaching kernel is slower than the vendor kernel,
and by how much. The README records the following measurements on the target
hardware:

| Measurement | Result |
|---|---|
| Triton FlashAttention against the vendor kernel | 0.63x at 8k tokens |
| FlashAttention peak memory against naive | 27.5x less |

The kernel is about a third slower than what
`F.scaled_dot_product_attention` dispatches to. That deserves an honest
explanation rather than a shrug.

Start with the arithmetic floor, for $L = 8192$, 8 heads, $d = 128$, and causal
masking. The two matmuls cost the following per head:

$$
2 \times 2 L^2 d = 4 \times 8192^2 \times 128 = 34.4 \text{ GFLOP}
$$

Causality halves that to 17.2 GFLOP, and $\times 8$ heads gives 137 GFLOP. The
A100's tensor cores peak at 312 TFLOP/s, trillion floating-point operations per
second, for bfloat16, so the floor is 0.44 ms.

The engine's docstring records 1.64 ms for this kernel at that geometry with the
$128 \times 64$ tiling, against 3.35 ms for a $64 \times 64$ tiling:

- This kernel reaches about 27% of peak.
- The vendor kernel, at 0.63x of 1.64 ms, lands near 1 ms, or about 43%.

Neither is close to peak, which is normal for attention. What needs explaining
is the gap between them, and four things account for most of it:

- **Non-matmul work on the wrong units.** Every block computes $B_r \times B_c$
  exponentials. Over a causal pass, that's $L^2/2$ exponentials per head, or 268
  million across 8 heads for one call. Exponentials run on the special function
  units, not the tensor cores, and they don't overlap the matmuls for free.
  FlashAttention-2's main contribution was reducing exactly this class of work.
- **The rescale in the inner loop.** `acc = acc * correction[:, None] + ...`
  touches $B_r \times d$ float32 registers on every iteration. A tuned kernel
  defers more of that work, keeping the correction as a scalar applied at tile
  boundaries rather than a full accumulator multiply each time.
- **Scheduling.** cuDNN and CUTLASS, NVIDIA's own kernel libraries, use warp
  specialization, where some warps do nothing but issue asynchronous copies
  while others do nothing but matmul. They add hand-tuned pipelining and, on
  newer hardware, TMA descriptors, which drive a dedicated copy engine. Triton
  generates a good generic schedule, and good generic loses to hand-tuned by
  tens of percent.
- **Autotuning.** The vendor kernel picks tile sizes per shape, per dtype, and
  per head dimension, from a table built by measurement. This kernel has two
  defaults. The measured $128 \times 64$ against $64 \times 64$ gap, 1.64 ms
  against 3.35 ms, is a factor of two from one parameter, and it shows how much
  is on the table.

Landing ==within 2x of a vendor kernel== with 40 lines you can read in one
sitting is a good result. The lab's threshold is 0.33x for that reason.

## Memory: 27.5x

This section checks the memory result, which isn't a near miss, and it's the
one that changes what you can serve. The lab measures peak allocation at
$L = 4096$, 8 heads, and $d = 128$.

> [!EXAMPLE] Check the 27.5x by hand
> The naive path materializes the scores in float32:
>
> $$
> 8 \times 4096 \times 4096 \times 4 = 536{,}870{,}912 \text{ bytes} = 512 \text{ MiB}.
> $$
>
> FlashAttention's live set is $Q$, $K$, $V$, and $O$: 8 MiB for $Q$, 2 MiB each
> for $K$ and $V$ with two KV heads, and 8 MiB for $O$, so 20 MiB. The ratio of
> $512 + \epsilon$ to 20 is 27.5 when $\epsilon$ is the roughly 38 MiB of float32
> copies the reference path makes of its inputs.

The measurement and the arithmetic agree, which is a good sign that you
understand what the measurement measured.

Peak memory, not clock time, is why every serving stack uses FlashAttention. A
27.5x reduction in the attention working set is ==the difference between 8k
context and 200k context== on the same card.

## What goes wrong

- **The reduction axis.** `tl.sum(p, axis=1)` sums over keys, and `axis=0` sums
  over queries. Both run. The second gives a result that's smooth, finite, and
  wrong, and it survives a shape check because the tile is square when
  `BLOCK_M == BLOCK_N`. Test with a non-square tiling, and the shape error
  appears immediately.
- **Forgetting the out-of-range mask on scores.** Columns past `kv_len` load as
  zeros, and $e^{0 - m}$ isn't zero. Without
  `tl.where(cols[None, :] < kv_len, ...)`, those phantom keys take a share of the
  probability mass. The symptom is an error that grows as the sequence length
  moves further from a multiple of `BLOCK_N`, which is why the lab includes a
  length of 130.
- **bfloat16 accumulators.** Keeping `l_i` or `acc` in bfloat16 saves registers
  and destroys the result at long context. The sum stops growing once the running
  total is 256 times larger than the new term. The symptom is an error that grows
  with `kv_len` and looks like a masking bug.
- **A fully masked row.** If every score in a row is $-\infty$, then
  $\hla{m^{(t)}} = -\infty$ and the correction computes
  $\exp(-\infty + \infty)$, which is NaN. Causal masking as written never
  produces one, because query row $i$ always sees key 0, which is in the first
  block. A custom mask can, and the NaN spreads to the whole output. The
  `l_i == 0.0` guard catches the division but not this.
- **The decode offset.** Covered earlier, and worth repeating: it's the bug that
  passes prefill tests and breaks in production.
- **TF32.** On float32 inputs, `tl.dot` uses the TF32 tensor cores, which keep
  10 mantissa bits. Results differ from a true float32 matmul by around
  $10^{-3}$ relative. That's why the lab's float32 tolerance is
  $5 \times 10^{-3}$ and not $10^{-6}$.

> [!RECAP]
> - Naive attention moves an $L^2$ score matrix through HBM four times, which is
>   97% of its traffic at 8192 tokens.
> - The online softmax keeps $\hla{m}$, $\hlb{\ell}$, and $\hlc{u}$ per query
>   row, and rescales $\hlb{\ell}$ and $\hlc{u}$ by
>   $\hld{\alpha} = e^{m_{\text{old}} - m_{\text{new}}} \in (0, 1]$ whenever the
>   maximum rises.
> - The result is exact: $\hlc{u^{(T)}} / \hlb{\ell^{(T)}}$ is the softmax
>   output, up to rounding.
> - $Q_i$ and the float32 accumulator live in registers, $K_j$ and $V_j$ in
>   shared memory, and only $Q$, $K$, $V$, and $O$ in HBM.
> - Traffic stays quadratic, $\Theta(L^2 d / M)$, but L2 absorbs most re-reads.
>   The kernel runs at 0.63x of the vendor kernel and uses 27.5x less memory.

## Check your understanding

> [!QUESTION] If the correction factor is $e^{m_{\text{old}} - m_{\text{new}}}$ and the maximum only rises, why is there no matching correction when the maximum doesn't move?
> There is; it's invisible. When $m^{(t+1)} = m^{(t)}$, the factor is $e^0 = 1$,
> and multiplying by it changes nothing. The kernel computes it anyway rather
> than branching, because a branch would diverge across the rows of a tile and
> cost more than the multiply.

> [!QUESTION] Why keep a running maximum at all? The algorithm would be simpler without it.
> Correctness would survive; range wouldn't. float32 overflows above $e^{88}$,
> and trained models produce score outliers far larger than a random-normal
> analysis predicts. Without the shift, one large score makes the numerator and
> the denominator both infinite and the output NaN. The maximum guarantees every
> exponent is at most zero, whatever the scores turn out to be.

> [!QUESTION] The tiled version re-reads $K$ and $V$ once per query tile, so it moves more key bytes than the naive version. Why is it still faster?
> Because the naive version's $L^2$ score traffic dwarfs both. At $L = 8192$ and
> $d = 128$, one head's $K$ and $V$ are 4.2 MB and its score matrix is 134 MB
> touched four times. Re-reading 4.2 MB sixty-four times is 268 MB of requests,
> most of which hit a 40 MB L2, against 537 MB of score traffic that can't hit
> anything because it's streamed once and never reused.

> [!QUESTION] Does the block size change the answer?
> No. It changes speed, register pressure, and how many blocks causality lets you
> skip, but the online softmax is exact for any $B_r$ and $B_c$, including
> $B_c = 1$. The lab checks a sequence length of 130 against a block size of 128
> for exactly this reason: the tail block has 2 valid columns, and the answer
> must still be right.

## Lab

> [!TRY]
> Write the FlashAttention forward pass in Triton. You pass when it matches a
> float32 naive implementation in all six cases plus the decode-against-prefill
> check, runs at least 0.33x of the vendor kernel at 8192 tokens, and uses at
> least 3x less peak memory than the naive path at 4096 tokens.

Match the signature in `starter.py`: `flash_attention(q, k, v, causal=True,
scale=None)`, with `q` of shape `(batch, heads, q_len, head_dim)` and `k`, `v`
of shape `(batch, kv_heads, kv_len, head_dim)`, where `kv_heads` divides
`heads`.

The harness checks correctness against a float32 naive implementation across six
cases, each of which must land within $5 \times 10^{-3}$ absolute:

- Non-causal square.
- Causal square.
- A single decode query behind a 1024-token prefix.
- A 64-token chunk behind a 512-token prefix.
- A length of 130 that isn't a multiple of the block size.
- A 1024-token causal pass.

A bfloat16 run must stay within $5 \times 10^{-2}$. One more check compares a
single decode query against the last row of a full prefill, which is where an
incorrect causal offset shows up.

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
