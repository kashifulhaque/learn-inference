---
title: FlashAttention
slug: 14-flash-attention
part: "Part 4 — Kernels"
summary: Deriving the online softmax, then writing the tiled kernel that never materializes a score matrix.
minutes: 100
gpu: true
objectives:
  - Derive the online softmax rescaling and prove it gives the same result.
  - Write a tiled attention kernel in Triton with correct causal masking.
  - Explain why FlashAttention is faster despite doing more arithmetic.
lab: 14-flash-attention
---

# FlashAttention

Naive attention writes an intermediate that's larger than the model. At 8k context
with 24 heads, the score matrix is 12 GB in bfloat16 for one sequence, and it
crosses the memory bus three times: written after the matmul, read for the
softmax, read again for the value multiply.

FlashAttention never writes it. The arithmetic is identical — this is an exact
algorithm, not an approximation — but the scores stay in registers and shared
memory.

## The problem with softmax

Tiling the two matrix multiplies is easy. The softmax between them is not, because
it needs a normalizer over the whole row, and you don't have the whole row until
you've seen every key.

The answer is to compute the softmax incrementally and fix up the result as you
go.

## The online softmax

Process keys in tiles. Maintain, for each query row, a running maximum `m`, a
running denominator `l`, and a running weighted sum of values `acc`.

When a new tile arrives with scores `s`:

```text
m_new = max(m, max(s))
correction = exp(m - m_new)
l_new = l * correction + sum(exp(s - m_new))
acc_new = acc * correction + exp(s - m_new) @ v_tile
```

The key line is the correction. Everything accumulated so far was scaled by
`exp(-m)`; the new maximum requires `exp(-m_new)`. Multiplying by
`exp(m - m_new)` converts one to the other exactly.

That the result equals the one-pass softmax is straightforward. After processing
every tile:

```text
acc = sum_j exp(s_j - m_final) v_j
l   = sum_j exp(s_j - m_final)
```

so `acc / l` is the softmax-weighted sum of values, with the maximum subtracted
for stability. The shift cancels, exactly as in the one-pass version. No
approximation anywhere.

## The kernel

One program handles one tile of queries against all keys:

```python
m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

for start_n in range(0, hi, BLOCK_N):
    k = tl.load(k_ptrs, mask=cols[:, None] < kv_len, other=0.0)
    v = tl.load(v_ptrs, mask=cols[:, None] < kv_len, other=0.0)

    scores = tl.dot(q, tl.trans(k)) * scale
    if IS_CAUSAL:
        scores = tl.where(cols[None, :] <= offs_m[:, None] + offset,
                          scores, float("-inf"))

    m_new = tl.maximum(m_i, tl.max(scores, axis=1))
    correction = tl.exp(m_i - m_new)
    p = tl.exp(scores - m_new[:, None])

    l_i = l_i * correction + tl.sum(p, axis=1)
    acc = acc * correction[:, None] + tl.dot(p.to(v.dtype), v)
    m_i = m_new

acc = acc / l_i[:, None]
```

`q` is loaded once, outside the loop, and stays in registers for the whole pass
over the keys. That's the reuse the whole design is built around.

The accumulators are float32 even when the inputs are bfloat16. The running sum
`l_i` accumulates thousands of exponentials, and bfloat16 loses the tail of that
reduction.

## Causal masking, twice

Causal attention masks about half the score matrix. Masking it after computing it
saves nothing, so the kernel skips whole tiles:

```python
offset = kv_len - q_len
hi = tl.minimum(kv_len, (start_m + 1) * BLOCK_M + offset)
```

Tiles entirely past the diagonal are never loaded. Only the tiles that straddle it
need the elementwise `tl.where`. For a square causal problem this halves the work.

The `offset` is the same one from chapter 7. During prefill it's zero. During
decode with a cached prefix it isn't, and getting it wrong gives you a kernel that
passes every prefill test and fails in production.

## Grouped queries in the kernel

The kernel indexes the shared KV head directly, rather than making the caller
duplicate it:

```python
kv_head = head // kv_group
```

Six query heads read the same KV head, so those reads hit L2 and the effective
bandwidth requirement drops by six. This is the real benefit of GQA, and it exists
only if the kernel does the sharing.

## Why it's faster

FlashAttention does *more* arithmetic than the naive version — the rescaling is
extra work — and it's several times faster anyway. Count bytes.

Naive, at sequence length *N*, head dimension *d*, one head:

```text
Read Q, K, V:        3 N d x 2 bytes
Write scores:        N² x 2
Read scores:         N² x 2
Write probabilities: N² x 2
Read probabilities:  N² x 2
Write output:        N d x 2
```

The `N²` terms dominate for any interesting *N*. Flash:

```text
Read Q, K, V: 3 N d x 2
Write output:   N d x 2
```

Linear in *N*, not quadratic. At `N = 8192` and `d = 256` that's a factor of
roughly 8 in bytes moved — and the operation was memory bound, so the time follows.

Memory follows too. Naive attention's peak allocation is O(N²) and it's what makes
long-context prefill run out of memory. Flash is O(N), and the score matrix never
exists.

## Checking it

Compare against the naive implementation in float32 on short sequences, where the
naive version fits. Check:

- Non-causal and causal.
- `q_len == kv_len`, the prefill case.
- `q_len == 1` with a long `kv_len`, the decode case, which exercises the offset.
- A sequence length that isn't a multiple of `BLOCK_M`, which exercises the masks.

In bfloat16 expect agreement to about 1e-2 absolute; in float32, 1e-5.

## Lab

Write the FlashAttention forward pass in Triton. The harness checks correctness
against the naive path across all four cases, then measures against
`F.scaled_dot_product_attention` at sequence lengths from 512 to 8192, and reports
peak memory for both.

## Further reading

- [FlashAttention: fast and memory-efficient exact attention with IO-awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: faster attention with better parallelism and work partitioning](https://arxiv.org/abs/2307.08691)
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
