---
title: The KV cache
slug: 09-the-kv-cache
part: "Part 3 — Making it fast"
summary: Deriving the quadratic-to-linear change, and paying for it in two different data structures.
minutes: 90
gpu: true
objectives:
  - Derive the total work of cached and uncached decode in closed form.
  - Say exactly which tensors are cacheable and why the others are not.
  - Explain why the cache stores post-RoPE keys.
  - Choose a cache layout and justify it from coalescing and read volume.
  - Implement a cache that holds both KV entries and recurrent state.
  - Compute the memory cost of both kinds of state at a given context and batch.
lab: 09-kv-cache
---

# The KV cache

A transformer is a function of the whole sequence. Nothing in its definition
says that generating token $n+1$ should be cheaper than generating token $n$.
Run the definition literally and it is not: each new token means another forward
pass over everything written so far.

The cache is the observation that almost all of that work is a recomputation of
something that cannot have changed. It is the single largest speedup in the
engine and the first one to implement.

## Before you start

This chapter assumes:

- **Chapter 0a's GPU vocabulary**, particularly HBM, L2, and what a coalesced
  read is. The layout section turns on coalescing.
- **Chapter 2's memory arithmetic**: 64 KiB of KV per token, 147.8 MiB of
  recurrent state per sequence, and the break-even derivation. This chapter
  re-derives the break-even point, so you do not need it memorized.
- **Chapters 6 and 7**: what the gated delta rule's state is, and what
  grouped-query attention reads.
- **Big-O notation**, and the two sums $\sum_{j=0}^{n-1} j = \tfrac{n(n-1)}{2}$
  and $\sum_{j=0}^{n-1} c = cn$.
- **PyTorch views versus copies.** `t[:, :, :stop]` is a view and allocates
  nothing. `torch.cat` allocates a new tensor and copies both inputs into it.
  The difference is the whole chapter.

Throughout, $p$ is the prompt length in tokens, $n$ is the number of tokens
generated, and $L$ is the total context at some point in time.

## The quadratic

Measure work in *token-forward-passes*: one token passing through all 64 layers
once. It is a crude unit, but it is proportional to FLOPs for everything except
attention itself, and it is exactly what the cache changes.

### Without a cache

To produce the first new token you run a forward pass over the $p$ prompt
tokens. To produce the second you append the first and run a pass over $p+1$
tokens. In general, the $j$-th generated token needs a pass over $p + j - 1$
tokens, for $j = 1, \ldots, n$. Total work:

$$
W_{\text{none}}(p, n) = \sum_{j=1}^{n} (p + j - 1) = \sum_{j=0}^{n-1} (p + j)
$$

Split the sum:

$$
W_{\text{none}}(p, n) = \sum_{j=0}^{n-1} p + \sum_{j=0}^{n-1} j = np + \frac{n(n-1)}{2}
$$

The leading term in $n$ is $n^{2}/2$, so $W_{\text{none}} = O(n^{2})$ for a
fixed prompt, and $O((p+n)^2)$ if the prompt grows with the output.

### With a cache

The prefill pass processes all $p$ prompt tokens at once and produces the first
generated token. Each subsequent step processes exactly one token, and there are
$n - 1$ of them:

$$
W_{\text{cache}}(p, n) = p + (n - 1) = O(p + n)
$$

Linear. The ratio is

$$
\frac{W_{\text{none}}}{W_{\text{cache}}} = \frac{np + \tfrac{n(n-1)}{2}}{p + n - 1}
\;\xrightarrow[\;n \gg p\;]{}\; \frac{n}{2} .
$$

For long generations the cache saves a factor of about $n/2$. That is the whole
result, and everything below is about what it costs.

### Real numbers for this model

Take a 512-token prompt and 128 generated tokens.

$$
W_{\text{none}} = 128 \cdot 512 + \frac{128 \cdot 127}{2} = 65{,}536 + 8{,}128 = 73{,}664
$$

$$
W_{\text{cache}} = 512 + 127 = 639
$$

| | Forward passes | Token-forward-passes |
|---|---|---|
| No cache | 128 | 73,664 |
| With cache | 1 prefill + 127 decode | 639 |

A ratio of $73{,}664 / 639 = 115.3$.

Turn that into FLOPs. A dense forward pass over one token costs about $2N$
FLOPs, where $N = 26.9 \times 10^{9}$ is the parameter count — one multiply and
one add per parameter. So $2N = 53.8$ GFLOP per token-forward-pass, and

$$
73{,}664 \times 53.8\ \text{GFLOP} = 3.96\ \text{PFLOP}
$$

against $639 \times 53.8\ \text{GFLOP} = 34.4$ TFLOP with the cache.

### Why the wall-clock ratio is smaller

Work is not time. The uncached passes each cover several hundred tokens, which
puts them well above the ridge point from chapter 10, so they run near the
tensor cores' peak. Cached decode steps cover one token each, which puts them
far below it, so they run at memory bandwidth.

Comparing the roofline floors for the case above, at the A100's rated 1935 GB/s
and 312 TFLOP/s:

| | Bound by | Floor |
|---|---|---|
| Uncached, 128 passes averaging 575 tokens | Compute | $3.96\times10^{15} / 312\times10^{12} = 12.7$ s |
| Prefill, 512 tokens | Compute | 88 ms |
| 127 decode steps, each reading 53.8 GB of weights | Memory | $127 \times 27.8\ \text{ms} = 3.53$ s |

So the arithmetic predicts about a $3.5\times$ wall-clock speedup at this size,
not $115\times$. The gap closes as $n$ grows: uncached time scales as $n^{2}$
while cached time scales as $n$, so the ratio grows linearly in $n$. Setting the
two expressions equal and solving gives a ratio of roughly

$$
\frac{n}{2} \cdot \frac{0.172\ \text{ms}}{27.8\ \text{ms}} \approx 0.0031\,n ,
$$

where 0.172 ms is one token-forward-pass of arithmetic at peak and 27.8 ms is
one decode step of weight reading. A $20\times$ speedup needs roughly 6500
generated tokens. Every figure in this section is arithmetic from the roofline,
not a measurement; the lab has you measure the work ratio, which is the part
that does not depend on how good your kernels are.

## What to cache

### Keys and values, for full-attention layers only

For a full-attention layer, the score between query at position $m$ and key at
position $n$ is

$$
s_{mn} = \frac{q_m^{\top} k_n}{\sqrt{d}} ,
$$

and $k_n$ depends only on the residual stream at position $n$. Causal masking
guarantees the residual stream at position $n$ is a function of tokens $0$
through $n$ only. So once token $n$ has been processed, $k_n$ and $v_n$ are
fixed forever: the key computed for token 5 is the same whether the sequence is
6 tokens long or 6000.

That is the entire justification, and it is worth being precise about it,
because it is exactly what a non-causal model would lose.

**Queries are not cached.** A query is consumed by the step that produces it and
never read again. Caching $q_m$ would let you recompute row $m$ of the attention
matrix later, which no one wants.

**Attention outputs are not cached either.** The output at position $m$ is a
weighted sum over positions $0 \ldots m$, so it is already final — but nothing
downstream ever asks for it again. The residual stream from an earlier position
is not re-read by a later step.

### The cache holds post-RoPE keys

RoPE rotates each key by a block-diagonal rotation $R_n$ whose angles depend on
the absolute position $n$. The engine applies it before the cache write:

```python
k = apply_rotary_partial(k, cos, sin, self.rotary_dim)
if kv_cache is not None:
    k, v = kv_cache.append(layer_idx, k, v)
```

The reason this is safe is the identity RoPE was designed around. For rotation
matrices, $R_m^{\top} R_n = R_{n-m}$, so

$$
(R_m q_m)^{\top} (R_n k_n) = q_m^{\top} R_m^{\top} R_n k_n = q_m^{\top} R_{n-m} k_n .
$$

The score depends only on the *difference* $n - m$, and $n$ is fixed the moment
token $n$ is written. Whatever query arrives later, it applies its own $R_m$ to
itself; the stored $R_n k_n$ never needs to change.

Store pre-RoPE keys instead and every decode step has to rotate the entire
prefix before using it: $O(L)$ extra arithmetic, an extra $O(L)$ read, and an
extra buffer to hold the result. You get nothing for it.

Values are never rotated, so the question does not arise for them.

### Recurrent state and the convolution window, for linear layers

The 48 linear-attention layers keep two things per sequence:

- The delta-rule state $S$, shape `(batch, 48, 128, 128)`, in float32.
- The causal convolution window, shape `(batch, 10240, 3)`, in bfloat16 — the
  previous $k - 1 = 3$ steps of a four-tap kernel.

Neither grows with sequence length, and neither is appended to. Both are
overwritten in full every step. That difference runs through the rest of this
chapter.

## Two data structures, not one

`HybridCache` holds four dictionaries, keyed by layer index:

```python
self.k_cache: dict[int, Tensor]      # 16 full-attention layers
self.v_cache: dict[int, Tensor]
self.states: dict[int, Tensor]       # 48 linear-attention layers
self.conv_states: dict[int, Tensor]
```

A dense list with 48 unused slots would work and waste nothing, but the
dictionary makes the hybrid structure visible in a debugger: `sorted(k_cache)`
prints `[3, 7, 11, ..., 63]` and you can see at a glance that you built the
right thing.

Calling all four "the cache" hides that they behave nothing alike:

| | KV cache | Recurrent state |
|---|---|---|
| Layers | 16 | 48 |
| Size | Grows by 4 KiB per token per layer | Fixed at 3.08 MiB per layer |
| Written per step | One token's slice | The whole tensor |
| Read per step | The whole prefix | The whole tensor |
| Can you recover position $j$? | Yes, it is still there | No, it has been folded in |
| Dtype | bfloat16 | float32 |
| Reset | Move the cursor to 0 | Zero the tensor |

The last two rows are where engineering consequences live.

**Float32 for the state.** The recurrence multiplies the state by a decay
$\alpha \in (0,1)$ thousands of times. In bfloat16, with 8 significand bits, the
repeated rounding of that product accumulates into visible drift over a long
sequence. The KV cache has no such problem because nothing is ever multiplied
into an entry after it is written.

**No rewind.** You can drop the last $r$ KV entries by moving the cursor back
$r$ places; the earlier entries are untouched. You cannot do that to the
recurrent state, because position $j$'s contribution was added into $S$ and the
addition is not invertible in floating point. That single fact costs you three
things later: prefix sharing across requests (chapter 15) works for KV and not
for state, preemption-by-recompute must replay the linear layers from the start,
and speculative decoding (chapter 20) must snapshot the state before proposing
rather than rolling it back afterwards.

## Layout

The cache is a four-dimensional tensor and there are two sensible orderings.
`HybridCache` uses the first:

```python
shape = (batch_size, config.num_key_value_heads, max_seq_len, config.head_dim)
```

That is `(batch, heads, seq, dim)`, or BHSD. The alternative is
`(batch, seq, heads, dim)`, BSHD. PyTorch stores tensors row-major, so the last
axis is contiguous and each earlier axis strides over everything after it.

**BHSD.** One head's keys occupy a contiguous $L \times 256$ block. Reading them
for the score matmul is one unbroken sequential stream, which is what the DRAM
row buffer and the L2 prefetcher reward. The block is also a dense GEMM operand:
you can hand it to cuBLAS with a leading dimension of 256 and no copy. Writing
one token's key touches 256 contiguous elements — 512 bytes — for each of the 4
KV heads, at four addresses separated by `max_seq_len * head_dim` elements. Four
small writes.

**BSHD.** One token's keys for all heads are contiguous:
$4 \times 256 = 1024$ elements, 2048 bytes, one run. The write becomes a single
coalesced store. Reading head $h$ across the sequence now strides by 1024
elements, so that head's keys arrive as $L$ separate 512-byte runs with 1536-byte
gaps between them.

Be precise about what that costs. At this geometry it is *not* wasted bytes: 256
bfloat16 values is 512 bytes, exactly four 128-byte cache lines, so an aligned
read of one head-token fetches four lines and uses all of them in either layout.
What BSHD loses is the length of the sequential run, which costs DRAM page
locality and prefetch efficiency rather than raw bandwidth. Shrink `head_dim` to
64, as many models do, and one head-token becomes 128 bytes — a single line — at
which point the strided layout starts costing real bandwidth too. The rule from
chapter 0a still holds: what matters is how much of each fetched line you use,
and how long the stream between jumps is.

Which one wins is decided by the ratio of reads to writes, and it is not close.
At 32k context, one decode step in one layer reads

$$
2 \cdot 4 \cdot 32768 \cdot 256 \cdot 2\ \text{bytes} = 134{,}217{,}728 = 128\ \text{MiB}
$$

and writes

$$
2 \cdot 4 \cdot 1 \cdot 256 \cdot 2\ \text{bytes} = 4096\ \text{bytes} = 4\ \text{KiB} .
$$

A ratio of 32,768 to 1. Optimize the read; the write is free at any layout. BHSD
it is.

The choice flips once the cache is paged. `PagedKVCache` in chapter 15 stores
blocks as `(num_blocks, block_size, kv_heads, head_dim)` — BSHD within a block —
because a paged write is a scatter to arbitrary slots, and scattering whole
tokens is cheaper than scattering four per-head fragments. The block is small
enough that the strided read inside it stays in L2.

## Preallocate, do not concatenate

The obvious implementation appends:

```python
self.k = torch.cat([self.k, new_k], dim=2)   # don't
```

Every `cat` allocates a new tensor and copies both inputs into it. At step $j$
the cache holds $p + j$ tokens, so the total copied over $n$ steps is the same
quadratic sum as before:

$$
\sum_{j=0}^{n-1} (p + j) = np + \frac{n(n-1)}{2}
$$

tokens of cache. For $p = 512$, $n = 128$ that is 73,664 tokens at 64 KiB each,
or 4.5 GiB read and 4.5 GiB written — about 8 ms of pure memcpy at the measured
1275 GB/s, to accomplish nothing. You reintroduced the quadratic you had
removed, in the data movement instead of the arithmetic.

The allocator damage is worse than the copying. Each `cat` requests a block one
token larger than the last and frees the previous one, so the caching allocator
accumulates a ladder of almost-but-not-quite reusable blocks. The symptom is an
out-of-memory error at 60 GB on an 80 GB card, with `torch.cuda.memory_reserved`
far above `memory_allocated`.

Allocate once at the maximum length and write into a slice:

```python
def append(self, layer_idx, k, v):
    seq = k.shape[2]
    start, stop = self.length, self.length + seq
    if stop > self.max_seq_len:
        raise RuntimeError(
            f"Cache overflow: {stop} tokens requested, "
            f"capacity {self.max_seq_len}."
        )
    self.k_cache[layer_idx][:, :, start:stop] = k      # in-place write
    self.v_cache[layer_idx][:, :, start:stop] = v
    return (self.k_cache[layer_idx][:, :, :stop],      # views
            self.v_cache[layer_idx][:, :, :stop])
```

The write is in place. The return is two views. Neither allocates, and the
returned tensors are exactly the shape the attention kernel wants:
`(batch, kv_heads, stop, head_dim)`.

Raise on overflow rather than growing. A cache that silently reallocates is a
cache that will silently reallocate in production, at the worst moment, under
the largest batch. The scheduler in chapter 16 is the component that should be
deciding what to do when capacity runs out, and it can only do that if the cache
tells it.

The cost of preallocation is reservation. A batch of 32 sequences that might
reach 32k tokens reserves

$$
32 \times 32768 \times 65{,}536\ \text{bytes} = 68.7\ \text{GB} = 64\ \text{GiB}
$$

even if every sequence stops at 200 tokens. Chapter 15 fixes that with paging.
Contiguous preallocation is still the right thing to build first: it is fifty
lines, it is correct by inspection, and it is the reference the paged version
gets checked against.

## Advance the cursor once

The cache length must advance after all 64 layers have run, not inside each
layer:

```python
for layer in self.layers:
    x = layer(x, cos, sin, cache=cache)
if cache is not None:
    cache.advance(seq)
```

Every layer in one forward pass writes at the same position, because they are
all processing the same tokens. Advancing per layer makes layer 7 believe the
sequence is one token longer than layer 3 does, which corrupts both the write
offset and the causal mask offset $L - T$. The symptom is a model that is
correct during prefill and wrong from the second decode step onwards — the
signature in chapter 8's bisection table.

## The memory arithmetic

Two independent terms. The KV cache costs 64 KiB per token of context, per
sequence:

$$
\text{KV bytes} = B \cdot L \cdot 65{,}536
$$

The recurrent state costs a fixed 147.8 MiB per sequence, independent of $L$:

$$
\text{state bytes} = B \cdot 154{,}927{,}104
$$

Per sequence, in GiB:

| Context | KV | State | Total |
|---|---|---|---|
| 1k | 0.06 | 0.14 | 0.21 |
| 4k | 0.25 | 0.14 | 0.39 |
| 32k | 2.00 | 0.14 | 2.14 |
| 128k | 8.00 | 0.14 | 8.14 |
| 262k | 16.00 | 0.14 | 16.14 |

Multiply by batch size for the total, since both terms are per sequence:

| | Batch 1 | Batch 8 | Batch 32 | Batch 64 |
|---|---|---|---|---|
| 4k context | 0.39 GiB | 3.2 GiB | 12.6 GiB | 25.2 GiB |
| 32k context | 2.14 GiB | 17.1 GiB | 68.6 GiB | 137 GiB |

Chapter 2's budget leaves roughly 19 GiB for cache after the weights, the CUDA
context, and an activation workspace. Read the table against that number: batch
8 at 32k fits with nothing to spare, batch 32 at 4k fits comfortably, batch 64
at 4k does not.

That last entry is the one worth pausing on. Chapter 2 quotes 19 GiB as "304k
tokens", which counts only the KV term. At batch 64 the fixed recurrent state is
$64 \times 147.8$ MiB $= 9.2$ GiB before a single token of context exists — half
the budget, spent on state that does not grow. The fixed cost is fixed *per
sequence*, not per GPU, so it scales with batch even though it does not scale
with context.

### Break-even against an all-full-attention model

An all-full-attention version of this geometry would pay $256$ KiB per token
across all 64 layers and carry no fixed state. The hybrid pays 64 KiB per token
plus 147.8 MiB. Setting them equal:

$$
262{,}144\,L = 65{,}536\,L + 154{,}927{,}104
$$

$$
196{,}608\,L = 154{,}927{,}104
$$

$$
L = \frac{154{,}927{,}104}{196{,}608} = 788\ \text{tokens}
$$

Below 788 tokens of context the fixed state costs more than the KV entries it
replaces. Above it the hybrid wins, and the margin grows by 192 KiB per further
token without limit. At 32k context the hybrid needs 2.14 GiB per sequence
against 8.0 GiB; at 128k, 8.14 GiB against 32 GiB.

`ModelConfig.hybrid_breakeven_tokens` computes this for any config, and chapter
2's lab has you derive it.

### What decode actually reads

Memory footprint and memory traffic are different questions. Per decode step,
per sequence, the engine reads:

- The weights: 53.8 GB, once, shared across the whole batch.
- The KV cache: $65{,}536 \cdot L$ bytes, read in full.
- The recurrent state: 147.8 MiB read and 147.8 MiB written, because every
  linear layer overwrites its state.

The state's traffic is $2 \times 154{,}927{,}104 = 309{,}854{,}208$ bytes per
sequence per step. The KV read matches that when

$$
65{,}536\,L = 309{,}854{,}208 \quad\Longrightarrow\quad L \approx 4728\ \text{tokens} .
$$

Below about 4.7k context, the 48 linear-attention layers move more bytes per
decode step than the 16 full-attention ones do. The hybrid design shifts memory
pressure; it does not remove it. That is also why quantizing the KV cache to
int8 (chapter 18) helps less at short context than the headline suggests.

## The state carry

The KV path is append-only and hard to get subtly wrong: either the prefix is
there or it is not. The linear path is the opposite.

`delta_rule_chunked` takes a `state` argument and returns the final state.
Prefill passes `None`, computes the whole chunked recurrence, and stores the
result. Each decode step passes the stored state back in, runs one
`delta_rule_step`, and stores the new one:

```python
prev_state = cache.recurrent_state(layer_idx) if cache is not None else None
# ... run the recurrence ...
if cache is not None:
    cache.set_linear_state(layer_idx, new_state, new_conv)
```

Forget to read `prev_state` and the model prefills correctly and then generates
as if the prompt never happened — fluent, on-topic for about one sentence, then
unmoored. Forget to write `new_state` and every decode step starts from the
prefill state, so the model repeats itself. Both produce output that reads as
plausible text, which is why the prefill-plus-decode equivalence test from
chapter 8 is the one that catches them.

The convolution window has the same structure and is easier to overlook: it is
the last three steps of input to the four-tap causal convolution, and it must be
shifted forward every step. `causal_depthwise_conv1d` returns the updated window
alongside its output; store both or neither.

## Check your understanding

**You generate 2000 tokens from a 100-token prompt. How many
token-forward-passes does each approach do?**

Uncached: $2000 \cdot 100 + \tfrac{2000 \cdot 1999}{2} = 200{,}000 +
1{,}999{,}000 = 2{,}199{,}000$. Cached: $100 + 1999 = 2099$. A ratio of 1048,
close to the $n/2 = 1000$ asymptote, because $n \gg p$ here.

**Why does the KV cache hold keys after RoPE but the query is rotated fresh
every step?**

Because the stored key's rotation angle depends on its own fixed position,
while the query's depends on the position of the step being computed, which is
different every time. $R_n k_n$ is a constant; $R_m q_m$ is not.

**A colleague proposes caching the attention weights — the post-softmax matrix —
to avoid recomputing them. What is wrong with that?**

They are never recomputed. Row $m$ of the attention matrix is used once, by step
$m$, and no later step reads it. What later steps need is the *keys and values*,
so that they can compute their own rows. Caching the weights would also cost
$O(L^{2})$ storage, which is the thing the whole design is trying to avoid.

**At batch 1 and 512 tokens of context, which moves more bytes per decode step:
the KV cache or the recurrent state?**

The state, by a wide margin. The KV read is $65{,}536 \times 512 = 32$ MiB; the
state read and write is 295.5 MiB, roughly nine times more. The crossover is at
about 4.7k tokens.

## Lab

Implement `Cache`, the hybrid cache the engine's model can run on.

The constructor takes the full-attention and linear-attention layer indices, a
batch size, a maximum sequence length, the KV head geometry, and the shapes of
the recurrent state and convolution window without their batch axis. It must
preallocate KV buffers for the full-attention layers only, and state buffers for
the linear-attention layers only. The harness checks `sorted(cache.k_cache)` and
`sorted(cache.states)` against the expected index lists.

`append(layer_idx, k, v)` writes into a slice and returns views of the whole
prefix. The harness prefills 10 tokens, checks the returned shape and contents,
advances, appends one more, and checks that the prefix is unchanged and the new
entry landed at the end. It writes to a second layer and checks the first layer's
buffer did not move. It also checks that exceeding `max_seq_len` raises
`RuntimeError`.

`recurrent_state`, `conv_state`, and `set_linear_state` must round-trip a state
and a window, and `recurrent_state` on a full-attention layer must return
`None`.

`memory_bytes()` returns `{"kv": ..., "recurrent": ..., "total": ...}`, and the
KV figure is checked against the exact preallocated size — `max_seq_len`, not
`length`, because preallocated bytes are spent whether you have used them or
not.

Finally the harness drives `engine.model.HybridLanguageModel` with your cache:
prefill 20 tokens of a 24-token sequence, decode the remaining 4 one at a time,
and require the logits to match a single full forward pass to $10^{-4}$.

## Further reading

- [Fast transformer decoding: one write-head is all you need](https://arxiv.org/abs/1911.02150)
- [Efficiently scaling transformer inference](https://arxiv.org/abs/2211.05102) — where the decode memory-traffic arithmetic is laid out in full.
- [RoFormer: enhanced transformer with rotary position embedding](https://arxiv.org/abs/2104.09864) — the relative-position identity that makes post-RoPE caching valid.
- [Efficient memory management for large language model serving with PagedAttention](https://arxiv.org/abs/2309.06180) — what chapter 15 builds.
