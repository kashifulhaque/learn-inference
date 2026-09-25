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

> [!TLDR]
> - Without a cache, generation does quadratic work; with one, it's linear. The
>   saving approaches a factor of $n/2$, though the wall-clock gain is far
>   smaller.
> - Cache post-RoPE keys and values for the 16 full-attention layers, and the
>   recurrent state and convolution window for the 48 linear ones.
> - The two behave nothing alike: KV grows and can rewind; state is fixed-size,
>   float32, and can't rewind.
> - Lay the KV cache out as `(batch, heads, seq, dim)`, preallocate it, and
>   write into slices. Never `torch.cat`.
> - KV costs 64 KiB per token; state costs a fixed 147.8 MiB per sequence.

A transformer is a function of the whole sequence. Nothing in its definition
says that generating token $n+1$ must be cheaper than generating token $n$.
Run the definition literally and it isn't: each new token means another forward
pass over everything written so far.

The cache is the observation that ==almost all of that work recomputes something
that can't have changed==. It's the single largest speedup in the engine and the
first one to implement.

## Before you start

This chapter assumes the following:

- **Chapter 0a's GPU vocabulary**, particularly HBM, L2, and what a coalesced
  read is. The layout section turns on coalescing.
- **Chapter 2's memory arithmetic**: 64 KiB of KV per token, 147.8 MiB of
  recurrent state per sequence, and the break-even derivation. This chapter
  re-derives the break-even point, so you don't need it memorized.
- **Chapters 6 and 7**: what the gated delta rule's state is, and what
  grouped-query attention reads.
- **Big-O notation**, and the two sums $\sum_{j=0}^{n-1} j = \tfrac{n(n-1)}{2}$
  and $\sum_{j=0}^{n-1} c = cn$.
- **PyTorch views versus copies.** `t[:, :, :stop]` is a view and allocates
  nothing. `torch.cat` allocates a new tensor and copies both inputs into it.
  The difference is the whole chapter.

Throughout, $p$ is the prompt length in tokens, $n$ is the number of tokens
generated, and $L$ is the total context at some point in time.

## From quadratic to linear

This section counts how much work generation does with and without a cache.
Measure work in *token-forward-passes*: one token passing through all 64 layers
once. It's a crude unit, but it's proportional to FLOPs for everything except
attention itself, and it's exactly what the cache changes.

### Without a cache

Each new token needs a forward pass over everything before it:

- The first new token needs a pass over the $p$ prompt tokens.
- The second needs a pass over $p+1$ tokens: the prompt plus the first.
- In general, the $j$-th generated token needs a pass over $p + j - 1$ tokens,
  for $j = 1, \ldots, n$.

Add up the passes, then split the sum into a prompt part and a growth part:

$$
W_{\text{none}}(p, n) = \sum_{j=0}^{n-1} (p + j) = \sum_{j=0}^{n-1} p + \sum_{j=0}^{n-1} j = \boxed{\hla{np} + \hlb{\frac{n(n-1)}{2}}}
$$

> [!INTUITION] Two ways to waste work
> The $\hla{np}$ term is the prompt, re-read once for every generated token. The
> $\hlb{n(n-1)/2}$ term is the generated tokens re-reading each other, and it
> grows as $n^{2}/2$. So $W_{\text{none}} = O(n^{2})$ for a fixed prompt, and
> $O((p+n)^2)$ if the prompt grows with the output.

### With a cache

The prefill pass processes all $p$ prompt tokens at once and produces the first
generated token. Each subsequent step processes exactly one token, and there are
$n - 1$ of them:

$$
W_{\text{cache}}(p, n) = p + (n - 1) = O(p + n)
$$

That's linear. Divide one by the other to get the saving:

$$
\frac{W_{\text{none}}}{W_{\text{cache}}} = \frac{np + \tfrac{n(n-1)}{2}}{p + n - 1}
\;\xrightarrow[\;n \gg p\;]{}\; \boxed{\frac{n}{2}}
$$

For long generations, ==the cache saves a factor of about $n/2$==. That's the
whole result; the rest of the chapter is about what it costs.

> [!EXAMPLE] A 512-token prompt and 128 generated tokens
> Plug in $p = 512$ and $n = 128$:
>
> $$
> W_{\text{none}} = 128 \cdot 512 + \frac{128 \cdot 127}{2} = 65{,}536 + 8{,}128 = 73{,}664
> $$
>
> $$
> W_{\text{cache}} = 512 + 127 = 639
> $$
>
> | | Forward passes | Token-forward-passes |
> |---|---|---|
> | No cache | 128 | 73,664 |
> | With cache | 1 prefill + 127 decode | 639 |
>
> That's a ratio of $73{,}664 / 639 = 115.3$.
>
> In FLOPs: a dense forward pass over one token costs about $2N$ FLOPs, where
> $N = 26.9 \times 10^{9}$ is the parameter count, one multiply and one add per
> parameter. So $2N = 53.8$ GFLOP per token-forward-pass, and the uncached run
> costs
>
> $$
> 73{,}664 \times 53.8\ \text{GFLOP} = 3.96\ \text{PFLOP}
> $$
>
> against $639 \times 53.8\ \text{GFLOP} = 34.4$ TFLOP with the cache.

## Why the wall-clock gain is smaller

Work isn't time. The uncached passes each cover several hundred tokens, which
puts them well above the ridge point from chapter 10, so they run near the
tensor cores' peak. Cached decode steps cover one token each, which puts them
far below it, so they run at memory bandwidth.

The following table compares the roofline floors for the preceding example, at
the A100's rated 1935 GB/s and 312 TFLOP/s:

| | Bound by | Floor |
|---|---|---|
| Uncached, 128 passes averaging 575 tokens | Compute | $3.96\times10^{15} / 312\times10^{12} = 12.7$ s |
| Prefill, 512 tokens | Compute | 88 ms |
| 127 decode steps, each reading 53.8 GB of weights | Memory | $127 \times 27.8\ \text{ms} = 3.53$ s |

> [!KEY] About 3.5 times faster, not 115
> The arithmetic predicts about a $3.5\times$ wall-clock speedup at this size,
> not $115\times$. The gap closes as $n$ grows, because uncached time scales as
> $n^{2}$ while cached time scales as $n$.

Dividing the two time expressions gives a ratio that grows linearly in $n$:

$$
\frac{n}{2} \cdot \frac{0.172\ \text{ms}}{27.8\ \text{ms}} \approx 0.0031\,n
$$

Here 0.172 ms is one token-forward-pass of arithmetic at peak, and 27.8 ms is
one decode step of weight reading. A $20\times$ speedup needs roughly 6500
generated tokens.

Every figure in this section is arithmetic from the roofline, not a
measurement. The lab has you measure the work ratio, which is the part that
doesn't depend on how good your kernels are.

## What to cache

This section answers which tensors can be stored once and reused, and why.

### Keys and values, for full-attention layers only

For a full-attention layer, the score between the query at position $m$ and the
key at position $n$ is:

$$
s_{mn} = \frac{q_m^{\top} k_n}{\sqrt{d}} .
$$

The key $k_n$ depends only on the residual stream at position $n$. Causal
masking guarantees that the residual stream at position $n$ is a function of
tokens $0$ through $n$ only.

> [!KEY] A key never changes after it's written
> Once token $n$ has been processed, $k_n$ and $v_n$ are fixed forever: the key
> computed for token 5 is the same whether the sequence is 6 tokens long or
> 6000. That's the entire justification for the cache, and it's exactly what a
> non-causal model would lose.

The other tensors in the layer aren't worth caching:

- **Queries.** A query is consumed by the step that produces it and never read
  again. Caching $q_m$ would let you recompute row $m$ of the attention matrix
  later, which no one wants.
- **Attention outputs.** The output at position $m$ is a weighted sum over
  positions $0 \ldots m$, so it's already final, but nothing downstream ever
  asks for it again. A later step doesn't re-read the residual stream from an
  earlier position.

### The cache holds post-RoPE keys

RoPE rotates each key by a block-diagonal rotation $R_n$ whose angles depend on
the absolute position $n$. The engine applies it before the cache write:

```python
k = apply_rotary_partial(k, cos, sin, self.rotary_dim)
if kv_cache is not None:
    k, v = kv_cache.append(layer_idx, k, v)
```

This is safe because of the identity RoPE was designed around. For rotation
matrices, $R_m^{\top} R_n = R_{n-m}$, so the rotated query $\hla{R_m q_m}$ and
the stored key $\hlb{R_n k_n}$ combine like this:

$$
\hla{(R_m q_m)}^{\top} \hlb{(R_n k_n)} = q_m^{\top} R_m^{\top} R_n k_n = q_m^{\top} \hlc{R_{n-m}} k_n .
$$

The score depends only on the difference $\hlc{n - m}$, and $n$ is fixed the
moment token $n$ is written. Whatever query arrives later applies its own $R_m$
to itself; ==the stored $\hlb{R_n k_n}$ never needs to change==.

Store pre-RoPE keys instead, and every decode step has to rotate the entire
prefix before using it: $O(L)$ extra arithmetic, an extra $O(L)$ read, and an
extra buffer to hold the result. You get nothing for it. Values are never
rotated, so the question doesn't arise for them.

### Recurrent state and the convolution window, for linear layers

The 48 linear-attention layers keep two things per sequence:

- The delta-rule state $S$, shape `(batch, 48, 128, 128)`, in float32.
- The causal convolution window, shape `(batch, 10240, 3)`, in bfloat16: the
  previous $k - 1 = 3$ steps of a four-tap kernel.

Neither grows with sequence length, and neither is appended to. ==Both are
overwritten in full every step==, and that difference runs through the rest of
this chapter.

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
prints `[3, 7, 11, ..., 63]`, and you can see at a glance that you built the
right thing.

Calling all four "the cache" hides that they behave nothing alike:

| | KV cache | Recurrent state |
|---|---|---|
| Layers | 16 | 48 |
| Size | Grows by 4 KiB per token per layer | Fixed at 3.08 MiB per layer |
| Written per step | One token's slice | The whole tensor |
| Read per step | The whole prefix | The whole tensor |
| Can you recover position $j$? | Yes, it's still there | No, it has been folded in |
| Dtype | bfloat16 | float32 |
| Reset | Move the cursor to 0 | Zero the tensor |

The last two rows are where the engineering consequences live.

**Float32 for the state.** The recurrence multiplies the state by a decay
$\alpha \in (0,1)$ thousands of times. In bfloat16, with 8 significand bits, the
repeated rounding of that product accumulates into visible drift over a long
sequence. The KV cache has no such problem, because nothing is ever multiplied
into an entry after it's written.

**No rewind.** You can drop the last $r$ KV entries by moving the cursor back
$r$ places; the earlier entries are untouched. You can't do that to the
recurrent state, because position $j$'s contribution was added into $S$, and the
addition isn't invertible in floating point.

> [!WARNING] The state can't rewind, and that costs you three times later
> - Prefix sharing across requests ([chapter 15](/c/15-paged-attention)) works
>   for KV and not for state.
> - Preemption-by-recompute must replay the linear layers from the start.
> - Speculative decoding ([chapter 20](/c/20-speculative-decoding)) must snapshot
>   the state before proposing, rather than rolling it back afterwards.

## Layout

This section picks the order of the KV cache's four axes. `HybridCache` uses
`(batch, heads, seq, dim)`, or BHSD:

```python
shape = (batch_size, config.num_key_value_heads, max_seq_len, config.head_dim)
```

The alternative is `(batch, seq, heads, dim)`, or BSHD. PyTorch stores tensors
row-major, so the last axis is contiguous and each earlier axis strides over
everything after it. The two layouts trade reads against writes:

| Layout | Reading one head across the sequence | Writing one token |
|---|---|---|
| BHSD | One contiguous $L \times 256$ block: an unbroken sequential stream, and a dense GEMM operand you can hand to cuBLAS with a leading dimension of 256 and no copy | Four small writes of 256 contiguous elements (512 bytes), one per KV head, at addresses `max_seq_len * head_dim` elements apart |
| BSHD | $L$ separate 512-byte runs with 1536-byte gaps, because each head strides by 1024 elements | One coalesced store of $4 \times 256 = 1024$ elements, 2048 bytes, in one run |

BHSD's sequential read is what the DRAM row buffer and the L2 prefetcher reward.
At this geometry, ==BSHD doesn't waste bytes==; it loses the length of the
sequential run, which costs DRAM page locality and prefetch efficiency rather
than raw bandwidth.

> [!DEEPDIVE] Why the strided read wastes no bandwidth at head_dim 256
> 256 bfloat16 values is 512 bytes, exactly four 128-byte cache lines. An
> aligned read of one head-token fetches four lines and uses all of them in
> either layout. Shrink `head_dim` to 64, as many models do, and one head-token
> becomes 128 bytes, a single line. At that point, the strided layout starts
> costing real bandwidth too. The rule from chapter 0a still holds: what matters
> is how much of each fetched line you use, and how long the stream between
> jumps is.

The ratio of reads to writes decides which layout wins, and it isn't close. At
32k context, one decode step in one layer reads this much:

$$
2 \cdot 4 \cdot 32768 \cdot 256 \cdot 2\ \text{bytes} = 134{,}217{,}728 = 128\ \text{MiB}
$$

It writes this much:

$$
2 \cdot 4 \cdot 1 \cdot 256 \cdot 2\ \text{bytes} = 4096\ \text{bytes} = 4\ \text{KiB} .
$$

That's a ratio of 32,768 to 1. Optimize the read; the write is free at any
layout. BHSD it is.

> [!NOTE] Paging flips the choice
> `PagedKVCache` in [chapter 15](/c/15-paged-attention) stores blocks as
> `(num_blocks, block_size, kv_heads, head_dim)`, which is BSHD within a block. A
> paged write is a scatter to arbitrary slots, and scattering whole tokens is
> cheaper than scattering four per-head fragments. The block is small enough
> that the strided read inside it stays in L2.

## Preallocate, don't concatenate

The obvious implementation appends:

```python
self.k = torch.cat([self.k, new_k], dim=2)   # don't
```

Every `cat` allocates a new tensor and copies both inputs into it. At step $j$,
the cache holds $p + j$ tokens, so the total copied over $n$ steps is the same
quadratic sum as before, counted in tokens of cache:

$$
\sum_{j=0}^{n-1} (p + j) = \hla{np} + \hlb{\frac{n(n-1)}{2}}
$$

For $p = 512$ and $n = 128$, that's 73,664 tokens at 64 KiB each, or 4.5 GiB
read and 4.5 GiB written: about 8 ms of pure memcpy at the measured 1275 GB/s,
to accomplish nothing. ==You reintroduced the quadratic you had removed==, in
the data movement instead of the arithmetic.

> [!WARNING] The allocator damage is worse than the copying
> Each `cat` requests a block one token larger than the last and frees the
> previous one, so the caching allocator accumulates a ladder of
> almost-but-not-quite reusable blocks. The symptom is an out-of-memory error at
> 60 GB on an 80 GB card, with `torch.cuda.memory_reserved` far above
> `memory_allocated`.

Instead, allocate once at the maximum length and write into a slice:

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

The write is in place, and the return is two views. Neither allocates, and the
returned tensors are exactly the shape the attention kernel wants:
`(batch, kv_heads, stop, head_dim)`.

Raise on overflow rather than growing. A cache that silently reallocates is a
cache that silently reallocates in production, at the worst moment, under the
largest batch. The scheduler in [chapter 16](/c/16-continuous-batching) is the
component that decides what to do when capacity runs out, and it can only do
that if the cache tells it.

The cost of preallocation is reservation. A batch of 32 sequences that might
reach 32k tokens reserves this much, even if every sequence stops at 200 tokens:

$$
32 \times 32768 \times 65{,}536\ \text{bytes} = 68.7\ \text{GB} = 64\ \text{GiB}
$$

Chapter 15 fixes that with paging. Contiguous preallocation is still the right
thing to build first: it's fifty lines, it's correct by inspection, and it's the
reference the paged version gets checked against.

## The memory arithmetic

The cache's footprint has two independent terms, both per sequence. The
$\hlc{\text{KV cache}}$ costs 64 KiB per token of context, and the
$\hld{\text{recurrent state}}$ costs a fixed 147.8 MiB, independent of $L$:

$$
\text{bytes} = \hlc{B \cdot L \cdot 65{,}536} + \hld{B \cdot 154{,}927{,}104}
$$

The following table shows the cost per sequence, in GiB:

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
context, and an activation workspace. Read the table against that number:
batch 8 at 32k fits with nothing to spare, batch 32 at 4k fits comfortably, and
batch 64 at 4k doesn't.

> [!KEY] The fixed cost is fixed per sequence, not per GPU
> Chapter 2 quotes 19 GiB as "304k tokens", which counts only the KV term. At
> batch 64, the fixed recurrent state is $64 \times 147.8$ MiB $= 9.2$ GiB
> before a single token of context exists: half the budget, spent on state that
> doesn't grow. The state scales with batch even though it doesn't scale with
> context.

### Break-even against an all-full-attention model

An all-full-attention version of this geometry would pay $256$ KiB per token
across all 64 layers and carry no fixed state. The hybrid pays
$\hlc{64\text{ KiB per token}}$ plus $\hld{147.8\text{ MiB}}$. Set the two
equal and solve for $L$:

$$
262{,}144\,L = \hlc{65{,}536\,L} + \hld{154{,}927{,}104}
\;\Longrightarrow\;
196{,}608\,L = 154{,}927{,}104
\;\Longrightarrow\;
\boxed{L = 788\ \text{tokens}}
$$

Below 788 tokens of context, the fixed state costs more than the KV entries it
replaces. Above it, the hybrid wins, and the margin grows by 192 KiB per further
token without limit. At 32k context, the hybrid needs 2.14 GiB per sequence
against 8.0 GiB; at 128k, 8.14 GiB against 32 GiB.

`ModelConfig.hybrid_breakeven_tokens` computes this for any config, and chapter
2's lab has you derive it.

### What decode actually reads

Memory footprint and memory traffic are different questions. Per decode step,
per sequence, the engine reads the following:

- The weights: 53.8 GB, once, shared across the whole batch.
- The KV cache: $\hlc{65{,}536 \cdot L}$ bytes, read in full.
- The recurrent state: 147.8 MiB read and 147.8 MiB written, because every
  linear layer overwrites its state.

The state's traffic is $\hld{2 \times 154{,}927{,}104 = 309{,}854{,}208}$ bytes
per sequence per step. The KV read matches it at this context length:

$$
\hlc{65{,}536\,L} = \hld{309{,}854{,}208} \quad\Longrightarrow\quad L \approx 4728\ \text{tokens} .
$$

==Below about 4.7k context, the 48 linear-attention layers move more bytes per
decode step== than the 16 full-attention ones do. The hybrid design shifts
memory pressure; it doesn't remove it. That's also why quantizing the KV cache
to int8 ([chapter 18](/c/18-quantization)) helps less at short context than the
headline suggests.

## The state carry

The KV path is append-only and hard to get subtly wrong: either the prefix is
there or it isn't. The linear path is the opposite.

`delta_rule_chunked` takes a `state` argument and returns the final state. Each
phase uses it differently:

- **Prefill** passes `None`, computes the whole chunked recurrence, and stores
  the result.
- **Each decode step** passes the stored state back in, runs one
  `delta_rule_step`, and stores the new one.

```python
prev_state = cache.recurrent_state(layer_idx) if cache is not None else None
# ... run the recurrence ...
if cache is not None:
    cache.set_linear_state(layer_idx, new_state, new_conv)
```

The convolution window has the same structure. It's the last three steps of
input to the four-tap causal convolution, and it must be shifted forward every
step. `causal_depthwise_conv1d` returns the updated window alongside its output;
store both or neither.

## What goes wrong

The cache's bugs produce output that reads as plausible text, which is why the
prefill-plus-decode equivalence test from
[chapter 8](/c/08-the-forward-pass) is the one that catches them.

**Forgetting to read `prev_state`.** The model prefills correctly and then
generates as if the prompt never happened: fluent, on-topic for about one
sentence, then unmoored.

**Forgetting to write `new_state`.** Every decode step starts from the prefill
state, so the model repeats itself.

**Forgetting the convolution window.** It has the same structure as the state
and is easier to overlook. Store the updated window and the output together, or
neither.

**Advancing the cursor inside each layer.** The cache length must advance after
all 64 layers have run, not inside each layer:

```python
for layer in self.layers:
    x = layer(x, cos, sin, cache=cache)
if cache is not None:
    cache.advance(seq)
```

Every layer in one forward pass writes at the same position, because they're all
processing the same tokens. Advancing per layer makes layer 7 believe the
sequence is one token longer than layer 3 does, which corrupts both the write
offset and the causal mask offset $L - T$. The symptom is a model that's correct
during prefill and wrong from the second decode step onwards, the signature in
chapter 8's bisection table.

> [!RECAP]
> - Uncached generation does $\hla{np} + \hlb{n(n-1)/2}$ token-forward-passes;
>   cached generation does $p + n - 1$. The work ratio tends to $n/2$, but the
>   wall-clock ratio is much smaller because decode is memory bound.
> - Cache post-RoPE keys and values, because a key never changes once written
>   and the score depends only on $n - m$.
> - KV grows and can rewind; the float32 recurrent state is fixed-size and
>   can't.
> - Use BHSD, preallocate, write into slices, raise on overflow, and advance the
>   cursor once per forward pass.
> - Memory is $\hlc{65{,}536\,BL} + \hld{154{,}927{,}104\,B}$ bytes. The hybrid
>   beats full attention above 788 tokens, and the state dominates decode
>   traffic below about 4.7k.

## Check your understanding

> [!QUESTION] You generate 2000 tokens from a 100-token prompt. How many token-forward-passes does each approach do?
> Uncached: $2000 \cdot 100 + \tfrac{2000 \cdot 1999}{2} = 200{,}000 +
> 1{,}999{,}000 = 2{,}199{,}000$. Cached: $100 + 1999 = 2099$. That's a ratio of
> 1048, close to the $n/2 = 1000$ asymptote, because $n \gg p$ here.

> [!QUESTION] Why does the KV cache hold keys after RoPE, but the query is rotated fresh every step?
> Because the stored key's rotation angle depends on its own fixed position,
> while the query's depends on the position of the step being computed, which is
> different every time. $R_n k_n$ is a constant; $R_m q_m$ isn't.

> [!QUESTION] A colleague proposes caching the attention weights, the post-softmax matrix, to avoid recomputing them. What's wrong with that?
> They're never recomputed. Row $m$ of the attention matrix is used once, by
> step $m$, and no later step reads it. What later steps need is the keys and
> values, so that they can compute their own rows. Caching the weights would
> also cost $O(L^{2})$ storage, which is the thing the whole design is trying to
> avoid.

> [!QUESTION] At batch 1 and 512 tokens of context, which moves more bytes per decode step: the KV cache or the recurrent state?
> The state, by a wide margin. The KV read is $65{,}536 \times 512 = 32$ MiB; the
> state read and write is 295.5 MiB, roughly nine times more. The crossover is at
> about 4.7k tokens.

## Lab

> [!TRY]
> Implement `Cache`, the hybrid cache the engine's model can run on. Passing
> means every harness check succeeds, ending with prefill-plus-decode logits
> that match a single full forward pass to $10^{-4}$.

The constructor takes the full-attention and linear-attention layer indices, a
batch size, a maximum sequence length, the KV head geometry, and the shapes of
the recurrent state and convolution window without their batch axis. It must
preallocate KV buffers for the full-attention layers only, and state buffers for
the linear-attention layers only. The harness checks `sorted(cache.k_cache)` and
`sorted(cache.states)` against the expected index lists.

`append(layer_idx, k, v)` writes into a slice and returns views of the whole
prefix. The harness checks it as follows:

- It prefills 10 tokens, checks the returned shape and contents, advances,
  appends one more, and checks that the prefix is unchanged and the new entry
  landed at the end.
- It writes to a second layer and checks that the first layer's buffer didn't
  move.
- It checks that exceeding `max_seq_len` raises `RuntimeError`.

`recurrent_state`, `conv_state`, and `set_linear_state` must round-trip a state
and a window, and `recurrent_state` on a full-attention layer must return
`None`.

`memory_bytes()` returns `{"kv": ..., "recurrent": ..., "total": ...}`. The KV
figure is checked against the exact preallocated size, using `max_seq_len`, not
`length`, because preallocated bytes are spent whether you've used them or not.

Finally, the harness drives `engine.model.HybridLanguageModel` with your cache:
it prefills 20 tokens of a 24-token sequence, decodes the remaining 4 one at a
time, and requires the logits to match a single full forward pass to $10^{-4}$.

## Further reading

- [Fast transformer decoding: one write-head is all you need](https://arxiv.org/abs/1911.02150)
- [Efficiently scaling transformer inference](https://arxiv.org/abs/2211.05102) — where the decode memory-traffic arithmetic is laid out in full.
- [RoFormer: enhanced transformer with rotary position embedding](https://arxiv.org/abs/2104.09864) — the relative-position identity that makes post-RoPE caching valid.
- [Efficient memory management for large language model serving with PagedAttention](https://arxiv.org/abs/2309.06180) — what chapter 15 builds.
