---
title: Grouped-query attention
slug: 07-grouped-query-attention
part: "Part 2 — A forward pass"
summary: Attention built up from a lookup, then the 16 layers that keep a real KV cache, with heads that share keys and values and an output gate that can switch a head off.
minutes: 115
gpu: true
objectives:
  - Derive scaled dot-product attention and show where the 1 over root d scale comes from.
  - Apply a causal mask correctly when a cached prefix is present, and explain why the mask goes before the softmax.
  - Explain what multi-head attention buys and what each head gives up.
  - Compare MHA, MQA, and GQA on this model's geometry with the cache arithmetic for each.
  - Derive the arithmetic intensity of decode attention and show that it equals the GQA group size.
  - Describe what the attention output gate does and why softmax attention needs one.
lab: 07-gqa
---

# Grouped-query attention

> [!TLDR]
> - Attention lets each token look back at earlier tokens and take a weighted
>   average of what they carry. Dividing the scores by $\sqrt{d_h}$, the square
>   root of a head's width, keeps that average from collapsing onto one token.
>   Here that's a divide by 16.
> - The causal mask hides the future. Apply it before the softmax, and shift it
>   by `kv_len - q_len` when earlier tokens come from a cache, or generation
>   quietly loses its own context.
> - Grouped-query attention (GQA) lets six query heads share one set of keys and
>   values. The stored keys and values, the KV cache, then cost 64 KiB per token:
>   a sixth of what giving every head its own would cost.
> - While generating tokens, attention does only 6 arithmetic operations per
>   byte it reads, at any context length or batch size. The GPU can't be kept
>   busy at that rate, so the only way to speed attention up is to read fewer
>   bytes.
> - A sigmoid output gate lets a head output nothing. Slicing `q_proj` instead of
>   chunking it drops the gate without raising an error.

Sixteen of the model's 64 layers use ordinary softmax attention, and chapter 6
covered the other 48. These sixteen are the layers that own a *KV cache*: a
store of every past token's keys and values, kept so they're never recomputed.
So they set the memory ceiling, and their share of each decode step grows with the context.

This chapter builds attention from the start instead of quoting the formula.
Everything in Parts 3 through 6 (the cache, the roofline, FlashAttention, paged
attention, and quantized caches) is an argument about the shapes and byte counts
you derive here. It's worth getting them exactly right once.

The route runs in five steps: what attention computes, why its scores are
scaled, how the causal mask works, how heads share keys and values, and why
generating tokens leaves the GPU's arithmetic mostly idle.

## Before you start

**Prefill and decode.** An engine runs a request in two phases. *Prefill* pushes
the whole prompt through the model in one forward pass. *Decode* then generates
one token per forward pass, each step reading back the cache that earlier steps
wrote.

**The residual stream.** Each token carries a 5120-wide vector from layer to
layer. Every layer reads it and adds its own contribution back in. That running
vector is the *residual stream*.

**Softmax.** Chapter 0a covers it, including shift invariance: subtracting a
constant from every score leaves the output unchanged, which makes a numerically
stable implementation possible. This chapter uses two more properties: the
outputs are non-negative, and they sum to 1.

**Variance of a sum.** For independent zero-mean random variables,
$\operatorname{Var}(\sum_i X_i) = \sum_i \operatorname{Var}(X_i)$. The scale
factor argument is one application of that.

**Tensor shapes.** Everything is `(batch, heads, seq, head_dim)`, where
`head_dim` is the width of one attention head. Chapter 0a covers the convention
and `torch.einsum`. Keep this chapter's shape table open while you read the
code.

**This model's geometry.** The arithmetic uses these values throughout:

| Field | Value | What it means |
|---|---|---|
| `hidden_size` | 5120 | Width of the residual stream |
| `num_attention_heads` | 24 | Query heads, $H$ |
| `num_key_value_heads` | 4 | Key and value heads, $H_{kv}$ |
| `head_dim` | 256 | Width of one head, $d_h$ |
| `attn_output_gate` | true | Each head's output passes through a gate |
| Full-attention layers | 16 of 64 | The layers this chapter covers |
| Bytes per element | 2, bfloat16 | The cache's storage format |

## Queries, keys, and values

What does attention compute? In one phrase, it's ==a differentiable lookup==
in a table whose rows are the tokens seen so far.

Picture a shelf of folders. Each has a label that says what's inside, and
contents that you take away. You arrive with a question.

A hard lookup opens the one folder whose label best fits the question.
Attention instead reads every folder, and blends the contents in proportion to
how well each label fits.

In attention's terms, the folders are past tokens $j$, and the question comes
from the current token $t$. Each is a vector of width $d_h$:

- A *key* $k_j \in \mathbb{R}^{d_h}$ is the label. It advertises what token $j$
  offers.
- A *value* $v_j \in \mathbb{R}^{d_h}$ is the contents that token $j$ hands
  over.
- A *query* $q_t \in \mathbb{R}^{d_h}$ is the question. It describes what token
  $t$ is looking for.

All three are linear projections of the residual stream, so the model learns
what they mean.

Picking the single best match isn't differentiable, so training couldn't adjust
it. Instead, score every key against the query, turn the scores into a
probability distribution, and return the weighted average of the values:

$$
\hla{s_{tj}} = \frac{q_t^\top k_j}{\sqrt{d_h}},
\qquad
\hlb{w_{tj}} = \frac{\exp(\hla{s_{tj}})}{\sum_{i \le t} \exp(s_{ti})},
\qquad
\hlc{o_t} = \sum_{j \le t} \hlb{w_{tj}} \, v_j
$$

> [!INTUITION]
> The score $\hla{s_{tj}}$ measures how well key $j$ matches the query. The
> softmax turns a row of scores into weights $\hlb{w_{tj}}$ that are positive
> and sum to 1. The output $\hlc{o_t}$ is a blend of the values, mixed in those
> proportions.

In matrix form, a whole block of queries is handled at once. Here $M$ is the
causal mask, covered later in this chapter:

$$
\operatorname{Attention}(Q, K, V)
= \operatorname{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_h}} + M \right) V
$$

Three properties follow immediately, and later chapters lean on all three:

- **The output is a convex combination**, a weighted average whose weights are
  non-negative and sum to 1. So $\hlc{o_t}$ lies inside the convex hull of the
  values it read. Attention can't amplify, and it can't return zero unless the
  values themselves sum to zero. The output gate later in this chapter
  exists to remove that limitation.
- **The query never leaves the exponential.** $\exp(q_t^\top k_j)$ doesn't
  factor into something depending on $q_t$ times something depending on $k_j$,
  so there's ==no fixed-size summary of the past==. Every past key and value has
  to be kept. That's the KV cache, and chapter 6 is what you get if you refuse to
  pay for it.
- **Cost is quadratic in prefill and linear in decode.** With $L$ tokens of
  context, scoring $L$ queries against $L$ keys is $O(L^2)$ work. Scoring one
  query against $L$ keys is $O(L)$ work, but it also reads $O(L)$ bytes, and
  that read turns out to be the binding constraint.

Next, you see why the formula divides by $\sqrt{d_h}$.

## Where the scale factor comes from

The $\hld{\sqrt{d_h}}$ divisor keeps the softmax from collapsing to a hard
argmax, which would pick one token and ignore the rest. It isn't cosmetic.

Start with the picture. A raw score adds up $d_h = 256$ products. Adding more
random terms widens the spread of the total, so scores in one row land far
apart. A softmax over widely spread scores hands nearly all the weight to the
largest one.

Now make that precise. Model the entries of $q$ and $k$ as independent, mean
zero, and unit variance, which is roughly what a well-initialized projection
followed by a normalization produces. The unscaled score is a sum of $d_h$
terms:

$$
q^\top k = \sum_{i=1}^{d_h} q_i k_i
$$

Each term has variance 1, and the terms are independent, so their variances add:

$$
\boxed{\operatorname{Var}(q^\top k) = d_h,
\qquad
\operatorname{sd}(q^\top k) = \hld{\sqrt{d_h}}}
$$

> [!DEEPDIVE] The variance of one term
> Each term has mean $\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$
> by independence. Its variance is the expected square, which also factors by
> independence:
>
> $$
> \operatorname{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] = \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = 1
> $$
>
> The $d_h$ terms are independent, so the variance of the sum is $d_h$ times 1.

So the typical raw score grows like $\hld{\sqrt{d_h}}$. Two scores separated by
a gap $\Delta$ produce softmax weights in the ratio $e^{\Delta}$, so a large
gap hands the largest score essentially all the weight. That hurts twice:

- **In training**, the gradient through a saturated softmax is about zero, so
  the layer stops learning.
- **At inference**, the head reads exactly one token and throws away the
  averaging.

Dividing by $\hld{\sqrt{d_h}}$ sets the score variance back to 1, whatever the
head width. Gaps are then of order 1, and weight ratios of order $e$.

> [!EXAMPLE] The scale for $d_h = 256$
> The raw score has a standard deviation of 16, so scores in a row routinely
> differ by tens. A gap of 16 gives a weight ratio of
> $e^{16} \approx 8.9 \times 10^{6}$. The scale that fixes it is
> $1/\sqrt{256} = 1/16 = 0.0625$, which is exactly `head_dim**-0.5` in
> `engine/layers/attention.py`.

The scale ==doesn't fix overflow==, and the two often get conflated. Softmax is
shift invariant, so a correct implementation subtracts the row maximum before
exponentiating and never overflows, whatever the scale of the scores. The scale
exists for saturation and gradients, not for numerical range. Chapter 14 uses
the same shift invariance to build FlashAttention's online softmax.

This model adds a second guard. Qwen3 applies RMSNorm per head to queries and
keys over the 256-wide head dimension, before the rotary embedding. That bounds
the score scale by the learned norm weights rather than by whatever the
projections produce. In the tensor listing these appear as `q_norm.weight` and
`k_norm.weight`, each shape `[256]`.

With the scores well scaled, the next question is which scores a token is
allowed to use at all.

## The causal mask

The causal mask stops a token from seeing its own future: it removes every
score where $j > t$. A language model predicts
token $t+1$ from tokens $1 \ldots t$. If position $t$ could attend to $t+1$,
training would be trivially satisfiable and the model useless at inference, when
the future doesn't exist.

### Why the mask goes before the softmax

The mask sets forbidden scores to $-\infty$ before exponentiating. Because
$\exp(-\infty) = 0$, those keys contribute nothing to either the numerator or
the denominator, and the surviving weights still sum to 1.

The alternative is to compute the full softmax, then multiply the forbidden
weights by zero. That's wrong because of the denominator: it still counts the
future keys. With $w_j$ for the unmasked weights and $m_j \in \{0,1\}$ for the
mask, the kept weights sum to less than 1:

$$
\sum_j m_j w_j = \frac{\sum_{j \le t} \exp(s_j)}{\sum_{\text{all } j} \exp(s_j)} < 1
$$

The output shrinks by a factor that depends on how attractive the future keys
were. A position whose future is highly relevant gets its output shrunk toward
zero, and one whose future is irrelevant doesn't. That's an attenuation that
depends on both position and content, and nobody intended it.

### The offset that matters

During prefill, queries and keys cover the same positions, and the mask is the
strict upper triangle. During decode, the keys include a cached prefix that the
queries don't cover, so the row and column indices no longer line up.

A concrete case shows the problem. Say nine tokens are already cached and one
new token arrives.

The key tensor has ten rows, positions 0 to 9, but the query
tensor has one row, numbered 0. That query really sits at position 9, and it may
see all ten keys. A mask that trusts the row number 0 would hide keys 1 to 9.

In general, let `q_len` be the number of queries in this call and `kv_len` the
total number of keys, cache included. The queries are always the last `q_len`
positions, so query row $i$ sits at the following absolute position, and it may
attend to key positions up to and including it:

$$
\text{pos}(i) = i + (\text{kv\_len} - \text{q\_len})
$$

The following code builds that mask:

```python
offset = kv_len - q_len
idx_q = torch.arange(q_len, device=device).unsqueeze(-1)   # (q_len, 1)
idx_k = torch.arange(kv_len, device=device).unsqueeze(0)   # (1, kv_len)
mask = idx_k > idx_q + offset                              # (q_len, kv_len), True = blocked
scores = scores.masked_fill(mask, float("-inf"))
```

Lab 07 checks three regimes:

| Call | `q_len` | `kv_len` | `offset` | Entries masked |
|---|---|---|---|---|
| Prefill | 4 | 4 | 0 | 6, the strict upper triangle |
| Decode step | 1 | 10 | 9 | 0, the single query sees everything |
| Chunked prefill | 3 | 10 | 7 | 3 |

> [!EXAMPLE] Check the chunked prefill row by hand
> The three queries are at positions 7, 8, and 9. The query at 7 must not see
> keys 8 and 9, which is two entries. The query at 8 must not see key 9, which
> is one. The query at 9 sees all ten. The total is 3.

> [!WARNING] A missing offset fails quietly
> Forgetting the offset produces a mask that's correct during prefill, when the
> offset is zero, and wrong during every decode step. The model generates well
> for its prompt and then drifts, because each decode step hides most of its own
> context. There's no crash, no NaN, and the output looks plausible.

There's a loud failure available too. If a row of the mask blocks every key, the
softmax computes $0/0$ and returns NaN. A stable implementation that subtracts
the row maximum gets $-\infty - (-\infty)$ first, which is also NaN. A correct
causal mask can't do that, because query $i$ can always see its own position,
but an off-by-one in the wrong direction does it immediately.

## Shapes, end to end

Getting the shapes right is most of the work in the lab, so it pays to see
them all in one place. The following table follows one full-attention layer for a batch of $B$ sequences and $T$ new
tokens against a cache of $L$ total positions:

| Tensor | Shape | Notes |
|---|---|---|
| `x` | $(B, T, 5120)$ | The residual stream |
| `q_proj(x)` | $(B, T, 12288)$ | $24 \times 256 \times 2$: queries and the gate |
| `q` after the split | $(B, T, 6144)$ | $24 \times 256$ |
| `gate` | $(B, T, 6144)$ | The other half |
| `k_proj(x)`, `v_proj(x)` | $(B, T, 1024)$ | $4 \times 256$ each |
| `q` reshaped | $(B, 24, T, 256)$ | After `view` and `transpose(1, 2)` |
| `k`, `v` reshaped | $(B, 4, T, 256)$ | |
| `k`, `v` after cache append | $(B, 4, L, 256)$ | $L = $ prefix $+ \, T$ |
| `k`, `v` after `repeat_kv` | $(B, 24, L, 256)$ | Naive path only |
| `scores` | $(B, 24, T, L)$ | The tensor FlashAttention avoids |
| `out` | $(B, 24, T, 256)$ | |
| `out` merged | $(B, T, 6144)$ | `transpose(1, 2).reshape(...)` |
| `o_proj(out)` | $(B, T, 5120)$ | Back to the residual stream |

Two of those rows deserve a second look.

**`head_dim` isn't `hidden_size / num_heads`.** $5120 / 24 = 213.33$, which
isn't an integer, and the model uses 256 anyway. Most transformers tie the two,
so most readers assume the tie holds. Here the attention block is wider than the
residual stream, $24 \times 256 = 6144$ against 5120, and `o_proj` projects back
down. Assuming the tie gives you a `view` that fails, or worse, ==one that
succeeds with the wrong stride==.

**The projections aren't square.** Add up the four projections in one
full-attention layer:

$$
\begin{aligned}
&\underbrace{5120 \times 12288}_{\text{q and gate}}
+ \underbrace{2 \times 5120 \times 1024}_{\text{k and v}}
+ \underbrace{6144 \times 5120}_{\text{o}} \\
&\quad = 104{,}857{,}600
\end{aligned}
$$

That total works out as follows:

- 104.9M parameters per layer, which is 210 MB in bfloat16.
- 3.4 GB across the 16 full-attention layers.
- Of the 104.9M, the key and value projections are only 10.5M.

That last figure is the first hint that shrinking the KV heads costs less than
it sounds.

## Multi-head attention

Real attention runs $H$ heads in parallel instead of one, and this section
explains what that buys. Each has its own $Q$, $K$, and $V$
projections into a $d_h$-dimensional subspace, and `o_proj` concatenates and
mixes their outputs.

### Why heads help

One head produces one score per query-key pair. That gives one probability
distribution per query, and so one weighted average. A token that needs the
subject of its clause, the topic of the paragraph, and the matching open bracket
can't gather all three from one head. Those are three different rankings of the
context, and a single softmax commits to one.

With 24 heads, a token gets 24 independent rankings and 24 separate averages.
That's 6144 channels of output that `o_proj` can combine however it likes. It's
why attention patterns in trained models are visibly specialized. Some heads
track syntax, some track repeated tokens, and some attend almost entirely to
position 1 as a null target.

Heads are ==free in arithmetic==. One head of width $H d_h$ costs
$2 \cdot T \cdot L \cdot H d_h$ FLOPs (floating-point operations) for the score
matmul. $H$ heads of width $d_h$ each cost $2 \cdot T \cdot L \cdot d_h$, for
the same total. Splitting the width into heads changes the expressiveness and
not the FLOP count.

What each head gives up is rank. A head's scores are inner products in a
256-dimensional subspace, so it can only express similarity structure within
that subspace. Narrow heads are cheap and weak, and wide heads are expressive
and few.

This model's 256 is unusually wide, where 64 or 128 is more common. It
pairs with the per-head RMSNorm and with chapter 5's partial rotary factor,
which rotates only the first 64 of the 256 channels.

## MHA, MQA, and GQA

How many key and value heads do you need to keep? It matters, because they're
the dominant memory cost at long context.

The idea is sharing. Queries are consumed on the spot, but keys and values stay
in the cache for the rest of the sequence. So let several query heads read the
same key and value head, and the cache shrinks by that factor. The three schemes
differ only in how far they take that:

- **MHA**, multi-head attention: $H_{kv} = H$. Every query head has private keys
  and values.
- **MQA**, multi-query attention: $H_{kv} = 1$. All query heads share one set.
- **GQA**, grouped-query attention: $1 < H_{kv} < H$. Query heads split into
  $H_{kv}$ groups of $g = H / H_{kv}$, and each group shares a KV head.

This model uses $H = 24$ and $H_{kv} = 4$, so the group size is
$g = 24/4 = 6$.

### The cache arithmetic

The cache stores $K$ and $V$ for every position in every full-attention layer.
Count the bytes per token, per layer:

$$
\text{bytes} = 2 \times \hlb{H_{kv}} \times d_h \times \hlc{b}
$$

The leading 2 counts $K$ and $V$, $\hlb{H_{kv}}$ is the number of KV heads, and
$\hlc{b} = 2$ is bytes per element in bfloat16. Over this model's 16
full-attention layers, the three schemes compare as follows:

| Scheme | $H_{kv}$ | Per layer per token | Over 16 layers | At 32k context |
|---|---|---|---|---|
| MHA | 24 | 24,576 B = 24 KiB | 384 KiB | 12 GiB |
| GQA | 4 | 4,096 B = 4 KiB | 64 KiB | 2 GiB |
| MQA | 1 | 1,024 B = 1 KiB | 16 KiB | 0.5 GiB |

> [!EXAMPLE] Check the GQA row
> $2 \times 4 \times 256 \times 2 = 4096$ bytes per layer. Times 16 layers,
> that's 65,536 bytes, exactly 64 KiB per token: the number chapter 2 uses and
> lab 07 checks. At 32,768 tokens, $65{,}536 \times 32{,}768 = 2$ GiB per
> sequence.

Now put that against the A100's budget:

- The weights take 53.8 GB of the 80 GB card, leaving about 26 GB.
- With GQA at 32k context, that holds 11 sequences, the batch chapter 2 derives.
- With MHA, it holds two.

The difference between those two numbers is ==the difference between a server
and a demo==.

> [!NOTE]
> Chapter 2's all-full-attention figure of 256 KiB per token is a different
> comparison: GQA applied to all 64 layers rather than 16. The 384 KiB in the
> table is MHA applied to the 16 layers this model has.

### What it costs in quality

The trade is real but small. The GQA paper takes a trained MHA checkpoint,
mean-pools its KV heads down to $H_{kv}$ groups, and uptrains briefly. MQA,
collapsing all the way to one KV head, loses measurable quality. A handful of
groups recovers nearly all of it and keeps most of the cache saving, because the
saving follows a $1/H_{kv}$ curve that's already mostly flat by $H_{kv} = 4$ or
8.

Two details make this model's case better than the paper's:

- **It was trained with four KV heads from the start**, so there's no
  conversion loss. The query heads in a group learned to share their keys
  rather than having sharing imposed on them.
- **Only 16 of its layers use attention at all**, so the layers that would
  suffer most from a narrow cache are the minority.

The trade also favors decode. Reducing $H_{kv}$ costs a little expressiveness on
every token and saves bytes on every token of context. At long context the
second effect dominates by orders of magnitude, which is why nothing serves long
context with MHA any more.

GQA shrinks what decode reads. The next section shows why that's the only kind
of saving that helps.

## Why decode attention is permanently memory bound

No kernel can make decode attention use the GPU's arithmetic well, and this
section shows why. Decode attention does exactly $g$ FLOPs per byte, whatever
the context length. Chapter 10 states that result, and it
explains the rest of the course.

Start with two terms:

- *Arithmetic intensity*, $I$, is the FLOPs an operation does per byte it moves
  from HBM, the GPU's main memory.
- The *ridge point* is the intensity at which a GPU's arithmetic and its memory
  finish at the same time. The A100 does 312 TFLOP/s (trillion FLOPs per
  second) and reads 1935 GB/s, so its ridge point is
  $312 \times 10^{12} / 1935 \times 10^{9} \approx 161$ FLOPs per byte.

Below the ridge point, an operation is *memory bound*: the arithmetic units
finish early and wait for data. Picture a kitchen whose cooks can plate 161
dishes in the time one delivery arrives. If each delivery holds ingredients for
only 6 dishes, the cooks spend most of their time waiting at the door.

Now count. Take one decode step for one sequence in one layer at context length
$\hla{L}$, so the query is a single token. Count the FLOPs of the two matmuls
and the bytes of the KV cache they read:

$$
\text{FLOPs} = 4 H \hla{L} d_h,
\qquad
\text{bytes} = 2 \hlb{H_{kv}} \hla{L} d_h \hlc{b}
$$

Divide the two. Both $d_h$ and the context length $\hla{L}$ cancel, because a
longer context means proportionally more arithmetic and proportionally more
bytes:

$$
I = \frac{4 H \hla{L} d_h}{2 \hlb{H_{kv}} \hla{L} d_h \, \hlc{b}}
= \frac{2H}{\hlb{H_{kv}} \, \hlc{b}}
$$

Substitute $\hlc{b} = 2$ for bfloat16:

$$
\boxed{I = \frac{H}{\hlb{H_{kv}}} = g = 6 \ \text{FLOPs per byte}}
$$

> [!DEEPDIVE] Counting the FLOPs and bytes
> The scores are $H$ dot products of length $d_h$ against $L$ keys each, which
> is $H L d_h$ multiply-adds, or $2 H L d_h$ FLOPs. The weighted sum over values
> is the same shape again, so the total is $4 H L d_h$.
>
> The whole KV cache for this layer comes off HBM: $K$ and $V$, each
> $H_{kv} \times L \times d_h$ elements at $b$ bytes. The query is $H d_h$
> elements, independent of $L$, so it drops out at any interesting context.

> [!KEY] The intensity is the group size
> $I = g$ exactly, not approximately, and not only for this model: the equality
> holds for any GQA model with a bfloat16 cache.

Three consequences follow, and they shape everything in Parts 3 to 6:

- **Context length doesn't help.** $\hla{L}$ cancelled. Attention at 128k context
  has the same arithmetic intensity as attention at 128 tokens. It takes a
  thousand times longer.
- **Batching doesn't help either.** The KV cache is per sequence, so a batch of
  $N$ sequences does $N$ times the FLOPs and reads $N$ times the bytes. The MLP
  is different: its weights are shared across the batch, so its intensity is
  roughly the batch size, as chapter 10's table shows. Batching is the main lever
  for everything except attention.
- **No kernel closes the gap.** At 6 FLOPs per byte against a ridge point of
  161, decode attention sits 27 times below the ridge, using under 4% of the
  card's arithmetic.

The only lever left is the denominator: ==read fewer bytes==. Each later
technique pulls it in its own way:

- GQA raises $g$, which is this chapter.
- Paged attention in chapter 15 stops reading padding.
- Cache quantization in chapter 18 shrinks each element. Set $\hlc{b} = 1$ for
  int8 and the intensity doubles to 12, because $\hlc{b}$ is in the
  denominator.

## The output gate

The query projection emits $24 \times 256 \times 2 = 12288$ channels, not 6144, and the
second half is a gate:

```python
q = self.q_proj(x)                    # (batch, seq, 12288)
q, gate = q.chunk(2, dim=-1)          # (batch, seq, 6144) each
# ... attention, producing out: (batch, seq, 6144) ...
out = out * torch.sigmoid(gate)       # elementwise, still (batch, seq, 6144)
return self.o_proj(out)               # (batch, seq, 5120)
```

The gate exists because of the convexity property. A head returns a weighted
average of values whatever the input, so it has no way to say "nothing in this
context is relevant."

Models without a gate learn a workaround. They dedicate one position, usually
the first token, as a null target, and heads dump their attention there when
they have nothing to contribute. Those *attention sinks* are a well-documented
nuisance. They complicate cache eviction, quantization, and long-context
extrapolation, all because the architecture gave the head no off switch.

The gate is ==the off switch==. A `sigmoid(gate)` near zero suppresses that
head's contribution to the residual stream entirely, per channel and per token.
It has the same motivation as the SiLU gate on chapter 6's linear-attention
layers. This one is a sigmoid, though, so it's bounded to $(0,1)$: it can
attenuate but not amplify.

It costs one extra $5120 \times 6144$ matrix per layer, which is 31,457,280
parameters or 62.9 MB in bfloat16. Across 16 layers that's about 1.0 GB of the
model's 53.8 GB, spent on the ability to output nothing.

> [!WARNING] Slicing instead of chunking drops the gate
> If you miss the split and slice the first 6144 channels instead, the shapes
> still work: you get a tensor of exactly the right size. Half the projection's
> learned behavior is discarded and the gate is never applied, so every head
> contributes at full strength on every token. Quality drops in a way that's hard
> to attribute to any one place.

## Share KV heads without copying them

Six query heads need to read one KV head without undoing what GQA saved. Most
attention kernels want one KV head per query head,
and the obvious way to get there ==gives back the whole GQA saving==. The naive
`repeat_kv` duplicates:

```python
def repeat_kv(x, repeats):
    # x: (batch, kv_heads, seq, head_dim)
    batch, heads, seq, dim = x.shape
    return (
        x[:, :, None, :, :]                             # (b, 4, 1, L, 256)
        .expand(batch, heads, repeats, seq, dim)        # (b, 4, 6, L, 256), a view
        .reshape(batch, heads * repeats, seq, dim)      # (b, 24, L, 256), a copy
    )
```

The two steps cost very different amounts:

- **`expand` is free.** It's a view with stride 0 along the new axis, so the six
  copies are the same memory read six times.
- **`reshape` isn't free.** The expanded tensor isn't contiguous, so `reshape`
  falls back to `contiguous()` and materializes every copy.

The bytes make it plain. At 32k context, for one sequence and one layer in
bfloat16:

$$
\underbrace{2 \times 4 \times 32768 \times 256 \times 2}_{\text{stored}} = 128 \ \text{MiB}
\qquad \longrightarrow \qquad
\times 6 = 768 \ \text{MiB}
$$

You've written 640 MiB of redundant data, and you're about to read it. The whole
point of GQA was to read 128 MiB instead of 768. The naive path gives back the
saving and adds a write on top.

Keep it anyway for correctness checks: it's three lines and plainly right. A
real kernel never calls it. It indexes the shared head instead, which is what
the `kv_group` parameter does in chapter 14's FlashAttention kernel:

```python
kv_head = head // kv_group     # query head 13 reads KV head 2
```

Query head $h$ reads KV head $\lfloor h/6 \rfloor$. The six queries in a group
read the same cache lines, which usually land in L2, the GPU's on-chip cache,
anyway. Lab 07 checks exactly this indexing convention.

## The naive implementation, and why you keep it

The naive implementation is four lines, and you can't serve with it. You still
want it, as the reference every fast kernel gets checked against:

```python
scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
scores = scores.masked_fill(mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)
out = torch.matmul(weights.to(v.dtype), v)
```

It materializes a $(B, H, T, L)$ tensor, one score for every query and key pair,
and memory grows fast:

- At 8k context with 24 heads and a batch of one, the scores alone are
  $24 \times 8192 \times 8192 \times 2$ bytes, or 3 GiB, in bfloat16.
- The listing computes in float32, which doubles that.
- It allocates the scores, the masked copy, and the softmax output as separate
  tensors.

Peak memory runs well past 12 GB for a single sequence. Measured against a
Triton FlashAttention kernel on the target hardware, the naive path uses 27.5
times the peak memory.

> [!TIP] Keep it as your ground truth
> The naive path is the reference for every kernel you write from chapter 14
> onward, and it's short enough to read in one sitting. Run it in float32 on
> short sequences and compare.

## Where decode time goes

With attention in place, you can account for a whole decode step. At batch
size 1, reading the weights dominates, and the cache is a small
addition that grows with context. A decode step reads the following, once per
token generated:

- All 53.8 GB of weights.
- The KV cache for the full-attention layers: 64 KiB per token of context.
- The recurrent state from the linear layers: 147.8 MiB, fixed.

Divide the 53.8 GB weight read by two bandwidths for the PCIe A100:

| Bandwidth | GB/s | Weight read | Tokens per second |
|---|---|---|---|
| Rated | 1935 | 27.8 ms | 36 |
| Measured device-to-device copy | 1275 | 42.2 ms | 24 |

Take the second number. The rated bandwidth is what the memory can do, and the
copy is what you get. Chapter 10 is about the gap.

At 32k context, the cache adds 2.0 GiB and the recurrent state 0.14 GiB. At
1275 GB/s, those two together take about 1.8 ms. That's small next to 42 ms,
but it grows with both context and batch size, and the weight read doesn't.

That asymmetry is ==the entire argument for batching==. Sixteen sequences read
the 53.8 GB of weights once and pay 16 times the cache. Throughput goes up
almost 16 times, and per-token latency barely moves. Chapter 16 builds the
scheduler that exploits it.

## What goes wrong

Each symptom points at one likely cause:

- **Fluent output that ignores the prompt after the first generated token.** The
  causal mask is missing its offset. It's correct during prefill and wrong during
  decode.
- **NaN logits on the first forward pass.** A mask row blocks every key, so the
  softmax divides zero by zero. Check the inequality direction.
- **Attention output with the wrong magnitude and no other symptom.** The mask is
  applied after the softmax instead of before, so the weights no longer sum to 1.
- **Out of memory during prefill at long context.** The naive path's
  $(B, H, T, L)$ score tensor. Use `F.scaled_dot_product_attention`, or chapter
  14.
- **Quality drop with no traceable cause.** The output gate is never applied,
  because `q_proj`'s output was sliced rather than chunked.
- **Slower than MHA despite GQA.** `repeat_kv` is in the hot path, materializing
  six copies of the cache on every step.
- **Shapes that pass and results that don't.** The code assumes
  `head_dim == hidden_size // num_heads`. It's 256 here, not 213.

> [!RECAP]
> - Attention output is a weighted average of values with weights that sum to 1,
>   and every past key and value stays in the cache.
> - The $\sqrt{d_h}$ scale guards against saturation, not overflow.
> - Mask before the softmax, shifted by `kv_len - q_len`.
> - GQA's cache is 64 KiB per token, and decode attention's intensity is
>   $H/H_{kv} = 6$ FLOPs per byte at any context or batch, so only reading fewer
>   bytes speeds it up.
> - Chunk `q_proj` into queries and a gate, and index the shared KV head
>   instead of copying it.

## Check your understanding

> [!QUESTION] Why doesn't the arithmetic intensity of decode attention improve at longer context?
> Both sides of the ratio are linear in $L$. A longer context means
> proportionally more dot products and proportionally more cache bytes to read,
> so $L$ cancels and the intensity stays at $2H/(H_{kv} b) = 6$.

> [!QUESTION] This model's KV projections are only 10% of a full-attention layer's parameters. Why is GQA worth doing at all?
> Because the saving is in the cache, not the weights. The weights are read once
> per forward pass regardless of context. The cache is also read once per forward
> pass, and it's 64 KiB per token of context, per sequence. At 32k context and
> batch 8, that's 16 GiB of the card, against 3.4 GB for all 16 layers' attention
> weights combined.

> [!QUESTION] What would happen if you removed the $1/\sqrt{d_h}$ scale but kept the per-head RMSNorm on queries and keys?
> The norm bounds the vector magnitudes but not the dimension count. With
> unit-norm rows the dot product is a cosine in $[-1, 1]$, so scores would be too
> small rather than too large: the softmax would flatten toward uniform instead
> of saturating. The scale and the normalization control different things, and
> this model uses both. The norm bounds what the projection produces, and the
> scale compensates for $d_h$.

> [!QUESTION] A batch of 16 sequences at 4k context. How many bytes does one decode step read from the KV cache?
> $64 \ \text{KiB} \times 4096 \times 16 = 4$ GiB. Against 53.8 GB of weights
> read once for the whole batch, the cache is now 7% of the traffic. At 32k it
> would be 32 GiB, more than half the size of the weight read. That's why the
> scheduler in chapter 16 cares about total context in the batch, not the number
> of sequences.

## Lab

> [!TRY]
> Implement `repeat_kv`, `causal_mask`, `attention`, `apply_output_gate`, and
> `kv_bytes_per_token`. You pass when every harness check that follows succeeds.

The harness checks that:

- `repeat_kv` produces 24 heads from 4, and query head $h$ reads KV head
  $\lfloor h/6 \rfloor$, not an interleaved order.
- `causal_mask` gives the strict upper triangle when `q_len == kv_len`, masks
  nothing for a single decode query behind a 9-token prefix, and masks exactly 3
  entries for a 3-token chunk behind a 7-token prefix.
- `attention` matches `F.scaled_dot_product_attention` to $10^{-4}$, causal and
  non-causal, on grouped heads.
- A single decode query against a cached prefix produces the same result as the
  last row of a full prefill.
- With all-ones values, every output entry is exactly 1. That's the convexity
  property, and it catches a missing or misplaced normalization.
- `apply_output_gate` is an elementwise sigmoid, and a gate of $-20$ suppresses
  the output to under $10^{-6}$.
- `kv_bytes_per_token` returns 65,536 for 4 KV heads over 16 layers, and exactly
  six times that for 24.

## Further reading

- [GQA: training generalized multi-query transformer models from multi-head checkpoints](https://arxiv.org/abs/2305.13245)
- [Fast transformer decoding: one write-head is all you need](https://arxiv.org/abs/1911.02150) — multi-query attention.
- [Attention is all you need](https://arxiv.org/abs/1706.03762) — where the scaled dot product and the multi-head split come from.
- [Efficient memory management for large language model serving with PagedAttention](https://arxiv.org/abs/2309.06180) — what chapter 15 builds.
- [Efficient streaming language models with attention sinks](https://arxiv.org/abs/2309.17453) — the null-target behavior the output gate removes the need for.
