---
title: Grouped-query attention
slug: 07-grouped-query-attention
part: "Part 2 — A forward pass"
summary: Scaled dot-product attention derived from scratch, then the 16 layers that keep a real KV cache, with head sharing and an output gate.
minutes: 110
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

Sixteen of the 64 layers use ordinary softmax attention. They are the layers that
own a KV cache, so they set the memory ceiling and dominate decode time. Chapter
6 covered the other 48.

This chapter builds attention from the start rather than quoting the formula.
Everything in Parts 3 through 6 — the cache, the roofline, FlashAttention, paged
attention, quantized caches — is an argument about the shapes and byte counts
derived here, so it is worth getting them exactly right once.

## Before you start

**Softmax.** Chapter 0a covers it, including shift invariance: subtracting a
constant from every logit leaves the output unchanged, which is what makes a
numerically stable implementation possible. The two properties used here are that
the outputs are non-negative and that they sum to 1.

**Variance of a sum.** For independent zero-mean random variables,
$\operatorname{Var}(\sum_i X_i) = \sum_i \operatorname{Var}(X_i)$. The scale
factor argument is one application of that.

**Tensor shapes.** Everything is `(batch, heads, seq, head_dim)`. Chapter 0a
covers the convention and `torch.einsum`. The shape table below is the thing to
keep open while reading the code.

**This model's geometry**, which the arithmetic uses throughout:

| Field | Value |
|---|---|
| `hidden_size` | 5120 |
| `num_attention_heads` | 24 |
| `num_key_value_heads` | 4 |
| `head_dim` | 256 |
| `attn_output_gate` | true |
| Full-attention layers | 16 of 64 |
| Bytes per element | 2, bfloat16 |

## Queries, keys, and values

Attention is a differentiable lookup in a table whose rows are the tokens seen so
far.

Each past token $j$ produces two vectors: a key $k_j \in \mathbb{R}^{d_h}$, which
advertises what that token has to offer, and a value
$v_j \in \mathbb{R}^{d_h}$, which is the content it hands over. The current token
$t$ produces a query $q_t \in \mathbb{R}^{d_h}$, which describes what it is
looking for. All three are linear projections of the residual stream, so the
model learns what "offer" and "want" mean.

A hard lookup would pick the single $j$ whose key best matches the query. That is
not differentiable. Instead, score every key, turn the scores into a probability
distribution, and return the weighted average:

$$
s_{tj} = \frac{q_t^\top k_j}{\sqrt{d_h}},
\qquad
w_{tj} = \frac{\exp(s_{tj})}{\sum_{i \le t} \exp(s_{ti})},
\qquad
o_t = \sum_{j \le t} w_{tj} \, v_j
$$

In matrix form, for a whole block of queries at once:

$$
\operatorname{Attention}(Q, K, V)
= \operatorname{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_h}} + M \right) V
$$

where $M$ is the causal mask, covered below. Three properties follow immediately
and are worth naming, because later chapters lean on all three.

**The output is a convex combination.** The weights are non-negative and sum to
1, so $o_t$ lies inside the convex hull of the values it read. Attention cannot
amplify, and it cannot return zero unless the values themselves sum to zero. That
limitation is what the output gate later in this chapter exists to remove.

**The query never leaves the exponential.** $\exp(q_t^\top k_j)$ does not factor
into something depending on $q_t$ times something depending on $k_j$, so there is
no fixed-size summary of the past. Every past key and value has to be kept. That
is the KV cache, and chapter 6 is what you get if you refuse to pay for it.

**Cost is quadratic in prefill and linear in decode.** Scoring $L$ queries
against $L$ keys is $O(L^2)$ work. Scoring one query against $L$ keys is $O(L)$
work but reads $O(L)$ bytes, which turns out to be the binding constraint.

## Where the 1 over root d comes from

The $\sqrt{d_h}$ divisor is not cosmetic. Here is the argument.

Model the entries of $q$ and $k$ as independent, mean zero, and unit variance —
which is roughly what a well-initialized projection followed by a normalization
produces. The unscaled score is a sum of $d_h$ terms:

$$
q^\top k = \sum_{i=1}^{d_h} q_i k_i
$$

Each term has mean $\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$ by
independence, and variance

$$
\operatorname{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] = \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = 1
$$

The $d_h$ terms are independent, so the variances add:

$$
\operatorname{Var}(q^\top k) = d_h,
\qquad
\operatorname{sd}(q^\top k) = \sqrt{d_h}
$$

The typical magnitude of a raw score grows like $\sqrt{d_h}$. With
$d_h = 256$ that is a standard deviation of 16, so scores in a row routinely
differ by tens.

Now feed that to a softmax. Two logits separated by a gap $g$ produce weights in
the ratio $e^{g}$. A gap of 16 gives

$$
e^{16} \approx 8.9 \times 10^{6}
$$

The largest score takes essentially all the mass and the distribution collapses
to a hard argmax. At training time the gradient through a saturated softmax is
about zero in every direction, so the layer stops learning. At inference time it
means the head reads exactly one token, throwing away the averaging that makes
attention useful.

Dividing by $\sqrt{d_h}$ sets the score variance back to 1, independent of head
width, so gaps are of order 1 and weight ratios of order $e$. For this model:

$$
\text{scale} = \frac{1}{\sqrt{256}} = \frac{1}{16} = 0.0625
$$

which is exactly `head_dim**-0.5` in `engine/layers/attention.py`.

One thing the scale does *not* fix is overflow, and it is worth being precise
about that because the two get conflated. Softmax is shift invariant, so a
correct implementation subtracts the row maximum before exponentiating and never
overflows regardless of logit scale. The scale exists for saturation and
gradients, not for numerical range. Chapter 14 uses the same shift invariance to
build FlashAttention's online softmax.

This model adds a second guard. Qwen3 applies RMSNorm per head to queries and
keys over the 256-wide head dimension, before the rotary embedding, so the logit
scale is bounded by the learned norm weights rather than by whatever the
projections happen to produce. In the tensor listing these appear as
`q_norm.weight` and `k_norm.weight`, each shape `[256]`.

## The causal mask

A language model predicts token $t+1$ from tokens $1 \ldots t$. If position $t$
could attend to position $t+1$, the training objective would be trivially
satisfiable and the model would be useless at inference, when the future does not
exist. So scores where $j > t$ must be removed.

### Why the mask goes before the softmax

The mask sets forbidden scores to $-\infty$ *before* exponentiating, so
$\exp(-\infty) = 0$ and those keys contribute nothing to either the numerator or
the denominator. The surviving weights still sum to 1.

The alternative — compute the full softmax, then multiply the forbidden weights
by zero — is wrong, and the reason is the denominator. Writing $w_j$ for the
unmasked weights and $m_j \in \{0,1\}$ for the mask:

$$
\sum_j m_j w_j = \frac{\sum_{j \le t} \exp(s_j)}{\sum_{\text{all } j} \exp(s_j)} < 1
$$

The output is scaled down by a factor that depends on how attractive the future
keys were. A position whose future is highly relevant gets its output shrunk
toward zero; a position whose future is irrelevant does not. That is a
position-dependent, content-dependent attenuation, which is not what anyone
intended.

### The offset that matters

During prefill, queries and keys cover the same positions and the mask is the
strict upper triangle. During decode, the keys include a cached prefix that the
queries do not cover, and the indices no longer line up.

Let `q_len` be the number of queries in this call and `kv_len` the total number
of keys, cache included. The queries are always the *last* `q_len` positions, so
query row $i$ sits at absolute position

$$
\text{pos}(i) = i + (\text{kv\_len} - \text{q\_len})
$$

and may attend to key positions up to and including that. In code:

```python
offset = kv_len - q_len
idx_q = torch.arange(q_len, device=device).unsqueeze(-1)   # (q_len, 1)
idx_k = torch.arange(kv_len, device=device).unsqueeze(0)   # (1, kv_len)
mask = idx_k > idx_q + offset                              # (q_len, kv_len), True = blocked
scores = scores.masked_fill(mask, float("-inf"))
```

Three regimes, all of which lab 07 checks:

| Call | `q_len` | `kv_len` | `offset` | Entries masked |
|---|---|---|---|---|
| Prefill | 4 | 4 | 0 | 6, the strict upper triangle |
| Decode step | 1 | 10 | 9 | 0, the single query sees everything |
| Chunked prefill | 3 | 10 | 7 | 3 |

Check the last row by hand. The three queries are at positions 7, 8, and 9. The
query at 7 must not see keys 8 and 9, which is two entries. The query at 8 must
not see key 9, which is one. The query at 9 sees all ten. Total 3.

Forgetting the offset produces a mask that is correct during prefill, when the
offset is zero, and wrong during every decode step. The symptom is a model that
generates well for its prompt and then drifts, because each decode step hides
most of its own context. It is a quiet failure: no crash, no NaN, plausible
output.

There is a loud failure available too. If a row of the mask blocks every key, the
softmax computes $0/0$ and returns NaN — and a stable implementation that
subtracts the row maximum gets $-\infty - (-\infty)$ first. That cannot happen
with a correct causal mask, because query $i$ can always see its own position,
but it happens immediately with an off-by-one in the wrong direction.

## Shapes, end to end

One full-attention layer, for a batch of $B$ sequences and $T$ new tokens against
a cache of $L$ total positions.

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

**`head_dim` is not `hidden_size / num_heads`.** $5120 / 24 = 213.33$, which is
not an integer, and the model uses 256 anyway. Most transformers tie the two, so
most readers assume the tie holds. Here the attention block is genuinely wider
than the residual stream: $24 \times 256 = 6144$ against 5120. The `o_proj` is
what projects back down. Assuming the tie gives you a `view` that fails, or worse,
one that succeeds with the wrong stride.

**The projections are not square.** Per full-attention layer:

$$
\underbrace{5120 \times 12288}_{\text{q and gate}}
+ \underbrace{2 \times 5120 \times 1024}_{\text{k and v}}
+ \underbrace{6144 \times 5120}_{\text{o}}
= 104{,}857{,}600
$$

That is 104.9M parameters, 210 MB in bfloat16, and across 16 layers, 3.4 GB.
The KV projections are the small ones — 10.5M of the 104.9M — which is the first
hint that shrinking the KV heads costs less than it sounds like it should.

## Multi-head attention

Everything above describes one head. Real attention runs $H$ of them in parallel,
each with its own $Q$, $K$, $V$ projections into a $d_h$-dimensional subspace,
then concatenates the outputs and mixes them with `o_proj`.

### Why heads help

One head produces one score per (query, key) pair, hence one probability
distribution per query, hence one weighted average. A token that needs to gather
the subject of its clause *and* the topic of the paragraph *and* the matching
open bracket cannot do it: those are three different rankings of the context, and
a single softmax commits to one.

With 24 heads it gets 24 independent rankings, 24 separate averages, and 6144
channels of output that `o_proj` can combine however it likes. Attention patterns
in trained models are visibly specialized for this reason — some heads track
syntax, some track repeated tokens, some attend almost entirely to position 1 as
a null target.

The striking part is that heads are free in arithmetic. One head of width
$H d_h$ costs $2 \cdot T \cdot L \cdot H d_h$ FLOPs for the score matmul.
$H$ heads of width $d_h$ each cost $2 \cdot T \cdot L \cdot d_h$, and there are
$H$ of them, for the same total. Splitting the width into heads changes the
expressiveness and not the FLOP count.

What each head gives up is rank. A head's scores are inner products in a
256-dimensional subspace, so the similarity structure it can express is limited
to that subspace. Narrow heads are cheap and weak; wide heads are expressive and
few. This model's choice of 256 is unusually wide — 64 or 128 is more common —
and pairs with the per-head RMSNorm and the partial rotary factor from chapter 5,
where only the first 64 of the 256 channels are rotated.

## MHA, MQA, and GQA

Multi-head attention gives every query head its own keys and values, which means
the cache holds $H$ heads' worth. That is the dominant memory cost at long
context, and the spectrum of fixes is about how many KV heads you keep.

- **MHA**: $H_{kv} = H$. Every query head has private keys and values.
- **MQA**: $H_{kv} = 1$. All query heads share one set.
- **GQA**: $1 < H_{kv} < H$. Query heads are partitioned into $H_{kv}$ groups of
  $g = H / H_{kv}$, and each group shares a KV head.

This model uses $H = 24$, $H_{kv} = 4$, so the group size is

$$
g = \frac{24}{4} = 6
$$

### The cache arithmetic

The cache stores $K$ and $V$ for every position, in every full-attention layer.
Per token, per layer:

$$
\text{bytes} = 2 \times H_{kv} \times d_h \times b
$$

The leading 2 counts $K$ and $V$; $b = 2$ is bfloat16. Over this model's 16
full-attention layers:

| Scheme | $H_{kv}$ | Per layer per token | Over 16 layers | At 32k context |
|---|---|---|---|---|
| MHA | 24 | 24,576 B = 24 KiB | 384 KiB | 12 GiB |
| GQA | 4 | 4,096 B = 4 KiB | 64 KiB | 2 GiB |
| MQA | 1 | 1,024 B = 1 KiB | 16 KiB | 0.5 GiB |

Check the GQA row: $2 \times 4 \times 256 \times 2 = 4096$ bytes per layer,
times 16 layers is 65,536 bytes, exactly 64 KiB per token. That is the number
chapter 2 uses and lab 07 checks. At 32,768 tokens,
$65{,}536 \times 32{,}768 = 2$ GiB per sequence.

Now put that against the A100's budget. The weights are 53.8 GB of an 80 GB card,
leaving about 26 GB. With GQA at 32k context you fit around eight sequences. With
MHA you fit two. The difference between those two numbers is the difference
between a server and a demo.

For the extra context, chapter 2's "all-full-attention" figure of 256 KiB per
token is a different comparison: it is GQA applied to all 64 layers rather than
16. The 384 KiB above is MHA applied to the 16 layers this model actually has.

### What it costs in quality

The trade is real but small, and the published result is worth stating plainly.
The GQA paper takes a trained MHA checkpoint, mean-pools its KV heads down to
$H_{kv}$ groups, and uptrains briefly. MQA — collapsing all the way to one KV
head — loses measurable quality. A handful of groups recovers nearly all of it
while keeping most of the cache saving, because the saving is a $1/H_{kv}$ curve
that is already mostly flat by the time $H_{kv}$ reaches 4 or 8.

Two details make this model's case better than the paper's. First, it was trained
with four KV heads from the start, so there is no conversion loss — the query
heads in a group learned to share their keys rather than having sharing imposed
on them. Second, only 16 of its layers use attention at all, so the layers that
would suffer most from a narrow cache are the minority.

The quality argument is also asymmetric in a way that favors decode. Reducing
$H_{kv}$ costs a little expressiveness on every token, and saves bytes on every
token *of context*. At long context the second effect dominates by orders of
magnitude, which is why nothing serves long context with MHA any more.

## Why decode attention is permanently memory bound

This is the calculation that explains the rest of the course. Chapter 10 states
the result; here is the derivation.

Take one decode step, one sequence, one layer, at context length $L$. The query
is a single token.

**FLOPs.** Two matmuls. The scores are $H$ dot products of length $d_h$ against
$L$ keys each:

$$
H \cdot L \cdot d_h \ \text{multiply-adds} = 2 H L d_h \ \text{FLOPs}
$$

Then the weighted sum over values, which is the same shape again:

$$
\text{FLOPs} = 2 H L d_h + 2 H L d_h = 4 H L d_h
$$

**Bytes.** The whole KV cache for this layer has to come off HBM: $K$ and $V$,
each $H_{kv} \times L \times d_h$ elements at $b$ bytes.

$$
\text{bytes} = 2 H_{kv} L d_h b
$$

The query itself is $H d_h$ elements, independent of $L$, so it drops out at any
interesting context length.

**The intensity.** Divide:

$$
I = \frac{4 H L d_h}{2 H_{kv} L d_h \, b}
$$

Now cancel. $d_h$ cancels, and so does $L$ — the numerator and denominator both
scale linearly with context length, because a longer context means proportionally
more arithmetic *and* proportionally more bytes:

$$
I = \frac{4H}{2 H_{kv} b} = \frac{2H}{H_{kv} \, b}
$$

Substitute $b = 2$ for bfloat16:

$$
I = \frac{H}{H_{kv}} = g = 6 \ \text{FLOPs per byte}
$$

**Exactly the GQA group size.** Not approximately, and not for this model in
particular — the equality $I = g$ holds for any GQA model with a bfloat16 cache.

Three consequences follow, and they shape everything in Parts 3 to 6.

**Context length does not help.** $L$ cancelled. Attention at 128k context has
the same arithmetic intensity as attention at 128 tokens. It takes a thousand
times longer.

**Batching does not help either.** The KV cache is per sequence, so a batch of
$N$ sequences does $N$ times the FLOPs and reads $N$ times the bytes. Compare
with the MLP, whose weights are shared across the batch and whose intensity is
therefore roughly the batch size — chapter 10 has that table. Batching is the
main lever for everything except attention.

**The A100's ridge point is 161 FLOPs per byte.** At 6, decode attention sits 27
times below it, using under 4% of the card's arithmetic. No kernel fixes that.
The only lever left is the denominator: read fewer bytes. That is precisely what
GQA does (raise $g$), what paged attention does in chapter 15 (stop reading
padding), and what cache quantization does in chapter 18 — set $b = 1$ for int8
and the intensity doubles to 12, because $b$ is in the denominator.

## The output gate

The query projection emits $24 \times 256 \times 2 = 12288$ channels, not 6144.
The second half is a gate:

```python
q = self.q_proj(x)                    # (batch, seq, 12288)
q, gate = q.chunk(2, dim=-1)          # (batch, seq, 6144) each
# ... attention, producing out: (batch, seq, 6144) ...
out = out * torch.sigmoid(gate)       # elementwise, still (batch, seq, 6144)
return self.o_proj(out)               # (batch, seq, 5120)
```

The motivation goes back to the convexity property from the start of the chapter.
Softmax weights are non-negative and sum to 1, so a head returns a convex
combination of values whatever the input. There is no way to say "nothing in this
context is relevant". Models without a gate learn a workaround: they dedicate one
position, usually the first token, as a null target, and heads dump their
attention mass there when they have nothing to contribute. Those attention sinks
are a well documented nuisance — they complicate cache eviction, quantization,
and long-context extrapolation, all because the architecture gave the head no off
switch.

The gate is the off switch. `sigmoid(gate)` near zero suppresses that head's
contribution to the residual stream entirely, per channel, per token. It is the
same motivation as the SiLU gate on the linear-attention layers in chapter 6,
though this one is a sigmoid and is therefore bounded to $(0,1)$: it can
attenuate but not amplify.

It costs one extra $5120 \times 6144$ matrix per layer:

$$
5120 \times 6144 = 31{,}457{,}280 \ \text{parameters} = 62.9 \ \text{MB in bfloat16}
$$

Across 16 layers that is about 1.0 GB of the model's 53.8 GB, spent on the
ability to output nothing.

The failure mode is quiet. Miss the split and slice the first 6144 channels
instead, and the shapes still work perfectly — you get a tensor of exactly the
right size. Half the projection's learned behavior is silently discarded and the
gate is never applied, so every head contributes at full strength on every token.
Quality drops in a way that is hard to attribute to any one place.

## repeat_kv, and why the obvious implementation defeats the purpose

Most attention kernels want one KV head per query head. The naive way to get
there duplicates:

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

`expand` is free: it is a view with stride 0 along the new axis, so the six
copies are the same memory read six times. `reshape` is not free. The expanded
tensor is not contiguous, so `reshape` falls back to `contiguous()` and
materializes every copy.

The bytes say it plainly. At 32k context, one sequence, one layer, bfloat16:

$$
\underbrace{2 \times 4 \times 32768 \times 256 \times 2}_{\text{stored}} = 128 \ \text{MiB}
\qquad \longrightarrow \qquad
\times 6 = 768 \ \text{MiB}
$$

You have written 640 MiB of redundant data and you are about to read it. The
whole point of GQA was to read 128 MiB instead of 768. The naive path gives back
the saving and adds a write on top.

Keep it anyway, for correctness checks — it is three lines and plainly right. A
real kernel never calls it. It indexes the shared head instead, which is what the
`kv_group` parameter does in the FlashAttention kernel in chapter 14:

```python
kv_head = head // kv_group     # query head 13 reads KV head 2
```

Query head $h$ reads KV head $\lfloor h/6 \rfloor$, and the six queries in a
group read the same cache lines, which usually land in L2 anyway. Lab 07 checks
exactly this indexing convention.

## The naive implementation, and why you keep it

```python
scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
scores = scores.masked_fill(mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)
out = torch.matmul(weights.to(v.dtype), v)
```

This materializes a $(B, H, T, L)$ tensor. At 8k context with 24 heads and a
batch of one:

$$
24 \times 8192 \times 8192 \times 2 \ \text{bytes} = 3 \ \text{GiB}
$$

in bfloat16 — and the listing above computes in float32, which doubles it, and
allocates the scores, the masked copy, and the softmax output as separate
tensors. Peak memory runs well past 12 GB for a single sequence. You cannot serve
with it. Measured against a Triton FlashAttention kernel on the target hardware,
the naive path uses 27.5 times the peak memory.

Keep it anyway. It is the ground truth for every kernel you write from chapter 14
onward, and it is short enough to read in one sitting. Run it in float32 on short
sequences and compare.

## Where decode time goes

A decode step at batch size 1 reads, once per token generated:

- All 53.8 GB of weights.
- The KV cache for the full-attention layers: 64 KiB per token of context.
- The recurrent state from the linear layers: 147.8 MiB, fixed.

Do the division. Against the PCIe A100's rated 1935 GB/s:

$$
\frac{53.8 \times 10^9}{1935 \times 10^9} = 27.8 \ \text{ms}
\quad \Longrightarrow \quad 36 \ \text{tokens per second}
$$

Against the 1275 GB/s a device-to-device copy actually achieves on this card:

$$
\frac{53.8 \times 10^9}{1275 \times 10^9} = 42.2 \ \text{ms}
\quad \Longrightarrow \quad 24 \ \text{tokens per second}
$$

Take the second number. The rated bandwidth is what the memory is capable of; the
copy is what you get. Chapter 10 is about the gap.

At 32k context the cache adds 2.0 GiB and the recurrent state 0.14 GiB, together
about 1.8 ms at 1275 GB/s. Small next to 42 ms — but it grows with both context
and batch size, and the weight read does not.

That asymmetry is the entire argument for batching. Sixteen sequences read the
53.8 GB of weights once and pay 16 times the cache. Throughput goes up almost 16
times; per-token latency barely moves. Chapter 16 builds the scheduler that
exploits it.

## What goes wrong

**Fluent output that ignores the prompt after the first generated token.** The
causal mask is missing its offset. Correct during prefill, wrong during decode.

**NaN logits on the first forward pass.** A mask row that blocks every key, so
the softmax divides zero by zero. Check the inequality direction.

**Attention output with the wrong magnitude, no other symptom.** The mask applied
after the softmax instead of before, so the weights no longer sum to 1.

**Out of memory during prefill at long context.** The naive path's
$(B, H, T, L)$ score tensor. Use `F.scaled_dot_product_attention`, or chapter 14.

**Quality drop with no traceable cause.** The output gate never applied, because
`q_proj`'s output was sliced rather than chunked.

**Slower than MHA despite GQA.** `repeat_kv` in the hot path, materializing six
copies of the cache on every step.

**Shapes that pass and results that do not.** Assuming
`head_dim == hidden_size // num_heads`. It is 256 here, not 213.

## Check your understanding

**Why does the arithmetic intensity of decode attention not improve at longer
context?**

Both sides of the ratio are linear in $L$. A longer context means proportionally
more dot products and proportionally more cache bytes to read, so $L$ cancels and
the intensity stays at $2H/(H_{kv} b) = 6$.

**This model's KV projections are only 10% of a full-attention layer's
parameters. Why is GQA worth doing at all?**

Because the saving is in the cache, not the weights. The weights are read once
per forward pass regardless of context; the cache is read once per forward pass
and is 64 KiB per token of context, per sequence. At 32k context and batch 8 that
is 16 GiB of the card, against 3.4 GB for all 16 layers' attention weights
combined.

**What would happen if you removed the $1/\sqrt{d_h}$ scale but kept the per-head
RMSNorm on queries and keys?**

The norm bounds the vector magnitudes but not the dimension count. With unit-norm
rows the dot product is a cosine in $[-1, 1]$, so scores would be too small rather
than too large — the softmax would flatten toward uniform instead of saturating.
Either way the scale and the normalization control different things, and this
model uses both: the norm bounds what the projection produces, and the scale
compensates for $d_h$.

**A batch of 16 sequences at 4k context. How many bytes does one decode step read
from the KV cache?**

$64 \ \text{KiB} \times 4096 \times 16 = 4$ GiB. Against 53.8 GB of weights read
once for the whole batch, the cache is now 7% of the traffic — and at 32k it would
be 32 GiB, more than half. This is why the scheduler in chapter 16 cares about
total context in the batch, not the number of sequences.

## Lab

Implement `repeat_kv`, `causal_mask`, `attention`, `apply_output_gate`, and
`kv_bytes_per_token`. The harness checks:

- `repeat_kv` produces 24 heads from 4, and query head $h$ reads KV head
  $\lfloor h/6 \rfloor$ — not an interleaved order.
- `causal_mask` gives the strict upper triangle when `q_len == kv_len`, masks
  nothing for a single decode query behind a 9-token prefix, and masks exactly 3
  entries for a 3-token chunk behind a 7-token prefix.
- `attention` matches `F.scaled_dot_product_attention` to $10^{-4}$, causal and
  non-causal, on grouped heads.
- A single decode query against a cached prefix produces the same result as the
  last row of a full prefill.
- With all-ones values, every output entry is exactly 1 — which is the convexity
  property, and catches a missing or misplaced normalization.
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
