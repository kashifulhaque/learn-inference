---
title: RMSNorm and the residual stream
slug: 04-rmsnorm-and-residuals
part: "Part 2 — A forward pass"
summary: The normalization every layer uses, why its sum has to run in float32, and how the residual stream and pre-norm keep a 64-layer model stable.
minutes: 70
gpu: true
objectives:
  - Write RMSNorm and LayerNorm as formulas, and say what dropping the mean subtraction costs and saves.
  - Derive, from the width of the bfloat16 significand, why a bfloat16 sum over 5120 elements comes out systematically low.
  - Describe the residual stream as a sum of layer contributions, and give the gradient argument for pre-norm.
  - Explain the per-head query and key norm and the bound it puts on attention scores.
  - Show that RMSNorm is bandwidth bound by comparing its arithmetic intensity with the A100 ridge point.
lab: 04-rmsnorm
---

# RMSNorm and the residual stream

> [!TLDR]
> - RMSNorm rescales each token's vector to a standard size, then multiplies
>   each channel by a learned gain. It's LayerNorm without the mean subtraction
>   or the shift.
> - Its one sum runs over 5120 numbers, and bfloat16 is too coarse for that.
>   Once the running total is about 256 times the next term, adding the term
>   changes nothing. So the sum runs in float32.
> - The residual stream is the running total that every layer adds its output
>   to. Normalizing each layer's input keeps a 64-layer model trainable, and
>   one last norm fixes the scale of the model's output scores.
> - Normalizing each attention head's queries and keys caps every attention
>   score at 16.
> - RMSNorm spends its time moving data, not computing, so the only way to
>   make it faster is to move fewer bytes.

This model has 64 layers, and each one normalizes its input twice: once before
the *token mixer*, the attention or linear-attention sublayer that moves
information between tokens, and once before the MLP. One more normalization
sits at the top of the stack. That's 129 invocations per forward pass, which
makes normalization the most executed operation in the model after the matrix
multiplies.

It's also the first place where numerics bite. RMSNorm is four arithmetic
operations per element and one sum. The sum runs over 5120 terms in a format
with 8 bits of precision, and that's enough to break it. The first half of this
chapter shows why and what the fix costs.

The second half is about what RMSNorm normalizes: the *residual stream*, the
running total that carries each token through the model. It's the model's
working memory, and it's the reason a 64-layer network trains at all.

## Before you start

**Root mean square.** For a vector $x \in \mathbb{R}^{d}$, the root mean square
is the square root of the average square:

$$
\operatorname{rms}(x) = \sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2}
$$

It measures the typical size of an entry. It equals the standard deviation of
$x$ only when the mean of $x$ is zero.

**Floating point, enough of it.** bfloat16 spends 1 bit on the sign, 8 on the
exponent, and 7 stored bits on the significand. Counting the implicit leading 1,
that's 8 bits of precision; float32 has 24. The gap between neighbouring
representable numbers near a value $S$ is one *ulp* (unit in the last place),
and it grows with $S$. For $2^{e} \le S < 2^{e+1}$, one ulp is $2^{e-7}$ in
bfloat16. The [notation chapter](/c/00a-notation-and-prerequisites) has the
formats in full.

**The ridge point.** An A100 80GB does about 312 TFLOP/s (trillion
floating-point operations per second) of bfloat16 matrix maths on its *tensor
cores*, the units built for matrix multiplies. It moves 1935 GB/s to and from
*HBM*, the GPU's main memory. Dividing the two, a kernel must do about 161 FLOPs
per byte it moves to keep the tensor cores busy. That ratio is the *ridge
point*, and a kernel below it is *memory bound*: it waits on memory, not on
arithmetic. [The roofline chapter](/c/10-roofline) develops this properly.

**PyTorch idioms.** `x.pow(2).mean(dim=-1, keepdim=True)` reduces over the last
axis and keeps it as size 1, so the result broadcasts back against `x`.
`torch.rsqrt` is the reciprocal square root, computed as one operation instead
of a square root followed by a division.

## The operation

This section says what RMSNorm computes. In words: take one token's vector,
measure the typical size of its entries, divide by that, and then multiply each
channel by a learned gain. Every token leaves with the same typical size,
whatever size it arrived with.

A two-element row shows the idea. Take $x = (3, 4)$:

1. The mean of squares is $(9 + 16)/2 = 12.5$.
2. Its square root is about 3.536.
3. Dividing gives about $(0.849, 1.131)$, whose root mean square is 1.

As a formula, for a row $x \in \mathbb{R}^{d}$, divide each element by the root
of the mean square plus a small constant, then multiply by the gain:

$$
y_i = \frac{x_i}{\sqrt{\hlc{\operatorname{ms}(x)} + \hla{\epsilon}}}\, \hlb{w_i},
\qquad
\hlc{\operatorname{ms}(x)} = \frac{1}{d}\sum_{j=1}^{d} x_j^2
$$

The pieces are the following:

- **The mean of squares** $\hlc{\operatorname{ms}(x)}$ runs over $d$ channels.
  For the residual-stream norms, $d$ is 5120.
- **The constant** $\hla{\epsilon}$ is `rms_norm_eps`, which is $10^{-6}$ in
  this config.
- **The gain** $\hlb{w}$ is a learned vector in $\mathbb{R}^{d}$, initialized to
  all ones.

Three properties follow directly:

- **Scale invariant.** Replacing $x$ with $cx$ for any $c \neq 0$ leaves $y$
  unchanged, up to the $\hla{\epsilon}$ term. Whatever magnitude the previous
  layer produced is discarded.
- **Not shift invariant.** Replacing $x$ with $x + c\mathbf{1}$ changes $y$.
  That's the difference from LayerNorm, and the next section covers what it
  costs.
- **Row-local.** Each of the $d$ channels in a row affects every other channel
  in that row, and nothing outside it. Rows are independent, which is why the
  kernel parallelizes one row per thread block.

## LayerNorm versus RMSNorm

You'll meet LayerNorm in older models and in papers, so it's worth knowing
exactly what RMSNorm leaves out. RMSNorm is LayerNorm with the mean subtraction
and the shift removed. LayerNorm first computes the mean and the variance of the
row:

$$
\mu = \frac{1}{d}\sum_{j=1}^{d} x_j,
\qquad
\sigma^2 = \frac{1}{d}\sum_{j=1}^{d} (x_j - \mu)^2
$$

Then it normalizes to zero mean and unit variance, and applies a gain and a
shift:

$$
y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \hla{\epsilon}}}\, \gamma_i + \beta_i
$$

Setting $\mu = 0$ and $\beta = 0$ gives RMSNorm. The variance and the mean of
squares are related exactly:

$$
\sigma^2 = \hlc{\operatorname{ms}(x)} - \mu^2
$$

So the two norms agree whenever the row's mean is already zero, and differ by
$\mu^2$ otherwise.

### What dropping re-centering saves

The saving is simplicity, not speed:

- **Parameters: negligible.** Dropping $\beta$ removes $d$ parameters per
  norm: $129 \times 5120 = 660{,}480$ parameters, which is 0.0025% of 26.9B.
- **Arithmetic: real but modest.** A naive LayerNorm reads the row twice, once
  for $\mu$ and once for $\sigma^2$, and on a bandwidth-bound kernel two passes
  cost twice one pass. A careful LayerNorm fuses them into one pass with two
  accumulators, $\sum x_j$ and $\sum x_j^2$. The saving then shrinks to one
  accumulator, one subtraction per element, and one addition per element.

Because the kernel is bandwidth bound, that saving is close to zero in
wall-clock terms. ==RMSNorm is simpler and no slower==, not much faster.

### What dropping re-centering costs

It costs an invariance. Under LayerNorm, a constant offset added to every
channel of a row is erased before the layer sees it. Under RMSNorm it survives,
scaled, so the residual stream is free to carry a shared constant component
that all channels see. Empirically, models use it.

The measured quality difference is nil. That's the result in the RMSNorm paper,
and it's why most large models since have dropped the mean subtraction. But
"nil" is an empirical finding about trained models, not a mathematical
equivalence.

### What epsilon is for

The constant $\hla{\epsilon}$ keeps the reciprocal square root finite when a
row is all zeros. That happens more often than you expect: padding positions,
masked tokens, and channels that a pruned or dead head never writes.

> [!WARNING] One zero row poisons the whole batch
> Without $\hla{\epsilon}$, $\hlc{\operatorname{ms}(x)} = 0$ gives $1/0$ and the
> row becomes NaN. The NaN propagates through the residual stream into every
> subsequent layer and out into the logits.

Where you put $\hla{\epsilon}$ matters. The reference adds it to the mean of
squares, inside the square root:

$$
\frac{1}{\sqrt{\hlc{\operatorname{ms}(x)} + \hla{\epsilon}}}
$$

It's not $1/(\sqrt{\operatorname{ms}(x)} + \epsilon)$, and it's not
$1/\sqrt{\operatorname{ms}(x) + d\epsilon}$. All three are finite at zero, and
only the first matches the checkpoint.

## Why the reduction runs in float32

This section answers the question the lab measures: why the sum inside RMSNorm
can't stay in bfloat16. The short answer is that a bfloat16 sum over 5120 terms
is ==biased low, not noisy==.

Start with a picture. Imagine a calculator that shows only three significant
digits. Ask it for $1000 + 1$ and it shows 1000: the 1 is real, but there's no
digit left to hold it. Add 1 a thousand times and the display still reads 1000,
though the true total is 2000. bfloat16 has about as many digits as that
calculator, between two and three decimal digits.

### When an addend disappears

Now make the picture exact. Take a running sum $S$ and the next addend $a$, both
positive, in bfloat16. The stored result of $S + a$ is the representable number
nearest the exact sum.

Suppose $2^{e} \le S < 2^{e+1}$, a range between two consecutive powers of 2
called a *binade*. The representable numbers in that range are $2^{e-7}$ apart,
because bfloat16 has 7 stored significand bits. Rounding to nearest returns $S$
itself whenever the addend is at most half that spacing:

$$
a \le \tfrac{1}{2}\,\mathrm{ulp}(S) = 2^{e-8}
$$

Rewrite that as a ratio. The addend is guaranteed to vanish once

$$
\frac{S}{a} \ge 2^{9} = 512
$$

and, when $S$ sits near the bottom of its binade, it can already vanish once

$$
\frac{S}{a} \ge 2^{8} = 256
$$

> [!KEY] Addends vanish at a ratio of 256
> Once the running sum is about $2^8$ times the next term, bfloat16 rounds the
> addition away entirely. It's not a small relative error: `S + a == S`
> returns true.

### What that does to a 5120-element reduction

Next, apply that threshold to the sum RMSNorm actually does. Run the reduction
the obvious way, left to right, over $d = 5120$ squared activations. Idealize
the row so that every $x_j^2$ is about the same value $s$. After $k$ successful
additions, the sum and its ratio to the next addend are:

$$
S_k \approx k\,s,
\qquad
\frac{S_k}{s} \approx k
$$

The ratio crosses $2^8 = 256$ at $k \approx 256$. From then on, each further
addend is at or below half an ulp of the running sum and contributes nothing.
The sum stalls:

$$
\boxed{S_{5120} \approx 256\,s \quad\text{instead of}\quad 5120\,s}
$$

> [!EXAMPLE] The stalled sum makes the output 4.5 times too large
> The computed mean of squares is a twentieth of the true value $s$:
>
> $$
> \widehat{\operatorname{ms}} \approx \frac{256\,s}{5120} = \frac{s}{20}
> $$
>
> So the normalizer is off by
> $\sqrt{\widehat{\operatorname{ms}}}/\sqrt{s} = 1/\sqrt{20} \approx 0.224$,
> and the output is about $1/0.224 \approx 4.5$ times too large.

Two things make this worse than ordinary rounding noise:

- **It's one-sided.** Dropping a positive addend can only make the sum too
  small. The error is a bias, so averaging over 5120 terms doesn't cancel it,
  and neither does averaging over the 129 norms in a forward pass. Every one
  of them is biased in the same direction.
- **It's length dependent.** The threshold is at 256 terms regardless of $d$.
  A reduction over 256 channels is barely affected and a reduction over 5120
  is crippled, so the same code is fine in one place and broken in another.

### Why the measured error is 1e-2 and not 4.5x

The stall assumes the sum runs strictly left to right, and GPU reductions don't.
A GPU splits the row into slices, sums each slice separately, then adds the
partial sums in pairs, then pairs of those, like a tournament bracket. No
running sum gets far ahead of what's added to it, so far fewer addends vanish.

The standard way to say this uses the *unit roundoff* $u$, the largest relative
error one rounding can introduce. It's $2^{-8}$ in bfloat16 and $2^{-24}$ in
float32. A left-to-right sum has a worst-case relative error of about $d\,u$,
and a pairwise tree cuts that to about $\log_2(d)\,u$. With
$\log_2 5120 \approx 12.3$, the bounds are the following:

| Scheme | Bound | Value |
|---|---|---|
| bfloat16, sequential | $5120 \times 2^{-8}$ | 20: no useful bound at all |
| bfloat16, pairwise | $12.3 \times 2^{-8}$ | about 0.048 |
| float32, pairwise | $12.3 \times 2^{-24}$ | about $7.3 \times 10^{-7}$ |

> [!INTUITION]
> The stall argument says bfloat16 accumulation is catastrophic. The tree
> reduction rescues most of it. What survives is a few percent of systematic
> low bias, the order of $10^{-2}$ relative error the lab measures on real
> activations. That's small enough that nothing crashes and large enough to
> change which token wins an argmax: the worst possible size for a bug.

The float32 bound, $7.3 \times 10^{-7}$, is more than an order of magnitude
below the $10^{-5}$ tolerance the lab checks against. That's why the reference
upcasts.

### The reference implementation

With the numerics settled, the code is short. The reference widens to float32,
reduces, scales, and rounds back once:

```python
def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    dtype = x.dtype
    x32 = x.float()
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    normed = x32 * torch.rsqrt(variance + eps)
    return (normed * weight.float()).to(dtype)
```

The following table walks through it line by line, with shapes for a prefill
batch of `(batch, seq, 5120)`:

| Line | Shape | What it does |
|---|---|---|
| `dtype = x.dtype` | — | Remember bfloat16 so the output can be cast back. |
| `x32 = x.float()` | `(batch, seq, 5120)` | Widen to 24 significand bits. Register-level only. |
| `x32.pow(2).mean(...)` | `(batch, seq, 1)` | The reduction, in float32. `keepdim` so it broadcasts. |
| `torch.rsqrt(variance + eps)` | `(batch, seq, 1)` | One reciprocal square root per row, not per element. |
| `normed * weight.float()` | `(batch, seq, 5120)` | The learned gain, also in float32. |
| `.to(dtype)` | `(batch, seq, 5120)` | Round once, at the end. |

The variable named `variance` isn't one: it's the mean of squares, with no mean
subtracted. The name follows the reference implementations everyone ports from,
so know that it lies.

==The upcast is free== in the sense that matters. The kernel is bandwidth
bound, and the bytes that cross between HBM and the chip are bfloat16 in both
directions. float32 exists only in registers, between the load and the store.
You pay register pressure, not bandwidth.

> [!WARNING] `x.var()` isn't the mean of squares
> PyTorch's `var` subtracts the mean and, by default, uses Bessel's
> correction, dividing by $d-1$. It computes $\sigma^2$, not
> $\hlc{\operatorname{ms}(x)}$. Substituting it gives you
> LayerNorm-without-the-centering-in-the-numerator, which matches nothing.

## The residual stream

You now know how RMSNorm works. This section is about what it normalizes. The
residual stream is the running sum that carries each token through the model.
Each layer is applied like this:

```python
residual = x
hidden = self.input_layernorm(x)
hidden = self.mixer(hidden, ...)
x = residual + hidden
return x + self.mlp(self.post_attention_layernorm(x))
```

The tensor `x` passes through the whole model untouched by anything except
addition. Each sublayer reads a normalized copy, computes something, and adds
the result back. That running sum is the residual stream.

### As a sum

Write the stream after $\ell$ sublayers as $x^{(\ell)}$. Each sublayer reads a
normalized copy and adds its contribution $\hld{\Delta^{(\ell)}}$:

$$
x^{(\ell)} = x^{(\ell-1)} + \hld{\Delta^{(\ell)}},
\qquad
\hld{\Delta^{(\ell)}} = F_\ell\!\left(N(x^{(\ell-1)})\right)
$$

Here $F_\ell$ is the mixer or the MLP, and $N$ is that sublayer's RMSNorm.
Unrolling from the embedding at $x^{(0)}$ gives the whole stack:

$$
\boxed{x^{(L)} = x^{(0)} + \sum_{\ell=1}^{L} \hld{\Delta^{(\ell)}}}
$$

With 64 layers and two sublayers each, $L = 128$.

> [!INTUITION]
> The final hidden state is the embedding plus 128 additive contributions. The
> model never overwrites, it only accumulates. Think of the stream as a shared
> bus, 5120 channels wide, that 128 writers append to and 128 readers read
> from.

### The stream grows

Because nothing ever subtracts, the stream tends to get larger as it goes up the
stack. If the contributions are roughly uncorrelated with each other and with
the embedding, and each has root mean square $\sigma$, then variance adds:

$$
\operatorname{rms}(x^{(L)}) \approx \sqrt{\operatorname{rms}(x^{(0)})^2 + L\sigma^2}
$$

With $\operatorname{rms}(x^{(0)}) \approx \sigma$ and $L = 128$, that's
$\sqrt{129}\,\sigma \approx 11.4\,\sigma$. The magnitude grows
==roughly as the square root of depth==.

This is a heuristic from the independence assumption, not a measurement. Real
contributions are correlated, but the direction is right, and it's observed in
practice. It has two consequences:

- Later layers see a larger input than earlier ones. That's why each sublayer
  normalizes its own input instead of trusting the stream's scale.
- The final state handed to `lm_head` has a depth-dependent magnitude, which is
  what the norm at the top of the stack fixes.

### Why the final norm exists

The final norm pins the scale of the *logits*, the scores, one per vocabulary
token, that become next-token probabilities. It runs once, before `lm_head`,
the matrix that turns the final hidden state into logits:

```python
x = self.norm(x)
if last_token_only:
    x = x[:, -1:, :]
return self.lm_head(x)
```

Each logit is an inner product between the final hidden state and one
vocabulary row. The logits go straight into a softmax, and softmax cares about
their absolute scale: multiply every logit by 2 and you've halved the sampling
temperature. So if the stream's magnitude drifts with depth, or between prompts,
the effective temperature drifts with it. The final RMSNorm gives `lm_head` a
unit-RMS input, up to its learned gain.

> [!WARNING] Skipping the final norm hides until you sample
> The model still produces sensible-looking text at greedy decoding, because
> argmax is invariant to a positive rescale. It then behaves strangely the
> moment you turn on temperature or top-p. The
> [sampling chapter](/c/11-sampling) is where that surfaces.

## Pre-norm versus post-norm

This section explains why the norm sits before each sublayer rather than after
it. It's a training argument, but it's the reason the layer code you're porting
looks the way it does. In short, pre-norm keeps a direct path for the gradient
through every layer, and that's why a 64-layer stack trains at all.

This model, like every model of its generation, uses *pre-norm*, which
normalizes the sublayer's input and leaves the stream itself alone:

$$
x^{(\ell)} = x^{(\ell-1)} + F_\ell\!\left(N(x^{(\ell-1)})\right)
$$

The original transformer used *post-norm*, which normalizes after the add, so
the stream itself passes through every norm:

$$
x^{(\ell)} = N\!\left(x^{(\ell-1)} + F_\ell(x^{(\ell-1)})\right)
$$

The difference shows up in the gradient from the top of the stack to the bottom.
That gradient is a product of per-layer *Jacobians*, the matrices that say how
much each output of a layer moves when each input moves. For pre-norm, write
$J_\ell$ for the Jacobian of $F_\ell \circ N$. The product over the stack
expands to an identity plus everything else:

$$
\boxed{
\begin{aligned}
\frac{\partial x^{(L)}}{\partial x^{(0)}}
&= \prod_{\ell=1}^{L} \left(\hla{I} + J_\ell\right) \\
&= \hla{I} + \sum_{\ell=1}^{L} J_\ell + \text{(products of two or more)}
\end{aligned}
}
$$

The leading term $\hla{I}$ doesn't depend on $L$. There's always a path from the
loss to every layer's input whose gradient is exactly 1. Depth can attenuate the
other terms, but it can't remove that one.

For post-norm, every factor also carries the normalization's Jacobian
$\hlb{N'_\ell}$, whose scale is roughly $1/\operatorname{rms}$ of its input:

$$
\frac{\partial x^{(L)}}{\partial x^{(0)}} = \prod_{\ell=1}^{L} \hlb{N'_\ell} \left(I + J_\ell\right)
$$

No identity term survives that product. If a typical factor has gain $c$, the
gradient reaching the bottom of a 64-layer stack scales like $c^{64}$.

> [!DEEPDIVE] Where the two products come from
> One pre-norm layer differentiates to $I + J_\ell$: the residual path gives
> the identity and the sublayer gives $J_\ell$. Expanding the product of $L$
> such factors, picking $I$ every time gives $I$, picking one $J_\ell$ gives
> the sum, and the rest are products of two or more Jacobians. One post-norm
> layer differentiates to $N'_\ell (I + J_\ell)$, because the norm wraps the
> whole sum, so every term in the expansion carries all $L$ normalization
> Jacobians, including the one that picks $I$ every time.

> [!EXAMPLE] A 5% per-layer error, 64 layers deep
> With a per-layer gain slightly below or above 1:
>
> $$
> 0.95^{64} \approx 0.038,
> \qquad
> 1.05^{64} \approx 22.7
> $$
>
> A 5% error in either direction changes the gradient at layer 0 by more than
> an order of magnitude.

That exponential sensitivity is why post-norm transformers need learning-rate
warmup and careful initialization, and why pre-norm ones mostly don't. The cost
of pre-norm is the growing stream, which the final norm handles. That's a good
trade.

## The per-head query and key norm

RMSNorm shows up in one more place, and this section covers what it does there.
Qwen3 normalizes each attention head's queries and keys separately, over the
*head dimension*, the 256-wide vector that one head works with. It does this
before the rotary position embedding of [chapter 5](/c/05-rotary-embeddings)
rotates them:

```python
q = q.view(batch, seq, self.num_heads, self.head_dim)
k = self.k_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim)
if self.q_norm is not None:
    q = self.q_norm(q)
if self.k_norm is not None:
    k = self.k_norm(k)
```

It's the same operation along a different axis. There are 24 query heads but
only 4 key heads, because grouped-query attention shares each key head across
six query heads, as [chapter 7](/c/07-grouped-query-attention) explains. The
following table compares the three places RMSNorm runs:

| Tensor | Shape | Rows normalized | Reduction length |
|---|---|---|---|
| `q` | `(batch, seq, 24, 256)` | 24 per token | 256 |
| `k` | `(batch, seq, 4, 256)` | 4 per token | 256 |
| residual stream | `(batch, seq, 5120)` | 1 per token | 5120 |

`HeadRMSNorm` in `engine/layers/rmsnorm.py` is the same function with
`hidden_size` replaced by `head_dim`.

### What it bounds

The per-head norm ==bounds every attention logit to 16==, whatever scale the
projections produce. Here an *attention logit* is the score between one query
and one key, before the softmax. The argument takes four steps:

1. An attention logit is $q \cdot k / \sqrt{256} = q \cdot k / 16$.
2. After per-head RMSNorm with a gain near 1, each vector has unit RMS over 256
   channels, so its Euclidean length is $\sqrt{256} = 16$.
3. By Cauchy-Schwarz, $|q \cdot k| \le 16 \times 16 = 256$.
4. Dividing by 16 bounds the logit:

$$
\left|\frac{q \cdot k}{16}\right| \le \frac{256}{16} = 16
$$

Softmax over logits bounded by 16 in magnitude can't overflow and can't
saturate to a one-hot distribution by accident. Without the norm, one head
whose projection drifted to twice the scale of the others produces logits four
times larger and attention that is effectively argmax. The per-head norm
equalizes heads.

**The reduction length is 256, and that's the threshold.** The stall
derivation puts the onset of total addend loss at about 256 terms, so a
256-element reduction sits right at the boundary. bfloat16 accumulation is far
less damaging here than over 5120 channels. The reference still upcasts,
because the same `rms_norm` function serves both.

> [!NOTE] Normalize first, then rotate
> With a gain of exactly 1, RMSNorm and RoPE commute: the rotation is
> orthogonal on the rotated channels and the identity elsewhere, so it
> preserves the root mean square of the head. With a learned per-channel gain
> they don't commute, because the gain applies in the unrotated basis. Follow
> the reference order.

### What it costs

The head norms add traffic, so it's worth knowing how much. Per token, per
full-attention layer, the q and k norms touch $(24 + 4) \times 256 = 7168$
elements. Each element costs 4 bytes, 2 read and 2 written:

$$
7168 \times 4 = 28{,}672\ \text{bytes}
$$

The totals per token compare as follows:

- Over the 16 full-attention layers, the head norms move 458,752 bytes.
- The 128 residual-stream norms move
  $128 \times 5120 \times 4 = 2{,}621{,}440$ bytes.

So the head norms add about 17% to the model's normalization traffic. That's
arithmetic, not a measurement.

## Where the time goes

This section answers the performance question: what limits how fast RMSNorm
runs? The answer is memory. RMSNorm is bandwidth bound: it does 1 FLOP per byte
against a ridge point of 161.

The count per element is the following:

- **Bytes:** it reads 2 and writes 2.
- **FLOPs:** about 4, namely a square, an accumulate, a scale, and the gain
  multiply.

The per-row reciprocal square root is amortized over 5120 channels and rounds
to nothing. The ratio of work to traffic, the *arithmetic intensity*, is:

$$
\frac{4\ \text{FLOP}}{4\ \text{bytes}} = 1\ \text{FLOP/byte}
$$

That's 1/161 of the way to compute bound. At the card's rated 1935 GB/s, one
FLOP per byte sustains 1.94 TFLOP/s, which is 0.6% of the A100's 312 TFLOP/s of
bfloat16 throughput.

> [!KEY] Only fewer bytes make it faster
> The tensor cores are idle. Cheaper arithmetic changes nothing; the only lever
> is the number of bytes the kernel moves.

To get a feel for the scale, the following example times the norms for one long
prompt.

> [!EXAMPLE] 128 norms over a 4096-token prefill
> One norm touches $4096 \times 5120 = 20{,}971{,}520$ elements, or
> $20{,}971{,}520 \times 4\ \text{bytes} = 83.9\ \text{MB}$. At the 1275 GB/s
> a device-to-device copy actually reaches on this card, the honest ceiling
> instead of the 1935 GB/s rating, one norm takes:
>
> $$
> \frac{83.9 \times 10^{6}}{1275 \times 10^{9}} = 65.8\ \mu\text{s}
> $$
>
> The totals are:
>
> - At the copy ceiling, 128 norms come to 8.4 ms.
> - The Triton kernel in [chapter 13](/c/13-fusion-in-triton) reaches
>   945 GB/s, 74% of the copy ceiling, which puts the same 128 norms at
>   11.4 ms.

Both totals are arithmetic from measured bandwidths, not timings of the whole
stack. They establish the scale: normalization is a visible slice of prefill,
and no amount of arithmetic cleverness touches it.

### Fusing with the residual add

If bytes are the only lever, the way to pull it is to stop writing
intermediates. Unfused, the residual add followed by the norm makes five trips
over the data:

1. Read the mixer's output.
2. Read the residual.
3. Write the sum.
4. Read the sum back.
5. Write the normalized result.

Fused, it's two reads and two writes: read both inputs, keep the sum in
registers, write the sum out for the next residual connection, and write the
normalized copy. Four trips instead of five predicts a saving of:

$$
1 - \frac{4}{5} = 20\%
$$

The measurement disagrees, and the reason is the cache. On this A100, the
fusion is 1.10x faster once the intermediate is too big for the cache, and
0.92x, which is slower, when it fits. The cache in question is the *L2*, the
40 MB on-chip cache that sits between the compute units and HBM. The
intermediate sum for 4096 tokens is slightly past it:

$$
4096 \times 5120 \times 2 = 41.9\ \text{MB}
$$

Below that size, ==the traffic the fusion eliminates never reached HBM== in the
first place, so there was nothing to save, and the fused kernel's extra register
pressure costs more than it gains. [Chapter 13](/c/13-fusion-in-triton) writes
the kernel and works through the measurement, and
`engine/kernels/rmsnorm_triton.py` has the finished version.

## What goes wrong

These are the mistakes the lab is most likely to provoke:

- **Reducing over the wrong axis.** `dim=0` instead of `dim=-1` normalizes
  across the batch instead of across channels. The shapes still broadcast,
  nothing raises, and the model emits fluent garbage. If a from-scratch
  implementation matches the reference on a square test tensor and fails on a
  rectangular one, this is why. Test at least one non-square shape.
- **Dropping `keepdim`.** The reduction returns `(batch, seq)` instead of
  `(batch, seq, 1)`. The subsequent multiply either raises a shape error or,
  worse, broadcasts against the wrong axis and silently produces a transposed
  result.
- **Using `x.var()`.** It subtracts the mean and divides by $d - 1$. RMSNorm
  wants neither.
- **Casting the weight but not the activations, or the reverse.** Mixing a
  float32 tensor with a bfloat16 one promotes silently in PyTorch, so nothing
  raises. Part of your reduction ran at 8 bits of precision after all.
- **Forgetting to cast back.** The layer returns float32, the next matmul runs
  at float32 instead of on the tensor cores, and prefill is several times slower
  with no error anywhere. The lab checks the output dtype for this reason.
- **Dropping `eps`.** It works on random test data and produces NaN the first
  time a padded or masked row is all zeros in production. The lab feeds you a
  zero row.

> [!RECAP]
> - RMSNorm divides each row by $\sqrt{\operatorname{ms}(x) + \epsilon}$ and
>   applies a gain. Dropping LayerNorm's centering buys simplicity, not speed.
> - A sequential bfloat16 sum stalls at about 256 terms. Tree reductions
>   shrink that to a few percent of low bias, which is still enough to flip an
>   argmax, so reduce in float32 and round once at the end.
> - The residual stream is the embedding plus 128 contributions, and its
>   magnitude grows roughly as the square root of depth.
> - Pre-norm keeps an identity term in the gradient; post-norm's gradient
>   scales like $c^{64}$. The final norm pins the scale for `lm_head`.
> - The per-head q and k norm bounds attention logits to 16.
> - At 1 FLOP per byte, RMSNorm is bandwidth bound, and fusion only pays once
>   the intermediate spills out of the 40 MB L2.

## Check your understanding

> [!QUESTION] The bfloat16 stall argument predicts the norm comes out 4.5 times too large. The lab measures an error around 1e-2 relative. Which is wrong?
> Neither. The stall argument assumes a strictly sequential sum, where the
> running total overtakes the addend after about 256 terms and never recovers.
> GPU reductions are tree-shaped, so each partial sum is over a short slice and
> the error bound falls from $d\,u$ to $\log_2(d)\,u$: from useless to a few
> percent. The residue is the systematic low bias you measure.

> [!QUESTION] Why does the same `rms_norm` function need float32 over 5120 channels but barely need it over 256?
> Because the threshold where addends start vanishing is a property of the
> format, not the reduction. It sits at a ratio of about $2^8$ between the
> running sum and the next term, which a sum of equal-sized terms reaches after
> about 256 of them. A 256-element reduction never gets far past it; a
> 5120-element one spends 95% of its length beyond it.

> [!QUESTION] Pre-norm makes the residual stream grow with depth. Why is that acceptable, and what would happen if you removed the final norm?
> It's acceptable because every sublayer normalizes its own input, so no layer
> ever sees the accumulated scale. The final norm protects the one consumer that
> doesn't normalize: `lm_head`. Remove it and the logits inherit the stream's
> magnitude, which changes the effective softmax temperature. Greedy decoding
> still works, because argmax ignores a positive rescale; anything
> temperature-dependent doesn't.

> [!QUESTION] RMSNorm has an arithmetic intensity of 1 FLOP per byte. Why does making the arithmetic cheaper not help?
> Because the ridge point is 161. At 1 FLOP per byte the kernel spends all its
> time waiting on HBM and the arithmetic units are 99.4% idle. Halving the
> FLOPs halves something that isn't the bottleneck. The only lever is bytes
> moved, which is what fusion attacks.

## Lab

> [!TRY]
> Implement three functions and run them on the GPU. Passing means `rms_norm`
> matches the reference to $10^{-5}$, keeps the input's dtype, stays finite on
> a zero row, and beats your low-precision version's error, and `bytes_moved`
> counts one read and one write.

The three functions are as follows:

- **`rms_norm(x, weight, eps)`** must match the reference to $10^{-5}$ on
  float32 inputs at shapes `(4, 5120)`, `(1, 5120)`, `(128, 5120)`, and
  `(7, 4097)`. That last one isn't a multiple of any convenient block size, on
  purpose. It must return the input's dtype when given bfloat16, and it must
  stay finite on an all-zero row.
- **`rms_norm_low_precision(x, weight, eps)`** is the same operation with the
  reduction left in the input dtype. Write it deliberately wrong, so the
  harness can compare: it checks that your float32 version has strictly lower
  maximum absolute error than your bfloat16 version, on bfloat16 activations
  scaled up by 8, and reports both as metrics.
- **`bytes_moved(x)`** returns what an ideal kernel must move: one read and one
  write, so `2 * x.numel() * x.element_size()`.

The harness then benchmarks `rms_norm` on a `(8192, 5120)` bfloat16 tensor and
reports achieved GB/s against the 1275 GB/s copy ceiling. Expect to land well
short of it. A PyTorch RMSNorm materializes every intermediate as its own
tensor, so it moves several times the ideal byte count;
[chapter 13](/c/13-fusion-in-triton) fuses them and closes most of the gap.

## Further reading

- [Root mean square layer normalization](https://arxiv.org/abs/1910.07467)
- [Layer normalization](https://arxiv.org/abs/1607.06450) — the original.
- [On layer normalization in the transformer architecture](https://arxiv.org/abs/2002.04745) — pre-norm versus post-norm, with the warmup analysis.
- [Deep residual learning for image recognition](https://arxiv.org/abs/1512.03385) — where the identity path came from.
