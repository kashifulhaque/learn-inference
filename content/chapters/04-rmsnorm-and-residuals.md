---
title: RMSNorm and the residual stream
slug: 04-rmsnorm-and-residuals
part: "Part 2 — A forward pass"
summary: The normalization every layer uses, the floating-point argument for accumulating in float32, and what pre-norm buys.
minutes: 70
gpu: true
objectives:
  - Write RMSNorm and LayerNorm as formulas and say exactly what dropping re-centering costs and saves.
  - Derive why a bfloat16 reduction over 5120 elements is systematically low, from the mantissa width.
  - Describe the residual stream as a sum of layer contributions, and give the gradient argument for pre-norm.
  - Explain the per-head query and key norm and what it bounds.
  - Show that RMSNorm is bandwidth bound by comparing its arithmetic intensity with the A100 ridge point.
lab: 04-rmsnorm
---

# RMSNorm and the residual stream

Every one of the 64 layers normalizes twice, once before the token mixer and once
before the MLP, and there is one more normalization at the top of the stack. That
is 129 invocations per forward pass. After the matrix multiplies, this is the most
executed operation in the model.

It is also the first place where numerics bite. RMSNorm is four arithmetic
operations per element and a sum, which sounds too simple to get wrong. The sum is
over 5120 terms, in a format with 8 bits of precision, and that is enough to break
it. This chapter derives why, rather than asserting it.

The second half of the chapter is about the thing RMSNorm is normalizing: the
residual stream. That tensor is the model's working memory and the reason a
64-layer network trains at all.

## Before you start

**Root mean square.** For a vector $x \in \mathbb{R}^{d}$,

$$
\operatorname{rms}(x) = \sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2}
$$

It is the standard deviation of $x$ if the mean of $x$ happens to be zero, and it
is not otherwise.

**Floating point, enough of it.** A floating-point number is a sign, an exponent,
and a fixed number of significand bits. bfloat16 spends 1 bit on the sign, 8 on
the exponent, and 7 stored bits on the significand — 8 bits of precision once you
count the implicit leading 1. float32 has 24. The spacing between representable
numbers near a value $S$ is called an *ulp* (unit in the last place), and it
scales with $S$: for $2^{e} \le S < 2^{e+1}$, one ulp is $2^{e-7}$ in bfloat16.
The [notation chapter](/c/00a-notation-and-prerequisites) has the formats in
full; that paragraph is all this chapter uses.

**The ridge point.** An A100 80GB delivers about 312 TFLOP/s in bfloat16 and
1935 GB/s of memory bandwidth, so it needs about 161 FLOPs per byte moved to keep
the tensor cores busy. Anything below that is memory bound. Chapter 10 develops
this properly; here it is a single number to compare against.

**PyTorch idioms.** `x.pow(2).mean(dim=-1, keepdim=True)` reduces over the last
axis and keeps it as size 1 so the result broadcasts back. `torch.rsqrt` is the
reciprocal square root, computed as one operation rather than a square root
followed by a division.

## The operation

RMSNorm scales each row of the input to unit root-mean-square, then applies a
learned per-channel gain. For a row $x \in \mathbb{R}^{d}$ with weight
$w \in \mathbb{R}^{d}$:

$$
y_i = \frac{x_i}{\sqrt{\operatorname{ms}(x) + \epsilon}}\, w_i,
\qquad
\operatorname{ms}(x) = \frac{1}{d}\sum_{j=1}^{d} x_j^2
$$

Here $d$ is 5120 for the residual-stream norms, $\epsilon$ is `rms_norm_eps`,
which is $10^{-6}$ in this config, and $w$ is a learned vector initialized to all
ones.

Three properties follow directly.

It is **scale invariant**: replacing $x$ with $cx$ for any $c \neq 0$ leaves $y$
unchanged, up to the $\epsilon$ term. Whatever magnitude the previous layer
produced is discarded.

It is **not shift invariant**: replacing $x$ with $x + c\mathbf{1}$ changes $y$.
This is the difference from LayerNorm, and the next section is about what it
costs.

It is **row-local**: each of the $d$ channels in a row affects every other channel
in that row, and nothing outside it. Rows are independent, which is why the kernel
parallelizes one row per thread block.

## LayerNorm versus RMSNorm

LayerNorm normalizes to zero mean and unit variance, then applies a gain and a
shift:

$$
\mu = \frac{1}{d}\sum_{j=1}^{d} x_j,
\qquad
\sigma^2 = \frac{1}{d}\sum_{j=1}^{d} (x_j - \mu)^2
$$

$$
y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}\, \gamma_i + \beta_i
$$

RMSNorm is what you get by setting $\mu = 0$ and $\beta = 0$ and keeping the rest.
The two are related exactly:

$$
\sigma^2 = \operatorname{ms}(x) - \mu^2
$$

so they agree whenever the row's mean is already zero, and they differ by
$\mu^2$ otherwise.

**What dropping re-centering saves.** Two things, and one of them is smaller than
it looks.

The parameter saving is negligible. Dropping $\beta$ removes $d$ parameters per
norm: $129 \times 5120 = 660{,}480$ parameters, which is 0.0025% of 26.9B. That is
not why anyone does it.

The arithmetic saving is real but modest. A naive LayerNorm reads the row twice,
once for $\mu$ and once for $\sigma^2$, and on a bandwidth-bound kernel two passes
cost twice one pass. A careful LayerNorm fuses them into a single pass with two
accumulators, $\sum x_j$ and $\sum x_j^2$, and then the saving shrinks to one
accumulator, one subtraction per element, and one addition per element. Since the
kernel is bandwidth bound, that saving is close to zero in wall-clock terms. The
honest summary is that RMSNorm is simpler and no slower, not that it is much
faster.

**What dropping re-centering costs.** An invariance. Under LayerNorm, a constant
offset added to every channel of a row is erased before the layer sees it. Under
RMSNorm it survives, scaled. The residual stream is therefore free to carry a
shared DC component that all channels see, and empirically models do use it. The
measured quality difference is nil — that is the result in the RMSNorm paper, and
it is why every recent model has dropped the mean subtraction — but "nil" is an
empirical finding about trained models, not a mathematical equivalence.

**What $\epsilon$ is for.** It keeps the reciprocal square root finite when a row
is all zeros, which happens more often than you would expect: padding positions,
masked tokens, and channels a pruned or dead head never writes. Without it,
$\operatorname{ms}(x) = 0$ gives $1/0$, the row becomes NaN, and the NaN
propagates through the residual stream into every subsequent layer and out into
the logits. One padded row poisons the whole batch.

Where you put $\epsilon$ matters. The reference adds it to the mean of squares,
inside the square root:

$$
\frac{1}{\sqrt{\operatorname{ms}(x) + \epsilon}}
$$

not $1/(\sqrt{\operatorname{ms}(x)} + \epsilon)$ and not
$1/\sqrt{\operatorname{ms}(x) + d\epsilon}$. All three are finite at zero and only
the first matches the checkpoint.

## Why the reduction runs in float32

Here is the argument the last version of this chapter asserted without showing.

### When an addend disappears

Take a running sum $S$ and the next addend $a$, both positive, in bfloat16. The
exact result is $S + a$. The stored result is the nearest representable number to
$S + a$.

Suppose $2^{e} \le S < 2^{e+1}$. The representable numbers in that range are
spaced $2^{e-7}$ apart, because bfloat16 has 7 stored significand bits. Rounding
to nearest returns $S$ itself whenever

$$
a \le \tfrac{1}{2}\,\mathrm{ulp}(S) = 2^{e-8}
$$

Rewriting in terms of the ratio $S/a$: the addend is guaranteed to vanish once

$$
\frac{S}{a} \ge 2^{9} = 512
$$

and it can already vanish, when $S$ sits near the bottom of its binade, as soon as

$$
\frac{S}{a} \ge 2^{8} = 256
$$

Take $2^8$ as the point where the loss starts. The key word is *entirely*: this is
not a small relative error on the addition, it is the addition doing nothing at
all. `S + a == S` returns true.

### What that does to a 5120-element reduction

Now run the reduction the obvious way, left to right, over $d = 5120$ squared
activations. Idealize the row so that every $x_j^2$ is about the same value $s$.
After $k$ successful additions,

$$
S_k \approx k\,s
$$

and the next addend is still about $s$, so the ratio is

$$
\frac{S_k}{s} \approx k
$$

which crosses $2^8 = 256$ at $k \approx 256$. From that point on, each further
addend is at or below half an ulp of the running sum and contributes nothing. The
sum stalls:

$$
S_{5120} \approx 256\,s \quad\text{instead of}\quad 5120\,s
$$

The computed mean of squares is then

$$
\widehat{\operatorname{ms}} \approx \frac{256\,s}{5120} = \frac{s}{20}
$$

against a true value of $s$. The normalizer is off by

$$
\frac{\sqrt{\widehat{\operatorname{ms}}}}{\sqrt{s}} = \frac{1}{\sqrt{20}} \approx 0.224
$$

so the output is about $1/0.224 \approx 4.5$ times too large.

Two things make this worse than ordinary rounding noise.

It is **one-sided**. Dropping a positive addend can only make the sum too small,
never too large. The error is a bias, not noise. Averaging over 5120 terms does
not cancel it, and neither does averaging over the 129 norms in a forward pass —
every one of them is biased the same direction.

It is **length dependent**. The threshold is at 256 terms regardless of $d$, so a
reduction over 256 channels is barely affected and a reduction over 5120 is
crippled. The same code is fine in one place and broken in another.

### Why the measured error is 1e-2 and not 4.5x

The derivation above assumes strictly sequential accumulation. Real GPU reductions
are not sequential: a thread block sums its own slice, then the partial sums are
combined pairwise in a tree. That changes the error bound completely.

Sequential summation of $d$ terms has a worst-case relative error bound of about
$d\,u$, where $u$ is the unit roundoff. Pairwise (tree) summation has a bound of
about $\log_2(d)\,u$. For this reduction, with $u = 2^{-8}$ in bfloat16 and
$\log_2 5120 \approx 12.3$:

| Scheme | Bound | Value |
|---|---|---|
| bfloat16, sequential | $5120 \times 2^{-8}$ | 20 — no useful bound at all |
| bfloat16, pairwise | $12.3 \times 2^{-8}$ | about 0.048 |
| float32, pairwise | $12.3 \times 2^{-24}$ | about $7.3 \times 10^{-7}$ |

That is the reconciliation. The stall argument says bfloat16 accumulation is
catastrophic; the tree reduction rescues most of it; what survives is a few
percent of systematic low bias, which is the order of $10^{-2}$ relative error the
lab measures on real activations. It is small enough that nothing crashes and
large enough to change which token wins an argmax, which is the worst possible
size for a bug.

The float32 bound, $7.3 \times 10^{-7}$, is more than an order of magnitude below
the $10^{-5}$ tolerance the lab checks against. That is why the reference upcasts.

### The reference implementation

```python
def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    dtype = x.dtype
    x32 = x.float()
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    normed = x32 * torch.rsqrt(variance + eps)
    return (normed * weight.float()).to(dtype)
```

Line by line, with shapes for a prefill batch of `(batch, seq, 5120)`:

| Line | Shape | What it does |
|---|---|---|
| `dtype = x.dtype` | — | Remember bfloat16 so the output can be cast back. |
| `x32 = x.float()` | `(batch, seq, 5120)` | Widen to 24 significand bits. Register-level only. |
| `x32.pow(2).mean(...)` | `(batch, seq, 1)` | The reduction, in float32. `keepdim` so it broadcasts. |
| `torch.rsqrt(variance + eps)` | `(batch, seq, 1)` | One reciprocal square root per row, not per element. |
| `normed * weight.float()` | `(batch, seq, 5120)` | The learned gain, also in float32. |
| `.to(dtype)` | `(batch, seq, 5120)` | Round once, at the end. |

The variable is named `variance` and is not one: it is the mean of squares, with
no mean subtracted. That naming follows the reference implementations everyone
ports from, so it is worth knowing that the name lies.

The upcast is free in the sense that matters. The kernel is bandwidth bound, and
the bytes that cross the bus are bfloat16 in both directions. float32 exists only
in registers, between the load and the store. You pay register pressure, not
bandwidth.

One trap: `x.var(dim=-1)` is not `x.pow(2).mean(dim=-1)`. PyTorch's `var`
subtracts the mean, and by default uses Bessel's correction, dividing by $d-1$. It
computes $\sigma^2$, not $\operatorname{ms}(x)$. Substituting it gives you
LayerNorm-without-the-centering-in-the-numerator, which matches nothing.

## The residual stream

Each layer is applied like this:

```python
residual = x
hidden = self.input_layernorm(x)
hidden = self.mixer(hidden, ...)
x = residual + hidden
return x + self.mlp(self.post_attention_layernorm(x))
```

The tensor `x` passes through the whole model untouched by anything except
addition. Layers read a normalized copy, compute something, and add the result
back. That running sum is the *residual stream*.

### As a sum

Write the stream after $\ell$ sublayers as $x^{(\ell)}$, and each sublayer's
contribution as $\Delta^{(\ell)}$. Then

$$
x^{(\ell)} = x^{(\ell-1)} + \Delta^{(\ell)},
\qquad
\Delta^{(\ell)} = F_\ell\!\left(N(x^{(\ell-1)})\right)
$$

where $F_\ell$ is the mixer or the MLP and $N$ is that sublayer's RMSNorm.
Unrolling from the embedding at $x^{(0)}$:

$$
x^{(L)} = x^{(0)} + \sum_{\ell=1}^{L} \Delta^{(\ell)}
$$

With 64 layers and two sublayers each, $L = 128$. The final hidden state is the
embedding plus 128 additive contributions, and the model never overwrites — it
only accumulates. This is why the stream is usefully thought of as a shared bus
that 128 writers append to and 128 readers read from, 5120 channels wide.

### The stream grows

If the contributions are roughly uncorrelated with each other and with the
embedding, and each has root-mean-square $\sigma$, then variance adds:

$$
\operatorname{rms}(x^{(L)}) \approx \sqrt{\operatorname{rms}(x^{(0)})^2 + L\sigma^2}
$$

With $\operatorname{rms}(x^{(0)}) \approx \sigma$ and $L = 128$, that is
$\sqrt{129}\,\sigma \approx 11.4\,\sigma$. The magnitude grows roughly as the
square root of depth. This is a heuristic from the independence assumption, not a
measurement, and real contributions are correlated, but the direction is right and
it is observed in practice.

Two consequences. Later layers see a larger input than earlier ones, which is
exactly why each sublayer normalizes its own input rather than trusting the
stream's scale. And the final state handed to `lm_head` has a depth-dependent
magnitude, which brings us to the norm at the top.

### Why the final norm exists

```python
x = self.norm(x)
if last_token_only:
    x = x[:, -1:, :]
return self.lm_head(x)
```

`lm_head` produces logits as inner products between the final hidden state and
each vocabulary row. Those logits go straight into a softmax, whose behaviour
depends on their absolute scale: multiply every logit by 2 and you have halved the
sampling temperature. If the stream's magnitude drifts with depth, or between
prompts, the effective temperature drifts with it. The final RMSNorm pins the
scale so `lm_head` always sees a unit-RMS input, up to its learned gain.

Skipping it is a common porting bug. The model still produces sensible-looking
text at greedy decoding, because argmax is invariant to a positive rescale, and
then behaves strangely the moment you turn on temperature or top-p. Chapter 11 is
where that surfaces.

### Pre-norm versus post-norm

Two placements of the normalization are possible. This model, like every recent
one, uses **pre-norm**:

$$
x^{(\ell)} = x^{(\ell-1)} + F_\ell\!\left(N(x^{(\ell-1)})\right)
$$

The original transformer used **post-norm**:

$$
x^{(\ell)} = N\!\left(x^{(\ell-1)} + F_\ell(x^{(\ell-1)})\right)
$$

The difference is a training-time argument, and it is worth writing out because it
explains why a 64-layer stack exists at all.

Differentiate the pre-norm form. Let $J_\ell$ be the Jacobian of
$F_\ell \circ N$ at $x^{(\ell-1)}$:

$$
\frac{\partial x^{(\ell)}}{\partial x^{(\ell-1)}} = I + J_\ell
$$

Chaining over the whole stack:

$$
\frac{\partial x^{(L)}}{\partial x^{(0)}} = \prod_{\ell=1}^{L} \left(I + J_\ell\right)
= I + \sum_{\ell=1}^{L} J_\ell + \text{(products of two or more)}
$$

The leading term is the identity, and it does not depend on $L$. There is always a
path from the loss to every layer's input whose gradient is exactly 1. Depth can
attenuate the other terms but it cannot remove that one.

Now the post-norm form. Writing $N'_\ell$ for the Jacobian of the normalization:

$$
\frac{\partial x^{(\ell)}}{\partial x^{(\ell-1)}} = N'_\ell \left(I + J_\ell\right)
$$

$$
\frac{\partial x^{(L)}}{\partial x^{(0)}} = \prod_{\ell=1}^{L} N'_\ell \left(I + J_\ell\right)
$$

Every factor now carries $N'_\ell$, whose scale is roughly $1/\operatorname{rms}$
of its input. No identity term survives the product. If a typical factor has gain
$c$, the gradient reaching the bottom of a 64-layer stack scales like $c^{64}$:

$$
0.95^{64} \approx 0.038,
\qquad
1.05^{64} \approx 22.7
$$

A 5% per-layer error in either direction changes the gradient at layer 0 by more
than an order of magnitude. That exponential sensitivity is why post-norm
transformers need learning-rate warmup and careful initialization, and why
pre-norm ones mostly do not.

The cost of pre-norm is the growing stream described above, which the final norm
handles. That is a good trade.

## The per-head query and key norm

Qwen3 applies RMSNorm in a second place: per attention head, over the 256-wide
head dimension, to queries and keys before the rotation.

```python
q = q.view(batch, seq, self.num_heads, self.head_dim)
k = self.k_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim)
if self.q_norm is not None:
    q = self.q_norm(q)
if self.k_norm is not None:
    k = self.k_norm(k)
```

| Tensor | Shape | Rows normalized | Reduction length |
|---|---|---|---|
| `q` | `(batch, seq, 24, 256)` | 24 per token | 256 |
| `k` | `(batch, seq, 4, 256)` | 4 per token | 256 |
| residual stream | `(batch, seq, 5120)` | 1 per token | 5120 |

Same operation, different axis. `HeadRMSNorm` in `engine/layers/rmsnorm.py` is the
same function with `hidden_size` replaced by `head_dim`.

**What it bounds.** An attention logit is $q \cdot k / \sqrt{256} = q \cdot k / 16$.
After per-head RMSNorm with a gain near 1, each vector has unit RMS over 256
channels, so its Euclidean norm is $\sqrt{256} = 16$. By Cauchy-Schwarz,
$|q \cdot k| \le 16 \times 16 = 256$, and the scaled logit is bounded by

$$
\frac{256}{16} = 16
$$

independent of how large the projections' outputs happened to be. Softmax over
logits bounded by 16 in magnitude cannot overflow and cannot saturate to a
one-hot distribution by accident. Without the norm, one head whose projection
drifted to twice the scale of the others produces logits four times larger and
attention that is effectively argmax. The per-head norm equalizes heads.

**The reduction length is 256, and that is the threshold.** The stall derivation
above puts the onset of total addend loss at about 256 terms. A 256-element
reduction sits right at the boundary, so bfloat16 accumulation is far less
damaging here than over 5120 channels. The reference still upcasts, because the
same `rms_norm` function serves both and there is no reason not to.

**Order matters.** The engine normalizes and then rotates. With a gain of exactly
1, RMSNorm and RoPE commute — the rotation is orthogonal on the rotated channels
and the identity elsewhere, so it preserves the root mean square of the head. With
a learned per-channel gain they do not commute, because the gain is applied in the
unrotated basis. Follow the reference order.

**What it costs.** Per token, per full-attention layer, the q and k norms touch
$(24 + 4) \times 256 = 7168$ elements, 4 bytes each read and written:

$$
7168 \times 4 = 28{,}672\ \text{bytes}
$$

Over the 16 full-attention layers that is 458,752 bytes per token. The 128
residual-stream norms move $128 \times 5120 \times 4 = 2{,}621{,}440$ bytes per
token. So the head norms add about 17% to the model's normalization traffic.
Arithmetic, not a measurement.

## Where the time goes

RMSNorm reads 2 bytes per element and writes 2, and does about 4 floating-point
operations per element: a square, an accumulate, a scale, and the gain multiply.
The per-row reciprocal square root is amortized over 5120 channels and rounds to
nothing. So the arithmetic intensity is

$$
\frac{4\ \text{FLOP}}{4\ \text{bytes}} = 1\ \text{FLOP/byte}
$$

against an A100 ridge point of about 161. It is 1/161 of the way to compute bound.
Put the other way: at the card's rated 1935 GB/s, one FLOP per byte sustains
1.94 TFLOP/s, which is 0.6% of the A100's 312 TFLOP/s of bfloat16 throughput. The
tensor cores are idle. The only way to make this faster is to move fewer bytes.

Work it through for a prefill of 4096 tokens. One norm touches

$$
4096 \times 5120 = 20{,}971{,}520 \ \text{elements}
$$

$$
20{,}971{,}520 \times 4\ \text{bytes} = 83.9\ \text{MB}
$$

At the 1275 GB/s a device-to-device copy actually reaches on this card — the
honest ceiling, not the 1935 GB/s rating — that is

$$
\frac{83.9 \times 10^{6}}{1275 \times 10^{9}} = 65.8\ \mu\text{s}
$$

and 128 of them come to 8.4 ms. The Triton kernel in chapter 13 reaches 945 GB/s,
74% of the copy ceiling, which puts the same 128 norms at 11.4 ms. Both of these
are arithmetic from measured bandwidths, not timings of the whole stack, but they
establish the scale: normalization is a visible slice of prefill, and no amount of
arithmetic cleverness touches it.

### Fusing with the residual add

The way to move fewer bytes is to stop writing intermediates. Unfused, the
residual add followed by the norm makes five trips over the data:

1. read the mixer's output,
2. read the residual,
3. write the sum,
4. read the sum back,
5. write the normalized result.

Fused, it is two reads and two writes: read both inputs, keep the sum in
registers, write the sum out for the next residual connection, and write the
normalized copy. Four trips instead of five, a predicted saving of

$$
1 - \frac{4}{5} = 20\%
$$

The measurement disagrees, in an interesting way. On this A100 the fusion is
1.10x faster once the intermediate is too big for cache, and 0.92x — slower — when
it fits. The A100's L2 is 40 MB. The intermediate sum for 4096 tokens is

$$
4096 \times 5120 \times 2 = 41.9\ \text{MB}
$$

which is just past it. Below that size the traffic the fusion eliminates never
reached HBM in the first place, so there was nothing to save, and the fused
kernel's extra register pressure costs more than it gains. Chapter 13 writes the
kernel and works through the measurement; `engine/kernels/rmsnorm_triton.py` has
the finished version.

## What goes wrong

**Reducing over the wrong axis.** `dim=0` instead of `dim=-1` normalizes across
the batch rather than across channels. The shapes still broadcast, nothing raises,
and the model emits fluent garbage. If a from-scratch implementation matches the
reference on a square test tensor and fails on a rectangular one, this is why —
always test at least one non-square shape.

**Dropping `keepdim`.** The reduction returns `(batch, seq)` instead of
`(batch, seq, 1)`, and the subsequent multiply either raises a shape error or,
worse, broadcasts against the wrong axis and silently produces a transposed
result.

**Using `x.var()`.** It subtracts the mean and divides by $d - 1$. Neither is what
RMSNorm wants.

**Casting the weight but not the activations, or the reverse.** Mixing a float32
tensor with a bfloat16 one promotes silently in PyTorch, so this does not raise;
it just means part of your reduction ran at 8 bits of precision after all.

**Forgetting to cast back.** The layer returns float32, the next matmul runs at
float32 instead of on the tensor cores, and prefill is several times slower with
no error anywhere. The lab checks the output dtype for exactly this reason.

**Dropping `eps`.** Works on random test data, produces NaN the first time a
padded or masked row is all zeros in production. The lab feeds you a zero row.

## Check your understanding

**The bfloat16 stall argument predicts the norm comes out 4.5 times too large.
The lab measures an error around 1e-2 relative. Which is wrong?**

Neither. The stall argument assumes a strictly sequential sum, where the running
total overtakes the addend after about 256 terms and never recovers. GPU
reductions are tree-shaped, so each partial sum is over a short slice and the
error bound falls from $d\,u$ to $\log_2(d)\,u$ — from useless to a few percent.
The residue is the systematic low bias you measure.

**Why does the same `rms_norm` function need float32 over 5120 channels but
barely need it over 256?**

Because the threshold where addends start vanishing is a property of the format,
not the reduction: it sits at a ratio of about $2^8$ between the running sum and
the next term, which a sum of equal-sized terms reaches after about 256 of them. A
256-element reduction never gets far past it; a 5120-element one spends 95% of its
length beyond it.

**Pre-norm makes the residual stream grow with depth. Why is that acceptable, and
what would happen if you removed the final norm?**

It is acceptable because every sublayer normalizes its own input, so no layer ever
sees the accumulated scale. The final norm is what protects the one consumer that
does not normalize: `lm_head`. Remove it and the logits inherit the stream's
magnitude, which changes the effective softmax temperature. Greedy decoding still
works, because argmax ignores a positive rescale; anything temperature-dependent
does not.

**RMSNorm has an arithmetic intensity of 1 FLOP per byte. Why does making the
arithmetic cheaper not help?**

Because the ridge point is 161. At 1 FLOP per byte the kernel spends all its time
waiting on HBM and the arithmetic units are 99.4% idle. Halving the FLOPs halves
something that is not the bottleneck. The only lever is bytes moved, which is what
fusion attacks.

## Lab

Implement three functions and run them on the GPU.

`rms_norm(x, weight, eps)` must match the reference to $10^{-5}$ at shapes
`(4, 5120)`, `(1, 5120)`, `(128, 5120)`, and `(7, 4097)` — that last one is not a
multiple of any convenient block size, on purpose. It must return the input's
dtype when given bfloat16, and it must stay finite on an all-zero row.

`rms_norm_low_precision(x, weight, eps)` is the same operation with the reduction
left in the input dtype. Write it deliberately wrong, so the harness can compare:
it checks that your float32 version has strictly lower maximum absolute error than
your bfloat16 version, on bfloat16 activations scaled up by 8, and reports both as
metrics.

`bytes_moved(x)` returns what an ideal kernel must move: one read and one write,
so `2 * x.numel() * x.element_size()`.

The harness then benchmarks `rms_norm` on a `(8192, 5120)` bfloat16 tensor and
reports achieved GB/s against the 1275 GB/s copy ceiling. Expect to land well
short of it. A PyTorch RMSNorm materializes every intermediate as its own tensor,
so it moves several times the ideal byte count; chapter 13 fuses them and closes
most of the gap.

## Further reading

- [Root mean square layer normalization](https://arxiv.org/abs/1910.07467)
- [Layer normalization](https://arxiv.org/abs/1607.06450) — the original.
- [On layer normalization in the transformer architecture](https://arxiv.org/abs/2002.04745) — pre-norm versus post-norm, with the warmup analysis.
- [Deep residual learning for image recognition](https://arxiv.org/abs/1512.03385) — where the identity path came from.
