---
title: Quantization on Ampere
slug: 18-quantization
part: "Part 6 — Scaling"
summary: What the bits of int8, bfloat16, and FP8 hold, how affine and symmetric quantization follow from them, why grouping fixes outliers, and why FP8 isn't available to you on an A100.
minutes: 115
gpu: true
objectives:
  - Say what each bit of bfloat16, FP8, int8, and int4 holds, and how a scale turns an integer into a real number.
  - Derive affine and symmetric quantization and bound the reconstruction error by the step size.
  - Explain why per-tensor scaling collapses under outliers, and derive per-channel and group-wise scaling as the fix.
  - Compute the metadata cost of a group size and the memory saved at model scale.
  - Explain why weight-only quantization helps decode and not prefill, with the byte arithmetic.
  - Describe what Ampere's int8 tensor cores compute, and why Ampere has no FP8.
  - Quantize a KV cache and say why keys tolerate it worse than values.
lab: 18-int8-quant
---

# Quantization on Ampere

> [!TLDR]
> - Quantization stores each weight as a small integer plus a scale shared by
>   many weights, so int8 weights take about half the bytes of bfloat16.
>   Rounding to the nearest step is off by at most half a step, so every design
>   choice is a way to make the step smaller.
> - One unusually large weight stretches the step for everything that shares its
>   scale. Giving each group of 128 weights its own scale contains the damage for
>   1.56% extra storage.
> - Decode spends its time reading weights, so halving the bytes makes it 1.97
>   times faster. Prefill spends its time computing, so it gains nothing.
> - The A100's int8 matrix units need one scale per tensor or per output row,
>   and the A100 has no FP8 arithmetic at all.
> - In a quantized KV cache, the softmax magnifies errors in keys while errors
>   in values average out, so keys need more precision.

Once the prompt is processed, each new token comes from a *decode* step that
reads all 53.8 GB of weights. At the A100's measured copy bandwidth of 1275 GB/s, that's a
floor of 42.2 ms per token, or 23.7 tokens per second, before a single FLOP.
Halve the bytes and you halve the floor.

That makes quantization, storing the weights in fewer bits, the most direct
throughput improvement available to this engine. It's also the easiest way to
wreck the model without noticing.

This chapter does the numerics properly. You start with what the bits of each
format hold. Quantization is a rounding scheme, and a rounding scheme has an
error bound you can derive rather than measure. Once you have the bound, every
later choice follows from it: symmetric against affine, per-tensor against
per-group, and weights only against weights and activations.

## Before you start

**The roofline, from [the roofline chapter](/c/10-roofline).** An operation's
*arithmetic intensity* is the FLOPs it does per byte it moves. The *ridge point*,
161 FLOPs per byte on the A100, separates *memory-bound* work below it from
*compute-bound* work above it. Decode runs at about 1 FLOP per byte, so it's
memory bound. *Prefill*, which processes the whole prompt at once, is compute
bound. Quantization changes the byte count, so it moves decode along the
memory-bound roof. It doesn't move prefill, which is already on the compute
roof.

**Norms.** The relative error used throughout is the Frobenius norm of the
difference over the Frobenius norm of the original. The Frobenius norm
$\lVert \cdot \rVert_F$ is the square root of the sum of every squared entry:

$$
\varepsilon = \frac{\lVert \hat{W} - W \rVert_F}{\lVert W \rVert_F}
$$

**Rounding as noise.** Round-to-nearest with step $s$ leaves a residual in
$[-s/2, s/2]$. Treating that residual as a uniform random variable is the
standard model, and it's accurate whenever $s$ is small compared with the
spread of the data.

## What the bits hold

Before you can reason about quantization error, you need a picture of what an
8-bit number can hold. This section compares the formats the chapter uses, bit
by bit. For more depth on floating point, see
[the notation chapter](/c/00a-notation-and-prerequisites).

A floating-point number splits its bits into three fields:

- A *sign* bit.
- An *exponent*, which picks a power of two.
- A *mantissa*, which picks one of evenly spaced points between that power of
  two and the next.

The span from one power of two to the next, such as 1 to 2 or 4 to 8, is an
*octave*. An integer format has no exponent: all its bits pick one of evenly
spaced integers. The following table shows what each format holds:

| Format | Bits: sign, exponent, mantissa | What it holds | Spacing of levels |
|---|---|---|---|
| float32 | 1, 8, 23 | About 7 significant digits | $2^{23}$ steps per octave |
| bfloat16 | 1, 8, 7 | float32's range, about 2 to 3 significant digits | 128 steps per octave |
| FP8 `e4m3` | 1, 4, 3 | About 1 significant digit | 8 steps per octave |
| int8 | 8 integer bits | The integers $-128$ to $127$ | Exactly 1, everywhere |
| int4 | 4 integer bits | The integers $-8$ to $7$ | Exactly 1, everywhere |

Each float format also has an implicit leading 1 that isn't stored, so
bfloat16 carries 8 significant bits and `e4m3` carries 4.

The floats spend the same number of levels on every octave, so the gap between
neighbors grows with the value. The integers space their levels evenly across
the whole range. So ==a floating-point format has roughly constant relative
precision, while a fixed-point format has constant absolute precision==. That
difference is what the outlier discussion later in this chapter is about.

An int8 on its own holds only an integer. To make it stand for a real weight,
you pair it with a *scale* $\hla{s}$, a real number stored separately that says
how much one integer step is worth. The stored integer $q$ then means
$\hat{w} = \hla{s}\,q$. Many weights share one scale, which is where the saving
comes from: 1 byte per weight, plus a small share of one scale.

> [!EXAMPLE] Four weights in int8
> Take the block $w = (0.834, -0.301, 0.027, -1.27)$. Choose the scale so the
> largest magnitude lands on the last integer level: $\hla{s} = 1.27/127 =
> 0.01$. Divide each weight by $\hla{s}$, round, and multiply back:
>
> | Weight $w$ | $w/\hla{s}$ | Stored $q$ | Dequantized $\hla{s}\,q$ | Error |
> |---|---|---|---|---|
> | 0.834 | 83.4 | 83 | 0.83 | 0.004 |
> | $-0.301$ | $-30.1$ | $-30$ | $-0.30$ | 0.001 |
> | 0.027 | 2.7 | 3 | 0.03 | 0.003 |
> | $-1.27$ | $-127$ | $-127$ | $-1.27$ | 0 |
>
> Every error is at most 0.005, which is half a step. Later sections derive
> that bound and ask what makes the step large.

Next, you make this recipe precise, starting with the general map from real
numbers to integers.

## Affine quantization, derived

An *affine* map sends a range of real numbers onto a range of integers and
back, with a scale and an offset. You have real values $w$ in a range
$[\beta_{\min}, \beta_{\max}]$, and you want to store them as integers in
$[q_{\min}, q_{\max}]$. For int8, that's $[-128, 127]$.

Start from the inverse map, *dequantization*, which turns an integer back into a
real number:

$$
\hat{w} = \hla{s}\,(q - \hlb{z})
$$

Here the *scale* $\hla{s} > 0$ is the real-valued width of one integer step, and
the *zero point* $\hlb{z}$ is the integer that dequantizes to exactly zero.

Require the two endpoints of the real range to land on the two endpoints of the
integer range. Solving those two equations gives the scale and the zero point:

$$
\boxed{
\hla{s} = \frac{\beta_{\max} - \beta_{\min}}{q_{\max} - q_{\min}},
\qquad
\hlb{z} = \operatorname{round}\!\left(q_{\min} - \frac{\beta_{\min}}{\hla{s}}\right)
}
$$

In words, the scale is the real range divided by the number of integer steps.
You round $\hlb{z}$ so that $w = 0$ maps to an exact integer and dequantizes
back to exactly zero.

> [!DEEPDIVE] Solving for the scale and zero point
> Require $\hat{w} = \beta_{\min}$ at $q = q_{\min}$ and $\hat{w} = \beta_{\max}$
> at $q = q_{\max}$. Subtracting the two equations eliminates $\hlb{z}$:
>
> $$
> \beta_{\max} - \beta_{\min} = \hla{s}\,(q_{\max} - q_{\min})
> \quad\Longrightarrow\quad
> \hla{s} = \frac{\beta_{\max} - \beta_{\min}}{q_{\max} - q_{\min}}
> $$
>
> Substituting back into the first equation gives the zero point before
> rounding, $\hlb{z} = q_{\min} - \beta_{\min}/\hla{s}$.
>
> To check the endpoints, substitute $\hlb{z}$ into the quantization map. At
> $w = \beta_{\min}$, the $\beta_{\min}/\hla{s}$ terms cancel and leave
> $q_{\min}$. At $w = \beta_{\max}$, you get
> $(\beta_{\max} - \beta_{\min})/\hla{s} + q_{\min} = q_{\max}$, using the
> definition of $\hla{s}$.

Quantization is the inverse of the dequantization map, rounded and clamped:

$$
q = \operatorname{clamp}\!\left(
\operatorname{round}\!\left(\frac{w}{\hla{s}}\right) + \hlb{z},\;
q_{\min},\; q_{\max}\right)
$$

> [!WARNING] Without the clamp, values wrap around
> You calibrate the range on one sample and apply it to another, so some values
> fall outside $[\beta_{\min}, \beta_{\max}]$. Without the clamp, those values
> wrap around in int8 and a large positive number becomes a large negative one.
> The failure is silent and catastrophic.

## Symmetric quantization

Weights sit close to symmetric around zero, so the zero point buys little. It
also costs an add per element on the dequantization path, so weights drop it.
Set $\beta_{\max} = -\beta_{\min} = \hld{\beta}$, where $\hld{\beta} = \max|w|$
is the largest magnitude. Then $\hlb{z} = 0$ falls out, and dequantization is
one multiply. This is the recipe the four-weight example used.

To keep the mapping symmetric, you also give up one integer level: use
$[-127, 127]$ rather than $[-128, 127]$, because $-128$ has no positive partner.
The three pieces are the following:

$$
\hla{s} = \frac{\hld{\beta}}{127},
\qquad
q = \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{w}{\hla{s}}\right),
\; -128,\; 127\right),
\qquad
\hat{w} = \hla{s}\,q
$$

The clamp keeps $-128$ as its lower bound, but a value inside the calibrated
range never reaches it, because $|w| / \hla{s} \le 127$.

The lab asks for this quantizer, one scale per output row:

```python
scale = weight.abs().amax(dim=-1, keepdim=True) / 127.0   # (out, 1)
quantized = (weight / scale).round().clamp(-128, 127).to(torch.int8)
dequantized = quantized.to(torch.bfloat16) * scale
```

Round before you clamp. Rounding first keeps every in-range value at its nearest
level. Clamping first moves a value to the boundary and then rounds it again,
which loses a level at each end.

The lab's weight is `(out_features, in_features) = (512, 1024)`. The following
table shows every tensor's shape:

| Tensor | Shape | Dtype |
|---|---|---|
| `weight` | `(512, 1024)` | float32 |
| `scale`, per tensor | `()` | float32 |
| `scale`, per channel | `(512, 1)` | float32 |
| `scale`, per group of 128 | `(512, 8)` | float32 |
| `quantized` | `(512, 1024)` | int8 |

## The error bound

How wrong can a quantized weight be? Never more than half a step. For any $w$
that doesn't clamp, the reconstruction error is the rounding residual scaled
back up:

$$
\hat{w} - w = \hla{s}\left(\operatorname{round}\!\left(\frac{w}{\hla{s}}\right)
- \frac{w}{\hla{s}}\right)
$$

Round-to-nearest satisfies $|\operatorname{round}(x) - x| \le 1/2$, which gives
the bound:

> [!KEY] The error is at most half a step
> $$
> \boxed{\lvert \hat{w} - w \rvert \le \frac{\hla{s}}{2}}
> $$
>
> Every later decision in this chapter is an attempt to make $\hla{s}$ smaller.

The bound ==doesn't depend on the magnitude of $w$==. A fixed-point format
spends the same absolute precision on a weight of 0.001 as on a weight of 3.0,
which is exactly what makes outliers so damaging.

The bound is the worst case. For the typical error, model the residual as
uniform on $[-\hla{s}/2, \hla{s}/2]$. A uniform variable on an interval of width
$\hla{s}$ has variance $\hla{s}^2/12$, so the root-mean-square (RMS) error per
weight is the following:

$$
\sigma_{\text{err}} = \frac{\hla{s}}{\sqrt{12}}
$$

## What the bound predicts

With the RMS error in hand, you can predict the error of a block of weights
before you run anything. Take a block of $\hlc{n}$ weights drawn from
$\mathcal{N}(0, \sigma^2)$, a normal distribution with standard deviation
$\sigma$. That's what the lab uses, and it's a fair first approximation to a
trained layer.

The block's scale is set by its largest magnitude. For $\hlc{n}$ independent
normals, the expected maximum grows as follows:

$$
\mathbb{E}\left[\max_i |w_i|\right] \approx \sigma\sqrt{2\ln(2\hlc{n})}
$$

So $\hla{s} = \sigma\sqrt{2\ln(2\hlc{n})}\,/\,127$. Substitute that into the RMS
error and divide by $\sigma$ to get the relative error:

$$
\boxed{
\varepsilon(\hlc{n}) \approx \frac{\sqrt{2\ln(2\hlc{n})}}{127\sqrt{12}}
= \frac{\sqrt{2\ln(2\hlc{n})}}{440}
}
$$

In words: the more weights share a scale, the larger their maximum tends to be,
and the coarser the step. The following table evaluates it for the three
granularities on the lab's `(512, 1024)` weight:

| Granularity | $n$ per scale | $\sqrt{2\ln 2n}$ | Predicted $\varepsilon$ |
|---|---|---|---|
| Per group of 128 | 128 | 3.33 | 0.76% |
| Per channel | 1024 | 3.90 | 0.89% |
| Per tensor | 524288 | 5.27 | 1.20% |

Three things follow:

- The ordering is right, which is what the lab's tests assert.
- The predicted group-wise error of 0.76% is under the lab's 1% threshold, so
  you can tell before running anything whether the test passes.
- The gap is small, a factor of 1.6 between per-tensor and per-group.

> [!INTUITION]
> The maximum of $\hlc{n}$ Gaussians grows like $\sqrt{\ln \hlc{n}}$, which is
> close to not growing at all. If weights really were Gaussian, per-tensor
> quantization would be almost fine. It isn't fine, because weights aren't
> Gaussian.

## Why per-tensor scaling collapses

Real weights break the Gaussian picture in one specific way: trained weight
matrices contain a small number of entries far outside the bulk distribution.
==A single outlier ruins every weight that shares its scale==, because it sets
$\hld{\beta}$ for all of them.

Go back to the four-weight example and make the last weight ten times larger,
$-12.7$. The scale becomes $\hla{s} = 12.7/127 = 0.1$:

| Weight $w$ | $w/\hla{s}$ | Stored $q$ | Dequantized | Error |
|---|---|---|---|---|
| 0.834 | 8.34 | 8 | 0.8 | 0.034 |
| $-0.301$ | $-3.01$ | $-3$ | $-0.3$ | 0.001 |
| 0.027 | 0.27 | 0 | 0 | 0.027 |
| $-12.7$ | $-127$ | $-127$ | $-12.7$ | 0 |

The outlier itself is exact. But the small weight 0.027 is gone entirely, and
the error on 0.834 grew from 0.004 to 0.034. The lab's outlier case is the same
effect at full size.

> [!EXAMPLE] One outlier of 500 in the lab's matrix
> The lab's outlier case is a `(512, 1024)` matrix of standard normals with a
> single entry set to 500. With one scale for the whole tensor:
>
> $$
> \hla{s} = \frac{500}{127} = 3.937,
> \qquad \frac{\hla{s}}{2} = 1.969
> $$
>
> Every weight with $|w| < 1.969$ rounds to the integer 0 and dequantizes to
> exactly zero. For a standard normal, that's this fraction:
>
> $$
> P(|w| < 1.969) = 2\Phi(1.969) - 1 = 0.951
> $$

So 95% of the matrix is replaced by zeros. The remaining 5% is quantized to
$\pm 1$ or $\pm 2$, which is three or four distinct levels for the entire
distribution.

Carry the arithmetic through: integrate the squared rounding error over the
normal density, and divide by the total squared norm, which the 500 itself
inflates. The relative error comes out near 0.76. Per group of 128, only the one
group holding the outlier is damaged, and the relative error stays near 0.013.
The ratio is about 58.

The lab asserts a ratio above 5 and measures the real values. The point of the
arithmetic is that you can predict the collapse rather than discover it.

> [!KEY] Quantization error is a property of the group, not of the weight
> The outlier isn't badly represented. It sets the endpoint, so it's represented
> exactly. It causes error by stretching $\hla{s}$, which every other weight in
> its group shares.

## Per-channel and group-wise scaling

If one outlier damages everything that shares its scale, the fix is to shrink
the set of weights that share a scale. There are two standard granularities:

- **Per channel.** One scale per output row. In a linear layer, each output row
  produces one output feature independently, so a row with an outlier damages
  only its own output. The scale shape is `(out_features, 1)`.
- **Per group.** Split the input dimension into contiguous blocks of $\hlc{g}$
  weights and give each block its own scale. The scale shape is
  `(out_features, in_features // g)`. This is the standard choice, and
  $\hlc{g} = 128$ is the standard size.

The reshape is the whole implementation:

```python
def quantize_per_group(weight, group_size=128):
    out, inp = weight.shape                                # (512, 1024)
    grouped = weight.reshape(out, inp // group_size, group_size)   # (512, 8, 128)
    scale = grouped.abs().amax(dim=-1, keepdim=True) / 127.0       # (512, 8, 1)
    q = (grouped / scale).round().clamp(-128, 127).to(torch.int8)
    return q.reshape(out, inp), scale.squeeze(-1)          # (512, 1024), (512, 8)
```

> [!WARNING] The `keepdim=True` on the amax
> Without it, the scale is `(512, 8)`, and broadcasting against `(512, 8, 128)`
> aligns the group axis with the within-group axis. That silently produces
> garbage rather than an error.

### The metadata cost of a group size

Smaller groups mean more scales, and storing a scale costs bytes too. With int8
weights and 2-byte scales, the bytes per weight are the following:

$$
\boxed{b(\hlc{g}) = 1 + \frac{2}{\hlc{g}}}
$$

At $\hlc{g} = 128$, that's $b(128) = 1 + 2/128 = 1.015625$ bytes per weight,
against 2 bytes for bfloat16:

- The stored size is $1.015625 / 2 = 50.78\%$ of bfloat16.
- The compression factor is $2 / 1.015625 = 1.969$.
- The scales cost 1.56% on top of the int8 weights.

That 1.56% is the number to weigh against the quality gain, and it's small
enough that going finer than 128 is rarely worth it:

| Group size $g$ | Bytes per weight | Fraction of bf16 | Scale overhead |
|---|---|---|---|
| 32 | 1.0625 | 53.1% | 6.25% |
| 64 | 1.03125 | 51.6% | 3.13% |
| 128 | 1.015625 | 50.8% | 1.56% |
| Per channel, 5120 wide | 1.00039 | 50.0% | 0.04% |
| Per tensor | 1.0 | 50.0% | ~0% |

> [!EXAMPLE] The lab's storage at $g = 128$
> The lab stores a `(512, 1024)` weight at $\hlc{g} = 128$:
>
> $$
> 512 \times 1024 + 512 \times \frac{1024}{128} \times 2
> = 524288 + 8192 = 532480 \text{ bytes}
> $$
>
> And $532480 / 1048576 = 50.78\%$, matching $b(128)/2$ exactly. The test checks
> that this stays under 55%.

For real weights rather than Gaussian ones, the measured quality gap between the
granularities is much wider than the $\sqrt{\ln n}$ argument predicts, because
of the outliers:

| Granularity | Typical weight error | Overhead |
|---|---|---|
| Per tensor | 2 to 5% | ~0% |
| Per channel | 0.5 to 1% | 0.02% |
| Per group of 128 | 0.1 to 0.3% | 1.5% |

## Weight-only against weight-and-activation

So far you've quantized weights. You can also quantize *activations*, the
values flowing between layers, and which scheme helps depends entirely on which
roof you're under. There are two schemes:

- **Weight-only (W8A16).** Store weights in int8. Read them, dequantize to
  bfloat16 in registers, and do a bfloat16 matmul against bfloat16 activations.
  The arithmetic never becomes integer arithmetic.
- **Weight-and-activation (W8A8).** Quantize activations too, and do the matmul
  with int8 inputs and int32 accumulation on the tensor cores, the GPU's
  matrix-multiply units.

### Decode time halves

Decode is where weight-only quantization pays off. A decode step at batch 1
reads every weight once and does two FLOPs per weight. The bytes dominate, so
the time is the weight bytes over the bandwidth:

$$
t_{\text{decode}} \approx \frac{\text{weight bytes}}{\text{bandwidth}}
$$

At bfloat16, the weights are 53.8 GB. Divide by the measured device-to-device
copy rate of 1275 GB/s:

$$
t_{\text{bf16}} = \frac{53.8 \times 10^9}{1275 \times 10^9} = 42.2 \text{ ms}
$$

In int8 with $\hlc{g} = 128$, the same weights take
$53.8 \times (1.015625/2) = 27.3$ GB:

$$
t_{\text{int8}} = \frac{27.3 \times 10^9}{1275 \times 10^9} = 21.4 \text{ ms}
$$

That's a speedup of $42.2/21.4 = 1.97$, or 23.7 tokens per second against 46.7.
==Halving the bytes halves the time, because nothing else was the bottleneck.==
The remaining 1.6% of the gap from a clean factor of two is the scales.

### Dequantization is free

You might object that you added work. Per weight, the kernel now reads 1 byte,
does 1 multiply to dequantize, and does 2 FLOPs for the multiply-add. That's 3
FLOPs per byte.

The ridge point is 161 FLOPs per byte. At 3, the kernel is memory bound by a
factor of 54, so the arithmetic finishes long before the next bytes arrive, and
dequantization costs nothing measurable.

For comparison, the bfloat16 version runs at 2 FLOPs per 2 bytes, or 1 FLOP per
byte. Quantization moved the kernel from 1 to 3 FLOPs per byte, which is still
nowhere near 161.

### Prefill gets nothing

Prefill at 2048 tokens has an arithmetic intensity around 2000, well past the
ridge point. It's compute bound, so its time is FLOPs over the tensor core rate.
Weight-only quantization doesn't change the FLOP count, because the matmul is
still bfloat16.

Expect decode to roughly double and prefill to stay flat or get slightly worse,
because the dequantization adds a small amount of real work to a kernel that's
already compute bound.

The only way to speed up prefill is W8A8, which halves the FLOP cost by using
the integer tensor cores. That's a different and much harder problem, for the
reasons the next sections give.

## What Ampere's int8 tensor cores do

W8A8 is attractive for prefill, so it's worth knowing exactly what the hardware
offers. Ampere's tensor cores have an integer mode. The `mma` (matrix
multiply-accumulate) instruction takes int8 operands for both matrices,
multiplies them, and accumulates into int32. The published integer rate on the
A100 is higher than its bfloat16 rate.

To use it, the matmul must be expressible entirely in integers. Write the
activation as $x_k = \hla{s_x} q_{x,k}$, with a per-tensor activation scale
$\hla{s_x}$. Write the weight as $W_{kj} = \hla{s_{w,j}} q_{w,kj}$, with a
per-output-channel weight scale $\hla{s_{w,j}}$. Then the scales factor out:

$$
y_j = \sum_{k} x_k W_{kj}
= \hla{s_x}\, \hla{s_{w,j}} \sum_{k} q_{x,k}\, q_{w,kj}
$$

> [!INTUITION]
> Neither scale depends on the summation index $k$, so both leave the sum. What
> remains is a pure integer dot product, exactly what the tensor core computes
> into its int32 accumulator. One float multiply per output element converts it
> back. That's the whole of W8A8.

**Accumulator headroom.** Each product is at most $127 \times 127 = 16129$. With
$K = 5120$ terms, the worst-case accumulator is
$5120 \times 16129 = 8.26 \times 10^{7}$, against an int32 range of
$2^{31} = 2.15 \times 10^{9}$. That's a factor of 26 of headroom, so int32
accumulation can't overflow at this model's dimensions. That's a genuine
advantage over float16 accumulation, which can and does overflow.

**Why group-wise scaling breaks it.** The derivation pulled $\hla{s_{w,j}}$ out
of the sum because it didn't depend on $k$. A group-wise scale does depend on
$k$: it changes every 128 steps. You'd have to split the $K$ loop into chunks of
128, convert each int32 partial sum to float, scale it, and add. A conversion
and a float add every 128 accumulation steps defeats much of the point.

> [!KEY] Group-wise scaling is a weight-only technique
> W8A8 kernels use per-tensor or per-channel scales. That's also why W8A8 needs
> the activation outlier machinery later in this chapter, and weight-only
> doesn't.

For this course, weight-only is the right choice. Decode is where the time goes,
decode is memory bound, and weight-only gets the full memory saving with none of
the activation difficulty.

## Ampere has no FP8

FP8 would suit the outlier problem well, but the A100 can't compute in it.
Recall the formats table: FP8 spends bits on an exponent, so its precision is
relative rather than absolute. A value ten times larger gets the same number of
significant digits, not ten times worse.

The `e4m3` variant has three mantissa bits. With the implicit leading one, that
gives eight steps per octave and a worst-case relative rounding error of
$2^{-4} = 6.25\%$ at every magnitude. An outlier costs an exponent increment,
not the mantissa of every other value in its block.

Ada and Hopper have tensor cores that do FP8 matmuls natively. The A100 is
Ampere, compute capability 8.0, and ==has no FP8 tensor core support at all==.

You can store weights in an FP8 layout and unpack them in software for the
bandwidth saving, but there's no compute speedup. The arithmetic still runs in
bfloat16, and the unpack is more work than an int8 dequantization. On Ampere,
int8 gives the same bandwidth saving with better tooling and a faster unpack.

> [!TIP] Check the capability before you write an FP8 kernel
> This is a hard constraint, worth knowing before you spend a day on a kernel
> that can't be fast. Check it in code rather than trusting the part number:
>
> ```python
> major, minor = torch.cuda.get_device_capability()
> has_fp8 = (major, minor) >= (8, 9)    # Ada and Hopper onward
> ```

The model publishes an official FP8 checkpoint, `Qwen/Qwen3.8-27B-FP8`, which is
useful on an H100 and not here.

## Activation outliers

If W8A8 is so attractive, why is weight-only the common choice? Because
activations are much harder to quantize than weights. Transformer activations
contain outliers up to 100 times the typical magnitude, concentrated in a small
number of channels, and the same channels are outliers for every token.

That structure makes the problem tractable: ==the outliers are per channel, not
scattered==. Two methods exploit it.

**SmoothQuant** moves the difficulty from activations into weights. For a linear
layer $y = xW$, insert a diagonal matrix and its inverse:

$$
y = x W = \left(x \operatorname{diag}(\mathbf{c})^{-1}\right)
\left(\operatorname{diag}(\mathbf{c}) W\right)
$$

The product is unchanged. Choosing $c_k$ large for the channels with large
activations shrinks those activation channels and grows the matching weight
rows. With the following choice and $\alpha$ around 0.5, the difficulty splits
evenly and both tensors become quantizable:

$$
c_k = \frac{(\max_t |x_{tk}|)^{\alpha}}{(\max_j |W_{kj}|)^{1-\alpha}}
$$

The scaling folds into the preceding layer's norm weights at no runtime cost.

**AWQ** observes that a small fraction of weight channels matter far more than
the rest. It identifies them from activation statistics rather than weight
magnitudes, and protects them with a per-channel scale. The important weights
are the ones multiplying large activations, which you can't see by looking at
the weights alone.

## Calibration

Activation quantization needs a step that weight quantization doesn't. Weight
quantization is data-free: the range of a weight tensor is a property of the
tensor. You can't know the range of an activation without running data through
the model.

*Calibration* finds that range. Run a few hundred representative sequences,
record per-channel activation statistics, and derive the scales.

The naive choice is the observed maximum. That's wrong for the same reason
per-tensor weight scaling is wrong: one outlier in the calibration set sets the
scale forever. The better choice is the clipping threshold $\tau$ that minimizes
reconstruction error:

$$
\tau^{\star} = \arg\min_{\tau}\;
\mathbb{E}\left[\left(Q_{\tau}(x) - x\right)^2\right]
$$

Here $Q_{\tau}$ clamps to $[-\tau, \tau]$ and then quantizes with
$\hla{s} = \tau/127$.

> [!INTUITION]
> Raising $\tau$ clips less but rounds more coarsely, because $\hla{s}$ grows
> with $\tau$. The best threshold sits where those two errors balance, which for
> heavy-tailed activations is well below the observed maximum.

The calibration set matters. If you calibrate on English prose and serve code,
the scales are wrong in a way no perplexity check on English reveals.

## Quantizing the KV cache

The *KV cache* holds each past token's attention keys and values, so decode
doesn't recompute them. Quantizing it halves it, but keys and values need
different treatment. The cache matters at long context and large batch, where
the cache read starts to rival the weight read.

> [!EXAMPLE] The cache at int8, per token
> This model spends 64 KiB per token on KV cache:
>
> $$
> 16 \text{ layers} \times 4 \text{ KV heads} \times 256 \text{ head dim}
> \times 2 \text{ tensors} \times 2 \text{ bytes} = 65536 \text{ bytes}
> $$
>
> Quantize to int8 with one scale per key or value vector, that is, per layer,
> per head, and per token. That gives 32 KiB of data plus this many bytes of
> scales:
>
> $$
> 16 \times 4 \times 2 \times 2 = 256 \text{ bytes}
> $$
>
> The total is 32.25 KiB per token, a compression factor of 1.98. At 32k
> context, a sequence's cache drops from 2.0 GiB to 1.008 GiB.

### Why keys tolerate it worse than values

Keys and values enter attention at different places, so their errors propagate
differently. Keys decide *which* tokens a head attends to, and values are
*what* it reads from them.

**Keys feed a softmax, so their error is exponentiated.** The score for key $j$
is $\ell_j = q^{\top} k_j / \sqrt{d}$, where $q$ is the query and $d$ the head
dimension. Perturb the key by $e_j$, with each element an independent rounding
residual of variance $\hla{s_k}^2/12$:

$$
\Delta \ell_j = \frac{q^{\top} e_j}{\sqrt{d}},
\qquad
\operatorname{Var}(\Delta \ell_j)
= \frac{\lVert q \rVert^2 \hla{s_k}^2}{12\,d}
$$

The attention weight is $a_j \propto e^{\ell_j}$, so an absolute score error
$\delta$ multiplies that token's unnormalized weight by $e^{\delta}$:

- A score error of 0.1 changes the weight by 10.5%.
- A score error of 0.3 changes it by 35%.

The error doesn't average out, because it changes the weights themselves before
any averaging happens. Two consequences fall out of the variance formula:

- The error grows with $\lVert q \rVert$, so confident queries suffer the most.
  Those are the ones whose attention is sharply peaked and matters most.
- The error grows with $\hla{s_k}$, which the largest element of the key vector
  sets. A single large key component degrades every score that key takes part
  in.

**Values are combined linearly, so their error averages out.** The output is
$o = \sum_j a_j v_j$, with $a_j \ge 0$ and $\sum_j a_j = 1$. Perturb each value
by an independent residual of variance $\hla{s_v}^2/12$:

$$
\operatorname{Var}(\Delta o) = \frac{\hla{s_v}^2}{12}\sum_j a_j^2
= \frac{\hla{s_v}^2}{12\,\hlc{n_{\text{eff}}}},
\qquad
\hlc{n_{\text{eff}}} = \frac{1}{\sum_j a_j^2}
$$

The quantity $\hlc{n_{\text{eff}}}$ is the effective number of tokens the head
attends to:

- Spread attention over 100 tokens roughly equally, and
  $\hlc{n_{\text{eff}}} \approx 100$, so the value error shrinks by a factor of
  10.
- In the worst case, the head attends to a single token and
  $\hlc{n_{\text{eff}}} = 1$. The output error equals the quantization error of
  that one value: no amplification, and no suppression either.

> [!KEY] Values never amplify error; keys always do
> That asymmetry is the reason for the standard configuration: keys in bfloat16
> or with per-head scales, and values in int8. Quantizing both to int8 with
> per-tensor scales quietly degrades long-context quality.

## The model at three precisions

Now scale everything up to the whole model: quantization buys both time and
memory. The weights are 53.8 GB in bfloat16, which is
$53.8 \times 10^9 / 2 = 26.9 \times 10^9$ parameters. Apply $b(\hlc{g})$ with
$\hlc{g} = 128$:

| Precision | Bytes per weight | Weight size | Decode floor at 1275 GB/s |
|---|---|---|---|
| bfloat16 | 2 | 53.8 GB | 42.2 ms |
| int8, $g = 128$ | 1.015625 | 27.3 GB | 21.4 ms |
| int4, $g = 128$ | 0.515625 | 13.9 GB | 10.9 ms |

The int4 row is $26.9 \times 10^9 \times (0.5 + 2/128) = 13.87 \times 10^9$
bytes. The speedups against bfloat16 are 1.97 and 3.87.

The memory freed matters as much as the speedup. Moving to int8 returns
$53.8 - 27.3 = 26.5$ GB of HBM, the GPU's main memory. At 64 KiB per token,
that's this much more KV cache:

$$
\frac{26.5 \times 10^{9}}{65536} \approx 404{,}000 \text{ tokens}
$$

At a batch of 32, that's roughly 12,600 more tokens of context per sequence. On
a card where the model already leaves only about 19 GiB free, that's the
difference between serving short chats and serving documents.

### int4 costs quality

int4 isn't free in quality the way int8 nearly is. Symmetric int4 has levels
$[-7, 7]$, so at the same group size the step is much larger:

$$
\hla{s_{\text{int4}}} = \frac{\hld{\beta}}{7} = \frac{127}{7}\,\hla{s_{\text{int8}}}
= 18.1\; \hla{s_{\text{int8}}}
$$

By the error bound, the reconstruction error grows by the same factor. The
$\sqrt{2\ln(2n)}$ estimate at $\hlc{g} = 128$ goes from 0.76% to about 13.7%.

That's why ==int4 is where round-to-nearest stops working==. Weight-only schemes
start needing the AWQ or GPTQ machinery, which spends a calibration pass
choosing rounding directions that minimize the *output* error rather than the
weight error.

## Measuring the damage

Weight error is a proxy. What matters is the model's output, so this section
covers how to check it. Use three checks, in increasing order of cost and
usefulness:

1. **Layer output error.** Run one layer with and without quantization on the
   same input and compare. It's fast, and it localizes the damage. This is what
   the lab checks last: `x @ dequantize(q, s, 128).T` against `x @ weight.T`,
   with a 1% threshold on the relative norm of the difference.
2. **Logit correlation.** Run a full forward pass on a few prompts and compare
   the *logits*, the model's raw scores for every vocabulary token. Correlation
   must stay above 0.999, and top-1 agreement above 0.98.
3. **Perplexity on held-out text.** *Perplexity* measures how surprised the
   model is by real text, and it's the real measure. An increase of more than
   about 1% means the scheme is too aggressive.

==Doing only the first check is how broken quantization schemes ship==.

It's worth knowing how the weight error and the layer output error relate. Take
$y = xW$ with a weight perturbation $\Delta W$ whose elements are independent of
$x$ and of each other:

$$
\operatorname{Var}(\Delta y_j) = \lVert x \rVert^2 \sigma_{\text{err}}^2,
\qquad
\operatorname{Var}(y_j) = \lVert x \rVert^2 \sigma_{W}^2
$$

The $\lVert x \rVert^2$ cancels in the ratio, so the relative output error
equals the relative weight error. The $K$ terms in the sum don't average the
error away, because they also carry the signal. That's why the lab uses the same
1% threshold for both.

The equality fails exactly when the perturbation correlates with the input: when
the weights you rounded badly are the ones multiplying the largest activations.
That correlation is what AWQ measures, and it's what makes the output error
worse than the weight error on real data.

## What goes wrong

These are the mistakes the lab and a real deployment are most likely to
provoke.

> [!WARNING] Dequantizing the whole tensor and then calling a matmul
> This is the most common mistake, and it undoes the entire benefit: you write
> 53.8 GB of bfloat16 back to HBM and read it again. Dequantization must happen
> inside the matmul kernel, in registers. Compare the measured time against the
> roofline. If it matches the bfloat16 figure, this is what happened.

**Forgetting `keepdim=True` on the amax.** The scale broadcasts along the wrong
axis. There's no shape error, because the sizes happen to be compatible; the
weights come back wrong. Print the scale's shape and check it against the shapes
table in the symmetric quantization section.

**Clamping before rounding.** This loses one level at each end. The symptom is a
slightly larger error than the bound predicts, with no other sign.

**Quantizing in float16 instead of float32.** The division `weight / scale` in
float16 loses precision before the rounding happens, so the error exceeds its
own bound. Do the arithmetic in float32 and cast at the end. The same applies to
the stored scale: a bfloat16 scale has 8 mantissa bits and carries a 0.4%
relative error that multiplies every weight in its group.

**Calibrating on the wrong data.** Activation scales fitted to one domain fail
silently on another. There's no runtime symptom, only worse outputs.

> [!RECAP]
> - Symmetric int8 uses $\hla{s} = \hld{\beta}/127$ and one multiply to
>   dequantize, and its error is at most $\hla{s}/2$ per weight.
> - An outlier stretches the shared scale. Per-group scales at $g = 128$ contain
>   the damage for 1.56% of metadata, 50.78% of bfloat16 in total.
> - Weight-only int8 takes decode from 42.2 ms to 21.4 ms per token and leaves
>   compute-bound prefill flat.
> - Ampere's int8 tensor cores need per-tensor or per-channel scales, and the
>   A100 has no FP8 tensor cores.
> - Key error in a KV cache is exponentiated by the softmax; value error is
>   averaged down by $n_{\text{eff}}$.
> - Judge a scheme by perplexity, not by weight error alone.

## Check your understanding

> [!QUESTION] Why does quantizing weights to int8 roughly double decode speed but do nothing for prefill?
> Decode at batch 1 has an arithmetic intensity near 1 FLOP per byte, far below
> the ridge point of 161, so its time is bytes over bandwidth and halving the
> bytes halves the time. Prefill at 2048 tokens has an intensity near 2000,
> above the ridge point, so its time is FLOPs over the tensor core rate.
> Weight-only quantization changes bytes, not FLOPs. It even makes prefill
> slightly worse, because dequantization adds FLOPs to a kernel that's already
> compute bound.

> [!QUESTION] A group size of 128 costs 1.56% in metadata. Why not use 32 and get better quality?
> You can, and the error does improve, but the arithmetic says the return is
> small and the cost isn't. Going from $g = 128$ to $g = 32$ takes the overhead
> from 1.56% to 6.25%, so the stored size goes from 50.8% to 53.1% of bfloat16:
> you give back 4.5% of the decode speedup. By the $\sqrt{2\ln(2n)}$ argument,
> the error improves only from 0.76% to 0.66%. Four times the metadata for a 13%
> reduction in error is a poor trade, which is why 128 is the standard.

> [!QUESTION] You quantize both keys and values to int8, and long-context quality drops while short-context quality is fine. Which tensor is at fault, and why?
> Keys. Key error enters the softmax exponent, so an absolute score error
> $\delta$ scales that token's attention weight by $e^{\delta}$, and the error
> grows with $\lVert q \rVert$. At long context, more keys compete, so a
> perturbation that reorders the top few of them changes the output
> substantially. Value error is a convex combination and is suppressed by
> $\sqrt{n_{\text{eff}}}$, which gets *better* at long context, not worse. Put
> keys back in bfloat16 and re-measure.

## Lab

> [!TRY]
> Implement symmetric int8 quantization at three granularities, plus the helpers
> that dequantize, measure error, and count bytes. Passing means every harness
> check in the following list holds.

Implement the following functions:

- `quantize_per_tensor`.
- `quantize_per_channel`, returning a `(out, 1)` scale.
- `quantize_per_group`, returning a `(out, in // g)` scale.
- `dequantize`, which broadcasts a group scale back across its group.
- `relative_error`, as the Frobenius norm ratio.
- `storage_bytes`, for the int8 data plus its scales.

The harness checks the following, in order:

1. Quantized values stay inside the int8 range, and the largest weight maps near
   127.
2. Finer granularity gives lower error.
3. Group-wise error stays under 1%.
4. A single outlier of 500 hurts a per-tensor scale more than five times as much
   as a per-group one.
5. `storage_bytes` matches
   $\text{out} \times \text{in} + \text{out} \times (\text{in}/g) \times 2$ and
   comes in under 55% of bfloat16.
6. A real matmul against the dequantized weights keeps the layer output error
   under 1%.

## Further reading

- [SmoothQuant: accurate and efficient post-training quantization for large language models](https://arxiv.org/abs/2211.10438)
- [AWQ: activation-aware weight quantization for LLM compression and acceleration](https://arxiv.org/abs/2306.00978)
- [GPTQ: accurate post-training quantization for generative pre-trained transformers](https://arxiv.org/abs/2210.17323)
- [LLM.int8(): 8-bit matrix multiplication for transformers at scale](https://arxiv.org/abs/2208.07339) — the paper that identified the activation outlier channels.
- [Quantization and training of neural networks for efficient integer-arithmetic-only inference](https://arxiv.org/abs/1712.05877) — the original derivation of the affine scheme used here.
- [KIVI: a tuning-free asymmetric 2bit quantization for KV cache](https://arxiv.org/abs/2402.02750)
