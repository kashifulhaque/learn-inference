---
title: Quantization on Ampere
slug: 18-quantization
part: "Part 6 — Scaling"
summary: Deriving affine and symmetric quantization, why grouping fixes outliers, and why FP8 is not available to you on an A100.
minutes: 105
gpu: true
objectives:
  - Derive affine and symmetric quantization and bound the reconstruction error by the step size.
  - Explain why per-tensor scaling collapses under outliers, and derive per-channel and group-wise scaling as the fix.
  - Compute the metadata cost of a group size and the memory saved at model scale.
  - Explain why weight-only quantization helps decode and not prefill, with the byte arithmetic.
  - Describe what Ampere's int8 tensor cores compute, and why Ampere has no FP8.
  - Quantize a KV cache and say why keys tolerate it worse than values.
lab: 18-int8-quant
---

# Quantization on Ampere

The weights are 53.8 GB and decode reads all of them for every token. At the
A100's measured copy bandwidth of 1275 GB/s that is a floor of 42.2 ms per
token, or 23.7 tokens per second, before a single FLOP. Halving the bytes halves
the floor. Quantization is the most direct throughput improvement available to
this engine, and it is also the easiest way to wreck the model without noticing.

This chapter does the numerics properly. Quantization is a rounding scheme, and
a rounding scheme has an error bound you can derive rather than measure. Once
you have the bound, every design choice in the rest of the chapter — symmetric
against affine, per-tensor against per-group, weights only against weights and
activations — follows from it.

## Before you start

**Fixed-point representation.** An int8 value is an integer in $[-128, 127]$. To
represent a real number with it, you need a scale (how much one integer step is
worth) and optionally an offset (which integer means zero). That pair is the
entire scheme.

**Floating-point formats.** Chapter 0a covers bfloat16, float16, and the FP8
variants: how many bits go to the exponent, how many to the mantissa, and what
that buys. The one fact you need here is that a floating-point format has
roughly constant *relative* precision across magnitudes, while a fixed-point
format has constant *absolute* precision. That difference is what the outlier
discussion is about.

**The roofline from chapter 10.** Decode is memory bound with an arithmetic
intensity around 1 FLOP per byte against a ridge point of 161. Quantization
changes the byte count, which is the denominator of that ratio, so it moves
decode along the memory-bound roof. It does not move prefill, which is already
on the compute roof.

**Norms.** The relative error used throughout is the Frobenius norm of the
difference over the Frobenius norm of the original:

$$
\varepsilon = \frac{\lVert \hat{W} - W \rVert_F}{\lVert W \rVert_F}
$$

**Rounding as noise.** Round-to-nearest with step $s$ leaves a residual in
$[-s/2, s/2]$. Treating that residual as a uniform random variable is the
standard model and is accurate whenever $s$ is small compared with the spread of
the data.

## Affine quantization, derived

You have real values $w$ lying in a range $[\beta_{\min}, \beta_{\max}]$, and
you want to store them as integers in $[q_{\min}, q_{\max}]$. For int8 that is
$[-128, 127]$.

Ask for an affine map that sends the two endpoints of the real range to the two
endpoints of the integer range. Write the inverse map — dequantization — as

$$
\hat{w} = s\,(q - z)
$$

where $s > 0$ is the **scale** (the real-valued width of one integer step) and
$z$ is the **zero point** (the integer that dequantizes to exactly zero).

Match the endpoints. Requiring $\hat{w} = \beta_{\min}$ at $q = q_{\min}$ and
$\hat{w} = \beta_{\max}$ at $q = q_{\max}$ gives two equations. Subtracting them
eliminates $z$:

$$
\beta_{\max} - \beta_{\min} = s\,(q_{\max} - q_{\min})
\quad\Longrightarrow\quad
s = \frac{\beta_{\max} - \beta_{\min}}{q_{\max} - q_{\min}}
$$

Substituting back into the first equation gives the zero point:

$$
z = q_{\min} - \frac{\beta_{\min}}{s}
$$

The zero point must itself be an integer, so that $w = 0$ maps to an exact
integer and dequantizes back to exactly zero. Round it:

$$
z = \operatorname{round}\!\left(q_{\min} - \frac{\beta_{\min}}{s}\right)
$$

Quantization is then the inverse of the dequantization map, rounded and clamped:

$$
q = \operatorname{clamp}\!\left(
\operatorname{round}\!\left(\frac{w}{s}\right) + z,\;
q_{\min},\; q_{\max}\right)
$$

The clamp matters. Without it, a value outside $[\beta_{\min}, \beta_{\max}]$ —
which happens whenever you calibrate the range on one sample and apply it to
another — wraps around in int8 and turns a large positive number into a large
negative one. That failure is silent and catastrophic.

Check the endpoints by substituting $z$ back in. At $w = \beta_{\min}$ the
$\beta_{\min}/s$ terms cancel and leave $q_{\min}$; at $w = \beta_{\max}$ you get
$(\beta_{\max} - \beta_{\min})/s + q_{\min} = q_{\max}$, using the definition of
$s$.

## Symmetric quantization

Weights are close to symmetric around zero, so the extra generality of an
offset buys little and costs an add per element on the dequantization path.

Set $\beta_{\max} = -\beta_{\min} = \beta$ where $\beta = \max|w|$. Then
$z = 0$ falls out, and dequantization is one multiply. To keep the mapping
genuinely symmetric you must also give up one integer level: use
$[-127, 127]$ rather than $[-128, 127]$, because $-128$ has no positive partner.

$$
s = \frac{\beta}{127},
\qquad
q = \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{w}{s}\right),
\; -128,\; 127\right),
\qquad
\hat{w} = s\,q
$$

In code, which is what the lab asks for:

```python
scale = weight.abs().amax(dim=-1, keepdim=True) / 127.0   # (out, 1)
quantized = (weight / scale).round().clamp(-128, 127).to(torch.int8)
dequantized = quantized.to(torch.bfloat16) * scale
```

Round before clamping, not after. Rounding first then clamping keeps every
in-range value at its nearest level; clamping first would move a value to the
boundary and then round it again, which loses a level at each end.

Shapes, for the lab's `(out_features, in_features) = (512, 1024)` weight:

| Tensor | Shape | Dtype |
|---|---|---|
| `weight` | `(512, 1024)` | float32 |
| `scale`, per tensor | `()` | float32 |
| `scale`, per channel | `(512, 1)` | float32 |
| `scale`, per group of 128 | `(512, 8)` | float32 |
| `quantized` | `(512, 1024)` | int8 |

## The error bound

For any $w$ that does not clamp, the reconstruction error is the rounding
residual scaled back up:

$$
\hat{w} - w = s\left(\operatorname{round}\!\left(\frac{w}{s}\right)
- \frac{w}{s}\right)
$$

Round-to-nearest satisfies $|\operatorname{round}(x) - x| \le 1/2$, so

$$
\lvert \hat{w} - w \rvert \le \frac{s}{2}
$$

That is the whole error bound, and every subsequent decision is an attempt to
make $s$ smaller. Note what it does *not* depend on: the magnitude of $w$. A
fixed-point format spends the same absolute precision on a weight of 0.001 as on
a weight of 3.0, which is exactly the behavior that makes outliers so damaging.

For the expected error rather than the worst case, model the residual as uniform
on $[-s/2, s/2]$. A uniform variable on an interval of width $s$ has variance
$s^2/12$, so the root-mean-square error per weight is

$$
\sigma_{\text{err}} = \frac{s}{\sqrt{12}}
$$

## What the bound predicts

Take a block of $n$ weights drawn from $\mathcal{N}(0, \sigma^2)$, which is what
the lab uses and a fair first approximation to a trained layer. Two facts
combine.

The block's scale is set by its largest magnitude. For $n$ independent standard
normals, the expected maximum absolute value grows as

$$
\mathbb{E}\left[\max_i |w_i|\right] \approx \sigma\sqrt{2\ln(2n)}
$$

so $s = \sigma\sqrt{2\ln(2n)}\,/\,127$. Substituting into the RMS error and
dividing by $\sigma$ to get the relative error:

$$
\varepsilon(n) \approx \frac{\sqrt{2\ln(2n)}}{127\sqrt{12}}
= \frac{\sqrt{2\ln(2n)}}{440}
$$

Now evaluate it for the three granularities on the lab's `(512, 1024)` weight:

| Granularity | $n$ per scale | $\sqrt{2\ln 2n}$ | Predicted $\varepsilon$ |
|---|---|---|---|
| Per group of 128 | 128 | 3.33 | 0.76% |
| Per channel | 1024 | 3.90 | 0.89% |
| Per tensor | 524288 | 5.27 | 1.20% |

Three things follow. The ordering is right, which is what the lab's tests
assert. The predicted group-wise error of 0.76% is under the lab's 1% threshold,
so you can tell before running anything whether the test should pass. And the
gap is surprisingly small — a factor of 1.6 between per-tensor and per-group —
because the maximum of $n$ Gaussians grows like $\sqrt{\ln n}$, which is close
to not growing at all.

That last point is the interesting one. If weights really were Gaussian,
per-tensor quantization would be almost fine. It is not fine, and the reason is
that weights are not Gaussian.

## Why per-tensor scaling collapses

Trained weight matrices contain a small number of entries far outside the bulk
distribution. One of them sets $\beta$ for everything that shares its scale.

Take the lab's outlier case: a `(512, 1024)` matrix of standard normals with a
single entry set to 500. Per-tensor:

$$
s = \frac{500}{127} = 3.937,
\qquad \frac{s}{2} = 1.969
$$

Every weight with $|w| < 1.969$ rounds to the integer 0 and dequantizes to
exactly zero. For a standard normal that is

$$
P(|w| < 1.969) = 2\Phi(1.969) - 1 = 0.951
$$

95% of the matrix is replaced by zeros. The remaining 5% is quantized to $\pm 1$
or $\pm 2$, which is three or four distinct levels for the entire distribution.
Carrying the arithmetic through — integrating $w^2$ over the region that rounds
to zero, and dividing by the total squared norm, which the 500 itself inflates —
gives a relative error near 0.71. Per group of 128, only the one group holding
the outlier is damaged, and the relative error stays near 0.013. The ratio is
about 56.

The lab asserts a ratio above 5 and measures the real values. The point of the
arithmetic is that you can predict the collapse rather than discover it.

Notice the mechanism precisely. The outlier does not introduce error by being
badly represented — it is represented *exactly*, since it sets the endpoint. It
introduces error by stretching $s$, which is shared. **Quantization error is a
property of the group, not of the weight.**

## Per-channel and group-wise scaling

The fix follows directly: shrink the set of weights that share a scale.

**Per channel.** One scale per output row. In a linear layer each output row
produces one output feature independently, so a row with an outlier only damages
its own output. Scale shape `(out_features, 1)`.

**Per group.** Split the input dimension into contiguous blocks of $g$ weights
and give each block its own scale. Scale shape
`(out_features, in_features // g)`. This is the standard choice and $g = 128$
is the standard size.

The reshape is the whole implementation:

```python
def quantize_per_group(weight, group_size=128):
    out, inp = weight.shape                                # (512, 1024)
    grouped = weight.reshape(out, inp // group_size, group_size)   # (512, 8, 128)
    scale = grouped.abs().amax(dim=-1, keepdim=True) / 127.0       # (512, 8, 1)
    q = (grouped / scale).round().clamp(-128, 127).to(torch.int8)
    return q.reshape(out, inp), scale.squeeze(-1)          # (512, 1024), (512, 8)
```

The shape that trips people up is the `keepdim=True` on the amax. Without it the
scale is `(512, 8)` and broadcasting against `(512, 8, 128)` aligns the group
axis with the within-group axis, which silently produces garbage rather than an
error.

### The metadata arithmetic for a group size of 128

Storing a scale costs bytes too. With int8 weights and 2-byte scales, the bytes
per weight are

$$
b(g) = 1 + \frac{2}{g}
$$

At $g = 128$:

$$
b(128) = 1 + \frac{2}{128} = 1.015625 \text{ bytes per weight}
$$

against 2 bytes for bfloat16, so the stored size is

$$
\frac{1.015625}{2} = 50.78\%
$$

of bfloat16, a compression factor of $2/1.015625 = 1.969$. The scales cost
1.56% on top of the int8 weights. That is the number to weigh against the
quality gain, and it is small enough that going finer than 128 is rarely worth
it:

| Group size $g$ | Bytes per weight | Fraction of bf16 | Scale overhead |
|---|---|---|---|
| 32 | 1.0625 | 53.1% | 6.25% |
| 64 | 1.03125 | 51.6% | 3.13% |
| 128 | 1.015625 | 50.8% | 1.56% |
| Per channel, 5120 wide | 1.00039 | 50.0% | 0.04% |
| Per tensor | 1.0 | 50.0% | ~0% |

Check against the lab, which stores a `(512, 1024)` weight at $g = 128$:

$$
512 \times 1024 + 512 \times \frac{1024}{128} \times 2
= 524288 + 8192 = 532480 \text{ bytes}
$$

and $532480 / 1048576 = 50.78\%$, matching $b(128)/2$ exactly. The test checks
this stays under 55%.

For real weights rather than Gaussian ones, the measured quality gap between the
granularities is much wider than the $\sqrt{\ln n}$ argument predicts, because
of the outliers:

| Granularity | Typical weight error | Overhead |
|---|---|---|
| Per tensor | 2 to 5% | ~0% |
| Per channel | 0.5 to 1% | 0.02% |
| Per group of 128 | 0.1 to 0.3% | 1.5% |

## Weight-only against weight-and-activation

Two schemes, and which one helps depends entirely on which roof you are under.

**Weight-only (W8A16).** Store weights in int8. Read them, dequantize to
bfloat16 in registers, and do a bfloat16 matmul against bfloat16 activations.
The arithmetic never becomes integer arithmetic.

**Weight-and-activation (W8A8).** Quantize activations too, and do the matmul
with int8 inputs and int32 accumulation on the tensor cores.

### The decode arithmetic

A decode step at batch 1 reads every weight once and does two FLOPs per weight.
The bytes dominate, so the time is

$$
t_{\text{decode}} \approx \frac{\text{weight bytes}}{\text{bandwidth}}
$$

At bfloat16 the weights are 53.8 GB. Divide by the measured device-to-device
copy rate of 1275 GB/s:

$$
t_{\text{bf16}} = \frac{53.8 \times 10^9}{1275 \times 10^9} = 42.2 \text{ ms}
$$

In int8 with $g = 128$ the same weights take $53.8 \times (1.015625/2) =
27.3$ GB:

$$
t_{\text{int8}} = \frac{27.3 \times 10^9}{1275 \times 10^9} = 21.4 \text{ ms}
$$

a speedup of $42.2/21.4 = 1.97$. In tokens per second that is 23.7 against
46.7. **Halving the bytes halves the time, because nothing else was the
bottleneck.**

The remaining 1.6% of the gap from a clean factor of two is the scales.

### Dequantization is free

The obvious objection: you added work. Per weight, the kernel now reads 1 byte,
does 1 multiply to dequantize, and 2 FLOPs for the multiply-accumulate. That is
3 FLOPs per byte.

The ridge point is 161 FLOPs per byte. At 3, the kernel is memory bound by a
factor of 54, so the arithmetic finishes long before the next bytes arrive. The
dequantization costs nothing measurable. For comparison, the bfloat16 version
runs at 2 FLOPs per 2 bytes, or 1 FLOP per byte — also deep in memory-bound
territory. Quantization moved the kernel from 1 to 3 FLOPs per byte, which is
still nowhere near 161.

### Prefill gets nothing

Prefill at 2048 tokens has an arithmetic intensity around 2000, well past the
ridge point. It is compute bound, so its time is FLOPs divided by the tensor
core rate, and weight-only quantization does not change the FLOP count. The
matmul is still bfloat16.

Expect decode to roughly double and prefill to stay flat or get slightly worse,
because the dequantization adds a small amount of real work to a kernel that is
already compute bound.

The only way to speed up prefill is W8A8, which halves the FLOP cost by using
the integer tensor cores. That is a different and much harder problem, for the
reason the next two sections give.

## What Ampere's int8 tensor cores actually do

Ampere's tensor cores have an integer mode. The `mma` instruction takes int8
operands for both matrices, multiplies them, and accumulates into int32. The
published integer rate on the A100 is higher than its bfloat16 rate, which is
why W8A8 is attractive for prefill.

To use it, the matmul must be expressible entirely in integers. Write the
activation as $x_k = s_x q_{x,k}$ and the weight as $W_{kj} = s_{w,j} q_{w,kj}$
with a per-tensor activation scale and a per-output-channel weight scale. Then

$$
y_j = \sum_{k} x_k W_{kj}
= s_x\, s_{w,j} \sum_{k} q_{x,k}\, q_{w,kj}
$$

The sum on the right is a pure integer dot product — exactly what the tensor
core computes into its int32 accumulator. The scales come out of the sum because
neither depends on $k$. One float multiply per output element converts the
int32 accumulator back. That is the whole of W8A8.

**Accumulator headroom.** Each product is at most $127 \times 127 = 16129$, so
with $K = 5120$ terms the worst-case accumulator is $5120 \times 16129 =
8.26 \times 10^{7}$ against an int32 range of $2^{31} = 2.15 \times 10^{9}$. A
factor of 26 of headroom, so int32 accumulation cannot overflow at this model's
dimensions — a genuine advantage over float16 accumulation, which can and does.

**Why group-wise breaks it.** The derivation pulled $s_{w,j}$ out of the sum
because it did not depend on $k$. A group-wise scale does depend on $k$: it
changes every 128 steps. You would have to split the $K$ loop into chunks of
128, convert each int32 partial sum to float, scale it, and add — a conversion
and a float add every 128 accumulation steps, which defeats much of the point.

So the rule: **group-wise scaling is a weight-only technique.** W8A8 kernels use
per-tensor or per-channel scales, which is also why W8A8 needs the activation
outlier machinery in the next section and weight-only does not.

For this course, weight-only is the right choice. Decode is where the time goes,
decode is memory bound, and weight-only gets the full memory saving with none of
the activation difficulty.

## Ampere has no FP8

FP8 looks like the ideal format for this problem. It spends bits on an exponent,
so its precision is relative rather than absolute: a value ten times larger gets
the same number of significant digits, not ten times worse. The `e4m3` variant
has three mantissa bits, so with the implicit leading one it gives eight steps
per octave and a worst-case relative rounding error of $2^{-4} = 6.25\%$ at
every magnitude. An outlier costs an exponent increment, not the mantissa of
every other value in its block.

Hopper and later have tensor cores that do FP8 matmuls natively. The A100 is
Ampere, compute capability 8.0, and has no FP8 tensor core support at all. You
can store weights in an FP8 layout and unpack them in software for the bandwidth
saving, but there is no compute speedup — the arithmetic still runs in bfloat16,
and the unpack is more work than an int8 dequantization. On Ampere, int8 gives
the same bandwidth saving with better tooling and a faster unpack.

This is a hard constraint, worth knowing before you spend a day on a kernel that
cannot be fast. Check it in code rather than trusting the part number:

```python
major, minor = torch.cuda.get_device_capability()
has_fp8 = (major, minor) >= (8, 9)    # Ada and Hopper onward
```

The model publishes an official FP8 checkpoint, `Qwen/Qwen3.8-27B-FP8`, which is
useful on an H100 and not here.

## Activation outliers

Weight quantization is the easy half. Transformer activations contain outliers
up to 100 times the typical magnitude, concentrated in a small number of
channels, and those same channels are outliers for every token. That is why
activation quantization is much harder, and why weight-only schemes are the
common choice.

The structure is what makes it tractable: the outliers are *per channel*, not
scattered. Two methods exploit that.

**SmoothQuant** moves the difficulty from activations into weights. For a linear
layer $y = xW$, insert a diagonal matrix and its inverse:

$$
y = x W = \left(x \operatorname{diag}(\mathbf{c})^{-1}\right)
\left(\operatorname{diag}(\mathbf{c}) W\right)
$$

The product is unchanged. Choosing $c_k$ large for the channels with large
activations shrinks those activation channels and grows the corresponding weight
rows. With $c_k = (\max_t |x_{tk}|)^{\alpha} / (\max_j |W_{kj}|)^{1-\alpha}$ and
$\alpha$ around 0.5, the difficulty splits evenly and both tensors become
quantizable. The scaling folds into the preceding layer's norm weights at no
runtime cost.

**AWQ** observes that a small fraction of weight channels matter far more than
the rest, identifies them from activation statistics rather than weight
magnitudes, and protects them with a per-channel scale. The important weights
are the ones multiplying large activations, which you cannot see by looking at
the weights alone.

## Calibration

Weight-only quantization is data-free: the range of a weight tensor is a
property of the tensor. Activation quantization is not, because you cannot know
the range of an activation without running data through the model.

Calibration is the process of finding it. Run a few hundred representative
sequences, record per-channel activation statistics, and derive the scales.

The naive choice is the observed maximum, which is wrong for the same reason
per-tensor weight scaling is wrong: one outlier in the calibration set sets the
scale forever. The better choice is a clipping threshold $\tau$ that minimizes
reconstruction error,

$$
\tau^{\star} = \arg\min_{\tau}\;
\mathbb{E}\left[\left(Q_{\tau}(x) - x\right)^2\right]
$$

where $Q_{\tau}$ clamps to $[-\tau, \tau]$ and then quantizes with
$s = \tau/127$. Raising $\tau$ reduces clipping error and raises rounding error,
because $s$ grows with $\tau$. The minimum sits where those two curves cross,
which for heavy-tailed activations is well below the observed maximum.

The calibration set matters: calibrate on English prose, serve code, and the
scales will be wrong in a way no perplexity check on English reveals.

## Quantizing the KV cache

The cache can be quantized too, and it matters at long context and large batch,
where the cache read starts to rival the weight read.

The arithmetic first. This model spends 64 KiB per token on KV cache:

$$
16 \text{ layers} \times 4 \text{ KV heads} \times 256 \text{ head dim}
\times 2 \text{ tensors} \times 2 \text{ bytes} = 65536 \text{ bytes}
$$

Quantizing to int8 with one scale per key or value vector — that is, per layer,
per head, per token — gives 32 KiB of data plus

$$
16 \times 4 \times 2 \times 2 = 256 \text{ bytes}
$$

of scales, for 32.25 KiB per token and a compression factor of 1.98. At 32k
context that takes a sequence's cache from 2.0 GiB to 1.008 GiB.

### Why keys tolerate it worse than values

The two tensors enter the computation at different places, and the error
propagates differently.

**Keys feed a softmax, so their error is exponentiated.** The score for key $j$
is $\ell_j = q^{\top} k_j / \sqrt{d}$. Perturb the key by $e_j$, with each
element an independent rounding residual of variance $s_k^2/12$:

$$
\Delta \ell_j = \frac{q^{\top} e_j}{\sqrt{d}},
\qquad
\operatorname{Var}(\Delta \ell_j)
= \frac{\lVert q \rVert^2 s_k^2}{12\,d}
$$

The attention weight is $a_j \propto e^{\ell_j}$, so an absolute score error
$\delta$ multiplies that token's unnormalized weight by $e^{\delta}$. A score
error of 0.1 changes the weight by 10.5%; an error of 0.3 changes it by 35%. The
error does not average out, because it changes the weights themselves before any
averaging happens.

Two consequences fall out of the variance formula. The error grows with
$\lVert q \rVert$, so confident queries — the ones whose attention is sharply
peaked and therefore matters most — suffer the most. And it grows with $s_k$,
which is set by the largest element of the key vector, so a single large key
component degrades every score that key participates in.

**Values are combined linearly, so their error averages out.** The output is
$o = \sum_j a_j v_j$ with $a_j \ge 0$ and $\sum_j a_j = 1$. Perturb each value by
an independent residual of variance $s_v^2/12$:

$$
\operatorname{Var}(\Delta o) = \frac{s_v^2}{12}\sum_j a_j^2
= \frac{s_v^2}{12\,n_{\text{eff}}},
\qquad
n_{\text{eff}} = \frac{1}{\sum_j a_j^2}
$$

The quantity $n_{\text{eff}}$ is the effective number of tokens the head is
attending to. Spread attention over 100 tokens roughly equally and
$n_{\text{eff}} \approx 100$, so the value error is suppressed by a factor of
10. In the worst case the head attends to a single token,
$n_{\text{eff}} = 1$, and the output error equals the quantization error of that
one value — no amplification, just no suppression either.

So values never amplify and usually shrink; keys always amplify. That asymmetry
is the reason for the standard configuration: keys in bfloat16 or with per-head
scales, values in int8. Quantizing both to int8 with per-tensor scales is the
configuration that quietly degrades long-context quality.

## The model at three precisions

The weights are 53.8 GB in bfloat16, which is $53.8 \times 10^9 / 2 =
26.9 \times 10^9$ parameters. Apply $b(g)$ with $g = 128$:

| Precision | Bytes per weight | Weight size | Decode floor at 1275 GB/s |
|---|---|---|---|
| bfloat16 | 2 | 53.8 GB | 42.2 ms |
| int8, $g = 128$ | 1.015625 | 27.3 GB | 21.4 ms |
| int4, $g = 128$ | 0.515625 | 13.9 GB | 10.9 ms |

The int4 row is $26.9 \times 10^9 \times (0.5 + 2/128) = 13.87 \times 10^9$
bytes. The speedups against bfloat16 are 1.97 and 3.87.

The memory freed matters as much as the speedup. Moving to int8 returns
$53.8 - 27.3 = 26.5$ GB of HBM. At 64 KiB per token that is

$$
\frac{26.5 \times 10^{9}}{65536} \approx 404{,}000
$$

additional tokens of KV cache, which at a batch of 32 is roughly 12,600 more
tokens of context per sequence. On a card where the model already leaves only
about 19 GiB free, that is the difference between serving short chats and
serving documents.

int4 is not free in quality the way int8 nearly is. Symmetric int4 has levels
$[-7, 7]$, so at the same group size the step is

$$
s_{\text{int4}} = \frac{\beta}{7} = \frac{127}{7}\,s_{\text{int8}}
= 18.1\; s_{\text{int8}}
$$

and by the error bound the reconstruction error grows by the same factor: the
$\sqrt{2\ln(2n)}$ estimate at $g = 128$ goes from 0.76% to about 13.7%. That is
why int4 is where weight-only schemes stop working with plain round-to-nearest
and start needing the AWQ or GPTQ machinery, which spends a calibration pass
choosing rounding directions that minimize the *output* error rather than the
weight error.

## Measuring the damage

Weight error is a proxy. What matters is the model's output. Three checks, in
increasing order of cost and usefulness:

1. **Layer output error.** Run one layer with and without quantization on the
   same input and compare. Fast, and it localizes the damage. This is what the
   lab checks last: `x @ dequantize(q, s, 128).T` against `x @ weight.T`, with a
   1% threshold on the relative norm of the difference.
2. **Logit correlation.** A full forward pass on a few prompts. Correlation
   should stay above 0.999 and top-1 agreement above 0.98.
3. **Perplexity on held-out text.** The real measure. An increase of more than
   about 1% means the scheme is too aggressive.

Doing only the first is how quantization schemes that look fine ship broken.

It is worth knowing how the weight error and the layer output error relate. For
$y = xW$ with a weight perturbation $\Delta W$ whose elements are independent of
$x$ and of each other,

$$
\operatorname{Var}(\Delta y_j) = \lVert x \rVert^2 \sigma_{\text{err}}^2,
\qquad
\operatorname{Var}(y_j) = \lVert x \rVert^2 \sigma_{W}^2
$$

so the relative output error equals the relative weight error. The $K$ terms in
the sum do not average the error away, because they also carry the signal. That
is why the lab uses the same 1% threshold for both.

The equality fails exactly when the perturbation correlates with the input —
when the weights you rounded badly are the ones multiplying the largest
activations. That correlation is what AWQ measures and what makes the output
error worse than the weight error on real data.

## What goes wrong

**Dequantizing the whole weight tensor and then calling a matmul.** The most
common mistake, and it undoes the entire benefit: you write 53.8 GB of bfloat16
back to HBM and read it again. Dequantization must happen inside the matmul
kernel, in registers. Compare the measured time against the roofline — if it
matches the bfloat16 figure, this is what happened.

**Forgetting `keepdim=True` on the amax.** The scale broadcasts along the wrong
axis. There is no shape error, because the sizes happen to be compatible; the
weights come back wrong. Print the scale's shape and check it against the
table above.

**Clamping before rounding.** Loses one level at each end. The symptom is a
slightly larger error than the bound predicts, with no other sign.

**Quantizing in float16 instead of float32.** The division `weight / scale` in
float16 loses precision before the rounding happens, so the error exceeds its own
bound. Do the arithmetic in float32 and cast at the end. The same applies to the
stored scale: a bfloat16 scale has 8 mantissa bits and carries a 0.4% relative
error that multiplies every weight in its group.

**Calibrating on the wrong data.** Activation scales fitted to one domain fail
silently on another. There is no runtime symptom, only worse outputs.

## Check your understanding

**Why does quantizing weights to int8 roughly double decode speed but do nothing
for prefill?**

Decode at batch 1 has an arithmetic intensity near 1 FLOP per byte, far below
the ridge point of 161, so its time is bytes over bandwidth and halving the
bytes halves the time. Prefill at 2048 tokens has an intensity near 2000, above
the ridge point, so its time is FLOPs over the tensor core rate. Weight-only
quantization changes bytes, not FLOPs — and makes prefill slightly worse, since
dequantization adds FLOPs to a kernel that is already compute bound.

**A group size of 128 costs 1.56% in metadata. Why not use 32 and get better
quality?**

You can, and the error does improve, but the arithmetic says the return is
small and the cost is not. Going from $g = 128$ to $g = 32$ takes the overhead
from 1.56% to 6.25%, so the stored size goes from 50.8% to 53.1% of bfloat16 —
you give back 4.5% of the decode speedup. The error, by the
$\sqrt{2\ln(2n)}$ argument, improves only from 0.76% to 0.66%. Four times the
metadata for a 13% reduction in error is a poor trade, which is why 128 is the
standard.

**You quantize both keys and values to int8 and long-context quality drops while
short-context quality is fine. Which tensor is at fault and why?**

Keys. Key error enters the softmax exponent, so an absolute score error $\delta$
scales that token's attention weight by $e^{\delta}$, and the error grows with
$\lVert q \rVert$. At long context there are more keys competing, so a
perturbation that reorders the top few of them changes the output substantially.
Value error is a convex combination and is suppressed by
$\sqrt{n_{\text{eff}}}$, which gets *better* at long context, not worse. Put
keys back in bfloat16 and re-measure.

## Lab

Implement symmetric int8 quantization at three granularities —
`quantize_per_tensor`, `quantize_per_channel` returning a `(out, 1)` scale, and
`quantize_per_group` returning a `(out, in // g)` scale — plus `dequantize` that
broadcasts a group scale back across its group, `relative_error` as the
Frobenius norm ratio, and `storage_bytes` for the int8 data plus its scales.

The harness checks that quantized values stay inside the int8 range and that the
largest weight maps near 127, that finer granularity gives lower error, that
group-wise error stays under 1%, and that a single outlier of 500 hurts a
per-tensor scale more than five times as much as a per-group one. It then checks
that `storage_bytes` matches $\text{out} \times \text{in} + \text{out} \times
(\text{in}/g) \times 2$ and comes in under 55% of bfloat16, and finally runs a
real matmul against the dequantized weights and requires the layer output error
to stay under 1%.

## Further reading

- [SmoothQuant: accurate and efficient post-training quantization for large language models](https://arxiv.org/abs/2211.10438)
- [AWQ: activation-aware weight quantization for LLM compression and acceleration](https://arxiv.org/abs/2306.00978)
- [GPTQ: accurate post-training quantization for generative pre-trained transformers](https://arxiv.org/abs/2210.17323)
- [LLM.int8(): 8-bit matrix multiplication for transformers at scale](https://arxiv.org/abs/2208.07339) — the paper that identified the activation outlier channels.
- [Quantization and training of neural networks for efficient integer-arithmetic-only inference](https://arxiv.org/abs/1712.05877) — the original derivation of the affine scheme used here.
- [KIVI: a tuning-free asymmetric 2bit quantization for KV cache](https://arxiv.org/abs/2402.02750)
