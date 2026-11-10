---
title: Int4 weights and KV quantization
slug: 18a-int4-and-kv-quantization
part: "Part 6 — Scaling"
summary: Pack int4, compensate quantization with GPTQ and AWQ, and quantize keys and values on different axes.
minutes: 100
gpu: true
objectives:
  - "Pack signed int4 codes into int32 words and unpack those words to the same integers."
  - "Keep group-wise int4 reconstruction inside half a step, and count 0.5 bytes plus the scale."
  - "Run the GPTQ column update and show it beats round-to-nearest on correlated inputs."
  - "Search an AWQ per-channel scale from activation magnitudes and match a brute-force grid."
  - "Quantize keys per channel and values per token, and compare the attention error."
  - "Round to FP8 E4M3 and E5M2 with ties to even, including when the float8 dtype is absent."
lab: 18a-int4-kv
---

# Int4 weights and KV quantization

> [!TLDR]
> - Eight signed int4 codes fill one int32 word. The first code is the low nibble, stored as `code + 8`. A group of 128 shares one 2-byte scale, so a weight costs $0.5 + 2/128 = 33/64$ bytes. [Chapter 13a](/c/13a-skinny-decode-gemm) places that layout on the ridge.
> - Round-to-nearest stays inside half a step of each weight. On correlated inputs the error that matters is the layer output. GPTQ's column update cuts that error from $0.041674$ to $0.015605$ on the lab's layer.
> - AWQ raises each input channel's activation magnitude to a power $\alpha$ and keeps the scale with the smallest output error. The lab's wide layer picks $\alpha = 0.75$.
> - Per-token int4 keys move a seeded attention output by $0.277654$. The same grid on values moves it by $0.079503$. Per-channel keys drop the key error to $0.071986$.
> - E4M3 holds at most $448$ and has no infinity. E5M2 holds at most $57344$ and overflows to infinity. The lab rounds ties to even in ordinary floats.

[Chapter 18](/c/18-quantization) derives the half-step bound and stops the lab at int8 round-to-nearest. It also says why int4 is where that rounding stops being enough, and why a key error is harder on attention than a value error. [Chapter 13a](/c/13a-skinny-decode-gemm) counts a W4A16 weight and grades a CPU matrix-vector unpack. Neither chapter builds the quantizer that produces the integers.

This chapter builds it. You pack eight codes into a word, quantize a group, and then stop minimizing the weight error. GPTQ spends the Hessian of the layer inputs to choose the rounding. AWQ spends the activation magnitudes to choose a per-channel scale.

The KV section measures the key-versus-value gap on one seeded head and changes the axis. FP8 is a cast you can run with no GPU.

## Before you start

**The symmetric int4 grid, from [chapter 18](/c/18-quantization).** The stored codes used by a symmetric scale run from $-7$ to $7$. The step is the group maximum divided by $7$. Reconstruction misses the original by at most half a step.

The nibble also has room for $-8$. Chapter 13a uses that code. A symmetric scale of $\mathrm{amax}/7$ never needs it, and the packer still has to round-trip it.

**The byte count, from [chapter 13a](/c/13a-skinny-decode-gemm).** A group of $128$ with one bfloat16 scale is $33/64$ bytes per weight. That chapter puts the up projection on the ridge and grades a byte-packed matrix-vector product. This chapter does not recompute that floor, and it does not repeat that product. The words here hold eight nibbles, not two.

**The layer output.** For a weight $W$ of shape $(\text{out}, \text{in})$ and inputs $X$ of shape $(\text{tokens}, \text{in})$, the product is $XW^{\mathsf T}$. Chapter 18 shows that the relative output error matches the relative weight error when the rounding is independent of $X$. Correlated columns of $X$ break that match. The error you then minimize is

$$
\frac{\lVert XW^{\mathsf T} - X\hat{W}^{\mathsf T} \rVert_F}{\lVert XW^{\mathsf T} \rVert_F}.
$$

**Keys and values.** A key error enters the softmax. A value error is mixed by the attention weights. Chapter 18 derives the two variances. This chapter measures them on a synthetic head that plants the outlier channel KIVI describes, and it quantizes the two tensors on different axes.

## Pack eight codes into one word

Eight signed codes share one int32 word, and the first code occupies the low nibble.

A code $q$ in $-8, \ldots, 7$ is stored as the nibble $q + 8$, which is an integer in $0, \ldots, 15$. Element $8t + i$ occupies bits $4i$ through $4i + 3$ of word $t$. Index $i = 0$ is the low nibble. A length that is not a multiple of $8$ leaves the high nibbles of the last word at $0$.

Nibble $0$ is the encoding of code $-8$. A pad value of $0$, followed by adding $8$, writes an $8$ into the tail and fails the unpack.

Take the row $(1, -2, 7, 0, -8, 3, -1, 4)$. The nibbles are $9, 6, 15, 8, 0, 11, 7, 12$. The low byte is $9 + 16 \cdot 6 = 105$, which is `0x69`.

The whole word is `0xc7b08f69`. As a signed int32 that bit pattern is $-944730263$.

| Element | Code | Nibble | Bit shift |
|---|---|---|---|
| 0 | 1 | 9 | 0 |
| 1 | $-2$ | 6 | 4 |
| 2 | 7 | 15 | 8 |
| 3 | 0 | 8 | 12 |
| 4 | $-8$ | 0 | 16 |
| 5 | 3 | 11 | 20 |
| 6 | $-1$ | 7 | 24 |
| 7 | 4 | 12 | 28 |

> [!EXAMPLE] A short row leaves the high nibbles at zero
> The row $(1, -2, 7)$ uses one word. The nibbles are $9$, $6$, and $15$:
>
> $$
> 9 + 16 \cdot 6 + 256 \cdot 15 = 3945.
> $$
>
> The other five nibbles stay $0$. A second row $(-8, 4, 0)$ packs to $2240$. The lab checks both rows, and a length of $13$ that spans two words.

==The low nibble is the first code.== Swapping the two nibbles of a byte stores $6$ in the low nibble of the worked row, and the word is no longer $-944730263$.

Reading a code back is the same shift in reverse. Mask the word to 32 bits before the shift, because a negative int32 sign-extends when you widen it:

```python
word = int(packed) & 0xFFFFFFFF
nibble = (word >> (4 * i)) & 0xF
code = nibble - 8
```

## A group scale stays inside half a step

Group-wise symmetric int4 misses each weight by at most half a step, and the bytes are the payload plus one scale per group.

Split the last dimension into contiguous groups. The last group can be short. For a group with maximum absolute value $\beta$, the scale and the code are

$$
\hla{s} = \frac{\beta}{7},
\qquad
q = \mathrm{clip}\!\left(\mathrm{round}(w / \hla{s}),\, -7,\, 7\right),
\qquad
\hat{w} = \hla{s}\, q.
$$

A group of all zeros has $\hla{s} = 0$ and codes $0$. Every other code lands in $-7, \ldots, 7$, and the element that sets $\beta$ lands on $\pm 7$.

Round-to-nearest misses a real number by at most half a unit, so

$$
\boxed{\lvert \hat{w} - w \rvert \le \frac{\hla{s}}{2}}
$$

whenever the clip does not move a code. With $\hla{s} = \beta / 7$, every $w / \hla{s}$ already sits in $[-7, 7]$, so the clip is idle except for a float that rounds onto the boundary. The lab allows a relative slack of $10^{-4}$ on that bound, which is much smaller than one code.

The storage is half a byte of payload, plus $2$ bytes for each scale:

$$
\boxed{\frac{1}{2} + \frac{2\,\lceil n / g \rceil}{n}}
$$

bytes per weight. When $g$ divides $n$, the ceiling drops out:

$$
\frac{1}{2} + \frac{2}{g}.
$$

For $g = 128$ that is $64/128 + 2/128 = 66/128 = 33/64 = 0.515625$. Chapter 13a uses this count for the up projection. A short tail still pays for a whole scale. Ten weights in groups of $4$ need three scales, which is $0.5 + 6/10 = 1.1$ bytes per weight.

The lab checks $0.515625$ on a full group of $128$, and $1.1$ on a tail of length $10$.

## Correlated inputs punish independent rounding

Round-to-nearest minimizes each weight on its own, and that is the wrong target once two input columns carry the same signal.

Suppose two columns of $X$ are nearly copies. The output depends on a weighted sum of the matching weights, not on either weight alone. Rounding each weight to the nearest code can push both of them the same way. The sum then misses by about a step, even though each weight missed by at most half a step.

A correction that moves the second weight the other way, before you round it, can put the sum back on the grid the inputs actually see.

The Hessian of $\lVert XW^{\mathsf T} - X\hat{W}^{\mathsf T} \rVert_F^2$ with respect to one output row is the Gram matrix $H = X^{\mathsf T} X$. Off-diagonal entries are large exactly when columns of $X$ move together. GPTQ uses those entries. Round-to-nearest ignores them.

> [!KEY] The error that matters is the layer output
> A code can move farther from its weight and still cut the output error, if the move cancels a residual the inputs cannot tell apart from another column.

The paper's published grids are asymmetric and per row. The lab uses the symmetric group grid from chapter 18, so GPTQ and round-to-nearest share one step size. The only difference is the update.

## Compensate with the GPTQ column update

GPTQ quantizes columns from left to right and subtracts each column's rounding residual from the columns it has not touched yet.

Frantar, Ashkboos, Hoefler, and Alistarh build the method on Optimal Brain Quantization. OBQ picks the next weight by how cheap it is to round, and it updates the inverse Hessian after every weight. GPTQ's observation is that a fixed order is nearly as accurate on a wide layer, and a fixed order lets every output row share one inverse. The paper then quantizes columns in blocks, and it replaces the repeated inverse update with one Cholesky factor.

You pass $H = X^{\mathsf T} X$. The paper writes $2XX^{\mathsf T}$. Damping is $1\%$ of the mean diagonal of whichever matrix you form, so a constant factor on $H$ cancels in the update ratio. Add that damping, invert, and take the upper Cholesky factor $U$:

$$
H \leftarrow H + 0.01 \cdot \mathrm{mean}(\mathrm{diag}\, H) \cdot I,
\qquad
U = \mathrm{chol}(H^{-1})^{\mathsf T}.
$$

$U$ is what Algorithm 1 stores. Its diagonal is not the diagonal of the raw inverse. Use $U$ in the update, not a second copy of $H^{-1}$.

For each group of columns, read the scale from the weights as they stand at the start of the group, one scale per output row. Then, for column $j$ inside the group, quantize and push the residual forward:

$$
\hat{w}_j = \hla{s}\, \mathrm{clip}(\mathrm{round}(w_j / \hla{s}), -7, 7),
\qquad
\hlb{e} = \frac{w_j - \hat{w}_j}{U_{jj}},
$$

$$
W_{:,\, j:} \leftarrow W_{:,\, j:} - \hlb{e}\, U_{j,\, j:}.
$$

After the group, apply the same residuals to the columns past the group. Later groups therefore see weights that have already absorbed the earlier residuals, and their scales are computed from those updated weights.

The first group, before any update, matches round-to-nearest. Skipping the subtraction leaves every later column on that same grid. The result is round-to-nearest.

> [!EXAMPLE] One row, three weights
> Take $w = (0.20, 0.10, 0.40)$ and $64$ input rows that are $0.9$ times one shared column plus $0.1$ times independent noise, drawn in `float64` with seed $0$. The group scale is $0.40 / 7$. Round-to-nearest stores codes $(4, 2, 7)$, with relative output error $0.0620$. The update stores $(4, 1, 7)$, and the output error falls to $0.0197$.
>
> Code $2$ dequantizes to $0.8/7 \approx 0.1143$, which is closer to $0.10$ than code $1$ at $0.4/7 \approx 0.0571$. The middle weight gets worse. The output gets better, because the first column had been rounded up and the inputs cannot separate the two columns. `scripts/figures/18a-int4-and-kv-quantization.py` prints both errors.

The same script sweeps a wider layer: $128$ rows, $32$ inputs, $4$ outputs, group size $8$, `float32` draws, seed $18$. A correlation $\rho$ mixes one shared column into the inputs. Round-to-nearest has the higher error at every $\rho$. The gap widens as the columns become redundant.

| Correlation | Round-to-nearest | GPTQ | Margin |
|---|---|---|---|
| 0.00 | 0.081093 | 0.075025 | 0.006068 |
| 0.25 | 0.071020 | 0.062835 | 0.008184 |
| 0.50 | 0.060018 | 0.047501 | 0.012517 |
| 0.75 | 0.049697 | 0.032426 | 0.017271 |
| 0.90 | 0.043692 | 0.020280 | 0.023412 |
| 0.95 | 0.041674 | 0.015605 | 0.026069 |
| 0.99 | 0.039994 | 0.011045 | 0.028950 |

```viz
18a-gptq-margin
```

The lab grades $\rho = 0.95$. GPTQ's relative output error there is $0.015605$, against $0.041674$ for round-to-nearest, a margin of $0.026069$. The check requires a margin of at least $0.01$, which is under half of $0.026069$, and it requires the GPTQ error to exceed $0.005$.

Returning the original weight scores $0$ and fails the floor. Returning round-to-nearest scores a margin of $0$ and fails the gap. On OPT-175B, the paper reports that GPTQ loses $0.03$ WikiText-2 perplexity at $4$ bits, while round-to-nearest drops $2.2$ points. That is their model and their calibration set, not this one.

> [!DEEPDIVE] Why the Cholesky factor replaces the running inverse
> OBQ deletes one row and column of $H^{-1}$ after every weight, by a rank-one correction. On a layer with thousands of columns those corrections accumulate, and the paper reports that $H^{-1}$ can stop being positive definite. The only entries the update needs, at the moment it quantizes column $j$, are $U_{j, j:}$. Those entries are exactly what one Cholesky factor of the inverse contains. Computing it once, after damping, is Algorithm 1. On the lab's $32 \times 32$ Gram matrix the damping alone is enough to keep the factor finite. The lab still uses the factor, so the update matches the algorithm rather than a raw inverse that happens to be stable at this size.

## Search the scale from the activations

AWQ protects the input channels that meet large activations, by scaling those columns up before the group quantizer runs.

Lin and coauthors start from a hardware constraint. Keeping one percent of salient weights in 16 bits cuts the error, but a mixed-precision matrix is a poor kernel. They instead multiply a weight column by $s > 1$ and divide the matching activation by $s$. The product is unchanged before quantization.

After quantization, a larger $s$ spends more of the shared group scale on that column, so its relative error shrinks, until $s$ grows the group maximum and hurts the other columns.

The search does not pick a separate $s$ per channel. The activation magnitude already ranks the channels. Let $m_k$ be the mean absolute value of input channel $k$, and search one exponent:

$$
s = \left(\frac{m}{\mathrm{mean}(m)}\right)^{\hlb{\alpha}},
\qquad
\hlb{\alpha} \in \{0, 0.25, 0.5, 0.75, 1\}.
$$

$\hlb{\alpha} = 0$ gives a vector of ones, which is round-to-nearest. $\hlb{\alpha} = 1$ is the most aggressive scale in the grid. For each candidate, scale the weight columns, run the group quantizer, divide the dequantized weight by $s$, and keep the $s$ with the smallest relative output error. A tie keeps the smaller $\hlb{\alpha}$.

The paper grids $\alpha$ over $[0, 1]$ and also clips weights. The lab searches the five-point grid and does not clip, so the brute-force check is the same function.

> [!KEY] The scale comes from the activation
> A large weight next to a tiny activation is cheap to round badly. A small weight next to a large activation is not. Looking at the weight matrix alone cannot see that.

The lab's wide layer has $48$ tokens, $6$ inputs, and $2$ outputs, drawn with seed $4$. Five channels have activation scale $0.2$. The last has scale $12$, and its weight column is multiplied by $0.2$ so it does not already own the group maximum. One group covers the row. The relative output errors are:

| $\alpha$ | Output error |
|---|---|
| 0.00 | 0.330230 |
| 0.25 | 0.139580 |
| 0.50 | 0.065805 |
| 0.75 | 0.036474 |
| 1.00 | 0.099206 |

$\alpha = 0.75$ has the smallest error, and the next grid point is $0.029$ higher, so a `float32` power and a `float64` power pick the same point. A second layer, seed $1$, four inputs, last channel scaled by $8$, picks $\alpha = 0.25$. Hard-coding $0.75$ fails that layer. The check compares your scale vector with $(m / \mathrm{mean}(m))^{\alpha}$ for the alpha the brute-force search selects.

## SmoothQuant moves the outlier the other way

SmoothQuant and AWQ write down the same identity, and they quantize different tensors afterward.

Chapter 18 already derives the SmoothQuant scale. For a layer $y = xW$, a diagonal $c$ cancels:

$$
y = \left(x\, \mathrm{diag}(c)^{-1}\right)\left(\mathrm{diag}(c)\, W\right),
\qquad
c_k = \frac{(\max_t \lvert x_{tk} \rvert)^{\hlb{\alpha}}}{(\max_j \lvert W_{kj} \rvert)^{1 - \hlb{\alpha}}}.
$$

Both the smoothed activation and the scaled weight are then quantized to int8. The weight term in the denominator keeps a large weight from absorbing the whole move. Xiao and coauthors report $\alpha = 0.5$ as the balanced point for OPT and BLOOM when both tensors share an int8 grid.

AWQ never quantizes the activation. The activation stays in 16 bits, so the search has no weight term to balance. That is why $s$ depends only on $m$.

Use SmoothQuant when you need an int8 activation for a tensor-core kernel. Use AWQ when the activation stays wide and only the weight is int4. They are not two names for one scale.

## Quantize keys and values on different axes

Per-token int4 is harder on keys than on values, and the fix is a different axis for each tensor.

KIVI measures KV caches from Llama-2, Mistral, and Falcon. A few key channels are large on every token. A value cache has no such channel. Per-channel key scales confine a large channel to its own step, so the ordinary channels keep a small step.

Per-token value scales confine each token's error to that token. Attention then mixes a few value rows, and a bad scale on an unused token stays there. Per-channel value scales do the opposite: one token's outlier sets the step for that channel across tokens the head actually reads.

The lab does not implement KIVI's streaming buffer. KIVI keeps a residual of recent tokens in full precision and quantizes older key groups along the sequence, because a per-channel scale needs many tokens and decode produces one token at a time. The lab quantizes a finished block, which is the axis choice without the buffer.

The measurement is a synthetic head, seed $18$, $32$ tokens, dimension $16$. Channel $0$ of the key is fixed at $12$. Token $7$ carries an extra $2.5$ on the other channels. The query is $4$ on channel $1$ and $0$ elsewhere, so it attends through a channel the outlier does not own. Symmetric int4, one scale per vector along the axis under test, gives these relative attention-output errors:

| Tensor | Axis | Relative attention error |
|---|---|---|
| Key | per token | 0.277654 |
| Key | per channel | 0.071986 |
| Value | per token | 0.079503 |
| Value | per channel | 0.147489 |

> [!EXAMPLE] The key gap is a factor of 3.492
> Divide the per-token errors:
>
> $$
> \frac{0.277654}{0.079503} = 3.492.
> $$
>
> Per-channel keys fall to $0.071986$, which is under half of $0.277654$. Per-channel values rise to $0.147489$, which is $1.855$ times the per-token value error. The figure script prints all four.

==Per-token keys hurt more than per-token values.== That is the chapter 18 claim, on this seed. `quantize_kv` then does the KIVI assignment: per-channel keys, per-token values. The lab requires the per-token key error to exceed twice the per-token value error, the per-channel key error to stay under half the per-token key error, and the per-channel value error to exceed $1.5$ times the per-token value error. Swapping the axes fails the reconstruction against those two reductions, before the ratios are checked.

These four numbers are not a Llama accuracy. They are one head built so the outlier channel is obvious enough to grade. The direction matches the paper's measurements. The magnitudes are this seed.

## Round FP8 with ties to even

E4M3 and E5M2 are ordinary rounding problems, and an A100 still has no FP8 tensor core to run them on.

Micikevicius and coauthors specify two 8-bit encodings. E4M3 is the format they recommend for weights and activations. E5M2 is the format they recommend for gradients.

E5M2 follows IEEE specials. E4M3 spends the infinity codes on one extra binade, and it keeps a single NaN mantissa.

| | E4M3 | E5M2 |
|---|---|---|
| Exponent bias | 7 | 15 |
| Max finite | $1.75 \times 2^{8} = 448$ | $1.75 \times 2^{15} = 57344$ |
| Min normal | $2^{-6}$ | $2^{-14}$ |
| Min subnormal | $2^{-9}$ | $2^{-16}$ |
| Infinity | none | exponent field all ones, mantissa 0 |
| NaN | `S.1111.111` | exponent field all ones, mantissa not 0 |

The paper leaves the rounding mode to the implementation. This chapter uses round ties to even, which is what `torch.round` does, and what `torch.float8_e4m3fn` and `torch.float8_e5m2` do on the values the lab checks. The lab does not call those dtypes.

The same float function runs when the dtype is missing. The figure script prints the dtype cast of the lab's sample, and the function matches it.

In a binade $[2^{e}, 2^{e+1})$ the unit in the last place is $2^{e - M}$, where $M$ is the number of stored mantissa bits. Subnormals use the minimum positive code as the unit: $2^{-9}$ for E4M3 and $2^{-16}$ for E5M2. Round the magnitude to a multiple of that unit, ties toward the even integer multiple, then put the sign back.

> [!EXAMPLE] Two ties, and the overflow each format chooses
> E4M3 in $[1, 2)$ has a step of $0.125$. The midpoint $1.0625$ sits between $1$ (mantissa $0$, even) and $1.125$ (mantissa $1$, odd), so it rounds to $1$. The midpoint $1.1875$ sits between $1.125$ and $1.25$ (mantissa $2$, even), so it rounds to $1.25$.
>
> The highest E4M3 binade is $[256, 512)$, step $32$. The finite codes stop at $448$. The bit pattern at $480$ is the NaN. The midpoint $464$ ties to $448$, because $448 / 32 = 14$ is the even multiple. A value that rounds to $480$ or beyond is NaN, not $448$. Infinities are NaN too.
>
> E5M2 in $[1, 2)$ has a step of $0.25$. The midpoint $1.125$ ties to $1$. The max finite is $57344$, and $60000$ still rounds there. The midpoint $61440$ ties toward $65536$, which this format encodes as infinity.

==E4M3 has no infinity.== Clamping every overflow to $448$ keeps $480$ finite, and the lab rejects that. E5M2 does the IEEE thing: $61440$ and $65536$ are infinities, and a NaN stays a NaN.

A per-tensor software scale is how you aim a tensor at this range. The paper keeps the IEEE biases and puts the scale outside the encoding. For E4M3, dividing by $\max|x| / 448$ lands the largest magnitude on $448$. The lab grades the cast, not that scale. Choosing the scale badly is the same outlier problem as int8, which chapter 18 already covers.

A training or serving stack does not read $\max|x|$ from the current tensor each step. It keeps an amax history, the largest magnitude of the last several steps, and divides by the largest entry of that window over $448$. The window makes the scale stable, and it lets the scale be known before the matmul runs. A spike larger than the window still overflows, and E4M3 turns that overflow into NaN. The lab does not grade the history, and the figure script does not compute one.

Chapter 18 records the other limit. Ampere has no FP8 tensor core, so this cast does not speed up a matmul on the A100. It is how you check a format, and how you would store a tensor that a later GPU multiplies natively.

## Unpack inside the matmul, and only on CUDA

A W4A16 kernel unpacks the word in registers. This machine has no CUDA device, so the lab does not time one.

Chapter 18's warning still holds: writing the dequantized matrix out to memory and reading it back spends the bfloat16 traffic you removed. Chapter 13a already grades that unpack as a CPU matrix-vector product on a byte layout, two nibbles per byte.

This lab's `w4a16_matmul` is a different function. It reads the int32 words from `pack_int4`, eight codes per word, and it is a Triton kernel. The harness calls it only when `torch.cuda.is_available()` is true. The check then compares the kernel with a dequantized matmul.

The reference kernel in `solution.py` is deliberately plain. One program computes one output element, walks the $k$ rows of the weight, shifts the nibble out of the int32 word, subtracts $8$, and multiplies by the group scale:

```python
nibble = (packed_word >> shift) & 0xF
scale = tl.load(scale_ptr + (t // group) * stride_sg + col * stride_sn)
acc += activation * (nibble.to(tl.float32) - 8.0) * scale
```

The dequantized value lives in a register and never returns to memory. This version is a correctness reference, not a fast kernel. It has no tile, no vector load, and no tensor core, so it is not the kernel that beats bfloat16. A fast W4A16 kernel, such as Marlin, tiles the weight so that each word is read once for many outputs. It wins at a short batch because the $4$-bit weight is the traffic, and it loses at a long batch because the product is compute bound and the unpack is extra work. Chapter 13a computes where that crossing sits.

Without a device the check passes as a skip. This chapter reports no kernel time and no speedup.

The byte argument for running that kernel at a short batch is chapter 13a's. At a long batch the product is compute bound and the $4$-bit weight no longer sets the time. That crossing is computed there. It is not recomputed here.

## What goes wrong

These are the mistakes the lab is built to catch.

> [!WARNING] Skipping the Hessian update
> The first column of a group matches round-to-nearest either way. If you quantize the rest of the group and never subtract $\hlb{e}\, U_{j, j:}$, every later column matches too. The output error on the seeded layer stays at $0.041674$ instead of $0.015605$. The margin is $0$, and the check wants at least $0.01$.

**Swapping the nibbles.** Storing element $0$ in bits $4$ to $7$ writes $6$ into the low nibble of the worked row. The word is not $-944730263$. A pack and an unpack that swap in the same way still round-trip each other. The lab compares each of them with the low-nibble-first word, so the consistent swap fails both checks.

**Padding the tail with zeros before adding $8$.** Unused lanes must encode nibble $0$. A pad of integer $0$ becomes nibble $8$. The short row $(1, -2, 7)$ then disagrees with $3945$. Pad with $-8$, or add $8$ only on the real codes.

**Returning the original weight from GPTQ.** The output error is $0$, which is under the $0.005$ floor. The update has to quantize.

**Using one scale per token for keys.** On this seed that error is $0.277654$, against $0.071986$ for a per-channel key scale. `quantize_kv` is checked against those two reductions.

**Saturating E4M3 at $448$.** $464$ does round to $448$. $480$ does not. It is a NaN. E5M2's matching edge, $61440$, is an infinity, not $57344$.

## Check your understanding

> [!RECAP]
> - Eight codes fill an int32 word, low nibble first, stored as `code + 8`. Unused nibbles are $0$.
> - Symmetric group int4 uses $\hla{s} = \beta / 7$ and stays inside half a step. Bytes per weight are $0.5 + 2\lceil n/g \rceil / n$, which is $0.515625$ at group $128$.
> - GPTQ quantizes columns left to right and pushes each residual through the upper Cholesky factor of the damped inverse Hessian. On the lab's layer the output error falls from $0.041674$ to $0.015605$.
> - AWQ searches $\hlb{\alpha}$ in $\{0, 0.25, 0.5, 0.75, 1\}$ on scales built from activation magnitudes. The wide lab layer picks $0.75$.
> - Per-token key error on the seeded head is $0.277654$, against $0.079503$ for per-token values. Per-channel keys and per-token values are the assignment `quantize_kv` uses.
> - E4M3 tops out at $448$ and turns a rounded magnitude past that into NaN. E5M2 tops out at $57344$ and overflows to infinity. Ties round to even.

> [!QUESTION] The symmetric scale never stores -8. Why does the packer still have to round-trip it?
> The nibble has sixteen patterns, and chapter 13a's unpack reads all of them, including nibble $0$ as code $-8$. A symmetric scale of $\beta / 7$ only emits $-7, \ldots, 7$. The packer is the layout, not the quantizer. A word with a high bit set is also a legal int32, so the unpack has to mask to 32 bits before it shifts. The worked word `0xc7b08f69` is negative when you read it as int32.

> [!QUESTION] In the three-weight example, code 2 is closer to 0.10 than code 1. Why does GPTQ store code 1?
> The output depends on $Xw$, and the two input columns are $0.9$ shared. Rounding the first weight from $0.20$ up to $1.6/7$ leaves a residual. The update pushes that residual onto the later weights before they are rounded. Code $1$ is farther from $0.10$ and closer to the sum the inputs can see. The relative output error falls from $0.0620$ to $0.0197$.

> [!QUESTION] What does alpha 0 mean in the AWQ search, and why is 0.75 the winner on the wide layer?
> $\alpha = 0$ raises the normalized activation magnitude to the power $0$, which is a vector of ones. The quantizer then sees the original weight, so $\alpha = 0$ is round-to-nearest. Its error on that layer is $0.330230$. $\alpha = 0.75$ brings the error down to $0.036474$, and $\alpha = 1$ lets the large channel grow the group scale far enough that the error rises again, to $0.099206$. The smallest error wins.

> [!QUESTION] Per-channel keys and per-token values have similar attention error, 0.072 and 0.080. Why does the chapter still call keys the sensitive tensor?
> The comparison that matches chapter 18 uses the same axis on both tensors. Per-token keys are at $0.277654$ and per-token values are at $0.079503$. Keys are sensitive under that shared grid. The per-channel key scale is the remedy, not evidence that keys were easy. Values get worse, not better, when you give them the key's axis: $0.147489$ against $0.079503$.

> [!QUESTION] Why does 464 stay at 448 in E4M3, while 480 becomes NaN?
> The top binade steps by $32$. $448 / 32 = 14$, an even multiple, and $480 / 32 = 15$, which is the NaN pattern rather than a finite code. $464$ is the exact midpoint, and ties round to the even multiple, so $464$ stays $448$. Anything that rounds to $15 \times 32$ or beyond has left the finite set. E5M2's midpoint $61440$ ties the other way, onto an infinity, because $65536 / 8192 = 8$ is the even multiple and that code is infinity.

## Lab

> [!TRY]
> Pack and unpack int4, quantize a group, run the GPTQ update and the AWQ search, quantize a key and a value, and cast to FP8. You pass when the worked word is $-944730263$, the group error stays inside half a step, GPTQ's margin on the seeded layer is at least $0.01$, the AWQ scales match the brute-force alphas $0.75$ and $0.25$, the seeded attention gaps hold, and the FP8 ties match the table in this chapter. `w4a16_matmul` is checked only when CUDA is available. Without a device that check skips.

Implement the following functions:

- `pack_int4` and `unpack_int4`. Eight codes per int32 word, low nibble first, stored as `code + 8`.
- `quantize_groupwise_int4`. Symmetric, scale $\beta / 7$, codes in $-7, \ldots, 7$, one scale per group including a short tail.
- `gptq_quantize`. Damped Hessian, upper Cholesky factor of the inverse, column update, symmetric group grid.
- `awq_scale_search`. Scales $(m / \mathrm{mean}(m))^{\alpha}$ on the five-point grid. The smallest output error wins.
- `quantize_kv`. Per-channel keys, per-token values, symmetric int4.
- `fp8_cast`. Round ties to even for `"e4m3"` and `"e5m2"`, with the overflow rules in this chapter.
- `w4a16_matmul`. Triton, int32 words, only invoked when CUDA is visible.

The first check calls only `pack_int4`. A correct pack passes that check before the other functions exist.

## Further reading

- [Chapter 18b, sparsity and pruning](/c/18b-sparsity-and-pruning). Removes weights instead of rounding them, and reuses the Hessian update from this chapter in SparseGPT.
- [GPTQ: accurate post-training quantization for generative pre-trained transformers](https://arxiv.org/abs/2210.17323). Algorithm 1 is the column update. The $0.03$ and $2.2$ perplexity figures are their OPT-175B result.
- [AWQ: activation-aware weight quantization for on-device LLM compression and acceleration](https://arxiv.org/abs/2306.00978). The scale is $s_X^{\alpha}$ with $\alpha$ searched on $[0, 1]$.
- [SmoothQuant: accurate and efficient post-training quantization for large language models](https://arxiv.org/abs/2211.10438). The per-channel scale that moves outliers from activations into weights.
- [KIVI: a tuning-free asymmetric 2bit quantization for KV cache](https://arxiv.org/abs/2402.02750). Per-channel keys, per-token values.
- [FP8 formats for deep learning](https://arxiv.org/abs/2209.05433). E4M3 and E5M2, including the maxima $448$ and $57344$.
- [MIT 6.5940, TinyML and Efficient Deep Learning Computing, Fall 2024](https://hanlab.mit.edu/courses/2024-fall-65940). Lecture 5, Quantization (Part I), 19 September. Lecture 6, Quantization (Part II), 24 September. Lecture 13, Efficient LLM Deployment, 22 October. The syllabus lists titles and dates. This chapter does not claim what each lecture covers beyond its title.
