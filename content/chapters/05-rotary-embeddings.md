---
title: Rotary position embeddings
slug: 05-rotary-embeddings
part: "Part 2 — A forward pass"
summary: RoPE derived from the requirement that attention scores depend only on relative position, then the partial and multimodal variants this model uses.
minutes: 80
gpu: true
objectives:
  - State the relative-position requirement and derive the two-dimensional rotation that satisfies it.
  - Extend the solution to a full head as a block-diagonal rotation with geometric frequencies.
  - Explain what rope_theta controls, and why long-context models raise it.
  - Implement partial rotary embeddings and say geometrically what the unrotated channels do.
  - Explain how mRoPE assigns different position coordinates to different channel groups.
lab: 05-rope
---

# Rotary position embeddings

> [!TLDR]
> - Attention can't see word order on its own. RoPE fixes that by rotating each
>   query and key by an angle proportional to its position, so the score
>   between two tokens depends only on how far apart they are.
> - Each head is split into pairs of channels that rotate at different speeds,
>   like the hands of a clock. `rope_theta` sets how slow the slowest hand
>   turns, and long-context models raise it so that hand doesn't lap itself.
> - This model rotates only the first 64 of each head's 256 channels. The other
>   192 give each head a content-matching score that doesn't fade with
>   distance.
> - mRoPE gives image patches three position coordinates. For text all three are
>   equal, so it's exactly standard RoPE.

Attention has no idea what order its inputs are in. Permute the tokens of a
sequence and the attention output permutes with them, identically. "The cat sat
on the mat" and "the mat sat on the cat" produce the same set of outputs,
reordered. Position has to be injected deliberately.

*Rotary position embeddings* (RoPE) do it by rotating each query and key vector
by an angle proportional to its position. When a query at position $m$ meets a
key at position $n$, the absolute angles cancel and only the gap $m - n$
survives in the score.

This chapter derives that rotation from the requirement, one dimension pair at a
time. Then it covers the two things this model does differently: it rotates only
a quarter of each head, and it carries three position coordinates instead of
one.

## Before you start

**Why attention is permutation invariant.** The attention output for query $i$
is a weighted sum over values, with weights
$\operatorname{softmax}_j(q_i \cdot k_j / \sqrt{d})$. Nothing in that expression
refers to $i$ or $j$ except through the vectors themselves, so reordering the
keys and values together leaves the sum unchanged. The causal mask decides
*which* tokens are visible, but says nothing about how far apart they are.

**Rotation matrices.** A rotation in the plane by angle $\alpha$ is the
following matrix:

$$
R(\alpha) =
\begin{bmatrix}
\cos\alpha & -\sin\alpha \\
\sin\alpha & \cos\alpha
\end{bmatrix}
$$

This chapter uses three of its properties repeatedly. In one word, rotations are
orthogonal:

- They compose: $R(\alpha)R(\beta) = R(\alpha + \beta)$.
- The transpose is the inverse: $R(\alpha)^{\top} = R(-\alpha)$.
- They preserve length: $\lVert R(\alpha)x \rVert = \lVert x \rVert$.

**Complex numbers as 2D rotations.** Identify $x = (x_0, x_1) \in \mathbb{R}^2$
with $z = x_0 + i x_1$. Multiplying by $e^{i\alpha}$ rotates by $\alpha$, and the
real inner product is $\operatorname{Re}[z_x \overline{z_y}]$, where the bar is
complex conjugation. The complex form makes the derivation two lines; the matrix
form is what you implement.

**Where this sits in the block.** [Chapter 4](/c/04-rmsnorm-and-residuals)
covered the per-head RMSNorm on queries and keys. RoPE comes immediately after
it, and immediately before the key and value are written to the *KV cache*, the
store of past keys and values that the engine keeps so it never recomputes them:

```python
q, k, v, gate = self.project(x)          # q_norm and k_norm happen inside
q = apply_rotary_partial(q, cos, sin, self.rotary_dim)
k = apply_rotary_partial(k, cos, sin, self.rotary_dim)
if kv_cache is not None:
    k, v = kv_cache.append(layer_idx, k, v)
```

That ordering means ==the cache stores keys that are already rotated==, so a
cached key carries its absolute position baked in.
[Chapter 15](/c/15-paged-attention) comes back to this when it tries to reuse a
prefix.

## The requirement

Before any formula, this section pins down what "encode position" should mean,
because the right requirement makes the answer almost fall out.

Start with a picture. Imagine every vector as a clock hand, and every step along
the sequence turns the hand by the same small angle. A token at position 10 has
turned 10 steps; a token at position 15 has turned 15. On top of whatever angle
the two hands started with, the extra angle between them is 5 steps' worth, and
it's still 5 steps' worth if you move both tokens to positions 1010 and 1015. That's the property you want: the model sees
how far apart two tokens are, not where they sit.

Now say it precisely. Write $f(x, p)$ for the function that takes a vector and a
position and produces the vector attention uses. The score between a query at
$\hla{m}$ and a key at $\hlb{n}$ must depend on the two vectors and on the gap
$\hlc{m - n}$, and on nothing else:

$$
\langle f(q, \hla{m}), f(k, \hlb{n}) \rangle = g(q, k, \hlc{m - n})
$$

This is a strong constraint, and it rules out the obvious approaches. Adding a
learned per-position vector, as the original transformer's learned embeddings
do, gives a score with cross terms in $m$ and $n$ separately. Concatenating a
position feature does the same. The constraint asks for the absolute positions
to cancel ==exactly, for every query and key==.

## The two-dimensional solution

The clock-hand picture already contains the answer, and this section checks it
in two dimensions. With $d = 2$, a pure rotation satisfies the requirement, and
the 2D case turns out to be the whole problem. Treat $q$ and $k$ as complex
numbers and rotate each by its position times a fixed frequency $\theta$:

$$
f(x, p) = x\,e^{i p \theta}
$$

Take the real inner product. The conjugate of a product is the product of
conjugates, and $\overline{e^{i n \theta}} = e^{-i n \theta}$, so the two
exponentials merge:

$$
\begin{aligned}
\langle f(q, \hla{m}), f(k, \hlb{n}) \rangle
&= \operatorname{Re}\!\left[q e^{i \hla{m} \theta} \cdot \overline{k e^{i \hlb{n} \theta}}\right] \\
&= \boxed{\operatorname{Re}\!\left[q \bar{k}\, e^{i (\hlc{m - n}) \theta}\right]}
\end{aligned}
$$

The absolute positions have cancelled. What remains is a function of $q$, $k$,
and $\hlc{m - n}$, which is exactly the requirement.

### The same thing as matrices

The code works with real numbers, so it helps to see the same result in real
coordinates. There, $f(x, p) = R(p\theta)x$. Write the gap as
$\hlc{\delta} = m - n$. The two rotations combine into one rotation by the gap,
and expanding it gives the score in terms of the four coordinates:

$$
\begin{aligned}
\langle R(\hla{m}\theta)q,\; R(\hlb{n}\theta)k \rangle
&= (q_0 k_0 + q_1 k_1)\cos(\hlc{\delta}\theta) \\
&\quad + (q_0 k_1 - q_1 k_0)\sin(\hlc{\delta}\theta)
\end{aligned}
$$

> [!INTUITION]
> The score mixes two familiar quantities: the ordinary dot product of $q$ and
> $k$, and the signed area of the parallelogram they span. The cosine and sine
> of the gap set the mix. At $\hlc{\delta} = 0$ it's the plain dot product, and
> as the gap grows the mix rotates smoothly between the two. That's the entire
> mechanism.

> [!DEEPDIVE] Derive the matrix form step by step
> Move the first rotation across the inner product as a transpose, then use
> $R(\alpha)^{\top} = R(-\alpha)$ and compose the two rotations into
> $R(n\theta - m\theta)$:
>
> $$
> \langle R(m\theta)q,\; R(n\theta)k \rangle
> = q^{\top} R(m\theta)^{\top} R(n\theta)\, k
> = q^{\top} R\big((n - m)\theta\big)\, k
> $$
>
> With $\delta = m - n$, that's $q^{\top} R(-\delta\theta) k$. Multiplying out
> the 2-by-2 product gives the expansion in the body.

Two consequences are worth naming now:

- **The transform is orthogonal.** It doesn't change $\lVert q \rVert$ or
  $\lVert k \rVert$, so the per-head RMSNorm from chapter 4 survives it.
- **It has no learned parameters.** $f$ is a fixed function of position.

## Extending to a full head

A real head isn't 2 channels wide, so this section scales the 2D answer up. A
head is 256 channels. Split it into $d/2$ consecutive pairs, give pair $i$ its
own frequency $\theta_i$, and rotate each pair independently. In matrix form
that's a block-diagonal rotation:

$$
R_p =
\begin{bmatrix}
R(p\theta_0) & & & \\
& R(p\theta_1) & & \\
& & \ddots & \\
& & & R(p\theta_{d/2 - 1})
\end{bmatrix}
$$

A block-diagonal matrix of rotations is itself orthogonal and composes
blockwise, so $R_m^{\top} R_n = R_{n-m}$ holds for the whole head. The inner
product splits into independent per-pair contributions, where $q^{(i)}$ is the
2-vector of channels $2i$ and $2i+1$:

$$
\langle R_{\hla{m}} q, R_{\hlb{n}} k \rangle
= \sum_{i=0}^{d/2 - 1} \left\langle R(\hla{m}\theta_i) q^{(i)},\; R(\hlb{n}\theta_i) k^{(i)} \right\rangle
$$

Every term depends only on $\hlc{m - n}$, so the sum does too. The requirement
holds for ==a head of any width==.

### The frequencies

Why give each pair a different speed? Think of a clock again. The second hand
tells apart moments a few seconds apart, but it can't tell 1:00 from 2:00. The
hour hand can, but it barely moves in a few seconds. Together they cover both
scales. The pairs of a head work the same way: fast pairs resolve nearby
positions, slow pairs resolve distant ones.

A single hyperparameter $\Theta$, `rope_theta` in the config, sets the
frequencies as a geometric sequence:

$$
\theta_i = \Theta^{-2i/d}, \qquad i = 0, 1, \ldots, \tfrac{d}{2} - 1
$$

So $\theta_0 = 1$ always, and the frequencies decrease geometrically to
$\Theta^{-(d-2)/d} \approx 1/\Theta$. Here $d$ is the *rotary* dimension, 64 for
this model, so there are 32 frequencies and $\theta_i = \Theta^{-i/32}$.

A frequency is easiest to read as a wavelength, the number of token positions it
takes to complete one full turn:

$$
\lambda_i = \frac{2\pi}{\theta_i} = 2\pi\,\Theta^{2i/d}
$$

Pair 0 turns once every $2\pi \approx 6.3$ positions and encodes fine
distinctions: adjacent, two apart, three apart. The last pair turns slowly and
encodes coarse position.

## What $\Theta$ controls, and why long context raises it

This section answers the one tuning question RoPE poses: how large should
`rope_theta` be? $\Theta$ sets the *range* of wavelengths, from $2\pi$ at the
fast end to about $2\pi\Theta$ at the slow end. It doesn't change the fast end
at all.

The design constraint is at the slow end. If the slowest wavelength is shorter
than the context you intend to serve, the slowest pair wraps around, like an
hour hand that can't tell 1:00 from 13:00. Positions a full wavelength apart then
look identical to the one pair whose job was to tell far-apart positions apart.
With $d = 64$, the slowest pair is $i = 31$:

$$
\lambda_{31} = 2\pi\,\Theta^{31/32}
$$

The following table works out this wavelength for two values of $\Theta$:

| $\Theta$ | $\Theta^{31/32}$ | $\lambda_{31}$ | Turns over 262,144 positions |
|---|---|---|---|
| $10^4$ | about $7.50 \times 10^{3}$ | about $4.71 \times 10^{4}$ | about 5.6 |
| $10^7$ | about $6.04 \times 10^{6}$ | about $3.80 \times 10^{7}$ | about 0.0069 |

Read the two rows as follows:

- **At the classic $\Theta = 10{,}000$**, the slowest pair completes more than
  five full turns across a 262,144-token context. Position 1,000 and position
  48,000 land at nearly the same angle in that pair. Every faster pair has
  wrapped even more often, so the model loses its coarse sense of which is
  which.
- **At $\Theta = 10^{7}$**, the slowest pair covers less than 1% of a turn,
  about 2.5 degrees, over the entire context. It has become a monotone coarse
  position signal instead of a periodic one.

> [!KEY] Raising $\Theta$ is free at inference time, but not free in general
> That's why long-context models raise it. $\Theta$ appears only in the table
> you precompute, so it's the cheapest possible change to serve. But stretching
> the wavelengths spreads the same 32 pairs over a longer range, so nearby
> positions become slightly harder to tell apart in the slow pairs. A model
> trained at one $\Theta$ can't be served at another without degradation.

The literature on extending context after training, including position
interpolation, NTK-aware scaling, and YaRN, is all about doing this rescaling in
a way the trained model tolerates. This model was trained at its $\Theta$, so the
engine reads the config.

## Implementation

You have the maths. This section turns it into the two pieces of code the lab
asks for: a table of angles built once, and a rotation applied to every query
and key.

### Precompute the tables

The angles depend only on position, never on activations, so they're identical
for every layer, every head, and every request. Build them once:

```python
half = rotary_dim // 2
inv_freq = 1.0 / (
    theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
)
positions = torch.arange(max_position, device=device, dtype=torch.float32)
angles = torch.outer(positions, inv_freq)
return angles.cos().to(dtype), angles.sin().to(dtype)
```

The tensors have the following shapes:

| Tensor | Shape | Notes |
|---|---|---|
| `inv_freq` | `(32,)` | $\theta_i = \Theta^{-i/32}$, descending from 1. |
| `positions` | `(P,)` | float32, always. |
| `angles` | `(P, 32)` | Outer product: `angles[p, i]` is $p\,\theta_i$. |
| `cos`, `sin` | `(P, 32)` | Cast to the working dtype after the trig. |

`RotaryEmbedding` builds `P = 32768` by default before growing on demand. Each
table is then $32768 \times 32 \times 4 = 4.19$ MB, so 8.4 MB for both. Against
53.8 GB of weights that's nothing, and it saves two transcendental functions
per channel that you never compute again.

> [!WARNING] Compute the angles in float32
> The angle at position $p$ for pair 0 is $p$ radians, and bfloat16 near 100,000
> has an ulp, the gap between neighbouring representable values, of
> $2^{9} = 512$. Positions there round to the nearest multiple of 512, so a
> whole block of neighbouring positions collapses onto one value and becomes
> indistinguishable. Compute the angles in float32, take the cosine and sine,
> which land in $[-1, 1]$ where bfloat16 is comfortable, and cast after.

### Applying the rotation

The derivation paired channel $2i$ with channel $2i+1$, but implementations
overwhelmingly use a different pairing, the *half split*: within the rotary
block, pair channel $j$ with channel $j + d/2$.

```python
def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

rotated = x * cos_full + _rotate_half(x) * sin_full
```

To check that this is the rotation, let $a = x_j$ and $b = x_{j + 32}$, and let
$c = \cos(p\theta_j)$, $s = \sin(p\theta_j)$. Because `cos_full` is `cos`
concatenated with itself, channels $j$ and $j + 32$ both read frequency
$\theta_j$. The two outputs are then the following:

$$
\text{out}_j = a\,c + (-b)\,s = a c - b s
$$

$$
\text{out}_{j+32} = b\,c + a\,s = a s + b c
$$

That's exactly $R(p\theta_j)$ applied to $(a, b)$.

The half split is faster because both halves are contiguous slices, so `chunk`
and `cat` move coalesced blocks instead of gathering every other element. The
two pairings give identical scores, as long as a checkpoint is served with the
layout it was trained with.

> [!DEEPDIVE] Why the two pairings are interchangeable
> The half-split scheme is the interleaved scheme composed with a fixed
> permutation of the head's channels, the perfect shuffle. The same permutation
> is applied to $q$ and to $k$, and the rotation is block-diagonal in both
> layouts, so the inner product is identical. The permutation amounts to
> relabelling channels, and the learned q and k projections absorb the
> relabelling during training.

> [!WARNING] The pairing is a checkpoint contract, not a free choice
> Serve a checkpoint trained with the interleaved layout with the half-split
> kernel and no permutation of the projection weights, and it pairs the wrong
> channels and hands them the wrong frequencies. The model still runs and still
> produces fluent text. It has no idea what order the words are in.
> [Chapter 8](/c/08-the-forward-pass)'s end-to-end validation exists partly to
> catch this.

## Partial rotary

This section covers the first way this model departs from textbook RoPE: it
rotates only a quarter of each head. `partial_rotary_factor` is 0.25 and
`head_dim`, the width of one head, is 256, so:

$$
\text{rotary\_dim} = 256 \times 0.25 = 64
$$

Only the first 64 channels of each head rotate, in 32 pairs. The remaining 192
pass through untouched:

```python
rot, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
cos_full = torch.cat((cos, cos), dim=-1)[None, None, :, :].to(x.dtype)
sin_full = torch.cat((sin, sin), dim=-1)[None, None, :, :].to(x.dtype)
rotated = rot * cos_full + _rotate_half(rot) * sin_full
return torch.cat((rotated, passthrough), dim=-1)
```

The tensors have the following shapes, for the 24 query heads and the 4 key and
value (KV) heads:

| Tensor | Shape for `q` | Shape for `k` |
|---|---|---|
| `x` | `(batch, 24, seq, 256)` | `(batch, 4, seq, 256)` |
| `rot` | `(batch, 24, seq, 64)` | `(batch, 4, seq, 64)` |
| `passthrough` | `(batch, 24, seq, 192)` | `(batch, 4, seq, 192)` |
| `cos`, `sin` in | `(seq, 32)` | `(seq, 32)` |
| `cos_full`, `sin_full` | `(1, 1, seq, 64)` | `(1, 1, seq, 64)` |
| return | `(batch, 24, seq, 256)` | `(batch, 4, seq, 256)` |

The two leading singleton axes on `cos_full` let one table serve all 24 query
heads and all 4 KV heads by broadcasting. Queries and keys use the same table;
only the head count differs.

> [!TIP] Check the boundary against the config
> `ModelConfig.rotary_dim` computes `head_dim * partial_rotary_factor` and is 64
> here. Splitting at 128, half the head, is what you get if you assume the
> half-split trick and the partial factor are the same thing. It produces a
> model that runs, generates fluent text, and ignores word order.

### What the unrotated channels do, geometrically

Why would a model leave three quarters of each head unrotated? The score
answers it. The transform on a full head is now block-diagonal with two blocks:
a rotation on the first 64 channels and the identity on the last 192. Because
it's block-diagonal, the score splits cleanly into a rotated part and a
$\hld{\text{passthrough}}$ part:

$$
\begin{aligned}
\langle f(q, \hla{m}), f(k, \hlb{n}) \rangle
&= \underbrace{q_{\text{rot}}^{\top} R_{\hlb{n}-\hla{m}}\, k_{\text{rot}}}_{\text{depends on } \hlc{m - n}} \\
&\quad + \underbrace{\hld{q_{\text{pass}}^{\top} k_{\text{pass}}}}_{\text{no position at all}}
\end{aligned}
$$

Each head therefore computes a sum of two scores:

- A relative-position score over a 64-dimensional subspace.
- A pure content-match score, $\hld{q_{\text{pass}}^{\top} k_{\text{pass}}}$,
  over a 192-dimensional subspace.

The model chooses, per head and per direction, how much of its score to put in
each.

**Why that isn't obviously harmful.** Full RoPE forces every direction in the
head to be position-modulated, and the modulation attenuates with distance: sum
enough pairs at different frequencies and the position-dependent term decays as
the gap grows. That's a useful inductive bias for local syntax. It's an actively
unhelpful one for a head whose job is "find the definition of this term,
wherever it appeared, 80,000 tokens ago".

> [!KEY] The passthrough channels are a retrieval path that distance doesn't weaken
> Reserving three quarters of each head for a score that doesn't decay with
> distance is a strong architectural bet, and one reason this model handles
> long context well.

**What it doesn't save.** The cache is unchanged: you still store all 256
channels of every KV head, because the passthrough channels are part of the key.
The saving is arithmetic, and the arithmetic was never the bottleneck. Applying
the rotation costs about 3 FLOPs per channel, so rotating 64 channels instead of
256 across 28 heads saves about 16,000 FLOPs per token per full-attention layer.
That's noise next to the 2.54 GFLOP per token of `lm_head`, the output
projection onto the vocabulary. Treat partial rotary as
==an architectural choice, not an optimization==.

## Multimodal RoPE

This section covers the second departure: positions with more than one
coordinate. This model accepts images, and an image isn't a sequence. A patch at
row 7, column 12 of a frame has three coordinates, not one, and flattening them
into a single index throws away the two-dimensional structure the model needs.

*mRoPE* (multimodal RoPE) keeps the same rotation and changes where each pair
reads its angle from. `mrope_section` is `[11, 11, 10]`, which sums to 32, so it
splits the 32 frequency pairs into three sections. Each section takes its angle
from a different coordinate: time, height, and width.

```python
def apply_mrope_sections(cos, sin, sections):
    cos_parts, sin_parts, start = [], [], 0
    for axis, width in enumerate(sections):
        cos_parts.append(cos[axis, :, start : start + width])
        sin_parts.append(sin[axis, :, start : start + width])
        start += width
    return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)
```

The following table shows which pairs read which coordinate:

| Tensor | Shape | Meaning |
|---|---|---|
| `cos`, `sin` in | `(3, seq, 32)` | One plane per coordinate: t, h, w. |
| `cos[0, :, 0:11]` | `(seq, 11)` | Pairs 0–10 read the time coordinate. |
| `cos[1, :, 11:22]` | `(seq, 11)` | Pairs 11–21 read the height coordinate. |
| `cos[2, :, 22:32]` | `(seq, 10)` | Pairs 22–31 read the width coordinate. |
| return | `(seq, 32)` | Reassembled, ready for `apply_rotary_partial`. |

> [!KEY] For a text token, all three coordinates hold the same number
> Every plane of the input is then identical, so slicing different sections
> from different planes reassembles exactly the single-plane table. mRoPE
> reduces to standard RoPE with no special case and no approximation.

That's why a text-only engine, including this course's, can ignore the
distinction entirely and still be correct. For an image patch the coordinates
differ, and the patch carries its row and column alongside its place in the
sequence.

`mrope_interleaved` is true in this config. The flag selects a different mapping
from sections to channel pairs: sections distributed across the frequency range
rather than taken as contiguous blocks. `apply_mrope_sections` implements the
contiguous layout, which is the one that makes the mechanism legible. Under
either mapping the text path is identical, for the same reason.

## Positions during decode

The last question is which position numbers to pass in, and it has a trap.
*Prefill*, the pass that processes the whole prompt at once, passes positions
$0, 1, \ldots, n-1$. *Decode*, which generates one token per step, passes a
single position: the index of the token being generated, which is the current
cache length. One expression covers both phases:

```python
start = cache.length if cache is not None else 0
positions = torch.arange(start, start + seq, device=input_ids.device)
```

During prefill, `cache.length` is 0 and `seq` is the prompt length. During
decode, `cache.length` is however many tokens are already cached and `seq` is 1.

> [!WARNING] Passing position 0 on every decode step
> It's an easy mistake when the decode path is written separately from prefill.
> Every generated token then behaves as if it's the first token in the sequence,
> the relative gap to every cached key is wrong by a growing amount, and output
> degenerates into repetition after a few dozen tokens. It looks like a sampling
> bug. It isn't.

## What goes wrong

**Splitting at the wrong index.** Fluent text that ignores word order. Check
`rotary_dim` against the config, and check that channels past it come back
bitwise unchanged.

**Rotating the values.** Position belongs in the score, not the payload.
Rotating $v$ rotates the content the model retrieves, by an angle that depends
on where the content happened to sit. Only $q$ and $k$ are rotated.

**Rotating keys after appending them to the cache.** The cache already holds
rotated keys. Rotating again applies the angle twice, and the effective gap
becomes $2m - n$. Rotate, then append.

**Building the angle table in bfloat16.** Fine at position 100, wrong at
position 100,000, where the ulp is 512. Positions collapse into buckets and long
context quietly stops working. Compute in float32, cast the cosines.

**Forgetting to duplicate `cos` to the full rotary width.** The tables are
`(seq, 32)` and the half-split rotation needs `(seq, 64)`. Without the `cat` you
get a shape error if you're lucky, and a silent broadcast against the head axis
if you're not.

**Forgetting `.to(x.dtype)` on the tables.** float32 tables promote the bfloat16
query, the attention matmul falls off the tensor cores, and prefill slows down
several-fold with nothing in the logs.

> [!RECAP]
> - Rotating $q$ and $k$ by their positions makes the score depend only on
>   $m - n$, because the absolute angles cancel.
> - $\Theta$ sets the slowest wavelength, which must outlast the context.
> - Build the angle tables once, in float32, and keep the checkpoint's pairing.
> - Only 64 of 256 channels rotate; the other 192 add a position-free content
>   score, and the cache still stores all 256.
> - mRoPE reduces to standard RoPE for text. Decode positions start at the cache
>   length.

## Check your understanding

> [!QUESTION] Why can't you satisfy the relative-position requirement by adding a learned vector $p_m$ to the query and $p_n$ to the key?
> Expand the score:
>
> $$
> (q + p_m) \cdot (k + p_n)
> = q \cdot k + q \cdot p_n + p_m \cdot k + p_m \cdot p_n
> $$
>
> The middle two terms depend on $m$ and $n$ separately, not on their
> difference, so there's no function $g$ of $m - n$ that reproduces the score
> for every $q$ and $k$. Rotation works because it acts on $q$ and $k$ by an
> orthogonal map, and orthogonality is what makes the two absolute angles
> combine into one relative angle.

> [!QUESTION] The model rotates 64 of 256 channels. Does that make the KV cache smaller?
> No. All 256 channels of every KV head go into the cache, rotated or not,
> because the passthrough channels are part of the key and are needed to compute
> the score. Partial rotary saves a small amount of arithmetic and changes what
> the head can represent. Chapter 9's cache arithmetic is unaffected by it.

> [!QUESTION] A model trained with $\Theta = 10{,}000$ is served at 200,000 tokens of context. What breaks first?
> The slow pairs. At $\Theta = 10^4$ and 64 rotary channels the slowest
> wavelength is about 47,000 positions, so over 200,000 tokens it wraps more
> than four times. Distant positions become aliased onto each other and the
> model loses its coarse sense of where it is, while local ordering, carried by
> the fast pairs, still works. The symptom is a model that reads nearby text
> correctly and can't locate anything far away.

> [!QUESTION] Your implementation passes every shape and norm check, but the relative-position test fails: scores for gap 5 differ depending on where in the sequence the pair sits. What's the most likely cause?
> Query and key aren't being rotated with the same frequency assignment. Either
> one of them is using a differently built table, or the pairing differs between
> the two paths: half split for one and interleaved for the other. The absolute
> angles only cancel when both sides use the same block-diagonal $R$.

## Lab

> [!TRY]
> Implement three functions that build the angle tables and rotate part of each
> head. You pass when a query and key at gap 5 score the same wherever they sit.
> The lab runs on GPU but is device agnostic, so the maths is checkable anywhere.

The three functions and their checks are as follows.

**`build_rope_cache(rotary_dim, max_position, theta, device, dtype)`** returns
`cos` and `sin` of shape `(max_position, rotary_dim // 2)`, with
$\theta_i = \Theta^{-2i/d}$. The harness checks the shape, that position 0 gives
$\cos = 1$ and $\sin = 0$, that $\cos^2 + \sin^2 = 1$ everywhere, and that the
first frequency rotates faster than the last.

**`rotate_half(x)`** maps $(x_1, x_2)$ to $(-x_2, x_1)$ over the last axis,
split in halves.

**`apply_rotary_partial(x, cos, sin, rotary_dim)`** rotates the first
`rotary_dim` channels of a `(batch, heads, seq, head_dim)` tensor and passes the
rest through. The harness runs it with `head_dim = 256`, `rotary_dim = 64`, and
$\Theta = 10^{7}$. It checks the following:

- The shape survives.
- Channels past the boundary are unchanged, to within $10^{-6}$.
- Channels before the boundary do change.
- The norm of the rotated part is preserved to $10^{-4}$, the orthogonality
  property.

Then comes the test the whole scheme exists for. The harness rotates a fixed
query at $m$ and a fixed key at $n$ for the pairs $(10, 5)$, $(20, 15)$,
$(100, 95)$, and $(300, 295)$, all gap 5, and requires the four scores to agree
to a relative spread below $10^{-4}$. It also checks that gap 5 and gap 10 give
different scores, so a function that ignores position can't pass.

## Further reading

- [RoFormer: enhanced transformer with rotary position embedding](https://arxiv.org/abs/2104.09864) — the RoPE paper.
- [Attention is all you need](https://arxiv.org/abs/1706.03762) — the sinusoidal positional encodings RoPE replaced.
- [Extending context window of large language models via positional interpolation](https://arxiv.org/abs/2306.15595)
- [YaRN: efficient context window extension of large language models](https://arxiv.org/abs/2309.00071)
- [Train short, test long: attention with linear biases enables input length extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi, the main alternative.
- [Qwen2-VL: enhancing vision-language model's perception of the world at any resolution](https://arxiv.org/abs/2409.12191) — where mRoPE was introduced.
