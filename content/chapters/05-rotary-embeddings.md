---
title: Rotary position embeddings
slug: 05-rotary-embeddings
part: "Part 2 — A forward pass"
summary: RoPE derived from the relative-position requirement, then the partial and multimodal variants this model uses.
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

Attention has no idea what order its inputs are in. Permute the tokens of a
sequence and the attention output permutes with them, identically. "The cat sat on
the mat" and "the mat sat on the cat" produce the same set of outputs, just
reordered. Position has to be injected deliberately.

Rotary position embeddings do it by rotating each query and key vector by an angle
proportional to its position. The rotation is chosen so that when a query at
position $m$ meets a key at position $n$, the absolute angles cancel and only
$m - n$ survives in the score. The model learns relative position without ever
being told an absolute index.

This chapter derives that from the requirement, extends it to a full 256-wide
head, then covers the two things this model does differently: it rotates only a
quarter of each head, and it carries three position coordinates instead of one.

## Before you start

**Why attention is permutation invariant.** The attention output for query $i$ is
a weighted sum over keys and values, with weights
$\operatorname{softmax}_j(q_i \cdot k_j / \sqrt{d})$. Nothing in that expression
refers to $i$ or $j$ except through the vectors themselves. Reorder the keys and
values together and each weight follows its own key, so the sum is unchanged. The
causal mask imposes an ordering on *which* tokens are visible, but it says nothing
about how far apart they are.

**Rotation matrices.** A rotation in the plane by angle $\alpha$ is

$$
R(\alpha) =
\begin{bmatrix}
\cos\alpha & -\sin\alpha \\
\sin\alpha & \cos\alpha
\end{bmatrix}
$$

with three properties this chapter uses repeatedly. Rotations compose,
$R(\alpha)R(\beta) = R(\alpha + \beta)$. The transpose is the inverse,
$R(\alpha)^{\top} = R(-\alpha)$. And they preserve length,
$\lVert R(\alpha)x \rVert = \lVert x \rVert$. In one word: rotations are
orthogonal.

**Complex numbers as 2D rotations.** Identify $x = (x_0, x_1) \in \mathbb{R}^2$
with $z = x_0 + i x_1$. Multiplying by $e^{i\alpha}$ rotates by $\alpha$, and the
real inner product is $\operatorname{Re}[z_x \overline{z_y}]$, where the bar is
complex conjugation. The complex form makes the derivation two lines; the matrix
form is what you implement.

**Where this sits in the block.** Chapter 4 covered the per-head RMSNorm applied
to queries and keys. RoPE comes immediately after it and immediately before the
key and value are written to the cache:

```python
q, k, v, gate = self.project(x)          # q_norm and k_norm happen inside
q = apply_rotary_partial(q, cos, sin, self.rotary_dim)
k = apply_rotary_partial(k, cos, sin, self.rotary_dim)
if kv_cache is not None:
    k, v = kv_cache.append(layer_idx, k, v)
```

Note what that ordering implies: the cache stores keys that have already been
rotated, so a cached key carries its absolute position baked in. Chapter 15 comes
back to this when it tries to reuse a prefix.

## The requirement

Write $f(x, p)$ for the function that takes a vector and a position and produces
the vector attention will actually use. The property you want is that the score
between a query at $m$ and a key at $n$ depends on the two vectors and on the gap,
and on nothing else:

$$
\langle f(q, m), f(k, n) \rangle = g(q, k, m - n)
$$

This is a strong constraint. It rules out the obvious approaches. Adding a
learned per-position vector, as the original transformer's learned embeddings do,
gives a score with cross terms in $m$ and $n$ separately. Concatenating a position
feature does the same. The constraint asks for the absolute positions to cancel
exactly, for every $q$ and $k$.

## The two-dimensional solution

Start with $d = 2$, which turns out to be the whole problem. Treat $q$ and $k$ as
complex numbers and try a pure rotation:

$$
f(x, p) = x\,e^{i p \theta}
$$

for a fixed frequency $\theta$. Then

$$
\langle f(q, m), f(k, n) \rangle
= \operatorname{Re}\!\left[q e^{i m \theta} \cdot \overline{k e^{i n \theta}}\right]
$$

The conjugate of a product is the product of conjugates, and
$\overline{e^{i n \theta}} = e^{-i n \theta}$, so

$$
= \operatorname{Re}\!\left[q \bar{k}\, e^{i (m - n) \theta}\right]
$$

The absolute positions have cancelled. What remains is a function of $q$, $k$, and
$m - n$, which is exactly the requirement.

### The same thing as matrices

In real coordinates, $f(x, p) = R(p\theta)x$. Then

$$
\langle R(m\theta)q,\; R(n\theta)k \rangle
= q^{\top} R(m\theta)^{\top} R(n\theta)\, k
= q^{\top} R\big((n - m)\theta\big)\, k
$$

using $R(\alpha)^{\top} = R(-\alpha)$, and then composing the two rotations into
$R(n\theta - m\theta)$. Write $\delta = m - n$ and expand:

$$
q^{\top} R(-\delta\theta) k
= (q_0 k_0 + q_1 k_1)\cos(\delta\theta) + (q_0 k_1 - q_1 k_0)\sin(\delta\theta)
$$

The score is the ordinary dot product and the signed area of the parallelogram
spanned by $q$ and $k$, mixed by the cosine and sine of the gap. At $\delta = 0$
it is the plain dot product. As the gap grows, the mix rotates smoothly between
the two. That is the entire mechanism.

Two consequences worth naming now. The transform is orthogonal, so it does not
change $\lVert q \rVert$ or $\lVert k \rVert$ — the per-head RMSNorm from chapter
4 survives it. And $f$ has no learned parameters at all; it is a fixed function of
position.

## Extending to a full head

A head is 256 channels, not 2. Split it into $d/2$ consecutive pairs, give pair
$i$ its own frequency $\theta_i$, and rotate each pair independently. In matrix
form that is a block-diagonal rotation:

$$
R_p =
\begin{bmatrix}
R(p\theta_0) & & & \\
& R(p\theta_1) & & \\
& & \ddots & \\
& & & R(p\theta_{d/2 - 1})
\end{bmatrix}
$$

A block-diagonal matrix of rotations is itself orthogonal, and it composes
blockwise, so $R_m^{\top} R_n = R_{n-m}$ holds for the whole head. The inner
product splits into independent per-pair contributions:

$$
\langle R_m q, R_n k \rangle
= \sum_{i=0}^{d/2 - 1} \left\langle R(m\theta_i) q^{(i)},\; R(n\theta_i) k^{(i)} \right\rangle
$$

where $q^{(i)}$ is the 2-vector of channels $2i$ and $2i+1$. Every term depends
only on $m - n$, so the sum does too. The requirement is satisfied for a head of
any width.

### The frequencies

The frequencies are a geometric sequence set by a single hyperparameter
$\Theta$ — `rope_theta` in the config:

$$
\theta_i = \Theta^{-2i/d}, \qquad i = 0, 1, \ldots, \tfrac{d}{2} - 1
$$

So $\theta_0 = 1$ always, and the frequencies decrease geometrically to
$\Theta^{-(d-2)/d} \approx 1/\Theta$. Here $d$ is the *rotary* dimension, 64 for
this model, so there are 32 frequencies and $\theta_i = \Theta^{-i/32}$.

The natural way to read a frequency is as a wavelength — how many token positions
it takes to complete one full turn:

$$
\lambda_i = \frac{2\pi}{\theta_i} = 2\pi\,\Theta^{2i/d}
$$

Pair 0 turns once every $2\pi \approx 6.3$ positions and encodes fine
distinctions: adjacent, two apart, three apart. The last pair turns slowly and
encodes coarse position.

## What $\Theta$ controls, and why long context raises it

$\Theta$ sets the *range* of wavelengths, from $2\pi$ at the fast end to about
$2\pi\Theta$ at the slow end. It does not change the fast end at all.

The design constraint is at the slow end. If the slowest wavelength is shorter
than the context you intend to serve, the slowest pair wraps around, and positions
separated by a full wavelength become indistinguishable in every channel. Work out
this model's numbers, with $d = 64$ so the slowest pair is $i = 31$:

$$
\lambda_{31} = 2\pi\,\Theta^{31/32}
$$

| $\Theta$ | $\Theta^{31/32}$ | $\lambda_{31}$ | Turns over 262,144 positions |
|---|---|---|---|
| $10^4$ | about $7.50 \times 10^{3}$ | about $4.71 \times 10^{4}$ | about 5.6 |
| $10^7$ | about $6.04 \times 10^{6}$ | about $3.80 \times 10^{7}$ | about 0.0069 |

With the classic $\Theta = 10{,}000$, the slowest pair completes more than five
full turns across a 262,144-token context. Position 1,000 and position 48,000 land
at nearly the same angle in every pair, and the model has no way to tell them
apart. With $\Theta = 10^{7}$, the slowest pair covers less than 1% of a turn —
about 2.5 degrees — over the entire context. It has become a monotone coarse
position signal instead of a periodic one.

That is why long-context models raise $\Theta$, and it is the cheapest possible
change: $\Theta$ appears only in the table you precompute, so raising it costs
nothing at inference time. It is not free in general, though. Stretching the
wavelengths spreads the same 32 pairs over a longer range, so nearby positions
become slightly harder to distinguish in the slow pairs, and a model trained at
one $\Theta$ cannot be served at another without degradation. The
literature on extending context after training — position interpolation, NTK-aware
scaling, YaRN — is all about doing this rescaling in a way the trained model
tolerates. This model was trained at its $\Theta$, so the engine just reads the
config.

## Implementation

### Precompute the tables

The angles depend only on position, never on activations, so they are identical
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

| Tensor | Shape | Notes |
|---|---|---|
| `inv_freq` | `(32,)` | $\theta_i = \Theta^{-i/32}$, descending from 1. |
| `positions` | `(P,)` | float32, always. |
| `angles` | `(P, 32)` | Outer product: `angles[p, i]` is $p\,\theta_i$. |
| `cos`, `sin` | `(P, 32)` | Cast to the working dtype after the trig. |

At `P = 32768`, which is what `RotaryEmbedding` builds by default before growing
on demand, each table is $32768 \times 32 \times 4 = 4.19$ MB, so 8.4 MB for both.
Against 53.8 GB of weights that is nothing, and it buys you two transcendental
functions per channel that you never compute again.

The float32 in that snippet is load bearing. The angle at position $p$ for pair 0
is $p$ radians, and bfloat16 near 100,000 has an ulp of $2^{9} = 512$: any two
positions within 256 of each other round to the same bfloat16 value, so a whole
block of positions becomes indistinguishable in every pair at once. Compute the
angles in float32, take the cosine and
sine — which land in $[-1, 1]$, where bfloat16 is perfectly comfortable — and cast
after.

### Applying the rotation

The textbook pairing puts channel $2i$ with channel $2i+1$. Implementations
overwhelmingly use a different pairing, the *half split*: within the rotary block,
pair channel $j$ with channel $j + d/2$.

```python
def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

rotated = x * cos_full + _rotate_half(x) * sin_full
```

Check that this is the rotation. Let $a = x_j$ and $b = x_{j + 32}$, and let
$c = \cos(p\theta_j)$, $s = \sin(p\theta_j)$. Because `cos_full` is `cos`
concatenated with itself, channels $j$ and $j + 32$ both read frequency
$\theta_j$. Then

$$
\text{out}_j = a\,c + (-b)\,s = a c - b s
$$

$$
\text{out}_{j+32} = b\,c + a\,s = a s + b c
$$

which is exactly $R(p\theta_j)$ applied to $(a, b)$.

**Why the two pairings are interchangeable.** The half-split scheme is the
interleaved scheme composed with a fixed permutation of the head's channels — the
perfect shuffle. The same permutation is applied to $q$ and to $k$, and the
rotation is block-diagonal in both layouts, so the inner product is identical. The
permutation amounts to relabelling channels, and the learned q and k projections
absorb the relabelling during training. The half split is faster because both
halves are contiguous slices, so `chunk` and `cat` move coalesced blocks instead
of gathering every other element.

**This is a checkpoint contract, not a free choice.** A checkpoint trained with
the interleaved layout, served with the half-split kernel and no permutation of
the projection weights, pairs the wrong channels and hands them the wrong
frequencies. The model still runs. It still produces fluent text. It has no idea
what order the words are in. Chapter 8's end-to-end validation exists partly to
catch this.

## Partial rotary

`partial_rotary_factor` is 0.25 and `head_dim` is 256, so

$$
\text{rotary\_dim} = 256 \times 0.25 = 64
$$

Only the first 64 channels of each head rotate, in 32 pairs. The remaining 192
pass through untouched.

```python
rot, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
cos_full = torch.cat((cos, cos), dim=-1)[None, None, :, :].to(x.dtype)
sin_full = torch.cat((sin, sin), dim=-1)[None, None, :, :].to(x.dtype)
rotated = rot * cos_full + _rotate_half(rot) * sin_full
return torch.cat((rotated, passthrough), dim=-1)
```

| Tensor | Shape for `q` | Shape for `k` |
|---|---|---|
| `x` | `(batch, 24, seq, 256)` | `(batch, 4, seq, 256)` |
| `rot` | `(batch, 24, seq, 64)` | `(batch, 4, seq, 64)` |
| `passthrough` | `(batch, 24, seq, 192)` | `(batch, 4, seq, 192)` |
| `cos`, `sin` in | `(seq, 32)` | `(seq, 32)` |
| `cos_full`, `sin_full` | `(1, 1, seq, 64)` | `(1, 1, seq, 64)` |
| return | `(batch, 24, seq, 256)` | `(batch, 4, seq, 256)` |

The two leading singleton axes on `cos_full` are what let one table serve all 24
query heads and all 4 KV heads by broadcasting. Queries and keys use the same
table; only the head count differs.

### What the unrotated channels do, geometrically

The transform on a full head is now block-diagonal with two blocks: a rotation on
the first 64 channels and the identity on the last 192. Because it is
block-diagonal, the score splits cleanly:

$$
\langle f(q, m), f(k, n) \rangle
= \underbrace{q_{\text{rot}}^{\top} R_{n-m}\, k_{\text{rot}}}_{\text{depends on } m - n}
\;+\;
\underbrace{q_{\text{pass}}^{\top} k_{\text{pass}}}_{\text{no position at all}}
$$

Each head therefore computes a sum of two scores: a relative-position score over a
64-dimensional subspace, and a pure content-match score over a 192-dimensional
subspace. The model chooses, per head and per direction, how much of its score to
put in each.

**Why that is not obviously harmful.** Full RoPE forces every direction in the
head to be position-modulated, and the modulation attenuates with distance: sum
enough pairs at different frequencies and the position-dependent term decays as
the gap grows. That is a useful inductive bias for local syntax and an actively
unhelpful one for a head whose job is "find the definition of this term, wherever
it appeared, 80,000 tokens ago". Reserving three quarters of each head for a score
that does not decay with distance gives the model a retrieval path that long
context does not weaken. It is a strong architectural bet, and it is one reason
this model handles long context well.

**What it does not save.** The cache is unchanged: you still store all 256
channels of every KV head, because the passthrough channels are part of the key.
The saving is arithmetic, and the arithmetic was never the bottleneck — applying
the rotation costs about 3 FLOPs per channel, so rotating 64 channels instead of
256 across 28 heads saves about 16,000 FLOPs per token per full-attention layer,
which is noise next to a 2.54 GFLOP output projection. Treat partial rotary as an
architectural choice, not an optimization.

**Check the boundary against the config.** `ModelConfig.rotary_dim` computes
`head_dim * partial_rotary_factor` and is 64 here. Splitting at 128 — half the
head, which is what you get if you assume the half-split trick and the partial
factor are the same thing — produces a model that runs, generates fluent text, and
ignores word order.

## Multimodal RoPE

This model accepts images, and an image is not a sequence. A patch at row 7,
column 12 of a frame has three coordinates, not one, and flattening them into a
single index throws away the two-dimensional structure the model needs.

mRoPE keeps the same rotation and changes where each pair reads its angle from.
The 32 frequency pairs are split into sections — `mrope_section` is `[11, 11, 10]`,
which sums to 32 — and each section takes its angle from a different coordinate:
time, height, width.

```python
def apply_mrope_sections(cos, sin, sections):
    cos_parts, sin_parts, start = [], [], 0
    for axis, width in enumerate(sections):
        cos_parts.append(cos[axis, :, start : start + width])
        sin_parts.append(sin[axis, :, start : start + width])
        start += width
    return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)
```

| Tensor | Shape | Meaning |
|---|---|---|
| `cos`, `sin` in | `(3, seq, 32)` | One plane per coordinate: t, h, w. |
| `cos[0, :, 0:11]` | `(seq, 11)` | Pairs 0–10 read the time coordinate. |
| `cos[1, :, 11:22]` | `(seq, 11)` | Pairs 11–21 read the height coordinate. |
| `cos[2, :, 22:32]` | `(seq, 10)` | Pairs 22–31 read the width coordinate. |
| return | `(seq, 32)` | Reassembled, ready for `apply_rotary_partial`. |

The important property: **for a text token, all three coordinates hold the same
number**. Every plane of the input is then identical, so slicing different
sections from different planes reassembles exactly the single-plane table, and
mRoPE reduces to standard RoPE with no special case and no approximation. That is
why a text-only engine — including this course's — can ignore the distinction
entirely and still be correct. For an image patch the coordinates differ, and the
patch carries its row and column alongside its place in the sequence.

`mrope_interleaved` is true in this config. The flag selects a different mapping
from sections to channel pairs: sections distributed across the frequency range
rather than taken as contiguous blocks. `apply_mrope_sections` implements the
contiguous layout, which is the one that makes the mechanism legible. Under either
mapping the text path is identical, for the reason above.

## Positions during decode

Prefill passes positions $0, 1, \ldots, n-1$. Decode passes a single position: the
index of the token being generated, which is the current cache length.

```python
start = cache.length if cache is not None else 0
positions = torch.arange(start, start + seq, device=input_ids.device)
```

One expression covers both phases. During prefill `cache.length` is 0 and `seq` is
the prompt length; during decode `cache.length` is however many tokens are already
cached and `seq` is 1.

Passing 0 every step — an easy mistake when the decode path is written separately
from prefill — makes every generated token believe it is the first token in the
sequence. The relative gap to every cached key is then wrong by a growing amount,
and output degenerates into repetition after a few dozen tokens. It looks like a
sampling bug. It is not.

## What goes wrong

**Splitting at the wrong index.** Fluent text that ignores word order. Check
`rotary_dim` against the config, and check that channels past it come back bitwise
unchanged.

**Rotating the values.** Position belongs in the score, not the payload. Rotating
$v$ rotates the content the model retrieves, by an angle that depends on where the
content happened to sit. Only $q$ and $k$ are rotated.

**Rotating keys after appending them to the cache.** The cache already holds
rotated keys. Rotating again applies the angle twice, and the effective gap
becomes $2m - n$. Rotate, then append.

**Building the angle table in bfloat16.** Fine at position 100, wrong at position
100,000, where the ulp is 512. Positions collapse into buckets and long context
quietly stops working. Compute in float32, cast the cosines.

**Forgetting to duplicate `cos` to the full rotary width.** The tables are
`(seq, 32)` and the half-split rotation needs `(seq, 64)`. Without the `cat` you
get a shape error if you are lucky, and a silent broadcast against the head axis
if you are not.

**Forgetting `.to(x.dtype)` on the tables.** float32 tables promote the bfloat16
query, the attention matmul falls off the tensor cores, and prefill slows down
several-fold with nothing in the logs.

## Check your understanding

**Why can't you satisfy the relative-position requirement by adding a learned
vector $p_m$ to the query and $p_n$ to the key?**

Expand the score:

$$
(q + p_m) \cdot (k + p_n)
= q \cdot k + q \cdot p_n + p_m \cdot k + p_m \cdot p_n
$$

The middle two terms depend on $m$ and $n$ separately, not on
their difference, so there is no function $g$ of $m - n$ that reproduces the score
for every $q$ and $k$. Rotation works because it acts on $q$ and $k$ by an
orthogonal map, and orthogonality is what makes the two absolute angles combine
into one relative angle.

**The model rotates 64 of 256 channels. Does that make the KV cache smaller?**

No. All 256 channels of every KV head go into the cache, rotated or not — the
passthrough channels are part of the key and are needed to compute the score.
Partial rotary saves a small amount of arithmetic and changes what the head can
represent. Chapter 9's cache arithmetic is unaffected by it.

**A model trained with $\Theta = 10{,}000$ is served at 200,000 tokens of context.
What breaks first?**

The slow pairs. At $\Theta = 10^4$ and 64 rotary channels the slowest wavelength
is about 47,000 positions, so over 200,000 tokens it wraps more than four times.
Distant positions become aliased onto each other and the model loses its coarse
sense of where it is, while local ordering, carried by the fast pairs, still
works. The symptom is a model that reads nearby text correctly and cannot locate
anything far away.

**Your implementation passes every shape and norm check, but the relative-position
test fails: scores for gap 5 differ depending on where in the sequence the pair
sits. What is the most likely cause?**

Query and key are not being rotated with the same frequency assignment. Either one
of them is using a differently built table, or the pairing differs between the two
paths — half split for one and interleaved for the other. The absolute angles only
cancel when both sides use the same block-diagonal $R$.

## Lab

Implement three functions. The lab runs on GPU but is device agnostic, so the
maths is checkable anywhere.

`build_rope_cache(rotary_dim, max_position, theta, device, dtype)` returns `cos`
and `sin` of shape `(max_position, rotary_dim // 2)`, with
$\theta_i = \Theta^{-2i/d}$. The harness checks the shape, that position 0 gives
$\cos = 1$ and $\sin = 0$, that $\cos^2 + \sin^2 = 1$ everywhere, and that the
first frequency rotates faster than the last.

`rotate_half(x)` maps $(x_1, x_2)$ to $(-x_2, x_1)$ over the last axis, split in
halves.

`apply_rotary_partial(x, cos, sin, rotary_dim)` rotates the first `rotary_dim`
channels of a `(batch, heads, seq, head_dim)` tensor and passes the rest through.
The harness runs it with `head_dim = 256`, `rotary_dim = 64`, and
$\Theta = 10^{7}$, and checks that the shape survives, that channels past the
boundary are bitwise unchanged, that channels before it do change, and that the
norm of the rotated part is preserved to $10^{-4}$ — the orthogonality property.

Then the test the whole scheme exists for. The harness rotates a fixed query at
$m$ and a fixed key at $n$ for the pairs $(10, 5)$, $(20, 15)$, $(100, 95)$, and
$(300, 295)$ — all gap 5 — and requires the four scores to agree to a relative
spread below $10^{-4}$. It also checks that gap 5 and gap 10 give different
scores, so a function that ignores position cannot pass.

## Further reading

- [RoFormer: enhanced transformer with rotary position embedding](https://arxiv.org/abs/2104.09864) — the RoPE paper.
- [Attention is all you need](https://arxiv.org/abs/1706.03762) — the sinusoidal positional encodings RoPE replaced.
- [Extending context window of large language models via positional interpolation](https://arxiv.org/abs/2306.15595)
- [YaRN: efficient context window extension of large language models](https://arxiv.org/abs/2309.00071)
- [Train short, test long: attention with linear biases enables input length extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi, the main alternative.
- [Qwen2-VL: enhancing vision-language model's perception of the world at any resolution](https://arxiv.org/abs/2409.12191) — where mRoPE was introduced.
