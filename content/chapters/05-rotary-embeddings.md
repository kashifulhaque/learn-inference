---
title: Rotary position embeddings
slug: 05-rotary-embeddings
part: "Part 2 — A forward pass"
summary: RoPE from the relative-position requirement, then the partial and multimodal variants this model uses.
minutes: 50
gpu: true
objectives:
  - Derive RoPE from the requirement that attention scores depend on relative position.
  - Implement partial rotary embeddings and explain what the unrotated channels do.
  - Explain how mRoPE assigns different position coordinates to different channel groups.
lab: 05-rope
---

# Rotary position embeddings

Attention is permutation invariant. Without position information, "the cat sat on
the mat" and "the mat sat on the cat" produce identical outputs. RoPE injects
position by rotating queries and keys, and it does so in a way that makes the
attention score depend only on the *distance* between two tokens.

## The derivation

You want a function *f* that takes a vector and a position and produces something
whose inner product depends only on the gap:

```text
<f(q, m), f(k, n)> = g(q, k, m - n)
```

Take a two-dimensional slice of a head and treat it as a complex number. Multiply
by e^(i·m·θ) — a rotation by angle m·θ. Then:

```text
<f(q, m), f(k, n)> = Re[(q e^(imθ)) · conj(k e^(inθ))]
                   = Re[q · conj(k) · e^(i(m-n)θ)]
```

The absolute positions cancel and only `m - n` survives. That's the whole idea.

Extend it to a full head by splitting the channels into pairs and giving each pair
its own frequency:

```text
θ_j = base^(-2j/d),  j = 0 .. d/2 - 1
```

Low-index pairs rotate fast and encode fine position distinctions. High-index
pairs rotate slowly — with a base of 10 million, the slowest pair completes a
fraction of a turn across the entire 262,144-token context, so it encodes coarse
position. A large base is what makes long context work: with a base of 10,000 the
slow frequencies wrap around several times over 262k positions, and the model
can't distinguish position 1000 from position 100,000.

## Implementation

The angles depend only on position, so precompute them once:

```python
inv_freq = 1.0 / (theta ** (torch.arange(0, half) / half))
angles = torch.outer(positions, inv_freq)
cos, sin = angles.cos(), angles.sin()
```

Applying the rotation uses the half-split trick. Rather than interleaving pairs,
split the head down the middle and treat channel `j` and channel `j + d/2` as a
pair:

```python
def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

rotated = x * cos + _rotate_half(x) * sin
```

This is the same rotation with the channels permuted, and the permutation is
consistent between queries and keys so it cancels. It's faster because both halves
stay contiguous.

Rotate queries and keys. Never rotate values: the position information belongs in
the score, and rotating values would rotate the content the model retrieves.

## Partial rotary

`partial_rotary_factor` is 0.25, so only the first 64 of each head's 256 channels
get rotated. The other 192 pass through unchanged.

The unrotated channels carry content that's independent of position. A head that
retrieves "the definition of this term, wherever it appeared" wants channels whose
match strength doesn't decay with distance. Reserving three quarters of the head
for that is a strong architectural bet, and it's one reason this model handles
262k context.

Nothing about the cache changes. You still store all 256 channels per head. The
saving is arithmetic only.

```python
rot, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
rotated = rot * cos_full + _rotate_half(rot) * sin_full
return torch.cat((rotated, passthrough), dim=-1)
```

Splitting at the wrong index produces a model that runs and generates fluent text
that ignores word order. Check the boundary against
`ModelConfig.rotary_dim`, which is 64 here.

## Multimodal RoPE

This model accepts images, and an image isn't a sequence. mRoPE splits the rotary
channels into sections — `mrope_section` is `[11, 11, 10]`, covering all 32
frequency pairs — and each section reads its angle from a different coordinate:
time, height, and width.

For a text token all three coordinates hold the same number, and mRoPE reduces
exactly to standard RoPE. For an image patch they differ, so the patch carries its
position in two spatial dimensions plus its place in the sequence.

The text-only path can ignore this, and this course's engine does. `mrope_interleaved`
is true, which changes how sections map onto channels; `apply_mrope_sections` in
`engine/layers/rope.py` shows the layout.

## Positions during decode

Prefill passes positions `0 .. n-1`. Decode passes a single position: the index of
the token being generated, which is the cache length. Passing 0 every step — an
easy mistake when you write the decode path separately — makes every generated
token believe it's the first, and output degenerates into repetition after a few
dozen tokens.

## Lab

Implement `build_rope_cache` and `apply_rotary_partial`. Then demonstrate the
relative-position property numerically: show that the inner product between a
rotated query at position *m* and a rotated key at position *n* depends only on
`m - n`. Finally, check that your partial variant leaves the last 192 channels
untouched.

## Further reading

- [RoFormer: enhanced transformer with rotary position embedding](https://arxiv.org/abs/2104.09864)
- [YaRN: efficient context window extension of large language models](https://arxiv.org/abs/2309.00071)
