---
title: Sampling
slug: 11-sampling
part: "Part 3 — Making it fast"
summary: Temperature, top-k, top-p, min-p, and penalties, applied in the order that makes them mean what they claim.
minutes: 40
gpu: false
objectives:
  - Apply sampling transforms in the correct order and explain why the order matters.
  - Implement top-p, min-p, and repetition penalty correctly.
  - Reproduce a generation exactly given a seed.
lab: 11-sampling
---

# Sampling

The model produces 248,320 logits. Sampling turns them into one token. The
operations are simple; the order is not, and getting it wrong yields a sampler
whose knobs don't do what their names say.

## The order

1. **Penalties**, on raw logits.
2. **Temperature**, dividing logits.
3. **Truncation** — top-k, then min-p, then top-p.
4. **Softmax and draw.**

Each step depends on the previous one having happened.

Penalties come first because they're defined on logits. Applying a repetition
penalty after temperature makes its strength depend on the temperature, so a
penalty of 1.1 means something different at 0.7 than at 1.0.

Temperature comes before truncation because truncation thresholds are defined on
the tempered distribution. Applying top-p to the untempered probabilities and then
scaling gives a different nucleus than the parameter describes.

## Temperature

```python
logits = logits / temperature
```

Below 1.0 this sharpens the distribution; above 1.0 it flattens it. As temperature
approaches 0 the distribution approaches a point mass on the argmax, so the engine
special-cases `temperature == 0` to a plain `argmax` rather than dividing by
something near zero.

Temperature scales *logits*, not probabilities. Scaling probabilities and
renormalizing is a different operation and doesn't produce the intended
distribution.

## Top-k

Keep the *k* highest-scoring tokens, discard the rest.

```python
threshold = logits.topk(k, dim=-1).values[..., -1, None]
return logits.masked_fill(logits < threshold, float("-inf"))
```

The weakness is that *k* is fixed while the model's confidence isn't. When the next
token is nearly determined, `k = 50` admits 49 tokens that shouldn't be there. When
the model is genuinely uncertain across hundreds of options, it cuts off reasonable
ones.

## Top-p

Sort by probability, take the smallest prefix whose cumulative mass reaches *p*.

The subtlety is the boundary. The token that pushes the cumulative sum past *p*
must be kept, not dropped — otherwise the retained mass is less than *p*, and at
`p = 0.9` with a first token at 0.95 you'd keep nothing.

```python
cumulative = sorted_probs.cumsum(dim=-1)
remove = cumulative - sorted_probs >= p   # compare the mass *before* this token
remove[..., 0] = False                    # always keep at least one
```

Comparing `cumulative >= p` instead of `cumulative - sorted_probs >= p` drops the
boundary token. It's the most common top-p bug and it's invisible in normal
output.

## Min-p

Keep tokens at least `min_p` times as likely as the most likely one.

```python
threshold = probs.max(dim=-1, keepdim=True).values * min_p
return logits.masked_fill(probs < threshold, float("-inf"))
```

Unlike top-k and top-p, the threshold is relative to the model's own confidence. A
sharp distribution keeps almost nothing; a flat one keeps a lot. That adaptivity is
why min-p holds up better at high temperature, and `min_p = 0.05` with
`temperature = 1.5` is a reasonable pairing where top-p would produce noise.

## Penalties

Repetition penalty divides positive logits of already-seen tokens and *multiplies*
negative ones:

```python
scores = torch.where(scores > 0, scores / penalty, scores * penalty)
```

The asymmetry is required. Dividing a negative logit by 1.1 makes it larger, which
encourages the token you meant to discourage. Roughly half the vocabulary has
negative logits at any position, so a naive implementation gets it backwards half
the time.

Presence and frequency penalties are subtractive and don't have this problem:

```python
logits -= presence * (counts > 0) + frequency * counts
```

Presence applies once per distinct token; frequency scales with the count. Both act
on logits, before temperature.

## Reproducibility

Given a seed and identical inputs, a generation must reproduce exactly. That means
threading an explicit `torch.Generator` through `torch.multinomial` rather than
relying on global state, which any other operation in the process can disturb.

Exact reproducibility across *batch sizes* is a stronger requirement and usually
not worth paying for: reductions run in different orders at different batch sizes,
logits differ in the last bits, and an argmax near a tie can flip. Reproducibility
at fixed batch size and fixed seed is the practical guarantee.

## Numerical stability

`exp` overflows float32 above about 88. Real logits reach the tens, and attention
scores go further. Subtract the row max first:

```python
shifted = x - x.max(dim=dim, keepdim=True).values
exp = shifted.exp()
return exp / exp.sum(dim=dim, keepdim=True)
```

The constant cancels between numerator and denominator, so the result is
unchanged, and every exponent is now at most 0. This is the same trick that
FlashAttention applies incrementally in chapter 14.

## Lab

Implement the full sampling pipeline: penalties, temperature, top-k, min-p, top-p,
and a seeded draw. The harness checks the top-p boundary case, the repetition
penalty sign handling, and that a fixed seed reproduces a fixed sequence. It also
tests that `temperature = 0` returns the argmax exactly.

## Further reading

- [The curious case of neural text degeneration](https://arxiv.org/abs/1904.09751) — the nucleus sampling paper.
- [Turning up the heat: min-p sampling for creative and coherent LLM outputs](https://arxiv.org/abs/2407.01082)
