---
title: Speculative decoding
slug: 20-speculative-decoding
part: "Part 6 — Scaling"
summary: Trading idle compute for tokens, with a draft model or the model's own MTP head.
minutes: 80
gpu: true
objectives:
  - Explain why verifying k tokens costs about the same as generating one.
  - Implement rejection sampling that preserves the target distribution exactly.
  - Describe how a multi-token prediction head removes the need for a draft model.
lab: 20-spec-decode
---

# Speculative decoding

Decode at batch 1 uses about 0.7% of an A100's arithmetic. Everything else is idle
while the memory system reads 53.8 GB of weights. Speculative decoding spends that
idle compute on tokens.

## The observation

A forward pass on one token and a forward pass on five tokens take almost the same
time. Both read all the weights; the second does five times the arithmetic, and
the arithmetic wasn't the bottleneck.

So: guess the next few tokens cheaply, then verify all of them in one forward pass
of the real model. Every guess that survives is a token you got for free.

## The algorithm

1. A cheap *draft* proposes *k* tokens autoregressively.
2. The *target* model runs one forward pass over all *k*, producing a distribution
   at each position.
3. Accept or reject each draft token in order, by a rule that preserves the
   target's distribution exactly.
4. On the first rejection, sample a corrected token from an adjusted distribution
   and discard the rest.

A round produces between 1 and *k+1* tokens. Even a total rejection yields one
token, so it never loses ground.

## Rejection sampling

This is the part that must be exactly right, because a plausible-looking
approximation changes the model's output distribution in ways no test will catch.

For a draft token *x* with draft probability *q(x)* and target probability *p(x)*:

```python
r = torch.rand(())
if r < min(1.0, p[x] / q[x]):
    accept(x)
else:
    residual = torch.clamp(p - q, min=0)
    corrected = torch.multinomial(residual / residual.sum(), 1)
    accept(corrected)
    break
```

Accept with probability `min(1, p/q)`. On rejection, sample from the normalized
positive part of `p - q`.

The result is provably distributed exactly as if you'd sampled from the target
model directly. Not approximately — exactly. That's what makes speculative
decoding safe to turn on by default: it changes speed, not output quality.

Two details cause most bugs. The residual must be clamped at zero before
normalizing, or you get negative probabilities. And the draft's distribution must
come from the same sampling parameters as the target's, or the acceptance rate
collapses.

## The acceptance rate decides everything

Let *α* be the probability a draft token is accepted. Expected tokens per round:

```text
E[tokens] = (1 - α^(k+1)) / (1 - α)
```

| α | k=3 | k=5 | k=8 |
|---|---|---|---|
| 0.5 | 1.88 | 1.97 | 2.00 |
| 0.7 | 2.53 | 2.94 | 3.24 |
| 0.9 | 3.44 | 4.69 | 6.13 |

Two things follow. High acceptance is worth much more than a long draft: at
α = 0.5, going from k=3 to k=8 gains 6%. And the returns to *k* saturate, because
one rejection discards everything after it.

Net speedup is `E[tokens] / (1 + k x draft_cost)`, where `draft_cost` is the
draft's forward pass as a fraction of the target's. A draft that's 5% of the target
with α = 0.8 and k = 4 gives about 2.4 times. A draft that's 30% of the target
gives almost nothing regardless of α.

The practical rule: the draft must be at least 15 to 20 times cheaper, and drawn
from the same family so the tokenizers match and the distributions agree.
Qwen3-0.6B drafting for Qwen3.8-27B is roughly the right ratio.

## Multi-token prediction

This model has an MTP head — `mtp_num_hidden_layers` is 1 in the config. It's a
single extra layer trained to predict the token *after* next from the same hidden
state.

That removes the draft model entirely. The head runs on hidden states the target
already computed, so its cost is one layer out of 64, under 2%. And because it was
trained jointly with the model, its acceptance rate is high — typically 0.7 to 0.85,
where a separate small model gives 0.6 to 0.7.

Self-speculation with an MTP head is the strongest option available here: cheaper
draft, higher acceptance, no second model to load or keep in memory.

## Batching makes it worse

At batch 1, speculative decoding gives 2 to 3 times. At batch 32, it often gives
nothing.

The reason follows from chapter 10. Large batches already push arithmetic intensity
up, so there's no idle compute left to spend. Worse, sequences in a batch reject at
different points, so the batch advances by the *minimum* accepted length while
paying for the maximum.

Production engines enable speculation adaptively: on at low batch size, off at
high. That's the correct behavior and it's worth implementing rather than treating
speculation as a global switch.

## Cache management

Rejected tokens must be removed from the KV cache. With a contiguous cache that's
a length rollback. With the paged cache from chapter 15, it's a rollback of the
block table's length and freeing any blocks the rejected tokens claimed.

The linear-attention layers are harder, and this is where a hybrid model
complicates speculation. Their state is overwritten in place, so you can't roll it
back — you have to snapshot the state before speculating and restore it on
rejection. At 147.8 MiB per sequence that's a real cost, and it's why the practical
approach is to run the speculative steps through the linear layers only after the
accepted length is known.

## Lab

Implement the draft-verify loop with correct rejection sampling. The harness
checks the distribution property statistically: over many rounds, the token
distribution must match direct sampling from the target within a tolerance. Then
measure acceptance rate and speedup at k from 1 to 8, and at batch sizes 1, 8, and
32, to show where speculation stops paying.

## Further reading

- [Fast inference from transformers via speculative decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating large language model decoding with speculative sampling](https://arxiv.org/abs/2302.01318)
- [Medusa: simple LLM inference acceleration framework with multiple decoding heads](https://arxiv.org/abs/2401.10774)
