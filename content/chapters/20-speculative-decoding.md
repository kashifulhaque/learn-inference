---
title: Speculative decoding
slug: 20-speculative-decoding
part: "Part 6 — Scaling"
summary: Trading idle compute for tokens, with the proof that the output distribution is exactly unchanged.
minutes: 110
gpu: true
objectives:
  - Explain why verifying gamma + 1 tokens costs about the same as generating one.
  - Prove that the accept-reject rule produces samples distributed exactly as the target.
  - Derive the expected tokens per round, the net speedup, and the optimal draft length.
  - "Choose a draft: a small model, a multi-token prediction head, n-gram lookup, or Medusa heads."
  - Roll back a rejected token from both a paged KV cache and a recurrent state.
lab: 20-spec-decode
---

# Speculative decoding

> [!TLDR]
> - Decode at batch 1 is memory bound, so verifying five tokens costs the same
>   42.2 ms as generating one.
> - Accept each draft token with probability $\min(1, p/q)$ and resample from
>   a residual on rejection, and the output is distributed exactly as the
>   target.
> - The acceptance rate is $1 - D_{\text{TV}}(p, q)$. The speedup is capped by
>   $1/(1-\alpha)$ and, at large batch sizes, by the ridge point.
> - Rejected tokens must be rolled back from the KV cache and the recurrent
>   state; verifying as one chunk avoids the second rollback.

Decode at batch 1 uses about 0.7% of an A100's arithmetic. Everything else is
idle while the memory system reads 53.8 GB of weights. Speculative decoding
spends that idle compute on tokens.

What makes it more than a heuristic is that it's exact. The accept-reject rule
produces samples distributed *identically* to sampling from the target model
directly: not approximately, not in the limit, but for every token. That proof
is the point of this chapter, and it's the part most treatments skip. Without
it, speculation is a quality risk you can't measure. With it, it's
==a pure speed knob== you can turn on by default.

## Before you start

**Sampling from a categorical distribution.** A distribution $p$ over a
vocabulary $V$ assigns $p(x) \ge 0$ with $\sum_{x \in V} p(x) = 1$. The
[sampling chapter](/c/11-sampling) covers how temperature, top-$k$, and top-$p$
transform the logits into $p$. Everything here operates on $p$ after those
transforms.

**Total variation distance.** For two distributions on the same support, it's
half the summed absolute difference:

$$
D_{\text{TV}}(p, q) = \frac{1}{2}\sum_{x \in V} \lvert p(x) - q(x) \rvert
$$

It lands in $[0, 1]$, is 0 when the distributions are identical, and turns out
to be exactly what the acceptance rate depends on. The lab measures it.

**The roofline.** Decode at batch 1 has an arithmetic intensity around 1 FLOP
per byte against a ridge point of 161, so its time is bytes over bandwidth. That
gap is the resource speculation spends. For more, see
[the roofline chapter](/c/10-roofline).

**The paged KV cache and the recurrent state.** Both are per-sequence state
that a rejected token must not advance. They come from
[the paged attention chapter](/c/15-paged-attention) and
[the gated delta rule chapter](/c/06-gated-delta-rule), and the last section of
this chapter is about rolling them back.

**Notation.** Write $\gamma$ for the number of tokens the draft proposes in one
round and $\alpha$ for the probability an individual draft token is accepted.
The lab calls $\gamma$ `k`.

## Why verifying many tokens costs one weight read

A forward pass on one token and a forward pass on five tokens take
==almost the same time==, because the pass is memory bound.

> [!EXAMPLE] One token versus five, at batch 1
> The weights are 53.8 GB, read once per forward pass regardless of how many
> tokens are in it. At the measured copy bandwidth of 1275 GB/s:
>
> $$
> t_{\text{memory}} = \frac{53.8 \times 10^{9}}{1275 \times 10^{9}}
> = 42.2 \text{ ms}
> $$
>
> The arithmetic for $T$ tokens at batch 1 is about
> $2 \times 26.9 \times 10^{9} \times T$ FLOPs. At $T = 5$:
>
> $$
> t_{\text{compute}} = \frac{2.69 \times 10^{11}}{312 \times 10^{12}}
> = 0.86 \text{ ms}
> $$
>
> The arithmetic intensity is $2.69 \times 10^{11} / 53.8 \times 10^{9} = 5.0$
> FLOPs per byte, against a ridge point of 161.

The pass is memory bound by a factor of 32, so the compute time hides entirely
inside the memory time:

$$
\boxed{t(1 \text{ token}) = t(5 \text{ tokens}) = 42.2 \text{ ms}}
$$

So guess the next few tokens cheaply, then verify all of them in one forward
pass. Every guess that survives is a token you got for free. This only works
because decode is memory bound, which is why speculation and batching, the
other way to fill idle compute, fight each other. That comes back at the end.

## The algorithm

One round of speculative decoding has five steps:

1. A cheap **draft** proposes $\gamma$ tokens autoregressively, each conditioned
   on the ones before it.
2. The **target** model runs one forward pass over all $\gamma$ draft tokens
   plus the current prefix, producing a distribution at each of the $\gamma + 1$
   positions.
3. Walk the draft tokens in order, accepting or rejecting each by the rule in
   the next section.
4. On the first rejection, sample a corrected token from an adjusted
   distribution and discard everything after it.
5. If every draft token is accepted, sample a bonus token from the target's
   distribution at position $\gamma + 1$, which the same forward pass already
   produced.

A round yields between 1 and $\gamma + 1$ tokens. Even total rejection yields
one, so ==speculation never loses ground==: the worst case is one token per
target pass, which is what you had before.

## The acceptance rule and its proof

This section states the rule and proves the output is distributed exactly as
the target. Fix one position. Let $p$ be the target's distribution there and
$q$ the draft's. The draft proposed token $x \sim q$.

**The rule.** Draw $u \sim \mathcal{U}(0,1)$ and accept $x$ if:

$$
u \le \min\!\left(1, \frac{p(x)}{q(x)}\right)
$$

On rejection, draw the replacement from the *residual distribution*, the
positive part of $p - q$ renormalized:

$$
p'(y) = \frac{\big(p(y) - q(y)\big)_{+}}{\sum_{z \in V}\big(p(z) - q(z)\big)_{+}}
$$

Here $(a)_{+} = \max(0, a)$.

**The claim.** The token this procedure outputs is distributed exactly as $p$.

### The proof

The output is $y$ in one of two ways: $y$ was drafted and accepted, or
something was rejected and the residual produced $y$. The proof adds up those
two paths.

**The accept path.** The draft draws $x$ with probability $q(x)$ and accepts
with probability $\min(1, p(x)/q(x))$. The $q(x)$ cancels cleanly in both
branches of the minimum, which is why the rule is stated as a ratio:

$$
\Pr[\text{draw } x \text{ and accept}]
= q(x)\min\!\left(1, \frac{p(x)}{q(x)}\right)
= \hlc{\min\big(q(x), p(x)\big)}
$$

Summing that over the vocabulary gives the acceptance probability
$\alpha = \sum_{y} \min\big(p(y), q(y)\big)$, so a rejection happens with
probability $1 - \alpha$.

**The reject path.** The residual's normalizer turns out to be exactly
$1 - \alpha$, the rejection probability. So the reject path contributes
$(1-\alpha)\,p'(y) = \hld{(p(y) - q(y))_{+}}$.

**Both paths together.** Add the two contributions, and use
$(a - b)_{+} = a - \min(a, b)$ on the second:

$$
\Pr[\text{output} = y]
= \hlc{\min\big(p(y), q(y)\big)} + \hld{p(y) - \min\big(p(y), q(y)\big)}
= \boxed{p(y)}
$$

That's the claim. The argument holds for every $q$, including a bad one:
==speculation with a terrible draft is slow, never wrong==.

> [!INTUITION]
> The draft over-proposes some tokens and under-proposes others. Where it
> over-proposes, the accept step trims the excess and keeps the
> $\hlc{\min(p, q)}$ part. Where it under-proposes, the shortfall
> $\hld{(p - q)_{+}}$ is exactly what the residual supplies. The two pieces
> fit together into $p$.

> [!DEEPDIVE] Why the residual's normalizer is the rejection probability
> For any reals $a, b$, $(a - b)_{+} = a - \min(a, b)$: if $a \ge b$ both
> sides are $a - b$, and if $a < b$ both sides are 0. Apply that identity term
> by term and sum:
>
> $$
> \sum_{y} \big(p(y) - q(y)\big)_{+}
> = \sum_{y} p(y) - \sum_{y} \min\big(p(y), q(y)\big)
> = 1 - \alpha
> $$
>
> This is the step that makes the whole thing work, and it isn't a
> coincidence: the residual holds precisely the probability mass the draft
> failed to deliver.

### The acceptance rate is one minus total variation distance

The acceptance rate is fixed by how close the draft is to the target. Start
from $\alpha = \sum_y \min(p(y), q(y))$, use
$\min(a,b) = \tfrac{1}{2}(a + b - |a - b|)$, and sum:

$$
\alpha = \frac{1}{2}\left(1 + 1 - \sum_y \lvert p(y) - q(y)\rvert\right)
= \boxed{1 - D_{\text{TV}}(p, q)}
$$

> [!KEY] Acceptance is a property of the draft, not a knob
> The only way to raise $\alpha$ is to make the draft agree with the target
> more often. That's why a smarter sampler never helps and a draft from the
> same model family always does.

### In code

The accept probability and the residual are a few lines each:

```python
def accept_probability(p, q, token):
    return min(1.0, (p[token] / q[token]).item())

def residual_distribution(p, q):
    residual = torch.clamp(p - q, min=0.0)
    total = residual.sum()
    if total <= 0:            # p == q: the branch is unreachable, but guard it
        return p.clone()
    return residual / total
```

Two details cause most bugs, and the lab tests both:

- **Clamp before normalizing.** Otherwise negative entries survive and
  `multinomial` either raises or samples nonsense.
- **Guard the zero sum.** When $p = q$ the residual sums to zero, so
  normalizing divides by zero. The branch is never taken, because
  $\Pr[\text{reject}] = 0$ in that case, but the code still must not produce a
  NaN.

> [!WARNING] Draft and target must use the same sampling parameters
> If the target applies temperature 0.7 and top-$p$ 0.95 and the draft samples
> raw, the ratio $p(x)/q(x)$ compares two different objects. The acceptance
> rate collapses, and the guarantee you proved is a guarantee about a
> distribution nobody wanted.

## Expected tokens per round

A round emits a truncated geometric number of tokens, and high acceptance
matters far more than a long draft. Model each draft position as accepting
independently with probability $\alpha$. Let $N$ be the number of draft tokens
accepted before the first rejection, so $N \in \{0, 1, \dots, \gamma\}$ and:

$$
\Pr[N \ge i] = \alpha^{i}, \qquad i = 0, 1, \dots, \gamma
$$

A round emits $N + 1$ tokens in every case: $N$ accepted plus one correction on
a rejection, or $\gamma$ accepted plus the bonus token on full acceptance. Sum
the tail probabilities, using $\mathbb{E}[N] = \sum_{i \ge 1} \Pr[N \ge i]$:

$$
\boxed{\hla{\mathbb{E}[N + 1]} = \sum_{i=0}^{\gamma} \alpha^{i}
= \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}}
$$

At $\alpha = 1$ the geometric sum degenerates, and the value is $\gamma + 1$.
The following table gives $\hla{\mathbb{E}[N + 1]}$ for a few settings:

| $\alpha$ | $\gamma = 3$ | $\gamma = 5$ | $\gamma = 8$ |
|---|---|---|---|
| 0.5 | 1.88 | 1.97 | 2.00 |
| 0.7 | 2.53 | 2.94 | 3.20 |
| 0.9 | 3.44 | 4.69 | 6.13 |

Two things follow:

- **Acceptance beats length.** At $\alpha = 0.5$, going from $\gamma = 3$ to
  $\gamma = 8$ gains 6%.
- **Returns to $\gamma$ saturate.** One rejection discards everything after
  it, so the series converges to $1/(1-\alpha)$ no matter how long the draft.

> [!NOTE] The independence assumption is optimistic
> In practice acceptance decays with position, because draft token 5 is
> conditioned on four earlier draft tokens that might already have drifted from
> what the target would have produced. Treat $\alpha$ as an average over
> positions, and expect the measured $\mathbb{E}[N+1]$ to fall slightly below
> the formula.

## Net speedup, and the optimal draft length

The draft isn't free, so the speedup divides the tokens a round yields by what
the round costs. Let $c$ be the cost of one draft forward pass as a fraction of
one target forward pass. A round costs one target pass plus $\gamma$ draft
passes, $\hlb{1 + \gamma c}$ target-pass units, and produces
$\hla{\mathbb{E}[N+1]}$ tokens. Plain decoding produces one token per target
pass, so:

$$
\boxed{\operatorname{speedup}(\alpha, \gamma, c)
= \frac{\hla{1 - \alpha^{\gamma+1}}}{(\hla{1 - \alpha})(\hlb{1 + \gamma c})}}
$$

> [!EXAMPLE] The lab's two assertions
> At $\alpha = 0.8$, $\gamma = 4$, $c = 0.05$, the numerator is
> $1 - 0.8^5 = 0.67232$, so the speedup is $0.67232/(0.2 \times 1.2) = 2.80$.
>
> Raise the draft cost to $c = 0.5$ and the denominator becomes
> $0.2 \times 3 = 0.6$, giving 1.12. An expensive draft destroys the benefit
> regardless of how good it is.

### Where the optimum sits

The optimal $\gamma$ has no closed form, but it solves in a line of arithmetic.
Treat $\gamma$ as continuous and write $A = \alpha^{\gamma+1}$ and
$L = \ln(1/\alpha)$. The derivative of the numerator is $AL$, and setting the
quotient's derivative to zero gives $AL\,(1 + \gamma c) = (1 - A)\,c$, which
rearranges to:

$$
\alpha^{\gamma+1}\big(L + c + Lc\,\gamma\big) = c
$$

At $c = 0.05$, the solutions are as follows:

| $\alpha$ | Optimal $\gamma$ | Speedup there |
|---|---|---|
| 0.5 | 3 | 1.63 |
| 0.7 | 5 to 6 | 2.35 |
| 0.8 | 8 | 3.09 |
| 0.9 | 13 | 4.67 |

Two structural facts are visible in the table:

- **The optimal $\gamma$ rises steeply with $\alpha$.** The whole cost of a
  long draft is the tokens discarded after a rejection, and a high $\alpha$
  makes rejections rare.
- **The speedup is bounded by $1/(1-\alpha)$**, even with a free draft. At
  $\alpha = 0.8$ you can never beat 5 times.

> [!TIP] You don't need to tune the draft length finely
> The optimum is flat near its peak: at $\alpha = 0.7$ the values at
> $\gamma = 5$ and $\gamma = 6$ agree to three decimal places. Getting $\gamma$
> within a factor of two of the optimum captures almost all of the benefit.

### The ridge point caps it too

The second ceiling comes from the roofline. The verification pass carries
$B(\gamma + 1)$ tokens at batch size $B$, and the MLP's arithmetic intensity is
roughly the token count. Verification is free only while:

$$
B(\gamma + 1) \lesssim 161
$$

At $B = 1$ that allows $\gamma \le 160$, so the cap never binds and the
acceptance-rate argument decides everything. At $B = 32$ it allows
$\gamma \le 4$. Past that, every extra draft token costs real compute time: the
"1" in the cost $\hlb{1 + \gamma c}$ becomes $B(\gamma+1)/161$.

## Choosing a draft

The best draft is cheap and agrees with the target. The four common options
trade those two properties as follows:

| Draft | Cost $c$ | Acceptance | Exact? |
|---|---|---|---|
| Small model, same family | 0.022 at the bandwidth floor | 0.6 to 0.7 | Yes |
| Multi-token prediction head | about 0.016 | 0.7 to 0.85 | Yes |
| N-gram or prompt lookup | 0 | Low open-ended, high when quoting input | Yes |
| Medusa heads | No sequential draft latency | Weaker per position, offset by a tree | No, as published |

**A small model from the same family.** The requirement is a shared tokenizer
and similar training data, so the distributions agree. Qwen3-0.6B drafting for
this 27B model is roughly the right ratio. At bfloat16 it's about 1.2 GB of
weights, so its memory floor is $1.2/1275 = 0.94$ ms against the target's 42.2
ms, giving:

$$
c = \frac{0.94}{42.2} = 0.022
$$

That's 45 times cheaper, comfortably inside the usual rule that the draft must
be at least 15 to 20 times cheaper. In practice $c$ is higher than the
bandwidth floor suggests, because a draft pass is a few hundred small kernel
launches, and launch overhead doesn't shrink with the model.

**Self-speculation with a multi-token prediction head.** This model ships one:
`mtp_num_hidden_layers` is 1 in the config. It's a single extra layer trained to
predict the token *after* next from a hidden state the target already computed,
so its cost is one layer out of 64: under 2%, or $c \approx 0.016$.

Because it was trained jointly with the model, its acceptance rate is high:
typically 0.7 to 0.85, against 0.6 to 0.7 for a separate small model. That's a
cheaper draft, higher acceptance, and no second model to load or keep resident.
==On this hardware it's the strongest option available.==

**N-gram and prompt-lookup drafting.** This option uses no model at all. Take
the last few generated tokens, search the prompt and the generated text so far
for a matching n-gram, and propose whatever followed it. Because there's no
draft forward pass, $c = 0$ and the speedup collapses to:

$$
\operatorname{speedup} = \hla{\mathbb{E}[N+1]}
$$

So *any* acceptance rate above zero is a win, and enabling it never costs you
anything. Acceptance is low on open-ended generation and high on anything that
quotes its input: summarization, RAG, code editing, and structured output.
Enable it unconditionally and combine it with a model draft.

**Medusa-style multi-head drafting.** Attach $K$ small heads to the target's
final hidden state $h_t$, where head $k$ predicts the token at position
$t + k + 1$. The heads run in parallel on a hidden state the target already
produced, so there's no sequential draft latency at all.

The catch is that the heads are conditionally independent given $h_t$: head 2
doesn't know what head 1 predicted. So they don't produce a chain; they produce
a *tree*. Take the top few candidates from each head, and every path through
the tree is a candidate continuation.

Verification uses a tree attention mask, so each candidate token attends only
to its own ancestors, and one forward pass checks every path at once. The tree
compensates for the weaker per-position prediction by trying several
continuations simultaneously.

> [!WARNING] Medusa as published isn't exact
> It uses a typical acceptance criterion instead of the exact rejection rule
> proved earlier, so it doesn't preserve the target distribution exactly.
> That's a deliberate trade. Make it knowingly, not by accident.

## Batching makes it worse

At batch 1, speculative decoding gives 2 to 3 times. At batch 32 it often gives
nothing, for two reasons that compound.

**The idle compute is gone.** Arithmetic intensity rises with the number of
tokens in the pass. At batch 32 a plain decode step already carries 32 tokens,
and a speculative step with $\gamma = 4$ carries 160, right at the ridge point
of 161. Beyond that, the extra tokens aren't free; you're buying them with
compute time instead of with idle bandwidth.

**Ragged acceptance wastes the difference.** Every sequence in the batch is
verified over $\gamma + 1$ positions, but each advances by its own $N_i + 1$.
==The pass pays for the maximum and delivers the average.==

> [!EXAMPLE] A third of the verification work is discarded
> At $\alpha = 0.8$ and $\gamma = 4$, the efficiency is:
>
> $$
> \frac{\mathbb{E}[N+1]}{\gamma+1} = \frac{3.36}{5} = 67\%
> $$
>
> Below the ridge point that discarded third is free. Above it, it's 33% of
> your compute.

Production engines enable speculation adaptively: on at low batch size, off at
high, with the threshold set by $B(\gamma+1)$ against the ridge point. That's
the correct behavior, so implement it instead of treating speculation as a
global switch.

## Rolling back a rejected token

A rejection means tokens $i+1$ through $\gamma$ never happened. Every piece of
per-sequence state they touched has to be undone, and this model has two kinds.

### The 16 full-attention layers are easy

Keys and values were appended at slots the scheduler assigned, so undoing them
is a length rollback. Set the sequence length back to prefix plus $i + 1$, and
free any paged blocks that only rejected tokens occupied. The block table
shortens and the blocks return to the free list, machinery
[the paged attention chapter](/c/15-paged-attention) already built.

You don't even have to zero the stale entries. Attention masks by
`context_len`, so nothing beyond the new length is ever read, and the next
token overwrites the slot.

### The 48 linear-attention layers are the wrinkle

Their state is overwritten in place. After $\gamma+1$ tokens the state has
advanced $\gamma+1$ times, and there's no earlier version to return to. Of the
obvious approaches, one is impossible, one is hopeless, one is expensive, and
one is right.

**You can't defer them.** The obvious idea is to run the recurrent layers only
once the accepted length is known. That doesn't work, because the layer types
interleave. Every fourth layer is full attention, so layers 0, 1, and 2 are
linear and their outputs are layer 3's input. They must process the
speculative tokens; the only question is where their state ends up.

**Inverting the update is exact and numerically hopeless.** The gated delta
update is invertible in closed form, so you could run the recurrence backwards.
Don't. Every backward step multiplies by $1/\alpha_t > 1$, where $\alpha_t$ is
the gate, not the acceptance rate, so rounding error is amplified instead of
damped. Over $\gamma = 8$ steps the amplification is $\prod_t 1/\alpha_t$,
which is $0.9^{-8} = 2.3$ for a slow-forgetting head with $\alpha_t \approx 0.9$
and $2^{8} = 256$ for a fast-forgetting head with $\alpha_t \approx 0.5$. In
bfloat16, with 8 mantissa bits, the second case destroys the state completely.

> [!DEEPDIVE] The closed-form inverse
> [The gated delta rule chapter](/c/06-gated-delta-rule)'s update rearranges
> into an affine map on the state:
>
> $$
> S_t = \alpha_t\big(I - \beta_t k_t k_t^{\top}\big) S_{t-1}
> + \beta_t\, k_t v_t^{\top}
> $$
>
> Since the keys are L2-normalized, $k_t^{\top} k_t = 1$, and the
> Sherman-Morrison formula gives the inverse in closed form:
>
> $$
> \big(I - \beta_t k_t k_t^{\top}\big)^{-1}
> = I + \frac{\beta_t}{1 - \beta_t}\,k_t k_t^{\top}
> $$
>
> On top of the $1/\alpha_t$ amplification, the $\beta_t/(1-\beta_t)$ term
> diverges as $\beta_t \to 1$.

**Snapshot and restore works, and you can price it.** Copy the state before the
round and restore it on rejection. The state is 147.8 MiB per sequence, and a
copy reads and writes it:

$$
t_{\text{snapshot}} = \frac{2 \times 155 \times 10^{6}}{1275 \times 10^{9}}
= 0.24 \text{ ms}
$$

At batch 1 that's 0.6% of a 42.2 ms step, which is cheap. At batch 32 it's
4.6 GiB of extra memory and this much copying per round:

$$
32 \times 0.243 = 7.8 \text{ ms}
$$

That's roughly 19% of the step, taken out of memory you wanted for KV cache.
The cost scales with batch size while the benefit shrinks with it, which is
another reason speculation and large batches don't mix.

**The chunked form avoids the problem entirely.** This is the design to reach
for, and it reuses machinery the gated delta rule chapter already built.

> [!KEY] Treat the speculative tokens as one chunk
> The chunked delta rule computes every output in a chunk from the chunk's
> *starting* state plus intra-chunk terms, and applies the state update only at
> the chunk boundary. So compute all $\gamma + 1$ outputs against the committed
> state without mutating it. Once the accepted length $i$ is known, apply the
> update for the first $i + 1$ tokens only.

Nothing has to be snapshotted, because nothing was overwritten. The commit is an
update over an $i+1$-token chunk, which is work you'd have done anyway. The only
cost is that the chunk kernel must accept a variable commit length, which is a
parameter, not a redesign.

## What goes wrong

These are the mistakes the lab and a real engine surface:

- **Continuing past the first rejection.** Draft tokens after a rejection were
  conditioned on a token that no longer exists. Accepting any of them breaks the
  proof, because $p$ and $q$ at those positions were computed for a prefix that
  didn't happen. Break at the first rejection.
- **Forgetting the bonus token.** On full acceptance the target's distribution
  at position $\gamma+1$ is already computed and costs nothing to sample from.
  Dropping it reduces $\mathbb{E}[N+1]$ by $\alpha^{\gamma}$, which at
  $\alpha = 0.9$ and $\gamma = 5$ is a 13% loss of speedup for no reason.
- **Not clamping the residual.** Negative entries survive into `multinomial`,
  which either raises or silently samples from something that isn't a
  distribution.
- **Mismatched sampling parameters between draft and target.** The acceptance
  rate collapses and the guarantee no longer applies to the distribution you
  wanted. Apply the same temperature and truncation to both before comparing.
- **Leaving rejected tokens in the KV cache.** The next step attends to tokens
  that were never generated. It's completely silent, and it corrupts everything
  after it.
- **Leaving the recurrent state advanced.** The same failure, harder to see,
  because there's no length to inspect. The symptom is output that degrades
  over a long generation while short generations look fine: the state
  accumulates the effect of every rejected token.

> [!RECAP]
> - Decode is memory bound, so verifying $\gamma + 1$ tokens costs one weight
>   read.
> - Accept with probability $\min(1, p/q)$ and resample from $(p - q)_{+}$ on
>   rejection; the output is exactly $p$, for any draft.
> - $\alpha = 1 - D_{\text{TV}}(p, q)$, and a round yields
>   $(1 - \alpha^{\gamma+1})/(1 - \alpha)$ tokens at a cost of $1 + \gamma c$.
> - The speedup is capped by $1/(1-\alpha)$ and by $B(\gamma+1) \lesssim 161$,
>   so turn speculation off at large batch sizes.
> - Roll back the KV cache by length; for the recurrent state, verify as one
>   chunk and commit only the accepted prefix.

## Check your understanding

> [!QUESTION] Why does the residual distribution use $\max(0, p - q)$ instead of $|p - q|$?
> Because the rejection only needs to restore the mass the draft *failed* to
> deliver. Where $q(y) > p(y)$ the draft over-proposed $y$, and the accept step
> already discards the excess by accepting with probability $p(y)/q(y) < 1$.
> Where $p(y) > q(y)$ the draft under-proposed, and that shortfall, exactly
> $(p-q)_+$, is what the residual must supply. The proof shows those shortfalls
> sum to the rejection probability, which is what makes the
> arithmetic close.

> [!QUESTION] Your draft has an acceptance rate of 0.9 and costs 5% of the target. Is $\gamma = 4$ a good choice?
> It's leaving speedup on the table. The formula gives
> $(1 - 0.9^5)/(0.1 \times 1.2) = 3.41$ times, while the optimum at
> $\gamma = 13$ gives 4.67 times. With $\alpha$ that high, rejections are rare
> enough that a long draft rarely wastes anything, so the optimal $\gamma$ is
> much larger than the usual 4 or 5. Check the ridge-point cap too: at batch 1,
> $\gamma = 13$ puts 14 tokens in the verification pass, far below 161, so it's
> still free.

> [!QUESTION] Speculation gives 2.4 times at batch 1 and 1.0 times at batch 32. Is something broken?
> No, that's the expected behavior. At batch 32 with $\gamma = 4$ the
> verification pass carries $32 \times 5 = 160$ tokens, right at the ridge point
> of 161, so the extra tokens are no longer riding on idle bandwidth. On top of
> that, ragged acceptance discards a third of the verification work, which was
> free below the ridge point and isn't free above it. Switch speculation off
> above a batch-size threshold instead of trying to fix it.

> [!QUESTION] Why can't you rerun the linear-attention layers after you know the accepted length?
> Because the layer types interleave: three linear layers then one
> full-attention layer, repeating. The linear layers' outputs are the
> full-attention layers' inputs, so they must run during verification; you
> can't verify anything without them. What you can do is keep their state
> update out of the verification pass, which is what the chunked form gives you
> for free.

## Lab

> [!TRY]
> Build the accept-reject rule and the speedup arithmetic. Passing means the
> arithmetic checks hold and, over 40,000 rounds with a deliberately poor
> draft, the first emitted token's distribution is within 0.02 total variation
> distance of the target.

Implement the following functions:

- `accept_probability`, returning $\min(1, p(x)/q(x))$.
- `residual_distribution`, returning the clamped and normalized positive part
  of $p - q$, with a fallback to $p$ when it sums to zero.
- `verify`, running the accept-reject walk and returning the accepted tokens
  plus the count of draft tokens accepted.
- `expected_tokens_per_round`, as the truncated geometric sum.
- `net_speedup`, dividing it by $1 + \gamma c$.

The harness checks the arithmetic first: acceptance of 1.0 for a token the
target prefers, 0.5 for the lab's mismatched pair, 4.6856 expected tokens at
$\alpha = 0.9$ and $\gamma = 5$, and a speedup above 2 for a cheap draft and
below 1.2 for an expensive one.

Then it runs the statistical test that matters. Over 40,000 rounds with a
deliberately poor draft over a 16-token vocabulary, it compares the empirical
distribution of the first emitted token against the target. It requires a total
variation distance under 0.02, and closer to the target than to the draft. It
also checks that a draft identical to the target is accepted more than 99% of
the time, and that a round always yields between 1 and $\gamma + 1$ tokens.

==That statistical test is the one to watch.== Every plausible-looking shortcut
in the rejection rule passes the arithmetic checks and fails this one.

## Further reading

- [Fast inference from transformers via speculative decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating large language model decoding with speculative sampling](https://arxiv.org/abs/2302.01318)
- [Medusa: simple LLM inference acceleration framework with multiple decoding heads](https://arxiv.org/abs/2401.10774)
- [EAGLE: speculative sampling requires rethinking feature uncertainty](https://arxiv.org/abs/2401.15077) — drafting on the target's own feature sequence rather than on tokens.
- [SpecInfer: accelerating generative LLM serving with tree-based speculative inference](https://arxiv.org/abs/2305.09781) — the tree verification Medusa relies on.
