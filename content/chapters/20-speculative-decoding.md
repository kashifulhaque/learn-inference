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
  - Choose a draft: a small model, a multi-token prediction head, n-gram lookup, or Medusa heads.
  - Roll back a rejected token from both a paged KV cache and a recurrent state.
lab: 20-spec-decode
---

# Speculative decoding

Decode at batch 1 uses about 0.7% of an A100's arithmetic. Everything else is
idle while the memory system reads 53.8 GB of weights. Speculative decoding
spends that idle compute on tokens.

What makes it more than a heuristic is that it is exact. The accept-reject rule
below produces samples distributed *identically* to sampling from the target
model directly — not approximately, not in the limit, but for every token. That
proof is the whole point of this chapter, and it is the part most treatments
skip. Without it, speculation is a quality risk you cannot measure; with it, it
is a pure speed knob you can turn on by default.

## Before you start

**Sampling from a categorical distribution.** A distribution $p$ over a
vocabulary $V$ assigns $p(x) \ge 0$ with $\sum_{x \in V} p(x) = 1$. Chapter 11
covers how temperature, top-$k$, and top-$p$ transform the logits into $p$.
Everything here operates on $p$ after those transforms.

**Total variation distance.** For two distributions on the same support,

$$
D_{\text{TV}}(p, q) = \frac{1}{2}\sum_{x \in V} \lvert p(x) - q(x) \rvert
$$

It lands in $[0, 1]$, is 0 when the distributions are identical, and turns out
to be exactly what the acceptance rate depends on. The lab measures it.

**The roofline from chapter 10.** Decode at batch 1 has an arithmetic intensity
around 1 FLOP per byte against a ridge point of 161, so its time is bytes over
bandwidth. That gap is the resource speculation spends.

**The paged KV cache from chapter 15 and the recurrent state from chapter 6.**
Both are per-sequence state that a rejected token must not be allowed to
advance. The last section of this chapter is about exactly that.

**Notation.** Write $\gamma$ for the number of tokens the draft proposes in one
round and $\alpha$ for the probability an individual draft token is accepted.
The lab calls $\gamma$ `k`.

## Why verifying many tokens costs one weight read

A forward pass on one token and a forward pass on five tokens take almost the
same time. Put numbers on it.

The weights are 53.8 GB, read once per forward pass regardless of how many
tokens are in it. At the measured copy bandwidth of 1275 GB/s,

$$
t_{\text{memory}} = \frac{53.8 \times 10^{9}}{1275 \times 10^{9}}
= 42.2 \text{ ms}
$$

The arithmetic for $T$ tokens at batch 1 is about $2 \times 26.9 \times 10^{9}
\times T$ FLOPs. At $T = 5$:

$$
t_{\text{compute}} = \frac{2.69 \times 10^{11}}{312 \times 10^{12}}
= 0.86 \text{ ms}
$$

The arithmetic intensity is $2.69 \times 10^{11} / 53.8 \times 10^{9} = 5.0$
FLOPs per byte, against a ridge point of 161. The pass is memory bound by a
factor of 32, so the compute time hides entirely inside the memory time and

$$
t(1 \text{ token}) = t(5 \text{ tokens}) = 42.2 \text{ ms}
$$

So: guess the next few tokens cheaply, then verify all of them in one forward
pass. Every guess that survives is a token you got for free. This only works
because decode is memory bound, which is why speculation and batching — the
other way to fill idle compute — fight each other. That comes back at the end.

## The algorithm

1. A cheap **draft** proposes $\gamma$ tokens autoregressively, each conditioned
   on the ones before it.
2. The **target** model runs one forward pass over all $\gamma$ draft tokens
   plus the current prefix, producing a distribution at each of the $\gamma + 1$
   positions.
3. Walk the draft tokens in order, accepting or rejecting each by the rule
   below.
4. On the first rejection, sample a corrected token from an adjusted
   distribution and discard everything after it.
5. If every draft token is accepted, sample a bonus token from the target's
   distribution at position $\gamma + 1$, which the same forward pass already
   produced.

A round yields between 1 and $\gamma + 1$ tokens. Even total rejection yields
one, so speculation never loses ground — the worst case is one token per target
pass, which is what you had before.

## The acceptance rule and its proof

Fix one position. Let $p$ be the target's distribution there and $q$ the
draft's. The draft proposed token $x \sim q$.

**The rule.** Draw $u \sim \mathcal{U}(0,1)$ and accept $x$ if

$$
u \le \min\!\left(1, \frac{p(x)}{q(x)}\right)
$$

On rejection, draw the replacement from the **residual distribution**

$$
p'(y) = \frac{\big(p(y) - q(y)\big)_{+}}{\sum_{z \in V}\big(p(z) - q(z)\big)_{+}}
$$

where $(a)_{+} = \max(0, a)$.

**The claim.** The token this procedure outputs is distributed exactly as $p$.

### The proof

Four steps. The whole thing rests on one identity, stated first.

**Identity.** For any reals $a, b$,

$$
(a - b)_{+} = a - \min(a, b)
$$

If $a \ge b$ both sides are $a - b$; if $a < b$ both sides are 0.

**Step 1: the probability that $x$ is drawn and accepted.** The draft draws $x$
with probability $q(x)$ and accepts with probability $\min(1, p(x)/q(x))$, so

$$
\Pr[\text{draw } x \text{ and accept}]
= q(x)\min\!\left(1, \frac{p(x)}{q(x)}\right)
= \min\big(q(x), p(x)\big)
$$

The $q(x)$ cancels cleanly in both branches of the minimum, which is why the
rule is stated as a ratio.

**Step 2: the probability of rejection.** Sum step 1 over the vocabulary:

$$
\Pr[\text{accept}] = \sum_{y \in V} \min\big(p(y), q(y)\big) =: \alpha,
\qquad
\Pr[\text{reject}] = 1 - \alpha
$$

**Step 3: the residual's normalizer is exactly the rejection probability.**
Apply the identity term by term and sum:

$$
\sum_{y} \big(p(y) - q(y)\big)_{+}
= \sum_{y} p(y) - \sum_{y} \min\big(p(y), q(y)\big)
= 1 - \alpha
$$

This is the step that makes the whole thing work, and it is not a coincidence —
the residual holds precisely the probability mass the draft failed to deliver.

**Step 4: combine.** The output is $y$ either because $y$ was drafted and
accepted, or because something was rejected and the residual produced $y$:

$$
\Pr[\text{output} = y]
= \min\big(p(y), q(y)\big) + (1-\alpha)\,p'(y)
$$

By step 3, $(1-\alpha)\,p'(y) = (p(y) - q(y))_{+}$, so

$$
\Pr[\text{output} = y]
= \min\big(p(y), q(y)\big) + p(y) - \min\big(p(y), q(y)\big)
= p(y)
$$

which is the claim. The argument holds for every $q$, including a bad one:
speculation with a terrible draft is slow, never wrong.

### The acceptance rate is one minus total variation distance

Step 2 defined $\alpha = \sum_y \min(p(y), q(y))$. Using
$\min(a,b) = \tfrac{1}{2}(a + b - |a - b|)$ and summing,

$$
\alpha = \frac{1}{2}\left(1 + 1 - \sum_y \lvert p(y) - q(y)\rvert\right)
= 1 - D_{\text{TV}}(p, q)
$$

The acceptance rate is not a tunable parameter. It is a fixed property of how
close the draft's distribution is to the target's, and the only way to raise it
is to make the draft agree with the target more often. That is why "use a
smarter sampler" never helps and "use a draft from the same model family" always
does.

### In code

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

Two details cause most bugs, and the lab tests both. The residual must be
clamped at zero *before* normalizing, or negative entries survive and
`multinomial` either raises or samples nonsense. And when $p = q$ the residual
sums to zero, so normalizing divides by zero — the branch is never taken,
because $\Pr[\text{reject}] = 0$ in that case, but the code still has to not
produce a NaN.

A third detail is not a code bug but a design one: $p$ and $q$ must come from
the *same* sampling parameters. If the target applies temperature 0.7 and
top-$p$ 0.95 and the draft samples raw, the ratio $p(x)/q(x)$ compares two
different objects, the acceptance rate collapses, and the guarantee you proved
above is a guarantee about a distribution nobody wanted.

## Expected tokens per round

Model each draft position as accepting independently with probability $\alpha$.
Let $N$ be the number of draft tokens accepted before the first rejection, so
$N \in \{0, 1, \dots, \gamma\}$ and

$$
\Pr[N \ge i] = \alpha^{i}, \qquad i = 0, 1, \dots, \gamma
$$

A round emits $N + 1$ tokens in every case: $N$ accepted plus one correction on
a rejection, or $\gamma$ accepted plus the bonus token on full acceptance. Using
$\mathbb{E}[N] = \sum_{i \ge 1} \Pr[N \ge i]$,

$$
\mathbb{E}[N + 1] = \sum_{i=0}^{\gamma} \alpha^{i}
= \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
$$

with the value $\gamma + 1$ at $\alpha = 1$, where the geometric sum degenerates.

| $\alpha$ | $\gamma = 3$ | $\gamma = 5$ | $\gamma = 8$ |
|---|---|---|---|
| 0.5 | 1.88 | 1.97 | 2.00 |
| 0.7 | 2.53 | 2.94 | 3.24 |
| 0.9 | 3.44 | 4.69 | 6.13 |

Two things follow. High acceptance is worth far more than a long draft: at
$\alpha = 0.5$, going from $\gamma = 3$ to $\gamma = 8$ gains 6%. And the
returns to $\gamma$ saturate, because one rejection discards everything after
it — the series converges to $1/(1-\alpha)$ no matter how long the draft.

The independence assumption is optimistic. In practice acceptance decays with
position, because draft token 5 is conditioned on four earlier draft tokens that
may already have drifted from what the target would have produced. Treat
$\alpha$ as an average over positions and expect the measured
$\mathbb{E}[N+1]$ to fall slightly below the formula.

## Net speedup, and the optimal draft length

The draft is not free. Let $c$ be the cost of one draft forward pass as a
fraction of one target forward pass. A round costs one target pass plus $\gamma$
draft passes:

$$
\text{cost} = 1 + \gamma c
$$

target-pass units, and produces $\mathbb{E}[N+1]$ tokens. Plain decoding
produces one token per target pass, so

$$
\operatorname{speedup}(\alpha, \gamma, c)
= \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(1 + \gamma c)}
$$

Check it against the lab's assertions. At $\alpha = 0.8$, $\gamma = 4$,
$c = 0.05$: the numerator is $1 - 0.8^5 = 0.67232$, so
$0.67232/(0.2 \times 1.2) = 2.80$. Raise the draft cost to $c = 0.5$ and the
denominator becomes $0.2 \times 3 = 0.6$, giving 1.12. An expensive draft
destroys the benefit regardless of how good it is.

### Where the optimum sits

Treat $\gamma$ as continuous and differentiate. Writing $A = \alpha^{\gamma+1}$
and $L = \ln(1/\alpha)$, the derivative of the numerator is $AL$, and setting
the quotient's derivative to zero gives

$$
AL\,(1 + \gamma c) = (1 - A)\,c
$$

which rearranges to

$$
\alpha^{\gamma+1}\big(L + c + Lc\,\gamma\big) = c
$$

There is no closed form, but it solves in a line of arithmetic. At $c = 0.05$:

| $\alpha$ | Optimal $\gamma$ | Speedup there |
|---|---|---|
| 0.5 | 3 | 1.63 |
| 0.7 | 5 to 6 | 2.35 |
| 0.8 | 8 | 3.09 |
| 0.9 | 13 | 4.67 |

The optimum is flat near its peak — at $\alpha = 0.7$ the values at $\gamma = 5$
and $\gamma = 6$ agree to three decimal places — so there is no need to tune
$\gamma$ finely. Getting it within a factor of two of the optimum captures
almost all of the benefit.

Two structural facts are visible in the table. The optimal $\gamma$ rises
steeply with $\alpha$, because the whole cost of a long draft is the tokens
discarded after a rejection, and a high $\alpha$ makes rejections rare. And the
speedup is bounded above by $1/(1-\alpha)$ even with a free draft: at
$\alpha = 0.8$ you can never beat 5 times, no matter what.

### The ridge point caps it too

There is a second ceiling, and it comes from chapter 10. The verification pass
carries $B(\gamma + 1)$ tokens at batch size $B$, and the MLP's arithmetic
intensity is roughly the token count. Verification is free only while

$$
B(\gamma + 1) \lesssim 161
$$

At $B = 1$ that allows $\gamma \le 160$, so the cap never binds and the
acceptance-rate argument decides everything. At $B = 32$ it allows $\gamma \le
4$, and past that every extra draft token costs real compute time — the "1" in
the cost formula becomes $B(\gamma+1)/161$.

## Choosing a draft

**A small model from the same family.** The requirement is a shared tokenizer
and similar training data, so the distributions agree. Qwen3-0.6B drafting for
this 27B model is roughly the right ratio: at bfloat16 it is about 1.2 GB of
weights, so its memory floor is $1.2/1275 = 0.94$ ms against the target's 42.2
ms, giving

$$
c = \frac{0.94}{42.2} = 0.022
$$

That is 45 times cheaper, comfortably inside the usual rule that the draft must
be at least 15 to 20 times cheaper. In practice $c$ is higher than the
bandwidth floor suggests, because a draft pass is a few hundred small kernel
launches and launch overhead does not shrink with the model.

**Self-speculation with a multi-token prediction head.** This model ships one:
`mtp_num_hidden_layers` is 1 in the config. It is a single extra layer trained
to predict the token *after* next from a hidden state the target already
computed, so its cost is one layer out of 64 — under 2%, or $c \approx 0.016$.
And because it was trained jointly with the model, its acceptance rate is high:
typically 0.7 to 0.85, against 0.6 to 0.7 for a separate small model. Cheaper
draft, higher acceptance, and no second model to load or keep resident. On this
hardware it is the strongest option available.

**N-gram and prompt-lookup drafting.** No model at all. Take the last few
generated tokens, search the prompt and the generated text so far for a matching
n-gram, and propose whatever followed it. Since there is no draft forward pass,
$c = 0$ and the speedup collapses to

$$
\operatorname{speedup} = \mathbb{E}[N+1]
$$

so *any* acceptance rate above zero is a win and there is no configuration in
which enabling it costs you anything. Acceptance is low on open-ended
generation and high on anything that quotes its input: summarization, RAG,
code editing, structured output. Enable it unconditionally and combine it with a
model draft.

**Medusa-style multi-head drafting.** Attach $K$ small heads to the target's
final hidden state $h_t$, where head $k$ predicts the token at position
$t + k + 1$. The heads run in parallel on a hidden state the target already
produced, so there is no sequential draft latency at all.

The catch is that the heads are conditionally independent given $h_t$: head 2
does not know what head 1 predicted. So they do not produce a chain, they
produce a *tree* — take the top few candidates from each head and every path
through the tree is a candidate continuation. Verification uses a tree attention
mask, so each candidate token attends only to its own ancestors, and one forward
pass checks every path at once. The tree compensates for the weaker
per-position prediction by trying several continuations simultaneously.

One caveat, given this chapter's emphasis: Medusa as published uses a typical
acceptance criterion rather than the exact rejection rule proved above, so it
does *not* preserve the target distribution exactly. That is a deliberate trade,
and it is one you should make knowingly rather than by accident.

## Batching makes it worse

At batch 1, speculative decoding gives 2 to 3 times. At batch 32 it often gives
nothing, for two reasons that compound.

**The idle compute is gone.** From chapter 10, arithmetic intensity rises with
the number of tokens in the pass. At batch 32 a plain decode step already
carries 32 tokens, and a speculative step with $\gamma = 4$ carries 160 — right
at the ridge point of 161. Beyond that, the extra tokens are no longer free;
you are buying them with compute time rather than with idle bandwidth.

**Ragged acceptance wastes the difference.** Every sequence in the batch is
verified over $\gamma + 1$ positions, but each advances by its own $N_i + 1$.
The pass pays for the maximum and delivers the average. At $\alpha = 0.8$ and
$\gamma = 4$ the efficiency is

$$
\frac{\mathbb{E}[N+1]}{\gamma+1} = \frac{3.36}{5} = 67\%
$$

so a third of the verification work is discarded. Below the ridge point that
third is free. Above it, it is 33% of your compute.

Production engines enable speculation adaptively: on at low batch size, off at
high, with the threshold set by $B(\gamma+1)$ against the ridge point. That is
the correct behavior, and it is worth implementing rather than treating
speculation as a global switch.

## Rolling back a rejected token

A rejection means tokens $i+1$ through $\gamma$ never happened. Every piece of
per-sequence state they touched has to be undone, and this model has two kinds.

### The 16 full-attention layers are easy

Keys and values were appended at slots the scheduler assigned. Undoing that is a
length rollback: set the sequence length back to prefix plus $i + 1$, and free
any paged blocks that only rejected tokens occupied. The block table shortens
and the blocks return to the free list — machinery chapter 15 already built.

You do not even have to zero the stale entries. Attention masks by
`context_len`, so nothing beyond the new length is ever read, and the next
token overwrites the slot.

### The 48 linear-attention layers are the wrinkle

Their state is overwritten in place. After $\gamma+1$ tokens the state has
advanced $\gamma+1$ times and there is no earlier version to return to. Three
approaches, of which one is wrong, one is expensive, and one is right.

**You cannot defer them.** The obvious idea — run the recurrent layers only once
the accepted length is known — does not work, because the layer types interleave.
Every fourth layer is full attention, so layers 0, 1, 2 are linear and their
outputs are layer 3's input. Skipping the linear layers during verification means
not running the model. They must process the speculative tokens; the only
question is where their state ends up.

**Inverting the update is exact and numerically hopeless.** Chapter 6's gated
delta update rearranges into an affine map on the state:

$$
S_t = \alpha_t\big(I - \beta_t k_t k_t^{\top}\big) S_{t-1}
+ \beta_t\, k_t v_t^{\top}
$$

Since the keys are L2-normalized, $k_t^{\top} k_t = 1$, and the
Sherman-Morrison formula gives the inverse in closed form:

$$
\big(I - \beta_t k_t k_t^{\top}\big)^{-1}
= I + \frac{\beta_t}{1 - \beta_t}\,k_t k_t^{\top}
$$

so you could run the recurrence backwards. Do not. Every backward step
multiplies by $1/\alpha_t > 1$, so rounding error is amplified rather than
damped. Over $\gamma = 8$ steps the amplification is $\prod_t 1/\alpha_t$, which
for a slow-forgetting head with $\alpha_t \approx 0.9$ is $0.9^{-8} = 2.3$ and
for a fast-forgetting head with $\alpha_t \approx 0.5$ is $2^{8} = 256$. In
bfloat16, with 8 mantissa bits, the second case destroys the state completely.
The $\beta_t/(1-\beta_t)$ term diverges as $\beta_t \to 1$ on top of that.

**Snapshot and restore works, and you can price it.** Copy the state before the
round and restore it on rejection. The state is 147.8 MiB per sequence, and a
copy reads and writes:

$$
t_{\text{snapshot}} = \frac{2 \times 155 \times 10^{6}}{1275 \times 10^{9}}
= 0.24 \text{ ms}
$$

At batch 1 that is 0.6% of a 42.2 ms step — cheap. At batch 32 it is 4.6 GiB of
extra memory and

$$
32 \times 0.24 = 7.8 \text{ ms}
$$

of copying per round, roughly 19% of the step, taken out of memory you wanted
for KV cache. The cost scales with batch size while the benefit shrinks with it,
which is another reason speculation and large batches do not mix.

**The chunked form avoids the problem entirely.** This is the design to reach
for, and it reuses machinery chapter 6 already built. The chunked delta rule
computes every output in a chunk from the chunk's *starting* state plus
intra-chunk terms, and only applies the state update at the chunk boundary. So
treat the $\gamma + 1$ speculative tokens as one chunk: compute all their
outputs against the committed state without mutating it, and once the accepted
length $i$ is known, apply the update for the first $i + 1$ tokens only.

Nothing has to be snapshotted, because nothing was overwritten. The commit is an
update over an $i+1$-token chunk, which is work you would have done anyway. The
only cost is that the chunk kernel must accept a variable commit length, which
is a parameter, not a redesign.

## What goes wrong

**Continuing past the first rejection.** Draft tokens after a rejection were
conditioned on a token that no longer exists. Accepting any of them breaks the
proof — the distributions $p$ and $q$ at those positions were computed for a
prefix that did not happen. Break at the first rejection.

**Forgetting the bonus token.** On full acceptance the target's distribution at
position $\gamma+1$ is already computed and costs nothing to sample from.
Dropping it reduces $\mathbb{E}[N+1]$ by $\alpha^{\gamma}$, which at
$\alpha = 0.9$ and $\gamma = 5$ is a 13% loss of speedup for no reason.

**Not clamping the residual.** Negative entries survive into `multinomial`,
which either raises or silently samples from something that is not a
distribution.

**Mismatched sampling parameters between draft and target.** The acceptance rate
collapses and the guarantee no longer applies to the distribution you wanted.
Apply the same temperature and truncation to both before comparing.

**Leaving rejected tokens in the KV cache.** The next step attends to tokens
that were never generated. Completely silent, and it corrupts everything after
it.

**Leaving the recurrent state advanced.** The same failure, harder to see,
because there is no length to inspect. The symptom is output that degrades over
a long generation while short generations look fine — the state accumulates the
effect of every rejected token.

## Check your understanding

**Why does the residual distribution use $\max(0, p - q)$ rather than
$|p - q|$?**

Because the rejection only needs to restore the mass the draft *failed* to
deliver. Where $q(y) > p(y)$ the draft over-proposed $y$, and the accept step
already discards the excess by accepting with probability $p(y)/q(y) < 1$.
Where $p(y) > q(y)$ the draft under-proposed, and that shortfall — exactly
$(p-q)_+$ — is what the residual must supply. Step 3 of the proof shows those
shortfalls sum to the rejection probability, which is what makes the arithmetic
close.

**Your draft has an acceptance rate of 0.9 and costs 5% of the target. Is
$\gamma = 4$ a good choice?**

It is leaving speedup on the table. The formula gives
$(1 - 0.9^5)/(0.1 \times 1.2) = 3.41$ times, while the optimum at $\gamma = 13$
gives 4.67 times. With $\alpha$ that high, rejections are rare enough that a
long draft rarely wastes anything, so the optimal $\gamma$ is much larger than
the usual 4 or 5. Check the ridge-point cap too: at batch 1, $\gamma = 13$ puts
14 tokens in the verification pass, far below 161, so it is still free.

**Speculation gives 2.4 times at batch 1 and 1.0 times at batch 32. Is
something broken?**

No, that is the expected behavior. At batch 32 with $\gamma = 4$ the
verification pass carries $32 \times 5 = 160$ tokens, right at the ridge point
of 161, so the extra tokens are no longer riding on idle bandwidth. On top of
that, ragged acceptance discards a third of the verification work, which was
free below the ridge point and is not free above it. Switch speculation off
above a batch-size threshold rather than trying to fix it.

**Why can't you rerun the linear-attention layers after you know the
accepted length?**

Because the layer types interleave: three linear layers then one full-attention
layer, repeating. The linear layers' outputs are the full-attention layers'
inputs, so they must run during verification — you cannot verify anything
without them. What you can do is keep their state update out of the verification
pass, which is what the chunked form gives you for free.

## Lab

Implement `accept_probability` returning $\min(1, p(x)/q(x))$,
`residual_distribution` returning the clamped and normalized positive part of
$p - q$ with a fallback to $p$ when it sums to zero, `verify` running the
accept-reject walk and returning the accepted tokens plus the count of draft
tokens accepted, `expected_tokens_per_round` as the truncated geometric sum, and
`net_speedup` dividing it by $1 + \gamma c$.

The harness checks the arithmetic — acceptance of 1.0 for a token the target
prefers, 0.5 for the lab's mismatched pair, 4.6856 expected tokens at
$\alpha = 0.9$ and $\gamma = 5$, speedup above 2 for a cheap draft and below 1.2
for an expensive one — and then runs the statistical test that matters. Over
40,000 rounds with a deliberately poor draft over a 16-token vocabulary, it
compares the empirical distribution of the first emitted token against the
target and requires a total variation distance under 0.02, and closer to the
target than to the draft. It also checks that a draft identical to the target is
accepted more than 99% of the time, and that a round always yields between 1 and
$\gamma + 1$ tokens.

That statistical test is the one to watch. Every plausible-looking shortcut in
the rejection rule passes the arithmetic checks and fails this one.

## Further reading

- [Fast inference from transformers via speculative decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating large language model decoding with speculative sampling](https://arxiv.org/abs/2302.01318)
- [Medusa: simple LLM inference acceleration framework with multiple decoding heads](https://arxiv.org/abs/2401.10774)
- [EAGLE: speculative sampling requires rethinking feature uncertainty](https://arxiv.org/abs/2401.15077) — drafting on the target's own feature sequence rather than on tokens.
- [SpecInfer: accelerating generative LLM serving with tree-based speculative inference](https://arxiv.org/abs/2305.09781) — the tree verification Medusa relies on.
