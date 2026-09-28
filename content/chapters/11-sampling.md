---
title: Sampling
slug: 11-sampling
part: "Part 3 — Making it fast"
summary: Turning 248,320 raw scores into one token — softmax and temperature from first principles, the truncations, the penalties, and the order that makes the knobs mean what they say.
minutes: 75
gpu: false
objectives:
  - Derive softmax with temperature and take its zero and infinite limits.
  - Show why subtracting the row maximum is mandatory in floating point.
  - Derive top-k, top-p, and min-p as truncations of a categorical distribution.
  - Say which pairs of sampling transforms commute and which don't.
  - Prove that the Gumbel-max trick samples from the right distribution.
  - Cost a full-vocabulary softmax in bytes, FLOPs, and kernel launches.
lab: 11-sampling
---

# Sampling

> [!TLDR]
> - The model ends with one raw score, a *logit*, per vocabulary entry. Softmax
>   turns the scores into probabilities, and temperature stretches or squeezes
>   the gaps between them to make the output more predictable or more varied.
> - Subtract the largest logit before you exponentiate. Without that step, a
>   single logit above 88.72 overflows float32 and turns the whole row into
>   `NaN`.
> - Top-k, top-p, and min-p all work the same way: they set the scores of
>   excluded tokens to $-\infty$ and let softmax share the probability among
>   the tokens that are left.
> - Order matters: penalties, then temperature, then top-k, min-p, and top-p.
>   Temperature after top-p leaves fewer choices than the user asked for.
> - Sampling moves almost no data, but a naive pipeline launches about twenty
>   tiny GPU functions per step, so production engines fuse it into one.

The forward pass ends with a vector of 248,320 numbers. The last linear layer,
the *LM head*, multiplies the final hidden state by a `(5120, 248320)` matrix
and produces one *logit* per vocabulary entry. Logits are unnormalized and
unbounded: they aren't probabilities, they aren't log-probabilities, and
nothing constrains their scale.

To generate text, you need three things from them:

1. A probability distribution, so that "how likely is this token" has an answer.
2. A way to trade fidelity against variety, because the maximum-probability
   token isn't always the token you want.
3. A draw, reproducible from a seed, cheap enough to run every step.

Each operation is small on its own. Composing them isn't: ==the same five
transforms in two different orders give two different distributions==, and
only one of them matches what the parameter names promise. This chapter builds
the three things in order, then works out the order, and ends with what the
whole pipeline costs on a GPU.

## Before you start

**Categorical distributions.** A categorical distribution over $V$ outcomes is a
vector $p \in \mathbb{R}^V$ with $p_i \ge 0$ and $\sum_{i=1}^{V} p_i = 1$. Here
$V = 248{,}320$, and outcome $i$ is "the next token is vocabulary entry $i$".
Sampling means drawing one index with those probabilities.

**Logits.** Write $z \in \mathbb{R}^V$ for the model's output.
[Chapter 0a](/c/00a-notation-and-prerequisites) covers the softmax that turns
$z$ into $p$. This chapter derives it again, with temperature, because the
derivation tells you where temperature is allowed to go.

**Odds and log-odds.** The odds of outcome $i$ against outcome $j$ are
$p_i / p_j$, and the log-odds are $\log(p_i/p_j)$. Adding a constant to one
logit shifts its log-odds by that constant, which makes log-odds the natural
unit for penalties.

**Floating point.** Chapter 0a has the full table of formats. This chapter
leans on two facts. A format's exponent field sets its range, the largest value
it can hold. bfloat16 and float32 share an 8-bit exponent, and float16 doesn't.

**PyTorch.** `topk`, `sort`, `cumsum`, `masked_fill`, `gather`, `scatter`, and
`multinomial`, all with an explicit `dim`. The reference implementation is
`engine/sampling.py`; read it alongside this chapter.

## From logits to a distribution

This section answers the first need, a probability distribution, and shows that
softmax is the only map from logits to probabilities that makes logit
differences into log-odds.

Start with what the answer looks like. In the lab's five-logit row, the top two
tokens have logits 3 and 2, and softmax gives them probabilities 0.6364 and
0.2341. The first is $0.6364 / 0.2341 = 2.72$ times as likely as the second,
which is $e^{1}$. A gap of one logit is a factor of $e$ in the odds, whatever
the logits' absolute values are.

That's no accident. The model was trained with a cross-entropy loss, so the
logits were fitted such that their *differences* carry the information. Adding
the same constant to every logit changes nothing: a bias in the last layer could
absorb it. So ask that logit differences *be* log-odds:

$$
\log \frac{p_i}{p_j} = z_i - z_j \quad \text{for every } i, j.
$$

Rearrange to $p_i = p_j e^{z_i - z_j}$, and sum over $i$ so that the left side
becomes 1:

$$
1 = \sum_{i=1}^{V} p_i = p_j e^{-z_j} \sum_{i=1}^{V} e^{z_i}.
$$

Solve for $p_j$:

$$
\boxed{\,p_j = \frac{e^{z_j}}{\sum_{i=1}^{V} e^{z_i}}\,}
$$

That's the softmax, and the derivation shows it's ==the *only* map with the
property you asked for==. It's positive and normalized without either being
imposed.

## Temperature

This section answers the second need, trading fidelity against variety, with one
knob that sharpens or flattens the distribution by scaling every log-odds.

Picture the two top tokens again, one logit apart, with odds of $e^{1} = 2.72$
to 1. Divide both logits by 2 and the gap halves to 0.5, so the odds fall to
$e^{0.5} = 1.65$ to 1: the runner-up gets more of a chance. Divide by 0.5
instead and the gap doubles to 2, so the odds rise to $e^{2} = 7.39$ to 1: the
favorite pulls ahead.

That divisor is the temperature $\hla{T} > 0$, and it's applied to the logits
before the softmax:

$$
p_i(\hla{T}) = \frac{e^{z_i / \hla{T}}}{\sum_{j=1}^{V} e^{z_j / \hla{T}}}.
$$

The name comes from the Boltzmann distribution in physics, which has the same
form. Apply the log-odds identity, and temperature divides every log-odds by
$\hla{T}$:

$$
\log \frac{p_i(\hla{T})}{p_j(\hla{T})} = \frac{z_i - z_j}{\hla{T}}.
$$

> [!INTUITION]
> Temperature is a zoom on the gaps between logits. Below 1, it stretches
> every gap and sharpens the distribution. Above 1, it compresses every gap and
> flattens it. At exactly 1, it does nothing.

### The two limits

The two extremes of temperature are greedy decoding and uniform noise:

- **As $\hla{T} \to 0^{+}$**, $p(\hla{T})$ becomes a point mass on the argmax.
  If $a$ logits tie for the maximum, each gets $1/a$.
- **As $\hla{T} \to \infty$**, $p_i(\hla{T}) \to 1/V$. The distribution is
  uniform over the whole vocabulary, and the model's output stops mattering.

In between, the distribution spreads out steadily. Its *entropy*, a measure of
how spread out it is, rises monotonically with $\hla{T}$. It goes from 0 at the
greedy end to $\log V = \log 248{,}320 \approx 12.42$ *nats*, entropy's
natural-log unit, at the uniform end.

The engine special-cases `temperature == 0` to a plain `argmax` instead of
dividing by something near zero. ==The limit exists, but the float arithmetic
on the way to it doesn't survive.==

> [!DEEPDIVE] Deriving the two limits
> Let $m = \max_j z_j$, and let $A = \{\, i : z_i = m \,\}$ be the set of
> maximizers, with $a = |A|$. Shift invariance, proved in the next section, lets
> you remove $m$ from the ratio:
>
> $$
> p_i(\hla{T}) = \frac{e^{(z_i - m)/\hla{T}}}{\sum_{j=1}^{V} e^{(z_j - m)/\hla{T}}}.
> $$
>
> Every exponent is now at most 0.
>
> As $\hla{T} \to 0^{+}$: for $i \notin A$, the exponent $(z_i - m)/\hla{T} \to
> -\infty$, so that term goes to 0. For $i \in A$, the exponent is exactly 0, so
> that term is 1. The denominator tends to $a$, and
>
> $$
> \lim_{\hla{T} \to 0^{+}} p_i(\hla{T}) = \begin{cases} 1/a & i \in A \\ 0 & i \notin A. \end{cases}
> $$
>
> With a unique maximizer, $a = 1$, and the limit is a point mass on the argmax.
>
> As $\hla{T} \to \infty$: every exponent $(z_j - m)/\hla{T} \to 0$, so every
> term tends to 1, the denominator tends to $V$, and
>
> $$
> \lim_{\hla{T} \to \infty} p_i(\hla{T}) = \frac{1}{V}.
> $$

### Temperature acts on logits, not probabilities

Two operations get confused here, and only one of them is temperature.

Raising probabilities to the power $1/\hla{T}$ and renormalizing *is*
temperature, because $p_i \propto e^{z_i}$ gives
$p_i^{1/\hla{T}} \propto e^{z_i/\hla{T}}$:

```python
tempered = probs.pow(1.0 / temperature)
tempered = tempered / tempered.sum(dim=-1, keepdim=True)   # same as softmax(z / T)
```

Multiplying probabilities by $1/\hla{T}$ and renormalizing is a no-op, because
the scale cancels:

$$
\frac{p_i / \hla{T}}{\sum_j p_j / \hla{T}} = \frac{p_i / \hla{T}}{(1/\hla{T}) \sum_j p_j} = p_i.
$$

If your sampler's temperature knob has no visible effect, check this first.

## Shift invariance and the max subtraction

This section shows that adding a constant to every logit leaves softmax
unchanged, and that in floating point the right constant is the difference
between a correct answer and a `NaN`.

The picture is a group photo where you measure everyone's height relative to
the tallest person. The relative heights are all that matter, so you can choose
any reference point. Softmax only sees logit differences, so the same freedom
applies.

For any $c \in \mathbb{R}$, with $\mathbf{1}$ for the all-ones vector, the
factor $e^c$ appears in the numerator and the denominator:

$$
\boxed{\,\operatorname{softmax}(z + c\mathbf{1})_i = \frac{e^{c} e^{z_i}}{e^{c} \sum_j e^{z_j}} = \operatorname{softmax}(z)_i\,}
$$

The cancellation is exact, not approximate. It's an algebraic identity that
holds for every $c$, so in exact arithmetic the choice of $c$ is free.

### The overflow arithmetic

In floating point the choice isn't free, because $e^x$ grows fast enough to
leave the format. It overflows to infinity once $x$ exceeds the natural log of
the format's largest finite value. The thresholds for the three formats are as
follows:

| Format | Exponent bits | Largest finite value | $e^x$ overflows above |
|---|---|---|---|
| float32 | 8 | $3.4028 \times 10^{38}$ | $\ln(3.4028 \times 10^{38}) = 88.72$ |
| bfloat16 | 8 | $3.3895 \times 10^{38}$ | $\ln(3.3895 \times 10^{38}) = 88.72$ |
| float16 | 5 | $65{,}504$ | $\ln(65{,}504) = 11.09$ |

- **float32 and bfloat16 agree to four digits**, because overflow is set by the
  exponent field and the two formats share it. What bfloat16 gives up is
  precision, not range: its 8-bit significand resolves $2^{-8} = 0.39\%$,
  against float32's $2^{-24} = 6 \times 10^{-8}$.
- **float16 is the outlier.** A logit of 11 is ordinary, which is why no serious
  inference engine runs a softmax in float16, and why `engine/sampling.py` opens
  with `logits = logits.float()`.
- **Underflow is harmless.** In float32 the smallest positive subnormal is
  $1.4013 \times 10^{-45}$, so $e^x$ flushes to zero below
  $\ln(1.4013 \times 10^{-45}) = -103.3$. A token that underflows had a
  probability below $10^{-45}$, and zero is correct to every digit anyone can
  represent.

> [!KEY] Overflow is fatal, underflow isn't
> Choose $c = -\max_j z_j$, so that every exponent is at most 0. Without the
> shift, a row with a logit of 100 produces `inf` in the numerator and in the
> sum, and `inf / inf` is `NaN`.

The stable softmax is three lines:

```python
def stable_softmax(x, dim=-1):
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp = shifted.exp()
    return exp / exp.sum(dim=dim, keepdim=True)
```

Three properties make this the only shift anyone uses:

- **Nothing overflows.** Every term is in $(0, 1]$, however large the raw logits
  were.
- **The sum is bounded.** The maximizing term is exactly $e^0 = 1$, so the
  denominator is at least 1 and at most $V = 248{,}320$. No division by zero,
  and no overflow of the sum, since $248{,}320 \ll 3.4 \times 10^{38}$.
- **The top probability is well conditioned.** It's computed as
  $1 / \text{sum}$, which is the best-conditioned way to get it.

A single `NaN` in the row makes `torch.multinomial` fail or return garbage.
Because the failure is at the very end of the step, the traceback points at the
sampler rather than at whatever produced the large logit.

This same identity, applied incrementally as tiles of a row arrive, is the whole
idea behind the online softmax in [chapter 14](/c/14-flash-attention).

## Greedy against sampling

This section covers the simplest sampler, always taking the top token, and why
it isn't enough.

At temperature 0, the sampler returns $\operatorname{arg\,max}_i z_i$. That's
*greedy decoding*: deterministic, reproducible without a seed, and the right
default for extraction, classification, and structured output. It isn't "the
most likely answer," though.

Greedy maximizes the probability of each token given the prefix, one step at a
time. That's not the same as maximizing the probability of the whole sequence:
a token that's second-best now can open a continuation that's far more likely
overall, and greedy never sees it. *Beam search*, which keeps several candidate
sequences alive at once, exists to find the high-probability sequence; greedy
doesn't attempt it.

Greedy also has a well-documented failure: on open-ended text, it falls into
loops, repeating a phrase indefinitely. The nucleus sampling paper measured
this, and it's the reason the whole truncation family exists.

## Truncation: one operation, three rules

This section sets up top-k, top-p, and min-p, which all do the same thing: cut
off the unlikely tail so that a random draw can't land in it.

They differ only in the rule for choosing a keep-set
$S \subseteq \{1, \dots, V\}$. Keep only those outcomes and renormalize:

$$
q_i = \frac{p_i \, \mathbf{1}[i \in S]}{\sum_{j \in S} p_j}.
$$

The code never writes that division. It sets the logits outside $S$ to
$-\infty$ and lets the softmax renormalize:

$$
\tilde z_i = \begin{cases} z_i & i \in S \\ -\infty & i \notin S \end{cases}
\qquad
\operatorname{softmax}(\tilde z)_i = \frac{e^{\tilde z_i}}{\sum_j e^{\tilde z_j}}
$$

Because $e^{-\infty} = 0$, the numerator for $i \notin S$ and the excluded terms
of the denominator vanish, leaving exactly $q_i$. ==Masking and renormalizing
are the same thing==, which is why every filter in `engine/sampling.py` returns
logits rather than probabilities.

Two consequences follow:

- **A rule that depends only on ranking** can run on raw logits without a
  softmax, because softmax is strictly increasing. Top-k is such a filter.
- **A rule that depends on ratios of probabilities** can skip the normalizer,
  because the normalizer cancels in a ratio. Min-p is such a filter.

### Top-k

Top-k keeps the $k$ most likely tokens and nothing else. Formally, it keeps the
indices $S_k$ of the $k$ largest logits, ties broken arbitrarily. The whole
implementation is a threshold:

```python
def top_k_filter(logits, k):                      # logits: (batch, vocab)
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    threshold = logits.topk(k, dim=-1).values[..., -1, None]   # (batch, 1)
    return logits.masked_fill(logits < threshold, float("-inf"))
```

`topk(k).values[..., -1]` is the $k$-th largest logit. Comparing with `<`
rather than `<=` keeps every token tied at the boundary, so the result can hold
more than $k$ entries. That's the safe direction: keeping an extra equally
likely token changes the distribution less than dropping one.

The weakness of top-k is that ==$k$ is fixed while the model's confidence
isn't==:

- After `def `, the next token might be nearly determined, and `k = 50` admits
  49 tokens with a combined probability under $10^{-3}$.
- Mid-sentence in open prose, the model may be spread over hundreds of
  reasonable continuations, and `k = 50` cuts most of them.

## Top-p: a fixed mass

This section covers top-p, or nucleus sampling, which fixes top-k's weakness by
keeping a fixed share of probability instead of a fixed count.

Picture pouring tokens into a bucket, most likely first, and stopping as soon as
the bucket holds at least $p$ of the probability. The token that tips it over
stays in. On a confident row, one or two tokens fill the bucket; on a flat row,
hundreds do.

Formally, sort the probabilities descending,
$p_{(1)} \ge p_{(2)} \ge \dots \ge p_{(V)}$, and write the cumulative sums as
follows:

$$
C_n = \sum_{r=1}^{n} p_{(r)}, \qquad C_0 = 0.
$$

The *nucleus* is the shortest prefix whose mass reaches $p$:

$$
n(p) = \min \{\, n \ge 1 : C_n \ge p \,\}.
$$

The set exists for any $p \le 1$ because $C_V = 1$, and $n(p) \ge 1$, so the
nucleus is never empty. Keep ranks 1 through $n(p)$ and renormalize by
$C_{n(p)}$.

### The per-element predicate

The difficulty is turning that definition into a test on each rank. Because $C$
is non-decreasing, rank $r$ is kept exactly when the mass before it is still
short of $p$:

$$
\boxed{\, r \le n(p) \iff \hlb{C_{r-1}} < p \,}
$$

> [!KEY] Test the mass before the token, not through it
> The predicate reads $\hlb{C_{r-1}}$, the mass *strictly before* rank $r$, not
> $C_r$. Since `cumsum` gives you $C_r$, subtract the token's own probability.

In code, the predicate is written as follows:

```python
sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
sorted_probs = sorted_logits.softmax(dim=-1)
cumulative = sorted_probs.cumsum(dim=-1)          # C_r
remove = cumulative - sorted_probs >= p           # C_{r-1} >= p
remove[..., 0] = False                            # always keep the top token
```

`cumulative - sorted_probs` is $C_r - p_{(r)} = \hlb{C_{r-1}}$. The last line
is a backstop: $C_0 = 0 < p$ already keeps rank 1 for any $p > 0$, but it
costs nothing and it protects against `p = 0`.

> [!WARNING]
> Writing `cumulative >= p` instead is the classic bug. It drops the token that
> crosses the threshold, so the retained mass is $C_{n(p) - 1} < p$, strictly
> less than the parameter asked for. The output is still fluent, which is why
> this bug survives code review.

> [!EXAMPLE] Top-p on the lab's five logits
> Take $z = (3, 2, 1, 0, -1)$. The exponentials are
> $20.086, 7.389, 2.718, 1.000, 0.368$, with sum $31.561$, so
>
> $$
> p = (0.6364,\ 0.2341,\ 0.0861,\ 0.0317,\ 0.0117).
> $$
>
> The cumulative sums are $C_1 = 0.6364$, $C_2 = 0.8705$, $C_3 = 0.9567$,
> $C_4 = 0.9883$, and $C_5 = 1$.
>
> - **At `p = 0.7`:** $C_0 = 0 < 0.7$ keeps rank 1, $C_1 = 0.6364 < 0.7$ keeps
>   rank 2, and $C_2 = 0.8705 \ge 0.7$ drops rank 3. Two tokens survive,
>   holding $0.8705$ of the mass, renormalized to $0.7311$ and $0.2689$.
> - **At `p = 0.5`:** $C_0 = 0 < 0.5$ keeps rank 1, and $C_1 = 0.6364 \ge 0.5$
>   drops rank 2. One token survives, which is correct: the top token alone
>   exceeds 0.5.
> - **The buggy predicate at `p = 0.7`** tests $C_r \ge p$: $C_1 = 0.6364 <
>   0.7$ keeps rank 1, and $C_2 = 0.8705 \ge 0.7$ drops rank 2. One token
>   instead of two, retaining $0.6364$ of the mass against the $0.7$ requested.
>
> The lab checks both of these cases by name.

## Min-p: a logit window

This section covers min-p, which scales its cutoff with the model's confidence:
it keeps every token at least `min_p` times as likely as the most likely one.
At `min_p = 0.1`, a token survives if it's at least a tenth as likely as the
favorite.

```python
def min_p_filter(logits, min_p):                       # logits: (batch, vocab)
    if min_p <= 0.0:
        return logits
    probs = logits.softmax(dim=-1)                     # (batch, vocab)
    threshold = probs.max(dim=-1, keepdim=True).values * min_p   # (batch, 1)
    return logits.masked_fill(probs < threshold, float("-inf"))
```

The rule is a ratio, so the normalizer $Z$ cancels. Write $\hlc{m}$ for the
`min_p` parameter and $z_{\max} = \max_j z_j$, then take logs:

$$
\begin{aligned}
p_i \ge \hlc{m} \, p_{\max}
&\iff \frac{e^{z_i}}{Z} \ge \hlc{m} \frac{e^{z_{\max}}}{Z} \\
&\iff e^{z_i - z_{\max}} \ge \hlc{m} \\
&\iff \boxed{\, z_i \ge z_{\max} + \ln \hlc{m} \,}
\end{aligned}
$$

> [!INTUITION]
> Min-p is a *logit window*: it keeps every token within $|\ln \hlc{m}|$ of the
> top logit and nothing else. At `min_p = 0.05`, the window is
> $\ln 0.05 = -3.00$, so "every token within 3 logits of the best one." At
> `min_p = 0.3`, it's $\ln 0.3 = -1.204$.

Check that against the five-logit example: $z_{\max} = 3$, so `min_p = 0.3`
cuts at $3 - 1.204 = 1.796$ and keeps the logits 3 and 2. That's the two tokens
the lab asserts.

Two things follow from the window form:

- **Renormalizing doesn't move it.** A preceding top-k changes $Z$ but no logit
  difference, so min-p's keep-set is the same before or after it.
- **Temperature does move it**, because temperature rescales the differences.
  On tempered logits $z/\hla{T}$, the window in *original* logit units is
  $\hla{T} |\ln \hlc{m}|$ wide. At `temperature = 1.5` and `min_p = 0.05`,
  that's $1.5 \times 3.00 = 4.49$ logits, half again as wide as at temperature 1.

That widening is why ==min-p holds up at high temperature where top-p doesn't==.
Top-p's threshold is a mass, and flattening the distribution pushes mass into
the tail, so the nucleus grows without bound. Min-p's threshold tracks the
model's own confidence, so it grows only in proportion to $\hla{T}$.

## The order, and which pairs commute

You now have five transforms. This section answers the question the opening
raised: in what order do they run, and which swaps change the answer?

Swapping two adjacent steps changes the output for only three kinds of pair,
and knowing which is more useful than memorizing the list. The engine applies
the transforms in this order:

1. **Penalties**, on raw logits. The penalties section later in this chapter
   covers them; for now, they push down the logits of tokens already
   generated.
2. **Temperature**, dividing logits.
3. **Truncation**: top-k, then min-p, then top-p.
4. **Softmax and draw.**

The following table shows which pairs *commute*, meaning that swapping them
gives the same result:

| Pair | Commutes? | Why |
|---|---|---|
| Repetition penalty, temperature | Yes | Both are multiplicative and $\hla{T} > 0$ preserves the sign test |
| Presence or frequency penalty, temperature | No | Additive against multiplicative |
| Temperature, top-k | Yes | Dividing by $\hla{T} > 0$ preserves the ranking |
| Temperature, min-p | No | The window scales with $\hla{T}$ |
| Temperature, top-p | No | The nucleus is a mass, not a ranking |
| Penalties, any truncation | No | Penalties change the ranking |

### Repetition penalty commutes with temperature

The repetition penalty maps a seen token's logit $z$ to $z/\rho$ when $z > 0$
and to $z\rho$ when $z \le 0$, for $\rho > 1$. Apply both steps in each order:

- **Temperature first** gives $z/(\hla{T}\rho)$ and $z\rho/\hla{T}$.
- **Penalty first** gives $(z/\rho)/\hla{T}$ and $(z\rho)/\hla{T}$, the same
  two expressions.

Dividing by a positive $\hla{T}$ can't change the sign, so the branch taken is
the same too.

### Additive penalties don't commute

Presence and frequency penalties subtract a constant $\alpha$. Before
temperature, the penalty becomes $(z - \alpha)/\hla{T} = z/\hla{T} -
\alpha/\hla{T}$; after it, $z/\hla{T} - \alpha$. The effective strength differs
by a factor of $\hla{T}$.

The convention is penalties first, so the effective log-odds shift is
$\alpha/\hla{T}$. At `temperature = 0.7`, a presence penalty of `0.5` behaves
like `0.714` in the space the truncations see. Document that rather than
"fixing" it, because users tune penalties at a fixed temperature and expect the
same numbers to work across engines.

### Temperature before truncation

This is the pair that changes output visibly: ==temperature after top-p gives a
narrower nucleus when the user asked for more variety==.

> [!EXAMPLE] Top-p 0.7 at temperature 2.0, in both orders
> Use the same five logits with `top_p = 0.7` and `temperature = 2.0`.
>
> **Correct order: temperature, then top-p.** The tempered logits are
> $(1.5, 1.0, 0.5, 0.0, -0.5)$, with exponentials
> $4.482, 2.718, 1.649, 1.000, 0.607$ summing to $10.455$:
>
> $$
> p(\hla{T}{=}2) = (0.4287,\ 0.2600,\ 0.1577,\ 0.0956,\ 0.0580).
> $$
>
> The cumulative sums are $0.4287$, $0.6886$, and $0.8463$. The test
> $\hlb{C_{r-1}} < 0.7$ keeps ranks 1, 2, and 3, because $C_2 = 0.6886$ is still
> below $0.7$. Three tokens survive, renormalized to $(0.5065, 0.3072, 0.1863)$.
>
> **Wrong order: top-p, then temperature.** Top-p on the untempered
> distribution keeps two tokens, as in the earlier example. Tempering those two
> gives logits $1.5$ and $1.0$, so $(0.6225, 0.3775)$.

The support shrinks from three tokens to two, and the top token's probability
rises from $0.51$ to $0.62$. Worse, the parameter no longer describes anything:
in the wrong order, the two surviving tokens hold $0.4287 + 0.2600 = 0.6886$ of
the tempered mass, under the $0.7$ that was asked for.

### Why top-k, then min-p, then top-p

Each truncation has its own reason for its place:

- **Top-k first**, because it's a hard cap that costs one `topk` over the row.
  It bounds the damage if the other two are left at their defaults.
- **Min-p anywhere**, because it's a logit window and truncation doesn't move
  logit differences. It sits in the middle because that reads well, not because
  it must.
- **Top-p last**, because it's the only one that reads the renormalized masses.
  Last, its nucleus is computed on the distribution you actually sample from.
  First, tokens that top-k is about to discard would inflate the cumulative sum.

## Penalties as logit transforms

This section covers the penalties, which discourage the model from repeating
itself by lowering the logits of tokens it has already produced. Both families
map $z \mapsto z'$ before temperature, and they differ in whether they respect
the log-odds geometry.

### Repetition penalty

The repetition penalty scales each seen token's logit, with a branch on its
sign:

```python
def apply_repetition_penalty(logits, previous, penalty):
    if penalty == 1.0:
        return logits
    scores = torch.gather(logits, 1, previous)                     # (batch, n_seen)
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits.scatter(1, previous, scores)                     # (batch, vocab)
```

The asymmetry is the whole function. The goal is $z' < z$ for every seen token:

- **For $z > 0$**, dividing by $\rho > 1$ moves the logit toward zero, which is
  down.
- **For $z < 0$**, dividing moves it *up*: $-2 / 1.1 = -1.82 > -2$, which
  encourages the token you meant to discourage. Multiplying gives
  $-2 \times 1.1 = -2.2 < -2$, which is down.

Both branches decrease the logit; that's the only invariant the function has.
Roughly half of a 248,320-entry vocabulary sits below zero at any position, so
an implementation that divides unconditionally gets the sign wrong about half
the time. ==The symptom is a model that repeats *more* when you raise the
penalty.==

The lab checks this with `logits = [2.0, -2.0, 0.5]`, `previous = [0, 1]`, and
`penalty = 2.0`. The result must be `[1.0, -4.0, 0.5]`: the positive logit
halved, the negative one doubled in magnitude, and the unseen one untouched. It
also checks that the input tensor isn't modified in place, which is why the code
uses `scatter` rather than `scatter_`.

> [!NOTE]
> The shift in log-odds is $-z(1 - 1/\rho)$ on the positive branch and
> $z(\rho - 1)$ on the negative one. Both are proportional to $|z|$, so a token
> with a logit near zero is barely penalized, and a token with a logit of
> exactly zero isn't penalized by any $\rho$.

### Presence and frequency penalties

Presence and frequency penalties subtract a constant, which makes them the
well-behaved family:

```python
def apply_presence_frequency_penalty(logits, counts, presence, frequency):
    # counts: (batch, vocab), occurrences of each token so far
    if presence == 0.0 and frequency == 0.0:
        return logits
    return (logits
            - presence * (counts > 0).to(logits.dtype)
            - frequency * counts.to(logits.dtype))
```

With $c_i$ for the count of token $i$ so far, the map is a shift:

$$
z_i' = z_i - \alpha_{\text{pres}} \mathbf{1}[c_i > 0] - \alpha_{\text{freq}} c_i.
$$

The log-odds of a penalized token against an unpenalized one drop by exactly
$\alpha_{\text{pres}} + \alpha_{\text{freq}} c_i$, whatever the logits were.
There's no sign branch and no dependence on scale:

- **Presence** applies once per distinct token seen. A presence penalty of
  `0.5` multiplies the odds by $e^{-0.5} = 0.607$, a 39% reduction, uniformly
  across the vocabulary.
- **Frequency** scales with the count, so it keeps biting as a token recurs.

The `counts` tensor is `(batch, vocab)`, the same size as the logits, which the
cost section takes into account.

## The Gumbel-max trick

This section covers the third need, the draw, and a way to do it with one
argmax: no normalization, no cumulative sum, and no search.

Everything so far ends at `torch.multinomial`, which wants a normalized
probability vector, builds a cumulative sum, draws a uniform, and searches.
The alternative is a race. Give every token a random bonus, add it to the
token's logit, and pick the highest total. With the right kind of random bonus,
each token wins with exactly its softmax probability.

The right kind is the *Gumbel* distribution. Let $\hld{G_1}, \dots, \hld{G_V}$
be independent standard Gumbel variables. You get one from a uniform
$U \sim \text{Uniform}(0,1)$ as $\hld{G} = -\ln(-\ln U)$. Then:

$$
\boxed{\,\operatorname{arg\,max}_{i} \, (z_i + \hld{G_i}) \sim \operatorname{softmax}(z)\,}
$$

The result is exact, not asymptotic, and ==the proof never needs the
normalizer==: it produces it.

> [!DEEPDIVE] Proof that Gumbel-max samples from softmax
> The standard Gumbel has CDF $F(g) = \exp(-e^{-g})$ and density
> $f(g) = e^{-g} \exp(-e^{-g})$. Condition on $\hld{G_i} = g$. Index $i$ wins
> when $z_j + \hld{G_j} < z_i + g$ for every $j \ne i$, that is,
> $\hld{G_j} < g + z_i - z_j$, which has probability
> $\exp(-e^{-(g + z_i - z_j)})$.
>
> Multiply over $j \ne i$ and integrate:
>
> $$
> P(i \text{ wins}) = \int_{-\infty}^{\infty} e^{-g} \exp(-e^{-g}) \prod_{j \ne i} \exp\!\left(-e^{-g} e^{z_j - z_i}\right) dg.
> $$
>
> The lone $\exp(-e^{-g})$ is the $j = i$ term of the same product, since
> $e^{z_i - z_i} = 1$. Fold it in and write $S = \sum_{j=1}^{V} e^{z_j - z_i}$,
> and the product collapses to $\exp(-S e^{-g})$:
>
> $$
> P(i \text{ wins}) = \int_{-\infty}^{\infty} e^{-g} \exp\!\left(-S e^{-g}\right) dg.
> $$
>
> Substitute $u = e^{-g}$, so $du = -e^{-g} dg$ and the limits flip:
>
> $$
> P(i \text{ wins}) = \int_{0}^{\infty} e^{-S u} \, du = \frac{1}{S} = \frac{e^{z_i}}{\sum_{j} e^{z_j}}.
> $$
>
> That's $p_i$, and the normalizer came out of $S$.

The argmax form has three properties that the cumulative-sum form doesn't:

- **It composes with masks for free.** A masked token has
  $z_i + \hld{G_i} = -\infty$ and can't win.
- **It's one fused pass.** Each element needs its own random number and nothing
  else, so there's no data-dependent control flow. The cumulative-sum form needs
  a prefix scan and a search.
- **It parallelizes across a batch.** Every row does its own argmax, and no
  row's work depends on another's.

> [!WARNING]
> $U$ must be strictly inside $(0, 1)$. $U = 0$ gives $-\ln 0 = \infty$ and then
> $-\ln(\infty) = -\infty$, and $U = 1$ gives $-\ln(0)$ the other way. Generate
> on the half-open interval and reject the endpoint. The double logarithm also
> loses precision in float32 for $U$ near 1; the equivalent *exponential race*,
> drawing $E_i \sim \text{Exp}(1)$ and taking
> $\operatorname{arg\,max}_i \, p_i / E_i$, avoids one of the logarithms.

## Seeding and reproducibility

This section answers when a seed reproduces a generation, and how far you can
promise that it does.

A generation must reproduce exactly given a seed and identical inputs, which
means threading an explicit generator rather than relying on global state:

```python
gen = torch.Generator(device=logits.device).manual_seed(1234)
token = torch.multinomial(probs, num_samples=1, generator=gen)   # (batch, 1)
```

The global random number generator (RNG) is shared with every other operation in the
process. A dropout layer, a shuffled dataloader, or another request's sampler
consumes draws from it, so the same seed reproduces only if nothing else ever
draws. An explicit `torch.Generator` is owned by the request.

Reproducibility *across batch sizes* is a much stronger requirement, and it's
usually not worth paying for. Floating-point addition isn't associative, so a
reduction that splits a row differently at batch 1 and batch 8 produces logits
that differ in the last bits.

That small difference is enough. The LM head runs in bfloat16, which resolves
logits to about `0.39%` relative, and with 248,320 candidates, several are
routinely within that of each other. An argmax near a tie flips, and every
subsequent token changes with it.

> [!KEY] Promise only what you can reproduce
> Same seed, same batch composition, same engine version: same output. Say that
> in your API documentation, and don't promise more.

## What sampling costs

This section prices the pipeline on a GPU: it's negligible by bytes and
expensive by kernel launches. Do the arithmetic before you dismiss it.

### One softmax

The vocabulary is $V = 248{,}320$, so one float32 row of logits is as follows:

$$
248{,}320 \times 4 = 993{,}280 \text{ bytes} = 0.993 \text{ MB}.
$$

One softmax pass reads that and writes it, $1.99$ MB. The arithmetic is one
subtraction, one exponential, and one division per element, about
$3V = 745{,}000$ FLOPs. Divide FLOPs by bytes:

$$
\frac{7.45 \times 10^{5}}{1.99 \times 10^{6}} = 0.375 \ \text{FLOPs per byte}.
$$

[Chapter 10](/c/10-roofline) put the A100's ridge point, the FLOPs per byte an
operation needs before arithmetic rather than memory limits it, at 161. Softmax
sits below it by a factor of $161 / 0.375 = 429$, as memory bound as anything in
the engine. The timings follow:

- **Memory.** At the measured copy bandwidth of 1275 GB/s, the pass takes
  $1.99 \times 10^{6} / 1.275 \times 10^{12} = 1.6$ microseconds.
- **Arithmetic.** At 312 TFLOP/s, it would take 2.4 nanoseconds.

==Only the bytes matter.== That also prices a naive softmax: PyTorch's multi-pass
form, with one kernel each for the max, the exponentials, the sum, and the
division, reads the row four times instead of once.
`engine/kernels/softmax_triton.py` does all four in one trip for this reason.

### The whole pipeline, at batch 1

Count each operation in `engine/sampling.py` as read plus write in float32:

| Step | Traffic |
|---|---|
| Cast bfloat16 to float32 | 1.49 MB |
| Repetition penalty (gather, where, scatter) | 1.99 MB |
| Presence and frequency (reads a `(batch, vocab)` count tensor) | 2.98 MB |
| Divide by temperature | 1.99 MB |
| Top-k (`topk`, then `masked_fill`) | 2.98 MB |
| Min-p (softmax, max, `masked_fill`) | 4.97 MB |
| Top-p (sort, softmax, `cumsum`, `scatter`, `masked_fill`) | at least 8 MB |
| Final softmax | 1.99 MB |
| `multinomial` | at least 2 MB |
| **Total** | **at least 26 MB** |

The sort is the entry to be suspicious of. A radix sort over a quarter of a
million float32 keys and their int64 indices makes several passes over both, so
8 MB is a floor, not an estimate.

At 1275 GB/s, 26 MB takes about 21 microseconds. The decode step it sits inside
reads the model's 53.8 GB of bfloat16 weights once:

$$
\frac{53.8 \times 10^{9}}{1.275 \times 10^{12}} = 42.2 \text{ ms}.
$$

By bytes, sampling is 0.05% of a decode step. By launches, it isn't. A *kernel*
is one function the GPU runs, and *launching* one costs the CPU a fixed
overhead before any work starts:

- The table is about twenty separate kernels, each on a single row of 248,320
  elements.
- Chapter 10 put launch overhead at 5 to 10 microseconds, so the pipeline spends
  100 to 200 microseconds in overhead against 21 microseconds of traffic.
  ==The launches cost five to ten times what the memory does.==
- A row of 248,320 elements is 970 blocks at 256 threads, barely nine per SM
  across the A100's 108 SMs, its independent processors. No kernel has enough
  work to hide the next launch.

Three further costs don't show up in either count:

- **A forced synchronization.** `torch.multinomial` and any `.item()` on the
  result make the GPU drain, and the CPU can't queue the next step's launches
  until it returns. On a 42 ms step the stall is absorbed, but the sampler is
  the one point where the pipeline is guaranteed to empty.
- **A hole in the CUDA graph.** A *CUDA graph* records a whole step's kernels
  and replays them as one launch. Data-dependent control flow, the sort in
  particular, can't be captured. Chapter 10 noted that a decode step runs
  several hundred kernels and that graph capture removes that overhead. The hole
  costs more than the sampler's own time.
- **Traffic that scales with batch.** At batch 128, the pipeline moves
  $128 \times 26 = 3.3$ GB, which is
  $3.3 \times 10^{9} / 1.275 \times 10^{12} = 2.6$ ms against the same 42.2 ms
  weight read. That's 6% of the step, and worth removing.

### Which is why it's one kernel

Production engines run one fused kernel per step, or very few, over the whole
batch at once:

- Penalties, temperature, and the masks are elementwise or row-reduction work,
  so they fuse into a single pass that keeps the row in registers.
- Top-k is served by a partial selection rather than a full sort.
- Top-p's cumulative sum runs on the top-k survivors, not on 248,320 entries.
- The draw is the Gumbel-max argmax, which needs neither a scan nor a
  synchronization, so the token index stays on the device and the next step's
  launches queue behind it without a stall.

The result is one launch, graph-capturable, and bound by the one unavoidable
read of the logits.

## What goes wrong

**`NaN` in the output distribution.** A missing max subtraction, or a softmax
run in float16. Symptom: `torch.multinomial` raises, or returns index 0
forever. While debugging, check for non-finite logits before the softmax.

**Temperature appears disconnected.** You scaled probabilities instead of
logits, and the scale cancelled. Check that the change happens before the
softmax.

**Top-p under-delivers its mass.** The boundary token is being dropped. Compare
`cumulative - sorted_probs >= p`, not `cumulative >= p`.

**Raising the repetition penalty makes repetition worse.** The negative branch is
dividing instead of multiplying.

**Everything is masked.** With `min_p` near 1 or `top_p` near 0, a keep-set can
come out empty if the guards are missing, and the softmax of an all-`-inf` row
is all `NaN`. Every filter needs a "keep at least one" floor.

**A fixed seed doesn't reproduce.** Either the global RNG is in use, something
else in the process is consuming draws, or the batch composition changed between
runs.

**A one-token vocabulary shift.** Somebody applied a filter to probabilities and
then re-softmaxed. Softmax isn't idempotent; applying it twice flattens the
distribution a second time.

> [!RECAP]
> - Softmax turns logit differences into log-odds, and temperature
>   $\hla{T}$ divides them. `temperature == 0` is a plain `argmax`.
> - Always subtract the row maximum, and run the softmax in float32. Overflow
>   starts at 88.72 in float32 and bfloat16, and at 11.09 in float16.
> - Truncation is masking to $-\infty$. Top-p keeps rank $r$ while
>   $\hlb{C_{r-1}} < p$; min-p keeps $z_i \ge z_{\max} + \ln \hlc{m}$.
> - Apply penalties, then temperature, then top-k, min-p, and top-p. The
>   repetition penalty multiplies negative logits and divides positive ones.
> - Gumbel-max samples exactly with one argmax, which is what lets a fused,
>   graph-capturable sampler avoid the sort and the synchronization.

## Check your understanding

> [!QUESTION] Why is subtracting the maximum safe but subtracting the mean unsafe?
> Both are shifts, and shift invariance holds for any constant, so both are
> *mathematically* safe. Only the maximum guarantees that every exponent is at
> most 0. Subtracting the mean leaves the largest logit above zero by however
> far it exceeds the mean, and on a peaked 248,320-entry row that gap can
> exceed 88.7.

> [!QUESTION] At `temperature = 1.5` and `min_p = 0.05`, how wide is the keep window in raw logits, and how many tokens does that admit?
> The window is $\hla{T} |\ln \hlc{m}| = 1.5 \times 3.00 = 4.49$ logits below
> the top. How many tokens it admits depends entirely on the row: on a peaked
> distribution, one or two; on a flat one, thousands. That adaptivity is the
> point. Min-p doesn't promise a count or a mass; it promises a confidence
> ratio.

> [!QUESTION] The pipeline moves 26 MB and takes 21 microseconds of traffic, inside a step that takes 42 ms. Why is it still worth fusing?
> Because 21 microseconds of traffic is delivered by twenty launches costing 100
> to 200 microseconds, because the sort prevents CUDA graph capture of the whole
> step, and because the traffic term scales with batch size while the step's
> 42 ms weight read doesn't. At batch 128, it's 6% of the step.

> [!QUESTION] Does applying top-k before min-p give a different result from applying min-p before top-k?
> No. Min-p's keep-set is $\{i : z_i \ge z_{\max} + \ln \hlc{m}\}$, which
> depends only on logit differences, and top-k changes no logit that survives
> it. The two keep-sets intersect in the same place either way. Temperature is
> the transform whose position matters.

## Lab

> [!TRY]
> Implement the full pipeline on CPU: `apply_repetition_penalty`,
> `top_k_filter`, `top_p_filter`, `min_p_filter`, and `sample`. You pass when
> every harness check succeeds, including the top-p boundary case at `p = 0.7`.

The harness works on the five-logit row from the worked example and checks,
among others, the following:

- Top-k keeps exactly `k` tokens, keeps the highest ones, and `k = 0` is a
  no-op.
- Top-p at `p = 0.7` keeps two tokens, the boundary case, and at `p = 0.5` keeps
  one. A tiny `p` still leaves something to sample from, and `p = 1` is a no-op.
- Min-p at `0.3` keeps two tokens on the peaked row and everything on a flat
  one.
- The repetition penalty turns `[2.0, -2.0, 0.5]` into `[1.0, -4.0, 0.5]` with
  `penalty = 2.0`, leaves unseen tokens alone, is a no-op at `1.0`, and doesn't
  modify its input in place.
- `temperature = 0` returns the argmax exactly, and `sample` returns shape
  `(batch, 1)`.
- The same `torch.Generator` seed reproduces the same tokens, a different seed
  usually doesn't, and `top_k = 1` forces the argmax.
- Over 200 draws from a row whose top five logits are `[5, 4, 3, 2, 1]` and
  whose other 507 are `-10`, every draw with `top_p = 0.9` comes from those
  five.

The lab reports `nucleus_size` and `seed_reproducible` as metrics. Its `sample`
takes its parameters as keyword arguments rather than a `SamplingParams` object,
so read the starter's signature before you begin.

## Further reading

- [The curious case of neural text degeneration](https://arxiv.org/abs/1904.09751) — the nucleus sampling paper.
- [Turning up the heat: min-p sampling for creative and coherent LLM outputs](https://arxiv.org/abs/2407.01082)
- [CTRL: a conditional transformer language model for controllable generation](https://arxiv.org/abs/1909.05858) — where the repetition penalty comes from.
- [Categorical reparameterization with Gumbel-softmax](https://arxiv.org/abs/1611.01144) — the Gumbel-max trick and its differentiable relaxation.
- [The Gumbel-max trick for discrete distributions](https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/)
