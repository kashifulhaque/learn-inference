---
title: Sampling
slug: 11-sampling
part: "Part 3 — Making it fast"
summary: Turning 248,320 logits into one token — softmax and temperature from first principles, the truncations, the penalties, and the order that makes the knobs mean what they say.
minutes: 75
gpu: false
objectives:
  - Derive softmax with temperature and take its zero and infinite limits.
  - Show why subtracting the row maximum is mandatory in floating point.
  - Derive top-k, top-p, and min-p as truncations of a categorical distribution.
  - Say which pairs of sampling transforms commute and which do not.
  - Prove that the Gumbel-max trick samples from the right distribution.
  - Cost a full-vocabulary softmax in bytes, FLOPs, and kernel launches.
lab: 11-sampling
---

# Sampling

The forward pass ends with a vector of 248,320 numbers. Sampling turns that
vector into one integer. Everything the model knows about what comes next is in
those numbers, and everything a user perceives as the model's "style" is in how
you collapse them.

This chapter is the last one before the engine gets fast. It belongs here
because sampling is the step that decides what the next forward pass sees, and
because it is the step most often implemented from a half-remembered blog post.
The operations are individually trivial. Composing them is not: the same five
transforms in two different orders give two different distributions, and only
one of them matches what the parameter names promise.

## Why this exists

The last linear layer of the model, the *LM head*, multiplies the final hidden
state by a matrix of shape `(hidden_size, vocab_size)` — here `(5120, 248320)` —
and produces one real number per vocabulary entry. These numbers are called
*logits*. They are unnormalized and unbounded. They are not probabilities, they
are not log-probabilities, and nothing constrains their scale.

To generate text you need three things from them:

1. A probability distribution, so that "how likely is this token" has an answer.
2. A way to trade fidelity against variety, because the maximum-probability
   token is not always the token you want.
3. A draw, reproducible from a seed, cheap enough to run every step.

The rest of the chapter builds those three things, in that order.

## Before you start

**Categorical distributions.** A categorical distribution over $V$ outcomes is a
vector $p \in \mathbb{R}^V$ with $p_i \ge 0$ and $\sum_{i=1}^{V} p_i = 1$. Here
$V = 248{,}320$ and outcome $i$ is "the next token is vocabulary entry $i$".
Sampling means drawing one index with those probabilities.

**Logits.** Write $z \in \mathbb{R}^V$ for the model's output. Chapter 0a covers
the softmax that turns $z$ into $p$; this chapter derives it again, with
temperature, because the derivation is what tells you where temperature is
allowed to go.

**Odds and log-odds.** The odds of outcome $i$ against outcome $j$ are
$p_i / p_j$. The log-odds are $\log(p_i/p_j)$. A transform that adds a constant
to one logit shifts log-odds by that constant. This is the natural unit for
talking about penalties.

**Floating point.** You need the exponent range of the formats the engine uses.
Chapter 0a has the full table; the two facts this chapter leans on are that
bfloat16 and float32 share an 8-bit exponent, and that float16 does not.

**PyTorch.** `topk`, `sort`, `cumsum`, `masked_fill`, `gather`, `scatter`, and
`multinomial`, all with an explicit `dim`. The reference implementation is
`engine/sampling.py`; read it alongside this chapter.

## From logits to a distribution

Nothing so far forces any particular map from $z$ to $p$. Pick one by deciding
what you want it to preserve.

The model was trained with a cross-entropy loss, which means the logits were
fitted so that *differences* between them carry the information. Adding the same
constant to every logit changes nothing about the model's prediction: the last
layer's bias, if it had one, could absorb it. So the map should depend on $z$
only through differences, and the natural way to state that is to ask that logit
differences *be* log-odds:

$$
\log \frac{p_i}{p_j} = z_i - z_j \quad \text{for every } i, j.
$$

That single requirement pins the map down. Rearrange to $p_i = p_j e^{z_i - z_j}$
and sum over $i$:

$$
1 = \sum_{i=1}^{V} p_i = p_j e^{-z_j} \sum_{i=1}^{V} e^{z_i}.
$$

Solve for $p_j$:

$$
p_j = \frac{e^{z_j}}{\sum_{i=1}^{V} e^{z_i}}.
$$

That is the softmax, and the derivation shows it is the *only* map with the
property asked for. It is automatically positive and automatically normalized;
neither had to be imposed.

## Temperature

Now add a knob. Scale the logits by $1/T$ for some $T > 0$ before applying
softmax:

$$
p_i(T) = \frac{e^{z_i / T}}{\sum_{j=1}^{V} e^{z_j / T}}.
$$

$T$ is the *temperature*, named after the Boltzmann distribution, which has the
same form. From the log-odds identity, temperature divides every log-odds by
$T$:

$$
\log \frac{p_i(T)}{p_j(T)} = \frac{z_i - z_j}{T}.
$$

Below 1 that stretches every gap and sharpens the distribution. Above 1 it
compresses every gap and flattens it. At $T = 1$ it does nothing.

### The two limits

Let $m = \max_j z_j$ and let $A = \{\, i : z_i = m \,\}$ be the set of
maximizers, with $a = |A|$. Use shift invariance, proved in the next section, to
rewrite the ratio with $m$ removed:

$$
p_i(T) = \frac{e^{(z_i - m)/T}}{\sum_{j=1}^{V} e^{(z_j - m)/T}}.
$$

Every exponent is now at most 0.

As $T \to 0^{+}$: for $i \notin A$ the exponent $(z_i - m)/T \to -\infty$, so
that term goes to 0. For $i \in A$ the exponent is exactly 0, so that term is 1.
The denominator therefore tends to $a$, and

$$
\lim_{T \to 0^{+}} p_i(T) = \begin{cases} 1/a & i \in A \\ 0 & i \notin A. \end{cases}
$$

With a unique maximizer, $a = 1$ and the limit is a point mass on the argmax.
This is why the engine special-cases `temperature == 0` to a plain `argmax`
instead of dividing by something near zero: the limit exists, but the float
arithmetic on the way to it does not survive.

As $T \to \infty$: every exponent $(z_j - m)/T \to 0$, so every term tends to 1,
the denominator tends to $V$, and

$$
\lim_{T \to \infty} p_i(T) = \frac{1}{V}.
$$

The distribution becomes uniform over the whole vocabulary. The model's output
stops mattering entirely. Between the two limits the entropy of $p(T)$ rises
monotonically with $T$, from 0 at the greedy end to $\log V = \log 248{,}320
\approx 12.42$ nats at the uniform end.

### Temperature acts on logits, not probabilities

Two operations get confused here, and only one of them is temperature.

Raising probabilities to the power $1/T$ and renormalizing *is* temperature,
because $p_i \propto e^{z_i}$ gives $p_i^{1/T} \propto e^{z_i/T}$:

```python
tempered = probs.pow(1.0 / temperature)
tempered = tempered / tempered.sum(dim=-1, keepdim=True)   # same as softmax(z / T)
```

Multiplying probabilities by $1/T$ and renormalizing is a no-op. The scale
cancels:

$$
\frac{p_i / T}{\sum_j p_j / T} = \frac{p_i / T}{(1/T) \sum_j p_j} = p_i.
$$

The symptom is a temperature knob that appears to be connected to nothing. If
you ever write a sampler whose temperature has no visible effect, this is the
first thing to check.

## Shift invariance, and why the max subtraction is mandatory

Softmax is invariant to adding a constant to every logit. For any
$c \in \mathbb{R}$, writing $\mathbf{1}$ for the all-ones vector:

$$
\operatorname{softmax}(z + c\mathbf{1})_i = \frac{e^{z_i + c}}{\sum_j e^{z_j + c}} = \frac{e^{c} e^{z_i}}{e^{c} \sum_j e^{z_j}} = \operatorname{softmax}(z)_i.
$$

The factor $e^c$ appears in both numerator and denominator and cancels. Exactly.
Not approximately — this is an algebraic identity, and it holds for every $c$.

In exact arithmetic the identity means the choice of $c$ is free. In floating
point it is the difference between a correct answer and a `NaN`.

### The overflow arithmetic

An IEEE binary format with an 8-bit exponent field has a largest finite value
a little under $2^{128}$. For float32 that value is $3.4028 \times 10^{38}$; for
bfloat16, which has the same 8 exponent bits and a shorter significand, it is
$3.3895 \times 10^{38}$. So $e^x$ overflows to infinity when

$$
x > \ln(3.4028 \times 10^{38}) = 88.72 \quad \text{(float32)},
$$

$$
x > \ln(3.3895 \times 10^{38}) = 88.72 \quad \text{(bfloat16)}.
$$

The two thresholds agree to four digits, because overflow is set by the exponent
field and the two formats share it. What bfloat16 gives up is precision, not
range: its significand is 8 bits, so its relative resolution is $2^{-8} = 0.39\%$,
against float32's $2^{-24} = 6 \times 10^{-8}$.

float16 is the outlier. Its exponent field is 5 bits, its largest finite value is
$65{,}504$, and

$$
x > \ln(65{,}504) = 11.09 \quad \text{(float16)}.
$$

A logit of 11 is ordinary. This is why no serious inference engine runs a softmax
in float16, and why `engine/sampling.py` opens with `logits = logits.float()`.

Underflow is the other end. In float32 the smallest positive subnormal is
$1.4013 \times 10^{-45}$, so $e^x$ flushes to zero below
$\ln(1.4013 \times 10^{-45}) = -103.3$. That is harmless: a token whose exponent
underflows had a probability below $10^{-45}$, and zero is the correct answer to
every digit anyone can represent.

So overflow is fatal and underflow is not. Choose $c = -\max_j z_j$, which makes
every exponent at most 0:

```python
def stable_softmax(x, dim=-1):
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp = shifted.exp()
    return exp / exp.sum(dim=dim, keepdim=True)
```

Three properties follow immediately, and together they are why this is the only
shift anyone uses.

Every term is in $(0, 1]$, so nothing overflows regardless of how large the raw
logits were. The maximizing term is exactly $e^0 = 1$, so the denominator is at
least 1 and at most $V = 248{,}320$ — no division by zero, and no overflow of the
sum either, since $248{,}320 \ll 3.4 \times 10^{38}$. And the largest
probability is computed as $1 / \text{sum}$, which is the best-conditioned way to
get it.

Without the shift, a row containing a logit of 100 produces `inf` in the
numerator and `inf` in the sum, and `inf / inf` is `NaN`. One `NaN` in the row
makes `torch.multinomial` fail or return garbage, and because the failure is at
the very end of the step, the traceback points at the sampler rather than at
whatever produced the large logit.

This same identity, applied incrementally as tiles of a row arrive, is the whole
idea behind the online softmax in chapter 14.

## Greedy against sampling

With temperature 0 the sampler returns $\operatorname{arg\,max}_i z_i$. That is
*greedy decoding*: deterministic, reproducible without a seed, and the right
default for anything you want to be stable — extraction, classification,
structured output.

Greedy is not "the most likely answer". It maximizes the probability of each
token given the prefix, one step at a time, which is not the same as maximizing
the probability of the whole sequence. A token that is second-best now can open
a continuation that is far more likely overall, and greedy will never see it.
Beam search exists to search for the high-probability sequence; greedy does not
attempt it.

Greedy also has a well-documented failure mode: on open-ended text it falls into
loops, repeating a phrase indefinitely. The nucleus sampling paper measured this
and it is the reason the whole truncation family exists.

## Truncation as a family

Top-k, top-p, and min-p are all the same operation with different rules for
choosing the keep-set. Fix a subset $S \subseteq \{1, \dots, V\}$, keep only
those outcomes, and renormalize:

$$
q_i = \frac{p_i \, \mathbf{1}[i \in S]}{\sum_{j \in S} p_j}.
$$

The code never writes that division. It sets the logits outside $S$ to
$-\infty$ and lets the softmax do the renormalizing. That works because

$$
\operatorname{softmax}(\tilde z)_i = \frac{e^{\tilde z_i}}{\sum_j e^{\tilde z_j}}, \qquad \tilde z_i = \begin{cases} z_i & i \in S \\ -\infty & i \notin S, \end{cases}
$$

and $e^{-\infty} = 0$, so both the numerator for $i \notin S$ and the excluded
terms of the denominator vanish, leaving exactly $q_i$. Masking and
renormalizing are the same thing, which is why every filter in
`engine/sampling.py` returns logits rather than probabilities.

Two consequences are worth stating.

Any filter whose rule depends only on the *ranking* of the logits can be applied
to raw logits without a softmax, because softmax is strictly increasing. Top-k is
such a filter.

Any filter whose rule depends on *ratios* of probabilities can also skip the
normalizer, because the normalizer cancels in a ratio. Min-p is such a filter,
as the next section but one shows.

### Top-k

Let $S_k$ be the indices of the $k$ largest logits, ties broken arbitrarily. The
whole implementation is a threshold:

```python
def top_k_filter(logits, k):                      # logits: (batch, vocab)
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    threshold = logits.topk(k, dim=-1).values[..., -1, None]   # (batch, 1)
    return logits.masked_fill(logits < threshold, float("-inf"))
```

`topk(k).values[..., -1]` is the $k$-th largest logit. Comparing with `<` rather
than `<=` means that a tie at the boundary keeps every tied token, so the result
can hold more than $k$ entries. That is the safe direction: keeping an extra
equally-likely token changes the distribution less than dropping one.

The weakness of top-k is that $k$ is fixed while the model's confidence is not.
After `def ` the next token might be nearly determined, and `k = 50` admits 49
tokens with a combined probability under $10^{-3}$. Mid-sentence in open prose
the model may be genuinely spread over hundreds of reasonable continuations, and
`k = 50` cuts most of them.

### Top-p, or nucleus sampling

Top-p replaces the fixed count with a fixed mass. Sort the probabilities
descending, $p_{(1)} \ge p_{(2)} \ge \dots \ge p_{(V)}$, and write the cumulative
sums as

$$
C_n = \sum_{r=1}^{n} p_{(r)}, \qquad C_0 = 0.
$$

The *nucleus* is the shortest prefix whose mass reaches $p$:

$$
n(p) = \min \{\, n \ge 1 : C_n \ge p \,\}.
$$

The set exists for any $p \le 1$ because $C_V = 1$, and $n(p) \ge 1$ always, so
the nucleus is never empty. Keep ranks $1$ through $n(p)$ and renormalize by
$C_{n(p)}$.

The whole difficulty is turning that definition into a per-element predicate.
Rank $r$ is kept exactly when $r \le n(p)$, and since $C$ is non-decreasing,

$$
r \le n(p) \iff C_{r-1} < p.
$$

Read that carefully. The test is on $C_{r-1}$, the mass *strictly before* rank
$r$, not on $C_r$. Since `cumsum` gives you $C_r$, the predicate is written as

```python
sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
sorted_probs = sorted_logits.softmax(dim=-1)
cumulative = sorted_probs.cumsum(dim=-1)          # C_r
remove = cumulative - sorted_probs >= p           # C_{r-1} >= p
remove[..., 0] = False                            # always keep the top token
```

`cumulative - sorted_probs` is $C_r - p_{(r)} = C_{r-1}$. The last line is
belt-and-braces: $C_0 = 0 < p$ already guarantees rank 1 survives for any
$p > 0$, but it costs nothing and it protects against `p = 0`.

Writing `cumulative >= p` instead is the classic bug. It drops the token that
crosses the threshold, so the retained mass is $C_{n(p) - 1} < p$ — strictly less
than the parameter asked for.

#### Worked example

Take the five logits the lab uses, $z = (3, 2, 1, 0, -1)$. Exponentiating gives
$20.086, 7.389, 2.718, 1.000, 0.368$ with sum $31.561$, so

$$
p = (0.6364,\ 0.2341,\ 0.0861,\ 0.0317,\ 0.0117).
$$

Cumulative: $C_1 = 0.6364$, $C_2 = 0.8705$, $C_3 = 0.9567$, $C_4 = 0.9883$,
$C_5 = 1$.

At `p = 0.7`: $C_0 = 0 < 0.7$ keeps rank 1; $C_1 = 0.6364 < 0.7$ keeps rank 2;
$C_2 = 0.8705 \ge 0.7$ drops rank 3. Two tokens survive, holding $0.8705$ of the
mass, renormalized to $0.7311$ and $0.2689$.

At `p = 0.5`: $C_0 = 0 < 0.5$ keeps rank 1; $C_1 = 0.6364 \ge 0.5$ drops rank 2.
One token survives, which is correct — the top token alone already exceeds 0.5.

Now run the buggy predicate at `p = 0.7`. It tests $C_r \ge p$: $C_1 = 0.6364 <
0.7$ keeps rank 1, $C_2 = 0.8705 \ge 0.7$ drops rank 2. One token instead of two,
and the retained mass is $0.6364$ against the $0.7$ the caller requested. The
output is still fluent, which is why this bug survives code review.

The lab checks both of these cases by name.

### Min-p

Min-p keeps every token at least `min_p` times as likely as the most likely one:

```python
def min_p_filter(logits, min_p):                       # logits: (batch, vocab)
    if min_p <= 0.0:
        return logits
    probs = logits.softmax(dim=-1)                     # (batch, vocab)
    threshold = probs.max(dim=-1, keepdim=True).values * min_p   # (batch, 1)
    return logits.masked_fill(probs < threshold, float("-inf"))
```

The rule is a ratio, so the normalizer cancels. With $m$ for the `min_p`
parameter and $z_{\max} = \max_j z_j$:

$$
p_i \ge m \, p_{\max} \iff \frac{e^{z_i}}{Z} \ge m \frac{e^{z_{\max}}}{Z} \iff e^{z_i - z_{\max}} \ge m \iff z_i \ge z_{\max} + \ln m.
$$

Min-p is a *logit window*. It keeps every token within $|\ln m|$ of the top
logit and nothing else. At `min_p = 0.05` the window is
$\ln 0.05 = -3.00$, so "every token within 3 logits of the best one". At
`min_p = 0.3` it is $\ln 0.3 = -1.204$.

Check that against the worked example: $z_{\max} = 3$, so `min_p = 0.3` cuts at
$3 - 1.204 = 1.796$ and keeps the logits 3 and 2. Two tokens, which is what the
lab asserts.

Two things follow from the window form. First, min-p is unaffected by anything
that only renormalizes — a preceding top-k changes $Z$ but not any logit
difference, so min-p's keep-set is the same whether it runs before or after.
Second, min-p is very much affected by temperature, because temperature rescales
the differences. Applied to tempered logits $z/T$, the window in *original*
logit units is $T |\ln m|$ wide. At `temperature = 1.5` and `min_p = 0.05` that
is $1.5 \times 3.00 = 4.49$ logits, half again as wide as at temperature 1.

That widening is the reason min-p holds up at high temperature where top-p
does not. Top-p's threshold is a mass, and flattening the distribution pushes
mass into the tail, so the nucleus grows without bound. Min-p's threshold tracks
the model's own confidence, so it grows only in proportion to $T$.

## The order, and which pairs commute

The engine applies the transforms in this order:

1. **Penalties**, on raw logits.
2. **Temperature**, dividing logits.
3. **Truncation** — top-k, then min-p, then top-p.
4. **Softmax and draw.**

Not every adjacent pair actually matters. Work out which, because knowing the
three that do is more useful than memorizing the list.

| Pair | Commutes? | Why |
|---|---|---|
| Repetition penalty, temperature | Yes | Both are multiplicative and $T > 0$ preserves the sign test |
| Presence or frequency penalty, temperature | No | Additive against multiplicative |
| Temperature, top-k | Yes | Dividing by $T > 0$ preserves the ranking |
| Temperature, min-p | No | The window scales with $T$ |
| Temperature, top-p | No | The nucleus is a mass, not a ranking |
| Penalties, any truncation | No | Penalties change the ranking |

### Repetition penalty commutes with temperature

The repetition penalty maps a seen token's logit $z$ to $z/\rho$ when $z > 0$
and $z\rho$ when $z \le 0$, for $\rho > 1$. Apply temperature first and you get
$(z/T)/\rho = z/(T\rho)$ on the positive branch and $(z/T)\rho = z\rho/T$ on the
negative one. Apply the penalty first and you get $(z/\rho)/T$ and $(z\rho)/T$ —
the same two expressions. Dividing by a positive $T$ also cannot change the sign,
so the branch taken is the same. The order is irrelevant.

### Additive penalties do not

Presence and frequency penalties subtract. Applied before temperature, a penalty
$\alpha$ becomes $(z - \alpha)/T = z/T - \alpha/T$. Applied after, it is
$z/T - \alpha$. The effective strength differs by a factor of $T$.

Applying penalties first is the convention, so the effective log-odds shift is
$\alpha/T$. At `temperature = 0.7`, a presence penalty of `0.5` behaves like
`0.714` in the space the truncations see. That is worth documenting rather than
"fixing": users tune penalties at a fixed temperature and expect the same
numbers to work across engines.

### Temperature before truncation

This is the one that changes output visibly. Take the same five logits and
`top_p = 0.7`, `temperature = 2.0`.

Correct order — temperature, then top-p. The tempered logits are
$(1.5, 1.0, 0.5, 0.0, -0.5)$, exponentials $4.482, 2.718, 1.649, 1.000, 0.607$
summing to $10.455$, so

$$
p(T{=}2) = (0.4287,\ 0.2600,\ 0.1577,\ 0.0956,\ 0.0580).
$$

Cumulative: $0.4287$, $0.6886$, $0.8463$. The test $C_{r-1} < 0.7$ keeps ranks 1,
2 and 3 — because $C_2 = 0.6886$ is still below $0.7$. Three tokens, renormalized
to $(0.5065, 0.3072, 0.1863)$.

Wrong order — top-p, then temperature. Top-p on the untempered distribution keeps
two tokens, as computed earlier. Tempering those two gives logits $1.5$ and
$1.0$, so $(0.6225, 0.3775)$.

The support went from three tokens to two and the top token's probability from
$0.51$ to $0.62$. Worse, the parameter no longer describes anything: in the wrong
order the two surviving tokens hold $0.4287 + 0.2600 = 0.6886$ of the tempered
mass, under the $0.7$ that was asked for. The user turned temperature up to get
more variety and got a *narrower* nucleus.

### Why top-k, then min-p, then top-p

Top-k first because it is a hard cap and costs one `topk` over the row; running
it first shrinks nothing downstream but bounds the damage if the other two are
left at their defaults.

Min-p anywhere, by the argument above — it is a logit window and truncation does
not move logit differences. It sits in the middle because that reads well next to
its neighbours, not because it must.

Top-p last because it is the only one of the three that reads the renormalized
masses. Running it after the others means its nucleus is computed on the
distribution you will actually sample from, which is what "the smallest set whose
mass reaches $p$" is supposed to mean. Running it first would let tokens that
top-k is about to discard inflate the cumulative sum.

## Penalties as logit transforms

Both penalty families are maps $z \mapsto z'$ applied before temperature. They
differ in whether they respect the log-odds geometry.

### Repetition penalty

```python
def apply_repetition_penalty(logits, previous, penalty):
    if penalty == 1.0:
        return logits
    scores = torch.gather(logits, 1, previous)                     # (batch, n_seen)
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits.scatter(1, previous, scores)                     # (batch, vocab)
```

The asymmetry is the whole function. The goal is $z' < z$ for every seen token.
For $z > 0$, dividing by $\rho > 1$ moves the logit toward zero and down. For
$z < 0$, dividing by $\rho$ moves it *up*: $-2 / 1.1 = -1.82 > -2$, which
encourages the token you meant to discourage. Multiplying instead gives
$-2 \times 1.1 = -2.2 < -2$, which is down. Both branches decrease the logit;
that is the only invariant the function has.

Roughly half of a 248,320-entry vocabulary sits below zero at any position, so an
implementation that divides unconditionally gets the sign wrong about half the
time — and the visible symptom is a model that repeats *more* when you raise the
penalty.

The lab checks this with `logits = [2.0, -2.0, 0.5]`, `previous = [0, 1]`,
`penalty = 2.0`. The result must be `[1.0, -4.0, 0.5]`: the positive logit halved,
the negative one doubled in magnitude, the unseen one untouched. It also checks
that the input tensor is not modified in place, which is why `scatter` is used
rather than `scatter_`.

The quirk worth knowing: the shift in log-odds is $-z(1 - 1/\rho)$ on the
positive branch and $z(\rho - 1)$ on the negative one. Both are proportional to
$|z|$, so a token with a logit near zero is barely penalized at all, and a token
with a logit of exactly zero is not penalized by any $\rho$ whatsoever.

### Presence and frequency penalties

```python
def apply_presence_frequency_penalty(logits, counts, presence, frequency):
    # counts: (batch, vocab), occurrences of each token so far
    if presence == 0.0 and frequency == 0.0:
        return logits
    return (logits
            - presence * (counts > 0).to(logits.dtype)
            - frequency * counts.to(logits.dtype))
```

These are constant shifts, so they are the well-behaved family:

$$
z_i' = z_i - \alpha_{\text{pres}} \mathbf{1}[c_i > 0] - \alpha_{\text{freq}} c_i.
$$

The log-odds of a penalized token against an unpenalized one drop by exactly
$\alpha_{\text{pres}} + \alpha_{\text{freq}} c_i$, whatever the logits were. A
presence penalty of `0.5` multiplies the odds by $e^{-0.5} = 0.607$, a 39%
reduction, uniformly across the vocabulary. There is no sign branch and no
dependence on scale.

Presence applies once per distinct token seen; frequency scales with the count,
so it keeps biting as a token recurs. The `counts` tensor is `(batch, vocab)` and
is the same size as the logits, which the cost section below takes into account.

## The Gumbel-max trick

Everything so far ends at `torch.multinomial`, which wants a normalized
probability vector, builds a cumulative sum, draws a uniform, and does a search.
There is a way to sample from a categorical distribution with no normalization,
no cumulative sum, and no search. One argmax.

Let $G_1, \dots, G_V$ be independent standard Gumbel variables. You get one from
a uniform $U \sim \text{Uniform}(0,1)$ as $G = -\ln(-\ln U)$. Then

$$
\operatorname{arg\,max}_{i} \, (z_i + G_i) \sim \operatorname{softmax}(z).
$$

### Why it works

The standard Gumbel has CDF $F(g) = \exp(-e^{-g})$ and density
$f(g) = e^{-g} \exp(-e^{-g})$. Condition on $G_i = g$. Index $i$ wins when
$z_j + G_j < z_i + g$ for every $j \ne i$, that is $G_j < g + z_i - z_j$, which
has probability $\exp(-e^{-(g + z_i - z_j)})$.

Multiply over $j \ne i$ and integrate:

$$
P(i \text{ wins}) = \int_{-\infty}^{\infty} e^{-g} \exp(-e^{-g}) \prod_{j \ne i} \exp\!\left(-e^{-g} e^{z_j - z_i}\right) dg.
$$

The lone $\exp(-e^{-g})$ is the $j = i$ term of the same product, since
$e^{z_i - z_i} = 1$. Folding it in and writing $S = \sum_{j=1}^{V} e^{z_j - z_i}$
collapses the product to $\exp(-S e^{-g})$:

$$
P(i \text{ wins}) = \int_{-\infty}^{\infty} e^{-g} \exp\!\left(-S e^{-g}\right) dg.
$$

Substitute $u = e^{-g}$, so $du = -e^{-g} dg$ and the limits flip:

$$
P(i \text{ wins}) = \int_{0}^{\infty} e^{-S u} \, du = \frac{1}{S} = \frac{e^{z_i}}{\sum_{j} e^{z_j}}.
$$

Which is $p_i$. The derivation is exact, not asymptotic, and it never needed the
normalizer — $S$ produced it.

### Why an engine wants this

The argmax form has properties the cumulative-sum form does not.

It needs no normalization, so it composes with $-\infty$ masks for free: a
masked token has $z_i + G_i = -\infty$ and cannot win.

It is one pass over the row with a per-element random number, so it is a single
fused kernel with no data-dependent control flow. The cumulative-sum form needs a
prefix scan and a search.

It parallelizes across a batch trivially. Every row does its own argmax, and no
row's work depends on another's.

Two cautions. $U$ must be strictly inside $(0, 1)$: $U = 0$ gives $-\ln 0 =
\infty$ and then $-\ln(\infty) = -\infty$, and $U = 1$ gives $-\ln(0)$ the other
way. Generate on the half-open interval and reject the endpoint. And the double
logarithm loses precision in float32 for $U$ very close to 1; the equivalent
*exponential race* form, drawing $E_i \sim \text{Exp}(1)$ and taking
$\operatorname{arg\,max}_i \, p_i / E_i$, avoids one of the logarithms.

## Seeding and reproducibility

Given a seed and identical inputs, a generation must reproduce exactly. That
means threading an explicit generator rather than relying on global state:

```python
gen = torch.Generator(device=logits.device).manual_seed(1234)
token = torch.multinomial(probs, num_samples=1, generator=gen)   # (batch, 1)
```

The global RNG is shared with every other operation in the process. A dropout
layer, a shuffled dataloader, or another request's sampler consumes draws from
it, so the same seed reproduces the same output only if nothing else in the
process ever draws. An explicit `torch.Generator` is owned by the request.

Reproducibility *across batch sizes* is a much stronger requirement and is
usually not worth paying for. Floating-point addition is not associative, so a
reduction that splits a row differently at batch 1 and batch 8 produces logits
that differ in the last bits. That would not matter if the winner were far ahead,
but the LM head runs in bfloat16, whose 8-bit significand resolves logits to
about `0.39%` relative — and with 248,320 candidates, several of them are
routinely within that of each other. An argmax near a tie flips, the token
changes, and every subsequent token changes with it.

The practical guarantee is: same seed, same batch composition, same engine
version, same output. Say that in your API documentation and do not promise more.

## What sampling costs

The per-step traffic is small in absolute terms and easy to dismiss. Do the
arithmetic before dismissing it.

### One softmax

The vocabulary is $V = 248{,}320$. In float32 one row of logits is

$$
248{,}320 \times 4 = 993{,}280 \text{ bytes} = 0.993 \text{ MB}.
$$

One softmax pass reads that and writes it: $1.99$ MB. The arithmetic is one
subtraction, one exponential, and one division per element, so call it
$3V = 745{,}000$ FLOPs. Arithmetic intensity:

$$
\frac{7.45 \times 10^{5}}{1.99 \times 10^{6}} = 0.375 \ \text{FLOPs per byte}.
$$

Chapter 10 put the A100's ridge point at 161 FLOPs per byte. Softmax sits below
it by a factor of $161 / 0.375 = 429$. It is as memory bound as anything in the
engine. At the measured copy bandwidth of 1275 GB/s the pass takes $1.99 \times
10^{6} / 1.275 \times 10^{12} = 1.6$ microseconds; the arithmetic, at 312
TFLOP/s, would take 2.4 nanoseconds. Only the bytes matter.

That also tells you what a naive softmax costs. PyTorch's multi-pass form — one
kernel for the max, one for the exponentials, one for the sum, one for the
division — reads the row four times instead of once.
`engine/kernels/softmax_triton.py` does all four in one trip for this reason.

### The whole pipeline, at batch 1

Count each operation in `engine/sampling.py` as read plus write in float32.

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

At 1275 GB/s, 26 MB takes about 21 microseconds. Set that against the decode step
it sits inside: the model's weights are 53.8 GB in bfloat16, and reading them
once at 1275 GB/s takes

$$
\frac{53.8 \times 10^{9}}{1.275 \times 10^{12}} = 42.2 \text{ ms}.
$$

So by bytes, sampling is 0.05% of a decode step. It is negligible.

By launches, it is not. The table is about twenty separate kernels, each
operating on a single row of 248,320 elements. Chapter 10 put kernel launch
overhead at 5 to 10 microseconds, so the pipeline spends 100 to 200 microseconds
in overhead against 21 microseconds of traffic — the launches cost five to ten
times what the memory does. And a row of 248,320 elements is 970 blocks at 256
threads, barely nine per SM across 108 SMs, so there is not enough work in any
one kernel to hide the next launch behind it.

Three further costs do not show up in either count.

`torch.multinomial` and any `.item()` on the result force a device
synchronization. The GPU drains, and the CPU cannot queue the next step's
launches until it returns. On a step whose ideal length is 42 ms that stall is
absorbed, but it also means the sampler is the one point in the loop where the
pipeline is guaranteed to empty.

Data-dependent control flow — the sort in particular — cannot be captured in a
CUDA graph. Chapter 10 noted that a decode step runs several hundred kernels and
that graph capture is how production engines remove that overhead. A sampler with
a sort in it is a hole in the graph, and the hole costs more than the sampler's
own time.

And the cost scales with batch size while the launch count does not. At batch
128 the same pipeline moves $128 \times 26 = 3.3$ GB, which is $3.3 \times
10^{9} / 1.275 \times 10^{12} = 2.6$ ms against the same 42.2 ms weight read —
6% of the step, and now genuinely worth removing.

### Which is why it is one kernel

Production engines do not run the table above. They run one fused kernel per
step, or a very small number, over the whole batch at once:

- Penalties, temperature, and the masks are all elementwise or row-reduction
  work, so they fuse into a single pass that keeps the row in registers.
- Top-k is served by a partial selection rather than a full sort.
- Top-p's cumulative sum runs on the top-k survivors, not on 248,320 entries.
- The draw is the Gumbel-max argmax, which needs neither a scan nor a
  synchronization, so the token index stays on the device and the next step's
  launches queue behind it without a stall.

The result is a sampler that is one launch, graph-capturable, and bound by the
one unavoidable read of the logits. That is the shape to aim for, and the
arithmetic above is why.

## What goes wrong

**`NaN` in the output distribution.** A missing max subtraction, or a softmax run
in float16. Symptom: `torch.multinomial` raises, or returns index 0 forever.
Check for non-finite logits before the softmax while debugging.

**Temperature appears disconnected.** You scaled probabilities instead of logits,
and the scale cancelled. Check that the change happens before the softmax.

**Top-p under-delivers its mass.** The boundary token is being dropped. Compare
`cumulative - sorted_probs >= p`, not `cumulative >= p`.

**Raising the repetition penalty makes repetition worse.** The negative branch is
dividing instead of multiplying.

**Everything is masked.** With `min_p` near 1 or `top_p` near 0, a keep-set can
come out empty if the guards are missing, and the softmax of an all-`-inf` row is
all `NaN`. Every filter needs a "keep at least one" floor.

**A fixed seed does not reproduce.** Either the global RNG is in use, or
something else in the process is consuming draws, or the batch composition
changed between runs.

**A one-token vocabulary shift.** Somebody applied a filter to probabilities and
then re-softmaxed. Applying softmax twice is not idempotent; it flattens the
distribution a second time.

## Check your understanding

**Why is subtracting the maximum safe but subtracting the mean unsafe?**

Both are shifts, and shift invariance holds for any constant, so both are
*mathematically* safe. Only the maximum guarantees every exponent is at most 0.
Subtracting the mean leaves the largest logit above zero by however far it exceeds
the mean, and on a peaked 248,320-entry row that gap can easily exceed 88.7.

**At `temperature = 1.5` and `min_p = 0.05`, how wide is the keep window in raw
logits, and how many tokens does that admit?**

The window is $T |\ln m| = 1.5 \times 3.00 = 4.49$ logits below the top. How many
tokens it admits depends entirely on the row: on a peaked distribution, one or
two; on a flat one, thousands. That adaptivity is the point — min-p does not
promise a count or a mass, it promises a confidence ratio.

**The pipeline moves 26 MB and takes 21 microseconds of traffic, inside a step
that takes 42 ms. Why is it still worth fusing?**

Because 21 microseconds of traffic is delivered by twenty launches costing 100 to
200 microseconds, because the sort prevents CUDA graph capture of the whole step,
and because the traffic term scales with batch size while the step's 42 ms weight
read does not. At batch 128 it is 6% of the step.

**Does applying top-k before min-p give a different result from applying min-p
before top-k?**

No. Min-p's keep-set is $\{i : z_i \ge z_{\max} + \ln m\}$, which depends only on
logit differences, and top-k changes no logit that survives it. The two keep-sets
intersect in the same place either way. Temperature is the transform whose
position genuinely matters.

## Lab

Implement the full pipeline on CPU: `apply_repetition_penalty`, `top_k_filter`,
`top_p_filter`, `min_p_filter`, and `sample`.

The harness works on the five-logit row from the worked example and checks, among
others:

- Top-k keeps exactly `k` tokens, keeps the highest ones, and `k = 0` is a no-op.
- Top-p at `p = 0.7` keeps two tokens — the boundary case — and at `p = 0.5`
  keeps one. A tiny `p` still leaves something to sample from, and `p = 1` is a
  no-op.
- Min-p at `0.3` keeps two tokens on the peaked row and everything on a flat one.
- The repetition penalty turns `[2.0, -2.0, 0.5]` into `[1.0, -4.0, 0.5]` with
  `penalty = 2.0`, leaves unseen tokens alone, is a no-op at `1.0`, and does not
  modify its input in place.
- `temperature = 0` returns the argmax exactly, and `sample` returns shape
  `(batch, 1)`.
- The same `torch.Generator` seed reproduces the same tokens, a different seed
  usually does not, and `top_k = 1` forces the argmax.
- Over 200 draws from a row whose top five logits are `[5, 4, 3, 2, 1]` and whose
  other 507 are `-10`, every draw with `top_p = 0.9` comes from those five.

It reports `nucleus_size` and `seed_reproducible` as metrics. The lab's `sample`
takes its parameters as keyword arguments rather than a `SamplingParams` object,
so read the starter's signature before you begin.

## Further reading

- [The curious case of neural text degeneration](https://arxiv.org/abs/1904.09751) — the nucleus sampling paper.
- [Turning up the heat: min-p sampling for creative and coherent LLM outputs](https://arxiv.org/abs/2407.01082)
- [CTRL: a conditional transformer language model for controllable generation](https://arxiv.org/abs/1909.05858) — where the repetition penalty comes from.
- [Categorical reparameterization with Gumbel-softmax](https://arxiv.org/abs/1611.01144) — the Gumbel-max trick and its differentiable relaxation.
- [The Gumbel-max trick for discrete distributions](https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/)
