---
title: Linear attention and the gated delta rule
slug: 06-gated-delta-rule
part: "Part 2 — A forward pass"
summary: The recurrence that runs 48 of this model's 64 layers, derived from softmax attention, and the chunked form that makes prefill possible.
minutes: 150
gpu: true
objectives:
  - Derive linear attention from softmax attention and name exactly what the kernel trick removes.
  - Quantify the capacity of a fixed-size state and show the interference term when it overflows.
  - Derive the delta rule as gradient descent on a squared error, and explain why the keys are L2-normalized.
  - Read a head's forget gate as a time constant in tokens.
  - Derive the chunked parallel form, including the triangular solve, and count its FLOPs.
  - Implement the chunked form and verify it against the sequential recurrence.
lab: 06-delta-rule
---

# Linear attention and the gated delta rule

Three quarters of this model's layers don't use attention. They run a recurrence
with a fixed-size state, which is why a sequence at 128k context carries the same
147.8 MiB of layer state as a sequence at 10 tokens.

This is the hardest chapter in the course. It is also the one where the payoff is
largest, because the recurrence is what makes the model's memory profile flat
instead of linear, and the chunked form is what stops that recurrence from making
prefill hopeless.

Take it in two halves. The first half derives the update rule one term at a time:
linear attention, then the delta rule, then the gate. The second half turns that
sequential rule into something a GPU can run in parallel. The second half is
longer and has more algebra in it, but nothing beyond an induction and a
triangular solve.

## Before you start

The chapter assumes the following, and nothing else.

**Outer products.** For column vectors $k \in \mathbb{R}^{d_k}$ and
$v \in \mathbb{R}^{d_v}$, the outer product $k v^\top$ is the
$d_k \times d_v$ matrix with entry $(a, b)$ equal to $k_a v_b$. It has rank 1.
Every state update in this chapter adds one outer product to a matrix.

**Reading an outer product back.** If $S = k v^\top$ then
$S^\top k = v \, (k^\top k) = \lVert k \rVert^2 v$. With a unit-norm key,
$S^\top k = v$ exactly. That one identity is the whole idea of an associative
memory: write with an outer product, read with a matrix-vector product.

**Matrix calculus for one scalar function.** You need the gradient of
$\tfrac{1}{2}\lVert S^\top k - v \rVert^2$ with respect to the matrix $S$. The
chapter does the differentiation entry by entry, so you only need partial
derivatives.

**Symmetric projectors.** For a unit-norm $k$, the matrix $P = k k^\top$
satisfies $P^2 = P$. Its eigenvalues are 1, with eigenvector $k$, and 0, with
multiplicity $d_k - 1$ on the subspace orthogonal to $k$.

**Tensor shapes and einsum.** Chapter 0a covers the `(batch, heads, seq, dim)`
convention and `torch.einsum`. Every listing here carries its shapes in comments.

**Floating point.** Chapter 0a covers float32's exponent range. You need two
numbers from it: the smallest normal float32 is about $1.18 \times 10^{-38}$,
and the smallest subnormal is about $1.40 \times 10^{-45}$. The underflow
section uses both.

Nothing in this chapter needs a GPU to understand. The lab runs on one to measure
the speedup, but every check in it is device agnostic.

## From softmax attention to a recurrence

Start with what you are replacing. Softmax attention computes, for query position
$t$ in a causal model:

$$
o_t = \frac{\sum_{j \le t} \exp(q_t^\top k_j / \sqrt{d_k}) \, v_j}
           {\sum_{j \le t} \exp(q_t^\top k_j / \sqrt{d_k})}
$$

Here $q_t, k_j \in \mathbb{R}^{d_k}$ are the query and key vectors, and
$v_j \in \mathbb{R}^{d_v}$ is the value. Chapter 7 derives this formula properly;
for now, the only feature that matters is where $q_t$ appears. It sits inside the
$\exp$, coupled to every $k_j$. You cannot pull it out of the sum, so you cannot
precompute anything that summarizes the past independently of the query. That is
precisely why softmax attention must keep every past key and value.

### The kernel trick

The standard way to break that coupling is to assume the exponential is a kernel
with a known feature map. Suppose there is a function
$\phi : \mathbb{R}^{d_k} \to \mathbb{R}^{m}$ with

$$
\exp(q^\top k / \sqrt{d_k}) \approx \phi(q)^\top \phi(k)
$$

Substitute it into the numerator and the associativity of matrix products does
the rest:

$$
\sum_{j \le t} \phi(q_t)^\top \phi(k_j) \, v_j
= \phi(q_t)^\top \left( \sum_{j \le t} \phi(k_j) v_j^\top \right)
$$

The bracketed term is an $m \times d_v$ matrix that does not mention $q_t$. Write
it as $S_t$ and you have a state that summarizes the entire past, at a size that
does not depend on $t$:

$$
S_t = S_{t-1} + \phi(k_t) v_t^\top, \qquad o_t = S_t^\top \phi(q_t)
$$

The denominator factors the same way, against a vector
$z_t = z_{t-1} + \phi(k_t)$, giving $o_t = S_t^\top \phi(q_t) / (z_t^\top \phi(q_t))$.

### What this model actually does

This model takes $\phi$ to be the identity map, so $m = d_k$, and drops the
denominator entirely:

$$
S_t = S_{t-1} + k_t v_t^\top, \qquad o_t = S_t^\top q_t
$$

That is linear attention. Be clear about what was given up to get here, because
both losses are real.

**The kernel identity is gone.** With $\phi$ the identity, $\phi(q)^\top \phi(k)$
is $q^\top k$, which is not an approximation of $\exp(q^\top k / \sqrt{d_k})$ in
any useful sense. This is not softmax attention computed more cheaply. It is a
different operator that happens to have the same input and output signature, and
it has to be trained as itself. The kernel derivation explains where the shape of
the recurrence comes from; it does not license the substitution.

**The normalizer is gone.** Softmax attention returns a convex combination of
values: the weights are non-negative and sum to 1, so the output is inside the
convex hull of the values it read. Linear attention's output is an unconstrained
linear combination. Weights can be negative and their sum can be anything. In
exchange, the layer gains something softmax attention does not have — the ability
to output near zero when nothing relevant is in the state. The gated RMSNorm at
the end of the layer, described later, is what keeps the unbounded magnitude in
check.

**Reading is now free.** $o_t = S_t^\top q_t$ costs $d_k d_v$ multiply-adds no
matter how long the context is. Generating a token is $O(1)$ in context length
instead of $O(L)$. For this model, $d_k = d_v = 128$, so one head's read is
$2 \times 128 \times 128 = 32{,}768$ FLOPs at token 10 and at token 100,000.

What you pay for that is capacity, and the price is worth quantifying.

## How much can a fixed state hold

The state $S$ is a $d_k \times d_v$ matrix. In this model, that is
$128 \times 128 = 16{,}384$ numbers per head, held in float32. Across 48 value
heads and 48 linear layers that comes to 147.8 MiB per sequence, which chapter 2
derives.

Ask what $S$ can store. Suppose you write $n$ associations with unit-norm keys:

$$
S = \sum_{i=1}^{n} k_i v_i^\top
$$

Now read it with key $k_m$:

$$
S^\top k_m = \sum_{i=1}^{n} v_i \, (k_i^\top k_m)
= v_m + \underbrace{\sum_{i \ne m} (k_i^\top k_m) \, v_i}_{\text{interference}}
$$

The first term is what you wanted. The second is everything else in the state,
weighted by how much its key overlaps with the one you asked for.

**When the interference vanishes.** If the keys are mutually orthogonal, every
$k_i^\top k_m$ with $i \ne m$ is zero and the read is exact. You can fit at most
$d_k$ mutually orthogonal directions in $\mathbb{R}^{d_k}$, so the hard ceiling is
$d_k = 128$ perfect associations per head. Write a 129th and the keys are
necessarily linearly dependent, so some interference is forced.

**How it grows before the ceiling.** Real keys are not orthogonal. For two
independent random unit vectors in $\mathbb{R}^{d}$, the expected squared inner
product is $1/d$. So the interference term has expected squared norm about
$(n-1)/d_k$ times the typical $\lVert v \rVert^2$, against a signal of
$\lVert v_m \rVert^2$. The signal-to-noise ratio is roughly

$$
\mathrm{SNR} \approx \frac{d_k}{n - 1}
$$

With $d_k = 128$, the noise energy matches the signal energy at about $n = 129$
writes. Past a few hundred tokens, a plain linear-attention head reading a key it
wrote long ago gets back mostly other keys' values. This is why plain linear
attention degrades over long sequences, and it degrades smoothly rather than
failing loudly, which is worse.

**The failure that motivates the delta rule.** The clearest case is writing the
same key twice. Suppose token 3 writes $(k, v_a)$ and token 90 writes $(k, v_b)$,
meaning "this slot now holds $v_b$ instead". With plain accumulation:

$$
S = k v_a^\top + k v_b^\top = k (v_a + v_b)^\top,
\qquad S^\top k = v_a + v_b
$$

The read returns the sum, not the update. The state has no way to express
replacement, only addition. Every overwrite makes the entry worse.

## The delta rule

The fix is to make the write *conditional on what is already there*. Before
writing, read the state at $k_t$, and write back only the part that is missing.

### Deriving it from an objective

State the goal as an optimization. At step $t$ you want a state that returns
$v_t$ when read with $k_t$. Define the loss

$$
\mathcal{L}(S) = \tfrac{1}{2} \lVert S^\top k_t - v_t \rVert_2^2
$$

Differentiate entry by entry. Write $r = S^\top k_t$, so
$r_b = \sum_a S_{ab} (k_t)_a$. Then

$$
\frac{\partial \mathcal{L}}{\partial S_{ab}}
= (r_b - (v_t)_b) \cdot \frac{\partial r_b}{\partial S_{ab}}
= (r_b - (v_t)_b) \, (k_t)_a
$$

Entry $(a, b)$ of the gradient is a product of the $a$-th entry of $k_t$ and the
$b$-th entry of the residual. That is exactly an outer product:

$$
\nabla_S \mathcal{L} = k_t \, (S^\top k_t - v_t)^\top
$$

Take one step of gradient descent with step size $\beta_t$:

$$
S_t = S_{t-1} - \beta_t \, \nabla_S \mathcal{L}
    = S_{t-1} + \beta_t \, k_t \, (v_t - S_{t-1}^\top k_t)^\top
$$

That is the delta rule, and it is the classical one from online learning. Read
$S_{t-1}^\top k_t$ as "what does the state currently return for this key". If it
already equals $v_t$, the correction is zero and nothing is written. If it
differs, the state moves toward $v_t$ in proportion to how far off it is.

In the engine, $\beta_t$ is produced per value head by a sigmoid, so it lies in
$(0, 1)$:

```python
b_raw, a_raw = ba.chunk(2, dim=-1)     # each (batch, seq, 48)
beta = torch.sigmoid(b_raw)            # (batch, seq, 48), in (0, 1)
```

### Why it stops the state from saturating

Expand the update and collect the terms in $S_{t-1}$:

$$
S_t = S_{t-1} - \beta_t k_t k_t^\top S_{t-1} + \beta_t k_t v_t^\top
    = (I - \beta_t k_t k_t^\top) \, S_{t-1} + \beta_t k_t v_t^\top
$$

The write is now two moves: erase along $k_t$, then add. Go back to the
double-write example with $\beta = 1$. The first write puts $k v_a^\top$ in the
state. The second write first applies $I - k k^\top$, which annihilates the $k$
direction and leaves $0$, then adds $k v_b^\top$. Reading gives $v_b$. The state
replaced rather than accumulated.

### Why the keys are L2-normalized

The erase factor $I - \beta k k^\top$ is only well behaved when $\lVert k \rVert = 1$.

With a unit-norm key, $P = k k^\top$ is an orthogonal projector onto the line
through $k$, so its eigenvalues are 1 and 0. The eigenvalues of $I - \beta P$ are
therefore

$$
1 - \beta \quad \text{along } k,
\qquad 1 \quad \text{on the } d_k - 1 \text{ directions orthogonal to } k
$$

Read off what the step size does. At $\beta = 0$ the matrix is $I$ and nothing is
erased. At $\beta = 1$ the $k$ direction is erased completely, which is a full
replacement. In between it is a partial erase, and every direction orthogonal to
$k$ is untouched — the write disturbs nothing it was not aimed at. Since
$\beta \in (0,1)$, the spectral radius is exactly 1 and repeated writes never
amplify the state.

Now drop the normalization. The eigenvalue along $k$ becomes
$1 - \beta \lVert k \rVert^2$. Stability needs
$\lvert 1 - \beta \lVert k \rVert^2 \rvert \le 1$, that is
$0 \le \beta \lVert k \rVert^2 \le 2$. With $\beta$ capped at 1 by the sigmoid,
you need $\lVert k \rVert^2 \le 2$, and a raw linear projection has no reason to
respect that. A key of norm 2 at $\beta = 0.8$ gives an eigenvalue of
$1 - 0.8 \times 4 = -2.2$: every write along that direction flips the sign of
whatever was there and multiplies it by 2.2. Over a few hundred tokens the state
overflows to infinity, and then to NaN.

So the engine normalizes:

```python
q = F.normalize(q, dim=-1)   # (batch, seq, 16, 128), each row unit norm
k = F.normalize(k, dim=-1)   # (batch, seq, 16, 128)
```

Queries are normalized too. That is not required for stability — the query never
touches the state — but it keeps the output magnitude from tracking the raw
projection scale, and it makes $q_t^\top k_j$ a cosine similarity in $[-1, 1]$.

## The gate

The delta rule fixes overwriting. It does not let the layer forget something it
was never asked about again. Add a scalar forget gate $\alpha_t \in (0,1)$ that
shrinks the whole state before the write:

$$
S_t = \alpha_t S_{t-1}
    + \beta_t \, k_t \, (v_t - \alpha_t S_{t-1}^\top k_t)^\top,
\qquad o_t = S_t^\top q_t
$$

Note where $\alpha_t$ appears inside the correction. The state is decayed first,
and the delta rule then corrects the *decayed* state, not the old one. In the
collected form:

$$
S_t = (I - \beta_t k_t k_t^\top)\,\alpha_t S_{t-1} + \beta_t k_t v_t^\top
$$

Every head gets its own gate, one scalar per token per value head, so a layer's
48 heads decay at 48 different rates.

### The parameterisation, and why it cannot escape (0, 1)

A gate that leaves $(0,1)$ breaks the layer: above 1 the state grows without
bound, below 0 it oscillates in sign. The engine guarantees the range by
construction rather than by clamping:

```python
decay_rate = F.softplus(a_raw.float() + self.dt_bias) * self.A_log.exp()
alpha = torch.exp(-decay_rate)
```

In symbols, with $a_t$ the raw projection output for this head and token:

$$
r_t = \operatorname{softplus}(a_t + b_{\Delta}) \cdot e^{A_{\log}},
\qquad \alpha_t = e^{-r_t}
$$

Follow the signs. $\operatorname{softplus}(x) = \log(1 + e^x)$ is strictly
positive for every real $x$, and $e^{A_{\log}}$ is strictly positive for every
real $A_{\log}$. So $r_t > 0$ always, and $\alpha_t = e^{-r_t}$ lands strictly
inside $(0, 1)$ for any input whatsoever. The limits are approached but not
reached: as $a_t \to -\infty$, softplus goes to 0 and $\alpha_t \to 1$, which is
perfect retention; as $a_t \to +\infty$, softplus grows like $a_t$ and
$\alpha_t \to 0$, which is a full reset.

Both $A_{\log}$ and the bias $b_{\Delta}$ are learned per value head, shapes
`(48,)` each. The token-dependent part is $a_t$; $A_{\log}$ sets the head's
baseline scale.

### Reading a head's time constant

This is the useful skill. Given $A_{\log}$, how long does this head remember?

Hold $r_t = r$ constant over $n$ steps. The state is multiplied by $\alpha$ each
step, so after $n$ steps whatever was written has been scaled by

$$
\alpha^n = e^{-rn}
$$

Solve for the horizon at which a fraction of the signal survives. For half:

$$
n_{1/2} = \frac{\ln 2}{r} \approx \frac{0.693}{r}
$$

For 90% forgotten, meaning a tenth of the signal left:

$$
n_{90} = \frac{\ln 10}{r} \approx \frac{2.303}{r}
$$

Work an example. Suppose $\operatorname{softplus}(a_t + b_{\Delta}) = 1$, so that
$r = e^{A_{\log}}$ and $A_{\log}$ alone sets the timescale. The arithmetic gives:

| $A_{\log}$ | $r = e^{A_{\log}}$ | $\alpha = e^{-r}$ | $n_{1/2}$ | $n_{90}$ |
|---|---|---|---|---|
| $2$ | $7.389$ | $0.00062$ | $0.09$ | $0.31$ |
| $0$ | $1.000$ | $0.368$ | $0.69$ | $2.3$ |
| $-2$ | $0.1353$ | $0.8734$ | $5.1$ | $17$ |
| $-4$ | $0.01832$ | $0.9819$ | $38$ | $126$ |
| $-6$ | $0.002479$ | $0.99752$ | $280$ | $929$ |
| $-8$ | $0.0003355$ | $0.99966$ | $2066$ | $6864$ |

Those are arithmetic, not measurements: they follow from the formula above and
the assumption that softplus returns 1. The checkpoint's actual $A_{\log}$ values
live in the weights and vary by head.

Two things fall out. First, the spread is enormous — two units of $A_{\log}$ move
the 90% horizon by a factor of $e^2 \approx 7.4$. A layer's 48 heads can cover
everything from "the previous two tokens" to "the last several thousand". That
spread is how a fixed 128 by 128 state serves both local syntax and long-range
recall: the short heads are effectively a wide n-gram window, the long heads are
the associative memory, and the capacity argument above applies separately to
each. Second, an exponential parameterisation is the right one. If $A_{\log}$
were the rate rather than its logarithm, gradient descent could never move a head
from a 2-token horizon to a 2000-token one.

## The sequential form

Here is the recurrence as code, one token at a time. This is `delta_rule_step` in
`engine/layers/linear_attn.py`, which is also the decode path.

```python
def delta_rule_step(state, q, k, v, alpha, beta):
    # state:       (batch, heads, k_dim, v_dim), float32
    # q, k:        (batch, heads, k_dim)
    # v:           (batch, heads, v_dim)
    # alpha, beta: (batch, heads)
    alpha = alpha[..., None, None].float()   # (batch, heads, 1, 1)
    beta = beta[..., None].float()           # (batch, heads, 1)
    q, k, v = q.float(), k.float(), v.float()

    # read = S^T k, the current association for this key.
    read = torch.einsum("bhkv,bhk->bhv", state, k)            # (b, h, v_dim)

    # delta = beta * (v - alpha * read), the correction to write.
    delta = beta * (v - alpha[..., 0] * read)                 # (b, h, v_dim)

    # S = alpha * S + k delta^T.
    state = alpha * state + torch.einsum("bhk,bhv->bhkv", k, delta)

    # o = S^T q, using the state *after* this token's own write.
    out = torch.einsum("bhkv,bhk->bhv", state, q)             # (b, h, v_dim)
    return state, out
```

Three details worth pinning down.

**Everything is float32.** The state is an accumulator that is read and rewritten
every token. bfloat16 has 8 significand bits; accumulating 100,000 updates into
it loses the small corrections entirely. The inputs arrive in bfloat16 and are
promoted on entry.

**`alpha[..., 0]` versus `alpha`.** The gate is broadcast against a matrix in one
place and a vector in the other, so it needs two different trailing shapes. Get
this wrong and PyTorch broadcasts silently to something plausible.

**The output uses the post-write state.** $o_t = S_t^\top q_t$, not
$S_{t-1}^\top q_t$. Token $t$ can read what token $t$ just wrote. This is the
single most common sign-and-index error in the chunked form, and lab 06's first
check catches it: with $\alpha = \beta = 1$ on a fresh state, the first output
must be exactly $(q_1^\top k_1) v_1$.

This code is correct and unusable for prefill. A 2000-token prompt becomes 2000
sequential steps, each launching a handful of tiny kernels that occupy a few
hundred of the A100's threads. The GPU spends its time on launch latency.

## The chunked parallel form

The goal is to compute a whole block of tokens at once. Split the sequence into
chunks of $C$ tokens — the engine uses 64 — and within a chunk, solve the
recurrence in closed form.

Throughout this section, indices $1 \le t \le C$ are *local* to the chunk,
$S_0$ is the state carried in from the previous chunk, and all vectors belong to
one head. Define the cumulative decay

$$
g_t = \prod_{i=1}^{t} \alpha_i, \qquad g_0 = 1
$$

and give the correction written at step $t$ a name:

$$
u_t = \beta_t \, (v_t - \alpha_t S_{t-1}^\top k_t) \in \mathbb{R}^{d_v}
$$

With that name, the recurrence reads:

$$
S_t = \alpha_t S_{t-1} + k_t u_t^\top
$$

The whole derivation is a matter of finding all $C$ of the $u_t$ without running
the recurrence.

### Step 1: unroll the recurrence

Claim:

$$
S_t = g_t S_0 + \sum_{j \le t} \frac{g_t}{g_j} \, k_j u_j^\top
$$

Prove it by induction. At $t = 1$, the right-hand side is
$g_1 S_0 + (g_1/g_1) k_1 u_1^\top = \alpha_1 S_0 + k_1 u_1^\top$, which is the
recurrence. Assume it holds at $t-1$ and substitute:

$$
S_t = \alpha_t \left( g_{t-1} S_0 + \sum_{j \le t-1} \frac{g_{t-1}}{g_j} k_j u_j^\top \right) + k_t u_t^\top
$$

Since $\alpha_t g_{t-1} = g_t$, the leading term becomes $g_t S_0$ and every
ratio becomes $g_t / g_j$. The final term is $k_t u_t^\top$, which is the
$j = t$ term because $g_t / g_t = 1$. That completes the induction.

Read the result: the state at any point inside the chunk is the carried-in state,
decayed, plus every correction written so far, each decayed by however much has
elapsed since it was written.

### Step 2: substitute back into the definition of u

The $u_t$ are still defined in terms of $S_{t-1}$, so the unrolled form is not
yet usable. Substitute it. Transpose the expression at index $t-1$ and multiply
through by $\alpha_t$:

$$
\alpha_t S_{t-1}^\top = g_t S_0^\top + \sum_{j < t} \frac{g_t}{g_j} \, u_j k_j^\top
$$

Apply that to $k_t$. Each term $u_j k_j^\top k_t$ is the vector $u_j$ scaled by
the scalar $k_j^\top k_t$:

$$
\alpha_t S_{t-1}^\top k_t = g_t S_0^\top k_t + \sum_{j < t} \frac{g_t}{g_j} \, (k_j^\top k_t) \, u_j
$$

Now put that in the definition of $u_t$:

$$
u_t = \beta_t \left( v_t - g_t S_0^\top k_t \right)
    - \sum_{j < t} \beta_t \frac{g_t}{g_j} (k_j^\top k_t) \, u_j
$$

Every $S$ has disappeared. What is left is a linear system: each $u_t$ depends on
the earlier $u_j$ and on quantities you can compute directly from the chunk's
inputs and $S_0$.

### Step 3: write it as a matrix equation

Collect the coefficients into a $C \times C$ matrix:

$$
A_{tj} =
\begin{cases}
\beta_t \, \dfrac{g_t}{g_j} \, (k_j^\top k_t) & j < t \\[4pt]
0 & j \ge t
\end{cases}
$$

Stack the $u_t$ as the rows of a $C \times d_v$ matrix $U$, and the
right-hand sides as the rows of $R$, where
$R_t = \beta_t (v_t - g_t S_0^\top k_t)$. The system from step 2 is

$$
U + A U = R, \qquad \text{that is} \qquad (I + A)\, U = R
$$

$A$ is **strictly** lower triangular: it is zero on the diagonal as well as
above it, because $u_t$ depends only on strictly earlier corrections. Therefore
$I + A$ is **unit** lower triangular — ones on the diagonal, $A$'s entries below
it.

That buys two guarantees, and they are worth stating separately.

**It is always invertible.** The determinant of a triangular matrix is the
product of its diagonal, which here is $1^C = 1$. Not "usually invertible" or
"invertible if well conditioned" — the determinant is exactly 1 for every input
the layer could ever see. There is no configuration of keys, gates, or step sizes
that makes this solve fail.

**The inverse is a finite series.** $A$ is nilpotent with $A^C = 0$, so

$$
(I + A)^{-1} = I - A + A^2 - \cdots + (-A)^{C-1}
$$

That is not how you should compute it, but it explains what the solve means: a
correction at step $t$ propagates its influence forward at most $C - 1$ times
before the chunk ends.

### Step 4: solve it

A unit lower triangular system is solved by forward substitution, which is
`torch.linalg.solve_triangular`. Row by row:

$$
u_t = R_t - \sum_{j < t} A_{tj} \, u_j
$$

Row $t$ needs rows $1$ through $t-1$, so the solve is sequential in $t$. The
depth of that sequential chain is $C$, not the sequence length, and each step is
a matrix operation over all $d_v$ columns at once. That is the trade the chunking
makes: a chain of length $L$ becomes $L/C$ chunks each containing a chain of
length $C$, with everything else in dense matmuls.

### Step 5: the outputs

With $U$ in hand, the outputs follow from step 1 and $o_t = S_t^\top q_t$:

$$
o_t = g_t \, S_0^\top q_t + \sum_{j \le t} \frac{g_t}{g_j} \, (q_t^\top k_j) \, u_j
$$

The sum runs to $j \le t$ inclusive, which is the post-write convention from the
sequential form. The first term is the carried-in state read by this query,
decayed. The second is a masked $C \times C$ matrix of scores multiplied into
$U$ — one small matmul for the whole chunk.

### Step 6: the chunk-final state

Evaluate step 1 at $t = C$:

$$
S_C = g_C S_0 + \sum_{j=1}^{C} \frac{g_C}{g_j} \, k_j u_j^\top
$$

This is the only value that crosses a chunk boundary. Everything else in the
chunk is independent of the next chunk, so a sequence of $L$ tokens runs $L/C$
iterations of this loop instead of $L$. At $L = 2048$ and $C = 64$, that is 32
iterations instead of 2048.

### The code

Here is `delta_rule_chunked` from `engine/layers/linear_attn.py`, with shapes,
matched against the derivation.

```python
for start in range(0, seq, chunk_size):
    stop = min(start + chunk_size, seq)
    size = stop - start                 # C, possibly short on the last chunk

    qc = q32[:, :, start:stop]          # (b, h, C, k_dim)
    kc = k32[:, :, start:stop]          # (b, h, C, k_dim)
    vc = v32[:, :, start:stop]          # (b, h, C, v_dim)
    bc = beta32[:, :, start:stop]       # (b, h, C)

    # g_t within the chunk, in log space. Step 1's cumulative decay.
    log_g = log_alpha[:, :, start:stop].cumsum(dim=-1)   # (b, h, C)
    g = log_g.exp()                                      # (b, h, C)

    # ratio[t, j] = g_t / g_j, kept only where j <= t.
    log_ratio = log_g[..., :, None] - log_g[..., None, :]  # (b, h, C, C)
    causal = torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))
    ratio = torch.where(causal, log_ratio.exp(), torch.zeros((), device=device))

    # A[t, j] = beta_t * ratio[t, j] * (k_t . k_j). Step 3.
    kk = torch.einsum("bhtd,bhjd->bhtj", kc, kc)           # (b, h, C, C)
    a_mat = bc[..., :, None] * ratio * kk                  # (b, h, C, C)
    a_mat = a_mat * torch.tril(                            # strictly lower
        torch.ones(size, size, device=device, dtype=a_mat.dtype), diagonal=-1
    )

    # R_t = beta_t * (v_t - g_t * S_0^T k_t). Step 3.
    read0 = torch.einsum("bhkv,bhtk->bhtv", state, kc)     # (b, h, C, v_dim)
    rhs = bc[..., None] * (vc - g[..., None] * read0)      # (b, h, C, v_dim)

    # (I + A) U = R. Step 4.
    u = torch.linalg.solve_triangular(
        eye[:size, :size] + a_mat, rhs, upper=False, unitriangular=False
    )                                                      # (b, h, C, v_dim)

    # o_t = g_t S_0^T q_t + sum_{j<=t} ratio[t,j] (q_t . k_j) u_j. Step 5.
    inter = g[..., None] * torch.einsum("bhkv,bhtk->bhtv", state, qc)
    qk = torch.einsum("bhtd,bhjd->bhtj", qc, kc) * ratio   # (b, h, C, C)
    outputs[:, :, start:stop] = inter + torch.einsum("bhtj,bhjv->bhtv", qk, u)

    # S_C = g_C S_0 + sum_j (g_C / g_j) k_j u_j^T. Step 6.
    tail = (log_g[..., -1:] - log_g).exp()                 # (b, h, C)
    state = g[..., -1, None, None] * state + torch.einsum(
        "bhjk,bhjv->bhkv", kc * tail[..., None], u
    )                                                      # (b, h, k_dim, v_dim)
```

Two masks appear, and they are not the same mask. `causal` is `tril` with the
diagonal included, because $g_t/g_t = 1$ is a legitimate entry used by the output
formula. The second `tril` has `diagonal=-1`, because $A$ must be strictly lower
triangular for $I + A$ to be unit triangular. Using the inclusive mask for $A$
puts $\beta_t \lVert k_t \rVert^2 = \beta_t$ on the diagonal and quietly solves a
different system.

### What it costs

Count FLOPs per chunk per head, with $d = d_k = d_v = 128$. Only the terms that
scale with $C^2 d$ or $C d^2$ matter; everything else is $O(C^2)$ or $O(Cd)$.

| Operation | Shape | FLOPs |
|---|---|---|
| `kk` | $(C \times d)(d \times C)$ | $2C^2 d$ |
| `read0` | $(C \times d_k)(d_k \times d_v)$ | $2C d^2$ |
| triangular solve | $C^2/2$ MACs per column, $d_v$ columns | $C^2 d$ |
| `inter` | $(C \times d_k)(d_k \times d_v)$ | $2C d^2$ |
| `qk` | $(C \times d)(d \times C)$ | $2C^2 d$ |
| `qk @ u` | $(C \times C)(C \times d_v)$ | $2C^2 d$ |
| final state | $(d_k \times C)(C \times d_v)$ | $2C d^2$ |

Total, per chunk:

$$
F_{\text{chunk}} \approx 7C^2 d + 6C d^2
\qquad \Longrightarrow \qquad
\frac{F_{\text{chunk}}}{C} \approx 7Cd + 6d^2 \ \text{per token}
$$

Now the sequential form. Per token: the read $S^\top k$ is $2d^2$, the update
$\alpha S + k \delta^\top$ is $3d^2$ (scale, outer product, add), and the output
$S^\top q$ is $2d^2$. That is $7d^2$ per token.

The two are equal when $7Cd + 6d^2 = 7d^2$, that is

$$
C = \frac{d}{7} \approx 18
$$

**Above a chunk size of about 18, the chunked form does strictly more
arithmetic than the sequential loop.** At the engine's $C = 64$ and $d = 128$:

$$
7 \times 64 \times 128 + 6 \times 128^2 = 57{,}344 + 98{,}304 = 155{,}648
$$

against $7 \times 128^2 = 114{,}688$ for the sequential form — about 1.36 times
the FLOPs.

So the chunked form is not a FLOP saving. It is a *shape* change. The sequential
form's $7d^2$ FLOPs per token arrive as four dependent kernel launches operating
on $128 \times 128$ tensors, which leaves an A100's 108 SMs almost entirely idle
and is bound by launch latency, not arithmetic. The chunked form's 1.36 times
more FLOPs arrive as dense matmuls with an inner dimension of 64 or 128, which
the tensor cores run near peak, and the dependent chain shrinks from $L$ steps to
$L/C$. Lab 06 requires the chunked form to beat the sequential loop by more than
5 times on a 2048-token sequence; the arithmetic above says it does that while
doing more work, which is the whole point.

Chunk size trades the two terms against each other. The $7Cd$ term grows with
$C$, so large chunks waste arithmetic; the $L/C$ sequential chain and the
depth-$C$ triangular solve push the other way. 64 is the usual compromise.

## Decay ratios and floating point

Every ratio $g_t / g_j$ with $j \le t$ satisfies

$$
\frac{g_t}{g_j} = \prod_{i=j+1}^{t} \alpha_i \le 1
$$

because each $\alpha_i$ lies in $(0, 1]$. The ratio is a product of gates over
the interval between the two positions, and mathematically it can never exceed 1.
Keeping that true in floating point is where the log-space computation earns its
place.

**The direct route underflows.** Compute $g_t$ as a running product, then divide.
Take a head with $\alpha = 0.5$, so $g_t = 2^{-t}$ exactly. float32 holds normal
values down to $2^{-126}$ and subnormals down to $2^{-149}$. So:

- Past $t = 126$, $g_t$ is subnormal and starts losing significand bits.
- At $t = 150$, $g_t$ is exactly zero.

A slower head is not much better. With $\alpha = 0.9$, $g_t = 0.9^t$, and
$0.9^t < 1.40 \times 10^{-45}$ once
$t > \ln(1.40 \times 10^{-45}) / \ln(0.9) = 103.3 / 0.1054 \approx 980$ tokens.
Within a few hundred tokens for a fast head, within a thousand for a slow one,
$g_t$ is zero.

**And zero is worse than inaccurate.** Once $g_t$ and $g_j$ are both zero, the
ratio is $0/0$, which is NaN, not a small number. One NaN in the coefficient
matrix propagates through the triangular solve to every subsequent row, then to
the state, then to every later chunk. The symptom is a model that produces normal
text for the first part of a long prompt and NaN logits after it.

**The log-space route does not.** The engine keeps logarithms:

```python
log_alpha = torch.log(alpha.float().clamp_min(1e-12))    # (b, h, seq)
log_g = log_alpha[:, :, start:stop].cumsum(dim=-1)       # (b, h, C)
log_ratio = log_g[..., :, None] - log_g[..., None, :]    # (b, h, C, C)
ratio = torch.where(causal, log_ratio.exp(), zeros)
```

$\log g_t$ is a sum of negative numbers with no lower limit that float32 cares
about: $-1000$ is an ordinary float32, and so is $-10^{30}$. The subtraction
$\log g_t - \log g_j$ produces the log of the ratio directly, and only that
difference is exponentiated. Since the difference is at most 0, the result is at
most 1, and when the true ratio is genuinely below $10^{-45}$ the exponential
returns 0 — which is the right answer, not a NaN.

Lab 06 tests this on purpose. One check runs the whole sequence with
$\alpha = 10^{-8}$, a gate that erases essentially everything each step. In log
space, $\log \alpha = -18.42$, and over a 64-token chunk the cumulative sum
reaches $-1179$ — an unremarkable float32. The true $g_{64}$ is $10^{-512}$,
which no float32 or float64 can hold. Computed as a quotient of products it is
NaN; computed as `exp(log_g_t - log_g_j)` every entry is correct.

Two more safeguards in that code. `clamp_min(1e-12)` stops $\log 0 = -\infty$,
which would make the difference $-\infty - (-\infty)$, which is NaN. And the cumulative
sum restarts at each chunk boundary rather than running over the whole sequence,
which bounds $\lvert \log g_t \rvert$ by $C$ times the largest per-token rate.

## The rest of the layer

The recurrence is the interesting part but not the whole layer. Two pieces sit
around it.

### The causal depthwise convolution

Before the recurrence sees them, $q$, $k$, and $v$ pass through a four-tap causal
depthwise convolution — one independent filter per channel, looking at the
current position and the three before it.

```python
def causal_depthwise_conv1d(x, weight, cache=None):
    # x:      (batch, seq, channels)
    # weight: (channels, kernel)
    # cache:  (batch, channels, kernel - 1) from the previous call
    xt = x.transpose(1, 2)                          # (b, channels, seq)
    if cache is None:
        cache = torch.zeros(batch, channels, kernel - 1, ...)
    padded = torch.cat([cache, xt], dim=-1)         # (b, channels, seq + 3)
    new_cache = padded[..., -(kernel - 1):]         # (b, channels, 3)
    out = F.conv1d(padded, weight.unsqueeze(1), groups=channels)
    return F.silu(out.transpose(1, 2)), new_cache   # (b, seq, channels)
```

The channel count is the concatenation of $q$, $k$, and $v$ across all heads:

$$
2 \times (16 \times 128) + 48 \times 128 = 4096 + 6144 = 10{,}240
$$

The output gate $z$ is projected alongside them but skips the convolution, so it
is not in that count.

`groups=channels` is what makes it depthwise: each output channel is a function
of its own input channel only, so the weight is `(10240, 4)` rather than a dense
`(10240, 10240, 4)`. That is 40,960 parameters per layer, next to nothing.
Per token the convolution costs $10{,}240 \times 4$ multiply-adds, about 82k
FLOPs, against roughly 5.5M FLOPs for the recurrence across 48 heads. It is under
2% of the layer's arithmetic.

What it buys is a genuine local window. The delta rule's state is an associative
memory addressed by content; it has no notion of "the token immediately before
this one" unless a head burns a key direction on encoding position. A four-tap
convolution supplies that for free.

During decode the `cache` argument holds the previous three steps, shape
`(batch, 10240, 3)`, so the convolution stays $O(1)$ per token like the rest of
the layer. Chapter 2's memory budget allocates a full kernel width of 4 columns
in bfloat16, giving $10{,}240 \times 4 \times 2 = 81{,}920$ bytes per layer, or
3.75 MiB across the 48 linear layers — the difference between 144 MiB of pure
recurrent state and the 147.8 MiB the course quotes.

### The gated RMSNorm output

The recurrence's output is an unconstrained linear combination, so its magnitude
is not bounded the way softmax attention's is. The layer normalizes each value
head, then gates it:

```python
out = out.transpose(1, 2).reshape(batch, seq, 48, 128)
variance = out.float().pow(2).mean(dim=-1, keepdim=True)   # (b, s, 48, 1)
out = out.float() * torch.rsqrt(variance + self.eps) * self.norm_weight
z = z.view(batch, seq, 48, 128)                            # (b, s, 48, 128)
out = (out * F.silu(z.float())).to(x.dtype)
return self.out_proj(out.reshape(batch, seq, 6144))        # (b, s, 5120)
```

The normalization is over the last axis only, 128 channels within one head, so
each of the 48 heads is scaled independently. `norm_weight` has shape `(128,)`
and is shared across heads. The gate `z` comes from the same input projection as
$q$, $k$, and $v$, is full width at `(batch, seq, 6144)`, and passes through
SiLU rather than a sigmoid, so it is not bounded to $(0,1)$ — it can amplify as
well as suppress. Chapter 4 covers RMSNorm; the only new thing here is that the
normalization is per head rather than over the residual stream.

## Head sharing: 16 key heads, 48 value heads

The layer projects 16 query heads and 16 key heads of width 128, but 48 value
heads of width 128. Each key and query head is then repeated across three value
heads:

```python
self.repeats = num_v_heads // num_k_heads            # 48 // 16 = 3
q = q.repeat_interleave(self.repeats, dim=2)         # (b, s, 16, 128) -> (b, s, 48, 128)
k = k.repeat_interleave(self.repeats, dim=2)
```

This is the same trick grouped-query attention plays in chapter 7, but the saving
lands somewhere different, and the difference matters.

**What it saves.** The $q$ and $k$ projections emit $2 \times 16 \times 128 =
4096$ channels instead of $2 \times 48 \times 128 = 12{,}288$. That is
$5120 \times 8192 = 41.9$M fewer parameters per layer, and across the 48 linear
layers, about 2.01 billion parameters or 4.0 GB in bfloat16. It also narrows the
convolution from 18,432 channels to 10,240.

**What it does not save.** The recurrent state is still 48 separate
$128 \times 128$ matrices per layer. Three value heads sharing a key head still
have their own values, their own $\alpha$, and their own $\beta$, so their states
diverge from the first token. Contrast with GQA, where sharing a KV head directly
shrinks the cache because the cache *is* the shared keys and values. Here the
sharing is a parameter saving, not a state saving, and quoting it as a memory
reduction is wrong.

What the three heads do share is a subspace: they all read and write along the
same 128-dimensional key directions, and differ only in what they associate with
those directions and how fast they forget it. Given the timescale spread in the
gate table above, that is a sensible grouping — the same address book, three
different retention policies.

## Verify the two forms against each other

Take this seriously. The chunked form is intricate enough that a sign error, an
off-by-one in a mask, or an inclusive-versus-exclusive triangle produces output
that looks entirely reasonable and is wrong. There is no way to eyeball it. Run
both forms on the same random inputs and compare.

The reference implementation agrees to float32 precision across chunk sizes, and
also agrees when you split the sequence and resume from a carried state — which
is what prefill followed by decode does:

```text
chunk=  1  out_maxerr=3.58e-07  state_maxerr=0.00e+00
chunk= 32  out_maxerr=4.77e-07  state_maxerr=3.58e-07
chunk=256  out_maxerr=1.79e-06  state_maxerr=1.31e-06
split-resume maxerr: 4.17e-07
prefill+decode maxerr: 4.77e-07
```

Read the trend. At `chunk=1` the chunked path degenerates to the sequential one
and the states match bit for bit. As the chunk grows, error grows slowly, because
a longer triangular solve accumulates more rounding. All of it stays within a few
multiples of float32's $10^{-7}$ epsilon.

If your errors are $10^{-3}$ rather than $10^{-6}$, you have a bug, not a
precision issue. Nothing about float32 arithmetic on these shapes produces
$10^{-3}$.

## What goes wrong

**Errors around 1e-3, uniformly.** Almost always a triangle boundary. Check that
$A$ excludes its diagonal (`diagonal=-1`) and that the output sum includes
$j = t$. These are the two places the inclusive and exclusive conventions differ.

**The first output is wrong, everything after is fine.** The output is reading
$S_{t-1}$ instead of $S_t$. With an empty initial state, every later token has
enough history to mask the error.

**NaN after a few hundred tokens of a long prompt.** Decay ratios computed as a
quotient of cumulative products. Move to log space.

**NaN immediately.** $\log 0$ from a gate that reached exactly zero in float32,
giving $-\infty - (-\infty)$. `clamp_min` on the gate before the logarithm.

**The state grows without bound.** Unnormalized keys, so
$1 - \beta \lVert k \rVert^2$ went below $-1$. Normalize.

**Correct outputs, wrong final state.** The chunk-final state uses the ratio
$g_C / g_j$, which is `(log_g[..., -1:] - log_g).exp()`, a different quantity
from the `ratio` matrix used for the outputs. Reusing the last row of `ratio`
works only when the chunk is full, so the bug appears on the last, short chunk of
a sequence whose length is not a multiple of 64.

**Silent shape broadcast.** The gate needs shape `(b, h, 1, 1)` against the state
and `(b, h, 1)` against a value vector. Getting it wrong broadcasts to something
with the correct shape and the wrong contents.

## Check your understanding

**A head has $A_{\log} = -5$, and its raw gate input is such that softplus
returns 1. How many tokens until it has forgotten 90% of what it knew?**

The rate is $r = e^{-5} = 0.006738$, and
$n_{90} = \ln(10)/r = 2.303 / 0.006738 \approx 342$ tokens. Its half-life is
$\ln(2)/r \approx 103$ tokens.

**Why is $I + A$ guaranteed invertible, with no condition on the inputs?**

$A$ is strictly lower triangular, because $u_t$ depends only on strictly earlier
corrections. So $I + A$ is unit lower triangular, and the determinant of a
triangular matrix is the product of its diagonal, which is $1^C = 1$. No choice
of keys, gates, or step sizes changes that.

**The chunked form does about 36% more arithmetic than the sequential loop at
$C = 64$. Why is it more than 5 times faster?**

Because the sequential loop is not limited by arithmetic. It issues four
dependent kernel launches per token on $128 \times 128$ tensors, leaving the
A100's SMs idle and paying launch latency 2048 times. The chunked form turns the
same work into dense matmuls the tensor cores can saturate, and shortens the
dependent chain from 2048 steps to 32.

**A head writes 200 distinct unit-norm keys into its 128 by 128 state, then reads
back the first one. What comes out?**

The correct value plus an interference term summing the other 199 values weighted
by their inner products with the query key. For random keys the expected squared
inner product is $1/128$, so the interference has roughly $199/128 \approx 1.6$
times the energy of the signal. The forget gate is what keeps this from
happening: by the time the 200th key is written, the first has been decayed
toward zero.

## Lab

Implement `delta_rule_step`, `delta_rule_recurrent`, and `delta_rule_chunked`,
and prove they agree. The harness checks:

- The sequential form returns the right shapes.
- With the gates fully open on a fresh state, the first output is exactly
  $(q_1^\top k_1) v_1$ — which catches an update applied in the wrong order.
- The chunked form matches the sequential one at chunk sizes 1, 7, 32, 64, and
  256, in both the outputs and the final state, to $10^{-4}$.
- A sequence split in two, with the second half resuming from the first half's
  carried state, matches one long run.
- A chunked prefill of 100 tokens followed by single `delta_rule_step` calls
  matches one long sequential run. This is the real decode path.
- A nearly closed forget gate, $\alpha = 10^{-8}$, agrees between both forms.
  This is the log-space check.

On a GPU it also times both forms on a 2048-token sequence and requires the
chunked form to be more than 5 times faster. That speedup is the point of the
whole exercise.

## Further reading

- [Gated delta networks: improving Mamba2 with delta rule](https://arxiv.org/abs/2412.06464)
- [Parallelizing linear transformers with the delta rule over sequence length](https://arxiv.org/abs/2406.06484)
- [Transformers are RNNs: fast autoregressive transformers with linear attention](https://arxiv.org/abs/2006.16236)
- [Linear transformers are secretly fast weight programmers](https://arxiv.org/abs/2102.11174) — where the delta rule enters this line of work.
- [Gated linear attention transformers with hardware-efficient training](https://arxiv.org/abs/2312.06635) — the chunked form, in more generality.
- [Mamba: linear-time sequence modeling with selective state spaces](https://arxiv.org/abs/2312.00752) — the input-dependent gate, from the state-space side.
