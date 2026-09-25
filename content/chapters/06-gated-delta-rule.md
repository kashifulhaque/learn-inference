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

> [!TLDR]
> - Linear attention replaces the KV cache with a fixed $128 \times 128$ state per head, so reading costs the same at token 10 and at token 100,000.
> - A fixed state holds about $d_k = 128$ clean associations. Plain accumulation can't overwrite a key; the delta rule can, because it writes only the difference from what the state already returns.
> - A per-head forget gate $\hla{\alpha_t}$ decays the state. Its parameter $A_{\log}$ reads directly as a memory horizon in tokens.
> - The chunked form solves 64 tokens at once with a unit lower-triangular solve. It does about 1.36 times the FLOPs of the sequential loop, and it's still more than 5 times faster.
> - Decay ratios must be computed in log space, or long prompts turn into NaN.

Three quarters of this model's layers don't use attention. They run a recurrence
with a fixed-size state, which is why a sequence at 128k context carries the same
147.8 MiB of layer state as a sequence at 10 tokens.

This is the hardest chapter in the course. It's also the one where the payoff is
largest: the recurrence makes the model's memory profile flat instead of linear,
and the chunked form stops that recurrence from making prefill hopeless.

Take it in two halves:

1. **The update rule, one term at a time.** Linear attention, then the delta
   rule, then the gate.
2. **A form a GPU can run in parallel.** This half is longer and has more
   algebra, but nothing beyond an induction and a triangular solve.

## Before you start

The chapter assumes the following, and nothing else.

**Outer products.** For column vectors $k \in \mathbb{R}^{d_k}$ and $v \in \mathbb{R}^{d_v}$, the outer product $k v^\top$ is the $d_k \times d_v$ matrix with entry $(a, b)$ equal to $k_a v_b$. It has rank 1, and every state update in this chapter adds one to a matrix.

**Reading an outer product back.** If $S = k v^\top$, then $S^\top k = v \, (k^\top k) = \lVert k \rVert^2 v$. With a unit-norm key, $S^\top k = v$ exactly. That identity is the whole idea of an associative memory: write with an outer product, read with a matrix-vector product.

**Matrix calculus for one scalar function.** You need the gradient of $\tfrac{1}{2}\lVert S^\top k - v \rVert^2$ with respect to the matrix $S$. The chapter differentiates entry by entry, so partial derivatives are enough.

**Symmetric projectors.** For a unit-norm $k$, the matrix $P = k k^\top$ satisfies $P^2 = P$. Its eigenvalues are 1, with eigenvector $k$, and 0, with multiplicity $d_k - 1$ on the subspace orthogonal to $k$.

**Tensor shapes and einsum.** [Chapter 0a](/c/00a-notation-and-prerequisites) covers the `(batch, heads, seq, dim)` convention and `torch.einsum`. Every listing here carries its shapes in comments.

**Floating point.** Chapter 0a also covers float32's exponent range. You need two numbers from it: the smallest normal float32 is about $1.18 \times 10^{-38}$, and the smallest subnormal is about $1.40 \times 10^{-45}$.

Nothing in this chapter needs a GPU to understand. The lab runs on one to measure
the speedup, but every check in it is device agnostic.

## From softmax attention to a recurrence

Linear attention keeps the shape of softmax attention and drops the part that
forces a KV cache. Start with what you're replacing. For query position $t$ in a
causal model, softmax attention computes:

$$
o_t = \frac{\sum_{j \le t} \exp(q_t^\top k_j / \sqrt{d_k}) \, v_j}
           {\sum_{j \le t} \exp(q_t^\top k_j / \sqrt{d_k})}
$$

Here $q_t, k_j \in \mathbb{R}^{d_k}$ are the query and key vectors, and
$v_j \in \mathbb{R}^{d_v}$ is the value. [Chapter 7](/c/07-grouped-query-attention)
derives this formula properly. For now, the only feature that matters is where
$q_t$ appears.

It sits inside the $\exp$, coupled to every $k_j$. You can't pull it out of the
sum, so you can't precompute anything that summarizes the past independently of
the query. That's why softmax attention ==must keep every past key and value==.

### The kernel trick

The standard way to break that coupling is to assume the exponential is a kernel
with a known feature map. Suppose a function $\phi : \mathbb{R}^{d_k} \to \mathbb{R}^{m}$
exists with:

$$
\exp(q^\top k / \sqrt{d_k}) \approx \phi(q)^\top \phi(k)
$$

Substitute it into the numerator, and the associativity of matrix products does
the rest:

$$
\sum_{j \le t} \phi(q_t)^\top \phi(k_j) \, v_j
= \phi(q_t)^\top \left( \sum_{j \le t} \phi(k_j) v_j^\top \right)
$$

The bracketed term is an $m \times d_v$ matrix that doesn't mention $q_t$. Call it
$S_t$, and you have a state that summarizes the entire past at a size that doesn't
depend on $t$:

$$
S_t = S_{t-1} + \phi(k_t) v_t^\top, \qquad o_t = S_t^\top \phi(q_t)
$$

The denominator factors the same way, against a vector $z_t = z_{t-1} + \phi(k_t)$, giving $o_t = S_t^\top \phi(q_t) / (z_t^\top \phi(q_t))$.

### What this model actually does

This model takes $\phi$ to be the identity map, so $m = d_k$, and drops the
denominator entirely:

$$
\boxed{S_t = S_{t-1} + k_t v_t^\top, \qquad o_t = S_t^\top q_t}
$$

That's linear attention. It gives up two things and gains one:

- **The kernel identity is gone.** With $\phi$ the identity, $\phi(q)^\top \phi(k)$
  is $q^\top k$, which isn't an approximation of $\exp(q^\top k / \sqrt{d_k})$ in
  any useful sense. This is ==a different operator, not cheap softmax==. It has
  the same input and output signature, and it has to be trained as itself. The
  kernel derivation explains the shape of the recurrence; it doesn't license the
  substitution.
- **The normalizer is gone.** Softmax attention returns a convex combination of
  values: the weights are non-negative and sum to 1. Linear attention's output is
  an unconstrained linear combination, so weights can be negative and sum to
  anything. In exchange, the layer can output near zero when nothing relevant is
  in the state. The gated RMSNorm at the end of the layer keeps the unbounded
  magnitude in check.
- **Reading is now free.** $o_t = S_t^\top q_t$ costs $d_k d_v$ multiply-adds no
  matter how long the context is, so generating a token is $O(1)$ in context
  length instead of $O(L)$. For this model, $d_k = d_v = 128$, so one head's read
  is $2 \times 128 \times 128 = 32{,}768$ FLOPs at token 10 and at token 100,000.

What you pay for that is capacity, and the price is worth quantifying.

## How much can a fixed state hold

A fixed state holds about $d_k$ clean associations, and past that, reads return
a blend of other values. The state $S$ is a $d_k \times d_v$ matrix: in this
model, $128 \times 128 = 16{,}384$ numbers per head, held in float32. Across 48
value heads and 48 linear layers, that's 147.8 MiB per sequence, which
[chapter 2](/c/02-memory-arithmetic) derives.

Write $n$ associations with unit-norm keys:

$$
S = \sum_{i=1}^{n} k_i v_i^\top
$$

Then read it back with key $k_m$. The sum splits into the value you asked for
and everything else:

$$
S^\top k_m = \sum_{i=1}^{n} v_i \, (k_i^\top k_m)
= v_m + \underbrace{\hld{\sum_{i \ne m} (k_i^\top k_m) \, v_i}}_{\text{interference}}
$$

The $\hld{\text{interference}}$ is every other value in the state, weighted by how
much its key overlaps with the one you asked for.

**When the interference vanishes.** If the keys are mutually orthogonal, every
$k_i^\top k_m$ with $i \ne m$ is zero and the read is exact. At most $d_k$
mutually orthogonal directions fit in $\mathbb{R}^{d_k}$, so the hard ceiling is
$d_k = 128$ perfect associations per head. Write a 129th and the keys are
necessarily linearly dependent, so some interference is forced.

**How it grows before the ceiling.** Real keys aren't orthogonal. For two
independent random unit vectors in $\mathbb{R}^{d}$, the expected squared inner
product is $1/d$. So the $\hld{\text{interference}}$ has expected squared norm
about $(n-1)/d_k$ times the typical $\lVert v \rVert^2$, against a signal of
$\lVert v_m \rVert^2$. The signal-to-noise ratio is roughly:

$$
\mathrm{SNR} \approx \frac{d_k}{n - 1}
$$

With $d_k = 128$, the noise energy matches the signal energy at about $n = 129$
writes. Past a few hundred tokens, a plain linear-attention head that reads a key
it wrote long ago gets back mostly other keys' values. Plain linear attention
degrades over long sequences, and it ==degrades smoothly rather than loudly==,
which is worse.

**The failure that motivates the delta rule.** The clearest case is writing the
same key twice. Suppose token 3 writes $(k, v_a)$ and token 90 writes $(k, v_b)$,
meaning "this slot now holds $v_b$ instead". Plain accumulation gives:

$$
S = k v_a^\top + k v_b^\top = k (v_a + v_b)^\top,
\qquad S^\top k = v_a + v_b
$$

> [!KEY] Plain accumulation can only add
> The read returns the sum, not the update. The state has no way to express
> replacement, so every overwrite makes the entry worse.

## The delta rule

The delta rule makes the write *conditional on what's already there*: before
writing, read the state at $k_t$, and write back only the part that's missing.

### Deriving it from an objective

State the goal as an optimization. At step $t$, you want a state that returns
$v_t$ when read with $k_t$, so define the loss:

$$
\mathcal{L}(S) = \tfrac{1}{2} \lVert S^\top k_t - v_t \rVert_2^2
$$

Its gradient with respect to $S$ is an outer product of the key and the residual:

$$
\nabla_S \mathcal{L} = k_t \, (S^\top k_t - v_t)^\top
$$

> [!DEEPDIVE] Differentiating the loss entry by entry
> Write $r = S^\top k_t$, so $r_b = \sum_a S_{ab} (k_t)_a$. Only $r_b$ depends on
> $S_{ab}$, so the chain rule gives:
>
> $$
> \frac{\partial \mathcal{L}}{\partial S_{ab}}
> = (r_b - (v_t)_b) \cdot \frac{\partial r_b}{\partial S_{ab}}
> = (r_b - (v_t)_b) \, (k_t)_a
> $$
>
> Entry $(a, b)$ of the gradient is the $a$-th entry of $k_t$ times the $b$-th
> entry of the residual. That's exactly the outer product
> $k_t \, (S^\top k_t - v_t)^\top$.

Take one step of gradient descent with step size $\hlb{\beta_t}$:

$$
\boxed{
S_t = S_{t-1} - \hlb{\beta_t} \, \nabla_S \mathcal{L}
    = S_{t-1} + \hlb{\beta_t} \, k_t \, (v_t - \hlc{S_{t-1}^\top k_t})^\top
}
$$

That's the delta rule, the classical one from online learning.

> [!INTUITION]
> Read $\hlc{S_{t-1}^\top k_t}$ as "what the state currently returns for this
> key". If it already equals $v_t$, the correction is zero and nothing is
> written. If it differs, the state moves toward $v_t$ in proportion to how far
> off it is, scaled by $\hlb{\beta_t}$.

In the engine, a sigmoid produces $\hlb{\beta_t}$ per value head, so it lies in
$(0, 1)$:

```python
b_raw, a_raw = ba.chunk(2, dim=-1)     # each (batch, seq, 48)
beta = torch.sigmoid(b_raw)            # (batch, seq, 48), in (0, 1)
```

### Why it stops the state from saturating

Expand the update and collect the terms in $S_{t-1}$:

$$
S_t = S_{t-1} - \hlb{\beta_t} k_t k_t^\top S_{t-1} + \hlb{\beta_t} k_t v_t^\top
    = (I - \hlb{\beta_t} k_t k_t^\top) \, S_{t-1} + \hlb{\beta_t} k_t v_t^\top
$$

The write is now two moves: ==erase along the key, then add==.

Go back to the double-write example with $\hlb{\beta} = 1$. The first write puts
$k v_a^\top$ in the state. The second first applies $I - k k^\top$, which
annihilates the $k$ direction and leaves $0$, then adds $k v_b^\top$. Reading
gives $v_b$: the state replaced rather than accumulated.

### Why the keys are L2-normalized

The erase factor $I - \hlb{\beta} k k^\top$ is well behaved only when
$\lVert k \rVert = 1$.

With a unit-norm key, $P = k k^\top$ is an orthogonal projector onto the line
through $k$, so its eigenvalues are 1 and 0. The eigenvalues of
$I - \hlb{\beta} P$ are therefore:

$$
1 - \hlb{\beta} \quad \text{along } k,
\qquad 1 \quad \text{on the } d_k - 1 \text{ directions orthogonal to } k
$$

Read off what the step size does:

- At $\hlb{\beta} = 0$, the matrix is $I$ and nothing is erased.
- At $\hlb{\beta} = 1$, the $k$ direction is erased completely: a full
  replacement.
- In between, it's a partial erase, and every direction orthogonal to $k$ is
  untouched. The write disturbs nothing it wasn't aimed at.

Since $\hlb{\beta} \in (0,1)$, the spectral radius is exactly 1 and repeated
writes never amplify the state.

Now drop the normalization. The eigenvalue along $k$ becomes
$1 - \hlb{\beta} \lVert k \rVert^2$. Stability needs
$\lvert 1 - \hlb{\beta} \lVert k \rVert^2 \rvert \le 1$, that is,
$0 \le \hlb{\beta} \lVert k \rVert^2 \le 2$. With $\hlb{\beta}$ capped at 1 by the
sigmoid, you need $\lVert k \rVert^2 \le 2$, and a raw linear projection has no
reason to respect that.

> [!WARNING] An unnormalized key blows up the state
> A key of norm 2 at $\hlb{\beta} = 0.8$ gives an eigenvalue of
> $1 - 0.8 \times 4 = -2.2$. Every write along that direction flips the sign of
> whatever was there and multiplies it by 2.2. Over a few hundred tokens, the
> state overflows to infinity, and then to NaN.

So the engine normalizes:

```python
q = F.normalize(q, dim=-1)   # (batch, seq, 16, 128), each row unit norm
k = F.normalize(k, dim=-1)   # (batch, seq, 16, 128)
```

Queries are normalized too. Stability doesn't require it, because the query never
touches the state, but it keeps the output magnitude from tracking the raw
projection scale, and it makes $q_t^\top k_j$ a cosine similarity in $[-1, 1]$.

## The forget gate

The gate lets the layer forget things it's never asked about again, which the
delta rule alone can't do. Add a scalar forget gate $\hla{\alpha_t} \in (0,1)$
that shrinks the whole state before the write:

$$
\boxed{
S_t = \hla{\alpha_t} S_{t-1}
    + \hlb{\beta_t} \, k_t \, (v_t - \hla{\alpha_t} \hlc{S_{t-1}^\top k_t})^\top,
\qquad o_t = S_t^\top q_t
}
$$

Note where $\hla{\alpha_t}$ appears inside the correction. The state is decayed
first, and the delta rule then corrects ==the decayed state, not the old one==.
In the collected form:

$$
S_t = (I - \hlb{\beta_t} k_t k_t^\top)\,\hla{\alpha_t} S_{t-1} + \hlb{\beta_t} k_t v_t^\top
$$

Every head gets its own gate, one scalar per token per value head, so a layer's
48 heads decay at 48 different rates.

### The parameterization, and why it can't escape (0, 1)

A gate that leaves $(0,1)$ breaks the layer: above 1, the state grows without
bound; below 0, it oscillates in sign. The engine guarantees the range by
construction rather than by clamping:

```python
decay_rate = F.softplus(a_raw.float() + self.dt_bias) * self.A_log.exp()
alpha = torch.exp(-decay_rate)
```

In symbols, with $a_t$ the raw projection output for this head and token:

$$
r_t = \operatorname{softplus}(a_t + b_{\Delta}) \cdot e^{A_{\log}},
\qquad \hla{\alpha_t} = e^{-r_t}
$$

Follow the signs:

- $\operatorname{softplus}(x) = \log(1 + e^x)$ is strictly positive for every
  real $x$, and $e^{A_{\log}}$ is strictly positive for every real $A_{\log}$.
- So the rate $r_t > 0$ always, and $\hla{\alpha_t} = e^{-r_t}$ lands strictly
  inside $(0, 1)$ for any input whatsoever.
- As $a_t \to -\infty$, softplus goes to 0 and $\hla{\alpha_t} \to 1$: perfect
  retention.
- As $a_t \to +\infty$, softplus grows like $a_t$ and $\hla{\alpha_t} \to 0$: a
  full reset.

Both limits are approached but never reached. $A_{\log}$ and the bias $b_{\Delta}$
are learned per value head, with shape `(48,)` each. The token-dependent part is
$a_t$; $A_{\log}$ sets the head's baseline scale.

### Reading a head's time constant

Given $A_{\log}$, how long does this head remember? This is the useful skill.

Hold the rate $r_t = r$ constant. The state is multiplied by $\hla{\alpha}$ each
step, so after $n$ steps whatever was written has been scaled by
$\hla{\alpha}^n = e^{-rn}$. Solve for the horizon at which half survives, and at
which a tenth survives (90% forgotten):

$$
n_{1/2} = \frac{\ln 2}{r} \approx \frac{0.693}{r},
\qquad
n_{90} = \frac{\ln 10}{r} \approx \frac{2.303}{r}
$$

Suppose $\operatorname{softplus}(a_t + b_{\Delta}) = 1$, so that $r = e^{A_{\log}}$
and $A_{\log}$ alone sets the timescale. The arithmetic gives:

| $A_{\log}$ | $r = e^{A_{\log}}$ | $\hla{\alpha} = e^{-r}$ | $n_{1/2}$ | $n_{90}$ |
|---|---|---|---|---|
| $2$ | $7.389$ | $0.00062$ | $0.09$ | $0.31$ |
| $0$ | $1.000$ | $0.368$ | $0.69$ | $2.3$ |
| $-2$ | $0.1353$ | $0.8734$ | $5.1$ | $17$ |
| $-4$ | $0.01832$ | $0.9819$ | $38$ | $126$ |
| $-6$ | $0.002479$ | $0.99752$ | $280$ | $929$ |
| $-8$ | $0.0003355$ | $0.99966$ | $2066$ | $6864$ |

These are arithmetic, not measurements: they follow from the formula and the
assumption that softplus returns 1. The checkpoint's actual $A_{\log}$ values live
in the weights and vary by head.

> [!KEY] Two units of $A_{\log}$ move the horizon by a factor of 7.4
> Two units of $A_{\log}$ move the 90% horizon by $e^2 \approx 7.4$, so a layer's
> 48 heads can cover everything from "the previous two tokens" to "the last
> several thousand". The short heads act as a wide n-gram window, and the long
> heads are the associative memory, each with its own capacity limit.

It also shows why an exponential parameterization is the right one. If $A_{\log}$
were the rate rather than its logarithm, gradient descent could never move a head
from a 2-token horizon to a 2000-token one.

## The sequential form

The sequential form is the recurrence as code, one token at a time. This is
`delta_rule_step` in `engine/layers/linear_attn.py`, which is also the decode
path.

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

Three details are worth pinning down:

- **Everything is float32.** The state is an accumulator that's read and
  rewritten every token. bfloat16 has 8 significand bits, and accumulating
  100,000 updates into it loses the small corrections entirely. The inputs arrive
  in bfloat16 and are promoted on entry.
- **`alpha[..., 0]` versus `alpha`.** The gate is broadcast against a matrix in
  one place and a vector in the other, so it needs two different trailing shapes.
  Get this wrong and PyTorch broadcasts silently to something plausible.
- **The output uses the post-write state.** $o_t = S_t^\top q_t$, not
  $S_{t-1}^\top q_t$, so token $t$ can read its own write. This is the
  single most common sign-and-index error in the chunked form. Lab 06's first
  check catches it: with $\hla{\alpha} = \hlb{\beta} = 1$ on a fresh state, the
  first output must be exactly $(q_1^\top k_1) v_1$.

This code is ==correct and unusable for prefill==. A 2000-token prompt becomes
2000 sequential steps, each launching a handful of tiny kernels that occupy a few
hundred of the A100's threads. The GPU spends its time on launch latency.

## The chunked parallel form

The chunked form computes a whole block of tokens at once by solving the
recurrence in closed form inside each chunk. Split the sequence into chunks of $C$
tokens (the engine uses 64).

Throughout this section, indices $1 \le t \le C$ are *local* to the chunk, $S_0$
is the state carried in from the previous chunk, and all vectors belong to one
head. Define the cumulative decay:

$$
\hla{g_t} = \prod_{i=1}^{t} \hla{\alpha_i}, \qquad \hla{g_0} = 1
$$

Give the correction written at step $t$ a name:

$$
u_t = \hlb{\beta_t} \, (v_t - \hla{\alpha_t} \hlc{S_{t-1}^\top k_t}) \in \mathbb{R}^{d_v}
$$

With that name, the recurrence reads $S_t = \hla{\alpha_t} S_{t-1} + k_t u_t^\top$.
The whole derivation is a matter of finding all $C$ of the $u_t$
==without running the recurrence==. It takes six steps:

1. Unroll the recurrence, so $S_t$ depends only on $S_0$ and the $u_j$.
2. Substitute that back into the definition of $u_t$, which removes every $S$.
3. Write the result as a triangular matrix equation.
4. Solve it by forward substitution.
5. Read the outputs off the solved $u_t$.
6. Compute the chunk-final state to carry forward.

### Step 1: unroll the recurrence

Inside the chunk, the state is the carried-in state, decayed, plus every
correction written so far, each decayed by however much has elapsed since it was
written:

$$
\boxed{S_t = \hla{g_t} S_0 + \sum_{j \le t} \hla{\frac{g_t}{g_j}} \, k_j u_j^\top}
$$

> [!DEEPDIVE] Proof by induction
> At $t = 1$, the right-hand side is
> $g_1 S_0 + (g_1/g_1) k_1 u_1^\top = \alpha_1 S_0 + k_1 u_1^\top$, which is the
> recurrence. Assume it holds at $t-1$ and substitute:
>
> $$
> S_t = \alpha_t \left( g_{t-1} S_0 + \sum_{j \le t-1} \frac{g_{t-1}}{g_j} k_j u_j^\top \right) + k_t u_t^\top
> $$
>
> Since $\alpha_t g_{t-1} = g_t$, the leading term becomes $g_t S_0$ and every
> ratio becomes $g_t / g_j$. The final term is $k_t u_t^\top$, which is the
> $j = t$ term because $g_t / g_t = 1$. That completes the induction.

### Step 2: substitute back into the definition of u

The $u_t$ are still defined in terms of $S_{t-1}$, so the unrolled form isn't yet
usable. Substituting step 1 at index $t-1$ into the definition of $u_t$ removes
every $S$ except the carried-in $S_0$:

$$
\boxed{
u_t = \hlb{\beta_t} \left( v_t - \hla{g_t} \hlc{S_0^\top k_t} \right)
    - \sum_{j < t} \hlb{\beta_t} \hla{\frac{g_t}{g_j}} (k_j^\top k_t) \, u_j
}
$$

> [!DEEPDIVE] The substitution, step by step
> Transpose the step 1 expression at index $t-1$ and multiply through by
> $\alpha_t$:
>
> $$
> \alpha_t S_{t-1}^\top = g_t S_0^\top + \sum_{j < t} \frac{g_t}{g_j} \, u_j k_j^\top
> $$
>
> Apply that to $k_t$. Each term $u_j k_j^\top k_t$ is the vector $u_j$ scaled by
> the scalar $k_j^\top k_t$:
>
> $$
> \alpha_t S_{t-1}^\top k_t = g_t S_0^\top k_t + \sum_{j < t} \frac{g_t}{g_j} \, (k_j^\top k_t) \, u_j
> $$
>
> Put that in $u_t = \beta_t (v_t - \alpha_t S_{t-1}^\top k_t)$ and distribute
> $\beta_t$ to get the boxed result.

> [!INTUITION]
> What's left is a linear system. Each $u_t$ depends on the earlier $u_j$, weighted
> by how much their keys overlap with $k_t$ and how much decay separates them, and
> on quantities you can compute directly from the chunk's inputs and
> $\hlc{S_0^\top k_t}$.

### Step 3: write it as a matrix equation

Collect the coefficients into a $C \times C$ matrix:

$$
A_{tj} =
\begin{cases}
\hlb{\beta_t} \, \hla{\dfrac{g_t}{g_j}} \, (k_j^\top k_t) & j < t \\[4pt]
0 & j \ge t
\end{cases}
$$

Stack the $u_t$ as the rows of a $C \times d_v$ matrix $U$, and the right-hand
sides $R_t = \hlb{\beta_t} (v_t - \hla{g_t} \hlc{S_0^\top k_t})$ as the rows of
$R$. The system from step 2 becomes:

$$
U + A U = R, \qquad \text{that is} \qquad \boxed{(I + A)\, U = R}
$$

$A$ is **strictly** lower triangular: it's zero on the diagonal as well as above
it, because $u_t$ depends only on strictly earlier corrections. Therefore $I + A$
is **unit** lower triangular, with ones on the diagonal and $A$'s entries below
it. That buys two guarantees.

> [!KEY] The solve can never fail
> The determinant of a triangular matrix is the product of its diagonal, which
> here is $1^C = 1$. Not "usually invertible" or "invertible if well conditioned":
> the determinant is exactly 1 for every input the layer could ever see. No
> configuration of keys, gates, or step sizes makes this solve fail.

**The inverse is a finite series.** $A$ is nilpotent with $A^C = 0$, so:

$$
(I + A)^{-1} = I - A + A^2 - \cdots + (-A)^{C-1}
$$

That's not how you should compute it, but it explains what the solve means: a
correction at step $t$ propagates its influence forward at most $C - 1$ times
before the chunk ends.

### Step 4: solve it

Solve a unit lower-triangular system by forward substitution, which is
`torch.linalg.solve_triangular`. Row by row:

$$
u_t = R_t - \sum_{j < t} A_{tj} \, u_j
$$

Row $t$ needs rows $1$ through $t-1$, so the solve is sequential in $t$. But the
depth of that chain is $C$, not the sequence length, and each step is a matrix
operation over all $d_v$ columns at once.

That's the trade chunking makes: a chain of length $L$ becomes $L/C$ chunks,
each containing a chain of length $C$, with everything else in dense matmuls.

### Step 5: the outputs

With $U$ in hand, the outputs follow from step 1 and $o_t = S_t^\top q_t$:

$$
\boxed{o_t = \hla{g_t} \, S_0^\top q_t + \sum_{j \le t} \hla{\frac{g_t}{g_j}} \, (q_t^\top k_j) \, u_j}
$$

The sum runs to $j \le t$ inclusive, which is the post-write convention from the
sequential form.

- The first term is the carried-in state read by this query, decayed.
- The second is a masked $C \times C$ matrix of scores multiplied into $U$: one
  small matmul for the whole chunk.

### Step 6: the chunk-final state

Evaluate step 1 at $t = C$:

$$
\boxed{S_C = \hla{g_C} S_0 + \sum_{j=1}^{C} \hla{\frac{g_C}{g_j}} \, k_j u_j^\top}
$$

This is ==the only value that crosses a chunk boundary==. Everything else in the
chunk is independent of the next chunk, so a sequence of $L$ tokens runs $L/C$
iterations of this loop instead of $L$. At $L = 2048$ and $C = 64$, that's 32
iterations instead of 2048.

### The code

Here's `delta_rule_chunked` from `engine/layers/linear_attn.py`, with shapes. The
comments name the step each line implements.

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

> [!WARNING] Two masks, and they differ
> `causal` is `tril` with the diagonal included, because $g_t/g_t = 1$ is a
> legitimate entry that the output formula uses. The second `tril` has
> `diagonal=-1`, because $A$ must be strictly lower triangular for $I + A$ to be
> unit triangular. Using the inclusive mask for $A$ puts
> $\hlb{\beta_t} \lVert k_t \rVert^2 = \hlb{\beta_t}$ on the diagonal and quietly
> solves a different system.

## What the chunked form costs

The chunked form does more arithmetic than the sequential loop, not less. Count
FLOPs per chunk per head, with $d = d_k = d_v = 128$. Only the terms that scale
with $C^2 d$ or $C d^2$ matter; everything else is $O(C^2)$ or $O(Cd)$.

| Operation | Shape | FLOPs |
|---|---|---|
| `kk` | $(C \times d)(d \times C)$ | $2C^2 d$ |
| `read0` | $(C \times d_k)(d_k \times d_v)$ | $2C d^2$ |
| triangular solve | $C^2/2$ MACs per column, $d_v$ columns | $C^2 d$ |
| `inter` | $(C \times d_k)(d_k \times d_v)$ | $2C d^2$ |
| `qk` | $(C \times d)(d \times C)$ | $2C^2 d$ |
| `qk @ u` | $(C \times C)(C \times d_v)$ | $2C^2 d$ |
| final state | $(d_k \times C)(C \times d_v)$ | $2C d^2$ |

Add up the rows, then divide by $C$ for the cost per token:

$$
F_{\text{chunk}} \approx 7C^2 d + 6C d^2
\qquad \Longrightarrow \qquad
\frac{F_{\text{chunk}}}{C} \approx 7Cd + 6d^2 \ \text{per token}
$$

The sequential form costs $7d^2$ per token: the read $S^\top k$ is $2d^2$, the
update $\hla{\alpha} S + k \delta^\top$ is $3d^2$ (scale, outer product, add), and
the output $S^\top q$ is $2d^2$.

The two are equal when $7Cd + 6d^2 = 7d^2$, that is:

$$
\boxed{C = \frac{d}{7} \approx 18}
$$

Above a chunk size of about 18, ==the chunked form does strictly more arithmetic==
than the sequential loop.

> [!EXAMPLE] The engine's chunk size
> At $C = 64$ and $d = 128$, the chunked form costs, per token:
>
> $$
> 7 \times 64 \times 128 + 6 \times 128^2 = 57{,}344 + 98{,}304 = 155{,}648
> $$
>
> The sequential form costs $7 \times 128^2 = 114{,}688$, so the chunked form does
> about 1.36 times the FLOPs.

> [!KEY] Chunking is a shape change, not a FLOP saving
> The sequential form's $7d^2$ FLOPs per token arrive as four dependent kernel
> launches on $128 \times 128$ tensors. That leaves an A100's 108 SMs almost
> entirely idle and is bound by launch latency, not arithmetic. The chunked form's
> 1.36 times more FLOPs arrive as dense matmuls with an inner dimension of 64 or
> 128, which the tensor cores run near peak, and the dependent chain shrinks from
> $L$ steps to $L/C$.

Lab 06 requires the chunked form to beat the sequential loop by more than 5 times
on a 2048-token sequence, while doing more work. Chunk size trades the two costs:
the $7Cd$ term grows with $C$, so large chunks waste arithmetic, while the $L/C$
sequential chain and the depth-$C$ triangular solve push the other way. 64 is the
usual compromise.

## Decay ratios and floating point

Compute every decay ratio in log space, because the direct route underflows to
zero and then divides zero by zero. Every ratio with $j \le t$ satisfies:

$$
\hla{\frac{g_t}{g_j}} = \prod_{i=j+1}^{t} \hla{\alpha_i} \le 1
$$

Each $\hla{\alpha_i}$ lies in $(0, 1]$, so the ratio is a product of gates over the
interval between the two positions and mathematically can never exceed 1. Keeping
that true in floating point is where the log-space computation earns its place.

**The direct route underflows.** Compute $\hla{g_t}$ as a running product, then
divide. Take a head with $\hla{\alpha} = 0.5$, so $\hla{g_t} = 2^{-t}$ exactly.
float32 holds normal values down to $2^{-126}$ and subnormals down to $2^{-149}$,
so:

- Past $t = 126$, $\hla{g_t}$ is subnormal and starts losing significand bits.
- At $t = 150$, $\hla{g_t}$ is exactly zero.

A slower head isn't much better. With $\hla{\alpha} = 0.9$, $\hla{g_t} = 0.9^t$,
and $0.9^t < 1.40 \times 10^{-45}$ once $t > \ln(1.40 \times 10^{-45}) / \ln(0.9) = 103.3 / 0.1054 \approx 980$
tokens. Within a few hundred tokens for a fast head, and within a thousand for a
slow one, $\hla{g_t}$ is zero.

> [!WARNING] Zero is worse than inaccurate
> Once $g_t$ and $g_j$ are both zero, the ratio is $0/0$, which is NaN, not a small
> number. One NaN in the coefficient matrix propagates through the triangular
> solve to every later row, then to the state, then to every later chunk. The
> symptom is a model that produces normal text for the first part of a long prompt
> and NaN logits after it.

**The log-space route doesn't underflow.** The engine keeps logarithms:

```python
log_alpha = torch.log(alpha.float().clamp_min(1e-12))    # (b, h, seq)
log_g = log_alpha[:, :, start:stop].cumsum(dim=-1)       # (b, h, C)
log_ratio = log_g[..., :, None] - log_g[..., None, :]    # (b, h, C, C)
ratio = torch.where(causal, log_ratio.exp(), zeros)
```

$\log g_t$ is a sum of negative numbers with no lower limit that float32 cares
about: $-1000$ is an ordinary float32, and so is $-10^{30}$. The subtraction
$\log g_t - \log g_j$ produces the log of the ratio directly, and only that
difference is exponentiated.

Since the difference is at most 0, the result is at most 1. When the true ratio is
genuinely below $10^{-45}$, the exponential returns 0, which is ==the right answer, not a NaN==.

Lab 06 tests this on purpose. One check runs the whole sequence with
$\hla{\alpha} = 10^{-8}$, a gate that erases essentially everything each step:

- In log space, $\log \alpha = -18.42$, and over a 64-token chunk the cumulative
  sum reaches $-1179$, an unremarkable float32.
- The true $g_{64}$ is $10^{-512}$, which no float32 or float64 can hold.
- Computed as a quotient of products, it's NaN. Computed as
  `exp(log_g_t - log_g_j)`, every entry is correct.

Two more safeguards:

- `clamp_min(1e-12)` stops $\log 0 = -\infty$, which would make the difference
  $-\infty - (-\infty)$, which is NaN.
- The cumulative sum restarts at each chunk boundary rather than running over the
  whole sequence, which bounds $\lvert \log g_t \rvert$ by $C$ times the largest
  per-token rate.

## The rest of the layer

The recurrence is the interesting part but not the whole layer: a short
convolution sits before it, and a gated RMSNorm sits after it.

### The causal depthwise convolution

Before the recurrence sees them, $q$, $k$, and $v$ pass through a four-tap causal
depthwise convolution: one independent filter per channel, looking at the current
position and the three before it.

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
isn't in that count.

`groups=channels` makes it depthwise: each output channel is a function of its own
input channel only. So the weight is `(10240, 4)` rather than a dense
`(10240, 10240, 4)`, which is 40,960 parameters per layer, next to nothing. Per
token, the convolution costs $10{,}240 \times 4$ multiply-adds, about 82k FLOPs,
against roughly 5.5M FLOPs for the recurrence across 48 heads. It's under 2% of
the layer's arithmetic.

What it buys is ==a genuine local window==. The delta rule's state is an
associative memory addressed by content. It has no notion of "the token
immediately before this one" unless a head spends a key direction on encoding
position, and a four-tap convolution supplies that for free.

During decode, the `cache` argument holds the previous three steps, shape
`(batch, 10240, 3)`, so the convolution stays $O(1)$ per token like the rest of
the layer. Chapter 2's memory budget allocates a full kernel width of 4 columns in
bfloat16, giving $10{,}240 \times 4 \times 2 = 81{,}920$ bytes per layer, or 3.75
MiB across the 48 linear layers. That's the difference between 144 MiB of pure
recurrent state and the 147.8 MiB the course quotes.

### The gated RMSNorm output

The recurrence's output is an unconstrained linear combination, so its magnitude
isn't bounded the way softmax attention's is. The layer normalizes each value
head, then gates it:

```python
out = out.transpose(1, 2).reshape(batch, seq, 48, 128)
variance = out.float().pow(2).mean(dim=-1, keepdim=True)   # (b, s, 48, 1)
out = out.float() * torch.rsqrt(variance + self.eps) * self.norm_weight
z = z.view(batch, seq, 48, 128)                            # (b, s, 48, 128)
out = (out * F.silu(z.float())).to(x.dtype)
return self.out_proj(out.reshape(batch, seq, 6144))        # (b, s, 5120)
```

- **The normalization is per head.** It runs over the last axis only, 128
  channels within one head, so each of the 48 heads is scaled independently.
  `norm_weight` has shape `(128,)` and is shared across heads.
- **The gate can amplify.** `z` comes from the same input projection as $q$, $k$,
  and $v$, and is full width at `(batch, seq, 6144)`. It passes through SiLU
  rather than a sigmoid, so it isn't bounded to $(0,1)$ and can amplify as well as
  suppress.

[Chapter 4](/c/04-rmsnorm-and-residuals) covers RMSNorm. The only new thing here
is that the normalization is per head rather than over the residual stream.

## Head sharing: 16 key heads, 48 value heads

Sharing key heads saves parameters, not state. The layer projects 16 query heads
and 16 key heads of width 128, but 48 value heads of width 128, and repeats each
key and query head across three value heads:

```python
self.repeats = num_v_heads // num_k_heads            # 48 // 16 = 3
q = q.repeat_interleave(self.repeats, dim=2)         # (b, s, 16, 128) -> (b, s, 48, 128)
k = k.repeat_interleave(self.repeats, dim=2)
```

This is the same trick grouped-query attention plays in chapter 7, but the saving
lands somewhere different, and the difference matters.

**What it saves.** The $q$ and $k$ projections emit $2 \times 16 \times 128 = 4096$
channels instead of $2 \times 48 \times 128 = 12{,}288$. That's
$5120 \times 8192 = 41.9$M fewer parameters per layer and, across the 48 linear
layers, about 2.01 billion parameters or 4.0 GB in bfloat16. It also narrows the
convolution from 18,432 channels to 10,240.

**What it doesn't save.** The recurrent state is still 48 separate
$128 \times 128$ matrices per layer. Three value heads sharing a key head still
have their own values, their own $\hla{\alpha}$, and their own $\hlb{\beta}$, so
their states diverge from the first token. In GQA, sharing a KV head directly
shrinks the cache, because the cache *is* the shared keys and values. Here,
==quoting the sharing as a memory reduction is wrong==.

What the three heads share is a subspace: the same 128-dimensional key
directions, with different values and different forgetting rates. Given the
timescale spread in the gate table, that's a sensible grouping: the same address
book, three different retention policies.

## Verify the two forms against each other

Run both forms on the same random inputs and compare, because nothing else catches
a chunked-form bug. A sign error, an off-by-one in a mask, or an
inclusive-versus-exclusive triangle produces output that looks entirely
reasonable and is wrong, and there's no way to eyeball it.

The reference implementation agrees to float32 precision across chunk sizes. It
also agrees when you split the sequence and resume from a carried state, which is
what prefill followed by decode does:

```text
chunk=  1  out_maxerr=3.58e-07  state_maxerr=0.00e+00
chunk= 32  out_maxerr=4.77e-07  state_maxerr=3.58e-07
chunk=256  out_maxerr=1.79e-06  state_maxerr=1.31e-06
split-resume maxerr: 4.17e-07
prefill+decode maxerr: 4.77e-07
```

Read the trend:

- At `chunk=1`, the chunked path degenerates to the sequential one, and the states
  match bit for bit.
- As the chunk grows, error grows slowly, because a longer triangular solve
  accumulates more rounding.
- All of it stays within a few multiples of float32's $10^{-7}$ epsilon.

> [!TIP] An error of 1e-3 is a bug
> If your errors are $10^{-3}$ rather than $10^{-6}$, you have a bug, not a
> precision issue. Nothing about float32 arithmetic on these shapes produces
> $10^{-3}$.

## What goes wrong

- **Errors around 1e-3, uniformly.** Almost always a triangle boundary. Check that
  $A$ excludes its diagonal (`diagonal=-1`) and that the output sum includes
  $j = t$. These are the two places the inclusive and exclusive conventions differ.
- **The first output is wrong, and everything after is fine.** The output is
  reading $S_{t-1}$ instead of $S_t$. With an empty initial state, every later
  token has enough history to mask the error.
- **NaN after a few hundred tokens of a long prompt.** Decay ratios computed as a
  quotient of cumulative products. Move to log space.
- **NaN immediately.** $\log 0$ from a gate that reached exactly zero in float32,
  giving $-\infty - (-\infty)$. Apply `clamp_min` to the gate before the logarithm.
- **The state grows without bound.** Unnormalized keys, so
  $1 - \hlb{\beta} \lVert k \rVert^2$ went below $-1$. Normalize.
- **Correct outputs, wrong final state.** The chunk-final state uses the ratio
  $g_C / g_j$, which is `(log_g[..., -1:] - log_g).exp()`, a different quantity
  from the `ratio` matrix used for the outputs. Reusing the last row of `ratio`
  works only when the chunk is full, so the bug appears on the last, short chunk
  of a sequence whose length isn't a multiple of 64.
- **Silent shape broadcast.** The gate needs shape `(b, h, 1, 1)` against the
  state and `(b, h, 1)` against a value vector. Getting it wrong broadcasts to
  something with the correct shape and the wrong contents.

> [!RECAP]
> - Linear attention drops softmax's kernel and normalizer to get a fixed
>   $d_k \times d_v$ state that costs $O(1)$ per token to read. It's a different
>   operator, trained as itself.
> - A state holds about $d_k = 128$ clean associations, with SNR near
>   $d_k / (n - 1)$. Plain accumulation can't overwrite a key.
> - The delta rule is one gradient step on $\tfrac{1}{2}\lVert S^\top k_t - v_t \rVert^2$:
>   erase along $k_t$, then add. Unit-norm keys keep the erase factor's spectral
>   radius at 1.
> - The forget gate $\hla{\alpha_t} = e^{-r_t}$ stays inside $(0, 1)$ by
>   construction, and $n_{90} = \ln 10 / r$ turns $A_{\log}$ into a horizon in
>   tokens.
> - The chunked form solves $(I + A)\,U = R$ with $A$ strictly lower triangular, so
>   the solve always succeeds. It does about 1.36 times the FLOPs at $C = 64$ and
>   wins on shape, not arithmetic.
> - Compute decay ratios as `exp(log_g_t - log_g_j)`, never as a quotient of
>   products.

## Check your understanding

> [!QUESTION] A head has $A_{\log} = -5$, and its raw gate input is such that softplus returns 1. How many tokens until it has forgotten 90% of what it knew?
> The rate is $r = e^{-5} = 0.006738$, and
> $n_{90} = \ln(10)/r = 2.303 / 0.006738 \approx 342$ tokens. Its half-life is
> $\ln(2)/r \approx 103$ tokens.

> [!QUESTION] Why is $I + A$ guaranteed invertible, with no condition on the inputs?
> $A$ is strictly lower triangular, because $u_t$ depends only on strictly earlier
> corrections. So $I + A$ is unit lower triangular, and the determinant of a
> triangular matrix is the product of its diagonal, which is $1^C = 1$. No choice
> of keys, gates, or step sizes changes that.

> [!QUESTION] The chunked form does about 36% more arithmetic than the sequential loop at $C = 64$. Why is it more than 5 times faster?
> Because the sequential loop isn't limited by arithmetic. It issues four
> dependent kernel launches per token on $128 \times 128$ tensors, leaving the
> A100's SMs idle and paying launch latency 2048 times. The chunked form turns the
> same work into dense matmuls the tensor cores can saturate, and shortens the
> dependent chain from 2048 steps to 32.

> [!QUESTION] A head writes 200 distinct unit-norm keys into its 128 by 128 state, then reads back the first one. What comes out?
> The correct value plus an interference term summing the other 199 values
> weighted by their inner products with the query key. For random keys, the
> expected squared inner product is $1/128$, so the interference has roughly
> $199/128 \approx 1.6$ times the energy of the signal. The forget gate is what
> keeps this from happening: by the time the 200th key is written, the first has
> been decayed toward zero.

## Lab

> [!TRY]
> Implement `delta_rule_step`, `delta_rule_recurrent`, and `delta_rule_chunked`,
> and prove they agree. Passing means the chunked form matches the sequential one
> to $10^{-4}$ and, on a GPU, runs more than 5 times faster.

The harness checks:

- The sequential form returns the right shapes.
- With the gates fully open on a fresh state, the first output is exactly
  $(q_1^\top k_1) v_1$, which catches an update applied in the wrong order.
- The chunked form matches the sequential one at chunk sizes 1, 7, 32, 64, and
  256, in both the outputs and the final state, to $10^{-4}$.
- A sequence split in two, with the second half resuming from the first half's
  carried state, matches one long run.
- A chunked prefill of 100 tokens followed by single `delta_rule_step` calls
  matches one long sequential run. This is the real decode path.
- A nearly closed forget gate, $\alpha = 10^{-8}$, agrees between both forms.
  This is the log-space check.
- On a GPU, the chunked form is more than 5 times faster than the sequential one
  on a 2048-token sequence. That speedup is the point of the whole exercise.

## Further reading

- [Gated delta networks: improving Mamba2 with delta rule](https://arxiv.org/abs/2412.06464)
- [Parallelizing linear transformers with the delta rule over sequence length](https://arxiv.org/abs/2406.06484)
- [Transformers are RNNs: fast autoregressive transformers with linear attention](https://arxiv.org/abs/2006.16236)
- [Linear transformers are secretly fast weight programmers](https://arxiv.org/abs/2102.11174) — where the delta rule enters this line of work.
- [Gated linear attention transformers with hardware-efficient training](https://arxiv.org/abs/2312.06635) — the chunked form, in more generality.
- [Mamba: linear-time sequence modeling with selective state spaces](https://arxiv.org/abs/2312.00752) — the input-dependent gate, from the state-space side.
