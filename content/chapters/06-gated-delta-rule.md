---
title: Linear attention and the gated delta rule
slug: 06-gated-delta-rule
part: "Part 2 — A forward pass"
summary: The recurrence that runs 48 of this model's 64 layers, and the chunked form that makes prefill possible.
minutes: 90
gpu: true
objectives:
  - Derive linear attention from softmax attention and explain what the kernel trick removes.
  - Explain what the delta rule fixes and why the keys are normalized.
  - Implement the chunked parallel form and verify it against the sequential recurrence.
lab: 06-delta-rule
---

# Linear attention and the gated delta rule

Three quarters of this model's layers don't use attention. They run a recurrence
with a fixed-size state. This is the chapter that explains why, and it's the
hardest one in the course.

## From softmax attention to a recurrence

Softmax attention computes, for query position *t*:

```text
o_t = sum_{j<=t} exp(q_t . k_j) v_j / sum_{j<=t} exp(q_t . k_j)
```

The `exp` couples every query to every key, which is why you must keep all past
keys and values. Replace it with a plain dot product and the sum factors:

```text
o_t = sum_{j<=t} (q_t . k_j) v_j = q_t . (sum_{j<=t} k_j v_j^T)
```

The bracketed term doesn't depend on *t* except through the range of the sum, so
you can maintain it incrementally:

```text
S_t = S_{t-1} + k_t v_t^T
o_t = S_t^T q_t
```

That's linear attention. The state `S` is a fixed `k_dim x v_dim` matrix. Reading
it costs the same at token 10 and token 100,000, and generating a token is O(1)
in context length instead of O(n).

The cost is capacity. `S` is a fixed-size associative memory holding a sum of
outer products, and once you've written more key-value pairs than it can keep
apart, new writes interfere with old ones. Plain linear attention degrades badly
over long sequences for exactly this reason.

## The delta rule

The fix is to stop accumulating and start *overwriting*. Before writing a new
value for key `k_t`, read what the state currently associates with that key and
write back only the difference:

```text
S_t = S_{t-1} + b_t * k_t (v_t - S_{t-1}^T k_t)^T
```

Read `S_{t-1}^T k_t` as "what does the state currently return for this key". If
that already equals `v_t`, the correction is zero and nothing is written. If it
differs, the state moves toward `v_t` with step size `b_t`.

This is the classic delta rule from online learning, and it's the same update as
one step of gradient descent on `||S^T k - v||²`. The state stops saturating,
because writing the same key twice doesn't double its contribution.

## The gate

Add a forget gate `a_t` in (0, 1) that shrinks everything already in the state:

```text
S_t = a_t * S_{t-1} + b_t * k_t (v_t - a_t * S_{t-1}^T k_t)^T
o_t = S_t^T q_t
```

Now the layer can discard. Each of the 48 value heads gets its own gate,
parameterized so it can't leave (0, 1) whatever the input does:

```python
decay_rate = F.softplus(a_raw + self.dt_bias) * self.A_log.exp()
alpha = torch.exp(-decay_rate)
```

`softplus` keeps the rate positive, so `alpha` stays in (0, 1). `A_log` is learned
per head and sets its timescale: a head with a large `A_log` forgets in a few
tokens and specializes in local structure; one with a small `A_log` holds
information for thousands of tokens. A layer's 48 heads end up covering a wide
range of timescales, which is how a fixed-size state serves both roles.

The keys are L2-normalized before use. With unit-norm keys, `k k^T` is a
projection matrix and `(I - b k k^T)` is a well-conditioned partial erase of one
direction. Without normalization the erase overshoots or undershoots depending on
key magnitude, and the recurrence becomes unstable over long sequences.

## The sequential form

```python
def delta_rule_step(state, q, k, v, alpha, beta):
    read = torch.einsum("bhkv,bhk->bhv", state, k)
    delta = beta * (v - alpha * read)
    state = alpha * state + torch.einsum("bhk,bhv->bhkv", k, delta)
    out = torch.einsum("bhkv,bhk->bhv", state, q)
    return state, out
```

This is exactly right and completely unusable for prefill. A 2000-token prompt
becomes 2000 sequential steps, each a handful of tiny operations on a GPU built
for large parallel ones.

## The chunked form

Split the sequence into chunks of 64. Within a chunk, unroll the recurrence.

Write `g_t` for the cumulative decay `prod_{i<=t} a_i`. Unrolling gives:

```text
S_t = g_t S_0 + sum_{j<=t} (g_t / g_j) k_j u_j^T
```

where `u_t = b_t (v_t - a_t S_{t-1}^T k_t)` is the correction written at step *t*.
Substituting the expansion of `S_{t-1}` into that definition:

```text
u_t = b_t (v_t - g_t S_0^T k_t) - sum_{j<t} b_t (g_t/g_j) (k_j . k_t) u_j
```

This is a linear system in `u`. Collect the coefficients into a matrix

```text
A[t, j] = b_t (g_t/g_j) (k_j . k_t)   for j < t,  0 otherwise
```

and it becomes `(I + A) u = rhs`, where `A` is strictly lower triangular. A
triangular solve gives every `u` in the chunk at once. Then the outputs and the
chunk-final state are matrix multiplies:

```text
o_t = g_t S_0^T q_t + sum_{j<=t} (g_t/g_j) (q_t . k_j) u_j
S_C = g_C S_0 + sum_j (g_C/g_j) k_j u_j^T
```

Only the chunk-boundary state stays sequential, so a 2000-token prompt runs 32
iterations instead of 2000, and each iteration is dense linear algebra.

Every ratio `g_t / g_j` with `j <= t` is at most 1, because every `a` is at most
1. Computing them as `exp(log g_t - log g_j)` keeps that true in floating point;
computing them as a quotient of cumulative products underflows to zero within a
few hundred tokens.

## Verify the two against each other

This is the part to take seriously. The chunked form is intricate enough that a
sign error produces output that looks reasonable and is wrong. Run both on the
same random inputs and compare.

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

If your errors are 1e-3 rather than 1e-6, you have a bug, not a precision issue.

## The rest of the layer

Two more pieces sit around the recurrence.

A **causal depthwise convolution** with a four-tap kernel runs over q, k, and v
before the recurrence. It gives each token a local window, which the delta rule
alone doesn't provide. During decode the previous three steps live in a small
cache, so it stays O(1).

A **gated RMSNorm** normalizes each value head's output and multiplies it by
`silu(z)`, where `z` comes from the same input projection. The output gate lets
the layer suppress its own contribution to the residual stream.

## Head sharing

There are 16 key heads and 48 value heads. Each key head serves three value
heads — the same trick as grouped-query attention, for the same reason: the key
projections are what you'd otherwise have to compute and store three times.

## Lab

Implement `delta_rule_recurrent` and `delta_rule_chunked`, and prove they agree.
The harness checks four things: the two forms match on random input, the chunked
form is invariant to chunk size, a carried-in state resumes correctly, and a
chunked prefill followed by single steps matches one long sequential run.

Then time both. The speedup on a 2048-token sequence is the point of the whole
exercise.

## Further reading

- [Gated delta networks: improving Mamba2 with delta rule](https://arxiv.org/abs/2412.06464)
- [Parallelizing linear transformers with the delta rule over sequence length](https://arxiv.org/abs/2406.06484)
- [Transformers are RNNs: fast autoregressive transformers with linear attention](https://arxiv.org/abs/2006.16236)
