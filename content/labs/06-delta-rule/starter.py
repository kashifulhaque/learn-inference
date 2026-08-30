"""Lab 06 — the gated delta rule.

The recurrence, with state S of shape (key_dim, value_dim) per head:

    S_t = a_t * S_{t-1} + b_t * k_t (v_t - a_t * S_{t-1}^T k_t)^T
    o_t = S_t^T q_t

Implement it sequentially first, then in chunk-parallel form, and show they
agree. The chunked version is what makes prefill possible.
"""

import torch


def delta_rule_step(state, q, k, v, alpha, beta):
    """One decode step.

    Args:
        state: (batch, heads, k_dim, v_dim), float32.
        q, k: (batch, heads, k_dim)
        v: (batch, heads, v_dim)
        alpha, beta: (batch, heads)

    Returns:
        (new_state, output) where output is (batch, heads, v_dim).
    """
    # TODO
    raise NotImplementedError


def delta_rule_recurrent(q, k, v, alpha, beta, state=None):
    """The sequential loop over the sequence.

    Args:
        q, k: (batch, heads, seq, k_dim)
        v: (batch, heads, seq, v_dim)
        alpha, beta: (batch, heads, seq)
        state: optional (batch, heads, k_dim, v_dim) carried in.

    Returns:
        (outputs, final_state), outputs shaped (batch, heads, seq, v_dim).
    """
    # TODO
    raise NotImplementedError


def delta_rule_chunked(q, k, v, alpha, beta, state=None, chunk_size=64):
    """The chunk-parallel form.

    Within a chunk, with g_t the cumulative decay:

        u = (I + A)^-1 [ b * (v - g * S_0^T k) ]
        A[t, j] = b_t (g_t/g_j) (k_j . k_t)   for j < t
        o_t     = g_t S_0^T q_t + sum_{j<=t} (g_t/g_j) (q_t . k_j) u_j
        S_C     = g_C S_0 + sum_j (g_C/g_j) k_j u_j^T

    Same signature and same results as delta_rule_recurrent, plus chunk_size.
    """
    # TODO
    raise NotImplementedError
