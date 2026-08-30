"""Reference solution for lab 06, matching engine/layers/linear_attn.py."""

from __future__ import annotations

import torch
from torch import Tensor


def delta_rule_step(
    state: Tensor, q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor
) -> tuple[Tensor, Tensor]:
    """One decode step.

    Args:
        state: (batch, heads, k_dim, v_dim), float32.
        q, k: (batch, heads, k_dim)
        v: (batch, heads, v_dim)
        alpha, beta: (batch, heads) — the forget gate and the write strength.

    Returns:
        The new state and the output, (batch, heads, v_dim).
    """
    alpha = alpha[..., None, None].float()
    beta = beta[..., None].float()
    q, k, v = q.float(), k.float(), v.float()

    read = torch.einsum("bhkv,bhk->bhv", state, k)
    delta = beta * (v - alpha[..., 0] * read)
    state = alpha * state + torch.einsum("bhk,bhv->bhkv", k, delta)
    out = torch.einsum("bhkv,bhk->bhv", state, q)
    return state, out


def delta_rule_recurrent(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
    state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """The plain sequential loop. Correct, slow, and the reference for the rest.

    Args:
        q, k: (batch, heads, seq, k_dim)
        v: (batch, heads, seq, v_dim)
        alpha, beta: (batch, heads, seq)
        state: optional (batch, heads, k_dim, v_dim) carried in.

    Returns:
        Outputs (batch, heads, seq, v_dim) and the final state.
    """
    batch, heads, seq, k_dim = q.shape
    v_dim = v.shape[-1]
    if state is None:
        state = torch.zeros(batch, heads, k_dim, v_dim, device=q.device, dtype=torch.float32)

    outputs = []
    for t in range(seq):
        state, out = delta_rule_step(
            state, q[:, :, t], k[:, :, t], v[:, :, t], alpha[:, :, t], beta[:, :, t]
        )
        outputs.append(out)
    return torch.stack(outputs, dim=2).to(v.dtype), state


def delta_rule_chunked(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
    state: Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[Tensor, Tensor]:
    """Chunk-parallel form of the same recurrence.

    Inside a chunk, write the cumulative decay g_t = prod_{i<=t} a_i. Unrolling
    the recurrence gives

        S_t = g_t S_0 + sum_{j<=t} (g_t / g_j) k_j u_j^T
        u_t = b_t (v_t - g_t S_0^T k_t) - sum_{j<t} b_t (g_t/g_j) (k_j . k_t) u_j

    The second line is a linear system in u with a strictly lower-triangular
    matrix, so (I + A) u = rhs solves exactly in one triangular solve. Every
    other term is a matrix multiply. Only the chunk-boundary state stays
    sequential, so the loop runs seq/chunk_size times instead of seq times.

    All ratios g_t/g_j with j <= t are at most 1 because every a is in (0, 1],
    and they are computed in log space, so nothing overflows.
    """
    batch, heads, seq, k_dim = q.shape
    v_dim = v.shape[-1]
    device = q.device

    q32, k32, v32 = q.float(), k.float(), v.float()
    log_alpha = torch.log(alpha.float().clamp_min(1e-12))
    beta32 = beta.float()

    if state is None:
        state = torch.zeros(batch, heads, k_dim, v_dim, device=device, dtype=torch.float32)
    else:
        state = state.float()

    eye = torch.eye(chunk_size, device=device, dtype=torch.float32)
    outputs = torch.empty(batch, heads, seq, v_dim, device=device, dtype=torch.float32)

    for start in range(0, seq, chunk_size):
        stop = min(start + chunk_size, seq)
        size = stop - start

        qc = q32[:, :, start:stop]
        kc = k32[:, :, start:stop]
        vc = v32[:, :, start:stop]
        bc = beta32[:, :, start:stop]

        # g_t within the chunk, in log space.
        log_g = log_alpha[:, :, start:stop].cumsum(dim=-1)              # (b,h,c)
        g = log_g.exp()

        # ratio[t, j] = g_t / g_j, kept only where j <= t.
        log_ratio = log_g[..., :, None] - log_g[..., None, :]           # (b,h,c,c)
        causal = torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))
        ratio = torch.where(causal, log_ratio.exp(), torch.zeros((), device=device))

        # A[t, j] = b_t * ratio[t, j] * (k_t . k_j), strictly lower triangular.
        kk = torch.einsum("bhtd,bhjd->bhtj", kc, kc)
        a_mat = bc[..., :, None] * ratio * kk
        a_mat = a_mat * torch.tril(
            torch.ones(size, size, device=device, dtype=a_mat.dtype), diagonal=-1
        )

        # rhs_t = b_t * (v_t - g_t * S_0^T k_t)
        read0 = torch.einsum("bhkv,bhtk->bhtv", state, kc)
        rhs = bc[..., None] * (vc - g[..., None] * read0)

        u = torch.linalg.solve_triangular(
            eye[:size, :size] + a_mat, rhs, upper=False, unitriangular=False
        )

        # o_t = g_t * S_0^T q_t + sum_{j<=t} ratio[t,j] (q_t . k_j) u_j
        inter = g[..., None] * torch.einsum("bhkv,bhtk->bhtv", state, qc)
        qk = torch.einsum("bhtd,bhjd->bhtj", qc, kc) * ratio
        outputs[:, :, start:stop] = inter + torch.einsum("bhtj,bhjv->bhtv", qk, u)

        # S_C = g_C S_0 + sum_j (g_C / g_j) k_j u_j^T
        tail = (log_g[..., -1:] - log_g).exp()
        state = g[..., -1, None, None] * state + torch.einsum(
            "bhjk,bhjv->bhkv", kc * tail[..., None], u
        )

    return outputs.to(v.dtype), state
