"""Gated delta-rule linear attention — 48 of this model's 64 layers.

Softmax attention keeps every past key and value, so its state grows with the
sequence and reading that state is what makes long-context decode slow. Linear
attention replaces the growing cache with a fixed-size matrix S and updates it
one token at a time:

    S_t = a_t * S_{t-1} + b_t * k_t (v_t - a_t * S_{t-1}^T k_t)^T
    o_t = S_t^T q_t

Read the update as three moves. `a_t` in (0, 1) forgets: it shrinks everything
already written. `S_{t-1}^T k_t` reads out whatever the state currently
associates with this key. Subtracting that from `v_t` and writing the
difference back with strength `b_t` is the delta rule — it overwrites the old
association instead of piling a new one on top, which is what keeps the state
from saturating. Drop the subtraction and you get plain linear attention, which
degrades badly over long sequences.

The state costs the same whether the sequence is 100 tokens or 100k. For this
model that is 48 heads x 128 x 128 floats per layer, about 3 MB, against a KV
cache that would grow by 4 KB per token per layer forever.

The catch is that the recurrence is sequential, which would make prefill
hopeless. `delta_rule_chunked` fixes that: it splits the sequence into chunks,
solves each chunk's internal dependencies in closed form with one triangular
solve, and passes only the chunk-boundary state forward. Within a chunk the
work becomes matrix multiplies; across chunks it stays a short loop.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# The recurrence, three ways. All three must agree; lab 06 checks that they do.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------


def causal_depthwise_conv1d(
    x: Tensor, weight: Tensor, cache: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Short causal depthwise convolution over the sequence.

    A four-tap window per channel. It is cheap and gives each token a local view
    before the recurrence sees it, which the delta rule alone does not provide.
    During decode the previous three steps are kept in `cache`, so the whole
    thing stays O(1) per token.

    Args:
        x: (batch, seq, channels)
        weight: (channels, kernel)
        cache: (batch, channels, kernel - 1) from the previous call.

    Returns:
        The convolved sequence and the updated cache.
    """
    batch, seq, channels = x.shape
    kernel = weight.shape[-1]
    xt = x.transpose(1, 2)                                   # (b, c, s)

    if cache is None:
        cache = torch.zeros(batch, channels, kernel - 1, device=x.device, dtype=x.dtype)
    padded = torch.cat([cache, xt], dim=-1)
    new_cache = padded[..., -(kernel - 1):] if kernel > 1 else cache

    out = F.conv1d(padded, weight.unsqueeze(1), groups=channels)
    return F.silu(out.transpose(1, 2)), new_cache


class GatedDeltaNet(nn.Module):
    """The linear-attention layer, gates and all.

    Shapes for Qwen3.8-27B: 16 key heads of width 128, 48 value heads of width
    128. Key and query heads are shared across three value heads each, the same
    trick GQA plays, for the same reason.
    """

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        k_head_dim: int,
        v_head_dim: int,
        conv_kernel: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.conv_kernel = conv_kernel
        self.repeats = num_v_heads // num_k_heads

        self.k_total = num_k_heads * k_head_dim
        self.v_total = num_v_heads * v_head_dim

        # q, k, v and the output gate z in one matrix.
        self.in_proj_qkvz = nn.Linear(
            hidden_size, 2 * self.k_total + 2 * self.v_total, bias=False
        )
        # beta (write strength) and a (forget gate), one scalar per value head.
        self.in_proj_ba = nn.Linear(hidden_size, 2 * num_v_heads, bias=False)

        self.conv_weight = nn.Parameter(
            torch.zeros(2 * self.k_total + self.v_total, conv_kernel)
        )
        self.A_log = nn.Parameter(torch.zeros(num_v_heads))
        self.dt_bias = nn.Parameter(torch.zeros(num_v_heads))
        self.norm_weight = nn.Parameter(torch.ones(v_head_dim))
        self.out_proj = nn.Linear(self.v_total, hidden_size, bias=False)
        self.eps = eps

    def gates(self, ba: Tensor) -> tuple[Tensor, Tensor]:
        """Turn the raw projection into alpha (forget) and beta (write).

        alpha = exp(-exp(A_log) * softplus(a + dt_bias)) sits in (0, 1) for any
        input, and exp(A_log) gives each head its own timescale: a head with a
        large A_log forgets in a few tokens, one with a small A_log holds on for
        thousands.
        """
        b_raw, a_raw = ba.chunk(2, dim=-1)
        beta = torch.sigmoid(b_raw)
        decay_rate = F.softplus(a_raw.float() + self.dt_bias) * self.A_log.exp()
        alpha = torch.exp(-decay_rate)
        return alpha, beta

    def forward(
        self, x: Tensor, cache=None, layer_idx: int = 0, chunk_size: int = 64
    ) -> Tensor:
        batch, seq, _ = x.shape

        qkvz = self.in_proj_qkvz(x)
        q, k, v, z = torch.split(
            qkvz, [self.k_total, self.k_total, self.v_total, self.v_total], dim=-1
        )

        conv_cache = cache.conv_state(layer_idx) if cache is not None else None
        mixed, new_conv = causal_depthwise_conv1d(
            torch.cat([q, k, v], dim=-1), self.conv_weight, conv_cache
        )
        q, k, v = torch.split(mixed, [self.k_total, self.k_total, self.v_total], dim=-1)

        q = q.view(batch, seq, self.num_k_heads, self.k_head_dim)
        k = k.view(batch, seq, self.num_k_heads, self.k_head_dim)
        v = v.view(batch, seq, self.num_v_heads, self.v_head_dim)

        # The delta rule needs unit-norm keys: k k^T is then a projection, and
        # (I - b k k^T) is a well-conditioned partial erase of that direction.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Share each key/query head across `repeats` value heads.
        q = q.repeat_interleave(self.repeats, dim=2).transpose(1, 2)
        k = k.repeat_interleave(self.repeats, dim=2).transpose(1, 2)
        v = v.transpose(1, 2)

        alpha, beta = self.gates(self.in_proj_ba(x))
        alpha = alpha.transpose(1, 2)
        beta = beta.transpose(1, 2)

        prev_state = cache.recurrent_state(layer_idx) if cache is not None else None
        if seq == 1:
            new_state, out = delta_rule_step(
                prev_state
                if prev_state is not None
                else torch.zeros(
                    batch, self.num_v_heads, self.k_head_dim, self.v_head_dim,
                    device=x.device, dtype=torch.float32,
                ),
                q[:, :, 0], k[:, :, 0], v[:, :, 0], alpha[:, :, 0], beta[:, :, 0],
            )
            out = out.unsqueeze(2)
        else:
            out, new_state = delta_rule_chunked(
                q, k, v, alpha, beta, prev_state, chunk_size=chunk_size
            )

        if cache is not None:
            cache.set_linear_state(layer_idx, new_state, new_conv)

        out = out.transpose(1, 2).reshape(batch, seq, self.num_v_heads, self.v_head_dim)

        # Gated RMSNorm over each value head, then project back.
        variance = out.float().pow(2).mean(dim=-1, keepdim=True)
        out = (out.float() * torch.rsqrt(variance + self.eps) * self.norm_weight)
        z = z.view(batch, seq, self.num_v_heads, self.v_head_dim)
        out = (out * F.silu(z.float())).to(x.dtype)

        return self.out_proj(out.reshape(batch, seq, self.v_total))
