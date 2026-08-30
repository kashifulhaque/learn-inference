"""Lab 14 — FlashAttention.

Tile the keys, keep a running maximum and denominator per query row, and rescale
the accumulator whenever the maximum moves. The score matrix never reaches HBM.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    n_heads, q_len, kv_len,
    kv_group: tl.constexpr,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """One program handles one tile of queries against every key.

    The online softmax update, for each new tile of scores s:

        m_new = max(m, max(s))
        l     = l * exp(m - m_new) + sum(exp(s - m_new))
        acc   = acc * exp(m - m_new) + exp(s - m_new) @ v
    """
    # TODO
    pass


def flash_attention(q, k, v, causal: bool = True, scale: float | None = None,
                    block_m: int = 64, block_n: int = 64):
    """FlashAttention forward.

    Args:
        q: (batch, heads, q_len, head_dim)
        k, v: (batch, kv_heads, kv_len, head_dim); kv_heads must divide heads.
    """
    # TODO
    raise NotImplementedError
