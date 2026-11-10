"""Lab 14 — FlashAttention.

Tile the keys, keep a running maximum and denominator per query row, and rescale
the accumulator whenever the maximum moves. The score matrix never reaches HBM.

The three helpers at the bottom count launches and rescale work. They run on
any machine, and the grader checks them even without a GPU.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # CPU-only machine: the helpers at the bottom still run
    HAS_TRITON = False

A100_SMS = 108


if HAS_TRITON:

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


def count_programs(batch: int, heads: int, q_len: int, kv_len: int, block_m: int) -> int:
    """Thread blocks (programs) your flash_attention launches.

    Look at the `grid` you wrote above. Count what it counts, and nothing else.
    """
    # TODO
    raise NotImplementedError("count_programs")


def sm_fill(programs: int, num_sms: int = A100_SMS) -> float:
    """Fraction of the SMs that hold at least one program in the first wave."""
    # TODO
    raise NotImplementedError("sm_fill")


def rescale_elements(q_len: int, kv_len: int, block_m: int, block_n: int,
                     head_dim: int, version: int, causal: bool = True) -> int:
    """Accumulator elements scaled by a row factor, per head.

    version=1: FlashAttention-1 rescales the old output and the new P @ V on
    every KV tile, and never divides at the end.
    version=2: FlashAttention-2 rescales the old output only, on every KV tile,
    then divides by the row sum once when the loop ends.
    Each scaling touches rows * head_dim elements, where rows is the number of
    valid query rows in the tile. With causal=True, query tile i stops at
    min(kv_len, (i + 1) * block_m + kv_len - q_len) keys.
    """
    # TODO
    raise NotImplementedError("rescale_elements")
