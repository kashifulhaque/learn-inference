"""Lab 15 — paged attention.

A contiguous cache reserves the maximum length for every sequence. Blocks let a
sequence hold only what it has filled.
"""

import torch


class PagedCache:
    def __init__(self, num_blocks: int, block_size: int, num_kv_heads: int,
                 head_dim: int, device="cpu", dtype=torch.float32):
        """Allocate the block pool and the free list.

        Storage is one tensor of shape
        (num_blocks, block_size, num_kv_heads, head_dim) for keys and one for
        values.
        """
        # TODO
        raise NotImplementedError

    @property
    def free_blocks(self) -> int:
        """How many blocks are unallocated."""
        # TODO
        raise NotImplementedError

    def blocks_needed(self, seq_id: str, extra_tokens: int) -> int:
        """How many new blocks appending `extra_tokens` to `seq_id` requires."""
        # TODO
        raise NotImplementedError

    def allocate(self, seq_id: str, extra_tokens: int) -> list[int]:
        """Reserve room for `extra_tokens` more tokens; return the block table.

        Raise RuntimeError when there are not enough free blocks.
        """
        # TODO
        raise NotImplementedError

    def release(self, seq_id: str) -> None:
        """Return a sequence's blocks to the free list."""
        # TODO
        raise NotImplementedError

    def slot_indices(self, seq_id: str, positions: torch.Tensor) -> torch.Tensor:
        """Map logical positions to flat slots in the block pool."""
        # TODO
        raise NotImplementedError

    def write(self, seq_id: str, positions: torch.Tensor, k: torch.Tensor,
              v: torch.Tensor) -> None:
        """Scatter K and V into their blocks.

        Args:
            k, v: (tokens, num_kv_heads, head_dim)
        """
        # TODO
        raise NotImplementedError

    def gather(self, seq_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect a sequence's whole prefix into contiguous tensors.

        Useful for checking a paged kernel. A real decode never does this.
        """
        # TODO
        raise NotImplementedError


def paged_decode_attention(cache: "PagedCache", seq_ids: list[str],
                           q: torch.Tensor, scale: float | None = None
                           ) -> torch.Tensor:
    """One decode step for a batch of sequences, reading through block tables.

    Args:
        q: (num_seqs, heads, head_dim)

    Returns:
        (num_seqs, heads, head_dim). Query head h reads KV head
        h // (heads // num_kv_heads).
    """
    # TODO
    raise NotImplementedError


def memory_waste(lengths: list[int], max_seq_len: int, block_size: int
                 ) -> tuple[float, float]:
    """Return (contiguous_waste, paged_waste) as fractions in [0, 1].

    Contiguous reserves `max_seq_len` per sequence. Paged reserves whole blocks.
    Waste is reserved-but-unused divided by reserved.
    """
    # TODO
    raise NotImplementedError
