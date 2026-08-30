"""Caches for a hybrid model.

Two kinds of layer means two kinds of state:

*Full-attention layers* keep every past key and value. The cache grows by
`2 * kv_heads * head_dim` elements per token per layer and is read in full on
every decode step, which makes decode memory bound.

*Linear-attention layers* keep one fixed-size matrix per sequence plus a
three-step convolution window. Size does not depend on sequence length.

`HybridCache` is the simple contiguous version: one preallocated tensor per
layer, sized to a maximum length. It is easy to read and wastes a lot of memory
when requests are shorter than that maximum. `PagedKVCache` fixes the waste by
handing out fixed-size blocks; see chapter 12.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import ModelConfig


class HybridCache:
    """Contiguous per-layer cache for one batch of sequences."""

    def __init__(
        self,
        config: ModelConfig,
        batch_size: int,
        max_seq_len: int,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)
        self.dtype = dtype
        self.length = 0

        self.k_cache: dict[int, Tensor] = {}
        self.v_cache: dict[int, Tensor] = {}
        self.states: dict[int, Tensor] = {}
        self.conv_states: dict[int, Tensor] = {}

        for layer in config.full_attention_layers:
            shape = (batch_size, config.num_key_value_heads, max_seq_len, config.head_dim)
            self.k_cache[layer] = torch.zeros(shape, device=device, dtype=dtype)
            self.v_cache[layer] = torch.zeros(shape, device=device, dtype=dtype)

        conv_channels = (
            2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim
        )
        for layer in config.linear_attention_layers:
            self.states[layer] = torch.zeros(
                batch_size,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
                device=device,
                dtype=torch.float32,
            )
            self.conv_states[layer] = torch.zeros(
                batch_size,
                conv_channels,
                config.linear_conv_kernel_dim - 1,
                device=device,
                dtype=dtype,
            )

    # --- full attention -----------------------------------------------------

    def append(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write this step's K and V, then return the whole prefix."""
        seq = k.shape[2]
        start = self.length
        stop = start + seq
        if stop > self.max_seq_len:
            raise RuntimeError(
                f"Cache overflow: {stop} tokens requested, capacity {self.max_seq_len}. "
                "Either raise max_seq_len or start evicting."
            )
        self.k_cache[layer_idx][:, :, start:stop] = k
        self.v_cache[layer_idx][:, :, start:stop] = v
        return (
            self.k_cache[layer_idx][:, :, :stop],
            self.v_cache[layer_idx][:, :, :stop],
        )

    # --- linear attention ---------------------------------------------------

    def recurrent_state(self, layer_idx: int) -> Tensor | None:
        return self.states.get(layer_idx)

    def conv_state(self, layer_idx: int) -> Tensor | None:
        return self.conv_states.get(layer_idx)

    def set_linear_state(self, layer_idx: int, state: Tensor, conv: Tensor) -> None:
        self.states[layer_idx] = state
        self.conv_states[layer_idx] = conv

    # --- bookkeeping --------------------------------------------------------

    def advance(self, tokens: int) -> None:
        """Move the write cursor after a full forward pass over all layers."""
        self.length += tokens

    def reset(self) -> None:
        self.length = 0
        for state in self.states.values():
            state.zero_()
        for conv in self.conv_states.values():
            conv.zero_()

    def memory_bytes(self) -> dict[str, int]:
        kv = sum(t.numel() * t.element_size() for t in self.k_cache.values())
        kv += sum(t.numel() * t.element_size() for t in self.v_cache.values())
        rec = sum(t.numel() * t.element_size() for t in self.states.values())
        rec += sum(t.numel() * t.element_size() for t in self.conv_states.values())
        return {"kv": kv, "recurrent": rec, "total": kv + rec}


# ---------------------------------------------------------------------------
# Paged KV cache
# ---------------------------------------------------------------------------


@dataclass
class BlockTable:
    """Which physical blocks hold a sequence's KV, in logical order."""

    blocks: list[int]
    length: int = 0


class OutOfBlocks(RuntimeError):
    """No free block left. The scheduler should preempt something."""


class PagedKVCache:
    """KV storage in fixed-size blocks, the way an operating system pages memory.

    A contiguous cache has to reserve `max_seq_len` per sequence up front, so a
    batch of 64 requests that could reach 32k tokens reserves for all of them
    even though most stop after a few hundred. Blocks remove that: a sequence
    holds only the blocks it has filled, and the tail block is the only place
    memory is wasted, at most `block_size - 1` tokens.

    The cost is one indirection. Attention can no longer read a contiguous
    range; it walks a block table. That is what the paged attention kernel in
    chapter 12 does.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_blocks: int,
        block_size: int = 16,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = torch.device(device)
        self.dtype = dtype

        shape = (num_blocks, block_size, config.num_key_value_heads, config.head_dim)
        self.k_blocks = {
            layer: torch.zeros(shape, device=device, dtype=dtype)
            for layer in config.full_attention_layers
        }
        self.v_blocks = {
            layer: torch.zeros(shape, device=device, dtype=dtype)
            for layer in config.full_attention_layers
        }

        self.free: list[int] = list(range(num_blocks))
        self.tables: dict[str, BlockTable] = {}

    # --- allocation ---------------------------------------------------------

    @property
    def free_blocks(self) -> int:
        return len(self.free)

    def blocks_needed(self, seq_id: str, extra_tokens: int) -> int:
        table = self.tables.get(seq_id)
        current = table.length if table else 0
        have = len(table.blocks) * self.block_size if table else 0
        return max(0, -(-(current + extra_tokens - have) // self.block_size))

    def allocate(self, seq_id: str, extra_tokens: int) -> BlockTable:
        table = self.tables.setdefault(seq_id, BlockTable(blocks=[]))
        need = self.blocks_needed(seq_id, extra_tokens)
        if need > len(self.free):
            raise OutOfBlocks(
                f"{seq_id} needs {need} blocks, {len(self.free)} free. Preempt a "
                "running sequence or lower the batch size."
            )
        for _ in range(need):
            table.blocks.append(self.free.pop())
        table.length += extra_tokens
        return table

    def release(self, seq_id: str) -> None:
        table = self.tables.pop(seq_id, None)
        if table:
            self.free.extend(table.blocks)

    def utilisation(self) -> float:
        return 1.0 - len(self.free) / self.num_blocks

    # --- reads and writes ---------------------------------------------------

    def slot_indices(self, seq_id: str, positions: Tensor) -> Tensor:
        """Map logical token positions to flat slots in the block pool."""
        table = self.tables[seq_id]
        blocks = torch.tensor(table.blocks, device=positions.device, dtype=torch.long)
        return blocks[positions // self.block_size] * self.block_size + (
            positions % self.block_size
        )

    def write(self, layer_idx: int, seq_id: str, positions: Tensor, k: Tensor, v: Tensor) -> None:
        """Scatter K and V for `positions` into their blocks.

        Args:
            k, v: (tokens, kv_heads, head_dim)
        """
        slots = self.slot_indices(seq_id, positions)
        kb = self.k_blocks[layer_idx].view(-1, self.config.num_key_value_heads, self.config.head_dim)
        vb = self.v_blocks[layer_idx].view(-1, self.config.num_key_value_heads, self.config.head_dim)
        kb[slots] = k.to(self.dtype)
        vb[slots] = v.to(self.dtype)

    def gather(self, layer_idx: int, seq_id: str) -> tuple[Tensor, Tensor]:
        """Collect a sequence's whole KV prefix into contiguous tensors.

        Convenient for checking a paged kernel against the naive path. A real
        decode never calls this: gathering defeats the point of paging.
        """
        table = self.tables[seq_id]
        positions = torch.arange(table.length, device=self.device)
        slots = self.slot_indices(seq_id, positions)
        kb = self.k_blocks[layer_idx].view(-1, self.config.num_key_value_heads, self.config.head_dim)
        vb = self.v_blocks[layer_idx].view(-1, self.config.num_key_value_heads, self.config.head_dim)
        return kb[slots], vb[slots]

    def memory_bytes(self) -> int:
        per_layer = self.k_blocks[next(iter(self.k_blocks))]
        return 2 * per_layer.numel() * per_layer.element_size() * len(self.k_blocks)
