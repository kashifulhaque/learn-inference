"""Lab 09 — the hybrid cache.

Fill in the cache the engine uses. Full-attention layers append keys and values;
linear-attention layers overwrite a fixed-size state and a convolution window.
"""

import torch


class Cache:
    def __init__(self, full_layers, linear_layers, batch_size, max_seq_len,
                 num_kv_heads, head_dim, state_shape, conv_shape,
                 device="cpu", dtype=torch.float32):
        """Preallocate every buffer.

        Args:
            full_layers: Indices of layers that keep a KV cache.
            linear_layers: Indices of layers that keep a recurrent state.
            state_shape: Per-layer recurrent state shape, without the batch axis.
            conv_shape: Per-layer convolution window shape, without the batch axis.
        """
        self.length = 0
        # TODO: allocate k_cache, v_cache, states, and conv_states.
        raise NotImplementedError

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Write this step's K and V, then return views of the whole prefix.

        Args:
            k, v: (batch, kv_heads, seq, head_dim)

        Returns:
            (keys, values) covering positions 0 through length + seq.

        Raise RuntimeError when the write would exceed max_seq_len.
        """
        # TODO
        raise NotImplementedError

    def recurrent_state(self, layer_idx: int):
        """Return the stored state for a linear-attention layer."""
        # TODO
        raise NotImplementedError

    def conv_state(self, layer_idx: int):
        """Return the stored convolution window for a linear-attention layer."""
        # TODO
        raise NotImplementedError

    def set_linear_state(self, layer_idx: int, state, conv) -> None:
        """Replace both linear-attention buffers for a layer."""
        # TODO
        raise NotImplementedError

    def advance(self, tokens: int) -> None:
        """Move the write cursor. Call once per forward pass, not per layer."""
        # TODO
        raise NotImplementedError

    def memory_bytes(self) -> dict:
        """Return {"kv": ..., "recurrent": ..., "total": ...} in bytes."""
        # TODO
        raise NotImplementedError
