import torch


class Cache:
    def __init__(self, full_layers, linear_layers, batch_size, max_seq_len,
                 num_kv_heads, head_dim, state_shape, conv_shape,
                 device="cpu", dtype=torch.float32):
        self.length = 0
        self.max_seq_len = max_seq_len
        shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.k_cache = {i: torch.zeros(shape, device=device, dtype=dtype)
                        for i in full_layers}
        self.v_cache = {i: torch.zeros(shape, device=device, dtype=dtype)
                        for i in full_layers}
        self.states = {
            i: torch.zeros((batch_size, *state_shape), device=device,
                           dtype=torch.float32)
            for i in linear_layers
        }
        self.conv_states = {
            i: torch.zeros((batch_size, *conv_shape), device=device, dtype=dtype)
            for i in linear_layers
        }

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        seq = k.shape[2]
        start, stop = self.length, self.length + seq
        if stop > self.max_seq_len:
            raise RuntimeError(
                f"Cache overflow: {stop} tokens requested, capacity {self.max_seq_len}."
            )
        self.k_cache[layer_idx][:, :, start:stop] = k
        self.v_cache[layer_idx][:, :, start:stop] = v
        return self.k_cache[layer_idx][:, :, :stop], self.v_cache[layer_idx][:, :, :stop]

    def recurrent_state(self, layer_idx: int):
        return self.states.get(layer_idx)

    def conv_state(self, layer_idx: int):
        return self.conv_states.get(layer_idx)

    def set_linear_state(self, layer_idx: int, state, conv) -> None:
        self.states[layer_idx] = state
        self.conv_states[layer_idx] = conv

    def advance(self, tokens: int) -> None:
        self.length += tokens

    def memory_bytes(self) -> dict:
        kv = sum(t.numel() * t.element_size()
                 for t in (*self.k_cache.values(), *self.v_cache.values()))
        recurrent = sum(t.numel() * t.element_size()
                        for t in (*self.states.values(), *self.conv_states.values()))
        return {"kv": kv, "recurrent": recurrent, "total": kv + recurrent}
