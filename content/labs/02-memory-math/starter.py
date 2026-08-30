"""Lab 02 — memory arithmetic.

Every function takes a config dict with these keys:

    hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads,
    head_dim, intermediate_size, vocab_size, full_attention_interval,
    attn_output_gate, linear_num_key_heads, linear_num_value_heads,
    linear_key_head_dim, linear_value_head_dim, linear_conv_kernel_dim,
    bytes_per_element
"""


def num_full_attention_layers(config: dict) -> int:
    """Count layers that use full attention.

    Full attention lands on every `full_attention_interval`-th layer, counting
    so that layer index 3 is the first one when the interval is 4.
    """
    # TODO
    raise NotImplementedError


def mlp_params_per_layer(config: dict) -> int:
    """Parameters in one SwiGLU block: gate, up, and down."""
    # TODO
    raise NotImplementedError


def embedding_params(config: dict) -> int:
    """Parameters in the embedding plus the untied output projection."""
    # TODO
    raise NotImplementedError


def kv_bytes_per_token(config: dict) -> int:
    """Bytes the KV cache grows by for one token, across all layers.

    Only full-attention layers contribute.
    """
    # TODO
    raise NotImplementedError


def dense_kv_bytes_per_token(config: dict) -> int:
    """What the same model would cost if every layer used full attention."""
    # TODO
    raise NotImplementedError


def recurrent_state_bytes(config: dict, state_dtype_bytes: int = 4) -> int:
    """Fixed linear-attention state for one sequence, across all such layers.

    Each layer holds a (value heads, key dim, value dim) matrix in float32, plus
    a causal convolution window of `linear_conv_kernel_dim` steps over the
    concatenated q, k, and v channels in the model's own dtype.
    """
    # TODO
    raise NotImplementedError


def breakeven_tokens(config: dict) -> float:
    """Context length past which the hybrid cache beats an all-full one."""
    # TODO
    raise NotImplementedError


def max_batch_size(config: dict, gpu_bytes: int, weight_bytes: int,
                   context_len: int, overhead_bytes: int = 6 * 1024**3) -> int:
    """Largest batch that fits, given weights, cache, and a fixed overhead.

    Returns 0 when even one sequence does not fit.
    """
    # TODO
    raise NotImplementedError
