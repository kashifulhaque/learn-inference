def num_full_attention_layers(config: dict) -> int:
    interval = config["full_attention_interval"]
    return sum(
        1 for i in range(config["num_hidden_layers"]) if (i + 1) % interval == 0
    )


def mlp_params_per_layer(config: dict) -> int:
    return 3 * config["hidden_size"] * config["intermediate_size"]


def embedding_params(config: dict) -> int:
    return 2 * config["vocab_size"] * config["hidden_size"]


def kv_bytes_per_token(config: dict) -> int:
    per_layer = (
        2 * config["num_key_value_heads"] * config["head_dim"]
        * config["bytes_per_element"]
    )
    return per_layer * num_full_attention_layers(config)


def dense_kv_bytes_per_token(config: dict) -> int:
    per_layer = (
        2 * config["num_key_value_heads"] * config["head_dim"]
        * config["bytes_per_element"]
    )
    return per_layer * config["num_hidden_layers"]


def recurrent_state_bytes(config: dict, state_dtype_bytes: int = 4) -> int:
    layers = config["num_hidden_layers"] - num_full_attention_layers(config)
    recurrent = (
        config["linear_num_value_heads"]
        * config["linear_key_head_dim"]
        * config["linear_value_head_dim"]
        * state_dtype_bytes
    )
    conv_channels = (
        2 * config["linear_num_key_heads"] * config["linear_key_head_dim"]
        + config["linear_num_value_heads"] * config["linear_value_head_dim"]
    )
    conv = conv_channels * config["linear_conv_kernel_dim"] * config["bytes_per_element"]
    return (recurrent + conv) * layers


def breakeven_tokens(config: dict) -> float:
    saving = dense_kv_bytes_per_token(config) - kv_bytes_per_token(config)
    if saving <= 0:
        return float("inf")
    return recurrent_state_bytes(config) / saving


def max_batch_size(config: dict, gpu_bytes: int, weight_bytes: int,
                   context_len: int, overhead_bytes: int = 6 * 1024**3) -> int:
    available = gpu_bytes - weight_bytes - overhead_bytes
    per_sequence = recurrent_state_bytes(config) + kv_bytes_per_token(config) * context_len
    if available <= 0 or per_sequence <= 0:
        return 0
    return max(0, available // per_sequence)
