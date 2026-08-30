"""Model geometry, and the memory arithmetic that falls out of it.

Qwen3.5-family models are hybrid: most layers use gated-delta linear attention
with a fixed-size recurrent state, and every `full_attention_interval`-th layer
uses ordinary grouped-query attention with a KV cache that grows with the
sequence. Nearly every interesting property of the engine follows from that
split, so this module makes it easy to ask questions about it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BYTES_PER_DTYPE = {"bfloat16": 2, "float16": 2, "float32": 4, "float8": 1, "int8": 1}


@dataclass
class ModelConfig:
    """The subset of the HF config the engine actually needs."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    layer_types: list[str]
    full_attention_interval: int
    attn_output_gate: bool
    # Linear-attention (gated delta) geometry
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    max_position_embeddings: int
    dtype: str = "bfloat16"
    tie_word_embeddings: bool = False
    mrope_section: list[int] = field(default_factory=list)
    mrope_interleaved: bool = True
    raw: dict[str, Any] = field(default_factory=dict)

    # --- construction -------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        text = raw.get("text_config", raw)
        rope = text.get("rope_parameters", {})
        layer_types = text.get("layer_types") or []
        interval = text.get("full_attention_interval", 4)
        if not layer_types:
            layer_types = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(text["num_hidden_layers"])
            ]
        return cls(
            hidden_size=text["hidden_size"],
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text.get("head_dim", text["hidden_size"] // text["num_attention_heads"]),
            intermediate_size=text["intermediate_size"],
            vocab_size=text["vocab_size"],
            rms_norm_eps=text.get("rms_norm_eps", 1e-6),
            rope_theta=rope.get("rope_theta", text.get("rope_theta", 10_000.0)),
            partial_rotary_factor=text.get(
                "partial_rotary_factor", rope.get("partial_rotary_factor", 1.0)
            ),
            layer_types=layer_types,
            full_attention_interval=interval,
            attn_output_gate=text.get("attn_output_gate", False),
            linear_num_key_heads=text.get("linear_num_key_heads", 0),
            linear_num_value_heads=text.get("linear_num_value_heads", 0),
            linear_key_head_dim=text.get("linear_key_head_dim", 0),
            linear_value_head_dim=text.get("linear_value_head_dim", 0),
            linear_conv_kernel_dim=text.get("linear_conv_kernel_dim", 4),
            max_position_embeddings=text.get("max_position_embeddings", 32768),
            dtype=text.get("dtype", text.get("torch_dtype", "bfloat16")),
            tie_word_embeddings=text.get("tie_word_embeddings", False),
            mrope_section=rope.get("mrope_section", []),
            mrope_interleaved=rope.get("mrope_interleaved", True),
            raw=raw,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "ModelConfig":
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        return cls.from_dict(json.loads(path.read_text()))

    # --- derived geometry ---------------------------------------------------

    @property
    def bytes_per_element(self) -> int:
        return BYTES_PER_DTYPE.get(self.dtype, 2)

    @property
    def rotary_dim(self) -> int:
        """RoPE touches only the first `rotary_dim` channels of each head."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def full_attention_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @property
    def linear_attention_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t != "full_attention"]

    @property
    def num_full_attention_layers(self) -> int:
        return len(self.full_attention_layers)

    @property
    def num_linear_attention_layers(self) -> int:
        return len(self.linear_attention_layers)

    @property
    def gqa_group_size(self) -> int:
        """How many query heads share one KV head."""
        return self.num_attention_heads // self.num_key_value_heads

    # --- memory arithmetic --------------------------------------------------

    def kv_bytes_per_token(self) -> int:
        """KV cache cost of one token, summed over the full-attention layers.

        Two tensors (K and V), `num_key_value_heads` of them, `head_dim` wide.
        The linear-attention layers contribute nothing here: their state does
        not grow with sequence length.
        """
        per_layer = 2 * self.num_key_value_heads * self.head_dim * self.bytes_per_element
        return per_layer * self.num_full_attention_layers

    def recurrent_state_bytes_per_sequence(self, state_dtype_bytes: int = 4) -> int:
        """Fixed cost of the linear-attention state for one sequence.

        The delta-rule state is (value heads, key dim, value dim) per layer and
        is kept in float32 for numerical stability, plus a small causal-conv
        window of `linear_conv_kernel_dim` steps.
        """
        recurrent = (
            self.linear_num_value_heads
            * self.linear_key_head_dim
            * self.linear_value_head_dim
            * state_dtype_bytes
        )
        conv_channels = (
            self.linear_num_key_heads * self.linear_key_head_dim * 2
            + self.linear_num_value_heads * self.linear_value_head_dim
        )
        conv = conv_channels * self.linear_conv_kernel_dim * self.bytes_per_element
        return (recurrent + conv) * self.num_linear_attention_layers

    def cache_bytes(self, seq_len: int, batch: int = 1) -> int:
        """Total cache footprint for `batch` sequences of `seq_len` tokens."""
        return batch * (
            self.recurrent_state_bytes_per_sequence() + self.kv_bytes_per_token() * seq_len
        )

    def dense_kv_bytes_per_token(self) -> int:
        """What the KV cache would cost if every layer were full attention.

        This is the number to compare against; the gap is what the hybrid
        design buys you.
        """
        return (
            2
            * self.num_key_value_heads
            * self.head_dim
            * self.bytes_per_element
            * self.num_hidden_layers
        )

    def hybrid_breakeven_tokens(self) -> float:
        """Context length past which the hybrid cache beats an all-full one.

        Below this length the fixed recurrent state costs more than the KV
        entries it replaces; above it, the hybrid wins and keeps winning.
        """
        saved_per_token = self.dense_kv_bytes_per_token() - self.kv_bytes_per_token()
        if saved_per_token <= 0:
            return float("inf")
        return self.recurrent_state_bytes_per_sequence() / saved_per_token

    # --- parameter counting -------------------------------------------------

    def param_counts(self) -> dict[str, int]:
        """Parameters by component, for the language model only."""
        h, d = self.hidden_size, self.head_dim
        embed = self.vocab_size * h
        lm_head = 0 if self.tie_word_embeddings else self.vocab_size * h

        q_out = self.num_attention_heads * d * (2 if self.attn_output_gate else 1)
        attn = (
            h * q_out
            + 2 * h * self.num_key_value_heads * d
            + self.num_attention_heads * d * h
        )
        mlp = 3 * h * self.intermediate_size

        k_total = self.linear_num_key_heads * self.linear_key_head_dim
        v_total = self.linear_num_value_heads * self.linear_value_head_dim
        linear_attn = (
            h * (2 * k_total + v_total)          # q, k, v projections
            + (2 * k_total + v_total) * self.linear_conv_kernel_dim  # depthwise conv
            + h * self.linear_num_value_heads * 2  # per-head decay and beta gates
            + h * v_total                          # output gate
            + v_total * h                          # output projection
        )

        full = self.num_full_attention_layers * (attn + mlp)
        linear = self.num_linear_attention_layers * (linear_attn + mlp)
        norms = self.num_hidden_layers * 2 * h + h

        total = embed + lm_head + full + linear + norms
        return {
            "embedding": embed,
            "lm_head": lm_head,
            "full_attention_layers": full,
            "linear_attention_layers": linear,
            "norms": norms,
            "total": total,
        }

    def weight_bytes(self, bits: int | None = None) -> int:
        per_param = self.bytes_per_element if bits is None else bits / 8
        return int(self.param_counts()["total"] * per_param)

    def summary(self) -> dict[str, Any]:
        counts = self.param_counts()
        return {
            "layers": self.num_hidden_layers,
            "full_attention_layers": self.num_full_attention_layers,
            "linear_attention_layers": self.num_linear_attention_layers,
            "hidden_size": self.hidden_size,
            "head_dim": self.head_dim,
            "gqa_group_size": self.gqa_group_size,
            "rotary_dim": self.rotary_dim,
            "params_billions": round(counts["total"] / 1e9, 2),
            "weights_gb_bf16": round(self.weight_bytes() / 1024**3, 1),
            "kv_kb_per_token": round(self.kv_bytes_per_token() / 1024, 2),
            "dense_kv_kb_per_token": round(self.dense_kv_bytes_per_token() / 1024, 2),
            "recurrent_state_mb": round(
                self.recurrent_state_bytes_per_sequence() / 1024**2, 1
            ),
            "hybrid_breakeven_tokens": round(self.hybrid_breakeven_tokens()),
        }
