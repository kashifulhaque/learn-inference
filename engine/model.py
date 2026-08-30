"""The 64-layer hybrid stack.

Every layer is pre-norm with a residual connection:

    x = x + mixer(norm1(x))
    x = x + mlp(norm2(x))

The only thing that changes down the stack is which mixer a layer uses. Layers
3, 7, 11, ... are full attention; the rest are gated delta linear attention.
The MLP is identical everywhere.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .cache import HybridCache
from .config import ModelConfig
from .layers.attention import GatedGroupedQueryAttention
from .layers.linear_attn import GatedDeltaNet
from .layers.mlp import SwiGLU
from .layers.rmsnorm import RMSNorm
from .layers.rope import RotaryEmbedding


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_full_attention = config.layer_types[layer_idx] == "full_attention"

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if self.is_full_attention:
            self.mixer = GatedGroupedQueryAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                rotary_dim=config.rotary_dim,
                output_gate=config.attn_output_gate,
                eps=config.rms_norm_eps,
            )
        else:
            self.mixer = GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=config.linear_num_key_heads,
                num_v_heads=config.linear_num_value_heads,
                k_head_dim=config.linear_key_head_dim,
                v_head_dim=config.linear_value_head_dim,
                conv_kernel=config.linear_conv_kernel_dim,
                eps=config.rms_norm_eps,
            )

        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        x: Tensor,
        cos: Tensor | None,
        sin: Tensor | None,
        cache: HybridCache | None = None,
        attn_impl: str = "sdpa",
    ) -> Tensor:
        residual = x
        hidden = self.input_layernorm(x)
        if self.is_full_attention:
            hidden = self.mixer(
                hidden, cos, sin, kv_cache=cache, layer_idx=self.layer_idx,
                attn_impl=attn_impl,
            )
        else:
            hidden = self.mixer(hidden, cache=cache, layer_idx=self.layer_idx)
        x = residual + hidden

        return x + self.mlp(self.post_attention_layernorm(x))


class HybridLanguageModel(nn.Module):
    """Text-only path. The vision tower is out of scope for this curriculum."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary = RotaryEmbedding(
            config.rotary_dim, min(config.max_position_embeddings, 32768), config.rope_theta
        )

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor | None = None,
        cache: HybridCache | None = None,
        attn_impl: str = "sdpa",
        last_token_only: bool = False,
    ) -> Tensor:
        batch, seq = input_ids.shape
        if positions is None:
            start = cache.length if cache is not None else 0
            positions = torch.arange(start, start + seq, device=input_ids.device)

        cos, sin = self.rotary(positions)
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x, cos, sin, cache=cache, attn_impl=attn_impl)

        if cache is not None:
            cache.advance(seq)

        x = self.norm(x)
        if last_token_only:
            # During decode only the final position matters, and the lm_head is
            # a 5120 x 248320 matrix. Slicing first turns a 248320-wide GEMM
            # over every position into one over a single row.
            x = x[:, -1:, :]
        return self.lm_head(x)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 64,
        sampler=None,
        eos_token_ids: set[int] | None = None,
        max_seq_len: int | None = None,
    ) -> Tensor:
        """A single-sequence generation loop: prefill once, then step."""
        from .sampling import SamplingParams, sample

        device = input_ids.device
        sampler = sampler or SamplingParams(temperature=0.0)
        eos_token_ids = eos_token_ids or set()

        cache = HybridCache(
            self.config,
            batch_size=input_ids.shape[0],
            max_seq_len=max_seq_len or input_ids.shape[1] + max_new_tokens + 8,
            device=device,
            dtype=self.embed_tokens.weight.dtype,
        )

        logits = self(input_ids, cache=cache, last_token_only=True)
        produced = []
        for _ in range(max_new_tokens):
            next_token = sample(logits[:, -1, :], sampler)
            produced.append(next_token)
            if next_token.numel() == 1 and int(next_token) in eos_token_ids:
                break
            logits = self(next_token.view(-1, 1), cache=cache, last_token_only=True)

        return torch.cat([input_ids, torch.cat(produced, dim=-1)], dim=-1)
