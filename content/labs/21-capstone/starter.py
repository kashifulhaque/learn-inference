"""Lab 21: assemble the engine end to end.

The pieces are the repository's own `engine/` package: the hybrid model, the
paged KV cache, the scheduler, and the sampler. This file adds the three things
no isolated lab builds, which are the glue between them:

* `StatePool`: one fixed-size recurrent-state slot per running request.
* `SeqCache`: the cache interface the model expects, backed by paged KV blocks
  and one state slot, for a single sequence.
* `Engine`: schedule, forward, sample, and commit, with preemption by
  recompute that restores every piece of per-request state.

`step_seconds` and `run_trace` put the engine on a simulated clock driven by
chapter 17a's step-time model. Nothing here is a GPU measurement.
"""

from __future__ import annotations

import torch
from torch import Tensor

from engine.cache import PagedKVCache
from engine.config import ModelConfig
from engine.model import HybridLanguageModel
from engine.sampling import SamplingParams, sample
from engine.scheduler import Request, Scheduler, State

# The hybrid recipe at toy size: 8 layers, every 4th full attention (layers 3
# and 7), the other six gated delta. Same layer_types rule as the 64-layer model.
TINY = {
    "text_config": {
        "hidden_size": 48, "num_hidden_layers": 8, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 96,
        "vocab_size": 128, "rms_norm_eps": 1e-6, "partial_rotary_factor": 0.5,
        "full_attention_interval": 4, "attn_output_gate": True,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        "linear_key_head_dim": 8, "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4, "max_position_embeddings": 512,
        "dtype": "float32",
        "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5},
    }
}


def build_tiny_model(seed: int = 21) -> HybridLanguageModel:
    """A seeded, randomly initialised tiny model.

    The engine's constructors zero the causal-conv weights and `A_log`, which
    makes every linear layer a no-op. A model like that cannot tell a restored
    recurrent state from a lost one, so this sets them to values with memory:
    decay near 0.9, and conv taps that mix the window.
    """
    torch.manual_seed(seed)
    config = ModelConfig.from_dict(TINY)
    model = HybridLanguageModel(config).eval()
    with torch.no_grad():
        for layer in model.layers:
            if layer.is_full_attention:
                continue
            mixer = layer.mixer
            mixer.conv_weight.copy_(torch.randn_like(mixer.conv_weight) * 0.6)
            mixer.A_log.copy_(torch.randn_like(mixer.A_log) * 0.3 - 2.0)
            mixer.dt_bias.copy_(torch.randn_like(mixer.dt_bias) * 0.3)
            mixer.norm_weight.copy_(1.0 + 0.2 * torch.randn_like(mixer.norm_weight))
    return model


def load_hf_weights(model: HybridLanguageModel, model_dir) -> None:
    """Copy a Hugging Face Qwen3.5-family checkpoint into the engine's layout.

    NOT VERIFIED. This mapping follows the layout of `modeling_qwen3_5.py` in
    transformers and has never run against real weights. The differences it
    absorbs:

    * Decoder, q, k, and final RMSNorms are zero-centred in the checkpoint
      (`x * (1 + w)`). The engine multiplies by `w`, so store `1 + w`. The
      gated norm inside the linear layer is not zero-centred.
    * The full-attention `q_proj` interleaves each head's query and gate. The
      engine expects all queries, then all gates.
    * The linear layer stores q/k/v and z, and b and a, as separate matrices.
      The engine fuses them as `in_proj_qkvz` and `in_proj_ba`.
    * The depthwise conv weight is (channels, 1, kernel). The engine's is
      (channels, kernel).
    * A tied checkpoint has no `lm_head.weight`.
    """
    from engine.weights import WeightIndex

    index = WeightIndex(model_dir)
    prefix = "model.language_model." if any(
        n.startswith("model.language_model.") for n in index.map
    ) else "model."
    param = next(model.parameters())
    cfg = model.config

    def get(name: str) -> Tensor:
        return index.get(prefix + name, device=param.device, dtype=param.dtype)

    def put(target: Tensor, value: Tensor) -> None:
        if target.shape != value.shape:
            raise ValueError(f"shape {tuple(value.shape)} does not fit {tuple(target.shape)}")
        target.copy_(value)

    with torch.no_grad():
        put(model.embed_tokens.weight, get("embed_tokens.weight"))
        if "lm_head.weight" in index:
            put(model.lm_head.weight, index.get("lm_head.weight", param.device, param.dtype))
        else:
            put(model.lm_head.weight, get("embed_tokens.weight"))
        put(model.norm.weight, 1.0 + get("norm.weight"))
        for i, layer in enumerate(model.layers):
            base = f"layers.{i}."
            put(layer.input_layernorm.weight, 1.0 + get(base + "input_layernorm.weight"))
            put(layer.post_attention_layernorm.weight,
                1.0 + get(base + "post_attention_layernorm.weight"))
            for name in ("gate_proj", "up_proj", "down_proj"):
                put(getattr(layer.mlp, name).weight, get(base + f"mlp.{name}.weight"))
            mixer = layer.mixer
            if layer.is_full_attention:
                a = base + "self_attn."
                q = get(a + "q_proj.weight").view(
                    cfg.num_attention_heads, 2, cfg.head_dim, cfg.hidden_size
                )
                width = cfg.num_attention_heads * cfg.head_dim
                put(mixer.q_proj.weight,
                    torch.cat([q[:, 0].reshape(width, -1), q[:, 1].reshape(width, -1)]))
                for name in ("k_proj", "v_proj", "o_proj"):
                    put(getattr(mixer, name).weight, get(a + f"{name}.weight"))
                put(mixer.q_norm.weight, 1.0 + get(a + "q_norm.weight"))
                put(mixer.k_norm.weight, 1.0 + get(a + "k_norm.weight"))
            else:
                a = base + "linear_attn."
                put(mixer.in_proj_qkvz.weight,
                    torch.cat([get(a + "in_proj_qkv.weight"), get(a + "in_proj_z.weight")]))
                put(mixer.in_proj_ba.weight,
                    torch.cat([get(a + "in_proj_b.weight"), get(a + "in_proj_a.weight")]))
                put(mixer.conv_weight, get(a + "conv1d.weight").squeeze(1))
                put(mixer.A_log, get(a + "A_log"))
                put(mixer.dt_bias, get(a + "dt_bias"))
                put(mixer.norm_weight, get(a + "norm.weight"))
                put(mixer.out_proj.weight, get(a + "out_proj.weight"))


class StatePool:
    """Recurrent state for the linear-attention layers, one slot per sequence.

    `states[layer]` is (num_slots, value_heads, k_dim, v_dim) in float32 and
    `convs[layer]` is (num_slots, channels, kernel - 1), for each linear layer.
    """

    def __init__(self, config: ModelConfig, num_slots: int, device, dtype) -> None:
        # TODO: allocate the two dicts and a free list.
        raise NotImplementedError

    @property
    def free_slots(self) -> int:
        raise NotImplementedError

    def slot_of(self, seq_id: str) -> int:
        raise NotImplementedError

    def alloc(self, seq_id: str) -> int:
        """Hand out a slot, zeroed. A freed slot still holds its last owner's state."""
        raise NotImplementedError

    def release(self, seq_id: str) -> None:
        """Return the slot. Releasing an id that holds none does nothing."""
        raise NotImplementedError


class SeqCache:
    """The `HybridCache` interface for one sequence, backed by shared pools.

    The model calls `append` from each full-attention layer, the recurrent-state
    methods from each linear layer, and `advance` once after all layers. It also
    reads `length` for the position of the next token. The scheduler has already
    reserved blocks in `paged.tables[seq_id]`; this class writes into them.
    """

    def __init__(self, paged: PagedKVCache, pool: StatePool, seq_id: str) -> None:
        raise NotImplementedError

    def append(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write K and V, shaped (1, kv_heads, s, d), at positions length..length+s-1.

        Return the whole prefix, positions 0..length+s-1, shaped (1, kv_heads, T, d).
        `PagedKVCache.write` and `slot_indices` do the block arithmetic.
        """
        raise NotImplementedError

    def recurrent_state(self, layer_idx: int) -> Tensor:
        """This sequence's state for one linear layer, shaped (1, heads, k_dim, v_dim)."""
        raise NotImplementedError

    def conv_state(self, layer_idx: int) -> Tensor:
        """This sequence's conv window, shaped (1, channels, kernel - 1)."""
        raise NotImplementedError

    def set_linear_state(self, layer_idx: int, state: Tensor, conv: Tensor) -> None:
        raise NotImplementedError

    def advance(self, tokens: int) -> None:
        raise NotImplementedError


class Engine:
    """Scheduler, paged cache, state slots, model, and sampler in one loop.

    Build a `PagedKVCache`, a `StatePool` with `max_batch_size` slots, and a
    `Scheduler` (watermark 0). Run each scheduled item as its own forward pass
    with a `SeqCache`. Use `attn_impl=self.attn_impl`, "naive" by default.
    """

    def __init__(
        self,
        model: HybridLanguageModel,
        *,
        num_blocks: int,
        block_size: int = 4,
        max_batch_size: int = 8,
        max_batched_tokens: int = 32,
        chunk_size: int = 8,
        eos_ids: set[int] | None = None,
        attn_impl: str = "naive",
    ) -> None:
        # Also keep: self.seqs (id -> SeqCache), self.requests, self.params,
        # self.generators, self.preemptions, self.steps.
        raise NotImplementedError

    @property
    def free_blocks(self) -> int:
        raise NotImplementedError

    @property
    def free_slots(self) -> int:
        raise NotImplementedError

    @property
    def has_work(self) -> bool:
        raise NotImplementedError

    def add_request(self, prompt_ids: list[int], max_new_tokens: int,
                    params: SamplingParams | None = None) -> Request:
        """Create a Request with id "r0", "r1", ... and queue it. Default params are greedy."""
        raise NotImplementedError

    def step(self) -> dict:
        """Run one scheduled batch and return what ran:

            {"prefill": [(id, chunk_tokens)], "decode": [(id, ctx_len_after)],
             "emitted": [ids that got a token], "preempted": [ids restarted]}

        Things this step has to get right, none of which an isolated lab sees:
        free the slot and generator of any request that left the running set;
        give a request that starts prefill at position 0 a fresh zeroed slot,
        even when the scheduler preempted and re-admitted it in this same step;
        skip decode items whose request was preempted after it joined the list;
        check each item against what its cache holds before you run it; and
        sample with one seeded generator per request.
        """
        raise NotImplementedError

    def generate(self, prompts: list[list[int]], max_new_tokens: int,
                 params: list[SamplingParams] | None = None,
                 max_steps: int = 5000) -> list[list[int]]:
        """Queue every prompt, step until the engine is idle, return each output list."""
        raise NotImplementedError

    def audit(self) -> list[str]:
        """List every disagreement between the pools and the requests; empty is healthy.

        Check: no block on the free list twice, none both held and free, held plus
        free equals num_blocks, and every state view belongs to a running request
        whose block-table length equals the view's length.
        """
        raise NotImplementedError


A100 = {
    "bf16_tflops": 312.0, "hbm_gbs": 1935.0, "launch_us": 5.0, "launches": 400,
    "allreduce_us": 8.0, "link_gbs": 600.0, "memory_bytes": 80 * 1024**3, "tp": 1,
}


def step_seconds(tm, hw: dict, n_decode: int, ctx_sum: float, chunks: list[int]) -> float:
    """Seconds one mixed step takes under chapter 17a's model.

    `tm` is the 17a solution module. One step reads the weights once however
    many sequences ride in it. Add the decode batch's FLOPs and bytes
    (`tm.decode_flops`, `tm.decode_traffic_bytes`, mean context ctx_sum/n_decode),
    then each chunk's FLOPs (`tm.prefill_flops`) and its bytes beyond the weights
    (`tm.prefill_traffic_bytes` minus `tm._weight_bytes`). With no decodes, start
    from `tm._weight_bytes("hybrid")`. Take max(flops/peak, bytes/bandwidth) plus
    `tm.launch_seconds(hw)`. Use kind "hybrid" and degree 1.
    """
    raise NotImplementedError


def run_trace(engine: Engine, trace: list[dict], tm, hw: dict) -> dict:
    """Replay arrivals on a simulated clock.

    Each row has "arrival", "prompt", "max_new". Add rows whose arrival is at or
    before the clock, step, and advance the clock by `step_seconds` for what ran.
    With nothing running, jump to the next arrival. Stamp a request's "first"
    time the first time it emits and its "finish" time when its last token lands.
    Return {"records": {id: {"arrival","first","finish","tokens"}}, "makespan",
    "steps", "preemptions"}.
    """
    raise NotImplementedError


def goodput(records: dict, makespan: float, slo_ttft: float, slo_tpot: float) -> float:
    """Output tokens per second from requests that met both SLOs.

    TTFT is first - arrival. TPOT is (finish - first) / (tokens - 1), and a
    one-token request meets any TPOT. Unfinished requests do not count.
    """
    raise NotImplementedError
