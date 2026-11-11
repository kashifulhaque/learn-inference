"""Reference solution for the capstone lab: one engine, wired end to end.

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
    """Recurrent state for the linear-attention layers, one slot per sequence."""

    def __init__(self, config: ModelConfig, num_slots: int, device, dtype) -> None:
        conv_channels = (
            2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim
        )
        self.num_slots = num_slots
        self.states = {
            layer: torch.zeros(
                num_slots, config.linear_num_value_heads, config.linear_key_head_dim,
                config.linear_value_head_dim, device=device, dtype=torch.float32,
            )
            for layer in config.linear_attention_layers
        }
        self.convs = {
            layer: torch.zeros(
                num_slots, conv_channels, config.linear_conv_kernel_dim - 1,
                device=device, dtype=dtype,
            )
            for layer in config.linear_attention_layers
        }
        self._free = list(range(num_slots))
        self._held: dict[str, int] = {}

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def slot_of(self, seq_id: str) -> int:
        return self._held[seq_id]

    def alloc(self, seq_id: str) -> int:
        """Hand out a slot and zero it. A reused slot holds the last owner's state."""
        if seq_id in self._held:
            raise RuntimeError(f"{seq_id} already holds slot {self._held[seq_id]}")
        if not self._free:
            raise RuntimeError("no free recurrent-state slot")
        slot = self._free.pop()
        for layer in self.states:
            self.states[layer][slot].zero_()
            self.convs[layer][slot].zero_()
        self._held[seq_id] = slot
        return slot

    def release(self, seq_id: str) -> None:
        slot = self._held.pop(seq_id, None)
        if slot is not None:
            self._free.append(slot)


class SeqCache:
    """The `HybridCache` interface for one sequence, backed by shared pools.

    The model calls `append` from each full-attention layer and the
    recurrent-state methods from each linear layer, then `advance` once at the
    end of the forward pass.
    """

    def __init__(self, paged: PagedKVCache, pool: StatePool, seq_id: str) -> None:
        self.paged = paged
        self.pool = pool
        self.seq_id = seq_id
        self.slot = pool.slot_of(seq_id)
        self.length = 0

    def append(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write K and V of shape (1, kv_heads, s, d), return the whole prefix."""
        s = k.shape[2]
        cfg = self.paged.config
        table = self.paged.tables.get(self.seq_id)
        if table is None or table.length < self.length + s:
            raise RuntimeError(
                f"{self.seq_id}: writing {s} tokens at {self.length} but the block "
                f"table covers {0 if table is None else table.length}"
            )
        positions = torch.arange(self.length, self.length + s, device=k.device)
        self.paged.write(layer_idx, self.seq_id, positions, k[0].transpose(0, 1), v[0].transpose(0, 1))
        upto = torch.arange(self.length + s, device=k.device)
        slots = self.paged.slot_indices(self.seq_id, upto)
        kb = self.paged.k_blocks[layer_idx].view(-1, cfg.num_key_value_heads, cfg.head_dim)
        vb = self.paged.v_blocks[layer_idx].view(-1, cfg.num_key_value_heads, cfg.head_dim)
        return (
            kb[slots].transpose(0, 1).unsqueeze(0).to(k.dtype),
            vb[slots].transpose(0, 1).unsqueeze(0).to(v.dtype),
        )

    def recurrent_state(self, layer_idx: int) -> Tensor:
        return self.pool.states[layer_idx][self.slot : self.slot + 1].clone()

    def conv_state(self, layer_idx: int) -> Tensor:
        return self.pool.convs[layer_idx][self.slot : self.slot + 1].clone()

    def set_linear_state(self, layer_idx: int, state: Tensor, conv: Tensor) -> None:
        self.pool.states[layer_idx][self.slot] = state[0]
        self.pool.convs[layer_idx][self.slot] = conv[0]

    def advance(self, tokens: int) -> None:
        self.length += tokens


class Engine:
    """Scheduler, paged cache, state slots, model, and sampler in one loop."""

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
        param = next(model.parameters())
        self.model = model
        self.device = param.device
        self.config = model.config
        self.eos = set(eos_ids or ())
        # "naive" builds its causal mask from kv_len - q_len, so a chunk that
        # follows a cached prefix sees the prefix. SDPA's is_causal aligns the
        # mask to the top-left corner and hides it.
        self.attn_impl = attn_impl
        self.paged = PagedKVCache(
            self.config, num_blocks, block_size, device=param.device, dtype=param.dtype
        )
        self.states = StatePool(self.config, max_batch_size, param.device, param.dtype)
        self.scheduler = Scheduler(
            self.paged, max_batch_size=max_batch_size,
            max_batched_tokens=max_batched_tokens, chunk_size=chunk_size, watermark=0.0,
        )
        self.seqs: dict[str, SeqCache] = {}
        self.requests: dict[str, Request] = {}
        self.params: dict[str, SamplingParams] = {}
        self.generators: dict[str, torch.Generator] = {}
        self.preemptions = 0
        self.steps = 0
        self._count = 0

    # --- pools --------------------------------------------------------------

    @property
    def free_blocks(self) -> int:
        return self.paged.free_blocks

    @property
    def free_slots(self) -> int:
        return self.states.free_slots

    @property
    def has_work(self) -> bool:
        return self.scheduler.has_work

    # --- requests -----------------------------------------------------------

    def add_request(
        self, prompt_ids: list[int], max_new_tokens: int,
        params: SamplingParams | None = None,
    ) -> Request:
        request = Request(
            prompt_ids=list(prompt_ids), max_new_tokens=max_new_tokens, id=f"r{self._count}"
        )
        self._count += 1
        self.requests[request.id] = request
        self.params[request.id] = params or SamplingParams(temperature=0.0)
        self.scheduler.add(request)
        return request

    def _generator(self, req_id: str) -> torch.Generator:
        """One generator per request, seeded from its own params."""
        if req_id not in self.generators:
            seed = self.params[req_id].seed
            self.generators[req_id] = torch.Generator(device=self.device).manual_seed(
                0 if seed is None else seed
            )
        return self.generators[req_id]

    def _drop(self, req_id: str) -> None:
        """Forget everything the engine holds for a request: slot, view, generator."""
        self.states.release(req_id)
        self.seqs.pop(req_id, None)
        self.generators.pop(req_id, None)

    def _start(self, req_id: str) -> SeqCache:
        self.states.alloc(req_id)
        self.seqs[req_id] = SeqCache(self.paged, self.states, req_id)
        return self.seqs[req_id]

    # --- one step -----------------------------------------------------------

    def step(self) -> dict:
        """Run one scheduled batch. Returns what ran, for the simulated clock."""
        batch = self.scheduler.step()
        self.steps += 1
        live = {r.id for r in self.scheduler.running}
        preempted: list[str] = []

        # A slot lives exactly as long as its request is in the running set.
        for rid in list(self.seqs):
            if rid not in live:
                if self.requests[rid].state is State.PREEMPTED:
                    preempted.append(rid)
                self._drop(rid)

        # The scheduler can preempt a victim and re-admit it inside one step.
        # That request is live again, but its cache was released. A request that
        # starts prefill at position 0 always gets a fresh slot and generator.
        prefill = [(r, c) for r, c in batch.prefill if r.id in live]
        for request, _ in prefill:
            if request.prefilled == 0 and request.id in self.seqs:
                preempted.append(request.id)
                self._drop(request.id)
        # A victim picked after it joined the decode list is still in
        # batch.decode, with its output already cleared. Skip it.
        decode = [
            r for r in batch.decode
            if r.id in live and not r.needs_prefill and r.output_ids
        ]
        self.preemptions += len(preempted)

        emitted: list[str] = []
        info = {"prefill": [], "decode": [], "emitted": emitted, "preempted": preempted}

        with torch.inference_mode():
            for request in decode:
                seq = self.seqs[request.id]
                expected = request.prompt_len + len(request.output_ids) - 1
                if seq.length != expected:
                    raise RuntimeError(
                        f"{request.id}: cache holds {seq.length} tokens, expected {expected}"
                    )
                ids = torch.tensor([[request.output_ids[-1]]], device=self.device)
                logits = self.model(ids, cache=seq, attn_impl=self.attn_impl, last_token_only=True)
                self._commit(request, logits[:, -1, :])
                info["decode"].append((request.id, seq.length))
                emitted.append(request.id)

            for request, chunk in prefill:
                seq = self.seqs.get(request.id) or self._start(request.id)
                if seq.length != request.prefilled:
                    raise RuntimeError(
                        f"{request.id}: cache holds {seq.length} tokens, chunk starts at "
                        f"{request.prefilled}"
                    )
                start = request.prefilled
                ids = torch.tensor([request.prompt_ids[start : start + chunk]], device=self.device)
                logits = self.model(ids, cache=seq, attn_impl=self.attn_impl, last_token_only=True)
                self.scheduler.commit_prefill(request, chunk)
                info["prefill"].append((request.id, chunk))
                if not request.needs_prefill:
                    self._commit(request, logits[:, -1, :])
                    emitted.append(request.id)
        return info

    def _commit(self, request: Request, logits: Tensor) -> None:
        params = self.params[request.id]
        generator = None if params.greedy else self._generator(request.id)
        token = int(sample(logits, params, generator=generator))
        self.scheduler.commit_token(request, token, self.eos)

    def generate(
        self, prompts: list[list[int]], max_new_tokens: int,
        params: list[SamplingParams] | None = None, max_steps: int = 5000,
    ) -> list[list[int]]:
        """Run every prompt to completion and return the generated ids in order."""
        requests = [
            self.add_request(p, max_new_tokens, None if params is None else params[i])
            for i, p in enumerate(prompts)
        ]
        for _ in range(max_steps):
            if not self.has_work:
                break
            self.step()
        else:
            raise RuntimeError(f"still working after {max_steps} steps: preemption livelock?")
        return [list(r.output_ids) for r in requests]

    def audit(self) -> list[str]:
        """Return every way the pools and the request state disagree. Empty is healthy."""
        problems = []
        held = [b for t in self.paged.tables.values() for b in t.blocks]
        if len(set(self.paged.free)) != len(self.paged.free):
            problems.append("a block is on the free list twice")
        if set(held) & set(self.paged.free):
            problems.append("a block is both held and free")
        if len(held) + len(self.paged.free) != self.paged.num_blocks:
            problems.append(
                f"{len(held)} held + {len(self.paged.free)} free != {self.paged.num_blocks} blocks"
            )
        running = {r.id for r in self.scheduler.running}
        for rid, seq in self.seqs.items():
            if rid not in running:
                problems.append(f"{rid} holds a slot but is not running")
                continue
            table = self.paged.tables.get(rid)
            if table is None or table.length != seq.length:
                problems.append(
                    f"{rid}: block table covers {None if table is None else table.length}, "
                    f"state view holds {seq.length}"
                )
        return problems


# ---------------------------------------------------------------------------
# Simulated clock
# ---------------------------------------------------------------------------

A100 = {
    "bf16_tflops": 312.0, "hbm_gbs": 1935.0, "launch_us": 5.0, "launches": 400,
    "allreduce_us": 8.0, "link_gbs": 600.0, "memory_bytes": 80 * 1024**3, "tp": 1,
}


def step_seconds(tm, hw: dict, n_decode: int, ctx_sum: float, chunks: list[int]) -> float:
    """Seconds one mixed step takes under chapter 17a's model.

    One step reads the weights once, however many sequences ride in it. Add the
    decode batch's FLOPs and bytes, then each prefill chunk's FLOPs and its
    bytes beyond the weights, and take the longer of compute and memory plus the
    launch floor. A pure decode step equals `tm.decode_step_time`, and one lone
    chunk equals `tm.prefill_time`.
    """
    kind = "hybrid"
    flops = 0.0
    if n_decode:
        ctx = ctx_sum / n_decode
        flops += tm.decode_flops(kind, n_decode, ctx)
        nbytes = tm.decode_traffic_bytes(kind, n_decode, ctx, 1)
    else:
        nbytes = tm._weight_bytes(kind)
    for chunk in chunks:
        flops += tm.prefill_flops(kind, chunk)
        nbytes += tm.prefill_traffic_bytes(kind, chunk, 1) - tm._weight_bytes(kind)
    compute = flops / (hw["bf16_tflops"] * 1e12)
    memory = nbytes / (hw["hbm_gbs"] * 1e9)
    return max(compute, memory) + tm.launch_seconds(hw)


def run_trace(engine: Engine, trace: list[dict], tm, hw: dict) -> dict:
    """Replay arrivals on a simulated clock and record each request's times.

    Each trace row has "arrival" (seconds), "prompt", and "max_new". The clock
    advances by `step_seconds` for the batch that ran. When nothing is running
    it jumps to the next arrival. `log` holds one row per step, for the chapter's figure. A request's first-token time is the first
    time it ever emitted, and its finish time is when its last token lands.
    """
    pending = sorted(range(len(trace)), key=lambda i: (trace[i]["arrival"], i))
    clock = 0.0
    records: dict[str, dict] = {}
    ids: dict[int, str] = {}
    cursor = 0
    log: list[dict] = []
    while cursor < len(pending) or engine.has_work:
        while cursor < len(pending) and trace[pending[cursor]]["arrival"] <= clock:
            i = pending[cursor]
            row = trace[i]
            req = engine.add_request(row["prompt"], row["max_new"])
            ids[i] = req.id
            records[req.id] = {"arrival": row["arrival"], "first": None, "finish": None,
                               "tokens": row["max_new"]}
            cursor += 1
        if not engine.has_work:
            clock = max(clock, trace[pending[cursor]]["arrival"])
            continue
        info = engine.step()
        ctx_sum = sum(ctx for _, ctx in info["decode"])
        dt = step_seconds(
            tm, hw, len(info["decode"]), ctx_sum, [c for _, c in info["prefill"]]
        )
        clock += dt
        log.append({"t": clock, "dt": dt, "decode": len(info["decode"]),
                    "prefill": sum(c for _, c in info["prefill"]),
                    "preempted": len(info["preempted"])})
        for rid in info["emitted"]:
            rec = records[rid]
            if rec["first"] is None:
                rec["first"] = clock
            if len(engine.requests[rid].output_ids) >= rec["tokens"]:
                rec["finish"] = clock
    return {"records": records, "makespan": clock, "steps": engine.steps,
            "preemptions": engine.preemptions, "log": log}


def goodput(records: dict, makespan: float, slo_ttft: float, slo_tpot: float) -> float:
    """Output tokens per second from requests that met both SLOs.

    TTFT is first token minus arrival. TPOT is (finish - first) over the n - 1
    gaps, and a one-token request has no gap, so it meets any TPOT.
    """
    good = 0
    for rec in records.values():
        if rec["finish"] is None:
            continue
        ttft = rec["first"] - rec["arrival"]
        n = rec["tokens"]
        tpot = (rec["finish"] - rec["first"]) / (n - 1) if n > 1 else 0.0
        if ttft <= slo_ttft and tpot <= slo_tpot:
            good += n
    return good / makespan
