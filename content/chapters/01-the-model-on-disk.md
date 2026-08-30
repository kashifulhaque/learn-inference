---
title: The model on disk
slug: 01-the-model-on-disk
part: "Part 1 — Ground truth"
summary: Safetensors, shards, and what the tensor names tell you about the architecture.
minutes: 45
gpu: false
objectives:
  - Read a safetensors index and locate any tensor without loading the model.
  - Infer a model's architecture from its tensor names alone.
  - Explain why memory mapping makes lazy, layer-by-layer loading possible.
lab: 01-inspect-weights
---

# The model on disk

Qwen3.8-27B ships as 18 safetensors shards, a JSON index, a tokenizer, and a
config. Together they're about 54 GB. Before you write a single layer, read what
the files actually contain: the tensor names are the most honest description of
the architecture you can get, and they often disagree with the model card.

## The safetensors format

A safetensors file is a header followed by raw tensor bytes. The header is JSON
that maps each tensor name to its dtype, shape, and a byte range in the rest of
the file. That layout has two useful properties.

Reading the header costs one small read. You can list every tensor, with shapes,
without touching the weight data.

Loading a tensor is a memory map, not a copy. The operating system pages in the
bytes you actually touch. This is what lets you move a model to the GPU layer by
layer while never holding more than one layer in host memory — the difference
between needing 54 GB of RAM and needing 2 GB.

Compare this with `torch.save`, which uses pickle: reading anything requires
deserializing arbitrary Python objects, so you must trust the file and you must
load it whole.

## Shards and the index

A single 54 GB file is awkward to download and impossible to resume, so the model
is split. `model.safetensors.index.json` maps each tensor name to the shard that
holds it:

```json
{
  "metadata": {"total_size": 53974302720},
  "weight_map": {
    "model.language_model.layers.0.mlp.gate_proj.weight": "model-00001-of-00018.safetensors",
    "model.language_model.layers.0.self_attn.q_proj.weight": "model-00001-of-00018.safetensors"
  }
}
```

To load a tensor, look up its shard, open that shard, and read the range. The
`WeightIndex` class in `engine/weights.py` does exactly this and caches the open
file handles.

## Reading the architecture from the names

Group the tensor names by layer and the model describes itself. In this model,
layer 0 and layer 3 hold different things:

```text
# Layer 0 — linear attention
model.language_model.layers.0.linear_attn.in_proj_qkvz.weight
model.language_model.layers.0.linear_attn.in_proj_ba.weight
model.language_model.layers.0.linear_attn.conv1d.weight
model.language_model.layers.0.linear_attn.A_log
model.language_model.layers.0.linear_attn.dt_bias
model.language_model.layers.0.linear_attn.norm.weight
model.language_model.layers.0.linear_attn.out_proj.weight

# Layer 3 — full attention
model.language_model.layers.3.self_attn.q_proj.weight
model.language_model.layers.3.self_attn.k_proj.weight
model.language_model.layers.3.self_attn.v_proj.weight
model.language_model.layers.3.self_attn.o_proj.weight
```

Three things follow immediately.

**The layers aren't uniform.** Some have `self_attn`, some have `linear_attn`.
Cross-check against `config.json`, where `layer_types` lists all 64 entries and
`full_attention_interval` is 4.

**`A_log` and `dt_bias` are state-space parameters.** A model with those is
running a recurrence, not attention, in those layers.

**The `q_proj` shape is a surprise.** With 24 heads of width 256, you'd expect
`[6144, 5120]`. The file says `[12288, 5120]` — exactly double. That's the output
gate: the projection emits queries and a gate side by side, and the layer splits
them. `config.json` confirms it with `attn_output_gate: true`. If you'd trusted
the shape arithmetic instead of reading the file, you'd have written a layer that
loads without error and produces nonsense.

## The config that matters

Qwen3.8-27B is multimodal, so `config.json` nests the language model under
`text_config` and puts a vision tower alongside it. This course uses the text
path only. `engine/config.py` handles both shapes:

```python
text = raw.get("text_config", raw)
```

The fields worth knowing before chapter 2:

| Field | Value | Why it matters |
|---|---|---|
| `hidden_size` | 5120 | Width of the residual stream. |
| `num_hidden_layers` | 64 | 16 full attention, 48 linear. |
| `num_attention_heads` | 24 | Query heads in full-attention layers. |
| `num_key_value_heads` | 4 | KV heads. Six queries share each one. |
| `head_dim` | 256 | Not `hidden_size / num_heads`. Read it, don't derive it. |
| `intermediate_size` | 17408 | MLP width. Most parameters live here. |
| `vocab_size` | 248320 | The embedding alone is 1.27B parameters. |
| `partial_rotary_factor` | 0.25 | Only 64 of 256 channels get rotated. |
| `linear_num_value_heads` | 48 | Value heads in the recurrent layers. |

`head_dim` is the one that catches people. 24 x 256 is 6144, not 5120, so the
attention block projects into a wider space than the residual stream and projects
back out. Deriving `head_dim` from `hidden_size / num_heads` gives 213 and every
shape downstream is wrong.

## Verifying the download

The repository includes `crc32.txt`. Checking it after a 54 GB download costs a
few minutes and rules out the failure mode where a truncated shard produces
plausible-looking garbage logits that you then spend an afternoon debugging.

## Lab

Open the model's index and answer questions about it without loading any weights:
count the tensors, group them by layer type, find the largest one, and detect the
output gate from shapes alone.

The lab runs on CPU against a cached copy of the index, so it starts in seconds.

## Further reading

- [The safetensors format specification](https://github.com/huggingface/safetensors)
- [Qwen3.8-27B on Hugging Face](https://huggingface.co/Qwen/Qwen3.8-27B)
