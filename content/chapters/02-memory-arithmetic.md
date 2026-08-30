---
title: Memory arithmetic
slug: 02-memory-arithmetic
part: "Part 1 — Ground truth"
summary: Parameter counts, KV cache growth, and the break-even point that justifies a hybrid model.
minutes: 60
gpu: false
objectives:
  - Count a model's parameters from its config and predict its weight footprint.
  - Compute KV cache growth per token and decide what batch size fits.
  - Find the context length where a hybrid cache beats a dense one.
lab: 02-memory-math
---

# Memory arithmetic

An A100 80GB gives you 80 GB, minus about 1 GB the driver keeps. Everything must
fit: weights, cache, activations, and the allocator's fragmentation. This chapter
works out where it all goes, because every later optimization is measured against
these numbers.

## Weights

Count the parameters block by block.

**Embedding and output.** The vocabulary is 248,320 tokens and the hidden size is
5120, so the embedding holds 1.27B parameters. `tie_word_embeddings` is false, so
the output projection is a separate matrix of the same size. Together that's 2.54B
parameters, 9.5% of the model, spent entirely on the vocabulary.

**The MLP, in every layer.** SwiGLU needs three matrices: gate and up, each
5120 x 17408, and down, 17408 x 5120. That's 3 x 5120 x 17408 = 267M parameters
per layer, and there are 64 layers. The MLP is 17.1B parameters, 64% of the model.

**Full-attention layers.** The query projection emits `24 x 256 x 2` because of
the output gate, so it's 5120 x 12288. Keys and values are 5120 x 1024 each. The
output projection is 6144 x 5120. That's about 105M per layer, over 16 layers.

**Linear-attention layers.** Queries and keys are 16 heads of 128, values are 48
heads of 128, and the output gate matches the values, so `in_proj_qkvz` is
5120 x 16384. The output projection is 6144 x 5120. Adding the convolution, the
per-head gates, and the biases gives about 122M per layer, over 48 layers.

The total lands at 26.9B, which matches the name. In bfloat16, at 2 bytes per
parameter:

```text
26.9e9 x 2 bytes = 53.8 GB
```

That leaves roughly 26 GB on an 80GB card for everything else. An A100 40GB can't
hold the weights at all, which is why this course uses the 80GB part.

## KV cache

Each full-attention layer stores a key and a value per token, per KV head:

```text
bytes/token/layer = 2 (K and V) x 4 KV heads x 256 head_dim x 2 bytes = 4096
```

Only the 16 full-attention layers pay this:

```text
bytes/token = 4096 x 16 = 65,536 = 64 KB
```

Grouped-query attention is already doing heavy lifting here. With 24 KV heads
instead of 4, the same cache would be 384 KB per token, six times larger.

Cache size scales with batch size and context, and nothing else:

| Context | Batch 1 | Batch 8 | Batch 32 |
|---|---|---|---|
| 4k | 0.25 GB | 2.0 GB | 8.0 GB |
| 32k | 2.0 GB | 16.0 GB | 64.0 GB |
| 128k | 8.0 GB | 64.0 GB | 256.0 GB |

With 26 GB of headroom, you can serve batch 8 at 32k context, or batch 32 at 4k.
The cache, not the weights, sets your maximum batch size. That single fact is why
chapter 15 is about packing it efficiently.

## Recurrent state

The 48 linear-attention layers keep a fixed state instead. Per layer, per
sequence:

```text
48 value heads x 128 key dim x 128 value dim x 4 bytes (float32) = 3.0 MiB
```

Adding the four-step convolution window brings each layer to 3.08 MiB, and across
48 layers the state is 147.8 MiB. It doesn't grow. A sequence at 128k context
carries the same 147.8 MiB as a sequence at 10 tokens.

## The break-even point

Now compare the hybrid against a hypothetical model where all 64 layers use full
attention.

The dense model pays 256 KiB per token and no fixed cost. The hybrid pays 64 KiB
per token plus 147.8 MiB fixed. Setting them equal:

```text
262144 x L = 65536 x L + 154,927,104
196608 x L = 154,927,104
L = 788 tokens
```

Below 788 tokens of context the fixed state costs more than the KV entries it
replaces. Above it the hybrid wins, and the margin grows by 192 KiB for every
further token. At 32k context the hybrid uses 2.14 GiB per sequence against
8.0 GiB. At 128k it uses 8.14 GiB against 32 GiB.

The design is a deliberate trade: pay a constant to make the slope shallower.
`ModelConfig.hybrid_breakeven_tokens` computes it for any config, and the lab has
you derive it.

## Activations

During prefill, intermediate tensors are transient but not small. The MLP's
hidden activation is `tokens x 17408` in bfloat16 — 143 MB for 4096 tokens, and
SwiGLU holds two of them at once. Attention scores, if you materialize them, are
`heads x q_len x kv_len`: at 8k context with 24 heads that's 12 GB for one
sequence. Not materializing them is what FlashAttention is for, and it's why
chunked prefill caps how many tokens enter a forward pass at once.

## Budgeting

A usable A100 80GB budget looks like this:

| Item | Size |
|---|---|
| Weights, bf16 | 53.8 GB |
| CUDA context and driver | ~1.0 GB |
| Activation workspace | ~3.0 GB |
| Fragmentation headroom | ~2.0 GB |
| **Left for cache** | **~19 GB** |

19 GB of KV cache is 304k tokens: batch 8 at 32k context, batch 64 at 4k, or any
mix. `PagedKVCache` in chapter 15 turns that into a block pool so short requests
don't reserve space for a length they never reach.

Quantizing the weights to int8 halves the weight footprint and roughly triples
the cache budget. Chapter 18 covers what that costs in quality, and why FP8 —
which does this better — isn't available to you on Ampere.

## Lab

Write the arithmetic as code. Given a config, produce parameter counts by
component, cache growth per token, the hybrid break-even point, and the largest
batch size that fits a given GPU at a given context length. The harness checks
your numbers against the real model's config.

## Further reading

- [NVIDIA A100 tensor core GPU architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf)
- [GQA: training generalized multi-query transformer models from multi-head checkpoints](https://arxiv.org/abs/2305.13245)
