---
title: The forward pass
slug: 08-the-forward-pass
part: "Part 2 — A forward pass"
summary: Assembling 64 layers, loading real weights, and checking your logits against the reference.
minutes: 75
gpu: true
objectives:
  - Assemble a hybrid decoder stack from the layers you have written.
  - Map checkpoint tensor names onto your modules and catch mismatches early.
  - Validate your implementation against Hugging Face transformers.
lab: 08-forward-pass
---

# The forward pass

You have RMSNorm, RoPE, both token mixers, and the MLP. This chapter stacks them,
loads real weights, and checks the result. It's the last point where correctness
is cheap to verify, so it's worth being thorough.

## The stack

Every layer has the same shape:

```python
x = x + mixer(input_layernorm(x))
x = x + mlp(post_attention_layernorm(x))
```

Only the mixer changes. `config.layer_types` decides:

```python
self.is_full_attention = config.layer_types[layer_idx] == "full_attention"
```

Around the 64 layers sit the embedding, a final norm, and the output projection.
That's `HybridLanguageModel` in `engine/model.py`, and it's about 80 lines.

## Loading weights

The tensor names in the checkpoint won't match your module names. Write an
explicit mapping rather than trying to make your class hierarchy mirror the file.
An explicit map is easier to read, and it fails loudly when a name is missing
instead of silently leaving a tensor at its initialized value.

Three checks catch nearly every loading bug:

**Every checkpoint tensor is consumed.** Track which names you read and assert the
set is complete. An unconsumed tensor means you missed a component.

**Every parameter is written.** Initialize the model on the `meta` device, or fill
it with NaN first. A parameter that's still NaN after loading tells you
immediately; a parameter left at its random initialization produces subtly worse
output that you might not notice for hours.

**Shapes match exactly.** PyTorch's `Linear` stores weights transposed relative to
the mathematical convention. `nn.Linear(in, out).weight` has shape `[out, in]`.
Checkpoints usually follow the same convention, but a transposed load is a real
failure mode — it raises only when the dimensions differ, and for a square matrix
it doesn't raise at all.

Load layer by layer, moving each layer to the GPU before reading the next. With
memory-mapped safetensors, peak host memory stays at roughly one layer.

## Validating against the reference

Load the same weights in `transformers`, run both on the same tokens, and compare.

Compare logits, not generated text. Greedy decoding is a hard argmax, so two
implementations can agree on every generated token while differing by a lot on the
underlying distribution. The comparison you want:

```python
max_abs_diff = (mine - reference).abs().max()
correlation  = torch.corrcoef(torch.stack([mine.flatten(), reference.flatten()]))[0, 1]
top1_match   = (mine.argmax(-1) == reference.argmax(-1)).float().mean()
```

In bfloat16, expect a max absolute difference around 1e-2 on logits of magnitude
10 to 30, correlation above 0.9999, and top-1 agreement of 1.0. Bit-exact
agreement is not achievable and not the goal: different kernels reduce in
different orders, and floating-point addition isn't associative.

## Bisecting a mismatch

When the logits disagree, compare layer by layer. Register a forward hook on both
models, run one prompt, and find the first layer where the residual stream
diverges. Everything downstream is downstream of that one layer.

A rough guide to what the first bad layer tells you:

| First divergence | Likely cause |
|---|---|
| Layer 0, immediately | Embedding lookup or weight transpose. |
| Layer 3, the first full-attention layer | The output gate split, or the mask offset. |
| Layer 0 but only past position 64 | The partial rotary boundary. |
| Gradual drift across all layers | A norm running in bfloat16 instead of float32. |
| Correct for one token, wrong after | Cache indexing or the decode position. |

## Two forward passes, one model

Prefill and decode run the same code with different shapes, and both must produce
the same numbers. The test is direct: run a full forward pass over a sequence, then
run a prefill on a prefix followed by single-token steps, and compare the logits at
matching positions. They must agree.

That test exercises the KV cache append, the linear-attention state carry, the
convolution window, and the decode position all at once. It's the single most
valuable test in the engine, and the reference implementation passes it:

```text
token-by-token vs full-forward maxerr: 5.96e-07
```

## Lab

Assemble the stack, load a small Qwen3 checkpoint, and validate against
`transformers`. The harness checks correlation, top-1 agreement, and the
prefill-then-decode equivalence.

The lab uses Qwen3-0.6B rather than the 27B model, because a wrong answer in three
seconds teaches you more than a wrong answer in four minutes. The architecture
differs — Qwen3-0.6B is dense — so the harness supplies a hybrid config with random
weights for the parts that only the 27B exercises.

## Further reading

- [The transformers modeling code for Qwen3](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3)
