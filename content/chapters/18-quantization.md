---
title: Quantization on Ampere
slug: 18-quantization
part: "Part 6 — Scaling"
summary: Weight-only int8, why grouping matters, and why FP8 is not available to you on an A100.
minutes: 70
gpu: true
objectives:
  - Implement group-wise int8 weight quantization and measure its error.
  - Explain why weight-only quantization helps decode and not prefill.
  - Explain what FP8 offers and why Ampere cannot use it.
lab: 18-int8-quant
---

# Quantization on Ampere

The weights are 53.8 GB and decode reads all of them for every token. Halving that
nearly doubles decode speed. Quantization is the most direct throughput
improvement available, and it's also where you can most easily wreck the model
without noticing.

## Weight-only, and why

Decode is memory bound. Time is proportional to bytes read, and the weights are
almost all of the bytes. Storing them in int8 halves the read and roughly doubles
decode speed.

The arithmetic doesn't have to be int8. The kernel reads int8, dequantizes to
bfloat16 in registers, and does a bfloat16 matmul. That sounds wasteful and isn't:
the dequantization is a multiply-add per element, and the operation was bound by
memory, not arithmetic.

Prefill sees almost no benefit, because prefill is compute bound. Quantizing
weights doesn't reduce its FLOPs. Expect decode to roughly double and prefill to
stay flat or get slightly worse.

## The scheme

Map a float range onto 256 integers:

```python
scale = weight.abs().amax(dim=-1, keepdim=True) / 127.0
quantized = (weight / scale).round().clamp(-128, 127).to(torch.int8)
dequantized = quantized.to(torch.bfloat16) * scale
```

Symmetric quantization has no zero point, so dequantization is one multiply.
Asymmetric quantization fits skewed distributions slightly better and costs an
extra add per element. Weights are close to symmetric around zero, so symmetric is
the right default.

## Group size decides quality

The choice that matters most is how many weights share a scale.

**Per tensor.** One scale for the whole matrix. Smallest overhead, worst quality —
a single outlier sets the scale for 89 million weights, and everything else
collapses into a few integer levels.

**Per channel.** One scale per output row. Much better, still vulnerable to a row
with one large entry.

**Per group.** One scale per 128 contiguous weights. This is the standard choice.
Overhead is 2 bytes per 128 weights, about 1.5%, and it confines an outlier's
damage to its own group.

Measure quality as relative error on the dequantized weights, and then on the layer
output, which is what actually matters:

| Granularity | Typical weight error | Overhead |
|---|---|---|
| Per tensor | 2 to 5% | ~0% |
| Per channel | 0.5 to 1% | 0.02% |
| Per group of 128 | 0.1 to 0.3% | 1.5% |

## Activation outliers

Transformer activations contain outliers up to 100 times the typical magnitude,
concentrated in a small number of channels. This is why *activation* quantization
is much harder than weight quantization, and why weight-only schemes are the
common choice.

Two ideas address it, and both are worth knowing:

**SmoothQuant** shifts the difficulty from activations to weights by scaling
channel-wise, dividing activations by a per-channel factor and multiplying the
corresponding weight rows by it. The product is unchanged and both tensors become
easier to quantize.

**AWQ** observes that a small fraction of weight channels matter far more than the
rest, identifies them from activation statistics, and protects them with a
per-channel scale.

## What FP8 would give you, and why you can't have it

FP8 is a natural fit: 8 bits with an exponent, so it handles outliers far better
than int8 at the same width. Hopper and later have tensor cores that do FP8
matmuls natively, at twice bfloat16 throughput.

The A100 is Ampere, compute capability 8.0. It has no FP8 tensor core support. You
can store weights in an FP8 layout and unpack them, but the arithmetic runs in
bfloat16 and there's no compute speedup — only the bandwidth saving, which int8
also gives you with better tooling.

On Ampere, int8 is the right choice. This is a real hardware constraint, and it's
worth knowing about before you spend a day on a kernel that can't be fast. The
model publishes an official FP8 checkpoint, `Qwen/Qwen3.8-27B-FP8`, which is
useful on an H100 and not here.

## Quantizing the KV cache

The cache can be quantized too, and it matters at long context and large batch,
where the cache read starts to rival the weight read.

Keys tolerate int8 poorly — they feed a dot product that goes through a softmax,
so errors amplify. Values tolerate it better, since they're combined linearly.
A common configuration keeps keys in bfloat16 and values in int8, or uses per-head
scales for keys.

## Measuring the damage

Weight error is a proxy. What matters is the model's output. Three checks, in
increasing order of cost and usefulness:

1. **Layer output error.** Run one layer with and without quantization on the same
   input, and compare. Fast, and it localizes the damage.
2. **Logit correlation.** Full forward pass on a few prompts. Correlation should
   stay above 0.999 and top-1 agreement above 0.98.
3. **Perplexity on held-out text.** The real measure. An increase of more than
   about 1% means the scheme is too aggressive.

Doing only the first is how quantization schemes that look fine ship broken.

## Lab

Implement group-wise int8 quantization and dequantization, measure the error at
three granularities, and demonstrate the memory saving. Then run a quantized MLP
layer against its bfloat16 version and report both the output error and the decode
speedup.

## Further reading

- [SmoothQuant: accurate and efficient post-training quantization for large language models](https://arxiv.org/abs/2211.10438)
- [AWQ: activation-aware weight quantization for LLM compression and acceleration](https://arxiv.org/abs/2306.00978)
- [GPTQ: accurate post-training quantization for generative pre-trained transformers](https://arxiv.org/abs/2210.17323)
