---
title: RMSNorm and the residual stream
slug: 04-rmsnorm-and-residuals
part: "Part 2 — A forward pass"
summary: The normalization every layer uses, why it accumulates in float32, and what pre-norm buys.
minutes: 40
gpu: true
objectives:
  - Implement RMSNorm and explain each term.
  - Show what happens when the reduction runs in bfloat16.
  - Describe the residual stream and why pre-norm architectures train deeper.
lab: 04-rmsnorm
---

# RMSNorm and the residual stream

Every one of the 64 layers normalizes twice, so this is the most-executed
operation in the model after the matrix multiplies. It's also the first place
numerics bite.

## The operation

```text
y = x / sqrt(mean(x^2) + eps) * w
```

LayerNorm subtracts the mean and divides by the standard deviation, then applies a
scale and a shift. RMSNorm drops the mean subtraction and the shift. It turns out
the re-centering contributes little, and dropping it saves a pass over the data
and a parameter vector per norm.

`eps` is 1e-6 here. It keeps the reciprocal square root finite when a row is all
zeros, which happens more often than you'd expect — padding positions, masked
tokens, and dead channels.

## Accumulate in float32

The reduction sums 5120 squared values. In bfloat16, which has 8 bits of mantissa,
a running sum stops being able to represent its next addend once the sum grows
past about 256 times the addend. The tail of the reduction contributes nothing and
the result is systematically low.

The reference implementation is explicit about this:

```python
dtype = x.dtype
x32 = x.float()
variance = x32.pow(2).mean(dim=-1, keepdim=True)
normed = x32 * torch.rsqrt(variance + eps)
return (normed * weight.float()).to(dtype)
```

Upcast, reduce, scale, downcast. The cast is free in the sense that matters: the
kernel is memory bound, and float32 registers cost no extra bandwidth because the
data crosses the bus in bfloat16 either way.

The lab has you run the reduction both ways and measure the error. On real
activations it's on the order of 1e-2 relative — large enough to change which
token wins an argmax.

## The residual stream

Each layer is applied like this:

```python
x = x + mixer(norm(x))
x = x + mlp(norm(x))
```

The tensor `x` passes through the whole model untouched by anything except
addition. Layers read a normalized copy and add their contribution back. That
running sum is the *residual stream*, and it's the model's working memory: 5120
channels that all 64 layers read from and write to.

Two consequences matter for the engine.

**Gradients and activations both stay well-scaled.** Normalizing the input to each
block rather than the output — pre-norm rather than post-norm — means the residual
path is a clean identity. This is what makes 64-layer models trainable without the
warmup schedules that post-norm transformers need.

**The stream grows.** Every layer adds to it, so its magnitude increases with
depth. The final `norm` before the output projection exists to undo that.

## Where the time goes

RMSNorm reads 2 bytes per element and writes 2, and does about 4 FLOPs. That's an
arithmetic intensity near 1, against an A100 ridge point of about 161. It is
entirely bandwidth bound, and the only way to speed it up is to move fewer bytes.

Fusing it with the residual add does exactly that. Unfused, the pair reads `x`,
reads `residual`, writes the sum, reads the sum back, and writes the normalized
output: five trips. Fused, it's two reads and two writes. Chapter 13 writes that
kernel; `engine/kernels/rmsnorm_triton.py` has the finished version.

Qwen3 also applies RMSNorm per attention head to queries and keys, over the
256-wide head dimension rather than the 5120-wide residual stream. Same operation,
different axis. It stabilizes the logit scale across heads.

## Lab

Implement `rms_norm` from scratch and match the reference to 1e-5. Then measure
the bfloat16-accumulation error, and time your implementation against a fused
version to see the bandwidth ceiling.

## Further reading

- [Root mean square layer normalization](https://arxiv.org/abs/1910.07467)
- [On layer normalization in the transformer architecture](https://arxiv.org/abs/2002.04745) — pre-norm versus post-norm.
