---
title: Grouped-query attention
slug: 07-grouped-query-attention
part: "Part 2 — A forward pass"
summary: The 16 layers that keep a real KV cache, with head sharing and an output gate.
minutes: 60
gpu: true
objectives:
  - Explain how GQA trades quality for cache size and why the trade favors decode.
  - Implement causal masking that is correct when a cached prefix is present.
  - Describe what the attention output gate does.
lab: 07-gqa
---

# Grouped-query attention

Sixteen of the 64 layers use ordinary softmax attention. They're the layers that
own a KV cache, so they set the memory ceiling and dominate decode time.

## The formula

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
```

The `sqrt(d)` divisor keeps the logits from growing with head dimension. If `q`
and `k` have unit-variance entries, `q . k` has variance `d`, so scores scale like
`sqrt(d)`. With `d = 256`, unscaled logits reach magnitudes where softmax
saturates into a hard argmax and gradients vanish.

Causal masking sets scores to negative infinity where a query would see a future
key. With a cached prefix, the offset matters:

```python
offset = kv_len - q_len
scores = scores.masked_fill(idx_k > idx_q + offset, float("-inf"))
```

Query row *i* of the current block is at absolute position `i + offset`. Forgetting
the offset produces a mask that's correct during prefill, when `offset` is zero,
and wrong during every decode step. The symptom is a model that generates well for
its prompt and then drifts.

## Head sharing

Multi-head attention gives every query head its own keys and values. This model
has 24 query heads and 4 KV heads, so six queries share each KV head.

The KV cache shrinks by six times. Since decode is bound by how fast you can read
that cache, decode gets roughly six times faster. Quality loss is small, and it's
smaller still because the heads were trained this way rather than merged
afterwards.

The naive way to implement it duplicates the KV heads:

```python
def repeat_kv(x, repeats):
    return x[:, :, None].expand(b, h, repeats, s, d).reshape(b, h * repeats, s, d)
```

That's correct and defeats the purpose: `reshape` on an expanded tensor
materializes the copies, so you're back to reading six times the bytes. It's fine
for checking correctness. A real kernel indexes the shared head instead, which is
what the `kv_group` parameter does in the FlashAttention kernel in chapter 14:

```python
kv_head = head // kv_group
```

## Query and key normalization

Qwen3 applies RMSNorm per head to queries and keys, over the 256-wide head
dimension, before the rotation. It bounds the logit scale independently of what
the projections produce. In the tensor listing these appear as `q_norm.weight` and
`k_norm.weight` with shape `[256]`.

## The output gate

The query projection emits `24 x 256 x 2 = 12288` channels, not 6144. The second
half is a gate:

```python
q = self.q_proj(x)
q, gate = q.chunk(2, dim=-1)
# ... attention ...
out = out * torch.sigmoid(gate)
```

Softmax attention always outputs a convex combination of values — the weights sum
to 1, so a head must return *something* even when nothing in the context is
relevant. The gate lets the head scale its own output toward zero instead. It's
the same motivation as the gate on the linear-attention layers, and it costs one
extra `hidden x heads*head_dim` matrix.

Miss the split and the shapes still work if you happen to slice the first 6144
channels, but half the projection's learned behavior is discarded and quality
drops in a way that's hard to attribute.

## The naive implementation, and why you keep it

```python
scores = torch.matmul(q, k.transpose(-1, -2)) * scale
scores = scores.masked_fill(mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)
out = torch.matmul(weights, v)
```

This materializes a `batch x heads x q_len x kv_len` tensor. At 8k context with 24
heads that's 12 GB in bfloat16 for a single sequence. You can't serve with it.

Keep it anyway. It's the ground truth for every kernel you write from chapter 14
onward, and it's small enough to read in one sitting. Run it in float32 on short
sequences and compare.

## Where decode time goes

A decode step at batch size 1 reads:

- All 53.8 GB of weights.
- The KV cache for the full-attention layers: 64 KB per token of context.
- The recurrent state: 147.8 MiB.

At 1935 GB/s, reading the weights alone takes 28 ms, which caps you at about 36
tokens per second no matter how good your kernels are. Measured bandwidth is
closer to 1275 GB/s, so 42 ms and 24 tokens per second is the realistic figure. At 32k context the cache
adds 2.1 GiB, another 1 ms — small next to the weights, but it grows with both
context and batch size while the weight read doesn't.

That asymmetry is the entire argument for batching. Sixteen sequences read the
weights once and pay 16 times the cache. Throughput goes up almost 16 times;
per-token latency barely moves.

## Lab

Implement grouped-query attention with correct causal masking under a cached
prefix, the query and key norms, and the output gate. Match `F.scaled_dot_product_attention`
to 1e-3 in bfloat16, then show that your mask is right by checking that a prefill
followed by decode steps produces the same logits as one long forward pass.

## Further reading

- [GQA: training generalized multi-query transformer models from multi-head checkpoints](https://arxiv.org/abs/2305.13245)
- [Fast transformer decoding: one write-head is all you need](https://arxiv.org/abs/1911.02150) — multi-query attention.
