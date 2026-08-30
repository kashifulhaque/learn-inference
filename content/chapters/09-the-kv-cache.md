---
title: The KV cache
slug: 09-the-kv-cache
part: "Part 3 — Making it fast"
summary: Turning quadratic regeneration into linear decode, and what the cache costs in return.
minutes: 50
gpu: true
objectives:
  - Explain why decode without a cache is quadratic.
  - Implement a cache that holds both KV entries and recurrent state.
  - Measure the speedup and the memory it costs.
lab: 09-kv-cache
---

# The KV cache

Without a cache, generating token *n* means running the whole prompt plus
everything generated so far through the model again. Generating 500 tokens from a
500-token prompt does about 375,000 token-forward-passes instead of 1000. The
cache is the single largest speedup in the engine and the first one to implement.

## What to cache

For each full-attention layer, keys and values for every past token. Those are
exactly the tensors that don't change when a new token arrives: the key computed
for token 5 is the same whether the sequence is 6 tokens long or 6000.

Queries aren't cached. A query is used once, in the step that produces it, and
never again.

For each linear-attention layer, the recurrent state and the convolution window.
Both are fixed size and both are overwritten every step rather than appended to.

`HybridCache` holds both:

```python
self.k_cache: dict[int, Tensor]      # full-attention layers
self.v_cache: dict[int, Tensor]
self.states: dict[int, Tensor]       # linear-attention layers
self.conv_states: dict[int, Tensor]
```

The dictionaries are keyed by layer index and only contain entries for the layers
that need them. A dense list with 48 unused slots works too, and wastes nothing,
but the dictionary makes the hybrid structure visible in the debugger.

## Preallocate, don't concatenate

The obvious implementation appends:

```python
self.k = torch.cat([self.k, new_k], dim=2)   # don't
```

Every append allocates a new tensor and copies the old one. Over 500 decode steps
that copies the cache 500 times, and the copies grow. It also fragments the
allocator badly enough to cause out-of-memory errors well below the real limit.

Allocate once at the maximum length and write into a slice:

```python
self.k_cache[layer][:, :, start:stop] = k
return self.k_cache[layer][:, :, :stop]
```

The write is in place and the read is a view. Neither allocates.

The cost is that you reserve `max_seq_len` for every sequence whether it needs it
or not. A batch of 32 sequences that might reach 32k tokens reserves 64 GiB even
if most stop at 200 tokens. Chapter 15 fixes that with paging; contiguous
preallocation is the right thing to build first.

## Advance the cursor once

The cache length must advance after all 64 layers have run, not inside each layer.
Advancing per layer makes layer 1 believe the sequence is one token longer than
layer 0 does, which corrupts the mask offset and the write position.

```python
for layer in self.layers:
    x = layer(x, cos, sin, cache=cache)
if cache is not None:
    cache.advance(seq)
```

## Measuring it

Generating 128 tokens from a 512-token prompt, on one A100:

| | Forward passes | Tokens processed |
|---|---|---|
| No cache | 128 | 73,792 |
| With cache | 1 prefill + 127 decode | 639 |

The work drops by about 115 times. Wall-clock speedup is smaller, because
uncached decode does its work in large efficient batches while cached decode does
tiny inefficient ones. Expect 20 to 40 times, and expect the gap to widen with
sequence length, since the uncached version is quadratic and the cached one is
linear.

## The cost

The cache is now the thing that limits your batch size. Chapter 2's arithmetic
applies: 64 KiB per token, times context, times batch. On an A100 80GB with the
weights loaded, you have roughly 19 GiB, or 304k tokens total.

The cache is also the thing decode reads. At batch 1 it's small next to the 53.8 GB
of weights, but it scales with batch size while the weight read doesn't. Past
batch 32 at long context, the cache read starts to dominate — which is where
quantizing the cache to int8 becomes worth its quality cost.

## What the hybrid changes

Only 16 layers append. The other 48 overwrite a fixed state, so 75% of the model
contributes nothing to cache growth. That's the whole point of the architecture,
and it's why this model can offer 262k context on a single GPU.

The state carry has to be exactly right, though. `delta_rule_chunked` accepts a
`state` argument and returns the final state; prefill passes `None` and stores the
result, and each decode step passes the stored state back. Getting this wrong
produces a model that prefills correctly and then generates as if the prompt never
happened.

## Lab

Implement `HybridCache.append` and the state carry, then run the equivalence test:
a prefill followed by single-token decodes must produce the same logits as one
full forward pass. Then measure cached against uncached generation and report the
speedup.

## Further reading

- [Fast transformer decoding: one write-head is all you need](https://arxiv.org/abs/1911.02150)
