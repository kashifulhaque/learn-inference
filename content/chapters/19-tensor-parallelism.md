---
title: Tensor parallelism
slug: 19-tensor-parallelism
part: "Part 6 — Scaling"
summary: Splitting matrices across GPUs so one all-reduce per block is enough.
minutes: 70
gpu: true
objectives:
  - Split linear layers column-wise and row-wise and explain why the pairing works.
  - Count the collective operations a transformer block needs and explain why it is two.
  - Describe how attention heads and a hybrid model's layers partition.
lab: 19-tensor-parallel
---

# Tensor parallelism

One A100 80GB holds this model in bfloat16 with about 19 GiB left for cache. Two
GPUs give you 27 GiB of weights each and far more cache room, and roughly halve
decode latency, because each GPU reads half the weights.

Tensor parallelism splits individual matrices across devices. Every GPU runs every
layer on a slice of the work.

## Column and row splits

A linear layer `Y = X W` can be split two ways.

**Column parallel** splits `W` by output columns. Each GPU computes a slice of the
output from the full input:

```text
W = [W_0 | W_1]        Y_0 = X W_0,  Y_1 = X W_1,  Y = [Y_0 | Y_1]
```

Input is replicated, output is sharded. No communication.

**Row parallel** splits `W` by input rows. Each GPU takes a slice of the input and
produces a partial sum of the full output:

```text
W = [W_0]              Y = X_0 W_0 + X_1 W_1
    [W_1]
```

Input is sharded, output needs an all-reduce.

## The pairing

Put them back to back and the sharded intermediate never needs gathering:

```text
X --[column parallel]--> sharded --[row parallel]--> all-reduce --> Y
```

This is exactly the shape of both blocks in a transformer.

**The MLP.** `gate_proj` and `up_proj` are column parallel, `down_proj` is row
parallel. SiLU and the elementwise multiply are per element, so they work fine on
a shard. One all-reduce at the end.

**Attention.** `q_proj`, `k_proj`, `v_proj` are column parallel, split by head.
Each GPU owns whole heads and computes their attention independently, including
their share of the KV cache. `o_proj` is row parallel. One all-reduce at the end.

So a transformer block needs exactly two all-reduces, one per sub-block. That's
the whole communication cost, and it's why tensor parallelism scales well within a
node.

## Splitting the heads

With 24 query heads and 4 KV heads across 2 GPUs, each GPU takes 12 query heads
and 2 KV heads. The KV heads must divide evenly by the parallelism degree, and 4
does not divide by 8 — so 8-way tensor parallelism on this model requires
replicating KV heads, which wastes cache.

The linear-attention layers split the same way, by value head: 48 value heads
become 24 per GPU, along with their recurrent state. The delta rule is independent
per head, so the state shards cleanly and no communication is needed until
`out_proj`.

## Communication cost

A ring all-reduce moves `2 x (world_size - 1) / world_size` times the tensor
size per GPU. For 4096 tokens at hidden size 5120 in bfloat16 the tensor is
42 MB, so 2-way moves 42 MB per rank per all-reduce. Two all-reduces per layer
over 64 layers gives 5.4 GB per forward pass.

NVLink between two A100s runs at 600 GB/s, so that's about 9 ms — noticeable but
small against a full prefill. Over PCIe at 64 GB/s it's 84 ms, which is not
small at all. Tensor parallelism needs fast interconnect, and this is the reason
it's a within-node technique.

During decode the tensors are tiny — one token, 10 KB — so latency dominates
rather than bandwidth. Each all-reduce costs 5 to 10 microseconds of latency, and
128 of them per step is around 1 ms. That's real overhead against a decode step of
a few milliseconds, and it's why tensor parallelism improves decode latency
sublinearly.

## What stays replicated

Not everything shards. Norms are elementwise and cheap, so every GPU keeps a full
copy. The embedding can be sharded by vocabulary, but it's a gather, so
replication is simpler and costs 2.5 GB per GPU. The output projection is
usually sharded by vocabulary with a gather at the end, since its output is
248,320 wide.

## Testing without two GPUs

You don't need two GPUs to check the math. Simulate both ranks on one device:
split the weights, run each shard, and combine. The result must match the unsharded
layer exactly, up to floating-point reduction order.

That test catches every partitioning bug — wrong split axis, wrong concatenation
order, missing all-reduce — and it's the lab for this chapter. What it doesn't
catch is NCCL configuration, which needs real devices.

## Lab

Implement column-parallel and row-parallel linear layers, then build a
tensor-parallel MLP and a tensor-parallel attention block. Simulate 2-way and
4-way parallelism on one GPU and show the outputs match the unsharded versions to
1e-3. Report the communication volume per layer.

## Further reading

- [Megatron-LM: training multi-billion parameter language models using model parallelism](https://arxiv.org/abs/1909.08053)
- [Efficiently scaling transformer inference](https://arxiv.org/abs/2211.05102)
