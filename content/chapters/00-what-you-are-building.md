---
title: What you are building
slug: 00-what-you-are-building
part: "Part 1 — Ground truth"
summary: The shape of an inference engine, the model this course targets, and how the labs run.
minutes: 20
gpu: false
objectives:
  - Describe the two phases of inference and why they have opposite bottlenecks.
  - Name the components you build over the next twenty chapters.
  - Run a lab and read its output.
lab: 00-hello-gpu
---

# What you are building

By the end of this course you have an inference engine: a program that loads open
weights and turns prompts into tokens, quickly, for many users at once. You write
the attention kernels, the cache, the scheduler, and the server. Nothing is
imported from vLLM or TensorRT.

The target model is [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). It
runs on a single NVIDIA A100 80GB, which you rent by the second from Modal.

## Two phases, opposite problems

Every generation splits into two phases, and almost every design decision in the
engine comes from the fact that they behave nothing alike.

*Prefill* processes the prompt. All the prompt's tokens go through the model at
once, as one large matrix multiply per layer. The GPU has plenty of arithmetic to
do and reads each weight once for the whole batch, so prefill saturates the
tensor cores. Prefill is **compute bound**.

*Decode* produces one token at a time. Each step reads all 50 GB of weights to
compute a single token's worth of arithmetic. The tensor cores idle while the
memory system works. Decode is **memory bound**.

The numbers make the gap concrete. An A100 80GB delivers about 312 TFLOP/s in
bfloat16 and about 2039 GB/s of memory bandwidth. Dividing one by the other gives
the *ridge point*: roughly 153 FLOPs per byte. An operation that does less
arithmetic than that per byte it touches can't reach peak compute, no matter how
good the kernel is. Prefill on a 2000-token prompt sits well above the ridge
point. Decode at batch size 1 sits at about 2. That is a 75x gap, and closing it
is most of what this course is about.

The techniques follow directly:

- Batching more requests raises decode's arithmetic intensity, because one read
  of the weights serves many sequences. This is why continuous batching exists.
- Shrinking what decode must read speeds it up proportionally. This is why
  grouped-query attention, quantization, and linear attention exist.
- Producing more than one token per weight read breaks the trade-off entirely.
  This is why speculative decoding exists.

## The model is a hybrid, and that matters

Qwen3.8-27B is not a stack of identical transformer blocks. Its 64 layers
alternate between two kinds of token mixer:

| Layer type | Count | State per sequence | Cost per token |
|---|---|---|---|
| Full attention (grouped-query) | 16 | Grows forever | 4 KB per layer |
| Gated delta linear attention | 48 | Fixed | 0 |

Every fourth layer is full attention. The rest use a linear attention that keeps
a fixed-size matrix instead of a growing cache. The consequence is large: a
64-layer model that used full attention everywhere would need 256 KB of KV cache
per token. This one needs 64 KB, plus a one-time 148 MB of recurrent state per
sequence. Past about 790 tokens of context, the hybrid is ahead, and the gap
widens without limit.

You implement both mixers. The linear attention is the more interesting one, and
chapter 6 spends a while on it.

## What you build

The engine comes together in this order:

1. **Read the weights.** Safetensors, shards, and the tensor names that tell you
   what the architecture really is.
2. **Do the arithmetic on paper.** Parameter counts, cache growth, and the
   roofline. Every later optimization is judged against these numbers.
3. **Write the layers.** RMSNorm, rotary embeddings, the delta rule, grouped-query
   attention, and the gated MLP, in PyTorch first.
4. **Assemble a forward pass** and check its logits against Hugging Face
   `transformers`. This is the last point where you have ground truth for free,
   so it's worth getting right.
5. **Add a cache** and turn an O(n²) loop into an O(n) one.
6. **Write kernels.** One CUDA kernel by hand to see what the hardware wants,
   then Triton for the rest: fused normalization, FlashAttention, paged
   attention.
7. **Schedule.** Continuous batching, chunked prefill, and preemption.
8. **Measure.** Time to first token, inter-token latency, throughput, and what
   each one hides.
9. **Scale.** Quantization, tensor parallelism, and speculative decoding.

## How the labs work

Each chapter ends with a lab. You edit code in the browser, click **Run**, and it
executes on a real A100 through Modal. Output streams back as it happens.

Every lab has a test harness that checks correctness first and speed second. A
kernel that returns wrong answers quickly fails, which is the correct outcome and
happens to everyone.

Some labs run on CPU, because arithmetic about memory doesn't need a GPU to be
worth doing. Those start in a couple of seconds instead of a minute.

Your first lab does nothing but confirm the plumbing works: it asks the GPU what
it is.

## Further reading

- [Efficient memory management for large language model serving with PagedAttention](https://arxiv.org/abs/2309.06180) — the vLLM paper.
- [FlashAttention: fast and memory-efficient exact attention with IO-awareness](https://arxiv.org/abs/2205.14135)
- [Gated delta networks: improving Mamba2 with delta rule](https://arxiv.org/abs/2412.06464)
