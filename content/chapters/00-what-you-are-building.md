---
title: What you are building
slug: 00-what-you-are-building
part: "Part 1 — Ground truth"
summary: What an inference engine does, why prefill and decode have opposite bottlenecks, and how a lab reaches the GPU.
minutes: 35
gpu: false
objectives:
  - Describe what one forward pass computes and what autoregressive generation adds to it.
  - Derive the arithmetic intensity of decode and prefill, and explain why they have opposite bottlenecks.
  - Name the components you build over the next twenty chapters.
  - Run a lab and read its output.
lab: 00-hello-gpu
---

# What you are building

> [!TLDR]
> - The engine wraps one pure function, the forward pass, and calls it in a
>   loop: prefill runs the whole prompt once, and decode runs one token per step.
> - Decode at batch 1 does 1 FLOP per byte against the A100's ridge point of
>   161, so it's memory bound. Prefill does about $T$, so it's compute bound.
> - Every optimization in the course raises the tokens per weight read, shrinks
>   the bytes, or commits more than one token per read.
> - Each lab runs on a rented A100 and prints one `[PASS]` or `[FAIL]` line per
>   check.

By the end of this course you have an inference engine: a program that loads open
weights and turns prompts into tokens, quickly, for many users at once. You write
the attention kernels, the cache, the scheduler, and the server. Nothing is
imported from vLLM or TensorRT.

The target model is [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). It
runs on a single NVIDIA A100 80GB, which you rent by the second from a GPU
provider.

## Before you start

- **Maths.** University linear algebra and calculus. The course assumes nothing
  about GPUs, transformers, or serving.
- **Python and PyTorch.** Tensor indexing, `reshape`, `transpose`,
  broadcasting, and `@` for matrix multiply. You don't need CUDA; chapter 12
  starts from nothing.
- **Everything else.** [Notation and prerequisites](/c/00a-notation-and-prerequisites)
  collects the rest: tensor shape conventions, `torch.einsum`, softmax and its
  shift invariance, what bfloat16 actually stores, the GPU vocabulary, and a
  table of the symbols later chapters use.

Read that page now if any of these are unfamiliar: arithmetic intensity, warp,
ULP, coalescing, KiB against KB. Otherwise, skim it and come back when a symbol
stops making sense.

## What a forward pass computes

An inference engine is built around one function. It takes a sequence of token
IDs and returns, for each position, a score for every word in the vocabulary.
Everything else in the engine exists to call this function often, cheaply, and
for many users at once.

For a prompt of $T$ tokens, the function runs three steps:

1. **Embed.** Each token ID indexes a row of a $V \times d$ matrix, where
   $V = 248{,}320$ is the vocabulary size and $d = 5120$ is the hidden size. The
   result is a tensor of shape `(T, 5120)`. This is the *residual stream*: one
   vector per token, carrying everything the model knows about that position so
   far.
2. **Run 64 layers.** Each layer reads the residual stream, computes a
   correction, and adds it back. A layer has two halves:
   - The *token mixer* lets positions see each other. This is attention, or in
     this model, sometimes a recurrence.
   - The *MLP* transforms each position independently.

   Both halves are wrapped in normalization, and both add to the stream rather
   than replacing it.
3. **Project to logits.** A final normalization, then a $d \times V$ matrix
   turns each position's 5120-vector into 248,320 scores, one per vocabulary
   entry. The result has shape `(T, 248320)`.

Only step 2's token mixer is sequential across tokens, and even there the
dependency runs one way: position $t$ may read positions $\leq t$, never the
future. That restriction is *causal masking*, and it's what makes the cache in
the next section possible.

> [!KEY] The forward pass has no memory of its own
> It's a pure function of the token IDs you hand it. Every bit of "state" in a
> chat session is the token sequence itself, plus caches that exist only to
> avoid recomputing what the function already computed.

## What autoregressive means

Generation is a loop around the forward pass. The model scores every position,
but only the last position's logits are useful for generating: they describe
what comes next.

```python
tokens = tokenizer.encode(prompt)        # e.g. 2000 token IDs
for _ in range(max_new_tokens):
    logits = model(tokens)               # (len(tokens), 248320)
    next_id = sample(logits[-1])         # one token ID
    tokens.append(next_id)
    if next_id == eos_id:
        break
```

Read that loop carefully, because ==its cost structure is the whole course==.

- The first iteration processes 2000 tokens.
- Every iteration after it processes 2001, then 2002, and so on, and recomputes
  from scratch everything it already computed.
- Generating 500 tokens this way runs the model over
  $2000 + 2001 + \cdots + 2499 = 1{,}124{,}750$ token positions, roughly 1.1
  million, to produce 500 useful ones. That's the $O(n^2)$ loop chapter 9
  replaces.

The fix follows from causal masking. Layer $\ell$'s output at position $t$
depends only on positions $\leq t$, and those positions haven't changed. So you
keep each layer's intermediate keys and values in a cache and feed the loop one
new token per step. Every step becomes $O(1)$ in the number of tokens
processed, though not in bytes read, as the next sections show.

That split gives generation two phases that behave nothing alike:

- **Prefill** is the first iteration: the whole prompt, at once.
- **Decode** is every iteration after it: one token, against a cache.

The word *autoregressive* means exactly this: the model's own output is appended
to its input and fed back. There's no way around the loop. You can't produce
token 10 without having produced token 9, so ==a 500-token reply takes 500
sequential steps== no matter how large your GPU is.

## Arithmetic intensity and the ridge point

Prefill and decode need different engineering because of one ratio, not a rule
of thumb. *Arithmetic intensity* is the FLOPs an operation performs divided by
the bytes it must move between HBM and the chip:

$$
I = \frac{\hla{\text{FLOPs}}}{\hlb{\text{bytes moved}}}
$$

A GPU has two peak rates. An A100 80GB delivers about 312 TFLOP/s in bfloat16
and has a rated memory bandwidth of 1935 GB/s. Divide the first by the second
to get the *ridge point*:

$$
I_{\text{ridge}} = \frac{312 \times 10^{12}\ \text{FLOP/s}}
{1935 \times 10^{9}\ \text{byte/s}} \approx \boxed{161\ \text{FLOP/byte}}
$$

> [!INTUITION]
> The ridge point is how many FLOPs the chip can do in the time it takes to read
> one byte. An operation with $I$ below 161 finishes its arithmetic before the
> next bytes arrive, so it can't saturate the tensor cores. No amount of kernel
> cleverness changes that. ==Only moving fewer bytes does.==

## Decode: one token per weight read

At batch 1, almost all of decode's work is multiplying a weight matrix by a
single vector. For an $n \times m$ weight matrix in bfloat16, count the FLOPs
and the bytes:

$$
\hla{\text{FLOPs}} = 2nm, \qquad \hlb{\text{bytes}} = 2nm
$$

The $\hla{2}$ in the FLOP count is one multiply plus one add per weight. The
$\hlb{2}$ in the byte count is bfloat16's two bytes per weight. They cancel:

$$
I_{\text{decode}} = \frac{\hla{2nm}}{\hlb{2nm}} = \boxed{1\ \text{FLOP/byte}}
$$

Every weight is read from HBM, used for exactly one multiply-add, and discarded.
Against a ridge point of 161, decode at batch 1 uses about $1/161$ of the A100's
arithmetic, ==under 1%==.

> [!EXAMPLE] One decode step of the whole model
> Chapter 2 counts 26.9B parameters, which is 53.8 GB in bfloat16. One decode
> step reads all of them and does $2 \times 26.9 \times 10^9 = 53.8$ GFLOP.
> Divide each by its rate:
>
> $$
> t_{\text{memory}} = \frac{53.8 \times 10^{9}\ \text{bytes}}
> {1275 \times 10^{9}\ \text{byte/s}} \approx 42\ \text{ms}
> $$
>
> $$
> t_{\text{compute}} = \frac{53.8 \times 10^{9}\ \text{FLOP}}
> {312 \times 10^{12}\ \text{FLOP/s}} \approx 0.17\ \text{ms}
> $$
>
> The bandwidth here is 1275 GB/s, which is what a device-to-device copy
> actually measures on this card, not the 1935 it's rated for.

Both times are arithmetic, not a benchmark: they're lower bounds on what a
perfect implementation could do. The gap between them is the point. Decode
spends 250 times longer waiting for weights than using them, and the tensor
cores idle through all 42 ms.

## Prefill: many tokens per weight read

In prefill, the same weight matrix multiplies $T$ vectors at once. The FLOPs
scale with $T$, but the weight bytes don't, because one read of the matrix
serves all $T$ tokens. Add the activations to the bytes and divide:

$$
I_{\text{prefill}} = \frac{\hla{2nmT}}{\hlb{2nm} + \hlc{2mT}}
= \frac{nT}{n + T} \approx \boxed{T}
\quad\text{when } T \ll n
$$

The $\hlc{2mT}$ term is the activations, small compared with the weights
$\hlb{2nm}$ until the batch gets very large. A 2000-token prompt gives $I \approx 2000$, twelve times past the ridge
point, and prefill saturates the tensor cores.

Same matrices, same hardware, same kernels. The only thing that changes is how
many tokens share one read of the weights:

| Tokens in the forward pass | $I$ (FLOP/byte) | Bound by |
|---|---|---|
| 1 — decode, batch 1 | 1 | Memory, by 161x |
| 16 | 16 | Memory, by 10x |
| 161 | 161 | Exactly the ridge point |
| 2048 — prefill | ~2000 | Compute |

> [!NOTE] FLOPs per parameter isn't FLOPs per byte
> You'll see "2 FLOPs per parameter" quoted as decode's intensity. That's per
> *parameter*. In bfloat16 each parameter costs two bytes, so per byte it's 1.
> Chapter 10 does this calculation again per operation and finds the one place
> where the answer is neither 1 nor $T$: attention during decode sits at exactly
> 6, the GQA group size, no matter the batch size.

## What follows from the ratio

Each family of optimization in the course attacks a different term of the
intensity fraction:

- **Raise $T$.** One read of the weights serves many sequences, so intensity
  rises with the number of tokens in flight. This is why continuous batching
  exists, and it's the single largest throughput lever in the engine.
  Chapter 16.
- **Shrink the bytes.** Decode's time is bytes divided by bandwidth, so reading
  half as much runs twice as fast. This is why grouped-query attention,
  quantization, paged caches, and linear attention exist. Chapters 7, 15, and
  18.
- **Get more than one token per read.** If a step can commit several tokens,
  the cost per token drops even though intensity per step doesn't. This is why
  speculative decoding exists. Chapter 20.

Nothing in the course is arbitrary: ==each technique moves one term of one
fraction==.

## The model is a hybrid, and that matters

Qwen3.8-27B isn't a stack of identical transformer blocks. Its 64 layers
alternate between two kinds of token mixer:

| Layer type | Count | State per sequence | Cost per token |
|---|---|---|---|
| Full attention, grouped-query | 16 | Grows with context | 4 KiB per layer |
| Gated delta linear attention | 48 | 147.8 MiB, fixed | 0 |

Every fourth layer is full attention. The rest use a linear attention that keeps
a fixed-size matrix, a recurrent state, instead of a growing cache.

The consequence is large:

- A 64-layer model with full attention everywhere would need 256 KiB of cache
  per token.
- This one needs 64 KiB per token, plus a one-time 147.8 MiB per sequence.
- Past about 788 tokens of context the hybrid is ahead, and ==the gap widens
  without limit==: at 32k context it needs 2.14 GiB per sequence against
  8.0 GiB.

Chapter 2 derives all four of those numbers. You implement both mixers;
chapter 6 spends a while on the linear attention, the more interesting one.

## What you build

The engine comes together in this order:

1. **Read the weights.** Safetensors, shards, and the tensor names that tell you
   what the architecture really is. Chapter 1.
2. **Do the arithmetic on paper.** Parameter counts, cache growth, and the
   roofline. Every later optimization is judged against these numbers.
   Chapter 2.
3. **Write the layers.** RMSNorm, rotary embeddings, the delta rule,
   grouped-query attention, and the gated MLP, in PyTorch first. Chapters 3
   to 7.
4. **Assemble a forward pass** and check its logits against Hugging Face
   `transformers`. This is the last point where you have ground truth for free,
   so it's worth getting right. Chapter 8.
5. **Add a cache** and turn that $O(n^2)$ loop into an $O(n)$ one. Chapter 9.
6. **Measure the roofline** and learn to tell a slow kernel from a memory-bound
   one. Chapter 10.
7. **Sample.** Temperature, top-*k*, top-*p*, and the numerical care they need.
   Chapter 11.
8. **Write kernels.** One CUDA kernel by hand to see what the hardware wants,
   then Triton for the rest: fused normalization, FlashAttention, paged
   attention. Chapters 12 to 15.
9. **Schedule.** Continuous batching, chunked prefill, and preemption.
   Chapter 16.
10. **Benchmark.** Time to first token, inter-token latency, throughput, and
    what each one hides. Chapter 17.
11. **Scale.** Quantization, tensor parallelism, and speculative decoding.
    Chapters 18 to 20.

By the end, the pieces fit together as one step that repeats:

1. A request arrives at the server.
2. The scheduler decides whether it joins the current batch, and whether its
   prompt is chunked across several steps.
3. The model runner gathers the batch's token IDs, looks up their embeddings,
   and walks the 64 layers. It reads each sequence's KV blocks from the paged
   cache and each sequence's recurrent state from a fixed slot.
4. The sampler turns the final logits into one token per sequence.
5. The scheduler appends those tokens, frees any sequence that finished, admits
   whatever is waiting, and runs the next step.

Every chapter builds one of those nouns.

## How a lab run reaches the GPU

Each chapter ends with a lab. When you edit code in the browser and click
**Run**, the following happens:

1. The backend packages your file together with the lab's test harness.
2. It ships both to a RunPod serverless endpoint.
3. The provider starts a container with an A100 80GB attached, imports your file
   as a module, and calls the harness's `run(submission)` with it.
4. Output streams back to the browser line by line as it's printed, so a
   long-running benchmark shows its progress instead of a spinner.

The harness prints one line per check:

```text
[PASS] 16 layers use full attention
[FAIL] KV cache grows 64 KiB per token — got 262,144 bytes
      kv_kib_per_token = 256

7/8 checks passed
```

==Correctness is checked before speed, always.== A kernel that returns wrong
answers quickly fails, which is the correct outcome and happens to everyone.

Lines beginning with a metric name are recorded against your progress, so you
can watch a number improve across attempts. That's where chapter 17's
benchmarks come from.

> [!TIP]
> Some labs run on CPU, because arithmetic about memory doesn't need a GPU to be
> worth doing. Those start in a couple of seconds instead of a minute. A lab's
> header says which it is.

## What goes wrong

This chapter's lab most often trips on these:

- **Comparing against the rated bandwidth.** The A100's 1935 GB/s is a rating;
  a copy measures 1275 GB/s. Judge kernels against the measured figure, or
  good kernels look broken.
- **Mixing units in the ridge point.** Divide FLOP/s by byte/s. If you divide
  312 by 1935 without converting TFLOP/s and GB/s to base units, you get 0.161
  instead of 161.
- **Assuming every run gets the same card.** Runs land on one of two A100
  variants with different rated bandwidths, so their ridge points differ. The
  lab prints which one you got.

> [!RECAP]
> - The forward pass is a pure function from token IDs to one score per
>   vocabulary entry per position. Generation calls it in a loop.
> - Prefill runs the whole prompt once; decode runs one token per step against
>   a cache.
> - The A100's ridge point is about 161 FLOP/byte. Decode at batch 1 sits at 1,
>   and prefill sits at about $T$.
> - One decode step of the whole model needs about 42 ms for memory and 0.17 ms
>   for compute, so bytes, not FLOPs, set its speed.
> - The hybrid model's linear-attention layers keep a fixed state, which cuts
>   cache growth from 256 KiB to 64 KiB per token.

## Check your understanding

> [!QUESTION] Why does batching help decode but not change prefill much?
> Decode's intensity is limited by how many tokens share one read of the
> weights, and at batch 1 that number is 1. Adding sequences multiplies the
> FLOPs while the weight bytes stay fixed, so intensity rises roughly linearly
> with batch size. Prefill already has thousands of tokens sharing each read, so
> it's past the ridge point and more tokens only add proportional work.

> [!QUESTION] A100 memory bandwidth is rated at 1935 GB/s but a copy measures 1275. Which should you compare a kernel against?
> The 1275. It's what the simplest possible bandwidth-bound kernel achieves on
> this hardware, so it's the real ceiling. Comparing against the rating makes
> good kernels look broken; chapter 10 has the numbers.

> [!QUESTION] If the weights were int8 instead of bfloat16, what would decode's batch-1 intensity become?
> Two. The FLOP count is unchanged at $2nm$, but the bytes halve to $nm$, so
> $I = 2$. That's still 80 times below the ridge point: quantization makes
> decode faster by halving the bytes, not by making it compute bound. Chapter 18
> covers what it costs in quality.

## Lab

> [!TRY]
> Confirm the plumbing works before anything harder depends on it. Write two
> functions:
>
> - `gpu_report`, which reads `torch.cuda.get_device_properties(0)` and returns
>   the device name, memory in GB, SM count, and compute capability.
> - `ridge_point`, which converts the two published peak rates into base units
>   and divides.
>
> You pass when a CUDA device is visible, the report has all four keys, the
> device has at least 39 GB and a nonzero SM count, and your ridge point lands
> within 2 of 161.2.

The harness also prints which A100 you were given. The SXM4 module is rated at
2039 GB/s rather than 1935, so its ridge point is 153, not 161. Runs land on
either variant, and chapter 10 covers what that does to your measurements.

## Further reading

- [Efficient memory management for large language model serving with PagedAttention](https://arxiv.org/abs/2309.06180) — the vLLM paper.
- [FlashAttention: fast and memory-efficient exact attention with IO-awareness](https://arxiv.org/abs/2205.14135)
- [Gated delta networks: improving Mamba2 with delta rule](https://arxiv.org/abs/2412.06464)
- [Making deep learning go brrrr from first principles](https://horace.io/brrr_intro.html) — arithmetic intensity, at length.
- [Attention is all you need](https://arxiv.org/abs/1706.03762)
