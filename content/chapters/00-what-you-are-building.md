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

By the end of this course you have an inference engine: a program that loads open
weights and turns prompts into tokens, quickly, for many users at once. You write
the attention kernels, the cache, the scheduler, and the server. Nothing is
imported from vLLM or TensorRT.

The target model is [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). It
runs on a single NVIDIA A100 80GB, which you rent by the second from a GPU
provider.

## What you need to know before starting

The course assumes university linear algebra and calculus, and nothing about
GPUs, transformers, or serving. Everything else it assumes is collected in
[Notation and prerequisites](/c/00a-notation-and-prerequisites): tensor
shape conventions, `torch.einsum`, softmax and its shift invariance, what
bfloat16 actually stores, the GPU vocabulary, and a table of the symbols later
chapters use.

Read it now if any of these are unfamiliar: arithmetic intensity, warp, ULP,
coalescing, KiB against KB. Skim it otherwise and come back when a symbol stops
making sense. This chapter uses two things from it — the arithmetic-intensity
ratio and the GB/GiB convention.

You also need working Python and PyTorch familiarity: tensor indexing,
`reshape`, `transpose`, broadcasting, and `@` for matrix multiply. You do not
need CUDA; chapter 12 starts from nothing.

## What a forward pass computes

An inference engine is built around one function. It takes a sequence of token
IDs and returns, for each position, a score for every word in the vocabulary.
That is all. Everything else in the engine exists to call this function often,
cheaply, and for many users at once.

Concretely, for a prompt of $T$ tokens:

1. **Embed.** Each token ID indexes a row of a $V \times d$ matrix, where
   $V = 248{,}320$ is the vocabulary size and $d = 5120$ is the hidden size. The
   result is a tensor of shape `(T, 5120)`. This is the *residual stream*: one
   vector per token, carrying everything the model knows about that position so
   far.
2. **Run 64 layers.** Each layer reads the residual stream, computes a
   correction, and adds it back. A layer has two halves. The *token mixer* lets
   positions see each other — this is attention, or in this model, sometimes a
   recurrence. The *MLP* transforms each position independently. Both are
   wrapped in normalization, and both add to the stream rather than replacing
   it.
3. **Project to logits.** A final normalization, then a $d \times V$ matrix
   turns each position's 5120-vector into 248,320 scores, one per vocabulary
   entry. The result has shape `(T, 248320)`.

Nothing in that list is sequential across tokens except step 2's token mixer,
and even there the dependency is one-directional: position $t$ may read
positions $\leq t$, never the future. That restriction is *causal masking*, and
it is what makes the next section possible.

The forward pass has no memory of its own. It is a pure function of the token
IDs you hand it. Every bit of "state" in a chat session is the token sequence
itself, plus caches that exist only to avoid recomputing what the function
already computed.

## What autoregressive means, concretely

The model scores every position, but only the last one is useful for generating:
its logits describe what should come next. Generation is a loop.

```python
tokens = tokenizer.encode(prompt)        # e.g. 2000 token IDs
for _ in range(max_new_tokens):
    logits = model(tokens)               # (len(tokens), 248320)
    next_id = sample(logits[-1])         # one token ID
    tokens.append(next_id)
    if next_id == eos_id:
        break
```

Read that loop carefully, because its cost structure is the whole course.

The first iteration processes 2000 tokens. Every iteration after it processes
2001, then 2002, and so on — and recomputes, from scratch, everything it already
computed. Generating 500 tokens this way runs the model over roughly 1.1 million
token positions to produce 500 useful ones. That is the $O(n^2)$ loop chapter 9
replaces.

The fix is to notice that layer $\ell$'s output at position $t$ depends only on
positions $\leq t$, and those positions have not changed. So you keep each
layer's intermediate keys and values in a cache, feed the loop one new token per
step, and every step becomes $O(1)$ in the number of tokens processed — though
not in bytes read, as the next section shows.

That split gives generation its two phases, which behave nothing alike:

- **Prefill** is the first iteration: the whole prompt, at once.
- **Decode** is every iteration after it: one token, against a cache.

The word *autoregressive* means exactly this — the model's own output is
appended to its input and fed back. There is no way around the loop. You cannot
produce token 10 without having produced token 9, which is why a 500-token reply
takes 500 sequential steps no matter how large your GPU is.

## The two phases, derived

The reason prefill and decode need different engineering is not a rule of thumb.
It falls out of one ratio.

Define *arithmetic intensity* as the FLOPs an operation performs divided by the
bytes it must move between HBM and the chip:

$$
I = \frac{\text{FLOPs}}{\text{bytes moved}}
$$

A GPU has two peak rates. An A100 80GB delivers about 312 TFLOP/s in bfloat16
and has a rated memory bandwidth of 1935 GB/s. Their ratio is the *ridge point*:

$$
I_{\text{ridge}} = \frac{312 \times 10^{12}\ \text{FLOP/s}}
{1935 \times 10^{9}\ \text{byte/s}} \approx 161\ \text{FLOP/byte}
$$

An operation with $I$ below 161 cannot saturate the tensor cores. It finishes its
arithmetic before the next bytes arrive, and no amount of kernel cleverness
changes that — only moving fewer bytes does.

**Decode, batch 1.** Almost all the work is multiplying a weight matrix by a
single vector. For an $n \times m$ weight matrix in bfloat16:

$$
\text{FLOPs} = 2nm, \qquad \text{bytes} = 2nm
$$

The $2$ in the FLOP count is one multiply plus one add per weight. The $2$ in
the byte count is bfloat16's two bytes per weight. They cancel:

$$
I_{\text{decode}} = \frac{2nm}{2nm} = 1\ \text{FLOP/byte}
$$

Every weight is read from HBM, used for exactly one multiply-add, and discarded.
Against a ridge point of 161, decode at batch 1 uses about $1/161$ of the A100's
arithmetic — under 1%.

It is worth doing the same sum for the whole model rather than one matrix.
Chapter 2 counts 26.9B parameters, which is 53.8 GB in bfloat16. One decode step
reads all of them and does $2 \times 26.9 \times 10^9 = 53.8$ GFLOP:

$$
t_{\text{memory}} = \frac{53.8 \times 10^{9}\ \text{bytes}}
{1275 \times 10^{9}\ \text{byte/s}} \approx 42\ \text{ms}
$$

$$
t_{\text{compute}} = \frac{53.8 \times 10^{9}\ \text{FLOP}}
{312 \times 10^{12}\ \text{FLOP/s}} \approx 0.17\ \text{ms}
$$

The bandwidth figure there is 1275 GB/s, which is what a device-to-device copy
actually measures on this card, not the 1935 it is rated for. Both numbers are
arithmetic, not a benchmark: they are lower bounds on what a perfect
implementation could do. The gap between them is the point. Decode spends
250 times longer waiting for weights than using them, and the tensor cores idle
through all 42 ms.

**Prefill.** Now the same weight matrix multiplies $T$ vectors at once. The FLOPs
scale with $T$; the bytes do not, because one read of the matrix serves all $T$
tokens:

$$
I_{\text{prefill}} = \frac{2nmT}{2nm + 2mT} \approx T
\quad\text{when } T \ll n
$$

The denominator's second term is the activations, small compared with the
weights until the batch gets very large. So intensity is roughly the token
count. A 2000-token prompt gives $I \approx 2000$, twelve times past the ridge
point, and prefill saturates the tensor cores.

Same matrices, same hardware, same kernels. The only thing that changed is how
many tokens share one read of the weights:

| Tokens in the forward pass | $I$ (FLOP/byte) | Bound by |
|---|---|---|
| 1 — decode, batch 1 | 1 | Memory, by 161x |
| 16 | 16 | Memory, by 10x |
| 161 | 161 | Exactly the ridge point |
| 2048 — prefill | ~2000 | Compute |

You will see "2 FLOPs per parameter" quoted as decode's intensity. That is FLOPs
per *parameter*, not per byte; in bfloat16 each parameter costs two bytes, so
per byte it is 1. Chapter 10 does this calculation again per operation, and
finds the one place where the answer is neither 1 nor $T$: attention during
decode sits at exactly 6, the GQA group size, no matter the batch size.

## What follows from the ratio

Three families of optimization, and each one attacks a different term:

- **Raise $T$.** One read of the weights serves many sequences, so intensity
  rises with the number of tokens in flight. This is why continuous batching
  exists, and it is the single largest throughput lever in the engine.
  Chapter 16.
- **Shrink the bytes.** Decode's time is bytes divided by bandwidth, so reading
  half as much runs twice as fast. This is why grouped-query attention,
  quantization, paged caches, and linear attention exist. Chapters 7, 15, and
  18.
- **Get more than one token per read.** If a step can commit several tokens,
  the cost per token drops even though intensity per step does not. This is why
  speculative decoding exists. Chapter 20.

Nothing in the course is arbitrary. Each technique moves one term of one
fraction.

## The model is a hybrid, and that matters

Qwen3.8-27B is not a stack of identical transformer blocks. Its 64 layers
alternate between two kinds of token mixer:

| Layer type | Count | State per sequence | Cost per token |
|---|---|---|---|
| Full attention, grouped-query | 16 | Grows with context | 4 KiB per layer |
| Gated delta linear attention | 48 | 147.8 MiB, fixed | 0 |

Every fourth layer is full attention. The rest use a linear attention that keeps
a fixed-size matrix — a recurrent state — instead of a growing cache.

The consequence is large. A 64-layer model with full attention everywhere would
need 256 KiB of cache per token. This one needs 64 KiB per token, plus a
one-time 147.8 MiB per sequence. Past about 788 tokens of context the hybrid is
ahead, and the gap widens without limit: at 32k context it needs 2.14 GiB per
sequence against 8.0 GiB. Chapter 2 derives all four of those numbers.

You implement both mixers. The linear attention is the more interesting one, and
chapter 6 spends a while on it.

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
   so it is worth getting right. Chapter 8.
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

By the end, the pieces fit together like this. A request arrives at the server.
The scheduler decides whether it joins the current batch, and whether its prompt
is chunked across several steps. The model runner gathers the batch's token IDs,
looks up their embeddings, and walks the 64 layers, reading each sequence's KV
blocks from the paged cache and each sequence's recurrent state from a fixed
slot. The sampler turns the final logits into one token per sequence. The
scheduler appends those tokens, frees any sequence that finished, admits
whatever is waiting, and runs the next step. Every chapter builds one of those
nouns.

## How a lab run reaches the GPU

Each chapter ends with a lab. You edit code in the browser and click **Run**.

What happens next: the backend packages your file together with the lab's test
harness and ships both to a GPU provider — RunPod by default, Modal as the
alternative; the app decides and you can override it in the lab pane. The
provider starts a container with an A100 80GB attached, imports your file as a
module, and calls the harness's `run(submission)` with it. Output streams back
to the browser line by line as it is printed, so a long-running benchmark shows
its progress instead of a spinner.

The harness prints one line per check:

```text
[PASS] 16 layers use full attention
[FAIL] KV cache grows 64 KiB per token — got 262,144 bytes
      kv_kib_per_token = 256

7/8 checks passed
```

Correctness is checked before speed, always. A kernel that returns wrong answers
quickly fails, which is the correct outcome and happens to everyone.

Lines beginning with a metric name are recorded against your progress, so you
can watch a number improve across attempts. That is where chapter 17's
benchmarks come from.

Some labs run on CPU, because arithmetic about memory does not need a GPU to be
worth doing. Those start in a couple of seconds instead of a minute. A lab's
header says which it is.

## Check your understanding

**Why does batching help decode but not change prefill much?** Because decode's
intensity is limited by how many tokens share one read of the weights, and at
batch 1 that number is 1. Adding sequences multiplies the FLOPs while the weight
bytes stay fixed, so intensity rises roughly linearly with batch size. Prefill
already has thousands of tokens sharing each read, so it is past the ridge point
and more tokens only add proportional work.

**A100 memory bandwidth is rated at 1935 GB/s but a copy measures 1275. Which
should you compare a kernel against?** The 1275. It is what the simplest
possible bandwidth-bound kernel achieves on this hardware, so it is the real
ceiling. Comparing against the rating makes good kernels look broken; chapter 10
has the numbers.

**If the weights were int8 instead of bfloat16, what would decode's batch-1
intensity become?** Two. The FLOP count is unchanged at $2nm$, but the bytes
halve to $nm$, so $I = 2$. Still 80 times below the ridge point — quantization
makes decode faster by halving the bytes, not by making it compute bound.
Chapter 18 covers what it costs in quality.

## Lab

The first lab does nothing but confirm the plumbing works before anything harder
depends on it. You write two functions: `gpu_report`, which reads
`torch.cuda.get_device_properties(0)` and returns the device name, memory in GB,
SM count, and compute capability; and `ridge_point`, which converts the two
published peak rates into base units and divides.

The harness checks that a CUDA device is visible, that the report has all four
keys, that the device has at least 39 GB and a nonzero SM count, and that your
ridge point lands within 2 of 161.2. It also prints which A100 you were given:
the SXM4 module is rated at 2039 GB/s rather than 1935, so its ridge point is
153, not 161. Runs land on either variant, and chapter 10 covers what that does
to your measurements.

## Further reading

- [Efficient memory management for large language model serving with PagedAttention](https://arxiv.org/abs/2309.06180) — the vLLM paper.
- [FlashAttention: fast and memory-efficient exact attention with IO-awareness](https://arxiv.org/abs/2205.14135)
- [Gated delta networks: improving Mamba2 with delta rule](https://arxiv.org/abs/2412.06464)
- [Making deep learning go brrrr from first principles](https://horace.io/brrr_intro.html) — arithmetic intensity, at length.
- [Attention is all you need](https://arxiv.org/abs/1706.03762)
