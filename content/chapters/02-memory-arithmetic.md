---
title: Memory arithmetic
slug: 02-memory-arithmetic
part: "Part 1 — Ground truth"
summary: Deriving the parameter count, the weight footprint, KV growth per token, the recurrent state, and the break-even context length.
minutes: 75
gpu: false
objectives:
  - Count a model's parameters component by component from its config alone.
  - Predict the weight footprint in bfloat16 and what it leaves on an 80 GB card.
  - Compute KV cache growth per token and the fixed cost of a recurrent state.
  - Solve for the context length where a hybrid cache beats a dense one.
  - Decide the largest batch that fits a given GPU at a given context length.
lab: 02-memory-math
---

# Memory arithmetic

An A100 80GB gives you 80 GiB, minus what the driver keeps. Everything must fit:
weights, cache, activations, and whatever the allocator wastes on fragmentation.
This chapter works out where it all goes.

Every number here is arithmetic, not measurement. You can check each one on
paper, and the lab makes you write them as code. They are also the numbers every
later chapter is judged against: chapter 15 exists because of the KV figure,
chapter 18 because of the weight figure, and chapter 16 because of the batch
figure.

## Before you start

Four conventions, then the counting.

**A linear layer's parameter count is the product of its shape.** PyTorch stores
`nn.Linear` weights as `(out_features, in_features)`, so a projection from 5120
to 17408 holds $5120 \times 17408$ parameters. This model has no biases on its
projections, which is normal for modern decoders, so there is nothing to add.

**Bytes are parameters times the element size.** bfloat16 is 2 bytes per
parameter, and the whole checkpoint is bfloat16. Chapter 18 changes this number
and nothing else.

**Units.** Weights are quoted in decimal GB, because that is how model sizes are
written everywhere. Caches are quoted in binary KiB, MiB, and GiB, because they
come from power-of-two shapes. $1\ \mathrm{GB} = 10^9$ bytes and
$1\ \mathrm{GiB} = 2^{30}$ bytes; the difference is 7%, which is large enough to
matter in a budget. The
[notation chapter](/c/00a-notation-and-prerequisites) has the full convention.

**The geometry.** Everything below comes from these config fields:

| Symbol | Field | Value |
|---|---|---|
| $d$ | `hidden_size` | 5120 |
| $N$ | `num_hidden_layers` | 64 |
| $h$ | `num_attention_heads` | 24 |
| $h_{kv}$ | `num_key_value_heads` | 4 |
| $d_h$ | `head_dim` | 256 |
| $d_{ff}$ | `intermediate_size` | 17408 |
| $V$ | `vocab_size` | 248320 |
| — | `full_attention_interval` | 4 |
| — | `attn_output_gate` | true |
| — | `linear_num_key_heads` | 16 |
| — | `linear_num_value_heads` | 48 |
| — | `linear_key_head_dim`, `linear_value_head_dim` | 128 |
| — | `linear_conv_kernel_dim` | 4 |
| $b$ | bytes per element | 2 |

Full attention lands on every fourth layer, counting so that layer index 3 is
the first:

$$
N_{\text{full}} = \frac{64}{4} = 16, \qquad N_{\text{lin}} = 64 - 16 = 48
$$

## Counting the parameters

Five components. Take them one at a time and keep the arithmetic visible.

### Embedding and output projection

The embedding is a $V \times d$ matrix, one row per vocabulary entry:

$$
248{,}320 \times 5120 = 1{,}271{,}398{,}400 \approx 1.27\text{B}
$$

`tie_word_embeddings` is false, so the output projection is a separate matrix of
the same shape. Together:

$$
2 \times 1{,}271{,}398{,}400 = 2{,}542{,}796{,}800 \approx 2.54\text{B}
$$

That is 9.5% of the model spent entirely on the vocabulary, and none of it does
any reasoning. A 248k vocabulary is unusually large; it buys shorter token
sequences for non-English text, and it costs 2.5 GB of the 80 you have.

### The MLP, in every layer

SwiGLU needs three matrices: a gate and an up projection, each $d \times d_{ff}$,
and a down projection, $d_{ff} \times d$. Per layer:

$$
3 \times 5120 \times 17408 = 267{,}386{,}880 \approx 267.4\text{M}
$$

Every one of the 64 layers has one, regardless of which token mixer it uses:

$$
64 \times 267{,}386{,}880 = 17{,}112{,}760{,}320 \approx 17.11\text{B}
$$

The MLP is 64% of the model. It is also the part with the simplest cost model —
three GEMMs — which is why chapter 10 uses it as the roofline example.

### Full-attention layers

Chapter 1 read these shapes off the checkpoint. Now count them.

The query projection emits queries *and* a gate, so its output width is
$2 \times h \times d_h$:

$$
5120 \times (2 \times 24 \times 256) = 5120 \times 12288 = 62{,}914{,}560
$$

Keys and values have only $h_{kv} = 4$ heads each:

$$
2 \times \bigl(5120 \times 4 \times 256\bigr) = 2 \times 5{,}242{,}880
= 10{,}485{,}760
$$

The output projection maps the 24 heads' concatenated width back to hidden:

$$
(24 \times 256) \times 5120 = 6144 \times 5120 = 31{,}457{,}280
$$

Adding the three:

$$
62{,}914{,}560 + 10{,}485{,}760 + 31{,}457{,}280 = 104{,}857{,}600
\approx 104.9\text{M}
$$

Without the output gate the query projection would be half its size and the block
would hold 73.4M. The gate costs 31.5M per layer, 0.5B across the model — worth
knowing before you decide it is a typo.

### Linear-attention layers

The fused `in_proj_qkvz` produces four things. Queries and keys use
$16 \times 128 = 2048$ channels each; values and the `z` gate use
$48 \times 128 = 6144$ each:

$$
2048 + 2048 + 6144 + 6144 = 16384
$$

$$
5120 \times 16384 = 83{,}886{,}080
$$

The depthwise convolution runs over the q, k, and v channels only — not `z` —
with a kernel of width 4. One filter per channel, so it is a sum, not a product
of head counts:

$$
(2048 + 2048 + 6144) \times 4 = 10{,}240 \times 4 = 40{,}960
$$

`in_proj_ba` produces one $\beta$ and one $a$ per value head, where $\beta$ is
the delta rule's write strength and $a$ its forget gate:

$$
5120 \times (2 \times 48) = 5120 \times 96 = 491{,}520
$$

The output projection maps the 6144 value channels back to hidden:

$$
6144 \times 5120 = 31{,}457{,}280
$$

Per layer:

$$
83{,}886{,}080 + 40{,}960 + 491{,}520 + 31{,}457{,}280 = 115{,}875{,}840
\approx 115.9\text{M}
$$

A linear-attention block is 11M *larger* than a full-attention block. Linear
attention is not cheaper in parameters — it is cheaper in state, which is the
point of the whole design and the subject of the next three sections.

### Norms

Two RMSNorms per layer, each a vector of length $d$, plus one final norm:

$$
64 \times 2 \times 5120 + 5120 = 655{,}360 + 5120 = 660{,}480
$$

Rounding error, at 0.0025% of the model. Count it anyway; the lab does.

### The total

Full-attention layers carry an attention block and an MLP; linear layers carry a
linear-attention block and an MLP:

$$
16 \times (104{,}857{,}600 + 267{,}386{,}880) = 5{,}955{,}911{,}680
$$

$$
48 \times (115{,}875{,}840 + 267{,}386{,}880) = 18{,}396{,}610{,}560
$$

Adding the embeddings and the norms:

| Component | Parameters | Share |
|---|---|---|
| Embedding and output projection | 2,542,796,800 | 9.5% |
| Full-attention layers, 16 | 5,955,911,680 | 22.1% |
| Linear-attention layers, 48 | 18,396,610,560 | 68.4% |
| Norms | 660,480 | 0.0% |
| **Total** | **26,895,979,520** | **100%** |

26.9B, which matches the name on the box. In bfloat16, at 2 bytes per parameter:

$$
26{,}895{,}979{,}520 \times 2 = 53{,}791{,}959{,}040\ \text{bytes}
= 53.8\ \mathrm{GB} = 50.1\ \mathrm{GiB}
$$

That leaves roughly 30 GiB on an 80 GiB card before anything else is allocated.
An A100 40GB cannot hold the weights at all, which is why this course uses the
80GB part and why the lab's `max_batch_size` returns 0 for a 40 GiB budget.

## KV cache

A full-attention layer stores one key and one value per token, per KV head, each
$d_h$ wide. Per token, per layer:

$$
\underbrace{2}_{K \text{ and } V} \times
\underbrace{4}_{h_{kv}} \times
\underbrace{256}_{d_h} \times
\underbrace{2}_{\text{bytes}} = 4096\ \text{bytes} = 4\ \mathrm{KiB}
$$

Only the 16 full-attention layers pay it. The 48 linear layers contribute
nothing, because their state does not grow with the sequence:

$$
4096 \times 16 = 65{,}536\ \text{bytes} = 64\ \mathrm{KiB\ per\ token}
$$

Grouped-query attention is already doing heavy lifting. Each KV head serves
$h / h_{kv} = 6$ query heads. Without it — 24 KV heads instead of 4 — the same
16 layers would cost

$$
2 \times 24 \times 256 \times 2 \times 16 = 393{,}216\ \text{bytes}
= 384\ \mathrm{KiB\ per\ token}
$$

six times larger, for the same model. Chapter 7 covers what that costs in
quality.

Cache size scales with batch and context, and with nothing else:

| Context | Batch 1 | Batch 8 | Batch 32 |
|---|---|---|---|
| 4k | 0.25 GiB | 2.0 GiB | 8.0 GiB |
| 32k | 2.0 GiB | 16.0 GiB | 64.0 GiB |
| 128k | 8.0 GiB | 64.0 GiB | 256.0 GiB |

Read the bottom-right cell again: 256 GiB of cache for 32 concurrent 128k
sequences, on a card with 80. The cache, not the weights, sets your maximum batch
size, and that single fact is why chapter 15 is about packing it efficiently.

## Recurrent state

The 48 linear-attention layers keep a fixed-size state instead of a growing
cache. Chapter 6 derives what is in it; here you only need its shape.

The delta-rule state is one matrix per value head, of size
$\text{key dim} \times \text{value dim}$, and it is kept in float32 for numerical
stability — a bfloat16 accumulator stops absorbing small updates after a few
hundred tokens, for the reason the
[notation chapter](/c/00a-notation-and-prerequisites) works through. Per layer:

$$
48 \times 128 \times 128 \times 4 = 3{,}145{,}728\ \text{bytes} = 3.0\ \mathrm{MiB}
$$

The causal convolution also needs the last 4 steps of its q, k, and v channels,
in the model's own dtype:

$$
\underbrace{(2 \times 16 \times 128 + 48 \times 128)}_{10{,}240\ \text{channels}}
\times 4 \times 2 = 81{,}920\ \text{bytes} = 80\ \mathrm{KiB}
$$

Per layer that is $3{,}145{,}728 + 81{,}920 = 3{,}227{,}648$ bytes, or 3.078 MiB.
Across 48 layers:

$$
3{,}227{,}648 \times 48 = 154{,}927{,}104\ \text{bytes} = 147.8\ \mathrm{MiB}
$$

It does not grow. A sequence at 128k context carries the same 147.8 MiB as a
sequence ten tokens long. That is the entire trade: a large constant in exchange
for a shallower slope.

## The break-even point

Compare the hybrid against a hypothetical model where all 64 layers use full
attention. The dense model's per-token cost is the same 4 KiB per layer, over
64 layers instead of 16:

$$
4096 \times 64 = 262{,}144\ \text{bytes} = 256\ \mathrm{KiB\ per\ token}
$$

Write both footprints as a function of context length $L$, in bytes per sequence:

$$
M_{\text{dense}}(L) = 262{,}144\,L
$$

$$
M_{\text{hybrid}}(L) = 65{,}536\,L + 154{,}927{,}104
$$

Two straight lines. The hybrid has the higher intercept and the shallower slope,
so they cross exactly once. Set them equal and solve for $L$:

$$
262{,}144\,L = 65{,}536\,L + 154{,}927{,}104
$$

$$
(262{,}144 - 65{,}536)\,L = 154{,}927{,}104
$$

$$
196{,}608\,L = 154{,}927{,}104
$$

$$
L = \frac{154{,}927{,}104}{196{,}608} = 788\ \text{tokens}
$$

The general form is worth remembering, because it is the only equation in this
chapter that is not a multiplication:

$$
L_{\text{break-even}}
= \frac{\text{fixed state per sequence}}{\text{KV bytes saved per token}}
$$

Below 788 tokens the fixed state costs more than the KV entries it replaces.
Above it the hybrid wins, and the margin grows by 192 KiB for every further
token. At 32k context the hybrid needs 2.14 GiB per sequence against 8.0 GiB; at
128k, 8.14 GiB against 32.0 GiB. The ratio tends to 4 as $L$ grows, which is
$64 / 16$ — the layer ratio, as it must be.

788 tokens is short. Almost every real request is longer than that, so in
practice the hybrid always wins; the break-even calculation exists to prove the
design is not free rather than to guide a decision.

`ModelConfig.hybrid_breakeven_tokens` computes this for any config, and the lab
has you derive it.

## The largest batch that fits

Now put the three quantities together. A sequence at context $L$ costs its
recurrent state plus its KV entries:

$$
M_{\text{seq}}(L) = 154{,}927{,}104 + 65{,}536\,L
$$

The GPU has to hold the weights and a fixed overhead first. Writing $C$ for the
card's capacity, $W$ for the weights, and $O$ for overhead:

$$
B_{\max} = \left\lfloor \frac{C - W - O}{M_{\text{seq}}(L)} \right\rfloor
$$

The lab uses $C = 80 \times 2^{30}$, $W = 53.8$ GB, and $O = 6\ \mathrm{GiB}$ for
the CUDA context, the driver, and an activation workspace. The numerator:

$$
85{,}899{,}345{,}920 - 53{,}800{,}000{,}000 - 6{,}442{,}450{,}944
= 25{,}656{,}894{,}976\ \text{bytes} = 23.9\ \mathrm{GiB}
$$

At 32k context, one sequence costs

$$
154{,}927{,}104 + 65{,}536 \times 32{,}768 = 2{,}302{,}410{,}752\ \text{bytes}
= 2.14\ \mathrm{GiB}
$$

$$
B_{\max} = \left\lfloor \frac{25{,}656{,}894{,}976}{2{,}302{,}410{,}752}
\right\rfloor = 11
$$

At 4k context, one sequence costs 423,362,560 bytes and $B_{\max} = 60$. On a
40 GiB card the numerator is negative before you divide anything, so the answer
is 0 at every context length.

Three things to take from that formula.

**The floor matters.** Memory is allocated in whole sequences; 11.14 sequences
is 11. Chapter 15's paged cache is what lets you stop rounding down so
aggressively, by allocating blocks instead of whole sequences.

**The fixed state is 7% of a 32k sequence and 37% of a 4k one.** The hybrid's
constant hurts most exactly where you want large batches, which is short
requests. This is a real cost, not a footnote.

**Every term is a lever.** Halving $W$ with int8 quantization moves the numerator
from 23.9 GiB to 48.9 GiB, which roughly doubles the batch. Chapter 18 covers
what that costs in quality, and why FP8 — which does it better — is not available
to you on Ampere.

## Activations

Weights and cache are persistent. Activations are transient, but not small, and
the allocator has to have room for them.

The MLP's hidden activation is `(tokens, 17408)` in bfloat16. For a 4096-token
prefill chunk:

$$
4096 \times 17{,}408 \times 2 = 142{,}606{,}336\ \text{bytes} = 136\ \mathrm{MiB}
$$

SwiGLU holds two of those at once — the gate branch and the up branch — before
the elementwise product frees one.

Attention scores are worse, if you materialize them. The score tensor is
`(heads, q_len, kv_len)`, and at 8k context with 24 heads:

$$
24 \times 8192 \times 8192 \times 2 = 3{,}221{,}225{,}472\ \text{bytes}
= 3.0\ \mathrm{GiB}
$$

for one sequence, and it grows with the square of context. Not materializing it
is exactly what FlashAttention is for, and chapter 14 gets this to zero by never
holding more than a tile. It is also why chunked prefill caps how many tokens
enter a forward pass at once: the cap bounds the activation peak.

## A budget that works

| Item | Size |
|---|---|
| Weights, bf16 | 50.1 GiB |
| CUDA context and driver | ~1.0 GiB |
| Activation workspace | ~3.0 GiB |
| Fragmentation headroom | ~2.0 GiB |
| **Left for cache** | **~23.9 GiB** |

23.9 GiB of pure KV is 391k tokens at 64 KiB each. Once you subtract each
sequence's 147.8 MiB of recurrent state, that is batch 11 at 32k context, batch
60 at 4k, or any mix in between. `PagedKVCache` in chapter 15 turns the budget
into a block pool so a short request does not reserve space for a length it never
reaches.

## What goes wrong

**Confusing GB with GiB.** 53.8 GB of weights is 50.1 GiB. Subtracting 53.8 from
80 when both are GiB loses 3.7 GiB, which is two more sequences at 32k. The lab's
tests use binary units for the card and decimal for the weights, exactly as this
chapter does, because that is what the tools report.

**Forgetting that the KV cache is per sequence, not per batch.** It is easy to
compute 2 GiB at 32k and treat it as the total. It is per sequence, so batch 8 is
16 GiB and the card is full.

**Sizing the cache for the average request.** Requests arrive at whatever length
they arrive. An engine that assumes 4k and meets a 32k prompt must either preempt
something or run out of memory mid-forward-pass, which is an out-of-memory error
inside a CUDA kernel and is not gracefully recoverable. Chapter 16 handles this
with admission control.

**Ignoring fragmentation.** The 2 GiB of headroom above is not superstition. A
caching allocator that has served many different sizes ends up unable to satisfy
a large contiguous request even when the free total is sufficient. Paged
allocation exists partly to avoid this.

## Check your understanding

**If the model used 24 KV heads instead of 4, how much cache would a batch of 8
at 32k context need?** Per token the cost is 384 KiB, six times the 64 KiB
figure, so per sequence at 32,768 tokens it is 12 GiB and for 8 sequences 96 GiB.
That does not fit on the card alongside 50 GiB of weights. Grouped-query
attention is not an optimization here; it is what makes the configuration
possible at all.

**Why is the break-even independent of batch size?** Because both sides of the
equation scale linearly with the number of sequences — each sequence has its own
recurrent state and its own KV entries — so the batch factor cancels. It depends
only on the per-sequence geometry.

**A colleague proposes keeping the recurrent state in bfloat16 to save 74 MiB per
sequence. What breaks?** The state is a running sum over thousands of tokens. In
bfloat16, an addend smaller than about $1/256$ of the accumulated magnitude
rounds away entirely, so the state stops updating from the tail of the sequence
while still looking finite and plausible. The saving is also small: 74 MiB per
sequence is 3% of a 32k sequence's footprint, and it does not change the batch
size at 32k at all.

**How would the break-even move if `full_attention_interval` were 8?** There
would be 8 full-attention layers and 56 linear ones. The hybrid's per-token cost
halves to 32 KiB, so the saving per token rises to 224 KiB, while the fixed state
grows to $56 \times 3{,}227{,}648 = 180{,}748{,}288$ bytes. The break-even is
$180{,}748{,}288 / 229{,}376 = 788$ tokens — unchanged, because both the state and
the saving scale with the same layer split. That invariance is a good check that
you have the formula right.

## Lab

Write the arithmetic as code. The harness hands every function a config dict with
the fields from the table at the top of this chapter.

You implement eight functions: `num_full_attention_layers`,
`mlp_params_per_layer`, `embedding_params`, `kv_bytes_per_token`,
`dense_kv_bytes_per_token`, `recurrent_state_bytes`, `breakeven_tokens`, and
`max_batch_size`.

The harness checks each against the real model: 16 full-attention layers, 267.4M
parameters in one MLP block, 2.54B in the embedding and output projection
together, 64 KiB of KV per token, 256 KiB for an all-full-attention model,
147.8 MiB of recurrent state per sequence to within a kilobyte, and a break-even
within 2 tokens of 788. For `max_batch_size` it checks that 32k context gives a
batch between 5 and 12, that 4k context gives a larger batch than 32k, and that
an A100 40GB returns 0 because the weights alone do not fit.

It also reports `params_billions`, `kv_kib_per_token`, `recurrent_state_mib`,
`breakeven_tokens`, and both batch sizes as metrics, so you can compare them
against the figures derived above.

The lab runs on CPU. Arithmetic about memory does not need a GPU to be worth
doing.

## Further reading

- [NVIDIA A100 tensor core GPU architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf)
- [GQA: training generalized multi-query transformer models from multi-head checkpoints](https://arxiv.org/abs/2305.13245)
- [Fast transformer decoding: one write-head is all you need](https://arxiv.org/abs/1911.02150) — multi-query attention, the limiting case of GQA.
- [Efficient memory management for large language model serving with PagedAttention](https://arxiv.org/abs/2309.06180)
