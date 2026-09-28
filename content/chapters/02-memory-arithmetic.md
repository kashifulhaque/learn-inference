---
title: Memory arithmetic
slug: 02-memory-arithmetic
part: "Part 1 — Ground truth"
summary: Counting the parameters and the weight footprint, then the per-token KV cache, the fixed recurrent state, the context length where the hybrid wins, and the largest batch that fits.
minutes: 85
gpu: false
objectives:
  - Count a model's parameters component by component from its config alone.
  - Predict the weight footprint in bfloat16 and what it leaves on an 80 GB card.
  - Compute how fast the KV cache grows per token, and the fixed cost of a recurrent state.
  - Solve for the context length where a hybrid cache beats a dense one.
  - Work out the largest batch that fits a given GPU at a given context length.
lab: 02-memory-math
---

# Memory arithmetic

> [!TLDR]
> - The model holds about 27 billion parameters. At 2 bytes each, the weights
>   take about 54 GB, which leaves roughly 30 GiB of an 80 GiB card for
>   everything else.
> - Only 16 of the 64 layers keep a cache that grows as a conversation gets
>   longer. It grows by 64 KiB for every token.
> - The other 48 layers keep a fixed-size memory for each conversation, about
>   148 MiB, no matter how long the conversation gets.
> - That fixed cost pays for itself once a conversation passes 788 tokens. Past
>   that, this hybrid design uses far less memory than a model with full
>   attention in every layer.
> - After the weights and some overhead, the card fits 11 conversations of 32k
>   tokens at once, or 60 of 4k tokens. A 40 GiB card can't even hold the
>   weights.

Suppose you're about to serve this model to real users. The first question
anyone asks is how many of them you can serve at once. The answer comes from
one budget: an A100 80GB gives you 80 GiB of memory, minus what the driver
keeps, and everything the engine needs has to fit inside it.

This chapter works out where every byte goes. Every number here is ==arithmetic,
not measurement==. You can check each one on paper, and the lab has you write
them as code.

They're also the numbers later chapters are judged against. Chapter 15 exists
because of the KV cache figure, chapter 18 because of the weight figure, and
chapter 16 because of the batch figure.

## Before you start

**A parameter is one learned number, and a linear layer's parameter count is
the product of its shape.** PyTorch stores `nn.Linear` weights as
`(out_features, in_features)`, so a projection from 5120 to 17408 holds
$5120 \times 17408$ parameters. This model has no biases on its projections,
which is normal for modern decoders, so there's nothing to add.

**Bytes are parameters times the element size.** bfloat16, the 16-bit
floating-point format the whole checkpoint uses, takes 2 bytes per parameter.
Chapter 18 changes this number and nothing else.

**Units.** Weights are quoted in decimal GB, because that's how model sizes are
written everywhere. Caches are quoted in binary KiB, MiB, and GiB, because they
come from power-of-two shapes. So $1\ \mathrm{GB} = 10^9$ bytes, and
$1\ \mathrm{GiB} = 2^{30}$ bytes. The two differ by 7%, which is enough to
matter in a budget. The [notation chapter](/c/00a-notation-and-prerequisites)
has the full convention.

**Sequence, context length, and batch.** A *sequence* is one conversation's
tokens: the prompt plus everything generated so far. Its *context length* $L$
is how many tokens it holds. A *batch* is the set of sequences the engine
processes together, and its size sets how many users you serve at once.

**The two kinds of layer.** Chapter 1 showed that this model mixes
*full-attention* layers, where each token looks back at every earlier token,
with *linear-attention* layers, which keep a fixed-size running summary instead.

**The geometry.** Everything in this chapter comes from the following config
fields:

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
the first. So a quarter of the layers are full attention, and the rest are
linear:

$$
N_{\text{full}} = \frac{64}{4} = 16, \qquad N_{\text{lin}} = 64 - 16 = 48
$$

## Where the memory goes

Before any counting, here's the map. Besides a fixed overhead for the driver,
four things compete for the card, and they behave differently as traffic grows:

| What | Grows with | Lifetime |
|---|---|---|
| Weights | Nothing: fixed for the model | Whole run |
| KV cache | Every token of every sequence | Until the sequence finishes |
| Recurrent state | Number of sequences, not their length | Until the sequence finishes |
| Activations | Tokens in one forward pass | One forward pass |

The chapter works through them roughly in that order: the weights first,
because they come off the top, then the two per-sequence costs, then the batch
they allow.

## Counting the parameters

The weights are the one cost you pay before serving anyone, so they come first.
The model holds 26.9B parameters, and ==the MLP is 64% of them==.

The method is the same for every component: multiply out each matrix's shape,
add up one layer, then multiply by how many layers have it.

### Embedding and output projection

The *embedding* turns each token id into a vector of width $d$, so it's a
$V \times d$ matrix with one row per vocabulary entry:

$$
248{,}320 \times 5120 = 1{,}271{,}398{,}400 \approx 1.27\text{B}
$$

At the other end of the model, the *output projection* turns the final vector
back into one score per vocabulary entry. `tie_word_embeddings` is false, so it's
a separate matrix of the same shape. Together they hold:

$$
2 \times 1{,}271{,}398{,}400 = 2{,}542{,}796{,}800 \approx 2.54\text{B}
$$

That's 9.5% of the model spent entirely on the vocabulary, and none of it does
any reasoning. A 248k vocabulary is unusually large. It buys shorter token
sequences for non-English text. It costs 2.54 GB of memory per matrix, 5.09 GB
for the pair, out of the 80 you have.

### The MLP, in every layer

Every layer ends in an *MLP*, the feed-forward block. This model's MLP is
*SwiGLU*, which needs three matrices:

- A gate projection and an up projection, each from $d$ to $d_{ff}$.
- A down projection, from $d_{ff}$ back to $d$.

| Matrix | Shape | Parameters |
|---|---|---|
| `gate_proj` | $17408 \times 5120$ | 89,128,960 |
| `up_proj` | $17408 \times 5120$ | 89,128,960 |
| `down_proj` | $5120 \times 17408$ | 89,128,960 |
| **One MLP** | | **267,386,880** |

The three matrices are the same size, so one MLP is three times one of them:

$$
3 \times 5120 \times 17408 = 267{,}386{,}880 \approx 267.4\text{M}
$$

Every one of the 64 layers has an MLP, whichever kind of attention it uses:

$$
64 \times 267{,}386{,}880 = 17{,}112{,}760{,}320 \approx 17.11\text{B}
$$

For the engine, that makes the MLP the biggest single target. It's also the
part with the simplest cost model, three matrix multiplies (*GEMMs*), which is
why chapter 10 uses it as the roofline example.

### Full-attention layers

[Chapter 1](/c/01-the-model-on-disk) read the full-attention shapes off the
checkpoint. Two facts from it matter here:

- The query projection emits queries *and* an output gate side by side, so its
  output width is $2 \times h \times d_h = 2 \times 24 \times 256 = 12288$.
- Keys and values have only $h_{kv} = 4$ heads each, so their projections are
  $4 \times 256 = 1024$ wide.

The four projections count up as follows:

| Projection | Shape | Parameters |
|---|---|---|
| `q_proj`, with gate | $5120 \times 12288$ | 62,914,560 |
| `k_proj` | $5120 \times 1024$ | 5,242,880 |
| `v_proj` | $5120 \times 1024$ | 5,242,880 |
| `o_proj` | $(24 \times 256) \times 5120 = 6144 \times 5120$ | 31,457,280 |

Adding them gives one full-attention block:

$$
\begin{aligned}
&62{,}914{,}560 + 2 \times 5{,}242{,}880 + 31{,}457{,}280 \\
&= 62{,}914{,}560 + 10{,}485{,}760 + 31{,}457{,}280 \\
&= \boxed{104{,}857{,}600 \approx 104.9\text{M}}
\end{aligned}
$$

Without the output gate, the query projection would be half its size, and the
block would hold 73.4M. So the gate costs 31.5M per layer, and 0.5B across the
16 layers. That's worth knowing before you decide the double-width `q_proj` is a
typo.

### Linear-attention layers

Each linear-attention layer has four weight tensors, and every width comes from
the head counts in the geometry table:

- Queries and keys use one 128-wide block per key head:
  $16 \times 128 = 2048$ channels each.
- Values and the `z` gate use one 128-wide block per value head:
  $48 \times 128 = 6144$ channels each.

With those two widths, each tensor's size follows:

- **`in_proj_qkvz`** stacks all four blocks, so it maps 5120 to
  $2048 + 2048 + 6144 + 6144 = 16384$.
- **`in_proj_ba`** produces two scalars per value head, $2 \times 48 = 96$:
  $\beta$, the delta rule's write strength, and $a$, its forget gate. Chapter 6
  explains both.
- **`out_proj`** maps the 6144 value channels back to the hidden size.
- **`conv1d`** is a short *depthwise* convolution over the q, k, and v channels
  only, not `z`. Depthwise means each channel has its own filter, so its size is
  the channel count times the kernel width of 4:

$$
(2048 + 2048 + 6144) \times 4 = 10{,}240 \times 4 = 40{,}960
$$

The following table collects the four:

| Tensor | Shape | Parameters |
|---|---|---|
| `in_proj_qkvz` | $5120 \times 16384$ | 83,886,080 |
| `conv1d` | $10{,}240 \times 4$ | 40,960 |
| `in_proj_ba` | $5120 \times 96$ | 491,520 |
| `out_proj` | $6144 \times 5120$ | 31,457,280 |

Adding the four gives the per-layer total:

$$
\begin{aligned}
&83{,}886{,}080 + 40{,}960 + 491{,}520 + 31{,}457{,}280 \\
&= \boxed{115{,}875{,}840 \approx 115.9\text{M}}
\end{aligned}
$$

> [!KEY] Linear attention saves state, not parameters
> A linear-attention block is 11M *larger* than a full-attention block. It's
> cheaper in state, the memory it keeps per conversation. That's the point of
> the whole design, and the subject of the next three sections.

### Norms

The last component is tiny. Each layer has two RMSNorms, which rescale a vector
using one learned gain per channel, so each holds $d$ parameters. One more norm
sits at the end of the model:

$$
64 \times 2 \times 5120 + 5120 = 655{,}360 + 5120 = 660{,}480
$$

That's rounding error, at 0.0025% of the model. Count it anyway, because the lab
does.

### The total

You now have every piece. Each layer holds its attention block plus an MLP, so
the two layer types add up as follows:

| Layer type | Count | Attention block | MLP | All layers of this type |
|---|---|---|---|---|
| Full attention | 16 | 104,857,600 | 267,386,880 | 5,955,911,680 |
| Linear attention | 48 | 115,875,840 | 267,386,880 | 18,396,610,560 |

The last column is the count times the sum of the two middle columns:

$$
\begin{aligned}
16 \times (104{,}857{,}600 + 267{,}386{,}880) &= 5{,}955{,}911{,}680 \\
48 \times (115{,}875{,}840 + 267{,}386{,}880) &= 18{,}396{,}610{,}560
\end{aligned}
$$

Add the embeddings and the norms to get the whole model:

| Component | Parameters | Share |
|---|---|---|
| Embedding and output projection | 2,542,796,800 | 9.5% |
| Full-attention layers, 16 | 5,955,911,680 | 22.1% |
| Linear-attention layers, 48 | 18,396,610,560 | 68.4% |
| Norms | 660,480 | 0.0% |
| **Total** | **26,895,979,520** | **100%** |

That's 26.9B, which matches the name on the box. Now convert it to bytes, at 2
bytes per parameter in bfloat16:

$$
\begin{aligned}
26{,}895{,}979{,}520 \times 2 &= 53{,}791{,}959{,}040\ \text{bytes} \\
&= \boxed{53.8\ \mathrm{GB} = 50.1\ \mathrm{GiB}}
\end{aligned}
$$

For the engine, this is the first and largest line in the budget. It leaves
==roughly 30 GiB on an 80 GiB card== before anything else is allocated.

An A100 40GB can't hold the weights at all. That's why this course uses the 80GB
part, and why the lab's `max_batch_size` returns 0 for a 40 GiB budget.

## KV cache

With the weights placed, the next question is what each conversation costs as it
grows. The answer for the full-attention layers is the KV cache: 64 KiB per
token, paid by only 16 layers.

Here's why the cache exists. When the model generates a token, each
full-attention layer compares that token's query against the key of every
earlier token, then mixes their values. Recomputing all those keys and values on
every step would redo the whole conversation each time. So the engine computes
each token's keys and values once and stores them. That store is the *KV cache*.

Each new token adds one key and one value per KV head in each full-attention
layer, and each is $d_h$ wide. Multiply the pieces together for one token in one
layer:

$$
\underbrace{2}_{K \text{ and } V} \times
\underbrace{\hla{4}}_{\hla{h_{kv}}} \times
\underbrace{256}_{d_h} \times
\underbrace{2}_{\text{bytes}} = 4096\ \text{bytes} = 4\ \mathrm{KiB}
$$

Only the 16 full-attention layers pay this. The 48 linear layers contribute
nothing per token, because their state doesn't grow with the sequence. So for
the whole model:

$$
4096 \times 16 = 65{,}536\ \text{bytes} = \boxed{64\ \mathrm{KiB\ per\ token}}
$$

In engine terms, a 4k-token conversation ($4096 \times 64$ KiB) holds 0.25 GiB
of KV cache, and that grows by the same amount for every 4k tokens more.

Grouped-query attention, where several query heads share one KV head, is already
doing heavy lifting here. Each of the $\hla{h_{kv}}$ KV heads serves
$h / \hla{h_{kv}} = 24 / 4 = 6$ query heads, so the cache stores 4 heads instead
of 24.

> [!EXAMPLE] The same cache without grouped-query attention
> Give every query head its own KV head, $\hla{h_{kv}} = 24$ instead of 4, and
> the same 16 layers cost:
>
> $$
> 2 \times \hla{24} \times 256 \times 2 \times 16 = 393{,}216\ \text{bytes}
> = 384\ \mathrm{KiB\ per\ token}
> $$
>
> That's six times larger, for the same model. Chapter 7 covers what that costs
> in quality.

The cache's size scales with batch size and context length, and with nothing
else. Multiply 64 KiB by the tokens per sequence and by the number of sequences:

| Context | Batch 1 | Batch 8 | Batch 32 |
|---|---|---|---|
| 4k | 0.25 GiB | 2.0 GiB | 8.0 GiB |
| 32k | 2.0 GiB | 16.0 GiB | 64.0 GiB |
| 128k | 8.0 GiB | 64.0 GiB | 256.0 GiB |

Read the bottom-right cell again: 256 GiB of cache for 32 concurrent 128k
sequences, on a card with 80.

> [!KEY] The cache sets the batch size
> The cache, not the weights, sets your maximum batch size. That single fact is
> why chapter 15 is about packing it efficiently.

## Recurrent state

The KV cache covers the 16 full-attention layers. This section answers the
matching question for the other 48: what do linear-attention layers keep per
conversation? The answer is a fixed-size state of 147.8 MiB per sequence.

Picture the difference this way. Full attention keeps a transcript and appends
to it every token. Linear attention keeps a fixed-size summary and overwrites it
every token, so its size never changes. Chapter 6 derives what's in the summary.
Here you need only its size, which has two parts.

**The delta-rule state.** Each value head keeps one matrix of size
$\text{key dim} \times \text{value dim}$, that is, $128 \times 128$. It's kept in
float32, 4 bytes per element, for numerical stability: a bfloat16 accumulator
stops absorbing small updates after a few hundred tokens. The
[notation chapter](/c/00a-notation-and-prerequisites) works through why.

**The convolution window.** The short causal convolution needs the last 4 steps
of its q, k, and v channels, stored in the model's own dtype, bfloat16.

The following table sizes both parts for one layer:

| Part | Calculation | Bytes |
|---|---|---|
| Delta-rule state | $48\ \text{heads} \times 128 \times 128 \times 4\ \text{bytes}$ | 3,145,728 (3.0 MiB) |
| Convolution window | $10{,}240\ \text{channels} \times 4\ \text{steps} \times 2\ \text{bytes}$ | 81,920 (80 KiB) |
| **One layer** | | **3,227,648 (3.078 MiB)** |

The 10,240 convolution channels are the q, k, and v widths added up:

$$
\underbrace{(2 \times 16 \times 128 + 48 \times 128)}_{10{,}240\ \text{channels}}
\times 4 \times 2 = 81{,}920\ \text{bytes} = 80\ \mathrm{KiB}
$$

Every one of the 48 linear layers keeps its own copy:

$$
\begin{aligned}
3{,}227{,}648 \times 48 &= \hlc{154{,}927{,}104}\ \text{bytes} \\
&= \boxed{147.8\ \mathrm{MiB}}
\end{aligned}
$$

==It doesn't grow.== A sequence at 128k context carries the same 147.8 MiB as a
sequence ten tokens long. That's the entire trade: a large constant in exchange
for a shallower slope.

## The break-even point

You now have two per-sequence costs, one that grows and one that's fixed. Was
the trade worth it? Yes, once a sequence passes 788 tokens: from there on, the
hybrid uses less memory than an all-full-attention model.

Think of it like two phone plans. One has no monthly fee but a high rate per
minute. The other charges a fixed fee but a lower rate. The fixed-fee plan
loses on a short call and wins on a long one, and somewhere in between they cost
the same.

Here, the "no fee" plan is a hypothetical dense model where all 64 layers use
full attention. It pays the same 4 KiB per layer per token, over 64 layers
instead of 16:

$$
4096 \times 64 = 262{,}144\ \text{bytes} = 256\ \mathrm{KiB\ per\ token}
$$

Write both footprints as a function of context length $L$, in bytes per
sequence. The dense model pays only per token:

$$
M_{\text{dense}}(L) = 262{,}144\,L
$$

The hybrid pays $\hlb{65{,}536}$ bytes per token plus a fixed
$\hlc{154{,}927{,}104}$ bytes of recurrent state:

$$
M_{\text{hybrid}}(L) = \hlb{65{,}536\,L} + \hlc{154{,}927{,}104}
$$

Those are two straight lines. The hybrid has the higher starting point and the
shallower slope, so they cross exactly once. To find the crossing, set them
equal, move the per-token terms to one side, and divide:

$$
\begin{aligned}
262{,}144\,L &= \hlb{65{,}536\,L} + \hlc{154{,}927{,}104} \\
(262{,}144 - 65{,}536)\,L &= 154{,}927{,}104 \\
196{,}608\,L &= 154{,}927{,}104 \\
L &= \frac{154{,}927{,}104}{196{,}608} = \boxed{788\ \text{tokens}}
\end{aligned}
$$

The general form is worth remembering, because it's the only equation in this
chapter that isn't a multiplication:

$$
L_{\text{break-even}}
= \frac{\hlc{\text{fixed state per sequence}}}{\text{KV bytes saved per token}}
$$

> [!INTUITION]
> Every token saves the hybrid 192 KiB of KV cache compared with the dense
> model: that's the 196,608 bytes in the denominator. The
> $\hlc{\text{fixed state}}$ is a one-time bill. The break-even is how many
> tokens of savings it takes to pay that bill off.

The following table shows both footprints on either side of the crossing:

| Context | Dense, per sequence | Hybrid, per sequence | Cheaper |
|---|---|---|---|
| 100 | 25.0 MiB | 154.0 MiB | Dense |
| 788 | 197.0 MiB | 197.0 MiB | Equal |
| 32k | 8.0 GiB | 2.14 GiB | Hybrid |
| 128k | 32.0 GiB | 8.14 GiB | Hybrid |

Below 788 tokens, the fixed state costs more than the KV entries it replaces.
Above it, the hybrid wins, and the margin grows by 192 KiB for every further
token. The ratio between them tends to 4 as $L$ grows, which is $64 / 16$: the
layer ratio, as it must be.

788 tokens is short. Almost every real request is longer than that, so in
practice ==the hybrid always wins==. The calculation exists to prove the design
isn't free, not to guide a decision.

`ModelConfig.hybrid_breakeven_tokens` computes this for any config, and the lab
has you derive it.

## The largest batch that fits

This is the question the chapter opened with: how many conversations can the
card hold at once? At 32k context, 11 sequences fit. At 4k, 60 do. This section
puts the weights, the KV cache, and the recurrent state together to show why.

Picture packing a suitcase: the weights go in first, then a fixed overhead, and
whatever space is left is divided into equal slots, one per sequence.

One sequence at context $L$ costs its recurrent state plus its KV entries:

$$
M_{\text{seq}}(L) = \hlc{154{,}927{,}104} + \hlb{65{,}536\,L}
$$

Write $C$ for the card's capacity, $\hld{W}$ for the weights, and $O$ for the
overhead. Subtract the weights and the overhead from the capacity, divide by the
cost of one sequence, and round down to whole sequences:

$$
\boxed{B_{\max} = \left\lfloor \frac{C - \hld{W} - O}{M_{\text{seq}}(L)} \right\rfloor}
$$

The lab uses the following values:

- $C = 80 \times 2^{30}$ bytes, the 80 GiB card.
- $\hld{W} = 53.8$ GB, the weights.
- $O = 6\ \mathrm{GiB}$ for the CUDA context, the driver, and a workspace for
  activations, the temporary tensors a forward pass creates.

> [!EXAMPLE] Batch size at 32k context
> The numerator is what's left after the weights and the overhead:
>
> $$
> \begin{aligned}
> &85{,}899{,}345{,}920 - \hld{53{,}800{,}000{,}000} - 6{,}442{,}450{,}944 \\
> &= 25{,}656{,}894{,}976\ \text{bytes} = 23.9\ \mathrm{GiB}
> \end{aligned}
> $$
>
> At 32k context, one sequence costs:
>
> $$
> \begin{aligned}
> &\hlc{154{,}927{,}104} + \hlb{65{,}536 \times 32{,}768} \\
> &= 2{,}302{,}410{,}752\ \text{bytes} = 2.14\ \mathrm{GiB}
> \end{aligned}
> $$
>
> Divide and round down:
>
> $$
> B_{\max} = \left\lfloor \frac{25{,}656{,}894{,}976}{2{,}302{,}410{,}752}
> \right\rfloor = 11
> $$

The same steps at 4k context, and on a 40 GiB card, give the following results:

| Case | Numerator | One sequence | Quotient | $B_{\max}$ |
|---|---|---|---|---|
| 80 GiB, 32k context | 25,656,894,976 | 2,302,410,752 | 11.14 | 11 |
| 80 GiB, 4k context | 25,656,894,976 | 423,362,560 | 60.6 | 60 |
| 40 GiB, any context | −17,292,777,984 | — | negative | 0 |

The 4k row's sequence cost is $154{,}927{,}104 + 65{,}536 \times 4096$. On a
40 GiB card, the numerator is negative before you divide anything, because the
weights alone are bigger than the card, so the answer is 0 at every context
length.

Three things follow from that formula:

- **The floor matters.** Memory is allocated in whole sequences, so 11.14
  sequences is 11. Chapter 15's paged cache is what lets you stop rounding down
  so aggressively, by allocating blocks instead of whole sequences.
- **The fixed state is 7% of a 32k sequence and 37% of a 4k one.** The hybrid's
  constant hurts most exactly where you want large batches, which is short
  requests. ==This is a real cost, not a footnote.==
- **Every term is a lever.** Halving $\hld{W}$ with int8 quantization, storing
  each weight in 1 byte instead of 2, moves the numerator from 23.9 GiB to
  48.9 GiB, which roughly doubles the batch. Chapter 18 covers what that costs
  in quality, and why FP8, which does it better, isn't available to you on the
  A100's Ampere architecture.

## Activations

The weights and the caches stay on the card while a sequence lives.
*Activations*, the intermediate tensors a forward pass creates and then frees,
don't. They're transient, but not small, and the allocator needs room for them:
that's the activation workspace in $O$.

Start with the MLP. Its hidden activation has shape `(tokens, 17408)` in
bfloat16. During *prefill*, when the engine processes a prompt, it can push a
4096-token chunk through at once:

$$
4096 \times 17{,}408 \times 2 = 142{,}606{,}336\ \text{bytes} = 136\ \mathrm{MiB}
$$

SwiGLU holds two of those at once, the gate branch and the up branch, before the
elementwise product frees one.

Attention scores are worse, if you materialize them. The score tensor is
`(heads, q_len, kv_len)`: one score for every pair of positions, in every head.
At 8k context with 24 heads:

$$
24 \times 8192 \times 8192 \times 2 = 3{,}221{,}225{,}472\ \text{bytes}
= 3.0\ \mathrm{GiB}
$$

That's for one sequence, and ==it grows with the square of context==. Not
materializing it is exactly what FlashAttention is for, and chapter 14 gets this
to zero by never holding more than a small tile of it.

It's also why *chunked prefill*, which splits a long prompt into pieces, caps how
many tokens enter a forward pass at once: the cap bounds the activation peak.

## A budget that works

Now you can put the whole card on one page. The weights, the overhead, and the
cache split it as follows:

| Item | Size | Where it comes from |
|---|---|---|
| Weights, bf16 | 50.1 GiB | $\hld{W}$, the parameter count times 2 bytes |
| CUDA context and driver | ~1.0 GiB | Part of $O$ |
| Activation workspace | ~3.0 GiB | Part of $O$ |
| Fragmentation headroom | ~2.0 GiB | Part of $O$ |
| **Left for cache** | **~23.9 GiB** | $C - \hld{W} - O$ |

If it held only KV cache, 23.9 GiB would store 391k tokens at 64 KiB each. Once
you subtract each sequence's 147.8 MiB of recurrent state, it holds batch 11 at
32k context, batch 60 at 4k, or any mix in between.

`PagedKVCache` in chapter 15 turns this budget into a pool of blocks, so a short
request doesn't reserve space for a length it never reaches.

## What goes wrong

These mistakes turn a budget that fits on paper into an out-of-memory error.

**Confusing GB with GiB.** 53.8 GB of weights is 50.1 GiB. Subtracting 53.8 from
80 as if both were GiB loses 3.7 GiB, which is two more sequences at 32k. The
lab's tests use binary units for the card and decimal units for the weights,
exactly as this chapter does, because that's what the tools report.

**Forgetting that the KV cache is per sequence, not per batch.** It's tempting
to compute 2 GiB at 32k and treat it as the total. It's per sequence, so batch 8
is 16 GiB and the card is full.

**Sizing the cache for the average request.** Requests arrive at whatever length
they arrive. An engine that assumes 4k and meets a 32k prompt must either pause
another sequence to free memory (*preempt* it) or run out of memory in the middle
of a forward pass. That's an out-of-memory error inside a CUDA kernel, and it
isn't gracefully recoverable. Chapter 16 handles this with admission control.

**Ignoring fragmentation.** The 2 GiB of headroom in the budget isn't
superstition. A caching allocator that has served many different sizes ends up
unable to satisfy a large contiguous request, even when the free total is
enough. Paged allocation exists partly to avoid this.

> [!RECAP]
> - The model has 26,895,979,520 parameters: 53.8 GB, or 50.1 GiB, in bfloat16.
>   The MLP holds 64% of them.
> - The KV cache costs 64 KiB per token, per sequence, from the 16
>   full-attention layers only. Grouped-query attention makes it six times
>   smaller.
> - The recurrent state is a fixed 147.8 MiB per sequence.
> - Break-even is the fixed state divided by the KV bytes saved per token: 788
>   tokens.
> - $B_{\max} = \lfloor (C - W - O) / M_{\text{seq}}(L) \rfloor$ gives 11 at 32k,
>   60 at 4k, and 0 on a 40 GiB card.

## Check your understanding

> [!QUESTION] If the model used 24 KV heads instead of 4, how much cache would a batch of 8 at 32k context need?
> Per token the cost is 384 KiB, six times the 64 KiB figure. Per sequence at
> 32,768 tokens that's 12 GiB, and for 8 sequences it's 96 GiB. That doesn't fit
> on the card alongside 50 GiB of weights. Grouped-query attention isn't an
> optimization here; it's what makes the configuration possible at all.

> [!QUESTION] Why is the break-even independent of batch size?
> Both sides of the equation scale linearly with the number of sequences: each
> sequence has its own recurrent state and its own KV entries. The batch factor
> cancels, so the break-even depends only on the per-sequence geometry.

> [!QUESTION] A colleague proposes keeping the recurrent state in bfloat16 to save 72 MiB per sequence. What breaks?
> The state is a running sum over thousands of tokens. In bfloat16, an addend
> smaller than about $1/256$ of the accumulated magnitude rounds away entirely.
> The state stops updating from the tail of the sequence while still looking
> finite and plausible. The saving is also small: 72 MiB per sequence is 3% of a
> 32k sequence's footprint, and it doesn't change the batch size at 32k at all.

> [!QUESTION] How would the break-even move if `full_attention_interval` were 8?
> There would be 8 full-attention layers and 56 linear ones. The hybrid's
> per-token cost halves to 32 KiB, so the saving per token rises to 224 KiB,
> while the fixed state grows to $56 \times 3{,}227{,}648 = 180{,}748{,}288$
> bytes. The break-even is $180{,}748{,}288 / 229{,}376 = 788$ tokens, unchanged,
> because both the state and the saving scale with the same layer split. That
> invariance is a good check that you have the formula right.

## Lab

> [!TRY]
> Write this chapter's arithmetic as code. You pass when each function
> reproduces the real model's figures: 64 KiB of KV per token, 147.8 MiB of
> recurrent state, a break-even near 788 tokens, and a batch size of 0 on an
> A100 40GB.

The harness hands every function a config dict with the fields from the
geometry table at the start of this chapter. You implement eight functions:
`num_full_attention_layers`, `mlp_params_per_layer`, `embedding_params`,
`kv_bytes_per_token`, `dense_kv_bytes_per_token`, `recurrent_state_bytes`,
`breakeven_tokens`, and `max_batch_size`.

The harness checks each one against the real model:

- 16 full-attention layers.
- 267.4M parameters in one MLP block.
- 2.54B parameters in the embedding and output projection together.
- 64 KiB of KV per token, and 256 KiB for an all-full-attention model.
- 147.8 MiB of recurrent state per sequence, to within a kilobyte.
- A break-even within 2 tokens of 788.
- For `max_batch_size`: a batch between 5 and 12 at 32k context, a larger batch
  at 4k context than at 32k, and 0 on an A100 40GB, because the weights alone
  don't fit.

It also reports `params_billions`, `kv_kib_per_token`, `recurrent_state_mib`,
`breakeven_tokens`, and both batch sizes as metrics, so you can compare them
against the figures derived in this chapter.

The lab runs on CPU. Arithmetic about memory doesn't need a GPU to be worth
doing.

## Further reading

- [NVIDIA A100 tensor core GPU architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf)
- [GQA: training generalized multi-query transformer models from multi-head checkpoints](https://arxiv.org/abs/2305.13245)
- [Fast transformer decoding: one write-head is all you need](https://arxiv.org/abs/1911.02150) — multi-query attention, the limiting case of GQA.
- [Efficient memory management for large language model serving with PagedAttention](https://arxiv.org/abs/2309.06180)
