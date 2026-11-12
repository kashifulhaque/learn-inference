---
title: "KV economy: MLA, windows, sinks, and eviction"
slug: 07a-kv-economy-attention-variants
part: "Part 2 — A forward pass"
summary: Four ways to spend fewer KV bytes than GQA does, each with what it costs in FLOPs and in quality risk, measured on this model's geometry.
minutes: 110
gpu: false
objectives:
  - "Count MLA's latent KV cache and its decoupled RoPE key, and compare bytes per token with GQA."
  - "Derive weight absorption, check that the naive and absorbed forms give the same output, and compute the FLOPs and arithmetic intensity of each."
  - "Implement a sliding-window mask and a rolling-buffer cache with attention sinks, and check them against masked full attention."
  - "Run H2O heavy-hitter eviction on planted heavy hitters and compare the attention mass it keeps with an oracle's."
  - "Put every scheme in one table of bytes per token, FLOPs per token, and quality risk for this model's geometry."
lab: 07a-kv-economy
---

# KV economy: MLA, windows, sinks, and eviction

> [!TLDR]
> - GQA already cut this model's cache to 64 KiB per token. Four more levers exist: cache a low-rank latent (MLA), keep only a window, keep a few sink tokens, or evict the tokens that attention ignores.
> - MLA caches $d_c + d_r$ numbers per layer per token instead of $2 H_{kv} d_h$. DeepSeek-V2's $512 + 64 = 576$ against $2 \cdot 128 \cdot 128 = 32{,}768$ is a ratio of 56.9, and it equals GQA with 2.25 KV groups.
> - Absorbing the key up-projection into the query lets decode attend in the latent space. It removes the per-step key and value expansion, and it raises decode intensity from GQA's group size to $\approx 242$ FLOPs per byte at V2's 128 heads, above the A100's ridge of 161.
> - A window with a rolling buffer holds `window + sinks` slots whatever the context length. On a planted head, four sink tokens bring the error against full attention from 1.069 to 0.000094.
> - H2O keeps the heavy hitters and the newest tokens. On planted heavy hitters it retains 0.976 of an oracle's attention mass, against 0.041 for a window of the same size.

[Chapter 7](/c/07-grouped-query-attention) shrinks the KV cache by sharing key and value heads, and [chapter 9](/c/09-the-kv-cache) shows that the cache is the part of memory that grows with every token. At 64 KiB per token, one 32k-token sequence holds 2 GiB. A server that wants more sequences per card, or longer ones, has to spend fewer bytes per token or keep fewer tokens.

This chapter covers both. MLA, from DeepSeek-V2 and V3, spends fewer bytes per token by caching a compressed latent. Sliding windows, attention sinks, and eviction keep fewer tokens. Each lever changes a different term of the cost, and each carries a different risk to quality.

The lab is CPU-only and runs in float64. No part of it is an A100 measurement, and the MLA rows for this course's model are hypothetical, because the model has no MLA layers.

## Before you start

**GQA bytes, from [chapter 7](/c/07-grouped-query-attention).** One full-attention layer stores $2 \times 4 \times 256 \times 2 = 4096$ bytes of keys and values per token. Over the 16 full-attention layers that is $65{,}536$ bytes, 64 KiB. The 48 gated-delta layers keep a fixed recurrent state and add nothing per token, so every number in this chapter is about the 16 layers that do.

**Decode intensity, from chapter 7.** Decode attention does $4 H d_h$ FLOPs per cached token and reads $2 H_{kv} d_h b$ bytes, so its intensity is $2H/(H_{kv} b)$. With $H = 24$, $H_{kv} = 4$, and $b = 2$ bytes that is the group size, 6. [Chapter 10](/c/10-roofline) puts the A100's ridge at 161.2 FLOPs per byte, so decode attention sits far on the memory side.

**Rotary embeddings, from [chapter 5](/c/05-rotary-embeddings).** RoPE rotates a query and a key by their own positions, so their dot product depends only on the distance between them. That property is the reason MLA needs a special case.

## MLA caches a latent instead of keys and values

*Multi-head latent attention* (MLA) stores one short vector per token per layer and rebuilds every head's key and value from it when it needs them.

Start from the picture. GQA keeps $H_{kv}$ pairs of keys and values and lets several query heads read each pair. MLA keeps one vector $\hla{c}$ of width $d_c$. A learned up-projection turns $\hla{c}$ into all $H$ keys, and a second one turns it into all $H$ values. The cache holds only $\hla{c}$.

DeepSeek-V2's equations 9 to 11 write the same thing in symbols. The hidden state $h_t$ of token $t$ is projected down once, and up twice:

$$
\hla{c_t} = W^{DKV} h_t,
\qquad
k^C_t = W^{UK} \hla{c_t},
\qquad
v^C_t = W^{UV} \hla{c_t}.
$$

Here $\hla{c_t}$ has $d_c$ entries, and $W^{UK}$ and $W^{UV}$ map it to $H d_h$ entries each. The paper requires $d_c \ll H d_h$.

The cache cost follows from what you store. Standard multi-head attention stores $2 H d_h$ elements per token per layer. MLA stores $d_c$ for the latent, plus the RoPE key that the next section adds, which has $d_r$ entries and is shared by every head:

$$
\boxed{\text{MLA elements per token per layer} = \hla{d_c} + \hlb{d_r}}
$$

The paper's section 3.1.2 sets DeepSeek-V2's dimensions to 60 layers, a hidden size of 5120, $H = 128$ heads of $d_h = 128$, $d_c = 512$, and $d_r = 64$.

> [!EXAMPLE] DeepSeek-V2 against its own MHA, in elements
> MHA would store $2 \times 128 \times 128 = 32{,}768$ elements per layer, or $1{,}966{,}080$ over 60 layers. MLA stores $512 + 64 = 576$ per layer, or $34{,}560$ over 60 layers. The ratio is $1{,}966{,}080 / 34{,}560 = 56.89$.
>
> In bfloat16 that is $69{,}120$ bytes, 67.5 KiB, per token, against $3{,}932{,}160$ bytes, 3.75 MiB, for MHA. The paper's Table 1 notes that $576 = 4.5\,d_h$, which equals GQA with $n_g = 2.25$ groups. DeepSeek-V3 keeps $d_c = 512$, $d_r = 64$, and $H = 128$, with 61 layers, which is $70{,}272$ bytes per token.

The paper's abstract also reports a 93.3% KV cache reduction against its DeepSeek 67B. That compares two different models under the paper's own settings, and this chapter doesn't reproduce it.

Now use this model's geometry. It already has $H_{kv} = 4$ groups of $d_h = 256$, so GQA costs 4096 bytes per layer per token. Suppose a model like this used MLA at DeepSeek-V2's proportions: $d_c = 4 d_h = 1024$ and $d_r = d_h/2 = 128$. That stores $(1024 + 128) \times 2 = 2304$ bytes per layer, a ratio of $4096 / 2304 = 1.78$, which equals $4 / 2.25$. At V2's absolute widths, $d_c = 512$ and $d_r = 64$, it stores $1152$ bytes, a ratio of $3.56$.

==MLA's saving depends on what you compare it with.== Against MHA it is 56.9 times. Against a model that already shares four KV heads it is under 2 times at the same proportions. The lab's `mla_kv_bytes` and `gqa_kv_bytes` make the comparison for any configuration.

The paper says MLA reaches better quality than MHA with this much smaller cache and compares the two in its Appendix D.2. This chapter doesn't test quality, and nothing in the lab does.

```viz
07a-kv-economy
```

## Why RoPE needs its own key

You can't rotate the compressed latent, because the rotation would stand between two matrices that the next section wants to multiply together.

Write the score between query $t$ and key $j$ with RoPE applied to both. With $R_t$ the rotation for position $t$, and $W^Q$ the query projection:

$$
(R_t W^Q h_t)^{\mathsf T} (R_j W^{UK} c_j)
= h_t^{\mathsf T} W^{Q\,\mathsf T} \hlc{R_{t-j}} W^{UK} c_j.
$$

The two rotations merge into one, $R_{t-j}$, which depends on the distance. It sits between $W^Q$ and $W^{UK}$, and matrices don't commute, so you can't fold $W^{UK}$ into the query side once and reuse it. The paper says the alternative is to recompute every prefix key on every step, which defeats the point of caching a latent.

DeepSeek's fix, in equations 14 to 18, splits each head's query and key into two parts. The content part, $q^C$ and $k^C$, carries no RoPE and goes through the latent. The position part, $q^R_{t,i}$ and $k^R_t$, carries RoPE and has $d_r$ entries. The key $k^R_t$ comes straight from the hidden state, and all heads share it:

$$
q_{t,i} = [q^C_{t,i};\, q^R_{t,i}],
\qquad
k_{t,i} = [k^C_{t,i};\, \hlb{k^R_t}],
\qquad
\text{score} = \frac{q_{t,i}^{\mathsf T} k_{j,i}}{\sqrt{d_h + d_r}}.
$$

The dot product of the two concatenations is the content score plus the position score. The score's scale is $1/\sqrt{d_h + d_r}$, the width of the concatenated vector, not $1/\sqrt{d_h}$.

The cost is $d_r$ extra cached numbers per layer per token, $64/576 = 11.1\%$ of V2's cache. The lab's `mla_kv_bytes` has to include it. Leaving it out gives $61{,}440$ bytes per token for V2, not $69{,}120$.

## Absorb the up-projections

Because the content key is linear in the latent, you can apply its up-projection to the query once instead of to every cached token.

The naive way to decode keeps the latent cache and, on every step, expands it. For each of the $T$ cached tokens and each head it computes a key $W^{UK} c_j$ and a value $W^{UV} c_j$. That is $4 T\, d_c\, H d_h$ FLOPs per layer per step, which grows with the context.

Absorption moves the work to the query. A dot product is associative, so

$$
q^{C\,\mathsf T} (W^{UK} c_j) = \underbrace{(W^{UK\,\mathsf T} q^C)}_{\hla{q^{\text{lat}}}}{}^{\mathsf T} c_j.
$$

The product $\hla{q^{\text{lat}}} = W^{UK\,\mathsf T} q^C$ has $d_c$ entries and is computed once per head per step. The score against every cached token is then a dot product with the cached latent itself.

> [!EXAMPLE] The identity on a 2-by-2 case
> Take $c = (1, 2)$, $W^{UK} = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}$, and $q = (2, 1)$. The expanded key is $W^{UK} c = (1, 3)$, and the score is $2 \cdot 1 + 1 \cdot 3 = 5$. The absorbed query is $W^{UK\,\mathsf T} q = (3, 1)$, and its dot product with $c$ is $3 \cdot 1 + 1 \cdot 2 = 5$.

The values work the same way, one step later. The weighted sum of values is linear in the values, so the sum can run over the latents and the up-projection can follow it:

$$
o = \sum_j p_j\, W^{UV} c_j = W^{UV} \Big( \sum_j p_j\, c_j \Big).
$$

Both forms end with the same numbers, and the lab checks it: `mla_attention_absorbed` must match `mla_attention_naive` to $10^{-9}$ in float64, and the naive form must match PyTorch's `scaled_dot_product_attention` on explicit keys and values.

> [!NOTE] The paper folds $W^{UV}$ into $W^O$. Deployed code usually doesn't.
> The paper says $W^{UV}$ can be absorbed into the output projection $W^O$. Per head, the merged matrix is $d \times d_c$, so for V2 it has $128 \times 5120 \times 512 = 335{,}544{,}320$ entries per layer. Keeping $W^{UV}$ ($8{,}388{,}608$ entries) and $W^O$ ($83{,}886{,}080$) separate stores $92{,}274{,}688$, which is 3.6 times fewer. The lab applies $W^{UV}$ after the weighted sum.

### What absorption does to the FLOPs

Absorption trades a large per-step expansion for a small per-step projection and a wider dot product.

Count 2 FLOPs per multiply-add, for the attention core only: the projections from the hidden state and $W^O$ are the same in both forms. For context length $T$ and one layer:

| Form | Scores | Weighted sum | Extra |
|---|---|---|---|
| Naive | $2H(d_h + d_r)\,T$ | $2H d_h\,T$ | $4\,d_c\,H d_h\,T$ to expand keys and values |
| Absorbed | $2H(d_c + d_r)\,T$ | $2H d_c\,T$ | $4H d_h d_c$ for the query and the output, independent of $T$ |

At V2's shape and $T = 4096$, over 60 layers, the absorbed form costs $70{,}464{,}307{,}200$ FLOPs, 70.5 GFLOP. The naive form costs $8{,}266{,}469{,}867{,}520$, 8.27 TFLOP. The naive form does 117 times the work, nearly all of it expansion.

The absorbed form's per-step cost has a part that grows with $T$, $278{,}528$ FLOPs per cached token per layer, and a fixed part, $33{,}554{,}432$. The two are equal at $T = 120.5$. Beyond that, the growing part dominates.

### Intensity, and why 128 heads matter

Absorbed decode reads $(d_c + d_r)\, b$ bytes per cached token per layer and does $2H(2d_c + d_r)$ FLOPs on them, so its intensity, for large $T$, is

$$
\boxed{I_{\text{MLA}} = \frac{2H\,(2\hla{d_c} + \hlb{d_r})}{(\hla{d_c} + \hlb{d_r})\, b}}
$$

and it scales with the number of heads $H$. For V2, $I = 278{,}528 / 1152 = 241.8$ FLOPs per byte. At $T = 4096$ the fixed part lifts it to $248.889$, the lab's `intensity_absorbed`. Against the ridge of 161.2, ==V2's absorbed decode is compute bound==. MHA decode at the same shape has intensity 1.

On a rated A100 (312 TFLOP/s, 1935 GB/s), the floor for one cached token in one layer is the larger of the two times. MHA at V2's shape needs $65{,}536$ bytes, or $33.87$ ns. MLA needs $1152$ bytes, $0.595$ ns, but $278{,}528$ FLOPs, $0.893$ ns, so compute sets it. The speedup floor is $33.87 / 0.893 = 37.9$, not 56.9, because MLA spent its bytes saving on arithmetic. These are floors from rated numbers, not measurements.

This model has 24 query heads, not 128. The same formula with $H = 24$ gives $I \approx 45.3$ at $d_c = 4d_h$, and $45.7$ at 32,768 tokens once the fixed part counts. That is below the ridge, so MLA on this geometry would stay memory bound. Intensity rises from 6 to 45.7, and the time still follows the bytes.

## A window caps the cache

A sliding window lets each query read only the last $W$ tokens, so a rolling buffer of $W$ slots is enough.

Mistral 7B uses $W = 4096$. The window counts the current token, so query $i$ reads keys $i - W + 1$ through $i$. Implementations differ by one here, so write the mask down before you write the cache. As a boolean matrix with $j \le i$:

$$
\text{allowed}(i, j) = (i - j < W)\ \text{or}\ (j < s).
$$

The second term is the $s$ sink tokens, which the section on sinks explains. Without sinks, $s = 0$.

Mistral's paper keeps a cache of fixed size $W$ and stores the keys and values of timestep $i$ at position $i \bmod W$. At 32k tokens that cut cache memory by 8 times, which is $32768 / 4096$. On this model, a window of 4096 with 4 sinks holds $4100$ slots, $4100 \times 65{,}536 = 268{,}697{,}600$ bytes, 256.25 MiB per sequence, against 2 GiB, a ratio of $7.992$.

A window doesn't mean a token never influences a distant one. In Mistral's argument, information moves forward by up to $k \times W$ tokens after $k$ layers, so 32 layers of $W = 4096$ give a theoretical span of about 131K tokens. This model has only 16 full-attention layers, which would give $16 \times 4096 = 65{,}536$ and rely on the 48 delta layers for the rest. How well information survives that many hops is a quality question the arithmetic can't answer.

The cache and the mask have to agree, and the lab checks it. The slot for token $p$, with $s$ sinks, is:

$$
\text{slot}(p) = \begin{cases} p & p < s \\ s + (p - s) \bmod W & p \ge s. \end{cases}
$$

> [!EXAMPLE] A ring of window 4 and 2 sinks, over ten tokens
> Tokens 0 and 1 take slots 0 and 1 and never move. Token 2 takes slot $2 + 0 = 2$, token 5 takes slot 5, and token 6 wraps to slot $2 + (6 - 2) \bmod 4 = 2$, overwriting token 2. The slots for tokens 0 to 9 are $[0, 1, 2, 3, 4, 5, 2, 3, 4, 5]$, and after token 9 the cache holds positions $[0, 1, 6, 7, 8, 9]$.

Gemma 2 interleaves the two designs. Its report alternates local sliding-window layers of 4096 tokens with global layers whose span is 8192, in every other layer. On this model's 16 layers, 8 local and 8 global would hold $8 \times 4096 \times 32{,}768 = 1$ GiB for the global layers and $8 \times 4096 \times 4096 = 128$ MiB for the local ones, which is 1152 MiB at 32k.

Applying a window to a model that was trained with full attention changes what the model computes. The sink section shows what breaks.

## Sinks keep the softmax honest

A softmax has to put its mass somewhere, so when you evict the tokens a head leans on, the mass lands on whatever remains.

StreamingLLM reports that window attention fails once the text outgrows the cache, and that keeping the key-value states of the first few tokens largely recovers it. It calls those tokens *attention sinks*. Its abstract reports stable language modeling with up to 4 million tokens and up to a 22.2x speedup over a sliding-window recomputation baseline. How a real engine assigns rotary positions to a rolled cache is a design choice that paper treats in detail, and the lab doesn't model it.

The lab plants a sink to make the effect measurable. In a head with 200 tokens and $d = 16$, token 0's key is set to $6\sqrt{d}$ times the direction every query shares, so full attention puts $0.9999$ of its mass on token 0. That is a toy head. It shows the mechanism, and it doesn't say how strong a real head's sink is.

Evicting token 0 from a window of 32 tokens moves the output a lot, and keeping it fixes it:

| Sinks kept | Relative error of the final token's output |
|---|---|
| 0 | 1.069187 |
| 1 | 0.000096 |
| 4 | 0.000094 |

A bare window gets an error above 1: the remaining tokens had almost no weight under full attention, and the softmax inflated them to fill the gap. The lab uses 4 sinks, and this planted head needs only one.

Chapter 7's output gate gives a head another way to output nothing. Whether this model needs sinks, and how many, is something to measure on its real attention before you rely on it.

## Eviction by accumulated attention

Instead of fixing which tokens survive, H2O scores each cached token by the attention it has received and evicts the lowest scorer.

Zhang and coauthors observe that the accumulated attention scores of tokens follow a power-law distribution, and they call the few tokens that hold most of the mass *heavy hitters*. Their policy keeps a balance of recent tokens and heavy hitters. In their baselines the budget is split evenly between the two.

The lab's version is online, like decoding. At step $t$ the new token joins the cache. The query softmaxes over only the cached keys, and each cached key's score grows by its probability $p_{t,k}$. If the cache exceeds the budget, drop the cached key with the lowest accumulated score outside the newest `recent` tokens, and let ties go to the oldest.

The best fixed set of $B$ keys has a closed form, which is what makes an oracle possible. With $A$ the full attention matrix, $\sum_q \sum_{k \in S} A_{qk} = \sum_{k \in S} \text{colsum}_k$, so the top-$B$ column sums maximize the retained mass. The oracle sees every future query, so no online policy can match it.

The lab plants heavy hitters. It builds logits $n = 128$, noise of standard deviation 0.5, a bonus of $+4$ on six columns, and a mild recency penalty of $0.04$ per step of distance. The budget is 24, of which `recent` is 8. The retained mass is the sum of the attention matrix over the kept columns. The matrix has 128 rows, so total mass is 128.

| Seed | Planted heavy hitters kept | Oracle | H2O | Window of 24 |
|---|---|---|---|---|
| 0 | 6 of 6 | 100.201 | 97.816 (0.9762) | 4.094 (0.0409) |
| 5 | 6 of 6 | 99.967 | 97.500 (0.9753) | 3.616 (0.0362) |

The last column is a sliding window of the same size, which keeps the newest 24 tokens. It keeps none of the planted tokens and retains four percent of the mass. The check requires at least 0.9 of the oracle's mass, and the reference solution gets 0.975.

Retained attention mass is a proxy. It says the kept keys carry most of this synthetic head's softmax, not that a model's answers stay correct. H2O's paper reports comparable task performance with less than 20% of the KV cache budget, on its models and benchmarks. That claim is theirs.

Two limits follow from how the policy works. An evicted token never comes back, so a token that turns out to matter after it was dropped is lost. And scoring needs the attention probabilities, which a fused kernel such as the one in [chapter 14](/c/14-flash-attention) never writes out, so an engine has to compute them separately.

### SnapKV scores once, at prefill

SnapKV does the same ranking in one shot, after the prompt and before decoding. It takes an observation window at the end of the prompt, 16 tokens in the paper's main setting. It sums the attention from those queries to the earlier prompt tokens, separately for each head, smooths the votes with max pooling of kernel size 5, and keeps the top $k = \lfloor p \times L_\text{prefix} \rfloor$ positions. Those positions, concatenated with the observation window's own entries, are the compressed prompt cache.

The paper compresses the prompt and not the generated tokens, because the prompt is typically the memory bottleneck. It reports a 3.6x generation speedup and 8.2x better memory efficiency at 16K-token inputs. Those figures are the paper's, on its setup, and the lab doesn't reproduce them.

## All the schemes on this model's geometry

Every scheme in this chapter changes bytes, FLOPs, or both, and none of them is free.

The table below takes 16 full-attention layers, 24 query heads, $d_h = 256$, bfloat16, and a context of 32,768 tokens. FLOPs are for one decode token's attention core, over all 16 layers. Every row is computed from the lab's solution by `scripts/figures/07a-kv-economy-attention-variants.py`. Rows marked hypothetical describe a design this model doesn't have.

| Scheme | KV per token | KV at 32k | Attention GFLOP per token | Intensity | Quality risk |
|---|---|---|---|---|---|
| MHA, 24 KV heads (hypothetical) | 384 KiB | 12 GiB | 12.885 | 1.0 | Reference |
| GQA, 4 KV heads (this model) | 64 KiB | 2 GiB | 12.885 | 6.0 | None: trained this way |
| MLA, $d_c = 1024$, $d_r = 128$ (hypothetical) | 36 KiB | 1.125 GiB | 55.163 | 45.7 | A different architecture: you train it, you can't switch it on |
| MLA, $d_c = 512$, $d_r = 64$ (hypothetical) | 18 KiB | 576 MiB | 27.582 | 45.7 | Same, with a tighter latent |
| Window 4096 plus 4 sinks | 64 KiB until the cap, then 0 | 256.25 MiB | 1.612 | 6.0 | Exact recall of tokens outside the window is gone |
| Half local, half global (hypothetical) | 64 KiB, growing | 1152 MiB | 7.248 | 6.0 | The model has to be trained that way |
| H2O or SnapKV, budget 4096 | 64 KiB until the budget, then 0 | 256 MiB | 1.611 | 6.0 | Depends on the task, and evictions are permanent |

Read the table in three passes. MLA is the only row that raises FLOPs, from 12.9 to 55.2 GFLOP, in exchange for fewer bytes. The window and eviction rows cut FLOPs to an eighth of GQA's, and the half-local row cuts them to 7.2 GFLOP, because they read fewer tokens. MHA and GQA have the same FLOPs and differ only in bytes, which is chapter 7's whole point.

On a rated A100 the floors for these 16 layers per decode token are $1110\ \mu s$ for GQA and $624\ \mu s$ for the first MLA row, both set by bytes. The window and H2O rows come to about $139\ \mu s$. These are floors from rated numbers and not measurements, and they ignore everything outside the attention core.

Two interactions are worth knowing before you choose. A cache that has overwritten or evicted tokens is no longer a prefix of the sequence, so [chapter 15a](/c/15a-prefix-caching-and-radix-trees) can't reuse a block that was overwritten or dropped. And none of this touches the 147.75 MiB of fixed recurrent state that the 48 delta layers keep per sequence. This chapter's levers act on the 64 KiB that grows.

## What goes wrong

These are the mistakes the lab is built to catch.

> [!WARNING] Counting only the latent
> MLA's cache has a second part, the shared RoPE key. Dropping it reports $61{,}440$ bytes per token for DeepSeek-V2 instead of $69{,}120$, an error of 11%. The first lab check fails on it.

**Scaling the absorbed scores by the latent width.** The absorbed form reads vectors of width $d_c + d_r$, which tempts you to use $1/\sqrt{d_c + d_r}$. The scale belongs to the original head, $1/\sqrt{d_h + d_r}$. The two forms then differ, and the lab's check against the naive form shows an error of about $0.08$ on its small configuration.

**Treating the window as exclusive.** If the window doesn't count the current token, a window of 3 reads 3 previous tokens and the current one. The lab's mask check shows the first differing entry.

**Letting the sinks roll.** Writing token $p$ to slot $p \bmod (s + W)$ lets the ring overwrite slot 0. After the first wrap the sink is gone. On the planted head the error against full attention is $1.04$ even with 4 sinks configured.

**Evicting by recency alone.** A policy that keeps the newest $B$ tokens is a sliding window. On the planted heavy hitters it drops all six and retains about 4% of the oracle's mass.

**Reading retained mass as accuracy.** The 0.976 against 0.041 contrast belongs to the planted logits. A real model's heavy hitters, and what its answers lose when you drop the rest, have to be measured on its own attention.

## Check your understanding

> [!RECAP]
> - MLA caches $d_c + d_r$ elements per layer per token. DeepSeek-V2 stores 576 against MHA's 32,768, a ratio of 56.9. The shared RoPE key $d_r$ is part of the cache.
> - RoPE sits between $W^Q$ and $W^{UK}$ if you rotate the content key, which blocks absorption. A separate $d_r$-wide rotary key and query fix it, and the score scale becomes $1/\sqrt{d_h + d_r}$.
> - Absorbed decode attends in the latent space. It costs 117 times fewer FLOPs than expanding keys and values at V2's shape and $T = 4096$. Its intensity grows with the head count, $241.8$ at 128 heads, above the ridge.
> - A window of $W$ with $s$ sinks needs $W + s$ slots, in a ring where token $p \ge s$ goes to slot $s + (p - s) \bmod W$. Four sinks bring a planted head's error from $1.069$ to $0.000094$.
> - H2O keeps the newest tokens and the highest accumulated attention, and retains $0.976$ of the oracle's mass on planted heavy hitters. SnapKV makes a similar selection once, at the end of the prompt.
> - MLA raises FLOPs and cuts bytes. Windows and eviction cut both and risk recall. GQA is the baseline all of them are measured against.

> [!QUESTION] Why does MLA at DeepSeek-V2's shape speed up decode attention by 37.9 times and not by 56.9 times?
> The bytes fall by 56.9 times, from $65{,}536$ to $1152$ per cached token per layer. But the absorbed form does $278{,}528$ FLOPs on those bytes, an intensity of $241.8$, which is above the A100's ridge of 161.2. Compute time, $0.893$ ns, is then larger than memory time, $0.595$ ns, and it sets the floor: $33.87 / 0.893 = 37.9$.

> [!QUESTION] Why can't you apply RoPE to the compressed latent and still absorb $W^{UK}$ into the query?
> The key's rotation $R_j$ and the query's rotation $R_t$ combine into $R_{t-j}$, which depends on the distance and sits between $W^Q$ and $W^{UK}$. You can't fold $W^{UK}$ into the query once and reuse it for every cached token. MLA carries position in a separate $d_r$-wide key that every head shares, and keeps the latent free of rotation.

> [!QUESTION] A 4096-token window with 4 sinks holds 4100 slots. How many bytes is that for this model, and how does it compare with 32k of GQA?
> Each token costs $65{,}536$ bytes, so $4100 \times 65{,}536 = 268{,}697{,}600$ bytes, 256.25 MiB. Full GQA at 32,768 tokens is 2 GiB, which is $7.992$ times larger. The window's bytes stop growing at the cap, and the full cache's don't.

> [!QUESTION] The planted sink gives full attention 0.9999 of its mass on token 0. Why does a window without sinks have an error above 1?
> Without token 0, the softmax renormalizes over the 32 remaining tokens. Those tokens carried almost no weight before, and now they carry all of it, so the weighted sum of values is mostly a different set of vectors. The error is about the size of the output itself.

> [!QUESTION] The oracle retains 100.2 of 128 units of attention mass and H2O retains 97.8. Why can't an online H2O reach the oracle?
> The oracle picks the top-$B$ columns by total attention over every query, including queries that haven't happened yet. H2O decides at each step from the scores it has so far, and it can't bring back a token it already dropped. It also has to keep the newest 8 tokens whether or not they are heavy.

## Lab

> [!TRY]
> Count MLA's and GQA's bytes, write the naive and absorbed MLA forms, count their FLOPs and intensity, build the window mask and a rolling cache with sinks, and run H2O eviction. You pass when the absorbed form equals the naive one to $10^{-9}$, the rolling cache equals masked full attention at every step, and H2O retains at least 0.9 of the oracle's attention mass on both planted layers.

Implement the following functions and class:

- `gqa_kv_bytes` and `mla_kv_bytes`. Bytes per token over all layers, bfloat16 by default.
- `window_kv_bytes`. GQA bytes in a cache of at most `window + sinks` tokens.
- `mla_attention_naive`. Expand keys and values from the latent, then attend with scale $1/\sqrt{d_h + d_r}$.
- `mla_attention_absorbed`. Absorb $W^{UK}$ into the query, sum the latents, then apply $W^{UV}$.
- `gqa_decode_cost` and `mla_decode_cost`. FLOPs, bytes, and intensity for one decode token.
- `sliding_window_mask`. A boolean matrix where the window counts the current token.
- `RollingKVCache`. A ring of `sinks + window` slots with `append`, `gather`, and `attend`.
- `evict_h2o`. The online policy above. Return the kept positions, sorted.

The first check calls only `mla_kv_bytes`. Each later check adds one function. The lab uses float64 and seeded draws throughout and runs in well under a second.

## Further reading

- [DeepSeek-V2: a strong, economical, and efficient mixture-of-experts language model](https://arxiv.org/abs/2405.04434). Sections 2.1.2 to 2.1.4 and Table 1 give MLA, the decoupled RoPE, and the cache comparison. Section 3.1.2 gives the dimensions used here.
- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437). Its hyperparameters section lists 61 layers and the same MLA widths.
- [Mistral 7B](https://arxiv.org/abs/2310.06825). The sliding window, the rolling buffer cache, and the 8x memory claim at 32k.
- [Efficient streaming language models with attention sinks](https://arxiv.org/abs/2309.17453). The StreamingLLM claims quoted here come from its abstract. The [paper page](https://papers.ifkash.dev/streamingllm/) covers the perplexity collapse of window attention and the learnable sink token.
- [H2O: heavy-hitter oracle for efficient generative inference of large language models](https://arxiv.org/abs/2306.14048). The power-law observation and the recent-plus-heavy-hitter policy.
- [SnapKV: LLM knows what you are looking for before generation](https://arxiv.org/abs/2404.14469). The observation window, pooling, and the 3.6x and 8.2x figures.
- [Gemma 2: improving open language models at a practical size](https://arxiv.org/abs/2408.00118). Section 2 describes the local and global alternation.
- [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556). DeepSeek Sparse Attention keeps the full cache but attends to only the top-k keys that a small lightning indexer picks for each query. The [paper page](https://papers.ifkash.dev/deepseek-v32/) builds the indexer and the top-k selection.
