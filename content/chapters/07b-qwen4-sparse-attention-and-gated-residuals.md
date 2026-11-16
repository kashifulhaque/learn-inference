---
title: "Qwen4: sparse attention, gated residuals, and n-gram embeddings"
slug: 07b-qwen4-sparse-attention-and-gated-residuals
part: "Part 2 — A forward pass"
summary: What Qwen's three published Qwen4-Max architecture changes cost and save at inference time, in KV bytes, bytes read per decode step, FLOPs, and parameter memory, with small illustrative versions you build and check.
minutes: 100
gpu: false
objectives:
  - "Separate what Qwen published about Qwen4-Max's architecture from what it didn't, and label every design detail in this chapter as illustrative."
  - "Explain why block-sparse attention cuts bytes read and FLOPs per decode step but not the KV cache, and compute both for the Qwen3.8 card's geometry."
  - "Implement a micro-block indexer, a top-k block selection, and a sparse decode step, and check it against dense attention restricted to the same blocks."
  - "Write a four-stream gated residual, show it reduces to a plain residual under identity-like gates, and count its activation bytes and gate FLOPs."
  - "Hash n-grams into an embedding table and account for its parameter memory against its per-token compute."
lab: 07b-qwen4-blocks
---

# Qwen4: sparse attention, gated residuals, and n-gram embeddings

> [!TLDR]
> - Qwen's Qwen4-Max blog names three architecture changes: GDN plus Qwen Sparse Attention (QSA), a four-branch Gated Residual, and an N-gram Embedding. It publishes no block size, top-k, indexer or gate equations, or hashing scheme, so everything below the headline is this course's illustrative design.
> - Sparse attention doesn't shrink the KV cache, because every key is still stored. It cuts the bytes read and the FLOPs per decode step to the kept blocks plus an index scan. On the Qwen3.8 card's geometry at 1,000,000 tokens, an illustrative budget cuts the per-layer read from $4.096$ GB to $18.8$ MB.
> - Four residual streams multiply the activation bytes a layer touches by 4 and add about $2.4 \times 10^{-4}$ of an MoE layer's FLOPs in gates. The KV cache doesn't change.
> - The n-gram table adds 328B parameters, about 13.7% of the 2.4T backbone, and almost no compute. It costs memory capacity, not FLOPs.
> - The three changes move different resources, and a serving engine has to budget for each.

[Chapter 7](/c/07-grouped-query-attention) and [chapter 7a](/c/07a-kv-economy-attention-variants) shrink the KV cache. This chapter covers a different lever that Qwen4-Max uses: keep the cache, and read less of it. It also covers two changes that have nothing to do with attention, a wider residual path and a big lookup table, because an engine pays for them in other ways.

The source is a short product blog, and it states the changes in a few sentences. So the chapter works in two layers. The first says what Qwen stated. The second builds a small version of each idea, so you can count its bytes and FLOPs. Where the chapter shows a block size, a gate equation, or a hash, it says that Qwen hasn't published its own.

The lab is CPU-only and runs in float64. Nothing in this chapter is a measurement.

## Before you start

**KV bytes per token, from [chapter 7](/c/07-grouped-query-attention).** One full-attention layer stores $2 H_{kv} d_h b$ bytes of keys and values per token, with $H_{kv}$ KV heads of width $d_h$ at $b$ bytes per element. Only full-attention layers hold a cache that grows with the context.

**Decode is byte-bound, from [chapter 10](/c/10-roofline).** One decode token does a few FLOPs per byte it reads, far below the A100's ridge point of $161.2$. Time follows bytes, so cutting bytes read is what speeds a step up.

**The Qwen3.8 card, a stand-in for Qwen4.** The Hugging Face card for Qwen3.8-2.4T-A95B, the predecessor, lists 92 layers in a pattern of three Gated DeltaNet layers then one gated-attention layer, repeated 23 times. Attention has 64 query heads, 4 KV heads, and head width 256. The hidden size is 8192. Qwen4-Max's blog states none of these, so every example that needs a geometry uses this card and says so.

## What Qwen published, and what it didn't

The blog states the three changes in one sentence each and gives three parameter counts. Everything else is open.

| Change | What the blog states | What it doesn't state |
|---|---|---|
| GDN + QSA | Three Gated DeltaNet layers for every QSA layer, a 3:1 ratio. A lightweight indexer selects context at micro-block granularity, which cuts both attention and indexing costs. | Block size, top-k, the indexer's form, layer count, head counts |
| Gated Residual (GR) | Four parallel residual branches with dynamic read and write gates, for information flow and training stability. | The gate equations, where the gates sit, what they're conditioned on |
| N-gram Embedding | Context-based lookups give representations of local token patterns at the second layer, with little extra compute. | The value of $n$, the hashing scheme, the table layout, the dtype |

The blog also states 2.4T backbone parameters, 71B activated per token, and 328B additional N-gram Embedding parameters. Against Qwen3.8's 95B active, 71B is $25.3\%$ fewer, which matches the blog's "25% fewer."

> [!NOTE] Where the facts come from
> The blog renders only with JavaScript, so a plain fetch returns an empty page. The statements above are a reading of it, collected on 2026-11-16, and the chapter's fact entries in `docs/FACTS.md` mark each one. The quality and cost claims in the blog, such as the benchmark gains and the lower cost per task, are Qwen's and this chapter doesn't evaluate them.

Three of the ideas have public relatives, and the chapter cites them only for what their own abstracts say. Native Sparse Attention (NSA) combines coarse token compression with fine token selection. MoBA applies mixture-of-experts routing to blocks of attention. The DeepSeek-V3.2 paper describes DeepSeek Sparse Attention (DSA), which scores earlier tokens with a *lightning indexer*. Hyper-Connections and mHC widen the residual stream, and Engram revisits N-gram embeddings. ==Nothing in the blog says Qwen's design matches any of them.== They're here to give you a known design to read next to the illustrative one.

## Three Gated DeltaNet layers for every sparse-attention layer

The 3:1 ratio sets how much of the model has a KV cache at all, and sparse attention then changes what that quarter costs per step.

The blog states the 3:1 ratio for Qwen4 but not a layer count, so take the Qwen3.8 card's 92-layer pattern as a stand-in. Of 92 layers, $92 / 4 = 23$ are attention and $69$ are Gated DeltaNet. Each attention layer stores keys and values for 4 KV heads of width 256 in bfloat16:

$$
\hla{2 \times 4 \times 256 \times 2} = 4096 \text{ bytes per token per layer},
\qquad
23 \times 4096 = \boxed{94{,}208 \text{ bytes per token}}
$$

That is 92 KiB per token. At 1,000,000 tokens one sequence holds $94.208$ GB, or $87.74$ GiB. The course's own 27B model has 16 attention layers and $65{,}536$ bytes per token, so the card's model is the bigger case.

The 69 delta layers don't grow. Each keeps a $d_k \times d_v$ state per value head, as [chapter 6](/c/06-gated-delta-rule) derives. With 128 value heads and head width 128, that is $69 \times 128 \times 128 \times 128 = 144{,}703{,}488$ elements per sequence, $289$ MB in bfloat16 or $579$ MB in float32, whatever the context. The count ignores any convolution state. It comes from the card's head counts, and Qwen4's are unpublished.

So at long context the cache is the 23 attention layers' $94.2$ GB, and the recurrent state is under a percent of it. A sparse-attention layer attacks the first number, but only the *read*.

## A block-sparse decode step, built from the idea

QSA's published description, a light indexer choosing micro-blocks, has a natural minimal form: score blocks cheaply, keep a few, attend only to those. The lab builds that form. ==It is this course's design, and Qwen hasn't published its own.==

For one decode token, take a query $q$ of width $d$ per head and cached keys $K$ and values $V$. The design adds a second, narrow set of *index keys* $k^{\text{idx}}_j$ of width $d_{\text{idx}}$ for every token $j$, and a narrow index query $q^{\text{idx}}$. Four steps follow:

1. **Pool.** Average the index keys of each block of $b$ tokens into one pooled key. Context $T$ gives $\lceil T/b \rceil$ pooled keys.
2. **Score.** Dot $q^{\text{idx}}$ with every pooled key. That is one score per block, not per token.
3. **Select.** Keep the $k$ blocks with the highest scores. A tie goes to the lower block.
4. **Attend.** Run ordinary softmax attention over the tokens in those $k$ blocks only.

This is the shape of DSA's lightning indexer with one change: DSA's paper gives a per-token score $I_{t,s} = \sum_j w_{t,j} \operatorname{ReLU}(q_{t,j} \cdot k_s)$ with a small number of indexer heads, and selects 2048 tokens. The lab scores blocks, because the blog says QSA works at micro-block granularity, and it uses one indexer head and a plain dot product to stay small.

### What one step reads

Count the bytes the step reads from the cache. The indexer scans the pooled keys, and attention reads the kept tokens' keys and values:

$$
\boxed{B_{\text{sparse}} = \hla{\left\lceil \tfrac{T}{b} \right\rceil d_{\text{idx}}\, s_{\text{idx}}} + \hlb{k\, b\;\, 2 H_{kv} d_h\, s}}
$$

Here $\hla{\cdot}$ is the index scan, with $s_{\text{idx}}$ bytes per index element, and $\hlb{\cdot}$ is the kept KV, with $s$ bytes per KV element. Dense attention reads $T \cdot 2 H_{kv} d_h s$ instead. The second term doesn't depend on $T$, and the first grows as $T/b$, so a larger block shrinks the scan.

> [!EXAMPLE] One Qwen3.8-shaped layer at 1,000,000 tokens
> Use $b = 64$, $k = 64$ blocks, so $4096$ kept tokens, and $d_{\text{idx}} = 64$ in bfloat16. Those three numbers are illustrative.
>
> - Dense read: $10^6 \times 4096 = 4{,}096{,}000{,}000$ bytes.
> - Index scan: $\lceil 10^6 / 64 \rceil = 15{,}625$ blocks times $64 \times 2$ bytes is $2{,}000{,}000$ bytes.
> - Kept KV: $4096 \times 4096 = 16{,}777{,}216$ bytes.
> - Total $18{,}777{,}216$ bytes, which is $218.1$ times fewer than dense.
>
> Over 23 layers the dense read is $94.208$ GB and the sparse read is $431.9$ MB.

The FLOPs follow the same split. Dense decode attention does $4 H d_h T$ FLOPs, so $65.5$ GFLOP per layer at $T = 10^6$ with 64 query heads of width 256. The sparse step does $4 H d_h \cdot kb = 268$ MFLOP for attention and $2 \lceil T/b \rceil d_{\text{idx}} = 2$ MFLOP for the scan. Over 23 layers that is $1.507$ TFLOP against $6.22$ GFLOP.

```viz
07b-sparse-decode-bytes
```

The table covers four context lengths with the same budget.

| Context | Blocks | Dense bytes per layer | Index scan | Kept KV | Dense over sparse |
|---|---|---|---|---|---|
| 32,768 | 512 | 134,217,728 | 65,536 | 16,777,216 | 8.0 |
| 131,072 | 2,048 | 536,870,912 | 262,144 | 16,777,216 | 31.5 |
| 262,144 | 4,096 | 1,073,741,824 | 524,288 | 16,777,216 | 62.1 |
| 1,000,000 | 15,625 | 4,096,000,000 | 2,000,000 | 16,777,216 | 218.1 |

The saving grows with the context, because the kept term is flat. At $32$k it is only $8.0$ times.

### The block size trades scan bytes against waste

A smaller block gives the indexer finer choices and costs more to scan. Hold the kept budget at 4096 tokens and the context at $10^6$. Block 16 needs $62{,}500$ pooled keys, an $8.0$ MB scan, and a total of $24.8$ MB. Block 256 needs $3{,}907$, a $0.5$ MB scan, and $17.3$ MB. The kept KV stays $16.8$ MB.

What the table can't show is quality. A coarse block keeps tokens it doesn't need and may miss a lone relevant token inside a block the pooled key scored low. The lab's needle check shows the good case, where one block holds the mass. How a real model's attention behaves is a question about its trained weights, and the blog's description of QSA gives no number for it.

> [!KEY] Sparse attention cuts the read, not the cache
> Every key is still stored, because a later query may select any block. The KV cache stays $94.2$ GB per 1M-token sequence, and the pooled index keys add $2$ MB per layer, $0.05\%$ of it. What shrinks is the bytes read and the FLOPs per decode step. A server's batch size is still bound by cache capacity.

That distinction decides what the engine gains. Capacity limits how many sequences fit, so the sparse step raises *speed* per sequence, not *count* of sequences. To raise the count, you'd combine it with the levers in [chapter 7a](/c/07a-kv-economy-attention-variants) or [chapter 18a](/c/18a-int4-and-kv-quantization).

### Where the saving matters at batch 1

At batch 1 the weights compete with the cache. Qwen4-Max activates 71B parameters per token, so if they're bfloat16 and all read once, a decode step reads $142$ GB of weights. That's an assumption: the blog doesn't give a dtype. Against it, the dense 1M-token KV read is $94.2$ GB, which is $0.66$ of the weight read. The sparse read is $431.9$ MB, $0.003$ of it.

The rated A100 bandwidth of 1935 GB/s sets floors of $48.7$ ms for the dense KV read and $0.223$ ms for the sparse one. These are rated-number floors for the attention reads alone. The model doesn't fit one A100 anyway, so they only show how the two reads compare. Weights are shared across a batch and KV isn't, so the cache's share of a step grows with the batch size.

## What the engine has to do differently

A sparse step changes the kernel and the cache layout, and each change is a design choice, not something Qwen has stated.

- **Gathers replace a contiguous scan.** The kept blocks are scattered across the cache. If a block is a multiple of a paged-attention page, as in [chapter 15](/c/15-paged-attention), a kept block is a set of whole pages and the gather needs no partial page. That alignment is a design choice for this chapter's engine.
- **The indexer has its own cache.** The pooled keys are extra state per sequence, and they must be updated as tokens arrive, because the last block is partial until it fills.
- **Selection is per layer, and it depends on the query.** Two sequences that share a prefix, as in [chapter 15a](/c/15a-prefix-caching-and-radix-trees), share keys and index keys, which depend only on the prefix, but not the blocks they select.
- **Top-k is an approximation.** A block the indexer misses is a key the step doesn't see. Every sparse design makes that trade, and its quality depends on training the indexer with the attention, which the lab doesn't do.

## A residual path with four streams

A standard residual adds each layer's output to one stream of width $d$. A four-stream gated residual keeps four streams, lets a *read gate* mix them into the layer input, and lets a *write gate* spread the output back.

Qwen's blog says only that Gated Residual has four parallel residual branches with dynamic read and write gates. Hyper-Connections gives a published form with the same shape. Its paper uses $n$ parallel streams, with $n = 4$ in its main experiments. A weighted sum of the streams forms the layer input, and the output is written back to the streams with a second set of weights.

The paper's formulation also mixes the streams among themselves with an $n \times n$ matrix, and it initializes the matrices to match a Pre-Norm residual. mHC reports that unconstrained Hyper-Connections break the identity mapping of a residual and cause instability and extra memory-access cost, and constrains the mixing to restore it. Whether Qwen's design resembles either is unstated.

The lab uses the smallest form that has a read and a write gate. With streams $X$ of shape $(n, d)$, a read gate $\hla{r} \in \mathbb{R}^n$, a write gate $\hlb{w} \in \mathbb{R}^n$, and a layer $f$:

$$
\boxed{X' = X + \hlb{w}\, f\!\left(\hla{r}^{\top} X\right)}
$$

The row vector $\hla{r}^{\top} X$ is the layer input, a weighted sum of the streams. The output $f(\cdot)$ has width $d$, and stream $i$ gets $\hlb{w_i}$ times it. The lab's dynamic gates are $r = \operatorname{softmax}(X w_r)$ and $w = \sigma(X w_w)$, both chosen here and not Qwen's.

### Identity-like gates give a plain residual

Two settings recover the ordinary residual, and the lab tests both.

- **Equal streams.** If every stream equals $x$, $r_i = 1/n$, and $w_i = 1$, then $r^{\top}X = x$ and every stream becomes $x + f(x)$.
- **One-hot gates.** If $r = w = e_k$, then stream $k$ becomes $x_k + f(x_k)$ and the other three don't change.

The second one matches how the Hyper-Connections paper initializes its read weights, one-hot on a stream, so training starts from a Pre-Norm residual. The identity check is the reason to build it: a gate set you can't reduce to a residual is a design you can't debug.

### What four streams cost

Four streams change the bytes a layer touches and add small gate arithmetic. Count both for the card's hidden size, $d = 8192$ in bfloat16:

| Quantity | One stream | Four streams | Ratio |
|---|---|---|---|
| Residual bytes held per token | 16,384 | 65,536 | 4 |
| Read plus write of the streams per layer | 32,768 | 131,072 | 4 |
| Gate FLOPs per token per layer | 0 | 262,144 | |

The gate FLOPs are $8nd$ in the lab's form: two gate dot products, the read mix, and the write-back. An MoE layer with 11 active experts, three matrices of $8192 \times 2048$ each, does $1{,}107{,}296{,}256$ FLOPs per token. The gates are $2.4 \times 10^{-4}$ of that.

The cost is bytes, not FLOPs. The elementwise reads and writes of the streams are memory-bound, so they want fusing with the surrounding norm, as [chapter 13](/c/13-fusion-in-triton) does for RMSNorm. If a pipeline stage boundary sends the streams, as in [chapter 19b](/c/19b-pipeline-and-context-parallelism), it sends four times the hidden-state bytes. That is arithmetic on the lab's form, not something Qwen has said about its deployment.

The KV cache doesn't change. The streams are activations that live for one token's pass, and they aren't cached. The blog's reason for GR is training stability and information flow, and those are training properties that this chapter doesn't test.

## A table of n-grams, and where its cost lands

An *N-gram Embedding* gives each position an extra vector chosen by the last few token ids, and adds it to the hidden state at the second layer.

The blog says the lookups are context-based, give representations of local token patterns, and add capacity with little extra compute. It adds 328B parameters to a 2.4T backbone. The ratio is $328 / 2400 = 13.67\%$, for a total of $2.728$T parameters.

The mechanism is a table $E$ of shape $(M, d)$ and a map from the last $n$ ids to a row. The lab's map is a hash, so a trigram lands in one of $M$ rows with no vocabulary-sized table:

$$
\hla{h_t} = \operatorname{hash}(x_{t-n+1}, \dots, x_t) \bmod M,
\qquad
\hlb{e_t} = E[h_t],
\qquad
\text{hidden}_t \mathrel{+}= e_t \text{ at layer } 2.
$$

Here $h_t$ is the row index and $e_t$ is the vector added. The lab pads positions before the start with a reserved id, so every position hashes exactly $n$ ids. ==Qwen hasn't published its hashing scheme, its $n$, or whether it uses several tables.== The lab's FNV-1a plus a finalizer, with $n$ a parameter, is a stand-in. Hash collisions are a design cost: two different n-grams that share a row share a vector.

### Parameters are capacity, FLOPs are tiny

A lookup does no arithmetic, and the add costs $d$ FLOPs. The parameters scale with the table and the FLOPs don't:

$$
\text{params} = \hla{M d},
\qquad
\text{FLOPs per token} = \hlb{d},
\qquad
\text{bytes read per token} = d\, s.
$$

> [!EXAMPLE] A table of $2^{20}$ rows at $d = 8192$
> Parameters: $2^{20} \times 8192 = 8{,}589{,}934{,}592$, which is $17.2$ GB in bfloat16. Per token the layer reads one row of $16{,}384$ bytes and does $8192$ FLOPs. A dense layer with the same parameter count would do $2 \times 8.59 \times 10^9 = 17.2$ GFLOP per token. Growing the table 16 times multiplies the parameters by 16 and leaves the per-token FLOPs and bytes read unchanged.

So the n-gram table is a capacity lever, in contrast to the MoE experts, which are also capacity but cost a matrix multiply when active. At the blog's size, 328B parameters in bfloat16 would be $656$ GB, if that's the dtype, against $4.8$ TB for the backbone. The per-token row read of $16{,}384$ bytes is $1.15 \times 10^{-7}$ of the $142$ GB active-weight read.

What the table costs a server is memory, not time. It has to live somewhere. Engram's abstract reports that deterministic addressing allows prefetching from host memory with negligible overhead. That is Engram's claim for its own design, and whether Qwen4-Max keeps its table in GPU memory, in host memory, or sharded is not stated. The row for a token depends only on the token ids, so an engine can compute every address for a prompt before running a layer. That's a property of any lookup keyed on token ids, and it's why the lab's causality check matters.

## Three changes, three different budgets

Each change spends a different resource, so a capacity plan has to track each one.

| Change | Cache bytes | Bytes read per step | FLOPs per step | Parameter memory | Activation bytes |
|---|---|---|---|---|---|
| Block-sparse attention | Same, plus a small pooled-key cache | Much lower at long context | Much lower at long context | Indexer weights, unstated | Same |
| Four-stream gated residual | Same | Streams read and written 4 times | Gates, $2.4 \times 10^{-4}$ of an MoE layer | Gate weights, unstated | 4 times the residual |
| N-gram embedding | Same | One row per token | $d$ per token | 328B more, $13.7\%$ of the backbone | Same |

The table gives the direction of each cost, with the magnitudes from this chapter's illustrative form. For Qwen4 itself, only the 3:1 layout and the parameter counts are published.

## What goes wrong

These are the mistakes the lab is built to catch.

> [!WARNING] Reading "sparse" as "smaller cache"
> A sparse step reads fewer bytes and stores the same number. If you size a batch from the sparse read, you'll run out of memory at $94.2$ GB per 1M-token sequence. The lab's `sparse_decode_cost` reports `extra_cache_bytes` beside `dense_bytes` so the two stay apart.

**Softmaxing over all keys, then masking.** If you softmax over $T$ tokens and zero the dropped ones afterward, the weights over the kept tokens don't sum to 1. The sparse step has to renormalize over the kept tokens, and the check against masked dense attention catches the difference.

**Scoring tokens, not blocks.** An indexer that keeps the best tokens returns token ids, and the step gathers the wrong rows. The first selection check expects block ids.

**Forgetting the short last block.** With $T = 203$ and $b = 16$ the last block has 11 tokens. Clipping it wrongly reads past the cache or drops those keys.

**Writing the gate before the skip.** If `write` scales the stream and the layer output is added to the scaled stream, the identity setting no longer gives $x + f(x)$. Check the equal-stream case first.

**Hashing with a sum.** `sum(ids)` is deterministic and in range, and it maps $(a, b)$ and $(b, a)$ to the same row. The lab checks order sensitivity and an even spread over 64 buckets.

## Check your understanding

> [!RECAP]
> - Qwen's blog names three changes: GDN + QSA at 3:1, a four-branch Gated Residual, and an N-gram Embedding. It gives parameter counts, 2.4T, 71B active, and 328B n-gram, and no block size, top-k, equations, or hash.
> - Block-sparse attention reads $B_{\text{sparse}} = \lceil T/b \rceil d_{\text{idx}} s_{\text{idx}} + kb \cdot 2H_{kv}d_h s$ bytes. The KV cache is unchanged. On the Qwen3.8 card at 1M tokens, the illustrative budget gives $218.1$ times fewer bytes per layer.
> - The saving grows with context because the kept term is flat. At 32k it is $8.0$ times.
> - Four streams hold 4 times the residual bytes and add gates worth $2.4 \times 10^{-4}$ of an MoE layer. Identity-like gates reduce the layer to a plain residual.
> - The n-gram table adds parameters, not FLOPs: $Md$ parameters, $d$ FLOPs and one row read per token. It costs memory capacity.

> [!QUESTION] Why doesn't block-sparse attention let a server fit more sequences?
> Every key stays in the cache, because a later query may select any block. The step reads less and stores the same, so capacity is the same $94.2$ GB per 1M-token sequence on the card's geometry. The pooled index keys even add a little.

> [!QUESTION] At 1,000,000 tokens, the example's indexer scan is 2.0 MB and the kept KV is 16.8 MB. What happens to each if the block size doubles with the same 4096-token budget?
> The scan halves, to $1.0$ MB, because there are $7{,}813$ pooled keys instead of $15{,}625$. The kept KV stays $16{,}777{,}216$ bytes, because the budget is in tokens. The total falls from $18.8$ MB to $17.8$ MB, a ratio of $230.4$ against $218.1$.

> [!QUESTION] Why is the 32k-token ratio of 8.0 so much smaller than the 1M-token ratio of 218.1?
> The kept term is $16{,}777{,}216$ bytes at both lengths, while dense grows with $T$. At 32,768 tokens dense is $134{,}217{,}728$ bytes, $8.0$ times the kept term, and the scan adds a little on top. At 1M tokens dense is $244$ times the kept term, and the $2$ MB scan trims the ratio to $218.1$.

> [!QUESTION] With $r = w = e_2$, what does the four-stream layer do to streams 0, 1, and 3?
> Nothing. The layer input is $r^{\top}X = x_2$, the output is $f(x_2)$, and the write gate adds it to stream 2 only. Streams 0, 1, and 3 are unchanged, and stream 2 becomes $x_2 + f(x_2)$, an ordinary residual.

> [!QUESTION] The n-gram table has 328B parameters and the model has 71B active. Why isn't the n-gram table's compute proportional to 328B?
> The lookup reads one row per token, $d$ elements, and adds it. The work is $d$ FLOPs per table. The table's parameters are storage that most tokens never touch, the way most experts in an MoE layer sit idle for a given token. Unlike experts, a row costs no matrix multiply when it is read.

## Lab

> [!TRY]
> Build a block-sparse decode step and its cost function, a four-stream gated residual, and a hashed n-gram embedding. You pass when the sparse step equals dense attention restricted to the same blocks, the cost function reproduces the bytes in this chapter, equal streams with identity gates give $x + f(x)$, the hash is deterministic, uniform, and order-sensitive, and the n-gram accounting shows parameters growing with the table while FLOPs don't.

Implement the following functions:

- `pool_blocks` and `select_blocks`. Mean-pool index keys per block, then keep the top-$k$ blocks, ties to the lower block, ids ascending.
- `block_token_indices` and `sparse_attention_step`. Gather the kept tokens, clipping the last block, and softmax over them only.
- `sparse_decode_cost`. Index-scan bytes, kept KV bytes, dense bytes, the ratio, and the FLOPs.
- `gated_residual` and `dynamic_gates`. $X + w\,f(r^{\top}X)$, with a softmax read gate and a sigmoid write gate.
- `gated_residual_cost`. Stream bytes, read-write bytes, and gate FLOPs.
- `ngram_hash`, `ngram_indices`, and `ngram_embedding`. A deterministic hash of the last $n$ ids, with padding before the start.
- `inject_ngram` and `ngram_accounting`. Add the rows at layer 2, and count parameters, bytes read, and FLOPs.

The first check calls only `pool_blocks`. Each later check adds a function. All 18 checks run in well under a second, and the designs are the lab's, not Qwen's.

## Further reading

- [Qwen4-Max: your new teammate has arrived](https://qwen.ai/blog?id=qwen4-max). The source for the three changes, the 3:1 ratio, and the 2.4T, 71B, and 328B counts. The page needs JavaScript to render.
- [Qwen3.8-2.4T-A95B model card](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B). The predecessor's layout and head counts, used here as the running example.
- [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556). DeepSeek Sparse Attention, its lightning indexer, and the 2048 selected tokens.
- [Native sparse attention](https://arxiv.org/abs/2502.11089). Coarse token compression with fine token selection, built for hardware.
- [MoBA: mixture of block attention](https://arxiv.org/abs/2502.13189). Mixture-of-experts routing applied to blocks of attention.
- [Hyper-connections](https://arxiv.org/abs/2409.19606). The $n$-stream residual with width and depth connections, $n = 4$ in the main experiments.
- [mHC: manifold-constrained hyper-connections](https://arxiv.org/abs/2512.24880). Restores the identity mapping that hyper-connections break.
- [Conditional memory via scalable lookup](https://arxiv.org/abs/2601.07372). Engram, a modern N-gram embedding with deterministic addressing.
