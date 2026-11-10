---
title: Disaggregated prefill and decode
slug: 17b-disaggregated-serving
part: "Part 5 — Serving"
summary: Why a colocated GPU trades one latency for the other even with chunked prefill, what a prefill GPU ships to a decode GPU, how fast the link must be, and when splitting the GPUs loses.
minutes: 120
gpu: false
objectives:
  - Show with a simulated clock that a prefill chunk trades the decoders' worst gap against the prompt's time to first token.
  - Count the bytes a prefill GPU ships for this hybrid, 16 growing attention layers plus a fixed 147.8 MiB state, and compare them with a dense stack.
  - Find the least link rate that keeps the copy inside a TTFT budget, with and without layer-wise streaming.
  - Search the prefill:decode GPU split by goodput and check it against brute force.
  - Name the traces and links on which disaggregation loses to a tuned colocated engine.
lab: 17b-disaggregation
---

# Disaggregated prefill and decode

> [!TLDR]
> - Prefill is compute-bound and decode is memory-bound. Putting both on one GPU makes each slow the other. On a simulated GPU, a decode step that takes 31.7 ms grows to 93.8 ms when a 512-token prompt chunk rides along. A chunk of 160 tokens or fewer costs 3% or less, but it stretches a long prompt's TTFT.
> - Disaggregation gives prefill and decode their own GPUs. The price is a copy: 659.75 MiB for an 8,192-token prompt on this hybrid, of which 147.75 MiB is the fixed recurrent state. A dense stack would ship 2,048 MiB.
> - At 300 GB/s the copy of an 8,192-token prompt takes 2.3 ms. At 25 Gbit/s Ethernet it takes 221 ms. Streaming each layer as it finishes hides most of it: the exposed wait falls to 0.11 ms and 10.7 ms.
> - On a prefill-heavy trace of 8 GPUs, the best split is 7 prefill to 1 decode, and it beats the best colocated chunk size by 56% in goodput. On a decode-heavy trace the best split ties colocation, and short prompts over a slow link lose outright.
> - Every number here is simulated from chapter 17a's step-time floor. None is an A100 or network measurement.

[Chapter 17a](/c/17a-performance-model) gave you the time of one step. [Chapter 16](/c/16-continuous-batching) gave you a scheduler that mixes prompt chunks into decode steps. This chapter asks what happens when you stop mixing: send the prompt to one set of GPUs, ship its state to another set, and decode there.

Large deployments do this. DistServe, Splitwise, Mooncake, and TetriInfer all split the two phases onto separate machines, and vLLM has a connector interface for it. The case rests on [chapter 0's](/c/00-what-you-are-building) asymmetry: prefill is limited by arithmetic and decode by bytes. This chapter tests that case on this model, with a simulator you write. The result is more nuanced than the papers' headlines. Disaggregation wins some traces by a wide margin, ties others, and loses when prompts are short or the link is slow.

The lab runs on the CPU. The simulator advances a clock, not a GPU, and the step time comes from chapter 17a's floor.

## Before you start

**The step-time floor, from [chapter 17a](/c/17a-performance-model).** One forward pass takes $\max(F/C, B/\beta)$ plus a launch term of 2.0 ms. A decode step at batch 1 and context 1 is 29.962 ms. A cold 2,048-token prefill is 360.094 ms. The lab's `step_time` generalizes that floor to a step with some decodes and some prompt chunks. It reproduces both numbers.

**Chunked prefill, from [chapter 16](/c/16-continuous-batching).** The scheduler splits a prompt into chunks and adds one chunk to each decode step, so decoders never wait for a whole prompt. Sarathi-Serve calls this stall-free batching. The chunk size is the knob.

**The state, from [chapter 2](/c/02-memory-arithmetic) and [chapter 9](/c/09-the-kv-cache).** The 16 full-attention layers keep 64 KiB of KV per prompt token. Each of the 48 linear layers keeps 3,227,648 bytes: a 3,145,728-byte delta-rule matrix plus an 81,920-byte convolution window. The 48 layers together hold 154,927,104 bytes, 147.75 MiB, however long the prompt is. Full attention sits on every fourth layer, counting from index 3.

**Link rates, one way.** Chapter 19 uses NVLink at 300 GB/s per A100 in each direction and PCIe 4.0 x16 at 32 GB/s in each direction. Chapter 17a's tensor-parallel model still uses 600 GB/s for NVLink. That figure adds both directions. A KV copy goes one way, so this chapter uses 300 GB/s. Between nodes, InfiniBand HDR is 200 Gbit/s and NDR is 400 Gbit/s per 4x port, which is 25 and 50 GB/s. Ethernet at 100 and 25 Gbit/s is 12.5 and 3.125 GB/s. Those last two are the line rate divided by 8, and they ignore protocol overhead.

## What interference looks like

This section asks how much a prompt chunk slows the decoders that share its step, and why a smaller chunk is not a free fix.

A step reads the weights once, however many tokens it carries. Eight decoders at context 2,048 take 31.652 ms. That step is memory-bound, so the arithmetic units sit mostly idle: the decodes use about 1.4 ms of compute. A prompt chunk adds compute and little else, so a small chunk fits under the weight read.

How small? The step is $\max(F/C, B/\beta) + t_{\text{launch}}$, and a chunk of $n$ tokens adds $2 \times 26.9 \times 10^9 \times n$ FLOPs. Compute passes the byte floor at about 164 tokens. Past that, every extra token costs its full 0.173 ms.

| Chunk added to 8 decodes | Step | Against decode only |
|---|---|---|
| none | 31.65 ms | 1.00x |
| 64 tokens | 32.01 ms | 1.01x |
| 160 tokens | 32.53 ms | 1.03x |
| 192 tokens | 37.25 ms | 1.18x |
| 256 tokens | 48.54 ms | 1.53x |
| 512 tokens | 93.76 ms | 2.96x |
| 1,024 tokens | 184.44 ms | 5.83x |

Every decoder in the step waits for the whole step. ==A 512-token chunk triples the gap between two tokens for every request that shares it.== The 160-token chunk is nearly free for the decoders. It is expensive for the prompt, because an 8,192-token prompt then needs 52 steps.

The lab's planted trace shows both sides. Eight decoders, each prompt 128 tokens and output 300, start at time 0. One 8,192-token prompt arrives at 2 s. The *worst gap* of a request is its longest gap between tokens, after the first gap, which holds the handoff.

| Policy | Decoders' worst gap | Long prompt's TTFT |
|---|---|---|
| Colocated, whole prompt in one step | 1,467.48 ms | 1.485 s |
| Colocated, 2,048-token chunks | 377.34 ms | 1.495 s |
| Colocated, 512-token chunks | 97.37 ms | 1.538 s |
| Colocated, 192-token chunks | 38.64 ms | 1.631 s |
| Colocated, 160-token chunks | 32.79 ms | 1.699 s |
| Colocated, 128-token chunks | 32.02 ms | 2.063 s |
| One prefill GPU plus one decode GPU, 300 GB/s | 31.60 ms | 1.468 s |

A prefill of 8,192 tokens alone takes 1,466.09 ms, so the whole-prompt row stalls the decoders for that long. Chunking cuts the stall by 45 times at 160 tokens, and costs the prompt 0.21 s. Shrinking the chunk further saves the decoders 0.8 ms and costs the prompt another 0.36 s. ==Chunking moves interference between the decoders and the prompt. It does not remove it.== Only the last row has neither cost, and it uses two GPUs where the others use one.

The mean hides all of this. The slowest decoder's mean gap in the whole-prompt row is 35.98 ms, against a worst gap of 1,467 ms. That is why the lab scores a request by its worst gap, not its mean gap, when it checks a TPOT SLO.

```viz
17b-chunk-vs-disagg
```

> [!KEY] The trade is real, and the comparison needs equal GPUs
> One GPU with a tuned chunk and two GPUs split into prefill and decode are not the same budget. The fair question is how many requests per second meet the SLO on the same number of GPUs. The later sections answer it.

## What a prefill GPU ships

This section counts the bytes that cross the link for one finished prompt.

Each full-attention layer ships its K and V for every prompt token: $2 \times 4 \times 256 \times 2 = 4{,}096$ bytes per layer per token. There are 16 such layers, so 65,536 bytes per token. Each linear layer ships its state and convolution window, 3,227,648 bytes, however many tokens the prompt has. For a prompt of $T$ tokens:

$$
\text{bytes}(T) = 65{,}536\, T + \hlc{154{,}927{,}104}
$$

A dense stack, with all 64 layers full attention, ships $262{,}144\, T$ and no state.

| Prompt tokens | Hybrid | Of which state | Dense |
|---|---|---|---|
| 128 | 155.75 MiB | 94.9% | 32.00 MiB |
| 512 | 179.75 MiB | 82.2% | 128.00 MiB |
| 2,048 | 275.75 MiB | 53.6% | 512.00 MiB |
| 8,192 | 659.75 MiB | 22.4% | 2,048.00 MiB |
| 32,768 | 2,195.75 MiB | 6.7% | 8,192.00 MiB |
| 131,072 | 8,339.75 MiB | 1.8% | 32,768.00 MiB |

The hybrid ships less than a dense stack once the prompt passes 788 tokens, where $65{,}536\,T + 154{,}927{,}104 = 262{,}144\,T$. Above 2,364 tokens the KV outweighs the state. Below 788 tokens the hybrid ships more, up to 4.9 times as much at 128 tokens, because the fixed state dominates a short prompt.

> [!EXAMPLE] An 8,192-token prompt
> The KV is $65{,}536 \times 8192 = 536{,}870{,}912$ bytes. The state adds 154,927,104. The total is 691,798,016 bytes, which is 659.75 MiB. At 300 GB/s that is 2.306 ms. The prefill that produced it takes 1,466.09 ms, so the copy is 0.16% of the time.

The state is a single contiguous matrix per layer, not a list of pages. [Chapter 15a](/c/15a-prefix-caching-and-radix-trees) shows that a cached prefix needs a snapshot of the state at the prefix boundary. A disaggregated engine that reuses a prefix has to ship that snapshot too.

## How fast the link has to be

This section turns the byte count into a link rate for a budget.

Let the budget be the extra wait you allow for the copy. A bulk copy starts after the prefill ends, so the wait is $\text{bytes}/\text{link}$ plus a setup latency $s$, and the least link rate follows:

$$
\text{link}_{\min} = \frac{\text{bytes}(T)}{\text{budget} - s}
$$

Two budgets are natural. One is a fraction of the prefill, such as 10%. The other is an absolute number of milliseconds. The first gets easier as prompts grow, because bytes grow slower than prefill time. The second gets harder.

| Prompt | Prefill | Bulk copy within 10% of the prefill | Bulk copy within 10 ms |
|---|---|---|---|
| 128 | 30.39 ms | 53.748 GB/s | 16.332 GB/s |
| 2,048 | 360.09 ms | 8.030 GB/s | 28.914 GB/s |
| 8,192 | 1,466.09 ms | 4.719 GB/s | 69.180 GB/s |
| 32,768 | 6,365.84 ms | 3.617 GB/s | 230.241 GB/s |

The same bulk copy, in time, over each link:

| Prompt | 300 GB/s NVLink | 50 GB/s IB NDR | 32 GB/s PCIe | 25 GB/s IB HDR | 12.5 GB/s Ethernet | 3.125 GB/s Ethernet |
|---|---|---|---|---|---|---|
| 128 | 0.54 ms (1.8%) | 3.27 ms (10.7%) | 5.10 ms (16.8%) | 6.53 ms (21.5%) | 13.07 ms (43.0%) | 52.26 ms (172.0%) |
| 2,048 | 0.96 ms (0.3%) | 5.78 ms (1.6%) | 9.04 ms (2.5%) | 11.57 ms (3.2%) | 23.13 ms (6.4%) | 92.53 ms (25.7%) |
| 8,192 | 2.31 ms (0.2%) | 13.84 ms (0.9%) | 21.62 ms (1.5%) | 27.67 ms (1.9%) | 55.34 ms (3.8%) | 221.38 ms (15.1%) |

The percentages are the copy's share of that prompt's prefill. Read the 128-token row. Over NVLink the copy is noise. Over 25 Gbit/s Ethernet it is longer than the prefill.

## Streaming the layers

This section shows how to hide the copy under the prefill.

A prefill runs layer by layer. Layer $i$'s state is final when layer $i$ finishes. A *layer-wise* transfer puts that layer's bytes on the link then, while the next layers compute. The prefill takes the same time. The only copy left to wait for is whatever the link has not drained when the last layer finishes.

Take the prefill as equal time per layer, $t_p/64$. Layer $i$ is ready at $r_i = (i+1)\,t_p/64$. The link is free at $f_{i-1}$ after the previous layer. Then:

$$
f_i = \max(r_i, f_{i-1}) + \frac{b_i}{\text{link}} + s, \qquad \text{wait} = \max(0,\; f_{63} - t_p)
$$

Here $b_i$ is the bytes of layer $i$: 4,096 per prompt token for a full-attention layer, and 3,227,648 for a linear one. The lab's `layerwise_exposed_seconds` is this recurrence.

> [!EXAMPLE] 8,192 tokens, 300 GB/s
> Each layer is ready 22.9 ms after the last, since $1.466/64 = 22.9$ ms. A full-attention layer ships 33,554,432 bytes, which is 0.11 ms at 300 GB/s. Every layer drains long before the next is ready. The link is idle most of the time, and the wait after the prefill is the last layer's copy, a full-attention layer: $33{,}554{,}432 / (300 \times 10^9) = 0.112$ ms. The bulk copy was 2.31 ms.

| Link | Bulk wait, 8,192 tokens | Streamed wait | Bulk wait, 128 tokens | Streamed wait |
|---|---|---|---|---|
| NVLink 300 GB/s | 2.31 ms | 0.112 ms | 0.54 ms | 0.002 ms |
| IB NDR 50 GB/s | 13.84 ms | 0.671 ms | 3.27 ms | 0.010 ms |
| IB HDR 25 GB/s | 27.67 ms | 1.342 ms | 6.53 ms | 0.021 ms |
| 100 Gbit/s Ethernet | 55.34 ms | 2.684 ms | 13.07 ms | 0.042 ms |
| PCIe 32 GB/s | 21.62 ms | 1.049 ms | 5.10 ms | 0.016 ms |
| 25 Gbit/s Ethernet | 221.38 ms | 10.737 ms | 52.26 ms | 22.350 ms |

Streaming helps most when the link can keep up with the average rate, $\text{bytes}/t_p$. For 8,192 tokens that average is 0.47 GB/s, far below any link in the table. The last row, 128 tokens on 25 Gbit/s Ethernet, is the case streaming cannot rescue: 163 MB at 3.125 GB/s takes 52 ms, and the whole prefill lasts 30 ms. The link is the bottleneck from layer 0, and the wait is the backlog.

The same logic gives the least link rate for a streamed budget. The wait falls as the rate rises, so the lab finds the rate by bisection.

| Prompt | Streamed within 5 ms | Streamed within 0.5 ms |
|---|---|---|
| 128 | 4.678 GB/s | 5.370 GB/s |
| 2,048 | 1.678 GB/s | 16.777 GB/s |
| 8,192 | 6.711 GB/s | 67.109 GB/s |
| 32,768 | 26.844 GB/s | 268.435 GB/s |

For prompts of 2,048 tokens and up, the answer is the last layer's bytes over the budget: $33{,}554{,}432/0.005 = 6.711$ GB/s for 8,192 tokens. For a short prompt the average rate, $\text{bytes}/t_p$, dominates instead.

Real engines stream in blocks, not per layer, and they overlap the copy with compute through a separate stream. Each transfer pays a setup cost. The lab's `link_latency_s` is that cost per layer, and a large value erases the benefit. DéjàVu builds a KV-cache streaming library around this idea, and Splitwise uses fast interconnects between the machines.

## A simulator for the whole system

This section describes the model that produces every load number below, and what it leaves out.

The simulator advances a clock. There are two modes.

**Colocated.** Requests go to $n$ GPUs in arrival order, round robin. Each GPU runs steps. A step takes all running decodes and fills a budget of `chunk` tokens from the waiting prompts, first come first served. The step time is `step_time`, chapter 17a's floor generalized to a mixed step. A prompt's first token is out when its last chunk's step ends.

**Disaggregated.** A prompt runs alone, whole, on the prefill GPU that frees first. Its KV then crosses that GPU's own link. TTFT includes the wait for the copy. The decode GPUs run pure decode steps. A ready request joins at the next step boundary.

Metrics follow chapter 16 and chapter 17. TTFT runs from arrival. A request's TPOT is its mean gap after the first token. Its worst gap is the longest gap after the first one. A request meets the SLO when its TTFT and its TPOT (mean, or worst gap) are within bounds. *Goodput* is the count of requests that meet it, per second of simulated span. That is DistServe's definition, taken on total GPUs.

What the model leaves out, so you know what the numbers cannot say:

- **No decode backpressure.** A prefill GPU never waits for a decode GPU. Decode memory shows up only as `max_batch`.
- **No prefix reuse and no batched prompts.** A prefill GPU runs one prompt at a time. At these lengths the prefill is compute-bound already.
- **Equal time per layer** in the streaming recurrence, and no contention between a copy and compute for HBM bandwidth.
- **The handoff gap is not scored.** The gap between token 1 and token 2 holds the wait for a step boundary on the decode GPU, up to about 2 steps. Excluding it flatters disaggregation by one gap per request.
- **Floors, not measurements.** Chapter 17a explains why. A real chunked step has overheads this model does not charge.

## Choosing the split

This section chooses how many of $G$ GPUs prefill, and compares the winner with colocation on the same GPUs.

With $G$ GPUs and $p$ of them on prefill, the system serves at the rate of its slower stage. Too few prefill GPUs and the TTFT SLO fails because prompts queue. Too few decode GPUs and the TPOT SLO fails because the decode batch grows. The best $p$ balances the two stages' work.

You can estimate that balance without running the simulator. A prefill GPU spends $\text{prefill}(T)$ seconds per request. A decode GPU runs the largest batch $b$ whose step meets the TPOT SLO, so it spends $(n_{\text{out}}-1)\,\text{step}(b)/b$ seconds per request. Split the GPUs in proportion to the two totals.

The lab's `best_split` does not estimate. It simulates every split and keeps the best goodput, and a tie goes to fewer prefill GPUs. Two planted traces show it working.

**Prefill-heavy.** 100 requests at 3 requests per second, 8,192 tokens in and 32 out, on 8 GPUs. The SLO is a TTFT of 1.6 s and a worst gap of 36 ms. The seed is 7.

| Split | Goodput | Requests that met the SLO |
|---|---|---|
| 1 prefill, 7 decode | 0.007/s | 1 of 100 |
| 4 prefill, 4 decode | 0.105/s | 4 of 100 |
| 5 prefill, 3 decode | 0.400/s | 13 of 100 |
| 6 prefill, 2 decode | 1.609/s | 52 of 100 |
| 7 prefill, 1 decode | 2.226/s | 72 of 100 |

| Colocated, 8 GPUs, chunk | Goodput | p99 worst gap |
|---|---|---|
| 128 | 0/s | 31.0 ms |
| 160 | 0/s | 31.2 ms |
| 192 | 1.357/s | 37.0 ms |
| 256 | 1.266/s | 49.1 ms |
| 512 | 1.360/s | 96.2 ms |
| 4,096 | 1.423/s | 745.2 ms |

Prefill dominates this trace. Each prompt costs 1.47 s of prefill and only 31 decode steps. The best split, 7:1, beats the best colocated goodput by 56%. ==Chunks small enough to protect the decoders starve the prompts==: at 128 and 160 tokens the TTFT SLO fails for every request, and the goodput is zero. With only 32 output tokens, most requests never overlap a stall, which is why the unchunked row scores well. The analytic split agrees with the search: 7.

**Decode-heavy.** 100 requests at 8 per second, 128 tokens in and 512 out, SLO 1.0 s and 40 ms, seed 11.

| Split | Goodput |
|---|---|
| 1 prefill, 7 decode | 3.486/s |
| 2 prefill, 6 decode | 3.470/s |
| 4 prefill, 4 decode | 3.410/s |
| 6 prefill, 2 decode | 2.584/s |
| 7 prefill, 1 decode | 1.420/s |

Every split meets all 100 requests. The goodput differs only through the drain time after the last arrival, and it falls once too few GPUs decode. Colocation, at any chunk size from 128 to 4,096, reaches 3.499/s. ==On a decode-heavy trace, disaggregation ties colocation at best.== The best split puts one GPU on prefill, and that GPU is mostly idle, because a 128-token prefill takes 30 ms and the arrivals are 125 ms apart.

> [!NOTE] These are simulated floors on one cost model
> A different model, GPU, or SLO moves the crossover. The lab gives you the tool to find it, not a rule to apply. Sarathi-Serve argues the colocated side and DistServe the split side. This simulator shows each winning on a different trace.

## When disaggregation loses

This section lists the conditions that tip the comparison, each with a number from the simulator.

- **Short prompts.** The fixed state is 147.75 MiB whether the prompt has 8 tokens or 8,000. A 128-token prompt ships 155.75 MiB for a prefill that takes 30.39 ms. Over 100 Gbit/s Ethernet, the median TTFT on a trace of such prompts is 43.45 ms, against 30.39 ms colocated. Over NVLink it is 30.93 ms.
- **Slow links.** On the same trace the median TTFT is 35.49 ms over PCIe, 36.92 ms over InfiniBand HDR, and 82.65 ms over 25 Gbit/s Ethernet. Streaming recovers most of the loss for long prompts and none of it when the link is slower than the prefill produces state.
- **Small clusters.** Splitting 2 GPUs into 1:1 leaves no room to match the stages. The best split on a trace is an integer, and on 8 GPUs the 7:1 split above leaves one decode GPU as the single point of failure and capacity.
- **Decode-heavy traces.** Above, a tie. The split adds a copy and removes the ability to move a GPU between roles with a load change.
- **A tuned chunk and a loose SLO.** If the worst-gap SLO allows a stall, a larger chunk gets most of the benefit with one engine to run.

vLLM's own documentation draws the same line: disaggregated prefill tunes TTFT and inter-token latency separately and controls tail latency, and it states that it does not improve throughput. Mooncake pairs the split with a KV-cache pool, a cache-aware scheduler, and early rejection.

## What goes wrong

- **Charging the state per token.** The state is 147.75 MiB once, not per prompt token. Multiplying the state by 1,024 tokens gives about 148 GiB. The lab checks `kv_transfer_bytes` at 0 tokens, where the answer is 154,927,104, and at three other lengths.
- **Leaving the copy out of TTFT.** The request cannot decode until its state arrives. A simulator that omits it makes every TTFT too small by $\text{bytes}/\text{link}$. The lab checks one request to the nanosecond, and the short-prompt trace against the exact copy time.
- **Answering the streaming question with the bulk bound.** The bulk bound is 69.18 GB/s for 10 ms at 8,192 tokens, and streaming meets 5 ms at 6.71 GB/s. The check bisects and requires the result to meet the budget while 99% of it does not.
- **Running a chunk as its own step.** A colocated step that runs the decodes and then the chunk charges the weight read twice and misses the sharing that makes small chunks cheap. The shared-step TTFT check fails it.
- **Reading NVLink's 600 GB/s as one way.** The datasheet figure counts both directions. A one-way copy sees 300 GB/s.
- **Scoring by the mean.** The mean gap of the slowest decoder is 36 ms while its worst gap is 1.47 s.

> [!RECAP]
> - A step with decodes and a chunk of $n$ tokens is memory-bound until $n$ passes about 164 tokens. Past that, the chunk triples the step by 512 tokens. A chunk moves interference between the decoders and the prompt.
> - The hybrid ships $65{,}536\,T + 154{,}927{,}104$ bytes. That is 659.75 MiB at 8,192 tokens, against 2,048 MiB for a dense stack, and more than a dense stack below 788 tokens.
> - A bulk copy needs $\text{bytes}/(\text{budget} - s)$. Streaming each layer as it finishes cuts the wait at 8,192 tokens to the last layer's copy, 0.112 ms at 300 GB/s.
> - On 8 GPUs, a prefill-heavy trace favors a 7:1 split by 56% in goodput. A decode-heavy trace ties colocation. A short prompt over a slow link loses.
> - `best_split` simulates every split and matches a brute-force loop. The analytic work balance agrees on the prefill-heavy trace.
> - Every number is a simulated floor, not a measurement.

## Check your understanding

> [!QUESTION] Why does a 512-token chunk slow eight decoders by 2.96 times when a 160-token chunk slows them by 3%?
> Both steps read the same weights. The 160-token chunk adds compute that fits under the byte floor, because compute alone passes that floor at about 164 tokens. The 512-token chunk makes the step compute-bound, so each added token costs 0.173 ms of arithmetic.

> [!QUESTION] Why is the state's share of the copy 94.9% at 128 tokens and 6.7% at 32,768?
> The state is 154,927,104 bytes at any length, and the KV grows by 65,536 bytes per token. At 128 tokens the KV is 8,388,608 bytes. At 32,768 tokens it is 2,147,483,648.

> [!QUESTION] Why does streaming not help 128 tokens on 25 Gbit/s Ethernet?
> The state is 163 MB and the link moves 3.125 GB/s, so the copy takes 52 ms. The prefill lasts 30 ms. The link is busy from the first layer, and the wait is the backlog left when the prefill ends: 22.35 ms.

> [!QUESTION] Why does the best split on the prefill-heavy trace sit at the edge of the search?
> A 7:1 split leaves one decode GPU, and the decode work is small: 32 output tokens per request. The search range stops at 7 prefill GPUs because each side needs one. The analytic balance gives the same answer.

> [!QUESTION] Why does the lab score the TPOT SLO by the worst gap?
> A mean over 300 tokens averages one 1.47 s stall into 5 ms. A client sees the stall. The lab excludes the first gap, which holds the handoff on a disaggregated run.

## Lab

> [!TRY] Count the bytes, then simulate
> Implement `kv_transfer_bytes`, `layerwise_exposed_seconds`, `min_link_gbs`, `simulate_colocated_gpu`, `simulate_disagg_prefill`, and `best_split`. You pass when the byte counts and exposed waits match the hand numbers, a single request on each mode matches the sum of `step_time` calls to the nanosecond, the planted trace reproduces the chunk trade, and `best_split` matches a brute-force loop and puts 7 of 8 GPUs on prefill for the prefill-heavy trace.

The starter gives you the cost model (`step_time`, `prefill_seconds`), the decode pool, the metric summary, `simulate`, `poisson_trace`, and `capacity_split`. The tests pin the cost model to chapter 17a's 29.962 ms and 360.094 ms first, so edit nothing else in it.

Checks run from the smallest piece to the whole: the byte count, the bulk link rate, the streamed wait, one colocated request, one disaggregated request and a queue, the interference trace, the prefill-heavy and decode-heavy traces, and the slow-link case. Every message shows the expected and actual value. A simulator that drops the copy from TTFT fails the single-request check. A step that runs decodes and chunk apart fails the shared-step check.

## Further reading

- Yinmin Zhong, Shengyu Liu, Junda Chen, and others, [DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670). Defines goodput and chooses the prefill and decode allocations.
- Pratyush Patel, Esha Choukse, Chaojie Zhang, and others, [Splitwise: Efficient generative LLM inference using phase splitting](https://arxiv.org/abs/2311.18677). Splits prompt computation and token generation onto separate machines and sends the state over the cluster's fast interconnects.
- Ruoyu Qin, Zheming Li, Weiran He, and others, [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079). Adds a disaggregated KV-cache pool, a cache-aware scheduler, and early rejection under overload.
- Foteini Strati, Sara McAllister, Amar Phanishayee, and others, [DéjàVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving](https://arxiv.org/abs/2403.01876). Streams the KV cache, separates prompt and token processing, and replicates state for recovery.
- Cunchen Hu, Heyang Huang, and others, [Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads](https://arxiv.org/abs/2401.11181). TetriInfer: chunked prefill, separate prefill and decode instances, and a two-level scheduler.
- Amey Agrawal, Nitin Kedia, Ashish Panwar, and others, [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310). The colocated side: chunked prefills and stall-free scheduling.
- [vLLM disaggregated prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/). Lists the connectors and states that the feature is experimental and does not improve throughput.
- [InfiniBand, data rates](https://en.wikipedia.org/wiki/InfiniBand). HDR is 200 Gbit/s and NDR is 400 Gbit/s on a 4x link.
