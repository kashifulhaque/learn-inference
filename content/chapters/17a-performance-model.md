---
title: A model of step time and goodput
slug: 17a-performance-model
part: "Part 5 — Serving"
summary: A floor on decode and prefill from the roofline, communication, and a launch term, then the batch and the parallel layout that meet a latency target.
minutes: 110
gpu: false
objectives:
  - Write decode and prefill time as the longer of compute and memory, plus communication and a launch floor.
  - Find the batch where decode would cross the ridge, and the batches that fit in memory.
  - Recover a planted bandwidth fraction and a fixed overhead from a synthetic trace.
  - Convert a token rate and a price you supply into tokens per dollar, and find the throughput an SLO allows.
  - Choose a tensor-parallel degree or a set of replicas for a fixed GPU count.
lab: 17a-performance-model
---

# A model of step time and goodput

> [!TLDR]
> - One step takes the longer of the compute floor and the memory floor, then adds communication and a launch floor. At batch 1 and context 1 that floor is 29.962 ms: 27.962 ms of memory and 2.000 ms of launches.
> - The MLP crosses the ridge near 161 tokens. The whole hybrid step does not, for any batch that fits on one to four of these GPUs. From context 340 on, the intensity cap itself sits short of the ridge.
> - A bandwidth fraction and a fixed overhead are the two constants a device would miss. The lab recovers them from a synthetic trace with planted noise. That trace is not a measurement.
> - Tokens per dollar takes an hourly price you supply. This course has no citable A100 price, so the formula keeps the price as an argument.
> - On four GPUs at context 1,024, degree 4 is the highest throughput on NVLink. On PCIe the same loose SLO picks degree 2 with two replicas. A 25 ms cap on the step scores plain replicas at zero.

[Chapter 17](/c/17-benchmarks) tells you which timestamps to record, and what each summary hides.
[Chapter 10](/c/10-roofline) tells you whether one operation waits on arithmetic or on bytes.
[Chapter 19](/c/19-tensor-parallelism) tells you what an all-reduce costs.
None of those chapters answers the capacity question: at this batch, this context, and this number of GPUs, which step time to plan for, and which layout meets the latency target.

This chapter writes that expectation as one floor. It uses peaks, byte counts, and collective costs the course already derived. It does not add a peak, a price, or a measured latency. The lab runs on the CPU. There is no weight file and no GPU.

The floor is not a promise about a device. Chapter 10's plain copy reaches 1275 GB/s of a 1935 GB/s rating. A real decode step misses the rating by its own fraction, and by a fixed overhead the launch count only approximates. You fit those two constants when you have a trace. Until then, the trace in the lab is synthetic, and it says so.

## Before you start

**The roofline, from [chapter 10](/c/10-roofline).** An operation that does $F$ FLOPs and moves $B$ bytes takes at least the longer of $F/C$ and $B/\beta$. On the A100 80GB PCIe card, $C$ is 312 TFLOP/s of bfloat16 tensor-core throughput and the rated $\beta$ is 1935 GB/s. Their ratio, the ridge point, is 161.2 FLOPs per byte. A plain copy on the card this course measured moves 1275 GB/s. Chapter 10 also prices a decode step's launches: 400 kernels at 5 µs is 2.0 ms, and at 10 µs is 4.0 ms. This chapter uses the 5 µs end, and it names that choice.

**The model, from [chapter 2](/c/02-memory-arithmetic).** Hidden size 5120, MLP intermediate 17408, 64 layers. Sixteen layers are full attention and keep 64 KiB of KV per token. Forty-eight are gated delta layers and keep one recurrent state of 154,927,104 bytes per sequence, which chapter 2 reports as 147.8 MiB. The hybrid weights are 26,895,979,520 parameters, 53,791,959,040 bytes in bfloat16. A hypothetical dense stack, every layer full attention, is the comparison chapter 2 uses for the cache. Its parameters are the same pieces counted 64 times:

$$
64 \times (104{,}857{,}600 + 267{,}386{,}880) + 2{,}542{,}796{,}800 + 660{,}480 = 26{,}367{,}104{,}000
$$

That is 52,734,208,000 bytes. The dense KV is 256 KiB per token, and it has no recurrent state.

**The linear layer's extra FLOPs, from [chapter 6](/c/06-gated-delta-rule).** Decode runs the sequential rule, $7 d^2$ FLOPs per value head with $d = 128$, which is 114,688 per head. Prefill runs the chunked form. At chunk 64 that is $7 \times 64 \times 128 + 6 \times 128^2 = 155{,}648$ FLOPs per head per token. This chapter does not re-derive the rule.

**Latency names, from [chapter 17](/c/17-benchmarks).** Time to first token (TTFT) is the wait until the first generated token. Inter-token latency is the mean gap after that. When every sequence in the step waits on the same decode, the gap equals the step time. DistServe calls that gap time per output token (TPOT). This chapter uses the step time as TPOT, and the prefill of one prompt as TTFT. It does not replay chapter 17's trace statistics.

**Communication, from [chapter 19](/c/19-tensor-parallelism).** This model has four KV heads, so the degrees worth considering are 1, 2, and 4. Each of the 64 layers ends its attention or linear sub-block and its MLP in one all-reduce, 128 collectives in a forward pass. Chapter 19 prices each at about 8 µs, and the hidden state at 10,240 bytes, so the ring moves $40{,}960 \times (P-1)/P$ bytes per token per layer per rank. NVLink between two A100s is 600 GB/s. PCIe Gen4 x16 is about 64 GB/s. The pairing that makes one all-reduce enough is chapter 19's subject. This chapter spends the cost and does not rebuild the split.

## The step, in one expression

This section writes the time of one forward pass. Compute and memory overlap. Communication and the launch do not, in this model, so they add.

Chapter 10's roofline is a lower bound: the hardware can move bytes while it does arithmetic, and it cannot finish before the slower of the two is done. Pope et al. split the same idea into a memory time and a compute time, and they analyze prefill and decode separately because the two phases do not share a batch shape. Kipply's floorline is the same max. This chapter keeps that max and then adds the two terms chapter 10 and chapter 19 already showed sitting outside it:

$$
t = \max\!\left(\frac{F}{\hla{C}}, \frac{B}{\hlb{\beta}}\right) + t_{\text{comm}} + \hlc{t_{\text{launch}}}
$$

$\hla{C}$ is the arithmetic peak and $\hlb{\beta}$ is the bandwidth you pass in. The rated pair gives the ridge of 161.2. $\hlc{t_{\text{launch}}}$ is a count of launches times a time per launch. $t_{\text{comm}}$ is zero at degree 1. Otherwise it is the 128 collective latencies plus the ring's bytes at the link rate:

$$
t_{\text{comm}} = 128 \times 8\,\mu\text{s} + \frac{64 \times T \times 40{,}960 \times (P-1)/P}{\text{link}}
$$

$T$ is the number of tokens in the forward pass: the batch, during decode, or the prompt length, during prefill. The 8 µs figure is chapter 19's "about 8 microseconds." $128 \times 8$ µs is 1.024 ms, which that chapter rounds to 1.02 ms.

> [!KEY] The number is a floor
> A kernel that misses the rated bandwidth takes longer than $B/\hlb{\beta}$. A fused graph takes less than the 2.0 ms launch term. Use the expression to compare layouts and to see which term dominates. Do not quote it as a measured step.

The launch term uses chapter 10's worked example of 400 launches at 5 µs, so $\hlc{t_{\text{launch}}} = 2.0$ ms, and it does not change with batch, context, or degree. That overstates a captured decode, which chapter 19 notes, and it is the right charge for an engine that has not captured the step. Prefill pays it too. Next to a prefill of hundreds of milliseconds, 2 ms is a small addend. Next to a decode, it is not.

## What a hybrid decode step moves

This section counts $F$ and $B$ for one decode step, then plugs in batch 1 and context 1.

**FLOPs.** Every parameter does two FLOPs per token, the count chapter 19 uses for the weight matmuls. Full attention adds chapter 10's scores and values, $4 \times 24 \times \text{ctx} \times 256$ per layer, on the 16 full-attention layers. Each linear layer adds $48 \times 7 \times 128^2$ FLOPs for the sequential rule. At batch $b$:

$$
F = b \times \big(2 \times 26{,}895{,}979{,}520 + 16 \times 4 \times 24 \times \text{ctx} \times 256 + 48 \times 48 \times 114{,}688\big)
$$

**Bytes on the bus.** The weights are read once. The KV prefix is read, chapter 10's count, not an extra write of the new token. The recurrent state is read and written, because the delta rule updates the whole matrix and 147.8 MiB does not fit in the 40 MB L2. MLP inputs and outputs, and the 129 RMSNorm reads and writes, stay at full width: chapter 19's all-reduce reassembles the hidden state on every rank, so that traffic is not divided by $P$. Weights, KV, and state are divided by $P$.

At degree 1 the byte count is:

$$
\begin{aligned}
B &= 53{,}791{,}959{,}040 \\
&\quad + 65{,}536 \times \text{ctx} \times b \\
&\quad + 2 \times 154{,}927{,}104 \times b \\
&\quad + (64 + 129) \times 2 \times b \times 5120 \times 2
\end{aligned}
$$

> [!EXAMPLE] Batch 1, context 1, one GPU
> The pieces, in bytes, are the weights 53,791,959,040, the KV 65,536, the state read and write 309,854,208, and the activations 3,952,640. The sum is 54,105,831,424 bytes.
>
> $$
> \frac{54{,}105{,}831{,}424}{1.935 \times 10^{12}} = 27.962\ \text{ms}
> $$
>
> The FLOPs are 54,056,593,408, and $54{,}056{,}593{,}408 / (312 \times 10^{12})$ is 0.173 ms. The launch adds 2.000 ms. Communication is zero. The step is ==the memory term plus the launch==:
>
> $$
> \boxed{t = 27.962\ \text{ms} + 2.000\ \text{ms} = 29.962\ \text{ms}}
> $$

The 27.962 ms splits further. The weights alone are $53{,}791{,}959{,}040 / 1.935 \times 10^{12} = 27.799$ ms, which is chapter 10's 27.8 ms with the exact byte count instead of 53.8 GB. The state adds 0.160 ms. Activations add 0.002 ms. One token of KV adds 0.034 µs. Chapter 19's 42.2 ms is a different quotient: the same weights at the measured 1275 GB/s, with no state, no KV, and no launch. The exact quotient is 42.190 ms. It answers "how long is the weight read on the copy we measured," not "how long is the step in this model."

The same arithmetic on the dense stack, at batch 1 and context 1, is 29.255 ms. The hybrid step is slower. The hybrid weights take 27.799 ms to read and the dense weights take 27.253 ms. Only the hybrid pays the 0.160 ms of state traffic. At context 1 the KV term is noise, so the dense stack wins this one comparison. It is not the model you serve.

**Prefill** uses the same sum, with three substitutions. $T$ tokens of one sequence replace the batch. Causal attention extends chapter 10's one-query count over prefixes $1 \ldots T$, so the attention FLOPs carry $T(T+1)/2$ instead of a single context. Linear layers use the chunked 155,648 FLOPs per head per token. The KV cache is written once, $65{,}536 \times T$ bytes, and the state is read and written once per chunk of 64 rather than once per token. A tiled attention kernel does not reread every earlier token from HBM for every query, so the byte count does not grow with $T^2$.

> [!EXAMPLE] A 2048-token prefill on one GPU
> Compute takes 358.094 ms. Memory takes 37.176 ms. The launch adds 2.000 ms. The step is compute-bound, at 360.094 ms.
>
> The weight matmuls inside that compute are $2 \times 26{,}895{,}979{,}520 \times 2048 / (312 \times 10^{12}) = 353.3$ ms. That is the same product chapter 19 writes as 353 ms for a 4096-token prefill split across two GPUs. The attention triangle and the chunked recurrence add the other 4.8 ms.

At degree 2, chapter 19's prefill example moves $64 \times 4096 \times 20{,}480 = 5.369 \times 10^{9}$ bytes, 5.37 GB, which is 8.95 ms on NVLink. This model adds the 1.024 ms of collective latency on top, 9.97 ms. Against a prefill of a third of a second, the latency term is small. Against a decode step it is most of the communication.

## The batch where decode would cross the ridge

This section asks when $F/\hla{C}$ exceeds $B/\hlb{\beta}$, and whether that batch fits.

Chapter 10's MLP, taken alone, has intensity $T / (1 + T/26112)$. At 161 tokens that intensity is 160, against a ridge of 161.2. The whole step is a different ratio. Its FLOPs grow with the batch, and so do the state, the KV, and the activations. The weights do not. Divide the per-sequence FLOPs by the per-sequence bytes and you get the intensity the step approaches as the batch grows. At context 1 that cap is 172.2 FLOPs per byte, a little over the ridge, so a crossing exists. At context 339 the cap is 161.27, still over. At context 340 it is 161.24, under. ==At context 340 and beyond, no batch is compute-bound==, because the KV and the state keep the asymptote short of the ridge.

> [!DEEPDIVE] Where the two floors meet
> Write $F = f_1 b$ and $B = W + p b$, with $W$ the weight bytes. Set the floors equal:
>
> $$
> \frac{f_1 b}{\hla{C}} = \frac{W + p b}{\hlb{\beta}}
> $$
>
> Solve for the batch. The crossing is real only when the denominator is positive, which is the same condition as $f_1 / p > \hla{C}/\hlb{\beta}$:
>
> $$
> b^{\ast} = \frac{\hla{C}\, W}{f_1 \hlb{\beta} - \hla{C}\, p}
> $$
>
> At context 1 and degree 1 the script evaluates this at 2,515.71. The floors meet at batch 2,516. One GPU holds 207 sequences at that context. The 80 GiB card minus the weights leaves $(85{,}899{,}345{,}920 - 53{,}791{,}959{,}040)$ bytes, and each sequence needs $65{,}536 + 154{,}927{,}104$ bytes, which floors to 207. Degree 4 holds 1,869, and the crossing has moved out to batch 5,648, because the replicated activations do not shrink when the weights do. Degree 2 sits between them: the crossing is batch 3,086, and 761 sequences fit.

The card runs out of memory before the crossing, at degree 1, 2, and 4, at context 1. Longer context makes the crossing later, or removes it, and it lowers the number of sequences that fit. So every decode step you can actually run on this model is memory-bound. The 161-token MLP result is still why batching helps the MLP. It is not the batch at which the step changes regime.

The memory cap is the useful number. Resident bytes are the weights, the KV, and one copy of the state, divided by the degree. The second trip of the state is bus traffic, not a second allocation. Activations are transient and are not reserved, which is chapter 2's budget.

| Context | Intensity cap | Crossing batch | Sequences that fit, degree 1 |
|---|---|---|---|
| 1 | 172.2 | 2,516 | 207 |
| 128 | 167.9 | 4,024 | 196 |
| 1,024 | 143.0 | none | 144 |
| 4,096 | 95.6 | none | 75 |
| 32,768 | 27.2 | none | 13 |

At 32,768 tokens the hybrid step at batch 1 is 31.071 ms and the dense step is 33.694 ms. The dense KV read is 4.439 ms against the hybrid's 1.110 ms, which is chapter 10's 1.11 ms for 2 GiB at 1935 GB/s. The dense read is four times that. The hybrid still pays 0.160 ms of state and a heavier weight read, 27.799 ms against 27.253 ms. The net is 2.6 ms in the hybrid's favor on the step, and a much larger gap on capacity: 13 sequences against 3. Chapter 2's break-even of 788 tokens is about bytes stored. The step-time comparison counts the state twice, on the bus, and it counts the weight difference. Those are different questions, and at batch 1 they do not cross until context 6,956, where both steps are 30.197 ms. The capacity gap shows up earlier and matters more.

## Throughput and the step

This section turns the step time into tokens per second, and shows why the curve climbs and then stops.

Decode throughput at batch $b$ is $b / t(b)$ output tokens per second. The step has the shape $\text{fixed} + \text{slope} \times b$ while it stays memory-bound: the weights and the launch are the fixed part, and the per-sequence traffic is the slope. Then

$$
\frac{b}{t(b)} = \frac{b}{\text{fixed} + \text{slope} \times b}
$$

which rises with $b$ and levels off at $1/\text{slope}$. It does not have a hump. The largest batch that fits is the fastest memory-bound operating point, until an SLO forbids it.

At context 1,024 on one GPU the curve is the following. Every row is memory-bound. Batch 144 is the last that fits.

| Batch | Step | Tokens per second |
|---|---|---|
| 1 | 30.00 ms | 33.3 |
| 8 | 31.37 ms | 255 |
| 32 | 36.10 ms | 886 |
| 64 | 42.40 ms | 1,510 |
| 144 | 58.15 ms | 2,476 |

```viz
17a-throughput-latency
```

The same shape at the memory cap, still one GPU:

| Context | Sequences that fit | Step at that batch | Tokens per second |
|---|---|---|---|
| 1,024 | 144 | 58.15 ms | 2,476 |
| 4,096 | 75 | 52.37 ms | 1,432 |
| 32,768 | 13 | 46.34 ms | 281 |

A 32k context does not make the batch-1 step much slower, because the weights still dominate one sequence. It cuts the batch you can hold, and the throughput falls with it. That is the capacity plan in one table.

## Two constants, from a synthetic trace

This section fits the two numbers a device adds on top of the rated floor. The rows are generated. They are not timings from an A100.

Write the rated floor as if the card hit $\hlb{\beta}$ and the launch were exactly 2.0 ms. A device does neither. While the step is memory-bound, a useful correction is two constants:

$$
t = \frac{B}{\alpha \, \hlb{\beta}} + \tau
$$

$\alpha$ is the fraction of rated bandwidth the step achieves. $\tau$ is a fixed overhead in seconds, absorbing the launch and anything else that does not scale with the bytes. In $x = B / \hlb{\beta}$, the equation is a line, $t = x / \alpha + \tau$. Ordinary least squares recovers the slope $1/\alpha$ and the intercept $\tau$. The fit uses bytes. A column of FLOPs is not an input to it. Using that column is the arithmetic-peak mistake from the decode step, moved into the regression.

This course has no bundled table of measured A100 step times, so the lab does not pretend to fit one. It builds five rows from the memory term at a planted $\alpha = 0.7$ and a planted $\tau = 0.002$ s, adds Gaussian noise of standard deviation 5 µs with seed 17, and asks you to recover the two constants. A trace from a device is what would replace those rows. The planted 0.7 is not the copy's $1275/1935 = 0.659$. That ratio is a measured copy. This one is a number the generator hid in the rows.

The five rows, at context 128, are the following. The script prints the full values.

| Batch | Bytes | Seconds |
|---|---|---|
| 1 | 54,114,154,496 | 0.041953 |
| 2 | 54,436,349,952 | 0.042189 |
| 4 | 55,080,740,864 | 0.042664 |
| 8 | 56,369,522,688 | 0.043620 |
| 16 | 58,947,086,336 | 0.045509 |

Least squares on these rows returns $\alpha = 0.7020$ and $\tau = 0.002114$ s. The acceptance window is six standard errors of that estimator at the planted noise: 0.00724 on $\alpha$ and 0.000426 s on $\tau$. Both errors sit inside the window. A fit that flips the sign of $\tau$ lands near $-0.002$ and does not.

> [!WARNING] Do not call this table a measurement
> The seconds were computed from the byte count, the planted constants, and seeded noise. Quoting them as A100 step times would invent a result this course did not run.

The line is the right model only while every row is memory-bound. A compute-bound row follows $\max(F/\hla{C}, B/(\alpha \hlb{\beta})) + \tau$, and a straight line through it recovers the wrong slope. The five rows are memory-bound: the weights alone are about 28 ms at the rated bandwidth, and 40 ms at $\alpha = 0.7$, against a compute term under 3 ms at batch 16.

## Tokens per dollar

This section converts a token rate into a cost. The hourly price is an argument, because this course does not have a citable A100 price.

Let $r$ be output tokens per second, $p$ the price in dollars per GPU-hour, and $G$ the number of GPUs you pay for. An hour is 3,600 seconds, so

$$
\boxed{\frac{\text{tokens}}{\text{dollar}} = \frac{r \times 3600}{p \times G}}
$$

At the one-GPU cap in the previous table, $r = 2476.5$. One GPU then returns $2476.5 \times 3600 / p$ tokens per dollar. The lab checks the arithmetic on a price you pass in. It does not check a market rate.

The Stanford CS336 syllabus lists public hourly rates for a B200, a different GPU from this course's A100. Those rates are not a stand-in for $p$. Vidur reports a dollar cost for a search over configurations of other models. That cost is not an A100-hour price for this one. Leave $p$ as a number you looked up for the card you rent, and keep it out of the floor.

## The throughput an SLO allows

This section searches for the fastest decode that meets both latency targets. Relaxing either target cannot make that maximum smaller.

A workload here is three integers: a prompt length, a context length for the decode step, and a maximum batch to search. TTFT is the prefill of that one prompt. TPOT is the decode step at that context. A batch counts only when all three of the following hold:

1. The prefill is at most the TTFT SLO.
2. The decode step is at most the TPOT SLO.
3. The weights, the KV, and the state fit in `memory_bytes`.

Goodput, in this chapter, is the largest $b / t(b)$ among the batches that pass. Chapter 17's goodput is a fraction of a recorded trace. DistServe's is the highest request rate that keeps both SLOs on a serving run. Both of those need arrivals. This search does not have arrivals. It answers a narrower question: if you could hold a steady batch, what is the fastest batch that still meets the two limits. When the prefill misses the TTFT SLO, every batch misses, and the result is 0.

A 128-token prompt takes 30.4 ms to prefill on one GPU. A TTFT SLO of 30 ms rejects it. A SLO of 40 ms accepts it. Raising the TTFT SLO from 30 ms to 40 ms turns a zero into the TPOT-limited rate, and raising it further does not lower that rate.

The TPOT side moves in steps, at context 1,024 on one GPU, with the search allowed to run out to the memory cap:

| TPOT SLO | Tokens per second | Why |
|---|---|---|
| 20 ms | 0 | Batch 1 already takes 30.00 ms |
| 30 ms | 33.3 | Only batch 1 qualifies |
| 50 ms | 2,045 | The largest qualifying batch, still under the cap of 144 |
| 1 s | 2,476 | The memory cap, batch 144 |

> [!KEY] Each SLO only adds batches
> A batch that met a tighter SLO still meets a looser one, and a batch that did not fit still does not fit. The maximum over a larger set cannot fall. The lab checks that shape against the same search written out as a loop.

The 50 ms row is not batch 144. Batch 144 takes 58.15 ms, so a 50 ms SLO stops earlier, and the throughput is lower than the memory cap. That is the sense in which a latency target costs you tokens. The search has to try the batches. Taking batch 1 whenever it meets the SLO, or taking the memory cap without looking at the step, both fail the check.

## A degree, or replicas

This section uses the floor to spend a fixed number of GPUs. Chapter 19 is where the all-reduce comes from. The choice here is which degree to run.

You have $G$ GPUs. A layout is a degree $P$ in $\{1, 2, 4\}$ that divides $G$, and $G/P$ independent replicas. A replica is a full copy of that degree: it serves its own batch, and the aggregate rate is the replica count times one replica's goodput. Degree 1 with four replicas is four separate models. Degree 4 with one replica is one model sharded across the four GPUs.

Sharding divides the weight read, the KV, and the state by $P$, and it adds $t_{\text{comm}}$. The launch stays 2.0 ms. Because the step is memory-bound in every batch that fits, cutting the bytes helps until the collective latency and the link are the larger term. The memory cap also moves. Each rank holds $1/P$ of the weights, so the room for KV grows, and the largest legal batch is larger than on one GPU.

The following table is four GPUs, a 128-token prompt, context 1,024, and a search through batch 4,096, which is past every memory cap in the row. The TTFT SLO is 1 s, so the 30.4 ms prefill is not the constraint. Rates are aggregate output tokens per second.

| TPOT SLO | 4 replicas | Degree 2, two replicas | Degree 4 |
|---|---|---|---|
| 25 ms | 0 | 6,332 | 11,123 |
| 40 ms | 5,121 | 11,352 | 13,890 |
| 80 ms | 9,906 | 14,981 | 16,203 |
| 1 s | 9,906 | 14,981 | 16,217 |

On NVLink, degree 4 wins every row. The 25 ms row is the sharp one: a single GPU's step is 30.0 ms, so a replica cannot meet 25 ms at any batch, and the sharded step can. On PCIe at 64 GB/s, with the same 1 s SLO, the order changes. Four replicas still do 9,906. Degree 2 does 13,175. Degree 4 does 11,222. ==The loose SLO on PCIe picks degree 2 with two replicas.== The extra ring bytes at degree 4, on the 64 GB/s link, cost more than the second split of the weights returns.

The same comparison at context 4,096 and at context 32,768, still on NVLink with a 1 s SLO, also picks degree 4: 10,569 tokens/s against 9,323 and 5,729 at context 4,096, and 2,484 against 2,060 and 1,122 at context 32,768. That is this workload, not a theorem about every model. The script prints the three contexts. A different prompt length changes the TTFT term and can zero out a layout the table accepts.

A search that stops at batch 64 never sees the batches where degree 4's extra room pays off. On that shorter search the loose SLO picks four replicas, and a TPOT SLO near 21 ms still picks degree 4, because replicas cannot meet it. The lab uses the shorter search so the brute-force check stays small. Set `max_batch` at least as high as the number of sequences that fit before you treat the answer as a capacity plan.

## What goes wrong

- **Pricing a memory-bound step at the arithmetic peak.** $F/\hla{C}$ at batch 1 is 0.173 ms. The step is 29.962 ms. Returning the compute term, or forgetting the $\max$ and adding the two floors, fails the hand-calculated case by milliseconds. The tolerance is 1 ns, which is under chapter 10's 10.6 ns RMSNorm floor at batch 1 and under the 5 µs launch.
- **Fitting $\tau$ with the wrong sign, or regressing on FLOPs.** The planted overhead is $+0.002$ s. A line that subtracts the overhead, or that uses $F/\hla{C}$ as $x$, does not land inside the six-standard-error window.
- **Calling the synthetic seconds a measurement.** The generator wrote them. A device trace is the replacement, and this chapter does not have one.
- **Stopping the search at a batch the card could still hold.** The loose-SLO winner at batch 64 is not the winner at the memory cap. Report the cap you searched.
- **Quoting chapter 19's 42.2 ms as this step.** That figure is the weight read at 1275 GB/s. This model's batch-1 floor at the rated bandwidth, with the state and the launch, is 29.962 ms. They answer different questions.

> [!RECAP]
> - $t = \max(F/\hla{C}, B/\hlb{\beta}) + t_{\text{comm}} + \hlc{t_{\text{launch}}}$, with $\hlc{t_{\text{launch}}} = 2.0$ ms and $t_{\text{comm}} = 0$ at degree 1.
> - Batch 1, context 1, hybrid, one GPU: 0.173 ms of compute, 27.962 ms of memory, 29.962 ms in all.
> - The MLP ridge near 161 tokens is not the step's ridge. Past context 340, hybrid decode stays memory-bound at every batch. Short of that, the crossing batch does not fit.
> - The lab's $\alpha$ and $\tau$ are planted in a synthetic trace. The copy's $1275/1935$ is a different number.
> - Tokens per dollar divides a rate by a price you supply. Goodput in this chapter is the best steady batch that meets both SLOs and fits.
> - On four GPUs at context 1,024, NVLink prefers degree 4. PCIe, at a loose SLO, prefers degree 2 with two replicas.

## Check your understanding

> [!QUESTION] Why is the batch-1 step 29.962 ms rather than the 0.173 ms of arithmetic?
> The memory floor is 27.962 ms and the compute floor is 0.173 ms, so the max is the memory term. The launch adds 2.000 ms and there is no all-reduce at degree 1. Returning $F/\hla{C}$ prices a memory-bound step at the arithmetic peak.

> [!QUESTION] Why can a looser TPOT SLO not lower the maximum goodput?
> Every batch that met the tighter SLO still meets the looser one, and the memory cap does not move. The maximum of a larger set of rates is at least as large. The same nesting holds for the TTFT SLO, which either rejects the prompt for every batch or accepts it for all of them.

> [!QUESTION] Why is the recovered bandwidth fraction not 1275/1935?
> $1275/1935 = 0.659$ is the fraction a measured copy reached, from chapter 10. The lab plants $\alpha = 0.7$ in a synthetic trace and adds noise. Recovering 0.70 means the fit found the planted constant. It does not mean a decode step was timed.

> [!QUESTION] Why does a 25 ms TPOT SLO score four replicas at zero?
> Each replica runs the unsharded step. At context 1,024 that step is 30.0 ms at batch 1 and longer after that. No batch meets 25 ms, so the aggregate is zero. Degree 4's weight read is a quarter of that, and its step does meet 25 ms.

## Lab

> [!TRY] Implement the floor and the two searches
> Write `decode_step_time`, `prefill_time`, `tokens_per_dollar`, `max_goodput`, `fit_efficiency`, and `choose_layout`. You pass when the batch-1 hybrid step matches the 29.962 ms hand total within 1 ns, the goodput search matches a brute-force loop, and the fit recovers the planted 0.7 and 0.002 s within six standard errors of the seed-17 noise.

The hardware dict carries the peaks and the degree: `bf16_tflops`, `hbm_gbs`, `launches`, `launch_us`, `allreduce_us`, `link_gbs`, `memory_bytes`, and `tp`. Missing `tp` means 1. `cfg["kind"]` is `"hybrid"` or `"dense"`.

`max_goodput` searches batches `1` through `workload["max_batch"]`. The workload also carries `prompt` and `ctx`. A missed TTFT SLO returns 0. `choose_layout` tries degrees 1, 2, and 4 that divide the GPU count, and on an exact tie it prefers more replicas.

`fit_efficiency` sees rows of `bytes`, `flops`, and `seconds`. Fit the line in bytes. The noise uses seed 17. The seconds are the synthetic trace from this chapter, not a device log.

The first check calls only `decode_step_time`. A step priced at the arithmetic peak fails it.

## Further reading

- Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, and others, [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102). Separates a memory time from a compute time, and analyzes prefill and decode as different batch shapes. The partitioning discussion is the source chapter 19 already uses.
- Kipply, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/). A floorline that takes the longer of compute and memory, and a batch size where a forward pass crosses from memory-bound to compute-bound. The bandwidth in that post is not this course's 1935 GB/s.
- Zhihang Yuan, Yuzhang Shang, Yang Zhou, and others, [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/abs/2402.16363). Uses a roofline to separate memory-bound decode from compute-bound prefill, and names prefill and decode as the two stages.
- Amey Agrawal, Nitin Kedia, Jayashree Mohan, and others, [Vidur: A Large-Scale Simulation Framework for LLM Inference](https://arxiv.org/abs/2405.05465). Predicts latency from profiled operators and searches configurations for a cost. That is a simulator fitted to a device. This chapter's floor is not that fit, and the lab's trace is synthetic.
- Yinmin Zhong, Shengyu Liu, Junda Chen, and others, [DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670). Defines goodput as the highest rate that meets a TTFT SLO and a TPOT SLO together.
- Stanford CS336, Language Modeling from Scratch, [syllabus](https://cs336.stanford.edu/). Lecture 2 is "PyTorch (einops), resource accounting (FLOPs, memory, arithmetic intensity)." Lecture 10 is "Inference." The syllabus also lists public B200 rates, which are not an A100 price.
- MIT 6.5940, TinyML and Efficient AI Computing, [Fall 2026 schedule](https://hanlab.mit.edu/courses/2026-fall-65940). "Chapter I: Efficient Inference" (16 September), "Transformer and LLM" (22 October), and "Guest Lecture: Efficient Inference in Production" (24 November).
