---
title: Sparsity and pruning
slug: 18b-sparsity-and-pruning
part: "Part 6 — Scaling"
summary: Score weights by magnitude and by Wanda, build a 2:4 mask, count the compressed bytes, and see why Ampere's sparse tensor cores win bytes more than FLOPs.
minutes: 90
gpu: false
objectives:
  - Prune a layer by magnitude and by Wanda at the same sparsity, and show Wanda keeps the layer output closer.
  - Build a 2:4 mask, check that every group of four keeps exactly two, and explain why a per-row 50% mask fails the check.
  - "Store a 2:4 matrix as values plus 2-bit indices, and count the metadata bytes."
  - Compare bytes per weight for dense bfloat16, 2:4 bfloat16, and group-wise int4, and say what that bound means for decode.
  - "Write the distillation loss, check its gradient against finite differences, and say why a draft for chapter 20 is distilled."
lab: 18b-sparsity
---

# Sparsity and pruning

> [!TLDR]
> - Pruning sets weights to zero. Magnitude pruning ranks weights by $|W|$ and ignores the inputs. Wanda ranks by $|W|$ times the L2 norm of the input channel. On the lab's layer, with four planted outlier channels, Wanda's 50% output error is $0.091117$ against $0.248337$.
> - A 2:4 mask keeps two of every four consecutive input weights. The compressed matrix stores the two values and a 2-bit position for each. For bfloat16 that is $1.125$ bytes per dense weight, against $2$ dense and $0.515625$ for group-128 int4.
> - Ampere's sparse tensor cores run a 2:4 matrix at twice the dense math rate. Decode is memory bound, so the win is bytes. If every byte of the 53.8 GB model were a prunable weight, the weight read would fall from $42.196$ ms to $23.735$ ms, a factor of $1.778$. That is an upper bound computed here, not a measurement.
> - SparseGPT is GPTQ's update applied to pruning. A distilled draft raises the acceptance rate in [chapter 20](/c/20-speculative-decoding). The loss is $T^2$ times the KL divergence from the teacher.

[Chapter 18](/c/18-quantization) and [chapter 18a](/c/18a-int4-and-kv-quantization) shrink each weight. This chapter removes weights. Both attack the same number: the bytes that decode reads from memory on every token.

The two ideas stack, and they fail in different ways. Quantization keeps every weight and rounds it. Pruning keeps the full precision of the weights it keeps and drops the others. The question in both cases is which error to accept, and the answer in both cases depends on the inputs the layer sees.

You build the scorers, the 2:4 mask, and the compressed format in the lab, all on the CPU. The GPU check is written and never run, and the chapter says what it would measure and what it would not prove.

## Before you start

**The layer output, from [chapter 18a](/c/18a-int4-and-kv-quantization).** For a weight $W$ of shape $(\text{out}, \text{in})$ and inputs $X$ of shape $(\text{tokens}, \text{in})$, the layer computes $XW^{\mathsf T}$. The error that matters after compression is the relative change in that product:

$$
\frac{\lVert XW^{\mathsf T} - X\hat{W}^{\mathsf T} \rVert_F}{\lVert XW^{\mathsf T} \rVert_F}.
$$

**The decode byte floor, from [chapter 10](/c/10-roofline) and [chapter 11a](/c/11a-constrained-decoding).** One decode step reads the 53.8 GB of bfloat16 weights. At the measured 1275 GB/s copy rate that is $53.8 / 1275 = 42.196$ ms. Anything that shrinks the bytes shrinks that floor in proportion, and nothing else about the step has to change.

**Activation outliers, from [chapter 18](/c/18-quantization).** A few input channels carry activations far larger than the rest. A weight that multiplies such a channel moves the output more than its size suggests. Quantization scales those channels. Pruning has to avoid deleting them.

## Why remove weights

Han and coauthors prune, quantize, and Huffman-code a trained network, and report a 35x to 49x storage reduction with no accuracy loss. Pruning alone cuts connections by 9x to 13x in their paper. Those are their networks, not language models.

For serving a language model, two costs matter. The stored bytes decide whether the model fits, and the bytes read per token decide the decode time. A zero that you still store and still read saves nothing. Sparsity pays only in a format that skips the zeros, and the format decides what you can win.

The most common format is unstructured: any weight can be zero. Store each kept weight with a column index and you pay for the index. At 50% sparsity, a bfloat16 value with a 16-bit column index costs $2 + 2 = 4$ bytes per kept weight, which is $2$ bytes per dense weight. That is exactly the dense size, and the gather it needs is irregular. ==Unstructured 50% sparsity saves no bytes in a naive index format.== The fix is a structure the hardware can decode with a fixed-size index, and Ampere has one.

## Magnitude pruning ranks by the weight alone

Magnitude pruning keeps the weights with the largest $|W_{oj}|$ and zeros the rest.

It is the baseline because it needs no data. It also has a clear failure. The output of a layer is a sum over input channels, $y_o = \sum_j W_{oj} x_j$, so a weight's contribution scales with the activation $x_j$ it meets. A small weight on a channel with huge activations can matter more than a large weight on a quiet channel. Magnitude pruning cannot see that.

Compare weights inside one output row, not across the whole matrix. Each row of $W$ produces one output, and the error in that output is the sum of what you removed from that row. A row-wise rule also gives every row the same number of survivors, so $\text{in} \cdot (1 - s)$ weights stay in each row at sparsity $s$. The lab's `prune_rowwise` does this, and a tie drops the lower column index first.

## Wanda scores the weight and its activation

Wanda multiplies each weight's magnitude by the L2 norm of its input channel, and then prunes by that score inside each output row.

Sun and coauthors call the method pruning by weights and activations. It needs no retraining and no weight update. The score is

$$
\boxed{S_{oj} = \hla{|W_{oj}|} \cdot \hlb{\lVert X_{:,j} \rVert_2}}
$$

where $\hlb{\lVert X_{:,j} \rVert_2}$ is the norm of column $j$ of the calibration inputs, taken over the tokens. Every output row shares that norm, so it is one number per input channel.

The score estimates the size of the term $W_{oj} x_j$ over the calibration tokens. It is the quantity a row's output error is made of, so dropping the smallest scores drops the smallest pieces of the output.

> [!EXAMPLE] Four weights, one heavy channel
> Take the row $W = (0.9, -0.1, 0.4, -0.7)$ and column norms $(1, 1, 3, 1)$, so channel 2 carries three times the activation. Magnitude scores are $(0.9, 0.1, 0.4, 0.7)$, and the top two are channels 0 and 3. Wanda scores are $(0.9, 0.1, 1.2, 0.7)$, and the top two are channels 0 and 2.
>
> For one token with $x = (1, 1, 3, 1)$ the full output is $0.9 - 0.1 + 1.2 - 0.7 = 1.3$. Keeping channels 0 and 3 gives $0.2$, a relative error of $0.846$. Keeping channels 0 and 2 gives $2.1$, a relative error of $0.615$. Wanda keeps the channel the input makes large.

Where does the product come from? Suppose you ignore how input channels correlate, so the Hessian $H = X^{\mathsf T}X$ is its diagonal. Pruning weight $W_{oj}$ then costs $\tfrac{1}{2} W_{oj}^2 H_{jj}$ in that loss, and $H_{jj} = \lVert X_{:,j} \rVert_2^2$. Up to the constant, the square root of that cost is $|W_{oj}| \cdot \lVert X_{:,j} \rVert_2$, which is the Wanda score. The next section says what you give up by ignoring the off-diagonal terms.

The lab plants the outlier the course keeps meeting. The layer has 32 outputs, 64 inputs, and 256 tokens drawn with seed 18, and channels 3, 17, 40, and 58 are scaled by 12. Those channels have a mean column norm of $191.16$, against $15.89$ for the other sixty. Each output row drops half its weights:

| Sparsity | Magnitude error | Wanda error |
|---|---|---|
| 0.25 | 0.115543 | 0.030871 |
| 0.50 | 0.248337 | 0.091117 |
| 0.75 | 0.549705 | 0.178512 |

The script's full sweep is in the figure, from 10% to 90%. Wanda is lower at every point.

```viz
18b-wanda-error
```

At 50%, the layer has 128 weights in the four outlier channels: 32 rows times 4. Magnitude pruning keeps $65$ of them, about half, because it ranks them with everything else. Wanda keeps $119$. The error drops by the ratio $0.091117 / 0.248337 = 0.367$, and the lab asks for under $0.5$.

==Wanda wins here because the activations are skewed, not because it is better in general.== With equal column norms the Wanda score is a constant multiple of the magnitude and the two methods pick the same weights. The planted layer exaggerates the effect. Sun and coauthors report that Wanda outperforms magnitude pruning significantly on LLaMA and LLaMA-2, and is competitive with methods that update weights. That is their models and their evaluations, not this layer.

## SparseGPT updates the weights that stay

SparseGPT is GPTQ's update applied to pruning: remove a weight, then correct the weights that remain so the layer output stays put.

Frantar and Alistarh start from Optimal Brain Surgeon. When you zero weight $q$ of a row, the loss $\tfrac{1}{2}\lVert XW^{\mathsf T} - X\hat{W}^{\mathsf T}\rVert_F^2$ for that row rises by $w_q^2 / (2 [H^{-1}]_{qq})$ at best, and you reach that best by adjusting the other weights in the row by

$$
\delta w = -\frac{w_q}{[H^{-1}]_{qq}}\, H^{-1}_{:,q}.
$$

Here $H = X^{\mathsf T}X$ is the same Gram matrix as in [chapter 18a](/c/18a-int4-and-kv-quantization), and the update is the one GPTQ applies to the residual of a rounded weight. GPTQ rounds and pushes the rounding residual into the later columns. SparseGPT zeros and pushes the removed weight into the later columns. Both use the Cholesky factor of the damped inverse so they can sweep column by column on a layer with thousands of columns.

The abstract of the SparseGPT paper reports that GPT-family models reach at least 50% sparsity in one shot with no retraining, and that it prunes OPT-175B and BLOOM-176B in under 4.5 hours. It also handles the 2:4 and 4:8 patterns. The paper reports minimal accuracy loss at 50% and a negligible perplexity increase at 60% unstructured sparsity.

The lab does not implement SparseGPT. The update is chapter 18a's, and the point of Wanda is that it skips it. The trade is the one from the last section: Wanda treats the channels as independent and costs one pass of norms, and SparseGPT uses the correlations between channels and costs a Cholesky factor per layer.

## A 2:4 mask keeps two of every four

A 2:4 mask keeps exactly two of every four consecutive input weights, so every group of four has the same size after compression.

Cut each row into groups of four columns. In each group, keep the two weights with the highest scores and zero the other two. The density is exactly 50%, and so is the density of every group. That regularity is what the hardware needs.

The score can be the magnitude or the Wanda score. On the seeded layer, the 2:4 errors are:

| Score | 2:4 error | Row-wise 50% error |
|---|---|---|
| Magnitude | 0.332604 | 0.248337 |
| Wanda | 0.123251 | 0.091117 |

2:4 costs error, because it forbids the best 50% mask: a group that holds three important weights must drop one. The cost here is $0.332604 / 0.248337 = 1.339$ times the magnitude error. Wanda's 2:4 error, $0.123251$, is still under half of magnitude's unstructured error.

**Check the structure, not the density.** A mask with 50% density is not a 2:4 mask. The row-wise Wanda mask above has 50% density and is not valid, because some groups keep three weights and others keep one. The lab's `is_valid_2_4` counts the kept weights in each group of four and requires exactly two. It also rejects a width that is not a multiple of four.

> [!WARNING] A mask with the right density can still be wrong
> Top-half per row, top-half per matrix, and the first half of each row all give 50% density. Only the per-group count makes the hardware's fixed-size index valid. The lab checks that the per-row mask fails `is_valid_2_4` on the seeded layer.

## The compressed format stores values and 2-bit indices

A compressed 2:4 matrix stores the two kept values of each group and the position of each inside the group.

For a matrix of shape $(\text{out}, \text{in})$ you store $\text{out} \cdot \text{in} / 2$ values. Each is one of four positions, so it needs $\log_2 4 = 2$ bits. The `compress_2_4` function returns the values and the positions as two arrays of shape $(\text{out}, \text{in}/2)$, with the two positions of a group in ascending column order. A kept weight that is exactly zero still takes a slot, because the mask decides what is kept.

> [!EXAMPLE] One group of eight
> The row $(0, 5, 0, -2,\ 7, 0, 0, 1)$ has the mask $(0,1,0,1,\ 1,0,0,1)$. Its values are $(5, -2, 7, 1)$ and its positions are $(1, 3, 0, 3)$. Eight bfloat16 weights take $16$ bytes dense. The compressed row takes $4 \cdot 2 = 8$ bytes of values and $4 \cdot 2 = 8$ bits, which is $1$ byte, of positions.

For a $4096 \times 4096$ bfloat16 matrix the counts are:

| Part | Bytes |
|---|---|
| Dense bfloat16 | 33,554,432 |
| Kept values | 16,777,216 |
| 2-bit indices | 2,097,152 |
| Compressed total | 18,874,368 |

The metadata is $1/8$ of the value bytes and $1/9$ of the compressed matrix. Per dense weight, the compressed matrix costs

$$
\boxed{\frac{1}{2}\cdot 2 + \frac{1}{2}\cdot\frac{2}{8} = 1.125\ \text{bytes}}
$$

against $2$ for dense bfloat16. A hardware library reorders the metadata into its own tiled layout. The lab keeps the logical layout, and the byte count is the same.

## What Ampere gives you, and what it does not

Ampere is the first NVIDIA generation with sparse tensor cores, and they run a 2:4 matrix at twice the dense math rate.

The NVIDIA paper on accelerating sparse networks states that the sparse tensor cores deliver twice the math throughput of dense matrix units for the 2:4 pattern. On this course's A100, whose dense bfloat16 rate is 312 TFLOP/s, twice that is $624$ TFLOP/s. The paper's second claim is that a training workflow, which prunes and then retrains, keeps accuracy across a range of tasks. The recipe retrains. Wanda and SparseGPT are one-shot, and this chapter makes no claim about how much accuracy a one-shot 2:4 model loses on a language model.

A doubled math rate does not help a decode step, because the step is not limited by math. At batch 1 the weight matrices run near one FLOP per byte, and the ridge point is $161.2$ ([chapter 10](/c/10-roofline)). The time is the weight read. What 2:4 changes is the byte count.

| Format | Bytes per weight | Weight bytes if all 53.8 GB were this format | Read at 1275 GB/s | Fewer bytes than dense |
|---|---|---|---|---|
| Dense bfloat16 | 2 | 53.800 GB | 42.196 ms | 1.000x |
| 2:4 bfloat16 | 1.125 | 30.262 GB | 23.735 ms | 1.778x |
| Group-128 int4 | 0.515625 | 13.870 GB | 10.879 ms | 3.879x |

These are upper bounds. The 53.8 GB includes weights that nobody prunes, and each row assumes a kernel that reaches the copy rate. No row is a measurement. This machine has no GPU, and the chapter reports no A100 time.

> [!KEY] On decode, 2:4 is a byte trick, and int4 is a bigger one
> Group-128 int4 cuts the bytes by $3.879$, against $1.778$ for 2:4 in bfloat16. A decode-bound deployment gets more from [chapter 18a](/c/18a-int4-and-kv-quantization)'s int4 than from 2:4 alone. The two combine, but the accuracy cost of each compounds, and the lab does not measure the combination.

2:4 does help where the math is the limit. A long prefill at hundreds of tokens per step runs above the ridge, and a doubled math rate applies there. Whether a real kernel reaches it depends on the matrix shapes and the library. The NVIDIA paper reports the math rate, and this chapter does not quote an end-to-end speedup, because it did not measure one.

PyTorch exposes the format as `torch.sparse.to_sparse_semi_structured`, which converts a 2:4 pruned half-precision CUDA tensor. The lab's `sparse_semi_structured_check` calls it and compares the sparse product with the dense one. ==The check skips on a CPU, and nobody has run it on a GPU for this chapter.== The documentation lists the dtype and shape limits, so read it before you rely on a shape.

## Distill a draft for chapter 20

A distilled draft is a small model trained to match the large model's output distribution, and that match sets the acceptance rate in speculative decoding.

The machine-learning course owns distillation as a topic, so this section keeps only what chapter 20 needs. For the full treatment, see the [machine-learning course](https://ml.ifkash.dev/).

[Chapter 20](/c/20-speculative-decoding) accepts a draft token with a probability that sums to $\sum_x \min(p(x), q(x))$, where $p$ is the target and $q$ is the draft. That sum equals $1 - \mathrm{TV}(p, q)$, one minus the total variation distance. Pinsker's inequality bounds the distance by the divergence:

$$
\mathrm{TV}(p, q) \le \sqrt{\tfrac{1}{2}\,\mathrm{KL}(p \,\Vert\, q)}.
$$

A draft that minimizes $\mathrm{KL}(p \,\Vert\, q)$ against the target on the target's own text therefore raises the lower bound on its acceptance rate. That is the argument for distilling the draft instead of training it on hard labels. It is a bound, not a measured acceptance.

Hinton, Vinyals, and Dean describe the loss. With teacher logits $z^{t}$ and student logits $z^{s}$, temperature $T$, and batch size $B$:

$$
\boxed{\mathcal{L} = T^{2}\,\frac{1}{B}\sum_{b}\mathrm{KL}\!\left(\mathrm{softmax}(z^{t}_b / T)\,\Vert\,\mathrm{softmax}(z^{s}_b / T)\right)}
$$

A temperature above 1 flattens both distributions, so the student also sees the teacher's ranking of the wrong tokens. The gradient of $\mathcal{L}$ with respect to the student logits is $T\,(p^{s} - p^{t})/B$, where $p$ is the tempered softmax. Without the $T^2$ factor the gradient is $(p^{s} - p^{t})/(TB)$. Hinton and coauthors note that it then scales as $1/T^2$ at large $T$, and the factor keeps its size comparable across temperatures.

> [!EXAMPLE] One row, two tokens
> Take teacher logits $(2, 0)$ and student logits $(0, 0)$, so $p^{t} = (0.880797, 0.119203)$ and the student says $(0.5, 0.5)$. The lab's `kd_loss` gives loss $0.327813$ and a gradient of $\mp 0.380797$ at $T = 1$. At $T = 2$ the loss is $0.443776$ and the gradient is $\mp 0.462117$. At $T = 4$ it is $0.484798$ and $\mp 0.489837$. Without $T^2$, the $T = 4$ gradient would be $16$ times smaller.
>
> At $T = 1$ the total variation distance is $0.380797$, so one position accepts a draft token with probability $0.619203$. Pinsker's bound gives $\sqrt{0.327813 / 2} = 0.404854$, which is above $0.380797$.

The lab checks the loss against a pure-Python reference, checks the autograd gradient against $T\,(p^{s} - p^{t})/B$, and checks it again with finite differences in float64. It does not train a student.

Pruning and distillation also combine. The Minitron paper compresses a 15B Nemotron-4 model to 8B and 4B by pruning depth, width, attention, and MLP, and then retraining the pruned model with distillation. It reports up to 40x fewer training tokens for each derived model than training from scratch. Those are the authors' numbers on their models.

## What goes wrong

These are the mistakes the lab is built to catch.

**Leaving the activations out of the Wanda score.** `wanda_scores` returns the magnitude, and the two methods tie at $0.248337$. The check wants Wanda under half of that.

**Averaging instead of taking the norm.** Two identical tokens must scale a column norm by $\sqrt{2}$. A mean leaves it unchanged. The worked check feeds a duplicated token to catch it.

**Checking density instead of structure.** A 50% mask built per row passes a density check and fails the group count. Fixed-size indices need exactly two per group of four.

**Dropping a kept zero from the compressed format.** A weight that is zero but kept still owns one of the two slots. Decide by the mask, or a group with a zero value loses its position.

**Forgetting $T^2$ in the distillation loss.** At $T = 2$ the gradient is a quarter of the one above. A finite-difference check of the function against itself cannot see this. The lab compares with $T\,(p^{s} - p^{t})/B$.

**Quoting the byte bound as a speedup.** $1.778$ is bytes, for a model that is entirely prunable, at a copy rate. It is a ceiling that no one measured here.

## Check your understanding

> [!RECAP]
> - Wanda scores $|W_{oj}| \cdot \lVert X_{:,j} \rVert_2$ and prunes inside each output row. On the seeded layer its 50% error is $0.091117$, against $0.248337$ for magnitude.
> - SparseGPT zeros a weight and corrects the rest with the inverse-Hessian update that GPTQ uses for rounding.
> - A 2:4 mask keeps two of every four columns. Check the count per group, not the density. On the seeded layer the Wanda 2:4 error is $0.123251$.
> - A compressed 2:4 matrix stores values and 2-bit indices, $1.125$ bytes per dense bfloat16 weight. Group-128 int4 is $0.515625$.
> - Ampere's 2:4 tensor cores double the dense math rate. Decode reads bytes, so the ceiling is $1.778$ on the weight read, and no number here is an A100 measurement.
> - The distillation loss is $T^2$ times the KL divergence, with gradient $T(p^{s} - p^{t})/B$. A smaller KL raises the lower bound on a draft's acceptance.

> [!QUESTION] Why is a 50% unstructured mask not a 2:4 mask, even though both keep half the weights?
> The hardware stores a fixed-size 2-bit index for exactly two weights per group of four. An unstructured mask keeps three in one group and one in another, so no fixed-size index describes it. The lab's per-row Wanda mask has density $0.5$ and fails `is_valid_2_4`. Only the count in each group decides validity.

> [!QUESTION] Unstructured 50% sparsity with a 16-bit column index stores $4$ bytes per kept bfloat16 weight. What does that save?
> Nothing. At 50% sparsity you store half the weights at $2 + 2 = 4$ bytes each, which is $2$ bytes per dense weight, the dense size. The 2-bit index of the 2:4 format is what brings it to $1.125$.

> [!QUESTION] The sparse tensor core doubles the math rate to 624 TFLOP/s. Why does a decode step not get twice as fast?
> At batch 1 the step is limited by reading the weights, near one FLOP per byte against a ridge point of $161.2$. The math rate is not the limit. 2:4 helps through the byte count: at most $1.778$ on the weight read, and only for weights that are pruned and read at the copy rate.

> [!QUESTION] On decode, why does group-128 int4 beat 2:4 bfloat16 on bytes?
> Int4 stores $0.515625$ bytes per weight and 2:4 bfloat16 stores $1.125$. The first cuts the bytes by $3.879$ and the second by $1.778$. The bound says nothing about accuracy, and the two formats lose accuracy differently.

> [!QUESTION] Why does the gradient of the distillation loss use $T^2$, and what does the finite-difference check not catch?
> The raw KL gradient is $(p^{s} - p^{t})/(TB)$. The chain rule gives the $1/T$, and at large $T$ the difference $p^{s} - p^{t}$ shrinks too. Multiplying the loss by $T^2$ gives $T\,(p^{s} - p^{t})/B$. A finite difference of the submitted function matches its own autograd gradient whether or not $T^2$ is present, so the lab also compares with the analytic gradient.

## Lab

> [!TRY]
> Score weights by magnitude and by Wanda, prune rows, build and validate a 2:4 mask, compress it, count the bytes, and write the distillation loss. You pass when Wanda's 50% error is under half of magnitude's on the seeded layer, the 2:4 mask is valid, the compressed matrix round-trips at $1.125$ bytes per weight, and the loss gradient matches finite differences. The semi-structured check passes as a skip on a CPU.

Implement the following functions:

- `magnitude_scores` and `wanda_scores`. The norm is over tokens, one per input column.
- `prune_rowwise`. Drop `round(sparsity * in)` lowest scores per row, and a tie drops the lower index.
- `prune_2_4` and `is_valid_2_4`. Keep two of four, and count per group.
- `compress_2_4`, `decompress_2_4`, and `compressed_bytes`. Values plus 2-bit positions.
- `output_error`. The relative layer-output error.
- `kd_loss`. $T^2$ times the batch-mean KL, with the teacher outside the graph.
- `sparse_semi_structured_check`. Returns `None` on a CPU. The reference version was never run on a GPU.

The first check calls only `magnitude_scores`.

## Further reading

- [Deep compression: compressing deep neural networks with pruning, trained quantization and Huffman coding](https://arxiv.org/abs/1510.00149). Pruning, quantization, and coding in sequence, with 35x to 49x storage reduction.
- [SparseGPT: massive language models can be accurately pruned in one-shot](https://arxiv.org/abs/2301.00774). The OBS update applied to pruning, including the 2:4 and 4:8 patterns.
- [A simple and effective pruning approach for large language models](https://arxiv.org/abs/2306.11695). Wanda, pruning by weights and activations.
- [Accelerating sparse deep neural networks](https://arxiv.org/abs/2104.08378). The 2:4 pattern, the Ampere sparse tensor cores, and the retraining workflow.
- [Compact language models via pruning and knowledge distillation](https://arxiv.org/abs/2407.14679). Pruning a pretrained model and retraining it with distillation.
- [Distilling the knowledge in a neural network](https://arxiv.org/abs/1503.02531). The temperature-scaled loss.
- [MIT 6.5940, TinyML and Efficient Deep Learning Computing, Fall 2024](https://hanlab.mit.edu/courses/2024-fall-65940). Lecture 3, Pruning and Sparsity (Part I), 12 September. Lecture 4, Pruning and Sparsity (Part II), 17 September. Lecture 9, Knowledge Distillation, 3 October. The syllabus lists titles and dates. This chapter does not claim what each lecture covers beyond its title.
