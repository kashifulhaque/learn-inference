---
title: Inference-aware scaling laws and model design
slug: 02a-inference-aware-model-design
part: "Part 1 — Ground truth"
summary: The Chinchilla compute-optimal split, the same problem with serving cost added, and a byte count per decoded token that ranks model shapes.
minutes: 55
gpu: false
objectives:
  - Find the compute-optimal parameter and token counts for a budget from the fitted loss and C = 6ND.
  - Minimize training plus lifetime inference FLOPs for a target loss, and show that over-training wins as the number of served tokens grows.
  - Price one decoded token in bytes from a config, with the weights shared across the batch and the KV cache and recurrent state paid per sequence.
  - Rank candidate shapes by decode cost at a batch and a context length, and say which term decides the ranking.
  - Explain what a 3:1 hybrid with 4 KV heads buys at serving time, and what this chapter cannot say about why it was chosen.
lab: 02a-model-design
---

# Inference-aware scaling laws and model design

> [!TLDR]
> - The fitted loss $L(N, D) = E + A/N^{\alpha} + B/D^{\beta}$ and the budget $C \approx 6ND$ give a closed form for the compute-optimal $N$ and $D$. With the published constants, a 26.9B model is compute-optimal at 1.471T tokens (54.7 per parameter). The Besiroglu refit puts it at 19.3 per parameter.
> - Add the cost of serving, 2N FLOPs per token, and the best model for a fixed loss gets smaller and trains on more data. At $10^{13}$ served tokens, the best model for the 26.9B model's loss has 12.07B parameters and trains on 4.28T tokens, which is 28.9% fewer lifetime FLOPs.
> - There is no closed form once served tokens count, so the lab searches over $N$. Each $N$ fixes the training tokens through the loss target.
> - One decoded token moves the weights divided by the batch, plus one sequence's KV read, plus its recurrent state read and written. At batch 1 the weights are 99% of the bytes. At batch 32 and 32,768 tokens the KV read is half.
> - A shape that wins at batch 1 loses at batch 32 and 32k. The 3:1 hybrid with 4 KV heads costs 4.14 GB per token there, against 10.24 GB for the same stack with full attention in every layer.

[Chapter 2](/c/02-memory-arithmetic) counts what the model costs to hold: 53.8 GB of weights, 64 KiB of KV per token, and 147.8 MiB of recurrent state. It treats the shape of the model as given. This chapter asks the question that comes before it. If you were choosing the size and shape of a model that you will serve for a long time, which numbers would you optimize?

Two bodies of work answer that. The scaling-law papers say how big a model to train for a compute budget, and how the answer changes once serving is part of the bill. The architecture papers, grouped-query attention among them, say what a shape costs once the model is running. Both reduce to arithmetic you can do from a config, so the lab is a CPU lab.

The course's own model is the running example. It has 26,895,979,520 parameters, 16 full-attention layers, and 4 KV heads. This chapter does not know how its authors chose those numbers, and it doesn't claim to. It prices them against the alternatives.

## Before you start

**Loss.** The *pre-training loss* is the average negative log-likelihood per token on held-out text. Lower is better. The loss values in this chapter come from a published fit on DeepMind's training data. They are in that fit's units, so don't read them as this model's loss.

**FLOPs per token.** A model with $N$ parameters does about $2N$ floating-point operations per token on a forward pass: one multiply and one add per parameter. Training adds a backward pass of about twice that, so about $6N$ per trained token. These are the counts [chapter 10](/c/10-roofline) and [chapter 15](/c/15-paged-attention) use, and the scaling-law papers use them too. The training budget is then:

$$
C \approx 6ND
$$

where $D$ is the number of training tokens.

**Served tokens.** *Inference demand*, written $D_{\mathrm{inf}}$, is every token the model processes after training: prompts and generations, summed over every request in its life.

**Over-training.** A model is *over-trained* when it sees more tokens than the compute-optimal $D$ for its size. It reaches a lower loss than a compute-optimal model of the same size, at a higher training cost.

**Bytes and the A100.** The A100 80GB moves 1935 GB/s from HBM and does 312 TFLOP/s of bfloat16, the ratings from [chapter 10](/c/10-roofline). Decode at a small batch is limited by the bytes, so this chapter counts bytes.

## The fitted loss and the compute-optimal split

Given a FLOP budget $C$, the question is how much of it to spend on a bigger model and how much on more data. Hoffmann et al. (the Chinchilla paper) answer by fitting a loss to many training runs. They write it in the following form:

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

The three terms have plain readings. $E$ is the loss of an ideal model, the entropy of the text. The $A/N^{\alpha}$ term is what a finite model gives up, and it shrinks as $N$ grows. The $B/D^{\beta}$ term is what finite data gives up, and it shrinks as $D$ grows.

### The constants

The paper's body prints the exponents of the resulting frontier, $a = 0.46$ and $b = 0.54$ for this fit, but not the digits of the fit. Besiroglu et al. recovered those digits from the paper's TeX source, and they are the ones this chapter uses:

| Constant | Hoffmann et al. | Besiroglu et al. refit |
|---|---|---|
| $E$ | 1.6934 | 1.8172 |
| $A$ | 406.4 | 482.01 |
| $B$ | 410.7 | 2085.43 |
| $\alpha$ | 0.3392 | 0.3478 |
| $\beta$ | 0.2849 | 0.3658 |

Besiroglu et al. refit the same functional form to a partial reconstruction of the data from the paper's Figure 4. They report that the original values, rounded as printed in the paper ($E = 1.69$, $\alpha = 0.34$, $\beta = 0.28$), fit the data poorly, and that the refit is consistent with the paper's other two approaches. Treat the two columns as two estimates of one law. The qualitative results in this chapter hold for both, and the numbers differ, which the lab lets you see.

### The closed form

Under $C = 6ND$, each $N$ fixes $D = C / (6N)$. Substitute that into the loss and one variable is left. Set its derivative to zero:

$$
\frac{d}{dN}\left[ E + A N^{-\alpha} + B \left(\frac{C}{6}\right)^{-\beta} N^{\beta} \right] = 0
\;\Longrightarrow\;
-\alpha A N^{-\alpha - 1} + \beta B \left(\frac{C}{6}\right)^{-\beta} N^{\beta - 1} = 0
$$

Solve for $N$. The result is a power law in the budget:

$$
\boxed{N_{\mathrm{opt}} = G \left(\frac{C}{6}\right)^{a}, \qquad
D_{\mathrm{opt}} = \frac{C}{6 N_{\mathrm{opt}}}}
\qquad\text{with}\quad
G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha + \beta}},\quad
a = \frac{\beta}{\alpha + \beta}
$$

In words: parameters and tokens grow as powers of the budget, with exponents $a$ and $1 - a$ that add to 1. The Hoffmann constants give $G = 1.3004$ and $a = 0.4565$. The refit gives $a = 0.5126$, so tokens and parameters scale almost one to one.

> [!EXAMPLE] A budget of $10^{24}$ FLOPs
> With the Hoffmann constants, $C/6 = 1.667 \times 10^{23}$, so $N_{\mathrm{opt}} = 1.3004 \times (1.667 \times 10^{23})^{0.4565} = 51.85\text{B}$ parameters. Then $D_{\mathrm{opt}} = 10^{24} / (6 \times 51.85 \times 10^{9}) = 3.214\text{T}$ tokens. Check the budget: $6 \times 51.85 \times 10^9 \times 3.214 \times 10^{12} = 10^{24}$.

### How many tokens per parameter

The ratio $D/N$ is the number people quote. Chinchilla's Table 3 gives 205.1B tokens for a 10B model, which is 20.5 per parameter, and the rule of thumb of 20 comes from there. The ratio from the parametric fit depends on which constants you use, and it moves with $N$:

| Model | Hoffmann constants | Besiroglu refit |
|---|---|---|
| $N$ = 1B | 29.2 tokens per parameter | 22.7 |
| $N$ = 10B | 45.3 | 20.3 |
| $N$ = 26.9B, this model | 54.7 (1.471T tokens) | 19.3 (0.519T tokens) |

The refit lands near the paper's 20 at every size, and the Hoffmann digits don't. That is the substance of the replication: a fit this sensitive to its constants shouldn't be quoted to three digits. Use the 54.7 and the 19.3 as two ends of a range, not as a prediction.

## Training plus serving

The compute-optimal split minimizes loss for a training budget. A team that will serve the model has a different problem. It wants a target loss, and it wants the cheapest way to get there once it counts the tokens it will serve.

### The problem

Sardana et al. ("Beyond Chinchilla-optimal") state it directly. Fix a loss $\ell$ and an inference demand $D_{\mathrm{inf}}$. Find the model that reaches $\ell$ for the fewest lifetime FLOPs, where a trained token costs $6N$ and a served token costs $2N$:

$$
\min_{N,\, D}\; \underbrace{6ND}_{\text{training}} + \underbrace{2N D_{\mathrm{inf}}}_{\text{serving}}
\qquad\text{subject to}\qquad L(N, D) = \ell
$$

In words: every parameter you remove makes each served token cheaper, but it must be made up by training on more tokens. The constraint is what ties the two together. It also gives each $N$ exactly one $D$, by inverting the loss:

$$
D(N) = \left( \frac{B}{\ell - E - A / N^{\alpha}} \right)^{1/\beta}
$$

The denominator must be positive, so $N$ can't go below $N_{\mathrm{floor}} = (A / (\ell - E))^{1/\alpha}$. At that size the training tokens needed are infinite, because no amount of data gets a too-small model to the loss.

> [!EXAMPLE] The sum with one model
> Take $N = 10^9$, $D = 2 \times 10^{10}$ tokens, and $D_{\mathrm{inf}} = 10^{11}$. Training costs $6 \times 10^9 \times 2 \times 10^{10} = 1.2 \times 10^{20}$ FLOPs. Serving costs $2 \times 10^9 \times 10^{11} = 2.0 \times 10^{20}$. The total is $3.2 \times 10^{20}$, and serving is 62.5% of it. Count a served token at $6N$ by mistake and the total is $7.2 \times 10^{20}$, more than twice too large.

### There is no closed form, so search

Setting the derivative of the total to zero gives an equation with two power-law terms in $D$ that has no general solution. Sardana et al. prove it in their appendix and solve it with Newton's method. Because each $N$ fixes $D$, the lab searches over $N$ alone, in log space, between $N_{\mathrm{floor}}$ and a size well above any answer. The cost is large near the floor, where $D$ explodes, and it grows with $N$ above the optimum, so the search finds one valley.

To check a solution, differentiate the cost with respect to $\log N$ by finite differences. At the optimum the slope is zero, and the cost is higher 10% above and 10% below.

### What the search finds

The target loss in the following table is the loss of a compute-optimal model with this model's 26.9B parameters, 1.9515 under the Hoffmann constants. The last column compares against training that 26.9B model on its own compute-optimal tokens and then serving it the same volume.

| Served tokens | Best $N$ | Training tokens | Tokens per parameter | Lifetime FLOPs saved |
|---|---|---|---|---|
| none | 26.896B | 1.471T | 54.7 | 0.0% |
| $10^{11}$ | 25.984B | 1.523T | 58.6 | 0.0% |
| $10^{12}$ | 21.012B | 1.922T | 91.5 | 2.3% |
| $10^{13}$ | 12.070B | 4.279T | 354.5 | 28.9% |
| $10^{14}$ | 6.920B | 15.762T | 2,277.7 | 63.7% |
| $10^{15}$ | 4.749B | 74.073T | 15,599.1 | 78.5% |

With no serving, the search returns the closed-form answer, which is the first check in the lab. As served tokens grow, the best model shrinks and trains on more data, because each extra parameter is paid for again on every served token. Below about $10^{11}$ served tokens the compute-optimal model is within 0.1% of the best, so a model that sees little traffic should be sized by the Chinchilla rule.

```viz
02a-lifetime-cost
```

The Besiroglu refit gives the same shape with different numbers: 17.62B at $10^{12}$ served tokens, 10.18B at $10^{13}$, and 6.74B at $10^{14}$, saving 8.7%, 44.3%, and 68.5%. The refit's baseline trains on fewer tokens, so serving is a larger share of its lifetime cost, and the savings arrive sooner.

> [!KEY] Over-training is the cheap way to a given loss when you serve a lot
> Training a smaller model on more tokens costs more to train and less to serve. The break-even moves with the number of tokens you expect to serve, which you must estimate before you train. Sardana et al. state that the method assumes you can estimate demand before training, because the model's size is fixed before the demand is known.

### FLOPs are not the cost

Sardana et al. flag the largest simplification in this problem. A served token and a trained token both count at their FLOPs, but decode runs far below the peak. The paper cites a model FLOP utilization as low as about 1% for inference, against 40% to 60% for training, because decode is limited by memory bandwidth. A served token therefore costs more than its FLOPs suggest.

The lab models that with `inference_weight`, a multiplier on the served-token term. At $10^{13}$ served tokens, a weight of 5 moves the best model from 12.07B to 8.04B parameters, and a weight of 50 moves it to 5.22B. The weight is a dimensionless knob you set. This course has no citable A100 price or utilization, so it doesn't pick one for you.

### What real models do

The Llama 3 report is a published example of this trade. It states that the 405B flagship, trained on 15.6T tokens, is approximately compute-optimal for its budget. It also says the smaller models are trained for much longer than is compute-optimal, and that they perform better than compute-optimal models at the same inference budget. The report gives a corpus of about 15T tokens for the herd. If an 8B model saw all of it, that would be about 1,900 tokens per parameter, which is far past the 20 to 55 that the fits above call optimal. The report doesn't give a per-model token count in the passage this chapter reads, so treat 1,900 as an illustration.

## Pricing a shape

The first half of the chapter chose $N$. The second half asks what a given shape costs to decode, so you can compare shapes at about the same $N$. The unit is bytes moved per generated token, because [chapter 10](/c/10-roofline) shows decode at serving batch sizes is limited by bytes.

### The cost of a token

A decode step processes one new token for each of $B$ sequences. It reads every weight once, however many sequences are in the batch. Each sequence also reads its own cache. Per generated token, that is:

$$
\boxed{
\text{bytes per token} =
\underbrace{\frac{b \, N_{\mathrm{params}}}{B}}_{\text{weights, shared}}
+ \underbrace{\kappa \, L}_{\text{KV read}}
+ \underbrace{2 S}_{\text{state read and written}}
}
$$

The symbols come from [chapter 2](/c/02-memory-arithmetic). $b = 2$ is bytes per element, $N_{\mathrm{params}}$ is the parameter count, $\kappa$ is the KV bytes per token (64 KiB for this model), $L$ is the context length, and $S$ is the recurrent state per sequence (154,927,104 bytes). The state counts twice because each step reads it and writes it back.

The weight term includes the embedding and the output projection, as [chapter 17a](/c/17a-performance-model) does. The output projection is a full matrix multiply, so it is read in full. The input embedding is a table lookup that reads one row per token, so counting all of it overstates the read by 2,542,796,800 bytes, 4.73% of the weights. The overstatement is the same for every shape that keeps the vocabulary, so it doesn't change those rankings.

> [!EXAMPLE] This model at batch 1 and at batch 32
> At batch 1 and context 0, the weights are 53,791,959,040 bytes and the state read and written is 309,854,208, so a token moves 54,101,813,248 bytes. At 1935 GB/s that is a 27.960 ms floor, and chapter 17a's 27.962 ms adds 3.95 MB of activation traffic and one token of KV. At batch 32 and 32,768 tokens, the weights share is 1,680,998,720 bytes, the KV read is 2,147,483,648, and the state is 309,854,208, a total of 4,138,336,576 bytes. The KV read is now the largest term.

### Where the KV read overtakes the weights

Setting the KV read equal to the weights share gives the context where the cache starts to dominate:

$$
L^{*} = \frac{b \, N_{\mathrm{params}}}{B \, \kappa}
$$

For this model, $L^{*}$ is 820,800 tokens at batch 1, 102,600 at batch 8, 25,650 at batch 32, and 6,413 at batch 128. A bigger batch shrinks each sequence's share of the weights, so the cache takes over at shorter contexts. For the same stack with full attention in every layer, $\kappa$ is four times larger and $L^{*}$ at batch 32 falls to 6,286 tokens.

Chapter 2's break-even of 788 tokens is a different comparison. It asks when the hybrid holds less memory than a dense stack. This one asks when the cache's traffic exceeds the weights'. Both come from the same $\kappa$.

## Ranking candidate shapes

With the cost formula, you can change one thing about the config and read off the price. Each row in the next table changes one field from this model, at batch 32 and 32,768 tokens of context. The parameter column shows that the comparison is at about the same size.

| Shape | Parameters | KV per token | Bytes per token | Floor per step |
|---|---|---|---|---|
| 15:1 hybrid (4 full-attention layers) | 27.028B | 16 KiB | 2.61 GB | 43.22 ms |
| 2 KV heads | 26.812B | 32 KiB | 3.06 GB | 50.59 ms |
| 128k vocabulary | 25.664B | 64 KiB | 4.06 GB | 67.16 ms |
| **This model: 3:1 hybrid, 4 KV heads** | 26.896B | 64 KiB | 4.14 GB | 68.44 ms |
| 1:1 hybrid | 26.720B | 128 KiB | 6.17 GB | 102.06 ms |
| 8 KV heads | 27.064B | 128 KiB | 6.30 GB | 104.12 ms |
| Full attention in every layer, 4 KV heads | 26.367B | 256 KiB | 10.24 GB | 169.31 ms |
| 24 KV heads (no grouping) | 27.735B | 384 KiB | 14.93 GB | 246.87 ms |
| Full attention in every layer, 24 KV heads | 29.723B | 1,536 KiB | 53.40 GB | 883.06 ms |

The floor column is the longer of the byte time and the $2N$-FLOP matmul time, with launch overhead and activation traffic left out. Every row here is memory-bound. All of the numbers are computed by the lab's solution, and none is an A100 measurement.

```viz
02a-decode-shapes
```

### One knob at a time

**KV heads.** Cutting from 24 to 4 KV heads cuts the cache by 6 times and saves 10.8 GB per token at this batch and context. Cutting from 4 to 2 saves another 1.08 GB. The saving is linear in the head count, so the biggest gain comes from the first cut, and each later cut gives less in absolute bytes. Grouped-query attention, the technique in [chapter 7](/c/07-grouped-query-attention), is this knob.

**Hybrid fraction.** Each layer you move from full to linear attention removes 4 KiB of cache per token and adds a fixed 3,227,648 bytes of state. Going from all full attention to 3:1 takes the per-token cost at this batch and context from 10.24 GB to 4.14 GB, a factor of 2.5. Going further, to 15:1, takes it to 2.61 GB. [Chapter 6](/c/06-gated-delta-rule) explains the linear layer.

**Depth against width.** The lab's `match_params` holds the parameter count near 26.9B and changes the number of layers, scaling the hidden size to match. With the head geometry held fixed, 32 layers need a hidden size of 7424 and have 8 full-attention layers, so the cache is 32 KiB per token. At 96 layers the hidden size is 4096 and there are 24 full-attention layers, 96 KiB per token. At batch 32 and 32,768 tokens the 32-layer shape costs 2.92 GB per token and the 96-layer shape 5.38 GB, against 4.14 GB for this model. Deeper stacks pay more cache and more state at the same size. Shallow ones pay in wider matrices, which this byte count doesn't price.

**Vocabulary size.** The vocabulary sets the size of two matrices. Going from 248,320 to 128,000 tokens removes 1,232,076,800 parameters, which is 2,464,153,600 bytes of weights. The lab counts 25.664B parameters against 26.896B. The weights share falls by 77 MB per token at batch 32, which is 1.9% of the bytes at 32k. The vocabulary is a weight-side cost, so it matters most at small batch and short context. [Chapter 2](/c/02-memory-arithmetic) notes that a large vocabulary buys shorter sequences for non-English text, and that benefit isn't in this count.

### The ranking flips

The shape that wins depends on the batch and the context, so the first question in a design review is which regime you serve. At batch 1 and 1,024 tokens, the all-full-attention shape with 4 KV heads ranks second of the eleven shapes in the figure script, at 53.00 GB per token against this model's 54.17 GB. Only the 128k vocabulary is ahead. The weights are 99% of the bytes, so parameter count decides. At batch 32 and 32,768 tokens, the same shape ranks ninth, at 10.24 GB. The lab's ranking check pins both orders for six shapes.

## Why a 3:1 hybrid with 4 KV heads

This section is about what the numbers say, not about what the model's designers intended. The course has no source for their reasoning, and a ranking by decode bytes leaves out the thing a designer cares about most, which is loss.

What the numbers do say is that both choices save a lot of serving cost at long context. Against the same stack with full attention in every layer, the 3:1 hybrid cuts bytes per token by 2.5 times at batch 32 and 32k, and the 4-head grouping against 24 heads cuts the cache by 6 times. Together, against full attention with 24 KV heads, the model moves 13 times fewer bytes per token (4.14 GB against 53.40 GB). Those savings are what makes the batch figures in chapter 2 possible, such as 11 sequences of 32k on one card.

The numbers also say the model is not the cheapest shape in the table. A 15:1 hybrid or 2 KV heads costs less at 32k. A designer weighs that against quality: fewer KV heads and fewer full-attention layers each give up something in modelling, and this chapter doesn't measure it. The cost side is the part you can compute from a config, and the lab computes it. The quality side needs training runs, which is where the scaling laws in the first half of the chapter come in. A design is a point on a frontier, and these two halves are its two axes.

## What goes wrong

- **Counting a served token at 6N.** The training rate is not the serving rate. Serving a token needs a forward pass only. The lab's lifetime check fails on the exact sum, and the optimum comes out too small, 9.07B instead of 12.07B at $10^{13}$ served tokens.
- **Leaving the embedding out of the weight read.** It looks like lookup and not computation, so a weight count that skips it is tempting. The output projection is read in full on every step. The input table is a row lookup, so counting all of it overstates the read by 4.73%, but dropping both matrices understates it by 5.09 GB, 9.5% of the weights. The lab's byte check pins 54,101,813,248 at batch 1 and context 0.
- **Quoting the tokens-per-parameter ratio as a constant.** It depends on the constants of the fit and on $N$. Quote the fit you used.
- **Reading the fit's loss as this model's loss.** The values come from a fit to another dataset. Use them to compare designs under the fit, not to predict your model's perplexity.
- **Ranking shapes at one batch and one context.** The ranking flips between the weight-bound and the cache-bound regimes. State the regime with the ranking.
- **Forgetting the demand estimate.** The optimum with inference is only as good as $D_{\mathrm{inf}}$. A model sized for $10^{14}$ served tokens that sees $10^{11}$ wasted most of its training.

> [!RECAP]
> - The fitted loss and $C \approx 6ND$ give $N_{\mathrm{opt}} = G (C/6)^{a}$ in closed form. The tokens per parameter it implies depends on the constants: 54.7 with the Hoffmann digits and 19.3 with the Besiroglu refit, for a 26.9B model.
> - Serving cost, $2N$ per token, moves the optimum for a fixed loss to a smaller model on more data. There is no closed form, so you search over $N$, with each $N$ fixing $D$ through the loss target.
> - At $10^{13}$ served tokens the best model for the 26.9B model's loss has 12.07B parameters on 4.28T tokens, and saves 28.9% of lifetime FLOPs. A heavier weight on a served token pushes it smaller.
> - A decoded token moves the weights divided by the batch, plus one sequence's KV read, plus its state read and written. The KV read overtakes the weights at 25,650 tokens for batch 32.
> - Ranking shapes means naming the regime. This model's 3:1 hybrid with 4 KV heads costs 4.14 GB per token at batch 32 and 32k, 2.5 times less than full attention in every layer.

## Check your understanding

> [!QUESTION] Why does the compute-optimal model get smaller when you add serving cost, if the loss target stays the same?
> A served token costs $2N$, so every parameter is paid for again on every token served. A smaller model is cheaper to serve. To reach the same loss, it must train on more tokens, which costs more to train. The optimum is where the extra training cost of removing a parameter equals the serving cost it saves, and more served tokens push that point toward fewer parameters.

> [!QUESTION] A team expects $10^{11}$ served tokens for a 26.9B model's loss. Should it over-train?
> Not meaningfully. At $10^{11}$ served tokens the best model has 25.98B parameters, and the saving over the compute-optimal 26.9B model is under 0.1%. Serving is a small share of the lifetime FLOPs, so the Chinchilla rule is close to optimal.

> [!QUESTION] At batch 1 the lab ranks a 128k vocabulary above a 15:1 hybrid. At batch 32 and 32,768 tokens the order reverses. Why?
> At batch 1 the weights are 99% of the bytes, so the shape with fewer parameters wins, and the vocabulary change removes 1.23B of them. At batch 32 and 32k the KV read is half of the bytes, and the 15:1 hybrid has a quarter of this model's cache. The cache term is the one that differs most between the two shapes there.

> [!QUESTION] Why does a bigger batch make the KV cache matter at shorter contexts?
> The weights are read once per step, so each sequence's share is $b N_{\mathrm{params}} / B$. A larger batch makes that share smaller, while each sequence's KV read stays $\kappa L$. The two are equal at $L^{*} = b N_{\mathrm{params}} / (B \kappa)$, which falls as $B$ grows: 820,800 tokens at batch 1 and 6,413 at batch 128.

## Lab

> [!TRY]
> Write the scaling-law arithmetic and the decode cost as code. You pass when the fitted loss and the compute-optimal split match to six digits, when the search with inference gives 12.07B parameters at $10^{13}$ served tokens, and when one decoded token at batch 32 and 32,768 tokens costs 4,138,336,576 bytes.

The lab runs on CPU and needs no weights. You implement the functions below.

- `chinchilla_loss` and `compute_optimal` give the fitted loss and the closed-form split under $C = 6ND$.
- `tokens_for_loss` inverts the loss, and returns infinity below the floor.
- `lifetime_flops` adds $6N$ per trained token and $2N$ per served token, with an optional weight on the served term.
- `optimal_with_inference` searches over $N$ for the cheapest model at a target loss.
- `param_count` reproduces chapter 2's 26,895,979,520 from a config, and moves when you change the shape.
- `inference_cost_per_token` prices one decoded token in bytes, and `rank_shapes` orders a dictionary of configs by it.
- `match_params` returns a config with a given number of layers and about the same parameter count.

The checks, from the smallest piece to the whole:

- The fitted loss at $10^9$ parameters and $2 \times 10^{10}$ tokens is 2.53054.
- The compute-optimal split at $10^{24}$ FLOPs is 51.85B parameters on 3.214T tokens, with a zero finite-difference slope.
- `tokens_for_loss` round-trips and returns infinity under the floor.
- Lifetime FLOPs for $(10^9, 2 \times 10^{10}, 10^{11})$ are exactly $3.2 \times 10^{20}$.
- The optimum with inference returns the closed-form answer for no serving, then 21.01B, 12.07B, and 6.92B at $10^{12}$, $10^{13}$, and $10^{14}$ served tokens, with a flat cost at the optimum.
- A served-token weight of 5 gives 8.04B at $10^{13}$ served tokens.
- `param_count` gives 26,895,979,520 for this model, and exact counts for three changed shapes.
- `inference_cost_per_token` gives 54,101,813,248 at batch 1 and context 0, and 4,138,336,576 at batch 32 and 32,768 tokens.
- `rank_shapes` gives one order at batch 32 and 32,768 tokens and a different one at batch 1 and 1,024 tokens.
- `match_params` returns 32 layers and a hidden size of 7424 for the 26.9B target.

The harness reports `optimal_N_billions_at_1e13_served` and `decode_gb_per_token_b32_32k` as metrics. Every number is arithmetic on the constants and the config, and none is a measurement.

## Further reading

- [Training compute-optimal large language models](https://arxiv.org/abs/2203.15556) — Hoffmann et al., the Chinchilla paper. Section 3.3 has the parametric loss.
- [Chinchilla scaling: a replication attempt](https://arxiv.org/abs/2404.10102) — Besiroglu et al., the refit and the digits of the original fit.
- [Beyond Chinchilla-optimal: accounting for inference in language model scaling laws](https://arxiv.org/abs/2401.00448) — Sardana et al., the lifetime-cost objective, the proof that it has no closed form, and the cost-per-FLOP extension.
- [Scaling laws for neural language models](https://arxiv.org/abs/2001.08361) — Kaplan et al., the source Sardana et al. cite for the $6N$ and $2N$ counts.
- [The Llama 3 herd of models](https://arxiv.org/abs/2407.21783) — the 405B model's 15.6T tokens and the over-training of the smaller models.
- [GQA: training generalized multi-query transformer models from multi-head checkpoints](https://arxiv.org/abs/2305.13245) — the KV-head knob.
- [Stanford CS336, Spring 2025](https://cs336.stanford.edu/spring2025/) — Lecture 9 and Lecture 11, both titled "Scaling laws", cover the same fits from the training side.
