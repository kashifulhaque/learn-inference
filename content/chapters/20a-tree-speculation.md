---
title: Tree speculation and speculation under load
slug: 20a-tree-speculation
part: "Part 6 — Scaling"
summary: Verify a tree of drafted tokens in one pass, keep the output distribution exact with several candidates per node, choose the tree and the draft length for a batch, and keep a recurrent state per branch.
minutes: 120
gpu: false
objectives:
  - Build a tree attention mask and position ids from a parent array, and check that one masked pass equals running every root-to-leaf path alone.
  - Verify several candidates per node with recursive rejection sampling, and prove the output distribution stays the target's.
  - Compute the expected accepted length of a tree from per-depth acceptance, and choose the best tree for a node budget.
  - Set the draft length and the node budget from the batch, using chapter 20's cost model.
  - Say what a gated-delta layer needs for a tree (a state per branch) and what that costs in bytes and time.
  - "Explain EAGLE's feature-level drafting, what EAGLE-3 changes, and how an MTP head differs from a Medusa head."
lab: 20a-tree-speculation
---

# Tree speculation and speculation under load

> [!TLDR]
> - A chain draft throws away everything after its first rejection. A tree tries several continuations in the same verification pass. A mask that lets each node see only its ancestors makes one pass equal to running every root-to-leaf path alone.
> - Several candidates per node stay exact if you draw them with replacement from the draft and test them in order, replacing the target after each rejection by the normalized positive part of $p - q$. The lab's total-variation check on the first two emitted tokens is 0.0100, against 0.4419 for the draft's own distribution.
> - Expected accepted length is the sum, over the tree's non-root nodes, of the product of the acceptance along each node's path. Picking nodes greedily by that product gives the best tree for a budget, and the lab checks it against every tree up to 8 nodes.
> - A verify pass carries $B(N+1)$ tokens. The ridge is 161.2, so the right draft length falls as the batch grows: 8 at batch 16, 4 at batch 32, 1 at batch 96, 0 at batch 160 for $\alpha = 0.8$.
> - A gated-delta layer cannot truncate a branch, so a tree needs a recurrent state per branch. At 17 nodes that is 2.453 GiB per sequence and 1.361 ms of writes, which cuts the useful node budget at batch 8 from 19 to 3.

[Chapter 20](/c/20-speculative-decoding) derives speculation for a chain: draft $\gamma$ tokens, verify them in one pass, keep the prefix the target accepts. It proves the output is exact, prices the draft, and finds the batch where speculation stops paying.

Production systems do more than that. They draft a tree, they draft from the target's own hidden states, and they turn speculation down when the batch is large. This chapter covers those three, and it keeps the arithmetic you can check. The lab runs on the CPU, with a toy attention head and toy distributions.

Every cost uses the pair chapter 20's lab uses: the rated 1935 GB/s and 312 TFLOP/s, with 53.8 GB of weights and 26.9 billion active parameters. That gives a plain decode step of 27.80 ms and a ridge of 161.2 tokens. Every number here is computed by the lab solution. None is an A100 measurement, and the acceptance rates in the tree examples come from synthetic distributions, not from a trained draft.

## Before you start

**The chain rule, from [chapter 20](/c/20-speculative-decoding).** Draft $\gamma$ tokens from $q$. Accept the draft token $x$ with probability $\min(1, p(x)/q(x))$. On a rejection, sample from the normalized positive part of $p - q$, then stop. A chain with acceptance $\alpha$ yields $(1 - \alpha^{\gamma+1})/(1 - \alpha)$ tokens per round, and its net speedup divides that by $1 + \gamma c$, where $c$ is the cost of a draft pass in target passes.

**The pass time, from chapter 20 and [chapter 17a](/c/17a-performance-model).** A pass over $B$ sequences with $T$ tokens each takes the longer of the weight read and the arithmetic: $\max(27.80\ \text{ms},\ 0.1724\,BT\ \text{ms})$. The two terms meet at $BT = 161.2$. Below that, extra tokens ride on idle bandwidth.

**The recurrent state, from [chapter 6](/c/06-gated-delta-rule) and [chapter 9](/c/09-the-kv-cache).** The 48 gated-delta layers keep 154,927,104 bytes of state per sequence, 147.8 MiB, and a step overwrites it in place. The 16 full-attention layers append keys and values, so a length change undoes their work.

**Rotary positions, from [chapter 5](/c/05-rotary-embeddings).** A key's rotation depends on its position id. Two keys at the same position get the same rotation, whatever their index in the array.

## Why a chain wastes its draft

This section asks what a tree buys that a longer chain does not.

A chain commits to one guess per position. If the draft's top token at position 1 is wrong, the whole rest of the chain was drafted from a context that never happened, and the round yields one token. The draft often has a decent second choice that a chain never tries.

A tree spends the same verification pass on both. Two candidates at position 1, each with its own continuation, is a tree of two branches. The pass costs the tokens it carries, so the price of the second branch is its nodes.

> [!EXAMPLE] A seven-node tree
> The parent array $[-1, 0, 0, 1, 1, 2, 3]$ describes a root (the last committed token, node 0), two candidates for the next token (nodes 1 and 2), two continuations of node 1 (nodes 3 and 4), one of node 2 (node 5), and one of node 3 (node 6). It has three root-to-leaf paths, $[0,1,4]$, $[0,2,5]$, and $[0,1,3,6]$. As a chain, the same seven tokens would be one path of length 7. The tree keeps all three and verifies them in one pass over 7 tokens.

Running the three paths as separate sequences would carry 10 tokens, because the root and node 1 repeat. The tree carries each shared prefix once. Making one pass behave like three separate runs is the job of the mask.

## The tree mask and position ids

This section builds the two arrays that let one forward pass verify a tree.

Node $i$ may attend to the whole committed context, and to itself and its ancestors in the tree. It must not attend to a sibling, a cousin, or a descendant, because in the path it represents those tokens do not exist. Write the tree as the parent array $\pi$ with $\pi_0 = -1$ and $\pi_i < i$. The mask is:

$$
\boxed{M_{ij} = \begin{cases} 1 & \text{if } j \text{ is } i \text{ or an ancestor of } i \\ 0 & \text{otherwise} \end{cases}}
\qquad
\boxed{\text{pos}_i = \hla{L} + \operatorname{depth}(i)}
$$

$\hla{L}$ is the committed length. Depth, not index, sets the position, because in node $i$'s path it is token number $\operatorname{depth}(i)$ after the context. Two siblings share a position, and so do nodes 3, 4, and 5 in the example.

> [!EXAMPLE] The seven-node mask and positions
> Row $i$ lists the columns it may attend to, beyond the context: row 0 is $\{0\}$, row 1 is $\{0,1\}$, row 2 is $\{0,2\}$, row 3 is $\{0,1,3\}$, row 4 is $\{0,1,4\}$, row 5 is $\{0,2,5\}$, and row 6 is $\{0,1,3,6\}$. That is 18 of 49 entries. With 7 committed tokens, the positions are $[7, 8, 8, 9, 9, 9, 10]$. A lower-triangular mask would set 28 entries, and it would let node 2 see node 1, a sibling.

The attention itself is the one you already have, with the mask applied to the tree block and nothing masked in the context block:

$$
\text{out}_i = \operatorname{softmax}\!\left(\frac{\operatorname{rope}(q_i,\ \text{pos}_i)\,\big[K_{\text{ctx}};\ \operatorname{rope}(K,\ \text{pos})\big]^{\top}}{\sqrt{d}} + \hlb{\log M}\right)[V_{\text{ctx}};\ V]
$$

$\hlb{\log M}$ is 0 where the mask is 1 and $-\infty$ where it is 0. The test that matters is an equivalence. Take each root-to-leaf path alone, give it the positions $L, L+1, \ldots$, and run ordinary causal attention over the context and the path. Every node's output must match the tree pass, because each node is on some path and sees the same keys at the same rotations. The lab checks this on two trees, with an error under $10^{-5}$.

> [!KEY] The mask and the positions fail separately
> A mask from the lower triangle leaves positions right and attention wrong. Positions from the array index leave the mask right and the rotations wrong. Both pass a shape check and both produce plausible text, which is why the lab compares against the path-by-path run.

For the 16 full-attention layers, the tree's keys and values land in scratch slots. After verification, the engine copies the accepted path's entries into consecutive slots after the context, advances the length by the accepted count, and frees the rest. For a chain, that copy is a no-op, which is why chapter 20 only truncates.

## Verifying several candidates and staying exact

This section extends chapter 20's proof from one candidate per position to $m$ candidates per node.

At a node, the target gives $p$ and the draft gave $q$. The draft proposes $m$ candidates $x_1, \ldots, x_m$, each drawn independently from $q$, with replacement. The target tests them in order. SpecInfer calls the whole procedure multi-step speculative sampling, and it picks the next child at random from those left. Independent draws are exchangeable, so index order is equivalent. In this chapter's terms:

1. Set $p_1 = p$.
2. For $k = 1, \ldots, m$: accept $x_k$ with probability $\min(1, p_k(x_k)/q(x_k))$. If you accept, emit $x_k$ and stop.
3. If you reject, set $p_{k+1}$ to the normalized positive part of $p_k - q$, and go on. $q$ does not change.
4. If every candidate is rejected, emit a sample from $p_{m+1}$.

The claim is that the emitted token is distributed as $p$.

> [!DEEPDIVE] Proof by induction on the candidates left
> Let $C_j$ be the claim: processing candidates $j, \ldots, m$ against the current target $p_j$ emits a sample from $p_j$.
>
> $C_{m+1}$ holds by construction: with no candidates left, step 4 samples from $p_{m+1}$.
>
> Assume $C_{j+1}$. Candidate $j$ is drawn from $q$, so token $x$ is emitted by acceptance with probability $q(x)\min(1, p_j(x)/q(x)) = \min(q(x), p_j(x))$. Write $\beta_j = \sum_x \min(q(x), p_j(x))$ for the total. The candidate is rejected with probability $1 - \beta_j$, and then step 3 moves to $p_{j+1} = (p_j - q)_+ / (1 - \beta_j)$, because $\sum_x (p_j - q)_+(x) = 1 - \beta_j$. By $C_{j+1}$, a rejection emits a sample from $p_{j+1}$. So:
>
> $$
> \Pr[\text{emit } x] = \min(q(x), p_j(x)) + (1 - \beta_j)\,\frac{(p_j(x) - q(x))_+}{1 - \beta_j} = p_j(x)
> $$
>
> because $\min(q, p) + (p - q)_+ = p$. That is $C_j$, and $C_1$ is the claim. The step uses that candidate $j$ is drawn from the same $q$, independently of the earlier rejections. That is why you draw with replacement and why $q$ stays fixed.

The proof needs the candidates to be samples. A tree built from the draft's top-$k$ tokens is not drawn from $q$, and the proof does not cover it. At temperature 0 the target's choice is its argmax, and any tree is exact by matching tokens. Methods that draft top-$k$ candidates at a temperature above 0 need a different proposal analysis. This chapter proves only the with-replacement rule.

Duplicates are legal. If $x_1 = x_2$, the second copy faces the residual $p_2$, not $p$, and the proof holds. Dropping a duplicate before verification would change the rule, and it would break the proof.

> [!EXAMPLE] What a second and third candidate buy
> Take $p = (0.5, 0.3, 0.2)$ and $q = (0.2, 0.3, 0.5)$. The first candidate is accepted with total probability $\beta_1 = 0.2 + 0.3 + 0.2 = 0.7$. A rejection moves to the residual $(0.3, 0, 0)/0.3 = (1, 0, 0)$, against which the second candidate is accepted with probability $\beta_2 = 0.2$. After that rejection the residual is $(1, 0, 0)$ again, and $\beta_3 = 0.2$. The chance that candidate $k$ is the accepted one is $a_k = \beta_k \prod_{j<k}(1 - \beta_j)$:
>
> $$
> a = (0.7,\ 0.3 \times 0.2,\ 0.3 \times 0.8 \times 0.2) = (0.7,\ 0.06,\ 0.048)
> $$
>
> Three candidates accept with probability $0.808$, against $0.7$ for one. The tree pays two more nodes for 0.108.

Walking a whole tree is one loop. Start at the root. Collect the node's children, run the rule over their tokens with that node's $q$ and $p$, emit the result, and move to the accepted child. At a leaf, or after a rejection, emit the last token and stop. A round emits the accepted path plus one token.

The lab's statistical check runs the walk on a toy Markov language model, where the target and the draft are different next-token tables over 5 tokens. Each round draws a tree of 3 children at the root and 2 under each, verifies it, and continues from the last emitted token until two tokens exist. Over 30,000 streams, the empirical joint distribution of the first two tokens is 0.0100 from the target's joint in total variation distance. It is 0.4419 from what the draft alone would give.

## Expected accepted length and the best tree

This section turns a tree into one number and picks the best tree for a budget.

Let $r_{d,k}$ be the probability that the $k$-th child at depth $d$ is the accepted one, given that its parent was accepted. The siblings are mutually exclusive, so $\sum_k r_{d,k} \le 1$. The previous section's $a_k$ values are exactly these numbers. A node is on the accepted path if every edge from the root to it was the accepted one. Its path probability is the product:

$$
P(v) = \prod_{e \in \text{path}(v)} r_{d(e),\,k(e)}
\qquad
\boxed{\mathbb{E}[\text{accepted drafts}] = \sum_{v \ne \text{root}} \hla{P(v)}}
$$

The expectation adds up because accepting is "this node is on the path," and the expected count of nodes on the path is the sum of their probabilities. The round yields that plus one token, the bonus or correction.

> [!EXAMPLE] A five-node tree by hand
> Take $r_1 = (0.6, 0.3)$ and $r_2 = (0.5, 0.2)$, and the tree $[-1, 0, 0, 1, 1]$: two children of the root, then two children of node 1. The path probabilities are $0.6$ and $0.3$ for nodes 1 and 2, then $0.6 \times 0.5 = 0.3$ and $0.6 \times 0.2 = 0.12$ for nodes 3 and 4. The sum is $1.32$ accepted drafts, so $2.32$ tokens per round.

### Choose the nodes greedily

You pick $N$ nodes to maximize that sum. Two facts make a greedy pick optimal:

- The objective is additive over nodes.
- A node's value is its parent's value times a number at most 1, and a node's value is at most its elder sibling's when $r_{d,k}$ falls with $k$. So the $N$ most valuable nodes always form a connected tree, with parents and elder siblings included.

So you keep a heap of candidates. Start with the root's first child. Each time you take a node, push its first child and its next sibling, each with its path probability. Take $N$ nodes. The lab checks the result against every ordered tree up to 8 nodes on three acceptance tables, to $10^{-9}$.

The numbers below come from a made-up pair: a Zipf target over 32 tokens and a draft whose logits carry more noise at each of six depths. They are the closed-form $a_k$ of four candidates at each depth, computed, not measured. Where the draft is wrong the first candidate is accepted with probability 0.395 to 0.795, and the second with 0.048 to 0.149.

| Node budget | Best tree | Chain | Tokens per round (tree) | Depth of tree |
|---|---|---|---|---|
| 1 | 0.795 | 0.795 | 1.795 | 1 |
| 4 | 1.775 | 1.775 | 2.775 | 4 |
| 6 | 1.899 | 1.867 | 2.899 | 5 |
| 8 | 1.996 | 1.867 | 2.996 | 5 |
| 12 | 2.146 | 1.867 | 3.146 | 6 |
| 16 | 2.258 | 1.867 | 3.258 | 6 |

The first column is the draft nodes, and the next two are expected accepted drafts. The chain stops at the table's six levels, so it cannot use more than 6 nodes. Up to four nodes the best tree is the chain, because deepening still beats widening. After that the greedy pick starts to branch.

```viz
20a-best-tree
```

The gain is modest here: 0.13 extra accepted tokens at 8 nodes, for 2 more nodes than the chain's 6. How much a tree helps depends on the table. When the second candidate is nearly as good as the first, as in the lab's second table where $r_1 = (0.5, 0.45, 0.04)$, a 3-node tree expects 1.05 accepted drafts and a 3-node chain expects 0.62. EAGLE-2 builds this table per context: it uses the draft model's confidence as an estimate of acceptance and grows a different tree for every context.

## Speculation under load

This section sets the draft length and the node budget from the batch.

A verify pass at batch $B$ with $N + 1$ tokens per sequence carries $B(N+1)$ tokens. Below the ridge of 161.2, extra tokens are free. Above it, each one costs $0.1724$ ms. Chapter 20 gives the closed form at batch 1, where the continuous optimum solves $\alpha^{\gamma+1}(L + c + Lc\gamma) = c$ with $L = \ln(1/\alpha)$. At $c = 0.05$ that gives $\gamma^{\ast} = 3.09,\ 5.48,\ 7.86,\ 13.29$ for $\alpha = 0.5,\ 0.7,\ 0.8,\ 0.9$, and the best integer draft length is $3,\ 6,\ 8,\ 13$. The lab's `adaptive_gamma` reproduces those at batch 1 and checks each against the closed form.

At a larger batch, you pick the $\gamma$ that maximizes the speedup of the cost model. It is the model of chapter 20's lab, one draft pass per drafted token and a verify pass from `step_time`:

$$
\operatorname{speedup}(B, \gamma) = \frac{\mathbb{E}[N+1]\ \cdot\ t(B, 1)}{\gamma c\ t(B, 1) + t(B, \gamma + 1)}
$$

where $t(B, T) = \max(27.80\ \text{ms},\ 0.1724\,BT\ \text{ms})$. Setting $\gamma = 0$ gives 1, plain decoding, so the best choice can be to turn speculation off.

| Batch | Best $\gamma$ | Speedup | Speedup at fixed $\gamma = 4$ |
|---|---|---|---|
| 1 | 8 | 3.092 | 2.801 |
| 16 | 8 | 3.092 | 2.801 |
| 32 | 4 | 2.801 | 2.801 |
| 64 | 2 | 1.890 | 1.539 |
| 96 | 1 | 1.451 | 1.058 |
| 128 | 1 | 1.099 | 0.806 |
| 160 | 0 | 1.000 | 0.651 |

The table uses $\alpha = 0.8$ and $c = 0.05$. A fixed $\gamma = 4$ keeps a speedup of at least 1 up to batch 101, which is chapter 20's break-even. Adapting the length keeps it to batch 141. Past that, plain decoding wins, and the right action is to drop speculation.

The batch where $\gamma$ starts to fall follows the ridge. At batch 32, $\gamma = 4$ puts $32 \times 5 = 160$ tokens in the pass, one under the ridge, so the best $\gamma$ drops from 8 to 4 between batches 16 and 32.

### The node budget plays the same role

For a tree, the lever is the node budget $N$. The verify pass carries $B(N+1)$ tokens, and the draft runs one pass per tree level, at cost $c$ each. The lab's `tree_speedup` uses that. With the table from the previous section and $c = 0.05$, the budget with the highest speedup is:

| Batch | 1 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| Best node budget | 32 | 32 | 19 | 9 | 4 | 2 | 1 |

A tree at batch 1 can use a large budget, because verification is free. At batch 32 the same table gives 4 nodes, which is a chain. Trees are a low-load technique.

Production systems close this loop with measurement. SmartSpec, which the arXiv page for 2406.14066 now titles TurboSpec, picks the speculation length that maximizes goodput, the useful tokens per second, from profiling and feedback rather than from a closed-form model. The vLLM documentation lists speculation as aimed at medium-to-low request rates and memory-bound workloads, which is the same observation. The lab's rule is the cost-model version of both.

> [!WARNING] A fixed length is a bet on the load
> A draft length tuned at batch 1 loses to plain decoding past batch 101 in this model. If the scheduler does not know the batch, it tunes for the wrong one. Pass the running batch into the choice, or turn speculation off above the break-even.

## What the hybrid needs for a tree

This section asks what the 48 gated-delta layers do with a tree.

For the 16 full-attention layers the tree is fine, as above. A gated-delta layer is a recurrence, and a tree has no single order. Node 4 follows node 1, not node 3, so a state advanced in index order would absorb node 3's token before node 4 reads it. That is the lab's mutant: one shared state, advanced over the nodes in index order, is off by 0.668 at node 2.

Each node's state must start from its parent's state. Chapter 20 rolls back a chain by snapshot, or by treating the chain as one chunk. A tree offers three designs, and all three are derivations from this chapter's arithmetic, not a description of a particular engine:

- **A state per node.** Node $i$ starts from its parent's state and writes its own. After verification, you keep the state of the last accepted node and drop the rest. Nothing is replayed. The cost is memory and writes.
- **One starting state and flattened paths.** Run each root-to-leaf path as its own chunk from the committed state, as chapter 20's chunked form allows, and apply the update only for the accepted path. Nothing is stored per node, but shared prefixes run once per leaf. The seven-node example runs 10 tokens instead of 7. A 16-node best tree runs 37 instead of 17.
- **Replay.** Verify without keeping states, then rerun the accepted path from the saved starting state. This costs one more pass over the delta layers.

The first design has a price you can compute. One state is 154,927,104 bytes. A tree of 17 nodes, root included, holds $17 \times 154{,}927{,}104 = 2{,}633{,}760{,}768$ bytes, 2.453 GiB per sequence. Writing them at 1935 GB/s takes 1.361 ms per sequence per round, 0.0801 ms per node. At batch 32 that is 43.6 ms, more than the whole 27.80 ms step. Chapter 20's single snapshot, read and written, costs 0.160 ms.

Adding the write time to `tree_speedup` moves the budgets:

| Batch | 1 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| Budget, no state writes | 32 | 32 | 19 | 9 | 4 | 2 | 1 |
| Budget, one state per node | 32 | 10 | 3 | 3 | 3 | 1 | 0 |

At batch 32, the 8-node tree's speedup falls from 1.471 to 1.046. The flattened design avoids that traffic, but its extra tokens count against the ridge: at batch 32, 37 tokens per sequence is 1,184 tokens in the pass, 7 times the ridge. Which design wins depends on the batch and the tree. The lab's check is the part all three share: the state of every node equals replaying its own path from the committed state.

## Where the draft comes from

The tree needs a $q$ at every node. Four designs produce it. This section covers them in prose, and the lab does not implement them.

**Medusa** puts extra heads on the target's last hidden state, each predicting a position further ahead. The heads do not see each other's choices, so their candidates form a tree, which the tree attention above verifies in one pass. The paper reports about 2.2 times for Medusa-1, which freezes the backbone, and 2.3 to 3.6 times for Medusa-2, which trains the backbone with the heads. As chapter 20 notes, its published acceptance is not the exact rule.

**EAGLE** drafts at the feature level. It autoregresses over the target's second-to-top-layer features, then maps a feature to a token with the target's own output layer. Feature prediction has uncertainty, because the same prefix can lead to different next features depending on the token sampled. EAGLE removes much of that by feeding the draft the token sequence advanced by one step, so it knows which token was sampled. The paper reports 2.7 to 3.5 times on LLaMA2-Chat 70B with the output distribution unchanged.

**EAGLE-2** keeps the EAGLE draft and makes the tree dynamic. A fixed tree ignores that some contexts are easy and some are hard. EAGLE-2 reads the draft's confidence as an estimate of acceptance and grows the tree where the estimate is high, which is the greedy pick in this chapter with a per-context table. It reports 3.05 to 4.26 times.

**EAGLE-3** changes two things. It drops the feature-prediction objective and predicts tokens directly, and it fuses features from several layers of the target, low, middle, and high, instead of only the top. Its training-time test trains the draft on its own multi-step outputs, so that the draft sees at training time what it sees when it drafts several steps in a row. The paper reports up to 6.5 times, about 1.4 times EAGLE-2, and a 1.38 times throughput gain at batch size 64 in SGLang. That last number is a measurement from the paper, and this course made none like it.

**MTP heads** are the multi-token prediction modules DeepSeek-V3 trains. Section 2.2 of the report describes sequential modules, each with its own transformer block, that share the embedding and the output head. Unlike Medusa's independent heads, each module keeps the causal chain at its depth, and the report sets the depth to 1. It says the modules can be repurposed for speculative decoding, and reports an 85% to 90% acceptance rate on the second token and 1.8 times the tokens per second. Chapter 20's config reading applies to this model: `mtp_num_hidden_layers` is 1, one layer out of 64, so $c \approx 0.016$. For a tree of depth $D$ with a head of that cost, the draft is $D$ passes, which is what `tree_speedup` charges.

## What goes wrong

**A lower-triangular mask.** It looks like causal attention and it is wrong for a tree. Node 2 sees node 1, so its output depends on a token that is not on its path. The path-by-path check fails it.

**Position ids from the array index.** The mask is right and every rotation is off for any node whose depth differs from its index. Use $L + \operatorname{depth}$.

**No residual update between siblings.** Testing each candidate against the original $p$ is the natural reading of "try several." It changes the output distribution. The lab's total-variation check catches it.

**Top-$k$ candidates under the with-replacement proof.** The exactness proof needs candidates drawn from $q$. Taking the draft's top tokens at a temperature above 0 is a different rule, and this chapter's proof does not cover it.

**Ranking nodes by edge acceptance.** A node deep in a chain has a high edge probability and a tiny path probability. Rank by the path product, or the best tree is not best.

**One recurrent state for the whole tree.** The output looks fine for a few tokens, because the state error is small at first. Give each node its parent's state.

**Leaving speculation on at high load.** At batch 128 with $\gamma = 4$ the speedup is 0.806. The scheduler has to choose a length from the batch, or turn speculation off.

**Forgetting to compact the KV path.** After verification the accepted path's keys and values sit in scratch slots. Copy them into consecutive slots, or the next token attends to the wrong keys.

> [!RECAP]
> - A tree mask lets node $i$ see the context, itself, and its ancestors. Position ids are the committed length plus depth. One masked pass then equals running every root-to-leaf path alone.
> - With candidates drawn with replacement from $q$, testing them in order and replacing the target by the normalized positive part of $p - q$ after each rejection keeps the output exactly $p$. The proof is an induction on the candidates left.
> - Expected accepted length is the sum of path probabilities over the non-root nodes. The greedy pick by path probability is the best tree for a budget.
> - A verify pass carries $B(N+1)$ tokens. The draft length and the node budget fall as the batch rises, and past the break-even they reach zero.
> - A gated-delta layer cannot truncate a branch. A tree needs a state per branch, a flattened path, or a replay, and each has a cost you can compute.
> - EAGLE drafts at the feature level, EAGLE-2 grows the tree per context, EAGLE-3 fuses several layers and trains on its own multi-step output, and an MTP module keeps the causal chain that Medusa's heads drop.

## Check your understanding

> [!QUESTION] Why does the tree mask give nodes 3, 4, and 5 in the example the same position id?
> They all have depth 2, and a position id counts tokens along a path, not slots in the array. In each of their paths the token is the second one after the context. Rotating by the array index would give them three different rotations, and none would match the path-by-path run.

> [!QUESTION] You drop duplicate candidates before verification to save nodes. What breaks?
> The rule stops being the one you proved. The proof draws $m$ candidates independently from $q$ and tests each against the current residual. A duplicate is a second independent draw, and removing it changes the acceptance probabilities from $a_k$ to something else. The output is then no longer $p$.

> [!QUESTION] The best tree for 4 nodes is a chain, and for 8 nodes it branches. Why?
> A node's value is its parent's value times an edge probability. In the example table, the chain's nodes are worth 0.795, 0.523, 0.329, and 0.130, and the best sibling, a second child of node 1, is worth $0.795 \times 0.08 = 0.064$. The fifth chain node is worth 0.060, below that sibling. So the greedy pick takes the sibling at the fifth node, and the tree starts to widen.

> [!QUESTION] At batch 32 with $\gamma = 4$, the verify pass carries 160 tokens and speculation still gains 2.8 times. At batch 64 it gains 1.54 times. What changed?
> At batch 64 the pass carries $64 \times 5 = 320$ tokens, twice the ridge of 161.2. The pass now takes $0.1724 \times 320 = 55.2$ ms, not 27.8 ms, so the extra tokens cost real time. The best $\gamma$ at batch 64 is 2, which gives 1.89 times, because it keeps the pass nearer the ridge.

> [!QUESTION] Why does the hybrid make trees costlier than they are on a plain transformer?
> The full-attention layers pay nothing extra, because their keys and values sit in scratch slots and a length change undoes them. The 48 gated-delta layers overwrite a 147.8 MiB state in place, and each branch needs its own copy of the state it starts from. A 17-node tree writes 2.453 GiB per sequence per round, 1.361 ms at the rated bandwidth, and that cost scales with the batch.

## Lab

> [!TRY]
> Build the tree mask and positions, tree attention that equals the path-by-path run, exact multi-candidate verification, the expected accepted length and best tree, the load-dependent draft length and node budget, and per-node recurrent state. Passing means all 10 checks hold, including a total-variation distance under 0.025 on the first two emitted tokens and a best tree that matches exhaustive search up to 8 nodes. The lab runs on the CPU in a few seconds.

The tree has one root, node 0, which holds the last committed token. `parents[i]` is node $i$'s parent, `parents[0]` is -1, and `parents[i] < i`. The starter gives you `rope`, `delta_step`, `tree_depths`, `ancestors`, `leaf_paths`, and the pass-time model. Implement the following functions:

- `tree_mask` and `tree_positions`, from a parent array and the committed length.
- `tree_attention`, one head over the context and the tree.
- `verify_node` and `verify_tree`, the recursive rule at one node and the walk down the tree.
- `branch_acceptance`, the closed-form $a_k$ for $m$ candidates.
- `path_probabilities`, `expected_accepted_length`, and `best_tree`.
- `chain_speedup`, `adaptive_gamma`, `tree_speedup`, and `adaptive_node_budget`.
- `tree_states`, `commit_by_replay`, and `snapshot_bytes`.

The checks go from the smallest piece to the whole. The first compares the mask and positions with the hand-built seven-node tree. The third compares tree attention with every root-to-leaf path run alone, on two trees. The fifth is the statistical test, and the eighth compares your best tree with every ordered tree up to 8 nodes on three tables. The ninth pins the draft length at batch 1 to chapter 20's closed form and the node budgets to the tables above. The tenth checks one state per node against replaying its path.

==The statistical check is the one to watch.== A rule that tests every sibling against the original target passes the arithmetic checks and fails it.

## Further reading

- [Fast inference from transformers via speculative decoding](https://arxiv.org/abs/2211.17192) and [Accelerating large language model decoding with speculative sampling](https://arxiv.org/abs/2302.01318) (arXiv 2211.17192 and 2302.01318), the chain rule this chapter extends.
- [SpecInfer: accelerating generative large language model serving with tree-based speculative inference and verification](https://arxiv.org/abs/2305.09781) (arXiv 2305.09781), the tree verification and the multi-step speculative sampling rule.
- [Medusa: simple LLM inference acceleration framework with multiple decoding heads](https://arxiv.org/abs/2401.10774) (arXiv 2401.10774).
- [EAGLE: speculative sampling requires rethinking feature uncertainty](https://arxiv.org/abs/2401.15077) (arXiv 2401.15077).
- [EAGLE-2: faster inference of language models with dynamic draft trees](https://arxiv.org/abs/2406.16858) (arXiv 2406.16858).
- [EAGLE-3: scaling up inference acceleration of large language models via training-time test](https://arxiv.org/abs/2503.01840) (arXiv 2503.01840).
- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) (arXiv 2412.19437), sections 2.2 and 5.4.3 for the MTP modules and their use in speculative decoding.
- [Optimizing speculative decoding for serving large language models using goodput](https://arxiv.org/abs/2406.14066) (arXiv 2406.14066), the goodput-based choice of speculation length.
- [vLLM speculative decoding documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/), the methods an engine supports and the load range it targets.
