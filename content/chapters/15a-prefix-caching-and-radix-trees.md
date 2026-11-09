---
title: Prefix caching and radix trees
slug: 15a-prefix-caching-and-radix-trees
part: "Part 4 — Kernels"
summary: A radix tree over token blocks, a hash that names each block by its parent, and where a hybrid model stores recurrent-state checkpoints.
minutes: 120
gpu: false
objectives:
  - Insert a token sequence into a radix tree, match the longest prefix, and split an edge where two sequences diverge.
  - Name a full block by a hash of its parent hash and its token ids, and check that name against the tree.
  - Keep a reference count, copy a partial tail instead of writing through it, and evict only unreferenced leaves.
  - Place recurrent-state checkpoints so a byte budget saves the most prefill tokens.
  - Compute the prefill time a given hit rate skips, from the 2NT count.
lab: 15a-prefix-cache
---

# Prefix caching and radix trees

> [!TLDR]
> - Two requests that share a prompt prefix can share the KV blocks of that prefix. The index is a radix tree: you insert a sequence, you match the longest prefix, and you split an edge where the next sequence diverges.
> - A full block's name is a hash of its parent's name and its own token ids. The same sixteen tokens under two prefixes are two blocks. Only a full block is a hit.
> - A block stays while its reference count is above zero. Eviction removes the least recently used leaf, and a partial tail is copied rather than overwritten.
> - The 48 linear layers keep one recurrent state of 154,927,104 bytes. A checkpoint of that state is reusable only at the token it was stored for, so the spacing is a budget.
> - Under the 2NT prefill count, the fraction of tensor-core time you skip equals the fraction of prompt tokens the cache hits.

[Chapter 15](/c/15-paged-attention) shows that two sequences can point at the same physical block, and that a 500-token system prompt is worth skipping on the second request. That chapter stops at one shared prefix and a copy when a sequence writes. A serving stack has more shapes than one prefix: a system prompt, then a tool description, then a document, then the turn that is unique.

The index for those shapes is a *radix tree* over token blocks. SGLang calls the technique RadixAttention: the runtime keeps finished and running KV in the tree, matches a new prompt from the root, and evicts the least recently used leaf whose reference count is zero. vLLM names the same blocks with a hash instead of a pointer. This chapter builds both, then asks where the hybrid model's recurrent state is allowed to sit.

The lab is CPU-only. Tokens are integers, and a block is a tuple of them. There is no weight file and no GPU.

## Before you start

**Pages, from [chapter 15](/c/15-paged-attention).** A block holds `block_size` tokens. This course uses 16. Two sequences share a block only when every token in it matches and every block before it matches. A tail shorter than a block is not shared, because the next token each sequence writes would land in the same slots.

**The recurrent state, from [chapter 2](/c/02-memory-arithmetic).** Each of the 48 linear layers keeps 3,227,648 bytes: a float32 delta-rule state and a short convolution window. Across the layers that is 154,927,104 bytes. Chapter 2 rounds that quotient to 147.8 MiB, and the exact value is 147.75 MiB.

[Chapter 6](/c/06-gated-delta-rule) is the update that overwrites this state in place. A snapshot from the end of a prompt is not a snapshot of a shorter prefix.

**Prefill FLOPs, from chapter 15.** A prefill of $L$ tokens on $N$ parameters is about $2NL$ FLOPs. $N$ here is the exact count from chapter 2, 26,895,979,520. The tensor-core peak $C$ is the A100's 312 TFLOP/s of bfloat16, from [chapter 10](/c/10-roofline).

[Chapter 17a](/c/17a-performance-model) prices a full prefill with a memory term and a launch floor. The times in this chapter are the $2NL$ term alone.

**The token budget, from [chapter 16](/c/16-continuous-batching).** A step admits prefill until a token budget is spent. Cached tokens are not new work. The lab's `schedule_prefill` is that rule on a list of prompts. It is not the scheduler from chapter 16, and this chapter does not edit that lab.

## A radix tree over token blocks

This section answers how you find the longest shared prefix without scanning every cached sequence.

A radix tree stores a sequence on the edges, not one token per edge. The edge label is a run of tokens, at most one block long in this chapter. The node at the end of an edge is the prefix those tokens spell, starting from the root. Two sequences that share a block share the edge. Two sequences that differ inside a block force that edge to split.

Take a block size of 4 and two prompts:

| Sequence | Tokens | Edges after both inserts |
|---|---|---|
| A | 1, 1, 1, 1, 5, 5, 5, 5 | `(1, 1, 1, 1)`, `(5, 5)`, `(5, 5)` |
| B | 1, 1, 1, 1, 5, 5, 9, 9 | `(1, 1, 1, 1)`, `(5, 5)`, `(9, 9)` |

Insert A first. The tree grows two edges, `(1, 1, 1, 1)` and `(5, 5, 5, 5)`. Insert B. The first edge matches. The second edge matches for two tokens and then diverges, so you split it:

1. Replace `(5, 5, 5, 5)` with a prefix edge `(5, 5)`.
2. Hang the old suffix `(5, 5)` off the new node, and hang B's suffix `(9, 9)` beside it.
3. Leave A's stored tokens as the concatenation of the new pieces. A still reads `1, 1, 1, 1, 5, 5, 5, 5`.

==The split is the copy.== You do not rewrite the old edge's tokens to B's tokens. A sequence that already holds the edge keeps a concatenation that equals what you inserted.

Matching walks from the root and consumes an edge only when the edge's tokens equal the next tokens of the prompt. The walk stops at the first miss. The hit length is the longest prefix of that walk whose token count is a multiple of the block size. In the example, B matches the first full block and then a two-token edge. The two-token edge is not a block, so B's hit is 4 tokens, not 6.

> [!KEY] A hit is a prefix of whole blocks
> Attention at position $p$ depends on every token before $p$. A cached block is reusable only when every block before it matched. A partial edge is stored so the tree can split, and it is not a hit.

That rule is why a system prompt padded to a multiple of 16 shares cleanly, which chapter 15 already recommends. The padding is not part of the tree. It is how you make the shared string land on a block boundary before you insert it.

SGLang's hit rate is the definition this chapter uses. For one request, or for a set of them:

$$
\text{hit rate} = \frac{\hla{\text{cached prompt tokens}}}{\text{prompt tokens}}
$$

The $\hla{\text{cached count}}$ is the sum of the page-aligned hits. It is not the number of requests that matched anything.

> [!EXAMPLE] One system prompt, 100 requests
> Each prompt is 32 shared tokens and 16 tokens of its own, so each prompt is 48 tokens and the batch is 4,800. The first request hits nothing. Each of the other 99 hits the 32-token prefix:
>
> $$
> \frac{99 \times 32}{100 \times 48} = \frac{3{,}168}{4{,}800} = 0.66
> $$
>
> The lab checks that fraction. `scripts/figures/15a-prefix-caching-and-radix-trees.py` prints the same ratio.

vLLM's cache is the same idea with a different index, which the next section builds. Preble keeps one radix tree per shared root on the scheduler, and it splits a node when a new prompt matches only a prefix of that node's label. The split in the example is that operation.

## Hash-chained block identity

This section answers how a block can be named without a pointer, and why the name has to include the parent.

A hash table does not store edges. It stores a map from a block's name to the block id. The name has to mean "these tokens, after this prefix."

vLLM hashes three things: the parent block's hash, the token ids in this block, and any extra key that must keep the block private, such as a LoRA id or a cache salt. This chapter uses the first two. Extra keys are out of scope, and the cache still stores full blocks only.

Number the full blocks of a prompt $t_0, t_1, \ldots$ and write $H$ for the hash. The first parent is a fixed stand-in, written $\mathrm{none}$ here:

$$
h_0 = H(\mathrm{none}, t_0), \qquad h_i = H(\hlb{h_{i-1}}, t_i)
$$

$\hlb{h_{i-1}}$ is the whole prefix, folded into one integer. Two blocks with the same token ids and different parents get different names. Mooncake uses the same shape in its CPU pool: a block's hash is fixed by the block and by the prefix in front of it.

The preimage in the lab is ASCII, so the hash is stable across processes. Python's built-in `hash` is not. It is salted per process. The preimage is the word `none`, or the decimal parent, then `|`, then the token ids joined by commas. The name is the first 8 bytes of SHA-256, read as an unsigned integer.

The following function is that name:

```python
def block_hash(parent: int | None, tokens: tuple[int, ...]) -> int:
    head = "none" if parent is None else str(int(parent))
    body = head + "|" + ",".join(str(int(token)) for token in tokens)
    digest = hashlib.sha256(body.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")
```

Walk the full blocks, feeding each hash back in as the next parent, and you have the chain. The tree and the chain describe one prefix. For two prompts, count the leading hashes that are equal, and multiply by the block size. That product is the hit length `match` returns.

> [!WARNING] A hash of the tokens alone collides
> The block `(9, 9, 9, 9)` is a common shape for padding and for repeated bytes. After the prefix `(1, 2, 3, 4)` and after `(8, 8, 8, 8)` it is not the same KV. A hash that drops $\hlb{h_{i-1}}$ returns one name for both, and the second request reads the first request's block.

The lab checks the collision directly. `block_hash(1, (9, 9, 9, 9))` and `block_hash(2, (9, 9, 9, 9))` differ, and each equals the hash of its own parent plus those tokens. The first check calls `block_hash` once, on the first block of a prompt, so that function stands on its own.

vLLM records the hash when the block fills, and clears it when the block is evicted. A partial tail has no hash. In the worked tree, B's second block is `(5, 5, 9, 9)`, which is not A's `(5, 5, 5, 5)`, so the chains share one hash and the hit is one block. The split that left a two-token edge does not create an extra hit. The page rule and the hash rule agree.

## Reference counts, copy-on-write, and leaf eviction

This section answers which blocks you can free, and which writes have to copy first.

A node carries a reference count: how many live sequences hold it. `insert` adds one along the path it stores. `release` subtracts one. `match` does not take a reference. It only reports the hit, and it refreshes the node's last-used time so a hit is not the next eviction.

SGLang's rule is the one the lab enforces. A node can be evicted when its reference count is zero. Among those, the victim is a leaf, and the leaf chosen is the least recently used.

An interior node still has a child. Freeing it would drop that child out of the tree, including a child some other sequence still matches. After the leaf is gone, the parent may itself be a leaf, and a later eviction can take it. Evicting leaves first keeps a shared ancestor until its children are gone.

Copy-on-write is the same discipline on a write. A partial tail is an edge shorter than a block. Two cases come up:

- A second sequence diverges inside the edge. Split, as the insert of B does. The first sequence's concatenation does not change.
- A second sequence continues past the tail with more tokens. Do not extend the edge's token tuple in place. Keep the short edge, and hang the new tokens on a new edge after it. The sequence that ended on the tail still materializes the short tail.

==A partial edge stays unchanged for a sequence that already holds it.== If you assign a longer tuple over that edge, the earlier sequence's tokens change, and the lab's `materialize` check fails.

Eviction has a capacity as well as an explicit `evict(n)`. When an insert would pass `capacity` edges, it evicts unreferenced leaves until it fits or until no such leaf remains. A sequence you have not released is not a source of free edges. The held block stays, and `evict` returns how many leaves it actually removed.

The free order in vLLM is the same preference, read from the tail. When a request finishes, its blocks join the free queue in reverse, last block first, because the last block hashes more tokens and is the least likely to be reused. The lab's LRU-among-leaves is the SGLang form of that preference. A block whose reference count is still above zero stays allocated.

> [!NOTE] Longest shared prefix first
> For one batch, SGLang states that walking the radix tree in depth-first order maximizes the hit rate when the cache can hold the longest request. Longest-shared-prefix-first is that walk. The lab does not reorder the queue. It only stops charging tokens the cache already holds.

## Where to checkpoint the recurrent state

This section answers where a hybrid model may store the 154,927,104-byte state, and what that storage costs against the tokens it saves.

The KV cache is append-only. Token $p$ is written once, and a prefix hit reuses every KV block up to the match. That is why the tree stores blocks at all.

The linear layers do not work that way. Each step overwrites one state. The state after token 2,048 is not the state after token 512, and you cannot roll it back. Marconi states the consequence for hybrid models: a recurrent state is reusable only on an exact match of the prefix it summarizes, so a cache that wants partial hits has to have stored a checkpoint at that prefix.

A checkpoint is one copy of the state, 154,927,104 bytes, at one token position. A later prompt of length $L$ uses the rightmost checkpoint $p$ with $p \le L$, and it skips $p$ tokens of prefill. If no checkpoint sits at or before $L$, it skips nothing. Write $P$ for the set of positions you stored:

$$
\mathrm{saved}(L) = \max\left(\{0\} \cup \{p \in P : p \le L\}\right)
$$

The total for a workload is the sum of $\mathrm{saved}(L)$ over the prompts. A position that is not itself a prompt length can move forward to the next length that still uses it, and the sum stays the same or grows. The search in the lab therefore tries subsets of the distinct lengths.

The byte budget buys a fixed number of snapshots:

$$
k = \left\lfloor \frac{\text{budget}}{\hlc{154{,}927{,}104}} \right\rfloor
$$

$\hlc{154{,}927{,}104}$ is one state, not one token. Checkpointing every 16-token boundary of a single 2,048-token prompt stores $2{,}048 / 16 = 128$ states:

$$
128 \times 154{,}927{,}104 = 19{,}830{,}669{,}312\ \text{bytes} = 18.46875\ \mathrm{GiB}
$$

That is one sequence. A full hit at position 2,048 needs one checkpoint, not 128. The extra 127 snapshots exist for prompts that match a shorter prefix. Marconi's traces, at a block size of 32, show how uneven that reuse is on their workloads: 25.0% of the attention blocks were reused, and 0.4% of the recurrent states were. The spacing has to earn its bytes.

> [!KEY] One checkpoint costs a whole recurrent state
> KV for 16 tokens is 1 MiB on this model, from chapter 15. The state beside it is 147.75 MiB. You can afford a KV block per page. You cannot afford a state per page.

The figure uses one hundred prompts: 40 of length 256, 30 of length 512, 20 of length 1,024, and 10 of length 2,048. For each $k$ from 0 to 4 the script picks the positions that maximize the sum, and on a tie it keeps the lexicographically smallest list. At $k = 1$ the positions 512 and 1,024 both save 30,720 tokens. The list `[512]` is the one the tie-break keeps.

| Checkpoints | Positions | Tokens saved | Mean per prompt | Bytes |
|---|---|---|---|---|
| 0 | none | 0 | 0 | 0 |
| 1 | 512 | 30,720 | 307.2 | 154,927,104 |
| 2 | 256, 1,024 | 48,640 | 486.4 | 309,854,208 |
| 3 | 256, 1,024, 2,048 | 58,880 | 588.8 | 464,781,312 |
| 4 | all four lengths | 66,560 | 665.6 | 619,708,416 |

```viz
15a-checkpoint-tradeoff
```

The second checkpoint adds 17,920 tokens. The fourth adds 7,680. The curve flattens because the short prompts are already covered, and a new position helps only the prompts at least that long. Four states cost 591 MiB, which is $4 \times 147.75$. The script prints each row.

> [!EXAMPLE] Why 512 beats 2,048 when you can store only one
> A checkpoint at 2,048 saves 2,048 tokens on each of the 10 long prompts, and nothing on the other 90:
>
> $$
> 10 \times 2{,}048 = 20{,}480
> $$
>
> A checkpoint at 512 saves 512 tokens on each of the 60 prompts that reach it:
>
> $$
> 60 \times 512 = 30{,}720
> $$
>
> The shorter position wins. It throws away the tail of the long prompts, and it covers six times as many requests.

The lab grades `plan_state_checkpoints` against a brute-force search on small multisets, including the tie of nine prompts of length 10 and one of length 100. One slot saves 100 tokens either at 10 or at 100. The lexicographically smaller list is `[10]`. A budget below 154,927,104 bytes buys no slot and returns an empty list.

## What a hit rate does to time to first token

This section answers how many milliseconds a hit rate skips, using the prefill count you already have.

Write $r$ for the hit rate, the fraction of prompt tokens served from the cache. Under the $2NL$ count, the skipped work is the same fraction of the prefill. The skipped tensor-core time is:

$$
t_{\mathrm{saved}} = \hla{r} \times \frac{2NL}{C} = \boxed{\hla{r}\,\frac{2NL}{C}}
$$

$\hla{r}$ is the hit rate, $N$ is the parameter count, $L$ is the prompt length, and $C$ is 312 TFLOP/s. The time you still pay is $(1 - \hla{r})$ of the same prefill. This is a count of skipped FLOPs, not a measured time to first token. It ignores the memory term and the launch floor that chapter 17a adds, and it ignores that full attention still reads the cached prefix. Chapter 15 uses the same count for the same reason: it prices the work you do not redo.

> [!EXAMPLE] A 500-token prompt at a hit rate of 0.75
> The exact parameter count gives:
>
> $$
> \frac{2 \times 26{,}895{,}979{,}520 \times 500}{312 \times 10^{12}} = 0.086205\ \mathrm{s} = 86.205\ \mathrm{ms}
> $$
>
> Chapter 15 rounds the same division, with 26.9 billion in place of the exact count, to 86 ms. Three quarters of 86.205 ms is 64.654 ms, which is 375 tokens left uncomputed. The remaining quarter is 21.551 ms.
>
> $$
> 0.75 \times 86.205\ \mathrm{ms} = 64.654\ \mathrm{ms}
> $$

The 100-request example has a hit rate of 0.66, so the same formula skips 0.66 of the $2NL$ time on that mix. The lab does not time it. It counts tokens.

`schedule_prefill` admits prompts in order and charges `len(prompt) - n_cached` against the budget. On three 48-token prompts with a 32-token shared prefix and a budget of 80, the first prompt pays 48, and each of the next two pays 16. The call admits all three, charges 80, and reports `prefill_tokens_saved` of 64. A scheduler that charges the full 48 on the second prompt stops after the first.

SGLang reports the latency side of this on a production trace, which is not this model and not this GPU. After a month in Chatbot Arena, RadixAttention's hit rate was 52.4% for LLaVA-Next-34B and 74.1% for Vicuna-33B, and the Vicuna deployment's average first-token latency fell by a factor of 1.7. The paper's throughput figure, up to 6.4× higher throughput than the systems they compare, is the whole runtime, not the tree alone. Treat those as their measurements. The number you can recompute here is the $2NL$ fraction.

## What goes wrong

**Hashing a block without its parent.** The tokens `(9, 9, 9, 9)` then share one name under every prefix. The second request reuses KV that belongs to the first. The check that passes with only `block_hash` still passes, because the first block has no parent. The next check, two parents and one token tuple, is the one that fails.

**Evicting an interior node.** The shared prefix disappears while a child edge is still in the tree. After one eviction the prefix hit drops from 4 tokens to 0, which is the lab's leaf check. A leaf whose reference count is above zero fails the same check from the other side: `evict` returns 0 and the held block still matches.

**Extending a partial tail in place.** The sequence that ended at `(2, 2)` now reads the tokens you appended for someone else. `materialize` on the earlier id returns the longer tuple. Split, or hang a new edge, and the earlier tuple stays.

**Charging cached tokens against the budget.** The second prompt pays 48 instead of 16. With a budget of 80 the scheduler admits one prompt instead of three, and `prefill_tokens_saved` stays 0.

**Checkpointing every block.** One 2,048-token sequence stores 18.46875 GiB of recurrent state and gains partial hits you may not see. One checkpoint at a length many prompts share saves more tokens per byte, which is the table's $k = 1$ row.

> [!RECAP]
> - The radix tree inserts a sequence, matches a page-aligned prefix, and splits an edge at the first divergence.
> - A block's name is a hash of its parent hash and its tokens. Full blocks only.
> - Evict an unreferenced leaf. Copy a partial tail instead of writing through it.
> - A recurrent checkpoint costs 154,927,104 bytes and saves tokens only for prompts at least that long.
> - Under the $2NL$ count, a hit rate $r$ skips a fraction $r$ of the tensor-core prefill. At 500 tokens and $r = 0.75$, that is 64.654 ms of the 86.205 ms.

## Check your understanding

> [!QUESTION] Why do two blocks with the same token ids need different hashes when their parents differ?
> Because the KV in a block depends on every token before it. The parent hash is that prefix, folded into the name. Without it, `(9, 9, 9, 9)` after two different prompts is one cache entry, and the second prompt reads the first prompt's keys and values.

> [!QUESTION] Eviction is allowed to free a node whose reference count is zero. Why does it still skip an interior node?
> The interior node has a child. Deleting it removes the path to that child, so a sequence that still matches the child loses the prefix in front of it. Free the leaf first. The interior node becomes a leaf when its last child is gone, and a later eviction can take it.

> [!QUESTION] The state after the last token of a 2,048-token prompt is 154,927,104 bytes. Why can't a 512-token prompt reuse it?
> The linear layers overwrite the state in place. The value at token 2,048 is not the value at token 512, and there is no per-token log to roll back. A 512-token prompt needs a checkpoint stored at a position at or before 512.

> [!QUESTION] A 500-token prefill is 86.205 ms of tensor-core time at the 2NT count. What does a hit rate of 0.75 skip?
> Three quarters of the prompt, 375 tokens, and three quarters of that time, 64.654 ms. The remaining prefill is 21.551 ms. The fraction equals the hit rate only under this count. Chapter 17a's floor adds memory and launches, and those terms do not all scale with the missed tokens.

## Lab

> [!TRY]
> Build the prefix cache and the checkpoint planner. You pass when `block_hash` mixes in the parent, 100 requests that share a 32-token prompt hit 3,168 of 4,800 tokens, eviction keeps every interior node and every block with a reference count above zero, and the checkpoint positions match a brute-force search on small lists.

The lab runs on the CPU. Implement the following:

- `block_hash(parent, tokens)`, the SHA-256 name from the hash section.
- `PrefixCache`, with `match`, `insert`, `release`, `evict`, and `materialize`. `match` returns the block ids and the page-aligned hit length. `insert` returns a sequence id and splits on divergence. `release` drops that sequence's references. `evict(n)` removes up to `n` unreferenced leaves and returns how many it removed.
- `plan_state_checkpoints(prompt_lens, budget_bytes)`, the positions that maximize saved prefill tokens. One snapshot costs 154,927,104 bytes. On a tie, return the lexicographically smallest list.
- `schedule_prefill(prompts, token_budget, cache)`, which charges only the tokens `match` missed. The metric is `prefill_tokens_saved`.

`match` on an empty cache returns no blocks and 0 cached tokens. A shared system prompt that is an exact multiple of the block size is a full hit for every request after the first. A partial tail stays private to the sequence that ends there.

## Further reading

- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104). RadixAttention keeps KV in a radix tree, counts a hit as cached prompt tokens over prompt tokens, and evicts the least recently used leaf whose reference count is zero. The Chatbot Arena figures in this chapter are from that paper.
- [Automatic Prefix Caching](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md). vLLM hashes the parent hash, the block's token ids, and any extra key, and it caches full blocks only.
- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079). The CPU pool names a block by a hash of the block and of the prefix in front of it.
- [Preble: Efficient Distributed Prompt Scheduling for LLM Serving](https://arxiv.org/abs/2407.00023). The global scheduler keeps radix trees and splits a node when a new prompt matches only part of the node's label.
- [Marconi: Prefix Caching for the Era of Hybrid LLMs](https://arxiv.org/abs/2411.19379). Recurrent state is updated in place, so a checkpoint is an exact-prefix match. The 25.0% and 0.4% reuse figures are Marconi's, at block size 32.
- [CS336: Language Modeling from Scratch, Lecture 10, Inference](https://cs336.stanford.edu/). Percy Liang's inference lecture on the course schedule.
