---
title: "Capstone: the engine, end to end"
slug: 21-capstone-the-engine
part: "Part 7 — Capstone"
summary: Wire the paged cache, the recurrent state, the scheduler, and the sampler into one engine, find the bugs that only appear when they run together, and score the result on a simulated clock.
minutes: 150
gpu: false
objectives:
  - Say which component owns each piece of per-request state, and when each piece is created and destroyed.
  - Run chunked prefill and preemption by recompute on a hybrid model with output identical to a no-cache forward pass.
  - Explain why a restart needs a fresh recurrent state and a fresh sampler generator, and detect a desynchronised pool with an audit.
  - Score an engine on a seeded trace with chapter 17a's step-time model, and write a design report from the result.
  - Run the same engine on a real checkpoint on an A100, and say what that run has and has not verified.
lab: 21-capstone
---

# Capstone: the engine, end to end

> [!TLDR]
> - Every earlier lab checks one part alone. An engine adds the glue: a slot for each request's recurrent state, a view that lets the model write into paged blocks, and a loop that keeps all of them in step.
> - Four bugs only show up when the parts run together: a chunk that cannot see its prefix, a restart that keeps the old recurrent state, a sampler whose draws depend on the batch, and a pool that leaks a block or a slot.
> - The graded core runs on the CPU, on a seeded tiny model with the same hybrid layout. The engine must match a no-cache forward pass token for token, under chunking and under 17 preemptions.
> - A simulated clock prices each step with [chapter 17a](/c/17a-performance-model)'s floor for the 27B model. The seeded 64-request trace scores 165.78 tokens/s of goodput, against 33.37 tokens/s one request at a time.
> - A gated check runs a real checkpoint against Hugging Face transformers on an A100. It has never run. This chapter says how to run it, and does not report a result.

You have built every part of this engine in isolation. [Chapter 8](/c/08-the-forward-pass) gives you a forward pass, and [chapter 9](/c/09-the-kv-cache) gives it a cache. [Chapter 15](/c/15-paged-attention) pages that cache, [chapter 16](/c/16-continuous-batching) schedules it, and [chapter 11](/c/11-sampling) turns logits into tokens. [Chapter 16a](/c/16a-the-serving-loop-and-api) wraps the loop in a stream.

Each of those labs passes against its own tests, and none of them runs next to the others. This chapter puts them in one loop and asks for one property: the output of the assembled engine does not depend on how the engine scheduled it. Batch the requests together, cut the prompts into chunks, or evict and recompute them, and every token stays the same.

That property fails in ways the isolated labs cannot see. The failures are not in any one component. They are in what one component assumes about another.

## Before you start

**The hybrid cache, from [chapter 9](/c/09-the-kv-cache).** The 16 full-attention layers keep K and V for every past token. The 48 linear layers keep one fixed-size state per sequence, plus a three-step conv window. The tiny model in this chapter has the same shape rule at a smaller size: 8 layers, with layers 3 and 7 full attention and the other six gated delta.

**Paged blocks, from [chapter 15](/c/15-paged-attention).** A sequence holds a list of fixed-size blocks, a block table. A new token needs a new block only when the last one is full.

**The scheduler, from [chapter 16](/c/16-continuous-batching).** Each step it picks decodes first, then prefill chunks within a token budget, and when memory runs out it preempts the newest request and recomputes it later. The scheduler owns the KV blocks. It knows nothing about recurrent state or the sampler.

**The step-time model, from [chapter 17a](/c/17a-performance-model).** One step takes the longer of the compute floor and the memory floor, plus a launch floor. At batch 1 that is 29.962 ms for this model on an A100, of which the weight read is 27.799 ms.

## How the pieces fit

This section answers which component owns what, because every bug in this chapter is a disagreement about ownership.

The engine in the lab is the repository's own `engine/` package plus three small classes. The package supplies the model, the paged KV cache, the scheduler, and the sampler. The lab adds the three things that sit between them:

| Piece | Comes from | Owns | Lives from, until |
|---|---|---|---|
| `PagedKVCache` and the block table | [chapters 9](/c/09-the-kv-cache) and [15](/c/15-paged-attention) | K and V for the full-attention layers | Admission, until finish or preemption |
| `StatePool` | this chapter | One recurrent state and conv window per sequence, for the linear layers | The same span |
| `SeqCache` | this chapter | The `HybridCache` interface for one sequence, over the two pools above | The same span |
| `Scheduler` | [chapter 16](/c/16-continuous-batching) | Which requests run, and how many tokens each | The whole run |
| Sampler and its generator | [chapter 11](/c/11-sampling) | One seeded random stream per request | The same span as the cache |
| `Engine` | this chapter | The loop that calls all of the above, once per step | The whole run |

The model's forward pass expects a cache object with a fixed interface: `append` for K and V, `recurrent_state`, `conv_state`, and `set_linear_state` for the linear layers, a `length` that is the next position, and `advance` once per pass. The repository's `HybridCache` implements that interface over contiguous tensors for a batch. `SeqCache` implements the same interface over blocks and a state slot for one sequence. The model does not change.

The tiny model forwards each scheduled item on its own, one sequence per pass. That is a simplification. A real engine fuses the whole step into one variable-length forward over every sequence, which is what the kernels in chapters 13 to 15 are for. The ownership rules do not change when you fuse, so the bugs below carry over.

> [!KEY] A state slot lives exactly as long as its request is running
> The block table, the recurrent-state slot, the sampler's generator, and the engine's view of the sequence are four copies of one fact: this request has run this many tokens. They are created together and destroyed together. Every bug in this chapter is one of them outliving, or being reset without, the others.

[Chapter 16a](/c/16a-the-serving-loop-and-api) already gave you the release rule for finish and cancel: blocks and the slot go back together. Preemption is a third way out of the running set, and it is the one that tests that rule hardest.

## One step, end to end

This section answers what the engine does between two calls to `step`.

`Engine.step` runs the same six stages as chapter 16a, with the stream stage left to the caller:

1. **Schedule.** Ask the scheduler for a batch: decode items first, then prefill chunks. It reserves blocks as it goes, and it may preempt.
2. **Reconcile.** Compare the engine's views with the running set. Free the slot and generator of any request that left it. Drop any decode item whose request was preempted after it joined the list.
3. **Forward.** For each item, build a `SeqCache`, run the model on that item's tokens, and take the logits of the last position.
4. **Sample.** Draw a token for each item that finishes a prefill or a decode, from that request's own generator.
5. **Commit.** Tell the scheduler: `commit_prefill` for a chunk, `commit_token` for a token.
6. **Report.** Return what ran, so a clock can price it.

Walk the first step of the seeded trace. Prefill chunks are at most 16 tokens, and the step budget is 48 tokens. Nothing is running, so the clock jumps to the first arrival, a request with a 9-token prompt. The scheduler reserves 3 blocks of 4 tokens and returns a batch with no decodes and 9 prefill tokens. The engine allocates a state slot, runs the chunk through the model, and commits it. The chunk reaches the end of the prompt, so the request samples its first token and joins the decode list on the next step. The simulated clock charges the step 29.978 ms, which is the weight read plus the launch floor and a sliver of prefill.

```viz
21-trace-clock
```

The figure replays all 94 steps of the trace. Step cost barely moves, from 29.8 to 31.3 ms across the whole trace, while the decode count rises from 0 to 8 and a step carries up to 45 prefill tokens. That flatness is chapter 17a's whole argument, and it is the reason batching pays.

## Chunk boundaries

This section answers what changes when a prompt arrives in pieces.

A prompt of 24 tokens with a chunk size of 5 runs as chunks of 5, 5, 5, 5, and 4. Each chunk after the first has a cached prefix, so its queries are fewer than its keys. Two components have to carry the right thing across each boundary, and they carry different things.

The linear layers carry a state. The chunked delta rule from [chapter 6](/c/06-gated-delta-rule) takes the previous state in and hands the new one out, and the conv layer does the same with its three-step window. If `SeqCache` stores those after each chunk and returns them at the next, chunking is exact. The state after chunk 2 must be the input to chunk 3, and not the state after chunk 1.

The attention layers carry a mask. A chunk's query $i$ sits at absolute position $s + i$, where $s$ is the length of the cached prefix. It may attend to every key up to that position, which is all $s$ cached keys plus the chunk's own keys up to $i$. In matrix terms the causal mask is aligned to the bottom-right corner of the score matrix.

The engine's `naive` attention builds that mask from `kv_len - q_len`, so it is right at every boundary. PyTorch's fused path, `attn_impl="sdpa"`, passes `is_causal=True`, which aligns the mask to the top-left corner. For a chunk with a prefix, query 0 can then see only key 0, and the prefix is hidden. With the engine's default of `naive`, 8 of 8 seeded prompts match a no-cache forward pass at chunk size 5. With `sdpa` forced on, 4 of 8 do.

> [!WARNING] A chunk that cannot see its prefix produces plausible text
> The wrong mask does not crash and does not produce garbage. The model sees a shorter context than it has, and its output is a fluent continuation of the wrong prompt. Only a comparison against a no-cache pass catches it, and only when a prompt is longer than one chunk. A test that uses prompts shorter than the chunk size passes with the bug in place.

This is the first reason a fixed-size chunk must not divide the prompts evenly in a test. A prompt that is an exact multiple of the chunk size never exercises a short final chunk.

## Preemption by recompute

This section answers what the engine must reset when the scheduler evicts a request and later runs it again.

When the block pool runs out, the scheduler releases the newest request's blocks, sets its `prefilled` count to 0, clears its output, and puts it back at the head of the waiting queue. Later it runs again from the start. That is recompute: no state is saved, and the prompt is prefilled again.

The scheduler only knows about blocks. A preempted request leaves behind three things that the scheduler did not touch:

- **A recurrent-state slot** that holds the state after the request's last token. If the restarted request keeps it, the six linear layers start from a state that has already read the whole prompt and part of the output. The attention layers are rebuilt from scratch and are right. The delta layers are wrong, and the logits drift.
- **A sampler generator** that has already been advanced by the draws it made before the eviction. If it is kept, the recomputed request draws different random numbers from the first run, and a seeded request returns a different answer depending on whether it was evicted.
- **A view of the sequence** whose length no longer matches the block table.

The rule is the one from the key idea: free all of them together, and give a request that starts prefill at position 0 a fresh, zeroed slot and a fresh generator. A reused slot is a trap on its own. The pool hands out a slot the previous owner has written in, so `alloc` must zero it. The lab's first check tests exactly that.

Two scheduler behaviours make this harder than it sounds, and the lab's engine handles both:

- **Preempted and re-admitted in one step.** The scheduler frees a victim's blocks, then finds room for it again within the same call. The victim is in the running set when your engine looks, so a check on "is it still running" sees nothing wrong, but its cache is gone. The engine detects the restart by the position: a request whose next chunk starts at 0 begins fresh.
- **A stale decode item.** The scheduler can pick as victim a request already on this step's decode list. The request is then in `batch.decode` with its output cleared. Running it would feed a token that does not exist. The engine skips any decode item that is no longer past its prefill.

> [!EXAMPLE] 16 blocks, 4 slots, 8 prompts
> The lab's preemption check uses block size 4, so 16 blocks hold 64 tokens, against 8 prompts of 6 to 30 tokens plus 10 generated each. A batch of 4, chunk size 5, and a 12-token step budget run to completion in 95 steps with 17 preemptions. All 8 outputs match the no-cache reference, and all 16 blocks and 4 slots come back.
>
> The same prompts with 14 blocks and a batch of 3 do not finish. The scheduler evicts the newest request, re-admits it at once, and evicts another to make room. In 300 steps it preempts 79 times and completes nothing. That is a preemption storm, and it is a scheduling problem, not an engine bug. The lab caps `generate` at 5,000 steps so that a storm fails with a message instead of running until the timeout.

## Leaks, and how to catch them

This section answers how you notice that a pool and a request disagree before the symptom shows up somewhere else.

A leak is silent. A block that is never returned shrinks the pool by one, and the engine keeps running with a little less memory until a preemption storm starts a few hundred requests later. A block returned twice is worse: two requests then write into it, and each reads the other's tokens.

The engine has an `audit` that returns every disagreement between the pools and the requests. The lab runs it after every step of the preemption check. It looks for five things:

- A block on the free list twice.
- A block that is both held and free.
- Held blocks plus free blocks not equal to the pool size.
- A state view that belongs to a request outside the running set.
- A block table whose length is not the state view's length. That is the desync that preemption can cause.

The pool itself refuses to hand one request two slots, so that case raises at `alloc`.

An audit does not find every bug, but it finds the class where the engine's own books disagree, and it finds it at the step that caused it. The end-of-run check is the same idea in one shot: after every request has finished, all blocks are free, none appears twice, all slots are free, and no sequence view or generator is left.

## Sampling, per request

This section answers what has to be true for a seeded sample to be reproducible when requests share a batch.

A sampler with one global random generator gives each request whatever numbers are next in the stream. Then a request's output depends on which requests share its steps and in what order. The same seed gives a different answer when the batch is different, which is the bug that is hardest to reproduce from a bug report.

The fix is one `torch.Generator` per request, seeded from that request's own seed, and dropped when the request is preempted so its recompute draws the same numbers again. With that rule the lab's check holds: four seeded requests give identical outputs run alone, run together with chunking, run twice, and run under 8 preemptions. A different seed changes the output.

Greedy decoding hides this bug completely, because argmax draws nothing. A suite that tests only greedy output cannot see a shared generator, so the lab tests both.

## Pricing the engine on a simulated clock

This section answers how you score an engine you cannot time on the hardware you care about.

The tiny model gives you the tokens. [Chapter 17a](/c/17a-performance-model) gives you the time: for each step the engine ran, the step cost is the floor for the 27B model on an A100. One step reads the weights once however many sequences ride in it, so a mixed step adds the decode batch's FLOPs and bytes, then each prefill chunk's FLOPs and its bytes beyond the weights. Here $F$ is the FLOPs, $B$ the bytes, $C$ the rated bfloat16 peak, $\beta$ the rated bandwidth, and $t_\text{launch}$ the launch floor:

$$
t_\text{step} = \max\!\left(\frac{\hla{F}}{C},\ \frac{\hlb{B}}{\beta}\right) + t_\text{launch}
$$

Read it as the longer of two clocks plus a fixed cost. The lab checks that a pure decode step equals chapter 17a's `decode_step_time` and a lone chunk equals its `prefill_time`, both to a nanosecond.

Putting both in one step is cheaper than running them as two passes. Sixteen decodes at context 24 and one 16-token chunk cost 32.601 ms fused, against 62.400 ms as two passes, because the second pass would read the weights again. That is why the scheduler mixes decodes and prefill chunks, and the simulated clock prices it that way.

The trace is 64 requests with seeded arrivals at 40 per second, prompts of 6 to 30 tokens, and 4 to 12 new tokens. The engine runs in 94 steps and 18 preemptions, finishes at 2.9075 s on the simulated clock, and returns every block and slot. Goodput counts the output tokens of requests that met both SLOs, here a first token within 1.0 s and 0.15 s per token after it. The median time to first token is 0.450 s, the maximum 0.941 s, and the maximum time per output token 0.134 s. Goodput is 165.78 tokens/s.

The floor in the test comes from the same model. One request at a time, the 64 requests take 14.444 s, which is 33.37 tokens/s. The lab requires twice that, 66.74 tokens/s, and the engine is at 4.97 times the serial rate. This is a correctness gate, not a performance claim: an engine that fails to batch, or that loses requests, falls under it. The numbers are 17a's floors on a CPU-run trace. They are not A100 measurements, and nobody has timed this engine on one.

> [!NOTE] Why the SLOs do not bind here
> With 482 output tokens spread over 64 requests, the 30 ms step and the 1.0 s TTFT limit leave the engine far inside both SLOs, so goodput equals throughput. A loaded engine at a higher arrival rate, or with a block budget small enough to thrash, is where the two separate. Lower `num_blocks` in the lab's trace check and watch the preemption count and the tail times move.

## Write the design report

This section answers how to turn the run into a document someone else can act on.

A design report says what you built, what the model predicted, what the run did, and where they disagree. Chapter 17a gives you the prediction half. Use this template and fill every row from a command you ran:

```markdown
# Engine design report

## Workload
Arrivals, prompt and output length distributions, seeds, SLOs (TTFT, TPOT).

## Configuration
Block size and count, state slots, max batch, token budget, chunk size, attention path.

## Floors (chapter 17a, 27B hybrid, one A100, rated bandwidth)
- Weight read alone:             __ ms
- Decode step at batch 1:        __ ms
- Decode step at your max batch: __ ms
- A prefill chunk, fused:        __ ms vs. as a separate pass: __ ms

## Result on the simulated clock
Steps, preemptions, makespan, median and max TTFT, max TPOT, goodput, and the
one-request-at-a-time floor. State that the clock prices a CPU-run trace.

## Correctness
Match against the no-cache reference (prompts, chunk sizes, block budgets).
Audit result. Pool state at the end. Seeded-sampling reproducibility.

## Where the model and the run disagree
The step you expected to cost X and the clock charged Y, and why.

## Not verified
Everything that did not run on the target hardware or with real weights.
```

The last two sections carry the weight. A report that lists only agreements is a summary, and one that names its unverified parts is a report. The checkpoint run below is a row in that last section until someone runs it.

## Run it on a real checkpoint

This section answers how to run the same engine on real weights, and what that run would show. It has not been run.

The course's GPU endpoint was paused when this chapter was written, so no check in this section has ever executed. Treat everything here as instructions.

**Which checkpoint.** The target model is `Qwen/Qwen3.8-27B`. In bfloat16 its weights alone are 53.8 GB, and the lab's gated check compares in float32, which needs about 4 bytes per parameter. A search of the Hugging Face hub found no small Qwen3.8 checkpoint. It did find `Qwen/Qwen3.5-0.8B`, the small member of the Qwen3.5 family that the engine's code is written for. Its configuration has the same layout at a smaller size: 24 layers where every fourth is full attention (6 full, 18 linear), 8 query heads and 2 KV heads of width 256, a partial rotary factor of 0.25, a vocabulary of 248,320, and tied embeddings. The lab's check defaults to that id through the `MODEL_ID` environment variable.

**Run it.** On a machine with an A100, put the checkpoint where `engine.weights.local_model_path` looks, then run the lab file:

```bash
hf download Qwen/Qwen3.5-0.8B
MODEL_ID=Qwen/Qwen3.5-0.8B python3 scripts/check_labs.py 21-capstone
```

With CUDA and the checkpoint present, the lab adds a tenth check. It builds the engine's model at float32, copies the weights in through `load_hf_weights`, generates 8 tokens greedily for three prompts, and compares them with `transformers` on the same prompts. Without CUDA or without the checkpoint it records `real_weights_check` as skipped and adds no check.

**What the loader has to do.** The engine's layers do not share a layout with the checkpoint's, and `load_hf_weights` follows the layout of the transformers `modeling_qwen3_5.py` source:

| Checkpoint | Engine | Conversion |
|---|---|---|
| Norm weights are zero-centred: $x(1 + w)$ | RMSNorm multiplies by $w$ | Store $1 + w$ in the decoder, q, k, and final norms. Not in the linear layer's gated norm. |
| Full-attention `q_proj` interleaves each head's query and gate | All queries, then all gates | Reshape to heads, split, concatenate. |
| `in_proj_qkv` and `in_proj_z`; `in_proj_b` and `in_proj_a` | `in_proj_qkvz`; `in_proj_ba` | Concatenate the rows, in that order. |
| `conv1d.weight` is (channels, 1, kernel) | (channels, kernel) | Squeeze the middle axis. |
| No `lm_head.weight` when embeddings are tied | Separate `lm_head` | Copy the embedding. |

**What a pass would and would not show.** Three of three prompts matching for 8 tokens would show that the loader's mapping is right and that the engine's forward pass agrees with the reference at float32 on that model. It would not show anything about speed, about bfloat16 (where greedy output can diverge legitimately after a few tokens), about the 27B model, or about preemption on real weights. A failure is as likely to be the loader as the engine. The mapping above comes from reading the source, not from loading a checkpoint, and a wrong row in it fails every prompt. If the check fails, bisect with [chapter 8](/c/08-the-forward-pass)'s first-divergent-layer tool before you touch the engine.

## What goes wrong

- **Testing with prompts shorter than the chunk.** The chunk-boundary bug needs a prefix to hide. A test whose prompts all fit in one chunk passes with the wrong mask.
- **Testing only greedy.** Argmax draws no random numbers, so a shared generator, a missing reset on preemption, and a seed that is never used all pass.
- **Testing without pressure.** With a roomy block pool nothing is preempted, so the restart path never runs. The lab's budget is small enough to force 17 preemptions, and it asserts that at least one happens.
- **A slot that is reused without being zeroed.** The first request in a slot is right, because the pool starts at zero. The bug shows up on the second owner.
- **Releasing blocks but not the slot on preemption.** The pool of slots runs dry after a few evictions, and the next admission raises. Free everything together, in one function.
- **Trusting `still running`.** A request can be preempted and re-admitted inside one scheduler call. Check the position the next chunk starts at, not only membership.
- **Reading the goodput as an A100 number.** It prices a tiny-model trace with a model of the 27B. It checks that the engine batches and loses nothing. It does not say how fast the real engine is.
- **Treating the gated check as passed.** On a CPU it skips. A green lab run says nothing about real weights.

> [!RECAP]
> - An engine is the glue between parts: a state slot, a cache view, and a loop. The model does not change; the cache interface does the work.
> - A request's blocks, state slot, generator, and engine view are created and destroyed together. Preemption is the path that tests this.
> - A chunk needs a bottom-right-aligned causal mask and the previous chunk's recurrent state. The wrong mask produces fluent, wrong text.
> - A restart needs a zeroed slot and a fresh generator. Detect it by the position, not by the running set.
> - An audit after every step finds a desynchronised pool at the step that caused it.
> - The simulated clock prices a step with chapter 17a's floor. The trace's goodput is a correctness gate. The real-weights check has never run.

## Check your understanding

> [!QUESTION] The model's attention layers are rebuilt from scratch after a preemption, so why is a stale recurrent state still wrong?
> The attention layers read K and V from blocks that the restart rewrites. The six linear layers read the slot. A stale slot already contains the effect of every token the request ran before the eviction, so the recompute feeds the prompt in again on top of a state that has read it once. The state is a function of everything it has seen. Only a zeroed slot makes the recompute a function of the prompt alone.

> [!QUESTION] The chunk size is 5 and every prompt in your test is exactly 10 tokens. What bug can the test not see?
> It cannot see a short final chunk, and it exercises only the two chunk positions it has. The wrong-mask bug is still visible, because the second chunk has a prefix. A test that also uses prompts of 6 to 30 tokens covers a final chunk of every length from 1 to 5, which is where an off-by-one in a chunk's start position hides.

> [!QUESTION] Greedy output is identical with and without a shared sampler generator. Does that mean the engine's sampling is right?
> No. Greedy decoding calls argmax and draws nothing, so the generator is never used. The bug needs a non-greedy request, and it needs a second request in the batch whose draws interleave with the first one's. The lab runs each seeded request alone and compares it with the batched and preempted runs.

> [!QUESTION] A fused step with 16 decodes and one 16-token chunk costs 32.601 ms, and two separate passes cost 62.400 ms. Where does the 29.8 ms go?
> The second pass reads the weights again. At the rated 1935 GB/s the 27B model's weights take 27.799 ms, and the second pass also pays another launch floor. The fused step reads them once and adds only the chunk's extra FLOPs and bytes. The saving is the reason the scheduler packs prefill chunks into decode steps.

## Lab

> [!TRY] Assemble the engine and keep its outputs exact
> Write `StatePool`, `SeqCache`, `Engine`, `step_seconds`, `run_trace`, and `goodput` in `starter.py`. The model builder, the A100 constants, and the Hugging Face loader are given. You pass when greedy output equals a no-cache forward pass for 8 seeded prompts under chunking and under preemption, all blocks and slots come back, seeded sampling is identical alone, batched, and preempted, and the 64-request trace scores at least twice the one-at-a-time goodput.

The lab runs on the CPU, on a seeded random-initialised model with 8 layers, hidden size 48, and a vocabulary of 128. The model builder sets the conv weights and `A_log` to values with memory, because the engine's default initialisation makes the linear layers a no-op and a no-op cannot tell a restored state from a lost one. Nine checks go from the state pool to the whole trace.

The tenth check is gated. It runs only with CUDA and a local checkpoint, loads `Qwen/Qwen3.5-0.8B` by default, and compares 8 greedy tokens for three prompts with `transformers`. **Not verified on an A100 or with real weights: the course's GPU endpoint was paused, so that check has never run.** `meta.json` sets `needs_weights` to false, because the nine CPU checks need no weights.

The tests also name wrong implementations the checks must catch: a restart that keeps the old state, prefill chunks through `sdpa`, one shared generator, and a preemption that frees blocks but not the slot.

## Further reading

- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm). A from-scratch reimplementation of vLLM with prefix caching, tensor parallelism, and CUDA graphs. Read its engine and scheduler beside this chapter's.
- [Mini-SGLang](https://github.com/sgl-project/mini-sglang). A compact implementation of SGLang, about 5,000 lines, written to explain how a serving stack fits together.
- [vLLM architecture overview](https://docs.vllm.ai/en/latest/design/arch_overview.html). The V1 process architecture, the scheduler, and its preemption and async scheduling sections.
- [Chapter 16a](/c/16a-the-serving-loop-and-api) for the loop and the release rule, [chapter 17a](/c/17a-performance-model) for the step-time model, and [chapter 20](/c/20-speculative-decoding) for the rollback problem this chapter's restart shares.
