---
title: CUDA graphs and launch overhead
slug: 10a-cuda-graphs-and-launch-overhead
part: "Part 3 — Making it fast"
summary: The launch floor under a decode step, the CUDA graph that replays that step as one launch, and the batch-size buckets the graph forces on you.
minutes: 90
gpu: true
objectives:
  - Price a decode step's launch floor from the kernel count and the 5 µs launch cost.
  - Capture a decode step and say which parts stay fixed and which parts you may change.
  - Pad a real batch to a captured size and compute the slots that padding wastes.
  - "Name the capture failures: a host sync, a dynamic shape, an allocation capture does not record, and `.item()`."
  - Say when `torch.compile` fuses a region and when a graph break splits it.
lab: 10a-cuda-graphs
---

# CUDA graphs and launch overhead

> [!TLDR]
> - A decode step's launch floor is the kernel count times 5 µs. At the 400 kernels from [chapter 10](/c/10-roofline), that floor is 2.000 ms, before any kernel does useful work.
> - A CUDA graph records the step and replays it with one launch. Addresses and shapes stay at the captured values. The values in those addresses may change.
> - You capture one graph per batch size and pad the real batch up to the smallest captured size that holds it. On a 200-step trace, powers of two leave 402 idle slots out of 1,472. A tighter set leaves 150 out of 1,220.
> - Capture rejects a host sync, including `.item()`, a shape that is not the captured shape, and `cudaMalloc`, which capture does not record. `torch.cuda.empty_cache()` is the allocator call the lab marks as a failure.
> - `torch.compile` with `mode="reduce-overhead"` replays CUDA graphs for CUDA-only regions that do not mutate their inputs. A `.item()` or a data-dependent branch is a graph break, and the region splits there.

[Chapter 10](/c/10-roofline) already prices the launches. A decode step runs on the order of 400 kernels, and each launch costs 5 to 10 µs. At 5 µs the launches alone are 2.0 ms, which is more than the arithmetic in a batch-1 step and 7% of the memory floor in that chapter's table. A CUDA graph removes that term. It records the kernels once and submits the record as one launch.

The record is rigid. The addresses and the shapes are the ones from capture, and a later step that arrives with a different batch size needs a different graph. Production engines capture a list of batch sizes and pad. Padding is wasted compute, and the list is how you buy less waste at the cost of more captured graphs.

This chapter prices the floor, captures one step, counts the waste, and then says what capture rejects. `torch.compile` is the other tool in the same box: it can fuse a region, and it can also cut the region in half.

## Before you start

- **The launch cost, from [chapter 10](/c/10-roofline).** A kernel launch is the CPU handing one GPU function to the driver. Chapter 10 states the cost as 5 to 10 µs end to end, and it works the 400-kernel example at both ends: 2.0 ms and 4.0 ms. This chapter uses the 5 µs end. [Chapter 17a](/c/17a-performance-model) uses that same 5 µs in its step-time model. Neither chapter measured a new per-launch time.
- **The two floors, from the same chapter.** At batch 1 the arithmetic floor is 0.172 ms and the memory floor, at the rated 1935 GB/s, is 27.8 ms. Launch overhead sits on top of the slower of those two. Capturing the step changes the launch term. It does not change the bytes.
- **A tensor's address.** Two Python names can refer to one storage, and a new `torch.empty` is a new storage. Capture records the storage address. Copying values into that storage leaves the address in place.
- **The block table, later.** [Chapter 15](/c/15-paged-attention) builds a block table that maps a token to a page. When you capture decode, that table has to be a tensor whose address and shape are fixed for the captured batch. The page ids inside it are contents, and those may change.

## Price the launch floor

This section answers what the kernel count costs before a kernel does any useful work.

Chapter 10's working value is 5 µs per launch. The launches in one step are serial, because the next kernel waits for the previous one to finish and for its own launch to be processed. Multiply:

$$
t_{\text{launch}} = \hla{N_k} \times 5\,\mu\text{s}
$$

$\hla{N_k}$ is the number of kernels in the step. For the 400-kernel count chapter 10 uses:

$$
\boxed{400 \times 5\,\mu\text{s} = 2.000\,\text{ms}}
$$

That 2.000 ms is arithmetic, not a measurement of a running engine. It is the floor under the launch term. A step cannot spend less than this on launches while it still launches $\hla{N_k}$ kernels one by one.

A graph replay submits the whole record with one launch. Price that launch with the same 5 µs, and label it the same way: an estimate from the course constant, not a measured graph-launch latency.

$$
1 \times 5\,\mu\text{s} = 0.005\,\text{ms}
$$

The launches you remove are the other 399:

$$
399 \times 5\,\mu\text{s} = 1.995\,\text{ms}
$$

Chapter 10's memory floor at batch 1 stays 27.8 ms. The graph takes 1.995 ms off the launch term and leaves the byte term where it was. The lab times eager replay against graph replay when a CUDA device is visible, and it reports the ratio as `speedup`. It does not require that ratio to clear any threshold. A step that is already one kernel has little launch cost left to remove.

> [!KEY] The floor is the kernel count times 5 µs
> ==Four hundred launches cost 2.000 ms== at the 5 µs this course uses. One replay of a captured step costs 0.005 ms on that same constant. Both figures are products, and `scripts/figures/10a-cuda-graphs-and-launch-overhead.py` prints them.

> [!EXAMPLE] Where 2.000 ms sits
> Chapter 10's launch table already lists 2.0 ms as 12 times the 0.172 ms arithmetic floor, and as 7% of the 27.8 ms memory floor. The 1.995 ms a graph takes off the launch term is the large one of those two comparisons, and it is paid once per generated token. The byte term stays 27.8 ms.

## Capture a decode step

This section answers what a CUDA graph records, and what you are allowed to change between replays.

A CUDA graph is a record of the work a stream and its dependent streams perform: the kernels and the arguments those kernels were launched with. PyTorch builds that record by stream capture. Work issued to a capturing stream is recorded instead of run. After capture, `replay` submits the record. PyTorch's note on CUDA graphs says each replay runs the same kernels with the same arguments, and for pointer arguments that means the same addresses. One call, `cudaGraphLaunch`, submits the whole record.

Two things are fixed, and one thing is not:

| Piece | At capture | On a later replay |
|---|---|---|
| Storage addresses | The tensors you pass in | The same addresses. A new tensor is a different graph. |
| Shapes and layouts | The sizes of those tensors | The same sizes. A different batch size is a different graph. |
| Values in those addresses | Whatever was there during capture | You may overwrite them, then replay. |

The usual way to overwrite a value and keep the address is `copy_`. The following sample captures a multiply and an add, then runs them on new contents. `fresh_x` and `fresh_scale` are tensors of shape `(4, 8)` that you already hold. On a machine with no CUDA device the sample does not run, and the lab skips the same way.

```python
g = torch.cuda.CUDAGraph()
static_x = torch.empty((4, 8), device="cuda")
static_scale = torch.empty((4, 8), device="cuda")

side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    for _ in range(3):
        static_y = static_x * static_scale + static_x
torch.cuda.current_stream().wait_stream(side)

with torch.cuda.graph(g):
    static_y = static_x * static_scale + static_x

static_x.copy_(fresh_x)
static_scale.copy_(fresh_scale)
g.replay()
```

The warmup loop runs eager, on a side stream, before capture. The caching allocator finishes its first-call setup there, and on a real decode step this is also where cuBLAS picks an algorithm, so capture records the steady-state kernels. `torch.cuda.graph` then captures on its own side stream. PyTorch's note includes this warmup in the capture sequence. `static_y` has to stay alive for the life of the graph, because replay writes that same storage.

> [!KEY] Addresses and shapes stay fixed
> Replay ==reads and writes the addresses from capture==. You change the step by copying new values into those addresses. Rebinding the name, `static_x = fresh_x`, points the Python name at a new address and leaves the graph on the old one.

The same split applies to the KV cache and, once you build it, to the block table. The cache storage for a captured batch is allocated up front, at the bucket's batch size and at the longest context that graph reads. Replay writes new keys into that storage. The block table is a tensor of page ids with the captured shape. You copy a new row of ids in between replays. A Python list of ids built on every step is a new address, and the kernel recorded during capture keeps the address it saw.

## Pad the batch to a captured size

This section answers which batch sizes to capture, and how many slots the padding wastes.

A graph has one shape. A decode step's real batch changes every iteration as requests arrive and finish. You capture a list of batch sizes, the *buckets*, and you run the real batch on the smallest bucket that can hold it. The extra rows are padding. They do the same kernels as a real row, and their output is discarded.

Define $\hlb{b(n)}$ as the smallest bucket that is at least the real batch $n$. The waste of a bucket set on a trace is the idle padded slots divided by all padded slots:

$$
W = \frac{\sum_i \left(\hlb{b(n_i)} - n_i\right)}{\sum_i \hlb{b(n_i)}}
$$

The numerator is work you launched for rows that are not real requests. The denominator is the work you launched. A set that needs a bucket larger than every captured size has no graph for that step. The lab's `pick_bucket` returns `None` in that case, and the step stays on the eager path.

> [!EXAMPLE] Three batches, counted by hand
> Take the batches 1, 3, and 5. On the buckets 1, 2, 4, and 8 the pads are 1, 4, and 8. Idle slots are 0 + 1 + 3 = 4, and padded slots are 13, so the waste is 4/13. On the buckets 1, 2, 4, 6, and 8 the pads are 1, 4, and 6. Idle slots are 2, padded slots are 11, and the waste is 2/11. The second set wins because a batch of 5 fits in 6.

The lab's trace has 200 steps. The following table is the whole trace. `scripts/figures/10a-cuda-graphs-and-launch-overhead.py` sums it.

| Batch | Steps | Power-of-two pad | Captured pad |
|---|---|---|---|
| 1 | 40 | 1 | 1 |
| 2 | 20 | 2 | 2 |
| 3 | 20 | 4 | 4 |
| 5 | 30 | 8 | 6 |
| 6 | 24 | 8 | 6 |
| 7 | 12 | 8 | 8 |
| 8 | 10 | 8 | 8 |
| 9 | 16 | 16 | 12 |
| 10 | 12 | 16 | 12 |
| 12 | 8 | 16 | 12 |
| 13 | 4 | 16 | 16 |
| 15 | 4 | 16 | 16 |

Powers of two are the buckets 1, 2, 4, 8, 16, and 32. They pad 1,070 real slots up to 1,472, and 402 of those slots are idle. The idle fraction is 402/1472, which reduces by 2 to 201/736.

The captured set in the figure is 1, 2, 4, 6, 8, 12, 16, 24, and 32. It pads the same 1,070 slots up to 1,220, and 150 of those are idle. That fraction is 150/1220, which reduces by 10 to 15/122. The tighter set launches 252 fewer padded slots on this trace. Both totals are the script's sums, not a profile.

```viz
10a-padding-waste
```

The figure draws the idle slots of one step, pad minus real batch, and the readout keeps the trace totals. A batch of 5 is the opening position: the power-of-two pad is 8 and the captured pad is 6.

Each extra bucket is another graph, another set of static buffers, and more capture time at startup. vLLM's design note says a uniform decode batch takes the smallest captured decode graph that covers it, and any other batch takes the smallest token-keyed graph. If no graph covers the batch, the mode is `NONE` and the step runs eager. With piecewise compilation, vLLM's default mode is `FULL_AND_PIECEWISE`: a full graph for uniform decode, and piecewise graphs for the other batches. Piecewise leaves operations that cannot be captured on the eager path.

SGLang exposes the list directly. `--cuda-graph-bs-decode` is the decode batch sizes to capture, and `--cuda-graph-max-bs-decode` is the maximum. `--disable-cuda-graph-padding` keeps the graph for a batch that already matches a captured size, and runs eager when padding would be required. Prefill has its own list and a maximum sequence length. A prefill longer than that length runs eager. The shape still has to be one you captured.

## What capture rejects

This section answers which four mistakes make a step illegal to capture.

PyTorch's capture rules split into two kinds. A host sync, and control flow that depends on a value the CPU has to read, are rejected when you capture. A dynamic shape, and a replay that uses a different address, are prohibited too, and those two can come back as a wrong tensor rather than an exception. The NVIDIA programming guide states the sync rule at the driver: it is invalid to synchronize a stream that is being captured, or to query its status. The runtime API adds the allocator rule: `cudaMalloc` is not enqueued on the stream, so capture does not observe it.

The four snippets the lab grades are these:

| Name | The step | Verdict |
|---|---|---|
| `static_copy` | `return x * scale`, after the caller has copied new values into `x` and `scale` | Captures. Addresses and the shape are fixed. |
| `host_item` | `gate = x.sum().item()`, then `return x * gate` | Fails. `.item()` copies a device scalar to the host and synchronizes. |
| `dynamic_rows` | `return x[:n] * 2`, and the Python int `n` changes with the real batch | Fails the static-shape rule. Capture bakes in the `n` it saw. |
| `allocator_cache` | `torch.cuda.empty_cache()`, then `return x + 1` | Fails. `empty_cache` synchronizes and returns cached blocks to the driver. Capture does not record that. |

> [!WARNING] `.item()` is a host sync
> A debug print of one element, a stopping rule that reads a logit on the CPU, or a dtype branch on a tensor value all call the same kind of sync. Put the decision outside the captured region. Inside it, capture rejects the step.

The shape case can look like a success. Capture of `x[:n]` at `n = 4` records a kernel for 4 rows. A later call with `n = 6` keeps that kernel if you replay without recapturing. The output length stays 4. That is a failed capture of the step you wrote, even when the capture call itself returned a graph.

> [!NOTE] The lab grades the names on the CPU
> `failing_captures` returns `allocator_cache`, `dynamic_rows`, and `host_item`, sorted. That list is the table in this section. The CPU check does not attempt capture, and a passing run on a machine with no GPU has not confirmed the table on a device. On a CUDA device, attempt capture. `host_item` and `allocator_cache` hit the sync rule. `dynamic_rows` keeps the length from capture when you change `n` and replay. `static_copy` replays the values you copied in.

> [!DEEPDIVE] Why cudaMalloc is not a captured node
> Stream capture records work that is enqueued on the capturing stream. `cudaMalloc` allocates and returns on the host. It is not a stream operation, so the graph has no node for it. If the rest of the step uses the pointer `cudaMalloc` returned, replay still uses that pointer, and nothing in the graph allocates it again.
>
> CUDA's own graph memory nodes are the exception the driver does record. `cudaMallocAsync` and `cudaFreeAsync` are stream-ordered, and capture turns them into allocation nodes whose virtual addresses stay fixed for the life of the node.
>
> PyTorch's `torch.cuda.graph` takes a private caching-allocator pool around the capture. An `empty` that runs inside that block comes out of the pool, and the address is then fixed, which is why a single allocation during capture can be legal. The allocation is still a fixed address. Calling the allocator again on the replay path, or freeing the tensor the graph holds, breaks the contract. `empty_cache` is worse than a private-pool `empty`: it synchronizes and gives memory back, which is the operation capture cannot record.

## Fuse with torch.compile

This section answers when the compiler turns several ops into fewer kernels, and when it stops and starts a new region.

`torch.compile` traces the step and hands the region to inductor, the default backend. While the ops stay in one region, inductor can fuse a pointwise chain. `x * scale + x` is a multiply and an add. A fused region writes the sum without a second kernel reloading the product. The `torch.compile` reference's `epilogue_fusion` option requests that fusion into a template, and the option requires the `max-autotune` mode. In the default mode, fusion is whatever inductor does with a region that stayed whole.

A *graph break* ends the region. The compiler runs the compiled prefix, runs the breaking op eager, and starts a new region afterward. The common cases, from PyTorch's graph-break note, are a data-dependent branch and a direct read of tensor data: `.item()`, and `.data_ptr()`. Printing does it too. `fullgraph=True` turns a break into an error, which is how you find the op that split the step.

`mode="reduce-overhead"` is the mode that tries to cut Python overhead with CUDA graphs. The `torch.compile` reference limits that attempt: it applies to CUDA-only graphs that do not mutate their inputs, it can keep workspace memory around so the next call skips the allocation, and a reduction in overhead is not guaranteed. A decode step that writes the KV cache in place mutates an input. That step is the explicit capture from the previous sections, which is what vLLM and SGLang schedule. `reduce-overhead` still helps a region that only reads its inputs and writes a fresh output, such as a fused pointwise epilogue that does not alias the KV cache.

A graph break ==puts the host back between two short graphs==. Each piece can be fused on its own, and each piece can be replayed on its own. The `.item()` between them is a sync on every token. Move that read out of the step. Capture and the compiler reject it for the same reason.

## What goes wrong

- **Rounding the batch up to a power of two.** `pick_bucket(5, [1, 2, 4, 6, 8])` is 6. An implementation that ignores the list and rounds 5 up to 8 returns 8, and its waste on `[1, 3, 5]` is 4/13. The lab's CPU checks catch that implementation. The tighter list's waste is 2/11.
- **Rebinding the static tensor.** `static_x = fresh_x` changes the address. Replay keeps writing the tensor from capture, and the new values sit in a tensor the graph does not read.
- **Capturing on every step.** Capture includes the eager launches plus the recording. You pay it at startup, once per bucket, and then you replay. Capturing inside the token loop puts the launch floor back and adds the recording on top.
- **Leaving `.item()` next to the matmul.** The sync fails capture, and on the eager path it drains the GPU before the next kernel is queued. Read the scalar after `replay` returns, on the host side of the step.
- **Treating a one-kernel speedup as a failed graph.** The 1.995 ms figure is 399 launches you no longer pay. A step that was already one kernel has nothing to remove. The lab records `speedup` and does not require it to exceed 1.2.

> [!RECAP]
> - The launch floor is $\hla{N_k} \times 5\,\mu\text{s}$. At 400 kernels that is 2.000 ms, and one graph replay priced the same way is 0.005 ms.
> - Replay uses the captured addresses and the captured shapes. Copy new values in with `copy_`. Warm up on a side stream before capture.
> - Pad to the smallest captured batch size. On the 200-step trace, powers of two waste 201/736 of the padded slots, and the tighter set wastes 15/122.
> - A host sync, `.item()`, a Python slice bound that changes, and `empty_cache` fail the capture rules. The CPU grades those names. A device confirms them by attempting capture.
> - `reduce-overhead` replays CUDA graphs for CUDA-only regions that do not mutate inputs. A graph break splits the region, and fusion stops at the split.
> - A decode step that writes the KV cache in place is an explicit graph with static buffers. Each bucket is one graph.

## Check your understanding

> [!QUESTION] A decode step launches 400 kernels. What is the launch floor at 5 µs, and what is it after you capture the step as one graph?
> $400 \times 5\,\mu\text{s} = 2.000\,\text{ms}$ before capture, and $1 \times 5\,\mu\text{s} = 0.005\,\text{ms}$ after, using the same 5 µs. Both are products of the course constant. Chapter 10's memory floor of 27.8 ms does not move. The lab measures the real ratio on a device and does not require a particular speedup.

> [!QUESTION] You copy a new batch into the static tensor and the output still matches the batch from capture. What did the copy miss?
> The graph is still reading the address it recorded. The assignment `static_x = fresh_x` binds the name to a new tensor. `static_x.copy_(fresh_x)` writes the new values into the recorded storage. Replay then sees them.

> [!QUESTION] A batch of 5 arrives, and the captured sizes are 1, 2, 4, 6, and 8. How many rows are idle, and what does a power-of-two rounder do with the same batch?
> The smallest bucket that holds 5 is 6, so one row is idle. A rounder that ignores the list and goes to the next power of two selects 8, and three rows are idle. On the hand trace 1, 3, 5 that mistake is the difference between waste 2/11 and waste 4/13.

> [!QUESTION] You compile a step with `mode="reduce-overhead"`, and a line in the middle calls `logits.argmax().item()`. What happens to fusion?
> The `.item()` is a graph break. The compiler can fuse the region before it and the region after it, and it cannot fuse across it. `reduce-overhead` tries CUDA graphs on the CUDA-only pieces that do not mutate their inputs. The `.item()` stays a host sync between those pieces, on every token.

## Lab

> [!TRY]
> Write `pick_bucket`, `padding_waste`, `failing_captures`, and `capture_decode`. You pass the CPU checks when a batch of 5 on the buckets 1, 2, 4, 6, and 8 returns 6, when the waste on `[1, 3, 5]` is 4/13 for powers of two and 2/11 for the tighter list, when your `BUCKETS` wastes less than powers of two on the 200-step trace, and when the three failing snippet names come back sorted. The capture checks compare replay to eager execution on five contents and check that replay does not allocate. They run on a CUDA device. With no device the harness skips them and still passes the CPU checks.

`pick_bucket(batch, buckets)` returns the smallest bucket greater than or equal to `batch`, or `None` when none fits. `buckets` may be unsorted.

`padding_waste(trace, buckets)` returns idle padded slots divided by padded slots. Raise `ValueError` if any batch has no bucket.

`BUCKETS` is your captured set. On the harness trace it has to waste less than `[1, 2, 4, 8, 16, 32]`. The set in this chapter does. Powers of two do not, because they are the baseline.

`failing_captures(snippets)` returns the sorted names that fail the table under What capture rejects. The harness passes the four snippets. Grade them from the rules. A CPU result is a classification, and a device confirms it by attempting capture.

`capture_decode(model_step, static_inputs)` warms up on a side stream, captures `model_step(*static_inputs)`, and returns an object whose `replay()` runs the graph and returns the output tensor from capture. The caller copies new contents in between replays. The check skips when `torch.cuda.is_available()` is false.

## Further reading

- [PyTorch CUDA semantics, section CUDA Graphs](https://docs.pytorch.org/docs/2.15/notes/cuda.html). Stream capture, the side-stream warmup, and the rule that replay uses the same addresses. The capture sample in this chapter follows that sequence.
- [torch.compile](https://docs.pytorch.org/docs/2.15/generated/torch.compile.html). `mode="reduce-overhead"` uses CUDA graphs for CUDA-only regions that do not mutate inputs, and it says a reduction in overhead is not guaranteed.
- [Common Graph Breaks](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html). Data-dependent branches and `.item()` end a compiled region.
- [CUDA Graphs, vLLM design note](https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md). `FULL_AND_PIECEWISE` is the default when piecewise compilation is on. A uniform decode batch takes the smallest covering graph.
- [SGLang server arguments](https://docs.sglang.io/advanced_features/server_arguments.html). `--cuda-graph-bs-decode`, `--cuda-graph-max-bs-decode`, and `--disable-cuda-graph-padding`.
- [CUDA Programming Guide, section 4.2 CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html). Synchronizing a stream that is being captured is invalid.
- [CUDA Runtime API, stream management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html). `cudaMalloc` during capture is not a stream operation, so capture does not record it.
- [CS336 syllabus](https://cs336.stanford.edu/). Lecture 6 is Kernels, Triton, and it is the lecture that profiles a step. Lecture 10 is Inference. This chapter does not take a launch-cost number from either lecture. The 5 µs is chapter 10's.
