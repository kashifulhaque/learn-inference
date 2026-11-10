# Fact entries for chapter 16a

This file records numbers introduced by the serving-loop chapter. The rest of
the course's numbers live in the chapters that derive them. A number in
chapter 16a is arithmetic shown there, a row below, or a value the lab
measures on its simulated clock.

| Tag | Meaning |
|---|---|
| **C** | Confirmed in the cited page. |
| **D** | Derived. The arithmetic is in the chapter and in `scripts/figures/16a-the-serving-loop-and-api.py`. |

| Claim | Value | Tag | Source |
|---|---|---|---|
| GPU step time for Llama-8B on an H100, as stated by the vLLM project | as low as about 5 ms | C | [VLLM-V1] |
| OpenAI stop sequences | up to 4; the returned text does not contain the stop sequence | C | [OPENAI-CHAT] |
| Usage chunk when `stream_options.include_usage` is true | an extra chunk before `data: [DONE]`; `choices` is empty; earlier chunks carry `usage: null` | C | [OPENAI-CHAT] |
| `finish_reason` of `stop` | the model hit a natural stop or a provided stop sequence | C | [OPENAI-STREAM] |
| `finish_reason` of `length` | the request reached its maximum token count | C | [OPENAI-STREAM] |
| Serial idle ticks, 8 steps, schedule 6, forward 10, post 2 | 64 of 144 wall ticks, fraction 4/9 | D | chapter 16a |
| Overlapped idle ticks, same counts, schedule at most the forward | 22 of 102 wall ticks, fraction 11/51 | D | chapter 16a |

This machine has no GPU. The tick counts are not an A100 measurement. The lab
reports the overlapped ratio as `gpu_idle_fraction`.

Chapter 0's decode-step lower bound of about 42 ms is that chapter's
arithmetic (53.8 GB at the measured 1275 GB/s copy). Chapter 16a cites it and
does not remeasure it.

## Sources

| Key | Page |
|---|---|
| [VLLM-V1] | vLLM V1: A Major Upgrade to vLLM's Core Architecture, vLLM Blog, 27 January 2025. https://vllm.ai/blog/2025-01-27-v1-alpha-release |
| [VLLM-CORE] | `vllm.v1.engine.core` API, EngineCore.step: "Schedule, execute, and make output." https://docs.vllm.ai/en/latest/api/vllm/v1/engine/core/ |
| [VLLM-SRC] | `vllm/v1/engine/core.py` on the vLLM main branch. The batch queue "enables us to asynchronously schedule and execute batches." https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py |
| [SGLANG] | SGLang `Engine` docstring: TokenizerManager, a Scheduler subprocess, and a DetokenizerManager subprocess, connected with ZeroMQ. https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/engine.py |
| [TRTLLM] | TensorRT-LLM Executor API. https://nvidia.github.io/TensorRT-LLM/advanced/executor.html |
| [OPENAI-CHAT] | OpenAI Chat Completions reference, streaming and `stop`. https://developers.openai.com/api/reference/resources/chat |
| [OPENAI-STREAM] | OpenAI Chat Completions streaming events. https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events |
| [CS336] | Stanford CS336, Language Modeling from Scratch, syllabus. Lecture 10 is "Inference", taught by Percy Liang. https://cs336.stanford.edu/ |

## Chapter 16

Numbers introduced by the trace-replay section. They come from the lab's own computation on a simulated clock, not from a source. A step costs 2 ms plus 0.125 ms per token. The constants are planted toy values, not an A100 measurement, and `content/labs/16-scheduler/tests.py` checks every row below.

| Claim | Value | Tag | Source |
|---|---|---|---|
| Decode-only step, 4 tokens | 2 + 0.125 × 4 = 2.5 ms | D | chapter 16 |
| Step that admits a 960-token prompt whole, with 4 decodes | 2 + 0.125 × 964 = 122.5 ms | D | chapter 16 |
| Step with 4 decodes and a 60-token chunk | 2 + 0.125 × 64 = 10.0 ms; 960 / 60 = 16 steps | D | chapter 16 |
| Four decoders with a 960-token prompt arriving at 10 ms, whole versus chunked | max gap and `p99_itl` 122.5 ms versus 10.0 ms | D | chapter 16 |
| Same trace, TTFT of the long prompt | 123.5 ms versus 161.0 ms | D | chapter 16 |
| Same trace, `p99_ttft` over ttft 6.0 four times and the long prompt | 118.8 ms versus 154.8 ms | D | chapter 16 |
| Same trace, end of run and mean ITL of each decoder | 223.875 ms for both, 217.875 / 39 = 5.5865 ms | D | chapter 16 |
| A, B at priority 0 and C at priority 5 arriving at 1 ms, budget 32, TTFT of B and C | FCFS 18.0 and 19.25 ms; priority 20.25 and 17.0 ms | D | chapter 16 |
| Same trace, TTFT from admission instead of arrival (wrong) | B 8.25 ms, C 12.0 ms | D | chapter 16 |
| Same trace, A's ITL over n instead of n - 1 gaps (wrong) | 3.0 ms instead of 6.0 ms | D | chapter 16 |
| Percentile with linear interpolation, ten values 0.2 and one 5.0 | P99 = 0.2 + 0.9 × 4.8 = 4.52 | D | chapter 17 |


## Chapter 11

Numbers introduced by the batched, sort-free, and hook section. They come from the lab's own computation, not from a source. The logits are `torch.randn(4, 248320) * 3` with `torch.manual_seed(2)`, and the exact kept counts are the ones `content/labs/11-sampling/tests.py` checks.

| Claim | Value | Tag | Source |
|---|---|---|---|
| Worked batch on logits (3, 2, 1, 0, -1), rows (T=1, p=0.7), (T=0.5, p=0.7), (T=2, k=3, p=0.9) | kept 2, 1, and 3 tokens | D | chapter 11 |
| Row 3 of that batch, top three at T=2 | probabilities 0.5065, 0.3072, 0.1863 | D | chapter 11 |
| Bisection on the T=1 row at p=0.7, 8 steps | threshold bracket 0.2337 to 0.2362, boundary probability 0.2341 | D | chapter 11 |
| Top-p nucleus size on 248,320 entries, p = 0.5, 0.9, 0.95, 0.99, float64 | 436, 12,048, 22,994, 60,458 tokens | D | chapter 11 |
| Sum of a float32 `cumsum` over one 248,320-entry softmax row | 1.00007 instead of 1 | D | chapter 11 |
| Sort-free top-p, 40 bisection steps, four rows on CPU | 0.02 s, about the same as one 248,320-entry sort | D | chapter 11 |

The timing row is a CPU run on the author's machine, not an A100 measurement. The lab reports it as `sortfree_seconds`.

## Chapter 17a

Numbers introduced by the performance-model chapter. A time here is a floor
from chapters 2, 6, 10, and 19, evaluated in
`scripts/figures/17a-performance-model.py`. It is not an A100 measurement.
The lab's bandwidth fraction 0.7 and overhead 0.002 s are planted in a
synthetic trace. They are not device constants, and they are not
$1275/1935$.

| Claim | Value | Tag | Source |
|---|---|---|---|
| Dense-stack parameters, 64 full-attention layers plus embeddings and norms | 26,367,104,000 | D | chapter 17a |
| Hybrid decode floor, batch 1, context 1, degree 1, rated 1935 GB/s, 400 launches at 5 µs | 29.962 ms (memory 27.962 ms, compute 0.173 ms, launch 2.000 ms) | D | chapter 17a |
| Context where the hybrid decode intensity cap falls through the 161.240 FLOP/byte ridge | 340 tokens (161.27 at 339, 161.24 at 340) | D | chapter 17a |
| Batch where compute and memory floors meet, hybrid, context 1, degree 1 | 2,515.71 | D | chapter 17a |
| Sequences that fit at context 1, degree 1, beside the hybrid weights | 207 | D | chapter 17a |
| Prefill floor, one hybrid sequence, 2048 tokens, degree 1 | 360.094 ms (compute 358.094 ms, memory 37.176 ms) | D | chapter 17a |
| Four GPUs, context 1024, 1 s TPOT SLO, NVLink 300 GB/s one way, search to batch 4096 | degree 4, 15,398 tokens/s | D | chapter 17a |
| Same workload on PCIe 32 GB/s one way | degree 2 with two replicas, 11,609 tokens/s | D | chapter 17a |
| Pope et al. split inference into a memory time and a compute time, and treat prefill and decode separately | analytical model, not a latency for this model | C | [POPE] |
| DistServe defines goodput from a TTFT SLO and a TPOT SLO together | definition | C | [DISTSERVE] |
| CS336 lecture titles used here | Lecture 2, resource accounting; Lecture 10, Inference | C | [CS336] |
| MIT 6.5940 titles used here | Chapter I: Efficient Inference; Transformer and LLM; Guest Lecture: Efficient Inference in Production | C | [MIT65940] |

| Key | Page |
|---|---|
| [POPE] | Reiner Pope and others, Efficiently Scaling Transformer Inference. https://arxiv.org/abs/2211.05102 |
| [KIPPLY] | Kipply, Transformer Inference Arithmetic. https://kipp.ly/transformer-inference-arithmetic/ |
| [ROOFLINE-SURVEY] | Zhihang Yuan and others, LLM Inference Unveiled: Survey and Roofline Model Insights. https://arxiv.org/abs/2402.16363 |
| [VIDUR] | Amey Agrawal and others, Vidur: A Large-Scale Simulation Framework for LLM Inference. https://arxiv.org/abs/2405.05465 |
| [DISTSERVE] | Yinmin Zhong and others, DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving. https://arxiv.org/abs/2401.09670 |
| [MIT65940] | MIT 6.5940, TinyML and Efficient AI Computing, Fall 2026 schedule. https://hanlab.mit.edu/courses/2026-fall-65940 |

## Chapter 15

Numbers the paged-attention lab checks. All are arithmetic from chapter 2's per-token KV bytes and recurrent state, evaluated in `content/labs/15-paged-attention/tests.py`. None is a measurement.

| Claim | Value | Tag | Source |
|---|---|---|---|
| Swap bytes out, 2,000 tokens at 65,536 bytes of KV each, plus 154,927,104 bytes of recurrent state | 285,999,104 (back in: the same, 571,998,208 in all) | D | chapter 15 |
| Recompute for the same sequence | 0 bytes moved; 2 × 26.9e9 × 2,000 = 107.6 TFLOP | D | chapter 15 |
| Swap for 2,001 tokens, KV only | 126 blocks of 16, 132,120,576 bytes each way | D | chapter 15 |
| Fork of a 20-token sequence, then 12 appends on the child | 62 free blocks after the fork, 61 after the first append, 61 after 12 appends, 60 after the 13th | D | chapter 15 |

## Chapter 15a

Numbers introduced by the prefix-cache chapter. A byte count here is chapter 2's recurrent state, or an integer multiple of it. A time is the $2NL$ tensor-core term from chapter 15, evaluated in `scripts/figures/15a-prefix-caching-and-radix-trees.py`. It is not an A100 measurement, and it is not chapter 17a's prefill floor.

| Claim | Value | Tag | Source |
|---|---|---|---|
| Recurrent state, one sequence | 154,927,104 bytes (147.75 MiB; chapter 2 rounds the same quotient to 147.8 MiB) | D | chapter 2, chapter 15a |
| Checkpoints on every 16-token boundary of one 2,048-token prompt | 128 states, 19,830,669,312 bytes (18.46875 GiB) | D | chapter 15a |
| One checkpoint on the 100-prompt mix (40×256, 30×512, 20×1024, 10×2048) | position 512, 30,720 tokens saved, 154,927,104 bytes | D | chapter 15a |
| Two checkpoints on that mix | positions 256 and 1,024, 48,640 tokens saved, 309,854,208 bytes | D | chapter 15a |
| Three checkpoints on that mix | positions 256, 1,024, and 2,048, 58,880 tokens saved, 464,781,312 bytes | D | chapter 15a |
| Four checkpoints on that mix | all four lengths, 66,560 tokens saved, 619,708,416 bytes | D | chapter 15a |
| Tensor-core prefill, 500 tokens, exact parameter count, 312 TFLOP/s | 86.205 ms | D | chapter 15a |
| Same prefill at hit rate 0.75 | 375 tokens and 64.654 ms skipped, 21.551 ms left | D | chapter 15a |
| Shared 32-token prefix across 100 prompts of 48 tokens | 3,168 / 4,800 = 0.66 | D | chapter 15a |
| SGLang hit rate | cached prompt tokens / prompt tokens | C | [SGLANG-PAPER] |
| SGLang eviction | least recently used leaf whose reference count is zero | C | [SGLANG-PAPER] |
| Chatbot Arena RadixAttention hit rate, LLaVA-Next-34B and Vicuna-33B | 52.4% and 74.1%; Vicuna first-token latency 1.7× lower on average | C | [SGLANG-PAPER] |
| vLLM block hash | parent hash, block token ids, and extra keys; full blocks only | C | [VLLM-APC] |
| Mooncake block hash | hash of the block and of its prefix | C | [MOONCAKE] |
| Preble prefix index | radix trees; a node splits when a new prompt matches only part of its label | C | [PREBLE] |
| Marconi reuse, their traces, block size 32 | 25.0% of attention blocks, 0.4% of recurrent states | C | [MARCONI] |
| CS336 lecture cited here | Lecture 10, Inference, Percy Liang | C | [CS336] |

| Key | Page |
|---|---|
| [SGLANG-PAPER] | Lianmin Zheng and others, SGLang: Efficient Execution of Structured Language Model Programs. https://arxiv.org/abs/2312.07104 |
| [VLLM-APC] | vLLM, Automatic Prefix Caching. https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md |
| [MOONCAKE] | Ruoyu Qin and others, Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving. https://arxiv.org/abs/2407.00079 |
| [PREBLE] | Vikranth Srivatsa and others, Preble: Efficient Distributed Prompt Scheduling for LLM Serving. https://arxiv.org/abs/2407.00023 |
| [MARCONI] | Rui Pan and others, Marconi: Prefix Caching for the Era of Hybrid LLMs. https://arxiv.org/abs/2411.19379 |

## Chapter 10a

Numbers introduced by the CUDA-graph chapter. The 5 µs launch cost is chapter 10's lower end of the 5 to 10 µs range, the same value chapter 17a uses. It is not a new measurement. The products and the padding totals are arithmetic in `scripts/figures/10a-cuda-graphs-and-launch-overhead.py`. This machine has no CUDA device, so the chapter does not report a graph speedup.

| Claim | Value | Tag | Source |
|---|---|---|---|
| Eager decode launches, 400 kernels at 5 µs | 2.000 ms | D | chapter 10, chapter 10a |
| One graph replay at the same 5 µs | 0.005 ms | D | chapter 10a |
| Launches removed by that replay, 399 × 5 µs | 1.995 ms | D | chapter 10a |
| Hand trace `[1, 3, 5]` on buckets `[1, 2, 4, 8]` | 4 idle slots out of 13 padded, waste 4/13 | D | chapter 10a |
| Same hand trace on buckets `[1, 2, 4, 6, 8]` | 2 idle slots out of 11 padded, waste 2/11 | D | chapter 10a |
| 200-step trace on powers of two `[1, 2, 4, 8, 16, 32]` | 1,070 real slots, 402 idle out of 1,472 padded, waste 201/736 | D | chapter 10a |
| Same trace on `[1, 2, 4, 6, 8, 12, 16, 24, 32]` | 150 idle out of 1,220 padded, waste 15/122 | D | chapter 10a |
| PyTorch capture replays the same kernels at the same addresses | one `cudaGraphLaunch` per replay | C | [PT-CUDA-GRAPHS] |
| `torch.compile` mode `reduce-overhead` | CUDA graphs for CUDA-only regions that do not mutate inputs; a speedup is not guaranteed | C | [PT-COMPILE] |
| `torch.compile` graph break | data-dependent control flow, and `.item()` | C | [PT-BREAKS] |
| vLLM default CUDA-graph mode with piecewise compilation | `FULL_AND_PIECEWISE` | C | [VLLM-GRAPHS] |
| SGLang decode capture sizes | `--cuda-graph-bs-decode` list, or a maximum via `--cuda-graph-max-bs-decode` | C | [SGLANG-ARGS] |
| Synchronizing a stream that is being captured | invalid | C | [NV-GRAPHS] |
| `cudaMalloc` during stream capture | not enqueued on the stream, and not observed by capture | C | [NV-CAPTURE] |
| CS336 lecture titles used here | Lecture 6, Kernels, Triton; Lecture 10, Inference | C | [CS336] |

| Key | Page |
|---|---|
| [PT-CUDA-GRAPHS] | PyTorch CUDA semantics, section CUDA Graphs. https://docs.pytorch.org/docs/2.15/notes/cuda.html |
| [PT-COMPILE] | torch.compile. https://docs.pytorch.org/docs/2.15/generated/torch.compile.html |
| [PT-BREAKS] | Common Graph Breaks. https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html |
| [VLLM-GRAPHS] | vLLM design note, CUDA Graphs. https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md |
| [SGLANG-ARGS] | SGLang server arguments, CUDA graph flags. https://docs.sglang.io/advanced_features/server_arguments.html |
| [NV-GRAPHS] | NVIDIA CUDA Programming Guide, section 4.2 CUDA Graphs. https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html |
| [NV-CAPTURE] | NVIDIA CUDA Runtime API, stream management, capture restrictions. https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html |
| [CS336] | Stanford CS336, Language Modeling from Scratch, syllabus. Lecture 6 is Kernels, Triton. Lecture 10 is Inference. https://cs336.stanford.edu/ |

## Chapter 14a

Numbers introduced by the split-KV chapter. The SM count is 108 and the copy bandwidth is 1275 GB/s, both from chapter 10. A time or a GB/s here is that pair, evaluated in `scripts/figures/14a-split-kv-decode-attention.py`. It is not an A100 kernel measurement. The chapter rounds the three microsecond figures to two decimal places.

| Claim | Value | Tag | Source |
|---|---|---|---|
| Decode grid, batch 1, 24 query heads, query length 1 | 24 blocks, utilization 24/108 | D | chapter 10, chapter 14a |
| Copy-share ceiling at 24 of 108 SMs | 283.333333 GB/s | D | chapter 14a |
| Four splits, 96 blocks | utilization 96/108, ceiling 1133.333333 GB/s | D | chapter 14a |
| One full-attention layer, 32,768 tokens | 134,217,728 bytes (4,096 bytes per token) | D | chapter 7, chapter 14a |
| That read at 1275 GB/s, at 283.333333 GB/s, and at 1133.333333 GB/s | 105.2688 µs, 473.7096 µs, 118.4274 µs | D | chapter 14a |
| Reduction bytes, 4 splits, 24 heads, head dim 256, fp32 | 98,688 bytes, fraction 7.352829e-4 of the KV read | D | chapter 14a |
| `choose_num_splits(1, 24, kv, 108)` | 1 at kv 64, 2 at 65, 3 at 192, 4 at 256 and at 65,536 | D | chapter 14a |
| `choose_num_splits(1, 108, 65536, 108)` and `(1, 87, 65536, 108)` | 1 | D | chapter 14a |
| `choose_num_splits(1, 86, 65536, 108)` | 5 | D | chapter 14a |
| `choose_num_splits(1, 1, 65536, 108)` | 92 | D | chapter 14a |
| Worked merge, scores 0, 2, 1 and values 4, 6, 10 | output 6.798853, log-sum-exp 2.407606 | D | chapter 14a |
| Unscaled mean of partials 5.761594 and 10 | 7.880797 | D | chapter 14a |
| Flash-Decoding, batch 1 versus 108 SMs | less than 1% of the GPU, their grid | C | [FLASH-DECODING] |
| Flash-Decoding on CodeLlama-34B | up to 8× end to end, attention up to 50× versus FlashAttention, attention time roughly flat to 32k | C | [FLASH-DECODING] |
| A100 non-matmul FP32 versus FP16/BF16 matmul | 19.5 TFLOP/s versus 312 TFLOP/s, a factor of 16 | C | [FA2] |
| FlashAttention-1 forward pass on A100 | 30% to 50% of peak FLOP/s, their measurement | C | [FA2] |
| FlashAttention-2 on H100, as measured in FlashAttention-3 | 35% utilization | C | [FA3] |
| Hopper features FlashAttention-3 uses | TMA, asynchronous WGMMA, FP8 tensor cores | C | [FA3] |
| FlashAttention-3 FP16 on H100 SXM5 | 1.5× to 2.0× over FlashAttention-2, up to 740 TFLOP/s | C | [FA3] |
| Split heuristic fractions | return 1 at 80% of the slots, then the smallest count within 85% of the highest wave efficiency, cap 128 | C | [FA-HEURISTIC] |
| Split-KV tile when head dimension is greater than 128 | 64 tokens | C | [FA-HEURISTIC] |
| FlashInfer attention state | $(O, \mathrm{LSE})$ composes by the log-sum-exp merge; `plan` then `run` on a fixed grid | C | [FLASHINFER] |

| Key | Page |
|---|---|
| [FLASH-DECODING] | Tri Dao, Daniel Haziza, Francisco Massa, and Grigory Sizov, Flash-Decoding for long-context inference. Stanford CRFM, 12 October 2023. https://crfm.stanford.edu/2023/10/12/flashdecoding.html |
| [FA2] | Tri Dao, FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. https://arxiv.org/abs/2307.08691 |
| [FA3] | Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao, FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. https://arxiv.org/abs/2407.08608 |
| [FLASHINFER] | Zihao Ye and others, FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving. https://arxiv.org/abs/2501.01005 |
| [FA-HEURISTIC] | `num_splits_heuristic` in FlashAttention v2.6.3, `csrc/flash_attn/flash_api.cpp`. https://github.com/Dao-AILab/flash-attention/blob/v2.6.3/csrc/flash_attn/flash_api.cpp |

## Chapter 14

Numbers added to the FlashAttention chapter by the grid and FlashAttention-2 and -3 sections. The SM count is 108, from chapter 10. A D row is arithmetic in the chapter and in `content/labs/14-flash-attention/solution.py` (`count_programs`, `sm_fill`, `rescale_elements`). None is an A100 kernel measurement. The chapter quotes no value for `speedup_decode_32k`, because no A100 run of that shape went into this revision. The paper keys are defined in the Chapter 14a section.

| Claim | Value | Tag | Source |
|---|---|---|---|
| Programs launched, $\lceil q_\text{len}/B_r \rceil \times$ batch $\times$ query heads, block_m 128 | 512 tokens, 8 heads: 32 programs. 8192 tokens, 8 heads: 512. 32,768 tokens, 24 heads: 6144. Decode, batch 1, 24 heads: 24. Decode, batch 4, 24 heads: 96 | D | chapter 14 |
| SM fill in the first wave of 108 | 32/108 = 0.2963, 24/108 = 0.2222, 96/108 = 0.8889, 1.0 for 512 and 6144 programs | D | chapter 14 |
| Waves of 108 slots | 512 programs: 5 waves, efficiency 0.9481. 6144 programs: 57 waves, efficiency 0.9981 | D | chapter 14 |
| Scaled accumulator elements per head, L 8192, causal, 128x64 tiles, d 128 (4160 tile steps, 64 query tiles) | FlashAttention-1: 136,314,880. FlashAttention-2: 69,206,016, ratio 0.5077 | D | chapter 14 |
| Causal matmul FLOPs per head at L 8192 | 17.18 GFLOP, as 4 L^2 d / 2 | D | chapter 14 |
| Rescale term priced at 16 matmul FLOPs each | about 6.4% (FlashAttention-2) and 12.7% (FlashAttention-1) of the matmul work, an estimate that omits the exponentials | D | chapter 14 |
| FlashAttention-2 output update | keeps an unscaled output and divides once at the end of the loop; stores the log-sum-exp instead of the max and the sum | C | [FA2] section 3.1.1 |
| FlashAttention-2 grid | FlashAttention-1 runs batch size times heads blocks, efficient when that is large (say 80 or more) on 108 SMs; FlashAttention-2 also parallelizes over the sequence length | C | [FA2] section 3.2 |
| FlashAttention-2 loop order | outer loop over row blocks, inner over column blocks; first suggested and implemented by Phil Tillet in Triton | C | [FA2] section 3.2 |
| FlashAttention-2 causal masking | skips blocks above the diagonal, about 1.7 to 1.8 times faster than without the mask | C | [FA2] section 3.1.1 |
| FlashAttention-2 forward benchmark on A100 | around 2 times FlashAttention, up to 230 TFLOP/s (73% of peak) | C | [FA2] abstract and section 4.1 |
| FlashAttention-3 exponential throughput on H100 SXM5 | 989 TFLOP/s FP16 matmul against 3.9 TFLOP/s special functions; exponentials can take 50% of the matmul cycle at head dimension 128 | C | [FA3] section 3.1 |
| FlashAttention-3 pingpong scheduling | FP16 forward, head dimension 128, length 8192: from 570 to 620-640 TFLOP/s | C | [FA3] section 3.1 |
| FlashAttention-3 warp specialization | producer warps issue TMA, consumer warps issue WGMMA, registers reallocated with `setmaxnreg` | C | [FA3] section 3.1 |
| FlashAttention-3 FP8 | block quantization (one scale per block of Q, K, V); incoherent processing multiplies Q and K by a random orthogonal matrix, ±1 diagonals times a Hadamard matrix, O(d log d); up to 2.6 times lower error than per-tensor FP8 | C | [FA3] section 3.3 and abstract |
| FlashAttention-3 FP8 throughput | close to 1.2 PFLOP/s | C | [FA3] abstract |

## Chapter 13a

Numbers introduced by the skinny-decode chapter. The arithmetic peak is 312 TFLOP/s and the rated bandwidth is 1935 GB/s, both from chapter 10. A time here is a byte count divided by that bandwidth, evaluated in `scripts/figures/13a-skinny-decode-gemm.py`. It is not an A100 kernel measurement. The layer is the up projection, K = 5120 and N = 17408.

| Claim | Value | Tag | Source |
|---|---|---|---|
| Bfloat16 bytes, FLOPs, and intensity at M = 1 | 178,302,976 bytes, 178,257,920 FLOPs, 0.999747 | D | chapter 13a |
| Same product, memory floor at 1935 GB/s and compute floor at 312 TFLOP/s | 92.1462 µs and 0.5713 µs | D | chapter 13a |
| Bfloat16 intensity at M = 8, 64, and 4096 | 7.983856, 62.981187, 2012.485549 | D | chapter 13a |
| Bfloat16 bound at those four batches, ridge 161.240310 | memory, memory, memory, compute | D | chapter 13a |
| Bfloat16 bound flips | M = 168 memory (161.156762), M = 169 compute (162.076732) | D | chapter 13a |
| Output omitted at M = 1 and at M = 4096 | 178,268,160 bytes and 220,200,960 bytes | D | chapter 13a |
| W4A16 weight, group 128, no zero point | 45,957,120 bytes, 33/64 bytes per weight | D | chapter 18, chapter 13a |
| W4A16 bytes and intensity at M = 1 | 46,002,176 bytes, intensity 3.874989, floor 23.7737 µs | D | chapter 13a |
| W4A16 intensity at M = 8, 64, and 4096 | 30.788822, 233.586045, 3167.565570 | D | chapter 13a |
| W4A16 bound flips | M = 43 memory (160.041050), M = 44 compute (163.609023) | D | chapter 13a |
| W4A16 at M = 64 against the 244.7 copy ridge | 233.586 is compute bound at 161.2 and memory bound at 244.7 | D | chapter 10, chapter 13a |
| Dequantization included at M = 1, one extra multiply per weight | intensity 5.812483 | D | chapter 13a |
| Column tiles of 128 on N = 17408 | 136 programs | D | chapter 13a |
| Worked int4 product, activation (1, 1, 1, 1) | output (1, 15.5, 7.25) | D | chapter 13a |
| Ampere FP16 FLOP-per-byte ratio, as stated by MARLIN | typically 100 to 200 | C | [MARLIN] |
| MARLIN 4-bit vs FP16 on an A10, their measurement | near 4× up to batch 16 to 32, toward 1.5× at batch 128 | C | [MARLIN] |
| MARLIN in vLLM, their measurement | up to 2.8× at batch 16 | C | [MARLIN] |
| CS336 lectures cited here | Lecture 2, resource accounting; Lecture 6, Kernels, Triton | C | [CS336] |
| CS 229S lecture cited here | 4 October 2023, Hardware Aware Algorithm Design | C | [CS229S] |

| Key | Page |
|---|---|
| [MARLIN] | Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten Hoefler, and Dan Alistarh, MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models. https://arxiv.org/abs/2408.11743 |
| [CS336] | Stanford CS336, Language Modeling from Scratch, schedule. Lecture 2 is resource accounting. Lecture 6 is Kernels, Triton. https://cs336.stanford.edu/ |
| [CS229S] | CS 229S calendar, Fall 2023. 4 October, Hardware Aware Algorithm Design. https://cs229s.stanford.edu/fall2023/calendar/ |
| [TRITON-GEMM] | Triton tutorial, Matrix Multiplication. https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html |

## Chapter 18a

Numbers introduced by the int4 and KV chapter. Errors are computed by `scripts/figures/18a-int4-and-kv-quantization.py` on synthetic tensors. They are not an A100 measurement, and they are not a perplexity for Qwen3.8-27B. The W4A16 byte count $33/64$ is chapter 13a's. This chapter does not recompute that product's roofline.

| Claim | Value | Tag | Source |
|---|---|---|---|
| Worked int32 word for codes $(1,-2,7,0,-8,3,-1,4)$, low nibble first | `-944730263` (`0xc7b08f69`) | D | chapter 18a |
| Short rows $(1,-2,7)$ and $(-8,4,0)$ | words $3945$ and $2240$ | D | chapter 18a |
| Bytes per weight, group 128, one 2-byte scale | $0.5 + 2/128 = 33/64 = 0.515625$ | D | chapter 18a, chapter 13a |
| Bytes per weight, 10 weights, group 4 | $0.5 + 6/10 = 1.1$ | D | chapter 18a |
| Toy row $(0.20, 0.10, 0.40)$, seed 0, correlation 0.9 | round-to-nearest codes $(4,2,7)$ error $0.0620$; GPTQ codes $(4,1,7)$ error $0.0197$ | D | chapter 18a |
| Seeded layer, $\rho = 0.95$, group 8 | round-to-nearest $0.041674$, GPTQ $0.015605$, margin $0.026069$ | D | chapter 18a |
| Lab GPTQ gate | error greater than $0.005$, margin at least $0.01$ | D | chapter 18a |
| AWQ wide layer, seed 4, errors at $\alpha = 0, 0.25, 0.5, 0.75, 1$ | $0.330230$, $0.139580$, $0.065805$, $0.036474$, $0.099206$ | D | chapter 18a |
| AWQ alphas selected by the brute-force grid | $0.75$ on the wide layer, $0.25$ on the narrow layer (seed 1) | D | chapter 18a |
| Seeded head, relative attention error | per-token key $0.277654$, per-channel key $0.071986$, per-token value $0.079503$, per-channel value $0.147489$ | D | chapter 18a |
| Per-token key error divided by per-token value error | $0.277654 / 0.079503 = 3.492$ | D | chapter 18a |
| E4M3 bias, max finite, min normal, min subnormal | $7$, $448$, $2^{-6}$, $2^{-9}$; no infinity; NaN is `S.1111.111` | C | [FP8] |
| E5M2 bias, max finite, min normal, min subnormal | $15$, $57344$, $2^{-14}$, $2^{-16}$; infinity at exponent field all ones and mantissa 0 | C | [FP8] |
| E4M3 cast, round ties to even | $1.0625 \to 1$, $1.1875 \to 1.25$, $464 \to 448$, $480 \to \mathrm{NaN}$ | D | chapter 18a |
| E5M2 cast, round ties to even | $1.125 \to 1$, $60000 \to 57344$, $61440 \to \infty$ | D | chapter 18a |
| GPTQ on OPT-175B, 4-bit WikiText-2, as reported | GPTQ loses $0.03$ perplexity; round-to-nearest drops $2.2$ | C | [GPTQ] |
| GPTQ damping | $1\%$ of the mean Hessian diagonal | C | [GPTQ] |
| AWQ scale | $s = s_X^{\alpha}$, $\alpha$ searched on $[0, 1]$, $s_X$ the per-channel mean activation magnitude | C | [AWQ] |
| KIVI axis choice | per-channel keys, per-token values | C | [KIVI] |

| Key | Page |
|---|---|
| [GPTQ] | Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh, GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. https://arxiv.org/abs/2210.17323 |
| [AWQ] | Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han, AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration. https://arxiv.org/abs/2306.00978 |
| [SMOOTHQUANT] | Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han, SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. https://arxiv.org/abs/2211.10438 |
| [KIVI] | Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu, KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. https://arxiv.org/abs/2402.02750 |
| [FP8] | Paulius Micikevicius, Dusan Stosic, Patrick Judd, John Kamalu, Stuart Oberman, Mohammad Shoeybi, Michael Siu, Hao Wu, Neil Burgess, Sangwon Ha, Richard Grisenthwaite, Naveen Mellempudi, Marius Cornea, Alexander Heinecke, and Pradeep Dubey, FP8 Formats for Deep Learning. https://arxiv.org/abs/2209.05433 |
| [MIT6.5940] | MIT 6.5940, TinyML and Efficient Deep Learning Computing, Fall 2024, schedule. Lecture 5, Quantization (Part I), 19 September. Lecture 6, Quantization (Part II), 24 September. Lecture 13, Efficient LLM Deployment, 22 October. https://hanlab.mit.edu/courses/2024-fall-65940 |

## Chapter 18

Numbers introduced by the lab's SmoothQuant check. They come from seeded synthetic tensors on the CPU (`torch.manual_seed(18)`, 128 tokens, 512 inputs, 256 outputs, weight scale 0.05, channels 7, 150, and 333 multiplied by 60), not from an A100 measurement. Activations use one int8 scale per tensor and weights use one int8 scale per output row. The scale is $c_k = x_{\max,k}^{\alpha} / w_{\max,k}^{1-\alpha}$, with $w_{\max}$ taken over the output rows.

| Claim | Value | Tag | Source |
|---|---|---|---|
| W8A8 relative output error with no planted outliers | $0.012403$ | D | chapter 18 |
| Same layer with three planted outlier channels, no smoothing | $0.099090$ | D | chapter 18 |
| Smoothed at $\alpha = 0$, $0.5$ | $0.088486$, $0.014136$ | D | chapter 18 |
| Wrong direction: activations times $c$, weight divided by $c$, at $\alpha = 0.5$ | $0.218857$ | D | chapter 18 |
| Product error of $(x / c)(c W)$ against $xW$ | about $8.5 \times 10^{-8}$ relative, float32 | D | chapter 18 |
| SmoothQuant balanced point $\alpha = 0.5$ for OPT and BLOOM | as reported | C | [SMOOTHQUANT] |
| FP8 cast rounding and ranges (E4M3 max 448, E5M2 max 57344) | see chapter 18a | C | [FP8] |

## Chapter 10

Numbers introduced by the launch-floor model in the roofline chapter. Each is
arithmetic on the chapter's A100 rates (312 TFLOP/s, 1935 GB/s) and its 5 µs
launch cost, evaluated by `scripts/figures/10-roofline.py` through the lab's
reference solution. None is an A100 measurement. The lab's CUDA timing of 400
tiny kernels reports `tiny_eager_ms` and `tiny_graph_ms` on a device and has no
expected value.

| Claim | Value | Tag | Source |
|---|---|---|---|
| Kernel time model | per kernel, max of 5 µs launch and roofline floor; step is the sum | D | chapter 10 |
| Floor of a batch-1 RMSNorm, 20,480 bytes | 10.6 ns | D | chapter 10 |
| Floor of one of 271 weight kernels sharing 53.8 GB, intensity 1 | 0.1026 ms | D | chapter 10 |
| 400-kernel decode step, 129 norms and 271 weight kernels, eager | 28.449 ms, memory bound | D | chapter 10 |
| Same step in one graph, one 5 µs launch | 27.810 ms | D | chapter 10 |
| Launch time a graph removes from that step | 0.639 ms, 2.2% of eager | D | chapter 10 |
| 400 norm-sized kernels, eager | 2.000 ms, launch bound | D | chapter 10 |
| Three norm-sized kernels and the one-token MLP, per-kernel sum against one max over the step | 0.2914 ms against 0.2764 ms | D | chapter 10 |

## Chapter 17

Numbers introduced by the coordinated-omission, in-process client, and knee sections. They come from the lab's own computation in `content/labs/17-benchmark/tests.py`, not from a source. None is a GPU measurement.

| Claim | Value | Tag | Source |
|---|---|---|---|
| Planted stall: 1000 requests of 5 ms, request 100 takes 5 s, sent every 10 ms; closed-loop p99 | 0.005 s, one slow sample | D | chapter 17 |
| Same stall, open loop, latency from the intended send time; request 100+k latency | 5 - 0.005k s for k = 0 to 899; p99 4.95005 s | D | chapter 17 |
| Tene named the effect of a client that stops sending while the server stalls | coordinated omission | C | [TENE] |
| Engine trace: 10-tick steps of 1 ms, A at 0 s with 3 tokens, B at 0.005 s with 2 tokens | B first token 0.020 s, TTFT 0.015 s | D | chapter 17 |
| Sweep: 4 slots, 10 tokens at 10 ms a step, SLO TTFT 100 ms and ITL 15 ms, 300 Poisson arrivals, seed 7 | slot capacity 40 req/s; G 0.997, 0.960, 0.813, 0.033 at 10, 20, 30, 40 req/s | D | chapter 17 |
| Knee at threshold 0.9 on that sweep | 20 req/s | D | chapter 17 |
| vLLM `vllm bench serve` flags | `--request-rate`, `--num-prompts`, `--seed` | C | [VLLM-BENCH] |

| Key | Page |
|---|---|
| [TENE] | Gil Tene, How NOT to measure latency. https://www.youtube.com/watch?v=lJ8ydIuPFeU |
| [VLLM-BENCH] | vLLM benchmark CLI. https://docs.vllm.ai/en/latest/benchmarking/cli/ |

## Chapter 19

Numbers introduced by the interconnect-bandwidth correction and the ring all-reduce latency section. A D row is arithmetic in the chapter, in `scripts/figures/19-tensor-parallelism.py`, and in `content/labs/19-tensor-parallel/solution.py` (`ring_all_reduce`, `all_reduce_time`, `step_all_reduce_time`). None is an A100 measurement. The hop latency of 4 µs is an assumption chosen so that a two-GPU all-reduce costs the chapter's 8 µs. The weight read of 42.2 ms per step is chapter 14a's 53.8 GB at 1275 GB/s.

| Claim | Value | Tag | Source |
|---|---|---|---|
| A100 NVLink rating | 600 GB/s; the datasheet does not say whether this counts both directions | C | [A100-DS] |
| A100 NVLink links and per-link rate | 12 links, 25 GB/s in each direction per link | C | [AMPERE-BLOG] |
| A100 NVLink, one direction | 300 GB/s (12 × 25); the 600 GB/s rating is both directions combined | D | [AMPERE-BLOG] |
| PCIe Gen4 as rated for A100 | 64 GB/s; this is the both-directions figure, so 32 GB/s one way | C | [A100-DS] |
| NVLink bridge for PCIe A100 | two GPUs only | C | [A100-DS] |
| Hop latency alpha | 4 µs, an assumption | A | chapter 19 |
| Ring all-reduce time | 2(N-1) alpha + 2(N-1)/N × bytes / bandwidth | D | chapter 19 |
| Decode message, hidden 5120, bfloat16, one token | 10,240 bytes | D | chapter 19 |
| That message at N = 2, 4, 8 | 8.03, 24.05, 56.06 µs; latency is 99.6%, 99.8%, 99.9% | D | chapter 19 |
| Prefill message, 4096 tokens | 41,943,040 bytes; at N = 2, 147.8 µs, latency 5.4% | D | chapter 19 |
| Crossover message size, N alpha beta | 2.4, 4.8, 9.6 MB at N = 2, 4, 8 | D | chapter 19 |
| 128 decode all-reduces per token at N = 2, 4, 8 | 1.028, 3.079, 7.176 ms | D | chapter 19 |
| Decode step with a plain ring at N = 2, 4, 8 | 22.1, 13.6, 12.5 ms; speedup 1.91, 3.10, 3.39 | D | chapter 19 |
| Prefill comm at N = 2, 5.37 GB, bandwidth only | 17.9 ms over NVLink at 300 GB/s, 168 ms over PCIe at 32 GB/s | D | chapter 19 |
| Same, as a share of 353 ms of compute | 5.1% over NVLink, 48% over PCIe | D | chapter 19 |

Tag **A** marks an assumption.

| Key | Page |
|---|---|
| [A100-DS] | NVIDIA A100 Tensor Core GPU datasheet, 2021. Interconnect row: NVLink Bridge for 2 GPUs 600 GB/s, PCIe Gen4 64 GB/s. https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf |
| [AMPERE-BLOG] | NVIDIA Ampere Architecture In-Depth, NVIDIA Technical Blog. "A single A100 NVLink provides 25-GB/second bandwidth in each direction." The total number of links is 12. https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/ |

## Chapter 20

Numbers introduced by the batch-dependent cost model and the rollback example
in the speculative decoding chapter. Each is arithmetic on the chapter's A100
rates (53.8 GB of weights, 26.9 billion active parameters, 312 TFLOP/s,
1935 GB/s) and on chapter 15's 154,927,104 bytes of recurrent state per
sequence, evaluated by `content/labs/20-spec-decode/solution.py`. None is an
A100 measurement.

| Claim | Value | Tag | Source |
|---|---|---|---|
| Plain decode step at batch 1, weight read at the rated bandwidth | 27.804 ms | D | chapter 20 |
| Compute time per token | 0.1724 ms; the two terms cross at 161.2 tokens | D | chapter 20 |
| Speedup at α 0.8, γ 4, c 0.05, batch 1 and batch 32 | 2.80 and 2.80 (160 verify tokens, under the ridge) | D | chapter 20 |
| Same configuration at batch 128 | 0.806; verify pass 110.4 ms against a 27.8 ms plain step | D | chapter 20 |
| Break-even batch, α 0.8, γ 4, c 0.05 | 101 (continuous 101.96); speedup 1.009 at 101 and 0.9996 at 102 | D | chapter 20 |
| Break-even batch, α 0.5 γ 4; α 0.8 γ 2; α 0.8 γ 8; α 0.9 γ 4 | 56; 125; 70; 125 | D | chapter 20 |
| Break-even batch with a 154,927,104-byte state snapshot per sequence | α 0.5 γ 4: 47; α 0.8 γ 2: 96; α 0.8 γ 4: 85; α 0.8 γ 8: 63; α 0.9 γ 4: 105 | D | chapter 20 |
| Snapshot copy time per sequence per round at 1935 GB/s | 0.160 ms; speedup at batch 32 falls from 2.80 to 2.43 | D | chapter 20 |
| Scalar rollback example, decay 0.9, start 2.0, writes 1.0, 0.5, 2.0, 1.5 | state after four steps 5.7462; correct state after keeping two inputs 3.02 | D | chapter 20 |

# Fact entries for chapter 11a

This section records numbers introduced by the constrained-decoding chapter.
A number below is a claim from a cited page (tag C), or arithmetic from
`content/labs/11a-constrained-decoding/solution.py`, printed by
`scripts/figures/11a-constrained-decoding.py` (tag D). Nothing here is an A100
measurement. The timing rows are pure-Python CPU timings on the author's machine.

| Claim | Value | Tag | Source |
|---|---|---|---|
| Outlines recasts generation as moves between states of a finite-state machine and builds an index over the vocabulary | stated in the abstract | C | [OUTLINES] |
| XGrammar splits the vocabulary into context-independent tokens, which are prechecked, and context-dependent tokens, which are interpreted at run time | stated in the abstract | C | [XGRAMMAR] |
| XGrammar co-designs the grammar engine with the inference engine so grammar work overlaps with GPU execution | stated in the abstract | C | [XGRAMMAR] |
| SGLang jump-forward: forced single-path FSM transitions are appended as one string, and the whole sequence is re-tokenized, which adds about 4% overhead | as stated in the blog | C | [SGLANG-CFSM] |
| SGLang compressed FSM, llama-7B on an A10 | up to 2x lower latency at batch 1, up to 2.5x higher throughput | C | [SGLANG-CFSM] |
| llguidance mask time, 128k tokenizer | on the order of 50 µs of CPU time per token | C | [LLGUIDANCE] |
| vLLM structured outputs backends and fields | xgrammar or guidance backends; `structured_outputs` with `choice`, `regex`, `json`, `grammar`, `structural_tag` | C | [VLLM-SO] |
| Toy vocabulary of the lab | 296 tokens: end, pad, 256 single bytes, 38 multi-character | D | chapter 11a |
| Worked draw, logits 4, 2, 1, 3, 0 for maybe, y, yes, n, no, T=1 | unconstrained 0.6364, 0.0861, 0.0317, 0.2341, 0.0117; masked 0, 0.2369, 0.0871, 0.6439, 0.0321 | D | chapter 11a |
| Legal tokens for `(yes\|no)` at the start, after `y`, after `ye`, after `yes` | 5, 2, 1, 1 of 296 | D | chapter 11a |
| JSON pattern `\{"name": "[a-z]{1,8}", "age": (0\|[1-9][0-9]{0,2})\}` | 34 DFA states; 2 legal tokens at the start; 10 forced bytes in 3 tokens; 19 of 24 bytes forced in `{"name": "bo", "age": 7}` | D | chapter 11a |
| `nested_array_regex` depths 1 to 5, unminimized subset DFA | 7, 17, 37, 77, 157 states | D | chapter 11a |
| Mask size at 248,320 tokens | 248,320 B per row as bool, 31,040 B bit-packed | D | chapter 11a |
| Mask size at batch 128 | 31,784,960 B bool, 3,973,120 B packed | D | chapter 11a |
| Copy time at 32 GB/s one way (chapter 19's PCIe figure), batch 128 | 0.993 ms bool, 0.124 ms packed; 2.4% of the 42.2 ms decode weight read (53.8 GB at 1275 GB/s) | D | chapter 11a |
| Mask cache for 34 states at 248,320 tokens, bool | 8,442,880 B | D | chapter 11a |
| Mask build, pure Python, random 248,320-token vocabulary | about 34 ms; cached lookup about 0.07 µs; CPU timing, varies by machine | D | chapter 11a |
| 250 seeded completions of the JSON pattern | 250/250 pass `re.fullmatch` and `json.loads`; the sampled token path differs from greedy longest-match encoding in 250/250 | D | chapter 11a |
| `[0-9]{1,3}` as the age | accepts `007` under `re.fullmatch`; `json.loads` rejects it | D | chapter 11a |

| Key | Page |
|---|---|
| [OUTLINES] | Willard and Louf, Efficient Guided Generation for Large Language Models, arXiv:2307.09702. https://arxiv.org/abs/2307.09702 |
| [XGRAMMAR] | Dong et al., XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models, arXiv:2411.15100. https://arxiv.org/abs/2411.15100 |
| [SGLANG-CFSM] | Fast JSON Decoding for Local LLMs with Compressed Finite State Machine, LMSYS blog, 2024-02-05. https://lmsys.org/blog/2024-02-05-compressed-fsm/ |
| [LLGUIDANCE] | llguidance README. https://github.com/guidance-ai/llguidance |
| [VLLM-SO] | vLLM, Structured Outputs. https://docs.vllm.ai/en/latest/features/structured_outputs/ |

## Fact entries for chapter 07a

Numbers introduced by the KV-economy chapter. The DeepSeek-V2 dimensions and the quoted paper claims are confirmed against the cited paper text. Every other number is arithmetic on those dimensions or on this course's geometry (16 full-attention layers, 24 query heads, 4 KV heads, head width 256, bfloat16), evaluated by `content/labs/07a-kv-economy/solution.py` and printed by `scripts/figures/07a-kv-economy-attention-variants.py`. The MLA rows for this course's model are hypothetical, because the model has no MLA layers. The sink and H2O numbers come from seeded synthetic data. None is an A100 measurement.

| Tag | Meaning |
|---|---|
| **C** | Confirmed in the cited paper. |
| **D** | Derived. The arithmetic is in the chapter and in the figure script. |
| **S** | Seeded synthetic data from the lab. It shows a mechanism and says nothing about a real model. |

| Claim | Value | Tag | Source |
|---|---|---|---|
| DeepSeek-V2 layers, hidden size, heads, head width | 60, 5120, 128, 128 | C | [DSV2] section 3.1.2 |
| DeepSeek-V2 latent width d_c, query latent d_c', RoPE key width d_r | 512, 1536, 64 | C | [DSV2] section 3.1.2 |
| DeepSeek-V3 layers, hidden size, heads, head width | 61, 7168, 128, 128 | C | [DSV3] hyperparameters |
| DeepSeek-V3 d_c, d_c', d_r | 512, 1536, 64 | C | [DSV3] hyperparameters |
| MLA cache per token | (d_c + d_r) l elements, about 9/2 d_h l; equal to GQA with 2.25 groups | C | [DSV2] section 2.1.4, Table 1 |
| MLA scores scale with the decoupled key | 1 / sqrt(d_h + d_r) | C | [DSV2] equation 18 |
| W_UK absorbs into W_Q, W_UV into W_O | stated as possible during inference | C | [DSV2] section 2.1.2 |
| DeepSeek-V2 KV cache reduction against DeepSeek 67B | 93.3%, as reported; not reproduced here | C | [DSV2] abstract |
| DeepSeek-V2 elements per token, MHA and MLA | 1,966,080 and 34,560; ratio 56.89 | D | chapter 07a |
| DeepSeek-V2 bytes per token in bfloat16, MHA and MLA | 3,932,160 and 69,120 (67.5 KiB) | D | chapter 07a |
| DeepSeek-V3 MLA bytes per token | 61 x 576 x 2 = 70,272 | D | chapter 07a |
| Decoupled key share of the MLA cache | 64 / 576 = 11.1% | D | chapter 07a |
| MLA on this geometry at d_c 1024, d_r 128, and at d_c 512, d_r 64 | 2304 and 1152 bytes per layer per token against GQA's 4096; ratios 1.78 and 3.56 | D | chapter 07a |
| MLA bytes per token on this geometry, 16 layers | 36 KiB and 18 KiB against GQA's 64 KiB | D | chapter 07a |
| DeepSeek-V2 absorbed FLOPs per cached token per layer, and the fixed part | 278,528 and 33,554,432; equal at T = 120.5 | D | chapter 07a |
| DeepSeek-V2 decode attention at T = 4096, 60 layers, absorbed and naive | 70,464,307,200 and 8,266,469,867,520 FLOPs; ratio 117.3 | D | chapter 07a |
| DeepSeek-V2 absorbed intensity at T = 4096, and for large T | 248.889 and 241.778 FLOPs per byte; ridge 161.2 | D | chapter 07a |
| DeepSeek-V2 rated-A100 floor per cached token per layer, MHA and absorbed MLA | 33.869 ns and 0.893 ns; ratio 37.9 | D | chapter 07a |
| Merged W_UV and W_O against separate, per layer, V2 | 335,544,320 against 92,274,688 entries | D | chapter 07a |
| MLA absorbed intensity on this geometry, 24 heads | 45.3 for large T, 45.667 at 32,768 | D | chapter 07a |
| Window 4096 plus 4 sinks, this model, 32,768 tokens | 4100 slots, 268,697,600 bytes (256.25 MiB) against 2 GiB; ratio 7.992 | D | chapter 07a |
| Half local (4096) and half global, this model, 32,768 tokens | 1152 MiB | D | chapter 07a |
| Mistral 7B window, layers, theoretical span | 4096, 32, about 131K tokens | C | [MISTRAL] |
| Mistral 7B rolling buffer | position i mod W; 8x less cache at 32k tokens | C | [MISTRAL] |
| Gemma 2 local window and global span, alternating in every other layer | 4096 and 8192 | C | [GEMMA2] section 2 |
| StreamingLLM claims | stable at up to 4 million tokens; up to 22.2x over sliding-window recomputation; window attention fails when the text exceeds the cache | C | [STREAMING] abstract |
| Planted sink head, seed 1, 200 tokens, d 16, window 32 | mass on token 0 is 0.9999; error 1.069187 with 0 sinks, 0.000096 with 1, 0.000094 with 4 | S | chapter 07a |
| H2O heavy hitters and policy | accumulated attention follows a power law; keeps recent and heavy-hitter tokens, budget split evenly in the baselines | C | [H2O] |
| H2O budget claim | comparable performance with under 20% of the KV budget, on the paper's models | C | [H2O] |
| Planted H2O layers, 128 tokens, budget 24, recent 8, seeds 0 and 5 | oracle 100.201 and 99.967; H2O 97.816 and 97.500 (0.9762, 0.9753); window 4.094 and 3.616 (0.0409, 0.0362) | S | chapter 07a |
| SnapKV selection | observation window of 16, max pooling kernel 5, top-k per head, prompt-only | C | [SNAPKV] sections 4.1 and 5.1.1 |
| SnapKV claims | 3.6x generation speedup and 8.2x memory efficiency at 16K-token inputs | C | [SNAPKV] abstract |
| This model's attention FLOPs per decode token at 32,768, 16 layers | 12.885 GFLOP for MHA and GQA; 1.612 for window plus sinks; 7.248 for half local; 55.163 and 27.582 for the two MLA rows | D | chapter 07a |

## Sources for chapter 07a

| Key | Page |
|---|---|
| [DSV2] | DeepSeek-AI, DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. https://arxiv.org/abs/2405.04434 |
| [DSV3] | DeepSeek-AI, DeepSeek-V3 Technical Report. https://arxiv.org/abs/2412.19437 |
| [MISTRAL] | Jiang et al., Mistral 7B. https://arxiv.org/abs/2310.06825 |
| [STREAMING] | Xiao et al., Efficient Streaming Language Models with Attention Sinks. https://arxiv.org/abs/2309.17453 |
| [H2O] | Zhang et al., H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. https://arxiv.org/abs/2306.14048 |
| [SNAPKV] | Li et al., SnapKV: LLM Knows What You Are Looking for Before Generation. https://arxiv.org/abs/2404.14469 |
| [GEMMA2] | Gemma Team, Gemma 2: Improving Open Language Models at a Practical Size. https://arxiv.org/abs/2408.00118 |

## Chapter 17b

Numbers introduced by the disaggregated-serving chapter. Each is arithmetic on
chapter 2's byte counts, chapter 19's link rates, and chapter 17a's step-time
floor, or the output of the simulator in
`content/labs/17b-disaggregation/solution.py`, printed by
`scripts/figures/17b-disaggregated-serving.py`. None is an A100 or a network
measurement. Link rates are one way.

| Claim | Value | Tag | Source |
|---|---|---|---|
| Bytes a prefill GPU ships for the hybrid | 65,536 per prompt token plus a fixed 154,927,104 (147.75 MiB) | D | chapter 17b |
| Same, 128, 2,048, 8,192, 131,072 tokens | 155.75, 275.75, 659.75, 8,339.75 MiB | D | chapter 17b |
| Dense stack (64 full-attention layers) | 262,144 per token, no state; 2,048 MiB at 8,192 tokens | D | chapter 17b |
| Prompt length where hybrid and dense ship equal bytes | 788 tokens; KV equals the state at 2,364 | D | chapter 17b |
| NVLink per A100, one way | 300 GB/s (the 600 GB/s rating adds both directions) | C | [A100-DS], [AMPERE-BLOG] |
| PCIe 4.0 x16, one way | 32 GB/s | C | [A100-DS] |
| InfiniBand 4x port, HDR and NDR | 200 and 400 Gbit/s, which is 25 and 50 GB/s | C | [IB-WIKI] |
| Ethernet at 100 and 25 Gbit/s | 12.5 and 3.125 GB/s, line rate divided by 8, no protocol overhead | A | chapter 17b |
| Bulk copy of 8,192 tokens at 300, 50, 32, 25, 12.5, 3.125 GB/s | 2.31, 13.84, 21.62, 27.67, 55.34, 221.38 ms | D | chapter 17b |
| Prefill floor, 128, 512, 2,048, 8,192 tokens | 30.39, 91.03, 360.09, 1,466.09 ms | D | chapter 17a, 17b |
| Layer-wise streamed wait, 8,192 tokens at 300 GB/s | 0.112 ms (the last full-attention layer, 33,554,432 bytes) | D | chapter 17b |
| Least link rate, 8,192 tokens, bulk within 10% of the prefill | 4.719 GB/s | D | chapter 17b |
| Least link rate, 8,192 tokens, streamed within 5 ms | 6.711 GB/s | D | chapter 17b |
| Decode step, 8 decodes at context 2,048 | 31.652 ms; with a 160-token chunk 32.53 ms (1.03x); with 512 tokens 93.76 ms (2.96x) | D | chapter 17b |
| Chunk size where compute passes the decode byte floor | about 164 tokens | D | chapter 17b |
| Planted trace, 8 decoders plus an 8,192-token prompt at 2 s: worst gap, whole prompt / 512 / 160 chunk / disaggregated | 1,467.48 / 97.37 / 32.79 / 31.60 ms | D | chapter 17b |
| Same trace, long prompt's TTFT, whole prompt / 160 / 128 chunk / disaggregated | 1.485 / 1.699 / 2.063 / 1.468 s | D | chapter 17b |
| Prefill-heavy trace, 8 GPUs, 8,192 in, 32 out, 3 requests/s, seed 7, TTFT 1.6 s, worst gap 36 ms | best split 7:1 at 2.226 goodput/s; best colocated chunk 1.423/s (4,096); 56% higher | D | chapter 17b |
| Decode-heavy trace, 8 GPUs, 128 in, 512 out, 8 requests/s, seed 11, TTFT 1.0 s, worst gap 40 ms | best split 1:7 at 3.486/s; colocated 3.499/s at every chunk size | D | chapter 17b |
| Short-prompt trace (128 in, 16 out, 2 requests/s, seed 5): median TTFT | 30.39 ms colocated; 30.93 ms NVLink; 43.45 ms 100 Gbit/s Ethernet; 82.65 ms 25 Gbit/s Ethernet | D | chapter 17b |
| vLLM disaggregated prefill | experimental; "does not improve throughput" | C | [VLLM-DISAGG] |

Tag **A** marks an assumption.

| Key | Page |
|---|---|
| [IB-WIKI] | InfiniBand, Wikipedia, data rates table. HDR 200 Gbit/s and NDR 400 Gbit/s at 4x. https://en.wikipedia.org/wiki/InfiniBand |
| [VLLM-DISAGG] | vLLM documentation, Disaggregated Prefilling. https://docs.vllm.ai/en/latest/features/disagg_prefill/ |
| [DISTSERVE] | Zhong et al., DistServe. https://arxiv.org/abs/2401.09670 |
| [SPLITWISE] | Patel et al., Splitwise. https://arxiv.org/abs/2311.18677 |
| [MOONCAKE] | Qin et al., Mooncake. https://arxiv.org/abs/2407.00079 |
| [SARATHI-SERVE] | Agrawal et al., Sarathi-Serve. https://arxiv.org/abs/2403.02310 |

# Fact entries for chapter 20a

Numbers introduced by the tree speculation chapter. Rows tagged D are
arithmetic on the chapter's A100 rates (53.8 GB of weights, 26.9 billion active
parameters, 312 TFLOP/s, 1935 GB/s) and on chapter 15's 154,927,104 bytes of
recurrent state per sequence, evaluated by
`content/labs/20a-tree-speculation/solution.py` and printed by
`scripts/figures/20a-tree-speculation.py`. None is an A100 measurement. The
acceptance table comes from a Zipf target and a noisy draft, in closed form, so
it is not a measured acceptance rate. Rows tagged C are what the cited paper's
abstract or section reports, and this course did not reproduce them.

| Tag | Meaning |
|---|---|
| **C** | Confirmed in the cited page. |
| **D** | Derived. The arithmetic is in the chapter and in the figure script. |

| Claim | Value | Tag | Source |
|---|---|---|---|
| Seven-node tree, parents [-1, 0, 0, 1, 1, 2, 3], committed length 7 | positions [7, 8, 8, 9, 9, 9, 10]; 18 of 49 mask entries set; lower-triangular mask would set 28 | D | chapter 20a |
| Same tree as separate path runs | 10 tokens (paths of length 3, 3, 4) against 7 as a tree | D | chapter 20a |
| Branch acceptance, p (0.5, 0.3, 0.2), q (0.2, 0.3, 0.5), 3 candidates | a = (0.7, 0.06, 0.048); total 0.808 against 0.7 for one | D | chapter 20a |
| Multi-candidate rule is exact for candidates drawn iid from q | induction on candidates left | D | chapter 20a, [SPECINFER] |
| Lab statistical check, first two emitted tokens, 30,000 streams | total variation 0.0100 from the target's joint, 0.4419 from the draft's | D | chapter 20a |
| Expected accepted drafts, tree [-1, 0, 0, 1, 1], r1 (0.6, 0.3), r2 (0.5, 0.2) | path probabilities 0.6, 0.3, 0.3, 0.12; sum 1.32 | D | chapter 20a |
| Best tree against a chain on the chapter's table, budgets 1, 4, 6, 8, 12, 16 | 0.795, 1.775, 1.899, 1.996, 2.146, 2.258 against 0.795, 1.775, 1.867, 1.867, 1.867, 1.867 (the table has six levels) | D | chapter 20a |
| Best-tree greedy equals exhaustive search | three tables, budgets 1 to 8, to 1e-9 | D | chapter 20a |
| Batch-1 optimum draft length at c 0.05 | continuous 3.088, 5.480, 7.857, 13.292 for alpha 0.5, 0.7, 0.8, 0.9; integer 3, 6, 8, 13 | D | chapter 20a |
| Adaptive draft length, alpha 0.8, c 0.05, batches 1, 8, 16, 32, 64, 96, 128, 160 | 8, 8, 8, 4, 2, 1, 1, 0 | D | chapter 20a |
| Largest batch with speedup of at least 1, alpha 0.8, c 0.05 | 101 for a fixed draft length 4; 141 for the adaptive length | D | chapter 20a |
| Node budget at c 0.05, batches 1, 4, 8, 16, 32, 64, 128 | 32, 32, 19, 9, 4, 2, 1 | D | chapter 20a |
| Same with one 154,927,104-byte state written per node | 32, 10, 3, 3, 3, 1, 0 | D | chapter 20a |
| Per-node state, 17 nodes | 2,633,760,768 bytes, 2.453 GiB per sequence; 1.361 ms of writes at 1935 GB/s; 0.0801 ms per node | D | chapter 20a |
| Same at batch 32 | 43.6 ms of writes against a 27.80 ms plain step | D | chapter 20a |
| Flattened root-to-leaf paths of the chapter's best 16-node tree | 37 tokens against 17 as a tree | D | chapter 20a |
| Medusa speedup | about 2.2 times (Medusa-1), 2.3 to 3.6 times (Medusa-2) | C | [MEDUSA] |
| EAGLE speedup on LLaMA2-Chat 70B | 2.7 to 3.5 times latency, distribution unchanged | C | [EAGLE] |
| EAGLE-2 speedup | 3.05 to 4.26 times, from draft confidence as an acceptance estimate | C | [EAGLE2] |
| EAGLE-3 speedup | up to 6.5 times, about 1.4 times EAGLE-2; 1.38 times throughput at batch 64 in SGLang | C | [EAGLE3] |
| DeepSeek-V3 MTP | depth D = 1 (section 4.2); second-token acceptance 85% to 90% and 1.8 times TPS (section 5.4.3) | C | [DSV3] |

Tag **A** marks an assumption. This chapter has none beyond the ones stated in
its text: the draft costs 0.05 target passes per tree level, and a tree
verification writes one recurrent state per node.

| Key | Page |
|---|---|
| [MEDUSA] | Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads, arXiv 2401.10774. |
| [EAGLE] | EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty, arXiv 2401.15077. |
| [EAGLE2] | EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees, arXiv 2406.16858. |
| [EAGLE3] | EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test, arXiv 2503.01840. |
| [DSV3] | DeepSeek-V3 Technical Report, arXiv 2412.19437, sections 2.2, 4.2, and 5.4.3. |
| [SPECINFER] | SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification, arXiv 2305.09781; the multi-step speculative sampling procedure. |

# Fact entries for chapter 10b

Numbers in the profiling chapter. A D row is arithmetic on the hand-built traces in
`content/labs/10b-profiling/fixture_traces.py` or on a hardware limit, printed by
`scripts/figures/10b-profiling.py`. A C row is a claim from a cited page. Nothing here is a GPU
measurement, and the fixtures are not profiler exports from a device.

| Claim | Value | Tag | Source |
|---|---|---|---|
| `torch.profiler` exports a Chrome trace with `traceEvents`; complete events carry `ph` `"X"`, `ts`, `dur`, `pid`, `tid`, `cat` and `name`; `cpu_op` events are `aten::` calls | field names checked by exporting a CPU trace with torch 2.11 | C | [PT-PROFILER] |
| GPU categories in a real export: `kernel`, `gpu_memcpy`, `gpu_memset`, `gpu_user_annotation`, `cuda_runtime` | from the Kineto trace format; only `cpu_op` and `Trace` were observed locally | C | [PT-PROFILER] |
| A100 (compute capability 8.0) limits | 64 resident warps, 64 K 32-bit registers, 2,048 threads, 32 shared-memory banks, 255 registers per thread | C | [CUDA-CC] |
| Occupancy at 32, 64, 128, 255 registers per thread | 64, 32, 16, 8 warps; 100%, 50%, 25%, 12.5% | D | chapter 10b |
| Sectors per request, 32 threads reading 4 B at stride 1, 2, 4, 8 floats | 4, 8, 16, 32 | D | chapter 10b |
| Bank conflict reading a 32x32 float tile by column | 32-way at row pitch 32, none at pitch 33 | D | chapter 10b |
| Launch-bound fixture | busy 70/193 = 0.3627; 19 gaps (ten of 6 µs, nine of 7 µs), 123 µs idle; median kernel 3.5 µs | D | chapter 10b |
| Sync-bound fixture | busy 810/930 = 0.8710; one 120 µs gap, cause `host_sync` | D | chapter 10b |
| Two-stream fixture | busy 210/260 = 21/26 = 0.8077; summing durations gives 1.0 | D | chapter 10b |
| Launch floor, 400 kernels at 5 µs | 2.000 ms (chapter 10's constant, not remeasured) | D | chapter 10, chapter 10a |
| Copy ceiling, measured over rated bandwidth | 1275/1935 = 65.9% | D | chapter 10 |
| Regression example: baseline 10.0, 10.1, 9.9, 10.0, 10.2 vs 10.0, 10.1, 9.9, 10.0, 30.0 | means 10.04 and 14.00 (+39.4%); medians 10.0 and 10.0 | D | chapter 10b |
| Overhead-bound as a regime: time goes to Python, dispatch, and launching kernels | stated in the essay | C | [BRRR] |
| CS336 lecture list has no profiling lecture: lecture 2 "PyTorch (einops), resource accounting (FLOPs, memory, arithmetic intensity)", lecture 6 "Kernels, Triton" | as listed on the syllabus page; Assignment 2 asks for profiling | C | [CS336] |
| Nsight Compute metric names in the lab | follow the documented naming structure; the exact strings were not found on the fetched Profiling Guide text, so confirm with `ncu --query-metrics` | C (unverified) | [NCU-GUIDE] |

| Key | Page |
|---|---|
| [PT-PROFILER] | PyTorch, torch.profiler. https://docs.pytorch.org/docs/stable/profiler.html |
| [CUDA-CC] | CUDA Programming Guide, compute capabilities. https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html |
| [NCU-GUIDE] | Nsight Compute Profiling Guide. https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html |
| [BRRR] | Horace He, Making Deep Learning Go Brrrr From First Principles. https://horace.io/brrr_intro.html |
| [CS336] | Stanford CS336 syllabus. https://cs336.stanford.edu/ |

# Fact entries for chapter 19b

This section records numbers introduced by the pipeline and context parallelism chapter. A number below is a claim from a cited page (tag C), or arithmetic from `content/labs/19b-pipeline-and-ring/solution.py`, printed by `scripts/figures/19b-pipeline-and-context-parallelism.py` (tag D). Nothing here is a multi-GPU or A100 measurement. The hardware inputs are the course's: 1275 GB/s measured copy, 300 GB/s NVLink one way, 32 GB/s PCIe one way, 312 TFLOP/s peak, 4 µs per ring hop (chapter 19's assumption), 53.8 GB of weights, and 65,536 bytes of KV per token across 16 full-attention layers.

| Claim | Value | Tag | Source |
|---|---|---|---|
| Fill-and-drain pipeline, one time unit per stage per micro-batch | `m + p - 1` slots, bubble `(p-1)/(m+p-1)`; table counted by the lab | D | chapter 19b |
| Bubble at p=4 for m=1, 4, 16 | 0.7500, 0.4286, 0.1579 | D | chapter 19b |
| Bubble at p=2, m=8, and at p=8, m=32 | 0.1111 and 0.1795 | D | chapter 19b |
| Interleaved form `(p-1)/(v m + p - 1)` at p=4, m=16, v=2 and 4 | 0.0857 and 0.0448 | D | chapter 19b |
| Megatron-LM interleaved schedule | proposed in the paper; the authors state a throughput gain of more than 10% at comparable memory | C | [MEGATRON-SCALE] |
| Decode stage time at 1275 GB/s, 53.8 GB over p stages | 21.098 ms at p=2, 10.549 ms at p=4 | D | chapter 19b |
| One-token hop, 10,240 bytes at 300 GB/s plus 4 µs | 4.034 µs | D | chapter 19b |
| Decode inter-token latency through the pipeline | 42.204 ms at p=2, 42.212 ms at p=4, independent of micro-batches in flight | D | chapter 19b |
| Decode utilization and trips per second, p=4, 1 and 4 in flight | 0.25 at 23.69, 1.00 at 94.76 | D | chapter 19b |
| Fixed batch split into m micro-batches at p=4 | step 42.20, 52.75, 73.84, 116.04 ms at m=1, 2, 4, 8 | D | chapter 19b |
| Causal ring, contiguous, 64 tokens, 4 ranks | pairs per rank 136, 392, 648, 904; imbalance 1.7385; critical path 904; efficiency 0.5752 | D | chapter 19b |
| Causal ring, striped, same setup | pairs per rank 496, 512, 528, 544; imbalance 1.0462; critical path 544; efficiency 0.9559 | D | chapter 19b |
| Causal ring, zigzag, same setup | 520 pairs per rank; imbalance 1.0; efficiency 1.0 | D | chapter 19b |
| Efficiency at 2 and 8 ranks, 64 tokens | contiguous 0.6701 and 0.5372; striped 0.9848 and 0.9028 | D | chapter 19b |
| Ring attention error against full attention, seed 19, 2 heads, dim 16 | at most 3.6e-7 for every layout | D | chapter 19b |
| Per-rank KV memory in the ring | resident 1/P; peak 2/P with one receive buffer; peak ratio 1.0 at P=2, 0.5 at P=4, 0.25 at P=8 | D | chapter 19b |
| Shard length where a balanced causal block's compute equals its transfer | 346.67 tokens at 300 GB/s, 3,250 tokens at 32 GB/s | D | chapter 19b |
| KV cache for one request at 1,048,576 tokens | 64.0 GiB; per-GPU budget 69.11 GiB (53.8 GB of weights plus 19 GiB) | D | chapter 19b |
| Prefill FLOPs at 1,048,576 tokens | 2.7259e17, of which 5.6413e16 dense | D | chapter 19b |
| Decode floor at batch 1, 1,048,576 tokens, 4 GPUs | data 96.09 ms; tensor 27.10 ms; pipeline 96.11 ms; context 55.87 ms | D | chapter 19b |
| Tensor-parallel all-reduce part of that floor | 3.079 ms for 128 all-reduces at P=4 | D | chapter 19b |
| Prefill bytes sent per rank at 1,048,576 tokens, 4 GPUs | tensor 2,061.58 GB; pipeline 10.74 GB; context ring 51.54 GB | D | chapter 19b |
| Prefill compute floor at the 312 TFLOP/s peak, 4 GPUs | 218.4 s without a bubble; 259.4 s for pipeline with 16 chunks; 873.7 s on one GPU | D | chapter 19b |
| Context-parallel decode merges | 16 all-gathers, 0.196 ms in total | D | chapter 19b |
| Ulysses against the ring, bytes per rank over 16 layers at 1,048,576 tokens, P=4 | 90.19 GB against 51.54 GB, under the assumption that Ulysses all-to-alls Q, K, V, and the output | D | chapter 19b |
| Llama 3 context parallelism | the input is split into 2 x CP chunks and rank i gets chunks i and 2 x CP - 1 - i | C | [LLAMA3] section 3.3.2 |
| Llama 3 uses an all-gather of K and V, not a ring | stated reasons: masks such as the document mask are easier, and K and V are much smaller than Q under GQA | C | [LLAMA3] section 3.3.2 |
| Striped Attention | assigns each device tokens spread uniformly through the sequence; up to 1.45x end-to-end throughput over Ring Attention at 256k tokens on A100 and TPUv4 | C | [STRIPED] |
| Ring Attention | overlaps K and V block communication with blockwise attention; supports sequences up to the device count times longer | C | [RING] |
| DeepSpeed Ulysses | sequence split plus all-to-all for attention; constant communication volume when sequence length and device count grow proportionally | C | [ULYSSES] |
| Stanford CS336, Spring 2026 | Lecture 7 "Parallelism" and Lecture 8 "Parallelism" on the course schedule | C | [CS336] |

| Key | Page |
|---|---|
| [RING] | Liu, Zaharia, and Abbeel, Ring Attention with Blockwise Transformers for Near-Infinite Context, arXiv:2310.01889. https://arxiv.org/abs/2310.01889 |
| [STRIPED] | Brandon et al., Striped Attention: Faster Ring Attention for Causal Transformers, arXiv:2311.09431. https://arxiv.org/abs/2311.09431 |
| [ULYSSES] | Jacobs et al., DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models, arXiv:2309.14509. https://arxiv.org/abs/2309.14509 |
| [GPIPE] | Huang et al., GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism, arXiv:1811.06965. https://arxiv.org/abs/1811.06965 |
| [MEGATRON-SCALE] | Narayanan et al., Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM, arXiv:2104.04473. https://arxiv.org/abs/2104.04473 |
| [LLAMA3] | Llama Team, The Llama 3 Herd of Models, arXiv:2407.21783, section 3.3.2. https://arxiv.org/abs/2407.21783 |
| [CS336] | Stanford CS336 schedule, fetched for this chapter. https://cs336.stanford.edu/ |

# Fact entries for chapter 19a

This section records numbers introduced by the mixture-of-experts chapter. A
number below is a claim from a cited page (tag C), or arithmetic from
`content/labs/19a-moe/solution.py` and the Qwen3-30B-A3B config, printed by
`scripts/figures/19a-moe-and-expert-parallelism.py` (tag D). Routing statistics
come from seeded simulations (seed 19) with synthetic routers, not from a
trained router. Nothing here is an A100 measurement. Link rates and the 4 µs hop
latency are chapter 19's.

| Claim | Value | Tag | Source |
|---|---|---|---|
| Qwen3-30B-A3B config | hidden 2048, 48 layers, 128 experts, top-8, expert width 768, 32 query heads, 4 KV heads, head dim 128, vocabulary 151,936, `norm_topk_prob` true, bfloat16, untied embeddings | C | [QWEN3-CFG] |
| DeepSeek-V3 config | hidden 7168, 61 layers, first 3 dense, 256 routed and 1 shared expert, top-8, expert width 2048, `n_group` 8, `topk_group` 4, sigmoid scoring, `topk_method` `noaux_tc` | C | [DSV3-CFG] |
| DeepSeek-V3 size | 671B total parameters, 37B activated per token | C | [DSV3] |
| Mixtral | 8 experts per layer, 2 per token, 47B total, 13B active | C | [MIXTRAL] |
| Switch expert capacity | (tokens per batch / experts) x capacity factor; overflowing tokens are not processed by the layer | C | [SWITCH] |
| MegaBlocks | recasts MoE as block-sparse operations and never drops tokens | C | [MEGABLOCKS] |
| EPLB | replicates heavily loaded experts and packs replicas onto GPUs; hierarchical policy for prefill, global policy for decode | C | [EPLB] |
| DeepEP | expert-parallel all-to-all dispatch and combine kernels, with FP8 dispatch | C | [DEEPEP] |
| vLLM fused MoE | Triton `fused_moe_kernel`; `moe_align_block_size` sorts and pads tokens by expert | C | [VLLM-FUSED] |
| Qwen3-30B-A3B expert size | 3 x 2048 x 768 = 4,718,592 parameters, 9,437,184 bytes (9.0 MiB) | D | chapter 19a |
| Qwen3-30B-A3B parameters | 30,532,122,624 total (28,991,029,248 in experts, 1,541,093,376 elsewhere); 3,353,032,704 active at top-8, 11.0% | D | chapter 19a |
| Qwen3-30B-A3B bytes | 61,064,245,248 (61.06 GB, 56.87 GiB); 23.13 GiB left on 80 GiB; KV 98,304 B per token; 252,635 tokens fit | D | chapter 19a |
| Expected distinct experts, E 128, k 8 | 8.00, 51.62, 111.77, 125.94, 127.97 at batch 1, 8, 32, 64, 128 | D | chapter 19a |
| Weights read per decode step | 6,706,065,408 B at batch 1; 53.71 GB at batch 32 (8.01 times batch 1); 61.05 GB at batch 128 | D | chapter 19a |
| Step floor at 1275 GB/s | 5.26 ms at batch 1, 42.13 ms at batch 32, 47.88 ms at batch 128; dense 27B 42.19 ms | D | chapter 19a |
| Expert-MLP intensity B k / D | 1.000 at batch 1, 2.290 at 32, 4.065 at 64, 8.002 at 128 | D | chapter 19a |
| Batch to reach the 161.24 ridge | 2,580 for E 128 k 8; 5,160 for E 256 k 8; 645 for E 8 k 2; dense 162 | D | chapter 19a |
| Skewed routing, Zipf 0.8, batch 32 | 88.61 distinct experts against 111.77 uniform; ridge batch unchanged at 2,580 | D | chapter 19a |
| Capacity drop, uniform routing, seed 19 | T 1024: 4.36% at factor 1.0, 0.01% at 1.25 (capacity 80, busiest expert 81); T 256: 8.84%, 1.66%; T 4096: 2.18%, 0.00% | D | chapter 19a |
| Padded bmm example | counts 5,0,17,2,0,9,1,4 at block 8: 192 rows for 38 real, 5.05 times | D | chapter 19a |
| All-to-all volume, 128 tokens, k 8, hidden 2048, bf16, 8 ranks | 3,670,016 B per rank per direction (3.5 MiB); 28,672 B per token | D | chapter 19a |
| All-to-all time at 300 GB/s one way, alpha 4 µs | 12.23 µs bandwidth, 28 µs latency, 40.23 µs total; 3.86 ms for 96 per step | D | chapter 19a |
| Imbalance, 1,024 tokens, 8 ranks, 16 experts each | uniform 1.049; Zipf 0.5 1.467; Zipf 0.8 1.758 | D | chapter 19a |
| EPLB plan on Zipf 0.8 | packing only 1.172; 8 spare slots 1.001 | D | chapter 19a |
| Hand example, loads 9, 4, 2, 1, 6 slots, 2 ranks | contiguous 13 and 3 (1.625); plan replicas 2,2,1,1, rank loads 8.5 and 7.5 (1.0625) | D | chapter 19a |

| Key | Page |
|---|---|
| [QWEN3-CFG] | Qwen3-30B-A3B config.json. https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json |
| [DSV3-CFG] | DeepSeek-V3 config.json. https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json |
| [DSV3] | DeepSeek-AI, DeepSeek-V3 Technical Report, arXiv:2412.19437. https://arxiv.org/abs/2412.19437 |
| [MIXTRAL] | Jiang et al., Mixtral of Experts, arXiv:2401.04088. https://arxiv.org/abs/2401.04088 |
| [SWITCH] | Fedus et al., Switch Transformers, arXiv:2101.03961, Figure 3. https://arxiv.org/abs/2101.03961 |
| [MEGABLOCKS] | Gale et al., MegaBlocks, arXiv:2211.15841. https://arxiv.org/abs/2211.15841 |
| [EPLB] | DeepSeek EPLB README. https://github.com/deepseek-ai/EPLB |
| [DEEPEP] | DeepSeek DeepEP README. https://github.com/deepseek-ai/DeepEP |
| [VLLM-FUSED] | vLLM, fused_moe.py. https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py |
| [CS336] | Stanford CS336 Spring 2025 syllabus: Lecture 4, Mixture of experts, Thursday 10 April. http://cs336.stanford.edu/spring2025/ |

# Fact entries for chapter 18b

Numbers introduced by the sparsity and pruning chapter. A C row is a claim from a cited abstract. A D row is arithmetic from `content/labs/18b-sparsity/solution.py`, printed by `scripts/figures/18b-sparsity-and-pruning.py`. An S row is seeded synthetic data from the lab. Nothing here is an A100 measurement or a perplexity for Qwen3.8-27B. The 53.8 GB weight size and the 1275 GB/s copy rate are chapter 2's and chapter 0's. The byte rows are upper bounds that assume every byte is a prunable weight.

| Claim | Value | Tag | Source |
|---|---|---|---|
| Deep Compression stages and result | prune, trained quantization, Huffman coding; 35x to 49x storage reduction; pruning alone 9x to 13x fewer connections | C | [DEEPCOMP] |
| Wanda score | magnitude times the matching input activation norm; no retraining and no weight update; outperforms magnitude pruning and is competitive with weight-update methods on LLaMA and LLaMA-2 | C | [WANDA] |
| SparseGPT claims | at least 50% sparsity one-shot on GPT-family models; OPT-175B and BLOOM-176B in under 4.5 hours; 2:4 and 4:8 supported; negligible perplexity increase at 60% unstructured | C | [SPARSEGPT] |
| NVIDIA 2:4 claim | sparse tensor cores give twice the math throughput of dense units; a training workflow keeps accuracy across tasks and models | C | [NVSPARSE] |
| Minitron | 15B Nemotron-4 pruned and distilled to 8B and 4B; up to 40x fewer training tokens per model than from scratch | C | [MINITRON] |
| Sparse tensor core peak at twice the dense 312 TFLOP/s | 624 TFLOP/s | D | chapter 18b |
| MIT 6.5940 Fall 2024 lecture titles | 3 Pruning and Sparsity (Part I), 12 Sep; 4 Pruning and Sparsity (Part II), 17 Sep; 9 Knowledge Distillation, 3 Oct | C | [MIT6.5940] |
| Worked row $(0.9,-0.1,0.4,-0.7)$, column norms $(1,1,3,1)$ | Wanda scores $(0.9,0.1,1.2,0.7)$; magnitude keeps channels 0 and 3, Wanda keeps 0 and 2; relative error 0.846 against 0.615 | D | chapter 18b |
| Seeded layer, 32 by 64, 256 tokens, seed 18, channels 3, 17, 40, 58 scaled by 12 | mean column norm 191.16 on the outlier channels, 15.89 on the other 60 | S | chapter 18b |
| Row-wise relative output error at 25%, 50%, 75% | magnitude 0.115543, 0.248337, 0.549705; Wanda 0.030871, 0.091117, 0.178512 | S | chapter 18b |
| Wanda over magnitude error at 50% | 0.091117 / 0.248337 = 0.367 | S | chapter 18b |
| 2:4 relative output error | magnitude 0.332604, Wanda 0.123251; 0.332604 / 0.248337 = 1.339 | S | chapter 18b |
| Kept weights in the 128 outlier-channel slots at 50% | 65 for magnitude, 119 for Wanda | S | chapter 18b |
| Unstructured 50%, bfloat16 value plus 16-bit column index | 4 bytes per kept weight, 2 bytes per dense weight, the dense size | D | chapter 18b |
| 4096 by 4096 bfloat16, 2:4 compressed | dense 33,554,432 B; values 16,777,216 B; 2-bit indices 2,097,152 B; total 18,874,368 B; metadata is 1/8 of the values and 1/9 of the total | D | chapter 18b |
| Bytes per dense weight | dense bfloat16 2; 2:4 bfloat16 1.125; group-128 int4 0.515625 | D | chapter 18b, chapter 18a |
| Decode weight read, 53.8 GB at 1275 GB/s, upper bound | 42.196 ms dense; 30.262 GB and 23.735 ms for 2:4 (1.778x); 13.870 GB and 10.879 ms for int4 (3.879x) | D | chapter 18b |
| KD loss, teacher $(2,0)$, student $(0,0)$ | loss 0.327813, 0.443776, 0.484798 and gradient magnitude 0.380797, 0.462117, 0.489837 at T = 1, 2, 4 | D | chapter 18b |
| Same row at T = 1 | total variation 0.380797; acceptance 0.619203; Pinsker bound 0.404854 | D | chapter 18b |
| KD loss and gradient | $T^2$ times the batch-mean KL; gradient with respect to student logits $T(p^s - p^t)/B$ | D | [HINTON], chapter 18b |

## Sources for chapter 18b

| Key | Page |
|---|---|
| [DEEPCOMP] | Han et al., Deep Compression, arXiv:1510.00149. https://arxiv.org/abs/1510.00149 |
| [SPARSEGPT] | Frantar and Alistarh, SparseGPT, arXiv:2301.00774. https://arxiv.org/abs/2301.00774 |
| [WANDA] | Sun et al., A Simple and Effective Pruning Approach for Large Language Models, arXiv:2306.11695. https://arxiv.org/abs/2306.11695 |
| [NVSPARSE] | Mishra et al., Accelerating Sparse Deep Neural Networks, arXiv:2104.08378. https://arxiv.org/abs/2104.08378 |
| [MINITRON] | Muralidharan et al., Compact Language Models via Pruning and Knowledge Distillation, arXiv:2407.14679. https://arxiv.org/abs/2407.14679 |
| [HINTON] | Hinton, Vinyals, and Dean, Distilling the Knowledge in a Neural Network, arXiv:1503.02531. https://arxiv.org/abs/1503.02531 |
| [MIT6.5940] | MIT 6.5940 Fall 2024 schedule. https://hanlab.mit.edu/courses/2024-fall-65940 |
