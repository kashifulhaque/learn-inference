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
| Four GPUs, context 1024, 1 s TPOT SLO, NVLink 600 GB/s, search to batch 4096 | degree 4, 16,217 tokens/s | D | chapter 17a |
| Same workload on PCIe 64 GB/s | degree 2 with two replicas, 13,175 tokens/s | D | chapter 17a |
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
