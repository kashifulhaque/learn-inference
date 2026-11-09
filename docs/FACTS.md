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
