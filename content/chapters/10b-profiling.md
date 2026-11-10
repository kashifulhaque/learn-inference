---
title: Profiling and performance debugging
slug: 10b-profiling
part: "Part 3 — Making it fast"
summary: Read a profiler timeline for gaps, tell launch-bound from sync-bound, and turn Nsight Compute counters into a diagnosis and a fix.
minutes: 100
gpu: false
objectives:
  - Compute GPU busy fraction from a Chrome trace by merging overlapping intervals.
  - Tell a launch-bound timeline from a sync-bound one by the gap between kernels and what ends just before it.
  - Compute theoretical occupancy and sectors per request from a launch configuration and an access pattern.
  - Map Nsight Compute throughput, bank-conflict, spill, and sector counters to a bound and a fix.
  - Gate a performance regression with a median and a noise band.
lab: 10b-profiling
---

# Profiling and performance debugging

> [!TLDR]
> - Profile in two passes. Nsight Systems or `torch.profiler` shows the timeline and answers "is the GPU busy?". Nsight Compute shows one kernel's counters and answers "why is this kernel slow?".
> - GPU busy fraction is the union of GPU intervals over the window. A launch-bound trace in the lab is 36.3% busy (70 of 193 µs), and a trace with one `.item()` is 87.1% busy with one 120 µs gap that begins where the sync ends.
> - A fully coalesced warp read is 4 sectors per request. A stride of 8 floats is 32, which moves 1,024 bytes to use 128. A 128-register kernel allows 16 of an A100 SM's 64 warps, so 25% theoretical occupancy.
> - The two throughput numbers decide the first split: DRAM throughput high means memory bound, SM throughput high means compute bound, both low means latency bound. Spills, bank conflicts, and sector counts override that split.
> - A regression gate compares medians inside a noise band. One 30 ms outlier among five 10 ms runs moves the mean by 39% and the median by 0%.

[Chapter 10](/c/10-roofline) tells you what a kernel's floor should be, and [chapter 10a](/c/10a-cuda-graphs-and-launch-overhead) removes the launch term. Neither says how to find out which of those a slow step is paying. Every later kernel chapter says "measure", and this chapter is the measuring.

The method is narrow to wide, then back. Start with the timeline for the whole step, find the gap or the one fat kernel, then open that kernel's counters. You change one thing, and you run the same measurement again with a gate that knows the noise.

This chapter has no GPU run. The lab parses hand-built traces shaped like a real `torch.profiler` export, diagnoses planted counter sets, and profiles a small loop on the CPU. Every number below is arithmetic on those fixtures or on a hardware limit. None is an A100 measurement.

## Before you start

- **The launch cost, from [chapter 10](/c/10-roofline) and [10a](/c/10a-cuda-graphs-and-launch-overhead).** A launch is 5 µs in this course, so 400 kernels is $400 \times 5\,\mu\text{s} = 2.000\,\text{ms}$. The lab reuses that 5 µs. It does not measure a new one.
- **Rated versus measured bandwidth, from [chapter 10](/c/10-roofline).** The A100 rates 1935 GB/s and a plain copy measures 1275 GB/s, which is $1275/1935 = 65.9\%$ of the rating. A copy kernel that Nsight Compute shows at 66% of peak DRAM throughput is at its ceiling, not at a third of it.
- **Warps and sectors.** A warp is 32 threads that issue together. Global memory moves in 32-byte sectors, so a warp reading 4 bytes per thread from adjacent addresses touches $32 \times 4 = 128$ bytes, which is 4 sectors.
- **Streams.** Work on one CUDA stream runs in order, and two streams may overlap. A trace lane is a stream.

## Read the timeline first

This section answers how to measure how busy the GPU is, and what the gaps tell you.

`torch.profiler` exports a Chrome trace with `prof.export_chrome_trace(path)`. The file is JSON with a `traceEvents` list. A complete event has `ph` equal to `"X"`, a start `ts` and a length `dur` in microseconds, a `pid` and `tid`, and a `cat` that says what it is. GPU work is `kernel`, `gpu_memcpy`, and `gpu_memset`. The host side is `cpu_op` (an `aten::` call) and `cuda_runtime` (a call such as `cudaLaunchKernel` or `cudaStreamSynchronize`). The profiler also writes a `gpu_user_annotation` for each labelled range. That event spans the region and is not GPU work, so you must not count it. A real export has metadata (`ph: "M"`) and instant (`ph: "i"`) events too, and you skip them.

The lab's fixtures use these field names and categories. They are built by hand so that the answers are exact. They are not exports from a device.

Take the GPU events, sort by start, and merge any that overlap. Busy time is the length of the merged intervals. The window runs from the first GPU start to the last GPU end:

$$
\hla{f_{\text{busy}}} = \frac{\left|\bigcup_i [s_i,\, e_i]\right|}{\hlb{e_{\max} - s_{\min}}}
$$

$\hla{f_{\text{busy}}}$ is the share of the window with at least one GPU event running. $\hlb{e_{\max} - s_{\min}}$ is the window. The union matters. Adding durations counts two overlapping streams twice.

> [!EXAMPLE] Overlap, counted by hand
> The `multistream` fixture has kernel A on stream 7 from 0 to 100 µs, kernel B on stream 8 from 50 to 150 µs, and a device-to-host memcpy from 200 to 260 µs. The union is $[0,150] \cup [200,260]$, which is 150 + 60 = 210 µs. The window is 260 µs, so $f_{\text{busy}} = 210/260 = 21/26 = 0.8077$. A parser that adds durations gets $100+100+60 = 260$ and reports 1.0.

An idle gap is a stretch between merged intervals. The gap is where the GPU waits, and the next question is what the host was doing.

```viz
10b-trace-timeline
```

## Launch-bound and sync-bound

This section answers what the two common gap patterns look like and how to tell them apart.

*Launch-bound* means the kernels are so short that the host cannot launch the next one before the last one ends. Horace He's "Making Deep Learning Go Brrrr From First Principles" names this regime *overhead-bound*, and it says PyTorch runs ahead of the GPU only while the framework stays ahead. In a trace you see many tiny kernels, a steady gap between them, and a launch call for each.

The `launch_bound` fixture has 20 elementwise kernels, 10 multiplies of 4 µs and 10 adds of 3 µs, started 10 µs apart. Busy time is $10 \times 4 + 10 \times 3 = 70$ µs. The window is 193 µs. The 19 gaps are ten of 6 µs and nine of 7 µs, which sum to 123, and $70 + 123 = 193$.

$$
\boxed{f_{\text{busy}} = 70/193 = 36.3\%}
$$

Each launch costs 5 µs on the host in this course's constant, and these kernels last 3 to 4 µs. The median kernel is 3.5 µs, under twice the launch cost, so the host cannot keep up. The fix is the one from chapter 10a: fewer launches by fusing, or one replay by capturing a graph.

*Sync-bound* means the host waits for the GPU and the GPU then waits for the host. The usual cause is `.item()`, `.cpu()`, or a `print` of a tensor, each of which blocks until the stream drains. The `sync_bound` fixture has eight 50 µs GEMMs, a 10 µs reduce, a `cudaStreamSynchronize` that ends when the reduce ends, 120 µs of Python on the host, and eight more GEMMs. The GPU is busy for 810 of 930 µs:

$$
f_{\text{busy}} = 810/930 = 87.1\%
$$

That number looks fine. The gap is the information: one 120 µs hole that begins where the sync call ends. A launch-bound trace has many small gaps and no sync. A sync-bound trace has few large gaps, and a sync event ends at the start of each. The lab's rule marks a gap `host_sync` when a sync event ends from 10 µs before the gap starts up to the gap's end.

> [!KEY] The gap and what ends before it
> ==Look at what the host was doing when the GPU went idle.== A launch call per tiny kernel is launch-bound. A synchronize that ends where the gap starts is sync-bound. The fix for the first is fewer launches, and the fix for the second is to move the read off the step, as in chapter 10a.

> [!WARNING] A high busy fraction is not a clean bill
> The sync-bound trace is 87.1% busy and still loses 120 µs on every token that runs through it. Read the gap list as well as the fraction.

## Count in the kernel, with Nsight Compute

This section answers how to turn one kernel's counters into a bound and a fix.

Once the timeline points at one kernel, Nsight Compute profiles that kernel and reports counters. The lab takes a dict of these metrics. The names follow Nsight Compute's metric naming (unit, then pipe or counter, then a roll-up such as `.sum` or `.avg.pct_of_peak_sustained_elapsed`), which the Profiling Guide documents. Confirm each one on your own GPU with `ncu --query-metrics`, because names change between Nsight Compute versions.

| Metric | What it measures |
|---|---|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy, the share of the SM's warp slots in use |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM throughput as a percent of peak |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM throughput as a percent of peak |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` and `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum` | Shared-memory bank conflicts, and the wavefronts they sit in |
| `l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum` | Bytes stored to local memory, where spilled registers go |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` and `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | Sectors and requests for global loads |

Three of them need a number you can compute by hand.

**Coalescing.** A warp reads 4 bytes per thread. With adjacent addresses the 32 threads cover 128 bytes, which is 4 sectors per request. With a stride of $k$ floats the threads land in more sectors until each has its own:

| Stride in floats | Sectors per request | Bytes moved for 128 used |
|---|---|---|
| 1 | 4 | 128 |
| 2 | 8 | 256 |
| 4 | 16 | 512 |
| 8 or more | 32 | 1,024 |

A strided read wastes bandwidth and still shows high DRAM throughput. It is the trap in the diagnosis: the kernel looks memory bound, and the fix is not fewer bytes but the right bytes.

**Bank conflicts.** Shared memory has 32 banks of 4 bytes on an A100. Reading down a column of a 32 by 32 float tile with a row pitch of 32 puts every thread in bank $(32r) \bmod 32 = 0$, a 32-way conflict. Pitching the rows at 33 floats gives 32 distinct banks and no conflict. That is the one-column pad.

**Occupancy.** An A100 SM holds 64 warps and has 65,536 32-bit registers. A kernel that uses $r$ registers per thread fits $\lfloor 65536/(32r) \rfloor$ warps, capped at 64:

$$
\hla{w}(r) = \min\!\left(64,\ \left\lfloor \frac{65536}{32\, \hlb{r}} \right\rfloor\right)
$$

At $\hlb{r}=128$ that is 16 warps, 25% theoretical occupancy. At 64 registers it is 32 warps (50%), and at 255 it is 8 (12.5%). A kernel with low occupancy has too little work in flight to hide memory latency.

### The decision order

The lab's `diagnose` checks the rules in this order and returns the first hit. The thresholds are this lab's choices, not NVIDIA's:

1. **`register_spill`** when spill bytes exceed 4,096. Fix: cut register use with a smaller tile, `__launch_bounds__`, or a lower `maxrregcount`.
2. **`shared_bank_conflict`** when conflicts over wavefronts is at least 0.25. Fix: pad the shared tile by one column.
3. **`uncoalesced_global`** when sectors per request exceed 8. Fix: make adjacent threads read adjacent addresses, or stage a transpose through shared memory.
4. **`memory_bandwidth`** when DRAM throughput is at least 60% and above SM throughput. Fix: fuse neighbouring ops or use a narrower dtype.
5. **`compute`** when SM throughput is at least 60% and at least DRAM throughput. Fix: tensor cores and a narrower dtype.
6. **`low_occupancy`** when occupancy is under 30%. Fix: more resident warps.
7. **`latency_bound`** otherwise. Fix: more independent work per thread, more bytes in flight, or software pipelining.

The order is the point. Rules 1 to 3 name a specific defect, and a defect explains the throughput numbers. Reversing 3 and 4 labels a stride-8 read at 78% DRAM throughput as plain memory bound.

> [!EXAMPLE] The strided read that looks memory bound
> The lab's trap kernel has DRAM throughput 78%, SM throughput 15%, and 32,000 sectors over 1,000 requests, which is 32 per request. Rule 4 alone says `memory_bandwidth`. Rule 3 fires first and says `uncoalesced_global`. Fixing the stride cuts the bytes moved by up to 8 times, from 32 sectors to 4 per request, without touching the roofline.

> [!NOTE] Peak is not the copy rate
> Percent of peak is against the hardware peak. A well-tuned copy reaches about $1275/1935 = 65.9\%$ on this course's numbers, so treat roughly two-thirds as the ceiling for a pure streaming kernel before you decide there is headroom.

## Profile on the CPU, where you can

This section answers what you can check without a GPU.

`torch.profiler` with `ProfilerActivity.CPU` exports the same JSON shape with `cpu_op` events only. A loop of 60 iterations of `x * 1.01 + 0.1` shows up as at least 60 `aten::mul` and 60 `aten::add` events and one `aten::mm`. `many_small_ops(trace, min_count=50)` returns the first two and not the third. That is the many-small-ops pattern from the launch-bound section, seen from the host. The lab asserts counts, because the durations on a CPU depend on the machine. On a CPU-only trace `parse_trace` reports a busy fraction of 0.0, since there is no GPU event.

With a CUDA device the lab also profiles 20 matmul and tanh pairs and checks that `parse_trace` finds the kernels. That check skips without a device.

## Gate a regression

This section answers how to decide a change made a kernel slower rather than noisier.

Run the benchmark several times and compare medians. A run can hit a GC pause or a clock change, and one bad run should not fail the build. With a baseline of 10.0, 10.1, 9.9, 10.0, 10.2 ms and a new set of 10.0, 10.1, 9.9, 10.0, 30.0 ms, the means are 10.04 and 14.00, a 39.4% rise. Both medians are 10.0 ms.

$$
\hla{\Delta} = \frac{\operatorname{med}(\text{new}) - \operatorname{med}(\text{base})}{\operatorname{med}(\text{base})},
\qquad \text{regression if } \hla{\Delta} > \hlb{\tau}
$$

$\hla{\Delta}$ is the relative change, and $\hlb{\tau}$ is the noise band, 5% in the lab. For a throughput metric the sign flips: a drop of more than $\tau$ is the regression. A change of exactly $\tau$ is inside the band.

Pick $\tau$ from repeated runs of an unchanged build. The lab's 5% is a default, not a measurement of any machine.

## What goes wrong

- **Counting `gpu_user_annotation` or every `"X"` event as GPU work.** The annotation spans the region, so the busy fraction reads 1.0 on the launch-bound trace instead of 36.3%.
- **Summing durations.** Two overlapping streams count twice. The two-stream fixture reports 1.0 instead of 21/26.
- **Trusting the busy fraction alone.** The sync-bound trace is 87.1% busy. Read the gaps.
- **Checking DRAM throughput first.** A stride-8 read reaches 78% and reads as memory bound. Test sectors per request before throughput.
- **Comparing means.** One outlier in five runs trips the gate. Use the median.
- **Profiling a cold run.** The first iterations include allocator setup and autotuning. Warm up, then profile, as in chapter 10a.

> [!RECAP]
> - Busy fraction is the merged GPU interval length over the window. Launch-bound is 70/193 (36.3%). Sync-bound is 810/930 (87.1%) with one 120 µs gap after a sync.
> - Launch-bound has many small gaps, and the fix is fewer launches. Sync-bound has a few large gaps that begin where a sync ends, and the fix is to remove the read.
> - Coalesced is 4 sectors per request, and a stride of 8 floats is 32. A 128-register kernel is 25% theoretical occupancy on an A100.
> - Check spills, bank conflicts, and sectors before the throughput split.
> - Gate on medians with a noise band.

## Check your understanding

> [!QUESTION] A trace shows 20 kernels averaging 3.5 µs, started 10 µs apart. What is the busy fraction, and what is the fix?
> Busy time is 70 µs in a 193 µs window, which is 36.3%. The median kernel is under twice the 5 µs launch cost, so the step is launch bound. Fuse the ops or capture a CUDA graph.

> [!QUESTION] A step is 87% busy but has one 120 µs gap, and a `cudaStreamSynchronize` ends at its start. What do you look for?
> A host read of a device value, such as `.item()`, `.cpu()`, or a print. Remove it from the step or move it after the step.

> [!QUESTION] A kernel shows 78% DRAM throughput, 15% SM throughput, and 32 sectors per request. Is it memory bound?
> It is a coalescing defect. A coalesced warp read is 4 sectors per request, so this kernel moves 8 times the bytes it uses. Fix the access pattern first.

> [!QUESTION] A kernel uses 128 registers per thread. What is its theoretical occupancy on an A100?
> $65536/(32 \times 128) = 16$ warps of 64, which is 25%.

> [!QUESTION] Five runs went 10.0, 10.1, 9.9, 10.0, and 30.0 ms against a 10 ms baseline. Does a mean-based gate fail?
> Yes. The mean is 14.00 ms, up 39.4%. The median is 10.0 ms, up 0%, so a median gate passes.

## Lab

> [!TRY]
> Write `parse_trace`, `classify_trace`, `many_small_ops`, `diagnose`, and `compare_to_baseline`. You pass when the two-stream trace is 21/26 busy, the launch-bound trace is 70/193 with nineteen gaps, the sync-bound trace has one 120 µs `host_sync` gap, the seven planted kernels get their bound and a fix that names the right idea, and a single outlier does not trip the gate. The CPU profiler check runs on any machine. The GPU check skips without a device.

`parse_trace(trace, top_n=3, min_gap_us=1.0)` returns `gpu_busy_fraction`, `top_kernels`, `idle_gaps`, `window_us`, `kernel_count`, and `median_kernel_us`. `classify_trace(parsed)` returns `sync_bound`, `launch_bound`, or `healthy`. The starter docstrings give the exact rules.

## Further reading

- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html). `export_chrome_trace` and the activities the lab uses.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html). The metric structure, sections, and the memory tables.
- [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html). The timeline view for kernel gaps and host calls.
- [CUDA programming guide, compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html). The 64 warps, 64 K registers, and 32 banks the occupancy and bank arithmetic use.
- Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html). Compute-, bandwidth-, and overhead-bound.
- [CS336 syllabus](https://cs336.stanford.edu/). Its lecture list has no profiling lecture. Lecture 2 is "PyTorch (einops), resource accounting (FLOPs, memory, arithmetic intensity)" and lecture 6 is "Kernels, Triton". Assignment 2 asks you to profile and benchmark the model.
