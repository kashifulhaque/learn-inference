"""Chrome-trace fixtures for lab 10b, shaped like `torch.profiler` exports.

Field names follow what `prof.export_chrome_trace(...)` writes: a top-level
`traceEvents` list; complete events with `ph: "X"`, `ts` and `dur` in
microseconds, `pid`, `tid`; a `cat` of `kernel`, `gpu_memcpy`,
`gpu_user_annotation`, `cuda_runtime`, `cpu_op`, or `Trace`. GPU events use the
device id as `pid` and the stream id as `tid`. The traces are built by hand,
so every number a check expects is exact. None of them is a measurement.
"""

import json

T0 = 5000  # microseconds. A real trace uses a large epoch value.
LAUNCH_US = 5  # chapter 10's per-launch cost.


def _x(cat, name, ts, dur, pid, tid, **args):
    return {"ph": "X", "cat": cat, "name": name, "pid": pid, "tid": tid,
            "ts": ts, "dur": dur, "args": args}


def _kernel(name, ts, dur, stream=7):
    return _x("kernel", name, ts, dur, 0, stream, stream=stream, device=0)


def _wrap(events, span=None):
    """Add the metadata, instant, flow, and annotation events a real trace has.

    These are noise. A parser must not count them as GPU work.
    """
    noise = [
        {"ph": "M", "name": "process_name", "pid": 0, "tid": 0,
         "args": {"name": "python"}},
        {"ph": "M", "name": "process_name", "pid": 1, "tid": 0,
         "args": {"name": "GPU 0"}},
        {"ph": "i", "name": "Iteration Start: PyTorch Profiler", "s": "g",
         "pid": 1, "tid": 0, "ts": T0 - 10, "args": {}},
        {"ph": "s", "id": 1, "pid": 1, "tid": 1, "ts": T0, "cat": "ac2g",
         "name": "ac2g"},
    ]
    if span is not None:
        # PyTorch emits a gpu_user_annotation for every record_function range.
        # It covers the whole region and is not GPU work.
        noise.append(_x("gpu_user_annotation", "ProfilerStep#1", span[0],
                        span[1] - span[0], 0, 7))
    return {"schemaVersion": 1, "displayTimeUnit": "ms",
            "traceEvents": noise + events}


def healthy():
    """Ten 100 µs kernels back to back on one stream. Busy 1.0, no gaps."""
    ev = [_kernel("ampere_bf16_s16816gemm_bf16_128x128_nn", T0 + 100 * i, 100)
          for i in range(10)]
    return _wrap(ev, (T0, T0 + 1000))


def launch_bound():
    """Twenty tiny elementwise kernels, one launch every 10 µs.

    Even i is a 4 µs multiply, odd i is a 3 µs add. Kernel i starts at
    T0 + 10 + 10 i. The host issues each launch with a cudaLaunchKernel
    event of LAUNCH_US inside an aten op.
    """
    ev = []
    for i in range(20):
        mul = i % 2 == 0
        start = T0 + 10 + 10 * i
        ev.append(_x("cpu_op", "aten::mul" if mul else "aten::add",
                     start - 9, 8, 100, 1))
        ev.append(_x("cuda_runtime", "cudaLaunchKernel", start - 7,
                     LAUNCH_US, 100, 1))
        ev.append(_kernel("vectorized_elementwise_kernel_mul" if mul
                          else "vectorized_elementwise_kernel_add",
                          start, 4 if mul else 3))
    return _wrap(ev, (T0, T0 + 210))


def sync_bound():
    """Sixteen 50 µs GEMMs split by an `.item()` that blocks the host.

    GEMMs fill [T0, T0+400). A 10 µs reduce kernel ends at T0+410. The host
    waits in cudaStreamSynchronize until T0+410, runs 120 µs of Python, then
    launches eight more GEMMs from T0+530. The GPU is idle for those 120 µs.
    """
    ev = [_kernel("ampere_bf16_s16816gemm_bf16_128x128_nn", T0 + 50 * i, 50)
          for i in range(8)]
    ev.append(_kernel("reduce_kernel_sum", T0 + 400, 10))
    ev.append(_x("cpu_op", "aten::item", T0 + 295, 118, 100, 1))
    ev.append(_x("cuda_runtime", "cudaStreamSynchronize", T0 + 300, 110, 100, 1))
    ev.append(_x("cpu_op", "aten::gt", T0 + 420, 100, 100, 1))
    ev += [_kernel("ampere_bf16_s16816gemm_bf16_128x128_nn",
                   T0 + 530 + 50 * i, 50) for i in range(8)]
    return _wrap(ev, (T0, T0 + 930))


def multistream():
    """Overlap and a memcpy.

    Stream 7: kernel A [0, 100). Stream 8: kernel B [50, 150). Then a
    device-to-host memcpy [200, 260). The union is 150 + 60 = 210 of a
    260 µs window.
    """
    ev = [
        _kernel("kernel_A", T0, 100, 7),
        _kernel("kernel_B", T0 + 50, 100, 8),
        _x("gpu_memcpy", "Memcpy DtoH (Device -> Pageable)", T0 + 200, 60, 0, 9,
           bytes=1048576),
    ]
    return _wrap(ev, (T0, T0 + 260))


def to_json(trace):
    return json.dumps(trace)
