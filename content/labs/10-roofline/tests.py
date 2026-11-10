from lab_common import Checks, pick_device

HIDDEN, INTERMEDIATE = 5120, 17408
HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256

# The decode step from chapter 10's launch section: 400 kernels at batch 1.
# 129 RMSNorms move 2 * 5120 * 2 = 20,480 bytes each at intensity 1. The other
# 271 kernels share the 53.8 GB of weights and also run at intensity 1.
NORM_BYTES = 2 * HIDDEN * 2
NORMS, WEIGHT_KERNELS, WEIGHT_BYTES = 129, 271, 53.8e9


def decode_step():
    norm = (float(NORM_BYTES), float(NORM_BYTES))
    weight = (WEIGHT_BYTES / WEIGHT_KERNELS, WEIGHT_BYTES / WEIGHT_KERNELS)
    return [norm] * NORMS + [weight] * WEIGHT_KERNELS


def _fmt(value):
    return f"{value:.6f} ms"


def _near(name, actual_fn, expected, tol=1e-6):
    """Return a (passed, message) pair that shows expected and actual."""
    try:
        actual = actual_fn()
    except Exception as exc:  # noqa: BLE001
        return False, f"expected {_fmt(expected)}, got {type(exc).__name__}: {exc}"
    ok = isinstance(actual, (int, float)) and abs(actual - expected) <= tol
    return ok, f"expected {_fmt(expected)}, got {actual!r}"


def run(submission):
    c = Checks()
    try:
        _run_checks(c, submission)
    except Exception as exc:  # noqa: BLE001
        # An unfinished stub raises. Report it as a failed check, not a crash.
        c.add("the roofline calculator runs", False,
              f"expected every function to return a value, got {type(exc).__name__}: {exc}")
    return c.finish()


def _run_checks(c, submission):
    needed = ("matmul_flops", "mlp_cost", "decode_attention_cost",
              "arithmetic_intensity", "ridge_point", "roofline")
    if not c.require(submission, *needed):
        return

    c.check("a matmul counts two FLOPs per multiply-add",
            lambda: submission.matmul_flops(4, 8, 16) == 2 * 4 * 8 * 16,
            f"got {submission.matmul_flops(4, 8, 16)}")

    ridge = submission.ridge_point()
    c.check("the A100 80GB PCIe ridge point is about 161 FLOPs/byte",
            lambda: abs(ridge - 161.2) < 2.0, f"got {ridge:.1f}")

    flops1, bytes1 = submission.mlp_cost(1, HIDDEN, INTERMEDIATE)
    expected_flops = 3 * 2 * HIDDEN * INTERMEDIATE
    c.check("one token through the MLP does 3 matmuls' worth of FLOPs",
            lambda: flops1 == expected_flops,
            f"got {flops1:,}, expected {expected_flops:,}")
    c.check("the MLP reads 535 MB of weights",
            lambda: abs(bytes1 - 3 * HIDDEN * INTERMEDIATE * 2) < 1e6,
            f"got {bytes1 / 1e6:.1f} MB")

    intensity_1 = submission.arithmetic_intensity(*submission.mlp_cost(1, HIDDEN, INTERMEDIATE))
    intensity_2048 = submission.arithmetic_intensity(
        *submission.mlp_cost(2048, HIDDEN, INTERMEDIATE))

    c.check("decode at batch 1 sits near intensity 1",
            lambda: 0.5 < intensity_1 < 4, f"got {intensity_1:.2f}")
    c.check("prefill at 2048 tokens is far above the ridge point",
            lambda: intensity_2048 > 5 * ridge, f"got {intensity_2048:.0f}")

    r1 = submission.roofline(*submission.mlp_cost(1, HIDDEN, INTERMEDIATE))
    r2048 = submission.roofline(*submission.mlp_cost(2048, HIDDEN, INTERMEDIATE))
    c.check("the MLP at batch 1 is memory bound",
            lambda: r1["bound_by"] == "memory", f"got {r1['bound_by']}")
    c.check("the MLP at 2048 tokens is compute bound",
            lambda: r2048["bound_by"] == "compute", f"got {r2048['bound_by']}")
    c.check("the floor is the larger of the two limits",
            lambda: abs(r1["floor_ms"] - max(r1["compute_ms"], r1["memory_ms"])) < 1e-9)

    # Attention during decode: intensity is the GQA group size times two, and it
    # does not move with context length.
    short = submission.arithmetic_intensity(
        *submission.decode_attention_cost(1024, HEADS, KV_HEADS, HEAD_DIM))
    long = submission.arithmetic_intensity(
        *submission.decode_attention_cost(65536, HEADS, KV_HEADS, HEAD_DIM))
    c.check("decode attention intensity does not depend on context length",
            lambda: abs(short - long) < 1e-6, f"1k -> {short:.2f}, 64k -> {long:.2f}")
    c.check("decode attention intensity equals the GQA group size",
            lambda: abs(short - HEADS / KV_HEADS) < 0.1, f"got {short:.2f}")
    c.check("decode attention is memory bound",
            lambda: submission.roofline(
                *submission.decode_attention_cost(4096, HEADS, KV_HEADS, HEAD_DIM)
            )["bound_by"] == "memory")

    # A model with 24 KV heads would read six times the bytes for the same work.
    mqa = submission.arithmetic_intensity(
        *submission.decode_attention_cost(4096, HEADS, HEADS, HEAD_DIM))
    c.check("sharing KV heads raises intensity six-fold",
            lambda: abs(short / mqa - 6.0) < 0.1, f"ratio {short / mqa:.2f}")

    launch_checks(c, submission)

    c.metric("ridge_point", round(ridge, 1))
    c.metric("decode_intensity", round(intensity_1, 2))
    c.metric("prefill_intensity", round(intensity_2048, 1))
    c.metric("decode_attention_intensity", round(short, 2))
    measure_tiny_kernels(c)


def launch_checks(c, submission):
    """Step time with a launch floor under every kernel, on the CPU.

    Wrong implementation 1: take one max over the whole step,
    max(N * launch, sum of floors). On three 20 KB kernels and one 535 MB MLP
    read, that returns 0.276411 ms, because the big kernel's time swallows the
    tiny kernels' launches. The per-kernel max returns 0.291379 ms.

    Wrong implementation 2: add the launch to every kernel's floor. On 400 tiny
    kernels that returns 2.004234 ms, not 2.000000 ms.
    """
    new = ("launch_floor_ms", "kernel_time_ms", "step_time_ms", "classify_step")
    if not c.require(submission, *new):
        return

    tiny = (float(NORM_BYTES), float(NORM_BYTES))
    mlp = submission.mlp_cost(1, HIDDEN, INTERMEDIATE)
    mlp_ms = mlp[1] / (1935.0 * 1e9) * 1e3

    c.check("400 launches at 5 µs cost 2.000 ms",
            lambda: _near("floor", lambda: submission.launch_floor_ms(400), 2.0))
    c.check("a 20 KB kernel runs in 10.6 ns, so its launch sets its time",
            lambda: _near("tiny", lambda: submission.kernel_time_ms(*tiny), 0.005))
    c.check("a kernel longer than a launch takes its roofline time",
            lambda: _near("mlp", lambda: submission.kernel_time_ms(*mlp), mlp_ms, 1e-9))

    # Per-kernel max catches both wrong implementations above.
    c.check("400 tiny kernels take 2.000 ms eager, not 2.004 ms",
            lambda: _near("400 tiny", lambda: submission.step_time_ms([tiny] * 400), 2.0))
    mixed = [tiny, tiny, tiny, mlp]
    c.check("three tiny kernels and one MLP read: launches add to the big kernel's time",
            lambda: _near("mixed", lambda: submission.step_time_ms(mixed),
                          3 * 0.005 + mlp_ms, 1e-9))

    steps = decode_step()
    eager = submission.step_time_ms(steps)
    graph = submission.step_time_ms(steps, graph=True)
    c.check("the 400-kernel decode step takes 28.4486 ms eager",
            lambda: _near("eager", lambda: eager, 28.448618, 1e-4))
    c.check("the same step takes 27.8100 ms with one graph launch",
            lambda: _near("graph", lambda: graph, 27.809983, 1e-4))
    c.check("a graph removes only the exposed launches, 0.6386 ms",
            lambda: _near("saved", lambda: eager - graph, 0.638635, 1e-4))

    def verdicts():
        got = {
            "400 tiny": submission.classify_step([tiny] * 400),
            "decode step": submission.classify_step(steps),
            "prefill MLP": submission.classify_step([submission.mlp_cost(2048, HIDDEN, INTERMEDIATE)]),
        }
        want = {"400 tiny": "launch", "decode step": "memory", "prefill MLP": "compute"}
        return got == want, f"expected {want}, got {got}"

    c.check("a step is classified as launch bound, memory bound, or compute bound", verdicts)
    c.metric("launch_floor_ms", round(submission.launch_floor_ms(400), 3))
    c.metric("step_eager_ms", round(eager, 4))
    c.metric("step_graph_ms", round(graph, 4))
    c.metric("launch_saved_ms", round(eager - graph, 4))


def measure_tiny_kernels(c):
    """Time 400 tiny kernels eager and under one graph. Skips without CUDA.

    The graph capture follows chapter 10a: warm up on a side stream, then
    capture. This check only reports numbers. It sets no threshold.
    """
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None or not torch.cuda.is_available():
        print("      No CUDA device is visible, so the tiny-kernel timing is skipped.", flush=True)
        c.metric("tiny_kernel_timing", "skipped")
        return

    x = torch.zeros(1024, device="cuda")

    def step():
        for _ in range(400):
            x.add_(1.0)

    def ms(fn, iters=20):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        eager_ms, graph_ms = ms(step), ms(graph.replay)
        c.metric("tiny_eager_ms", round(eager_ms, 4))
        c.metric("tiny_graph_ms", round(graph_ms, 4))
        c.metric("tiny_speedup", round(eager_ms / graph_ms, 2) if graph_ms else 0.0)
    except Exception as exc:  # noqa: BLE001
        c.add("tiny-kernel timing", False,
              f"expected tiny_eager_ms and tiny_graph_ms, got {type(exc).__name__}: {exc}")
