"""Chapter 12, your first CUDA kernel.

Teaches content/chapters/12-your-first-cuda-kernel.md in its order. Every
number below was recomputed in Python and matches the chapter.

The A100 (chapter values): 108 SMs, 40 MB L2, 80 GB HBM2e rated at
1935 GB/s; chapter 10's measured copy is 1275 GB/s.

How the GPU runs a kernel:
- The lab's vector add: 2^24 = 16,777,216 threads, "about 16.8 million".
- A warp is 32 threads; a 256-thread block is 256 / 32 = 8 warps; a block
  holds at most 1024 threads.
- A launch returns to the host in 5 to 10 microseconds (chapter value).

The thread index:
- Theater seat: 3 x 256 + 7 = 775.
- Wrong order at G = 65,536 blocks: neighbors land 65,536 elements apart,
  65,536 x 4 B = 262,144 B = 256 KiB, which the chapter writes "256 KB".
- The drawn example uses G = 4: lanes 0..7 land on elements 0, 4, ..., 28.

Grid geometry and the tail (n = 1,000,003, B = 256):
- blocks = ceil(1,000,003 / 256) = (1,000,003 + 255) // 256 = 3907.
- threads = 3907 x 256 = 1,000,192; spare = 1,000,192 - 1,000,003 = 189.
- The last block starts at 3906 x 256 = 999,936 and ends at 1,000,191;
  live lanes 1,000,003 - 999,936 = 67, so warps 0 and 1 (lanes 0-63) are
  live, warp 2 (lanes 64-95) has lanes 64, 65, 66 live, and warps 3-7 are
  dead. Warps in the launch: 3907 x 8 = 31,256.
- Filling the machine: 2 x 108 = 216 blocks. n = 2^24: 65,536 blocks,
  65,536 / 108 = 606.8, "607 per SM". n = 2^20: 4096 blocks, 4096 / 108 =
  37.9, "38 per SM".

Grid-stride loop (drawn example): 2 blocks x 4 threads = stride 8 over 21
elements; passes cover 0-7, 8-15, 16-20 (threads 5-7 of the last pass fail
the loop condition).

Coalescing:
- Coalesced: 32 x 4 B = 128 B = 4 sectors of 32 B, 100% used.
- Strided by 32 floats: 128 B apart, 32 sectors x 32 B = 1024 B for 128
  useful bytes; 128 / 1024 = 12.5%; 1024 / 128 = 8x. 32 - 4 = 28 B wasted
  per sector.

The lab's 6.9x (n = 2^24):
- 1304 / 189 = 6.8995, "6.9".
- Strided bytes per element: 32 (read sector) + 4 (write) = 36 against
  8 counted; 36 / 8 = 4.5.
- 1304 / 4.5 = 289.8, "290 GB/s"; 290 / 189 = 1.534, "1.53";
  4.5 x 1.53 = 6.885, "6.9".
- Strided read footprint: 2^24 / 32 = 524,288 elements x 32 B =
  16,777,216 B = 16.8 MB, inside the 40 MB L2.
- 1304 / 1275 = 1.023: the vector add is level with the copy.

Memory hierarchy: 108 x 256 KB = 27,648 KB = 27.6 MB of registers;
108 x 164 KB = 17,712 KB = 17.7 MB of shared memory; both under 40 MB.

Shared memory (RMSNorm, 4096 x 1024, 256 threads):
- 1024 / 256 = 4 loop iterations per thread; log2 256 = 8 reduction
  rounds; log2 8 = 3 rounds on the drawn 8 slots (offsets 4, 2, 1).
- Shared bytes 256 x 4 = 1024.
- Contiguous chunks of 4: a warp's load spans 32 x 4 x 4 = 512 B = 16
  sectors, against 128 B = 4 sectors when striding by the block size.
- From 200: 100, 50, 25, 12, 6, 3, 1, losing element 24 (chapter value).

Timing: queueing about 5 microseconds; engine/bench.py runs 5 warm-up and
20 timed iterations; 80% of 1275 GB/s = 1020 GB/s.
"""

from manim import *

import kit
from kit import theme as th

# The chapter's maths colours: b (block) is \hla, t (thread) is \hlb,
# 4.5 is \hlc, 1.53 is \hld.
B_C = th.HL_A
T_C = th.HL_B
BYTES_C = th.HL_C
REQ_C = th.HL_D

LIVE = th.GOOD
DEAD = th.INK_4

FIRST_KERNEL = """\
__global__ void add_kernel(const float* a, const float* b,
                           float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a.data_ptr<float>(),
        b.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

STRIDE_LOOP = """\
__global__ void add_kernel(const float* a, const float* b,
                           float* out, int n) {
    int stride = blockDim.x * gridDim.x;   // threads in grid
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride)
        out[i] = a[i] + b[i];
}
"""

REDUCE = """\
// Tree reduction over the 256 partial sums: 8 halving steps.
for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
        partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
}
"""

CHECKS = """\
add_kernel<<<blocks, threads>>>(...);
// Launch-time: bad geometry, too much shared memory.
C10_CUDA_CHECK(cudaGetLastError());
// Execution-time: illegal address, misaligned access.
C10_CUDA_CHECK(cudaDeviceSynchronize());
"""


def region(title, width, height, color=None, sub=None, size=th.SIZE_SMALL):
    """A titled region (a machine, an SM) with its label in the top-left corner."""
    stroke = th.RULE_STRONG if color is None else color
    rect = RoundedRectangle(width=width, height=height, corner_radius=0.12, stroke_color=stroke,
                            stroke_width=th.STROKE_BOX, fill_color=th.tint(stroke, 0.06), fill_opacity=1)
    name = kit.label(title, size=size, color=th.INK, weight=MEDIUM)
    parts = [name]
    if sub:
        parts.append(kit.label(sub, size=th.SIZE_LABEL, color=th.INK_3))
    head = VGroup(*parts).arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
    head.move_to(rect.get_corner(UL) + RIGHT * (head.width / 2 + 0.2) + DOWN * (head.height / 2 + 0.16))
    group = VGroup(rect, head)
    group.rect, group.head = rect, head
    return group


def chip(width, height, color, text=None, tint=0.35, font=th.MONO, size=th.SIZE_LABEL):
    """A small filled rectangle with an optional label: a block, a command, a slot."""
    rect = Rectangle(width=width, height=height, stroke_color=color, stroke_width=th.STROKE_BOX,
                     fill_color=th.tint(color, tint), fill_opacity=1)
    group = VGroup(rect)
    if text:
        t = kit.label(text, size=size, color=th.INK, font=font)
        if t.width > width * 0.88:
            t.scale_to_fit_width(width * 0.88)
        t.move_to(rect)
        group.add(t)
    group.rect = rect
    return group


def swatch(text, color, tint=0.35, size=th.SIZE_LABEL):
    sq = Square(0.26, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                fill_color=th.tint(color, tint), fill_opacity=1)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


def crate(used, color, slot=0.095, height=0.46):
    """One 32-byte sector drawn as 8 float slots; the first `used` slots are filled."""
    slots = VGroup()
    for k in range(8):
        fill = th.tint(color, 0.75) if k < used else th.RAISE
        slots.add(Rectangle(width=slot, height=height, stroke_width=0, fill_color=fill, fill_opacity=1))
    slots.arrange(RIGHT, buff=0)
    edge = Rectangle(width=slots.width, height=height, stroke_color=th.INK_3, stroke_width=th.STROKE_AXIS,
                     fill_opacity=0).move_to(slots)
    group = VGroup(slots, edge)
    group.slots = slots
    return group


def math_rows(rows, size=th.SIZE_MATH_SMALL, buff=0.3):
    """Formula lines stacked and left-aligned, without Derivation's realignment."""
    return VGroup(*[kit.formula(r, size=size) for r in rows]).arrange(DOWN, buff=buff, aligned_edge=LEFT)


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "Your first CUDA kernel. "
                "In this lecture, you picture how a GPU runs your code: threads, blocks, warps, and coalescing. "
                "Then you write a kernel from Python, and derive the number the lab measures on an A100."
            )
        )
        self.a100_baseline()
        self.host_device()
        self.launch_hierarchy()
        self.launch_queue()
        self.first_kernel()
        self.thread_index()
        self.index_order()
        self.tail_warp()
        self.block_size()
        self.grid_stride()
        self.divergence()
        self.coalescing()
        self.lab_ratio()
        self.two_factors()
        self.memory_hierarchy()
        self.tree_reduction()
        self.five_details()
        self.async_errors()
        self.timing()
        self.closing_slide(
            [
                r"$i = b \cdot B + t$, so a warp reads 32 neighbors.",
                r"Round the blocks up; the tail splits one warp.",
                r"Sectors are 32 bytes: $6.9 = 4.5 \times 1.53$.",
                r"Errors are late and sticky; sync, then time.",
            ],
            narration=(
                "Here's what to keep. "
                "The index is block times block size, plus thread, so a warp reads 32 neighbors. "
                "Round the block count up; the tail splits only one warp. "
                "Memory moves in 32-byte sectors, and the lab's 6.9 is wasted bytes times wasted requests. "
                "And errors arrive late, so synchronize before you read results or timings."
            ),
            book_section="s-what-goes-wrong",
        )

    # ------------------------------------------------------------------
    # Before you start, the host and the device

    def a100_baseline(self):
        self.slide("a100-baseline", "Compare a kernel against 1275 GB/s, not 1935",
                   book_section="s-before-you-start")
        parts = VGroup(
            kit.box("108 SMs", sublabel="the processors", color=self.color, width=3.4, size=th.SIZE_BODY),
            kit.box("40 MB L2", sublabel="cache shared by all SMs", color=self.colors[2], width=3.4,
                    size=th.SIZE_BODY),
            kit.box("80 GB HBM2e", sublabel="main memory", color=self.colors[1], width=3.4, size=th.SIZE_BODY),
        ).arrange(RIGHT, buff=0.45)
        name = kit.label("NVIDIA A100, compute capability 8.0", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        bars = kit.bar_compare(
            [("Rated, PCIe card", 1935, th.INK_4, "1935 GB/s"),
             ("Measured copy, chapter 10", 1275, self.color, "1275 GB/s")],
            width=6.4, bar_height=0.5, gap=0.3,
        )
        note = kit.callout("key", r"Outside the matrix multiplies, almost everything is \textit{memory bound}: "
                                  r"limited by how fast bytes move, not by arithmetic.",
                           title="Chapter 10's conclusion", width=11.0, size=th.SIZE_SMALL)
        col = VGroup(VGroup(name, parts).arrange(DOWN, buff=0.25), bars, note).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "Every chapter so far has called someone else's kernels. "
            "This one writes a kernel by hand, compiled at run time from a Python string. "
            "The target is an A100 with 108 streaming multiprocessors. "
            "It has a 40 MB L2 cache, and 80 GB of HBM rated at 1935 GB/s. "
            "But chapter 10 measured a plain copy at 1275 GB/s, and that's your real yardstick. "
            "Almost everything outside the matrix multiplies is memory bound, so this lecture is about how bytes "
            "move."
        ) as t:
            t.wait_until("The target is")
            self.play(FadeIn(name), FadeIn(parts[0], shift=UP * 0.1), run_time=0.7)
            t.wait_until("It has a 40 MB")
            self.play(FadeIn(parts[1], shift=UP * 0.1), FadeIn(parts[2], shift=UP * 0.1), run_time=0.7)
            self.play(FadeIn(bars.labels[0]), GrowFromEdge(bars.bars[0], LEFT), FadeIn(bars.values[0]), run_time=0.9)
            t.wait_until("But chapter 10")
            self.play(FadeIn(bars.labels[1]), GrowFromEdge(bars.bars[1], LEFT), FadeIn(bars.values[1]), run_time=0.9)
            t.wait_until("Almost everything")
            self.play(FadeIn(note, shift=UP * 0.1), run_time=0.7)

    def host_device(self):
        self.slide("host-device", "The kernel sees one raw pointer", book_section="s-the-host-and-the-device")
        host = region("Host", 5.2, 2.5, color=self.colors[1], sub="the CPU and its memory")
        dev = region("Device", 6.6, 2.6, color=self.color, sub="the GPU and its memory")
        machines = VGroup(host, dev).arrange(RIGHT, buff=1.3)
        h_body = VGroup(kit.mono("vector_add(a, b)", size=th.SIZE_SMALL, color=th.INK),
                        kit.label("Python, PyTorch, host C++", size=th.SIZE_LABEL, color=th.INK_3)
                        ).arrange(DOWN, buff=0.18)
        h_body.move_to(host.rect.get_center() + DOWN * 0.25)
        strip = kit.grid(1, 8, cell=(0.42, 0.42))
        strip.paint(strip.block(0, 1, 0, 8), self.color, tint=0.2)
        ptr = kit.mono("float* a", size=th.SIZE_SMALL, color=th.INK)
        inner = VGroup(ptr, strip).arrange(RIGHT, buff=0.6)
        no_shape = kit.label("no shape, no strides, no dtype: pass n", size=th.SIZE_LABEL, color=th.BAD)
        VGroup(inner, no_shape).arrange(DOWN, buff=0.25).move_to(dev.rect.get_center() + DOWN * 0.25)
        top = VGroup(machines, h_body, inner, no_shape)
        tab = kit.table(
            [
                [r"\texttt{\_\_host\_\_} (the default)", "Host", "Host"],
                [r"\texttt{\_\_global\_\_}", "Device", r"Host: this is a \textit{kernel}"],
                [r"\texttt{\_\_device\_\_}", "Device", "Device"],
            ],
            header=["Qualifier", "Runs on", "Callable from"],
            size=th.SIZE_SMALL,
        )
        col = VGroup(top, tab).arrange(DOWN, buff=0.55)
        self.place(col)
        launch = kit.arrow(host, dev, text="launch", color=th.INK_3)
        aim = Arrow(ptr.get_right(), strip.cell(0, 0).get_left(), buff=0.1, color=self.color,
                    stroke_width=th.STROKE_BOX + 1, tip_length=0.18)
        with self.narrate(
            "A CUDA program runs on two machines. "
            "The host is the CPU, where your Python runs. "
            "The device is the GPU, where kernels run. "
            "A global function runs on the device, and the host launches it: that's a kernel. "
            "A device function runs on the device, called only from device code. "
            "The kernel sees one raw pointer to the tensor's data. "
            "It knows nothing about shapes or dtypes, so you pass the element count separately."
        ) as t:
            t.wait_until("The host is")
            self.play(FadeIn(host), FadeIn(h_body), run_time=0.7)
            t.wait_until("The device is")
            self.play(FadeIn(dev), FadeIn(strip), run_time=0.7)
            t.wait_until("A global function")
            self.play(FadeIn(tab.rules), FadeIn(VGroup(*tab.cells[0])), FadeIn(VGroup(*tab.cells[1])),
                      FadeIn(VGroup(*tab.cells[2]), shift=RIGHT * 0.1), GrowArrow(launch.line), FadeIn(launch.label),
                      run_time=0.8)
            t.wait_until("A device function")
            self.play(FadeIn(VGroup(*tab.cells[3]), shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("The kernel sees")
            self.play(FadeIn(ptr), GrowArrow(aim), run_time=0.8)
            t.wait_until("It knows nothing")
            self.play(FadeIn(no_shape, shift=UP * 0.08), run_time=0.6)

    # ------------------------------------------------------------------
    # How the GPU runs a kernel, and what a launch is

    def launch_hierarchy(self):
        self.slide("launch-hierarchy", "Threads march in warps, warps fill blocks",
                   book_section="s-how-the-gpu-runs-a-kernel")
        names = ["b0", "b1", "b2", "b3", "b4", "b5", "..."]
        blocks = kit.grid(1, 7, cell=(0.78, 0.5), values=[names], gap=0.08)
        blocks.paint(blocks.block(0, 1, 0, 6), self.colors[1], tint=0.2)
        grid_lab = kit.label("grid: every block of one launch", size=th.SIZE_LABEL, color=th.INK_3)
        warps = kit.grid(8, 32, cell=(0.19, 0.3), row_labels=[f"warp {k}" for k in range(8)])
        blk_lab = kit.label("block 0: 256 threads = 8 warps of 32 lanes", size=th.SIZE_LABEL, color=th.INK_3)
        top = VGroup(grid_lab, blocks).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        bottom = VGroup(blk_lab, warps).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        left = VGroup(top, bottom).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        sms = VGroup(*[region(f"SM {k}", 2.15, 1.35, color=self.color, size=th.SIZE_LABEL) for k in range(4)])
        sms.arrange_in_grid(2, 2, buff=0.3)
        sm_lab = kit.label("4 of the A100's 108 SMs", size=th.SIZE_LABEL, color=th.INK_3)
        right = VGroup(sm_lab, sms).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        layout = VGroup(left, right).arrange(RIGHT, buff=0.7, aligned_edge=UP)
        self.place(layout)

        def slot(sm, k):
            f = sms[sm].rect.width / 2.15
            return sms[sm].rect.get_center() + (DOWN * 0.2 + RIGHT * (-0.48 + 0.96 * k)) * f

        deal = [(3, 0, 0), (0, 2, 0), (5, 1, 0), (1, 3, 0), (4, 0, 1), (2, 1, 1)]
        frame = SurroundingRectangle(warps.block(0, 8, 0, 32), color=self.colors[1], buff=0.06, stroke_width=3)
        with self.narrate(
            "A GPU runs a huge number of threads that each do a little. "
            "A thread runs your kernel on one small piece of data, and the lab launches about 16.8 million. "
            "Threads march in warps of 32, every lane running the same instruction at once. "
            "The warp is what the hardware schedules. "
            "A block is a team of up to 1024 threads, so 256 threads make 8 warps. "
            "The grid is every block of one launch. "
            "The hardware hands each block to an SM whenever one has room. "
            "The A100 has 108 SMs, and each hosts several blocks at once."
        ) as t:
            self.play(FadeIn(blk_lab), FadeIn(warps), run_time=0.8)
            t.wait_until("A thread runs")
            self.play(warps.fill(warps.cell(0, 0), self.color, tint=0.8), run_time=0.6)
            t.wait_until("Threads march")
            self.play(warps.fill(warps.block(0, 1, 1, 32), self.color, tint=0.45), run_time=0.8)
            t.wait_until("The warp is what")
            self.play(Indicate(warps.row_labels[0], color=self.color), run_time=0.8)
            t.wait_until("A block is a team")
            self.play(LaggedStart(*[warps.fill(warps.row(r), self.colors[1], tint=0.25) for r in range(1, 8)],
                                  lag_ratio=0.15), Create(frame), run_time=1.4)
            t.wait_until("The grid is")
            self.play(FadeIn(grid_lab), FadeIn(blocks), run_time=0.7)
            t.wait_until("The hardware hands")
            self.play(FadeIn(sm_lab), FadeIn(sms), run_time=0.6)
            for b, sm, k in deal:
                c = chip(0.78, 0.42, self.colors[1], names[b], tint=0.45)
                c.scale(blocks.cell(0, 0).square.width / 0.78)
                c.move_to(blocks.cell(0, b))
                self.play(c.animate.move_to(slot(sm, k)), run_time=0.45)

    def launch_queue(self):
        self.slide("launch-queue", "A launch is a queued command, not a call",
                   book_section="s-what-a-kernel-launch-is")
        line = kit.code("kernel<<<grid, block, shared_bytes, stream>>>(args...);", "cuda", line_numbers=False)
        host = kit.box("Host", sublabel="your Python thread", color=self.colors[1], width=2.8)
        stream = kit.grid(1, 4, cell=(1.55, 0.66), gap=0.08)
        s_lab = kit.label("stream: an ordered queue of GPU work", size=th.SIZE_LABEL, color=th.INK_3)
        queue = VGroup(s_lab, stream).arrange(DOWN, buff=0.15)
        gpu = kit.box("GPU", sublabel="starts when it gets there", color=self.color, width=2.8)
        row = VGroup(host, queue, gpu).arrange(RIGHT, buff=0.9)
        warn = kit.callout("warning", r"Blocks run in no particular order. A kernel whose correctness depends on "
                                      r"block order is wrong even when it passes.",
                           title="Block order is never guaranteed", width=11.5, size=th.SIZE_SMALL)
        col = VGroup(line, row, warn).arrange(DOWN, buff=0.75)
        row.shift(DOWN * 0.1)
        self.place(col)
        glyphs = line.line(1)
        spans = [(9, 13, "blocks"), (14, 19, "threads"), (20, 32, "shared bytes"), (33, 39, "queue")]
        marks = VGroup()
        for i, (a, b, text) in enumerate(spans):
            part = VGroup(*glyphs[a:b])
            rule = kit.underline(part, self.colors[i % 4] if i else self.color, buff=0.06, stroke=3)
            lab = kit.label(text, size=th.SIZE_LABEL, color=th.INK_3)
            lab.move_to([rule.get_center()[0], line.panel.get_bottom()[1] - 0.24, 0])
            if i == 0:
                lab.align_to(rule, RIGHT)
            marks.add(VGroup(rule, lab))
        a1 = kit.arrow(host, stream, color=th.INK_4)
        a2 = kit.arrow(stream, gpu, color=th.INK_4)
        old = [chip(1.4, 0.5, th.INK_4, "earlier", tint=0.3) for _ in range(2)]
        for k, c in enumerate(old):
            c.move_to(stream.cell(0, k))
        cmd = chip(1.4, 0.5, self.color, "add_kernel", tint=0.4)
        sc = stream.cell(0, 0).square.width / 1.55
        for c in old + [cmd]:
            c.scale(sc)
        for k, c in enumerate(old):
            c.move_to(stream.cell(0, k))
        cmd.move_to(host.rect.get_center())
        back = kit.label("back in 5 to 10 µs", size=th.SIZE_LABEL, color=th.GOOD, weight=MEDIUM)
        back.next_to(host, DOWN, buff=0.18)
        with self.narrate(
            "A launch is a queued command, not a function call. "
            "NVIDIA's compiler turns the triple angle brackets into a call that pushes a command onto a stream, an "
            "ordered queue of GPU work. "
            "It returns in 5 to 10 microseconds, long before the GPU starts. "
            "That's chapter 10's launch overhead. "
            "The full form takes four parameters: blocks, threads per block, shared memory bytes, and the stream. "
            "Blocks then run in no particular order. "
            "A kernel that depends on block order is wrong, even when it passes."
        ) as t:
            self.play(FadeIn(line), run_time=0.7)
            t.wait_until("NVIDIA's compiler")
            self.play(FadeIn(host), FadeIn(queue), FadeIn(gpu), GrowArrow(a1.line), GrowArrow(a2.line),
                      *[FadeIn(c) for c in old], run_time=0.9)
            self.play(FadeIn(cmd, scale=0.8), run_time=0.4)
            self.play(cmd.animate.move_to(stream.cell(0, 2)), run_time=1.0)
            t.wait_until("It returns in")
            self.play(FadeIn(back, shift=UP * 0.08), Indicate(host.rect, color=th.GOOD, scale_factor=1.04),
                      run_time=0.8)
            self.play(old[0].animate.move_to(gpu.rect.get_center()).set_opacity(0), old[1].animate.move_to(
                stream.cell(0, 0)), cmd.animate.move_to(stream.cell(0, 1)), run_time=1.2)
            t.wait_until("The full form")
            self.play(LaggedStart(*[FadeIn(m, shift=UP * 0.05) for m in marks], lag_ratio=0.5), run_time=2.4)
            t.wait_until("Blocks then run")
            self.play(FadeIn(warn, shift=UP * 0.1), run_time=0.7)

    # ------------------------------------------------------------------
    # Compile CUDA from Python

    def first_kernel(self):
        self.slide("first-kernel", "Three lines on the device, one host function",
                   book_section="s-compile-cuda-from-python")
        panel = kit.code(FIRST_KERNEL, "cuda", caption="source string for load_inline")
        self.place(panel)
        dev_band = panel.band(1, 2, 3, 4, 5, color=th.tint(self.color, 0.08))
        with self.narrate(
            "Here's your first kernel, as a source string that PyTorch compiles at run time. "
            "The kernel is three lines. "
            "It's declared global and returns nothing, so results leave through the output pointer. "
            "The second line works out which element this thread owns. "
            "The third does two loads, one add, and one store, guarded because the grid rounds up. "
            "The host function is the only one Python calls. "
            "It allocates the output, picks 256 threads per block, and rounds the block count up. "
            "Then it queues the launch and returns, while the kernel may still be running. "
            "The first build takes 30 to 60 seconds, and any edit to the source costs that again."
        ) as t:
            self.play(FadeIn(panel.frame), run_time=0.5)
            t.wait_until("The kernel is three")
            self.play(FadeIn(dev_band), *panel.reveal(1, 2, 5), run_time=0.7)
            t.wait_until("It's declared global")
            self.play(panel.highlight(1, 2), run_time=0.5)
            t.wait_until("The second line")
            self.play(*panel.reveal(3), panel.highlight(3), run_time=0.6)
            t.wait_until("The third does")
            self.play(*panel.reveal(4), panel.highlight(4), run_time=0.6)
            t.wait_until("The host function")
            self.play(*panel.reveal(6, 14), run_time=0.6)
            t.wait_until("It allocates")
            self.play(*panel.reveal(7, 8), run_time=0.6)
            self.play(*panel.reveal(9, 10), run_time=0.6)
            t.wait_until("Then it queues")
            self.play(*panel.reveal(11, 12), panel.highlight(11, 12), run_time=0.7)
            self.play(*panel.reveal(13), run_time=0.5)

    # ------------------------------------------------------------------
    # The thread index

    def thread_index(self):
        self.slide("thread-index", "The block is the row, the thread is the seat",
                   book_section="s-the-thread-index")
        seats = kit.grid(4, 16, cell=0.42, row_labels=[f"block {r}" for r in range(4)])
        cap = kit.label("rows of 256 seats; 16 drawn", size=th.SIZE_LABEL, color=th.INK_3)
        cap.next_to(seats, UP, buff=0.15).align_to(seats, RIGHT)
        left = VGroup(seats, cap)
        main = kit.formula(r"i", r"=", r"b", r"\cdot B +", r"t", size=th.SIZE_MATH)
        main[2].set_color(B_C)
        main[4].set_color(T_C)
        box = SurroundingRectangle(main, color=th.INK_3, buff=0.18, corner_radius=0.08, stroke_width=2.5)
        ex = kit.formula(r"3", r"\cdot 256 +", r"7", r"= 775", size=th.SIZE_MATH_SMALL)
        ex[0].set_color(B_C)
        ex[2].set_color(T_C)
        inv = kit.formula(r"b = \lfloor i / B \rfloor, \quad t = i \bmod B", size=th.SIZE_MATH_SMALL)
        right = VGroup(VGroup(main, box), ex, inv).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        top = VGroup(left, right).arrange(RIGHT, buff=0.8)
        line = kit.code("int i = blockIdx.x * blockDim.x + threadIdx.x;", "cuda", line_numbers=False)
        col = VGroup(top, line).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "Every thread runs the same body, so it must work out which element it owns. "
            "Picture theater seats in rows of 256. "
            "Row 3, seat 7, is seat number 775. "
            "The block is the row, and the thread is the seat. "
            "So the index is the block index times the block size, plus the thread index. "
            "Divide to recover the block, and take the remainder for the thread. "
            "In code, it's one line."
        ) as t:
            self.play(FadeIn(seats), FadeIn(cap), run_time=0.8)
            t.wait_until("Row 3, seat 7")
            self.play(seats.fill(seats.block(0, 3, 0, 16), B_C, tint=0.15), run_time=0.9)
            self.play(seats.fill(seats.block(3, 4, 0, 7), T_C, tint=0.25), run_time=0.6)
            self.play(seats.fill(seats.cell(3, 7), self.color, tint=0.85), run_time=0.5)
            t.wait_until("The block is the row")
            self.play(FadeIn(ex), run_time=0.7)
            t.wait_until("So the index is")
            self.play(FadeIn(main), Create(box), run_time=0.9)
            t.wait_until("Divide to recover")
            self.play(FadeIn(inv), run_time=0.7)
            t.wait_until("In code")
            self.play(FadeIn(line, shift=UP * 0.1), run_time=0.7)

    def index_order(self):
        self.slide("index-order", "The thread index must be the fastest digit",
                   book_section="s-the-thread-index")
        cols = 32
        rows = VGroup()
        stuff = []
        for k, (tex, step) in enumerate([(r"$i = b \cdot B + t$: one contiguous run", 1),
                                        (r"$i = t \cdot G + b$: neighbors $G$ elements apart", 4)]):
            lanes = kit.grid(1, 8, cell=0.42, values=[[str(j) for j in range(8)]])
            lanes.paint(lanes.block(0, 1, 0, 8), self.color, tint=0.3)
            mem = kit.grid(1, cols, cell=(0.34, 0.42))
            lanes.next_to(mem, UP, buff=0.9).align_to(mem, LEFT)
            title = kit.prose(tex, size=th.SIZE_SMALL, color=th.INK, width=None)
            title.next_to(lanes, RIGHT, buff=0.6)
            lane_lab = kit.label("8 lanes of a warp", size=th.SIZE_LABEL, color=th.INK_3)
            lane_lab.next_to(lanes, UP, buff=0.1).align_to(lanes, LEFT)
            mem_lab = kit.label("memory", size=th.SIZE_LABEL, color=th.INK_3).next_to(mem, LEFT, buff=0.15)
            rows.add(VGroup(lane_lab, lanes, title, mem, mem_lab))
            stuff.append((lanes, mem, step))
        rows.arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        far = kit.prose(r"At $G = 65{,}536$ blocks: 256 KB apart.", size=th.SIZE_SMALL, color=th.BAD, width=None)
        col = VGroup(rows, far).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        self.place(col)
        links = []
        for lanes, mem, step in stuff:
            g = VGroup()
            for j in range(8):
                g.add(Line(lanes.cell(0, j).get_bottom(), mem.cell(0, j * step).get_top(), color=self.color,
                           stroke_width=2.5))
            links.append(g)
        with self.narrate(
            "Swap the digits, thread times grid size plus block, and every element is still covered once. "
            "But it's wrong. "
            "With the first order, neighboring threads get neighboring elements, so a warp reads one contiguous run. "
            "With the swapped order, neighbors land a whole grid apart. "
            "For 65,536 blocks, that's 256 KB apart. "
            "The thread index must be the fastest digit."
        ) as t:
            self.play(FadeIn(rows[0][:2]), FadeIn(rows[0][3:]), FadeIn(rows[1][:2]), FadeIn(rows[1][3:]),
                      run_time=0.8)
            t.wait_until("With the first order")
            lanes, mem, _ = stuff[0]
            self.play(FadeIn(rows[0][2]), LaggedStart(*[Create(l) for l in links[0]], lag_ratio=0.1),
                      mem.fill(mem.block(0, 1, 0, 8), self.color), run_time=1.2)
            t.wait_until("With the swapped")
            lanes, mem, _ = stuff[1]
            self.play(FadeIn(rows[1][2]), LaggedStart(*[Create(l) for l in links[1]], lag_ratio=0.1),
                      mem.fill(VGroup(*[mem.cell(0, 4 * j) for j in range(8)]), self.color), run_time=1.4)
            t.wait_until("For 65,536")
            self.play(FadeIn(far, shift=UP * 0.08), run_time=0.6)
            t.wait_until("The thread index must")
            self.play(Circumscribe(rows[0], color=self.color, buff=0.1), run_time=1.2)

    # ------------------------------------------------------------------
    # Grid geometry and the tail

    def tail_warp(self):
        self.slide("tail-warp", "Round up, guard, and only one warp diverges",
                   book_section="s-grid-geometry-and-the-tail")
        maths = math_rows([
            r"\text{blocks} = \lceil 1{,}000{,}003 / 256 \rceil = 3907",
            r"\text{threads} = 3907 \times 256 = 1{,}000{,}192",
            r"\text{spare} = 1{,}000{,}192 - 1{,}000{,}003 = 189",
        ], buff=0.25)
        code = kit.code("int blocks = (n + threads - 1) / threads;\nif (i < n) out[i] = a[i] + b[i];", "cuda",
                        line_numbers=False)
        top = VGroup(maths, code).arrange(RIGHT, buff=0.7)
        g = kit.grid(8, 32, cell=(0.3, 0.3), row_labels=[f"warp {k}" for k in range(8)])
        cap = kit.label("the last block: indices 999,936 to 1,000,191", size=th.SIZE_LABEL, color=th.INK_3)
        cap.next_to(g, UP, buff=0.12).align_to(g, LEFT)
        key = VGroup(swatch("live: 67 lanes", LIVE, 0.6), swatch("dead: 189 lanes", DEAD, 0.35)).arrange(
            RIGHT, buff=0.5)
        key.next_to(g, DOWN, buff=0.15).align_to(g, RIGHT)
        fig = VGroup(g, cap, key)
        col = VGroup(top, fig).arrange(DOWN, buff=0.5)
        self.place(col)
        live = [g.cell(k // 32, k % 32) for k in range(67)]
        dead = [g.cell(k // 32, k % 32) for k in range(67, 256)]
        split = g.outline(2, 3, 0, 32, color=self.color, stroke=4)
        split_lab = kit.label("the one split warp", size=th.SIZE_LABEL, color=self.color, weight=MEDIUM)
        split_lab.next_to(g, DOWN, buff=0.15).align_to(g, LEFT)
        with self.narrate(
            "To cover n elements, launch n over the block size, rounded up. "
            "Integer division truncates, so add the block size minus one first. "
            "Plain division silently drops the last partial block. "
            "Take the lab's awkward size, 1,000,003, at 256 per block. "
            "That's 3907 blocks. "
            "They launch 1,000,192 threads, so 189 sit idle. "
            "Here's the last block, as 8 warps of 32. "
            "Only its first 67 lanes are live. "
            "Warps 0 and 1 are fully live, and warps 3 to 7 are fully dead. "
            "Only warp 2 splits: one warp in 31,256, so the guard is free."
        ) as t:
            t.wait_until("Integer division")
            self.play(FadeIn(code.frame), *code.reveal(1), run_time=0.7)
            t.wait_until("That's 3907")
            self.play(FadeIn(maths[0]), run_time=0.6)
            t.wait_until("They launch")
            self.play(FadeIn(maths[1]), run_time=0.5)
            self.play(FadeIn(maths[2]), *code.reveal(2), run_time=0.6)
            t.wait_until("Here's the last block")
            self.play(FadeIn(g), FadeIn(cap), run_time=0.7)
            t.wait_until("Only its first 67")
            self.play(LaggedStart(*[g.fill(c, LIVE, tint=0.6) for c in live], lag_ratio=0.02),
                      FadeIn(key[0]), run_time=1.2)
            t.wait_until("Warps 0 and 1")
            self.play(g.fill(VGroup(*dead), DEAD, tint=0.35), FadeIn(key[1]), run_time=0.9)
            t.wait_until("Only warp 2")
            self.play(Create(split), FadeIn(split_lab), run_time=0.8)

    def block_size(self):
        self.slide("block-size", "Keep enough warps resident to hide the wait",
                   book_section="s-grid-geometry-and-the-tail")
        sm = region("One SM", 4.6, 4.4, color=self.color)
        pills = VGroup()
        for k in range(8):
            r = Rectangle(width=2.6, height=0.36, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                          fill_color=th.tint(LIVE, 0.3), fill_opacity=1)
            lab = kit.label(f"warp {k}", size=th.SIZE_LABEL, color=th.INK_2).move_to(r)
            pills.add(VGroup(r, lab))
        pills.arrange(DOWN, buff=0.08)
        pills.move_to(sm.rect.get_center() + DOWN * 0.2 + RIGHT * 0.45)
        ptr = Triangle(color=th.INK, fill_color=th.INK, fill_opacity=1, stroke_width=0).scale(0.12).rotate(-PI / 2)
        sched = kit.label("issue", size=th.SIZE_LABEL, color=th.INK_3)
        key = VGroup(swatch("ready", LIVE, 0.3), swatch("waiting on HBM", DEAD, 0.35)).arrange(DOWN, buff=0.12,
                                                                                               aligned_edge=LEFT)
        fig = VGroup(sm, pills)
        tab = kit.table(
            [["$2^{24}$", "65,536", "607"], ["$2^{20}$", "4096", "38"]],
            header=["$n$", "Blocks at 256", "Per SM"],
            align=["l", "r", "r"],
            size=th.SIZE_SMALL,
        )
        floor = kit.prose(r"Fewer than about 216 blocks, two per SM, leaves SMs idle.", size=th.SIZE_SMALL,
                          width=5.6)
        start = kit.prose(r"Start at \textbf{256 threads}: 8 warps, and several blocks fit on one SM.",
                          size=th.SIZE_SMALL, width=5.6)
        side = VGroup(key, start, floor, tab).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        layout = VGroup(fig, side).arrange(RIGHT, buff=0.8)
        self.place(layout)

        def at(k):
            return pills[k].get_left() + LEFT * 0.22

        ptr.move_to(at(0))
        sched.next_to(ptr, LEFT, buff=0.1)
        with self.narrate(
            "How big should a block be? "
            "A load from HBM takes hundreds of cycles. "
            "The SM doesn't wait; it switches to another ready warp, like a cook stirring a second pot. "
            "So you hide latency by keeping many warps resident. "
            "Start with 256 threads: 8 warps, small enough that several blocks share an SM. "
            "Then fill the machine. "
            "Fewer than about 216 blocks, two per SM, leaves SMs idle. "
            "At 2 to the 24 elements, you get 65,536 blocks, 607 per SM. "
            "At 2 to the 20, it's 4096 blocks, 38 per SM, and both are ample."
        ) as t:
            self.play(FadeIn(sm), FadeIn(pills), FadeIn(key), run_time=0.8)
            t.wait_until("A load from HBM")
            self.play(FadeIn(ptr), FadeIn(sched), run_time=0.4)
            self.play(pills[0][0].animate.set_fill(th.tint(self.color, 0.6)), run_time=0.4)
            self.play(pills[0][0].animate.set_fill(th.tint(DEAD, 0.35)), run_time=0.5)
            t.wait_until("The SM doesn't wait")
            for k in range(1, 6):
                self.play(ptr.animate.move_to(at(k)), sched.animate.next_to(at(k), LEFT, buff=0.22), run_time=0.3)
                self.play(pills[k][0].animate.set_fill(th.tint(self.color, 0.6)), run_time=0.25)
                anims = [pills[k][0].animate.set_fill(th.tint(DEAD, 0.35))]
                if k == 4:
                    anims.append(pills[0][0].animate.set_fill(th.tint(LIVE, 0.3)))
                if k == 5:
                    anims.append(pills[1][0].animate.set_fill(th.tint(LIVE, 0.3)))
                self.play(*anims, run_time=0.3)
            t.wait_until("Start with 256")
            self.play(FadeIn(start), run_time=0.6)
            t.wait_until("Fewer than about")
            self.play(FadeIn(floor), run_time=0.6)
            t.wait_until("At 2 to the 24")
            self.play(FadeIn(tab.rules), FadeIn(VGroup(*tab.cells[0])), FadeIn(VGroup(*tab.cells[1])), run_time=0.7)
            t.wait_until("At 2 to the 20")
            self.play(FadeIn(VGroup(*tab.cells[2])), run_time=0.6)

    def grid_stride(self):
        self.slide("grid-stride", "Size the grid to the machine, then stride",
                   book_section="s-the-grid-stride-loop")
        panel = kit.code(STRIDE_LOOP, "cuda")
        n, cells = 21, 24
        strip = kit.grid(1, cells, cell=(0.48, 0.5), values=[[str(c) for c in range(cells)]], size=th.SIZE_LABEL)
        for c in range(cells):
            strip.cell(0, c).text.set_z_index(3)
        for c in range(n, cells):
            strip.paint(strip.cell(0, c), th.INK_4, tint=0.15)
        past = kit.label("past n", size=th.SIZE_LABEL, color=th.INK_4).next_to(strip.block(0, 1, n, cells), DOWN,
                                                                             buff=0.12)
        cap = kit.label("2 blocks × 4 threads: stride 8, n = 21", size=th.SIZE_LABEL, color=th.INK_3)
        cap.next_to(strip, UP, buff=0.55).align_to(strip, LEFT)
        fig = VGroup(cap, strip, past)
        col = VGroup(panel, fig).arrange(DOWN, buff=0.5)
        self.place(col)
        rake = VGroup()
        for j in range(8):
            colr = self.color if j < 4 else self.colors[1]
            d = Dot(radius=0.11, color=colr)
            d.next_to(strip.cell(0, j), UP, buff=0.12)
            rake.add(d)
        bar = Line(rake[0].get_center() + UP * 0.2, rake[-1].get_center() + UP * 0.2, color=th.INK_3,
                   stroke_width=3)
        rake_g = VGroup(bar, rake)
        step = (strip.cell(0, 8).get_x() - strip.cell(0, 0).get_x())

        def touch(lo):
            hi = min(lo + 8, n)
            items = [strip.fill(strip.cell(0, c), self.color if (c - lo) < 4 else self.colors[1], tint=0.4)
                     for c in range(lo, hi)]
            return items

        with self.narrate(
            "The alternative is to size the grid to the machine, and let each thread walk the array. "
            "Picture the grid as a rake. "
            "Each pass covers one element per thread. "
            "Then the rake moves forward by its own width. "
            "The grid size no longer depends on n. "
            "The loop condition is the guard. "
            "And within each pass, a warp still reads consecutive elements, so access stays coalesced."
        ) as t:
            self.play(FadeIn(panel), run_time=0.7)
            t.wait_until("Picture the grid")
            self.play(FadeIn(fig), FadeIn(rake_g), run_time=0.7)
            t.wait_until("Each pass covers")
            self.play(*touch(0), run_time=0.7)
            t.wait_until("Then the rake moves")
            self.play(rake_g.animate.shift(RIGHT * step), run_time=0.8)
            self.play(*touch(8), run_time=0.6)
            self.play(rake_g.animate.shift(RIGHT * step), run_time=0.8)
            self.play(*touch(16), run_time=0.6)
            t.wait_until("The loop condition")
            self.play(panel.highlight(5), Indicate(past, color=th.BAD), run_time=0.8)
            t.wait_until("And within each")
            self.play(panel.highlight(3), run_time=0.6)

    # ------------------------------------------------------------------
    # Warps and divergence

    def divergence(self):
        self.slide("divergence", "A split warp runs both paths", book_section="s-warps-and-divergence")
        lanes = kit.grid(1, 32, cell=(0.34, 0.42))
        lab = kit.label("warp 0: lanes 0 to 31, one instruction at a time", size=th.SIZE_LABEL, color=th.INK_3)
        branch = kit.mono("if (i % 2 == 0) { path A } else { path B }", size=th.SIZE_SMALL, color=th.INK)
        head = VGroup(lab, branch).arrange(RIGHT, buff=0.8)
        unit = 1.6
        seg_a = Rectangle(width=2 * unit, height=0.42, stroke_width=0, fill_color=self.color, fill_opacity=0.8)
        seg_b = Rectangle(width=2 * unit, height=0.42, stroke_width=0, fill_color=self.colors[1], fill_opacity=0.8)
        time = VGroup(seg_a, seg_b).arrange(RIGHT, buff=0)
        ta = kit.label("path A", size=th.SIZE_LABEL, color=th.CANVAS, weight=MEDIUM).move_to(seg_a)
        tb = kit.label("path B", size=th.SIZE_LABEL, color=th.CANVAS, weight=MEDIUM).move_to(seg_b)
        cost = kit.label("cost: the sum of both paths", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        tl = VGroup(time, ta, tb)
        trow = VGroup(tl, cost).arrange(RIGHT, buff=0.5)
        tab = kit.table(
            [
                [r"\texttt{i < n}, the tail guard", "1 of 31,256", "Nothing"],
                [r"\texttt{i \% 2 == 0}", "Every warp", "Doubles the kernel"],
                [r"\texttt{blockIdx.x \% 2 == 0}", "None", "Nothing: a warp shares one block"],
            ],
            header=["Branch condition", "Warps that diverge", "Cost"],
            size=th.SIZE_SMALL,
        )
        col = VGroup(head, lanes, trow, tab).arrange(DOWN, buff=0.45)
        trow.align_to(lanes, LEFT)
        self.place(col)
        even = VGroup(*[lanes.cell(0, c) for c in range(0, 32, 2)])
        odd = VGroup(*[lanes.cell(0, c) for c in range(1, 32, 2)])
        with self.narrate(
            "The warp is the hardware: its 32 lanes issue one instruction at a time. "
            "Picture a squad at a fork, some going left and some right. "
            "The squad walks one path while the others wait, then walks the other. "
            "So when lanes disagree on a branch, the warp runs both sides with lanes masked. "
            "A split warp pays the sum of both paths. "
            "Divergence counts per warp. "
            "The tail guard splits 1 warp in 31,256, which costs nothing. "
            "Branching on an even index splits every warp, doubling the kernel. "
            "Branching on the block index splits none."
        ) as t:
            self.play(FadeIn(lab), FadeIn(lanes), run_time=0.7)
            t.wait_until("Picture a squad")
            self.play(FadeIn(branch), run_time=0.6)
            t.wait_until("The squad walks")
            self.play(lanes.fill(even, self.color, tint=0.6), lanes.fill(odd, DEAD, tint=0.2),
                      GrowFromEdge(seg_a, LEFT), FadeIn(ta), run_time=1.0)
            t.wait_until("So when lanes")
            self.play(lanes.fill(odd, self.colors[1], tint=0.6), lanes.fill(even, DEAD, tint=0.2),
                      GrowFromEdge(seg_b, LEFT), FadeIn(tb), run_time=1.0)
            t.wait_until("A split warp pays")
            self.play(FadeIn(cost), lanes.fill(even, self.color, tint=0.35), run_time=0.7)
            t.wait_until("Divergence counts")
            self.play(FadeIn(tab.rules), FadeIn(VGroup(*tab.cells[0])), run_time=0.5)
            for r, cue in zip((1, 2, 3), ["The tail guard splits", "Branching on an even", "Branching on the block"]):
                t.wait_until(cue)
                self.play(FadeIn(VGroup(*tab.cells[r]), shift=RIGHT * 0.1), run_time=0.6)

    # ------------------------------------------------------------------
    # Coalescing and the lab's 6.9x

    def coalescing(self):
        self.slide("coalescing", "A warp's load moves whole 32-byte sectors",
                   book_section="s-coalescing-sector-by-sector")
        good = VGroup(*[crate(8, self.color) for _ in range(4)]).arrange(RIGHT, buff=0.08)
        bad = VGroup(*[crate(1, self.colors[1]) for _ in range(32)]).arrange_in_grid(2, 16, buff=(0.08, 0.14))
        g_lab = VGroup(kit.label("Coalesced", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM),
                       kit.mono("out[i] = a[i];", size=th.SIZE_LABEL)).arrange(RIGHT, buff=0.4)
        b_lab = VGroup(kit.label("Strided by 32 floats", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM),
                       kit.mono("out[i] = a[i * 32];", size=th.SIZE_LABEL)).arrange(RIGHT, buff=0.4)
        g_note = kit.label("4 sectors, every byte used", size=th.SIZE_LABEL, color=th.INK_3)
        g_row = VGroup(good, g_note).arrange(RIGHT, buff=0.4)
        b_note = kit.label("32 sectors, 4 of every 32 bytes used", size=th.SIZE_LABEL, color=th.INK_3)
        top = VGroup(VGroup(g_lab, g_row).arrange(DOWN, buff=0.2, aligned_edge=LEFT),
                     VGroup(b_lab, bad, b_note).arrange(DOWN, buff=0.2, aligned_edge=LEFT)).arrange(
            DOWN, buff=0.45, aligned_edge=LEFT)
        bars = kit.bar_compare([("Coalesced", 128, self.color, "128 bytes"),
                                ("Strided", 1024, self.colors[1], "1024 bytes")],
                               width=5.2, bar_height=0.38, gap=0.2, size=th.SIZE_SMALL)
        verdict = kit.pill("12.5% efficiency: 8× read amplification, not 32×", self.color, size=th.SIZE_SMALL)
        bottom = VGroup(bars, verdict).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        col = VGroup(top, bottom).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        self.place(col)
        with self.narrate(
            "Now the property that decides every memory-bound kernel. "
            "Memory ships only whole 32-byte sectors, like crates. "
            "A warp's 32 threads each want one 4-byte float. "
            "When thread t reads element t, the floats span 128 bytes: exactly 4 sectors, every byte used. "
            "Now stride by 32 floats, so neighbors land 128 bytes apart. "
            "Each float sits alone in its own sector, and 28 bytes of every crate are wasted. "
            "The warp moves 1024 bytes to deliver 128: 12.5 percent efficiency. "
            "That's an eightfold amplification, not 32, because the unit is the sector."
        ) as t:
            t.wait_until("When thread t")
            self.play(FadeIn(g_lab), LaggedStart(*[FadeIn(c, shift=DOWN * 0.1) for c in good], lag_ratio=0.25),
                      run_time=1.0)
            self.play(FadeIn(g_note), run_time=0.5)
            t.wait_until("Now stride")
            self.play(FadeIn(b_lab), run_time=0.5)
            t.wait_until("Each float sits")
            self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.1) for c in bad], lag_ratio=0.06), run_time=1.6)
            self.play(FadeIn(b_note), run_time=0.4)
            t.wait_until("The warp moves")
            self.play(FadeIn(bars.labels), *bars.grow(run_time=1.0), run_time=1.0)
            self.play(FadeIn(bars.values), run_time=0.4)
            t.wait_until("That's an eightfold")
            self.play(FadeIn(verdict, scale=0.9), run_time=0.6)

    def lab_ratio(self):
        self.slide("lab-ratio", "The lab measures a 6.9× gap", book_section="s-derive-the-lab-s-6-9x")
        tab = kit.table(
            [
                ["Coalesced vector add: 2 reads, 1 write", r"$3n \times 4$", "1304 GB/s"],
                ["Strided copy: 1 read, 1 write", r"$2n \times 4$", "189 GB/s"],
            ],
            header=[r"Kernel, $n = 2^{24}$ floats", "Counted bytes", "Measured"],
            align=["l", "l", "r"],
            size=th.SIZE_SMALL,
        )
        bars = kit.bar_compare(
            [("Measured copy", 1275, th.INK_4, "1275 GB/s"),
             ("Vector add", 1304, self.color, "1304 GB/s"),
             ("Strided copy", 189, self.colors[1], "189 GB/s")],
            width=6.4, bar_height=0.42, gap=0.25,
        )
        ratio = kit.formula(r"\frac{1304}{189} = 6.9", size=th.SIZE_MATH)
        box = SurroundingRectangle(ratio, color=self.color, buff=0.16, corner_radius=0.08, stroke_width=3)
        low = VGroup(bars, VGroup(ratio, box)).arrange(RIGHT, buff=0.9)
        col = VGroup(tab, low).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "The lab runs both patterns on 2 to the 24 floats. "
            "The coalesced vector add reaches 1304 GB/s. "
            "That's level with the 1275 GB/s copy, so nothing is left to optimize. "
            "The strided copy reaches only 189 GB/s. "
            "The ratio is 6.9. "
            "The sector model predicted 8, so where did the difference go?"
        ) as t:
            self.play(FadeIn(tab.rules), FadeIn(VGroup(*tab.cells[0])), run_time=0.6)
            t.wait_until("The coalesced vector")
            self.play(FadeIn(VGroup(*tab.cells[1])), FadeIn(bars.labels[1]), GrowFromEdge(bars.bars[1], LEFT),
                      FadeIn(bars.values[1]), run_time=0.9)
            t.wait_until("That's level")
            self.play(FadeIn(bars.labels[0]), GrowFromEdge(bars.bars[0], LEFT), FadeIn(bars.values[0]), run_time=0.9)
            t.wait_until("The strided copy")
            self.play(FadeIn(VGroup(*tab.cells[2])), FadeIn(bars.labels[2]), GrowFromEdge(bars.bars[2], LEFT),
                      FadeIn(bars.values[2]), run_time=0.9)
            t.wait_until("The ratio is")
            self.play(FadeIn(ratio), Create(box), run_time=0.8)

    def two_factors(self):
        self.slide("two-factors", "6.9 = 4.5 × 1.53: wasted bytes, then wasted requests",
                   book_section="s-derive-the-lab-s-6-9x")
        u = 0.19
        rd, wr = self.colors[1], self.colors[2]

        def bar(parts):
            g = VGroup()
            for n_bytes, colr, op in parts:
                g.add(Rectangle(width=n_bytes * u, height=0.45, stroke_color=th.CANVAS, stroke_width=1.5,
                                fill_color=colr, fill_opacity=op))
            return g.arrange(RIGHT, buff=0)

        counted = bar([(4, rd, 0.85), (4, wr, 0.85)])
        real = bar([(4, rd, 0.85), (28, rd, 0.25), (4, wr, 0.85)])
        n1 = kit.label("counted", size=th.SIZE_SMALL, color=th.INK_2)
        n2 = kit.label("really moved", size=th.SIZE_SMALL, color=th.INK_2)
        v1 = kit.label("8 bytes", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        v2 = kit.label("36 bytes", size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
        r1 = VGroup(n1, counted, v1)
        r2 = VGroup(n2, real, v2)
        for r in (r1, r2):
            r[1].next_to(r[0], RIGHT, buff=0.3)
            r[2].next_to(r[1], RIGHT, buff=0.25)
        rows = VGroup(r1, r2).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        counted.align_to(real, LEFT)
        v1.next_to(counted, RIGHT, buff=0.25)
        key = VGroup(swatch("read: a whole sector for 4 useful bytes", rd, 0.6),
                     swatch("write: coalesced, 4 bytes", wr, 0.6)).arrange(RIGHT, buff=0.5)
        steps = VGroup(
            kit.formula(r"32 + 4 = 36 \text{ bytes per element}, \quad 36 / 8 = 4.5", size=th.SIZE_MATH_SMALL),
            kit.formula(r"1304 / 4.5 = 290 \text{ GB/s predicted},", r"\quad 290 / 189 = 1.53",
                        size=th.SIZE_MATH_SMALL),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        result = kit.formula(r"6.9", r"=", r"4.5", r"\times", r"1.53", size=th.SIZE_MATH)
        result[2].set_color(BYTES_C)
        result[4].set_color(REQ_C)
        rbox = SurroundingRectangle(result, color=th.INK_3, buff=0.16, corner_radius=0.08, stroke_width=2.5)
        l2 = kit.prose(r"The strided reads touch 16.8 MB, inside the 40 MB L2, which flatters them.",
                       size=th.SIZE_SMALL, color=th.INK_3, width=None)
        col = VGroup(VGroup(rows, key).arrange(DOWN, buff=0.3), steps, VGroup(result, rbox), l2).arrange(
            DOWN, buff=0.45)
        self.place(col)
        b_lab = kit.label("bytes", size=th.SIZE_LABEL, color=BYTES_C).next_to(rbox, LEFT, buff=0.3)
        q_lab = kit.label("requests", size=th.SIZE_LABEL, color=REQ_C).next_to(rbox, RIGHT, buff=0.3)
        with self.narrate(
            "Count the bytes that really move. "
            "The strided copy counts 8 per element: 4 read and 4 written. "
            "The write is coalesced, so it moves 4. "
            "But the read pulls a whole 32-byte sector. "
            "So each element really moves 36 bytes, an amplification of 4.5. "
            "At the same real rate, the strided kernel would show 290 GB/s. "
            "It measures 189, so 1.53 is left over. "
            "That factor is requests, not bytes: 32 sector requests per warp instead of 4, queueing on the way to L2. "
            "And the strided reads fit in L2, so a larger footprint would cost more than 6.9."
        ) as t:
            self.play(FadeIn(n1), FadeIn(n2), run_time=0.5)
            t.wait_until("The strided copy counts")
            self.play(GrowFromEdge(counted, LEFT), FadeIn(v1), run_time=0.8)
            t.wait_until("The write is")
            self.play(FadeIn(real[2]), FadeIn(key[1]), run_time=0.6)
            t.wait_until("But the read")
            self.play(FadeIn(real[0]), GrowFromEdge(real[1], LEFT), FadeIn(key[0]), run_time=0.9)
            t.wait_until("So each element")
            self.play(FadeIn(v2), FadeIn(steps[0]), run_time=0.7)
            t.wait_until("At the same real")
            self.play(FadeIn(steps[1][0]), run_time=0.7)
            t.wait_until("It measures 189")
            self.play(FadeIn(steps[1][1]), run_time=0.6)
            self.play(FadeIn(result), Create(rbox), FadeIn(b_lab), run_time=0.8)
            t.wait_until("That factor is")
            self.play(FadeIn(q_lab), Indicate(result[4], color=REQ_C), run_time=0.8)
            t.wait_until("And the strided reads")
            self.play(FadeIn(l2), run_time=0.6)

    # ------------------------------------------------------------------
    # The memory hierarchy, shared memory

    def memory_hierarchy(self):
        self.slide("memory-hierarchy", "The fast storage is small", book_section="s-the-memory-hierarchy")
        levels = [
            ("Registers", "256 KB per SM", "~1 cycle", "one thread", self.color, 2.8),
            ("Shared memory", "164 KB per SM", "~30 cycles", "one block", self.colors[2], 3.8),
            ("L2 cache", "40 MB", "~200 cycles", "whole GPU", self.colors[3], 4.8),
            ("HBM", "80 GB", "~400 cycles", "whole GPU", self.colors[1], 5.8),
        ]
        rows = VGroup()
        for name, size, lat, scope, colr, w in levels:
            rect = Rectangle(width=w, height=0.62, stroke_color=colr, stroke_width=th.STROKE_BOX,
                             fill_color=th.tint(colr, 0.18), fill_opacity=1)
            nm = kit.label(name, size=th.SIZE_SMALL, color=th.INK, weight=MEDIUM)
            nm.move_to(rect.get_left() + RIGHT * (0.2 + nm.width / 2))
            sz = VGroup()
            info = kit.label(f"{size}  ·  {lat}  ·  {scope}", size=th.SIZE_LABEL, color=th.INK_2)
            rows.add(VGroup(rect, nm, sz, info))
        rows.arrange(DOWN, buff=0.14)
        for r in rows:
            r.align_to(rows, LEFT)
        for r in rows:
            r[1].move_to(r[0].get_left() + RIGHT * (0.2 + r[1].width / 2))
            r[3].move_to([rows.get_right()[0] + 0.4 + r[3].width / 2, r[0].get_y(), 0])
        for r in rows:
            r[3].align_to(rows[0][3], LEFT)
        totals = kit.bar_compare(
            [("Registers, all 108 SMs", 27.6, self.color, "27.6 MB"),
             ("Shared memory, all SMs", 17.7, self.colors[2], "17.7 MB"),
             ("L2 cache", 40, self.colors[3], "40 MB")],
            width=4.6, bar_height=0.36, gap=0.2, size=th.SIZE_SMALL,
        )
        theme = kit.callout("key", r"Move data up this hierarchy once, and use it many times: FlashAttention, "
                                   r"fusion, and tiling.", title="Every later optimization", width=5.4,
                            size=th.SIZE_SMALL)
        low = VGroup(totals, theme).arrange(RIGHT, buff=0.6)
        col = VGroup(rows, low).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "Where can data live? "
            "Registers are one thread's own, about one cycle away. "
            "Shared memory belongs to one block, about 30 cycles away. "
            "The L2 cache serves the whole GPU, at about 200 cycles. "
            "HBM is about 400. "
            "Across 108 SMs, the registers total 27.6 MB, and shared memory 17.7 MB. "
            "Both are smaller than the 40 MB L2. "
            "Every later optimization moves data up this hierarchy once, and reuses it many times."
        ) as t:
            for i, cue in enumerate(["Registers are", "Shared memory belongs", "The L2 cache serves", "HBM is about"]):
                t.wait_until(cue)
                self.play(FadeIn(rows[i], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Across 108")
            self.play(FadeIn(totals.labels[:2]), GrowFromEdge(totals.bars[0], LEFT), GrowFromEdge(totals.bars[1], LEFT),
                      FadeIn(totals.values[:2]), run_time=1.0)
            t.wait_until("Both are smaller")
            self.play(FadeIn(totals.labels[2]), GrowFromEdge(totals.bars[2], LEFT), FadeIn(totals.values[2]),
                      run_time=0.8)
            t.wait_until("Every later")
            self.play(FadeIn(theme, shift=UP * 0.1), run_time=0.7)

    def tree_reduction(self):
        self.slide("tree-reduction", "256 partial sums, 8 halving rounds",
                   book_section="s-shared-memory-and-synchronization")
        panel = kit.code(REDUCE, "cuda", first_line=17, caption="rms_norm_kernel")
        slots = kit.grid(1, 8, cell=(0.8, 0.6), values=[[f"s{k}" for k in range(8)]], size=th.SIZE_SMALL)
        for c in range(8):
            slots.cell(0, c).text.set_z_index(3)
        s_lab = kit.label("shared array, 8 slots", size=th.SIZE_LABEL, color=th.INK_3)
        s_lab.next_to(slots, DOWN, buff=0.15).align_to(slots, LEFT)
        log = VGroup(*[kit.label(text, size=th.SIZE_LABEL, color=th.INK_2) for text in
                       ["offset 4: threads 0 to 3 add", "offset 2: threads 0 and 1 add",
                        "offset 1: slot 0 holds all 8"]]).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        rounds = kit.prose(r"$\log_2 8 = 3$ rounds; at 256 threads, 8.", size=th.SIZE_SMALL, color=self.color,
                           width=None)
        side = VGroup(log, rounds).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        low = VGroup(VGroup(slots, s_lab), side).arrange(RIGHT, buff=0.7, aligned_edge=DOWN)
        layout = VGroup(panel, low).arrange(DOWN, buff=0.85)
        self.place(layout)

        def arcs(off):
            g = VGroup()
            for j in range(off):
                a = slots.cell(0, j + off).get_top() + UP * 0.05
                b = slots.cell(0, j).get_top() + UP * 0.05
                g.add(CurvedArrow(a, b, angle=PI / 2.4, color=self.color, stroke_width=2.5, tip_length=0.14))
            return g

        with self.narrate(
            "The lab's RMSNorm kernel uses one block per row, and must add up 256 partial sums of squares. "
            "Each thread writes its partial sum to a shared slot, then waits at a barrier until the whole block "
            "arrives. "
            "Then the block reduces as a tree. "
            "On 8 slots, the lower 4 threads add the slot 4 places up. "
            "After a barrier, the lower 2 add the slot 2 places up. "
            "One more round, and slot 0 holds all 8. "
            "That's 3 rounds for 8 threads, and 8 rounds for 256."
        ) as t:
            self.play(FadeIn(panel.frame), run_time=0.5)
            t.wait_until("Each thread writes")
            self.play(FadeIn(slots), FadeIn(s_lab), run_time=0.6)
            self.play(LaggedStart(*[slots.fill(slots.cell(0, c), self.color, tint=0.15) for c in range(8)],
                                  lag_ratio=0.1), run_time=0.8)
            t.wait_until("Then the block reduces")
            self.play(*panel.reveal(17, 18, 19, 20, 21, 22, 23), panel.highlight(20), run_time=0.8)
            prev = None
            for off, cue, k in [(4, "On 8 slots", 0), (2, "After a barrier", 1), (1, "One more round", 2)]:
                t.wait_until(cue)
                a = arcs(off)
                anims = [Create(a), FadeIn(log[k], shift=RIGHT * 0.1)]
                if prev is not None:
                    anims.append(FadeOut(prev))
                if off == 2:
                    anims.append(panel.highlight(22))
                self.play(*anims, run_time=0.8)
                self.play(slots.fill(slots.block(0, 1, 0, off), self.color, tint=0.25 + 0.2 * (3 - off.bit_length())),
                          slots.fill(slots.block(0, 1, off, 2 * off), DEAD, tint=0.15), run_time=0.5)
                prev = a
            t.wait_until("That's 3 rounds")
            self.play(FadeIn(rounds), run_time=0.6)

    def five_details(self):
        self.slide("five-details", "Five details decide whether it works",
                   book_section="s-shared-memory-and-synchronization")
        items = kit.bullets(
            [
                r"The shared array's size comes from the launch: 256 floats, 1024 bytes.",
                r"Stride by the block size: 4 sectors per load, not 16.",
                r"\textbf{The barrier sits outside the \texttt{if}.}",
                r"The halving tree needs a power of two: from 200, element 24 is lost.",
                r"A tree adds similar magnitudes, so rounding grows like $\log_2 B$, not $B$.",
            ],
            size=th.SIZE_SMALL, width=11.6, buff=0.28,
        )
        warn = kit.callout("warning", r"Inside \texttt{if (threadIdx.x < offset)}, the threads above \texttt{offset} "
                                      r"never arrive, and the block stops responding.",
                           title="A barrier inside a branch stops the block", width=11.6, size=th.SIZE_SMALL)
        col = VGroup(items, warn).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        self.place(col)
        with self.narrate(
            "Five details decide whether it works. "
            "The shared array's size comes from the launch: 1024 bytes. "
            "The loop strides by the block size, so each load hits 4 sectors, not 16. "
            "The barrier must sit outside the if. "
            "Inside it, the upper threads never arrive, and the block hangs. "
            "The halving tree needs a power of two. "
            "And a tree's rounding grows with the log of the block size, not the size itself."
        ) as t:
            for i, cue in enumerate(["The shared array's", "The loop strides", "The barrier must"]):
                t.wait_until(cue)
                self.play(FadeIn(items[i], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("Inside it")
            self.play(FadeIn(warn, shift=UP * 0.1), Indicate(items[2], color=th.BAD, scale_factor=1.03),
                      run_time=0.9)
            t.wait_until("The halving tree needs")
            self.play(FadeIn(items[3], shift=RIGHT * 0.1), run_time=0.6)
            t.wait_until("And a tree's")
            self.play(FadeIn(items[4], shift=RIGHT * 0.1), run_time=0.6)

    # ------------------------------------------------------------------
    # Check your work, time a kernel

    def async_errors(self):
        self.slide("async-errors", "Errors surface late, and they stick", book_section="s-check-your-work")
        names = [("launch", "reads out of bounds"), ("next op", "looks fine"), ("next op", "looks fine"),
                 ("synchronize", "raises here")]
        calls = VGroup(*[kit.box(a, sublabel=b, width=2.6, size=th.SIZE_SMALL) for a, b in names])
        calls.arrange(RIGHT, buff=0.55)
        lane = kit.label("host calls, in order", size=th.SIZE_LABEL, color=th.INK_3)
        lane.next_to(calls, UP, buff=0.18).align_to(calls, LEFT)
        panel = kit.code(CHECKS, "cuda", line_numbers=False)
        habits = kit.label("Debug with CUDA_LAUNCH_BLOCKING=1, and check with compute-sanitizer.",
                           size=th.SIZE_SMALL, color=th.INK_2)
        col = VGroup(VGroup(lane, calls), panel, habits).arrange(DOWN, buff=0.5)
        panel.shift(DOWN * 0.45)
        habits.shift(DOWN * 0.45)
        self.place(col)
        links = VGroup(*[kit.arrow(a, b, color=th.INK_4) for a, b in zip(calls[:-1], calls[1:])])
        cause = CurvedArrow(calls[3].get_bottom() + DOWN * 0.08, calls[0].get_bottom() + DOWN * 0.08, angle=-PI / 9,
                            color=th.BAD, stroke_width=3, tip_length=0.16)
        cause_lab = kit.label("the real cause", size=th.SIZE_LABEL, color=th.BAD)
        cause_lab.next_to(cause, DOWN, buff=0.02)
        with self.narrate(
            "Catch wrong answers by comparing with PyTorch at awkward sizes, like 1,000,003. "
            "Silent failures are worse, because CUDA errors are asynchronous. "
            "A kernel that reads out of bounds doesn't fail at the launch. "
            "It fails at the next synchronizing call, on an innocent line. "
            "So check two channels. "
            "The last error after the launch catches bad configurations. "
            "A synchronize reports what the kernel did. "
            "Errors are sticky, so the first one in your log is the real one."
        ) as t:
            t.wait_until("A kernel that reads")
            self.play(FadeIn(lane), FadeIn(calls[0]), run_time=0.6)
            for k in range(1, 4):
                self.play(GrowArrow(links[k - 1].line), FadeIn(calls[k]), run_time=0.45)
            t.wait_until("It fails at")
            self.play(calls[3].rect.animate.set_stroke(th.BAD, width=3).set_fill(th.tint(th.BAD, 0.12)),
                      Create(cause), FadeIn(cause_lab), run_time=1.0)
            t.wait_until("So check two")
            self.play(FadeIn(panel.frame), *panel.reveal(1), run_time=0.6)
            t.wait_until("The last error")
            self.play(*panel.reveal(2, 3), panel.highlight(3), run_time=0.7)
            t.wait_until("A synchronize reports")
            self.play(*panel.reveal(4, 5), panel.highlight(5), run_time=0.7)
            t.wait_until("Errors are sticky")
            self.play(*[c.rect.animate.set_stroke(th.BAD, width=3) for c in calls[1:3]], run_time=0.7)
            self.play(Indicate(calls[0].rect, color=th.BAD, scale_factor=1.05), FadeIn(habits), run_time=0.8)

    def timing(self):
        self.slide("timing", "Time the device, not the queue", book_section="s-time-a-kernel")
        u = 1.0
        x0 = 0.0
        host_l = kit.label("Host", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        gpu_l = kit.label("GPU", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        axis_h = Line([x0, 0, 0], [x0 + 9.0 * u, 0, 0], color=th.RULE_STRONG, stroke_width=th.STROKE_AXIS)
        axis_g = Line([x0, -1.3, 0], [x0 + 9.0 * u, -1.3, 0], color=th.RULE_STRONG, stroke_width=th.STROKE_AXIS)
        host_l.next_to(axis_h, LEFT, buff=0.3)
        gpu_l.next_to(axis_g, LEFT, buff=0.3).align_to(host_l, LEFT)
        launch = chip(0.5, 0.42, self.colors[1], None, tint=0.5).move_to([x0 + 0.5, 0.25, 0])
        kern = chip(5.0, 0.42, self.color, "kernel runs", tint=0.4, font=th.SANS).move_to([x0 + 4.4, -1.05, 0])
        cpu_b = BraceBetweenPoints([x0 + 0.25, 0.55, 0], [x0 + 0.75, 0.55, 0], UP, color=th.BAD)
        cpu_t = kit.label("CPU timer: about 5 µs of queueing", size=th.SIZE_LABEL, color=th.BAD)
        cpu_t.next_to(cpu_b, UP, buff=0.08).align_to(cpu_b, LEFT)
        ev_s = Line([x0 + 1.8, -0.7, 0], [x0 + 1.8, -1.45, 0], color=th.GOOD, stroke_width=4)
        ev_e = Line([x0 + 7.0, -0.7, 0], [x0 + 7.0, -1.45, 0], color=th.GOOD, stroke_width=4)
        ev_b = BraceBetweenPoints([x0 + 1.8, -1.55, 0], [x0 + 7.0, -1.55, 0], DOWN, color=th.GOOD)
        ev_t = kit.label("start and end events: device time", size=th.SIZE_LABEL, color=th.GOOD)
        ev_t.next_to(ev_b, DOWN, buff=0.08)
        fig = VGroup(host_l, gpu_l, axis_h, axis_g, launch, kern, cpu_b, cpu_t, ev_s, ev_e, ev_b, ev_t)
        rules = kit.bullets(
            [
                r"Warm up: \texttt{engine/bench.py} runs 5 untimed, then 20 timed.",
                r"Report the median, not the mean.",
                r"Convert to GB/s; above 80\% of 1275 GB/s, the kernel is finished.",
            ],
            size=th.SIZE_SMALL, width=11.0, buff=0.22,
        )
        col = VGroup(fig, rules).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "A launch returns before the GPU finishes. "
            "So a CPU timer around it measures only the queueing, about 5 microseconds, whatever the kernel does. "
            "CUDA events fix that: the GPU timestamps them in the stream, so they measure device time. "
            "Then get three things right. "
            "Warm up: the course's benchmark runs 5 untimed iterations, then 20 timed. "
            "Report the median, because one preempted run skews a mean. "
            "And compare bandwidth against the 1275 GB/s copy: above 80 percent of it, the kernel is finished."
        ) as t:
            self.play(FadeIn(host_l), FadeIn(gpu_l), Create(axis_h), Create(axis_g), run_time=0.7)
            self.play(FadeIn(launch), run_time=0.4)
            self.play(GrowFromEdge(kern, LEFT), run_time=1.2)
            t.wait_until("So a CPU timer")
            self.play(FadeIn(cpu_b), FadeIn(cpu_t), run_time=0.7)
            t.wait_until("CUDA events fix")
            self.play(Create(ev_s), Create(ev_e), FadeIn(ev_b), FadeIn(ev_t), run_time=0.9)
            for i, cue in enumerate(["Warm up:", "Report the median", "And compare"]):
                t.wait_until(cue)
                self.play(FadeIn(rules[i], shift=RIGHT * 0.1), run_time=0.6)
