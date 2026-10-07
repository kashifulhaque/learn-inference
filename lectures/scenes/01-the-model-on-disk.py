"""Chapter 01, the model on disk.

Teaches content/chapters/01-the-model-on-disk.md in its order. Every number
below was recomputed in Python and matches the chapter.

Files:
- 18 shards, about 54 GB together: 53,974,302,720 B / 18 = 3.00e9 B, "about
  3 GB each". Index total_size 53,974,302,720 B = 53.97 GB; chapter 2's
  language model alone is 53.79 GB; the difference (0.18 GB) is tensors
  outside the text path.
- One layer of host memory: 53.79 GB / 64 layers = 0.84 GB, the chapter's
  "about 0.8 GB".

Row-major layout:
- Toy 3 x 4 bf16 matrix: one row is 4 x 2 = 8 bytes, so rows start at byte
  0, 8, and 16.
- Element [i, j] of a (248320, 5120) tensor sits at (i x 5120 + j) x bytes
  per element.

Safetensors:
- k_proj: 1024 x 5120 x 2 = 10,485,760 bytes = end - begin; begin = 0, so the
  first byte is at 8 + N.
- Check question: up_proj 17408 x 5120 x 2 = 178,257,920 (not shown).

Full attention (head_dim 256 from the config):
- k_proj 1024 / 256 = 4 KV heads; o_proj 6144 / 256 = 24 query heads;
  24 / 4 = 6 query heads per KV head.
- Expected q_proj rows 24 x 256 = 6144; file 12,288 = 2 x 6144.

Linear attention (16 key heads, 48 value heads, both 128 wide):
- q, k: 16 x 128 = 2048 each; v, z: 48 x 128 = 6144 each;
  2048 + 2048 + 6144 + 6144 = 16,384. The alternative 4 x 4096 = 16,384.
- in_proj_ba: 2 x 48 = 96. conv1d: 2048 + 2048 + 6144 = 10,240 channels,
  kernel 4. A_log, dt_bias: 48. out_proj: 48 x 128 = 6144.

Config:
- Full attention where (i + 1) % 4 == 0: layers 3, 7, 11, ..., 63; 16 of
  them, 64 - 16 = 48 linear. The off-by-one version (i % 4 == 0) gives 0, 4,
  8, ....
- 24 x 256 = 6144, not 5120; 5120 / 24 = 213.3, "about 213".
- partial_rotary_factor 0.25 x 256 = 64 rotated channels.
- Embedding 248,320 x 5120 = 1,271,398,400 = 1.27B parameters.
"""

from manim import *

import kit
from kit import theme as th

LEN_C = th.HL_A  # the 8-byte length prefix (\hla in the chapter)
HEAD_C = th.HL_B  # the N-byte header (\hlb)
DATA_C = th.HL_C  # the data buffer and begin (\hlc)
TENSOR_C = th.VIOLET  # one tensor's bytes
KEY_C = th.HL_A  # blocks that come from the 16 key heads
VAL_C = th.HL_B  # blocks that come from the 48 value heads

HEADER_JSON = """\
{
  "model.language_model.layers.3.self_attn.k_proj.weight": {
    "dtype": "BF16",
    "shape": [1024, 5120],
    "data_offsets": [0, 10485760]
  },
  "__metadata__": {"format": "pt"}
}
"""

LOAD_LOOP = """\
for layer_idx in range(config.num_hidden_layers):
    for suffix, name in index.layer_tensors(layer_idx):
        # page in, then H2D copy
        tensor = index.get(name, device="cuda")
"""

INIT = """\
class WeightIndex:
    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        index_file = self.dir / "model.safetensors.index.json"
        if index_file.is_file():
            self.map = json.loads(index_file.read_text())["weight_map"]
        else:
            single = "model.safetensors"
            with safe_open(self.dir / single, framework="pt") as f:
                self.map = {name: single for name in f.keys()}
        self._handles = {}
"""

GET = """\
def get(self, name, device="cpu", dtype=None):
    shard = self.map[name]
    tensor = self._handle(shard).get_tensor(name)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor.to(device)
"""

NAMES = """\
# layer 0: linear attention
linear_attn.in_proj_qkvz   [16384, 5120]
linear_attn.in_proj_ba     [96, 5120]
linear_attn.conv1d         [10240, 1, 4]
linear_attn.A_log          [48]
linear_attn.dt_bias        [48]
linear_attn.norm
linear_attn.out_proj       [5120, 6144]
# layer 3: full attention
self_attn.q_proj           [12288, 5120]
self_attn.k_proj           [1024, 5120]
self_attn.v_proj           [1024, 5120]
self_attn.o_proj           [5120, 6144]
"""

LAYER_TYPES = """\
text = raw.get("text_config", raw)
layer_types = [
    "full_attention" if (i + 1) % interval == 0
    else "linear_attention"
    for i in range(text["num_hidden_layers"])
]
"""


def block(width, height, color, text=None, size=th.SIZE_SMALL, tint=0.3, font=None):
    """A filled rectangle standing for a run of bytes, with an optional label."""
    rect = Rectangle(width=width, height=height, stroke_color=color, stroke_width=th.STROKE_BOX,
                     fill_color=th.tint(color, tint), fill_opacity=1)
    group = VGroup(rect)
    group.rect = rect
    group.text = None
    if text:
        t = kit.prose(text, size=size, color=th.INK, width=None) if "$" in text else kit.label(
            text, size=size, color=th.INK, font=font or th.SANS)
        t.move_to(rect)
        group.add(t)
        group.text = t
    return group


def panel(title, width, height, color=None, sub=None, size=th.SIZE_SMALL):
    """A titled region (a memory or a device) with its label in the top-left corner."""
    stroke = th.RULE_STRONG if color is None else color
    rect = RoundedRectangle(width=width, height=height, corner_radius=0.12, stroke_color=stroke,
                            stroke_width=th.STROKE_BOX, fill_color=th.tint(stroke, 0.06), fill_opacity=1)
    name = kit.label(title, size=size, color=th.INK, weight=MEDIUM)
    parts = [name]
    if sub:
        parts.append(kit.label(sub, size=th.SIZE_LABEL, color=th.INK_3))
    head = VGroup(*parts).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
    head.move_to(rect.get_corner(UL) + RIGHT * (head.width / 2 + 0.2) + DOWN * (head.height / 2 + 0.16))
    group = VGroup(rect, head)
    group.rect, group.head = rect, head
    return group


def swatch(text, color, size=th.SIZE_LABEL):
    sq = Square(0.26, stroke_color=th.RULE_STRONG, stroke_width=th.STROKE_HAIRLINE,
                fill_color=th.tint(color, 0.35), fill_opacity=1)
    return VGroup(sq, kit.label(text, size=size, color=th.INK_2)).arrange(RIGHT, buff=0.16)


def brace_label(mob, text, direction=DOWN, color=th.INK_3, size=th.SIZE_LABEL):
    """A brace under (or over) `mob` with a short label."""
    b = Brace(mob, direction, buff=0.08, color=th.INK_4)
    tex = "$" in text or "\\" in text
    t = kit.prose(text, size=size, color=color, width=None) if tex else kit.label(text, size=size, color=color)
    t.next_to(b, direction, buff=0.08)
    return VGroup(b, t)


def lift_text(g):
    """Keeps a grid's cell text above its squares, which an animated fill raises."""
    for row in g.cells:
        for c in row:
            if c.text is not None:
                c.text.set_z_index(2)
    return g


def pointer(color=th.INK):
    return Triangle(color=color, fill_color=color, fill_opacity=1, stroke_width=0).scale(0.12).rotate(PI)


class Lecture(kit.Lecture):
    def construct(self):
        self.title_slide(
            narration=(
                "The model on disk. "
                "In this lecture, you parse a safetensors header by hand, and find any tensor through the index. "
                "Then you read the architecture from the tensor names and shapes alone."
            )
        )
        self.folder_of_files()
        self.bytes_and_layout()
        self.memory_mapping()
        self.three_parts()
        self.header_entry()
        self.absolute_offset()
        self.why_bare()
        self.layer_by_layer()
        self.load_loop()
        self.shards_index()
        self.weight_index()
        self.get_copies()
        self.two_layers()
        self.head_counts()
        self.output_gate()
        self.qkvz()
        self.linear_shapes()
        self.layer_types()
        self.config_fields()
        self.what_goes_wrong()
        self.closing_slide(
            [
                r"A tensor starts at $8 + N + \text{begin}$, and $\text{end} - \text{begin}$ is elements "
                r"$\times$ bytes per element.",
                r"Headers and the index give every name, dtype, and shape without reading a weight.",
                r"Mapping and layer-by-layer loading keep host memory near one layer, about 0.8 GB.",
                r"\texttt{q\_proj} is double width; read \texttt{head\_dim}; full attention sits on layers "
                r"3, 7, 11, \dots",
            ],
            narration=(
                "Here's what to keep. "
                "A tensor starts at 8 plus N plus begin, and its byte range must match its shape. "
                "Headers and the index give you every name, type, and shape without reading a weight. "
                "Memory mapping keeps host memory near one layer. "
                "The query projection is double width, the head width comes from the config, and full attention "
                "starts at layer 3. "
                "Next, you'll turn these shapes into a memory budget."
            ),
            book_section="s-what-goes-wrong",
        )

    # Introduction ------------------------------------------------------------

    def folder_of_files(self):
        self.slide("folder-of-files", "The model ships as a folder of files", book_section="s-before-you-start")
        shards = kit.grid(3, 6, cell=(1.15, 0.75), gap=0.1,
                          values=[[f"{r * 6 + c + 1:02d}" for c in range(6)] for r in range(3)])
        sh_t = kit.label("18 weight shards, about 3 GB each", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        left = VGroup(sh_t, shards).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        idx = kit.box("Index", sublabel="which shard holds each tensor", color=self.color, width=4.6,
                      size=th.SIZE_BODY)
        cfg = kit.box("Config", sublabel="the exact numbers", color=th.INFO, width=4.6, size=th.SIZE_BODY)
        tok = kit.box("Tokenizer", width=4.6, size=th.SIZE_BODY, height=1.0)
        right = VGroup(idx, cfg, tok).arrange(DOWN, buff=0.25)
        top = VGroup(left, right).arrange(RIGHT, buff=0.8, aligned_edge=UP)
        total = kit.pill("about 54 GB together", self.color, size=th.SIZE_SMALL)
        quote = kit.prose(r"Tensor names: the most honest description of the architecture.",
                          size=th.SIZE_BODY, width=None, color=th.INK_2)
        total.next_to(shards, DOWN, buff=0.35).align_to(shards, LEFT)
        col = VGroup(VGroup(top, total), quote).arrange(DOWN, buff=0.7)
        self.place(col)
        band = kit.mark(quote, buff=0.06)
        with self.narrate(
            "The model this course serves ships as a folder of files. "
            "The weights are split across 18 shards of about 3 GB each. "
            "A small index file says which shard holds each tensor. "
            "A config file holds the exact numbers, and a tokenizer sits alongside. "
            "Together, they come to about 54 GB. "
            "Inside the shards, the tensor names are the most honest description of the architecture you can get. "
            "They often disagree with the model card."
        ) as t:
            t.wait_until("The weights are split")
            self.play(FadeIn(sh_t), LaggedStart(*[FadeIn(c, scale=0.85) for row in shards.cells for c in row],
                                                lag_ratio=0.06), run_time=1.6)
            t.wait_until("A small index")
            self.play(FadeIn(idx, shift=LEFT * 0.1), run_time=0.6)
            t.wait_until("A config file")
            self.play(FadeIn(cfg, shift=LEFT * 0.1), FadeIn(tok, shift=LEFT * 0.1), run_time=0.7)
            t.wait_until("Together, they")
            self.play(FadeIn(total, scale=0.9), run_time=0.6)
            t.wait_until("Inside the shards")
            self.play(FadeIn(quote), run_time=0.7)
            self.play(FadeIn(band), run_time=0.5)

    def bytes_and_layout(self):
        self.slide("bytes-and-layout", "A tensor is a run of bytes, last index fastest",
                   book_section="s-before-you-start")
        rows, cols = 3, 4
        vals = [[f"{i},{j}" for j in range(cols)] for i in range(rows)]
        mat = lift_text(kit.grid(rows, cols, cell=(0.9, 0.6), values=vals, size=th.SIZE_LABEL))
        mat_t = kit.label("shape (3, 4), bf16", size=th.SIZE_LABEL, color=th.INK_3)
        m = VGroup(mat_t, mat).arrange(DOWN, buff=0.2)
        flat = kit.grid(1, rows * cols, cell=(0.9, 0.6), values=[[v for r in vals for v in r]], size=th.SIZE_LABEL)
        row_cols = [LEN_C, HEAD_C, DATA_C]
        for r in range(rows):
            flat.paint(flat.block(0, 1, r * cols, (r + 1) * cols), row_cols[r])
        offs = VGroup()
        for r in range(rows):
            o = kit.label(f"byte {r * cols * 2}", size=th.SIZE_LABEL, color=row_cols[r])
            o.next_to(flat.cell(0, r * cols), DOWN, buff=0.15).align_to(flat.cell(0, r * cols), LEFT)
            offs.add(o)
        flat_t = kit.label("in the file", size=th.SIZE_LABEL, color=th.INK_3)
        flat_t.next_to(flat, UP, buff=0.15).align_to(flat, LEFT)
        strip = VGroup(flat_t, flat, offs)
        lead = kit.prose(r"Bytes $=$ elements in the shape $\times$ bytes per element. Nothing else is stored.",
                         size=th.SIZE_SMALL, width=None)
        f = kit.formula(r"\text{offset}[i, j] = (i \times 5120 + j) \times \text{bytes per element}",
                        size=th.SIZE_MATH_SMALL)
        f_t = kit.prose(r"for shape $(248320, 5120)$: row-major, also called C order", size=th.SIZE_SMALL,
                        width=None, color=th.INK_3)
        col = VGroup(lead, m, strip, VGroup(f, f_t).arrange(DOWN, buff=0.2)).arrange(DOWN, buff=0.45)
        self.place(col)
        with self.narrate(
            "Three ideas carry this chapter. "
            "First, a tensor is a contiguous run of bytes in a file. "
            "Its data type fixes how many bytes each element takes. "
            "Multiply the shape out, times the element size, and you have its length. "
            "Nothing else is stored. "
            "Second, the order is row-major: the last index moves fastest. "
            "Row 0's values go first, then row 1's, then row 2's. "
            "In a tensor 5120 wide, element i, j sits at i times 5120, plus j, times the element size. "
            "This is C order, the only layout safetensors stores."
        ) as t:
            self.play(FadeIn(lead), run_time=0.7)
            t.wait_until("Its data type")
            self.play(FadeIn(m), run_time=0.7)
            t.wait_until("Second, the order")
            self.play(FadeIn(flat_t), run_time=0.5)
            for r in range(rows):
                if r == 1:
                    t.wait_until("Row 0's values")
                self.play(mat.fill(mat.row(r), row_cols[r]), run_time=0.3)
                targets = VGroup(*[flat.cell(0, r * cols + c) for c in range(cols)])
                self.play(ReplacementTransform(mat.row(r).copy(), targets), FadeIn(offs[r]), run_time=0.9)
            t.wait_until("In a tensor 5120")
            self.play(FadeIn(f, shift=UP * 0.1), run_time=0.8)
            t.wait_until("This is C order")
            self.play(FadeIn(f_t), run_time=0.6)

    def memory_mapping(self):
        self.slide("memory-mapping", "Memory mapping reads only the pages you touch",
                   book_section="s-before-you-start")
        n = 20
        cell = (0.5, 0.55)
        disk = kit.grid(1, n, cell=cell)
        cache = kit.grid(1, n, cell=cell)
        addr = kit.grid(1, n, cell=cell)
        for g in (cache, addr):
            for sq in g.squares():
                sq.set_fill(th.CANVAS, opacity=1).set_stroke(th.RULE_STRONG, opacity=0.6)
        names = ["File on disk: 3 GB", "Page cache: the kernel's copy in RAM", "Your mapped addresses"]
        rows = VGroup()
        for g, name in zip((disk, cache, addr), names):
            lab = kit.label(name, size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
            rows.add(VGroup(lab, g).arrange(DOWN, buff=0.15, aligned_edge=LEFT))
        rows.arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        note = kit.prose(r"Touch 100 bytes of a 3 GB file: you read 100 bytes, not 3 GB.",
                         size=th.SIZE_SMALL, width=None)
        col = VGroup(rows, note).arrange(DOWN, buff=0.55)
        self.place(col)
        page_lab = kit.label("one 4 KB page per touch", size=th.SIZE_LABEL, color=self.color)
        page_lab.next_to(rows[1][0], RIGHT, buff=0.4)
        nothing = kit.label("reserved by mmap, read on touch", size=th.SIZE_LABEL, color=th.INK_3)
        nothing.next_to(rows[2][0], RIGHT, buff=0.4)

        def touch(c, first):
            p = pointer(self.color).next_to(addr.cell(0, c), DOWN, buff=0.08).rotate(PI)
            self.play(FadeIn(p, shift=UP * 0.1), addr.fill(addr.cell(0, c), self.color, tint=0.15), run_time=0.5)
            self.play(disk.fill(disk.cell(0, c), self.color, tint=0.8), run_time=0.4)
            src = disk.cell(0, c).copy()
            self.play(src.animate.move_to(cache.cell(0, c)), run_time=0.6)
            self.remove(src)
            self.play(cache.fill(cache.cell(0, c), self.color, tint=0.6),
                      addr.fill(addr.cell(0, c), self.color, tint=0.6), run_time=0.4)
            if first:
                self.play(FadeIn(page_lab), run_time=0.4)
            return p

        with self.narrate(
            "Third, memory mapping. "
            "The mmap system call asks the kernel to reserve a range of addresses backed by a file, not by RAM. "
            "At that moment, it reads nothing. "
            "When your program touches an address in that range, the kernel fetches the 4 KB page that contains it. "
            "That page comes from disk, or from the page cache, the kernel's in-RAM copy of recently read files. "
            "Touch a hundred bytes of a 3 GB file, and you've read a hundred bytes, not 3 GB."
        ) as t:
            self.play(FadeIn(rows[0]), run_time=0.6)
            t.wait_until("The mmap system call")
            self.play(FadeIn(rows[2]), run_time=0.7)
            t.wait_until("At that moment")
            self.play(FadeIn(rows[1]), FadeIn(nothing), run_time=0.6)
            t.wait_until("When your program")
            touch(6, True)
            t.wait_until("That page comes")
            touch(14, False)
            t.wait_until("Touch a hundred")
            self.play(FadeIn(note), run_time=0.7)

    # The safetensors format --------------------------------------------------------

    def file_strip(self, len_w=1.2, head_w=3.0, data_w=7.4, h=0.7, cells=True):
        """The file layout: 8 length bytes, the header, then the data buffer."""
        if cells:
            prefix = VGroup(*[block(len_w / 8, h, LEN_C, tint=0.35) for _ in range(8)]).arrange(RIGHT, buff=0)
        else:
            prefix = block(len_w, h, LEN_C, "8")
        head = block(head_w, h, HEAD_C, "JSON header")
        data = block(data_w, h, DATA_C, None, tint=0.22)
        row = VGroup(prefix, head, data).arrange(RIGHT, buff=0)
        return row, prefix, head, data

    def three_parts(self):
        self.slide("three-parts", "A safetensors file has three parts", book_section="s-the-safetensors-format")
        row, prefix, head, data = self.file_strip()
        widths = [1.4, 2.2, 0.9, 2.9]
        x = data.get_left()[0]
        cuts = VGroup()
        for w in widths[:-1]:
            x += w
            cuts.add(Line([x, data.get_bottom()[1], 0], [x, data.get_top()[1], 0], color=DATA_C,
                          stroke_width=th.STROKE_BOX))
        b1 = brace_label(prefix, r"8 bytes: $N$, unsigned 64-bit, least significant byte first", direction=UP,
                         color=LEN_C)
        b1[1].align_to(prefix, LEFT)
        b2 = brace_label(head, r"$N$ bytes of UTF-8 JSON", color=HEAD_C)
        b3 = brace_label(data, "raw tensor bytes, back to back", color=DATA_C)
        fig = VGroup(row, cuts, b1, b2, b3)
        notes = kit.bullets(
            [
                r"No separators between tensors.",
                r"No padding.",
                r"No metadata of any kind in the data.",
            ],
            size=th.SIZE_SMALL, width=8.0,
        )
        col = VGroup(fig, notes).arrange(DOWN, buff=0.8)
        self.place(col)
        with self.narrate(
            "Now the format itself. "
            "A safetensors file has three parts, in this order. "
            "The first 8 bytes hold one number, N: the length of the header. "
            "It's stored as an unsigned 64-bit integer, least significant byte first. "
            "The next N bytes are the header, written as UTF-8 JSON. "
            "Everything after that is a flat buffer of tensor data. "
            "There are no separators, no padding between tensors, and no metadata of any kind."
        ) as t:
            t.wait_until("The first 8 bytes")
            self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.1) for c in prefix], lag_ratio=0.15),
                      run_time=1.2)
            t.wait_until("It's stored as")
            self.play(FadeIn(b1), run_time=0.6)
            t.wait_until("The next N bytes")
            self.play(GrowFromEdge(head, LEFT), run_time=0.8)
            self.play(FadeIn(b2), run_time=0.5)
            t.wait_until("Everything after that")
            self.play(GrowFromEdge(data, LEFT), run_time=1.0)
            self.play(LaggedStart(*[Create(c) for c in cuts], lag_ratio=0.3), FadeIn(b3), run_time=0.9)
            t.wait_until("There are no separators")
            self.play(LaggedStart(*kit.build(notes), lag_ratio=0.4), run_time=1.5)

    def header_entry(self):
        self.slide("header-entry", "The header is a table of contents in JSON",
                   book_section="s-the-safetensors-format")
        panel_ = kit.code(HEADER_JSON, "json", caption="header")
        notes = VGroup(
            kit.prose(r"\texttt{dtype}: the number format, 2 bytes per element", size=th.SIZE_SMALL, width=None),
            kit.prose(r"\texttt{shape}: the dimensions, in row-major order", size=th.SIZE_SMALL, width=None),
            kit.prose(r"\texttt{data\_offsets}: half-open $[\text{begin}, \text{end})$, counted from the "
                      r"\textbf{end of the header}", size=th.SIZE_SMALL, width=None),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        col = VGroup(panel_, notes).arrange(DOWN, buff=0.45)
        self.place(col)
        with self.narrate(
            "The header maps each tensor name to three fields. "
            "Here's the entry for layer 3's key projection. "
            "Its data type is bf16, which takes 2 bytes per element. "
            "Its shape is 1024 by 5120. "
            "Its data offsets give a byte range that includes begin, and stops just before end. "
            "That range is measured from the end of the header, not from the start of the file. "
            "One key is reserved for metadata: a flat map of strings, with no tensor data."
        ) as t:
            self.play(FadeIn(panel_.frame), *panel_.reveal(1, 2, 6, 8), run_time=0.8)
            t.wait_until("Its data type")
            self.play(*panel_.reveal(3), panel_.highlight(3), FadeIn(notes[0]), run_time=0.7)
            t.wait_until("Its shape is")
            self.play(*panel_.reveal(4), panel_.highlight(4), FadeIn(notes[1]), run_time=0.7)
            t.wait_until("Its data offsets")
            self.play(*panel_.reveal(5), panel_.highlight(5), FadeIn(notes[2]), run_time=0.7)
            t.wait_until("One key is reserved")
            self.play(*panel_.reveal(7), run_time=0.6)
            self.play(panel_.highlight(7, color=th.tint(th.INK_4, 0.2)), run_time=0.5)

    def absolute_offset(self):
        self.slide("absolute-offset", "A tensor's first byte is at 8 + N + begin",
                   book_section="s-the-safetensors-format")
        row, prefix, head, data = self.file_strip(len_w=1.2, head_w=3.0, data_w=7.6, cells=False)
        start_in, width = 2.6, 2.4
        ten = block(width, data.height, TENSOR_C, "k_proj", font=th.MONO, tint=0.4, size=th.SIZE_LABEL)
        ten.move_to(data.get_left() + RIGHT * (start_in + width / 2))
        beg = Line(data.get_left(), data.get_left() + RIGHT * start_in)
        b1 = brace_label(prefix, "8", direction=UP, color=LEN_C, size=th.SIZE_SMALL)
        b2 = brace_label(head, "$N$", direction=UP, color=HEAD_C, size=th.SIZE_SMALL)
        b3 = brace_label(beg, r"$\text{begin}$", direction=UP, color=DATA_C, size=th.SIZE_SMALL)
        fig = VGroup(row, ten, b1, b2, b3)
        f = kit.formula(r"\text{absolute offset} =", r"8", r"+", r"N", r"+", r"\text{begin}")
        f[1].set_color(LEN_C)
        f[3].set_color(HEAD_C)
        f[5].set_color(DATA_C)
        ident = kit.formula(r"\text{end} - \text{begin} = \text{elements} \times \text{bytes per element}",
                            size=th.SIZE_MATH_SMALL)
        ex = kit.formula(r"1024 \times 5120 \times 2 = 10{,}485{,}760", size=th.SIZE_MATH_SMALL, color=th.INK_2)
        col = VGroup(fig, f, ident, ex).arrange(DOWN, buff=0.5)
        self.place(col)
        box = SurroundingRectangle(ident, color=self.color, buff=0.14, stroke_width=th.STROKE_AXIS)
        ptr = pointer(th.INK).next_to(row, DOWN, buff=0.1).set_x(row.get_left()[0])
        with self.narrate(
            "So where does a tensor's first byte live? "
            "Skip the 8-byte length prefix. "
            "Skip the N-byte header. "
            "Then count forward by begin. "
            "Because nothing else is stored, the range's length has to match the shape and the type exactly. "
            "The key projection has 1024 times 5120 elements, at 2 bytes each. "
            "That's 10,485,760 bytes, exactly end minus begin. "
            "Checking this identity is how you catch a truncated or corrupt file."
        ) as t:
            self.play(FadeIn(row), FadeIn(ptr), run_time=0.8)
            t.wait_until("Skip the 8-byte")
            self.play(ptr.animate.set_x(prefix.get_right()[0]), FadeIn(b1), FadeIn(f[:2]), run_time=0.8)
            t.wait_until("Skip the N-byte")
            self.play(ptr.animate.set_x(head.get_right()[0]), FadeIn(b2), FadeIn(f[2:4]), run_time=0.9)
            t.wait_until("Then count forward")
            self.play(ptr.animate.set_x(ten.get_left()[0]), FadeIn(b3), FadeIn(f[4:]), run_time=0.9)
            self.play(FadeIn(ten), run_time=0.5)
            t.wait_until("Because nothing else")
            self.play(FadeIn(ident, shift=UP * 0.1), run_time=0.8)
            t.wait_until("The key projection has")
            self.play(FadeIn(ex, shift=UP * 0.1), run_time=0.8)
            t.wait_until("Checking this identity")
            self.play(Create(box), run_time=0.7)

    # Why the format looks like this ----------------------------------------------------

    def why_bare(self):
        self.slide("why-bare", "Four properties of a bare format", book_section="s-why-the-format-looks-like-this")
        w = 6.0
        cards = VGroup(
            kit.callout("key", r"Read 8 bytes, then $N$ more, and parse the JSON. No weight data touched.",
                        title="1  Listing is one small read", width=w, size=th.SIZE_SMALL),
            kit.callout("key", r"Every tensor is stored contiguous. The header's shape is the shape in memory.",
                        title="2  No strides", width=w, size=th.SIZE_SMALL),
            kit.callout("key", r"The tensor's storage points into the mapping. Pages arrive when you touch them.",
                        title="3  A map, not a copy", width=w, size=th.SIZE_SMALL),
            kit.callout("key", r"JSON with a fixed schema, then numbers. \texttt{torch.save} uses pickle, "
                               r"which runs code.", title="4  Nothing executes", width=w, size=th.SIZE_SMALL),
        )
        top = VGroup(cards[0], cards[1]).arrange(RIGHT, buff=0.35, aligned_edge=UP)
        bot = VGroup(cards[2], cards[3]).arrange(RIGHT, buff=0.35, aligned_edge=UP)
        for a, b in ((cards[0], cards[1]), (cards[2], cards[3])):
            hgt = max(a.panel.height, b.panel.height)
            for c in (a, b):
                c.panel.stretch_to_fit_height(hgt, about_edge=UP)
                c[1].stretch_to_fit_height(hgt - 0.2).align_to(c.panel, UP).shift(DOWN * 0.1)
        grid_ = VGroup(top, bot).arrange(DOWN, buff=0.3)
        dtypes = ["F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8", "BOOL"]
        chips = VGroup(*[kit.mono(d, size=th.SIZE_SMALL, color=th.INK_2) for d in dtypes]).arrange(RIGHT, buff=0.38)
        dt_t = kit.label("Dtypes:", size=th.SIZE_SMALL, color=th.INK_3)
        dt_row = VGroup(dt_t, chips).arrange(RIGHT, buff=0.3)
        col = VGroup(grid_, dt_row).arrange(DOWN, buff=0.4)
        col.scale(0.97)
        self.place(col)
        bf = kit.mark(chips[3], buff=0.04)
        with self.narrate(
            "Each property of this bare layout is something your engine relies on. "
            "First, listing tensors is one small read. "
            "Read 8 bytes, read N more, and parse the JSON, without touching weight data. "
            "Second, there are no strides. "
            "Every tensor is stored contiguous, so the shape in the header is the shape in memory. "
            "Third, reading a tensor maps the file, rather than copying it. "
            "Fourth, nothing executes. "
            "PyTorch's own save format uses pickle, so reading it can run a stranger's code. "
            "This model's weights are all bf16."
        ) as t:
            for card, cue in zip(cards, ["First, listing", "Second, there", "Third, reading", "Fourth, nothing"]):
                t.wait_until(cue)
                self.play(FadeIn(card, shift=UP * 0.1), run_time=0.7)
            t.wait_until("This model's weights")
            self.play(FadeIn(dt_row), run_time=0.6)
            self.play(FadeIn(bf), run_time=0.6)

    # Why memory mapping matters here -----------------------------------------------------

    def layer_by_layer(self):
        self.slide("layer-by-layer", "Peak host memory is one layer, not 54 GB",
                   book_section="s-why-memory-mapping-matters-here")
        pw, ph = 3.4, 2.8
        disk = panel("Mapped file", pw, ph, color=DATA_C, sub="54 GB on disk")
        host = panel("Host RAM", pw, ph, color=th.INFO, sub="the CPU's memory")
        gpu = panel("GPU", pw, ph, color=self.color, sub="80 GB")
        row = VGroup(disk, host, gpu).arrange(RIGHT, buff=1.3)
        bars = kit.bar_compare(
            [("Read it all first", 54, th.INFO, "54 GB"), ("Layer by layer", 0.8, self.color, "0.8 GB")],
            width=6.0, bar_height=0.45, gap=0.28,
        )
        col = VGroup(row, bars).arrange(DOWN, buff=0.6)
        self.place(col)
        a1 = kit.arrow(disk.rect, host.rect, text="page in")
        a2 = kit.arrow(host.rect, gpu.rect, text="PCIe")
        inner_h = ph - 1.1
        base_y = disk.rect.get_bottom()[1] + 0.2
        # The naive load: the whole checkpoint lands in host RAM.
        full = Rectangle(width=pw - 0.6, height=inner_h, fill_color=th.tint(th.INFO, 0.4), fill_opacity=1,
                         stroke_width=0)
        full.move_to([host.rect.get_x(), base_y + inner_h / 2, 0])
        full_t = kit.label("all 54 GB", size=th.SIZE_SMALL, color=th.INK).move_to(full)
        # The GPU fills one layer slab at a time.
        k = 8
        slab_h = inner_h / k
        slabs = VGroup(*[Rectangle(width=pw - 0.6, height=slab_h - 0.04, fill_color=th.tint(self.color, 0.5),
                                   fill_opacity=1, stroke_width=0) for _ in range(k)])
        for i, s in enumerate(slabs):
            s.move_to([gpu.rect.get_x(), base_y + slab_h * (i + 0.5), 0])
        with self.narrate(
            "How do you get 54 GB onto the GPU without first needing 54 GB of CPU RAM? "
            "The naive way reads the whole checkpoint into host memory, then copies it across. "
            "That needs 54 GB of RAM you might not have. "
            "With a mapped file, you go layer by layer instead. "
            "Each layer pages in, crosses PCIe to the GPU, and the kernel reclaims its pages behind you. "
            "Peak host memory is now one layer, about 0.8 GB."
        ) as t:
            self.play(FadeIn(disk), FadeIn(host), FadeIn(gpu), run_time=0.8)
            self.play(GrowArrow(a1.line), FadeIn(a1.label), GrowArrow(a2.line), FadeIn(a2.label), run_time=0.7)
            t.wait_until("The naive way")
            self.play(GrowFromEdge(full, DOWN), run_time=1.6)
            self.play(FadeIn(full_t), run_time=0.4)
            t.wait_until("That needs 54 GB")
            self.play(FadeIn(bars.labels[0]), GrowFromEdge(bars.bars[0], LEFT), FadeIn(bars.values[0]),
                      run_time=0.9)
            t.wait_until("With a mapped file")
            self.play(FadeOut(full), FadeOut(full_t), run_time=0.5)
            for i in range(k):
                chip = slabs[i].copy().move_to([disk.rect.get_x(), base_y + inner_h / 2, 0])
                rt = 0.5 if i < 3 else 0.3
                self.play(FadeIn(chip), run_time=rt * 0.5)
                self.play(chip.animate.move_to([host.rect.get_x(), base_y + inner_h / 2, 0]), run_time=rt)
                self.play(ReplacementTransform(chip, slabs[i]), run_time=rt)
            t.wait_until("Peak host memory")
            self.play(FadeIn(bars.labels[1]), GrowFromEdge(bars.bars[1], LEFT), FadeIn(bars.values[1]),
                      run_time=0.8)

    def load_loop(self):
        self.slide("load-loop", "Pages from an earlier run come back for free",
                   book_section="s-why-memory-mapping-matters-here")
        code_ = kit.code(LOAD_LOOP, "python")
        notes = kit.bullets(
            [
                r"The second load is much faster than the first: the pages are still in the page cache.",
                r"Mapped tensors are read-only in practice: pages are shared.",
                r"A mapped tensor reads from disk on first touch. An ``instant'' load timed the mapping.",
            ],
            size=th.SIZE_SMALL, width=11.0,
        )
        col = VGroup(code_, notes).arrange(DOWN, buff=0.6)
        self.place(col)
        with self.narrate(
            "In code, the loop runs over the layers, then over each layer's tensors. "
            "Each fetch pages the tensor in, then copies it to the GPU over PCIe. "
            "Pages still in the page cache from a previous run come back for free. "
            "Two gotchas. "
            "Mapped pages are shared, so writing to one either faults or dirties the page cache. "
            "And a mapped tensor still reads from disk on first touch. "
            "If a load looks instant, you timed the mapping, not the read."
        ) as t:
            self.play(FadeIn(code_.frame), *code_.reveal(1, 2), run_time=0.8)
            t.wait_until("Each fetch pages")
            self.play(*code_.reveal(3, 4), code_.highlight(4), run_time=0.8)
            t.wait_until("Pages still in")
            self.play(FadeIn(notes[0], shift=RIGHT * 0.1), run_time=0.7)
            t.wait_until("Mapped pages are shared")
            self.play(FadeIn(notes[1], shift=RIGHT * 0.1), run_time=0.7)
            t.wait_until("And a mapped tensor")
            self.play(FadeIn(notes[2], shift=RIGHT * 0.1), run_time=0.7)

    # Shards and the index ----------------------------------------------------------

    def shards_index(self):
        self.slide("shards-index", "The index maps every tensor name to its shard",
                   book_section="s-shards-and-the-index")
        name = kit.mono("model.language_model.layers.0.mlp.gate_proj.weight", size=th.SIZE_SMALL,
                        color=th.INK)
        name_box = SurroundingRectangle(name, color=th.RULE_STRONG, buff=0.14, stroke_width=th.STROKE_BOX,
                                        fill_color=th.RAISE, fill_opacity=1)
        name_g = VGroup(name_box, name)
        idx = kit.box("model.safetensors.index.json", sublabel="weight_map: tensor name to shard file",
                      font=th.MONO, color=self.color, size=th.SIZE_SMALL)
        shards = kit.grid(3, 6, cell=(0.78, 0.5), gap=0.07,
                          values=[[f"{r * 6 + c + 1:02d}" for c in range(6)] for r in range(3)])
        sh_t = kit.label("18 shards", size=th.SIZE_LABEL, color=th.INK_3)
        sh = VGroup(sh_t, shards).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        flow = VGroup(idx, sh).arrange(RIGHT, buff=1.4)
        tab = kit.table(
            [
                [r"Index \texttt{total\_size}: tensor bytes, no headers", "53.97 GB"],
                [r"Chapter 2: the language model alone", "53.79 GB"],
            ],
            size=th.SIZE_SMALL, align=["l", "r"], row_gap=0.25,
        )
        col = VGroup(name_g, flow, tab).arrange(DOWN, buff=0.65)
        self.place(col)
        a1 = kit.arrow([idx.get_x(), name_g.get_bottom()[1], 0], idx)
        a2 = kit.arrow(idx, shards)
        with self.narrate(
            "With 18 files, how do you know which one to open? "
            "A single 54 GB file is awkward to download and impossible to resume, so the model is split. "
            "The index file maps each tensor name to the shard that holds it. "
            "Look up the name, open that shard, and read the byte range. "
            "The index also records a total size of about 53.97 GB, counting tensor bytes but not headers. "
            "Chapter 2 counts 53.79 GB for the language model alone. "
            "The checkpoint also carries tensors outside the text path."
        ) as t:
            self.play(FadeIn(sh), run_time=0.7)
            t.wait_until("The index file maps")
            self.play(FadeIn(name_g), run_time=0.5)
            self.play(GrowArrow(a1.line), FadeIn(idx), run_time=0.8)
            t.wait_until("Look up the name")
            self.play(GrowArrow(a2.line), run_time=0.7)
            self.play(shards.fill(shards.cell(0, 0), self.color, tint=0.6), run_time=0.5)
            self.play(Indicate(shards.cell(0, 0), color=self.color, scale_factor=1.15), run_time=0.8)
            t.wait_until("The index also records")
            self.play(FadeIn(tab.rules), *[FadeIn(c) for c in tab.cells[0]], run_time=0.7)
            t.wait_until("Chapter 2 counts")
            self.play(*[FadeIn(c) for c in tab.cells[1]], run_time=0.7)

    def weight_index(self):
        self.slide("weight-index", "One map from name to shard, either way", book_section="s-shards-and-the-index")
        code_ = kit.code(INIT, "python", caption="engine/weights.py", size=th.SIZE_CODE - 2)
        self.place(code_)
        with self.narrate(
            "The engine's weight index class does exactly this lookup. "
            "If the index file exists, it reads the weight map from it. "
            "Small models ship as one file with no index. "
            "For those, it opens the single file, and maps every tensor name in it to that file. "
            "Either way, the rest of the class has one code path instead of two. "
            "It also caches the open file handles. "
            "That way, loading 64 layers doesn't reopen 18 files 64 times."
        ) as t:
            self.play(FadeIn(code_.frame), *code_.reveal(1, 2, 3, 4), run_time=0.8)
            t.wait_until("If the index file")
            self.play(*code_.reveal(5, 6), code_.highlight(5, 6), run_time=0.8)
            t.wait_until("Small models ship")
            self.play(*code_.reveal(7, 8), run_time=0.6)
            t.wait_until("For those, it opens")
            self.play(*code_.reveal(9, 10), code_.highlight(9, 10, color=th.tint(th.INFO, 0.14)), run_time=0.8)
            t.wait_until("Either way")
            self.play(code_.focus(6, 10), run_time=0.6)
            t.wait_until("It also caches")
            self.play(*code_.reveal(11), code_.focus(), code_.highlight(11, color=th.tint(th.GOOD, 0.16)),
                      run_time=0.8)

    def get_copies(self):
        self.slide("get-copies", "Only a dtype or device change copies", book_section="s-shards-and-the-index")
        code_ = kit.code(GET, "python", size=th.SIZE_CODE - 2)
        bw = 3.3
        mapped = kit.box("Mapped pages", sublabel="a view: no copy", color=DATA_C, width=bw)
        conv = kit.box("dtype change", sublabel="a full copy", color=th.BAD, width=bw)
        dev = kit.box("GPU", sublabel="a copy over PCIe", color=self.color, width=bw)
        flow = VGroup(mapped, conv, dev).arrange(RIGHT, buff=0.9)
        sl = kit.prose(r"\texttt{get\_slice} reads only the header entry: shape and dtype, never the data.",
                       size=th.SIZE_SMALL, width=None)
        col = VGroup(code_, flow, sl).arrange(DOWN, buff=1.0)
        flow.shift(DOWN * 0.15)
        self.place(col)
        a1 = kit.arrow(mapped, conv)
        a2 = kit.arrow(conv, dev)
        skip = CurvedArrow(mapped.rect.get_top() + RIGHT * 0.6, dev.rect.get_top() + LEFT * 0.6, angle=-PI / 4,
                           color=self.color, stroke_width=th.STROKE_BOX + 1, tip_length=0.2)
        skip_t = kit.label("cheap path: one copy", size=th.SIZE_LABEL, color=self.color)
        skip_t.next_to(skip, UP, buff=0.08)
        conv_dim = VGroup(conv, a1, a2)
        with self.narrate(
            "To fetch one tensor, look up its shard, then ask that shard's mapping for the tensor. "
            "That returns a view into the mapping, so it copies nothing. "
            "The two conversion calls are where real work happens. "
            "A data type change copies the tensor, so keep the engine in bf16 end to end. "
            "A device change copies it across PCIe. "
            "Ask for the GPU with no type change, and you get the cheap path: one copy, from the page cache. "
            "To inspect a tensor without loading it, the slice handle reads only its header entry."
        ) as t:
            self.play(FadeIn(code_.frame), *code_.reveal(1), run_time=0.6)
            self.play(*code_.reveal(2, 3), run_time=0.7)
            t.wait_until("That returns a view")
            self.play(code_.highlight(3), FadeIn(mapped), run_time=0.7)
            t.wait_until("The two conversion")
            self.play(*code_.reveal(4, 5, 6), run_time=0.7)
            t.wait_until("A data type change")
            self.play(code_.highlight(5, color=th.tint(th.BAD, 0.14)), GrowArrow(a1.line), FadeIn(conv), run_time=0.8)
            t.wait_until("A device change")
            self.play(code_.highlight(6), GrowArrow(a2.line), FadeIn(dev), run_time=0.8)
            t.wait_until("Ask for the GPU")
            self.play(conv_dim.animate.set_opacity(0.3), run_time=0.5)
            self.play(Create(skip), FadeIn(skip_t), run_time=1.0)
            t.wait_until("To inspect a tensor")
            self.play(FadeIn(sl, shift=UP * 0.1), run_time=0.7)

    # Reading the architecture from the names --------------------------------------------

    def two_layers(self):
        self.slide("two-layers", "Layer 0 and layer 3 hold different tensors",
                   book_section="s-reading-the-architecture-from-the-names")
        code_ = kit.code(NAMES, "python", line_numbers=False, size=th.SIZE_CODE - 2,
                         caption="after model.language_model.layers.i")

        def layer(mixer, color):
            m = kit.box(mixer, color=color, width=3.6, size=th.SIZE_SMALL)
            mlp = kit.box("MLP", width=3.6, size=th.SIZE_SMALL)
            return VGroup(m, mlp).arrange(DOWN, buff=0.12)

        lin = layer("linear attention", th.INFO)
        full = layer("full attention", self.color)
        lin_t = kit.label("layer 0", size=th.SIZE_LABEL, color=th.INK_3)
        full_t = kit.label("layer 3", size=th.SIZE_LABEL, color=th.INK_3)
        diag = VGroup(VGroup(lin_t, lin).arrange(DOWN, buff=0.12, aligned_edge=LEFT),
                      VGroup(full_t, full).arrange(DOWN, buff=0.12, aligned_edge=LEFT)).arrange(DOWN, buff=0.5)
        mixer_t = kit.label("token mixer, then MLP", size=th.SIZE_SMALL, color=th.INK_2, weight=MEDIUM)
        right = VGroup(mixer_t, diag).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        layout = VGroup(code_, right).arrange(RIGHT, buff=0.7)
        self.place(layout)
        with self.narrate(
            "Now, what does the model look like inside? "
            "Group the tensor names by layer, and the model describes itself. "
            "Every layer has a token mixer, which lets each position look at earlier ones, then an MLP. "
            "Layer 0's mixer is linear attention. "
            "It replaces comparing queries with keys by a fixed-size running summary. "
            "Layer 3's mixer is full attention, the standard kind, split across several heads. "
            "The two layers hold different tensors."
        ) as t:
            self.play(FadeIn(code_.frame), run_time=0.5)
            t.wait_until("Every layer has")
            self.play(FadeIn(mixer_t), run_time=0.5)
            t.wait_until("Layer 0's mixer")
            self.play(FadeIn(lin_t), FadeIn(lin), *code_.reveal(*range(1, 9)), run_time=1.2)
            t.wait_until("Layer 3's mixer")
            self.play(FadeIn(full_t), FadeIn(full), *code_.reveal(*range(9, 14)), run_time=1.2)
            t.wait_until("The two layers hold")
            self.play(Indicate(lin[0].rect, color=th.INFO, scale_factor=1.05),
                      Indicate(full[0].rect, color=self.color, scale_factor=1.05), run_time=1.0)

    def head_counts(self):
        self.slide("head-counts", "Six query heads share each KV head",
                   book_section="s-reading-the-architecture-from-the-names")
        lead = kit.prose(r"Stored as (out, in): the input width is 5120.", size=th.SIZE_SMALL, width=None,
                         color=th.INK_3)
        d = kit.Derivation(
            [
                r"\texttt{k\_proj}: \quad \tfrac{1024}{256} &= 4 \ \text{KV heads}",
                r"\texttt{o\_proj}: \quad \tfrac{6144}{256} &= 24 \ \text{query heads}",
                r"\tfrac{24}{4} &= 6 \ \text{query heads per KV head}",
            ],
            size=th.SIZE_MATH_SMALL, dim=False,
        )
        qn = 24
        q = kit.grid(1, qn, cell=(0.42, 0.5), gap=0.05)
        kv = VGroup(*[block(6 * 0.42 + 5 * 0.05, 0.5, self.colors[i % 4], f"KV {i}", size=th.SIZE_LABEL)
                      for i in range(4)]).arrange(RIGHT, buff=0.05)
        q_t = kit.label("24 query heads", size=th.SIZE_LABEL, color=th.INK_3)
        kv_t = kit.label("4 KV heads", size=th.SIZE_LABEL, color=th.INK_3)
        kv.next_to(q, DOWN, buff=0.75).align_to(q, LEFT)
        q_t.next_to(q, LEFT, buff=0.3)
        kv_t.next_to(kv, LEFT, buff=0.3)
        links = VGroup()
        fig = VGroup(q, kv, q_t, kv_t)
        col = VGroup(lead, d, fig).arrange(DOWN, buff=0.55)
        self.place(col)
        for i in range(qn):
            g = i // 6
            links.add(Line(q.cell(0, i).get_bottom(), kv[g].get_top(), color=self.colors[g],
                           stroke_width=th.STROKE_HAIRLINE + 0.5))
        with self.narrate(
            "Start with the full-attention layer. "
            "PyTorch stores a linear weight output first, so the second dimension is the input width. "
            "Here it's 5120, the hidden size. "
            "The key projection has 1024 output channels. "
            "One head is 256 wide, so that's 4 key heads, and the values match. "
            "The output projection maps 6144 channels back to the hidden size. "
            "Divide by 256, and that's 24 query heads. "
            "So six query heads share each key head. "
            "That's grouped-query attention, GQA, the subject of chapter 7."
        ) as t:
            t.wait_until("PyTorch stores")
            self.play(FadeIn(lead), run_time=0.7)
            t.wait_until("One head is 256")
            self.play(d.show(0), FadeIn(kv), FadeIn(kv_t), run_time=0.9)
            t.wait_until("Divide by 256")
            self.play(d.show(1), FadeIn(q), FadeIn(q_t), run_time=0.9)
            t.wait_until("So six query heads")
            self.play(d.show(2), run_time=0.7)
            self.play(LaggedStart(*[q.fill(q.cell(0, i), self.colors[i // 6], tint=0.45) for i in range(qn)],
                                  lag_ratio=0.05), LaggedStart(*[Create(l) for l in links], lag_ratio=0.05),
                      run_time=1.8)

    def output_gate(self):
        self.slide("output-gate", "The query projection is twice the width you'd expect",
                   book_section="s-reading-the-architecture-from-the-names")
        u = 9.0 / 12288
        exp = block(6144 * u, 0.6, th.INK_4, "6144", tint=0.25)
        exp_t = kit.prose(r"Expected: $24 \times 256$", size=th.SIZE_SMALL, width=None, color=th.INK_2)
        qs = block(6144 * u, 0.6, self.color, "queries: 6144", tint=0.35)
        gate = block(6144 * u, 0.6, th.VIOLET, "gate: 6144", tint=0.35)
        fil = VGroup(qs, gate).arrange(RIGHT, buff=0)
        whole = block(12288 * u, 0.6, self.color, "12,288", tint=0.35)
        fil_t = kit.prose(r"In the file: \texttt{q\_proj} $[12288, 5120]$", size=th.SIZE_SMALL, width=None,
                          color=th.INK_2)
        names = VGroup(exp_t, fil_t)
        bars = VGroup(exp, fil)
        bars.arrange(DOWN, buff=0.8, aligned_edge=LEFT)
        for nm, b in zip(names, bars):
            nm.next_to(b, UP, buff=0.15).align_to(b, LEFT)
        whole.move_to(fil)
        cfg = kit.mono('"attn_output_gate": true', size=th.SIZE_SMALL, color=th.INK)
        cfg_t = kit.label("config.json", size=th.SIZE_LABEL, color=th.INK_3)
        cfg_g = VGroup(cfg_t, cfg).arrange(RIGHT, buff=0.4)
        warn = kit.callout("warning", r"Size \texttt{q\_proj} from the head counts, and the layer loads without "
                                      r"error and produces nonsense.", title="Trust the file, not the arithmetic",
                           width=10.0, size=th.SIZE_SMALL)
        col = VGroup(VGroup(names, bars), cfg_g, warn).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "The query projection is a surprise. "
            "With 24 heads of width 256, you'd expect 6144 output channels. "
            "The file says 12,288, exactly double. "
            "That's the output gate. "
            "The projection emits the queries and a gate side by side, and the layer splits them apart. "
            "The config confirms it with its attention output gate flag. "
            "Trust the file, not the shape arithmetic. "
            "Size it from the head counts, and your layer loads fine, then produces nonsense."
        ) as t:
            t.wait_until("With 24 heads")
            self.play(FadeIn(exp_t), GrowFromEdge(exp, LEFT), run_time=0.9)
            t.wait_until("The file says")
            self.play(FadeIn(fil_t), GrowFromEdge(whole, LEFT), run_time=1.1)
            t.wait_until("The projection emits")
            self.play(FadeOut(whole), FadeIn(qs), FadeIn(gate), run_time=0.8)
            self.play(gate.animate.shift(RIGHT * 0.25), run_time=0.5)
            self.play(gate.animate.shift(LEFT * 0.25), run_time=0.4)
            t.wait_until("The config confirms")
            self.play(FadeIn(cfg_g), run_time=0.6)
            t.wait_until("Trust the file")
            self.play(FadeIn(warn, shift=UP * 0.1), run_time=0.8)

    # The linear-attention layer --------------------------------------------------------

    def qkvz(self):
        self.slide("qkvz", "One fused matrix holds q, k, v, and z", book_section="s-the-linear-attention-layer")
        u = 11.5 / 16384
        parts = [("q", 2048, KEY_C), ("k", 2048, KEY_C), ("v", 6144, VAL_C), ("z", 6144, VAL_C)]
        segs = VGroup(*[block(n * u, 0.7, c, s, tint=0.35, font=th.MONO) for s, n, c in parts]).arrange(RIGHT,
                                                                                                      buff=0)
        labs = VGroup(
            brace_label(VGroup(segs[0], segs[1]), r"$16 \times 128 = 2048$ each", color=KEY_C),
            brace_label(VGroup(segs[2], segs[3]), r"$48 \times 128 = 6144$ each", color=VAL_C),
        )
        total = brace_label(segs, r"\texttt{in\_proj\_qkvz}: 16,384 rows", direction=UP, color=th.INK_2,
                            size=th.SIZE_SMALL)
        alt = VGroup(*[block(4096 * u, 0.5, th.INK_4, "4096", tint=0.15, size=th.SIZE_LABEL) for _ in range(4)])
        alt.arrange(RIGHT, buff=0)
        alt_t = kit.label("also sums to 16,384: the config's head counts pick the split", size=th.SIZE_LABEL,
                          color=th.INK_3)
        alt_g = VGroup(alt, alt_t).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        f = kit.formula(r"2048 + 2048 + 6144 + 6144 = 16{,}384", size=th.SIZE_MATH_SMALL)
        gemm = kit.pill("one GEMM, not four", self.color, size=th.SIZE_SMALL)
        frow = VGroup(f, gemm).arrange(RIGHT, buff=0.6)
        col = VGroup(VGroup(total, segs, labs), frow, alt_g).arrange(DOWN, buff=0.55)
        self.place(col)
        with self.narrate(
            "The linear-attention layer takes more decoding. "
            "The config gives 16 key heads and 48 value heads, all 128 wide. "
            "The fused input projection's name lists four blocks: queries, keys, values, and a gate, z. "
            "Queries and keys get one block per key head: 2048 rows each. "
            "Values and the gate get one per value head: 6144 rows each. "
            "Add them up, and you get the file's 16,384. "
            "Fusing them means one GEMM instead of four. "
            "The shape alone doesn't fix the split, though. "
            "Four blocks of 4096 fit too, and the config's head counts decide."
        ) as t:
            t.wait_until("The fused input")
            self.play(FadeIn(total), run_time=0.5)
            t.wait_until("Queries and keys get")
            self.play(GrowFromEdge(segs[0], LEFT), GrowFromEdge(segs[1], LEFT), FadeIn(labs[0]), run_time=0.9)
            t.wait_until("Values and the gate")
            self.play(GrowFromEdge(segs[2], LEFT), GrowFromEdge(segs[3], LEFT), FadeIn(labs[1]), run_time=0.9)
            t.wait_until("Add them up")
            self.play(FadeIn(f, shift=UP * 0.1), run_time=0.8)
            t.wait_until("Fusing them")
            self.play(FadeIn(gemm, scale=0.9), run_time=0.6)
            t.wait_until("Four blocks of 4096")
            self.play(LaggedStart(*[FadeIn(a, shift=DOWN * 0.1) for a in alt], lag_ratio=0.2), FadeIn(alt_t),
                      run_time=1.2)

    def linear_shapes(self):
        self.slide("linear-shapes", "Every other shape factors into the heads",
                   book_section="s-the-linear-attention-layer")
        tab = kit.table(
            [
                [r"\texttt{in\_proj\_ba}", r"$[96, 5120]$", r"$2 \times 48$: one $\beta$ and one $a$ per value head"],
                [r"\texttt{conv1d}", r"$[10240, 1, 4]$", r"$2048 + 2048 + 6144$ channels, kernel width 4"],
                [r"\texttt{A\_log}, \texttt{dt\_bias}", r"$[48]$", r"one scalar per value head"],
                [r"\texttt{out\_proj}", r"$[5120, 6144]$", r"$48 \times 128$ value channels back to hidden"],
            ],
            header=["Tensor", "Shape", "Decomposes into"],
            size=th.SIZE_SMALL, row_gap=0.3,
        )
        ch = 5
        conv = kit.grid(ch, 4, cell=(0.5, 0.36), gap=0.05)
        for r in range(ch):
            conv.paint(conv.row(r), self.colors[r % 4], tint=0.3)
        conv_t = kit.label("conv1d is depthwise: one 4-tap filter per channel", size=th.SIZE_SMALL,
                           color=th.INK_2)
        state = kit.label("A_log and dt_bias: state-space parameters", size=th.SIZE_SMALL, color=th.INK_2)
        notes = VGroup(conv_t, state).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        low = VGroup(conv, notes).arrange(RIGHT, buff=0.5)
        col = VGroup(tab, low).arrange(DOWN, buff=0.5)
        self.place(col)
        with self.narrate(
            "The other shapes decompose the same way. "
            "A small projection holds one write strength, beta, and one forget gate, a, per value head. "
            "The convolution covers the query, key, and value channels, with a kernel 4 wide. "
            "Its middle dimension is 1, and that's the tell: it's depthwise. "
            "Each channel has its own filter, and channels never mix. "
            "Two tensors hold one number per value head. "
            "They're state-space parameters, so these layers run a recurrence, not attention. "
            "One stores the log of a decay rate, which keeps the rate between 0 and 1. "
            "The output projection maps the value channels back to the hidden size."
        ) as t:
            self.play(FadeIn(tab.rules), *[FadeIn(c) for c in tab.cells[0]], run_time=0.6)
            t.wait_until("A small projection")
            self.play(*[FadeIn(c) for c in tab.cells[1]], run_time=0.6)
            t.wait_until("The convolution covers")
            self.play(*[FadeIn(c) for c in tab.cells[2]], run_time=0.6)
            t.wait_until("Its middle dimension")
            self.play(LaggedStart(*[FadeIn(conv.row(r), shift=RIGHT * 0.1) for r in range(ch)], lag_ratio=0.2),
                      run_time=1.0)
            t.wait_until("Each channel has")
            self.play(FadeIn(conv_t), run_time=0.6)
            t.wait_until("Two tensors hold")
            self.play(*[FadeIn(c) for c in tab.cells[3]], run_time=0.6)
            t.wait_until("They're state-space")
            self.play(FadeIn(state), run_time=0.6)
            t.wait_until("The output projection")
            self.play(*[FadeIn(c) for c in tab.cells[4]], run_time=0.6)

    # The config that matters -------------------------------------------------------------

    def layer_types(self):
        self.slide("layer-types", "Full attention ends each group of four layers",
                   book_section="s-the-config-that-matters")
        g = lift_text(kit.grid(8, 8, cell=0.55, values=[[str(r * 8 + c) for c in range(8)] for r in range(8)]))
        g_t = kit.label("layers 0 to 63", size=th.SIZE_LABEL, color=th.INK_3)
        left = VGroup(g_t, g).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        code_ = kit.code(LAYER_TYPES, "python", size=th.SIZE_CODE - 2)
        legend = VGroup(swatch("16 full attention", self.color), swatch("48 linear attention", th.INFO),
                        swatch("off by one", th.BAD)).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        right = VGroup(code_, legend).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        layout = VGroup(left, right).arrange(RIGHT, buff=0.7)
        self.place(layout)
        full = VGroup(*[g.cell(r, c) for r in range(8) for c in (3, 7)])
        lin = VGroup(*[g.cell(r, c) for r in range(8) for c in range(8) if c not in (3, 7)])
        wrong = VGroup(*[pointer(th.BAD).next_to(g.cell(0, c), UP, buff=0.08) for c in (0, 4)])
        wrong.add(kit.label("off by one", size=th.SIZE_LABEL, color=th.BAD).next_to(wrong[1], RIGHT, buff=0.15))
        with self.narrate(
            "The config gives the exact numbers. "
            "This model is multimodal, so its config nests the language model in a text section, beside a vision "
            "tower. "
            "It also builds the list of layer types when the config omits it. "
            "Note the i plus one. "
            "Full attention lands on layers 3, 7, 11, and so on: the last layer of each group of four. "
            "That gives 16 full-attention layers and 48 linear ones. "
            "Get it off by one, and every KV cache entry goes to the wrong layer, and the model still runs."
        ) as t:
            self.play(FadeIn(left), run_time=0.7)
            t.wait_until("This model is multimodal")
            self.play(FadeIn(code_.frame), *code_.reveal(1), run_time=0.7)
            t.wait_until("It also builds")
            self.play(*code_.reveal(2, 3, 4, 5, 6), run_time=0.9)
            t.wait_until("Note the i plus one")
            self.play(code_.highlight(3), run_time=0.6)
            t.wait_until("Full attention lands on layers")
            self.play(LaggedStart(*[g.fill(c, self.color, tint=0.7) for c in full], lag_ratio=0.1),
                      FadeIn(legend[0]), run_time=1.6)
            t.wait_until("That gives 16")
            self.play(g.fill(lin, th.INFO, tint=0.2), FadeIn(legend[1]), run_time=0.8)
            t.wait_until("Get it off by one")
            self.play(FadeIn(wrong, shift=DOWN * 0.1), FadeIn(legend[2]), run_time=0.8)
            self.play(Indicate(VGroup(g.col(0), g.col(4)), color=th.BAD, scale_factor=1.03), run_time=1.0)

    def config_fields(self):
        self.slide("config-fields", "Read the head width; don't derive it", book_section="s-the-config-that-matters")
        rows = [
            ("hidden_size", "5120"),
            ("num_hidden_layers", "64"),
            ("num_attention_heads", "24"),
            ("num_key_value_heads", "4"),
            ("head_dim", "256"),
            ("intermediate_size", "17408"),
            ("vocab_size", "248320"),
            ("partial_rotary_factor", "0.25"),
        ]
        tab = kit.table([[kit.mono(f, size=th.SIZE_SMALL), v] for f, v in rows], header=["Field", "Value"],
                        size=th.SIZE_SMALL, align=["l", "r"], row_gap=0.2, highlight_rows=[4])
        right = VGroup(
            kit.formula(r"24 \times 256 = 6144 \neq 5120", size=th.SIZE_MATH_SMALL),
            kit.formula(r"5120 / 24 \approx 213", size=th.SIZE_MATH_SMALL, color=th.BAD),
            kit.prose(r"Derive the head width, and \textbf{every shape downstream is wrong}.", size=th.SIZE_SMALL,
                      width=5.2),
            kit.prose(r"Only $0.25 \times 256 = 64$ channels per head get rotated.", size=th.SIZE_SMALL,
                      width=5.2, color=th.INK_3),
        ).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        layout = VGroup(tab, right).arrange(RIGHT, buff=1.0)
        self.place(layout)
        tab.bands.set_opacity(0)
        with self.narrate(
            "Here are the fields to know before chapter 2. "
            "The hidden size, the residual stream's width, is 5120. "
            "There are 64 layers. "
            "Full-attention layers have 24 query heads and 4 KV heads. "
            "The MLP width holds most of the parameters. "
            "The vocabulary is so large that the embedding alone is 1.27 billion parameters. "
            "Only a quarter of each head's channels get rotated. "
            "The head width is the one that catches people. "
            "Here, 24 heads times 256 is 6144, not 5120. "
            "Derive it by division instead, and every shape downstream is wrong."
        ) as t:
            self.play(FadeIn(tab.rules), *[FadeIn(c) for c in tab.cells[0]], run_time=0.6)
            for r, cue in zip([(1,), (2,), (3, 4), (6,), (7,), (8,)],
                              ["The hidden size", "There are 64", "Full-attention layers have", "The MLP width",
                               "The vocabulary is so", "Only a quarter"]):
                t.wait_until(cue)
                self.play(*[FadeIn(c) for i in r for c in tab.cells[i]], run_time=0.5)
            t.wait_until("Only a quarter")
            self.play(FadeIn(right[3]), run_time=0.5)
            t.wait_until("The head width is")
            self.play(*[FadeIn(c) for c in tab.cells[5]], tab.bands.animate.set_opacity(1), run_time=0.7)
            t.wait_until("Here, 24 heads")
            self.play(FadeIn(right[0], shift=UP * 0.1), run_time=0.7)
            t.wait_until("Derive it by")
            self.play(FadeIn(right[1], shift=UP * 0.1), run_time=0.6)
            self.play(FadeIn(right[2]), run_time=0.6)

    # What goes wrong ---------------------------------------------------------------------

    def what_goes_wrong(self):
        self.slide("what-goes-wrong", "Four loading bugs that fail quietly", book_section="s-what-goes-wrong")
        w = 6.0
        cards = VGroup(
            kit.callout("warning", r"Front tensors load; the last ones read past the end, or zeros. Check "
                                   r"\texttt{crc32.txt}.", title="A truncated shard", width=w, size=th.SIZE_SMALL),
            kit.callout("warning", r"\texttt{model.layers.i} or \texttt{model.language\_model.layers.i}: "
                                   r"know one, load nothing.", title="The wrong name prefix", width=w,
                        size=th.SIZE_SMALL),
            kit.callout("warning", r"Asking for float32 doubles the tensor's memory and copies it.",
                        title="A silent dtype conversion", width=w, size=th.SIZE_SMALL),
            kit.callout("note", r"The flag is false here, so \texttt{lm\_head.weight} exists too.",
                        title="Tied embeddings", width=w, size=th.SIZE_SMALL),
        )
        top = VGroup(cards[0], cards[1]).arrange(RIGHT, buff=0.35, aligned_edge=UP)
        bot = VGroup(cards[2], cards[3]).arrange(RIGHT, buff=0.35, aligned_edge=UP)
        for a, b in ((cards[0], cards[1]), (cards[2], cards[3])):
            hgt = max(a.panel.height, b.panel.height)
            for c in (a, b):
                c.panel.stretch_to_fit_height(hgt, about_edge=UP)
                c[1].stretch_to_fit_height(hgt - 0.2).align_to(c.panel, UP).shift(DOWN * 0.1)
        grid_ = VGroup(top, bot).arrange(DOWN, buff=0.3)
        noise = kit.pill("every shape right, every output noise", th.BAD, size=th.SIZE_SMALL)
        col = VGroup(grid_, noise).arrange(DOWN, buff=0.45)
        self.place(col)
        with self.narrate(
            "These loading mistakes fail quietly, with no error. "
            "A truncated shard still parses, and its front tensors still load. "
            "The last ones read past the end, or read zeros, and the model degenerates after a few fluent tokens. "
            "Checking the repository's checksums rules this out in minutes. "
            "Some checkpoints use a different name prefix, and a loader that knows only one loads nothing. "
            "Every shape is right, and every output is noise. "
            "Asking for float32 doubles a tensor's memory, and over 64 layers, mapping's advantage disappears. "
            "Finally, this model doesn't tie its embeddings, so chapter 2 counts the embedding twice."
        ) as t:
            t.wait_until("A truncated shard")
            self.play(FadeIn(cards[0], shift=UP * 0.1), run_time=0.7)
            t.wait_until("Some checkpoints use")
            self.play(FadeIn(cards[1], shift=UP * 0.1), run_time=0.7)
            t.wait_until("Every shape is right")
            self.play(FadeIn(noise, scale=0.9), run_time=0.6)
            t.wait_until("Asking for float32")
            self.play(FadeIn(cards[2], shift=UP * 0.1), run_time=0.7)
            t.wait_until("Finally, this model")
            self.play(FadeIn(cards[3], shift=UP * 0.1), run_time=0.7)
