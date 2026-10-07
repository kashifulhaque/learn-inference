"""Code panels: syntax-highlighted source with addressable lines.

`code()` sets each source line as its own row of monospaced text, coloured
with the site's syntax palette (`--syn-*` in `frontend/src/index.css`), on a
recessed panel. Rows are addressable, so a slide can reveal, highlight, or
focus individual lines while the voice talks about them:

    panel = kit.code(SRC, "python", highlight_lines=[3])
    self.place(panel)
    self.play(FadeIn(panel.frame))           # the panel and line numbers
    self.play(*panel.reveal(1, 2, 3))        # lines 1 to 3
    self.play(panel.highlight(5))            # a band behind line 5
    self.play(panel.focus(5, 6))             # dim every other line

Line numbers are 1-based, as the panel shows them. Keep a code slide to
14 lines or fewer and about 64 characters wide; `place()` shrinks anything
larger, and shrunken code is hard to read.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    AnimationGroup,
    FadeIn,
    MarkupText,
    Rectangle,
    RoundedRectangle,
    VGroup,
)

from . import theme as th
from .text import label, pango

# Pygments token types to (colour, italic), following the highlight.js
# classes the site styles.
_STYLE_ORDER = (
    ("Comment", th.SYN_COMMENT, True),
    ("Literal.String", th.SYN_STRING, False),
    ("Literal.Number", th.SYN_NUMBER, False),
    ("Keyword.Type", th.SYN_TYPE, False),
    ("Keyword", th.SYN_KEYWORD, False),
    ("Name.Builtin.Pseudo", th.SYN_TYPE, False),
    ("Name.Builtin", th.SYN_KEYWORD, False),
    ("Name.Function", th.SYN_FUNCTION, False),
    ("Name.Decorator", th.SYN_FUNCTION, False),
    ("Name.Class", th.SYN_TYPE, False),
    ("Comment.Preproc", th.SYN_KEYWORD, False),
    ("Operator.Word", th.SYN_KEYWORD, False),
    ("Operator", th.SYN_PUNCT, False),
    ("Punctuation", th.SYN_PUNCT, False),
)

_ALIASES = {"triton": "python", "cu": "cuda", "c++": "cpp", "sh": "bash", "shell": "bash", "text": "text"}


def _style(ttype) -> tuple[object, bool]:
    from pygments import token

    best = None
    for name, color, italic in _STYLE_ORDER:
        t = token.string_to_tokentype(name)
        if ttype in t and (best is None or len(t) > len(best[0])):
            best = (t, color, italic)
    return (best[1], best[2]) if best else (th.INK, False)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tokenize(src: str, language: str) -> list[list[tuple[str, object, bool]]]:
    """Tokenises `src` into lines of `(text, colour, italic)` runs."""
    from pygments.lexers import get_lexer_by_name
    from pygments.util import ClassNotFound

    lang = _ALIASES.get(language.lower(), language.lower())
    try:
        lexer = get_lexer_by_name(lang, stripnl=False, ensurenl=False)
    except ClassNotFound as err:
        raise ValueError(f"No syntax highlighter for language {language!r}.") from err
    lines: list[list[tuple[str, object, bool]]] = [[]]
    for ttype, value in lexer.get_tokens(src):
        color, italic = _style(ttype)
        parts = value.split("\n")
        for i, part in enumerate(parts):
            if i:
                lines.append([])
            if part:
                lines[-1].append((part, color, italic))
    return lines


class CodePanel(VGroup):
    """A code panel built by `code()`. See the module docstring.

    Attributes:
      panel: The background rectangle.
      numbers: Line numbers, one mobject per line (empty without numbers).
      rows: Each line's glyphs, a list indexed from 0. Use `line(n)` for
        1-based access.
      frame: The panel, line numbers, and caption, without the code, for
        revealing the lines one at a time.
      bands: The highlight bands added so far.
    """

    def line(self, n: int) -> VGroup:
        """Returns line `n` (1-based, as numbered on the panel)."""
        i = n - self.first_line
        if not 0 <= i < len(self.rows):
            raise IndexError(f"Line {n} isn't in this panel (lines {self.first_line}..{self.first_line + len(self.rows) - 1}).")
        return self.rows[i]

    def lines(self, *ns: int) -> VGroup:
        """Returns several lines as one VGroup. `lines()` with no arguments returns every line."""
        if not ns:
            return VGroup(*self.rows)
        return VGroup(*[self.line(n) for n in ns])

    def band(self, *ns: int, color=None, opacity: float = 1.0) -> VGroup:
        """Returns new highlight bands behind lines `ns`, not yet added."""
        color = th.tint(th.ACCENT, 0.14) if color is None else color
        out = VGroup()
        for n in ns:
            slot = self._slots[n - self.first_line]
            b = Rectangle(width=slot.width, height=slot.height, fill_color=color, fill_opacity=opacity, stroke_width=0)
            b.move_to(slot)
            b.set_z_index(-1)
            out.add(b)
        return out

    def highlight(self, *ns: int, color=None, run_time: float = 0.5):
        """Returns an animation that fades in a band behind lines `ns`."""
        bands = self.band(*ns, color=color)
        self.bands.add(*bands)
        self.add(bands)
        return FadeIn(bands, run_time=run_time)

    def reveal(self, *ns: int, shift: float = 0.08) -> list:
        """Returns one FadeIn per line, for revealing lines in turn."""
        targets = ns or tuple(range(self.first_line, self.first_line + len(self.rows)))
        return [FadeIn(self.line(n), shift=RIGHT * shift) for n in targets if len(self.line(n))]

    def focus(self, *ns: int, dim: float = 0.28, run_time: float = 0.6):
        """Returns an animation that dims every line except `ns`.

        Call `focus()` with no lines to bring every line back.
        """
        keep = set(ns)
        anims = []
        for i, row in enumerate(self.rows):
            if not len(row):
                continue
            n = i + self.first_line
            anims.append(row.animate.set_opacity(1.0 if not keep or n in keep else dim))
        return AnimationGroup(*anims, run_time=run_time)


def code(
    src: str,
    language: str = "python",
    highlight_lines: Iterable[int] = (),
    size: float = th.SIZE_CODE,
    line_numbers: bool = True,
    first_line: int = 1,
    caption: str | None = None,
    min_width: float = 0.0,
    pad: float = 0.34,
) -> CodePanel:
    """A syntax-highlighted code panel in the course's mono font.

    Args:
      src: The source. Leading and trailing blank lines are dropped, and
        tabs expand to four spaces. Keep it to 14 lines or fewer.
      language: A Pygments language name: `python`, `cuda`, `cpp`, `c`,
        `bash`, `json`, `text`. `triton` means Python.
      highlight_lines: Lines (1-based) to mark with a band from the start.
      size: Text size in `Text` units. The default suits up to 14 lines
        of about 64 characters; don't go below `th.SIZE_LABEL`.
      line_numbers: Show a faint line-number gutter.
      first_line: The number of the first line, for an excerpt.
      caption: A short label in the panel's top-right corner, such as a
        file name.
      min_width: Minimum panel width in Manim units, to line up panels.
      pad: Inner padding in Manim units.

    Returns:
      A `CodePanel` laid out at the origin. Its rows are addressable with
      `line(n)`; see the class for `highlight`, `reveal`, and `focus`.
    """
    text = src.expandtabs(4).strip("\n")
    lines = tokenize(text, language)
    if len(lines) > 18:
        raise ValueError(f"A code panel holds at most 18 lines; this one has {len(lines)}. Cut the excerpt.")

    probe = pango(MarkupText, "|", size, oversample=1, font=th.MONO, disable_ligatures=True)
    strut_h = probe.height
    pitch = strut_h * 1.16
    gutter = 0.0
    numbers = VGroup()
    if line_numbers:
        last = first_line + len(lines) - 1
        digits = len(str(last))
        for i in range(len(lines)):
            n = label(str(first_line + i).rjust(digits), size=size * 0.85, color=th.INK_4, font=th.MONO)
            numbers.add(n)
        gutter = max(n.width for n in numbers) + 0.42

    rows: list[VGroup] = []
    slots = VGroup()
    x0 = 0.0
    top = 0.0
    for i, runs in enumerate(lines):
        plain = "".join(t for t, _, _ in runs)
        markup = "".join(_span(t, c, it) for t, c, it in runs)
        mob = pango(MarkupText, f"|{markup}|", size, oversample=1, font=th.MONO, color=th.INK,
                    disable_ligatures=True)
        glyphs = list(mob.submobjects)
        expected = sum(1 for ch in plain if not ch.isspace()) + 2
        if len(glyphs) != expected:
            raise ValueError(
                f"Line {first_line + i} set as {len(glyphs)} glyphs, expected {expected}; "
                "it may hold a character the code font lacks."
            )
        bar = glyphs[0]
        row_top = top - i * pitch
        mob.shift([x0 - bar.get_left()[0], row_top - (pitch - strut_h) / 2 - bar.get_top()[1], 0])
        row = VGroup(*glyphs[1:-1])
        rows.append(row)
        slot = Rectangle(width=1, height=pitch, stroke_width=0, fill_opacity=0)
        slot.move_to([0, row_top - pitch / 2, 0])
        slots.add(slot)
        if line_numbers:
            numbers[i].move_to([0, 0, 0])
            numbers[i].align_to([x0 - 0.42, 0, 0], RIGHT)
            # Sit the number on the same baseline as the code's lowercase.
            numbers[i].set_y(row_top - pitch / 2 - 0.05 * strut_h)

    code_right = max((r.get_right()[0] for r in rows if len(r)), default=x0 + 1.0)
    content_w = code_right - (x0 - gutter)
    cap = None
    if caption:
        cap = label(caption, size=th.SIZE_LABEL - 4, color=th.INK_4, font=th.MONO)
    width = max(min_width, content_w + 2 * pad, (cap.width + 2 * pad + 0.4) if cap else 0)
    head = (cap.height + 0.06) if cap is not None else 0.0
    height = len(lines) * pitch + 2 * pad + head
    left = x0 - gutter - pad
    panel = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.12,
        fill_color=th.CODE_BG,
        fill_opacity=1,
        stroke_color=th.RULE,
        stroke_width=th.STROKE_HAIRLINE,
    )
    panel.move_to([left + width / 2, top + pad + head - height / 2, 0])
    panel.set_z_index(-2)
    inner_left = left + 0.08
    for slot in slots:
        slot.stretch_to_fit_width(width - 0.16)
        slot.set_x(inner_left + (width - 0.16) / 2)
    if cap is not None:
        cap.move_to(panel.get_corner(UP + RIGHT) + LEFT * (cap.width / 2 + pad) + DOWN * (cap.height / 2 + pad * 0.6))

    result = CodePanel()
    result.first_line = first_line
    result.panel = panel
    result.numbers = numbers
    result.rows = rows
    result._slots = slots
    result.bands = VGroup()
    result.caption = cap
    frame = VGroup(panel, numbers) if cap is None else VGroup(panel, numbers, cap)
    result.frame = frame
    result.add(panel, slots, result.bands, numbers, *rows)
    if cap is not None:
        result.add(cap)
    marks = list(highlight_lines)
    if marks:
        bands = result.band(*marks)
        result.bands.add(*bands)
        frame.add(bands)
    result.move_to([0, 0, 0])
    return result


def _span(text: str, color, italic: bool) -> str:
    style = ' style="italic"' if italic else ""
    return f'<span foreground="{_hex(color)}"{style}>{_escape(text)}</span>'


def _hex(color) -> str:
    """A colour as `#rrggbb` for Pango markup."""
    from manim import ManimColor

    return ManimColor(color).to_hex()[:7]


__all__ = ["CodePanel", "code", "tokenize"]
