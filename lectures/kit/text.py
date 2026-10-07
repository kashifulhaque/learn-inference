"""Text in the site's typefaces: labels, prose with inline maths, formulas.

Three kinds of text, three tools:

- `label()` is one line of Pango text (the sans or serif family) with no
  mathematics: kickers, axis titles, legend entries, pill badges.
- `prose()` is a wrapped paragraph in the serif family, set by XeLaTeX, so it
  can hold inline maths such as `$\\bar X$` and `\\texttt{...}` in the mono
  family. The maths stays in Computer Modern, which is what KaTeX draws on
  the site.
- `formula()` is display mathematics (`MathTex`), sized to sit beside prose.
- `mono()` is one line of code-font text with no highlighting; use
  `kit.code()` for code.

All sizes are in `Text` units (see `theme.TEX_SCALE`).
"""

from __future__ import annotations

from manim import (
    config,
    DOWN,
    LEFT,
    NORMAL,
    SEMIBOLD,
    MathTex,
    Tex,
    TexTemplate,
    Text,
    VGroup,
)

from . import theme as th

# XeLaTeX with fontspec's `no-math` option: prose in the serif family, code
# in the mono family, maths untouched in Computer Modern.
SERIF_TEMPLATE = TexTemplate(
    tex_compiler="xelatex",
    output_format=".xdv",
    preamble=rf"""
\usepackage[no-math]{{fontspec}}
\setmainfont{{{th.SERIF}}}[Numbers=Lining]
\setmonofont{{{th.MONO}}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{bm}}
\usepackage{{xcolor}}
\linespread{{1.18}}
""",
)

# The same maths template for `formula()`: pdfLaTeX, AMS packages, `bm`.
MATH_TEMPLATE = TexTemplate(
    preamble=r"""
\usepackage[english]{babel}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{xcolor}
""",
)

# Measured: at `font_size` 48, one TeX point is 0.049813 Manim units.
_UNITS_PER_PT_AT_48 = 0.049813


def _tex_size(size: float) -> float:
    return size * th.TEX_SCALE


def pt_for_width(width: float, size: float) -> float:
    """Converts a width in Manim units to TeX points at a given text size."""
    return width / (_UNITS_PER_PT_AT_48 * _tex_size(size) / 48)


# Pango hints glyph advances to whole pixels at the requested size, which
# spaces small text unevenly ("ari th metic"). Setting the text four times
# larger and scaling it down keeps the font's own spacing.
_OVERSAMPLE = 4


def pango(cls, text: str, size: float, oversample: int = _OVERSAMPLE, **kwargs):
    """Builds a `Text` or `MarkupText` at `size`, set large and scaled down.

    `MarkupText` wraps at a fixed Pango width, so long markup lines need
    `oversample=1`.
    """
    if oversample == 1:
        return cls(text, font_size=size, **kwargs)
    # Pango lays `Text` out on a canvas as wide as the video, so a large
    # setting would wrap at low quality. Widen the canvas while it's set.
    saved = config.pixel_width
    config.pixel_width = max(saved, 1920) * oversample
    try:
        mob = cls(text, font_size=size * oversample, **kwargs)
    finally:
        config.pixel_width = saved
    mob.scale(1 / oversample)
    return mob


def label(
    text: str,
    size: float = th.SIZE_LABEL,
    color=th.INK_3,
    font: str = th.SANS,
    weight=NORMAL,
    slant: str = "NORMAL",
) -> Text:
    """One line of text with no mathematics."""
    return pango(Text, text, size, font=font, color=color, weight=weight, slant=slant)


def mono(text: str, size: float = th.SIZE_LABEL, color=th.INK_2, weight=NORMAL) -> Text:
    """One line of text in the code font, with no highlighting."""
    return pango(Text, text, size, font=th.MONO, color=color, weight=weight, disable_ligatures=True)


def heading(text: str, size: float = th.SIZE_HEADING, color=th.INK) -> Text:
    """A one-line serif heading, semibold."""
    return pango(Text, text, size, font=th.SERIF, color=color, weight=SEMIBOLD)


def prose(
    text: str,
    size: float = th.SIZE_BODY,
    color=th.INK_2,
    width: float | None = th.MEASURE,
    align: str = "left",
) -> Tex:
    """A paragraph of serif prose that may hold inline `$...$` maths.

    Args:
      text: LaTeX source. Escape `%`, `&`, `#`, and `_` outside maths.
      size: Text size in `Text` units.
      color: Text colour.
      width: Wrap width in Manim units. `None` sets one unwrapped line.
      align: `left`, `center`, or `right`.
    """
    if width is None:
        body = text
    else:
        flush = {"left": r"\raggedright", "center": r"\centering", "right": r"\raggedleft"}[align]
        body = rf"\begin{{minipage}}{{{pt_for_width(width, size):.1f}pt}}{flush} {text}\end{{minipage}}"
    mob = Tex(body, tex_template=SERIF_TEMPLATE, font_size=_tex_size(size), color=color)
    return mob


def formula(
    *parts: str,
    size: float = th.SIZE_MATH,
    color=th.INK,
    colors: dict[str, object] | None = None,
) -> MathTex:
    """Display mathematics.

    Pass the formula in several strings to address the pieces later:
    `formula(r"\\bar X", r"\\sim", r"N(\\mu, \\sigma^2/n)")[2]` is the right-hand
    side. `colors` maps a substring to a colour, for example
    `{r"\\sigma": ORANGE}`.
    """
    mob = MathTex(*parts, tex_template=MATH_TEMPLATE, font_size=_tex_size(size), color=color)
    if colors:
        for tex, col in colors.items():
            mob.set_color_by_tex(tex, col)
    return mob


def math_label(tex: str, size: float = th.SIZE_LABEL, color=th.INK_3) -> MathTex:
    """Small inline maths for axis labels and annotations."""
    return MathTex(tex, tex_template=MATH_TEMPLATE, font_size=_tex_size(size), color=color)


def stack(*mobs, buff: float = 0.3, aligned_edge=LEFT) -> VGroup:
    """Stacks mobjects top to bottom, left-aligned by default."""
    return VGroup(*mobs).arrange(DOWN, buff=buff, aligned_edge=aligned_edge)
