"""The lecture palette, type scale, and frame geometry.

Every colour is a token from the light theme in `frontend/src/index.css`, so
a lecture reads as part of the site. The course accent and the font families
come from `lectures/course.json`. Scenes name a role (`INK`, `RULE`,
`ACCENT`) and never a raw hex value.

Geometry is in Manim units. The frame is 14.222 by 8 units, and at 1920x1080
one unit is 135 pixels.
"""

from __future__ import annotations

from manim import ManimColor, config

from .course import config as course_config

_COURSE = course_config()

# Colour tokens (frontend/src/index.css, :root) ------------------------------

CANVAS = ManimColor("#fcfbf8")  # --paper: the slide background
CHROME = ManimColor("#f4f2ed")  # --well: sidebars and bars
SUNK = ManimColor("#f4f2ed")  # --code-bg: recessed panels and code
RAISE = ManimColor("#ffffff")  # --card: raised panels
TINT = ManimColor("#ebe8e1")  # --tint: selected rows, empty cells
RULE = ManimColor("#e4e0d7")  # --line: hairlines and grids
RULE_STRONG = ManimColor("#cdc7ba")  # --line-strong: axes and box borders

INK = ManimColor("#1d1b18")  # --fg: headings and formulas
INK_2 = ManimColor("#48443d")  # --fg-muted: body text
INK_3 = ManimColor("#706a60")  # --fg-subtle: secondary text, axis labels
INK_4 = ManimColor("#a09a8f")  # --fg-faint: kickers, captions, faint marks

ACCENT = ManimColor(_COURSE["accent"])  # --accent: the course colour
ACCENT_STRONG = ACCENT.interpolate(INK, 0.25)  # close to --accent-strong
GOOD = ManimColor("#2d7a4c")  # --ok
BAD = ManimColor("#b42318")  # --bad
WARN = ManimColor("#9a5b06")  # --warn
INFO = ManimColor("#2c5ea8")  # --info
VIOLET = ManimColor("#6d4fb0")  # --violet

# The site's highlighter: --warn at 30% over the page, as `mark` draws it.
MARK = CANVAS.interpolate(WARN, 0.30)

# The four maths colours, \hla to \hld.
HL_A = ManimColor("#2d7a4c")
HL_B = ManimColor("#2c5ea8")
HL_C = ManimColor("#a15c07")
HL_D = ManimColor("#b4335a")

# Extra data-series names kept for plots.
ORANGE = HL_C
PINK = HL_D
BLUE = INFO
GREEN = GOOD
PURPLE = VIOLET

# Syntax colours (--syn-*), shared with the site's code blocks.
CODE_BG = ManimColor("#f4f2ed")
SYN_KEYWORD = ManimColor("#a13d12")
SYN_STRING = ManimColor("#3b7738")
SYN_NUMBER = ManimColor("#8a5a00")
SYN_COMMENT = ManimColor("#8d877c")
SYN_FUNCTION = ManimColor("#2c5ea8")
SYN_TYPE = ManimColor("#7446b0")
SYN_PUNCT = ManimColor("#6b655b")


def series() -> list[ManimColor]:
    """Returns four data-series colours, the course accent first.

    Use them in order: the first for the quantity the slide is about, the
    second for the thing it's compared with, and so on.
    """
    rest = [c for c in (INFO, GOOD, VIOLET, HL_D, HL_C) if c.to_hex() != ACCENT.to_hex()]
    return [ACCENT, *rest[:3]]


def tint(color, amount: float = 0.08) -> ManimColor:
    """Mixes `amount` of `color` into the canvas, for panel backgrounds."""
    return CANVAS.interpolate(ManimColor(color), amount)


# Fonts ----------------------------------------------------------------------
# The site sets prose in Source Serif 4, interface text in IBM Plex Sans, and
# code in IBM Plex Mono. course.json can name other installed families.

_FONTS = _COURSE["fonts"]
SERIF = _FONTS.get("serif", "Source Serif 4")  # prose, as in the reader
SERIF_DISPLAY = _FONTS.get("serif_display", "Source Serif 4 Display")  # large titles
SANS = _FONTS.get("sans", "IBM Plex Sans")  # labels, kickers, numbers in charts
MONO = _FONTS.get("mono", "IBM Plex Mono")  # code

# Type scale (Manim `font_size`). The body size is the smallest size a slide
# uses for sentences; LABEL is the floor for anything at all.
SIZE_DISPLAY = 72  # title card
SIZE_TITLE = 46  # slide header
SIZE_HEADING = 36  # box titles, sub-headings
SIZE_BODY = 32  # sentences and bullets
SIZE_SMALL = 28  # secondary sentences, table cells
SIZE_LABEL = 24  # axis labels, kickers, legends
SIZE_CODE = 26  # code panels
SIZE_MATH = 38  # display formulas
SIZE_MATH_SMALL = 32  # formulas inside boxes, derivation steps

# Every size above is in `Text` units. `Tex` and `MathTex` measure
# `font_size` differently, so the kit multiplies by this factor for them:
# at equal nominal size, the lowercase letters of both come out the same
# height (measured: x-height 0.211 units for Text at 32, 0.158 for Tex).
TEX_SCALE = 1.34

# Geometry -------------------------------------------------------------------

FRAME_W = config.frame_width  # 14.222
FRAME_H = config.frame_height  # 8.0

# Safe margins: nothing may touch the outer 0.6 units (81 px) at the sides
# or 0.45 units (61 px) at top and bottom.
MARGIN_X = 0.6
MARGIN_Y = 0.45

SAFE_LEFT = -FRAME_W / 2 + MARGIN_X
SAFE_RIGHT = FRAME_W / 2 - MARGIN_X
SAFE_TOP = FRAME_H / 2 - MARGIN_Y
SAFE_BOTTOM = -FRAME_H / 2 + MARGIN_Y
SAFE_WIDTH = SAFE_RIGHT - SAFE_LEFT

# The content area sits below the slide header.
CONTENT_TOP = 2.1
CONTENT_BOTTOM = SAFE_BOTTOM
CONTENT_HEIGHT = CONTENT_TOP - CONTENT_BOTTOM
CONTENT_CENTER_Y = (CONTENT_TOP + CONTENT_BOTTOM) / 2

# Text blocks wider than this are hard to read; wrap them.
MEASURE = 11.5

# Stroke widths.
STROKE_HAIRLINE = 1.5
STROKE_AXIS = 2.0
STROKE_BOX = 2.0
STROKE_CURVE = 4.0

# Timing (seconds).
HOLD = 1.2  # every section ends on a still frame at least this long
FADE = 0.5  # the clear at the start of a section
NARRATION_TAIL = 0.35  # breath after each narration block
