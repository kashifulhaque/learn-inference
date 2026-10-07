"""The `Lecture` scene: slides, narration, the settled-frame rule, metadata.

A lecture is a Manim scene cut into sections, one section per slide. Scene
files subclass `Lecture` and write `construct()` as a run of slides:

    class Lecture(kit.Lecture):
        def construct(self):
            self.title_slide()
            self.slide("kv-per-token", "Every token adds 64 KiB",
                       book_section="s-kv-cache")
            with self.narrate("Each new token adds one key and one value ...") as t:
                ...

The kit enforces three rules so that every slide is a clean still:

1. `slide()` clears the previous slide at the *start* of the new section.
2. Every section ends on a frame held still for at least `theme.HOLD`
   seconds. The kit adds the hold if the scene did not leave one, and it
   strips updaters first, so nothing moves during the hold.
3. At the end of each section the kit saves the frame (lossless PNG) and
   checks that nothing sits outside the safe area. A mobject outside it
   fails the render.

The kit writes `meta.json` (sections, timings, narration, caption cues) and
the frames into the directory named by the `LECTURE_OUT` environment
variable, which `lectures/build.py` sets.

A scene lives at `lectures/scenes/<slug>.py`; the kit reads the slug from
the file name and the chapter from the path pattern in `course.json`.
"""

from __future__ import annotations

import inspect
import json
import os
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from manim import (
    DL,
    DOWN,
    LEFT,
    MEDIUM,
    SEMIBOLD,
    UP,
    FadeIn,
    FadeOut,
    Group,
    Mobject,
    Scene,
    Wait,
    config,
)

from . import blocks
from . import course
from . import theme as th
from .narration import Speech, Voice
from .text import label, stack

_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# Top-level classes whose bounding boxes are compared for overlap warnings.
_TEXTY = ("Text", "MarkupText", "Tex", "MathTex", "SingleStringMathTex", "Paragraph")


class LectureError(RuntimeError):
    """A lecture broke a kit rule: a bad id, a clipped slide, and so on."""


@dataclass
class _Section:
    index: int
    id: str
    title: str
    book_section: str | None
    start: float
    end: float = 0.0
    notes: list[str] = field(default_factory=list)
    cues: list[dict] = field(default_factory=list)
    frame: str = ""
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False


class Narration:
    """The handle `Lecture.narrate` yields: timings for one narration block.

    Times are seconds. `start` is the scene time when the voice starts.
    """

    def __init__(self, scene: "Lecture", speech: Speech, start: float):
        self.scene = scene
        self.speech = speech
        self.start = start

    @property
    def duration(self) -> float:
        """Length of the spoken audio."""
        return self.speech.duration

    @property
    def elapsed(self) -> float:
        """Seconds since the voice started."""
        return self.scene.renderer.time - self.start

    @property
    def remaining(self) -> float:
        """Seconds of speech left, never less than one frame.

        Pass it as `run_time` to make an animation end with the sentence.
        """
        return max(1 / config.frame_rate, self.duration - self.elapsed)

    @property
    def sentences(self) -> int:
        return len(self.speech.sentences)

    def index(self, which: int | str) -> int:
        """Resolves a sentence: an index, or a fragment of the sentence's text.

        A fragment is safer than an index, because it survives edits that add
        or remove a sentence earlier in the block.
        """
        if isinstance(which, int):
            if not 0 <= which < len(self.speech.sentences):
                raise LectureError(
                    f"Sentence {which} does not exist; this narration has {len(self.speech.sentences)} "
                    f"sentences: {[c.text[:40] for c in self.speech.sentences]}"
                )
            return which
        hits = [i for i, c in enumerate(self.speech.sentences) if which in c.text]
        if len(hits) != 1:
            raise LectureError(
                f"{which!r} matches {len(hits)} sentences of this narration; it must match exactly one."
            )
        return hits[0]

    def sentence_start(self, which: int | str) -> float:
        """Offset of a sentence from the start of the block."""
        return self.speech.sentences[self.index(which)].start

    def sentence_end(self, which: int | str) -> float:
        return self.speech.sentences[self.index(which)].end

    def until(self, which: int | str) -> float:
        """Seconds from now until the sentence starts, never negative."""
        return max(0.0, self.start + self.sentence_start(which) - self.scene.renderer.time)

    def until_end(self, which: int | str) -> float:
        """Seconds from now until the sentence ends, at least one frame."""
        return max(1 / config.frame_rate, self.start + self.sentence_end(which) - self.scene.renderer.time)

    def wait_until(self, which: int | str) -> None:
        """Holds the frame until the sentence starts. No-op if it has.

        `which` is a 0-based sentence index or a fragment of its text:
        `t.wait_until("Second, it is narrower")`.
        """
        dt = self.until(which)
        if dt >= 1 / config.frame_rate:
            self.scene.wait(dt)


class Lecture(Scene):
    """Base class for every lecture scene. See the module docstring.

    A scene may set `slug`, the chapter slug such as `02-memory-arithmetic`;
    by default the kit takes it from the file name,
    `lectures/scenes/<slug>.py`.

    Attributes the kit sets up:

    - `self.color`: the course accent colour.
    - `self.colors`: four data-series colours, the accent first.
    - `self.chapter`: the chapter's front matter plus `number`,
      `part_label`, and `part_name`.
    """

    slug: str | None = None

    # Setup ------------------------------------------------------------------

    def setup(self):
        super().setup()
        path = Path(inspect.getfile(type(self))).resolve()
        self.slug = self.slug or path.stem
        self.color = th.ACCENT
        self.colors = th.series()
        try:
            self.chapter = read_chapter(self.slug)
        except FileNotFoundError as err:
            raise LectureError(str(err)) from err
        self.camera.background_color = th.CANVAS
        self.voice = Voice()
        self.out_dir = Path(os.environ.get("LECTURE_OUT", Path(config.media_dir) / "lecture"))
        (self.out_dir / "frames").mkdir(parents=True, exist_ok=True)
        self.strict = os.environ.get("LECTURE_LAYOUT", "strict") != "warn"
        only = os.environ.get("LECTURE_ONLY", "").strip()
        self.only = {s for s in only.split(",") if s} if only else None
        self.sections: list[_Section] = []
        self._current: _Section | None = None
        self._last_motion = 0.0
        self._narration: Narration | None = None
        # Scene time spent in sections skipped by LECTURE_ONLY. Those sections
        # leave no video, so later audio shifts back by this much.
        self._skipped_time = 0.0

    # Motion tracking --------------------------------------------------------
    # The hold rule needs to know when the frame last changed. `wait()` goes
    # through `play(Wait(...))`, so a play made only of waits is not motion.

    def play(self, *args, **kwargs):
        super().play(*args, **kwargs)
        if not all(isinstance(a, Wait) for a in args):
            self._last_motion = self.renderer.time

    def add(self, *mobjects):
        result = super().add(*mobjects)
        # Wait() adds an empty placeholder mobject; that is not a change.
        if any(sub.points.size for m in mobjects for sub in m.get_family()):
            self._last_motion = self.renderer.time
        return result

    def remove(self, *mobjects):
        result = super().remove(*mobjects)
        self._last_motion = self.renderer.time
        return result

    # Slides -----------------------------------------------------------------

    def slide(
        self,
        id: str,
        title: str | None,
        book_section: str | None = None,
        *,
        keep: tuple[Mobject, ...] = (),
        header: bool = True,
    ) -> None:
        """Starts a new slide: a new section, a cleared screen, a header.

        Args:
          id: The section id, lowercase words joined by hyphens, unique in
            the lecture. Lessons refer to the slide as `lecture:<id>`.
          title: The slide title, in sentence case. It also becomes the
            section title in the manifest.
          book_section: The id of the reader's `##` section this slide
            teaches, such as `s-the-idea`. The build checks that it exists.
          keep: Mobjects from the previous slide to carry over instead of
            fading out.
          header: False leaves the header off (the title card uses this).
        """
        if self._narration is not None:
            raise LectureError("slide() inside a narrate() block; close the narration first.")
        if not _ID.fullmatch(id):
            raise LectureError(f"Slide id {id!r} must be lowercase words joined by hyphens.")
        if any(s.id == id for s in self.sections):
            raise LectureError(f"Slide id {id!r} is used twice.")
        if book_section is not None and not book_section.startswith("s-"):
            raise LectureError(f"book_section {book_section!r} must be a reader section id starting with 's-'.")
        self._finish_section()
        skip = self.only is not None and id not in self.only
        self.next_section(id, skip_animations=skip)
        section = _Section(
            index=len(self.sections) + 1,
            skipped=skip,
            id=id,
            title=title or self.chapter["title"],
            book_section=book_section,
            start=self.renderer.time,
        )
        self.sections.append(section)
        self._current = section
        # Clear the previous slide now, at the start of this section.
        leaving = [m for m in self.mobjects if m not in keep]
        if leaving:
            self.play(FadeOut(Group(*leaving)), run_time=th.FADE)
        for m in list(self.mobjects):
            if m not in keep:
                self.remove(m)
        if header and title:
            self.header = self.make_header(title)
            self.play(FadeIn(self.header, shift=DOWN * 0.08), run_time=0.5)
        else:
            self.header = None

    def make_header(self, title: str):
        """The slide header: a kicker line in the accent colour over the title."""
        kicker = label(
            f"{self.chapter['number']}  ·  {self.chapter['title']}",
            size=th.SIZE_LABEL - 2,
            color=self.color,
            weight=MEDIUM,
        )
        name = label(title, size=th.SIZE_TITLE, color=th.INK, font=th.SERIF, weight=SEMIBOLD)
        blocks.fit(name, width=th.SAFE_WIDTH)
        group = stack(kicker, name, buff=0.2)
        group.to_corner(UP + LEFT, buff=0).shift([th.SAFE_LEFT - group.get_left()[0], th.SAFE_TOP - group.get_top()[1], 0])
        return group

    def title_slide(self, id: str = "title", narration: str | None = None, summary: str | None = None) -> None:
        """The opening slide: chapter number, part, title, and summary.

        Args:
          id: The slide id.
          narration: What the voice says. Defaults to the chapter title and
            its summary from the front matter.
          summary: The italic line under the title. Defaults to the chapter's
            `summary`; pass a shorter one if the summary runs past three lines.
        """
        self.slide(id, self.chapter["title"], header=False)
        number = self.chapter["number"]
        card = blocks.title_card(
            self.chapter["title"],
            summary if summary is not None else self.chapter.get("summary", ""),
            f"{course.config()['chapter_label']} {number}" if number else course.config()["chapter_label"],
            self.chapter.get("part") or None,
            course.config()["name"],
            color=self.color,
        )
        text = narration or f"{self.chapter['title']}. {self.chapter.get('summary', '')}".strip()
        with self.narrate(text):
            self.play(FadeIn(card.kicker), FadeIn(card.title, shift=UP * 0.1), run_time=0.9)
            self.play(FadeIn(card.rule), FadeIn(card.summary, shift=UP * 0.1), run_time=0.8)
            if len(card.badge):
                self.play(FadeIn(card.badge), run_time=0.4)

    def closing_slide(
        self,
        recap: list[str],
        id: str = "where-next",
        title: str = "Where next",
        narration: str | None = None,
        next_hook: str | None = None,
        book_section: str | None = None,
    ) -> None:
        """The closing slide: a recap, and the next chapter in course order.

        Args:
          recap: Two to four short lines, one per idea to keep. Prose, so
            inline maths is allowed.
          narration: What the voice says. Defaults to the recap lines.
          next_hook: One line on what the next chapter does. Defaults to the
            next chapter's `summary`; pass "" to show only its title.
          book_section: The reader section the recap draws on, if any.
        """
        self.slide(id, title, book_section=book_section)
        nxt = next_chapter(self.slug)
        hook = None
        if nxt is not None:
            hook = blocks.tex_escape(nxt.get("summary", "")) if next_hook is None else next_hook
        label_text = "Next chapter" + (f"  ·  {nxt['number']}" if nxt and nxt.get("number") else "")
        card = blocks.where_next(
            recap,
            nxt["title"] if nxt else None,
            hook or None,
            next_label=label_text,
            color=self.color,
        )
        # The AI-voice disclosure sits along the bottom edge, so the card
        # fits in the space above it.
        disclosure = course.config().get("narration_disclosure")
        footer = None
        if disclosure:
            footer = label(disclosure, size=th.SIZE_LABEL, color=th.INK_4)
            footer.move_to([0, 0, 0]).align_to([th.SAFE_LEFT, th.SAFE_BOTTOM, 0], DL)
            blocks.fit(card, th.SAFE_WIDTH, th.CONTENT_HEIGHT - footer.height - 0.35)
        self.place(card, "top", fit=footer is None)
        text = narration or " ".join(recap)
        with self.narrate(text) as t:
            if footer is not None:
                self.add(footer)
            self.play(FadeIn(card.left_col[0]), run_time=0.4)
            for i, row in enumerate(card.items):
                self.play(FadeIn(row, shift=0.1 * np.array([1, 0, 0])), run_time=0.5)
            self.play(FadeIn(card.right_col, shift=UP * 0.1), run_time=0.7)

    # Layout -----------------------------------------------------------------

    def place(self, mob: Mobject, where: str = "center", fit: bool = True) -> Mobject:
        """Fits `mob` into the content area and positions it.

        `where` is `center`, `left`, `right`, `top`, or `top-left`.
        """
        if fit:
            blocks.fit(mob, th.SAFE_WIDTH, th.CONTENT_HEIGHT)
        cx, cy = 0.0, th.CONTENT_CENTER_Y
        mob.move_to([cx, cy, 0])
        if "left" in where:
            mob.shift([th.SAFE_LEFT - mob.get_left()[0], 0, 0])
        if "right" in where:
            mob.shift([th.SAFE_RIGHT - mob.get_right()[0], 0, 0])
        if "top" in where:
            mob.shift([0, th.CONTENT_TOP - mob.get_top()[1], 0])
        return mob

    # Narration --------------------------------------------------------------

    @contextmanager
    def narrate(self, text: str, say: str | None = None, tail: float = th.NARRATION_TAIL):
        """Speaks `text` over the animations played inside the block.

        On exit the block waits until the voice has finished, plus `tail`
        seconds, so narration never spills into the next slide. The text is
        the caption and becomes part of the slide's speaker notes.

        Args:
          text: What the voice says, in plain sentences. Captions show it
            verbatim, symbols and all.
          say: A spoken form, if the automatic one reads a symbol wrongly.
          tail: Silence after the voice ends before the block closes.

        Yields:
          A `Narration` with `duration`, `remaining`, `until(i)`, and
          `wait_until(i)` for syncing animations to sentence `i`.
        """
        if self._current is None:
            raise LectureError("narrate() before the first slide().")
        if self._narration is not None:
            raise LectureError("narrate() blocks cannot nest.")
        speech = self.voice.speak(text, say)
        start = self.renderer.time
        if not self._current.skipped:
            self.add_sound(str(speech.path), time_offset=-self._skipped_time)
        handle = Narration(self, speech, start)
        self._narration = handle
        try:
            yield handle
        finally:
            self._narration = None
        rest = start + speech.duration + tail - self.renderer.time
        if rest >= 1 / config.frame_rate:
            self.wait(rest)
        self._current.notes.append(speech.text)
        for cue in speech.cues:
            self._current.cues.append({"start": start + cue.start, "end": start + cue.end, "text": cue.text})

    def pause(self, seconds: float = 0.6) -> None:
        """Holds the current frame, with no narration."""
        self.wait(seconds)

    # Section end ------------------------------------------------------------

    def _finish_section(self) -> None:
        section = self._current
        if section is None:
            return
        for m in self.mobjects:
            for sub in m.get_family():
                sub.clear_updaters()
        held = self.renderer.time - self._last_motion
        if held < th.HOLD:
            self.wait(th.HOLD - held + 0.5 / config.frame_rate)
        self._check_layout(section)
        # Draw every mobject onto a clean background. The renderer's cached
        # static image already holds the static mobjects, and drawing over it
        # would composite translucent fills twice.
        self.renderer.static_image = None
        self.renderer.update_frame(self)
        frame = self.renderer.get_frame()
        name = f"{section.index:02d}-{section.id}.png"
        from PIL import Image

        Image.fromarray(np.asarray(frame)).convert("RGB").save(self.out_dir / "frames" / name)
        section.frame = f"frames/{name}"
        section.end = self.renderer.time
        if section.skipped:
            self._skipped_time += section.end - section.start
        self._current = None

    def _check_layout(self, section: _Section) -> None:
        tol = 0.02
        left, right = -th.FRAME_W / 2 + th.MARGIN_X - tol, th.FRAME_W / 2 - th.MARGIN_X + tol
        bottom, top = -th.FRAME_H / 2 + th.MARGIN_Y - tol, th.FRAME_H / 2 - th.MARGIN_Y + tol
        problems = []
        for m in self.mobjects:
            if not _visible(m):
                continue
            x0, y0 = m.get_corner(DOWN + LEFT)[:2]
            x1, y1 = m.get_corner(UP + np.array([1, 0, 0]))[:2]
            if x0 < left or x1 > right or y0 < bottom or y1 > top:
                problems.append(
                    f"{_describe(m)} spans x {x0:.2f}..{x1:.2f}, y {y0:.2f}..{y1:.2f}; "
                    f"the safe area is x {left:.2f}..{right:.2f}, y {bottom:.2f}..{top:.2f}"
                )
        if problems:
            message = f"Slide {section.id!r} leaves the safe area:\n  " + "\n  ".join(problems)
            if self.strict:
                raise LectureError(message)
            section.warnings.append(message)
            print("WARNING:", message)
        # Overlap between text belonging to different top-level mobjects.
        leaves = []
        for top_i, m in enumerate(self.mobjects):
            for sub in m.get_family():
                if type(sub).__name__ in _TEXTY and _visible(sub):
                    leaves.append((top_i, sub))
        for i in range(len(leaves)):
            for j in range(i + 1, len(leaves)):
                (ti, a), (tj, b) = leaves[i], leaves[j]
                if ti == tj or _contains(a, b) or _contains(b, a):
                    continue
                if _overlap(a, b, shrink=0.03):
                    msg = f"Slide {section.id!r}: text overlaps: {_describe(a)} and {_describe(b)}"
                    section.warnings.append(msg)
                    print("WARNING:", msg)

    def tear_down(self):
        self._finish_section()
        meta = {
            "slug": self.slug,
            "course": course.config()["course"],
            "title": self.chapter["title"],
            "fps": config.frame_rate,
            "width": config.pixel_width,
            "height": config.pixel_height,
            "duration": self.renderer.time,
            "only": sorted(self.only) if self.only is not None else None,
            "tts": {
                "mode": self.voice.mode,
                "synthesised": self.voice.synthesised,
                "cached": self.voice.cached,
            },
            "sections": [asdict(s) for s in self.sections],
        }
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        super().tear_down()


# Helpers --------------------------------------------------------------------


def _visible(m: Mobject) -> bool:
    if type(m).__name__ in ("ValueTracker", "ComplexValueTracker"):
        return False
    fam = m.get_family()
    for sub in fam:
        fill = getattr(sub, "fill_opacity", None)
        stroke = getattr(sub, "stroke_opacity", None)
        try:
            op = max(float(np.max(sub.get_fill_opacities())), float(np.max(sub.get_stroke_opacities())))
        except Exception:
            op = 1.0 if fill is None and stroke is None else 0.0
        if op > 0.01 and getattr(sub, "points", np.zeros((0, 3))).size:
            return True
    return False


def _describe(m: Mobject) -> str:
    name = type(m).__name__
    text = getattr(m, "text", None) or getattr(m, "tex_string", None) or getattr(m, "original_text", None)
    if text:
        text = " ".join(str(text).split())
        return f"{name}({text[:48]!r})"
    return name


def _box(m: Mobject):
    x0, y0 = m.get_corner(DOWN + LEFT)[:2]
    x1, y1 = m.get_corner(UP + np.array([1, 0, 0]))[:2]
    return x0, y0, x1, y1


def _overlap(a: Mobject, b: Mobject, shrink: float) -> bool:
    ax0, ay0, ax1, ay1 = _box(a)
    bx0, by0, bx1, by1 = _box(b)
    return (min(ax1, bx1) - max(ax0, bx0)) > shrink and (min(ay1, by1) - max(ay0, by0)) > shrink


def _contains(a: Mobject, b: Mobject) -> bool:
    return b in a.get_family()


def read_chapter(slug: str) -> dict:
    """Reads a chapter's front matter: title, part, summary, and the rest."""
    return course.read_chapter(slug)


def next_chapter(slug: str) -> dict | None:
    """The chapter after `slug` in course order, or None for the last one."""
    return course.next_chapter(slug)
