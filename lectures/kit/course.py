"""The course this pipeline renders for: config, chapters, and section ids.

Everything course-specific lives in `lectures/course.json`, so the same kit
and build script serve every course. This module reads that file and the
chapter front matter with the Python standard library only, because
`lectures/build.py` loads it on a host that has no manim, NumPy, or PyYAML.
Load it as `kit.course` inside a scene, or by file path from `build.py`.

`LECTURE_COURSE_FILE` overrides the config path, which the tests use.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

LECTURES = Path(__file__).resolve().parents[1]
REPO = LECTURES.parent


@lru_cache(maxsize=None)
def config() -> dict:
    """Returns `lectures/course.json` as a dict, with defaults filled in."""
    path = Path(os.environ.get("LECTURE_COURSE_FILE", LECTURES / "course.json"))
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("course", "chapter_path", "manifest_path", "bucket", "public_base", "key_prefix", "voice"):
        if key not in cfg:
            raise KeyError(f"{path} has no {key!r}.")
    if "{slug}" not in cfg["chapter_path"] or "{slug}" not in cfg["manifest_path"]:
        raise ValueError(f"{path}: chapter_path and manifest_path must contain '{{slug}}'.")
    cfg.setdefault("name", cfg["course"])
    cfg.setdefault("chapter_label", "Chapter")
    cfg.setdefault("accent", "#c2410c")
    cfg.setdefault("fonts", {})
    cfg.setdefault("estimate_words_per_second", 2.6)
    cfg["_path"] = str(path)
    return cfg


def config_file() -> Path:
    return Path(config()["_path"])


def chapter_file(slug: str) -> Path:
    """The chapter's Markdown file, from `chapter_path` in the config."""
    return REPO / config()["chapter_path"].format(slug=slug)


def chapter_slugs() -> list[str]:
    """Every chapter slug, in course order (sorted by file name).

    A file name's stem is its slug, so `00-…` sorts before `00a-…`, which
    sorts before `01-…`, the order the site shows.
    """
    pattern = config()["chapter_path"]
    head, tail = pattern.split("{slug}", 1)
    if head and not head.endswith("/"):
        raise ValueError("chapter_path must put {slug} at the start of a path segment.")
    folder = REPO / head
    slugs = []
    for path in folder.iterdir():
        name = path.name
        if "/" in tail:  # one directory per chapter: content/chapters/{slug}/chapter.md
            if (path / tail.lstrip("/")).is_file():
                slugs.append(name)
        elif path.is_file() and name.endswith(tail) and len(name) > len(tail):
            slugs.append(name[: len(name) - len(tail)] if tail else name)
    return sorted(slugs)


# Front matter ---------------------------------------------------------------
# A small reader for the flat YAML the chapters use: scalars, quoted scalars,
# and lists of scalars. It avoids a PyYAML dependency on the host and gives
# the build and the kit identical values.

_FRONT = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n", re.S)


def _scalar(raw: str):
    raw = raw.strip()
    if not raw:
        return ""
    if raw[0] == '"' and raw[-1] == '"' and len(raw) >= 2:
        return json.loads(raw)
    if raw[0] == "'" and raw[-1] == "'" and len(raw) >= 2:
        return raw[1:-1].replace("''", "'")
    if raw in ("true", "false"):
        return raw == "true"
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    return raw


def front_matter_text(path: Path) -> str:
    match = _FRONT.match(path.read_text(encoding="utf-8"))
    return match.group(1) if match else ""


def front_matter(path: Path) -> dict:
    """Parses a file's YAML front matter (scalars and lists of scalars)."""
    out: dict = {}
    key = None
    block: list[str] | None = None
    for line in front_matter_text(path).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith((" ", "\t")):
            item = line.strip()
            if block is not None:
                block.append(item)
            elif key is not None and item.startswith("- "):
                if not isinstance(out.get(key), list):
                    out[key] = []
                out[key].append(_scalar(item[2:]))
            continue
        if block is not None and key is not None:
            out[key] = " ".join(block)
            block = None
        k, _, v = line.partition(":")
        key = k.strip()
        v = v.strip()
        if v in (">", ">-", "|", "|-"):
            block = []
        else:
            out[key] = _scalar(v) if v else []
    if block is not None and key is not None:
        out[key] = " ".join(block)
    return out


_NUMBER = re.compile(r"^\d+[a-z]?")


def chapter_number(slug: str) -> str:
    """The number a chapter goes by: the slug's leading digits and letter.

    Mirrors `chapterNumber` in `frontend/src/lib/chapters.ts`: `02` for
    `02-memory-arithmetic`, `00a` for `00a-notation-and-prerequisites`. A
    slug without a number gets its position in the course, two digits wide.
    """
    match = _NUMBER.match(slug)
    if match:
        return match.group(0)
    slugs = chapter_slugs()
    return f"{slugs.index(slug):02d}" if slug in slugs else ""


def read_chapter(slug: str) -> dict:
    """Reads a chapter's front matter and adds `number`, `part_label`, `part_name`."""
    path = chapter_file(slug)
    if not path.exists():
        raise FileNotFoundError(
            f"No chapter at {path}; name the scene file after a chapter slug."
        )
    fm = front_matter(path)
    fm.setdefault("title", slug)
    fm.setdefault("summary", "")
    fm["slug"] = slug
    fm["number"] = chapter_number(slug)
    part = str(fm.get("part", "") or "")
    label, sep, name = part.partition("—")
    fm["part_label"] = label.strip() if sep else ""
    fm["part_name"] = name.strip() if sep else part.strip()
    return fm


def next_chapter(slug: str) -> dict | None:
    """The chapter after `slug` in course order, or None for the last one."""
    slugs = chapter_slugs()
    i = slugs.index(slug)
    return read_chapter(slugs[i + 1]) if i + 1 < len(slugs) else None


def shown_fields(slug: str) -> dict:
    """The front matter a lecture shows, for the build's version hash."""
    fm = read_chapter(slug)
    return {k: fm.get(k) for k in ("title", "part", "summary", "number")}


# Reader section ids ---------------------------------------------------------
# The site gives each `##` section a positional anchor, `s-1`, `s-2`, ... in
# document order (`rehypeSectionize` in frontend/src/lib/markdownPlugins.ts).
# Positions shift when a section is added, so scenes name a section by a
# stable slug of its heading instead, using the rule from the stats course's
# `frontend/src/lib/outline.ts`: lowercase; drop `` ` ``, `*`, `_`, `$`, and
# `\`; replace each run of other non-alphanumeric characters with `-`; trim
# hyphens. The manifest carries both the slug and the positional anchor.

_SLUG_STRIP = re.compile(r"[`*_$\\]")
_SLUG_JOIN = re.compile(r"[^a-z0-9]+")
_SLUG_TRIM = re.compile(r"^-+|-+$")
_FENCE = re.compile(r"^(```|~~~)")
_H2 = re.compile(r"^## (.+?)\s*#*\s*$")


def slugify(text: str) -> str:
    text = _SLUG_STRIP.sub("", text.lower())
    return _SLUG_TRIM.sub("", _SLUG_JOIN.sub("-", text))


def headings(markdown: str) -> list[str]:
    """The `##` headings outside fenced code blocks, in order."""
    out = []
    fence = None
    for line in markdown.split("\n"):
        line = line.rstrip("\r")
        m = _FENCE.match(line.lstrip())
        if m:
            if fence is None:
                fence = m.group(1)
            elif m.group(1) == fence:
                fence = None
            continue
        if fence is None:
            h = _H2.match(line)
            if h:
                out.append(h.group(1))
    return out


def section_anchors(slug: str) -> dict[str, str]:
    """Maps each `s-<heading slug>` id to the site's positional anchor `s-N`."""
    text = chapter_file(slug).read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for i, heading in enumerate(headings(text), start=1):
        out.setdefault(f"s-{slugify(heading)}", f"s-{i}")
    return out


# Publishing -----------------------------------------------------------------


def manifest_file(slug: str) -> Path:
    return REPO / config()["manifest_path"].format(slug=slug)


def key_prefix(slug: str, version: str) -> str:
    """The R2 key prefix of one built lecture: `<key_prefix>/<slug>/<version>`."""
    return f"{config()['key_prefix'].strip('/')}/{slug}/{version}"


def base_url(slug: str, version: str) -> str:
    """The public URL every manifest path is relative to, with a trailing slash."""
    return f"{config()['public_base'].rstrip('/')}/{key_prefix(slug, version)}/"
