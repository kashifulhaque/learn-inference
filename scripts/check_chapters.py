"""Checks each chapter's shape against content/STYLE.md.

    python3 scripts/check_chapters.py [file ...]

Reports callout markers the reader doesn't recognize, ```viz fences that name
no registered figure, and chapters missing the parts every chapter has: one
summary, one recap, the standard sections, and questions written as callouts.
Exits nonzero if anything fails.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KINDS = {
    "TLDR", "KEY", "INTUITION", "EXAMPLE", "NOTE", "TIP", "WARNING",
    "DEEPDIVE", "QUESTION", "RECAP", "TRY", "IMPORTANT", "CAUTION",
}
REQUIRED_SECTIONS = ["Check your understanding", "Further reading"]
MARKER = re.compile(r"^((?:>[ \t]?)+)\[!([A-Za-z]+)\]")
FENCE = re.compile(r"^(?:>[ \t]?)*[ \t]*```")
VIZ_FENCE = re.compile(r"^(?:>[ \t]?)*[ \t]*```viz\s*$")
VIZ_DIR = ROOT / "frontend" / "src" / "components" / "viz"
# A registry key: a quoted name that starts with a chapter number.
VIZ_KEY = re.compile(r'^\s*"(\d\d[a-z]?-[a-z0-9-]+)"\s*:', re.MULTILINE)


def figure_names() -> set[str]:
    """The figure names registered in each group's index.ts."""
    names: set[str] = set()
    for index in VIZ_DIR.glob("*/index.ts"):
        names.update(VIZ_KEY.findall(index.read_text()))
    return names


def check(path: Path, figures: set[str]) -> list[str]:
    problems: list[str] = []
    counts: dict[str, int] = {}
    headings: list[str] = []
    in_fence = False
    in_viz = False
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if FENCE.match(line):
            in_viz = not in_fence and bool(VIZ_FENCE.match(line))
            in_fence = not in_fence
            continue
        if in_fence:
            if in_viz and line.strip():
                name = re.sub(r"^(?:>[ \t]?)*", "", line).strip()
                if name not in figures:
                    problems.append(f"line {number}: no figure is registered as {name!r}")
                in_viz = False
            continue
        if line.startswith("## "):
            headings.append(line[3:].strip())
        match = MARKER.match(line)
        if match:
            kind = match.group(2).upper()
            counts[kind] = counts.get(kind, 0) + 1
            if kind not in KINDS:
                problems.append(f"line {number}: unknown callout [!{match.group(2)}]")
            if kind == "DEEPDIVE" and not line[match.end():].strip():
                problems.append(f"line {number}: a deep dive needs a title")
            if kind == "QUESTION" and not line[match.end():].strip():
                problems.append(f"line {number}: a question callout needs the question as its title")
        elif re.match(r"^\s*\[![A-Za-z]+\]", line):
            problems.append(f"line {number}: callout marker without a leading '>'")
    if in_fence:
        problems.append("a code fence is never closed")
    for kind in ("TLDR", "RECAP"):
        if counts.get(kind, 0) != 1:
            problems.append(f"expected one [!{kind}], found {counts.get(kind, 0)}")
    for section in REQUIRED_SECTIONS:
        if section not in headings:
            problems.append(f"missing the '## {section}' section")
    if counts.get("QUESTION", 0) == 0:
        problems.append("no [!QUESTION] callouts")
    return problems


def main() -> int:
    files = [Path(arg) for arg in sys.argv[1:]] or sorted(
        (ROOT / "content" / "chapters").glob("*.md")
    )
    failed = 0
    figures = figure_names()
    for path in files:
        problems = check(path, figures)
        if problems:
            failed += 1
            for problem in problems:
                print(f"{path.relative_to(ROOT) if path.is_absolute() else path}: {problem}")
    print(f"{len(files)} chapter(s) checked, {failed} with problems")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
