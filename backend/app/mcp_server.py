"""An MCP server that gives AI agents the whole course.

The server speaks the MCP Streamable HTTP transport at `/mcp`, statelessly and
with plain JSON responses, so each request stands alone and needs no session.
It exposes everything the site shows a reader as tools: chapters, labs with
their checks and solutions, lecture slides and narration, the course documents,
the PDF book's details, the GPU providers' status, and the reader's progress,
notes, drafts, and lab runs. It also lets an agent run a lab as the reader.

Every request needs `Authorization: Bearer <token>`, where the token comes
from `mcp_tokens`. The token alone decides who the caller is. The SSO headers
that `auth.current_user` trusts are never read here, because Caddy routes
`/mcp` without Authelia and so can't vouch for them.
"""

from __future__ import annotations

import functools
import hmac
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Literal

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from . import course, curriculum, db, infra, lectures
from .config import REPO_ROOT, get_settings
from .mcp_tokens import user_for_token
from .providers import provider_status
from .runs import start_run

# The ASGI scope key that carries the authenticated reader to the tools.
USER_SCOPE_KEY = "learn_inference.mcp_user"

COURSE_TITLE = "Build an inference engine"
COURSE_DESCRIPTION = (
    "A hands-on course that teaches LLM inference by having you build an engine "
    "for Qwen3.8-27B, a hybrid of gated delta linear attention and full attention. "
    "Chapters run from the model on disk through the forward pass, the KV cache, "
    "CUDA and Triton kernels, continuous batching, quantization, parallelism, and "
    "speculative decoding to a capstone engine. Each chapter links an auto-graded "
    "Python lab that runs on an A100 through RunPod, and many have a narrated lecture."
)

INSTRUCTIONS = (
    f"{COURSE_DESCRIPTION}\n\n"
    "Start with `course_overview` for the syllabus and the reader's progress. Use "
    "`search` to find where a topic is taught, `get_chapter` to read a chapter, and "
    "`get_lab` for a lab's brief, starter code, and checks. `get_lecture` has the "
    "narration for a chapter's slides. `run_lab` runs code as the reader on RunPod's "
    "GPU endpoint, which bills the course owner's account for GPU time, exactly as "
    "the editor's Run button does, so call it only when the reader wants a run. Only "
    "call `get_solution` when the reader asks for the answer."
)

# Files an agent may read with `get_document`. Nothing else is served. Each
# entry is (folder, description), where the folder is a key of `_DOCUMENT_DIRS`.
DOCUMENTS = {
    "FACTS.md": (
        "docs",
        "The fact sheet: every number the chapters state, tagged confirmed, derived, "
        "or simulated, with its source.",
    ),
    "STYLE.md": ("content", "The house style every chapter, lab, and lecture follows."),
    "lab_common.py": (
        "labs",
        "The helper module that lab tests import, such as the performance thresholds.",
    ),
}

_DOCUMENT_DIRS: dict[str, Callable[[], Path]] = {
    "docs": lambda: REPO_ROOT / "docs",
    "content": lambda: get_settings().content_dir,
    "labs": lambda: get_settings().content_dir / "labs",
}

# Bounds that keep a tool result readable.
MAX_CODE_LENGTH = 200_000
MAX_NOTE_LENGTH = 200_000
MAX_STREAM_CHARS = 16_000
MAX_CHECK_TEXT = 4_000
MAX_DOCUMENT_CHARS = 40_000
MAX_SEARCH_RESULTS = 50
MAX_RUNS = 100

mcp = FastMCP(
    name="learn-inference",
    instructions=INSTRUCTIONS,
    stateless_http=True,
    json_response=True,
    # The bearer token is the protection here. DNS-rebinding checks guard
    # unauthenticated servers on localhost, and would reject `qwen.ifkash.dev`.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


# --- plumbing ---------------------------------------------------------------


def _tool(*, writes: bool = False, open_world: bool = False) -> Callable[[Callable[..., Any]], Any]:
    """Register a tool that returns JSON text only.

    Structured output would send every result twice, as text and as an object,
    which doubles a chapter's size for no gain.
    """
    return mcp.tool(
        structured_output=False,
        annotations=ToolAnnotations(
            readOnlyHint=not writes, destructiveHint=False, openWorldHint=open_world
        ),
    )


def _threaded(fn: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
    """Run a blocking tool body (SQLite, file reads) off the event loop."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))

    return wrapper


def _user(ctx: Context) -> str:
    request = ctx.request_context.request
    user = request.scope.get(USER_SCOPE_KEY) if request is not None else None
    if not user:
        # The endpoint authenticates before the server runs, so this means a
        # transport that bypassed it.
        raise ToolError("Not authenticated.")
    return str(user)


def _lab_or_error(lab_id: str) -> dict[str, Any]:
    lab = curriculum.get_lab(lab_id)
    if lab is None:
        raise ToolError(f"No lab with ID {lab_id!r}. Chapters name theirs in `lab`.")
    return lab


def _chapter_or_error(slug: str) -> dict[str, Any]:
    chapter = curriculum.get_chapter(slug)
    if chapter is None:
        raise ToolError(f"No chapter with slug {slug!r}. Call `list_chapters` for the slugs.")
    return chapter


def _chapters_for(lab_id: str) -> list[str]:
    return [c["slug"] for c in curriculum.chapter_list() if c.get("lab") == lab_id]


def _clip(text: str, limit: int) -> str:
    """Keep the start and the end of long output, where the useful parts are."""
    if len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n... [{omitted} characters omitted] ...\n{text[-tail:]}"


def _chapter_summary(chapter: dict[str, Any], progress: dict[str, str]) -> dict[str, Any]:
    return {
        "slug": chapter["slug"],
        "title": chapter["title"],
        "part": chapter["part"],
        "summary": chapter["summary"],
        "minutes": chapter.get("minutes"),
        "lab": chapter.get("lab"),
        "gpu": chapter.get("gpu", False),
        "has_lecture": lectures.get_lecture(chapter["slug"]) is not None,
        "status": progress.get(chapter["slug"]),
    }


# --- headings and snippets --------------------------------------------------

_HEADING_LINE = re.compile(r"^(#{1,3})[ \t]+(.+?)[ \t#]*$")
_FENCE_LINE = re.compile(r"^\s*(```|~~~)")


def _headings(body: str) -> list[tuple[int, str, int]]:
    """The `#`, `##`, and `###` headings as (level, text, offset), skipping code blocks."""
    found = []
    fenced = False
    offset = 0
    for line in body.splitlines(keepends=True):
        if _FENCE_LINE.match(line):
            fenced = not fenced
        elif not fenced:
            match = _HEADING_LINE.match(line.rstrip("\r\n"))
            if match:
                found.append((len(match.group(1)), match.group(2).strip(), offset))
        offset += len(line)
    return found


def _section(headings: list[tuple[int, str, int]], at: int) -> str | None:
    """The heading of the section that position `at` falls in."""
    heading = None
    for _, text, offset in headings:
        if offset > at:
            break
        heading = text
    return heading


def _snippet(text: str, at: int, length: int) -> str:
    start = max(0, at - 80)
    end = min(len(text), at + length + 160)
    snippet = re.sub(r"\s+", " ", text[start:end]).strip()
    return ("..." if start > 0 else "") + snippet + ("..." if end < len(text) else "")


# --- course and chapters ----------------------------------------------------


@_tool()
@_threaded
def course_overview(ctx: Context) -> dict[str, Any]:
    """Get the whole syllabus and the reader's progress in one call.

    Returns the course title and description; the model and GPU the labs target;
    every part with its chapters (slug, title, summary, minutes, linked lab ID,
    whether a lecture exists, and the reader's status: null, "in_progress", or
    "done"); counts of chapters, labs, and lectures; the PDF book's details; a
    progress summary; and the documents `get_document` serves. Call this first
    to orient yourself, then `get_chapter` with a slug to read a chapter.
    """
    user = _user(ctx)
    settings = get_settings()
    progress = db.get_progress(user)
    chapters = curriculum.chapter_list()
    parts: list[dict[str, Any]] = []
    for chapter in chapters:
        if not parts or parts[-1]["part"] != chapter["part"]:
            parts.append({"part": chapter["part"], "chapters": []})
        parts[-1]["chapters"].append(_chapter_summary(chapter, progress))
    return {
        "title": COURSE_TITLE,
        "description": COURSE_DESCRIPTION,
        "reader": user,
        "model": settings.model_id,
        "small_model": settings.small_model_id,
        "gpu": settings.gpu_type,
        "parts": parts,
        "counts": {
            "parts": len(parts),
            "chapters": len(chapters),
            "labs": len(curriculum.load_labs()),
            "lectures": len(lectures.load_lectures()),
        },
        "book": course.book_info(settings),
        "progress": course.progress_summary(user),
        "documents": {name: description for name, (_, description) in DOCUMENTS.items()},
    }


@_tool()
@_threaded
def list_chapters(ctx: Context, part: str | None = None) -> dict[str, Any]:
    """List chapters in course order, with their headings and objectives.

    Args:
        part: Only chapters whose part name contains this text, ignoring case,
            for example "Part 4" or "kernels". Omit it for every chapter.

    Each chapter has its slug, title, part, summary, minutes, objectives, `##`
    and `###` headings, linked lab ID, whether it has a lecture, and the
    reader's status. Bodies are omitted; call `get_chapter` for the full text.
    """
    progress = db.get_progress(_user(ctx))
    needle = (part or "").strip().lower()
    chapters = []
    for index, chapter in enumerate(curriculum.load_chapters(), start=1):
        if needle and needle not in str(chapter["part"]).lower():
            continue
        entry = _chapter_summary(chapter, progress)
        entry.update(
            index=index,
            objectives=chapter["objectives"],
            headings=[text for level, text, _ in _headings(chapter["body"]) if level > 1],
        )
        chapters.append(entry)
    return {"count": len(chapters), "chapters": chapters}


@_tool()
@_threaded
def get_chapter(ctx: Context, slug: str) -> dict[str, Any]:
    """Read one chapter: its full Markdown body and its front matter.

    Args:
        slug: The chapter's slug, such as "09-the-kv-cache".

    Returns the title, part, summary, minutes, objectives, the Markdown `body`
    (math is LaTeX between `$` or `$$`; `viz` blocks mark interactive figures),
    the previous and next slugs, the linked lab ID and title, a summary of the
    chapter's lecture or null, the reader's status, and the reader's saved
    note. Reading a chapter here doesn't mark it started.
    """
    user = _user(ctx)
    chapter = _chapter_or_error(slug)
    payload = dict(chapter)
    lab = curriculum.get_lab(chapter["lab"]) if chapter.get("lab") else None
    payload["lab_title"] = lab["title"] if lab else None
    lecture = lectures.get_lecture(slug)
    payload["lecture"] = lectures.summary(lecture) if lecture else None
    payload["status"] = db.get_progress(user).get(slug)
    payload["note"] = db.get_note(user, slug)
    return payload


def _search_documents() -> list[dict[str, Any]]:
    documents = []
    for chapter in curriculum.load_chapters():
        headings = _headings(chapter["body"])
        documents.append(
            {
                "kind": "chapter",
                "slug": chapter["slug"],
                "title": str(chapter["title"]),
                "part": chapter["part"],
                "summary": str(chapter.get("summary", "")),
                "headings": " \n".join(text for _, text, _ in headings),
                "heading_list": headings,
                "body": chapter["body"],
            }
        )
    for lab in curriculum.load_labs().values():
        body = "\n\n".join([lab["brief"], *lab["hints"]])
        documents.append(
            {
                "kind": "lab",
                "slug": lab["id"],
                "title": str(lab["title"]),
                "part": None,
                "summary": lab["brief"],
                "headings": "",
                "heading_list": [],
                "body": body,
            }
        )
    for slug, lecture in lectures.load_lectures().items():
        if curriculum.get_chapter(slug) is None:
            continue
        body = "\n\n".join(f"## {s['title']}\n{s['notes']}" for s in lecture["sections"])
        documents.append(
            {
                "kind": "lecture",
                "slug": slug,
                "title": lecture["title"],
                "part": None,
                "summary": "",
                "headings": " \n".join(s["title"] for s in lecture["sections"]),
                "heading_list": _headings(body),
                "body": body,
            }
        )
    return documents


@_tool()
@_threaded
def search(query: str, limit: int = 10) -> dict[str, Any]:
    """Search chapters, labs, and lecture narration by text, ignoring case.

    Args:
        query: Words to find, such as "block table" or "roofline". A document
            matches when it contains every word; an exact phrase match ranks
            higher, as do matches in titles and headings.
        limit: The most results to return, 1 to 50. Default 10.

    Each result has its kind ("chapter", "lab", or "lecture"), slug (a lab's
    ID for a lab), title, part, the section heading the hit falls under, and a
    short snippet around the hit. Pass a chapter or lecture slug to
    `get_chapter` or `get_lecture`, and a lab ID to `get_lab`.
    """
    phrase = " ".join(query.lower().split())
    terms = phrase.split()
    if not terms:
        raise ToolError("The query is empty.")
    limit = max(1, min(int(limit), MAX_SEARCH_RESULTS))

    results = []
    for doc in _search_documents():
        title = doc["title"].lower()
        headings = doc["headings"].lower()
        body_lower = doc["body"].lower()
        haystack = " ".join((title, doc["summary"].lower(), headings, body_lower))
        if not all(term in haystack for term in terms):
            continue
        score = 0.0
        if phrase in title:
            score += 20
        score += 6 * sum(term in title for term in terms)
        if phrase in headings:
            score += 8
        score += 2 * sum(term in headings for term in terms)
        score += min(body_lower.count(phrase), 10) * 1.5
        score += sum(min(body_lower.count(term), 10) for term in terms) * 0.2

        at = body_lower.find(phrase)
        length = len(phrase)
        if at < 0:
            at = next((body_lower.find(t) for t in terms if t in body_lower), -1)
            length = len(terms[0])
        if at >= 0:
            snippet = _snippet(doc["body"], at, length)
            section = _section(doc["heading_list"], at)
        else:
            snippet = doc["summary"]
            section = None
        results.append(
            {
                "kind": doc["kind"],
                "slug": doc["slug"],
                "title": doc["title"],
                "part": doc["part"],
                "section": section,
                "snippet": snippet,
                "score": round(score, 2),
            }
        )
    results.sort(key=lambda result: result["score"], reverse=True)
    return {"query": query, "total": len(results), "results": results[:limit]}


# --- labs -------------------------------------------------------------------


@_tool()
@_threaded
def get_lab(lab_id: str) -> dict[str, Any]:
    """Get a lab: its brief, hints, starter code, and the checks' source.

    Args:
        lab_id: The lab ID, which a chapter names in its `lab` field, such as
            "09-kv-cache".

    Returns the title, brief, hints, `starter` (the file the reader edits),
    `tests` (the source of tests.py, the checks a run applies), `fixtures` (the
    data files that tests import, by name), the metrics the lab reports, the
    timeout in seconds, the accelerator it asks for, whether it needs a GPU or
    model weights, and the chapters that link to it. The worked solution is
    withheld; see `get_solution`.
    """
    lab = _lab_or_error(lab_id)
    payload = {k: v for k, v in lab.items() if k != "solution"}
    payload["chapters"] = _chapters_for(lab_id)
    return payload


@_tool()
@_threaded
def get_solution(lab_id: str) -> dict[str, Any]:
    """Get a lab's worked solution.

    Args:
        lab_id: The lab ID.

    The solution gives the answer away. Only fetch it when the reader asks to
    see it, or to compare after they have passed the lab.
    """
    lab = _lab_or_error(lab_id)
    return {"id": lab_id, "solution": lab["solution"]}


# --- lectures, documents, and the book --------------------------------------


@_tool()
@_threaded
def list_lectures() -> dict[str, Any]:
    """List every chapter's narrated lecture, in chapter order, without sections.

    Each has its chapter slug, title, the chapter's title and part, version,
    duration in seconds, poster URL, and section count. Call `get_lecture` for
    the slides and narration.
    """
    found = lectures.load_lectures()
    out = []
    for index, chapter in enumerate(curriculum.load_chapters()):
        lecture = found.get(chapter["slug"])
        if lecture:
            out.append(
                {
                    **lectures.summary(lecture),
                    "chapter_title": chapter["title"],
                    "order": index,
                    "part": chapter["part"],
                }
            )
    return {"lectures": out}


@_tool()
@_threaded
def get_lecture(slug: str) -> dict[str, Any]:
    """Get a chapter's lecture: the video, captions, and each slide's narration.

    Args:
        slug: The chapter's slug. A lecture shares its chapter's slug.

    Each section is one slide: its title, start and end in seconds, the slide
    image URL, the narration (`notes`), and the chapter heading it teaches.
    Every URL is absolute. The video and slide images are media, which this
    server links to and doesn't return.
    """
    lecture = lectures.get_lecture(slug)
    if lecture is None or curriculum.get_chapter(slug) is None:
        raise ToolError(f"No lecture for {slug!r}. Call `list_lectures` for the slugs.")
    return lecture


@_tool()
@_threaded
def get_document(name: str, section: str | None = None) -> dict[str, Any]:
    """Read one of the course's reference documents.

    Args:
        name: "FACTS.md" for the fact sheet behind every number in the
            chapters, "STYLE.md" for the house style, or "lab_common.py" for the
            helper module the lab tests import. Case is ignored.
        section: For a long Markdown document, part of a heading's text, such
            as "chapter 14". Returns that section only. Omit it for the whole
            document, or, when it is too long, for a list of its sections.
    """
    wanted = next((key for key in DOCUMENTS if key.lower() == name.strip().lower()), None)
    if wanted is None:
        raise ToolError(f"No document {name!r}. Choose one of: {', '.join(DOCUMENTS)}.")
    folder, description = DOCUMENTS[wanted]
    path = _DOCUMENT_DIRS[folder]() / wanted
    if not path.is_file():
        raise ToolError(f"{wanted} isn't on this server.")
    text = path.read_text("utf-8")
    result: dict[str, Any] = {"name": wanted, "description": description}
    if not wanted.endswith(".md"):
        return {**result, "text": text}

    headings = _headings(text)
    if section:
        needle = section.strip().lower()
        for index, (level, heading, start) in enumerate(headings):
            if needle in heading.lower():
                end = next(
                    (off for lvl, _, off in headings[index + 1 :] if lvl <= level), len(text)
                )
                return {**result, "section": heading, "markdown": text[start:end]}
        raise ToolError(f"No section of {wanted} has a heading that contains {section!r}.")
    if len(text) <= MAX_DOCUMENT_CHARS:
        return {**result, "markdown": text}
    return {
        **result,
        "characters": len(text),
        "note": "This document is long. Call again with `section` set to part of a heading.",
        "sections": [heading for level, heading, _ in headings if level <= 2],
    }


@_tool()
@_threaded
def get_book_info() -> dict[str, Any]:
    """Get the PDF book's details: title, page count, size, build time, and commit.

    `book` is null on a server that has no PDF. The PDF itself is a download
    for the reader's browser, so this server returns its details, not its
    bytes. The book holds every chapter, lab brief, and solution, which
    `get_chapter`, `get_lab`, and `get_solution` return as text.
    """
    return {"book": course.book_info(get_settings())}


# --- GPU providers ----------------------------------------------------------


@_tool()
@_threaded
def get_providers() -> dict[str, Any]:
    """Get which GPU providers are configured, and the model and GPU the labs target.

    Each provider has its name, whether it is available, why not when it isn't,
    and whether it is preferred. `default` is the provider `run_lab` uses. This
    reads this server's configuration only and doesn't contact a provider.
    """
    settings = get_settings()
    return {
        **provider_status(),
        "model": settings.model_id,
        "small_model": settings.small_model_id,
        "gpu_type": settings.gpu_type,
    }


@_tool(open_world=True)
async def get_compute() -> dict[str, Any]:
    """Get what is running on the GPU providers right now, as the Compute page shows.

    Returns each provider's instances (pods and serverless workers, with
    whether each is active and billed), volumes, notices, and facts such as the
    balance and recent spend, plus runs this app started and never saw finish.
    This only reads: it can't stop or cancel anything. It calls the providers'
    APIs, which can take several seconds.
    """
    return await infra.snapshot()


# --- the reader's state -----------------------------------------------------


@_tool()
@_threaded
def get_progress(ctx: Context) -> dict[str, Any]:
    """Get the reader's progress: per-chapter status, totals, and recent runs.

    `chapters` lists every chapter the reader has started or finished, in
    course order, with its status ("in_progress" or "done") and when it last
    changed (Unix seconds). `summary` has the done count, the chapter total,
    and the last chapter touched. `runs` has the 10 most recent lab runs,
    newest first.
    """
    user = _user(ctx)
    rows = {row["chapter"]: row for row in db.get_progress_rows(user)}
    chapters = [
        {
            "slug": chapter["slug"],
            "title": chapter["title"],
            "status": rows[chapter["slug"]]["status"],
            "updated_at": rows[chapter["slug"]]["updated_at"],
        }
        for chapter in curriculum.chapter_list()
        if chapter["slug"] in rows
    ]
    return {
        "reader": user,
        "summary": course.progress_summary(user),
        "chapters": chapters,
        "runs": db.list_runs(user, limit=10),
    }


@_tool(writes=True)
@_threaded
def set_progress(
    ctx: Context, chapter: str, status: Literal["in_progress", "done"]
) -> dict[str, Any]:
    """Mark a chapter in progress or done for the reader.

    Args:
        chapter: The chapter's slug.
        status: "in_progress" or "done". Unlike the editor, `run_lab` never
            marks a chapter done, even when every check passes, so call this
            when the reader says they have finished.
    """
    _chapter_or_error(chapter)
    db.set_progress(_user(ctx), chapter, status)
    return {"chapter": chapter, "status": status}


@_tool()
@_threaded
def get_note(ctx: Context, chapter: str) -> dict[str, Any]:
    """Get the reader's note for a chapter.

    Args:
        chapter: The chapter's slug.

    `note` is an empty string when the reader has written none.
    """
    _chapter_or_error(chapter)
    return {"chapter": chapter, "note": db.get_note(_user(ctx), chapter)}


@_tool(writes=True)
@_threaded
def save_note(ctx: Context, chapter: str, body: str) -> dict[str, Any]:
    """Save the reader's note for a chapter.

    Args:
        chapter: The chapter's slug.
        body: The whole note, which replaces the reader's current one. The
            reader sees it the next time they open the chapter.
    """
    _chapter_or_error(chapter)
    if len(body) > MAX_NOTE_LENGTH:
        raise ToolError(f"The note is longer than {MAX_NOTE_LENGTH} characters.")
    db.save_note(_user(ctx), chapter, body)
    return {"chapter": chapter, "saved": True}


@_tool()
@_threaded
def get_draft(ctx: Context, lab_id: str) -> dict[str, Any]:
    """Get the reader's saved code for a lab.

    Args:
        lab_id: The lab ID.

    `code` is null when the reader has no draft; the editor then shows the
    starter code from `get_lab`.
    """
    _lab_or_error(lab_id)
    return {"lab_id": lab_id, "code": db.get_draft(_user(ctx), lab_id)}


@_tool(writes=True)
@_threaded
def save_draft(ctx: Context, lab_id: str, code: str) -> dict[str, Any]:
    """Save code as the reader's draft for a lab.

    Args:
        lab_id: The lab ID.
        code: The whole file, which replaces the reader's current draft. The
            reader sees it in the editor the next time they open the chapter.
    """
    _lab_or_error(lab_id)
    if len(code) > MAX_CODE_LENGTH:
        raise ToolError(f"The code is longer than {MAX_CODE_LENGTH} characters.")
    db.save_draft(_user(ctx), lab_id, code)
    return {"lab_id": lab_id, "saved": True}


@_tool()
@_threaded
def list_runs(ctx: Context, lab_id: str | None = None, limit: int = 20) -> dict[str, Any]:
    """List the reader's lab runs, newest first, without their logs.

    Args:
        lab_id: Only this lab's runs. Omit it for every lab.
        limit: The most runs to return, 1 to 100. Default 20.

    Each run has its ID, lab, provider, status ("running", "finished", "error",
    "out_of_credits", or "cancelled"), whether it passed (null when it didn't
    finish), its metrics, and start and finish times in Unix seconds. Call
    `get_run` for a run's log.
    """
    limit = max(1, min(int(limit), MAX_RUNS))
    return {"runs": db.list_runs(_user(ctx), lab_id, limit=limit)}


@_tool()
@_threaded
def get_run(ctx: Context, run_id: str) -> dict[str, Any]:
    """Get one of the reader's lab runs, with its log and error.

    Args:
        run_id: A run ID from `list_runs` or `run_lab`.

    Long logs are trimmed in the middle.
    """
    run = db.get_run(_user(ctx), run_id)
    if run is None:
        raise ToolError(f"No run with ID {run_id!r}.")
    run["log"] = _clip(run.get("log") or "", MAX_STREAM_CHARS)
    return run


@_tool(writes=True, open_world=True)
async def run_lab(ctx: Context, lab_id: str, code: str) -> dict[str, Any]:
    """Run code for a lab on the course's GPU endpoint, as the reader.

    Args:
        lab_id: The lab ID.
        code: The whole submission file, starting from the starter code.

    This does what the editor's Run button does, through the same code as
    `/api/run`: it saves the code as the reader's draft, sends a job to the
    default provider (a RunPod serverless endpoint), records the run, and waits
    for it to finish. A run costs real money. It bills the GPU time of the
    course owner's RunPod account, including the worker's cold start, for every
    lab, whether or not the lab itself needs a GPU, so don't loop or retry a
    failing lab without the reader's say-so. It can take up to the lab's
    timeout (see `get_lab`, 900 seconds by default). It doesn't mark the
    chapter done; call `set_progress` for that.

    Returns `passed`, `seconds`, the run ID and provider, each check's name,
    result, and detail, the lab's metrics, the log (long logs are trimmed in
    the middle), and any errors. `out_of_credits` is true when the provider
    rejected the job for lack of credit, and `errors` is empty when the job ran.
    """
    user = _user(ctx)
    if len(code) > MAX_CODE_LENGTH:
        raise ToolError(f"The code is longer than {MAX_CODE_LENGTH} characters.")
    try:
        run = await anyio.to_thread.run_sync(lambda: start_run(user, lab_id, code))
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    if run is None:
        raise ToolError(f"No lab with ID {lab_id!r}.")

    log: list[str] = []
    errors: list[str] = []
    result: dict[str, Any] = {}
    done: dict[str, Any] = {}
    out_of_credits = False
    async for event in run.events:
        kind = event.get("type")
        if kind == "log":
            log.append(str(event.get("line", "")))
        elif kind == "result":
            result = event
        elif kind == "error":
            errors.append(str(event.get("message", "")))
        elif kind == "out_of_credits":
            out_of_credits = True
            errors.append(f"{event.get('hint', '')} {event.get('message', '')}".strip())
        elif kind == "done":
            done = event

    checks = []
    for check in result.get("checks") or []:
        if isinstance(check, dict):
            check = {**check, "detail": _clip(str(check.get("detail", "")), MAX_CHECK_TEXT)}
        checks.append(check)
    return {
        "lab_id": lab_id,
        "run_id": run.run_id,
        "provider": run.provider,
        "passed": done.get("passed"),
        "seconds": done.get("seconds"),
        "checks": checks,
        "metrics": result.get("metrics", {}),
        "log": _clip("\n".join(log), MAX_STREAM_CHARS),
        "errors": errors,
        "out_of_credits": out_of_credits,
    }


# --- resources --------------------------------------------------------------


@mcp.resource(
    "inference://chapters/{slug}",
    name="chapter",
    description="A chapter's Markdown body. The slug comes from `list_chapters`.",
    mime_type="text/markdown",
)
@_threaded
def chapter_resource(slug: str) -> str:
    chapter = curriculum.get_chapter(slug)
    if chapter is None:
        raise ValueError(f"No chapter with slug {slug!r}.")
    return chapter["body"]


@mcp.resource(
    "inference://labs/{lab_id}",
    name="lab-starter",
    description="A lab's starter code. The ID comes from a chapter's `lab` field.",
    mime_type="text/x-python",
)
@_threaded
def lab_resource(lab_id: str) -> str:
    lab = curriculum.get_lab(lab_id)
    if lab is None:
        raise ValueError(f"No lab with ID {lab_id!r}.")
    return lab["starter"]


# --- the HTTP endpoint ------------------------------------------------------

_manager: StreamableHTTPSessionManager | None = None


@asynccontextmanager
async def lifespan() -> AsyncIterator[None]:
    """Run the MCP session manager for the life of the app.

    A session manager runs only once, so each app start makes a new one. That
    lets tests start the app more than once in one process.
    """
    global _manager
    manager = StreamableHTTPSessionManager(
        app=mcp._mcp_server,
        json_response=True,
        stateless=True,
        security_settings=mcp.settings.transport_security,
    )
    async with manager.run():
        _manager = manager
        try:
            yield
        finally:
            _manager = None


def _unauthorized(error: str | None, description: str) -> JSONResponse:
    challenge = 'Bearer realm="learn-inference-mcp"'
    if error:
        challenge += f', error="{error}"'
    return JSONResponse(
        {"error": error or "unauthorized", "error_description": description},
        status_code=401,
        headers={"WWW-Authenticate": challenge},
    )


class _Endpoint:
    """The ASGI app at `/mcp`: check the bearer token, then hand over to MCP.

    A class rather than a function, because Starlette wraps a plain function
    route as a request handler instead of mounting it as an ASGI app.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await _serve(scope, receive, send)


def _hub_user(headers: Headers) -> str | None:
    """The reader the MCP hub vouches for, or None without a valid hub secret.

    The hub sends its shared secret and the reader's Authelia username. The
    username maps through SSO_USERS, as a signed-in browser session does.
    Caddy strips both headers from public requests to /mcp.
    """
    settings = get_settings()
    secret = settings.mcp_hub_secret
    sent = headers.get("x-mcp-hub-secret", "")
    name = headers.get("x-mcp-hub-user", "").strip()
    if not secret or not sent or not name:
        return None
    if not hmac.compare_digest(sent.encode(), secret.encode()):
        return None
    return settings.sso_names.get(name.lower(), name)


async def _serve(scope: Scope, receive: Receive, send: Send) -> None:
    headers = Headers(scope=scope)
    hub_user = _hub_user(headers)
    if hub_user is not None:
        await _dispatch(hub_user, scope, receive, send)
        return
    scheme, _, token = headers.get("authorization", "").partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        response = _unauthorized(None, "Send an MCP token as 'Authorization: Bearer <token>'.")
        await response(scope, receive, send)
        return
    user = await anyio.to_thread.run_sync(user_for_token, token)
    if user is None:
        response = _unauthorized("invalid_token", "The token is unknown or revoked.")
        await response(scope, receive, send)
        return
    await _dispatch(user, scope, receive, send)


async def _dispatch(user: str, scope: Scope, receive: Receive, send: Send) -> None:
    if _manager is None:
        response = JSONResponse({"error": "The MCP server isn't running."}, status_code=503)
        await response(scope, receive, send)
        return
    await _manager.handle_request({**scope, USER_SCOPE_KEY: user}, receive, send)


endpoint = _Endpoint()
