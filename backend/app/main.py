"""learn-inference — HTTP API and static file server.

Private API routes and the PDF book require a trusted Authelia identity from
the reverse proxy. `/api/health` and `/api/me` are the only public API routes.
The MCP server at `/mcp` is the exception to the private routes: it ignores the
proxy identity and takes a bearer token. Everything else is the single-page app.
"""

import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import course, curriculum, db, infra, lectures, mcp_server, mcp_tokens
from .auth import current_user, sso_user
from .config import Settings, get_settings
from .providers import ProviderError, provider_status
from .runs import start_run


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    db.init_db()
    # Read the lecture manifests now, so a bad one is logged at startup
    # rather than on the first chapter someone opens.
    lectures.load_lectures()
    async with mcp_server.lifespan():
        yield


app = FastAPI(title="learn-inference", docs_url=None, redoc_url=None, lifespan=lifespan)


# The hub at this origin reads the progress summary from the browser.
HUB_ORIGIN = "https://courses.ifkash.dev"
SUMMARY_PATH = "/api/progress/summary"


@app.middleware("http")
async def hub_cors(request: Request, call_next: Any) -> Any:
    """Let the hub read `/api/progress/summary`, and no other path.

    This is middleware rather than `CORSMiddleware` so that it applies to one
    path only, and so that the 401 from `current_user` carries the headers too.
    """
    response = await call_next(request)
    if request.url.path == SUMMARY_PATH:
        response.headers.append("Vary", "Origin")
        if request.headers.get("origin") == HUB_ORIGIN:
            response.headers["Access-Control-Allow-Origin"] = HUB_ORIGIN
            response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


class ProgressBody(BaseModel):
    chapter: str
    status: str = Field(pattern="^(in_progress|done)$")


class DraftBody(BaseModel):
    lab: str
    code: str


class NoteBody(BaseModel):
    chapter: str
    body: str


class TokenBody(BaseModel):
    name: str = Field(min_length=1, max_length=mcp_tokens.MAX_NAME_LENGTH)


class RunBody(BaseModel):
    lab: str
    code: str
    provider: str | None = None


class ActionBody(BaseModel):
    provider: str
    action: str
    target: str = ""


@app.get("/api/me")
def me(request: Request, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return {
        "name": sso_user(request, settings),
        "model": settings.model_id,
        "small_model": settings.small_model_id,
        "gpu": settings.gpu_type,
    }


# --- content ----------------------------------------------------------------


@app.get("/api/chapters")
def chapters(user: str = Depends(current_user)) -> dict[str, Any]:
    return {"chapters": curriculum.chapter_list(), "progress": db.get_progress(user)}


@app.get("/api/chapters/{slug}")
def chapter(slug: str, user: str = Depends(current_user)) -> dict[str, Any]:
    found = curriculum.get_chapter(slug)
    if not found:
        raise HTTPException(status_code=404, detail="No such chapter")
    payload = dict(found)
    payload["note"] = db.get_note(user, slug)
    payload["lecture"] = lectures.get_lecture(slug)
    if found.get("lab"):
        lab = curriculum.get_lab(found["lab"])
        if lab:
            public = curriculum.public_lab(lab)
            public["draft"] = db.get_draft(user, lab["id"])
            payload["lab_detail"] = public
    return payload


@app.get("/api/lectures")
def lecture_list(user: str = Depends(current_user)) -> dict[str, Any]:
    """Every chapter's lecture, in chapter order, without its sections."""
    found = lectures.load_lectures()
    out: list[dict[str, Any]] = []
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


@app.get("/api/lectures/{slug}")
def one_lecture(slug: str, user: str = Depends(current_user)) -> dict[str, Any]:
    found = lectures.get_lecture(slug)
    if found is None or curriculum.get_chapter(slug) is None:
        raise HTTPException(status_code=404, detail="No such lecture")
    return found


@app.get("/api/labs/{lab_id}")
def lab(lab_id: str, user: str = Depends(current_user)) -> dict[str, Any]:
    found = curriculum.get_lab(lab_id)
    if not found:
        raise HTTPException(status_code=404, detail="No such lab")
    payload = curriculum.public_lab(found)
    payload["draft"] = db.get_draft(user, lab_id)
    payload["runs"] = db.list_runs(user, lab_id, limit=20)
    return payload


@app.get("/api/labs/{lab_id}/solution")
def lab_solution(lab_id: str, user: str = Depends(current_user)) -> dict[str, str]:
    """The worked solution. Deliberately behind its own click in the UI."""
    found = curriculum.get_lab(lab_id)
    if not found:
        raise HTTPException(status_code=404, detail="No such lab")
    return {"solution": found["solution"]}


@app.get("/api/book")
def book(
    user: str = Depends(current_user), settings: Settings = Depends(get_settings)
) -> dict[str, Any]:
    """The whole course as one PDF, or None when this server has no PDF."""
    return {"book": course.book_info(settings)}


@app.get("/book/{name}")
def book_file(
    name: str, user: str = Depends(current_user), settings: Settings = Depends(get_settings)
) -> FileResponse:
    """The PDF, sent as a download. Only the file the manifest names is served."""
    info = course.book_info(settings)
    if info is None or info["url"] != f"/book/{name}":
        raise HTTPException(status_code=404, detail="No such book")
    return FileResponse(settings.book_dir / name, media_type="application/pdf", filename=name)


# --- user state -------------------------------------------------------------


@app.get("/api/progress/summary")
def progress_summary(user: str = Depends(current_user)) -> dict[str, Any]:
    """How far the user is through the course, for the hub."""
    return course.progress_summary(user)


@app.post("/api/progress")
def set_progress(body: ProgressBody, user: str = Depends(current_user)) -> dict[str, bool]:
    db.set_progress(user, body.chapter, body.status)
    return {"ok": True}


@app.post("/api/drafts")
def save_draft(body: DraftBody, user: str = Depends(current_user)) -> dict[str, bool]:
    db.save_draft(user, body.lab, body.code)
    return {"ok": True}


@app.post("/api/notes")
def save_note(body: NoteBody, user: str = Depends(current_user)) -> dict[str, bool]:
    db.save_note(user, body.chapter, body.body)
    return {"ok": True}


@app.get("/api/runs")
def runs(
    lab: str | None = None, user: str = Depends(current_user)
) -> dict[str, Any]:
    return {"runs": db.list_runs(user, lab)}


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str, user: str = Depends(current_user)) -> dict[str, Any]:
    found = db.get_run(user, run_id)
    if not found:
        raise HTTPException(status_code=404, detail="No such run")
    return found


@app.get("/api/providers")
def providers(user: str = Depends(current_user)) -> dict[str, Any]:
    return provider_status()


# --- compute panel ----------------------------------------------------------


@app.get("/api/infra")
async def infra_snapshot(user: str = Depends(current_user)) -> dict[str, Any]:
    """What is running on every configured provider right now."""
    return await infra.snapshot()


@app.post("/api/infra/action")
async def infra_action(
    body: ActionBody, user: str = Depends(current_user)
) -> dict[str, Any]:
    """Stop a container, pod, or job. Never deletes an endpoint or a volume."""
    try:
        return await infra.act(body.provider, body.action, body.target)
    except ProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - the provider's message is the answer
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/infra/volume")
async def infra_volume(
    provider: str,
    name: str,
    path: str = "/",
    user: str = Depends(current_user),
) -> dict[str, Any]:
    """List one directory of a provider volume, for the weight-cache browser."""
    try:
        return await infra.browse(provider, name, path)
    except ProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# --- lab execution ----------------------------------------------------------


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/run")
async def run_lab(body: RunBody, user: str = Depends(current_user)) -> StreamingResponse:
    started = start_run(user, body.lab, body.code, body.provider)
    if started is None:
        raise HTTPException(status_code=404, detail="No such lab")

    async def stream() -> Any:
        async for event in started.events:
            yield _sse(event)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- MCP --------------------------------------------------------------------


def _token_view(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in ("id", "name", "prefix", "created_at", "last_used_at")}


@app.get("/api/mcp/tokens")
def list_mcp_tokens(
    request: Request,
    user: str = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """The MCP endpoint's URL and the caller's active tokens, without secrets."""
    url = settings.mcp_public_url or str(request.base_url).rstrip("/") + "/mcp"
    return {"url": url, "tokens": [_token_view(t) for t in db.list_tokens(user)]}


@app.post("/api/mcp/tokens")
def create_mcp_token(body: TokenBody, user: str = Depends(current_user)) -> dict[str, Any]:
    """Create a token for the caller. The response is the only copy of the secret."""
    try:
        record = mcp_tokens.create_token(user, body.name)
    except mcp_tokens.TokenLimitError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {key: record[key] for key in ("id", "name", "prefix", "created_at", "token")}


@app.delete("/api/mcp/tokens/{token_id}")
def revoke_mcp_token(token_id: int, user: str = Depends(current_user)) -> dict[str, Any]:
    if not db.revoke_token(token_id, user=user):
        raise HTTPException(status_code=404, detail="No such token")
    return {"revoked": True}


# Both spellings are routes of their own, so neither redirects: MCP clients
# don't follow a redirect on POST. They are registered before the SPA catch-all.
app.router.add_route("/mcp", mcp_server.endpoint, include_in_schema=False)
app.router.add_route("/mcp/", mcp_server.endpoint, include_in_schema=False)


# --- health & static --------------------------------------------------------


@app.get("/api/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return {
        "ok": True,
        "chapters": len(curriculum.chapter_list()),
        # Set at build time from the commit being deployed.
        "commit": os.environ.get("GIT_SHA", "unknown"),
        "book_pages": (course.book_info(settings) or {}).get("pages", 0),
    }


def _mount_frontend() -> None:
    settings = get_settings()
    dist = settings.frontend_dist
    if not dist.is_dir():
        return

    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str) -> Any:
        # .well-known/ too: an MCP client probes it for OAuth metadata, and an
        # HTML page with 200 there reads as a broken OAuth server.
        if full_path.startswith(("api/", "mcp/", ".well-known/")):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        candidate = dist / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")


_mount_frontend()
