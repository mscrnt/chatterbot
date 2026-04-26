"""FastAPI dashboard.

Server-rendered with Jinja2 + HTMX. Mobile-first Tailwind. Streamer-only —
nothing here writes to Twitch chat. The "Ask Qwen" RAG output renders in the
streamer's browser only.

Routes
  GET  /                              chatters list (search, sort, paginate)
  GET  /chatters                      chatters list partial (HTMX target)
  GET  /users/{twitch_id}             user detail page
  GET  /users/{twitch_id}/messages    paginated message partial
  GET  /users/{twitch_id}/ask         SSE stream of LLM tokens for RAG
  POST /users/{twitch_id}/forget      delete all rows for user
  POST /users/{twitch_id}/opt-out     toggle opt-out
  PATCH /notes/{note_id}              update note text
  DELETE /notes/{note_id}             delete note
  GET  /topics                        topics page (auto-refresh via hx-trigger)
  GET  /events                        events feed (filter by type)
  GET  /static/...                    static assets
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    Form,
    HTTPException,
    Path as PathParam,
    Query,
    Request,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import (
    EDITABLE_SETTING_KEYS,
    SECRET_SETTING_KEYS,
    Settings,
    get_settings,
    get_settings_with_sources,
)
from ..llm.ollama_client import OllamaClient
from ..repo import ChatterRepo
from .auth import make_auth_dependency
from .rag import answer_for_user

logger = logging.getLogger(__name__)


WEB_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(WEB_DIR / "templates"))


PAGE_SIZE = 50
MESSAGE_PAGE_SIZE = 50


def _short_ts(ts: str | None) -> str:
    if not ts:
        return ""
    return ts.replace("T", " ")[:16]


TEMPLATES.env.filters["short_ts"] = _short_ts


def _from_json(s: str | None):
    if not s:
        return None
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return None


TEMPLATES.env.filters["from_json"] = _from_json


def create_app(repo: ChatterRepo, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    llm = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        embed_model=settings.ollama_embed_model,
    )

    auth_dep = make_auth_dependency(settings)
    deps = [Depends(auth_dep)] if settings.dashboard_basic_auth_enabled else []
    app = FastAPI(
        title="chatterbot dashboard",
        docs_url=None,
        redoc_url=None,
        dependencies=deps,
    )
    app.mount(
        "/static",
        StaticFiles(directory=str(WEB_DIR / "static")),
        name="static",
    )

    # ---------------- chatters list ----------------

    @app.get("/", response_class=HTMLResponse)
    async def index(
        request: Request,
        q: str = Query(""),
        sort: str = Query("last_seen"),
        page: int = Query(1, ge=1),
    ):
        return _render_chatters(request, q, sort, page, partial=False)

    @app.get("/chatters", response_class=HTMLResponse)
    async def chatters_partial(
        request: Request,
        q: str = Query(""),
        sort: str = Query("last_seen"),
        page: int = Query(1, ge=1),
    ):
        return _render_chatters(request, q, sort, page, partial=True)

    def _render_chatters(
        request: Request, q: str, sort: str, page: int, *, partial: bool
    ) -> HTMLResponse:
        offset = (page - 1) * PAGE_SIZE
        rows = repo.list_chatters(query=q, sort=sort, limit=PAGE_SIZE, offset=offset)
        total = repo.count_chatters(query=q)
        pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        ctx = {
            "rows": rows,
            "q": q,
            "sort": sort,
            "page": page,
            "pages": pages,
            "total": total,
        }
        tpl = "partials/chatters_table.html" if partial else "chatters.html"
        return TEMPLATES.TemplateResponse(request, tpl, ctx)

    # ---------------- user detail ----------------

    @app.get("/users/{twitch_id}", response_class=HTMLResponse)
    async def user_detail(
        request: Request,
        twitch_id: str = PathParam(..., min_length=1),
    ):
        user = repo.get_user(twitch_id)
        if not user:
            raise HTTPException(404, "user not found")
        notes = repo.get_notes(twitch_id)
        events = repo.get_user_events(twitch_id, limit=50)
        summary = repo.get_user_event_summary(twitch_id)
        messages = repo.get_messages(twitch_id, limit=MESSAGE_PAGE_SIZE)
        msg_total = repo.count_messages(twitch_id)
        aliases = repo.get_user_aliases(twitch_id)
        prior_aliases = [a for a in aliases if a.name != user.name]
        return TEMPLATES.TemplateResponse(
            request,
            "user.html",
            {
                "user": user,
                "notes": notes,
                "events": events,
                "summary": summary,
                "messages": messages,
                "msg_total": msg_total,
                "msg_page": 1,
                "msg_pages": max(1, (msg_total + MESSAGE_PAGE_SIZE - 1) // MESSAGE_PAGE_SIZE),
                "prior_aliases": prior_aliases,
            },
        )

    @app.get("/users/{twitch_id}/messages", response_class=HTMLResponse)
    async def user_messages_partial(
        request: Request,
        twitch_id: str,
        q: str = Query(""),
        page: int = Query(1, ge=1),
    ):
        offset = (page - 1) * MESSAGE_PAGE_SIZE
        messages = repo.get_messages(
            twitch_id, query=q, limit=MESSAGE_PAGE_SIZE, offset=offset
        )
        total = repo.count_messages(twitch_id, query=q)
        pages = max(1, (total + MESSAGE_PAGE_SIZE - 1) // MESSAGE_PAGE_SIZE)
        return TEMPLATES.TemplateResponse(
            request,
            "partials/messages.html",
            {
                "messages": messages,
                "q": q,
                "msg_page": page,
                "msg_pages": pages,
                "msg_total": total,
                "user": {"twitch_id": twitch_id},
            },
        )

    # ---------------- ask qwen (SSE) ----------------

    @app.get("/users/{twitch_id}/ask")
    async def user_ask(
        twitch_id: str,
        q: str = Query(..., min_length=1, max_length=1000),
    ):
        async def stream():
            try:
                rag = await answer_for_user(repo, llm, twitch_id, q)
            except Exception:
                logger.exception("rag setup failed")
                yield _sse("error", "internal error during retrieval")
                yield _sse("done", "")
                return

            yield _sse(
                "citations",
                json.dumps(
                    {
                        "notes": [
                            {"id": n.id, "ts": n.ts, "text": n.text} for n in rag.notes
                        ],
                        "messages": [
                            {
                                "id": m.id,
                                "ts": m.ts,
                                "content": m.content[:500],
                            }
                            for m in rag.messages
                        ],
                    }
                ),
            )
            try:
                async for chunk in rag.stream:
                    yield _sse("chunk", chunk)
            except Exception:
                logger.exception("rag stream failed")
                yield _sse("error", "stream interrupted")
            yield _sse("done", "")

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---------------- mutating actions ----------------

    @app.post("/users/{twitch_id}/forget")
    async def forget(twitch_id: str):
        repo.forget_user(twitch_id)
        return RedirectResponse(url="/", status_code=303)

    @app.post("/users/{twitch_id}/opt-out", response_class=HTMLResponse)
    async def toggle_opt_out(
        request: Request,
        twitch_id: str,
    ):
        user = repo.get_user(twitch_id)
        if not user:
            raise HTTPException(404, "user not found")
        repo.set_opt_out(twitch_id, not user.opt_out)
        user = repo.get_user(twitch_id)
        return TEMPLATES.TemplateResponse(
            request,
            "partials/opt_out_badge.html",
            {"user": user},
        )

    @app.post("/users/{twitch_id}/notes", response_class=HTMLResponse)
    async def add_user_note(
        request: Request,
        twitch_id: str,
        text: Annotated[str, Form()],
    ):
        text = text.strip()
        if not text:
            # Empty submissions: just re-render the current list, no insert.
            user = repo.get_user(twitch_id)
            notes = repo.get_notes(twitch_id) if user else []
            return TEMPLATES.TemplateResponse(
                request, "partials/notes_list.html", {"notes": notes}
            )
        user = repo.get_user(twitch_id)
        if not user:
            raise HTTPException(404, "user not found")
        # Embed for RAG; non-fatal if Ollama is unreachable.
        embedding: list[float] | None
        try:
            embedding = await llm.embed(text[:500])
        except Exception:
            logger.exception("embed failed for manual note; storing without vector")
            embedding = None
        repo.add_note(twitch_id, text[:500], embedding)
        notes = repo.get_notes(twitch_id)
        return TEMPLATES.TemplateResponse(
            request, "partials/notes_list.html", {"notes": notes}
        )

    @app.patch("/notes/{note_id}", response_class=HTMLResponse)
    async def patch_note(
        request: Request,
        note_id: int,
        text: Annotated[str, Form()],
    ):
        repo.update_note(note_id, text.strip())
        return TEMPLATES.TemplateResponse(
            request,
            "partials/note.html",
            {"note": _find_note(repo, note_id)},
        )

    @app.delete("/notes/{note_id}")
    async def delete_note(note_id: int):
        repo.delete_note(note_id)
        return Response(status_code=200)

    # ---------------- settings (Twitch + StreamElements creds) ----------------
    # Plaintext credentials in SQLite. Fine for a single-user local tool bound
    # to 127.0.0.1; flagged to the user on the page.

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request, saved: int = Query(0)):
        live, db_sources = get_settings_with_sources()
        rows = []
        for key in EDITABLE_SETTING_KEYS:
            value = getattr(live, key)
            is_secret = key in SECRET_SETTING_KEYS
            is_bool = isinstance(value, bool)
            rows.append(
                {
                    "key": key,
                    "label": key.replace("_", " "),
                    "value": value,
                    "is_secret": is_secret,
                    "is_bool": is_bool,
                    "is_set": bool(value) if not is_bool else True,
                    "source": "db" if key in db_sources else "env",
                }
            )
        return TEMPLATES.TemplateResponse(
            request,
            "settings.html",
            {"rows": rows, "saved": bool(saved)},
        )

    @app.post("/settings")
    async def settings_save(request: Request):
        form = await request.form()
        for key in EDITABLE_SETTING_KEYS:
            if key == "streamelements_enabled":
                checked = form.get(key) is not None
                repo.set_app_setting(key, "true" if checked else "false")
                continue
            submitted = form.get(key)
            if submitted is None:
                continue
            submitted = str(submitted)
            if key in SECRET_SETTING_KEYS and submitted.strip() == "":
                # Blank submission for a secret = preserve existing.
                continue
            if submitted.strip() == "":
                repo.delete_app_setting(key)
            else:
                repo.set_app_setting(key, submitted.strip())
        return RedirectResponse(url="/settings?saved=1", status_code=303)

    # ---------------- diagnostic bundle ----------------

    @app.get("/diagnose")
    async def diagnose(with_recent_activity: int = Query(0)):
        """Build a privacy-safe .cbreport zip and stream it back as a download.
        See diagnose.py for what's in / out of the bundle."""
        from ..diagnose import build_diagnostic_bundle, default_bundle_filename
        import tempfile
        from fastapi import BackgroundTasks

        fname = default_bundle_filename()
        # Build into a tempfile we own, then hand it to FileResponse with a
        # cleanup task to delete it after the download finishes.
        tmp = tempfile.NamedTemporaryFile(
            prefix="chatterbot-diagnose-", suffix=".cbreport", delete=False
        )
        tmp.close()
        from pathlib import Path as _P
        out_path = _P(tmp.name)
        await asyncio.to_thread(
            build_diagnostic_bundle,
            out_path, settings,
            with_recent_activity=bool(with_recent_activity),
        )
        bg = BackgroundTasks()
        bg.add_task(lambda p=out_path: p.unlink(missing_ok=True))
        return FileResponse(
            out_path,
            filename=fname,
            media_type="application/octet-stream",
            background=bg,
        )

    # ---------------- modal partials ----------------
    # Each returns a self-contained HTML fragment (backdrop + shell + body)
    # swapped into #modal-root. Dismissal happens client-side by swapping the
    # root to empty.

    @app.get("/modals/forget/{twitch_id}", response_class=HTMLResponse)
    async def modal_forget(request: Request, twitch_id: str):
        user = repo.get_user(twitch_id)
        if not user:
            raise HTTPException(404, "user not found")
        return TEMPLATES.TemplateResponse(
            request,
            "modals/_forget.html",
            {
                "user": user,
                "msg_count": repo.count_messages(twitch_id),
                "note_count": repo.note_count(twitch_id),
                "event_count": repo.count_user_events(twitch_id),
                "summary": repo.get_user_event_summary(twitch_id),
            },
        )

    @app.get("/modals/note/{note_id}/edit", response_class=HTMLResponse)
    async def modal_edit_note(request: Request, note_id: int):
        note = repo.get_note(note_id)
        if not note:
            raise HTTPException(404, "note not found")
        return TEMPLATES.TemplateResponse(
            request, "modals/_edit_note.html", {"note": note}
        )

    @app.get("/modals/note/{note_id}/delete", response_class=HTMLResponse)
    async def modal_delete_note(request: Request, note_id: int):
        note = repo.get_note(note_id)
        if not note:
            raise HTTPException(404, "note not found")
        return TEMPLATES.TemplateResponse(
            request, "modals/_delete_note.html", {"note": note}
        )

    @app.get("/modals/event/{event_id}", response_class=HTMLResponse)
    async def modal_event(request: Request, event_id: int):
        event = repo.get_event(event_id)
        if not event:
            raise HTTPException(404, "event not found")
        raw = getattr(event, "raw_json", None)
        try:
            raw_pretty = json.dumps(json.loads(raw), indent=2) if raw else "(no payload)"
        except Exception:
            raw_pretty = raw or "(unparseable)"
        return TEMPLATES.TemplateResponse(
            request,
            "modals/_event_detail.html",
            {"event": event, "raw_pretty": raw_pretty},
        )

    @app.get("/modals/settings", response_class=HTMLResponse)
    async def modal_settings_env(request: Request):
        return TEMPLATES.TemplateResponse(
            request,
            "modals/_settings_env.html",
            {"s": settings, "opt_out_users": repo.list_opt_out_users()},
        )

    @app.get("/modals/shortcuts", response_class=HTMLResponse)
    async def modal_shortcuts(request: Request):
        return TEMPLATES.TemplateResponse(request, "modals/_shortcuts.html", {})

    @app.get("/modals/topic/{snapshot_id}/{topic_index}", response_class=HTMLResponse)
    async def modal_topic(request: Request, snapshot_id: int, topic_index: int):
        from .topic_rag import _parse_range  # local import to avoid cycle at boot
        from ..llm.schemas import TopicsResponse

        snapshot = repo.get_topic_snapshot(snapshot_id)
        if not snapshot or not snapshot.topics_json:
            raise HTTPException(404, "snapshot not found or pre-dates structured topics")
        try:
            parsed = TopicsResponse.model_validate_json(snapshot.topics_json)
        except Exception:
            raise HTTPException(500, "snapshot has unparseable topics_json")
        if not (0 <= topic_index < len(parsed.topics)):
            raise HTTPException(404, "topic index out of range")
        entry = parsed.topics[topic_index]

        # Resolve each cited driver to a profile link if we know them. Renames
        # are followed via aliases.
        driver_links = []
        for name in entry.drivers:
            user = repo.find_user_by_alias_or_name(name)
            driver_links.append({"name": name, "user_id": user.twitch_id if user else None})

        first_id, last_id = _parse_range(snapshot.message_id_range)
        messages = repo.messages_in_id_range_for_names(
            first_id, last_id, entry.drivers, limit=80
        )

        return TEMPLATES.TemplateResponse(
            request,
            "modals/_topic.html",
            {
                "snapshot": snapshot,
                "topic_index": topic_index,
                "topic_title": entry.topic,
                "driver_links": driver_links,
                "messages": messages,
            },
        )

    @app.get("/topics/{snapshot_id}/{topic_index}/explain")
    async def topic_explain(snapshot_id: int, topic_index: int):
        from .topic_rag import explain_topic

        async def stream():
            try:
                ctx = await explain_topic(repo, llm, snapshot_id, topic_index)
            except Exception:
                logger.exception("topic explain setup failed")
                yield _sse("error", "internal error during topic retrieval")
                yield _sse("done", "")
                return
            if ctx is None:
                yield _sse("error", "topic not found or snapshot pre-dates structured topics")
                yield _sse("done", "")
                return
            try:
                async for chunk in ctx.stream:
                    yield _sse("chunk", chunk)
            except Exception:
                logger.exception("topic explain stream failed")
                yield _sse("error", "stream interrupted")
            yield _sse("done", "")

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---------------- live chat (widget + full page) ----------------

    @app.get("/live", response_class=HTMLResponse)
    async def live(
        request: Request,
        limit: int = Query(30, ge=1, le=500),
        partial: int = Query(0),
    ):
        messages = repo.recent_global_messages(limit=limit)
        ctx = {"messages": messages, "limit": limit}
        tpl = "partials/live_rows.html" if partial else "live.html"
        return TEMPLATES.TemplateResponse(request, tpl, ctx)

    # ---------------- topics ----------------

    @app.get("/topics", response_class=HTMLResponse)
    async def topics(
        request: Request,
        partial: int = Query(0),
    ):
        snapshots = repo.list_topic_snapshots(limit=30)
        ctx = {"snapshots": snapshots, "settings": settings}
        tpl = "partials/topics_list.html" if partial else "topics.html"
        return TEMPLATES.TemplateResponse(request, tpl, ctx)

    # ---------------- events ----------------

    @app.get("/events", response_class=HTMLResponse)
    async def events(
        request: Request,
        type: str = Query(""),
        page: int = Query(1, ge=1),
        partial: int = Query(0),
    ):
        type_filter = type or None
        offset = (page - 1) * PAGE_SIZE
        events = repo.list_events(type_filter=type_filter, limit=PAGE_SIZE, offset=offset)
        ctx = {
            "events": events,
            "type_filter": type_filter or "",
            "page": page,
        }
        tpl = "partials/events_list.html" if partial else "events.html"
        return TEMPLATES.TemplateResponse(request, tpl, ctx)

    return app


def _sse(event: str, data: str) -> str:
    # SSE multi-line data is split by \n and reassembled by the client; keep
    # each chunk on one logical message.
    payload = data.replace("\r\n", "\n").replace("\n", "\\n")
    return f"event: {event}\ndata: {payload}\n\n"


def _find_note(repo: ChatterRepo, note_id: int):
    note = repo.get_note(note_id)
    if not note:
        raise HTTPException(404, "note not found")
    return note
