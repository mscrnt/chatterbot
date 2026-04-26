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
from ..insights import InsightsService
from ..llm.ollama_client import OllamaClient
from ..obs import OBSStatusService
from ..repo import ChatterRepo
from ..twitch import TwitchService
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

    # Expose runtime settings flags to every template (nav uses these).
    TEMPLATES.env.globals["mod_mode_enabled"] = bool(settings.mod_mode_enabled)
    # Re-read on each request so toggling in /settings takes effect after the
    # next page navigation without a process restart.
    def _live_widget_enabled() -> bool:
        try:
            return bool(get_settings().live_widget_enabled)
        except Exception:
            return True
    TEMPLATES.env.globals["live_widget_enabled"] = _live_widget_enabled

    # Newcomers nav pill — count of first-timers in the last 24h NEWER than
    # the streamer's last acknowledgment. Visiting /insights acks. Pill goes
    # away after the visit until a fresh chatter shows up.
    def _newcomers_today_count() -> int:
        try:
            return repo.count_first_timers_unacked()
        except Exception:
            return 0
    TEMPLATES.env.globals["newcomers_today_count"] = _newcomers_today_count

    # OBS live-status snapshot for the nav pill. Returns the cached status —
    # the background poller refreshes it every ~10s.
    def _obs_status():
        return obs_status.status
    TEMPLATES.env.globals["obs_status"] = _obs_status

    # Twitch live-status snapshot — viewer count + thumbnail for the nav.
    def _twitch_status():
        return twitch_status.status
    TEMPLATES.env.globals["twitch_status"] = _twitch_status

    # Helper to materialize the Twitch thumbnail URL with width/height.
    TEMPLATES.env.globals["twitch_thumbnail"] = TwitchService.thumbnail_url

    # Fired-but-not-dismissed reminders count for the nav pill.
    def _fired_reminders_count() -> int:
        try:
            return repo.count_fired_reminders()
        except Exception:
            return 0
    TEMPLATES.env.globals["fired_reminders_count"] = _fired_reminders_count

    # Cutoff used by the activity-badge component to label first-timers
    # (first_seen >= now - 24h). Recomputed per render so it stays accurate.
    from datetime import datetime, timedelta, timezone
    def _day_floor_iso() -> str:
        return (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(timespec="seconds")
    TEMPLATES.env.globals["day_floor_iso"] = _day_floor_iso

    auth_dep = make_auth_dependency(settings)
    deps = [Depends(auth_dep)] if settings.dashboard_basic_auth_enabled else []

    insights = InsightsService(repo, llm, settings)
    obs_status = OBSStatusService(settings)
    # The Helix poller consults OBS so it can pause itself while we're
    # not actually streaming. Same defaults-on-uncertainty contract.
    twitch_status = TwitchService(settings, obs=obs_status)
    _bg_tasks: set[asyncio.Task] = set()

    async def _lifespan(app):
        # Background talking-points refresher. Runs inside the dashboard
        # process so the page render is always served from cache.
        _bg_tasks.add(
            asyncio.create_task(insights.refresh_loop(), name="insights_refresh")
        )
        # Optional OBS poller; no-op when OBS_ENABLED=false.
        _bg_tasks.add(
            asyncio.create_task(obs_status.poll_loop(), name="obs_poll")
        )
        # Twitch Helix poller (viewer count + thumbnail). Auto-disables if
        # token validate fails or creds are missing.
        _bg_tasks.add(
            asyncio.create_task(twitch_status.poll_loop(), name="twitch_poll")
        )
        try:
            yield
        finally:
            for t in _bg_tasks:
                t.cancel()
            await asyncio.gather(*_bg_tasks, return_exceptions=True)

    app = FastAPI(
        title="chatterbot dashboard",
        docs_url=None,
        redoc_url=None,
        dependencies=deps,
        lifespan=_lifespan,
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
        notes_with_sources = repo.get_notes_with_sources(twitch_id)
        events = repo.get_user_events(twitch_id, limit=50)
        summary = repo.get_user_event_summary(twitch_id)
        messages = repo.get_messages(twitch_id, limit=MESSAGE_PAGE_SIZE)
        msg_total = repo.count_messages(twitch_id)
        aliases = repo.get_user_aliases(twitch_id)
        prior_aliases = [a for a in aliases if a.name != user.name]
        incidents = (
            repo.get_user_incidents(twitch_id, limit=15)
            if settings.mod_mode_enabled
            else []
        )
        reminders = repo.get_reminders_for_user(twitch_id)
        msg_stats = repo.user_message_stats(twitch_id)
        merged_children = repo.list_merged_children(twitch_id)
        merged_parent = (
            repo.get_user(user.merged_into) if user.merged_into else None
        )
        return TEMPLATES.TemplateResponse(
            request,
            "user.html",
            {
                "user": user,
                "notes_with_sources": notes_with_sources,
                "events": events,
                "summary": summary,
                "messages": messages,
                "msg_total": msg_total,
                "msg_page": 1,
                "msg_pages": max(1, (msg_total + MESSAGE_PAGE_SIZE - 1) // MESSAGE_PAGE_SIZE),
                "prior_aliases": prior_aliases,
                "incidents": incidents,
                "reminders": reminders,
                "msg_stats": msg_stats,
                "merged_children": merged_children,
                "merged_parent": merged_parent,
                "mod_mode_enabled": settings.mod_mode_enabled,
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
            nws = repo.get_notes_with_sources(twitch_id) if repo.get_user(twitch_id) else []
            return TEMPLATES.TemplateResponse(
                request, "partials/notes_list.html", {"notes_with_sources": nws}
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
        # Manual notes have no source_message_ids — they show "manual / unlinked"
        # in the template instead of an expandable source list.
        repo.add_note(twitch_id, text[:500], embedding)
        nws = repo.get_notes_with_sources(twitch_id)
        return TEMPLATES.TemplateResponse(
            request, "partials/notes_list.html", {"notes_with_sources": nws}
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

    # Settings groups — drives the section layout on /settings. Each entry is
    # (group_id, title, icon_classes, blurb, [keys]). icon_classes is the
    # full Font Awesome class string ("fa-solid fa-tv" or "fa-brands fa-youtube").
    _SETTINGS_GROUPS: tuple[tuple[str, str, str, str, tuple[str, ...]], ...] = (
        (
            "twitch", "Twitch", "fa-brands fa-twitch",
            "Bot identity + chat connection. Restart bot after changes.",
            (
                "twitch_bot_nick", "twitch_oauth_token", "twitch_channel",
                "twitch_client_id", "twitch_client_secret",
            ),
        ),
        (
            "obs", "OBS", "fa-solid fa-circle-dot",
            "Read-only WebSocket peek at live + scene state. Disabled by default.",
            ("obs_enabled", "obs_host", "obs_port", "obs_password"),
        ),
        (
            "streamelements", "StreamElements", "fa-solid fa-coins",
            "Pulls tip / sub / cheer / raid / follow events into the dashboard.",
            (
                "streamelements_enabled", "streamelements_jwt",
                "streamelements_channel_id",
            ),
        ),
        (
            "youtube", "YouTube", "fa-brands fa-youtube",
            "Stub — listener wired, no API polling yet. Configure to reserve credentials.",
            ("youtube_enabled", "youtube_api_key", "youtube_channel_id"),
        ),
        (
            "discord", "Discord", "fa-brands fa-discord",
            "Stub — listener wired, no gateway connection yet. Channel IDs comma-separated.",
            ("discord_enabled", "discord_bot_token", "discord_channel_ids"),
        ),
        (
            "moderation", "Moderation", "fa-solid fa-shield-halved",
            "Opt-in advisory classifier. The bot never takes chat action.",
            ("mod_mode_enabled",),
        ),
        (
            "dashboard", "Dashboard UI", "fa-solid fa-sliders",
            "Display preferences for this dashboard. No bot restart needed.",
            ("live_widget_enabled",),
        ),
    )

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request, saved: int = Query(0)):
        live, db_sources = get_settings_with_sources()

        def _row(key: str) -> dict:
            value = getattr(live, key)
            is_secret = key in SECRET_SETTING_KEYS
            is_bool = isinstance(value, bool)
            return {
                "key": key,
                "label": key.replace("_", " "),
                "value": value,
                "is_secret": is_secret,
                "is_bool": is_bool,
                "is_set": bool(value) if not is_bool else True,
                "source": "db" if key in db_sources else "env",
            }

        grouped: list[dict] = []
        seen: set[str] = set()
        for gid, title, icon, blurb, keys in _SETTINGS_GROUPS:
            grouped.append({
                "id": gid,
                "title": title,
                "icon": icon,
                "blurb": blurb,
                "rows": [_row(k) for k in keys if k in EDITABLE_SETTING_KEYS],
            })
            seen.update(keys)
        leftover = [k for k in EDITABLE_SETTING_KEYS if k not in seen]
        if leftover:
            grouped.append({
                "id": "other", "title": "Other", "icon": "fa-ellipsis",
                "blurb": "", "rows": [_row(k) for k in leftover],
            })

        from ..diagnose import make_github_issue_url
        return TEMPLATES.TemplateResponse(
            request,
            "settings.html",
            {
                "groups": grouped,
                "saved": bool(saved),
                "github_issue_url": make_github_issue_url(),
            },
        )

    _BOOL_SETTING_KEYS: frozenset[str] = frozenset(
        {
            "streamelements_enabled",
            "mod_mode_enabled",
            "obs_enabled",
            "live_widget_enabled",
            "youtube_enabled",
            "discord_enabled",
        }
    )

    @app.post("/settings")
    async def settings_save(request: Request):
        form = await request.form()
        for key in EDITABLE_SETTING_KEYS:
            if key in _BOOL_SETTING_KEYS:
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
        tab = (form.get("_tab") or "").strip()
        url = f"/settings?saved=1&tab={tab}" if tab else "/settings?saved=1"
        return RedirectResponse(url=url, status_code=303)

    # ---------------- moderation (opt-in via MOD_MODE_ENABLED) ----------------

    @app.get("/moderation", response_class=HTMLResponse)
    async def moderation_page(
        request: Request,
        status: str = Query("open"),
        page: int = Query(1, ge=1),
        partial: int = Query(0),
    ):
        if status not in ("open", "reviewed", "dismissed", "all"):
            status = "open"
        status_filter = None if status == "all" else status
        offset = (page - 1) * PAGE_SIZE
        rows = repo.list_incidents(
            status=status_filter, limit=PAGE_SIZE, offset=offset
        )
        total = repo.count_incidents(status=status_filter)
        ctx = {
            "rows": rows,
            "status": status,
            "page": page,
            "total": total,
            "pages": max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE),
            "mod_mode_enabled": settings.mod_mode_enabled,
        }
        tpl = "partials/incidents_list.html" if partial else "moderation.html"
        return TEMPLATES.TemplateResponse(request, tpl, ctx)

    @app.post("/moderation/{incident_id}/status", response_class=HTMLResponse)
    async def update_incident(
        request: Request,
        incident_id: int,
        new_status: Annotated[str, Form()],
    ):
        if new_status not in ("open", "reviewed", "dismissed"):
            raise HTTPException(400, "invalid status")
        repo.update_incident_status(incident_id, new_status)
        row = repo.get_incident(incident_id)
        if not row:
            raise HTTPException(404, "incident not found")
        return TEMPLATES.TemplateResponse(
            request, "partials/incident_card.html", {"row": row}
        )

    # ---------------- reminders ----------------

    @app.post("/users/{twitch_id}/reminders", response_class=HTMLResponse)
    async def add_user_reminder(
        request: Request,
        twitch_id: str,
        text: Annotated[str, Form()],
    ):
        text = text.strip()
        if text:
            user = repo.get_user(twitch_id)
            if not user:
                raise HTTPException(404, "user not found")
            repo.add_reminder(twitch_id, text[:500])
        reminders = repo.get_reminders_for_user(twitch_id)
        return TEMPLATES.TemplateResponse(
            request, "partials/reminders_section.html",
            {"reminders": reminders, "user": {"twitch_id": twitch_id}},
        )

    @app.post("/reminders/{reminder_id}/dismiss", response_class=HTMLResponse)
    async def dismiss_reminder_route(request: Request, reminder_id: int):
        repo.dismiss_reminder(reminder_id)
        return Response(status_code=200)

    @app.delete("/reminders/{reminder_id}")
    async def delete_reminder_route(reminder_id: int):
        repo.delete_reminder(reminder_id)
        return Response(status_code=200)

    @app.get("/reminders", response_class=HTMLResponse)
    async def reminders_page(request: Request):
        fired = repo.list_fired_reminders()
        return TEMPLATES.TemplateResponse(
            request, "reminders.html", {"fired": fired},
        )

    # ---------------- bot restart ----------------
    # Sends SIGTERM to the bot process via its pid file. The Makefile's
    # `make all` target re-launches the bot automatically (fail-fast loop),
    # so the practical effect is "pick up new credentials." Bare `make bot`
    # users will see the bot exit and need to re-run it themselves.

    @app.post("/restart-bot", response_class=HTMLResponse)
    async def restart_bot(request: Request):
        import os
        import signal as _signal
        from pathlib import Path as _P

        pid_file = _P("data/.bot.pid")
        ctx: dict[str, Any] = {"ok": False, "message": ""}
        if not pid_file.exists():
            ctx["message"] = (
                "no bot pid file at data/.bot.pid. If you just updated "
                "chatterbot, restart `make all` once manually — the bot only "
                "writes its pid here on versions that include this restart "
                "feature. After that one restart, this button works going "
                "forward."
            )
            return TEMPLATES.TemplateResponse(
                request, "partials/restart_bot_result.html", ctx
            )
        try:
            pid = int(pid_file.read_text().strip())
        except (OSError, ValueError) as e:
            ctx["message"] = f"could not read pid file: {e}"
            return TEMPLATES.TemplateResponse(
                request, "partials/restart_bot_result.html", ctx
            )
        try:
            os.kill(pid, _signal.SIGTERM)
        except ProcessLookupError:
            ctx["message"] = (
                f"bot pid {pid} not running. If you started with `make all`, "
                "the auto-restart loop will pick up the next launch."
            )
            return TEMPLATES.TemplateResponse(
                request, "partials/restart_bot_result.html", ctx
            )
        except PermissionError:
            ctx["message"] = (
                f"can't signal pid {pid} — bot is owned by a different user."
            )
            return TEMPLATES.TemplateResponse(
                request, "partials/restart_bot_result.html", ctx
            )
        ctx["ok"] = True
        ctx["message"] = (
            "sent SIGTERM to bot. If you launched via `make all`, it'll "
            "respawn within a few seconds with the new settings. "
            "Otherwise re-run `make bot` to start it again."
        )
        return TEMPLATES.TemplateResponse(
            request, "partials/restart_bot_result.html", ctx
        )

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

    @app.get("/modals/merge/{twitch_id}", response_class=HTMLResponse)
    async def modal_merge(
        request: Request,
        twitch_id: str,
        q: str = Query(""),
    ):
        child = repo.get_user(twitch_id)
        if not child:
            raise HTTPException(404, "user not found")
        candidates = repo.search_users_for_merge(q, exclude_id=twitch_id, limit=20) if q else []
        return TEMPLATES.TemplateResponse(
            request,
            "modals/_merge.html",
            {"child": child, "candidates": candidates, "q": q},
        )

    @app.post("/users/{twitch_id}/merge", response_class=HTMLResponse)
    async def merge_user(
        request: Request,
        twitch_id: str,
        parent_id: Annotated[str, Form()],
    ):
        try:
            counts = repo.merge_users(twitch_id, parent_id)
        except ValueError as e:
            raise HTTPException(400, str(e))
        # Tell the browser to navigate to the parent — the child is now
        # provenance-only and would be confusing to land on.
        resp = Response(status_code=204)
        resp.headers["HX-Redirect"] = f"/users/{parent_id}"
        return resp

    @app.get("/modals/message/{message_id}", response_class=HTMLResponse)
    async def modal_message(request: Request, message_id: int):
        ctx = repo.get_message_context(message_id, before=3, after=3)
        if not ctx["focal"]:
            raise HTTPException(404, "message not found")
        return TEMPLATES.TemplateResponse(
            request,
            "modals/_message_context.html",
            {
                "focal": ctx["focal"],
                "before": ctx["before"],
                "after": ctx["after"],
            },
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

    # ---------------- stats ----------------

    @app.get("/stats", response_class=HTMLResponse)
    async def stats_page(request: Request):
        import random as _random

        totals = repo.stats_totals()
        ev = repo.stats_event_totals()
        per_day = repo.stats_messages_per_day(days=30)
        per_hour = repo.stats_messages_per_hour()
        top_chatters = repo.stats_top_chatters_lifetime(limit=10)
        top_supporters = repo.stats_top_supporters(limit=5)
        new_per_week = repo.stats_new_chatters_per_week(weeks=12)

        # "Did you know" pool — pick 3 at random per page load.
        avg_len = repo.stats_avg_message_length()
        longest = repo.stats_longest_message()
        chatty_hour = repo.stats_most_chatty_hour()
        busiest_day = repo.stats_busiest_day()
        oldest = repo.stats_oldest_chatter()

        facts: list[str] = []
        if totals["chatters"]:
            facts.append(
                f"Your chat has logged {totals['messages']:,} messages from "
                f"{totals['chatters']:,} unique chatters."
            )
        if top_chatters:
            n, c = top_chatters[0]
            facts.append(
                f"Most prolific chatter: <b>{n}</b> with {c:,} messages — they're carrying."
            )
        if longest:
            ln_name, _, ln_len = longest
            facts.append(
                f"Longest message ever was {ln_len:,} characters, "
                f"courtesy of <b>{ln_name}</b>."
            )
        if avg_len:
            facts.append(
                f"Average message length is <b>{avg_len:.0f}</b> characters. "
                "(Twitch's max is 500.)"
            )
        if chatty_hour is not None:
            facts.append(
                f"Chat is most active around <b>{chatty_hour:02d}:00</b> — "
                "stream around then for max engagement."
            )
        if busiest_day:
            d, c = busiest_day
            facts.append(
                f"All-time busiest day: <b>{d}</b> with {c:,} messages. "
                "Whatever you did, do it again."
            )
        if oldest:
            n, when = oldest
            facts.append(
                f"Your founding chatter is <b>{n}</b> — first seen "
                f"{when.replace('T', ' ')[:10]}."
            )
        if ev["tip_total"]:
            facts.append(
                f"Lifetime tips: <b>${ev['tip_total']:,.2f}</b> from "
                f"{ev['unique_tippers']:,} different supporters. "
                "Cheers to them."
            )
        if ev["bits_total"]:
            facts.append(
                f"<b>{ev['bits_total']:,}</b> bits cheered across "
                f"{ev['unique_cheerers']:,} chatters."
            )
        if not facts:
            facts.append(
                "No data yet — your stats grow as chatters speak."
            )

        random_facts = _random.sample(facts, min(3, len(facts)))

        # Compose payload. JSON-friendly only — Jinja's tojson handles it all.
        # Word cloud — top words across all chat (cheap-ish; cap at 100).
        try:
            top_words = repo.stats_top_words(limit=120, min_count=2)
        except Exception:
            logger.exception("stats: word-cloud query failed")
            top_words = []

        try:
            sessions = repo.stream_sessions(gap_minutes=30, limit=12)
        except Exception:
            logger.exception("stats: session-bucket query failed")
            sessions = []

        return TEMPLATES.TemplateResponse(
            request,
            "stats.html",
            {
                "totals": totals,
                "ev": ev,
                "per_day_labels": [d for d, _ in per_day],
                "per_day_values": [n for _, n in per_day],
                "per_hour_values": [n for _, n in per_hour],
                "top_chatter_names": [n for n, _ in top_chatters],
                "top_chatter_counts": [c for _, c in top_chatters],
                "top_supporters": top_supporters,
                "new_per_week_labels": [w for w, _ in new_per_week],
                "new_per_week_values": [n for _, n in new_per_week],
                "facts": random_facts,
                "longest": longest,
                "word_cloud": [[w, c] for w, c in top_words],
                "sessions": sessions,
            },
        )

    # ---------------- insights (engagement helper) ----------------

    # Time windows offered by the Insights window selector. The regulars +
    # lapsed sections use this; "Active right now" and "New today" stay at
    # their fixed semantic windows (10 min / 24 h).
    _INSIGHT_WINDOWS = [
        ("7d",       "-7 days",   "last 7 days"),
        ("30d",      "-30 days",  "last 30 days"),
        ("90d",      "-90 days",  "last 90 days"),
        ("ytd",      "start of year", "year to date"),
        ("1y",       "-365 days", "last 1 year"),
        ("lifetime", None,        "lifetime"),
    ]
    _INSIGHT_WINDOW_LOOKUP = {key: (modifier, label) for key, modifier, label in _INSIGHT_WINDOWS}

    @app.get("/insights", response_class=HTMLResponse)
    async def insights_page(
        request: Request,
        partial: int = Query(0),
        window: str = Query("7d"),
    ):
        import time as _t
        if window not in _INSIGHT_WINDOW_LOOKUP:
            window = "7d"
        modifier, label = _INSIGHT_WINDOW_LOOKUP[window]

        # Visiting /insights = "I've reviewed the newcomers." Acks the pill.
        # Skip on the HTMX-poll partial path so an auto-refresh in the
        # background doesn't silently swallow new arrivals.
        if not partial:
            try:
                repo.set_newcomers_ack()
            except Exception:
                logger.exception("failed to ack newcomers")

        cache = insights.cache
        regulars = repo.list_regulars(since=modifier, limit=10)
        lapsed = repo.list_lapsed_regulars(
            active_since=modifier, lapsed_for="-7 days", limit=10
        )
        newcomers = repo.list_first_timers_today(limit=20)
        try:
            anniversaries = repo.users_with_anniversary_today()
        except Exception:
            logger.exception("insights: anniversaries lookup failed")
            anniversaries = []
        age = (
            int(_t.time() - cache.refreshed_at)
            if cache.refreshed_at is not None
            else None
        )
        # Drop "Skip:" rows from talking points — the LLM emits them when it
        # has nothing useful to say about a chatter; they're not for display.
        tps = [
            tp for tp in cache.talking_points
            if not (tp.point or "").lstrip().lower().startswith("skip:")
        ]
        # Per-surface "last read" — items with a ts newer than this get a
        # NEW badge in the templates. Distinct from newcomers ack so the
        # streamer can scan-and-dismiss without losing context for items
        # they just want to monitor.
        insights_acked_at = repo.get_surface_ack("insights")
        ctx = {
            "talking_points": tps,
            "talking_points_age_seconds": age,
            "talking_points_error": cache.error,
            "regulars": regulars,
            "lapsed": lapsed,
            "newcomers": newcomers,
            "anniversaries": anniversaries,
            "window": window,
            "window_label": label,
            "window_options": _INSIGHT_WINDOWS,
            "acked_at": insights_acked_at,
        }
        tpl = "partials/insights_body.html" if partial else "insights.html"
        return TEMPLATES.TemplateResponse(request, tpl, ctx)

    @app.post("/insights/mark-read", response_class=HTMLResponse)
    async def insights_mark_read(request: Request):
        repo.set_surface_ack("insights")
        # Re-render the body partial so HTMX can swap it in place.
        return await insights_page(request, partial=1, window="7d")

    # ---------------- live chat (widget + full page) ----------------

    @app.get("/live", response_class=HTMLResponse)
    async def live(
        request: Request,
        limit: int = Query(30, ge=1, le=500),
        partial: int = Query(0),
    ):
        messages = repo.recent_global_messages(limit=limit)
        signals = repo.arrival_signals_for_users(
            {m.user_id for m in messages}
        )
        ctx = {"messages": messages, "limit": limit, "arrival_signals": signals}
        tpl = "partials/live_rows.html" if partial else "live.html"
        return TEMPLATES.TemplateResponse(request, tpl, ctx)

    # ---------------- topics (now thread-grouped) ----------------

    _THREAD_STATUSES = ("active", "dormant", "archived", "all")

    @app.get("/topics", response_class=HTMLResponse)
    async def topics(
        request: Request,
        partial: int = Query(0),
        status: str = Query("all"),
        q: str = Query(""),
    ):
        if status not in _THREAD_STATUSES:
            status = "all"
        status_filter = None if status == "all" else status

        threads = repo.list_threads(
            status_filter=status_filter, query=q, limit=100
        )
        # Bucket the resulting threads for the UI sections. When the user
        # explicitly filters, only the requested bucket is non-empty.
        buckets = {"active": [], "dormant": [], "archived": []}
        for t in threads:
            buckets.setdefault(t.status, []).append(t)

        # Latest snapshot pinned at the top — the "what's chat doing right
        # now" anchor independent of the threading view.
        snapshots = repo.list_topic_snapshots(limit=10)
        latest = snapshots[0] if snapshots else None

        # Parse the structured topics on the latest snapshot so each bullet
        # becomes a clickable row that opens the per-topic detail modal.
        latest_topics: list[dict] = []
        if latest and latest.topics_json:
            try:
                from ..llm.schemas import TopicsResponse
                parsed = TopicsResponse.model_validate_json(latest.topics_json)
                latest_topics = [
                    {
                        "index": i,
                        "topic": t.topic,
                        "drivers": t.drivers,
                        "category": t.category,
                    }
                    for i, t in enumerate(parsed.topics)
                ]
            except Exception:
                logger.exception("topics: failed to parse latest topics_json")

        # Legacy text-only snapshots (no topics_json => never threaded).
        # Tucked at the bottom so the new view dominates.
        legacy = [s for s in snapshots if not s.topics_json][:5]

        topics_acked_at = repo.get_surface_ack("topics")
        ctx = {
            "active":   buckets["active"],
            "dormant":  buckets["dormant"],
            "archived": buckets["archived"],
            "status": status,
            "q": q,
            "latest": latest,
            "latest_topics": latest_topics,
            "legacy": legacy,
            "settings": settings,
            "total": len(threads),
            "acked_at": topics_acked_at,
        }
        tpl = "partials/topics_body.html" if partial else "topics.html"
        return TEMPLATES.TemplateResponse(request, tpl, ctx)

    @app.post("/topics/mark-read", response_class=HTMLResponse)
    async def topics_mark_read(request: Request):
        repo.set_surface_ack("topics")
        return await topics(request, partial=1, status="all", q="")

    @app.get("/modals/thread/{thread_id}", response_class=HTMLResponse)
    async def modal_thread(request: Request, thread_id: int):
        thread = repo.get_thread(thread_id)
        if not thread:
            raise HTTPException(404, "thread not found")
        members = repo.get_thread_members(thread_id)
        # Resolve drivers to user_ids (alias-aware) for clickable pills.
        driver_links = []
        for name in thread.drivers:
            user = repo.find_user_by_alias_or_name(name)
            driver_links.append({"name": name, "user_id": user.twitch_id if user else None})
        # Sample a handful of messages for at-a-glance context.
        messages = repo.get_thread_messages(thread_id, limit=30)
        return TEMPLATES.TemplateResponse(
            request,
            "modals/_thread.html",
            {
                "thread": thread,
                "members": members,
                "driver_links": driver_links,
                "messages": messages,
            },
        )

    @app.get("/threads/{thread_id}/explain")
    async def thread_explain(thread_id: int):
        from .thread_rag import explain_thread

        async def stream():
            try:
                ctx = await explain_thread(repo, llm, thread_id)
            except Exception:
                logger.exception("thread explain setup failed")
                yield _sse("error", "internal error during retrieval")
                yield _sse("done", "")
                return
            if ctx is None:
                yield _sse("error", "thread not found")
                yield _sse("done", "")
                return
            try:
                async for chunk in ctx.stream:
                    yield _sse("chunk", chunk)
            except Exception:
                logger.exception("thread explain stream failed")
                yield _sse("error", "stream interrupted")
            yield _sse("done", "")

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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
