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
from datetime import datetime, timezone
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
    JSONResponse,
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
NOTES_PAGE_SIZE = 20


def _short_ts(ts: str | None) -> str:
    """Render a stored UTC ISO timestamp in the host's local timezone.

    All writes go through `_now_iso()` which stamps UTC, so we have to
    convert at display time. `astimezone()` with no argument coerces to
    the system local zone — same one OBS, the bot, and the streamer's
    OS all already agree on. Falls back to the raw string slice if the
    value isn't a parseable ISO timestamp."""
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
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
    # `make_llm_client` selects Ollama / Anthropic / OpenAI per
    # settings.llm_provider. Embeddings always go to local Ollama.
    from ..llm.providers import make_llm_client
    llm = make_llm_client(settings)

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

    # Snoozed insight cards whose due_ts has passed — shown as a separate
    # nav pill so the streamer sees follow-ups they asked to come back to.
    def _due_snoozes_count() -> int:
        try:
            return repo.count_due_snoozes()
        except Exception:
            return 0
    TEMPLATES.env.globals["due_snoozes_count"] = _due_snoozes_count

    # Cutoff used by the activity-badge component to label first-timers
    # (first_seen >= now - 24h). Recomputed per render so it stays accurate.
    from datetime import datetime, timedelta, timezone
    def _day_floor_iso() -> str:
        return (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(timespec="seconds")
    TEMPLATES.env.globals["day_floor_iso"] = _day_floor_iso
    # Now ISO — used by Insights to compare each card's due_ts against
    # "now" so snoozed items can resurface as soon as they're due.
    def _now_iso_str() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    TEMPLATES.env.globals["now_iso"] = _now_iso_str

    auth_dep = make_auth_dependency(settings)
    deps = [Depends(auth_dep)] if settings.dashboard_basic_auth_enabled else []

    obs_status = OBSStatusService(settings)
    # The Helix poller consults OBS so it can pause itself while we're
    # not actually streaming. Same defaults-on-uncertainty contract.
    twitch_status = TwitchService(settings, obs=obs_status)
    # InsightsService takes the twitch_status so the engaging-subjects
    # extractor can include channel context (current game, title) in
    # its prompt. Helps the LLM disambiguate game-specific jargon.
    insights = InsightsService(repo, llm, settings, twitch_status=twitch_status)
    # Real-time transcript service. Owns the whisper model (lazy-loaded
    # on first audio chunk) + match-to-card cache. No-op when disabled.
    from ..transcript import TranscriptService
    transcript_service = TranscriptService(repo, llm, settings, obs=obs_status)
    # Wire the talking-points provider so the transcript matcher can
    # auto-address them when the streamer speaks about them.
    transcript_service._talking_points_provider = lambda: (
        [tp for tp in insights.cache.talking_points
         if not (tp.point or "").lstrip().lower().startswith("skip:")]
        if insights.cache and insights.cache.talking_points else []
    )
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
        # Whisper auto-confirm loop — promotes auto_pending insight
        # states to addressed after the confirm timeout. No-op when
        # whisper is disabled.
        _bg_tasks.add(
            asyncio.create_task(
                transcript_service.auto_confirm_loop(),
                name="transcript_auto_confirm",
            )
        )
        # Batched LLM transcript matcher. Periodically reviews the window
        # of utterances since the last watermark and flips cards the
        # streamer demonstrably engaged with. Streamer-aware prompt
        # (knows most utterances are game reactions) keeps false
        # positives down. No-op when whisper or the matcher is disabled
        # in settings.
        _bg_tasks.add(
            asyncio.create_task(
                transcript_service.llm_match_loop(),
                name="transcript_llm_match",
            )
        )
        # Transcript group summariser — replaces the per-utterance
        # live strip with one observational line per window.
        _bg_tasks.add(
            asyncio.create_task(
                transcript_service.transcript_group_loop(),
                name="transcript_group_summary",
            )
        )
        # Topic-thread recap loop — observational summaries for the
        # Live Conversations panel on engagement view.
        _bg_tasks.add(
            asyncio.create_task(
                insights.thread_recap_loop(),
                name="thread_recap_loop",
            )
        )
        # Engaging-subjects extractor — separate subject-level pass
        # over recent chat (distinct from topic_threads' cosine
        # clustering). Filters religion/politics/controversy at the
        # prompt level.
        _bg_tasks.add(
            asyncio.create_task(
                insights.engaging_subjects_loop(),
                name="engaging_subjects_loop",
            )
        )
        # OBS screenshot capture — pairs visual context with each
        # transcript group summary. No-op if OBS is unreachable.
        _bg_tasks.add(
            asyncio.create_task(
                transcript_service.screenshot_loop(),
                name="transcript_screenshot_loop",
            )
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
    # Serve transcript-group screenshots directly out of the data
    # directory. Created lazily by the screenshot loop; mounting an
    # empty directory at startup is fine — StaticFiles tolerates it.
    from pathlib import Path as _Path
    _shot_dir = _Path(settings.db_path).parent / "transcript_screenshots"
    _shot_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/transcript-screenshots",
        StaticFiles(directory=str(_shot_dir)),
        name="transcript_screenshots",
    )

    # ---------------- chatters list ----------------

    @app.get("/", response_class=HTMLResponse)
    async def index(
        request: Request,
        view: str = Query("engagement"),
        partial: int = Query(0),
        window: str = Query("7d"),
        status: str = Query("all"),
        q: str = Query(""),
    ):
        # Home is the Insights page. The chatters list moved to
        # /chatters as a sibling — the streamer hits Insights first and
        # drills into chatters from there. Existing bookmarks of `/`
        # land here unchanged.
        return await insights_page(
            request, view=view, partial=partial,
            window=window, status=status, q=q,
        )

    @app.get("/chatters", response_class=HTMLResponse)
    async def chatters_page(
        request: Request,
        q: str = Query(""),
        sort: str = Query("last_seen"),
        page: int = Query(1, ge=1),
    ):
        # Full chatters list. HTMX-partial path covers both the
        # search-as-you-type / sort / pagination interactions and the
        # 15s auto-refresh tick on the chatters_table partial.
        partial = request.headers.get("hx-request", "").lower() == "true"
        return _render_chatters(request, q, sort, page, partial=partial)

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
        q: str = Query(""),
        sort: str = Query("last_seen"),
        page: int = Query(1, ge=1),
    ):
        user = repo.get_user(twitch_id)
        if not user:
            raise HTTPException(404, "user not found")
        notes_with_sources = repo.get_notes_filtered(
            twitch_id, query="", origin="all",
            limit=NOTES_PAGE_SIZE, offset=0,
        )
        notes_total = repo.count_notes_filtered(twitch_id, query="", origin="all")
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
        # Prev/next navigation within the streamer's current
        # filter+sort context. None when the chatter falls outside
        # the scan cap or no neighbour exists at the boundary.
        prev_id, next_id = repo.neighbours_in_chatters(
            twitch_id, query=q, sort=sort,
        )
        # Preserved query string for the "back to list" link and for
        # forwarding context to the next/prev user pages.
        from urllib.parse import urlencode as _ue
        ctx_qs = _ue({"q": q, "sort": sort, "page": page})
        return TEMPLATES.TemplateResponse(
            request,
            "user.html",
            {
                "user": user,
                "notes_with_sources": notes_with_sources,
                "notes_total": notes_total,
                "notes_page": 1,
                "notes_pages": max(1, (notes_total + NOTES_PAGE_SIZE - 1) // NOTES_PAGE_SIZE),
                "notes_q": "",
                "notes_origin": "all",
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
                "prev_id": prev_id,
                "next_id": next_id,
                "ctx_qs": ctx_qs,
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

    @app.post("/users/{twitch_id}/star", response_class=HTMLResponse)
    async def toggle_star(request: Request, twitch_id: str):
        """Streamer-personal favorite toggle. The star pill reads back its
        new state inline via HTMX swap."""
        user = repo.get_user(twitch_id)
        if not user:
            raise HTTPException(404, "user not found")
        repo.set_user_starred(twitch_id, not user.is_starred)
        user = repo.get_user(twitch_id)
        return TEMPLATES.TemplateResponse(
            request, "partials/star_button.html", {"user": user},
        )

    def _notes_partial_ctx(
        twitch_id: str,
        *,
        q: str,
        origin: str,
        page: int,
    ) -> dict:
        """Shared context builder for the notes partial — used by add, the
        GET partial route, and any time we need to re-render the list."""
        page = max(1, page)
        offset = (page - 1) * NOTES_PAGE_SIZE
        nws = repo.get_notes_filtered(
            twitch_id, query=q, origin=origin,
            limit=NOTES_PAGE_SIZE, offset=offset,
        )
        total = repo.count_notes_filtered(twitch_id, query=q, origin=origin)
        pages = max(1, (total + NOTES_PAGE_SIZE - 1) // NOTES_PAGE_SIZE)
        return {
            "notes_with_sources": nws,
            "notes_total": total,
            "notes_page": page,
            "notes_pages": pages,
            "notes_q": q,
            "notes_origin": origin,
            "user": {"twitch_id": twitch_id},
        }

    @app.get("/users/{twitch_id}/notes", response_class=HTMLResponse)
    async def user_notes_partial(
        request: Request,
        twitch_id: str,
        q: str = Query(""),
        origin: str = Query("all"),
        page: int = Query(1, ge=1),
    ):
        if origin not in ("all", "manual", "llm", "suspect"):
            origin = "all"
        ctx = _notes_partial_ctx(twitch_id, q=q, origin=origin, page=page)
        return TEMPLATES.TemplateResponse(request, "partials/notes_list.html", ctx)

    @app.post("/users/{twitch_id}/notes", response_class=HTMLResponse)
    async def add_user_note(
        request: Request,
        twitch_id: str,
        text: Annotated[str, Form()],
        source_message_ids: Annotated[str, Form()] = "",
    ):
        """Add a manual note. Optional `source_message_ids` is a CSV of
        message ids the streamer wants to cite (e.g., from the
        click-message-to-note flow). Validation against this user's
        own messages happens inside repo.add_note."""
        text = text.strip()
        user = repo.get_user(twitch_id)
        if not user:
            raise HTTPException(404, "user not found")
        if text:
            embedding: list[float] | None
            try:
                embedding = await llm.embed(text[:500])
            except Exception:
                logger.exception("embed failed for manual note; storing without vector")
                embedding = None
            cited: list[int] | None = None
            if source_message_ids:
                try:
                    cited = [
                        int(x.strip()) for x in source_message_ids.split(",")
                        if x.strip()
                    ]
                except ValueError:
                    cited = None
            repo.add_note(
                twitch_id, text[:500], embedding,
                source_message_ids=cited,
                origin="manual",
            )
        # Always reset to page 1 / no filter so the streamer sees what they
        # just added at the top of the list.
        ctx = _notes_partial_ctx(twitch_id, q="", origin="all", page=1)
        return TEMPLATES.TemplateResponse(request, "partials/notes_list.html", ctx)

    @app.get("/modals/note-from-message/{message_id}", response_class=HTMLResponse)
    async def modal_note_from_message(request: Request, message_id: int):
        """Open a "make note from this message" modal. The message text
        is pre-filled into the textarea so the streamer can paraphrase
        before saving; the message id rides as a hidden field so the
        new note auto-cites it."""
        ctx = repo.get_message_context(message_id, before=0, after=0)
        focal = ctx.get("focal") if ctx else None
        if focal is None:
            raise HTTPException(404, "message not found")
        return TEMPLATES.TemplateResponse(
            request, "modals/_note_from_message.html",
            {"message": focal},
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

    @app.get("/modals/notes/merge", response_class=HTMLResponse)
    async def modal_merge_notes(
        request: Request,
        ids: str = Query("", max_length=1000),
        twitch_id: str = Query("", max_length=64),
    ):
        """Merge-multiple-notes modal. `ids` is a comma-separated list
        of note ids the streamer selected; `twitch_id` is the user
        whose notes those are. We sanity-check ownership server-side
        so a malicious URL can't merge across users."""
        if not ids or not twitch_id:
            raise HTTPException(400, "missing ids or twitch_id")
        try:
            id_list = [int(x.strip()) for x in ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(400, "invalid ids")
        if len(id_list) < 2:
            raise HTTPException(400, "need at least 2 notes to merge")
        if not repo.get_user(twitch_id):
            raise HTTPException(404, "user not found")
        # Pull each note + verify all belong to twitch_id.
        notes = []
        for nid in id_list:
            n = repo.get_note(nid)
            if n is None or n.user_id != twitch_id:
                raise HTTPException(
                    400, f"note {nid} does not belong to {twitch_id}"
                )
            notes.append(n)
        # Pre-fill with concatenation. Streamer can clear and
        # write-from-scratch or click "suggest" for an LLM proposal.
        prefill = "\n\n".join(n.text for n in notes)
        return TEMPLATES.TemplateResponse(
            request, "modals/_merge_notes.html",
            {
                "notes": notes,
                "twitch_id": twitch_id,
                "ids_csv": ",".join(str(n.id) for n in notes),
                "prefill": prefill,
            },
        )

    @app.post("/notes/merge/suggest", response_class=HTMLResponse)
    async def merge_notes_suggest(
        request: Request,
        ids: Annotated[str, Form()],
        twitch_id: Annotated[str, Form()],
    ):
        """Single-shot LLM suggestion for the merged umbrella note.
        Returns a tiny partial that swaps into the textarea via
        hx-target. The streamer can edit before saving."""
        try:
            id_list = [int(x.strip()) for x in ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(400, "invalid ids")
        if len(id_list) < 2 or not repo.get_user(twitch_id):
            raise HTTPException(400, "need ≥2 notes from a real user")
        notes = []
        for nid in id_list:
            n = repo.get_note(nid)
            if n is None or n.user_id != twitch_id:
                raise HTTPException(400, "note ownership mismatch")
            notes.append(n)
        bullet = "\n".join(f"- {n.text}" for n in notes)
        prompt = (
            f"Merge these {len(notes)} notes about a single chatter into ONE "
            "umbrella note that captures the shared subject. Keep it under "
            "300 characters. Stay observational — paraphrase what was said, "
            "don't add facts the notes don't contain. Output the umbrella "
            "note as a single line, no quotes, no preamble.\n\n"
            f"NOTES:\n{bullet}"
        )
        try:
            # think=True — streamer clicked "suggest merge" and is
            # waiting; getting a faithful paraphrase right beats getting
            # a fast one wrong (the suggestion replaces a textarea full
            # of work).
            suggestion = await llm.generate(
                prompt=prompt,
                system_prompt=(
                    "You consolidate per-chatter notes into an umbrella "
                    "summary. Stay grounded in the notes' content. No "
                    "advice, no extrapolation — just the merged paraphrase."
                ),
                think=True,
            )
        except Exception:
            logger.exception("merge_notes_suggest: llm.generate failed")
            return Response(
                status_code=200,
                content="<textarea name='text' rows='5' "
                "class='w-full tap bg-canvas border border-subtle rounded "
                "px-3 py-2 text-base text-primary focus:outline-none "
                "focus:ring-2 focus:ring-accent' "
                "id='merge-text-field'>(LLM suggestion failed — type the "
                "umbrella note yourself)</textarea>",
                media_type="text/html",
            )
        suggestion = (suggestion or "").strip()[:500]
        # Return the textarea with the suggested text. hx-swap='outerHTML'
        # replaces the existing textarea in place.
        return TEMPLATES.TemplateResponse(
            request, "partials/merge_textarea.html",
            {"prefill": suggestion},
        )

    @app.post("/notes/merge", response_class=HTMLResponse)
    async def merge_notes_endpoint(
        request: Request,
        ids: Annotated[str, Form()],
        twitch_id: Annotated[str, Form()],
        text: Annotated[str, Form()],
    ):
        """Apply a merge: combine selected notes into one new umbrella
        note. Originals are deleted, sources unioned. Returns the
        refreshed notes list partial so the page picks up the change
        immediately."""
        try:
            id_list = [int(x.strip()) for x in ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(400, "invalid ids")
        if len(id_list) < 2:
            raise HTTPException(400, "need at least 2 notes to merge")
        if not repo.get_user(twitch_id):
            raise HTTPException(404, "user not found")
        # Verify ownership before mutating.
        for nid in id_list:
            n = repo.get_note(nid)
            if n is None or n.user_id != twitch_id:
                raise HTTPException(
                    400, f"note {nid} does not belong to {twitch_id}"
                )
        # Embed the umbrella text for RAG; non-fatal if it fails.
        embedding: list[float] | None
        try:
            embedding = await llm.embed(text[:500])
        except Exception:
            logger.exception("merge_notes: embed failed; saving without vector")
            embedding = None
        new_id = await asyncio.to_thread(
            repo.merge_notes, id_list, text[:500], embedding=embedding,
        )
        if new_id is None:
            raise HTTPException(400, "merge produced no note (empty text?)")
        # Re-render the notes list partial so the user page picks up
        # the new merged note + the deletion of originals.
        ctx = _notes_partial_ctx(twitch_id, q="", origin="all", page=1)
        resp = TEMPLATES.TemplateResponse(
            request, "partials/notes_list.html", ctx,
        )
        # Trigger client-side close of the modal once the response lands.
        resp.headers["HX-Trigger"] = "merge-notes-done"
        return resp

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
            "Live-chat ingestion via YouTube Data API v3. Read-only API key. "
            "Adaptive polling keeps quota usage in check on long streams; "
            "raise the minimum poll interval if you stream past 6h.",
            (
                "youtube_enabled", "youtube_api_key", "youtube_channel_id",
                "youtube_min_poll_seconds", "youtube_max_poll_seconds",
            ),
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
        (
            "whisper", "Whisper", "fa-solid fa-microphone",
            "Real-time stream transcription via the OBS audio relay script. "
            "Auto-marks insight cards 'addressed' when you speak about them. "
            "First time the model loads it'll download ~75MB-1GB depending on size.",
            (
                "whisper_enabled", "whisper_model",
                "whisper_buffer_seconds", "whisper_match_threshold",
                "whisper_unnamed_match_threshold",
                "whisper_min_silence_ms",
                "whisper_llm_match_enabled",
                "whisper_llm_match_interval_seconds",
                "whisper_llm_match_min_chunks",
                "whisper_llm_match_confidence",
                "whisper_auto_confirm_seconds",
                "whisper_group_interval_seconds",
                "whisper_group_min_chunks",
                "thread_recap_interval_seconds",
                "thread_recap_max_messages_per_thread",
                "screenshot_interval_seconds",
                "screenshot_max_age_hours",
                "screenshot_jpeg_quality",
                "screenshot_width",
                "screenshot_grid_max",
            ),
        ),
        (
            "ai", "AI Provider", "fa-solid fa-robot",
            "Which backend handles generation calls (notes, recaps, "
            "engaging-subjects, etc). Embeddings ALWAYS run on local "
            "Ollama regardless — vec_messages dim is locked. "
            "Set `llm_provider` to ollama / anthropic / openai. "
            "Restart the dashboard after changes.",
            (
                "llm_provider",
                "anthropic_api_key", "anthropic_model",
                "anthropic_thinking_budget_tokens",
                "openai_api_key", "openai_model",
                "openai_reasoning_model", "openai_organization",
            ),
        ),
        (
            "internal", "Internal bus", "fa-solid fa-bolt",
            "Cross-process notification bus — the bot pushes to the "
            "dashboard's SSE stream when chat arrives so clients see "
            "updates with ~10ms latency instead of waiting for the "
            "10s watermark fallback. Empty URL disables push (the "
            "fallback poll still works).",
            (
                "dashboard_internal_url",
                "internal_notify_secret",
            ),
        ),
    )

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request, saved: int = Query(0)):
        live, db_sources = get_settings_with_sources()

        # Enum-style fields render as a <select> with these options.
        # Adding a new entry here AND a corresponding `is_set` /
        # validation rule is the only place to touch — the template
        # falls back to text input otherwise.
        ENUM_FIELDS: dict[str, list[str]] = {
            "llm_provider": ["ollama", "anthropic", "openai"],
        }

        def _row(key: str) -> dict:
            value = getattr(live, key)
            is_secret = key in SECRET_SETTING_KEYS
            is_bool = isinstance(value, bool)
            options = ENUM_FIELDS.get(key)
            return {
                "key": key,
                "label": key.replace("_", " "),
                "value": value,
                "is_secret": is_secret,
                "is_bool": is_bool,
                "options": options,
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
            "whisper_enabled",
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

    @app.get("/modals/profile/{twitch_id}/edit", response_class=HTMLResponse)
    async def modal_edit_profile(request: Request, twitch_id: str):
        user = repo.get_user(twitch_id)
        if not user:
            raise HTTPException(404, "user not found")
        # interests is stored JSON-encoded in the column; the user model
        # already deserialises it. Render as comma-joined for the form.
        interests = user.interests or []
        return TEMPLATES.TemplateResponse(
            request,
            "modals/_edit_profile.html",
            {"user": user, "interests_csv": ", ".join(interests)},
        )

    @app.post("/users/{twitch_id}/profile", response_class=HTMLResponse)
    async def update_user_profile_form(
        request: Request,
        twitch_id: str,
        pronouns: Annotated[str, Form()] = "",
        location: Annotated[str, Form()] = "",
        demeanor: Annotated[str, Form()] = "",
        interests: Annotated[str, Form()] = "",
    ):
        if not repo.get_user(twitch_id):
            raise HTTPException(404, "user not found")
        # demeanor: only persist when one of the known buckets so we don't
        # poison the LLM-extractor's signal with free-form values. Empty =>
        # clear it.
        valid_demeanors = {"hype", "chill", "supportive", "snarky",
                            "quiet", "analytical", "unknown"}
        d = demeanor.strip().lower() if demeanor else ""
        if d and d not in valid_demeanors:
            d = ""
        interest_list = [
            tag.strip() for tag in (interests or "").split(",") if tag.strip()
        ]
        repo.set_user_profile(
            twitch_id,
            pronouns=pronouns,
            location=location,
            demeanor=d or None,
            interests=interest_list or None,
        )
        user = repo.get_user(twitch_id)
        return TEMPLATES.TemplateResponse(
            request, "partials/profile_section.html", {"user": user},
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
        # Bidirectional link: if this message looks like a response to
        # something the streamer said on stream in the last few minutes,
        # surface the matching transcript chunk so the streamer can see
        # "yeah, this chat message was answering my question."
        related_transcript = None
        related_sim = None
        if transcript_service.enabled:
            try:
                qv = await llm.embed(ctx["focal"].content)
                hit = repo.find_related_transcript(
                    qv, max_age_minutes=5, threshold=0.40,
                )
                if hit:
                    related_transcript, related_sim = hit
            except Exception:
                logger.exception("modal_message: related-transcript lookup failed")
        return TEMPLATES.TemplateResponse(
            request,
            "modals/_message_context.html",
            {
                "focal": ctx["focal"],
                "before": ctx["before"],
                "after": ctx["after"],
                "parent_ids": ctx.get("parent_ids", {}),
                "related_transcript": related_transcript,
                "related_transcript_sim": related_sim,
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

    # /stats is a heavy aggregate page (18+ repo queries) that
    # changes slowly. Cache the assembled context for STATS_TTL_SECONDS
    # so back-to-back loads don't re-run the SQL. The cache is tiny
    # (one dict) and lives on the closure.
    _STATS_TTL_SECONDS = 60.0
    _stats_cache: dict[str, object] = {"ts": 0.0, "ctx": None, "match_threshold": None}

    @app.get("/stats", response_class=HTMLResponse)
    async def stats_page(request: Request):
        import random as _random
        import time as _t

        live_settings = get_settings()
        match_threshold = float(live_settings.whisper_match_threshold)

        cached_ctx = _stats_cache.get("ctx")
        cached_age = _t.time() - float(_stats_cache.get("ts") or 0)
        cached_threshold = _stats_cache.get("match_threshold")
        if (
            cached_ctx is not None
            and cached_age < _STATS_TTL_SECONDS
            and cached_threshold == match_threshold
        ):
            # Re-pick "did you know" facts on every render so the page
            # feels alive even when the underlying numbers are cached.
            facts_pool = cached_ctx.get("_facts_pool", [])
            ctx = dict(cached_ctx)
            ctx["facts"] = _random.sample(facts_pool, min(3, len(facts_pool)))
            return TEMPLATES.TemplateResponse(request, "stats.html", ctx)

        # Cache miss — fan out every repo call in parallel via
        # asyncio.gather + to_thread. Each query is independent; the
        # event loop frees up while SQLite (WAL-mode, parallel-read
        # safe) services them on the thread pool. Wall-clock collapses
        # from sum-of-queries (~95 ms) to slowest-single-query (~50 ms).
        # Word-cloud queries are NOT in this batch — they're scanning
        # the full message body and have their own /stats/wordcloud
        # endpoint with a longer (5-min) cache, lazy-loaded by the
        # template after page render.
        (
            totals, ev, per_day, per_hour, top_chatters, top_supporters,
            new_per_week, transcripts, transcripts_per_day, longest_utterance,
            avg_len, longest, chatty_hour, busiest_day, oldest,
            sessions,
        ) = await asyncio.gather(
            asyncio.to_thread(repo.stats_totals),
            asyncio.to_thread(repo.stats_event_totals),
            asyncio.to_thread(repo.stats_messages_per_day, days=30),
            asyncio.to_thread(repo.stats_messages_per_hour),
            asyncio.to_thread(repo.stats_top_chatters_lifetime, limit=10),
            asyncio.to_thread(repo.stats_top_supporters, limit=5),
            asyncio.to_thread(repo.stats_new_chatters_per_week, weeks=12),
            asyncio.to_thread(repo.stats_transcripts, match_threshold=match_threshold),
            asyncio.to_thread(repo.stats_transcripts_per_day, days=30),
            asyncio.to_thread(repo.stats_longest_utterance),
            asyncio.to_thread(repo.stats_avg_message_length),
            asyncio.to_thread(repo.stats_longest_message),
            asyncio.to_thread(repo.stats_most_chatty_hour),
            asyncio.to_thread(repo.stats_busiest_day),
            asyncio.to_thread(repo.stats_oldest_chatter),
            asyncio.to_thread(repo.stream_sessions, gap_minutes=30, limit=12),
            return_exceptions=False,
        )

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
        if transcripts["chunks"]:
            mins = transcripts["total_seconds"] / 60.0
            facts.append(
                f"Whisper has logged <b>{transcripts['chunks']:,}</b> "
                f"utterances totalling <b>{mins:,.0f}</b> minutes of stream voice."
            )
            if transcripts["auto_addressed"]:
                pct = 100.0 * transcripts["auto_addressed"] / transcripts["chunks"]
                facts.append(
                    f"Of those, <b>{transcripts['auto_addressed']:,}</b> "
                    f"({pct:.0f}%) auto-addressed an open insight card."
                )
        if not facts:
            facts.append(
                "No data yet — your stats grow as chatters speak."
            )

        random_facts = _random.sample(facts, min(3, len(facts)))

        # Compose payload. JSON-friendly only — Jinja's tojson handles it all.
        ctx = {
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
            # word_cloud + word_cloud_voice removed — lazy-loaded via
            # /stats/wordcloud now. Template flips a boolean instead
            # so the cloud section renders with a placeholder + fetch.
            "wordcloud_enabled": True,
            "sessions": sessions,
            "transcripts": transcripts,
            "transcripts_per_day_labels": [d for d, _ in transcripts_per_day],
            "transcripts_per_day_values": [n for _, n in transcripts_per_day],
            "longest_utterance": longest_utterance,
            "match_threshold": match_threshold,
        }
        # Stash for the next 60 s of /stats hits. We keep the unselected
        # facts pool too so the next render can re-sample without
        # rebuilding the whole context.
        _stats_cache["ctx"] = dict(ctx, _facts_pool=facts)
        _stats_cache["ts"] = _t.time()
        _stats_cache["match_threshold"] = match_threshold
        return TEMPLATES.TemplateResponse(request, "stats.html", ctx)

    # The word cloud is the most expensive query on /stats — it scans
    # every message body in the lookback window and tokenises in
    # Python. Pulling it onto its own endpoint with a longer cache
    # lets /stats render fast and lets the cloud lazy-load on the
    # client. 5 min TTL: word frequencies don't shift minute-to-minute.
    _WORDCLOUD_TTL_SECONDS = 5 * 60
    _wordcloud_cache: dict[str, object] = {
        "ts": 0.0, "lookback": None, "payload": None,
    }

    @app.get("/stats/wordcloud")
    async def stats_wordcloud(
        lookback: int = Query(30, ge=0, le=3650),
    ):
        """Lazy-loaded word-cloud payload for /stats. Bounded to the
        last `lookback` days by default (30) — bounding the scan keeps
        the cost flat as the message log grows. Pass `?lookback=0`
        for the lifetime view.

        Cached for 5 min in process memory keyed on the lookback
        window. Returns JSON: `{"chat": [[word, count], ...],
        "voice": [[word, count], ...]}`."""
        import time as _t
        cache_key = int(lookback)
        cached_payload = _wordcloud_cache.get("payload")
        cached_age = _t.time() - float(_wordcloud_cache.get("ts") or 0)
        cached_lookback = _wordcloud_cache.get("lookback")
        if (
            cached_payload is not None
            and cached_age < _WORDCLOUD_TTL_SECONDS
            and cached_lookback == cache_key
        ):
            return JSONResponse(cached_payload, headers={"Cache-Control": "no-store"})

        # `lookback=0` means "lifetime" — pass None down to the repo
        # so the WHERE clause is omitted entirely.
        repo_lookback = None if cache_key == 0 else cache_key
        chat_words, voice_words = await asyncio.gather(
            asyncio.to_thread(
                repo.stats_top_words,
                limit=120, min_count=2, lookback_days=repo_lookback,
            ),
            asyncio.to_thread(
                repo.stats_top_words_transcripts,
                limit=120, min_count=2, lookback_days=repo_lookback,
            ),
        )
        payload = {
            "chat": [[w, c] for w, c in chat_words],
            "voice": [[w, c] for w, c in voice_words],
            "lookback_days": cache_key,
        }
        _wordcloud_cache["payload"] = payload
        _wordcloud_cache["ts"] = _t.time()
        _wordcloud_cache["lookback"] = cache_key
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})

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

    _INSIGHTS_VIEWS = ("engagement", "topics")
    _THREAD_STATUSES = ("active", "dormant", "archived", "all")

    def _talking_point_key(tp) -> str:
        """Stable identity for a talking point so its addressed/snoozed
        state survives across cache refreshes. Hash is deterministic but
        short — fine for an index key."""
        import hashlib
        h = hashlib.sha1(
            (tp.user_id + "|" + (tp.point or "")).encode("utf-8")
        ).hexdigest()[:16]
        return f"{tp.user_id}:{h}"

    def _build_engagement_ctx(window: str) -> dict:
        import time as _t
        if window not in _INSIGHT_WINDOW_LOOKUP:
            window = "7d"
        modifier, label = _INSIGHT_WINDOW_LOOKUP[window]
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
        tps = [
            tp for tp in cache.talking_points
            if not (tp.point or "").lstrip().lower().startswith("skip:")
        ]
        # Stamp each talking point with its insight_state key so the
        # template can apply addressed/snoozed/pinned styling.
        for tp in tps:
            tp.item_key = _talking_point_key(tp)

        return {
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
            "acked_at": repo.get_surface_ack("insights"),
            # Per-card state, keyed for each section.
            "tp_states": repo.get_insight_states("talking_point"),
            "anniv_states": repo.get_insight_states("anniversary"),
            "newc_states": repo.get_insight_states("newcomer"),
            "reg_states": repo.get_insight_states("regular"),
            "lapsed_states": repo.get_insight_states("lapsed"),
            # Per-row dismissal for the "Talking to you" + "Active but
            # not engaged" panels. Keyed by user_id so snoozing once
            # hides every entry from that chatter for the window.
            "dm_states": repo.get_insight_states("direct_mention"),
            "nl_states": repo.get_insight_states("neglected_lurker"),
            # Activity pulse for the sparkline at the top.
            "pulse": repo.messages_per_minute(60),
            # Direct mentions (recent questions / @-style addresses).
            # Pull a wider candidate list than we display by default —
            # the template reveals 8, surfaces "show N more" so the
            # streamer can audit the full queue without an extra round
            # trip.
            "direct_mentions": repo.recent_direct_mentions(limit=30),
            # Most recent end-of-stream recap, if any.
            "latest_recap": (repo.list_stream_recaps(limit=1) or [None])[0],
            # Recap deltas — last 5 recaps for the cross-stream KPI strip.
            "recent_recaps": repo.list_stream_recaps(limit=5),
            # Streamer-personal favorites currently in chat.
            "starred_active": repo.list_starred_active(within_minutes=30, limit=12),
            # Active-now regulars the streamer hasn't engaged with
            # recently. Same expansion pattern as direct_mentions —
            # template renders 8, surfaces "show N more" to walk the
            # rest of the queue.
            "neglected_lurkers": repo.list_neglected_lurkers(
                active_within_minutes=30, neglected_for_days=7, limit=30,
            ),
            # Per-stream goals + computed progress.
            "goals_state": _build_goals_state(),
            # Live transcript strip — what whisper just heard.
            "transcript_chunks": repo.list_transcript_chunks(limit=15),
            "transcript_groups": repo.list_transcript_groups(limit=15),
            "transcript_status": transcript_service.status(),
            # Recent LLM-transcript matches — feed of cards the matcher
            # has flipped to auto_pending in the last hour, regardless of
            # current state. Auto_pendings auto-confirm (default 5 min)
            # so without this surface they vanish before you notice.
            "recent_matches": repo.list_recent_transcript_matches(
                limit=15, window_minutes=60,
            ),
            # Live conversation threads — surface "what's the room
            # actually talking about right now" alongside the per-
            # chatter hooks. Streamer feedback: thread summaries with
            # participants are more useful than hallucinated per-
            # chatter hooks; this panel grounds suggestions in actual
            # clustered chat content.
            "live_threads": repo.list_threads(
                status_filter="active", query="", limit=12,
            ),
            "live_thread_states": repo.get_insight_states("thread"),
            # Quiet cohorts — clusters of chatters who shared a topic
            # thread but have all gone silent. Streamer can pivot back
            # to a topic to re-engage that group of people. Same skip/
            # addressed state machine as live threads (kind='thread').
            # Engaging subjects — LLM-curated distinct conversation
            # subjects from recent chat. Sensitive topics (religion /
            # politics / controversy) filtered at the prompt level.
            "engaging_subjects": insights.subjects_cache.subjects,
            "engaging_subjects_age_seconds": (
                int(_t.time() - insights.subjects_cache.refreshed_at)
                if insights.subjects_cache.refreshed_at is not None else None
            ),
            "engaging_subjects_error": insights.subjects_cache.error,
            # High-impact subjects — cross-references currently-active
            # chatters against topic_thread driver history. Identifies
            # which subject would re-engage the most of THIS audience.
            "high_impact_subjects": repo.list_high_impact_subjects(
                active_within_minutes=int(getattr(
                    settings, "high_impact_active_within_minutes", 30,
                )),
                lookback_days=int(getattr(
                    settings, "high_impact_lookback_days", 14,
                )),
                min_overlap=int(getattr(
                    settings, "high_impact_min_overlap", 2,
                )),
                limit=int(getattr(
                    settings, "high_impact_limit", 6,
                )),
            ),
            "quiet_cohorts": repo.list_quiet_thread_cohorts(
                silence_minutes=int(getattr(settings, "quiet_cohort_silence_minutes", 15)),
                lookback_hours=int(getattr(settings, "quiet_cohort_lookback_hours", 24)),
                min_drivers=int(getattr(settings, "quiet_cohort_min_drivers", 2)),
                limit=int(getattr(settings, "quiet_cohort_limit", 6)),
            ),
        }

    def _build_topics_ctx(status: str, q: str) -> dict:
        if status not in _THREAD_STATUSES:
            status = "all"
        status_filter = None if status == "all" else status
        threads = repo.list_threads(
            status_filter=status_filter, query=q, limit=100
        )
        buckets = {"active": [], "dormant": [], "archived": []}
        for t in threads:
            buckets.setdefault(t.status, []).append(t)
        snapshots = repo.list_topic_snapshots(limit=10)
        latest = snapshots[0] if snapshots else None
        latest_topics: list[dict] = []
        if latest and latest.topics_json:
            try:
                from ..llm.schemas import TopicsResponse
                parsed = TopicsResponse.model_validate_json(latest.topics_json)
                latest_topics = [
                    {
                        "index": i, "topic": t.topic,
                        "drivers": t.drivers, "category": t.category,
                    }
                    for i, t in enumerate(parsed.topics)
                ]
            except Exception:
                logger.exception("topics: failed to parse latest topics_json")
        legacy = [s for s in snapshots if not s.topics_json][:5]
        return {
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
            "acked_at": repo.get_surface_ack("topics"),
            "thread_states": repo.get_insight_states("thread"),
            "thread_velocity": repo.thread_velocity(),
        }

    @app.get("/insights", response_class=HTMLResponse)
    async def insights_page(
        request: Request,
        view: str = Query("engagement"),
        partial: int = Query(0),
        window: str = Query("7d"),
        status: str = Query("all"),
        q: str = Query(""),
    ):
        if view not in _INSIGHTS_VIEWS:
            view = "engagement"

        # Visiting /insights = "I've reviewed the newcomers." Acks the pill.
        # Skip on HTMX-partial path so background polls don't silently
        # swallow new arrivals.
        if not partial:
            try:
                repo.set_newcomers_ack()
            except Exception:
                logger.exception("failed to ack newcomers")

        ctx: dict = {"view": view}
        if view == "engagement":
            ctx.update(_build_engagement_ctx(window))
            tpl_partial = "partials/insights_body.html"
        else:  # topics
            ctx.update(_build_topics_ctx(status, q))
            tpl_partial = "partials/topics_body.html"

        if partial:
            return TEMPLATES.TemplateResponse(request, tpl_partial, ctx)
        return TEMPLATES.TemplateResponse(request, "insights.html", ctx)

    @app.get("/modals/insight/{kind}/{twitch_id}", response_class=HTMLResponse)
    async def modal_insight(
        request: Request,
        kind: str,
        twitch_id: str,
        meta: str = Query("", max_length=2000),
        focus: int | None = Query(None),
    ):
        from .insight_rag import KIND_DISPLAY, VALID_KINDS

        if kind not in VALID_KINDS:
            raise HTTPException(404, f"unknown insight kind: {kind}")
        user = repo.get_user(twitch_id)
        if not user:
            raise HTTPException(404, "user not found")
        rows, focal_ids = repo.recent_user_messages_with_context(twitch_id)
        # `focus` is the message id the streamer clicked from the row —
        # add it to the focal set so the modal highlights that exact
        # line. Necessary for direct-mention rows where the row IS one
        # specific message; without this, the modal just shows "all
        # recent messages" with nothing to anchor to.
        if focus is not None:
            focal_ids = set(focal_ids) | {int(focus)}
        # Subjects this chatter has driven across topic_threads —
        # ranked by drive count + recency. Replaces the previous
        # LLM-streamed "what to say" prescription with an
        # observational data view: "here are the subjects they engage
        # with; you (the streamer) decide what to lean into."
        engaging_subjects = repo.subjects_engaging_chatter(twitch_id, limit=10)
        return TEMPLATES.TemplateResponse(
            request,
            "modals/_insight_chatter.html",
            {
                "kind": kind,
                "display": KIND_DISPLAY[kind],
                "user": user,
                "meta": meta,
                "rows": rows,
                "focal_ids": focal_ids,
                "engaging_subjects_for_user": engaging_subjects,
            },
        )

    @app.get("/insights/insight/{kind}/{twitch_id}/explain")
    async def insight_explain(
        kind: str,
        twitch_id: str,
        meta: str = Query("", max_length=2000),
    ):
        from .insight_rag import explain_insight, VALID_KINDS

        if kind not in VALID_KINDS:
            raise HTTPException(404, f"unknown insight kind: {kind}")

        async def stream():
            try:
                ctx = await explain_insight(repo, llm, kind, twitch_id, meta)
            except Exception:
                logger.exception("insight explain setup failed")
                yield _sse("error", "internal error during retrieval")
                yield _sse("done", "")
                return
            if ctx is None:
                yield _sse("error", "user not found")
                yield _sse("done", "")
                return
            try:
                async for chunk in ctx.stream:
                    yield _sse("chunk", chunk)
            except Exception:
                logger.exception("insight explain stream failed")
                yield _sse("error", "stream interrupted")
            yield _sse("done", "")

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    _SNOOZE_MINUTES = {"10m": 10, "30m": 30, "1h": 60, "24h": 60 * 24}

    @app.post("/insights/state", response_class=HTMLResponse)
    async def insights_state(
        request: Request,
        kind: Annotated[str, Form()],
        item_key: Annotated[str, Form()],
        action: Annotated[str, Form()],
        snooze: Annotated[str, Form()] = "",
        note: Annotated[str, Form()] = "",
    ):
        """Single endpoint for all per-card actions: addressed / snooze /
        pin / skip / open. Returns 204 with HX-Trigger so the page can
        refresh the affected section in place.

        - action='addressed': mark as handled-on-stream; sinks card.
        - action='snooze' + snooze='10m|30m|1h|24h': come back later.
        - action='pin': hold at top of section.
        - action='skip': hide entirely.
        - action='open': clear any state (the undo path).
        """
        from datetime import datetime, timedelta, timezone
        action = action.strip().lower()
        valid_actions = (
            "addressed", "snooze", "pin", "skip", "open",
            # transcript-driven confirm/reject of an auto_pending card.
            "confirm_pending", "reject_pending",
        )
        if action not in valid_actions:
            raise HTTPException(400, f"unknown action: {action}")

        state = action
        due_ts: str | None = None
        if action == "snooze":
            mins = _SNOOZE_MINUTES.get(snooze)
            if mins is None:
                raise HTTPException(400, "snooze must be 10m|30m|1h|24h")
            due_ts = (
                datetime.now(timezone.utc) + timedelta(minutes=mins)
            ).isoformat(timespec="seconds")
            state = "snoozed"
        elif action == "pin":
            state = "pinned"
        elif action == "skip":
            state = "skipped"
        elif action == "addressed":
            state = "addressed"
        elif action == "confirm_pending":
            # Streamer says "yes, I addressed that one." Promote
            # auto_pending → addressed; preserve the existing (auto)
            # transcript note as the rationale.
            state = "addressed"
        elif action == "reject_pending":
            # Streamer says "no, that wasn't really addressing it."
            # Clear the state — the card returns to the open list.
            state = "open"

        repo.set_insight_state(
            kind, item_key, state,
            due_ts=due_ts, note=(note.strip() or None),
        )
        # 204 + HX-Trigger lets the caller drop the card without a full
        # body re-render. The Insights page re-fetches its body partial
        # via auto-refresh anyway.
        resp = Response(status_code=204)
        resp.headers["HX-Trigger"] = "insights-state-changed"
        return resp

    @app.get("/insights/subject/{slug}/expand", response_class=HTMLResponse)
    async def subject_expand(request: Request, slug: str):
        """Render the expand-on-click body for an engaging subject:
        the brief + angles (already in cache) + recent verbatim
        messages from the subject's drivers within the lookback
        window. No LLM call — pure SQL."""
        # Find the cached subject by slug
        subject = next(
            (s for s in insights.subjects_cache.subjects if s.slug == slug),
            None,
        )
        if subject is None:
            raise HTTPException(404, "subject not found in cache")
        within = int(getattr(
            settings, "engaging_subjects_lookback_minutes", 20,
        ))
        msgs = (
            repo.messages_for_names_within(
                subject.drivers, within_minutes=within, limit=40,
            )
            if subject.drivers else []
        )
        return TEMPLATES.TemplateResponse(
            request,
            "partials/engaging_subject_expand.html",
            {"subject": subject, "messages": msgs, "within_minutes": within},
        )

    @app.post("/insights/subject/{slug}/reject", response_class=HTMLResponse)
    async def subject_reject(request: Request, slug: str):
        """Streamer flagged a subject as hallucinated / wrong / not
        actually being discussed. Add to the blocklist so subsequent
        extraction passes exclude it. Returns an empty body so the
        UI can `hx-swap="outerHTML"` to remove the row."""
        subject = next(
            (s for s in insights.subjects_cache.subjects if s.slug == slug),
            None,
        )
        if subject is None:
            # Already gone from cache (refresh raced) — still record the
            # rejection if we got the slug, to prevent the next pass from
            # re-extracting whatever name eventually maps to it.
            await asyncio.to_thread(insights.reject_subject, slug, "")
        else:
            await asyncio.to_thread(
                insights.reject_subject, subject.slug, subject.name,
            )
        return Response(status_code=200, content="")

    @app.post("/insights/subjects/clear-rejections", response_class=HTMLResponse)
    async def subject_clear_rejections(request: Request):
        """Wipe the subject blocklist. The next extraction pass starts
        from a clean slate. Useful at stream boundaries."""
        n = await asyncio.to_thread(insights.clear_subject_blocklist)
        return Response(status_code=200, content=f"cleared {n} rejection(s)")

    _GOAL_KINDS = (
        "address_first_timers", "clear_due_snoozes", "address_returning_regulars",
    )

    @app.post("/insights/goals", response_class=HTMLResponse)
    async def set_goals(
        request: Request,
        kind: Annotated[str, Form()],
        count: Annotated[int, Form()] = 1,
    ):
        """Set/update the current stream's goal targets. Append-only — each
        new POST adds another goal to the list. POST kind='clear' to reset."""
        if kind == "clear":
            repo.clear_stream_goals()
        elif kind in _GOAL_KINDS:
            cur = repo.get_stream_goals()
            targets = list(cur.get("targets") or [])
            # Replace any existing goal of the same kind.
            targets = [t for t in targets if t.get("kind") != kind]
            targets.append({"kind": kind, "count": max(1, int(count))})
            repo.set_stream_goals(targets)
        # Re-render the goals panel partial.
        return TEMPLATES.TemplateResponse(
            request, "partials/goals_panel.html",
            {"goals": _build_goals_state()},
        )

    def _build_goals_state() -> dict:
        """Compute current progress against each goal target."""
        goals = repo.get_stream_goals()
        targets = goals.get("targets") or []
        if not targets:
            return {"targets": [], "set_at": None}
        from datetime import datetime, timezone, timedelta
        # Goals scoped to "since the goals were set" so they reset each
        # time the streamer hits the start-of-stream button.
        since = goals.get("set_at") or (
            datetime.now(timezone.utc) - timedelta(hours=24)
        ).isoformat(timespec="seconds")
        out: list[dict] = []
        for t in targets:
            kind = t.get("kind")
            target = max(1, int(t.get("count") or 1))
            # progress count depends on the goal kind.
            done = 0
            if kind == "address_first_timers":
                # Count first-timers (newcomers) the streamer has marked
                # addressed since `since`.
                done = repo.count_state_changes_since(since, state="addressed")
                # The above includes ALL kinds; filter to newcomers via
                # a follow-up check in repo.
            elif kind == "clear_due_snoozes":
                # 1 means "all clear right now", target == count of due.
                done = max(0, target - repo.count_due_snoozes())
            elif kind == "address_returning_regulars":
                done = repo.count_state_changes_since(since, state="addressed")
            out.append({
                "kind": kind,
                "target": target,
                "done": min(done, target),
                "pct": min(100, int(100 * done / target)) if target else 0,
            })
        return {"targets": out, "set_at": goals.get("set_at")}

    @app.get("/audit", response_class=HTMLResponse)
    async def audit_page(request: Request):
        """Audit trail of recent state transitions. Lets the streamer
        see what they acted on across this and prior streams."""
        rows = repo.list_state_history(limit=200)
        return TEMPLATES.TemplateResponse(
            request, "audit.html", {"rows": rows},
        )

    @app.post("/audio/ingest")
    async def audio_ingest(request: Request):
        """OBS audio relay endpoint. Body is raw float32 mono PCM. The
        sample rate is in the X-Sample-Rate header (default 16000).
        Returns 204 — fire-and-forget; transcription runs as a task.

        Auth/security: this is bound to the dashboard's listen address
        (default 127.0.0.1). Keep it that way unless you trust the
        network — anyone who can reach this endpoint can inject audio
        into the streamer's transcript log.
        """
        if not transcript_service.enabled:
            return Response(status_code=204)
        try:
            sr = int(request.headers.get("x-sample-rate") or 16000)
        except (TypeError, ValueError):
            sr = 16000
        body = await request.body()
        if body:
            await transcript_service.ingest_chunk(body, sr)
        return Response(status_code=204)

    @app.get("/modals/transcript-group/{group_id}", response_class=HTMLResponse)
    async def modal_transcript_group(request: Request, group_id: int):
        """Detail view for one transcript group — summary + the
        underlying utterances clipped to the group's id range, plus
        the OBS screenshots captured during the group's window."""
        group = repo.get_transcript_group(group_id)
        if not group:
            raise HTTPException(404, "transcript group not found")
        chunks = repo.transcript_chunks_in_id_range(
            group.first_chunk_id, group.last_chunk_id,
        )
        screenshots = repo.screenshots_in_range(
            group.start_ts, group.end_ts,
            max_count=int(getattr(settings, "screenshot_grid_max", 4)),
        )
        return TEMPLATES.TemplateResponse(
            request, "modals/_transcript_group.html",
            {"group": group, "chunks": chunks, "screenshots": screenshots},
        )

    @app.get("/modals/transcript/{chunk_id}", response_class=HTMLResponse)
    async def modal_transcript(request: Request, chunk_id: int):
        """Detail view for one transcript chunk: the utterance itself,
        adjacent transcript chunks for chronological context, and the
        insight card it matched (if any).

        `matched_card` is a dict {kind, label, href, secondary?} that
        the template renders generically. None when the chunk didn't
        match anything (similarity below threshold), or matched but
        the underlying card has aged out of cache.
        """
        ctx = repo.transcript_context_around(chunk_id, before=3, after=3)
        if not ctx["focal"]:
            raise HTTPException(404, "transcript chunk not found")
        focal = ctx["focal"]
        matched_card: dict | None = None

        if focal.matched_kind == "thread" and focal.matched_item_key:
            try:
                t = repo.get_thread(int(focal.matched_item_key))
            except Exception:
                t = None
            if t is not None:
                matched_card = {
                    "kind": "topic thread",
                    "label": t.title,
                    "href": f"/modals/thread/{t.id}",
                    "secondary": (t.category or "").strip(),
                }

        elif focal.matched_kind == "talking_point" and focal.matched_item_key:
            # item_key is "{user_id}:{sha1_digest}". Walk the live talking-
            # points cache to find the original (name, point) pair; if it's
            # aged out we still have the user_id half and can link to them.
            user_id = focal.matched_item_key.split(":", 1)[0]
            user = repo.get_user(user_id) if user_id else None
            chatter_name = user.name if user else user_id

            point_text: str | None = None
            try:
                cache = insights.cache
                tps = list(cache.talking_points) if (cache and cache.talking_points) else []
            except Exception:
                tps = []
            import hashlib as _h
            for tp in tps:
                p = (tp.point or "").strip()
                if not p:
                    continue
                digest = _h.sha1(f"{tp.user_id}|{p}".encode("utf-8")).hexdigest()[:16]
                if f"{tp.user_id}:{digest}" == focal.matched_item_key:
                    point_text = p
                    chatter_name = tp.name
                    break

            if user_id:
                matched_card = {
                    "kind": "talking point",
                    "label": (
                        f"{chatter_name}: {point_text}"
                        if point_text else
                        f"{chatter_name} (talking-point card has aged out of the cache)"
                    ),
                    "href": f"/users/{user_id}",
                    "secondary": "",
                }

        return TEMPLATES.TemplateResponse(
            request, "modals/_transcript_context.html",
            {
                "focal": focal,
                "before": ctx["before"],
                "after": ctx["after"],
                "matched_card": matched_card,
            },
        )

    @app.get("/transcript", response_class=HTMLResponse)
    async def transcript_partial(
        request: Request,
        limit: int = Query(20, ge=1, le=100),
        last_pending: int = Query(0, ge=0),
        view: str = Query("summary"),
    ):
        """Live transcript strip for /insights. Two view modes:

          - `summary` (default) — one row per LLM-summarised group,
            ~60 s windows. Less distracting; less granular.
          - `log` — every raw whisper utterance with its match icon.
            Granular; noisier; polls fast for near-real-time updates.

        Accepts the legacy `verbatim` value as an alias for `log` so a
        streamer with the older value in localStorage keeps the right
        view through the rename.

        Streamer toggles client-side via localStorage; the choice rides
        on the auto-refresh `hx-vals` so each tick honors it.
        """
        if view == "verbatim":
            view = "log"
        if view not in ("summary", "log"):
            view = "summary"
        if view == "log":
            groups = []
            chunks = repo.list_transcript_chunks(limit=limit)
        else:
            groups = repo.list_transcript_groups(limit=limit)
            # Warming-up fallback: while no groups exist yet, show
            # chunks so the strip isn't empty during the first ~60 s
            # of audio.
            chunks = (
                [] if groups else repo.list_transcript_chunks(limit=limit)
            )
        status = transcript_service.status()
        resp = TEMPLATES.TemplateResponse(
            request, "partials/transcript_strip.html",
            {"groups": groups, "chunks": chunks, "service_status": status,
             "pending_count": status["pending_count"], "view": view},
        )
        if status["pending_count"] > last_pending:
            resp.headers["HX-Trigger"] = "insights-state-changed"
        return resp

    @app.get("/health", response_class=HTMLResponse)
    async def health_partial(request: Request):
        """Ops-style health snapshot for the Settings page. Uses the
        existing settings + ad-hoc Ollama probe; non-blocking."""
        import time as _t
        ollama_ok = False
        ollama_ms: int | None = None
        ollama_err: str | None = None
        try:
            t0 = _t.time()
            ollama_ok = await llm.health_check()
            ollama_ms = int((_t.time() - t0) * 1000)
        except Exception as e:
            ollama_err = f"{type(e).__name__}: {e}"
        indexed, total = repo.messages_embedding_coverage()
        obs = obs_status.status
        ctx = {
            "ollama_ok": ollama_ok,
            "ollama_ms": ollama_ms,
            "ollama_err": ollama_err,
            "ollama_url": settings.ollama_base_url,
            "ollama_model": settings.ollama_model,
            "ollama_embed_model": settings.ollama_embed_model,
            "embed_indexed": indexed,
            "embed_total": total,
            "due_snoozes": repo.count_due_snoozes(),
            "fired_reminders": repo.count_fired_reminders(),
            "obs": obs,
            "obs_host": f"{settings.obs_host}:{settings.obs_port}",
            "transcript_status": transcript_service.status(),
        }
        return TEMPLATES.TemplateResponse(
            request, "partials/health_panel.html", ctx,
        )

    @app.post("/insights/mark-read", response_class=HTMLResponse)
    async def insights_mark_read(request: Request):
        repo.set_surface_ack("insights")
        # Re-render the engagement body partial so HTMX can swap it in place.
        return await insights_page(
            request, view="engagement", partial=1, window="7d",
        )

    # ---------------- live chat (widget + full page) ----------------

    # ---------------- semantic message search ----------------

    @app.get("/search", response_class=HTMLResponse)
    async def search_page(
        request: Request,
        q: str = Query(""),
        partial: int = Query(0),
        k: int = Query(20, ge=1, le=100),
    ):
        results: list[tuple] = []
        error: str | None = None
        if q.strip():
            try:
                q_vec = await llm.embed(q.strip())
                results = repo.search_global_messages(q_vec, k=k)
            except Exception as e:
                logger.exception("search: embedding/query failed")
                error = f"{type(e).__name__}: {e}"
        indexed, total = repo.messages_embedding_coverage()
        ctx = {
            "q": q,
            "results": results,
            "indexed": indexed,
            "total": total,
            "error": error,
        }
        tpl = "partials/search_results.html" if partial else "search.html"
        return TEMPLATES.TemplateResponse(request, tpl, ctx)

    def _live_context(messages):
        """Shared context build for live-chat templates — resolves
        arrival signals, callbacks, and starred users for a given
        message list. Used by both the initial GET /live render and
        the SSE delta stream.

        Uses the canonical batch helpers (`get_users_by_ids` instead
        of N `get_user()` calls). On a 30-message render the win is
        ~30 round-trips collapsed into 1."""
        user_ids = {m.user_id for m in messages}
        signals = repo.arrival_signals_for_users(user_ids)
        returning_ids = {uid for uid, sig in signals.items() if sig == "returning"}
        callbacks = repo.last_callback_for_users(returning_ids) if returning_ids else {}
        users = repo.get_users_by_ids(user_ids)
        starred = {uid for uid, u in users.items() if u.is_starred}
        return {
            "messages": messages,
            "arrival_signals": signals,
            "callbacks": callbacks,
            "starred": starred,
        }

    @app.get("/live", response_class=HTMLResponse)
    async def live(
        request: Request,
        limit: int = Query(30, ge=1, le=500),
        partial: int = Query(0),
    ):
        messages = repo.recent_global_messages(limit=limit)
        ctx = _live_context(messages) | {"limit": limit}
        tpl = "partials/live_rows.html" if partial else "live.html"
        return TEMPLATES.TemplateResponse(request, tpl, ctx)

    @app.post("/internal/notify")
    async def internal_notify(request: Request):
        """Cross-process notification endpoint. The bot posts here
        when state changes so the dashboard's SSE stream can fan out
        to connected clients with ~10 ms latency instead of waiting
        for the watermark poll.

        Auth: optional shared secret in `X-Internal-Secret`. When
        `settings.internal_notify_secret` is set, requests without
        a matching header are rejected. Empty secret = unauth (dev
        only — fine on a localhost-bound dashboard).

        Body: `{"channel": "<name>", "version": "<opaque>"}`. Channel
        must be one of the registered `/events/stream` channels;
        unknown channels are silently dropped (avoids letting a
        misconfigured publisher spam the bus)."""
        from ..eventbus import get_bus
        secret = settings.internal_notify_secret or ""
        if secret:
            sent = request.headers.get("x-internal-secret") or ""
            if sent != secret:
                raise HTTPException(401, "internal secret mismatch")
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(400, "invalid JSON body")
        ch = (payload.get("channel") or "").strip() if isinstance(payload, dict) else ""
        if not ch:
            raise HTTPException(400, "channel required")
        version = ""
        if isinstance(payload, dict) and payload.get("version") is not None:
            version = str(payload["version"])
        get_bus().publish(ch, version)
        return Response(status_code=204)

    @app.get("/events/stream")
    async def events_stream(request: Request):
        """Multiplexed change-notification SSE for HTMX-driven panels.

        One persistent connection per page open. Every ~1.5 s the
        server polls a tiny "version" extractor for each channel; on
        change, emits one SSE event per channel:

            event: <channel>
            data: <version stamp>
            \n

        The client-side driver in base.html dispatches a corresponding
        DOM event on `body` so HTMX panels can subscribe via
        `hx-trigger="<channel> from:body"`. Replaces the old
        `hx-trigger="every Xs"` polls — panels render on actual state
        change, not on a fixed cadence.

        Channel registry below; add channels here + a watermark fn on
        repo. Watermarks must be cheap (MAX(id) / MAX(ts), no joins)
        because we run all of them every tick."""
        # Channel → (watermark function, refresh-the-status-pills?).
        # The status flag is a side-channel: the in-memory pill
        # statuses (twitch_status, obs_status) live on services; we
        # poll their `refreshed_at` to know if anything changed.
        def _twitch_v():
            ts = getattr(twitch_status, "status", None)
            return f"{getattr(ts, 'refreshed_at', 0)}|{getattr(ts, 'is_live', False)}|{getattr(ts, 'viewer_count', 0)}"
        def _obs_v():
            o = getattr(obs_status, "status", None)
            return f"{getattr(o, 'refreshed_at', 0)}|{getattr(o, 'streaming', False)}|{getattr(o, 'recording', False)}|{getattr(o, 'scene', '')}"
        def _insights_engagement_v():
            tp = getattr(insights, "cache", None)
            es = getattr(insights, "subjects_cache", None)
            return f"{getattr(tp, 'refreshed_at', 0)}|{getattr(es, 'refreshed_at', 0)}"

        channels = {
            "chatters":            lambda: repo.latest_user_change_version(),
            "messages":            lambda: repo.latest_message_id(),
            "events":              lambda: repo.latest_event_id(),
            "transcript":          lambda: repo.latest_transcript_chunk_id(),
            "insights:engagement": _insights_engagement_v,
            "insights:topics":     lambda: repo.latest_topic_thread_version(),
            "twitch":              _twitch_v,
            "obs":                 _obs_v,
        }
        last: dict[str, object] = {ch: None for ch in channels}

        # Hybrid push + poll. Cross-process events come from the bot
        # via /internal/notify (low-latency, ~10 ms). Same-process
        # services publish to the bus directly. The watermark poll is
        # a slower fallback (10 s) for state changes not broadcast —
        # e.g. SQL writes from a future TUI / external tool.
        from ..eventbus import get_bus
        bus = get_bus()

        async def gen():
            yield _sse("hello", "")
            poll_every = 10.0       # fallback poll cadence
            heartbeat_every = 15.0  # seconds of silence before ping

            # Single shared queue. Both push (bus subscriber) and
            # poll (watermark loop) write into it; the main gen
            # loop reads from it with a heartbeat timeout.
            outq: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=128)

            async def push_pump():
                """Forward bus publications into outq."""
                async for ch, version in bus.subscribe():
                    if ch not in channels:
                        continue
                    if version and version == last.get(ch):
                        continue
                    last[ch] = version
                    try:
                        outq.put_nowait((ch, version))
                    except asyncio.QueueFull:
                        pass  # client behind; drop

            async def poll_pump():
                """Slow watermark fallback for changes the bus missed."""
                while True:
                    for ch, vfn in channels.items():
                        try:
                            v = await asyncio.to_thread(vfn)
                        except Exception:
                            logger.exception(
                                "events_stream: watermark %s failed", ch,
                            )
                            continue
                        v_str = str(v)
                        if last.get(ch) is None:
                            last[ch] = v_str
                            continue
                        if v_str != last[ch]:
                            last[ch] = v_str
                            try:
                                outq.put_nowait((ch, v_str))
                            except asyncio.QueueFull:
                                pass
                    await asyncio.sleep(poll_every)

            push_task = asyncio.create_task(push_pump())
            poll_task = asyncio.create_task(poll_pump())
            try:
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        ch, v = await asyncio.wait_for(
                            outq.get(), timeout=heartbeat_every,
                        )
                        yield _sse(ch, v)
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
            finally:
                for t in (push_task, poll_task):
                    if not t.done():
                        t.cancel()

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/live/stream")
    async def live_stream(
        request: Request,
        limit: int = Query(30, ge=1, le=500),
        since: int = Query(0, ge=0),
    ):
        """Server-sent events stream of new chat messages. Replaces
        the old `hx-trigger="every 2s"` HTMX poll on the live widget
        and the /live page. The client opens one connection, gets the
        starting watermark from the initial page render, and receives
        only the rows that arrive after that.

        Cadence: 1 s server-side DB poll. When no new messages, sends
        a comment-line heartbeat to keep the connection alive through
        proxies that drop idle TCP. When messages arrive, renders
        them with the same partial template and pushes one `rows`
        event with the HTML fragment.

        Idle-stream cost is 0 template renders, just one tiny SQL
        `SELECT MAX(id)` per second. Compare to the old poll: ~30
        full-page partial renders / minute / page open."""
        async def gen():
            # Watermark — start from `since` if the caller passed one
            # (matches the largest id on their initial render); fall
            # back to "current latest" so a freshly opened SSE doesn't
            # replay the last hour of chat.
            watermark = since
            if watermark <= 0:
                try:
                    watermark = repo.latest_message_id()
                except Exception:
                    watermark = 0
            # Tell the client where we're picking up so it can compare
            # against its initial render and dedupe if needed.
            yield _sse("watermark", str(watermark))
            heartbeat_every = 15  # seconds
            poll_every = 1.0      # seconds
            ticks_since_heartbeat = 0
            while True:
                if await request.is_disconnected():
                    return
                try:
                    new_msgs = await asyncio.to_thread(
                        repo.recent_global_messages_after_id,
                        watermark, limit=limit,
                    )
                except Exception:
                    logger.exception("live_stream: db poll failed")
                    new_msgs = []
                if new_msgs:
                    # newest-first by query; push as-is and let the
                    # client prepend each row in order.
                    watermark = max(watermark, new_msgs[0].id)
                    ctx = _live_context(new_msgs)
                    try:
                        html = TEMPLATES.get_template(
                            "partials/live_rows.html"
                        ).render(ctx)
                    except Exception:
                        logger.exception("live_stream: template render failed")
                        html = ""
                    if html:
                        yield _sse("rows", html)
                    ticks_since_heartbeat = 0
                else:
                    ticks_since_heartbeat += 1
                    if ticks_since_heartbeat >= heartbeat_every:
                        # SSE comment line; not delivered to event
                        # listeners but keeps proxies from culling
                        # the connection.
                        yield ": ping\n\n"
                        ticks_since_heartbeat = 0
                await asyncio.sleep(poll_every)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ---------------- topics (merged into /insights as a sub-view) ----------------

    @app.get("/topics", response_class=HTMLResponse)
    async def topics_redirect(
        request: Request,
        partial: int = Query(0),
        status: str = Query("all"),
        q: str = Query(""),
    ):
        # Topics now lives as a sub-view of /insights. Forward HTMX
        # auto-refresh polls + bookmark visits through the merged route.
        return await insights_page(
            request, view="topics", partial=partial, status=status, q=q,
        )

    @app.post("/topics/mark-read", response_class=HTMLResponse)
    async def topics_mark_read(request: Request):
        repo.set_surface_ack("topics")
        return await insights_page(
            request, view="topics", partial=1, status="all", q="",
        )

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
