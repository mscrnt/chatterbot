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

    # Personal-dataset capture unlock — opt-in. Same shape as the bot
    # process: no-op unless the streamer enabled capture, ran
    # `chatterbot dataset setup`, and set CHATTERBOT_DATASET_PASSPHRASE
    # in the environment. Two processes (bot + dashboard) both unlock
    # independently — they share the same wrapped DEK in app_settings
    # so the unlocked DEK ends up identical.
    from ..dataset import try_unlock_at_startup
    try_unlock_at_startup(repo, llm)

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

    # Personal-dataset nav link visibility. Hidden by default to keep
    # the nav clean for installs that don't care about capture; shown
    # the moment the streamer opts in OR runs setup. Both cases mean
    # they're actively engaging with the feature and want easy access.
    def _dataset_nav_visible() -> bool:
        try:
            return (
                repo.dataset_capture_enabled()
                or bool(repo.get_app_setting("dataset_key_wrapped"))
            )
        except Exception:
            return False
    TEMPLATES.env.globals["dataset_nav_visible"] = _dataset_nav_visible

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
    # Takes twitch_status so the group-summary call can prefix the
    # prompt with authoritative CHANNEL CONTEXT — stops the LLM from
    # writing "Valheim, judging by the graphics" when the Helix poll
    # already knows the game.
    from ..transcript import TranscriptService
    transcript_service = TranscriptService(
        repo, llm, settings, obs=obs_status, twitch_status=twitch_status,
    )
    # Wire the talking-points provider so the transcript matcher can
    # auto-address them when the streamer speaks about them.
    transcript_service._talking_points_provider = lambda: (
        [tp for tp in insights.cache.talking_points
         if not (tp.point or "").lstrip().lower().startswith("skip:")]
        if insights.cache and insights.cache.talking_points else []
    )
    _bg_tasks: set[asyncio.Task] = set()

    async def _dataset_context_snapshot_loop():
        """Defer the dataset-loop import — installs without the
        optional `dataset` extra (no cryptography / zstandard) must
        still boot the dashboard. The loop body itself gates on
        `dataset_capture_enabled` + DEK presence, so it's idle if
        capture is off."""
        try:
            from ..dataset.loops import context_snapshot_loop
        except Exception:
            logger.exception("dataset extra missing — context_snapshot_loop disabled")
            return
        await context_snapshot_loop(repo, twitch_status=twitch_status)

    async def _dataset_retention_loop():
        try:
            from ..dataset.loops import retention_loop
        except Exception:
            logger.exception("dataset extra missing — retention_loop disabled")
            return
        await retention_loop(repo)

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
        # Open-questions LLM filter — runs the heuristic
        # `recent_questions` clusters through an LLM that has full
        # chat + transcript context, so already-answered or
        # directed-at-other-chatter asks get dropped before the
        # panel renders.
        _bg_tasks.add(
            asyncio.create_task(
                insights.open_questions_loop(),
                name="open_questions_loop",
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
        # Personal-dataset context snapshots — writes one self-
        # contained CONTEXT_SNAPSHOT every ~5 min while capture is
        # active so future fine-tune bundles don't need the full
        # chatters.db attached to make sense. No-op when capture is
        # off OR the DEK isn't loaded; both checked every iteration
        # so toggling capture on at runtime starts producing
        # snapshots without a restart.
        _bg_tasks.add(
            asyncio.create_task(
                _dataset_context_snapshot_loop(),
                name="dataset_context_snapshot_loop",
            )
        )
        # Daily retention/compaction — prunes events older than
        # `dataset_retention_days` (default 30, 0 = forever) and
        # trims the on-disk size to `dataset_retention_max_mb`
        # (default 5000, 0 = unbounded). Drops orphan shards along
        # the way. Same opt-in gates as the snapshot loop.
        _bg_tasks.add(
            asyncio.create_task(
                _dataset_retention_loop(),
                name="dataset_retention_loop",
            )
        )
        # Chat-lag auto-tuner — every ~10 min, re-run the
        # cross-correlation between transcript text and chat content
        # and silently update `chat_lag_seconds` if the result is
        # confident. No streamer action needed; converges over the
        # course of a stream. Set the interval to 0 in /settings →
        # Whisper to disable.
        _bg_tasks.add(
            asyncio.create_task(
                transcript_service.chat_lag_calibration_loop(),
                name="chat_lag_auto_tune",
            )
        )
        # Transcript-chunk embedding backfill — fills in vec_transcripts
        # for historical chunks that don't have an embedding yet.
        # Powers the /search "Streamer voice" tab. RETRIEVAL only;
        # live LLM calls don't read from the index.
        _bg_tasks.add(
            asyncio.create_task(
                transcript_service.transcript_embed_backfill_loop(),
                name="transcript_embed_backfill",
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
    # Expose key services on app.state so TestClient + diagnostic
    # tooling can introspect cache state without monkeypatching
    # the create_app closure's bindings. Production routes still
    # access these via closure references — app.state is purely a
    # secondary hook.
    app.state.insights_service = insights
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
        return await _render_chatters(request, q, sort, page, partial=partial)

    async def _render_chatters(
        request: Request, q: str, sort: str, page: int, *, partial: bool
    ) -> HTMLResponse:
        offset = (page - 1) * PAGE_SIZE
        # Pull list + count in parallel via to_thread. Both queries
        # are independent (one paginated, one COUNT(*)) and on a 50k+
        # message DB each was costing 400-600ms sequentially;
        # gathering halves perceived latency. asyncio.to_thread is
        # required: the route handler is async and the repo calls
        # are blocking sqlite work — running them inline pegs the
        # event loop and stalls /health, /transcript, etc.
        rows, total = await asyncio.gather(
            asyncio.to_thread(
                repo.list_chatters,
                query=q, sort=sort, limit=PAGE_SIZE, offset=offset,
            ),
            asyncio.to_thread(repo.count_chatters, query=q),
        )
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

    @app.get("/users/{twitch_id}/preview", response_class=HTMLResponse)
    async def user_preview(
        request: Request,
        twitch_id: str = PathParam(..., min_length=1),
    ):
        """Tiny popover for the chatter-pill hover preview. Avatar +
        soft-profile fields + last 1-2 LLM notes + last 1-2 messages
        + a 'open profile' link. Loaded lazily by the chatter_pill
        macro on first hover, cached in the DOM via HTMX `once`.

        Falls back to a minimal "user not found" partial when the
        twitch_id doesn't resolve — keeps merged-away or
        not-yet-seen ids from spamming 404s in the streamer's network
        tab on a hover."""
        user = repo.get_user(twitch_id)
        # Follow merge pointer so a hover on a child id surfaces the
        # canonical profile's data, not a stub. The pill's link still
        # points at the original id; the preview shows the parent
        # profile so the streamer sees the real notes / messages.
        if user and user.merged_into:
            parent = repo.get_user(user.merged_into)
            if parent is not None:
                user = parent
        if user is None:
            return TEMPLATES.TemplateResponse(
                request,
                "partials/_chatter_preview.html",
                {"user": None, "notes": [], "messages": []},
            )
        notes = repo.get_notes(user.twitch_id)[:2]
        messages = repo.get_messages(user.twitch_id, limit=2)
        return TEMPLATES.TemplateResponse(
            request,
            "partials/_chatter_preview.html",
            {
                "user": user,
                "notes": notes,
                "messages": messages,
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

    # Section layout + per-field metadata (labels, tooltips, help
    # text, input types, options, depends_on rules) live in
    # web/settings_meta.py. Adding a new editable setting:
    #   1) add the key to EDITABLE_SETTING_KEYS in config.py
    #   2) add a Field entry in settings_meta.FIELDS
    #   3) reference the key in a Section's `fields` list

    # Build the per-field metadata + defaults map ONCE at app
    # creation time, not per request. Both are static — `field_meta()`
    # walks settings_meta.FIELDS and `Settings.model_fields[k].default`
    # is set when the class is defined. Per-render the GET handler
    # was rebuilding both for ~70 keys (200ms-ish on a busy box per
    # the benchmark) when a single dict copy + one dict write per row
    # is all we actually need.
    from .settings_meta import field_meta as _field_meta
    from ..config import Settings as _SettingsCls
    _SETTINGS_FIELD_META: dict[str, dict] = {
        k: _field_meta(k) for k in EDITABLE_SETTING_KEYS
    }
    _SETTINGS_DEFAULTS: dict[str, object] = {}
    for _k in EDITABLE_SETTING_KEYS:
        _fi = _SettingsCls.model_fields.get(_k)
        if _fi is None:
            continue
        _d = _fi.default
        if repr(_d) == "PydanticUndefined":
            continue
        _SETTINGS_DEFAULTS[_k] = _d

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request, saved: int = Query(0)):
        live, db_sources = get_settings_with_sources()
        from .settings_meta import SECTIONS

        def _row(key: str) -> dict:
            value = getattr(live, key, None)
            is_bool = isinstance(value, bool)
            return {
                "key": key,
                "value": value,
                "is_set": bool(value) if not is_bool else True,
                "source": "db" if key in db_sources else "env",
                "meta": _SETTINGS_FIELD_META[key],
                "default": _SETTINGS_DEFAULTS.get(key),
            }

        # Flat key→row dict so the template looks up rows by name
        # without re-walking the section hierarchy.
        rows: dict[str, dict] = {
            k: _row(k) for k in EDITABLE_SETTING_KEYS
        }
        defaults = _SETTINGS_DEFAULTS
        form_values: dict[str, object] = {}
        for k, r in rows.items():
            v = r["value"]
            if isinstance(v, bool):
                form_values[k] = v
            elif v is None:
                # Fall back to the default rather than empty so the
                # input renders SOMETHING and Alpine's x-model has a
                # value to bind. Keeps audio buffer length etc. from
                # showing blank when the user hasn't customised it.
                d = defaults.get(k)
                if d is None or isinstance(d, bool):
                    form_values[k] = ""
                else:
                    form_values[k] = d
            else:
                form_values[k] = (
                    v if isinstance(v, (int, float, str)) else str(v)
                )

        from ..diagnose import make_github_issue_url
        # Build the Prompts tab state — one card per editable prompt,
        # current mode + guided values + custom text for each.
        # Cheap (in-memory dict + a few app_settings lookups). Only
        # rendered when the streamer switches to the Prompts tab.
        from ..llm import prompts as _prompts_mod
        prompt_defs = _prompts_mod.all_prompt_defs()
        prompts_card_state: dict = {}
        for pd in prompt_defs:
            prompts_card_state[pd.call_site] = {
                "mode": _prompts_mod.get_mode(pd.call_site, repo),
                "guided_values": _prompts_mod.get_guided_values(pd.call_site, repo),
                "custom_text": _prompts_mod.get_custom_text(pd.call_site, repo),
            }
        # Group by section in the order the registry declares so the
        # UI's section blocks render predictably.
        prompt_sections: list = []
        seen_sections: dict = {}
        for pd in prompt_defs:
            seen_sections.setdefault(pd.section, []).append(pd)
        for sec_id in _prompts_mod.SECTION_ORDER:
            if sec_id in seen_sections:
                prompt_sections.append((
                    sec_id,
                    _prompts_mod.SECTION_TITLES.get(sec_id, sec_id.title()),
                    seen_sections[sec_id],
                ))

        return TEMPLATES.TemplateResponse(
            request,
            "settings.html",
            {
                "sections": SECTIONS,
                "section_ids": [s["id"] for s in SECTIONS],
                "rows": rows,
                "form_values": form_values,
                "saved": bool(saved),
                "github_issue_url": make_github_issue_url(),
                # Initial state for the chat-lag calibrator partial:
                # current setting, no calibration result yet (the
                # streamer hits "Run" to populate it).
                "chat_lag_current": int(getattr(live, "chat_lag_seconds", 6)),
                "calibration": None,
                "chat_lag_lookback": 5,
                "chat_lag_lookback_options": (5, 15, 30),
                "chat_lag_auto_tuned_at": repo.get_app_setting("chat_lag_auto_tuned_at"),
                "chat_lag_auto_tuned_value": repo.get_app_setting("chat_lag_auto_tuned_value"),
                "chat_lag_auto_tune_enabled": int(
                    getattr(live, "chat_lag_auto_tune_interval_seconds", 600)
                ) > 0,
                # Prompts tab state — see the registry in
                # chatterbot/llm/prompts.py for what's editable.
                "prompt_sections": prompt_sections,
                "prompts_card_state": prompts_card_state,
                # Streamer-facts editor state — sits at the top of the
                # Prompts tab. Same shape the POST handler returns so
                # the HTMX swap behaves identically.
                **_build_streamer_facts_context(),
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
            "whisper_llm_match_enabled",
            "dataset_capture_enabled",
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

    # ============================================================
    # Streamer-customizable prompts.
    #
    # Each editable call site gets its own card on the Prompts tab.
    # Cards POST to /settings/prompts/<site> with the chosen mode
    # plus the appropriate payload (adlib values OR custom text).
    # The route returns the re-rendered card so HTMX can swap it
    # in place without a full page reload.
    #
    # Streamers who never visit this tab see no behavior change —
    # `resolve_prompt` falls back to the factory prompt when no
    # override is stored.
    # ============================================================

    def _render_prompt_card(
        request: Request,
        call_site: str,
        *,
        flash: str | None = None,
        flash_kind: str = "success",
    ) -> Response:
        """Render one prompt card with the current saved state +
        an optional flash banner. Used as the response from save /
        revert routes so the streamer sees their action confirmed
        without a full-page reload."""
        from ..llm import prompts as _prompts_mod
        pd = _prompts_mod.get_prompt_def(call_site)
        if pd is None:
            raise HTTPException(404, "prompt not found")
        return TEMPLATES.TemplateResponse(
            request,
            "partials/prompt_card.html",
            {
                "prompt": pd,
                "mode": _prompts_mod.get_mode(call_site, repo),
                "guided_values": _prompts_mod.get_guided_values(call_site, repo),
                "custom_text": _prompts_mod.get_custom_text(call_site, repo),
                "flash": flash,
                "flash_kind": flash_kind,
            },
        )

    @app.post("/settings/prompts/{call_site}", response_class=HTMLResponse)
    async def settings_prompts_save(request: Request, call_site: str):
        """Save mode + payload for one prompt card. The mode field
        determines which payload field is meaningful:
          - factory  → no payload, just the mode flip
          - guided   → form fields named `guided__<slot>`, packed
                       into a JSON dict
          - custom   → `custom` textarea contents"""
        from ..llm import prompts as _prompts_mod

        pd = _prompts_mod.get_prompt_def(call_site)
        if pd is None:
            raise HTTPException(404, "prompt not editable")

        form = await request.form()
        mode = (form.get("mode") or "factory").strip().lower()
        if not _prompts_mod.save_mode(call_site, mode, repo):
            return _render_prompt_card(
                request, call_site,
                flash=f"Invalid mode: {mode!r}.",
                flash_kind="error",
            )

        # Persist the mode-specific payload regardless of which
        # mode is currently active — so a streamer can switch
        # between modes without losing the values they typed in
        # the others.
        guided_values = {}
        for slot in pd.guided_slots:
            v = form.get(f"guided__{slot.name}")
            if v is not None:
                guided_values[slot.name] = str(v)
        if guided_values:
            _prompts_mod.save_guided_values(call_site, guided_values, repo)
        custom_payload = form.get("custom")
        if custom_payload is not None:
            _prompts_mod.save_custom_text(
                call_site, str(custom_payload), repo,
            )

        # Friendly mode-aware flash text — Guided / Custom take
        # effect on the next refresh of the affected feature, so
        # we explicitly tell the streamer their save IS persisted
        # but won't be visible until then.
        mode_flash = {
            "factory": "Reverted to the factory prompt.",
            "guided":  "Saved. Guided answers take effect on the next refresh.",
            "custom":  "Saved. Custom prompt takes effect on the next refresh.",
        }.get(mode, "Saved.")
        return _render_prompt_card(
            request, call_site, flash=mode_flash, flash_kind="success",
        )

    @app.post(
        "/settings/prompts/{call_site}/revert", response_class=HTMLResponse,
    )
    async def settings_prompts_revert(request: Request, call_site: str):
        """Wipe every override for a prompt and reset to factory.
        Idempotent — re-clicking is a no-op."""
        from ..llm import prompts as _prompts_mod
        if not _prompts_mod.revert_to_factory(call_site, repo):
            raise HTTPException(404, "prompt not editable")
        return _render_prompt_card(
            request, call_site,
            flash="Reverted to factory. Guided answers and custom text cleared.",
            flash_kind="success",
        )

    # Soft cap matching the loader's truncation point so the editor's
    # counter and the loader agree on what gets sent to the LLM.
    _STREAMER_FACTS_MAX_CHARS = 4000

    def _streamer_facts_paths() -> tuple[Path | None, str]:
        """Resolve the configured streamer-facts path. Returns
        `(absolute_path, display_path)` — `display_path` is the value
        as configured (relative or absolute) so the editor can show
        the streamer where they actually pointed it. `absolute_path`
        is None when the setting is empty / unset."""
        live = get_settings()
        configured = (
            getattr(live, "streamer_facts_path", "") or ""
        ).strip()
        if not configured:
            return None, ""
        p = Path(configured)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p, configured

    def _build_streamer_facts_context(
        *, flash: str | None = None, flash_kind: str = "success",
    ) -> dict:
        """Pull the editor's render context — current text + writable
        check + size + cap — in one place so the GET path on /settings
        and the POST handler return the same shape."""
        abs_path, display = _streamer_facts_paths()
        text = ""
        size = 0
        writable = False
        if abs_path is not None:
            try:
                if abs_path.exists() and abs_path.is_file():
                    text = abs_path.read_text(encoding="utf-8")
                    size = len(text.encode("utf-8"))
                    writable = True
                elif not abs_path.exists():
                    # File doesn't exist yet but we can still write to
                    # it as long as the parent directory is real and
                    # writable. Empty editor surfaces with a "save to
                    # create" affordance.
                    parent = abs_path.parent
                    writable = parent.is_dir()
                # Path resolves to a directory or other non-file thing
                # → leave writable=False so the editor surfaces the
                # warning state instead of silently overwriting.
            except (OSError, UnicodeDecodeError):
                writable = False
        return {
            "facts_path": str(abs_path) if abs_path else "",
            "facts_path_display": display,
            "facts_text": text,
            "facts_size": size,
            "facts_max": _STREAMER_FACTS_MAX_CHARS,
            "facts_writable": writable,
            "flash": flash,
            "flash_kind": flash_kind,
        }

    @app.post("/settings/streamer-facts", response_class=HTMLResponse)
    async def settings_streamer_facts_save(request: Request):
        """Write the streamer-facts file and re-render the editor card.

        On success, the InsightsService picks up the new content via
        its mtime-cached `_load_streamer_facts` on the next LLM call
        — no explicit cache flush needed. On failure (path unset,
        directory unwritable, payload too large), surface a flash and
        leave the file untouched."""
        abs_path, display = _streamer_facts_paths()
        if abs_path is None:
            ctx = _build_streamer_facts_context(
                flash=(
                    "No path configured. Set "
                    "streamer_facts_path in Settings → Insights first."
                ),
                flash_kind="error",
            )
            return TEMPLATES.TemplateResponse(
                request, "partials/streamer_facts_editor.html", ctx,
            )
        if abs_path.exists() and not abs_path.is_file():
            ctx = _build_streamer_facts_context(
                flash=(
                    f"Path {display!r} exists but is not a regular "
                    "file. Refusing to overwrite."
                ),
                flash_kind="error",
            )
            return TEMPLATES.TemplateResponse(
                request, "partials/streamer_facts_editor.html", ctx,
            )
        form = await request.form()
        content = (form.get("content") or "").replace("\r\n", "\n")
        # Hard cap = soft cap × 2 so the editor's counter going
        # slightly red doesn't reject a save outright; a runaway
        # paste 10× the cap does.
        if len(content) > _STREAMER_FACTS_MAX_CHARS * 2:
            ctx = _build_streamer_facts_context(
                flash=(
                    f"Content too long ({len(content)} chars). Trim "
                    f"to under {_STREAMER_FACTS_MAX_CHARS * 2} chars."
                ),
                flash_kind="error",
            )
            return TEMPLATES.TemplateResponse(
                request, "partials/streamer_facts_editor.html", ctx,
            )
        try:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content, encoding="utf-8")
        except OSError as e:
            ctx = _build_streamer_facts_context(
                flash=f"Couldn't write file: {e}",
                flash_kind="error",
            )
            return TEMPLATES.TemplateResponse(
                request, "partials/streamer_facts_editor.html", ctx,
            )
        ctx = _build_streamer_facts_context(
            flash=(
                f"Saved ({len(content)} chars). The next AI refresh "
                "picks up the change."
            ),
            flash_kind="success",
        )
        return TEMPLATES.TemplateResponse(
            request, "partials/streamer_facts_editor.html", ctx,
        )

    # ============================================================
    # Personal training dataset (opt-in capture system).
    #
    # Three browser-facing routes:
    #   GET  /dataset           — status page, no decryption.
    #   POST /dataset/setup     — generate + wrap a fresh DEK,
    #                              persist, flash recovery string.
    #   POST /dataset/export    — produce a .cbds bundle for
    #                              download.
    #
    # The capture machinery itself lives in chatterbot.dataset; the
    # routes here are thin wrappers that surface those CLI commands
    # in the dashboard. Bot/dashboard processes still need
    # CHATTERBOT_DATASET_PASSPHRASE in their env to actually unlock
    # the DEK at startup — the dashboard alone can't notify the bot
    # process about an unlock yet (slice 5+).
    # ============================================================

    def _dataset_status() -> dict:
        """Snapshot of the capture system's runtime state for the
        /dataset page. Read-only; never decrypts anything. Mirrors
        the shape `chatterbot dataset info` prints to stdout."""
        configured = bool(repo.get_app_setting("dataset_key_wrapped"))
        from .. import dataset as dataset_pkg
        from ..dataset import capture as _cap
        per_kind = {}
        total = 0
        total_bytes = 0
        try:
            total = repo.dataset_event_count()
            for k in (
                _cap.EVENT_LLM_CALL,
                _cap.EVENT_STREAMER_ACTION,
                _cap.EVENT_CONTEXT_SNAPSHOT,
            ):
                n = repo.dataset_event_count(kind=k)
                if n:
                    per_kind[k] = n
            with repo._cursor() as cur:  # noqa: SLF001 — small read
                cur.execute(
                    "SELECT COALESCE(SUM(byte_length), 0) AS s "
                    "FROM dataset_events"
                )
                row = cur.fetchone()
                total_bytes = int(row["s"]) if row else 0
        except Exception:
            logger.exception("dataset status query failed")
        # Suppress unused-import warning while keeping the lazy
        # import in this function (slice-4 surface keeps a reference
        # so a future call site doesn't have to re-add it).
        _ = dataset_pkg
        return {
            "configured": configured,
            "enabled": repo.dataset_capture_enabled(),
            "unlocked": repo.dataset_dek() is not None,
            "fingerprint": repo.get_app_setting("dataset_key_fingerprint") or "",
            "total_events": total,
            "total_bytes": total_bytes,
            "per_kind": per_kind,
        }

    @app.get("/dataset", response_class=HTMLResponse)
    async def dataset_page(
        request: Request,
        flash: str | None = Query(None),
        flash_kind: str | None = Query(None),
        recovery: str | None = Query(None),
    ):
        # Recent-events list — index-only (no decryption). Last 20
        # rows newest-first.
        recent = []
        try:
            rows = list(repo.iter_dataset_events())
            recent = list(reversed(rows[-20:]))
        except Exception:
            logger.exception("dataset: recent events query failed")
        return TEMPLATES.TemplateResponse(
            request, "dataset.html",
            {
                "status": _dataset_status(),
                "recent_events": recent,
                "flash": flash,
                "flash_kind": (flash_kind or "info"),
                # Setup flashes the recovery string ONCE via the
                # redirect querystring. The page renders it inside a
                # warning banner and tells the streamer to save it.
                "recovery_string": recovery,
            },
        )

    @app.post("/dataset/setup")
    async def dataset_setup(request: Request):
        """Generate a fresh DEK, wrap it under the streamer's
        passphrase, persist. Refuses to overwrite an existing wrapped
        DEK — the streamer would lose access to past events. Use
        the CLI's `--force` flag for that case (intentionally not
        exposed in the browser to avoid an accidental click)."""
        if repo.get_app_setting("dataset_key_wrapped"):
            return RedirectResponse(
                url="/dataset?flash=Dataset+is+already+initialised.+"
                    "Use+the+CLI+with+--force+to+rotate.&flash_kind=error",
                status_code=303,
            )
        form = await request.form()
        passphrase = (form.get("passphrase") or "").strip()
        confirm = (form.get("passphrase_confirm") or "").strip()
        if not passphrase or len(passphrase) < 6:
            return RedirectResponse(
                url="/dataset?flash=Passphrase+must+be+at+least+6+characters."
                    "&flash_kind=error",
                status_code=303,
            )
        if passphrase != confirm:
            return RedirectResponse(
                url="/dataset?flash=Passphrases+don%27t+match.&flash_kind=error",
                status_code=303,
            )

        try:
            from ..dataset import cipher
        except ImportError:
            return RedirectResponse(
                url="/dataset?flash=The+dataset+extra+isn%27t+installed."
                    "+Run+%60uv+sync+--extra+dataset%60+first.&flash_kind=error",
                status_code=303,
            )

        # Argon2id is intentionally slow (100s of ms) — run it off the
        # event loop so /dataset/setup doesn't block other requests.
        def _do_setup() -> tuple[str, str]:
            dek = cipher.generate_dek()
            wrapped = cipher.wrap_dek(dek, passphrase)
            fingerprint = cipher.fingerprint_dek(dek)
            recovery = cipher.dek_to_recovery_string(dek)
            repo.set_app_setting("dataset_key_wrapped", wrapped.to_json())
            repo.set_app_setting("dataset_key_fingerprint", fingerprint)
            if repo.get_app_setting("dataset_capture_enabled") is None:
                repo.set_app_setting("dataset_capture_enabled", "false")
            # Install the unlocked DEK so this dashboard process can
            # immediately capture without restarting. The bot process
            # (separate) still needs CHATTERBOT_DATASET_PASSPHRASE on
            # its next restart.
            repo.set_dataset_dek(dek)
            llm.attach_dataset_capture(repo) if hasattr(
                llm, "attach_dataset_capture"
            ) else None
            return fingerprint, recovery

        try:
            _, recovery = await asyncio.to_thread(_do_setup)
        except Exception:
            logger.exception("dataset setup failed")
            return RedirectResponse(
                url="/dataset?flash=Setup+failed+%E2%80%94+see+server+logs."
                    "&flash_kind=error",
                status_code=303,
            )

        # URL-encode the recovery string carefully — it has hyphens
        # and base32 chars only, all URL-safe, but go through quote
        # for safety.
        from urllib.parse import quote
        return RedirectResponse(
            url=(
                "/dataset"
                "?flash=Dataset+key+generated.+Save+the+recovery+string+below."
                f"&flash_kind=success&recovery={quote(recovery)}"
            ),
            status_code=303,
        )

    @app.post("/dataset/export")
    async def dataset_export(request: Request):
        """Decrypt every indexed event under the streamer's
        passphrase, repack into a single passphrase-protected .cbds
        bundle, return as a download. Mirrors `chatterbot dataset
        export` from the CLI."""
        form = await request.form()
        passphrase = (form.get("passphrase") or "").strip()
        since = (form.get("since") or "").strip() or None
        until = (form.get("until") or "").strip() or None
        # Browser checkbox — anonymises chatter names in the bundle
        # using `redactor.build_plan`. Defaults to off so a streamer
        # who exports for personal fine-tuning gets verbatim data.
        redact_users = form.get("redact_users") is not None
        if not passphrase:
            return RedirectResponse(
                url="/dataset?flash=Passphrase+required.&flash_kind=error",
                status_code=303,
            )

        wrapped_raw = repo.get_app_setting("dataset_key_wrapped")
        if not wrapped_raw:
            return RedirectResponse(
                url="/dataset?flash=No+dataset+key+%E2%80%94+run+setup+first."
                    "&flash_kind=error",
                status_code=303,
            )

        try:
            from ..dataset import cipher
        except ImportError:
            return RedirectResponse(
                url="/dataset?flash=The+dataset+extra+isn%27t+installed."
                    "&flash_kind=error",
                status_code=303,
            )

        def _do_export() -> bytes | None:
            """Run the export pipeline off the event loop. Returns
            the bundle bytes on success, None on a recoverable
            error (wrong passphrase / no events). Reuses the CLI's
            cmd_export logic by writing to a temp file then reading
            back — the function already handles the encrypt-and-
            tar pipeline correctly, no need to duplicate it."""
            try:
                wrapped = cipher.WrappedDEK.from_json(wrapped_raw)
                cipher.unwrap_dek(wrapped, passphrase)  # validates passphrase
            except Exception:
                return None
            import tempfile
            from pathlib import Path
            from ..dataset.cli import cmd_export

            class _S:  # cmd_export reads .db_path / .ollama_embed_dim
                db_path = repo.db_path
                ollama_embed_dim = settings.ollama_embed_dim

            # cmd_export prompts for the passphrase via getpass —
            # patch it for this single invocation. The patch is
            # local: re-imports of the cli module after this call
            # see the original prompt function back.
            from ..dataset import cli as _cli
            saved = _cli._prompt_passphrase
            _cli._prompt_passphrase = lambda *a, **k: passphrase
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".cbds", delete=False,
                ) as tf:
                    out_path = Path(tf.name)
                rc = cmd_export(
                    _S(), out_path,
                    since=since, until=until,
                    redact_users=redact_users,
                )
                if rc != 0 or not out_path.exists():
                    out_path.unlink(missing_ok=True)
                    return None
                data = out_path.read_bytes()
                out_path.unlink(missing_ok=True)
                return data
            finally:
                _cli._prompt_passphrase = saved

        try:
            bundle_bytes = await asyncio.to_thread(_do_export)
        except Exception:
            logger.exception("dataset export failed")
            return RedirectResponse(
                url="/dataset?flash=Export+failed+%E2%80%94+see+server+logs."
                    "&flash_kind=error",
                status_code=303,
            )
        if bundle_bytes is None:
            return RedirectResponse(
                url="/dataset?flash=Export+failed+%E2%80%94+wrong+passphrase+"
                    "or+no+events+in+range.&flash_kind=error",
                status_code=303,
            )

        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        filename = f"chatterbot-dataset-{stamp}.cbds"
        return Response(
            content=bundle_bytes,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    _CHAT_LAG_LOOKBACK_OPTIONS = (5, 15, 30)

    @app.post(
        "/settings/calibrate-chat-lag",
        response_class=HTMLResponse,
    )
    async def settings_calibrate_chat_lag(request: Request):
        """Run chat-lag cross-correlation. Default lookback dropped
        to 5 min so the calibration is useful within a couple minutes
        of going live; streamer can widen via the panel selector when
        chat is too thin to clear the activity guard at 5 min.

        `lookback` is read from EITHER the URL query string (the
        manual Run button hardcodes it there) OR the form body (the
        dropdown sends it via hx-include='this'). Falling back across
        both keeps a single endpoint serving both call shapes."""
        from ..latency import calibrate_chat_lag
        lookback_raw = request.query_params.get("lookback")
        if not lookback_raw:
            try:
                form = await request.form()
                lookback_raw = form.get("lookback")
            except Exception:
                lookback_raw = None
        try:
            lookback = int(lookback_raw or 5)
        except (TypeError, ValueError):
            lookback = 5
        # Snap to the supported options so a tampered value can't
        # trigger an unbounded run.
        lookback = min(_CHAT_LAG_LOOKBACK_OPTIONS, key=lambda x: abs(x - lookback))
        try:
            result = await asyncio.to_thread(
                calibrate_chat_lag, repo, lookback_minutes=lookback,
            )
        except Exception as e:
            logger.exception("chat-lag calibration failed")
            result = {
                "ok": False,
                "reason": f"{type(e).__name__}: {e}",
                "samples": {"chunks": 0, "messages": 0},
                "lookback_minutes": lookback,
                "offsets": [],
                "best_offset": None,
                "second_best_offset": None,
            }
        live, _ = get_settings_with_sources()
        return TEMPLATES.TemplateResponse(
            request,
            "partials/chat_lag_calibrator.html",
            {
                "calibration": result,
                "chat_lag_current": int(getattr(live, "chat_lag_seconds", 6)),
                "chat_lag_lookback": lookback,
                "chat_lag_lookback_options": _CHAT_LAG_LOOKBACK_OPTIONS,
                "chat_lag_auto_tuned_at": repo.get_app_setting("chat_lag_auto_tuned_at"),
                "chat_lag_auto_tuned_value": repo.get_app_setting("chat_lag_auto_tuned_value"),
                "chat_lag_auto_tune_enabled": int(getattr(live, "chat_lag_auto_tune_interval_seconds", 600)) > 0,
            },
        )

    @app.post(
        "/settings/calibrate-chat-lag/apply",
        response_class=HTMLResponse,
    )
    async def settings_calibrate_chat_lag_apply(
        request: Request,
        offset: int = Query(..., ge=0, le=60),
        lookback: int = Query(5, ge=1, le=60),
    ):
        """Save the picked offset to chat_lag_seconds and re-render
        the calibrator partial so the streamer sees the new "current"
        value reflected immediately. Re-runs calibration so the bar
        chart is preserved across the swap."""
        repo.set_app_setting("chat_lag_seconds", str(int(offset)))
        lookback = min(_CHAT_LAG_LOOKBACK_OPTIONS, key=lambda x: abs(x - lookback))
        from ..latency import calibrate_chat_lag
        try:
            result = await asyncio.to_thread(
                calibrate_chat_lag, repo, lookback_minutes=lookback,
            )
        except Exception:
            logger.exception("chat-lag calibration after apply failed")
            result = None
        live, _ = get_settings_with_sources()
        return TEMPLATES.TemplateResponse(
            request,
            "partials/chat_lag_calibrator.html",
            {
                "calibration": result,
                "chat_lag_current": int(getattr(live, "chat_lag_seconds", 6)),
                "chat_lag_lookback": lookback,
                "chat_lag_lookback_options": _CHAT_LAG_LOOKBACK_OPTIONS,
                "chat_lag_auto_tuned_at": repo.get_app_setting("chat_lag_auto_tuned_at"),
                "chat_lag_auto_tuned_value": repo.get_app_setting("chat_lag_auto_tuned_value"),
                "chat_lag_auto_tune_enabled": int(getattr(live, "chat_lag_auto_tune_interval_seconds", 600)) > 0,
            },
        )

    @app.post(
        "/settings/chat-lag/save",
        response_class=HTMLResponse,
    )
    async def settings_chat_lag_save(request: Request):
        """Save manual override + auto-tune toggle from the calibrator
        panel without going through the full /settings form. Two
        controls, both optional in the body:

          - chat_lag_seconds: int (manual override)
          - chat_lag_auto_tune: 'on' | absent (checkbox)

        Saving a manual override clears the auto-tuned-at timestamp
        so the panel doesn't keep showing a stale "auto-tuned" pill
        next to a streamer-overridden value. Toggling auto-tune
        just sets the interval to 0 (disabled) or back to its
        previous value (re-enabled) — we cache the last enabled
        interval in `chat_lag_auto_tune_last_enabled` so re-enabling
        respects whatever the streamer had configured."""
        form = await request.form()
        manual = form.get("chat_lag_seconds")
        auto_on = form.get("chat_lag_auto_tune") is not None
        if manual is not None and str(manual).strip():
            try:
                v = max(0, min(60, int(str(manual).strip())))
            except (TypeError, ValueError):
                v = None
            if v is not None:
                repo.set_app_setting("chat_lag_seconds", str(v))
                # Streamer's manual pick overrides any prior auto-tune.
                repo.delete_app_setting("chat_lag_auto_tuned_at")
                repo.delete_app_setting("chat_lag_auto_tuned_value")
        # Toggle: store last-enabled interval before disabling so we
        # can restore it (defaults to 600 if never set).
        cur_interval = repo.get_app_setting("chat_lag_auto_tune_interval_seconds")
        try:
            cur_interval_n = int(cur_interval) if cur_interval else 600
        except (TypeError, ValueError):
            cur_interval_n = 600
        if auto_on and cur_interval_n == 0:
            last_enabled = repo.get_app_setting(
                "chat_lag_auto_tune_last_enabled",
            )
            try:
                restore = int(last_enabled) if last_enabled else 600
            except (TypeError, ValueError):
                restore = 600
            repo.set_app_setting(
                "chat_lag_auto_tune_interval_seconds", str(restore),
            )
        elif not auto_on and cur_interval_n != 0:
            repo.set_app_setting(
                "chat_lag_auto_tune_last_enabled", str(cur_interval_n),
            )
            repo.set_app_setting(
                "chat_lag_auto_tune_interval_seconds", "0",
            )
        live, _ = get_settings_with_sources()
        return TEMPLATES.TemplateResponse(
            request,
            "partials/chat_lag_calibrator.html",
            {
                "calibration": None,
                "chat_lag_current": int(getattr(live, "chat_lag_seconds", 6)),
                "chat_lag_lookback": 5,
                "chat_lag_lookback_options": _CHAT_LAG_LOOKBACK_OPTIONS,
                "chat_lag_auto_tuned_at": repo.get_app_setting("chat_lag_auto_tuned_at"),
                "chat_lag_auto_tuned_value": repo.get_app_setting("chat_lag_auto_tuned_value"),
                "chat_lag_auto_tune_enabled": int(getattr(live, "chat_lag_auto_tune_interval_seconds", 600)) > 0,
            },
        )

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
    async def diagnose(
        with_recent_activity: int = Query(0),
        anonymize: int = Query(0),
    ):
        """Build a privacy-safe .cbreport zip and stream it back as a download.
        See diagnose.py for what's in / out of the bundle.

        `anonymize=1` is meaningful only with `with_recent_activity=1` —
        runs the dataset redactor over the activity slice so the
        bundle can be shared without revealing chatter usernames."""
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
            anonymize_recent_activity=bool(anonymize) and bool(with_recent_activity),
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
            # General chat questions — LLM-curated OPEN questions
            # only. The heuristic `recent_questions` pass clusters
            # `?`-bearing chat by token overlap and excludes @-
            # mentions / Twitch reply rows; the background
            # `open_questions_loop` then runs that pool through the
            # LLM with full chat + streamer-voice context to drop
            # already-answered, rhetorical, or directed-at-another-
            # chatter asks. The cache exposes the same dict shape
            # the template was reading (question / count / drivers /
            # latest_ts / last_msg_id), so no template change needed.
            "chat_questions": insights.open_questions_cache.questions,
            # Per-cluster dismissal state — kind='chat_question',
            # item_key=str(last_msg_id). When a fresh ask comes in
            # the cluster's last_msg_id changes so the dismissal
            # naturally stops applying and the question re-surfaces.
            "cq_states": repo.get_insight_states("chat_question"),
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

    @app.get("/modals/subject/{slug}", response_class=HTMLResponse)
    async def modal_subject(request: Request, slug: str):
        """Render the engaging-subject modal body. Replaces the
        slice-7 `/insights/subject/{slug}/expand` inline-expand
        route — the streamer now opens a focused modal instead of
        unfolding card content in place.

        The modal is built from cached state (subject metadata +
        cited messages); the AI talking-points section is loaded
        async on first paint via /insights/subject/{slug}/talking-points
        so the modal opens instantly even when the LLM call is
        slow."""
        subject = next(
            (s for s in insights.subjects_cache.subjects if s.slug == slug),
            None,
        )
        if subject is None:
            raise HTTPException(404, "subject not found in cache")
        # Cited messages = exactly what the LLM grounded the subject
        # in. Falls back to a name-based lookup when msg_ids is
        # empty (defensive — slice-7 always populates it).
        if getattr(subject, "msg_ids", None):
            messages = await asyncio.to_thread(
                repo.get_messages_by_ids, list(subject.msg_ids),
            )
        elif subject.drivers:
            within = int(getattr(
                settings, "engaging_subjects_lookback_minutes", 20,
            ))
            messages = repo.messages_for_names_within(
                subject.drivers, within_minutes=within, limit=40,
            )
        else:
            messages = []
        # Resolve drivers to twitch_ids (alias-aware) so the modal's
        # driver pills can link to /users/<twitch_id>. Mirrors the
        # thread-modal route's pattern.
        driver_links = []
        for name in subject.drivers:
            user = repo.find_user_by_alias_or_name(name)
            driver_links.append({
                "name": name,
                "user_id": user.twitch_id if user else None,
            })
        return TEMPLATES.TemplateResponse(
            request,
            "modals/_engaging_subject.html",
            {
                "subject": subject,
                "messages": messages,
                "driver_links": driver_links,
            },
        )

    @app.get(
        "/insights/subject/{slug}/talking-points",
        response_class=HTMLResponse,
    )
    async def subject_talking_points(request: Request, slug: str):
        """On-demand talking-points partial. Fired by the modal's
        HTMX `intersect once` trigger so the modal opens instantly
        and this lazy-loads in the background.

        Cached on the persistent subject for as long as the subject
        hasn't been resurfaced (= same chat context) — re-opening
        the modal within the same engaging-subjects refresh cycle
        is free, no LLM call."""
        try:
            points, error = await insights.generate_subject_talking_points(slug)
        except Exception:
            logger.exception("subject talking-points: unexpected error")
            points, error = [], "internal error generating points"
        return TEMPLATES.TemplateResponse(
            request,
            "partials/engaging_subject_talking_points.html",
            {"points": points, "error": error, "slug": slug},
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

    @app.get("/modals/question/{last_msg_id}", response_class=HTMLResponse)
    async def modal_question(request: Request, last_msg_id: int):
        """Open-question modal body. Built from the cached question
        entry plus a freshly-pulled chat window (so the streamer sees
        what's been said since the last refresh). The answer-angles
        section loads async via /insights/question/{id}/answer-angles
        so the modal opens instantly even when the LLM call is slow.

        Mirrors the engaging-subject modal pattern."""
        entry = next(
            (q for q in insights.open_questions_cache.questions
             if q.last_msg_id == last_msg_id),
            None,
        )
        if entry is None:
            raise HTTPException(404, "question not found in cache")

        window_min = int(getattr(
            settings, "open_questions_lookback_minutes",
            insights.OPEN_QUESTIONS_LOOKBACK_MINUTES,
        ))
        try:
            chat_msgs = await asyncio.to_thread(
                repo.recent_messages,
                limit=80, within_minutes=window_min,
            )
        except Exception:
            logger.exception("question modal: recent_messages failed")
            chat_msgs = []

        # Verbatim asks = messages from the cluster's drivers within
        # the window that contain '?'. The cluster identity is by
        # token-overlap not exact ids, so we re-derive here rather
        # than carrying message_ids on OpenQuestionEntry.
        driver_ids = {d.user_id for d in entry.drivers}
        verbatim_asks = [
            m for m in chat_msgs
            if m.user_id in driver_ids and "?" in (m.content or "")
        ][-12:]
        verbatim_ids = {m.id for m in verbatim_asks}

        return TEMPLATES.TemplateResponse(
            request,
            "modals/_open_question.html",
            {
                "entry": entry,
                "chat_msgs": chat_msgs,
                "verbatim_asks": verbatim_asks,
                "verbatim_ids": verbatim_ids,
                "window_min": window_min,
            },
        )

    @app.get(
        "/insights/question/{last_msg_id}/answer-angles",
        response_class=HTMLResponse,
    )
    async def question_answer_angles(request: Request, last_msg_id: int):
        """Async-loaded answer angles for the open-question modal.
        Fired by the modal's HTMX `intersect once` trigger so the
        modal opens instantly and this lazy-loads in the background.

        Cached on `_question_angles_cache` keyed by last_msg_id —
        re-opening the modal within the same question identity is
        free, no LLM call."""
        try:
            angles, error = await insights.generate_question_answer_angles(
                last_msg_id,
            )
        except Exception:
            logger.exception("question answer-angles: unexpected error")
            angles, error = [], "internal error generating angles"
        return TEMPLATES.TemplateResponse(
            request,
            "partials/open_question_answer_angles.html",
            {
                "angles": angles, "error": error,
                "last_msg_id": last_msg_id,
            },
        )

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
        # Optional capture-time header — when the OBS audio_client
        # buffered chunks during a dashboard outage, this is the ISO
        # time the audio was actually recorded. Used by the buffer to
        # timestamp the resulting transcript chunk against actual
        # audio time instead of wall-clock arrival time, so a chunk
        # that took 30s to be POSTed still appears at its real moment
        # on the live transcript timeline. Older audio_client
        # versions don't send this header — falls back to now()
        # transparently.
        captured_at = (request.headers.get("x-captured-at") or "").strip() or None
        body = await request.body()
        if body:
            await transcript_service.ingest_chunk(
                body, sr, captured_at=captured_at,
            )
        return Response(status_code=204)

    @app.get("/modals/transcript-group/{group_id}", response_class=HTMLResponse)
    async def modal_transcript_group(request: Request, group_id: int):
        """Detail view for one transcript group — summary + the
        underlying utterances clipped to the group's id range, plus
        the OBS screenshots captured during the group's window AND
        the exact chat messages the LLM saw when building the
        summary.

        Chat-context message IDs are persisted with the group at
        summary-write time (transcript_groups.context_message_ids).
        Older groups created before that column existed return
        empty `chat_messages` — the modal renders without a chat
        section in that case."""
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
        # Hydrate the persisted chat IDs into Message rows. Empty list
        # for older groups (pre-feature) — template renders cleanly.
        chat_messages: list = []
        if group.context_message_ids:
            chat_messages = await asyncio.to_thread(
                repo.get_messages_by_ids, list(group.context_message_ids),
            )
        # Window duration for the modal header. start_ts/end_ts are
        # ISO strings on the dataclass; parse defensively so a
        # malformed row doesn't crash the modal.
        duration_s = 0
        try:
            _start = datetime.fromisoformat(group.start_ts)
            _end = datetime.fromisoformat(group.end_ts)
            duration_s = max(0, int((_end - _start).total_seconds()))
        except (TypeError, ValueError):
            pass
        return TEMPLATES.TemplateResponse(
            request, "modals/_transcript_group.html",
            {
                "group": group,
                "chunks": chunks,
                "screenshots": screenshots,
                "chat_messages": chat_messages,
                "duration_s": duration_s,
            },
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

    @app.get("/debug/transcript-prompt")
    async def debug_transcript_prompt(
        group_id: int | None = Query(
            None,
            description="Inspect a specific past group; default = next "
            "(unsummarised) chunks the LLM would see right now.",
        ),
        include_image: bool = Query(
            False,
            description="Set true to also stitch + base64 the screenshot "
            "grid. Off by default since the b64 payload is large.",
        ),
    ):
        """Inspect what the group-summary LLM is actually being told.

        Default: shows the prompt that WOULD be sent for the next pass
        (unsummarised chunks past the watermark). Pass `?group_id=N`
        to inspect what was sent for an existing group.

        Useful for triaging "the LLM keeps calling it the wrong game"
        bugs — if the response shows `known_game=""` or
        `channel_context=""`, the Helix poll isn't wired or offline,
        not a prompting issue. If `known_game="Log Riders"` and the
        LLM still says Fall Guys, that's purely a vision-bias problem
        and the prompt itself is fine.
        """
        if group_id is not None:
            grp = await asyncio.to_thread(repo.get_transcript_group, group_id)
            if grp is None:
                raise HTTPException(status_code=404, detail="group not found")
            chunks = await asyncio.to_thread(
                repo.transcript_chunks_in_id_range,
                grp.first_chunk_id, grp.last_chunk_id,
            )
            source = f"group_id={group_id}"
        else:
            watermark = await asyncio.to_thread(
                repo.latest_transcript_group_last_chunk_id,
            )
            chunks = await asyncio.to_thread(
                repo.list_transcripts_after_id, watermark, limit=400,
            )
            source = f"unsummarised after watermark={watermark}"

        if not chunks:
            return JSONResponse({
                "ok": False,
                "reason": "no chunks to summarise",
                "source": source,
            })

        bundle = await transcript_service.build_group_summary_prompt(
            chunks, include_image=include_image,
        )
        # Drop the (large) base64 image payload from the JSON response
        # unless explicitly requested. Length is shown either way so
        # the streamer sees "yes, an image is being sent."
        grid_b64 = bundle.pop("screenshot_grid_b64", None)
        bundle["screenshot_grid_b64_len"] = len(grid_b64) if grid_b64 else 0
        if include_image and grid_b64:
            bundle["screenshot_grid_b64"] = grid_b64
        bundle["source"] = source
        bundle["chunk_count"] = len(chunks)
        bundle["chunk_ids"] = [chunks[0].id, chunks[-1].id]
        bundle["chunk_ts_range"] = [chunks[0].ts, chunks[-1].ts]
        return JSONResponse(bundle)

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

    # ---------------- live chat (widget + full page) ----------------

    # ---------------- semantic message search ----------------

    @app.get("/search", response_class=HTMLResponse)
    async def search_page(
        request: Request,
        q: str = Query(""),
        partial: int = Query(0),
        k: int = Query(20, ge=1, le=100),
        scope: str = Query(
            "chat",
            description="'chat' = chat messages, 'transcripts' = "
            "streamer voice (whisper utterances).",
        ),
    ):
        # `scope` chooses which embedding index to search against.
        # The default 'chat' preserves old URLs; 'transcripts' routes
        # to vec_transcripts so the streamer can answer "when have I
        # talked about X?". Both share a single query embedding —
        # nomic-embed-text indexes both halves so the same q_vec works.
        scope = scope if scope in ("chat", "transcripts") else "chat"
        message_results: list[tuple] = []
        transcript_results: list[tuple] = []
        error: str | None = None
        if q.strip():
            try:
                q_vec = await llm.embed(q.strip())
                if scope == "transcripts":
                    transcript_results = repo.search_transcripts(q_vec, k=k)
                else:
                    message_results = repo.search_global_messages(q_vec, k=k)
            except Exception as e:
                logger.exception("search: embedding/query failed")
                error = f"{type(e).__name__}: {e}"
        msg_indexed, msg_total = repo.messages_embedding_coverage()
        tx_indexed, tx_total = repo.transcripts_embedding_coverage()
        ctx = {
            "q": q,
            "scope": scope,
            "results": message_results,            # legacy template var
            "transcript_results": transcript_results,
            "indexed": msg_indexed,                # legacy template var
            "total": msg_total,
            "msg_indexed": msg_indexed,
            "msg_total": msg_total,
            "tx_indexed": tx_indexed,
            "tx_total": tx_total,
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

    @app.get(
        "/insights/high-impact/{thread_id}/openers",
        response_class=HTMLResponse,
    )
    async def high_impact_openers(
        request: Request,
        thread_id: int,
        live_drivers: str = Query(""),
    ):
        """Async-loaded openers for the high-impact modal. Fired by
        the modal's HTMX `intersect once` trigger so the modal opens
        instantly and this lazy-loads in the background.

        `live_drivers` is the comma-separated twitch_id list of
        chatters currently in chat that the modal computed at render
        time — passed through so the cache key (which factors in the
        audience) stays consistent between the modal render and the
        opener lookup."""
        ids = [d for d in live_drivers.split(",") if d.strip()]
        try:
            openers, error = await insights.generate_high_impact_openers(
                thread_id, ids,
            )
        except Exception:
            logger.exception("high-impact openers: unexpected error")
            openers, error = [], "internal error generating openers"
        return TEMPLATES.TemplateResponse(
            request,
            "partials/high_impact_openers.html",
            {
                "openers": openers, "error": error,
                "thread_id": thread_id,
            },
        )

    @app.get("/modals/high-impact/{thread_id}", response_class=HTMLResponse)
    async def modal_high_impact(request: Request, thread_id: int):
        """Dedicated modal for the "What to say next" card.

        Differs from the generic thread modal in framing: organized
        around the chatters who are LIVE RIGHT NOW and would re-engage
        with this topic. Each live chatter gets a section showing
        their own past messages on the topic so the streamer can
        address them by name with what they actually argued.

        Past discussions / historical drivers ride at the bottom as
        secondary context.
        """
        thread = repo.get_thread(thread_id)
        if not thread:
            raise HTTPException(404, "thread not found")
        members = repo.get_thread_members(thread_id)

        # Cross-reference active set against this thread's drivers
        # — the same logic used by list_high_impact_subjects when
        # ranking, but for ONE thread.
        active_window = int(getattr(
            settings, "high_impact_active_within_minutes", 30,
        ))
        active_ids = await asyncio.to_thread(
            repo.active_chatter_ids, within_minutes=active_window,
        )
        live_drivers: list[dict] = []
        other_drivers: list[dict] = []
        seen_ids: set[str] = set()
        for name in thread.drivers:
            user = repo.find_user_by_alias_or_name(name)
            if user is None:
                other_drivers.append({"name": name, "user_id": None})
                continue
            if user.twitch_id in seen_ids:
                continue
            seen_ids.add(user.twitch_id)
            entry = {"name": user.name, "user_id": user.twitch_id}
            if user.twitch_id in active_ids:
                live_drivers.append(entry)
            else:
                other_drivers.append(entry)

        # Per-live-driver quotes from inside this thread's snapshot
        # windows. Cap at 3 quotes per chatter so the modal stays
        # scannable; ordered newest-first so the most recent take
        # is the first thing the streamer sees.
        live_driver_quotes: dict[str, list] = {}
        for d in live_drivers:
            live_driver_quotes[d["user_id"]] = await asyncio.to_thread(
                repo.thread_messages_for_user,
                thread_id, d["user_id"], limit=3,
            )

        return TEMPLATES.TemplateResponse(
            request,
            "modals/_high_impact.html",
            {
                "thread": thread,
                "members": members,
                "live_drivers": live_drivers,
                "other_drivers": other_drivers,
                "live_driver_quotes": live_driver_quotes,
                "active_window": active_window,
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
