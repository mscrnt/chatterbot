"""CLI entrypoint.

Usage:
  chatterbot bot         # silent listener: TwitchIO + summarizer + SE + topics loop
  chatterbot tui         # streamer-only Textual viewer
  chatterbot dashboard   # streamer-only FastAPI dashboard
  chatterbot diagnose    # write a privacy-safe .cbreport bundle for bug reports

The first three are independent processes — they share the SQLite DB via WAL
so they can run simultaneously. `diagnose` is a one-shot.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from .bot import ChatterListener
from .config import Settings, get_settings
from .diagnose import build_diagnostic_bundle, default_bundle_filename
from .discord_bot import DiscordListener
from .llm.ollama_client import OllamaClient
from .logging_setup import setup_logging
from .moderator import Moderator
from .obs import OBSStatusService
from .repo import ChatterRepo
from .streamelements import StreamElementsListener
from .summarizer import Summarizer
from .tui import run_tui
from .youtube import YouTubeListener

logger = logging.getLogger(__name__)


_PID_FILE = Path("data/.bot.pid")


def _write_pid_file() -> None:
    try:
        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(str(__import__("os").getpid()))
    except OSError:
        logger.warning("could not write %s — restart-bot button won't find this process", _PID_FILE)


def _clear_pid_file() -> None:
    try:
        _PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


async def run_bot(settings: Settings) -> None:
    if not settings.twitch_oauth_token or not settings.twitch_channel:
        logger.error("missing TWITCH_OAUTH_TOKEN or TWITCH_CHANNEL — refusing to start")
        return

    _write_pid_file()

    repo = ChatterRepo(settings.db_path, embed_dim=settings.ollama_embed_dim)
    llm = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        embed_model=settings.ollama_embed_model,
    )

    if not await llm.health_check():
        logger.warning(
            "ollama health check failed at %s — starting anyway",
            settings.ollama_base_url,
        )

    summarizer = Summarizer(repo, llm, settings)
    listener = ChatterListener(settings, repo, summarizer)
    se = StreamElementsListener(repo, settings)
    # OBS is the authoritative "are we streaming?" signal. The bot process
    # owns its own poller (the dashboard has a separate one for the nav)
    # so YouTube + other quota-sensitive pollers can pause when offline.
    obs = OBSStatusService(settings)

    # One-time backfill of any pre-existing topic_snapshots into the
    # topic_threads index. Idempotent — only operates on snapshots that
    # don't yet have thread members. Non-fatal: bot keeps running on error.
    try:
        n = await summarizer._threader.backfill()
        if n:
            logger.info("threader: backfilled %d existing snapshots into threads", n)
    except Exception:
        logger.exception("threader: backfill failed — clustering will catch up on the next snapshot")

    tasks: list[asyncio.Task] = [
        asyncio.create_task(listener.start(), name="twitch_listener"),
        asyncio.create_task(summarizer.idle_loop(), name="summarizer_idle"),
        asyncio.create_task(summarizer.topics_loop(), name="summarizer_topics"),
    ]
    if settings.message_embed_interval_seconds > 0:
        tasks.append(
            asyncio.create_task(summarizer.embed_loop(), name="message_embedder")
        )
    if settings.obs_enabled:
        tasks.append(asyncio.create_task(obs.poll_loop(), name="obs_poll"))
        # End-of-stream recap: watch OBS state, fire LLM debrief on
        # streaming → not-streaming transition. No-op if OBS is disabled.
        tasks.append(asyncio.create_task(
            summarizer.recap_loop(obs), name="recap_loop"
        ))
    if settings.streamelements_enabled:
        tasks.append(asyncio.create_task(se.run(), name="streamelements"))
    if settings.mod_mode_enabled:
        moderator = Moderator(repo, llm, settings)
        tasks.append(asyncio.create_task(moderator.review_loop(), name="moderator"))
        logger.info("moderation mode ENABLED — advisory-only classifier active")
    # Cross-platform listeners. Both no-op when disabled or unconfigured.
    # YouTube takes the OBS poller so it can skip its 100-unit search.list
    # while we're not actually streaming (OBS-confirmed offline).
    if settings.youtube_enabled:
        yt = YouTubeListener(settings, repo, summarizer, obs=obs)
        tasks.append(asyncio.create_task(yt.start(), name="youtube_listener"))
    if settings.discord_enabled:
        dc = DiscordListener(settings, repo, summarizer)
        tasks.append(asyncio.create_task(dc.start(), name="discord_listener"))

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        repo.close()
        _clear_pid_file()


def run_dashboard(settings: Settings) -> None:
    import uvicorn

    from .web.app import create_app

    repo = ChatterRepo(settings.db_path, embed_dim=settings.ollama_embed_dim)
    app = create_app(repo, settings)
    if not settings.dashboard_basic_auth_enabled and settings.dashboard_host not in (
        "127.0.0.1",
        "localhost",
    ):
        logger.warning(
            "dashboard bound to %s without basic auth — anyone on this network can read it",
            settings.dashboard_host,
        )
    try:
        uvicorn.run(
            app,
            host=settings.dashboard_host,
            port=settings.dashboard_port,
            log_level="info",
        )
    finally:
        repo.close()


def run_diagnose(settings: Settings, with_recent_activity: bool) -> Path:
    out = Path(default_bundle_filename())
    path = build_diagnostic_bundle(out, settings, with_recent_activity=with_recent_activity)
    print(f"wrote {path}  ({path.stat().st_size:,} bytes)")
    if with_recent_activity:
        print("(includes opt-in recent_activity.json — usernames + per-user message counts)")
    else:
        print("(minimal mode — no chat content, no usernames, no secrets)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(prog="chatterbot")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("bot", "tui", "dashboard", "diagnose"),
        default=None,
        help="bot = silent listener; tui = Textual viewer; "
             "dashboard = FastAPI viewer; diagnose = write a .cbreport bundle",
    )
    parser.add_argument(
        "--with-recent-activity",
        action="store_true",
        help="(diagnose only) include usernames + per-user message counts "
             "from the last 24h. Off by default for privacy.",
    )
    args = parser.parse_args()

    settings = get_settings()
    mode = args.mode or settings.run_mode

    setup_logging(mode)

    if mode == "bot":
        try:
            asyncio.run(run_bot(settings))
        except KeyboardInterrupt:
            pass
    elif mode == "tui":
        repo = ChatterRepo(settings.db_path, embed_dim=settings.ollama_embed_dim)
        # Optional LLM client so manual notes added via the TUI can be embedded
        # for RAG. Failures inside the TUI are non-fatal — note still saves.
        llm = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            embed_model=settings.ollama_embed_model,
        )
        try:
            run_tui(repo, settings, llm=llm)
        finally:
            repo.close()
    elif mode == "dashboard":
        run_dashboard(settings)
    elif mode == "diagnose":
        run_diagnose(settings, with_recent_activity=args.with_recent_activity)
    else:
        parser.error(f"unknown mode: {mode!r}")


if __name__ == "__main__":
    main()
