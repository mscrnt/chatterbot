"""CLI entrypoint.

Usage:
  chatterbot bot         # silent listener: TwitchIO + summarizer + SE + topics loop
  chatterbot tui         # streamer-only Textual viewer
  chatterbot dashboard   # streamer-only FastAPI dashboard

All three modes are independent processes. They share the SQLite DB via WAL,
so they can run simultaneously.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .bot import ChatterListener
from .config import Settings, get_settings
from .llm.ollama_client import OllamaClient
from .repo import ChatterRepo
from .streamelements import StreamElementsListener
from .summarizer import Summarizer
from .tui import run_tui

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def run_bot(settings: Settings) -> None:
    if not settings.twitch_oauth_token or not settings.twitch_channel:
        logger.error("missing TWITCH_OAUTH_TOKEN or TWITCH_CHANNEL — refusing to start")
        return

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

    tasks: list[asyncio.Task] = [
        asyncio.create_task(listener.start(), name="twitch_listener"),
        asyncio.create_task(summarizer.idle_loop(), name="summarizer_idle"),
        asyncio.create_task(summarizer.topics_loop(), name="summarizer_topics"),
    ]
    if settings.streamelements_enabled:
        tasks.append(asyncio.create_task(se.run(), name="streamelements"))

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        repo.close()


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


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(prog="chatterbot")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("bot", "tui", "dashboard"),
        default=None,
        help="bot = silent listener; tui = Textual viewer; dashboard = FastAPI viewer",
    )
    args = parser.parse_args()

    settings = get_settings()
    mode = args.mode or settings.run_mode

    if mode == "bot":
        try:
            asyncio.run(run_bot(settings))
        except KeyboardInterrupt:
            pass
    elif mode == "tui":
        repo = ChatterRepo(settings.db_path, embed_dim=settings.ollama_embed_dim)
        try:
            run_tui(repo, settings)
        finally:
            repo.close()
    elif mode == "dashboard":
        run_dashboard(settings)
    else:
        parser.error(f"unknown mode: {mode!r}")


if __name__ == "__main__":
    main()
