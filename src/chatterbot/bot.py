"""TwitchIO chat listener.

This module is intentionally write-only with respect to the chat platform. It
does not register any commands. It does not emit any messages. Its job is:

  - upsert the user row for every chatter we observe
  - persist the message text into the (retained) chat log
  - hand off summarization triggers to the Summarizer

HARD RULE — DO NOT REMOVE WITHOUT READING README:
The bot must never produce chat output, and profile / event / topic / message
data must never enter any LLM prompt that produces chat-facing text. There is
no `!ask`, no `!whois`, no auto-reply. If you find yourself adding a command
handler here, stop — that's the prompt-injection / privacy attack surface this
project deliberately avoids. (The dashboard's "Ask Qwen" feature is fine
because its output renders in the streamer's browser only.)
"""

from __future__ import annotations

import asyncio
import logging

from twitchio.ext import commands

from .config import Settings
from .repo import ChatterRepo
from .summarizer import Summarizer

logger = logging.getLogger(__name__)


class ChatterListener(commands.Bot):
    def __init__(self, settings: Settings, repo: ChatterRepo, summarizer: Summarizer):
        channel = settings.twitch_channel.strip()
        super().__init__(
            token=settings.twitch_oauth_token,
            prefix="\x00",  # unreachable prefix; we register no commands anyway
            initial_channels=[channel] if channel else [],
        )
        self.settings = settings
        self.repo = repo
        self.summarizer = summarizer

    async def event_ready(self) -> None:
        logger.info(
            "twitch listener ready as %s in channel: %s",
            self.nick,
            self.settings.twitch_channel,
        )

    async def event_message(self, message) -> None:  # noqa: ANN001 - twitchio type
        if message.echo or message.author is None:
            return

        author = message.author
        twitch_id = str(getattr(author, "id", "") or "")
        name = getattr(author, "name", "") or ""
        content = (message.content or "").strip()

        if not twitch_id or not name or not content:
            return

        try:
            await asyncio.to_thread(self.repo.upsert_user, twitch_id, name)

            if await asyncio.to_thread(self.repo.is_opted_out, twitch_id):
                return  # honor opt_out — no logging, no summarization

            await asyncio.to_thread(self.repo.insert_message, twitch_id, content)
            unsummarized = await asyncio.to_thread(
                self.repo.unsummarized_count, twitch_id
            )
        except Exception:
            logger.exception("failed to persist message from %s", name)
            return

        try:
            await self.summarizer.maybe_summarize_user(twitch_id, unsummarized)
        except Exception:
            logger.exception("summarizer trigger failed for %s", name)
