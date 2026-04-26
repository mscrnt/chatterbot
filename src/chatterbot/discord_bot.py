"""Discord listener — STUB.

This module reserves the wiring for cross-platform ingestion. It does not
yet connect to Discord. When you wire it up, the contract matches
`bot.ChatterListener`:

  - upsert the user row (twitch_id = 'dc:<discord_user_id>', source='discord')
  - persist the message text into the same `messages` table
  - hand off summarization triggers to the existing Summarizer

The merge UI on the dashboard is what reconciles a Discord member with
their Twitch identity when they're the same person.

What's needed to make this real:
  - A bot token created at https://discord.com/developers/applications, plus
    the Message Content intent enabled on the application settings page.
  - The `discord.py` library (add to pyproject.toml as needed).
  - One or more channel IDs the streamer wants to ingest (set in Settings).
  - Honor `is_opted_out` exactly like the Twitch listener.

For now: when `discord_enabled` is true and creds are set, we just log
"would connect to Discord" once and idle. Disabled (default) = total no-op.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from .config import Settings
from .repo import ChatterRepo
from .summarizer import Summarizer

logger = logging.getLogger(__name__)


@dataclass
class DiscordStatus:
    enabled: bool = False
    connected: bool = False
    guild_count: int = 0
    channels: list[str] = field(default_factory=list)
    last_error: str | None = None


class DiscordListener:
    """No-op stub for Discord ingestion. Replace `_run` with a real
    discord.py client once a token + intents are in place."""

    SOURCE = "discord"

    def __init__(self, settings: Settings, repo: ChatterRepo, summarizer: Summarizer):
        self.settings = settings
        self.repo = repo
        self.summarizer = summarizer
        self.status = DiscordStatus(enabled=bool(settings.discord_enabled))

    @property
    def configured(self) -> bool:
        s = self.settings
        return bool(s.discord_enabled and s.discord_bot_token and s.discord_channel_ids)

    async def start(self) -> None:
        """Entry point used by main.run_bot. Returns when the bot exits."""
        if not self.configured:
            if self.settings.discord_enabled:
                logger.warning(
                    "discord: enabled but missing bot_token / channel_ids — skipping"
                )
            return
        chans = [c.strip() for c in self.settings.discord_channel_ids.split(",") if c.strip()]
        logger.info(
            "discord: stub active for %d channel(s) — no gateway connection implemented yet",
            len(chans),
        )
        self.status.connected = False
        self.status.channels = chans
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    @staticmethod
    def make_user_id(discord_id: str | int) -> str:
        """Namespace Discord snowflakes so they can't collide with Twitch
        user IDs."""
        return f"dc:{discord_id}"
