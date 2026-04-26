"""YouTube Live Chat listener — STUB.

This module exists to reserve the wiring for cross-platform ingestion.
It does not yet poll YouTube. When you wire it up, the goal is the same
contract as `bot.ChatterListener`:

  - upsert the user row (twitch_id = 'yt:<channel_id>', source='youtube')
  - persist the message text into the same `messages` table
  - hand off summarization triggers to the existing Summarizer

The merge UI on the dashboard is what reconciles a YouTube viewer with
their Twitch identity if they're the same person.

What's needed to make this real:
  - YouTube Data API v3 credentials (API key for read-only, OAuth for the
    streamer's own live broadcast lookups). Set in Settings → YouTube.
  - Poll `liveBroadcasts.list` to find the active broadcast id, then
    `liveChatMessages.list` with pageToken loop. Quota: each call costs
    5 units; default daily quota is 10,000 → mind cost.
  - Map `authorChannelId` → user_id 'yt:<id>'; `authorDisplayName` → name.

For now: when `youtube_enabled` is true and creds are set, we just log
"would connect to YouTube" once and idle. Disabled (default) = total no-op.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from .config import Settings
from .repo import ChatterRepo
from .summarizer import Summarizer

logger = logging.getLogger(__name__)


@dataclass
class YouTubeStatus:
    enabled: bool = False
    connected: bool = False
    broadcast_id: str | None = None
    last_error: str | None = None


class YouTubeListener:
    """No-op stub for YouTube live chat ingestion. Replace `_run` with the
    real polling loop once API credentials are in place."""

    SOURCE = "youtube"

    def __init__(self, settings: Settings, repo: ChatterRepo, summarizer: Summarizer):
        self.settings = settings
        self.repo = repo
        self.summarizer = summarizer
        self.status = YouTubeStatus(enabled=bool(settings.youtube_enabled))

    @property
    def configured(self) -> bool:
        s = self.settings
        return bool(s.youtube_enabled and s.youtube_api_key and s.youtube_channel_id)

    async def start(self) -> None:
        """Entry point used by main.run_bot. Returns when the bot exits."""
        if not self.configured:
            if self.settings.youtube_enabled:
                logger.warning(
                    "youtube: enabled but missing api_key / channel_id — skipping"
                )
            return
        logger.info(
            "youtube: stub active for channel=%s — no chat polling implemented yet",
            self.settings.youtube_channel_id,
        )
        self.status.connected = False
        # Idle forever so the gather() in main.py doesn't return early.
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    @staticmethod
    def make_user_id(channel_id: str) -> str:
        """Namespace YouTube author channel IDs so they can't collide with
        Twitch user IDs. Used by both the listener and the dashboard's
        merge picker."""
        return f"yt:{channel_id}"
