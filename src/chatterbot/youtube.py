"""YouTube Live Chat listener.

Polling-based read-only ingestion of the streamer's live chat. Drops every
message into the same `messages` table the Twitch listener uses, so the
dashboard surfaces (live widget, summarizer, RAG, merge UI) treat YouTube
chatters identically to Twitch chatters.

Hard rule (mirrors bot.py): write-only with respect to the platform. We
never call any send-message endpoint on YouTube. Profile / event / topic /
note data must never enter any LLM prompt that produces a chat-facing
response. Same architectural rule as the Twitch listener.

Quota model
-----------
The YouTube Data API v3 default quota is 10,000 units / day.
  - search.list             → 100 units (used to find a live broadcast)
  - videos.list             → 1 unit   (used to fetch activeLiveChatId)
  - liveChatMessages.list   → 5 units  (used to poll messages)

We minimize search.list:
  - At startup we run one search.
  - When chat returns "stream offline", we wait OFFLINE_RECHECK_SECONDS
    (default 300s = 5 min) before re-searching.
  - When chat is live we never call search; we just keep paging the chat
    using the next pageToken at the server-suggested pollingIntervalMillis.

Six-hour-stream cost: 1 search + 1 videos + ~720 chat polls/hr * 6 = ~21,600
units, which exceeds the daily quota. The streamer should request a quota
bump in Google Cloud Console for sustained YouTube ingestion.

Auth
----
Read-only public live chat works with an API key (no OAuth). No special
scope is required because the live-chat data is technically public.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from .config import Settings
from .obs import OBSStatusService
from .repo import ChatterRepo
from .summarizer import Summarizer

logger = logging.getLogger(__name__)


YT_API = "https://www.googleapis.com/youtube/v3"


@dataclass
class YouTubeStatus:
    enabled: bool = False
    connected: bool = False
    is_live: bool = False
    video_id: str | None = None
    live_chat_id: str | None = None
    # True when we've skipped a discovery cycle because OBS told us the
    # streamer isn't live. Surfaces in the dashboard as an "auto-paused" hint.
    auto_paused: bool = False
    last_error: str | None = None


class YouTubeListener:
    """Polls YouTube live chat and persists messages via ChatterRepo."""

    SOURCE = "youtube"
    OFFLINE_RECHECK_SECONDS = 300
    DEFAULT_POLL_MS = 5000
    AUTO_PAUSE_RECHECK_SECONDS = 30

    def __init__(
        self,
        settings: Settings,
        repo: ChatterRepo,
        summarizer: Summarizer,
        *,
        obs: OBSStatusService | None = None,
    ):
        self.settings = settings
        self.repo = repo
        self.summarizer = summarizer
        # Optional OBS coupling: when present, we skip the 100-unit
        # search.list discovery while OBS reports the streamer offline.
        # search.list runs every 5 min when offline, so this saves up to
        # 1,200 quota units / hr of wasted work between streams.
        self.obs = obs
        self.status = YouTubeStatus(enabled=bool(settings.youtube_enabled))
        self._page_token: str | None = None
        self._last_logged_error: str | None = None
        self._was_auto_paused: bool = False

    @property
    def configured(self) -> bool:
        s = self.settings
        return bool(s.youtube_enabled and s.youtube_api_key and s.youtube_channel_id)

    @staticmethod
    def make_user_id(channel_id: str) -> str:
        """Namespace YouTube author channel IDs so they can't collide with
        Twitch user IDs."""
        return f"yt:{channel_id}"

    async def start(self) -> None:
        """Top-level loop. Returns when the bot is cancelled."""
        if not self.configured:
            if self.settings.youtube_enabled:
                logger.warning("youtube: enabled but missing api_key / channel_id")
            return
        logger.info(
            "youtube: starting listener for channel=%s",
            self.settings.youtube_channel_id,
        )
        try:
            while True:
                # OBS-driven auto-pause: if OBS confirms the streamer isn't
                # live, skip discovery (saves the 100-unit search.list call).
                # Recheck every 30s so we re-engage quickly when the stream
                # comes online. Defaults to "not paused" on any uncertainty.
                if self.obs is not None and self.obs.status.is_streamer_offline():
                    if not self._was_auto_paused:
                        logger.info(
                            "youtube: auto-paused (OBS reports offline) — "
                            "skipping discovery to save quota"
                        )
                        self._was_auto_paused = True
                    self.status.auto_paused = True
                    self.status.is_live = False
                    await asyncio.sleep(self.AUTO_PAUSE_RECHECK_SECONDS)
                    continue
                if self._was_auto_paused:
                    logger.info("youtube: auto-resumed (OBS reports online)")
                    self._was_auto_paused = False
                self.status.auto_paused = False
                try:
                    found = await self._discover_live_chat()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self._log_error(f"discover failed: {type(e).__name__}: {e}")
                    found = False
                if not found:
                    self.status.is_live = False
                    await asyncio.sleep(self.OFFLINE_RECHECK_SECONDS)
                    continue
                self.status.is_live = True
                self.status.connected = True
                try:
                    await self._poll_chat_until_offline()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self._log_error(f"chat poll failed: {type(e).__name__}: {e}")
                # Either chat ended cleanly or errored — reset and re-discover.
                self._page_token = None
                self.status.is_live = False
                self.status.live_chat_id = None
                self.status.video_id = None
        except asyncio.CancelledError:
            logger.info("youtube: listener cancelled")
            raise

    async def _discover_live_chat(self) -> bool:
        """Find the channel's currently-live broadcast and its chat id."""
        api_key = self.settings.youtube_api_key
        channel_id = self.settings.youtube_channel_id.strip()
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{YT_API}/search",
                params={
                    "part": "id",
                    "channelId": channel_id,
                    "eventType": "live",
                    "type": "video",
                    "maxResults": 1,
                    "key": api_key,
                },
            )
            if r.status_code != 200:
                self._log_error(
                    f"search.list returned {r.status_code}: {r.text[:200]}"
                )
                return False
            items = (r.json().get("items") or [])
            if not items:
                return False
            video_id = items[0].get("id", {}).get("videoId")
            if not video_id:
                return False

            r = await client.get(
                f"{YT_API}/videos",
                params={
                    "part": "liveStreamingDetails",
                    "id": video_id,
                    "key": api_key,
                },
            )
            if r.status_code != 200:
                self._log_error(
                    f"videos.list returned {r.status_code}: {r.text[:200]}"
                )
                return False
            v_items = r.json().get("items") or []
            if not v_items:
                return False
            chat_id = (v_items[0].get("liveStreamingDetails") or {}).get(
                "activeLiveChatId"
            )
            if not chat_id:
                return False
            self.status.video_id = video_id
            self.status.live_chat_id = chat_id
            self._page_token = None
            self._last_logged_error = None
            logger.info(
                "youtube: found live broadcast video=%s chat=%s",
                video_id, chat_id[:16],
            )
            return True

    async def _poll_chat_until_offline(self) -> None:
        """Page through liveChatMessages until the API tells us the chat
        ended (no more messages, or 403/404)."""
        api_key = self.settings.youtube_api_key
        chat_id = self.status.live_chat_id
        if not chat_id:
            return
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                params: dict[str, Any] = {
                    "liveChatId": chat_id,
                    "part": "snippet,authorDetails",
                    "maxResults": 200,
                    "key": api_key,
                }
                if self._page_token:
                    params["pageToken"] = self._page_token
                r = await client.get(
                    f"{YT_API}/liveChat/messages", params=params
                )
                if r.status_code in (403, 404):
                    logger.info(
                        "youtube: chat ended (%d) — going offline",
                        r.status_code,
                    )
                    return
                if r.status_code != 200:
                    self._log_error(
                        f"liveChatMessages returned {r.status_code}: {r.text[:200]}"
                    )
                    await asyncio.sleep(self.DEFAULT_POLL_MS / 1000)
                    continue
                payload = r.json()
                self._page_token = payload.get("nextPageToken")
                interval_ms = int(
                    payload.get("pollingIntervalMillis") or self.DEFAULT_POLL_MS
                )
                for item in payload.get("items") or []:
                    try:
                        await self._persist(item)
                    except Exception:
                        logger.exception(
                            "youtube: failed to persist message"
                        )
                await asyncio.sleep(max(1.0, interval_ms / 1000))

    async def _persist(self, item: dict) -> None:
        snippet = item.get("snippet") or {}
        author = item.get("authorDetails") or {}
        message_type = snippet.get("type")
        # YouTube emits non-text events too (memberships, super-chats, etc).
        # Persist only textMessageEvent for the chat log; super-chats could
        # later route into events, but that's a separate feature.
        if message_type != "textMessageEvent":
            return
        text_payload = snippet.get("textMessageDetails") or {}
        content = (text_payload.get("messageText") or "").strip()
        author_channel_id = author.get("channelId") or snippet.get("authorChannelId")
        display_name = author.get("displayName") or "?"
        if not author_channel_id or not content:
            return
        user_id = self.make_user_id(author_channel_id)

        is_mod = bool(author.get("isChatModerator"))
        is_owner = bool(author.get("isChatOwner"))
        # YouTube channel members ≈ Twitch subs. We mark them as a "subscriber"
        # tier 1000 so the dashboard's existing sub-pill code lights up.
        is_member = bool(author.get("isChatSponsor"))
        sub_tier = "1000" if is_member else None

        await asyncio.to_thread(
            self.repo.upsert_user, user_id, display_name, source=self.SOURCE
        )
        await asyncio.to_thread(
            self.repo.update_user_badges, user_id,
            sub_tier=sub_tier, sub_months=0,
            is_mod=is_mod, is_vip=False, is_founder=is_owner,
        )

        if await asyncio.to_thread(self.repo.is_opted_out, user_id):
            return

        try:
            fired = await asyncio.to_thread(
                self.repo.fire_pending_reminders, user_id
            )
            if fired:
                logger.info(
                    "youtube: reminder fired %d for %s on incoming message",
                    fired, display_name,
                )
        except Exception:
            logger.exception("youtube: fire-pending failed")

        await asyncio.to_thread(self.repo.insert_message, user_id, content)
        unsummarized = await asyncio.to_thread(
            self.repo.unsummarized_count, user_id
        )
        try:
            await self.summarizer.maybe_summarize_user(user_id, unsummarized)
        except Exception:
            logger.exception("youtube: summarizer trigger failed")

    def _log_error(self, msg: str) -> None:
        """Log once per distinct error spelling — avoids log floods when the
        API is down or rate-limited."""
        if msg != self._last_logged_error:
            logger.warning("youtube: %s", msg)
            self._last_logged_error = msg
        self.status.last_error = msg
