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
    using the next pageToken at an adaptive interval.

Adaptive poll cadence keeps active polling under quota:
  - `youtube_min_poll_seconds` (default 10) — minimum interval. We never
    poll faster than this, even if the server suggests a smaller value.
  - `youtube_max_poll_seconds` (default 30) — cap on the backoff window.
  - Empty polls double the current interval (capped at max). Non-empty
    polls reset to the minimum. The server's `pollingIntervalMillis`
    is also honored as a floor.

A 6-hour stream with mixed activity (some chatter, some quiet stretches)
typically lands at ~7,000 quota units with the defaults — comfortably
under the free tier. Worst-case all-active 6-hour stream at 10s polls:
360/hr × 6 × 5 = 10,800 units, still within striking distance of the
free quota; raise to 15-20 s minimum for guaranteed headroom.

Event types we persist beyond chat:
  - superChatEvent       → events(type='superchat', amount, currency, message)
  - superStickerEvent    → events(type='superchat', amount, currency, message)
  - newSponsorEvent      → events(type='member',    sub_tier='1000')
  - memberMilestoneChat  → events(type='member',    sub_months from API)

This puts YouTube tips + memberships on the dashboard's events feed
alongside Twitch / StreamElements activity.

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
    AUTO_PAUSE_RECHECK_SECONDS = 30
    # Transient-error backoff: when the API returns 5xx, sleep this long
    # before retrying. Short enough that we recover from a blip, long
    # enough that we don't hammer a struggling endpoint.
    TRANSIENT_RETRY_SECONDS = 30

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
        # Adaptive poll interval — starts at min, doubles on empty polls
        # up to max, resets on activity.
        self._current_poll_seconds: float = float(
            getattr(settings, "youtube_min_poll_seconds", 10)
        )

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
        """Find the channel's currently-live broadcast and its chat id.
        Treats network blips and 5xx responses as "try again later"
        rather than "stream offline" — the latter would needlessly burn
        a search.list re-discovery 5 minutes from now."""
        api_key = self.settings.youtube_api_key
        channel_id = self.settings.youtube_channel_id.strip()
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
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
            except httpx.RequestError as e:
                self._log_error(f"search.list network: {type(e).__name__}: {e}")
                return False
            if 500 <= r.status_code < 600:
                self._log_error(
                    f"search.list 5xx ({r.status_code}) — will retry next cycle"
                )
                return False
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

            try:
                r = await client.get(
                    f"{YT_API}/videos",
                    params={
                        "part": "liveStreamingDetails",
                        "id": video_id,
                        "key": api_key,
                    },
                )
            except httpx.RequestError as e:
                self._log_error(f"videos.list network: {type(e).__name__}: {e}")
                return False
            if 500 <= r.status_code < 600:
                self._log_error(
                    f"videos.list 5xx ({r.status_code}) — will retry next cycle"
                )
                return False
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
        ended (no more messages, or 403/404). Adaptive backoff between
        polls keeps daily quota usage in check on long streams."""
        api_key = self.settings.youtube_api_key
        chat_id = self.status.live_chat_id
        if not chat_id:
            return
        min_s = max(1, int(getattr(self.settings, "youtube_min_poll_seconds", 10)))
        max_s = max(min_s, int(getattr(self.settings, "youtube_max_poll_seconds", 30)))
        # Reset to min at the start of every chat session so backoff
        # state doesn't leak across reconnects.
        self._current_poll_seconds = float(min_s)

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
                try:
                    r = await client.get(
                        f"{YT_API}/liveChat/messages", params=params
                    )
                except httpx.RequestError as e:
                    # Network blip / DNS / timeout — back off + retry.
                    self._log_error(f"liveChatMessages network: {type(e).__name__}: {e}")
                    await asyncio.sleep(self.TRANSIENT_RETRY_SECONDS)
                    continue
                if r.status_code in (403, 404):
                    logger.info(
                        "youtube: chat ended (%d) — going offline",
                        r.status_code,
                    )
                    return
                if 500 <= r.status_code < 600:
                    self._log_error(
                        f"liveChatMessages 5xx ({r.status_code}): {r.text[:200]} "
                        f"— backing off {self.TRANSIENT_RETRY_SECONDS}s"
                    )
                    await asyncio.sleep(self.TRANSIENT_RETRY_SECONDS)
                    continue
                if r.status_code == 429:
                    # Rate-limited: jam the poll cadence to max for this
                    # cycle and try again. The server is telling us we're
                    # going too fast (rare with our adaptive defaults).
                    self._log_error("liveChatMessages 429 — quota / rate-limit hit")
                    self._current_poll_seconds = float(max_s)
                    await asyncio.sleep(max_s)
                    continue
                if r.status_code != 200:
                    self._log_error(
                        f"liveChatMessages returned {r.status_code}: {r.text[:200]}"
                    )
                    await asyncio.sleep(self._current_poll_seconds)
                    continue

                payload = r.json()
                self._page_token = payload.get("nextPageToken")
                items = payload.get("items") or []
                # Server-suggested interval acts as a floor we must
                # honor — YouTube tells us "don't poll faster than X".
                server_floor_s = max(
                    1.0, int(payload.get("pollingIntervalMillis") or 0) / 1000
                )

                for item in items:
                    try:
                        await self._persist(item)
                    except Exception:
                        logger.exception(
                            "youtube: failed to persist message"
                        )

                # Adaptive backoff: empty polls double the interval
                # (capped at max); non-empty polls reset to min. Always
                # at least the server-suggested floor.
                if items:
                    self._current_poll_seconds = float(min_s)
                else:
                    self._current_poll_seconds = min(
                        float(max_s), self._current_poll_seconds * 2,
                    )
                sleep_s = max(self._current_poll_seconds, server_floor_s)
                await asyncio.sleep(sleep_s)

    async def _persist(self, item: dict) -> None:
        snippet = item.get("snippet") or {}
        author = item.get("authorDetails") or {}
        message_type = snippet.get("type")
        author_channel_id = author.get("channelId") or snippet.get("authorChannelId")
        display_name = author.get("displayName") or "?"
        if not author_channel_id:
            return
        user_id = self.make_user_id(author_channel_id)

        # Always upsert the user + badge state, regardless of message
        # type — super-chats and memberships also need the sender on
        # the chatters list so the events feed can link to their profile.
        is_mod = bool(author.get("isChatModerator"))
        is_owner = bool(author.get("isChatOwner"))
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

        # Route by message type. Most are chat; the donation /
        # membership types go to the events table so they show up in
        # the dashboard's events feed alongside Twitch / SE activity.
        if message_type == "textMessageEvent":
            await self._persist_text_message(user_id, display_name, snippet, item)
        elif message_type == "superChatEvent":
            await self._persist_super_chat(user_id, display_name, snippet, item)
        elif message_type == "superStickerEvent":
            await self._persist_super_sticker(user_id, display_name, snippet, item)
        elif message_type == "newSponsorEvent":
            await self._persist_new_member(user_id, display_name, snippet, item)
        elif message_type == "memberMilestoneChatEvent":
            await self._persist_member_milestone(user_id, display_name, snippet, item)
        # Other types (messageDeletedEvent, userBannedEvent, chatEndedEvent)
        # are intentionally ignored — they're moderation artifacts the
        # dashboard doesn't surface.

    async def _persist_text_message(
        self, user_id: str, display_name: str, snippet: dict, raw: dict,
    ) -> None:
        text_payload = snippet.get("textMessageDetails") or {}
        content = (text_payload.get("messageText") or "").strip()
        if not content:
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

    @staticmethod
    def _parse_amount(details: dict) -> tuple[float | None, str | None]:
        """super-chat / super-sticker amounts come back two ways:
        `amountMicros` (string of micros) + `currency` (ISO-4217), and
        a pre-formatted `amountDisplayString` like "$5.00". We prefer
        the precise micros parse; fall back to displayString if missing."""
        currency = details.get("currency")
        micros = details.get("amountMicros")
        if micros:
            try:
                return float(int(micros)) / 1_000_000.0, currency
            except (TypeError, ValueError):
                pass
        # Fallback parse from "$5.00", "€2.50", etc. — strip non-digit
        # / non-dot prefix.
        disp = (details.get("amountDisplayString") or "").strip()
        if disp:
            digits = "".join(c for c in disp if c.isdigit() or c == ".")
            try:
                return float(digits), currency
            except ValueError:
                pass
        return None, currency

    async def _persist_super_chat(
        self, user_id: str, display_name: str, snippet: dict, raw: dict,
    ) -> None:
        details = snippet.get("superChatDetails") or {}
        amount, currency = self._parse_amount(details)
        message = (details.get("userComment") or "").strip() or None
        await asyncio.to_thread(
            self.repo.record_event_for_user_id,
            user_id, display_name, "superchat",
            amount=amount, currency=currency, message=message, raw=raw,
        )
        logger.info(
            "youtube: super-chat from %s — %s %s",
            display_name, amount, currency or "",
        )

    async def _persist_super_sticker(
        self, user_id: str, display_name: str, snippet: dict, raw: dict,
    ) -> None:
        details = snippet.get("superStickerDetails") or {}
        amount, currency = self._parse_amount(details)
        sticker_meta = details.get("superStickerMetadata") or {}
        # Stickers don't have a userComment field; use the alt-text /
        # sticker name so the events log has SOMETHING readable.
        message = sticker_meta.get("altText") or sticker_meta.get("stickerId") or None
        await asyncio.to_thread(
            self.repo.record_event_for_user_id,
            user_id, display_name, "superchat",
            amount=amount, currency=currency, message=message, raw=raw,
        )
        logger.info(
            "youtube: super-sticker from %s — %s %s",
            display_name, amount, currency or "",
        )

    async def _persist_new_member(
        self, user_id: str, display_name: str, snippet: dict, raw: dict,
    ) -> None:
        details = snippet.get("newSponsorDetails") or {}
        tier = details.get("memberLevelName") or "member"
        await asyncio.to_thread(
            self.repo.record_event_for_user_id,
            user_id, display_name, "member",
            amount=None, currency=None,
            message=f"new member ({tier})", raw=raw,
        )
        logger.info("youtube: new member %s (tier=%s)", display_name, tier)

    async def _persist_member_milestone(
        self, user_id: str, display_name: str, snippet: dict, raw: dict,
    ) -> None:
        details = snippet.get("memberMilestoneChatDetails") or {}
        months = details.get("memberMonth")
        comment = (details.get("userComment") or "").strip()
        tier = details.get("memberLevelName") or "member"
        message = (
            f"{months}-month milestone ({tier})"
            + (f": {comment}" if comment else "")
        )
        try:
            months_int = int(months) if months else 0
        except (TypeError, ValueError):
            months_int = 0
        # Bump the badge sub_months too so the user-detail page shows
        # months alongside the member tier.
        if months_int:
            await asyncio.to_thread(
                self.repo.update_user_badges, user_id,
                sub_tier="1000", sub_months=months_int,
                is_mod=False, is_vip=False, is_founder=False,
            )
        await asyncio.to_thread(
            self.repo.record_event_for_user_id,
            user_id, display_name, "member",
            amount=float(months_int) if months_int else None,
            currency=None, message=message, raw=raw,
        )
        logger.info(
            "youtube: member milestone %s — %s months", display_name, months,
        )

    def _log_error(self, msg: str) -> None:
        """Log once per distinct error spelling — avoids log floods when the
        API is down or rate-limited."""
        if msg != self._last_logged_error:
            logger.warning("youtube: %s", msg)
            self._last_logged_error = msg
        self.status.last_error = msg
