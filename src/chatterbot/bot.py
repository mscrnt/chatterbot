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


def _parse_badges(s: str) -> dict[str, str]:
    """Parse a Twitch IRCv3 `badges` / `badge-info` tag value into
    {name: value}. Format: 'subscriber/3,vip/1,moderator/1' → keys."""
    out: dict[str, str] = {}
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "/" in part:
            k, v = part.split("/", 1)
            out[k] = v
        else:
            out[part] = ""
    return out


def _is_emote_only(content: str, emotes_tag: str | None) -> bool:
    """Decide whether a chat message is just emotes + whitespace.

    Twitch IRCv3 emits the exact spans of every native emote in the
    `emotes` tag, formatted as `id:start-end,start-end/id:start-end`.
    Positions are 0-indexed character offsets into the message content,
    counting Unicode code points (NOT UTF-16 like JS, NOT bytes).

    We mark the message emote-only when subtracting every emote span
    leaves nothing but whitespace. BTTV/FFZ emotes aren't in the IRC
    tag (they're rendered client-side) so we treat them as plain text —
    a 'bawkCrazy bawkCrazy bawkCrazy' message from a Twitch channel
    emote will be marked emote-only because Twitch IS aware of it.
    """
    if not emotes_tag or not content:
        return False
    # IRC offsets are over Unicode code points; index a code-point list,
    # not the UTF-16 string, so emotes after a multi-byte char align.
    chars = list(content)
    keep = [True] * len(chars)
    for chunk in emotes_tag.split("/"):
        if ":" not in chunk:
            continue
        _, spans = chunk.split(":", 1)
        for span in spans.split(","):
            if "-" not in span:
                continue
            try:
                start_s, end_s = span.split("-", 1)
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            for i in range(start, min(end + 1, len(chars))):
                keep[i] = False
    leftover = "".join(c for c, k in zip(chars, keep) if k).strip()
    return leftover == ""


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

        # Twitch's native Reply feature surfaces the parent message via IRCv3
        # tags. Capture them denormalized so the dashboard + classifier can
        # show what was being replied to without us tracking Twitch's UUIDs.
        tags = getattr(message, "tags", None) or {}
        reply_parent_login = (tags.get("reply-parent-user-login") or "").strip() or None
        reply_parent_body_raw = tags.get("reply-parent-msg-body") or ""
        # Twitch IRC encodes spaces in tag values as \s; decode back.
        reply_parent_body = (
            reply_parent_body_raw.replace("\\s", " ").strip() or None
            if reply_parent_body_raw
            else None
        )

        # IRCv3 badges → role/sub state. `badges` is the display set
        # (e.g. "subscriber/3,vip/1,moderator/1"); `badge-info` carries the
        # accurate sub_months ("subscriber/12") even when the badge image
        # rounds. We snap last-seen state per chatter.
        badges_str = tags.get("badges") or ""
        badge_info_str = tags.get("badge-info") or ""
        badges = _parse_badges(badges_str)
        badge_info = _parse_badges(badge_info_str)
        sub_tier = badges.get("subscriber")  # '1000'/'2000'/'3000' tier code
        # If the user has the founder badge they're a sub even without a
        # subscriber badge — promote to tier 1000 so the rest of the app
        # treats them as a subscriber.
        if not sub_tier and "founder" in badges:
            sub_tier = "1000"
        if "premium" in badges or "twitch-prime" in badges:
            sub_tier = sub_tier or "Prime"
        try:
            sub_months = int(badge_info.get("subscriber") or 0)
        except (TypeError, ValueError):
            sub_months = 0
        is_mod = "moderator" in badges or "broadcaster" in badges
        is_vip = "vip" in badges
        is_founder = "founder" in badges

        # Twitch IRCv3 `emotes` tag → ground-truth emote spans. If after
        # removing every emote the message is just whitespace, flag it so
        # the summarizer + moderator skip it (still kept for live display).
        emotes_tag = tags.get("emotes") or ""
        is_emote_only = _is_emote_only(content, emotes_tag)

        try:
            await asyncio.to_thread(self.repo.upsert_user, twitch_id, name)
            await asyncio.to_thread(
                self.repo.update_user_badges, twitch_id,
                sub_tier=sub_tier, sub_months=sub_months,
                is_mod=is_mod, is_vip=is_vip, is_founder=is_founder,
            )

            if await asyncio.to_thread(self.repo.is_opted_out, twitch_id):
                return  # honor opt_out — no logging, no summarization

            # Fire any pending reminders attached to this chatter — the
            # dashboard's nav pill picks them up on the next render.
            try:
                fired = await asyncio.to_thread(
                    self.repo.fire_pending_reminders, twitch_id
                )
                if fired:
                    logger.info(
                        "reminder: fired %d for %s on incoming message",
                        fired, name,
                    )
            except Exception:
                logger.exception("reminder: fire-pending failed for %s", name)

            await asyncio.to_thread(
                self.repo.insert_message,
                twitch_id, content,
                reply_parent_login=reply_parent_login,
                reply_parent_body=reply_parent_body,
                is_emote_only=is_emote_only,
            )
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
