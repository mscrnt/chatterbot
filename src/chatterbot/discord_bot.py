"""Discord listener.

Read-only ingestion of messages from a configured set of Discord channels.
Persists into the same `messages` table the Twitch listener uses, so the
dashboard surfaces (live widget, summarizer, RAG, merge UI) treat Discord
chatters identically to Twitch chatters.

Hard rule (mirrors bot.py): write-only with respect to Discord. We never
call any send-message endpoint. Profile / event / topic / note data must
never enter any LLM prompt that produces a chat-facing response.

Required setup
--------------
1. Create an application + bot at https://discord.com/developers/applications
2. Under "Bot" → privileged intents, enable **Message Content Intent**
   (and optionally **Server Members** for richer display names).
3. Use the OAuth2 URL generator to invite the bot to your server with
   the `bot` scope and at minimum the `Read Messages/View Channels` and
   `Read Message History` permissions.
4. Configure in dashboard `/settings`:
     - discord_enabled = on
     - discord_bot_token = <token>
     - discord_channel_ids = comma-separated channel IDs to ingest
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import discord

from .config import Settings
from .repo import ChatterRepo
from .summarizer import Summarizer

logger = logging.getLogger(__name__)


@dataclass
class DiscordStatus:
    enabled: bool = False
    connected: bool = False
    user: str | None = None
    guild_count: int = 0
    channel_count: int = 0
    last_error: str | None = None
    channels: list[str] = field(default_factory=list)


class DiscordListener:
    """discord.py Client subclass that mirrors ChatterListener's contract:
    upsert user, snapshot mod-state, persist message, trigger summarizer."""

    SOURCE = "discord"

    def __init__(self, settings: Settings, repo: ChatterRepo, summarizer: Summarizer):
        self.settings = settings
        self.repo = repo
        self.summarizer = summarizer
        self.status = DiscordStatus(enabled=bool(settings.discord_enabled))
        self._channel_ids: set[int] = set()
        self._client: discord.Client | None = None

    @property
    def configured(self) -> bool:
        s = self.settings
        return bool(
            s.discord_enabled and s.discord_bot_token and s.discord_channel_ids
        )

    @staticmethod
    def make_user_id(discord_id: str | int) -> str:
        """Namespace Discord snowflakes so they can't collide with Twitch ids."""
        return f"dc:{discord_id}"

    def _parse_channels(self) -> set[int]:
        out: set[int] = set()
        for raw in (self.settings.discord_channel_ids or "").split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.add(int(raw))
            except ValueError:
                logger.warning("discord: ignoring non-numeric channel id %r", raw)
        return out

    async def start(self) -> None:
        """Top-level loop. Runs the client until cancelled."""
        if not self.configured:
            if self.settings.discord_enabled:
                logger.warning(
                    "discord: enabled but missing bot_token / channel_ids"
                )
            return
        self._channel_ids = self._parse_channels()
        if not self._channel_ids:
            logger.warning("discord: no usable channel ids — disabling")
            return

        intents = discord.Intents.default()
        intents.message_content = True  # privileged — must be enabled in dev portal
        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready():
            assert self._client is not None
            self.status.connected = True
            self.status.user = str(self._client.user) if self._client.user else None
            self.status.guild_count = len(self._client.guilds)
            self.status.channels = [str(cid) for cid in self._channel_ids]
            self.status.channel_count = len(self._channel_ids)
            logger.info(
                "discord: ready as %s in %d guild(s), watching %d channel(s)",
                self.status.user,
                self.status.guild_count,
                self.status.channel_count,
            )

        @self._client.event
        async def on_message(message: discord.Message):
            await self._handle_message(message)

        try:
            await self._client.start(self.settings.discord_bot_token)
        except asyncio.CancelledError:
            logger.info("discord: listener cancelled")
            await self._client.close()
            raise
        except discord.LoginFailure as e:
            self.status.last_error = "login failure — bad token"
            logger.error("discord: %s", e)
        except Exception:
            self.status.last_error = "client crashed — see logs"
            logger.exception("discord: client crashed")

    async def _handle_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return  # never log bot output, including this bot's own
        if message.channel.id not in self._channel_ids:
            return
        content = (message.content or "").strip()
        if not content:
            return  # attachments / embeds without text — skip for now

        author = message.author
        user_id = self.make_user_id(author.id)
        # Prefer guild nickname when present; falls back to global name.
        display_name = (
            getattr(author, "nick", None) or author.display_name or author.name or "?"
        )

        is_mod = False
        is_founder = False
        try:
            perms = author.guild_permissions  # only on Member, not User
            is_mod = bool(perms.manage_messages or perms.administrator)
        except AttributeError:
            pass
        guild = message.guild
        if guild and guild.owner_id == author.id:
            is_founder = True
        # Server boosters approximate sub semantics on Twitch.
        sub_tier = None
        try:
            if getattr(author, "premium_since", None):
                sub_tier = "1000"
        except Exception:
            pass

        try:
            await asyncio.to_thread(
                self.repo.upsert_user, user_id, display_name, source=self.SOURCE
            )
            await asyncio.to_thread(
                self.repo.update_user_badges, user_id,
                sub_tier=sub_tier, sub_months=0,
                is_mod=is_mod, is_vip=False, is_founder=is_founder,
            )

            if await asyncio.to_thread(self.repo.is_opted_out, user_id):
                return

            try:
                fired = await asyncio.to_thread(
                    self.repo.fire_pending_reminders, user_id
                )
                if fired:
                    logger.info(
                        "discord: reminder fired %d for %s on incoming message",
                        fired, display_name,
                    )
            except Exception:
                logger.exception("discord: fire-pending failed")

            # Discord's native reply: stash the parent's author + body for
            # the dashboard's reply-context UI, mirroring the Twitch listener.
            reply_login = None
            reply_body = None
            ref = message.reference
            if ref and ref.resolved and isinstance(ref.resolved, discord.Message):
                parent = ref.resolved
                reply_login = parent.author.display_name or parent.author.name
                reply_body = (parent.content or "")[:200] or None

            await asyncio.to_thread(
                self.repo.insert_message,
                user_id, content,
                reply_parent_login=reply_login,
                reply_parent_body=reply_body,
            )
            unsummarized = await asyncio.to_thread(
                self.repo.unsummarized_count, user_id
            )
        except Exception:
            logger.exception("discord: failed to persist message from %s", display_name)
            return

        try:
            await self.summarizer.maybe_summarize_user(user_id, unsummarized)
        except Exception:
            logger.exception("discord: summarizer trigger failed for %s", display_name)
