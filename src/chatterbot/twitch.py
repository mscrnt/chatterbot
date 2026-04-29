"""Twitch Helix poller — viewer count + stream thumbnail.

Runs as a background task in the dashboard process. We don't require the
streamer to set TWITCH_CLIENT_ID separately; we derive it by validating
the existing TWITCH_OAUTH_TOKEN at startup. Twitch's /oauth2/validate
returns the client_id that issued the token, which is the same one Helix
needs in the `Client-Id` header.

Quietly does nothing if:
  - the token is missing or invalid
  - the configured TWITCH_CHANNEL is empty
  - Helix is unreachable

No special scopes required for `Get Streams`. Returns:
  - viewer_count    (int, when live)
  - thumbnail_url   (CDN URL, with `{width}x{height}` placeholder)
  - is_live         (bool)
  - game_name, title, started_at (light extras for dashboard surfacing)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

from .config import Settings
from .obs import OBSStatusService

logger = logging.getLogger(__name__)


@dataclass
class TwitchStatus:
    enabled: bool = False
    is_live: bool = False
    viewer_count: int = 0
    thumbnail_url_template: str | None = None
    game_name: str | None = None
    title: str | None = None
    started_at: str | None = None
    # Streamer-set descriptors on the channel. Returned by
    # /helix/streams alongside the live state. Useful as channel
    # context for the LLM ("Souls-like", "Speedrunning", etc.).
    tags: list[str] | None = None
    # Self-applied content classifications from /helix/channels:
    # "Mature", "Profanity", etc. Lets the LLM respect content
    # boundaries the streamer has already declared.
    content_classification_labels: list[str] | None = None
    # Cached broadcaster id from /oauth2/validate — needed to call
    # /helix/channels by id without an extra lookup each poll.
    broadcaster_id: str | None = None
    # True when we skipped a poll because OBS confirmed the streamer is
    # offline. Helix would just answer "not live" anyway, so we save the
    # call (and the rate-limit budget).
    auto_paused: bool = False
    error: str | None = None
    refreshed_at: float | None = None

    @property
    def stream_uptime_seconds(self) -> int:
        """How long the streamer has been live, derived from
        `started_at`. 0 when offline / unparseable. The LLM uses this
        as energy context — a 4-hour stream isn't the same vibe as a
        20-minute one."""
        if not self.is_live or not self.started_at:
            return 0
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
            delta = datetime.now(timezone.utc) - dt
            return max(0, int(delta.total_seconds()))
        except (TypeError, ValueError):
            return 0

    def format_for_llm(self, *, authoritative: bool = False) -> str:
        """Render this status as a CHANNEL CONTEXT block ready to
        prepend to an LLM prompt. Single source of truth shared by
        every "informed" call site (talking points, engaging subjects,
        thread recaps, transcript group summaries) so the LLM sees the
        same Helix snapshot in the same shape regardless of caller.

        Returns "" when offline / no signals — caller can concat
        unconditionally.

        `authoritative=True` adds an explicit "do NOT guess from the
        screenshot" instruction. Used by callers that also pass an
        image, where we want to head off "Valheim, judging by the
        graphics" hallucinations when Helix already named the game."""
        if not self.is_live:
            return ""
        bits: list[str] = []
        if self.game_name:
            bits.append(f"playing/streaming: {self.game_name}")
        if self.title:
            bits.append(f"stream title: {self.title}")
        if self.tags:
            bits.append("tags: " + ", ".join(self.tags[:8]))
        if self.content_classification_labels:
            bits.append(
                "content labels: "
                + ", ".join(self.content_classification_labels)
            )
        viewers = int(self.viewer_count or 0)
        if viewers > 0:
            # Coarse tier so the LLM gets scale signal without
            # overfitting to a specific count.
            if viewers < 25:
                tier = f"{viewers} viewers (small chat)"
            elif viewers < 200:
                tier = f"{viewers} viewers (medium)"
            elif viewers < 1000:
                tier = f"{viewers} viewers (busy)"
            else:
                tier = f"{viewers} viewers (very busy — chat is fast)"
            bits.append(tier)
        uptime = self.stream_uptime_seconds
        if uptime > 0:
            hours, rem = divmod(uptime, 3600)
            mins = rem // 60
            if hours:
                bits.append(f"live for {hours}h {mins}m")
            else:
                bits.append(f"live for {mins}m")
        if not bits:
            return ""
        # Lead with a hard KNOWN GAME line when authoritative — vision
        # models over-weight the screenshot, and burying the game name
        # inside a paragraph of "channel context" lets them skim past
        # it. A standalone all-caps line at the top of the prompt is
        # much harder to ignore.
        prefix = ""
        if authoritative and self.game_name:
            prefix = (
                f"KNOWN GAME: {self.game_name}\n"
                "(This game name comes from Twitch's live API and is "
                "authoritative. The screenshot may LOOK like a "
                "different game — IGNORE THAT and use this name.)\n\n"
            )
        if authoritative:
            header = (
                "CHANNEL CONTEXT (authoritative — from Twitch's live "
                "API. Use these names directly; do NOT guess the game "
                "or title from the screenshot)"
            )
        else:
            header = (
                "CHANNEL CONTEXT (the streamer is currently live; "
                "chat may reference this)"
            )
        return prefix + f"{header}: " + " · ".join(bits) + "\n\n"


class TwitchService:
    POLL_SECONDS = 60

    def __init__(self, settings: Settings, *, obs: OBSStatusService | None = None):
        self.settings = settings
        self._status = TwitchStatus(enabled=bool(settings.twitch_oauth_token and settings.twitch_channel))
        self._lock = asyncio.Lock()
        self._client_id: str | None = None
        self._broadcaster_id: str | None = None
        self._last_logged_error: str | None = None
        # Optional OBS coupling — when present, the Helix poll is skipped
        # while OBS reports the streamer offline.
        self.obs = obs
        self._was_auto_paused: bool = False

    @property
    def status(self) -> TwitchStatus:
        return self._status

    async def poll_loop(self) -> None:
        if not (self.settings.twitch_oauth_token and self.settings.twitch_channel):
            return  # no creds → no Helix
        # Initial 5s grace so the dashboard finishes booting cleanly.
        await asyncio.sleep(5)
        if not await self._init_client_id():
            return
        while True:
            # OBS-driven auto-pause: when OBS confirms the streamer is
            # offline, Helix would just answer is_live=false. Save the
            # call. Defaults to "not paused" on any uncertainty.
            if self.obs is not None and self.obs.status.is_streamer_offline():
                if not self._was_auto_paused:
                    logger.info(
                        "twitch: helix poll auto-paused (OBS reports offline)"
                    )
                    self._was_auto_paused = True
                self._status = TwitchStatus(
                    enabled=True, is_live=False, auto_paused=True,
                    refreshed_at=time.time(),
                )
                await asyncio.sleep(self.POLL_SECONDS)
                continue
            if self._was_auto_paused:
                logger.info("twitch: helix poll auto-resumed")
                self._was_auto_paused = False
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("twitch poll failed")
            await asyncio.sleep(self.POLL_SECONDS)

    async def _init_client_id(self) -> bool:
        token = self.settings.twitch_oauth_token.removeprefix("oauth:").strip()
        if not token:
            return False
        login = self.settings.twitch_channel.strip()
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get(
                    "https://id.twitch.tv/oauth2/validate",
                    headers={"Authorization": f"OAuth {token}"},
                )
                if r.status_code != 200:
                    logger.warning(
                        "twitch: token validate returned %d — viewer-count + thumbnail disabled",
                        r.status_code,
                    )
                    self._status = TwitchStatus(
                        enabled=True, error=f"token validate {r.status_code}",
                        refreshed_at=time.time(),
                    )
                    return False
                data = r.json()
                cid = (data or {}).get("client_id")
                if not cid:
                    logger.warning("twitch: no client_id in validate response")
                    return False
                self._client_id = cid
                token_owner_login = (data or {}).get("login") or "?"
                token_owner_id = (data or {}).get("user_id") or None

                # Resolve the *watched* channel's broadcaster_id.
                #   - Self-streaming case (token owner == watched
                #     channel): the validate response already carries
                #     the user_id we need. Skip the extra Helix call.
                #   - Watching-someone-else case: the token's user_id
                #     would route /helix/channels at the wrong account
                #     (the bug we just fixed); resolve the channel's
                #     own id via /helix/users?login=<channel>.
                self._broadcaster_id = None
                self_streaming = (
                    bool(login)
                    and bool(token_owner_login)
                    and login.lower() == token_owner_login.lower()
                )
                if self_streaming:
                    self._broadcaster_id = token_owner_id
                elif login:
                    try:
                        r2 = await client.get(
                            "https://api.twitch.tv/helix/users",
                            headers={
                                "Authorization": f"Bearer {token}",
                                "Client-Id": cid,
                            },
                            params={"login": login},
                        )
                        r2.raise_for_status()
                        users = (r2.json() or {}).get("data") or []
                        if users:
                            self._broadcaster_id = users[0].get("id") or None
                    except Exception:
                        # Non-fatal: we just won't get content
                        # classification labels. Game / title / tags
                        # all still come from /helix/streams.
                        logger.exception(
                            "twitch: helix /users lookup for %r failed", login,
                        )

            # One log line that makes the relationship obvious — the
            # `validated as X` phrasing was confusing when the token
            # owner and watched channel are different people.
            logger.info(
                "twitch: token owner=%s, watching=%s (client_id ok, broadcaster_id=%s)",
                token_owner_login, login or "—",
                self._broadcaster_id or "unresolved",
            )
            return True
        except Exception as e:
            logger.exception("twitch: token validate raised")
            self._status = TwitchStatus(
                enabled=True, error=type(e).__name__,
                refreshed_at=time.time(),
            )
            return False

    async def _poll_once(self) -> None:
        if not self._client_id:
            return
        token = self.settings.twitch_oauth_token.removeprefix("oauth:").strip()
        login = self.settings.twitch_channel.strip()
        headers = {
            "Authorization": f"Bearer {token}",
            "Client-Id": self._client_id,
        }
        async with self._lock:
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    r = await client.get(
                        "https://api.twitch.tv/helix/streams",
                        headers=headers,
                        params={"user_login": login},
                    )
                r.raise_for_status()
                data = r.json().get("data") or []
            except Exception as e:
                err = type(e).__name__
                if err != self._last_logged_error:
                    logger.warning("twitch: helix /streams failed: %s", err)
                    self._last_logged_error = err
                self._status = TwitchStatus(
                    enabled=True, error=err,
                    broadcaster_id=self._broadcaster_id,
                    refreshed_at=time.time(),
                )
                return
            self._last_logged_error = None
            if not data:
                # Stream is offline — keep enabled, clear live state.
                self._status = TwitchStatus(
                    enabled=True, is_live=False,
                    broadcaster_id=self._broadcaster_id,
                    refreshed_at=time.time(),
                )
                return
            stream = data[0]
            # Tags now ride along with /streams responses (Twitch
            # deprecated the old per-tag endpoint years ago). Stored
            # as-is — display layers handle truncation.
            stream_tags = [
                str(t).strip() for t in (stream.get("tags") or []) if t
            ]

            # Best-effort second call to /helix/channels for the
            # content-classification labels. /streams doesn't return
            # those. Failure is non-fatal — we just don't surface the
            # field. Same headers, same lock window.
            content_labels: list[str] = []
            if self._broadcaster_id:
                try:
                    async with httpx.AsyncClient(timeout=8.0) as client:
                        r2 = await client.get(
                            "https://api.twitch.tv/helix/channels",
                            headers=headers,
                            params={"broadcaster_id": self._broadcaster_id},
                        )
                    r2.raise_for_status()
                    cdata = (r2.json().get("data") or [])
                    if cdata:
                        labels = cdata[0].get("content_classification_labels") or []
                        content_labels = [str(x).strip() for x in labels if x]
                except Exception as e:
                    # Don't escalate — channels-info is bonus context,
                    # not load-bearing. Log once per error type so the
                    # log doesn't fill up.
                    err = type(e).__name__
                    if err != self._last_logged_error:
                        logger.info(
                            "twitch: helix /channels skipped (%s)", err,
                        )
                        self._last_logged_error = err

            self._status = TwitchStatus(
                enabled=True,
                is_live=True,
                viewer_count=int(stream.get("viewer_count") or 0),
                thumbnail_url_template=stream.get("thumbnail_url") or None,
                game_name=stream.get("game_name") or None,
                title=stream.get("title") or None,
                started_at=stream.get("started_at") or None,
                tags=stream_tags or None,
                content_classification_labels=content_labels or None,
                broadcaster_id=self._broadcaster_id,
                refreshed_at=time.time(),
            )

    @staticmethod
    def thumbnail_url(template: str | None, width: int = 320, height: int = 180) -> str | None:
        """Substitute `{width}` / `{height}` placeholders in a Twitch
        thumbnail URL. Returns None when no template (offline)."""
        if not template:
            return None
        return template.replace("{width}", str(width)).replace("{height}", str(height))
