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
    error: str | None = None
    refreshed_at: float | None = None


class TwitchService:
    POLL_SECONDS = 60

    def __init__(self, settings: Settings):
        self.settings = settings
        self._status = TwitchStatus(enabled=bool(settings.twitch_oauth_token and settings.twitch_channel))
        self._lock = asyncio.Lock()
        self._client_id: str | None = None
        self._last_logged_error: str | None = None

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
            logger.info("twitch: validated as %s (client_id ok)", data.get("login"))
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
        async with self._lock:
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    r = await client.get(
                        "https://api.twitch.tv/helix/streams",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Client-Id": self._client_id,
                        },
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
                    enabled=True, error=err, refreshed_at=time.time(),
                )
                return
            self._last_logged_error = None
            if not data:
                # Stream is offline — keep enabled, clear live state.
                self._status = TwitchStatus(
                    enabled=True, is_live=False, refreshed_at=time.time(),
                )
                return
            stream = data[0]
            self._status = TwitchStatus(
                enabled=True,
                is_live=True,
                viewer_count=int(stream.get("viewer_count") or 0),
                thumbnail_url_template=stream.get("thumbnail_url") or None,
                game_name=stream.get("game_name") or None,
                title=stream.get("title") or None,
                started_at=stream.get("started_at") or None,
                refreshed_at=time.time(),
            )

    @staticmethod
    def thumbnail_url(template: str | None, width: int = 320, height: int = 180) -> str | None:
        """Substitute `{width}` / `{height}` placeholders in a Twitch
        thumbnail URL. Returns None when no template (offline)."""
        if not template:
            return None
        return template.replace("{width}", str(width)).replace("{height}", str(height))
