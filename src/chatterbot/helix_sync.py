"""Twitch Helix roster sync.

Periodically polls Helix for chatter-roster data the bot can't get from
IRC alone:

  - VIPs       (channel:read:vips)
  - Moderators (moderation:read)
  - Subscribers + tier (channel:read:subscriptions)
  - Followers + follow date (moderator:read:followers)

Distinct from `twitch.TwitchService` which polls the LIVE-STREAM data
(viewer count, thumbnail, current game). This service polls the
CHATTER ROSTER and merges it into the users table.

Cadences (live-only — the loop pauses while OBS reports offline):
  - VIPs / mods:      every 15 min
  - Subs:             every 30 min
  - Followers (page): every 5 min  (first page only — newest first)

Token scopes are checked once at startup. Endpoints we lack scopes for
are silently skipped — the rest still applies. No retries on transient
5xx; we just try again on the next tick.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

from .config import Settings
from .obs import OBSStatusService
from .repo import ChatterRepo

logger = logging.getLogger(__name__)


HELIX = "https://api.twitch.tv/helix"
VALIDATE = "https://id.twitch.tv/oauth2/validate"


@dataclass
class HelixSyncStatus:
    enabled: bool = False
    connected: bool = False
    broadcaster_id: str | None = None
    broadcaster_login: str | None = None
    available_endpoints: list[str] = field(default_factory=list)
    last_vips_at: float | None = None
    last_mods_at: float | None = None
    last_subs_at: float | None = None
    last_followers_at: float | None = None
    counts_vips: int = 0
    counts_mods: int = 0
    counts_subs: int = 0
    counts_new_followers: int = 0
    last_error: str | None = None


class HelixSyncService:
    POLL_TICK_SECONDS = 60
    VIP_INTERVAL = 15 * 60
    MOD_INTERVAL = 15 * 60
    SUBS_INTERVAL = 30 * 60
    FOLLOWERS_INTERVAL = 5 * 60
    FOLLOWERS_PAGE_SIZE = 100  # newest-first page; we don't walk the full follower list

    def __init__(
        self,
        settings: Settings,
        repo: ChatterRepo,
        *,
        obs: OBSStatusService | None = None,
    ):
        self.settings = settings
        self.repo = repo
        self.obs = obs
        self.status = HelixSyncStatus(
            enabled=bool(
                settings.twitch_oauth_token and settings.twitch_channel
            ),
        )
        self._client_id: str | None = None
        self._scopes: set[str] = set()

    @property
    def configured(self) -> bool:
        s = self.settings
        return bool(s.twitch_oauth_token and s.twitch_channel)

    async def run(self) -> None:
        """Top-level loop. No-op when not configured.

        Holds a single `httpx.AsyncClient` for the entire loop
        lifetime — connection-pooled keep-alive saves the TCP /
        TLS handshake on every tick. Old behavior opened a fresh
        client per tick which paid handshake on each call."""
        if not self.configured:
            return
        # Stagger boot vs other Helix-touching services.
        await asyncio.sleep(8)
        if not await self._init():
            return
        last = {"vips": 0.0, "mods": 0.0, "subs": 0.0, "followers": 0.0}
        try:
            async with httpx.AsyncClient(
                timeout=10.0,
                headers={
                    "Authorization": f"Bearer {self._token()}",
                    "Client-Id": self._client_id or "",
                },
            ) as client:
                while True:
                    # Auto-pause off-stream — Helix data doesn't change
                    # much between streams and we'd rather save the
                    # rate-limit budget.
                    if self.obs is not None and self.obs.status.is_streamer_offline():
                        await asyncio.sleep(self.POLL_TICK_SECONDS)
                        continue

                    now = time.time()
                    if self._has_vip_scope() and now - last["vips"] >= self.VIP_INTERVAL:
                        await self._sync_vips(client)
                        last["vips"] = now
                    if self._has_mod_scope() and now - last["mods"] >= self.MOD_INTERVAL:
                        await self._sync_mods(client)
                        last["mods"] = now
                    if "channel:read:subscriptions" in self._scopes and now - last["subs"] >= self.SUBS_INTERVAL:
                        await self._sync_subs(client)
                        last["subs"] = now
                    if self._has_followers_scope() and now - last["followers"] >= self.FOLLOWERS_INTERVAL:
                        await self._sync_followers(client)
                        last["followers"] = now
                    await asyncio.sleep(self.POLL_TICK_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("helix_sync: unrecoverable loop error")

    def _token(self) -> str:
        return (self.settings.twitch_oauth_token or "").removeprefix("oauth:").strip()

    def _has_vip_scope(self) -> bool:
        return bool(self._scopes & {"channel:read:vips", "moderator:read:vips"})

    def _has_mod_scope(self) -> bool:
        return bool(self._scopes & {"moderation:read", "channel:manage:moderators"})

    def _has_followers_scope(self) -> bool:
        return bool(
            self._scopes & {
                "moderator:read:followers", "channel:read:subscriptions",
            }
        )

    async def _init(self) -> bool:
        """Validate token, get scopes + broadcaster_id. Returns True on
        success. Subsequent ticks reuse the cached client_id + scope set."""
        token = self._token()
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get(
                    VALIDATE,
                    headers={"Authorization": f"OAuth {token}"},
                )
            if r.status_code != 200:
                self.status.last_error = f"validate {r.status_code}"
                logger.warning("helix_sync: token validate %d", r.status_code)
                return False
            data = r.json() or {}
            self._client_id = data.get("client_id")
            login = data.get("login") or self.settings.twitch_channel
            self._scopes = set(data.get("scopes") or [])
            self.status.broadcaster_login = login
            self.status.connected = True
            self.status.available_endpoints = sorted(
                ep for ep, ok in (
                    ("vips",       self._has_vip_scope()),
                    ("mods",       self._has_mod_scope()),
                    ("subs",       "channel:read:subscriptions" in self._scopes),
                    ("followers",  self._has_followers_scope()),
                ) if ok
            )
            if not self.status.available_endpoints:
                logger.info(
                    "helix_sync: token has no roster scopes — "
                    "VIP/mod/sub/follower sync will be disabled. "
                    "Add channel:read:vips, moderation:read, "
                    "channel:read:subscriptions, moderator:read:followers."
                )
                return False
            # Resolve broadcaster_id by login.
            async with httpx.AsyncClient(
                timeout=8.0,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Client-Id": self._client_id or "",
                },
            ) as client:
                r = await client.get(f"{HELIX}/users", params={"login": login})
                r.raise_for_status()
                rows = r.json().get("data") or []
            if not rows:
                self.status.last_error = "broadcaster lookup failed"
                logger.warning(
                    "helix_sync: couldn't resolve broadcaster_id for login=%s",
                    login,
                )
                return False
            self.status.broadcaster_id = rows[0]["id"]
            logger.info(
                "helix_sync: ready (broadcaster=%s, scopes-ok=%s)",
                self.status.broadcaster_login,
                ", ".join(self.status.available_endpoints) or "(none)",
            )
            return True
        except Exception as e:
            self.status.last_error = f"{type(e).__name__}: {e}"
            logger.exception("helix_sync: init failed")
            return False

    async def _paged(
        self,
        client: httpx.AsyncClient,
        path: str,
        params: dict,
        *,
        max_pages: int = 200,
    ) -> list[dict]:
        """Walk a Helix cursor-paginated endpoint."""
        out: list[dict] = []
        cursor: str | None = None
        for _ in range(max_pages):
            p = dict(params)
            if cursor:
                p["after"] = cursor
            r = await client.get(f"{HELIX}{path}", params=p)
            if r.status_code in (401, 403):
                # Scope missing or token rotated — bail; init can re-run.
                self.status.last_error = (
                    f"{path} {r.status_code}: " + (r.json() or {}).get("message", "")[:120]
                )
                logger.warning("helix_sync: %s", self.status.last_error)
                return out
            r.raise_for_status()
            data = r.json()
            out.extend(data.get("data") or [])
            cursor = (data.get("pagination") or {}).get("cursor")
            if not cursor:
                break
        return out

    async def _sync_vips(self, client: httpx.AsyncClient) -> None:
        try:
            rows = await self._paged(
                client, "/channels/vips",
                {"broadcaster_id": self.status.broadcaster_id, "first": 100},
            )
        except Exception:
            logger.exception("helix_sync: /channels/vips failed")
            return
        snap = {r["user_id"]: r["user_login"] for r in rows}
        counts = await asyncio.to_thread(
            self.repo.helix_apply_role_snapshot, vips=snap,
        )
        self.status.last_vips_at = time.time()
        self.status.counts_vips = len(snap)
        logger.info(
            "helix_sync: VIPs synced — %d active (touched %d rows)",
            len(snap), counts["vips"],
        )

    async def _sync_mods(self, client: httpx.AsyncClient) -> None:
        try:
            rows = await self._paged(
                client, "/moderation/moderators",
                {"broadcaster_id": self.status.broadcaster_id, "first": 100},
            )
        except Exception:
            logger.exception("helix_sync: /moderation/moderators failed")
            return
        snap = {r["user_id"]: r["user_login"] for r in rows}
        counts = await asyncio.to_thread(
            self.repo.helix_apply_role_snapshot, mods=snap,
        )
        self.status.last_mods_at = time.time()
        self.status.counts_mods = len(snap)
        logger.info(
            "helix_sync: mods synced — %d active (touched %d rows)",
            len(snap), counts["mods"],
        )

    async def _sync_subs(self, client: httpx.AsyncClient) -> None:
        try:
            rows = await self._paged(
                client, "/subscriptions",
                {"broadcaster_id": self.status.broadcaster_id, "first": 100},
            )
        except Exception:
            logger.exception("helix_sync: /subscriptions failed")
            return
        snap: dict[str, dict] = {
            r["user_id"]: {
                "login": r.get("user_login") or r.get("user_name", ""),
                "tier": str(r.get("tier") or ""),
            }
            for r in rows
        }
        counts = await asyncio.to_thread(
            self.repo.helix_apply_role_snapshot, subs=snap,
        )
        self.status.last_subs_at = time.time()
        self.status.counts_subs = len(snap)
        logger.info(
            "helix_sync: subs synced — %d active (touched %d rows)",
            len(snap), counts["subs"],
        )

    async def _sync_followers(self, client: httpx.AsyncClient) -> None:
        """First-page-only poll. Helix returns followers newest-first,
        so this catches new followers without walking the full list
        (which on a 100k-follower channel would burn quota for nothing
        new). Older follower rows are populated via the one-time
        backfill script; this loop only ever ADDs followed_at, never
        clears. Schema: COALESCE(existing, snapshot)."""
        try:
            r = await client.get(
                f"{HELIX}/channels/followers",
                params={
                    "broadcaster_id": self.status.broadcaster_id,
                    "first": self.FOLLOWERS_PAGE_SIZE,
                },
            )
            if r.status_code in (401, 403):
                self.status.last_error = f"followers {r.status_code}"
                return
            r.raise_for_status()
            data = r.json().get("data") or []
        except Exception:
            logger.exception("helix_sync: /channels/followers failed")
            return
        snap: dict[str, dict] = {}
        for row in data:
            uid = row.get("user_id")
            if not uid:
                continue
            snap[uid] = {
                "login": row.get("user_login") or row.get("user_name", ""),
                "followed_at": row.get("followed_at"),
            }
        counts = await asyncio.to_thread(
            self.repo.helix_apply_role_snapshot, followers=snap,
        )
        self.status.last_followers_at = time.time()
        self.status.counts_new_followers = counts["followers"]
        logger.info(
            "helix_sync: followers page synced — %d in latest page (touched %d rows)",
            len(snap), counts["followers"],
        )
