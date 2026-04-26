"""StreamElements realtime socket listener.

Connects to https://realtime.streamelements.com (socket.io v3) using JWT auth and
forwards tip / subscriber / cheer / follower / raid events into the events table.

Auth flow:
  1. Open socket, transport=websocket.
  2. On `connect`, emit `authenticate` with {method: "jwt", token: <JWT>}.
  3. Server emits `authenticated` with {channelId} on success, or `unauthorized`.
  4. Listen on `event` (real) and `event:test` (test buttons in SE dashboard).

Event payload shape (`event`):
  {
    "channel": "<channelId>",
    "type": "tip" | "subscriber" | "cheer" | "follower" | "raid",
    "provider": "twitch" | ...,
    "data": { username, displayName, amount, currency, message, tier, ... },
    "createdAt": "...",
    "_id": "..."
  }

We log the full payload to events.raw_json for audit so future field tweaks don't
cost us historical data.

Hard rule (same as elsewhere): nothing here writes to chat. The bot has no
chat-output surface.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import socketio

from .config import Settings
from .repo import ChatterRepo

logger = logging.getLogger(__name__)


SE_REALTIME_URL = "https://realtime.streamelements.com"


class StreamElementsListener:
    def __init__(self, repo: ChatterRepo, settings: Settings):
        self.repo = repo
        self.settings = settings
        self.sio = socketio.AsyncClient(
            reconnection=True,
            reconnection_delay=2,
            reconnection_delay_max=30,
            randomization_factor=0.3,
            logger=False,
            engineio_logger=False,
        )
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.sio.event
        async def connect() -> None:
            logger.info("streamelements: socket connected, authenticating")
            await self.sio.emit(
                "authenticate",
                {"method": "jwt", "token": self.settings.streamelements_jwt},
            )

        @self.sio.event
        async def disconnect() -> None:
            logger.warning("streamelements: socket disconnected")

        @self.sio.event
        async def connect_error(data: Any) -> None:
            logger.error("streamelements: connect_error: %r", data)

        @self.sio.on("authenticated")
        async def on_authenticated(data: Any) -> None:
            logger.info("streamelements: authenticated channelId=%s", _safe(data, "channelId"))

        @self.sio.on("unauthorized")
        async def on_unauthorized(data: Any) -> None:
            logger.error("streamelements: unauthorized — check JWT. payload=%r", data)

        @self.sio.on("event")
        async def on_event(data: Any) -> None:
            await self._handle_event(data, test=False)

        @self.sio.on("event:test")
        async def on_event_test(data: Any) -> None:
            # SE dashboard "test" buttons fire these; useful for local checking.
            await self._handle_event(data, test=True)

    async def _handle_event(self, payload: Any, *, test: bool) -> None:
        if not isinstance(payload, dict):
            return
        event_type = payload.get("type")
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            data = {}

        # SE provides displayName + username; either may be missing for some types.
        twitch_name = (
            data.get("displayName") or data.get("username") or data.get("name") or ""
        ).strip()

        normalized: tuple[str, float | None, str | None, str | None] | None = (
            _normalize(event_type, data)
        )
        if normalized is None:
            return
        norm_type, amount, currency, message = normalized

        try:
            await asyncio.to_thread(
                self.repo.record_event,
                twitch_name,
                norm_type,
                amount,
                currency,
                message,
                payload,
            )
            logger.info(
                "streamelements: %s%s name=%s amount=%s currency=%s",
                norm_type,
                " (test)" if test else "",
                twitch_name or "?",
                amount,
                currency,
            )
        except Exception:
            logger.exception("failed to persist SE event")

    async def run(self) -> None:
        if not self.settings.streamelements_enabled:
            logger.info("streamelements: disabled")
            return
        if not self.settings.streamelements_jwt:
            logger.warning("streamelements: enabled but JWT is empty — skipping")
            return
        try:
            await self.sio.connect(
                SE_REALTIME_URL,
                transports=["websocket"],
                wait_timeout=10,
            )
            await self.sio.wait()
        except asyncio.CancelledError:
            await self.sio.disconnect()
            raise
        except Exception:
            logger.exception("streamelements: connection failed")


def _normalize(
    event_type: str | None,
    data: dict[str, Any],
) -> tuple[str, float | None, str | None, str | None] | None:
    """Return (normalized_type, amount, currency, message) or None to drop the event."""
    if not event_type:
        return None
    msg = (data.get("message") or "").strip() or None

    if event_type == "tip":
        return "tip", _as_float(data.get("amount")), data.get("currency"), msg

    if event_type == "subscriber":
        # SE conflates new/resub/gift here. amount = months when present.
        # tier values: "1000" | "2000" | "3000" | "prime"
        months = _as_float(data.get("amount"))
        tier = data.get("tier")
        # If gifted, decorate the message so it shows up in the TUI panel.
        sender = data.get("sender")
        if data.get("gifted") and sender:
            base = msg or ""
            msg = (f"[gift from {sender}] " + base).strip()
        return "sub", months, str(tier) if tier else None, msg

    if event_type == "cheer":
        return "cheer", _as_float(data.get("amount")), None, msg

    if event_type == "follower":
        return "follow", None, None, None

    if event_type == "raid":
        return "raid", _as_float(data.get("amount")), None, msg

    return None


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe(d: Any, key: str) -> Any:
    if isinstance(d, dict):
        return d.get(key)
    return None
