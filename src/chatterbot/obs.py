"""OBS status poller.

Optional, read-only. When `OBS_ENABLED=true`, a background asyncio task in
the dashboard process polls the OBS WebSocket every 10 seconds for two
things:

  - whether the stream output is currently active
  - the current program scene name

The result is cached in memory and surfaced as a small `🔴 LIVE · <scene>`
pill in the dashboard nav. We deliberately don't pull anything else from
OBS — chatterbot is a profiling sidecar, not a broadcast tool. If we ever
want to tag events / topics with the scene that was live at the time, the
state's already here for it.

If `obsws-python` isn't installed (or OBS is unreachable), the poller
goes quiet and the pill stays hidden — no crashes, no log spam.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from .config import Settings

logger = logging.getLogger(__name__)

try:
    from obsws_python import ReqClient as _ReqClient  # type: ignore
    _OBS_AVAILABLE = True
except Exception:
    _ReqClient = None  # type: ignore
    _OBS_AVAILABLE = False

# obsws_python logs full tracebacks at ERROR every time a connection fails.
# We already capture and surface those errors in OBSStatus, so the library's
# logger just doubles up and floods the dashboard log. Pin its threshold to
# CRITICAL so it stays out of the way (we don't emit CRITICAL anywhere).
logging.getLogger("obsws_python").setLevel(logging.CRITICAL)
logging.getLogger("obsws_python.baseclient").setLevel(logging.CRITICAL)
logging.getLogger("websocket").setLevel(logging.CRITICAL)


def _classify_error(exc: Exception) -> str:
    """Turn an obsws/websocket exception into a short streamer-readable hint."""
    name = type(exc).__name__
    msg = str(exc).strip()
    if isinstance(exc, ConnectionRefusedError):
        return (
            "connection refused — OBS WebSocket isn't enabled, "
            "or wrong port. (OBS → Tools → WebSocket Server Settings)"
        )
    if isinstance(exc, TimeoutError):
        return (
            "timed out — host unreachable. Check OBS_HOST in /settings; "
            "if OBS is on Windows and chatterbot is in WSL, try the Windows "
            "host IP or enable mirrored networking."
        )
    if "auth" in msg.lower() or "password" in msg.lower():
        return "authentication failed — wrong OBS_PASSWORD"
    if "not found" in msg.lower():
        return f"name not resolvable — DNS for OBS_HOST failed ({name})"
    return f"{name}: {msg or '(no detail)'}"


@dataclass
class OBSStatus:
    enabled: bool = False
    connected: bool = False
    streaming: bool = False     # OBS streaming output is active
    recording: bool = False     # OBS recording output is active
    scene: str | None = None
    error: str | None = None
    refreshed_at: float | None = None


class OBSStatusService:
    POLL_SECONDS = 10

    def __init__(self, settings: Settings):
        self.settings = settings
        self._status = OBSStatus(enabled=settings.obs_enabled)
        self._lock = asyncio.Lock()
        # Last error string we emitted to the application log. Used so we
        # only log when the error CHANGES — a missing OBS shouldn't spam.
        self._last_logged_error: str | None = None
        self._was_connected: bool = False

    @property
    def status(self) -> OBSStatus:
        return self._status

    async def poll_loop(self) -> None:
        if not self.settings.obs_enabled:
            return  # nothing to do; status stays disabled
        if not _OBS_AVAILABLE:
            self._status = OBSStatus(
                enabled=True, connected=False,
                error="obsws-python not installed", refreshed_at=time.time(),
            )
            logger.warning(
                "OBS_ENABLED=true but obsws-python isn't importable — pill hidden"
            )
            return

        # Initial small delay so the dashboard finishes booting cleanly.
        await asyncio.sleep(2)
        while True:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("obs poll iteration failed")
            await asyncio.sleep(self.POLL_SECONDS)

    async def _poll_once(self) -> None:
        async with self._lock:
            try:
                status = await asyncio.to_thread(self._poll_sync)
            except Exception as e:
                err = _classify_error(e)
                # Only log when the error message changes — an unreachable
                # OBS shouldn't spam the dashboard log every 10s.
                if err != self._last_logged_error:
                    logger.warning("obs unreachable: %s", err)
                    self._last_logged_error = err
                self._status = OBSStatus(
                    enabled=True, connected=False, streaming=False,
                    scene=None, error=err, refreshed_at=time.time(),
                )
                self._was_connected = False
                return
            self._status = status
            # Log the recovery (and re-prime the error dedupe).
            if not self._was_connected:
                logger.info(
                    "obs connected (%s:%d) — streaming=%s recording=%s",
                    self.settings.obs_host, self.settings.obs_port,
                    status.streaming, status.recording,
                )
                self._was_connected = True
            self._last_logged_error = None

    def _poll_sync(self) -> OBSStatus:
        """Open a short-lived ReqClient, pull state, close. Keeping the
        connection short-lived avoids dealing with reconnects ourselves."""
        if _ReqClient is None:
            raise RuntimeError("obsws-python not available")
        client = _ReqClient(
            host=self.settings.obs_host,
            port=self.settings.obs_port,
            password=self.settings.obs_password,
            timeout=4,
        )
        try:
            stream = client.get_stream_status()
            record = client.get_record_status()
            scene_resp = client.get_current_program_scene()
            streaming = bool(getattr(stream, "output_active", False))
            recording = bool(getattr(record, "output_active", False))
            scene_name = (
                getattr(scene_resp, "current_program_scene_name", None)
                or getattr(scene_resp, "scene_name", None)
            )
        finally:
            try:
                client.disconnect()
            except Exception:
                pass
        return OBSStatus(
            enabled=True, connected=True,
            streaming=streaming, recording=recording, scene=scene_name,
            error=None, refreshed_at=time.time(),
        )
