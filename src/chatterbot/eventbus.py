"""Process-wide async event bus for SSE notifications.

Replaces (well, supplements) the watermark-poll loop in
`/events/stream`. Two publisher paths:

  1. In-process services (insights cache refresh, transcript chunk
     landing, etc.) call `bus.publish(channel, version)` directly.
  2. The bot — which runs in a separate container — POSTs to the
     dashboard's `/internal/notify` endpoint with a shared-secret
     header, which then publishes into THIS bus.

Subscribers are SSE generators that get their own `asyncio.Queue`.
Slow subscribers don't block fast publishers — the put_nowait
silently drops if a subscriber's queue is full (capped at 64
events). The watermark-poll loop in `/events/stream` is still
the durable fallback for events the bot misses or for cases where
something writes to the DB outside our notification pathway.

Design choices:

- One bus per process (lives on the FastAPI app's lifespan, or
  injected via closure). Cross-process delivery is via HTTP, NOT
  shared-memory or filesystem signals.
- Channels are strings — keep the registry centralised in the
  /events/stream route so adding a channel touches one file.
- Versions are opaque strings; consumers compare for equality.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


class EventBus:
    """In-process pub/sub for SSE notifications.

    Subscribers get an async iterator yielding `(channel, version)`
    tuples. Multiple subscribers run independently; one slow client
    can't block another. Bounded queues (64 events) drop oldest on
    overflow so a runaway publisher doesn't OOM the dashboard.
    """

    QUEUE_MAXSIZE = 64

    def __init__(self):
        self._subscribers: list[asyncio.Queue[tuple[str, str]]] = []
        self._lock = asyncio.Lock()

    async def subscribe(self) -> AsyncIterator[tuple[str, str]]:
        """Async iterator over (channel, version) pairs. Caller is
        expected to consume in a long-running task; bail out by
        breaking from the iteration loop."""
        q: asyncio.Queue[tuple[str, str]] = asyncio.Queue(
            maxsize=self.QUEUE_MAXSIZE,
        )
        async with self._lock:
            self._subscribers.append(q)
        try:
            while True:
                yield await q.get()
        finally:
            async with self._lock:
                if q in self._subscribers:
                    self._subscribers.remove(q)

    def publish(self, channel: str, version: str = "") -> int:
        """Push a notification onto every subscriber's queue. Returns
        the count of subscribers who received it. Drops on overflow
        (subscribers should be quick — one slow consumer doesn't
        deserve to back-pressure the publisher)."""
        delivered = 0
        for q in list(self._subscribers):
            try:
                q.put_nowait((channel, version))
                delivered += 1
            except asyncio.QueueFull:
                # Drop oldest, requeue. Keeps recent state-change
                # signals fresh even when a client is slow.
                try:
                    q.get_nowait()
                    q.put_nowait((channel, version))
                    delivered += 1
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
        return delivered

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


# Singleton instance for the dashboard process. The bot has no
# subscribers — it only talks to the dashboard's bus via HTTP.
_global_bus: EventBus | None = None


def get_bus() -> EventBus:
    """Return the dashboard's process-wide event bus, creating it
    lazily on first call. Safe to import from any module."""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus
