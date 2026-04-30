"""Background loops that the dashboard process runs once dataset
capture is active.

Two loops:

  - `context_snapshot_loop` — every ~5 min, writes one CONTEXT_SNAPSHOT
    event with the current "shape" of chat / transcripts / threads /
    channel context. Lets future fine-tune bundles be self-contained
    without needing the full SQLite DB attached.

  - `retention_loop` — every ~24 h, runs `retention.compact(repo)`.
    Prunes old events + drops orphan shards.

Both loops:
  - Run in the dashboard process only (single source of truth — the
    bot process never starts these).
  - Silently no-op when capture is off OR the DEK isn't loaded.
  - Boot delays so the dashboard finishes startup before the first
    pass; jitter keeps wakeups from aligning with other loops.
  - Hot-path checks before doing any work so a streamer with capture
    off pays nothing.

Loops live under `dataset/` (not `web/app.py`) so the optional
`dataset` extra absence stays a single import-error catch in
`try_unlock_at_startup` rather than scattering try/except across
the web layer.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..repo import ChatterRepo

logger = logging.getLogger(__name__)


# Cadence defaults. Loops read app_settings overrides so streamers
# can dial them without restarting; falling back here keeps the
# defaults visible in one place.
CONTEXT_SNAPSHOT_INTERVAL_SECONDS = 300            # 5 min
CONTEXT_SNAPSHOT_LOOKBACK_MINUTES = 10
CONTEXT_SNAPSHOT_MAX_MESSAGES = 100
CONTEXT_SNAPSHOT_MAX_TRANSCRIPTS = 30

RETENTION_LOOP_INTERVAL_SECONDS = 24 * 3600        # 24 h


# ---- context snapshot composition ----


def _build_snapshot(
    repo: "ChatterRepo",
    *,
    lookback_minutes: int = CONTEXT_SNAPSHOT_LOOKBACK_MINUTES,
    max_messages: int = CONTEXT_SNAPSHOT_MAX_MESSAGES,
    max_transcripts: int = CONTEXT_SNAPSHOT_MAX_TRANSCRIPTS,
    twitch_status=None,
) -> dict:
    """Compose one CONTEXT_SNAPSHOT payload. Pure function — reads
    only from the repo + (optional) twitch_status. Doesn't itself
    decide whether to write; the loop does that gating.

    Each section is best-effort: if a sub-query fails, that section
    drops to an empty default rather than aborting the whole snapshot."""
    snapshot: dict = {
        "lookback_minutes": int(lookback_minutes),
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "messages": [],
        "transcripts": [],
        "threads": [],
        "channel_context": "",
    }

    # Recent messages — already opt-out filtered at SQL level by
    # _CLEAN_MSG_WHERE inside `recent_messages`.
    try:
        msgs = repo.recent_messages(
            limit=int(max_messages),
            within_minutes=int(lookback_minutes),
        )
        snapshot["messages"] = [
            {
                "id": int(m.id),
                "user_id": m.user_id,
                "name": m.name,
                "ts": m.ts,
                "content": m.content[:400],
                "reply_parent_login": m.reply_parent_login,
            }
            for m in msgs
        ]
    except Exception:
        logger.debug("snapshot: recent_messages failed", exc_info=True)

    # Streamer voice — recent transcript chunks.
    try:
        transcripts = repo.recent_transcripts(
            within_minutes=int(lookback_minutes),
            limit=int(max_transcripts),
        )
        snapshot["transcripts"] = [
            {"ts": t.ts, "text": t.text[:600]}
            for t in transcripts
            if (t.text or "").strip()
        ]
    except Exception:
        logger.debug("snapshot: recent_transcripts failed", exc_info=True)

    # Active conversation threads — what chat is grouping into.
    try:
        threads = repo.list_threads(
            status_filter="active", query="", limit=10,
        )
        snapshot["threads"] = [
            {
                "id": int(t.id),
                "title": t.title,
                "drivers": list(t.drivers or []),
                "recap": getattr(t, "recap", "") or "",
                "msg_count": int(getattr(t, "msg_count", 0) or 0),
            }
            for t in threads
        ]
    except Exception:
        logger.debug("snapshot: list_threads failed", exc_info=True)

    # Channel context (game / title / viewer count) — only when
    # TwitchService is wired into the dashboard. Format string
    # mirrors what the LLM informed-call prompts inline.
    if twitch_status is not None:
        try:
            ts = getattr(twitch_status, "status", None)
            if ts is not None:
                snapshot["channel_context"] = ts.format_for_llm(authoritative=False)
        except Exception:
            logger.debug("snapshot: channel_context failed", exc_info=True)

    return snapshot


# ---- the loops ----


async def context_snapshot_loop(
    repo: "ChatterRepo",
    *,
    twitch_status=None,
    interval_seconds: int = CONTEXT_SNAPSHOT_INTERVAL_SECONDS,
) -> None:
    """Every `interval_seconds`, write one CONTEXT_SNAPSHOT to the
    dataset (if capture is on + DEK is loaded). Silent no-op
    otherwise — checks the gate every iteration so toggling capture
    on at runtime starts producing snapshots without a restart."""
    # Boot delay + jitter so we don't align with the other dashboard
    # loops (talking_points / engaging_subjects / open_questions all
    # do their first pass within ~25s).
    await asyncio.sleep(60 + random.uniform(0, 15))
    while True:
        try:
            if repo.dataset_capture_enabled() and repo.dataset_dek() is not None:
                snapshot = await asyncio.to_thread(
                    _build_snapshot, repo,
                    twitch_status=twitch_status,
                )
                from .capture import record_context_snapshot_safe
                await asyncio.to_thread(
                    record_context_snapshot_safe, repo,
                    snapshot=snapshot,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("dataset context-snapshot iteration failed")
        await asyncio.sleep(int(interval_seconds))


async def retention_loop(
    repo: "ChatterRepo",
    *,
    interval_seconds: int = RETENTION_LOOP_INTERVAL_SECONDS,
) -> None:
    """Daily compaction: prune by retention_days + retention_max_mb,
    drop orphan shard files. Same opt-in / DEK gates as the snapshot
    loop — capture-off installs pay nothing."""
    # Big boot delay so a freshly-started dashboard isn't doing disk
    # housekeeping during the streamer's first 10 minutes when they
    # might be tweaking settings + watching live updates.
    await asyncio.sleep(600 + random.uniform(0, 60))
    while True:
        try:
            if repo.dataset_capture_enabled() and repo.dataset_dek() is not None:
                from .retention import compact
                await asyncio.to_thread(compact, repo)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("dataset retention iteration failed")
        await asyncio.sleep(int(interval_seconds))
