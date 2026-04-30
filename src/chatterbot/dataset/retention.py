"""Dataset retention + compaction.

Two knobs control how much captured data we keep:

  - `dataset_retention_days`    — drop events older than N days. 0 = keep
                                  forever.
  - `dataset_retention_max_mb`  — once the total encrypted size exceeds
                                  this cap, drop oldest events until we're
                                  back under. 0 = unbounded.

Compaction runs once a day from a dashboard background loop. It:

  1. Computes the time-cutoff (now - retention_days), if set.
  2. Deletes `dataset_events` rows older than the cutoff.
  3. If total bytes still exceeds `retention_max_mb`, deletes oldest
     events one shard-batch at a time until under the cap.
  4. Removes shard files that no longer have any index rows pointing
     at them. Partial shards (some rows still pointing in) stay on
     disk — partial-rewrite compaction is a slice 6+ concern.

The 'drop oldest' policy is intentional: a streamer who's been
capturing for months and then suddenly hits the 5 GB cap shouldn't
silently lose RECENT events. Old data goes first.

Capture-time guarantee: shards stay append-only. The writer never
reads an existing record back, so a compaction run that deletes
old shards while the writer is producing new ones is safe — the
writer either points at the active shard (which compaction never
touches) or at a fresh shard it just rotated to.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..repo import ChatterRepo

logger = logging.getLogger(__name__)


DEFAULT_RETENTION_DAYS = 30
DEFAULT_RETENTION_MAX_MB = 5000


@dataclass
class CompactionResult:
    """Outcome of one compaction pass — surfaced for the dashboard
    to display "last run pruned X events / Y MB" and so tests can
    assert specific behaviour."""
    rows_pruned_by_age: int = 0
    rows_pruned_by_size: int = 0
    bytes_freed: int = 0
    shards_deleted: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def rows_pruned(self) -> int:
        return self.rows_pruned_by_age + self.rows_pruned_by_size

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.finished_at - self.started_at)


# ---- settings access ----


def _read_int_setting(repo: "ChatterRepo", key: str, default: int) -> int:
    """Tolerant int read: missing / non-numeric values return the
    default. Settings are stored as strings in app_settings."""
    raw = repo.get_app_setting(key)
    if not raw:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def get_retention_days(repo: "ChatterRepo") -> int:
    """Effective retention window in days. 0 = forever."""
    return max(0, _read_int_setting(
        repo, "dataset_retention_days", DEFAULT_RETENTION_DAYS,
    ))


def get_retention_max_mb(repo: "ChatterRepo") -> int:
    """Effective on-disk cap in MiB. 0 = unbounded."""
    return max(0, _read_int_setting(
        repo, "dataset_retention_max_mb", DEFAULT_RETENTION_MAX_MB,
    ))


# ---- the compaction itself ----


def compact(repo: "ChatterRepo") -> CompactionResult:
    """One full compaction pass. Idempotent — running twice produces
    no extra deletions on the second run. Returns a CompactionResult
    so the caller can log / surface metrics."""
    result = CompactionResult(started_at=time.time())
    days = get_retention_days(repo)
    max_mb = get_retention_max_mb(repo)
    cap_bytes = max_mb * 1024 * 1024 if max_mb > 0 else 0

    # 1) Age-based prune.
    if days > 0:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).isoformat(timespec="seconds")
        try:
            with repo._cursor() as cur:  # noqa: SLF001 — internal compaction
                cur.execute(
                    "SELECT COUNT(*) AS n, COALESCE(SUM(byte_length), 0) AS sz "
                    "FROM dataset_events WHERE ts < ?",
                    (cutoff,),
                )
                row = cur.fetchone()
                aged_n = int(row["n"]) if row else 0
                aged_bytes = int(row["sz"]) if row else 0
                if aged_n:
                    cur.execute(
                        "DELETE FROM dataset_events WHERE ts < ?",
                        (cutoff,),
                    )
            result.rows_pruned_by_age = aged_n
            result.bytes_freed += aged_bytes
            if aged_n:
                logger.info(
                    "dataset retention: pruned %d events older than %s",
                    aged_n, cutoff,
                )
        except Exception:
            logger.exception(
                "dataset retention: age-based prune failed",
            )

    # 2) Size-based prune (only when cap is set).
    if cap_bytes > 0:
        try:
            current = _current_total_bytes(repo)
            while current > cap_bytes:
                # Pull a batch of the OLDEST 200 events. Batching keeps
                # the SQL fast on big indexes while still converging.
                with repo._cursor() as cur:  # noqa: SLF001
                    cur.execute(
                        "SELECT id, byte_length FROM dataset_events "
                        "ORDER BY id ASC LIMIT 200"
                    )
                    rows = cur.fetchall()
                    if not rows:
                        break
                    ids = [int(r["id"]) for r in rows]
                    freed = sum(int(r["byte_length"]) for r in rows)
                    placeholders = ",".join("?" for _ in ids)
                    cur.execute(
                        f"DELETE FROM dataset_events WHERE id IN ({placeholders})",
                        ids,
                    )
                result.rows_pruned_by_size += len(ids)
                result.bytes_freed += freed
                current -= freed
            if result.rows_pruned_by_size:
                logger.info(
                    "dataset retention: pruned %d events to fit %d MiB cap",
                    result.rows_pruned_by_size, max_mb,
                )
        except Exception:
            logger.exception(
                "dataset retention: size-based prune failed",
            )

    # 3) Shard cleanup. A shard with no index rows pointing at it is
    # dead weight — delete the file. Partial shards (some rows still
    # pointing in) stay; we don't rewrite mid-shard.
    try:
        result.shards_deleted = _delete_orphan_shards(repo)
    except Exception:
        logger.exception("dataset retention: shard cleanup failed")

    result.finished_at = time.time()
    if (result.rows_pruned or result.shards_deleted):
        logger.info(
            "dataset retention complete: %d events pruned, %d shards "
            "deleted, %.1f MB freed in %.2fs",
            result.rows_pruned,
            result.shards_deleted,
            result.bytes_freed / (1024 * 1024),
            result.duration_seconds,
        )
    return result


def _current_total_bytes(repo: "ChatterRepo") -> int:
    """Total bytes currently indexed. Same query the /dataset page
    uses for the status panel — single SUM."""
    with repo._cursor() as cur:  # noqa: SLF001
        cur.execute(
            "SELECT COALESCE(SUM(byte_length), 0) AS s FROM dataset_events"
        )
        row = cur.fetchone()
        return int(row["s"]) if row else 0


def _delete_orphan_shards(repo: "ChatterRepo") -> int:
    """Delete shard files that have no events pointing at them. Walks
    the shards/ directory once, joins against `dataset_events`, drops
    the difference. The active (most-recently-modified) shard is
    NEVER deleted even if empty — the writer has it open for append
    and would crash on next write."""
    from .storage import shards_dir, SHARD_SUBDIR

    data_root = Path(repo.db_path).parent
    sd = shards_dir(data_root)
    if not sd.exists():
        return 0

    # Live shards: any path still referenced by dataset_events.
    live: set[str] = set()
    with repo._cursor() as cur:  # noqa: SLF001
        cur.execute("SELECT DISTINCT shard_path FROM dataset_events")
        for r in cur.fetchall():
            live.add(r["shard_path"])
    # The shard_path stored in the index is relative to the data dir
    # (see capture._write_event_sync). Normalize for comparison.
    live_filenames = {
        Path(p).name for p in live
    }

    files = sorted(sd.glob("*.cbds.bin"), key=lambda p: p.stat().st_mtime)
    if not files:
        return 0
    # Spare the most recent file — it's likely the active append target.
    files_to_check = files[:-1]

    deleted = 0
    for f in files_to_check:
        if f.name in live_filenames:
            continue
        try:
            f.unlink()
            deleted += 1
            logger.info("dataset retention: deleted orphan shard %s", f.name)
        except OSError:
            logger.exception(
                "dataset retention: could not delete shard %s", f.name,
            )
    return deleted
