"""Retention / compaction tests.

The compaction loop is the only thing that PRUNES dataset rows. A
regression here means streamers either lose data they wanted (if it
prunes too aggressively) or fill their disk silently (if it doesn't
prune enough). Both are bad for trust in the feature.

Tests use direct `insert_dataset_event` to seed rows with controlled
timestamps and byte_lengths so we don't have to time-travel the
actual capture pipeline.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chatterbot.dataset import capture
from chatterbot.dataset.retention import (
    DEFAULT_RETENTION_DAYS,
    DEFAULT_RETENTION_MAX_MB,
    compact,
    get_retention_days,
    get_retention_max_mb,
)
from chatterbot.dataset.storage import shards_dir


def _seed_event(repo, *, ts: str, byte_length: int, kind: str = "llm_call",
                shard_filename: str = "test.cbds.bin") -> int:
    """Write one fake index row with a fixed timestamp + size. Doesn't
    encrypt anything — these tests only care about the PRUNE behaviour
    of the SQL queries."""
    return repo.insert_dataset_event(
        ts=ts,
        event_kind=kind,
        shard_path=f"dataset/shards/{shard_filename}",
        byte_offset=0,
        byte_length=byte_length,
        schema_version=1,
    )


def _iso(days_ago: int = 0) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(days=days_ago)
    ).isoformat(timespec="seconds")


# ---- settings access ----


def test_default_retention_settings(tmp_repo):
    """No app_setting → default retention values. Pinning the
    defaults so an inadvertent change shows up in tests, not as
    silent data loss for streamers."""
    assert get_retention_days(tmp_repo) == DEFAULT_RETENTION_DAYS
    assert get_retention_max_mb(tmp_repo) == DEFAULT_RETENTION_MAX_MB


def test_retention_settings_read_from_app_settings(tmp_repo):
    tmp_repo.set_app_setting("dataset_retention_days", "7")
    tmp_repo.set_app_setting("dataset_retention_max_mb", "100")
    assert get_retention_days(tmp_repo) == 7
    assert get_retention_max_mb(tmp_repo) == 100


def test_retention_settings_zero_means_unlimited(tmp_repo):
    """0 = forever / unbounded. The compaction skip-paths depend
    on this contract."""
    tmp_repo.set_app_setting("dataset_retention_days", "0")
    tmp_repo.set_app_setting("dataset_retention_max_mb", "0")
    assert get_retention_days(tmp_repo) == 0
    assert get_retention_max_mb(tmp_repo) == 0


def test_retention_settings_tolerate_garbage(tmp_repo):
    """A typo or hand-edited bad value falls back to the default —
    we never want compaction to error-loop because of a stray
    'forever' string in app_settings."""
    tmp_repo.set_app_setting("dataset_retention_days", "lots")
    tmp_repo.set_app_setting("dataset_retention_max_mb", "")
    assert get_retention_days(tmp_repo) == DEFAULT_RETENTION_DAYS
    assert get_retention_max_mb(tmp_repo) == DEFAULT_RETENTION_MAX_MB


# ---- age-based pruning ----


def test_compact_prunes_age(tmp_repo):
    """Events older than retention_days drop; newer ones stay."""
    tmp_repo.set_app_setting("dataset_retention_days", "30")
    tmp_repo.set_app_setting("dataset_retention_max_mb", "0")

    _seed_event(tmp_repo, ts=_iso(60), byte_length=100)   # too old
    _seed_event(tmp_repo, ts=_iso(45), byte_length=100)   # too old
    new_id = _seed_event(tmp_repo, ts=_iso(1), byte_length=100)   # fresh

    result = compact(tmp_repo)
    assert result.rows_pruned_by_age == 2
    assert result.bytes_freed == 200

    rows = list(tmp_repo.iter_dataset_events())
    assert len(rows) == 1
    assert rows[0]["id"] == new_id


def test_compact_age_zero_keeps_everything(tmp_repo):
    """retention_days=0 means 'keep forever' — even ancient events
    must not get pruned."""
    tmp_repo.set_app_setting("dataset_retention_days", "0")
    tmp_repo.set_app_setting("dataset_retention_max_mb", "0")

    _seed_event(tmp_repo, ts=_iso(1000), byte_length=100)
    _seed_event(tmp_repo, ts=_iso(1), byte_length=100)
    result = compact(tmp_repo)
    assert result.rows_pruned == 0
    assert tmp_repo.dataset_event_count() == 2


# ---- size-based pruning ----


def test_compact_prunes_to_size_cap(tmp_repo):
    """Once total bytes exceed the MB cap, oldest rows drop until
    we're under. The 'oldest first' policy is intentional —
    streamers shouldn't lose recent data when an old burst of
    captures pushes them over the cap."""
    tmp_repo.set_app_setting("dataset_retention_days", "0")
    # 1 MiB cap → ~1 MB
    tmp_repo.set_app_setting("dataset_retention_max_mb", "1")

    # Each event is 500 KB; 5 events = 2.5 MiB total, expect prune
    # to bring us back under 1 MiB.
    five_hundred_kb = 500 * 1024
    ids = []
    for d in (5, 4, 3, 2, 1):
        ids.append(_seed_event(tmp_repo, ts=_iso(d), byte_length=five_hundred_kb))

    result = compact(tmp_repo)
    assert result.rows_pruned_by_size > 0

    rows = list(tmp_repo.iter_dataset_events())
    # Whatever survived must be the NEWEST events (highest ids).
    surviving_ids = {r["id"] for r in rows}
    pruned_ids = set(ids) - surviving_ids
    # Every pruned id must be smaller than every surviving id —
    # confirms the 'oldest first' policy.
    if pruned_ids and surviving_ids:
        assert max(pruned_ids) < min(surviving_ids)


def test_compact_size_zero_skips_size_check(tmp_repo):
    """retention_max_mb=0 means unbounded. Even when the index is
    huge, no size-based pruning should fire."""
    tmp_repo.set_app_setting("dataset_retention_days", "0")
    tmp_repo.set_app_setting("dataset_retention_max_mb", "0")

    for d in range(1, 11):
        _seed_event(tmp_repo, ts=_iso(d), byte_length=10 * 1024 * 1024)

    result = compact(tmp_repo)
    assert result.rows_pruned == 0
    assert tmp_repo.dataset_event_count() == 10


# ---- shard cleanup ----


def test_compact_drops_orphan_shard_files(tmp_repo):
    """When every event pointing at a shard file is pruned, the
    file itself must get removed — otherwise old shards live on
    disk forever even though their metadata is gone."""
    data_root = Path(tmp_repo.db_path).parent
    sd = shards_dir(data_root)
    # Two shard files; one will be orphaned, one stays alive.
    orphan = sd / "old.cbds.bin"
    alive = sd / "new.cbds.bin"
    orphan.write_bytes(b"\x00" * 64)
    alive.write_bytes(b"\x00" * 64)

    tmp_repo.set_app_setting("dataset_retention_days", "30")
    tmp_repo.set_app_setting("dataset_retention_max_mb", "0")

    # Old event points at orphan (will be pruned by age).
    tmp_repo.insert_dataset_event(
        ts=_iso(60), event_kind="llm_call",
        shard_path="dataset/shards/old.cbds.bin",
        byte_offset=0, byte_length=64, schema_version=1,
    )
    # New event points at alive (stays).
    tmp_repo.insert_dataset_event(
        ts=_iso(1), event_kind="llm_call",
        shard_path="dataset/shards/new.cbds.bin",
        byte_offset=0, byte_length=64, schema_version=1,
    )
    # Active-shard guard: scanner spares the most-recently-modified
    # file. Make `alive` newer than `orphan` so orphan is eligible.
    import os
    import time as _time
    old_mtime = _time.time() - 600
    os.utime(orphan, (old_mtime, old_mtime))

    result = compact(tmp_repo)
    assert result.rows_pruned_by_age == 1
    assert result.shards_deleted >= 1
    assert not orphan.exists()
    assert alive.exists()  # alive shard untouched


def test_compact_idempotent(tmp_repo):
    """Running compaction twice produces no extra deletions on the
    second pass — the first pass should leave the index in a
    steady state for current settings."""
    tmp_repo.set_app_setting("dataset_retention_days", "30")
    _seed_event(tmp_repo, ts=_iso(60), byte_length=100)
    _seed_event(tmp_repo, ts=_iso(1), byte_length=100)

    first = compact(tmp_repo)
    second = compact(tmp_repo)

    assert first.rows_pruned == 1
    assert second.rows_pruned == 0
    assert tmp_repo.dataset_event_count() == 1
