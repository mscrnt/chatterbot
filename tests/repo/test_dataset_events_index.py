"""Index-table tests for `dataset_events` — direct SQL pinning so
the index migration doesn't drift away from what `iter_dataset_events`
expects.
"""

from __future__ import annotations


def test_table_has_expected_columns(tmp_repo):
    """The migration must produce these columns. If a future change
    drops one, the export pipeline silently breaks; test pins it."""
    with tmp_repo._cursor() as cur:  # noqa: SLF001 — schema introspection
        cur.execute("PRAGMA table_info(dataset_events)")
        cols = {r["name"] for r in cur.fetchall()}
    expected = {
        "id", "ts", "event_kind", "shard_path",
        "byte_offset", "byte_length", "schema_version",
    }
    assert expected.issubset(cols)


def test_indexes_exist(tmp_repo):
    """Indexed lookups by ts and event_kind back the export filters
    and `dataset_event_count(kind=...)`. Missing indexes don't break
    correctness but turn O(log n) walks into O(n)."""
    with tmp_repo._cursor() as cur:  # noqa: SLF001
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='dataset_events'"
        )
        idx_names = {r["name"] for r in cur.fetchall()}
    assert "idx_dataset_events_ts" in idx_names
    assert "idx_dataset_events_kind" in idx_names


def test_insert_and_iter_roundtrip(tmp_repo):
    """The two-method contract: insert_dataset_event writes a row,
    iter_dataset_events yields it back with the same fields."""
    new_id = tmp_repo.insert_dataset_event(
        ts="2026-04-30T00:00:00Z",
        event_kind="llm_call",
        shard_path="dataset/shards/test.cbds.bin",
        byte_offset=0,
        byte_length=100,
        schema_version=1,
    )
    assert new_id > 0
    rows = list(tmp_repo.iter_dataset_events())
    assert len(rows) == 1
    r = rows[0]
    assert r["ts"] == "2026-04-30T00:00:00Z"
    assert r["event_kind"] == "llm_call"
    assert r["shard_path"] == "dataset/shards/test.cbds.bin"
    assert r["byte_offset"] == 0
    assert r["byte_length"] == 100
    assert r["schema_version"] == 1


def test_event_count_filters_by_kind(tmp_repo):
    """The CLI's `info` command relies on per-kind counts."""
    tmp_repo.insert_dataset_event(
        ts="2026-04-30T00:00:00Z", event_kind="llm_call",
        shard_path="x", byte_offset=0, byte_length=1, schema_version=1,
    )
    tmp_repo.insert_dataset_event(
        ts="2026-04-30T00:00:01Z", event_kind="llm_call",
        shard_path="x", byte_offset=0, byte_length=1, schema_version=1,
    )
    tmp_repo.insert_dataset_event(
        ts="2026-04-30T00:00:02Z", event_kind="streamer_action",
        shard_path="x", byte_offset=0, byte_length=1, schema_version=1,
    )
    assert tmp_repo.dataset_event_count() == 3
    assert tmp_repo.dataset_event_count(kind="llm_call") == 2
    assert tmp_repo.dataset_event_count(kind="streamer_action") == 1
    assert tmp_repo.dataset_event_count(kind="never_emitted") == 0
