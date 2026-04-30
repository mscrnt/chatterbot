"""app_settings KV store — cache + write-through invariants.

Background loops hammer `get_app_setting` once per tick (engaging-
subjects blocklist, watermarks, dataset toggle). The TTL'd cache
guarantees that's a dict lookup, not a per-call SELECT. These tests
pin that contract.
"""

from __future__ import annotations

import time

import pytest


def test_get_returns_none_for_missing_key(tmp_repo):
    """Missing key returns None, not "" or KeyError. Lots of code
    relies on `(value or default)` being safe."""
    assert tmp_repo.get_app_setting("never-set") is None


def test_set_then_get_roundtrip(tmp_repo):
    tmp_repo.set_app_setting("foo", "bar")
    assert tmp_repo.get_app_setting("foo") == "bar"


def test_set_overwrites(tmp_repo):
    """Same key set twice keeps the latest value (KV semantics)."""
    tmp_repo.set_app_setting("k", "v1")
    tmp_repo.set_app_setting("k", "v2")
    assert tmp_repo.get_app_setting("k") == "v2"


def test_delete_removes_key(tmp_repo):
    tmp_repo.set_app_setting("k", "v")
    tmp_repo.delete_app_setting("k")
    assert tmp_repo.get_app_setting("k") is None


def test_set_is_write_through_to_cache(tmp_repo):
    """set_app_setting updates the in-memory cache so the very next
    `get_app_setting` returns the new value without a DB round trip.
    Without write-through, callers waiting on the TTL would briefly
    read stale values."""
    tmp_repo.set_app_setting("hot", "first")
    assert tmp_repo.get_app_setting("hot") == "first"

    tmp_repo.set_app_setting("hot", "second")
    # Read immediately — the cache MUST reflect the new value, not
    # wait for the 60s TTL to expire.
    assert tmp_repo.get_app_setting("hot") == "second"


def test_get_all_filters_nones(tmp_repo):
    """get_all_app_settings is used by the dashboard's debug surface.
    None-valued cache entries (from a delete) shouldn't appear."""
    tmp_repo.set_app_setting("a", "1")
    tmp_repo.set_app_setting("b", "2")
    tmp_repo.delete_app_setting("a")
    all_kv = tmp_repo.get_all_app_settings()
    assert "b" in all_kv
    assert "a" not in all_kv


def test_dataset_capture_enabled_helper(tmp_repo):
    """The hot-path toggle is case-insensitive on the value because
    historical configs sometimes wrote 'True' instead of 'true'.
    Documenting the exact contract here."""
    assert tmp_repo.dataset_capture_enabled() is False
    tmp_repo.set_app_setting("dataset_capture_enabled", "true")
    assert tmp_repo.dataset_capture_enabled() is True
    tmp_repo.set_app_setting("dataset_capture_enabled", "TRUE")
    assert tmp_repo.dataset_capture_enabled() is True
    tmp_repo.set_app_setting("dataset_capture_enabled", "false")
    assert tmp_repo.dataset_capture_enabled() is False
    # Anything else is treated as off — no silent yes-from-noise.
    tmp_repo.set_app_setting("dataset_capture_enabled", "yes")
    assert tmp_repo.dataset_capture_enabled() is False
