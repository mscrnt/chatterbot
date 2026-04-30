"""CONTEXT_SNAPSHOT capture — opt-in gate, payload composition,
self-contained shape.

The CONTEXT_SNAPSHOT event makes a captured bundle self-contained:
even without the chatters.db attached, the bundle reader has the
recent-message window, transcript chunks, active threads, and
channel context that surrounded each LLM_CALL. These tests pin
both the gate and the snapshot's shape so a refactor that drops a
field shows up loudly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chatterbot.dataset import capture
from chatterbot.dataset.loops import _build_snapshot
from chatterbot.dataset.storage import read_record


# ---- opt-in gate ----


def test_context_snapshot_off_drops(tmp_repo):
    """Default state: capture off. record_context_snapshot_safe must
    no-op even if a snapshot is provided."""
    capture.record_context_snapshot_safe(
        tmp_repo, snapshot={"messages": [], "transcripts": []},
    )
    assert tmp_repo.dataset_event_count() == 0


def test_context_snapshot_no_dek_drops(tmp_repo):
    """Toggle on but DEK not loaded — still drops, same behaviour
    as LLM_CALL and STREAMER_ACTION captures."""
    tmp_repo.set_app_setting("dataset_capture_enabled", "true")
    capture.record_context_snapshot_safe(
        tmp_repo, snapshot={"messages": [], "transcripts": []},
    )
    assert tmp_repo.dataset_event_count() == 0


def test_context_snapshot_writes_when_enabled(unlocked_repo):
    """Full opt-in: one snapshot lands as a CONTEXT_SNAPSHOT
    event in the index."""
    capture.record_context_snapshot_safe(
        unlocked_repo,
        snapshot={"lookback_minutes": 10, "messages": []},
    )
    rows = list(unlocked_repo.iter_dataset_events())
    assert len(rows) == 1
    assert rows[0]["event_kind"] == capture.EVENT_CONTEXT_SNAPSHOT


# ---- payload roundtrip ----


def test_context_snapshot_payload_decrypts_to_input(unlocked_repo):
    """End-to-end: pass a snapshot dict, decrypt the captured event,
    verify the snapshot field matches what we sent. Pinning the
    `snapshot` key shape because the dataset reader will dispatch
    on it."""
    payload_in = {
        "lookback_minutes": 10,
        "messages": [
            {"id": 1, "name": "alice", "content": "hello"},
            {"id": 2, "name": "bob", "content": "hey"},
        ],
        "transcripts": [{"ts": "2026-04-30T12:00:00Z", "text": "playing through"}],
        "threads": [],
        "channel_context": "now playing: re9",
    }
    capture.record_context_snapshot_safe(unlocked_repo, snapshot=payload_in)

    rows = list(unlocked_repo.iter_dataset_events())
    assert len(rows) == 1
    data_root = Path(unlocked_repo.db_path).parent
    rec = read_record(
        data_root / rows[0]["shard_path"],
        rows[0]["byte_offset"], rows[0]["byte_length"],
    )
    payload_out = capture.decrypt_event(
        unlocked_repo.dataset_dek(), rows[0]["ts"],
        rec.nonce, rec.ciphertext,
    )
    assert payload_out["kind"] == capture.EVENT_CONTEXT_SNAPSHOT
    assert payload_out["snapshot"] == payload_in


# ---- _build_snapshot composition ----


def test_build_snapshot_includes_recent_messages(tmp_repo, make_message):
    """The loop helper that composes a snapshot must surface the
    last N messages with their core fields. Using `tmp_repo`
    (no capture) so this test only exercises the pure composer."""
    make_message(content="first message", name="alice")
    make_message(content="second message", name="bob")

    snapshot = _build_snapshot(tmp_repo, lookback_minutes=15)
    assert snapshot["lookback_minutes"] == 15
    assert "captured_at" in snapshot
    contents = [m["content"] for m in snapshot["messages"]]
    assert "first message" in contents
    assert "second message" in contents
    # Each message dict has the fields the bundle reader needs.
    sample = snapshot["messages"][0]
    assert {"id", "user_id", "name", "ts", "content"} <= sample.keys()


def test_build_snapshot_truncates_long_content(tmp_repo, make_message):
    """Snapshot messages clip to 400 chars so a chat full of pasted
    walls of text doesn't blow up the encrypted blob size. Pin the
    truncation behaviour."""
    long_text = "x" * 1000
    make_message(content=long_text, name="alice")
    snapshot = _build_snapshot(tmp_repo, lookback_minutes=15)
    assert len(snapshot["messages"][0]["content"]) == 400


def test_build_snapshot_resilient_to_missing_twitch_status(tmp_repo):
    """`twitch_status=None` is the common case for installs without
    a Helix token. Snapshot should fall back to empty channel_context
    rather than crashing."""
    snapshot = _build_snapshot(tmp_repo, twitch_status=None)
    assert snapshot["channel_context"] == ""
    # Skeleton fields all present.
    assert snapshot["messages"] == []
    assert snapshot["transcripts"] == []
    assert snapshot["threads"] == []


def test_build_snapshot_uses_twitch_status_format_for_llm(tmp_repo):
    """When TwitchService is wired, channel_context picks up the
    same `format_for_llm` block the LLM informed-call prompts use.
    Pin the integration point."""
    class _FakeStatus:
        def format_for_llm(self, *, authoritative=False):
            return "now streaming: ng4 hardmode"

    class _FakeTwitch:
        @property
        def status(self):
            return _FakeStatus()

    snapshot = _build_snapshot(tmp_repo, twitch_status=_FakeTwitch())
    assert "ng4 hardmode" in snapshot["channel_context"]
