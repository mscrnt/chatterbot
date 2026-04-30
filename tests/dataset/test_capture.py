"""Capture pipeline — opt-in gate, schema_version, async-safety.

The opt-in invariant is the most important property of this module:
no event ever lands on disk unless the streamer explicitly enabled
capture AND the DEK is loaded. Every other test in this file
exercises a different angle of that gate or the encrypt/index path.
"""

from __future__ import annotations

import asyncio

import pytest

from chatterbot.dataset import capture, cipher


# ---- opt-in gate: every "off" branch must drop the event ----


async def _fire_one_call(repo, *, call_site: str = "unit-test"):
    """Helper: simulate one structured LLM call landing in the
    capture pipeline. Async-safe; returns once any background write
    has completed."""
    await capture.record_llm_call(
        repo,
        call_site=call_site,
        model_id="qwen3.5",
        provider="ollama",
        system_prompt=None,
        prompt="hi",
        response_text='{"x":1}',
        response_schema_name="X",
        num_ctx=100,
        num_predict=100,
        think=False,
        latency_ms=1,
    )
    # asyncio.to_thread schedules the actual write — yield once so
    # the worker has a chance to finish before we assert.
    await asyncio.sleep(0)


async def test_capture_off_drops_event(tmp_repo):
    """Default state (toggle off, no DEK): no event ever lands."""
    await _fire_one_call(tmp_repo)
    assert tmp_repo.dataset_event_count() == 0


async def test_capture_on_without_dek_drops_event(tmp_repo):
    """Toggle flipped on, but the DEK was never loaded into memory.
    Capture still drops — there's no key to encrypt under."""
    tmp_repo.set_app_setting("dataset_capture_enabled", "true")
    await _fire_one_call(tmp_repo)
    assert tmp_repo.dataset_event_count() == 0


async def test_capture_on_with_dek_writes(unlocked_repo):
    """The single fully-enabled state writes one row per call."""
    await _fire_one_call(unlocked_repo)
    assert unlocked_repo.dataset_event_count() == 1


async def test_re_disabling_toggle_stops_writes(unlocked_repo):
    """Flipping the toggle off mid-process must immediately stop
    writes — the streamer can pause capture without restarting
    bot/dashboard."""
    await _fire_one_call(unlocked_repo)
    assert unlocked_repo.dataset_event_count() == 1
    unlocked_repo.set_app_setting("dataset_capture_enabled", "false")
    await _fire_one_call(unlocked_repo)
    # No NEW event after the toggle flip.
    assert unlocked_repo.dataset_event_count() == 1


# ---- schema + roundtrip ----


async def test_event_schema_version_persisted(unlocked_repo):
    """Each row stores the schema version code knew about at write
    time. Reader code dispatches on this; wrong version means data
    silently mis-parses."""
    await _fire_one_call(unlocked_repo)
    rows = list(unlocked_repo.iter_dataset_events())
    assert len(rows) == 1
    assert rows[0]["schema_version"] == capture.CAPTURE_SCHEMA_VERSION


async def test_recorded_event_decrypts_to_original_payload(unlocked_repo):
    """End-to-end: write one event through `record_llm_call`, walk the
    index, decrypt, and assert the JSON matches what we passed in."""
    from pathlib import Path
    from chatterbot.dataset.storage import read_record

    await capture.record_llm_call(
        unlocked_repo,
        call_site="test.roundtrip",
        model_id="qwen3.5",
        provider="ollama",
        system_prompt="be terse",
        prompt="who are you?",
        response_text='{"text":"a chatbot"}',
        response_schema_name="ProfileResponse",
        num_ctx=8192,
        num_predict=512,
        think=True,
        latency_ms=42,
    )
    await asyncio.sleep(0)

    rows = list(unlocked_repo.iter_dataset_events())
    assert len(rows) == 1
    row = rows[0]

    data_root = Path(unlocked_repo.db_path).parent
    shard_path = data_root / row["shard_path"]
    rec = read_record(shard_path, row["byte_offset"], row["byte_length"])
    payload = capture.decrypt_event(
        unlocked_repo.dataset_dek(), row["ts"], rec.nonce, rec.ciphertext,
    )

    assert payload["kind"] == capture.EVENT_LLM_CALL
    assert payload["call_site"] == "test.roundtrip"
    assert payload["model_id"] == "qwen3.5"
    assert payload["provider"] == "ollama"
    assert payload["system_prompt"] == "be terse"
    assert payload["prompt"] == "who are you?"
    assert payload["response_text"] == '{"text":"a chatbot"}'
    assert payload["response_schema_name"] == "ProfileResponse"
    assert payload["num_ctx"] == 8192
    assert payload["num_predict"] == 512
    assert payload["think"] is True
    assert payload["latency_ms"] == 42
    assert payload["error"] is None


async def test_capture_failure_records_error_field(unlocked_repo):
    """When the production caller passes `error=...`, that field is
    persisted — useful negative signal for prompt iteration."""
    await capture.record_llm_call(
        unlocked_repo,
        call_site="test.error",
        model_id="qwen3.5",
        provider="ollama",
        system_prompt=None,
        prompt="x",
        response_text="",
        response_schema_name="X",
        num_ctx=None,
        num_predict=None,
        think=False,
        latency_ms=10,
        error="ValidationError: missing field 'foo'",
    )
    await asyncio.sleep(0)

    from pathlib import Path
    from chatterbot.dataset.storage import read_record
    rows = list(unlocked_repo.iter_dataset_events())
    data_root = Path(unlocked_repo.db_path).parent
    rec = read_record(
        data_root / rows[0]["shard_path"],
        rows[0]["byte_offset"], rows[0]["byte_length"],
    )
    payload = capture.decrypt_event(
        unlocked_repo.dataset_dek(), rows[0]["ts"], rec.nonce, rec.ciphertext,
    )
    assert "ValidationError" in payload["error"]


# ---- dataset_event_count by-kind ----


async def test_event_count_by_kind(unlocked_repo):
    """The CLI's `dataset info` calls `dataset_event_count(kind=...)`.
    Verify the SQL filter actually filters."""
    await _fire_one_call(unlocked_repo, call_site="a")
    await _fire_one_call(unlocked_repo, call_site="b")
    await _fire_one_call(unlocked_repo, call_site="c")
    assert unlocked_repo.dataset_event_count() == 3
    assert unlocked_repo.dataset_event_count(kind=capture.EVENT_LLM_CALL) == 3
    assert unlocked_repo.dataset_event_count(kind=capture.EVENT_STREAMER_ACTION) == 0


# ---- since/until filtering on the index ----


async def test_iter_dataset_events_with_since_until(unlocked_repo):
    """Export uses `iter_dataset_events(since_ts=, until_ts=)` to slice
    a date range — verify the SQL bounds work as advertised."""
    # Three events at different timestamps. We can't easily inject
    # custom ts values from outside the capture path, so we write
    # then UPDATE the ts in place — same trick the message factory
    # uses.
    for _ in range(3):
        await _fire_one_call(unlocked_repo)
    rows_all = list(unlocked_repo.iter_dataset_events())
    assert len(rows_all) == 3

    with unlocked_repo._cursor() as cur:  # noqa: SLF001 — test-only
        cur.execute("UPDATE dataset_events SET ts = ? WHERE id = ?",
                    ("2026-04-29T00:00:00Z", rows_all[0]["id"]))
        cur.execute("UPDATE dataset_events SET ts = ? WHERE id = ?",
                    ("2026-04-30T00:00:00Z", rows_all[1]["id"]))
        cur.execute("UPDATE dataset_events SET ts = ? WHERE id = ?",
                    ("2026-05-01T00:00:00Z", rows_all[2]["id"]))

    only_30 = list(unlocked_repo.iter_dataset_events(
        since_ts="2026-04-30T00:00:00Z", until_ts="2026-04-30T23:59:59Z",
    ))
    assert len(only_30) == 1
    assert only_30[0]["ts"] == "2026-04-30T00:00:00Z"
