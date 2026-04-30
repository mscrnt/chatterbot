"""STREAMER_ACTION capture — opt-in gate, hook coverage, schema.

Production mutators on `repo` and `insights` fire the capture hook
right after the SQL write succeeds. Tests confirm:

  - Opt-in gate: capture-off + DEK-not-loaded both drop the event.
  - Hook coverage: every wired mutator (set_insight_state, note
    CRUD, reject_subject, clear_subject_blocklist) actually fires.
  - Schema: each captured event carries the right action_kind /
    item_key / action shape so the dataset reader can dispatch
    cleanly.

Reads back through the index → shard → decrypt → parse pipeline,
same as the LLM_CALL roundtrip tests, so the encrypted-write path
is exercised end-to-end on every test.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chatterbot.dataset import capture
from chatterbot.dataset.capture import (
    ACTION_KIND_INSIGHT_STATE,
    ACTION_KIND_NOTE,
    ACTION_KIND_SUBJECT,
    ACTION_KIND_SUBJECT_BLOCKLIST,
    EVENT_STREAMER_ACTION,
)
from chatterbot.dataset.storage import read_record


# ---- helpers ----


def _decrypt_all(repo) -> list[dict]:
    """Walk the index, decrypt every event, return the parsed payloads
    in id order. Tests filter on `event_kind` / `action_kind` after."""
    data_root = Path(repo.db_path).parent
    out: list[dict] = []
    for row in repo.iter_dataset_events():
        rec = read_record(
            data_root / row["shard_path"], row["byte_offset"], row["byte_length"],
        )
        out.append(capture.decrypt_event(
            repo.dataset_dek(), row["ts"], rec.nonce, rec.ciphertext,
        ))
    return out


# ---- opt-in gate ----


def test_action_capture_off_drops(tmp_repo, make_user):
    """Default state: no DEK + toggle off. Mutator fires, no event
    lands."""
    uid = make_user(name="alice")
    tmp_repo.add_note(uid, "manual note text", embedding=None)
    assert tmp_repo.dataset_event_count() == 0


def test_action_capture_on_without_dek_drops(tmp_repo, make_user):
    """Toggle on but DEK not loaded — capture must still no-op."""
    tmp_repo.set_app_setting("dataset_capture_enabled", "true")
    uid = make_user(name="alice")
    tmp_repo.add_note(uid, "manual note text", embedding=None)
    assert tmp_repo.dataset_event_count() == 0


# ---- set_insight_state hook ----


def test_set_insight_state_captures_action(unlocked_repo):
    """The single chokepoint for dashboard card actions. Verify a
    state transition lands as a STREAMER_ACTION with composed
    item_key (`<insight_kind>:<item_key>`) and the new state in
    `action`."""
    unlocked_repo.set_insight_state(
        kind="talking_point", item_key="42", state="addressed",
        note="said it on stream",
    )
    events = _decrypt_all(unlocked_repo)
    assert len(events) == 1
    e = events[0]
    assert e["kind"] == EVENT_STREAMER_ACTION
    assert e["action_kind"] == ACTION_KIND_INSIGHT_STATE
    assert e["item_key"] == "talking_point:42"
    assert e["action"] == "addressed"
    assert e["note"] == "said it on stream"


def test_set_insight_state_open_clears_and_captures(unlocked_repo):
    """Setting state='open' (or any non-canonical value) clears the
    insight_states row entirely. The capture must still fire — the
    dataset reader cares about both transitions."""
    # First write an addressed state, then re-open it.
    unlocked_repo.set_insight_state(
        kind="thread", item_key="t-7", state="addressed",
    )
    unlocked_repo.set_insight_state(
        kind="thread", item_key="t-7", state="open",
    )
    events = _decrypt_all(unlocked_repo)
    actions = [e["action"] for e in events]
    assert actions == ["addressed", "open"]


# ---- note CRUD hooks ----


def test_add_note_captures_with_origin(unlocked_repo, make_user):
    """add_note records origin (manual / llm) in the action label so
    the dataset reader can tell human-authored from LLM-extracted
    notes — both are useful but mean different things for fine-tuning."""
    uid = make_user(name="alice")
    note_id = unlocked_repo.add_note(
        uid, "alice plays speedruns", embedding=None,
    )
    events = _decrypt_all(unlocked_repo)
    assert len(events) == 1
    e = events[0]
    assert e["action_kind"] == ACTION_KIND_NOTE
    assert e["item_key"] == str(note_id)
    assert e["action"] == "created:manual"
    assert e["note"] == "alice plays speedruns"


def test_update_note_captures_correction(unlocked_repo, make_user):
    """Streamer correcting an LLM-extracted note is gold supervision
    data. Verify the new text rides in `note` so the dataset reader
    can build "before vs after" pairs by joining update/create."""
    uid = make_user(name="alice")
    note_id = unlocked_repo.add_note(uid, "old version", embedding=None)
    unlocked_repo.update_note(note_id, "corrected version")
    events = _decrypt_all(unlocked_repo)
    actions = [(e["action"], e["note"]) for e in events]
    assert actions == [
        ("created:manual", "old version"),
        ("updated", "corrected version"),
    ]


def test_delete_note_captures_negative_signal(unlocked_repo, make_user):
    """Delete is the strongest negative signal — the streamer
    rejected this note outright. Verify it lands."""
    uid = make_user(name="alice")
    note_id = unlocked_repo.add_note(uid, "wrong note", embedding=None)
    unlocked_repo.delete_note(note_id)
    events = _decrypt_all(unlocked_repo)
    delete_events = [e for e in events if e["action"] == "deleted"]
    assert len(delete_events) == 1
    assert delete_events[0]["item_key"] == str(note_id)


# ---- subject reject hooks ----


def test_reject_subject_captures(unlocked_repo, mock_llm):
    """Engaging-subjects rejection — the streamer says "this isn't a
    real subject". Negative supervision for the engaging-subjects
    extractor."""
    from chatterbot.insights import InsightsService
    settings = SimpleNamespace(db_path="ignored", screenshot_interval_seconds=0)
    svc = InsightsService(unlocked_repo, mock_llm, settings)
    svc.reject_subject(slug="abc123", name="Hallucinated Subject")
    events = _decrypt_all(unlocked_repo)
    assert len(events) == 1
    e = events[0]
    assert e["action_kind"] == ACTION_KIND_SUBJECT
    assert e["item_key"] == "abc123"
    assert e["action"] == "rejected"
    assert e["note"] == "Hallucinated Subject"


def test_reject_subject_idempotent_no_double_capture(unlocked_repo, mock_llm):
    """reject_subject is idempotent at the SQL layer (it returns
    early when the subject is already in the blocklist). Capture
    must respect that — re-rejecting the same slug shouldn't write
    a second event."""
    from chatterbot.insights import InsightsService
    settings = SimpleNamespace(db_path="ignored", screenshot_interval_seconds=0)
    svc = InsightsService(unlocked_repo, mock_llm, settings)
    svc.reject_subject(slug="abc123", name="Subject")
    svc.reject_subject(slug="abc123", name="Subject")
    events = _decrypt_all(unlocked_repo)
    assert len(events) == 1


def test_clear_subject_blocklist_captures(unlocked_repo, mock_llm):
    """Clearing the blocklist is a regime-reset signal. Captured so
    the dataset reader can split sessions: "everything before this
    is one feedback regime, everything after is another"."""
    from chatterbot.insights import InsightsService
    settings = SimpleNamespace(db_path="ignored", screenshot_interval_seconds=0)
    svc = InsightsService(unlocked_repo, mock_llm, settings)
    # Pre-load the blocklist so clear has something to clear.
    svc.reject_subject(slug="x", name="X")
    svc.clear_subject_blocklist()
    events = _decrypt_all(unlocked_repo)
    cleared = [e for e in events if e["action_kind"] == ACTION_KIND_SUBJECT_BLOCKLIST]
    assert len(cleared) == 1
    assert cleared[0]["action"] == "cleared"


# ---- schema_version + ts ----


def test_streamer_action_carries_schema_version(unlocked_repo, make_user):
    """Each row stamps the schema version code knew about at write
    time. Reader dispatches on this; pinning the value here so a
    forgotten bump shows up in the test diff."""
    uid = make_user(name="alice")
    unlocked_repo.add_note(uid, "anything", embedding=None)
    rows = list(unlocked_repo.iter_dataset_events())
    assert rows[0]["schema_version"] == capture.CAPTURE_SCHEMA_VERSION
    assert rows[0]["event_kind"] == EVENT_STREAMER_ACTION


# ---- integration: action + LLM events coexist on the same dataset ----


async def test_action_and_llm_events_share_index(unlocked_repo, make_user):
    """Streamer actions and LLM calls land in the same
    dataset_events index but with distinct event_kind. Verify a
    round of mixed events comes back in id-order with kinds
    correctly distinguished."""
    uid = make_user(name="alice")
    unlocked_repo.add_note(uid, "first note", embedding=None)

    # Fire one LLM_CALL too.
    await capture.record_llm_call(
        unlocked_repo,
        call_site="test.mixed",
        model_id="qwen3.5", provider="ollama",
        system_prompt=None, prompt="hi", response_text="{}",
        response_schema_name="X",
        num_ctx=None, num_predict=None,
        think=False, latency_ms=1,
    )
    import asyncio as _asyncio
    await _asyncio.sleep(0)

    unlocked_repo.add_note(uid, "second note", embedding=None)

    rows = list(unlocked_repo.iter_dataset_events())
    kinds = [r["event_kind"] for r in rows]
    assert kinds == [
        EVENT_STREAMER_ACTION,
        capture.EVENT_LLM_CALL,
        EVENT_STREAMER_ACTION,
    ]
