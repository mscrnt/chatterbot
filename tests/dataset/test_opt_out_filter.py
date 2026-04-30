"""Capture-time opt-out filter — `referenced_user_ids` gates LLM_CALL
events.

The promise: a chatter who is opted out NEVER ends up encrypted on
disk via the dataset capture. Existing repo queries already filter
opt_out=0 at SQL time so opted-out users don't appear in prompts in
the first place; this test layer confirms the explicit
declaration mechanism (callers passing `referenced_user_ids=[...]`)
also drops the entire event when any listed user is opted out.

Also verifies the inverse: a non-opted-out user list produces a
captured event with the user_ids preserved in the payload, so a
future redaction pass can target specific chatters without
re-running the original opt-out check.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from chatterbot.dataset import capture
from chatterbot.dataset.storage import read_record


async def _fire(repo, *, referenced_user_ids=None):
    """One canned LLM_CALL through the capture pipeline. Returns
    after the (asyncio.to_thread) write has had a chance to land."""
    await capture.record_llm_call(
        repo,
        call_site="test.opt_out",
        model_id="qwen3.5",
        provider="ollama",
        system_prompt=None,
        prompt="prompt referencing chatters",
        response_text='{"x":1}',
        response_schema_name="X",
        num_ctx=None,
        num_predict=None,
        think=False,
        latency_ms=10,
        referenced_user_ids=referenced_user_ids,
    )
    await asyncio.sleep(0)


# ---- baseline: no filter when referenced_user_ids is None ----


async def test_no_user_ids_captures_unconditionally(unlocked_repo):
    """`referenced_user_ids=None` (default) means "I don't know /
    don't care" — capture proceeds as before. Pinning this so a
    refactor that accidentally treats None as "drop everything"
    fails loudly."""
    await _fire(unlocked_repo, referenced_user_ids=None)
    assert unlocked_repo.dataset_event_count() == 1


async def test_empty_user_ids_captures(unlocked_repo):
    """Empty list is also "nothing to filter on." Same outcome as
    None — capture proceeds."""
    await _fire(unlocked_repo, referenced_user_ids=[])
    assert unlocked_repo.dataset_event_count() == 1


# ---- positive case: non-opt-out users captured + payload stamped ----


async def test_non_opt_out_users_captured(unlocked_repo, make_user):
    """When all referenced users are opted in, the event lands AND
    the user_ids are persisted in the payload (lets a future
    redaction pass target specific chatters without rerunning the
    opt-out query)."""
    alice = make_user(name="alice", opt_out=False)
    bob = make_user(name="bob", opt_out=False)
    await _fire(unlocked_repo, referenced_user_ids=[alice, bob])

    rows = list(unlocked_repo.iter_dataset_events())
    assert len(rows) == 1

    # Decrypt and verify the user_ids made it into the payload.
    data_root = Path(unlocked_repo.db_path).parent
    rec = read_record(
        data_root / rows[0]["shard_path"],
        rows[0]["byte_offset"],
        rows[0]["byte_length"],
    )
    payload = capture.decrypt_event(
        unlocked_repo.dataset_dek(), rows[0]["ts"], rec.nonce, rec.ciphertext,
    )
    assert set(payload["referenced_user_ids"]) == {alice, bob}


# ---- negative case: any opt-out user drops the entire event ----


async def test_one_opt_out_user_drops_entire_event(unlocked_repo, make_user):
    """The promise: even if 9 of 10 referenced users are opted in,
    one opt-out chatter triggers a full skip. Capture is all-or-
    nothing per event; partial redaction is not a thing here."""
    alice = make_user(name="alice", opt_out=False)
    bob = make_user(name="bob", opt_out=True)  # the opt-out one
    carol = make_user(name="carol", opt_out=False)

    await _fire(unlocked_repo, referenced_user_ids=[alice, bob, carol])
    assert unlocked_repo.dataset_event_count() == 0


async def test_unknown_user_ids_dont_block_capture(unlocked_repo, make_user):
    """Unknown user_ids (not in the users table) shouldn't block —
    they have no opt-out preference, so we don't have consent to
    honour. Pinning this so a refactor that fails-closed on unknown
    ids doesn't silently kill capture."""
    alice = make_user(name="alice", opt_out=False)
    await _fire(unlocked_repo, referenced_user_ids=[alice, "unknown-id"])
    assert unlocked_repo.dataset_event_count() == 1


# ---- mixed: opted-out user filter doesn't affect future events ----


async def test_filter_is_per_event_not_per_repo(unlocked_repo, make_user):
    """Dropping one event because of an opt-out chatter must NOT
    leave the capture system in a wedged state. The next event
    (without that user) should land normally."""
    alice = make_user(name="alice", opt_out=False)
    bob = make_user(name="bob", opt_out=True)

    # First event drops on the opt-out filter.
    await _fire(unlocked_repo, referenced_user_ids=[alice, bob])
    assert unlocked_repo.dataset_event_count() == 0

    # Second event has only opted-in users — must land.
    await _fire(unlocked_repo, referenced_user_ids=[alice])
    assert unlocked_repo.dataset_event_count() == 1


# ---- repo helper: any_opted_out semantics ----


def test_any_opted_out_helper(tmp_repo, make_user):
    """The `any_opted_out` helper is also used directly by the
    capture pipeline. Pin its semantics here independently so a
    refactor in either place catches divergence at this test, not
    in production."""
    a = make_user(name="alice", opt_out=False)
    b = make_user(name="bob", opt_out=True)
    c = make_user(name="carol", opt_out=False)

    assert tmp_repo.any_opted_out([a]) is False
    assert tmp_repo.any_opted_out([b]) is True
    assert tmp_repo.any_opted_out([a, c]) is False
    assert tmp_repo.any_opted_out([a, b, c]) is True
    assert tmp_repo.any_opted_out([]) is False
    # Unknown ids = no consent on file = don't filter.
    assert tmp_repo.any_opted_out(["never-seen"]) is False
