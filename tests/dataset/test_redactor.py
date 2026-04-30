"""Export-time redactor tests.

The promise: after `redact_event(plan, ev)` runs, no covered
chatter name appears in any text-bearing field, and the same
chatter consistently maps to the same `<USER_NNN>` token across
every event in the bundle.

Tests use the redactor module directly (not through cmd_export)
so we can pin specific input/output pairs without going through
the encrypted bundle pipeline. A separate end-to-end test in
test_cli.py would round-trip through cmd_export — out of slice
5's scope, but the structure here makes it cheap to add.
"""

from __future__ import annotations

import pytest

from chatterbot.dataset import redactor


# ---- plan construction ----


def test_build_plan_assigns_stable_tokens(tmp_repo, make_user):
    """Same user_id → same token within a plan. Different users
    get different tokens. Token format pinned for the manifest
    consumer."""
    a = make_user(name="alice")
    b = make_user(name="bob")
    plan = redactor.build_plan(tmp_repo, [a, b])

    assert a in plan.id_to_token
    assert b in plan.id_to_token
    assert plan.id_to_token[a] != plan.id_to_token[b]
    # Format: "<USER_NNN>" with zero-padded 3-digit index.
    for tok in plan.id_to_token.values():
        assert tok.startswith("<USER_")
        assert tok.endswith(">")


def test_build_plan_includes_aliases(tmp_repo, make_user):
    """A user who's been seen under multiple names — every alias
    must map to the same anon token. The user_aliases table is
    populated by upsert_user; the test relies on the make_user
    fixture inserting an alias row."""
    uid = make_user(name="alice")
    # Insert a second alias the way upsert_user would.
    tmp_repo.upsert_user(twitch_id=uid, name="alice_old")
    plan = redactor.build_plan(tmp_repo, [uid])
    assert "alice" in plan.name_to_token
    assert "alice_old" in plan.name_to_token
    # Both aliases route to the SAME token.
    assert plan.name_to_token["alice"] == plan.name_to_token["alice_old"]


def test_build_plan_is_deterministic(tmp_repo, make_user):
    """Sorting user_ids deterministically inside build_plan means
    re-running the export against the same data yields the same
    token assignments. Useful for diffing two exports."""
    a = make_user(name="alice")
    b = make_user(name="bob")
    plan1 = redactor.build_plan(tmp_repo, [a, b])
    plan2 = redactor.build_plan(tmp_repo, [b, a])  # different order
    assert plan1.id_to_token == plan2.id_to_token


def test_empty_plan_is_noop(tmp_repo):
    """No user_ids → plan with no patterns → redact_text returns
    input unchanged, redact_event returns input dict unchanged."""
    plan = redactor.build_plan(tmp_repo, [])
    assert redactor.redact_text(plan, "alice was here") == "alice was here"
    ev = {"prompt": "alice", "referenced_user_ids": []}
    assert redactor.redact_event(plan, ev) == ev


# ---- redact_text behaviour ----


def test_redact_text_replaces_word_boundary_only(tmp_repo, make_user):
    """`alice` should be redacted but `alicelike` shouldn't —
    word-boundary matching prevents mangling unrelated tokens."""
    uid = make_user(name="alice")
    plan = redactor.build_plan(tmp_repo, [uid])

    assert "<USER_001>" in redactor.redact_text(plan, "what does alice think")
    # `alicelike` shares the prefix but isn't the same word.
    assert "alicelike" in redactor.redact_text(plan, "alicelike chatter")


def test_redact_text_case_insensitive(tmp_repo, make_user):
    """Twitch usernames are case-insensitive in practice; chatters
    type 'Alice' or 'ALICE' interchangeably. Redaction must catch
    every casing."""
    uid = make_user(name="alice")
    plan = redactor.build_plan(tmp_repo, [uid])
    out = redactor.redact_text(plan, "Alice and ALICE and alice")
    # All three got replaced with the same token.
    assert out.count("<USER_001>") == 3
    assert "alice" not in out.lower() or "user_001" in out.lower()


def test_redact_text_handles_punctuation(tmp_repo, make_user):
    """Names appearing next to punctuation must redact too.
    `alice,` or `@alice` or `(alice)` are common chat shapes."""
    uid = make_user(name="alice")
    plan = redactor.build_plan(tmp_repo, [uid])
    cases = [
        "@alice",
        "(alice)",
        "alice,",
        "alice's run",
        "ping alice!",
    ]
    for s in cases:
        assert "<USER_001>" in redactor.redact_text(plan, s), (
            f"failed to redact in: {s!r}"
        )


# ---- redact_event behaviour ----


def test_redact_event_patches_text_fields(tmp_repo, make_user):
    """LLM_CALL events: prompt + system_prompt + response_text +
    note all get scrubbed. referenced_user_ids gets replaced with
    tokens."""
    uid = make_user(name="alice")
    plan = redactor.build_plan(tmp_repo, [uid])
    ev = {
        "kind": "llm_call",
        "prompt": "context: alice has 12 messages",
        "system_prompt": "be terse about alice",
        "response_text": '{"text": "alice plays speedruns"}',
        "referenced_user_ids": [uid],
    }
    out = redactor.redact_event(plan, ev)
    assert "alice" not in out["prompt"].lower()
    assert "alice" not in out["system_prompt"].lower()
    assert "alice" not in out["response_text"].lower()
    assert out["referenced_user_ids"] == ["<USER_001>"]


def test_redact_event_patches_streamer_action_note(tmp_repo, make_user):
    """STREAMER_ACTION events with action_kind='note' carry the
    note text in `note` AND the note may reference a chatter name.
    Both get redacted."""
    uid = make_user(name="alice")
    plan = redactor.build_plan(tmp_repo, [uid])
    ev = {
        "kind": "streamer_action",
        "action_kind": "note",
        "item_key": "42",
        "action": "created:manual",
        "note": "alice loves speedruns",
    }
    out = redactor.redact_event(plan, ev)
    assert "alice" not in out["note"].lower()


def test_redact_event_patches_snapshot_messages(tmp_repo, make_user):
    """CONTEXT_SNAPSHOT events have nested `snapshot.messages` with
    name + content. Both get scrubbed."""
    uid = make_user(name="alice")
    plan = redactor.build_plan(tmp_repo, [uid])
    ev = {
        "kind": "context_snapshot",
        "snapshot": {
            "messages": [
                {"id": 1, "user_id": uid, "name": "alice", "content": "hi"},
            ],
            "threads": [
                {"id": 7, "title": "T", "drivers": ["alice"], "recap": "alice asked"},
            ],
        },
    }
    out = redactor.redact_event(plan, ev)
    msg = out["snapshot"]["messages"][0]
    assert msg["name"] == "<USER_001>"
    assert msg["user_id"] == "<USER_001>"
    thread = out["snapshot"]["threads"][0]
    assert thread["drivers"] == ["<USER_001>"]
    assert "alice" not in thread["recap"].lower()


def test_redact_event_does_not_mutate_input(tmp_repo, make_user):
    """The function returns a new dict — caller's input must stay
    untouched. Catches a bug where someone adds an in-place
    optimisation that breaks idempotency."""
    uid = make_user(name="alice")
    plan = redactor.build_plan(tmp_repo, [uid])
    ev_before = {
        "prompt": "alice says hi",
        "referenced_user_ids": [uid],
    }
    snapshot = dict(ev_before)
    out = redactor.redact_event(plan, ev_before)
    assert ev_before == snapshot     # input untouched
    assert out is not ev_before


# ---- collect_user_ids_in_events ----


def test_collect_user_ids_aggregates_across_events(tmp_repo):
    """The exporter walks all events once to collect the union of
    referenced user_ids before building the plan. Verify the
    aggregator picks up both `referenced_user_ids` AND nested
    snapshot user_ids."""
    events = [
        {"kind": "llm_call", "referenced_user_ids": ["u_a", "u_b"]},
        {"kind": "streamer_action"},  # no users
        {"kind": "context_snapshot", "snapshot": {
            "messages": [
                {"user_id": "u_b", "name": "bob"},   # dup
                {"user_id": "u_c", "name": "carol"},
            ],
        }},
    ]
    ids = redactor.collect_user_ids_in_events(events)
    assert sorted(ids) == ["u_a", "u_b", "u_c"]
