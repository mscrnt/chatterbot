"""Diagnostic bundle (.cbreport) — pin the file list, the safety
contract, and the dataset / insight / app_settings additions.

The bundle ends up attached to GitHub issues, so its privacy
contract matters: NO secrets, NO message content, NO chatter
identities by default. These tests check the contract holds and
that the new slices we added don't leak anything they shouldn't.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from chatterbot.config import Settings
from chatterbot.diagnose import (
    _APP_SETTINGS_SECRET_PATTERNS,
    _is_secret_key,
    _redact_secret_value,
    build_diagnostic_bundle,
)


# ---- helpers ----


@pytest.fixture
def settings_for_repo(tmp_repo) -> Settings:
    """Settings shaped for the diagnose helpers — they need
    db_path + ollama_embed_dim only."""
    s = Settings(_env_file=None)
    s.db_path = tmp_repo.db_path
    return s


def _open_bundle(path: Path) -> dict[str, bytes]:
    """Return a {filename: bytes} mapping for every entry in the
    bundle. Cheaper than calling readzip for every member."""
    out: dict[str, bytes] = {}
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            out[name] = zf.read(name)
    return out


# ---- file presence ----


def test_bundle_contains_expected_files(tmp_path, tmp_repo, settings_for_repo):
    """Every default file must be present. Catches a refactor that
    accidentally drops one of the new bundle slices (dataset.json /
    insight_states.json / app_settings.json) without test coverage."""
    out = tmp_path / "x.cbreport"
    build_diagnostic_bundle(out, settings_for_repo)
    files = _open_bundle(out)
    expected = {
        "meta.json", "system.txt", "packages.txt", "env.txt",
        "db_stats.json", "ollama.json",
        "app_settings.json", "insight_states.json", "dataset.json",
        "README.txt",
    }
    assert expected.issubset(files.keys()), (
        f"missing: {expected - files.keys()}"
    )
    # README mentions every new slice so the maintainer reading
    # the bundle knows what to look at.
    readme = files["README.txt"].decode("utf-8")
    assert "app_settings.json" in readme
    assert "insight_states.json" in readme
    assert "dataset.json" in readme


def test_bundle_recent_activity_only_when_opted_in(tmp_path, tmp_repo, settings_for_repo):
    """Recent activity slice is opt-in. Absent by default,
    present when `with_recent_activity=True`."""
    minimal = tmp_path / "minimal.cbreport"
    build_diagnostic_bundle(minimal, settings_for_repo)
    assert "recent_activity.json" not in _open_bundle(minimal)

    full = tmp_path / "full.cbreport"
    build_diagnostic_bundle(full, settings_for_repo, with_recent_activity=True)
    assert "recent_activity.json" in _open_bundle(full)


# ---- F: db_stats includes dataset_events ----


def test_db_stats_includes_dataset_events_count(tmp_path, tmp_repo, settings_for_repo):
    """`row_counts` must include dataset_events (added in slice 6+)
    so the maintainer can see at a glance whether capture has been
    recording without decrypting anything."""
    # Seed one row so the count is non-zero and we can tell "table
    # exists + counts work" from "table missing".
    tmp_repo.insert_dataset_event(
        ts="2026-04-30T00:00:00Z", event_kind="llm_call",
        shard_path="x", byte_offset=0, byte_length=1, schema_version=1,
    )
    out = tmp_path / "x.cbreport"
    build_diagnostic_bundle(out, settings_for_repo)
    db_stats = json.loads(_open_bundle(out)["db_stats.json"])
    assert "dataset_events" in db_stats["row_counts"]
    assert db_stats["row_counts"]["dataset_events"] == 1
    # insight_states + history also surfaced.
    assert "insight_states" in db_stats["row_counts"]
    assert "insight_state_history" in db_stats["row_counts"]


# ---- A: dataset.json content ----


def test_dataset_json_off_default(tmp_path, tmp_repo, settings_for_repo):
    """Fresh repo: capture not configured, not enabled, no events.
    dataset.json must reflect this verbatim — used as the maintainer's
    first-look "is capture even on?" check."""
    out = tmp_path / "x.cbreport"
    build_diagnostic_bundle(out, settings_for_repo)
    payload = json.loads(_open_bundle(out)["dataset.json"])
    assert payload["enabled"] is False
    assert payload["configured"] is False
    assert payload["fingerprint"] == ""
    assert payload["total_events"] == 0


def test_dataset_json_enabled_with_events(tmp_path, tmp_repo, settings_for_repo):
    """Capture configured + toggle on + some events captured. The
    bundle must surface the counts so a maintainer reading the
    .cbreport can answer "is the user's capture working?" instantly."""
    tmp_repo.set_app_setting("dataset_capture_enabled", "true")
    tmp_repo.set_app_setting("dataset_key_wrapped", "{}")
    tmp_repo.set_app_setting("dataset_key_fingerprint", "abc123")
    for kind in ("llm_call", "llm_call", "streamer_action"):
        tmp_repo.insert_dataset_event(
            ts="2026-04-30T00:00:00Z", event_kind=kind,
            shard_path="x", byte_offset=0, byte_length=128, schema_version=1,
        )
    out = tmp_path / "x.cbreport"
    build_diagnostic_bundle(out, settings_for_repo)
    payload = json.loads(_open_bundle(out)["dataset.json"])
    assert payload["enabled"] is True
    assert payload["configured"] is True
    assert payload["fingerprint"] == "abc123"
    assert payload["total_events"] == 3
    by_kind = {e["kind"]: e["count"] for e in payload["events_by_kind"]}
    assert by_kind == {"llm_call": 2, "streamer_action": 1}
    # Total bytes preserved.
    assert payload["total_encrypted_bytes"] == 3 * 128


def test_dataset_json_never_decrypts(tmp_path, tmp_repo, settings_for_repo):
    """Pinning the privacy contract: even if capture has events
    written, the .cbreport must NOT contain decrypted payloads.
    Walks the bundle looking for the well-known capture fields that
    only appear inside decrypted events."""
    tmp_repo.set_app_setting("dataset_capture_enabled", "true")
    tmp_repo.insert_dataset_event(
        ts="2026-04-30T00:00:00Z", event_kind="llm_call",
        shard_path="x", byte_offset=0, byte_length=1, schema_version=1,
    )
    out = tmp_path / "x.cbreport"
    build_diagnostic_bundle(out, settings_for_repo)
    blob = b"".join(_open_bundle(out).values()).lower()
    # These keys come from inside the encrypted payload — if any
    # show up the bundle accidentally exposed decrypted content.
    assert b"system_prompt" not in blob
    assert b"response_text" not in blob
    assert b"call_site" not in blob


# ---- E: insight_states summary ----


def test_insight_states_summary_counts_only(tmp_path, tmp_repo, settings_for_repo):
    """Insight-state slice exposes COUNTS — never the `note` field
    or item_key content."""
    tmp_repo.set_insight_state(
        kind="talking_point", item_key="42", state="addressed",
        note="this should never appear",
    )
    tmp_repo.set_insight_state(
        kind="talking_point", item_key="43", state="skipped",
    )
    out = tmp_path / "x.cbreport"
    build_diagnostic_bundle(out, settings_for_repo)
    payload = json.loads(_open_bundle(out)["insight_states.json"])
    counts = {(r["kind"], r["state"]): r["count"]
              for r in payload["live_by_kind_state"]}
    assert counts == {
        ("talking_point", "addressed"): 1,
        ("talking_point", "skipped"): 1,
    }
    # The note text MUST NOT appear anywhere in the bundle.
    blob = b"".join(_open_bundle(out).values())
    assert b"this should never appear" not in blob


# ---- D: app_settings.json filtering ----


def test_app_settings_redacts_secret_keys(tmp_path, tmp_repo, settings_for_repo):
    """Any key matching the secret pattern list gets length-only
    in the bundle. Pinning the redaction list so a regression that
    accidentally drops `_token` from the patterns shows up here."""
    tmp_repo.set_app_setting("twitch_oauth_token", "supersecretvalue")
    tmp_repo.set_app_setting("dataset_key_wrapped", "{...}")
    tmp_repo.set_app_setting("internal_notify_secret", "shhh")
    tmp_repo.set_app_setting("chat_lag_seconds", "6")  # not secret

    out = tmp_path / "x.cbreport"
    build_diagnostic_bundle(out, settings_for_repo)
    payload = json.loads(_open_bundle(out)["app_settings.json"])
    values = payload["values"]

    assert values["twitch_oauth_token"].startswith("<redacted:")
    assert values["dataset_key_wrapped"].startswith("<redacted:")
    assert values["internal_notify_secret"].startswith("<redacted:")
    # Non-secret value preserved verbatim.
    assert values["chat_lag_seconds"] == "6"

    # Verify the actual secret value never lands in the bundle text.
    blob = b"".join(_open_bundle(out).values())
    assert b"supersecretvalue" not in blob


def test_secret_key_classifier_covers_known_shapes():
    """Direct unit test of `_is_secret_key` so the patterns list
    has its own coverage even when no app_setting uses them."""
    assert _is_secret_key("twitch_oauth_token")
    assert _is_secret_key("openai_api_key")
    assert _is_secret_key("dataset_passphrase_hint")
    assert _is_secret_key("DATASET_KEY_WRAPPED")  # case-insensitive
    assert not _is_secret_key("chat_lag_seconds")
    assert not _is_secret_key("engaging_subjects_blocklist")


def test_redact_secret_value_is_length_only():
    out = _redact_secret_value("hunter2-supersecret")
    assert "supersecret" not in out
    assert "19" in out  # length leaks shape, not bytes


def test_app_settings_long_value_truncated(tmp_path, tmp_repo, settings_for_repo):
    """Streamer-facts and similar long blobs get truncated so the
    bundle stays small. Truncation marker must include the original
    length so it's clear data was elided."""
    tmp_repo.set_app_setting("engaging_subjects_blocklist", "x" * 5000)
    out = tmp_path / "x.cbreport"
    build_diagnostic_bundle(out, settings_for_repo)
    payload = json.loads(_open_bundle(out)["app_settings.json"])
    val = payload["values"]["engaging_subjects_blocklist"]
    assert len(val) < 700  # 500 chars + truncation marker
    assert "more chars" in val


# ---- B: anonymise option ----


def test_anonymize_recent_activity_replaces_names(
    tmp_path, tmp_repo, settings_for_repo, make_user, make_message,
):
    """With anonymise on, the recent_activity.json must NOT contain
    real chatter names — just stable <USER_NNN> tokens."""
    # Stable user_ids so two messages from alice GROUP BY into one
    # row (the per-user counts are keyed on twitch_id, not name).
    alice = make_user(name="alice")
    bob = make_user(name="bob")
    make_message(user_id=alice, name="alice", content="msg one")
    make_message(user_id=alice, name="alice", content="msg two")
    make_message(user_id=bob, name="bob", content="msg three")

    out = tmp_path / "x.cbreport"
    build_diagnostic_bundle(
        out, settings_for_repo,
        with_recent_activity=True,
        anonymize_recent_activity=True,
    )
    payload = json.loads(_open_bundle(out)["recent_activity.json"])
    names_in_output = {row["name"] for row in payload["per_user_message_counts"]}
    # Real names gone.
    assert "alice" not in names_in_output
    assert "bob" not in names_in_output
    # Tokens present.
    assert all(n.startswith("<USER_") for n in names_in_output)
    # Counts preserved (the SHAPE of activity).
    counts = sorted(row["msg_count"] for row in payload["per_user_message_counts"])
    assert counts == [1, 2]
    # Anonymised flag is asserted.
    assert payload.get("anonymized") is True


def test_anonymize_off_keeps_names(
    tmp_path, tmp_repo, settings_for_repo, make_message,
):
    """Default behaviour (anonymise off, with-recent-activity on):
    real chatter names appear. Pinning so a refactor that flips
    the default to "always anonymise" doesn't surprise streamers
    expecting verbatim data for their own debugging."""
    make_message(name="alice", content="msg one")
    out = tmp_path / "x.cbreport"
    build_diagnostic_bundle(
        out, settings_for_repo,
        with_recent_activity=True,
        anonymize_recent_activity=False,
    )
    payload = json.loads(_open_bundle(out)["recent_activity.json"])
    names = {row["name"] for row in payload["per_user_message_counts"]}
    assert "alice" in names
