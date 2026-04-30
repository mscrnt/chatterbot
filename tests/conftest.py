"""Shared pytest fixtures.

Anything that more than one test file needs lives here. The fixtures
are intentionally narrow: each one builds the minimum state the test
needs and tears it down on exit. Adding a new test category typically
means adding 1-2 fixtures here and N tests in the leaf file.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pytest

from chatterbot.repo import ChatterRepo

from ._mock_llm import MockLLMClient


# ---- repo fixtures ----


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Iterator[ChatterRepo]:
    """Build a ChatterRepo against a fresh, throwaway SQLite DB. The
    full schema is initialised — all migrations run. Closed on
    teardown so file handles don't leak."""
    db_path = tmp_path / "chatters.db"
    repo = ChatterRepo(str(db_path), embed_dim=768)
    try:
        yield repo
    finally:
        repo.close()
        # Reset the dataset-capture singleton between tests so each
        # test gets a clean shard writer rooted at its own data_root.
        from chatterbot.dataset import capture as _cap
        _cap.reset_writer()


# ---- mock LLM ----


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """Fresh MockLLMClient per test. Tests queue canned responses then
    drive production code through it."""
    return MockLLMClient()


# ---- dataset-capture fixtures ----


@pytest.fixture
def unlocked_repo(tmp_repo: ChatterRepo) -> ChatterRepo:
    """A repo with a fresh DEK loaded into memory and capture enabled.
    Setup mirrors what `chatterbot dataset setup` + `enable` + the
    bot startup unlock all do, collapsed into one fixture."""
    from chatterbot.dataset import cipher
    dek = cipher.generate_dek()
    wrapped = cipher.wrap_dek(dek, "test-passphrase")
    tmp_repo.set_app_setting("dataset_key_wrapped", wrapped.to_json())
    tmp_repo.set_app_setting(
        "dataset_key_fingerprint", cipher.fingerprint_dek(dek),
    )
    tmp_repo.set_app_setting("dataset_capture_enabled", "true")
    tmp_repo.set_dataset_dek(dek)
    return tmp_repo


# ---- factories for repo rows ----


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@pytest.fixture
def make_user(tmp_repo: ChatterRepo):
    """Insert a User row with sensible defaults, return its twitch_id.
    Defaults can be overridden per call: `make_user(name="alice")`."""

    def _factory(
        *,
        twitch_id: str | None = None,
        name: str = "alice",
        opt_out: bool = False,
        source: str = "twitch",
    ) -> str:
        # Use the public upsert so any future schema additions land
        # via a stable code path rather than raw INSERT. The repo
        # method stamps its own timestamps internally — tests just
        # supply id/name/source.
        uid = twitch_id or f"u_{name}_{os.urandom(2).hex()}"
        tmp_repo.upsert_user(twitch_id=uid, name=name, source=source)
        if opt_out:
            tmp_repo.set_opt_out(uid, True)
        return uid

    return _factory


@pytest.fixture
def make_message(tmp_repo: ChatterRepo, make_user):
    """Insert a Message row via the public `insert_message` path.

    Production `insert_message` stamps `ts` with `_now_iso()` and has no
    test hook to override — so when a test needs a specific timestamp
    (e.g. "this message is 20 minutes old"), we update the row in
    place after the insert. That keeps production code free of
    test-only parameters."""

    def _factory(
        *,
        user_id: str | None = None,
        name: str = "alice",
        content: str = "hello",
        ts: str | None = None,
        reply_parent_login: str | None = None,
        is_emote_only: bool = False,
    ) -> int:
        # Ensure a user row exists for the FK on messages. When a
        # caller passes their own user_id we still upsert against
        # that id (idempotent) so a test can repeatedly insert as
        # the same chatter without pre-creating the row.
        if user_id is None:
            uid = make_user(name=name)
        else:
            uid = make_user(twitch_id=user_id, name=name)
        msg_id = tmp_repo.insert_message(
            user_id=uid,
            content=content,
            reply_parent_login=reply_parent_login,
            is_emote_only=is_emote_only,
        )
        if ts:
            with tmp_repo._cursor() as cur:  # noqa: SLF001 — test-only
                cur.execute(
                    "UPDATE messages SET ts = ? WHERE id = ?",
                    (ts, msg_id),
                )
        return msg_id

    return _factory


# ---- async test helper ----


@pytest.fixture
def event_loop_policy():
    # Pinning the policy keeps test isolation tighter on Windows
    # runners (default policy on Win Python 3.12 is ProactorEventLoop
    # which produces noisy warnings on close). Linux already uses the
    # right one — this is a no-op there.
    return asyncio.DefaultEventLoopPolicy()
