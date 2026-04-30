"""CLI smoke tests — setup → enable → export → verify roundtrip.

Patches `getpass.getpass` so the prompts run unattended. We exercise
the actual command functions (not the argparse layer) because that's
where the bugs live; argparse is a thin dispatch layer.

These tests deliberately drive a freshly-initialised SQLite DB so a
regression in the migration path or app_settings persistence shows up
here too.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from chatterbot.dataset import capture, cipher, cli


# ---- helpers ----


class _FakeSettings:
    """Minimum surface the CLI commands read off `settings`. The real
    Settings has a hundred fields; this one has the two."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.ollama_embed_dim = 768


@pytest.fixture
def settings(tmp_path: Path) -> _FakeSettings:
    return _FakeSettings(str(tmp_path / "chatters.db"))


@pytest.fixture
def patched_passphrase(monkeypatch):
    """Stub the passphrase prompt to a constant. Tests can override
    by re-patching with a different value."""
    PASS = "test-passphrase"
    monkeypatch.setattr(cli, "_prompt_passphrase", lambda *a, **k: PASS)
    return PASS


# ---- cmd_setup ----


def test_setup_initialises_wrapped_dek(settings, patched_passphrase):
    """First-run setup writes a wrapped DEK + fingerprint to
    app_settings and leaves the toggle OFF — streamer must opt in
    explicitly via `dataset enable`."""
    rc = cli.cmd_setup(settings)
    assert rc == 0
    repo = cli._open_repo(settings)
    try:
        assert repo.get_app_setting("dataset_key_wrapped")
        assert len(repo.get_app_setting("dataset_key_fingerprint")) == 16
        assert repo.get_app_setting("dataset_capture_enabled") == "false"
    finally:
        repo.close()


def test_setup_refuses_to_clobber_existing_dek(settings, patched_passphrase):
    """Running setup twice on the same DB must NOT silently regenerate
    the DEK — that would lock the streamer out of any past events."""
    assert cli.cmd_setup(settings) == 0
    rc = cli.cmd_setup(settings)
    assert rc == 1  # refuse without --force


def test_setup_force_clobbers(settings, patched_passphrase):
    """--force overrides the safety check. The fingerprint must
    actually change (proving a new DEK was generated, not the same
    one re-wrapped)."""
    cli.cmd_setup(settings)
    repo = cli._open_repo(settings)
    fp1 = repo.get_app_setting("dataset_key_fingerprint")
    repo.close()

    assert cli.cmd_setup(settings, force=True) == 0
    repo = cli._open_repo(settings)
    fp2 = repo.get_app_setting("dataset_key_fingerprint")
    repo.close()
    assert fp1 != fp2


# ---- cmd_enable / cmd_disable ----


def test_enable_requires_setup_first(settings, capsys):
    """`dataset enable` before `dataset setup` must error with a
    helpful message — not silently turn on a toggle that points at
    no key."""
    rc = cli.cmd_enable(settings)
    assert rc == 1
    err = capsys.readouterr().err
    assert "setup" in err.lower()


def test_enable_then_disable_flips_toggle(settings, patched_passphrase):
    cli.cmd_setup(settings)
    assert cli.cmd_enable(settings) == 0
    repo = cli._open_repo(settings)
    assert repo.get_app_setting("dataset_capture_enabled") == "true"
    repo.close()

    assert cli.cmd_disable(settings) == 0
    repo = cli._open_repo(settings)
    assert repo.get_app_setting("dataset_capture_enabled") == "false"
    repo.close()


# ---- cmd_info ----


def test_info_does_not_decrypt(settings, patched_passphrase, capsys):
    """`info` reads only metadata. It must NEVER prompt for the
    passphrase. Verifying via stdout — a successful info run prints
    "wrapped DEK : present" without ever calling _prompt_passphrase."""
    cli.cmd_setup(settings)

    # Make sure info doesn't prompt — counter on the patch.
    calls = {"n": 0}

    def _spy(*a, **k):
        calls["n"] += 1
        return "should-never-be-called"
    cli._prompt_passphrase = _spy  # type: ignore[assignment]

    rc = cli.cmd_info(settings)
    assert rc == 0
    assert calls["n"] == 0
    out = capsys.readouterr().out
    assert "wrapped DEK     : present" in out


# ---- end-to-end: setup → enable → simulate capture → export → verify ----


def test_full_roundtrip(settings, patched_passphrase, tmp_path: Path):
    """The big integration sweep: every CLI surface in order, with a
    real (mocked-LLM-free) capture in the middle. If any production
    refactor breaks one link, this test fails loudly."""
    # 1) setup + enable
    assert cli.cmd_setup(settings) == 0
    assert cli.cmd_enable(settings) == 0

    # 2) simulate the bot's startup unlock by putting the DEK in
    #    process memory.
    repo = cli._open_repo(settings)
    wrapped = cipher.WrappedDEK.from_json(repo.get_app_setting("dataset_key_wrapped"))
    dek = cipher.unwrap_dek(wrapped, patched_passphrase)
    repo.set_dataset_dek(dek)

    # 3) Fire two LLM calls through the capture path.
    async def _drive():
        for i in range(2):
            await capture.record_llm_call(
                repo,
                call_site=f"test.event_{i}",
                model_id="qwen3.5",
                provider="ollama",
                system_prompt=None,
                prompt=f"prompt {i}",
                response_text=f'{{"i":{i}}}',
                response_schema_name="X",
                num_ctx=100,
                num_predict=100,
                think=False,
                latency_ms=1,
            )
            await asyncio.sleep(0)
    asyncio.run(_drive())
    assert repo.dataset_event_count() == 2
    repo.close()
    capture.close_writer()

    # 4) export
    out_bundle = tmp_path / "out.cbds"
    assert cli.cmd_export(settings, out_bundle) == 0
    assert out_bundle.exists()
    assert out_bundle.stat().st_size > 0

    # 5) verify
    assert cli.cmd_verify(out_bundle) == 0
