"""Browser-facing /dataset routes — setup, status, export.

Builds a fresh FastAPI app per test against a throwaway repo. We
don't mock at HTTP transport: TestClient hits the real ASGI app, the
real templates render, the real form-handling runs. The only thing
we substitute is `make_llm_client`, since live network LLMs aren't
relevant to dataset capture flows.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from chatterbot.config import Settings
from chatterbot.dataset import capture as cap_mod
from chatterbot.repo import ChatterRepo


# ---- per-test app + client ----


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch):
    """Construct a real FastAPI app pointed at a fresh sqlite DB.
    `make_llm_client` is monkeypatched to a stub so the app boots
    without a live Ollama. Yields (TestClient, repo) so tests can
    seed state on the same DB the app reads."""
    db_path = tmp_path / "chatters.db"
    repo = ChatterRepo(str(db_path), embed_dim=768)

    # Stub out the LLM client factory — the dataset routes never
    # call generate_structured. A SimpleNamespace with the
    # `attach_dataset_capture` method is enough.
    class _StubLLM:
        def __init__(self):
            self._dataset_repo = None

        def attach_dataset_capture(self, repo):
            self._dataset_repo = repo

        async def health_check(self):
            return True

        async def embed(self, text):
            return [0.0] * 768

    from chatterbot.llm import providers as _providers
    monkeypatch.setattr(_providers, "make_llm_client", lambda settings: _StubLLM())

    from chatterbot.web.app import create_app
    settings = Settings(_env_file=None)
    settings.db_path = str(db_path)
    app = create_app(repo, settings)
    client = TestClient(app)
    try:
        yield client, repo
    finally:
        client.close()
        repo.close()
        cap_mod.reset_writer()


# ---- GET /dataset status page ----


def test_dataset_page_unconfigured(app_client):
    """Default install: no DEK, no events. Page renders with the
    'Setup' card visible and 'no key yet' status."""
    client, _ = app_client
    r = client.get("/dataset")
    assert r.status_code == 200
    body = r.text
    # Status panel pins all three tri-state values.
    assert "Configured" in body
    assert "Capture enabled" in body
    assert "Unlocked in this process" in body
    # Setup form is shown when no key exists.
    assert 'action="/dataset/setup"' in body
    # Export form must NOT show when there's no key.
    assert 'action="/dataset/export"' not in body


def test_dataset_page_configured_no_events(app_client):
    """After setup but no captures yet — setup form gone, export form
    still hidden (no events to export)."""
    client, repo = app_client
    from chatterbot.dataset import cipher
    dek = cipher.generate_dek()
    wrapped = cipher.wrap_dek(dek, "test-pass")
    repo.set_app_setting("dataset_key_wrapped", wrapped.to_json())
    repo.set_app_setting(
        "dataset_key_fingerprint", cipher.fingerprint_dek(dek),
    )
    r = client.get("/dataset")
    assert r.status_code == 200
    assert 'action="/dataset/setup"' not in r.text
    # Still no export form — total_events == 0.
    assert 'action="/dataset/export"' not in r.text


def test_dataset_page_enabled_but_locked_warning(app_client):
    """Toggle on, no DEK loaded in process — page shows the warning
    that capture is dropping events."""
    client, repo = app_client
    from chatterbot.dataset import cipher
    dek = cipher.generate_dek()
    wrapped = cipher.wrap_dek(dek, "test-pass")
    repo.set_app_setting("dataset_key_wrapped", wrapped.to_json())
    repo.set_app_setting("dataset_capture_enabled", "true")
    # Don't call repo.set_dataset_dek — simulates the bot/dashboard
    # process not having the env var set.
    r = client.get("/dataset")
    assert r.status_code == 200
    assert "DEK isn't loaded in this process" in r.text


# ---- POST /dataset/setup ----


def test_setup_success_flashes_recovery(app_client):
    """Happy path: passphrase + matching confirm → wrapped DEK lands
    in app_settings, page redirects with the recovery string in the
    flash querystring."""
    client, repo = app_client
    r = client.post(
        "/dataset/setup",
        data={"passphrase": "secret123", "passphrase_confirm": "secret123"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    parsed = urlparse(r.headers["location"])
    qs = parse_qs(parsed.query)
    assert "recovery" in qs and qs["recovery"][0]
    assert "key generated" in qs.get("flash", [""])[0].lower()
    # Wrapped DEK is now in app_settings, capture toggle defaulted to false.
    assert repo.get_app_setting("dataset_key_wrapped")
    assert repo.get_app_setting("dataset_capture_enabled") == "false"
    # And the dashboard's own repo got the unlocked DEK in memory —
    # this dashboard process can capture without a restart.
    assert repo.dataset_dek() is not None


def test_setup_short_passphrase_rejected(app_client):
    client, repo = app_client
    r = client.post(
        "/dataset/setup",
        data={"passphrase": "short", "passphrase_confirm": "short"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    qs = parse_qs(urlparse(r.headers["location"]).query)
    assert qs.get("flash_kind") == ["error"]
    assert repo.get_app_setting("dataset_key_wrapped") is None


def test_setup_passphrase_mismatch_rejected(app_client):
    client, repo = app_client
    r = client.post(
        "/dataset/setup",
        data={"passphrase": "secret123", "passphrase_confirm": "different"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    qs = parse_qs(urlparse(r.headers["location"]).query)
    assert qs.get("flash_kind") == ["error"]
    assert repo.get_app_setting("dataset_key_wrapped") is None


def test_setup_refuses_to_clobber_existing(app_client):
    """Setup is intentionally not destructive — re-running it on a
    repo that already has a wrapped DEK refuses (would lock the
    streamer out of past events). The CLI's --force is the escape
    hatch; deliberately not exposed via the browser."""
    client, repo = app_client
    from chatterbot.dataset import cipher
    dek = cipher.generate_dek()
    wrapped = cipher.wrap_dek(dek, "first-pass")
    repo.set_app_setting("dataset_key_wrapped", wrapped.to_json())
    fingerprint_before = cipher.fingerprint_dek(dek)
    repo.set_app_setting("dataset_key_fingerprint", fingerprint_before)

    r = client.post(
        "/dataset/setup",
        data={"passphrase": "second-pass", "passphrase_confirm": "second-pass"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    qs = parse_qs(urlparse(r.headers["location"]).query)
    assert qs.get("flash_kind") == ["error"]
    # Fingerprint unchanged — we did NOT regenerate.
    assert repo.get_app_setting("dataset_key_fingerprint") == fingerprint_before


# ---- POST /dataset/export ----


def test_export_returns_cbds_bundle(app_client, monkeypatch):
    """End-to-end: setup → fire one capture event → export from the
    browser → response is a valid .cbds tarball."""
    client, repo = app_client
    # 1) Setup via the route so flow matches a real streamer.
    client.post(
        "/dataset/setup",
        data={"passphrase": "test-pass", "passphrase_confirm": "test-pass"},
        follow_redirects=False,
    )
    repo.set_app_setting("dataset_capture_enabled", "true")
    # The setup route already installed the DEK on the dashboard's
    # repo; capture should now write.
    assert repo.dataset_dek() is not None

    # 2) Fire one LLM_CALL synchronously (we own the event loop here).
    import asyncio
    asyncio.run(cap_mod.record_llm_call(
        repo, call_site="test.web_export",
        model_id="qwen3.5", provider="ollama",
        system_prompt=None, prompt="hi", response_text='{"x":1}',
        response_schema_name="X",
        num_ctx=None, num_predict=None,
        think=False, latency_ms=1,
    ))
    assert repo.dataset_event_count() == 1

    # 3) Export via the route.
    r = client.post(
        "/dataset/export",
        data={"passphrase": "test-pass"},
    )
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("application/octet-stream")
    cd = r.headers.get("content-disposition", "")
    assert "chatterbot-dataset-" in cd and ".cbds" in cd

    # 4) Parse the bundle: must be a valid tar containing the
    # expected entries from cmd_export.
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r") as tar:
        names = set(tar.getnames())
    assert {"manifest.json", "bundle_dek.wrapped",
            "payload.nonce", "payload.bin"}.issubset(names)


def test_export_wrong_passphrase_redirects_with_error(app_client):
    """Wrong passphrase → no leaking of bytes. Redirect with an
    error flash, status 303."""
    client, repo = app_client
    client.post(
        "/dataset/setup",
        data={"passphrase": "right-pass", "passphrase_confirm": "right-pass"},
        follow_redirects=False,
    )
    # No events yet — but the export pipeline checks the passphrase
    # before walking the index, so wrong passphrase fails with the
    # passphrase-error path, not the no-events-error path.
    r = client.post(
        "/dataset/export",
        data={"passphrase": "WRONG"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    qs = parse_qs(urlparse(r.headers["location"]).query)
    assert qs.get("flash_kind") == ["error"]


def test_export_without_setup_redirects_with_error(app_client):
    client, _ = app_client
    r = client.post(
        "/dataset/export",
        data={"passphrase": "anything"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    qs = parse_qs(urlparse(r.headers["location"]).query)
    assert qs.get("flash_kind") == ["error"]
    # Specifically the "run setup first" branch.
    flash = qs.get("flash", [""])[0].lower()
    assert "setup" in flash


# ---- nav visibility ----


def test_nav_link_hidden_by_default(app_client):
    """Default install has no DEK and toggle off — Dataset shouldn't
    appear in the nav. Keeps the UI clean for installs that don't
    use the feature."""
    client, _ = app_client
    r = client.get("/")
    assert r.status_code == 200
    # The /chatters link is always in nav (sanity check).
    assert 'href="/chatters"' in r.text
    # /dataset link must NOT be in nav.
    assert 'href="/dataset"' not in r.text


def test_nav_link_appears_after_setup(app_client):
    client, repo = app_client
    client.post(
        "/dataset/setup",
        data={"passphrase": "test-pass", "passphrase_confirm": "test-pass"},
        follow_redirects=False,
    )
    r = client.get("/")
    assert 'href="/dataset"' in r.text


def test_nav_link_appears_when_only_toggle_on(app_client):
    """Toggle on but no DEK yet — nav still shows so the streamer
    can navigate to /dataset and run setup."""
    client, repo = app_client
    repo.set_app_setting("dataset_capture_enabled", "true")
    r = client.get("/")
    assert 'href="/dataset"' in r.text
