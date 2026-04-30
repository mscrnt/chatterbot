"""Browser-facing routes for the streamer-customizable prompts.

Slice-9 added:
  - GET  /settings              renders Prompts tab inline
  - POST /settings/prompts/<call_site>          save mode + payload
  - POST /settings/prompts/<call_site>/revert   wipe overrides

These tests use TestClient against the real ASGI app and assert
both the HTMX-friendly response shape (returns the re-rendered
card partial) and the persisted app_settings state.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chatterbot.config import Settings
from chatterbot.llm import prompts as P
from chatterbot.repo import ChatterRepo


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "chatters.db"
    repo = ChatterRepo(str(db_path), embed_dim=768)

    class _StubLLM:
        model = "stub"
        embed_model = "stub-embed"
        def attach_dataset_capture(self, repo): pass
        async def health_check(self): return True
        async def embed(self, t): return [0.0] * 768

    from chatterbot.llm import providers as _providers
    monkeypatch.setattr(
        _providers, "make_llm_client", lambda settings: _StubLLM(),
    )
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


# ---- /settings renders the Prompts tab inline ----


def test_settings_page_includes_prompts_tab(app_client):
    """The /settings page server-side-renders the Prompts panel as
    one of its tabs. Verify the tab strip + at least one card show
    up so a regression that drops the prompts wiring fails here."""
    client, _ = app_client
    r = client.get("/settings")
    assert r.status_code == 200
    body = r.text
    # Prompts tab button is present (matches the Alpine click expr).
    assert "tab = 'prompts'" in body
    # A specific card from each section to prove all 7 render.
    assert "Per-chatter talking points" in body
    assert "Engaging subjects extraction" in body
    assert "Per-subject talking points" in body
    assert "Open questions filter" in body
    assert "Thread recaps" in body
    assert "Channel topic snapshots" in body
    assert "Streamer-voice group summaries" in body


def test_settings_page_card_renders_factory_mode_by_default(app_client):
    """Fresh repo: every card defaults to factory mode. Verify by
    looking at one specific card's HTML for the radio's checked
    state."""
    client, _ = app_client
    r = client.get("/settings")
    body = r.text
    # The factory-mode preview is rendered under a <details> in
    # the card; pin that the factory text is present and the
    # mode picker initialises with `mode = 'factory'`.
    assert "mode: 'factory'" in body
    assert "Show factory prompt" in body


# ---- POST /settings/prompts/<site> ----


def test_save_factory_mode_writes_setting(app_client):
    client, repo = app_client
    r = client.post(
        "/settings/prompts/insights.talking_points",
        data={"mode": "factory"},
    )
    assert r.status_code == 200
    # Card re-rendered with success flash.
    assert "Reverted to the factory prompt" in r.text or "Saved" in r.text
    assert repo.get_app_setting("prompts.insights.talking_points.mode") == "factory"


def test_save_adlibs_persists_form_values_as_json(app_client):
    """Adlib slots are submitted as `adlib__<slot>` form fields.
    The route packs them into a JSON dict and stores under
    `prompts.<site>.adlibs`."""
    client, repo = app_client
    r = client.post(
        "/settings/prompts/insights.talking_points",
        data={
            "mode": "adlibs",
            "adlib__tone": "spicy and unfiltered",
            "adlib__avoid": "office stuff",
        },
    )
    assert r.status_code == 200
    saved = P.get_adlib_values("insights.talking_points", repo)
    assert saved == {"tone": "spicy and unfiltered", "avoid": "office stuff"}
    # Resolved prompt now contains the streamer's values.
    rendered = P.resolve_prompt("insights.talking_points", repo)
    assert "spicy and unfiltered" in rendered


def test_save_custom_persists_textarea_contents(app_client):
    client, repo = app_client
    r = client.post(
        "/settings/prompts/insights.talking_points",
        data={
            "mode": "custom",
            "custom": "Be brief and helpful. No hallucinations.",
        },
    )
    assert r.status_code == 200
    assert P.get_custom_text("insights.talking_points", repo) == (
        "Be brief and helpful. No hallucinations."
    )
    # resolve uses the saved custom text in custom mode.
    assert P.resolve_prompt("insights.talking_points", repo) == (
        "Be brief and helpful. No hallucinations."
    )


def test_save_invalid_mode_returns_card_with_error_flash(app_client):
    client, repo = app_client
    r = client.post(
        "/settings/prompts/insights.talking_points",
        data={"mode": "garbage"},
    )
    assert r.status_code == 200
    # Card re-renders with an error flash.
    assert "Invalid mode" in r.text
    # Setting was NOT persisted.
    assert repo.get_app_setting(
        "prompts.insights.talking_points.mode",
    ) is None


def test_save_unknown_call_site_returns_404(app_client):
    client, _ = app_client
    r = client.post(
        "/settings/prompts/not.a.real.site",
        data={"mode": "factory"},
    )
    assert r.status_code == 404


def test_save_keeps_unrelated_payload_for_other_modes(app_client):
    """Switching from adlibs → custom shouldn't wipe the saved
    adlib values. The streamer might want to switch back. Pin
    that we persist all submitted payloads, regardless of mode."""
    client, repo = app_client
    # Save adlibs with values.
    client.post(
        "/settings/prompts/insights.talking_points",
        data={
            "mode": "adlibs",
            "adlib__tone": "low-key",
            "adlib__avoid": "(none)",
        },
    )
    # Switch to custom WITHOUT submitting adlib fields again.
    # Note: form fields not in the post body just won't be in the
    # form dict; the route persists what's there. So switching
    # modes without re-typing adlib fields would NOT wipe them
    # (we only persist when the field is present).
    client.post(
        "/settings/prompts/insights.talking_points",
        data={"mode": "custom", "custom": "my custom prompt"},
    )
    # Adlib values still around for when the streamer flips back.
    saved = P.get_adlib_values("insights.talking_points", repo)
    assert saved.get("tone") == "low-key"


# ---- POST /settings/prompts/<site>/revert ----


def test_revert_clears_all_overrides_and_returns_card(app_client):
    client, repo = app_client
    # Prime with a custom-mode override.
    client.post(
        "/settings/prompts/insights.talking_points",
        data={"mode": "custom", "custom": "anything"},
    )
    r = client.post(
        "/settings/prompts/insights.talking_points/revert",
    )
    assert r.status_code == 200
    # Card re-renders with success flash.
    assert "Reverted to factory" in r.text
    # Every override key is gone.
    for key in (
        "prompts.insights.talking_points.mode",
        "prompts.insights.talking_points.adlibs",
        "prompts.insights.talking_points.custom",
    ):
        assert repo.get_app_setting(key) is None


def test_revert_unknown_site_returns_404(app_client):
    client, _ = app_client
    r = client.post("/settings/prompts/not.a.real.site/revert")
    assert r.status_code == 404


# ---- HTMX target shape: returned HTML is the card, not full page ----


def test_save_response_is_card_partial(app_client):
    """HTMX swap target is the card div; the response body must be
    the card markup ALONE, not the whole settings page. Verify by
    checking the response is short + lacks the page chrome."""
    client, _ = app_client
    r = client.post(
        "/settings/prompts/insights.talking_points",
        data={"mode": "factory"},
    )
    body = r.text
    # The card root id is present.
    assert "prompt-card-insights-talking_points" in body
    # No top-of-page elements should appear (those would mean the
    # full settings.html re-rendered).
    assert "<title>" not in body
    assert "<header" not in body or "tabindex" not in body  # no nav
