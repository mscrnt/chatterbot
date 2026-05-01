"""Tests for `LLMClientHandle` — the stable proxy that lets services
hold one reference while the inner LLM client is swapped during a
hot reload (provider switch / API-key rotation / model change)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from chatterbot.llm.providers import LLMClientHandle


class _FakeClient:
    """Minimal stand-in for an LLMProvider — just enough surface to
    exercise the proxy. Each instance gets a unique id so tests can
    tell whether the handle is forwarding to the right one."""

    _next_id = 0

    def __init__(self, settings):
        type(self)._next_id += 1
        self.id = type(self)._next_id
        self.settings = settings

    async def health_check(self) -> bool:
        return True

    def echo(self, x):
        return (self.id, x)


@pytest.fixture(autouse=True)
def _patch_make_llm_client(monkeypatch):
    """Redirect the factory to our fake so tests don't reach real
    HTTP / API surfaces. Each call returns a fresh _FakeClient with
    an incrementing id."""
    monkeypatch.setattr(
        "chatterbot.llm.providers.make_llm_client",
        lambda settings: _FakeClient(settings),
    )


def _settings(**kwargs):
    """Bare SimpleNamespace stand-in — the fake factory ignores it
    anyway. Pass kwargs through so tests can express "different
    settings instance"."""
    return SimpleNamespace(**kwargs)


def test_handle_forwards_attribute_access():
    """A handle's `.echo(x)` should land on the inner client's
    `.echo(x)` via __getattr__ — services don't see the proxy."""
    handle = LLMClientHandle(_settings(name="boot"))
    inner_id, payload = handle.echo("hi")
    assert payload == "hi"
    assert inner_id == handle.inner.id


def test_reconfigure_swaps_inner():
    """After reconfigure(), .echo() should be served by the new
    inner client (different id), not the old one. This is the whole
    point of the proxy — services that captured `handle` before the
    reload still see fresh state."""
    handle = LLMClientHandle(_settings(name="boot"))
    boot_id = handle.inner.id

    handle.reconfigure(_settings(name="reloaded"))
    new_id, _ = handle.echo("after-reload")

    assert new_id != boot_id, (
        "reconfigure should produce a fresh inner client"
    )
    assert handle.inner.id == new_id


def test_reconfigure_failure_keeps_old_client(monkeypatch):
    """If `make_llm_client` raises during reconfigure (e.g.
    misconfigured API key), the handle should keep the previous
    inner client rather than blow up. The streamer's bot keeps
    serving the old config until they fix the misconfig."""
    handle = LLMClientHandle(_settings(name="good"))
    boot_id = handle.inner.id

    # Make the next factory call raise.
    def _broken(settings):
        raise ValueError("bad provider config")
    monkeypatch.setattr(
        "chatterbot.llm.providers.make_llm_client", _broken,
    )

    handle.reconfigure(_settings(name="bad"))

    # Inner is still the boot-time client.
    assert handle.inner.id == boot_id


def test_settings_attribute_resolves_from_handle_first():
    """The handle's own attrs (_settings, _inner, reconfigure,
    inner) take precedence over the inner client's. Without this
    the handle's own state would shadow / get shadowed by the
    inner's confusingly. Verifies attribute lookup order."""
    handle = LLMClientHandle(_settings(marker="from-handle"))
    # _settings is on the handle. Even though inner has settings.
    assert handle._settings.marker == "from-handle"
    # The handle's `inner` property returns the inner directly.
    assert handle.inner is not handle


def test_inner_settings_is_distinct_from_handle_settings():
    """Sanity: handle._settings and the inner client's `.settings`
    are separate fields. Reconfigure swaps both consistently."""
    handle = LLMClientHandle(_settings(name="boot"))
    assert handle._settings.name == "boot"
    assert handle.inner.settings.name == "boot"

    handle.reconfigure(_settings(name="reloaded"))
    assert handle._settings.name == "reloaded"
    assert handle.inner.settings.name == "reloaded"
