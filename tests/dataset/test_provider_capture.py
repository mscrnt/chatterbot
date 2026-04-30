"""Provider parity — every LLM provider's `generate_structured` must
fire the dataset capture finally-block when a repo is attached.

We don't go through real HTTP: each provider exposes
`generate_structured` as "schema → call generate → validate". The
capture wrapper sits in the `try/except/finally` around that pipeline
and is what we want to verify. So every test here subclasses the
provider with a canned `generate()` that bypasses httpx entirely —
the finally-block still runs against the real provider class.

Parametrising over all three providers means a slice-3 refactor that
adds a fourth backend gets a one-line addition here, not a copy-paste.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from chatterbot.llm.ollama_client import OllamaClient
from chatterbot.llm.providers import AnthropicClient, OpenAIClient


# ---- canned response ----


class _Echo(BaseModel):
    """Trivial schema the test queues a response into. Each provider
    parses via `model_validate_json`, so the canned generate() output
    must serialise to a valid Echo."""
    text: str


_CANNED_JSON = '{"text":"stubbed"}'


# ---- per-provider stub subclasses ----


class _OllamaStub(OllamaClient):
    """OllamaClient that bypasses httpx — `generate()` returns a
    canned JSON string so the finally-block in `generate_structured`
    still runs against real provider code."""
    async def generate(self, *a, **kw):  # noqa: ARG002 — match parent signature loosely
        return _CANNED_JSON


class _AnthropicStub(AnthropicClient):
    async def generate(self, *a, **kw):  # noqa: ARG002
        return _CANNED_JSON


class _OpenAIStub(OpenAIClient):
    async def generate(self, *a, **kw):  # noqa: ARG002
        return _CANNED_JSON


# ---- factories ----


def _make_ollama() -> OllamaClient:
    return _OllamaStub(
        base_url="http://stub", model="qwen3.5",
        embed_model="nomic-embed",
    )


def _make_anthropic() -> AnthropicClient:
    return _AnthropicStub(
        api_key="stub-key",
        model="claude-opus-4-7",
        embed_via=_make_ollama(),
    )


def _make_openai() -> OpenAIClient:
    return _OpenAIStub(
        api_key="stub-key",
        model="gpt-4o",
        embed_via=_make_ollama(),
    )


# ---- the parametrized test ----


@pytest.mark.parametrize(
    "factory, expected_provider",
    [
        (_make_ollama, "ollama"),
        (_make_anthropic, "anthropic"),
        (_make_openai, "openai"),
    ],
    ids=["ollama", "anthropic", "openai"],
)
async def test_provider_attaches_capture_and_records_call_site(
    unlocked_repo, factory, expected_provider,
):
    """The whole point of slice 2: every provider goes through the
    same capture pipeline. Attach an unlocked_repo, run a structured
    call, assert one event lands with the right `provider` and the
    same `call_site` we passed in. Coverage gap here means a
    streamer who flips llm_provider= to openai loses dataset
    capture silently."""
    client = factory()
    client.attach_dataset_capture(unlocked_repo)

    result = await client.generate_structured(
        prompt="hi",
        response_model=_Echo,
        call_site="test.provider_parity",
    )
    assert result.text == "stubbed"

    # The capture write hops through asyncio.to_thread — give the
    # worker a tick to land before we walk the index.
    await asyncio.sleep(0)

    rows = list(unlocked_repo.iter_dataset_events())
    assert len(rows) == 1, (
        f"{expected_provider}: expected 1 event captured, got {len(rows)}"
    )

    # Decrypt + verify the captured fields. We don't go through
    # `decrypt_event` directly — the test_capture.py module already
    # exercises that round trip; here we just want to confirm the
    # finally-block fired with the right provider/call_site.
    from pathlib import Path
    from chatterbot.dataset.capture import decrypt_event
    from chatterbot.dataset.storage import read_record

    data_root = Path(unlocked_repo.db_path).parent
    rec = read_record(
        data_root / rows[0]["shard_path"],
        rows[0]["byte_offset"],
        rows[0]["byte_length"],
    )
    payload = decrypt_event(
        unlocked_repo.dataset_dek(), rows[0]["ts"], rec.nonce, rec.ciphertext,
    )
    assert payload["provider"] == expected_provider
    assert payload["call_site"] == "test.provider_parity"
    assert payload["response_text"] == _CANNED_JSON


@pytest.mark.parametrize(
    "factory",
    [_make_ollama, _make_anthropic, _make_openai],
    ids=["ollama", "anthropic", "openai"],
)
async def test_provider_without_attached_repo_drops_silently(factory):
    """Default state — no repo attached. The finally-block still
    runs but `record_llm_call_safe` no-ops on `repo is None`. Verify
    the call returns the parsed response and nothing crashes."""
    client = factory()
    # No attach_dataset_capture call — capture stays unwired.
    result = await client.generate_structured(
        prompt="hi",
        response_model=_Echo,
        call_site="test.no_capture",
    )
    assert result.text == "stubbed"


@pytest.mark.parametrize(
    "factory, expected_provider",
    [
        (_make_ollama, "ollama"),
        (_make_anthropic, "anthropic"),
        (_make_openai, "openai"),
    ],
    ids=["ollama", "anthropic", "openai"],
)
async def test_provider_records_error_on_failure(
    unlocked_repo, factory, expected_provider,
):
    """When the underlying call raises, the capture row must still be
    written with `error` populated. Negative signal is useful for
    prompt iteration — failed validations / timeouts shouldn't be
    invisible in the dataset."""
    client = factory()
    client.attach_dataset_capture(unlocked_repo)

    # Override generate() to raise. Keep the stub-canned response
    # path elsewhere intact.
    async def _boom(*a, **kw):  # noqa: ARG001
        raise RuntimeError("simulated provider outage")
    client.generate = _boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        await client.generate_structured(
            prompt="hi",
            response_model=_Echo,
            call_site="test.error_capture",
        )

    await asyncio.sleep(0)
    rows = list(unlocked_repo.iter_dataset_events())
    assert len(rows) == 1, (
        f"{expected_provider}: error event was not captured"
    )

    from pathlib import Path
    from chatterbot.dataset.capture import decrypt_event
    from chatterbot.dataset.storage import read_record
    data_root = Path(unlocked_repo.db_path).parent
    rec = read_record(
        data_root / rows[0]["shard_path"],
        rows[0]["byte_offset"],
        rows[0]["byte_length"],
    )
    payload = decrypt_event(
        unlocked_repo.dataset_dek(), rows[0]["ts"], rec.nonce, rec.ciphertext,
    )
    assert payload["provider"] == expected_provider
    assert "RuntimeError" in (payload["error"] or "")
    # response_text stays empty when the call fails before validate.
    assert payload["response_text"] == ""
