"""MockLLMClient — a deterministic stand-in for the live OllamaClient
/ AnthropicClient / OpenAIClient surface.

Implements the same `LLMProvider` Protocol the real clients do (see
`chatterbot/llm/providers.py`). Tests register canned responses keyed
by `(call_site, response_model)`; when production code makes a
`generate_structured(...)` call, the mock pops the matching expectation
and returns the canned `BaseModel` instance.

Design choices:

  - **Protocol-level mocking**, not HTTP-level. Our concern is "did the
    cache shape change", "did the prompt assembly miss a field", "did
    capture get wired up". HTTP-transport bugs (timeouts, malformed
    JSON) are rare and need a separate live-Ollama integration test.

  - **Strict matching by call_site.** Tests fail loudly if production
    code changes the `call_site` string the test was expecting — that
    keeps "rename the call site, forget to update the test" from
    silently passing.

  - **Records every call**, so a test can assert "exactly one
    summarizer.note_extraction call fired with prompt containing X."

  - **Replay-shaped API.** Captured `.cbds` events from the dataset
    capture system can drop straight in as fixtures later (`record_*`
    methods take the same shape that's persisted on disk).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


@dataclass
class RecordedCall:
    """One captured generate_structured invocation. Tests assert against
    these to verify that production code wired the right prompt /
    system_prompt / model into the LLM call."""
    call_site: str
    response_model_name: str
    prompt: str
    system_prompt: str | None
    num_ctx: int | None
    num_predict: int | None
    think: bool
    images: list[str] | None


@dataclass
class _Expectation:
    """One queued canned response. Match by `call_site` AND
    `response_model_name`; both must align before we serve. Anything
    not provided defaults to a permissive match."""
    call_site: str
    response_model_name: str
    response: BaseModel


class MockLLMClient:
    """Drop-in replacement for any `LLMProvider` in tests.

    Usage:

        mock = MockLLMClient(model_id="qwen3.5-mock")
        mock.queue_response(
            call_site="insights.open_questions",
            response=OpenQuestionsResponse(questions=[...]),
        )
        # ... run the production code that calls generate_structured ...
        assert len(mock.calls) == 1
        assert mock.calls[0].call_site == "insights.open_questions"

    `queue_response` is FIFO per (call_site, model). Tests that need
    multiple calls in a fixed order queue them in that order.
    """

    def __init__(self, *, model_id: str = "mock-model"):
        self.model = model_id
        self.embed_model = "mock-embed"
        self.calls: list[RecordedCall] = []
        self._queue: list[_Expectation] = []
        # Static embedding payload — tests that exercise the embed path
        # only care that *something* came back; semantics aren't tested
        # without a real embedding model.
        self._embed_dim = 768
        # Optional dataset-capture handle, mirrors OllamaClient's surface
        # so the `attach_dataset_capture` wiring can be exercised.
        self._dataset_repo = None

    # ---- test-side controls ----

    def queue_response(
        self,
        *,
        call_site: str,
        response: BaseModel,
    ) -> None:
        """Add one canned response to the FIFO queue. Pops on the next
        matching `generate_structured` call."""
        self._queue.append(_Expectation(
            call_site=call_site,
            response_model_name=type(response).__name__,
            response=response,
        ))

    def reset(self) -> None:
        """Drop every recorded call + every queued expectation. Useful
        for sharing one fixture across multiple sub-tests within a
        single test function."""
        self.calls.clear()
        self._queue.clear()

    @property
    def pending(self) -> int:
        """How many queued responses remain unconsumed. Tests can assert
        `mock.pending == 0` to verify production code consumed exactly
        what was queued."""
        return len(self._queue)

    # ---- production-side surface (LLMProvider Protocol) ----

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        format_schema: dict[str, Any] | None = None,
        model_override: str | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        images: list[str] | None = None,
        think: bool = False,
    ) -> str:
        # We keep `generate` defined for Protocol conformance, but
        # production paths in this codebase always go through
        # `generate_structured`. If a test ever needs the raw path we
        # can extend; for now, raise so we're alerted.
        raise NotImplementedError(
            "MockLLMClient.generate is not wired — tests should drive "
            "production code through generate_structured(...)."
        )

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        model_override: str | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        images: list[str] | None = None,
        think: bool = False,
        call_site: str = "unknown",
        referenced_user_ids: list[str] | None = None,
    ) -> T:
        rec = RecordedCall(
            call_site=call_site,
            response_model_name=response_model.__name__,
            prompt=prompt,
            system_prompt=system_prompt,
            num_ctx=num_ctx,
            num_predict=num_predict,
            think=think,
            images=list(images) if images else None,
        )
        self.calls.append(rec)

        match_idx = next(
            (
                i for i, exp in enumerate(self._queue)
                if exp.call_site == call_site
                and exp.response_model_name == response_model.__name__
            ),
            None,
        )
        if match_idx is None:
            raise AssertionError(
                f"MockLLMClient: unexpected call to {call_site!r} expecting "
                f"{response_model.__name__} — no matching response queued. "
                f"Queue currently: "
                + ", ".join(
                    f"{e.call_site}:{e.response_model_name}" for e in self._queue
                )
                + "."
            )
        exp = self._queue.pop(match_idx)
        # Sleep zero — preserves "this is a coroutine" semantics so
        # asyncio scheduling order in tests matches production.
        await asyncio.sleep(0)

        # If a dataset_repo is attached, fire the capture call the same
        # way OllamaClient does. Mirrors the real wrapper's finally
        # block so tests exercise the full path.
        if self._dataset_repo is not None:
            try:
                from chatterbot.dataset.capture import record_llm_call
                await record_llm_call(
                    self._dataset_repo,
                    call_site=call_site,
                    model_id=model_override or self.model,
                    provider="mock",
                    system_prompt=system_prompt,
                    prompt=prompt,
                    response_text=exp.response.model_dump_json(),
                    response_schema_name=response_model.__name__,
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                    think=think,
                    latency_ms=0,
                    referenced_user_ids=referenced_user_ids,
                )
            except Exception:
                pass  # capture must never break the call path
        return exp.response  # type: ignore[return-value]

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_override: str | None = None,
        think: bool = False,
    ) -> AsyncIterator[str]:
        async def _empty() -> AsyncIterator[str]:
            if False:
                yield ""  # pragma: no cover
        return _empty()

    async def embed(self, text: str) -> list[float]:
        # Deterministic but content-aware: hash the text and unfold it
        # into a 768-vector. Lets tests verify "the same input
        # produces the same embedding" without actually loading a
        # real model.
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat the digest until we hit embed_dim, then map to floats
        # in [-1, 1]. Cheap, deterministic, distinct-per-input.
        out: list[float] = []
        i = 0
        while len(out) < self._embed_dim:
            byte = h[i % len(h)]
            out.append((byte / 127.5) - 1.0)
            i += 1
        return out

    async def health_check(self) -> bool:
        return True

    # ---- dataset capture wiring (mirrors OllamaClient surface) ----

    def attach_dataset_capture(self, repo) -> None:
        self._dataset_repo = repo
