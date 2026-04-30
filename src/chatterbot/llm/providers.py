"""LLM provider abstraction.

Lets chatterbot route generation calls to Ollama (default), Anthropic
(Claude), or OpenAI based on `Settings.llm_provider`. Each backend
implements the same surface — `generate`, `generate_structured`,
`stream_generate`, `health_check` — so the rest of the codebase
doesn't care which one is in use.

Design constraints:
- **Embeddings always run on Ollama.** vec_messages and vec_threads
  are locked to `nomic-embed-text`'s 768-dim vectors; switching the
  embedding model would require re-embedding every row, and neither
  Anthropic nor OpenAI's embedding APIs match nomic's geometry. The
  factory wires `embed()` to the local Ollama regardless of the
  provider chosen for generation.
- **Structured output** is mapped to each provider's native shape:
    Ollama       — `format=<json_schema>` (constrains decoding)
    OpenAI       — `response_format={"type":"json_schema",...}`
    Anthropic    — tool-use with a single tool whose input schema
                   is the response model (Anthropic's recommended
                   approach for strict JSON since they don't have a
                   first-class structured-output flag).
- **`think=True`** maps to:
    Ollama       — the existing think-mode flag
    OpenAI       — uses an o-series reasoning model (configurable)
    Anthropic    — `extended thinking` (server-side reasoning)

The factory returns a `LLMProvider` instance; consumers just call
its methods. The shared semaphore in OllamaClient still serialises
generation calls (single-slot by default) to keep the GPU happy;
the OpenAI/Anthropic clients have their own concurrency since
those are remote APIs that auto-scale.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Protocol, TypeVar, runtime_checkable

import httpx
from pydantic import BaseModel

from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMProvider(Protocol):
    """Common interface for all generation backends.

    `embed()` always delegates to a local Ollama embedding model
    regardless of the active provider — see the module docstring."""

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
    ) -> str: ...

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
    ) -> T: ...

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_override: str | None = None,
        think: bool = False,
    ) -> AsyncIterator[str]: ...

    async def embed(self, text: str) -> list[float]: ...

    async def health_check(self) -> bool: ...


class _EmbeddingDelegator:
    """Mixin that pins embeddings to a local Ollama instance even
    when generation runs against OpenAI / Anthropic. The `embed_via`
    OllamaClient is constructed by the factory and shared across
    providers — it has its own connection pool + semaphore."""

    def __init__(self, embed_via: OllamaClient):
        self._embed_client = embed_via

    async def embed(self, text: str) -> list[float]:
        return await self._embed_client.embed(text)


# ============================================================
# Anthropic (Claude)
# ============================================================

class AnthropicClient(_EmbeddingDelegator):
    """Anthropic Messages API client. Structured output uses
    tool-use — we register a single tool whose `input_schema` is the
    response model's JSON Schema, then force the model to call that
    tool. The tool's input is the structured response. This is
    Anthropic's recommended pattern for strict JSON output.

    `think=True` enables `thinking` (extended-reasoning mode). We
    floor `max_tokens` so the reasoning trace doesn't truncate the
    final answer.
    """

    BASE_URL = "https://api.anthropic.com"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-7",
        embed_via: OllamaClient | None = None,
        timeout: float = 120.0,
        thinking_budget_tokens: int = 4096,
    ):
        if embed_via is None:
            raise ValueError(
                "AnthropicClient needs an OllamaClient to delegate "
                "embeddings to (vec_messages dim is locked to nomic-embed)."
            )
        super().__init__(embed_via)
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.thinking_budget_tokens = thinking_budget_tokens
        # Optional ChatterRepo handle for personal-dataset capture —
        # mirrors OllamaClient's surface. None until the
        # bot/dashboard startup unlock attaches one.
        self._dataset_repo = None

    def attach_dataset_capture(self, repo) -> None:
        """Wire a ChatterRepo into this client so structured calls
        record LLM_CALL events into the encrypted dataset shards.
        Capture is still gated by the repo's own toggle; attaching
        without the streamer opting in is a no-op."""
        self._dataset_repo = repo

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }

    def _build_payload(
        self,
        prompt: str,
        system_prompt: str | None,
        num_predict: int | None,
        think: bool,
        format_schema: dict[str, Any] | None,
        model_override: str | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        max_tokens = num_predict or 2048
        if think and max_tokens < self.thinking_budget_tokens + 1024:
            # Reasoning trace eats into max_tokens; give the answer
            # at least 1024 tokens of headroom.
            max_tokens = self.thinking_budget_tokens + 1024
        payload: dict[str, Any] = {
            "model": model_override or self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if stream:
            payload["stream"] = True
        if system_prompt:
            payload["system"] = system_prompt
        if think:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }
        if format_schema is not None:
            # Tool-use as the structured-output channel: register one
            # tool, force the model to call it, parse its input.
            payload["tools"] = [{
                "name": "respond",
                "description": "Provide the structured response.",
                "input_schema": format_schema,
            }]
            payload["tool_choice"] = {"type": "tool", "name": "respond"}
        return payload

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Pull the first text or tool_use block out of a Messages
        API response. Tool-use input is returned as a JSON string;
        callers parse it via the response_model in
        generate_structured()."""
        for block in data.get("content", []):
            btype = block.get("type")
            if btype == "tool_use":
                return json.dumps(block.get("input") or {})
            if btype == "text":
                return block.get("text", "")
        return ""

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        format_schema: dict[str, Any] | None = None,
        model_override: str | None = None,
        num_ctx: int | None = None,  # unused — Anthropic auto-sizes
        num_predict: int | None = None,
        images: list[str] | None = None,  # ignored — text-only path
        think: bool = False,
    ) -> str:
        payload = self._build_payload(
            prompt, system_prompt, num_predict, think, format_schema,
            model_override,
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.BASE_URL}/v1/messages",
                json=payload,
                headers=self._headers(),
            )
            r.raise_for_status()
            return self._extract_text(r.json())

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
    ) -> T:
        import time as _time
        schema = response_model.model_json_schema()
        started = _time.monotonic()
        raw = ""
        error: str | None = None
        try:
            raw = await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                format_schema=schema,
                model_override=model_override,
                num_ctx=num_ctx,
                num_predict=num_predict,
                images=images,
                think=think,
            )
            return response_model.model_validate_json(raw)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            from ..dataset.capture import record_llm_call_safe
            await record_llm_call_safe(
                self._dataset_repo,
                call_site=call_site,
                model_id=model_override or self.model,
                provider="anthropic",
                system_prompt=system_prompt,
                prompt=prompt,
                response_text=raw,
                response_schema_name=response_model.__name__,
                num_ctx=num_ctx,
                num_predict=num_predict,
                think=think,
                latency_ms=int((_time.monotonic() - started) * 1000),
                error=error,
            )

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_override: str | None = None,
        think: bool = False,
    ) -> AsyncIterator[str]:
        payload = self._build_payload(
            prompt, system_prompt, None, think, None, model_override,
            stream=True,
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}/v1/messages",
                json=payload,
                headers=self._headers(),
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    body = line[6:].strip()
                    if not body or body == "[DONE]":
                        continue
                    try:
                        event = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            text = delta.get("text") or ""
                            if text:
                                yield text
                        elif delta.get("type") == "thinking_delta":
                            text = delta.get("thinking") or ""
                            if text:
                                yield text

    async def health_check(self) -> bool:
        try:
            # No public ping endpoint; smallest-possible message.
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(
                    f"{self.BASE_URL}/v1/messages",
                    json={
                        "model": self.model,
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "ping"}],
                    },
                    headers=self._headers(),
                )
                return r.status_code in (200, 400)  # 400 = model rejected ping but auth OK
        except Exception:
            return False


# ============================================================
# OpenAI (and compatible APIs — Azure, Together, etc.)
# ============================================================

class OpenAIClient(_EmbeddingDelegator):
    """OpenAI Responses / Chat Completions client. Structured output
    uses the native `response_format={"type":"json_schema",...}`
    field, which constrains decoding server-side and is the
    cleanest mapping for our pydantic-validated calls.

    `think=True` is interpreted as "use the configured reasoning
    model" — caller routes to e.g. `gpt-5-thinking` or `o4-mini`
    via `model_override`, otherwise the plain model is used.
    """

    BASE_URL = "https://api.openai.com"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        reasoning_model: str | None = None,
        embed_via: OllamaClient | None = None,
        timeout: float = 120.0,
        organization: str | None = None,
    ):
        if embed_via is None:
            raise ValueError(
                "OpenAIClient needs an OllamaClient to delegate "
                "embeddings to (vec_messages dim is locked)."
            )
        super().__init__(embed_via)
        self.api_key = api_key
        self.model = model
        self.reasoning_model = reasoning_model
        self.timeout = timeout
        self.organization = organization
        # Optional ChatterRepo handle for personal-dataset capture —
        # mirrors OllamaClient + AnthropicClient. None until the
        # bot/dashboard startup unlock attaches one.
        self._dataset_repo = None

    def attach_dataset_capture(self, repo) -> None:
        """Wire a ChatterRepo into this client so structured calls
        record LLM_CALL events into the encrypted dataset shards.
        Capture is still gated by the repo's own toggle; attaching
        without the streamer opting in is a no-op."""
        self._dataset_repo = repo

    def _headers(self) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        if self.organization:
            h["OpenAI-Organization"] = self.organization
        return h

    def _pick_model(self, override: str | None, think: bool) -> str:
        if override:
            return override
        if think and self.reasoning_model:
            return self.reasoning_model
        return self.model

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        format_schema: dict[str, Any] | None = None,
        model_override: str | None = None,
        num_ctx: int | None = None,  # unused — OpenAI auto-sizes
        num_predict: int | None = None,
        images: list[str] | None = None,  # not wired — text-only path
        think: bool = False,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self._pick_model(model_override, think),
            "messages": messages,
        }
        if num_predict:
            payload["max_completion_tokens"] = num_predict
        if format_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "respond",
                    "schema": format_schema,
                    "strict": True,
                },
            }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.BASE_URL}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                return ""
            msg = (choices[0] or {}).get("message") or {}
            return msg.get("content") or ""

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
    ) -> T:
        import time as _time
        schema = response_model.model_json_schema()
        started = _time.monotonic()
        raw = ""
        error: str | None = None
        try:
            raw = await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                format_schema=schema,
                model_override=model_override,
                num_ctx=num_ctx,
                num_predict=num_predict,
                images=images,
                think=think,
            )
            return response_model.model_validate_json(raw)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            from ..dataset.capture import record_llm_call_safe
            await record_llm_call_safe(
                self._dataset_repo,
                call_site=call_site,
                model_id=self._pick_model(model_override, think),
                provider="openai",
                system_prompt=system_prompt,
                prompt=prompt,
                response_text=raw,
                response_schema_name=response_model.__name__,
                num_ctx=num_ctx,
                num_predict=num_predict,
                think=think,
                latency_ms=int((_time.monotonic() - started) * 1000),
                error=error,
            )

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_override: str | None = None,
        think: bool = False,
    ) -> AsyncIterator[str]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self._pick_model(model_override, think),
            "messages": messages,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    body = line[6:].strip()
                    if not body or body == "[DONE]":
                        continue
                    try:
                        event = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    delta = ((event.get("choices") or [{}])[0].get("delta") or {})
                    chunk = delta.get("content")
                    if chunk:
                        yield chunk

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    f"{self.BASE_URL}/v1/models",
                    headers=self._headers(),
                )
                return r.status_code == 200
        except Exception:
            return False


# ============================================================
# Factory
# ============================================================

def make_llm_client(settings) -> LLMProvider:
    """Build the right LLMProvider for the configured backend.

    Always builds a local OllamaClient first — that's the embedding
    path regardless of which backend handles generation. For
    `llm_provider == "ollama"` the same client services both. For
    `anthropic` / `openai` we wrap the Ollama instance for embeddings
    and use the remote API for generation.

    Raises ValueError on missing-key configurations so a misconfig
    fails loudly at startup instead of silently falling back to
    Ollama (which would defeat the point of switching providers).
    """
    embed_client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        embed_model=settings.ollama_embed_model,
        max_concurrent_generations=settings.ollama_max_concurrent_generations,
    )

    provider = (settings.llm_provider or "ollama").strip().lower()

    if provider == "ollama":
        return embed_client

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "llm_provider=anthropic but anthropic_api_key is empty"
            )
        return AnthropicClient(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model or "claude-opus-4-7",
            embed_via=embed_client,
            thinking_budget_tokens=settings.anthropic_thinking_budget_tokens,
        )

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "llm_provider=openai but openai_api_key is empty"
            )
        return OpenAIClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model or "gpt-4o",
            reasoning_model=settings.openai_reasoning_model or None,
            organization=settings.openai_organization or None,
            embed_via=embed_client,
        )

    raise ValueError(f"unknown llm_provider: {provider!r}")
