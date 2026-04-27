"""Ollama HTTP client.

Pattern copied from streamlored/src/streamlored/llm/ollama_client.py with the
following changes:

  - `embed()` method (we need embeddings for note storage / RAG).
  - `stream_generate()` for the dashboard's "Ask Qwen" SSE endpoint.
  - `generate_structured(..., response_model=PydanticModel)` — the **gold
    standard** for any LLM call that expects parseable output. Passes the
    model's JSON Schema to Ollama as `format`, then validates the response
    with the same pydantic model. See llm/schemas.py.
  - `think: false` is hard-coded in every generate call (Qwen3.5 thinks by
    default and it's slow; we never want CoT here).

If you find yourself doing `json.loads()` on `generate()` output anywhere
else in the codebase, replace it with a `generate_structured()` call.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, TypeVar

import httpx
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        embed_model: str,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        format_schema: dict[str, Any] | None = None,
        model_override: str | None = None,
        num_ctx: int | None = None,
    ) -> str:
        """Low-level generate. Returns raw response string.

        For structured output, prefer `generate_structured()` — it ensures the
        same schema is enforced at generation time and validated on receipt.

        `num_ctx` overrides Ollama's per-call context window. Ollama defaults
        to 2048 tokens which is fine for short prompts; bump for long bundles
        (transcript windows, large summaries). Qwen 2.5 family supports up
        to 131072 at the model level.
        """
        payload: dict[str, Any] = {
            "model": model_override or self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if format_schema is not None:
            # Ollama 0.5+ accepts a JSON Schema here and constrains decoding.
            payload["format"] = format_schema
        if num_ctx is not None:
            payload["options"] = {"num_ctx": int(num_ctx)}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        model_override: str | None = None,
        num_ctx: int | None = None,
    ) -> T:
        """Run a generation that returns a validated `response_model` instance.

        The model's JSON Schema constrains decoding server-side; the response
        is then parsed and validated through the same model. Raises
        `pydantic.ValidationError` if the output doesn't conform (rare with
        structured output, but fail-loudly is the right default).
        """
        schema = response_model.model_json_schema()
        raw = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            format_schema=schema,
            model_override=model_override,
            num_ctx=num_ctx,
        )
        return response_model.model_validate_json(raw)

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_override: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield response chunks as Ollama streams them.

        Used by the dashboard's RAG endpoint so the streamer sees tokens land
        progressively. The output is free-form prose (rendered to the
        streamer's browser) — no schema is applied here.
        """
        payload: dict[str, Any] = {
            "model": model_override or self.model,
            "prompt": prompt,
            "stream": True,
            "think": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/generate", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = event.get("response")
                    if chunk:
                        yield chunk
                    if event.get("done"):
                        break

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
