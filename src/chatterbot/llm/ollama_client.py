"""Ollama HTTP client.

Pattern copied from streamlored/src/streamlored/llm/ollama_client.py with the
following changes:

  - `embed()` method (we need embeddings for note storage / RAG).
  - `stream_generate()` for the dashboard's "Ask Qwen" SSE endpoint.
  - `generate_structured(..., response_model=PydanticModel)` — the **gold
    standard** for any LLM call that expects parseable output. Passes the
    model's JSON Schema to Ollama as `format`, then validates the response
    with the same pydantic model. See llm/schemas.py.
  - Optional `think=True` per call. Off by default because Qwen3.5's chain-
    of-thought is slow; turn it on for accuracy-over-latency calls (note
    extraction, profile summaries, recap). Generation calls also get a
    larger num_predict cap when thinking, since CoT consumes its own budget
    before the answer.
  - A single async semaphore (default capacity 1) serialises generate calls
    across the process so background loops don't dogpile Ollama. Embeddings
    are excluded — they're cheap and frequently parallel during backfill.

If you find yourself doing `json.loads()` on `generate()` output anywhere
else in the codebase, replace it with a `generate_structured()` call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, TypeVar

import httpx
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


# When think=True the model spends tokens on chain-of-thought before the
# answer. The defaults below are the floor we apply if the caller hasn't
# explicitly raised them — keeps a small num_predict from truncating mid-
# thought.
_THINK_NUM_CTX_FLOOR = 16384
_THINK_NUM_PREDICT_FLOOR = 4096


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        embed_model: str,
        timeout: float = 120.0,
        max_concurrent_generations: int = 1,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.timeout = timeout
        # Single-slot by default — Ollama serialises on the GPU anyway, so
        # parallel client requests just queue at the network layer with
        # unbounded latency. A semaphore at the client surface gives us a
        # FAIR FIFO queue and lets callers reason about wait time.
        # Embeddings bypass this lock (see `embed()` below) since they're
        # cheap, parallel-safe in Ollama, and called in tight backfill
        # loops.
        cap = max(1, int(max_concurrent_generations))
        self._gen_sem = asyncio.Semaphore(cap)

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
        """Low-level generate. Returns raw response string.

        For structured output, prefer `generate_structured()` — it ensures the
        same schema is enforced at generation time and validated on receipt.

        `num_ctx` overrides Ollama's per-call context window. Ollama defaults
        to 2048 tokens which is fine for short prompts; bump for long bundles
        (transcript windows, large summaries). Qwen 2.5/3.5 family supports
        up to 131072+ at the model level.

        `num_predict` caps response tokens (Ollama default ~128). Bump it
        whenever the response can be long — recaps, multi-subject extraction.

        `images` is a list of base64-encoded image strings (no `data:` URI
        prefix — just the b64 payload). Multimodal models (Qwen3.5-9B has
        a vision encoder) consume them alongside the prompt. Non-vision
        models silently ignore the field.

        `think=True` enables chain-of-thought reasoning for accuracy-over-
        latency calls. Floors num_ctx and num_predict if the caller didn't
        raise them, so the CoT trace doesn't truncate mid-thought.
        """
        if think:
            num_ctx = max(num_ctx or 0, _THINK_NUM_CTX_FLOOR)
            num_predict = max(num_predict or 0, _THINK_NUM_PREDICT_FLOOR)

        payload: dict[str, Any] = {
            "model": model_override or self.model,
            "prompt": prompt,
            "stream": False,
            "think": bool(think),
        }
        if system_prompt:
            payload["system"] = system_prompt
        if format_schema is not None:
            # Ollama 0.5+ accepts a JSON Schema here and constrains decoding.
            payload["format"] = format_schema
        options: dict[str, Any] = {}
        if num_ctx is not None:
            options["num_ctx"] = int(num_ctx)
        if num_predict is not None:
            options["num_predict"] = int(num_predict)
        if options:
            payload["options"] = options
        if images:
            payload["images"] = list(images)

        # Fair FIFO across all callers in this process.
        wait_started = time.monotonic()
        async with self._gen_sem:
            waited = time.monotonic() - wait_started
            if waited > 1.0:
                logger.info(
                    "ollama queue wait %.1fs (think=%s, model=%s)",
                    waited, think, payload["model"],
                )
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
        num_predict: int | None = None,
        images: list[str] | None = None,
        think: bool = False,
    ) -> T:
        """Run a generation that returns a validated `response_model` instance.

        The model's JSON Schema constrains decoding server-side; the response
        is then parsed and validated through the same model. Raises
        `pydantic.ValidationError` if the output doesn't conform (rare with
        structured output, but fail-loudly is the right default).

        See `generate()` for `think`, `num_ctx`, `num_predict` semantics.
        """
        schema = response_model.model_json_schema()
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

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_override: str | None = None,
        think: bool = False,
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
            "think": bool(think),
        }
        if think:
            payload["options"] = {
                "num_ctx": _THINK_NUM_CTX_FLOOR,
                "num_predict": _THINK_NUM_PREDICT_FLOOR,
            }
        if system_prompt:
            payload["system"] = system_prompt

        wait_started = time.monotonic()
        async with self._gen_sem:
            waited = time.monotonic() - wait_started
            if waited > 1.0:
                logger.info(
                    "ollama stream queue wait %.1fs (think=%s)",
                    waited, think,
                )
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
        # Embeddings bypass _gen_sem on purpose — they're independent of
        # generation throughput on Ollama, and tight backfill loops would
        # otherwise stall behind a slow generation call.
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
