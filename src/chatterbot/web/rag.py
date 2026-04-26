"""Per-user RAG for the dashboard's "Ask Qwen about this user" feature.

Streamer-only. Output renders in the streamer's browser via SSE — never to
Twitch chat. The bot has no chat-output surface; this RAG path lives entirely
inside the dashboard process.

Flow:
  1. Embed the streamer's question.
  2. Lazily embed any of the user's messages that don't yet have an embedding
     (cap to a recent window so we never block on a huge backfill).
  3. Top-K notes + top-K messages by cosine distance.
  4. Build a prompt: notes block + messages block + question.
  5. Stream the answer.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from ..llm.ollama_client import OllamaClient
from ..repo import ChatterRepo, Message, Note

logger = logging.getLogger(__name__)


RAG_SYSTEM = """You are a research assistant helping a Twitch streamer remember details about a single viewer.

You will be given:
  - a small set of facts ("Notes") that were previously extracted from this viewer's chat
  - a small set of recent messages this viewer sent in the streamer's chat
  - one question from the streamer

RULES:
- Answer ONLY from the supplied notes and messages. Do not invent facts.
- If the supplied data does not answer the question, say so plainly.
- Be candid and concise. Don't pad. Don't editorialize.
- This output goes to the streamer's private dashboard — not to chat.
- When you reference a specific message, quote it briefly so the streamer can recognize it.
"""


# Cap on how many missing-embedding messages we'll embed in a single request.
# Avoids long stalls if a user has thousands of unprocessed messages.
EMBED_BACKFILL_CAP = 200


class RagAnswer:
    def __init__(
        self,
        notes: list[Note],
        messages: list[Message],
        stream: AsyncIterator[str],
    ):
        self.notes = notes
        self.messages = messages
        self.stream = stream


async def answer_for_user(
    repo: ChatterRepo,
    llm: OllamaClient,
    user_id: str,
    question: str,
) -> RagAnswer:
    user = await _to_thread(repo.get_user, user_id)
    if not user:
        async def _none() -> AsyncIterator[str]:
            yield "User not found."
            return
        return RagAnswer([], [], _none())

    # Lazy embedding backfill for messages we haven't embedded yet.
    pending = await _to_thread(repo.messages_missing_embedding, user_id, EMBED_BACKFILL_CAP)
    for msg in pending:
        try:
            vec = await llm.embed(msg.content)
        except Exception:
            logger.exception("embed failed for message %d; skipping", msg.id)
            continue
        await _to_thread(repo.upsert_message_embedding, msg.id, vec)

    try:
        q_vec = await llm.embed(question)
    except Exception:
        logger.exception("embed failed for RAG question")
        async def _err() -> AsyncIterator[str]:
            yield "Embedding failed — check the Ollama connection."
            return
        return RagAnswer([], [], _err())

    notes = await _to_thread(repo.search_user_notes, user_id, q_vec, 5)
    messages = await _to_thread(repo.search_user_messages, user_id, q_vec, 10)

    prompt = _build_prompt(user.name, question, notes, messages)
    stream = llm.stream_generate(prompt=prompt, system_prompt=RAG_SYSTEM)
    return RagAnswer(notes=notes, messages=messages, stream=stream)


def _build_prompt(
    name: str, question: str, notes: list[Note], messages: list[Message]
) -> str:
    parts: list[str] = [f"Viewer: {name}"]
    if notes:
        parts.append("\n[Notes]")
        for n in notes:
            parts.append(f"- {n.text}")
    else:
        parts.append("\n[Notes]\n- (none)")
    if messages:
        parts.append("\n[Messages]")
        for m in messages:
            ts = m.ts.replace("T", " ")[:16]
            parts.append(f"- [{ts}] {m.content}")
    else:
        parts.append("\n[Messages]\n- (none)")
    parts.append(f"\nQuestion: {question}")
    return "\n".join(parts)


async def _to_thread(fn, *args, **kwargs):
    import asyncio

    return await asyncio.to_thread(fn, *args, **kwargs)
