"""Per-topic "tell me more" RAG for the topics modal.

Streamer-only. Output renders in the streamer's browser via SSE — never
returns to Twitch chat.

Inputs:
  - a topic_snapshot row (with topics_json + message_id_range)
  - a topic index within that snapshot

Process:
  1. Look up the topic + driver names from the parsed topics_json.
  2. Pull the actual messages those drivers (resolving aliases) sent within
     the snapshot's message_id_range.
  3. Build a strict prompt and stream qwen's expansion.

Output is free-form prose, so this rides `OllamaClient.stream_generate()`
rather than the structured-output path. Schemas are for parseable shapes;
prose answers stay prose.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from ..llm.ollama_client import OllamaClient
from ..llm.schemas import TopicsResponse
from ..repo import ChatterRepo, Message, TopicSnapshot

logger = logging.getLogger(__name__)


TOPIC_EXPLAIN_SYSTEM = """You explain a single discussion topic from a Twitch chat snapshot to the streamer.

You will be given:
  - the topic title and the usernames driving it
  - the actual chat messages those users sent during the snapshot's window

RULES:
- Explain in 2-4 short sentences what was actually being said about this topic.
- Quote or paraphrase specific messages — don't invent.
- If the messages don't actually substantiate the topic, say so plainly.
- No editorializing, no inferred sentiment. Just what was said.
- This output renders to the streamer's private dashboard — not to chat.
"""


class TopicContext:
    def __init__(
        self,
        snapshot: TopicSnapshot,
        topic_title: str,
        drivers: list[str],
        messages: list[Message],
        stream: AsyncIterator[str],
    ):
        self.snapshot = snapshot
        self.topic_title = topic_title
        self.drivers = drivers
        self.messages = messages
        self.stream = stream


async def explain_topic(
    repo: ChatterRepo,
    llm: OllamaClient,
    snapshot_id: int,
    topic_index: int,
) -> TopicContext | None:
    snapshot = await asyncio.to_thread(repo.get_topic_snapshot, snapshot_id)
    if not snapshot or not snapshot.topics_json:
        return None

    try:
        parsed = TopicsResponse.model_validate_json(snapshot.topics_json)
    except Exception:
        logger.exception("snapshot %d has unparseable topics_json", snapshot_id)
        return None

    if not (0 <= topic_index < len(parsed.topics)):
        return None

    entry = parsed.topics[topic_index]
    first_id, last_id = _parse_range(snapshot.message_id_range)
    messages = await asyncio.to_thread(
        repo.messages_in_id_range_for_names,
        first_id,
        last_id,
        entry.drivers,
        50,
    )

    prompt = _build_prompt(entry.topic, entry.drivers, messages)
    stream = llm.stream_generate(prompt=prompt, system_prompt=TOPIC_EXPLAIN_SYSTEM)
    return TopicContext(
        snapshot=snapshot,
        topic_title=entry.topic,
        drivers=list(entry.drivers),
        messages=messages,
        stream=stream,
    )


def _parse_range(rng: str | None) -> tuple[int, int]:
    if not rng or "-" not in rng:
        return (0, 0)
    try:
        a, b = rng.split("-", 1)
        return (int(a), int(b))
    except ValueError:
        return (0, 0)


def _build_prompt(topic: str, drivers: list[str], messages: list[Message]) -> str:
    drv = ", ".join(drivers) if drivers else "(unspecified)"
    parts = [f"Topic: {topic}", f"Drivers: {drv}", "", "Messages from those drivers:"]
    if not messages:
        parts.append("- (none in the snapshot's window)")
    else:
        for m in messages:
            ts = m.ts.replace("T", " ")[:16]
            parts.append(f"- [{ts}] {m.name}: {m.content}")
    parts.append("")
    parts.append("Explain what was being discussed.")
    return "\n".join(parts)
