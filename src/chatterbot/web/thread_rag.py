"""Pull-thread RAG — generate a "what they were saying + here's a follow-up
you could ask" summary for an entire topic thread.

Streamer-only. The modal triggers an SSE stream into the streamer's
browser; nothing returns to chat.

Inputs:
  - a topic_thread + its members (cumulative across snapshots)

Process:
  1. Look up the thread + members + cumulative drivers via repo.
  2. Pull every message that the cumulative drivers sent inside any
     member's snapshot id range (deduped, oldest first).
  3. Send to qwen via stream_generate with a strict prompt: 2-sentence
     recap + ONE concrete follow-up suggestion the streamer could use.

This rides `OllamaClient.stream_generate()` (free-form prose, no schema)
because the output is for the streamer to read, not parse.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from ..llm.ollama_client import OllamaClient
from ..repo import ChatterRepo, Message, TopicThread, TopicThreadMember

logger = logging.getLogger(__name__)


THREAD_FOLLOWUP_SYSTEM = """You help a Twitch streamer pick up a thread of conversation that's gone quiet — or refresh themselves on what was just being said.

You'll get:
  - a topic title
  - the chatters who drove it (across one or more snapshot moments)
  - the actual messages they exchanged

Output, in this exact order:

**Recap.** 2 sentences max. What were they actually talking about? Reference specific things they said.

**Pick this back up.** ONE concrete line the streamer could say (or ask) in chat to revive the conversation. Make it specific to what was discussed — not generic. Address the chatter(s) by name where natural.

RULES:
- Only use what's in the messages. Don't invent.
- Stay observational. No speculation about feelings or intent.
- This output renders to the streamer's private dashboard — not to chat.
- Keep it tight. The streamer is reading it on a second monitor while live.
"""


class ThreadContext:
    def __init__(
        self,
        thread: TopicThread,
        members: list[TopicThreadMember],
        messages: list[Message],
        stream: AsyncIterator[str],
    ):
        self.thread = thread
        self.members = members
        self.messages = messages
        self.stream = stream


async def explain_thread(
    repo: ChatterRepo, llm: OllamaClient, thread_id: int
) -> ThreadContext | None:
    thread = await asyncio.to_thread(repo.get_thread, thread_id)
    if thread is None:
        return None
    members = await asyncio.to_thread(repo.get_thread_members, thread_id)
    messages = await asyncio.to_thread(repo.get_thread_messages, thread_id, 200)

    prompt = _build_prompt(thread, messages)
    stream = llm.stream_generate(prompt=prompt, system_prompt=THREAD_FOLLOWUP_SYSTEM)
    return ThreadContext(thread=thread, members=members, messages=messages, stream=stream)


def _build_prompt(thread: TopicThread, messages: list[Message]) -> str:
    drv = ", ".join(thread.drivers) if thread.drivers else "(unspecified)"
    parts = [
        f"Thread title: {thread.title}",
        f"Drivers across all snapshots: {drv}",
        f"First seen: {thread.first_ts} · last seen: {thread.last_ts}",
        f"Member snapshots: {thread.member_count}",
        "",
        "Messages from those drivers (oldest first):",
    ]
    if not messages:
        parts.append("- (none — drivers may have been outside their snapshot windows)")
    else:
        for m in messages:
            ts = m.ts.replace("T", " ")[:16]
            line = f"- [{ts}] {m.name}: {m.content}"
            if m.reply_parent_body:
                snip = m.reply_parent_body[:120].replace('"', "'")
                line += f'  (replying to {m.reply_parent_login or "?"}: "{snip}")'
            parts.append(line)
    return "\n".join(parts)
