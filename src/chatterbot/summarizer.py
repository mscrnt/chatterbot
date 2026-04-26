"""Background summarization pipeline.

Two loops:

1. Per-user note extraction. When a user's count of unsummarized messages
   reaches the threshold OR they've been idle long enough with messages
   pending, ship the unsummarized messages to the LLM with a strict
   factual-extraction prompt, embed each surviving note, store it, and
   advance the watermark. Messages are NOT deleted.

2. Channel-wide topic snapshot. Every M minutes, summarize the most recent
   K messages across all opted-in users into a "what's chat talking about
   right now" snapshot. Streamer-only — never enters a chat-facing prompt.

Both LLM calls go through `OllamaClient.generate_structured()` with pydantic
schemas (see llm/schemas.py). That gives us schema-constrained generation
plus parse-time validation in one shot. Don't reach for `json.loads` here.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from pydantic import ValidationError

from .config import Settings
from .llm.ollama_client import OllamaClient
from .llm.schemas import NoteExtractionResponse, TopicEntry, TopicsResponse
from .repo import ChatterRepo

logger = logging.getLogger(__name__)


NOTE_EXTRACTION_SYSTEM = """You analyze recent Twitch chat messages from one viewer and extract short factual notes about that viewer.

RULES:
- Extract 0 to 3 notes maximum.
- Only facts the viewer explicitly stated about themselves: interests, pets, gear, location, games they play, jobs, family, etc.
- No personality judgments. No inferred sentiment. No speculation about mood or intent.
- If nothing notable was stated, return an empty list.
- Each note: one short third-person sentence about the viewer (e.g., "Has a cat named Loki.").
- Ignore stream meta-chatter, reactions to gameplay, emote spam, and questions to the streamer.
- Some lines are tagged `(replying to X: "...")` — that's a Twitch native reply.
  Use the quoted parent only as context to understand what the viewer is responding to.
  Do NOT extract facts about person X or about the parent message itself.
"""


TOPICS_SYSTEM = """You summarize what a Twitch chat is currently talking about.

RULES:
- Identify 3 to 5 main topics from the messages provided.
- One short line per topic.
- Cite which usernames are driving each topic.
- No editorializing. No inferred sentiment. Just topic + drivers.
- If chat is essentially silent or unfocused, return fewer topics or an empty list.
"""


class Summarizer:
    def __init__(self, repo: ChatterRepo, llm: OllamaClient, settings: Settings):
        self.repo = repo
        self.llm = llm
        self.settings = settings
        self._user_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._inflight: set[str] = set()

    # ---------------- per-user note extraction ----------------

    async def maybe_summarize_user(self, user_id: str, unsummarized_count: int) -> None:
        if unsummarized_count < self.settings.summarize_after_messages:
            return
        if user_id in self._inflight:
            return
        self._inflight.add(user_id)
        asyncio.create_task(self._summarize_user_safe(user_id))

    async def _summarize_user_safe(self, user_id: str) -> None:
        try:
            await self._summarize_user(user_id)
        except Exception:
            logger.exception("summarize_user failed for %s", user_id)
        finally:
            self._inflight.discard(user_id)

    async def _summarize_user(self, user_id: str) -> None:
        async with self._user_locks[user_id]:
            if await asyncio.to_thread(self.repo.is_opted_out, user_id):
                # Don't summarize, but advance the watermark so we don't keep
                # re-checking these messages forever.
                pending = await asyncio.to_thread(self.repo.messages_since_watermark, user_id)
                if pending:
                    last_id = pending[-1][0]
                    await asyncio.to_thread(self.repo.set_watermark, user_id, last_id)
                return

            rows = await asyncio.to_thread(self.repo.messages_since_watermark, user_id)
            if not rows:
                return

            user = await asyncio.to_thread(self.repo.get_user, user_id)
            display_name = user.name if user else user_id

            # Look up reply parents per message so the LLM can interpret
            # short responses ("yes", "me too", "no way") in context.
            full_msgs = await asyncio.to_thread(
                self.repo.get_messages_by_ids, [mid for mid, _ in rows]
            )
            by_id = {m.id: m for m in full_msgs}
            corpus_lines: list[str] = []
            for mid, content in rows:
                m = by_id.get(mid)
                if m and m.reply_parent_body:
                    snippet = m.reply_parent_body[:160].replace('"', "'")
                    parent = m.reply_parent_login or "?"
                    corpus_lines.append(
                        f'- (replying to {parent}: "{snippet}") {content}'
                    )
                else:
                    corpus_lines.append(f"- {content}")
            corpus = "\n".join(corpus_lines)
            prompt = f"Viewer username: {display_name}\n\nMessages:\n{corpus}"

            try:
                response = await self.llm.generate_structured(
                    prompt=prompt,
                    system_prompt=NOTE_EXTRACTION_SYSTEM,
                    response_model=NoteExtractionResponse,
                )
            except ValidationError:
                logger.exception("note extraction validation failed for %s", user_id)
                # Don't advance watermark — try again next pass.
                return
            except Exception:
                logger.exception("LLM generate failed for user %s", user_id)
                return

            for text in response.notes:
                try:
                    embedding = await self.llm.embed(text)
                except Exception:
                    logger.exception("embed failed for note; storing without vector")
                    embedding = None
                await asyncio.to_thread(self.repo.add_note, user_id, text, embedding)

            last_id = rows[-1][0]
            await asyncio.to_thread(self.repo.set_watermark, user_id, last_id)
            logger.info(
                "summarized user=%s msgs=%d -> notes=%d watermark=%d",
                display_name,
                len(rows),
                len(response.notes),
                last_id,
            )

    async def idle_loop(self) -> None:
        interval = self.settings.idle_sweep_interval_seconds
        idle_minutes = self.settings.summarize_idle_minutes
        while True:
            try:
                await asyncio.sleep(interval)
                idle_users = await asyncio.to_thread(
                    self.repo.users_with_idle_unsummarized, idle_minutes
                )
                for uid in idle_users:
                    if uid in self._inflight:
                        continue
                    self._inflight.add(uid)
                    asyncio.create_task(self._summarize_user_safe(uid))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("idle_loop iteration failed")

    # ---------------- channel-wide topic snapshots ----------------

    async def topics_loop(self) -> None:
        interval_seconds = max(60, self.settings.topics_interval_minutes * 60)
        max_msgs = self.settings.topics_max_messages
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                await self._take_topic_snapshot(max_msgs)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("topics_loop iteration failed")

    async def _take_topic_snapshot(self, max_messages: int) -> None:
        rows = await asyncio.to_thread(
            self.repo.recent_messages_for_topics, max_messages
        )
        if not rows:
            return
        first_id = rows[0][0]
        last_id = rows[-1][0]
        formatted = "\n".join(f"{name}: {content}" for _, name, content in rows)

        try:
            response = await self.llm.generate_structured(
                prompt=f"Recent chat (oldest first):\n{formatted}",
                system_prompt=TOPICS_SYSTEM,
                response_model=TopicsResponse,
            )
        except ValidationError:
            logger.exception("topics extraction validation failed")
            return
        except Exception:
            logger.exception("topic LLM generate failed")
            return

        if not response.topics:
            return

        summary = _render_topics(response.topics)
        msg_range = f"{first_id}-{last_id}"
        # Persist the structured topics alongside the rendered string so the
        # dashboard can drive the per-topic "tell me more" modal.
        topics_json = response.model_dump_json()
        await asyncio.to_thread(
            self.repo.add_topic_snapshot, summary, msg_range, topics_json
        )
        logger.info("topic snapshot saved range=%s topics=%d", msg_range, len(response.topics))


def _render_topics(topics: list[TopicEntry]) -> str:
    """Flatten the validated topics list into the bullet-string we currently
    persist in `topic_snapshots.summary`. Kept here so the on-disk format
    stays under one roof."""
    lines: list[str] = []
    for t in topics:
        if t.drivers:
            drv = ", ".join(t.drivers)
            lines.append(f"\u2022 {t.topic} ({drv})")
        else:
            lines.append(f"\u2022 {t.topic}")
    return "\n".join(lines)
