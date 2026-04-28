"""Engagement / community-insights service.

Powers the dashboard's Insights tab. Streamer-only — none of this output
ever returns to Twitch chat (same hard rule as everything else). What it
produces is a short conversation-hook line per currently-active chatter
that the streamer can riff on while reading their dashboard on a second
monitor.

Background loop refreshes the cache every ~3 min so the page render is
instant and we don't pay an LLM call per HTMX poll. Repo-derived sections
(regulars / lapsed / first-timers) are cheap and computed at request time.

Goes through `OllamaClient.generate_structured()` with the gold-standard
pydantic `TalkingPointsResponse`, so the output is constrained at
generation time AND validated on receipt.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from pydantic import ValidationError

from .config import Settings
from .llm.ollama_client import OllamaClient
from .llm.schemas import TalkingPointsResponse
from .repo import ChatterRepo

logger = logging.getLogger(__name__)


TALKING_POINTS_SYSTEM = """You help a Twitch streamer remember conversation hooks for chatters who are active in their chat right now. The streamer reads your output on a second monitor while streaming and uses it to engage with chat — your output never returns to chat itself.

You'll get a numbered list of active chatters, each with:
  - their username
  - extracted notes (facts they've previously stated about themselves)
  - a few of their most recent messages
  - optionally, the current channel-wide topic context

For each chatter, produce AT MOST ONE short conversation-starter the streamer could use. Keep each line under 25 words. Skipping is the expected default — only emit a hook when it's clearly grounded.

HARD RULES (violations make the dashboard worse, not better):
- ONLY paraphrase content that literally appears in this chatter's notes or recent messages. Do not bridge, infer, or extrapolate.
- A chatter mentioning a topic ONCE in their recent messages is NOT a hook. Wait for them to return to it. Single-mention notes are background, not chat-current.
- NEVER invent or assume the existence of products, releases, events, places, people, or facts that aren't directly attested in their messages. If a chatter says "excited for the FF8 remake," you do NOT know an FF8 remake exists — you only know the chatter said that. Phrase the hook as "they mentioned being excited for an FF8 remake" rather than "the FF8 remake."
- If a hook would require asserting something beyond what you can quote, SKIP this chatter. An empty `points` list is fine.
- Don't speculate about feelings or intent. Stay observational.
- Don't recycle the same hook the streamer has clearly already addressed (their recent messages may show this).
- Reply with chatter_index matching the number in the input, and `point` = the line.

When in doubt, skip. The cost of a missed hook is zero; the cost of a hallucinated one is the streamer publicly riffing on something that doesn't exist.
"""


@dataclass
class TalkingPointEntry:
    user_id: str
    name: str
    point: str


@dataclass
class InsightsCache:
    talking_points: list[TalkingPointEntry]
    refreshed_at: float | None  # unix ts
    error: str | None = None


class InsightsService:
    """Caches the LLM-derived talking points; refreshed by a background task."""

    DEFAULT_REFRESH_SECONDS = 180  # 3 min
    ACTIVE_WINDOW_MINUTES = 10
    RECENT_MESSAGES_PER_USER = 6

    def __init__(self, repo: ChatterRepo, llm: OllamaClient, settings: Settings):
        self.repo = repo
        self.llm = llm
        self.settings = settings
        self._cache = InsightsCache(talking_points=[], refreshed_at=None, error=None)
        self._lock = asyncio.Lock()

    @property
    def cache(self) -> InsightsCache:
        return self._cache

    async def refresh_loop(self, interval_seconds: int | None = None) -> None:
        interval = interval_seconds or self.DEFAULT_REFRESH_SECONDS
        # First pass after a small delay so the dashboard finishes booting.
        await asyncio.sleep(5)
        last_processed_id = 0
        while True:
            try:
                # Skip when no new messages have arrived since the last
                # refresh — same active chatters, same recent messages,
                # would just produce the same talking points.
                latest = await asyncio.to_thread(self.repo.latest_message_id)
                if latest > last_processed_id:
                    await self._refresh()
                    last_processed_id = latest
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("insights refresh failed")
            await asyncio.sleep(interval)

    async def _refresh(self) -> None:
        async with self._lock:
            active = await asyncio.to_thread(
                self.repo.list_active_chatters,
                self.ACTIVE_WINDOW_MINUTES,
                20,
            )
            if not active:
                self._cache = InsightsCache(
                    talking_points=[], refreshed_at=time.time(), error=None
                )
                return

            # Build a numbered prompt block.
            blocks: list[str] = []
            id_by_index: dict[int, tuple[str, str]] = {}
            for i, user in enumerate(active, start=1):
                notes = await asyncio.to_thread(self.repo.get_notes, user.twitch_id)
                msgs = await asyncio.to_thread(
                    self.repo.get_messages, user.twitch_id,
                    limit=self.RECENT_MESSAGES_PER_USER,
                )
                note_lines = (
                    "\n      - " + "\n      - ".join(n.text for n in notes)
                    if notes else " (none)"
                )
                msg_lines = "\n      - " + "\n      - ".join(
                    m.content[:160] for m in reversed(msgs)
                ) if msgs else " (none)"
                blocks.append(
                    f"[{i}] {user.name}\n"
                    f"    notes:{note_lines}\n"
                    f"    recent messages:{msg_lines}"
                )
                id_by_index[i] = (user.twitch_id, user.name)

            # Channel context: the most recent topic snapshot, if any.
            topics = await asyncio.to_thread(self.repo.list_topic_snapshots, 1)
            topic_block = (
                f"Current channel-wide topics:\n{topics[0].summary}\n\n"
                if topics else ""
            )

            prompt = (
                f"{topic_block}"
                f"Active chatters (last {self.ACTIVE_WINDOW_MINUTES} minutes):\n\n"
                + "\n\n".join(blocks)
            )

            try:
                response = await self.llm.generate_structured(
                    prompt=prompt,
                    system_prompt=TALKING_POINTS_SYSTEM,
                    response_model=TalkingPointsResponse,
                )
            except ValidationError as e:
                logger.exception("talking-points validation failed")
                self._cache = InsightsCache(
                    talking_points=[], refreshed_at=time.time(),
                    error=f"validation failed: {e!s}",
                )
                return
            except Exception as e:
                logger.exception("talking-points LLM call failed")
                self._cache = InsightsCache(
                    talking_points=[], refreshed_at=time.time(),
                    error=str(e),
                )
                return

            entries: list[TalkingPointEntry] = []
            for tp in response.points:
                pair = id_by_index.get(tp.chatter_index)
                if pair is None:
                    continue  # hallucinated index; drop
                uid, name = pair
                entries.append(
                    TalkingPointEntry(user_id=uid, name=name, point=tp.point)
                )
            self._cache = InsightsCache(
                talking_points=entries, refreshed_at=time.time(), error=None
            )
            logger.info(
                "insights refreshed: %d active, %d talking points",
                len(active), len(entries),
            )
