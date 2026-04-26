"""Moderation classifier — opt-in advisory background loop.

When `MOD_MODE_ENABLED=true`, this loop periodically pulls the chunk of
recent chat messages newer than the moderation watermark, batches them
through a strict-rubric LLM classifier (gold-standard pydantic structured
output), and persists any flagged messages as `incidents` rows for the
streamer to review on the dashboard's Moderation tab.

Hard rule: this is **advisory only**. The bot never produces chat output,
never times anyone out, never bans anyone. The streamer reviews each
incident in the dashboard and decides what to do.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import ValidationError

from .config import Settings
from .llm.ollama_client import OllamaClient
from .llm.schemas import ModerationBatchResponse
from .repo import ChatterRepo

logger = logging.getLogger(__name__)


MOD_REVIEW_SYSTEM = """You are reviewing recent Twitch chat messages for potential community-rule violations on behalf of the streamer.

For each message you receive (each has an integer message_id and a username), decide whether it is a violation:

- is_violation: true ONLY when the message clearly violates community standards.
- severity: 1 = minor (mild rule-bending), 2 = warning (clear violation, low harm), 3 = serious (hate, credible threats, doxxing).
- categories: zero or more of harassment, hate_speech, threats, spam, doxxing, other.
- rationale: 1-2 sentence factual explanation referencing what was said.

RULES:
- Be conservative. When in doubt, is_violation = false. False positives harm innocent viewers.
- Heated gaming reactions, sarcasm, casual profanity, in-jokes between regulars, and general silliness are NOT violations.
- Quoting or referencing a slur to call it out, criticize, or report it is NOT a violation.
- DO flag: hateful slurs targeting people / groups, credible threats of violence, doxxing attempts (sharing of personal info), persistent targeted harassment of a specific user, blatant spam / scam links / follow-bot output.

Return ONLY classifications where is_violation = true. Empty `classifications` array is the normal, expected result.
"""


class Moderator:
    def __init__(self, repo: ChatterRepo, llm: OllamaClient, settings: Settings):
        self.repo = repo
        self.llm = llm
        self.settings = settings

    async def review_loop(self) -> None:
        interval_seconds = max(60, self.settings.mod_review_interval_minutes * 60)
        max_msgs = self.settings.mod_review_max_messages
        # Don't fire immediately on boot; let the bot accumulate some chat first.
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                await self._review_batch(max_msgs)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("moderator review_loop iteration failed")

    async def _review_batch(self, max_messages: int) -> None:
        rows = await asyncio.to_thread(
            self.repo.messages_for_mod_review, max_messages
        )
        if not rows:
            return

        # Build a numbered prompt block. Use just username + content; no
        # historical context or notes (those are unrelated to whether THIS
        # message is a violation).
        lines = [f"[{m.id}] {m.name}: {m.content}" for m in rows]
        prompt = "Messages to review:\n" + "\n".join(lines)

        try:
            response = await self.llm.generate_structured(
                prompt=prompt,
                system_prompt=MOD_REVIEW_SYSTEM,
                response_model=ModerationBatchResponse,
            )
        except ValidationError:
            logger.exception("moderation classifier validation failed")
            # Don't advance the watermark — try again next pass.
            return
        except Exception:
            logger.exception("moderation LLM generate failed")
            return

        # Map message_id -> Message so we can recover user_id when storing.
        by_id = {m.id: m for m in rows}
        flagged = 0
        for entry in response.classifications:
            if not entry.is_violation:
                continue
            msg = by_id.get(entry.message_id)
            if msg is None:
                # The model invented an id outside the batch; ignore.
                continue
            try:
                await asyncio.to_thread(
                    self.repo.add_incident,
                    user_id=msg.user_id,
                    message_id=msg.id,
                    severity=int(entry.severity),
                    categories=list(entry.categories),
                    rationale=entry.rationale,
                )
                flagged += 1
            except Exception:
                logger.exception("failed to persist incident for msg %d", msg.id)

        # Advance the watermark to the highest id we sent through, regardless
        # of whether anything was flagged. We never re-review the same message.
        last_id = rows[-1].id
        await asyncio.to_thread(self.repo.set_mod_watermark, last_id)
        logger.info(
            "moderator reviewed=%d flagged=%d watermark=%d",
            len(rows), flagged, last_id,
        )
