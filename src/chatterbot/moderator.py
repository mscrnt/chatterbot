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

The prompt has two sections:

  [CONTEXT] — earlier chat messages, included so you understand what was being
             discussed. **DO NOT classify these.** They are background only.

  [REVIEW]  — the messages you must judge. Return one classification per
             violating message_id. Empty list is the normal result.

For each [REVIEW] message, decide:
- is_violation: true ONLY when the message clearly violates community standards.
- severity: 1 = minor (mild rule-bending), 2 = warning (clear violation, low harm), 3 = serious (hate, credible threats, doxxing).
- categories: zero or more of harassment, hate_speech, threats, spam, doxxing, other.
- rationale: 1-2 sentence factual explanation referencing what was said.

RULES:
- Be conservative. When in doubt, is_violation = false. False positives harm innocent viewers.
- USE THE CONTEXT. Banter inside an established friendly thread, gaming-react
  hyperbole ("kill the boss", "die already"), heated reactions, sarcasm,
  casual profanity, in-jokes between regulars — NOT violations.
- A message tagged `(replying to X: "...")` is a Twitch native reply; treat
  the quoted parent as authoritative context for what's being responded to.
- Quoting or referencing a slur to call it out, criticize, or report it is
  NOT a violation.
- DO flag: hateful slurs targeting people / groups, credible threats of
  violence, doxxing attempts (sharing of personal info), persistent targeted
  harassment of a specific user, blatant spam / scam links / follow-bot output.

Return ONLY classifications where is_violation = true and message_id is in [REVIEW].
"""

# How many messages to include as look-back context before the batch.
MOD_LOOKBACK_CONTEXT = 10


def _format_message_line(m) -> str:  # noqa: ANN001 — repo.Message
    base = f"[{m.id}] {m.name}"
    if m.reply_parent_body:
        snippet = m.reply_parent_body[:160].replace('"', "'")
        parent = m.reply_parent_login or "?"
        base += f' (replying to {parent}: "{snippet}")'
    return f"{base}: {m.content}"


class Moderator:
    def __init__(self, repo: ChatterRepo, llm: OllamaClient, settings: Settings):
        self.repo = repo
        self.llm = llm
        self.settings = settings

    async def review_loop(self) -> None:
        interval_seconds = max(60, self.settings.mod_review_interval_minutes * 60)
        max_msgs = self.settings.mod_review_max_messages
        # Don't fire immediately on boot; let the bot accumulate some chat first.
        # Watermark guard — same pattern as insights.refresh_loop. If no new
        # messages have arrived since the last batch, skip the LLM call
        # entirely. Saves a wake-up + classifier eval cycle on idle chat.
        last_processed_id = 0
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                latest = await asyncio.to_thread(self.repo.latest_message_id)
                if latest <= last_processed_id:
                    continue  # no new chat → no new work
                await self._review_batch(max_msgs)
                last_processed_id = latest
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

        # Pull a small look-back window so the model sees what was being said
        # immediately before the batch starts. The watermark moves only past
        # the [REVIEW] block, never past [CONTEXT].
        first_id = rows[0].id
        context = await asyncio.to_thread(
            self.repo.messages_before_id, first_id, MOD_LOOKBACK_CONTEXT
        )

        parts: list[str] = ["[CONTEXT — do NOT classify these, background only]"]
        if context:
            parts.extend(_format_message_line(m) for m in context)
        else:
            parts.append("(no earlier messages in our store)")
        parts.append("")
        parts.append("[REVIEW — judge each]")
        parts.extend(_format_message_line(m) for m in rows)
        prompt = "\n".join(parts)

        try:
            response = await self.llm.generate_structured(
                prompt=prompt,
                system_prompt=MOD_REVIEW_SYSTEM,
                response_model=ModerationBatchResponse,
                # Route through a smaller / faster model when configured —
                # moderation is the highest-frequency LLM call and benefits
                # most from running on dedicated capacity.
                model_override=self.settings.ollama_mod_model or None,
                call_site="moderator.incident_classification",
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
