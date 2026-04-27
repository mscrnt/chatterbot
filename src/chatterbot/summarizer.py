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
from .llm.schemas import (
    NoteExtractionResponse,
    ProfileExtractionResponse,
    TopicEntry,
    TopicsResponse,
)
from .repo import ChatterRepo
from .threader import Threader

logger = logging.getLogger(__name__)


_AUDIENCE_BLOCK = """STREAM CONTEXT (read first):
- This is a Twitch stream focused on zombie / horror / gore games.
- The audience skews 18-35, predominantly male. Casual profanity, dark
  humor, gallows reactions to violence, and snark are baseline-normal —
  not personality red flags.
- Hype reactions to gory or violent moments ("LMAO they exploded", "lol
  send him to hell", "kill them all") are a standard chat dialect for
  this genre. Treat them as reaction noise, not as personality data.
- Sarcasm is common. A line that sounds positive ("yeah totally fine",
  "great strat bro", "this is going so well") often means the opposite
  when the surrounding chat shows something just went wrong (death,
  jump-scare, fail, controller throw). Use surrounding context lines
  (prefixed `[ctx N]`) to disambiguate before extracting anything.
- The streamer is the broadcaster, not a chat member. Reactions
  ("ggs Bawk", "you got this", "ty for stream") are addressed AT them
  and tell us nothing factual about the speaker.
"""


_INPUT_FORMAT_BLOCK = """INPUT FORMAT:
- Each line starts with `[id]` (focal user's own message — extract from
  these) or `[ctx id] otherUser:` (a chat-wide context line — DO NOT
  extract from these; they're only there so you can read the moment).
- A line tagged `(replying to X: "...")` used the platform's native
  reply feature. Use the quoted parent for context only — do not extract
  facts about person X or about the parent message itself.
"""


PROFILE_EXTRACTION_SYSTEM = (_AUDIENCE_BLOCK + "\n" + _INPUT_FORMAT_BLOCK + """
TASK: build a soft profile of ONE chat viewer (the focal user) from their
own messages. This is DIFFERENT from note extraction — notes are hard
cited facts. This is the squishier "who is this person" view: pronouns,
location, vibe, things they care about. Partial signals are fine; they
accumulate across batches.

Fields and rules:
- `pronouns`: only set when the focal viewer explicitly used pronouns
  about themselves ("she/her", "they/them", "i'm a dude"). Otherwise
  null. Don't infer from username, display name, or assumed demographic.
- `location`: only set when the viewer explicitly mentioned where they
  are ("from Sydney", "I'm in Texas", "2am here in Berlin"). Otherwise
  null. Don't infer from time-of-day or vocabulary alone.
- `demeanor`: pick ONE bucket that best fits the focal viewer's dominant
  tone in THIS batch, judged AGAINST the genre baseline above (so casual
  profanity / dark humor on its own is not "snarky" — it's the genre).
  Acceptable buckets:
    hype        — heavy caps, exclamations, big reactions to clutch moments
    chill       — measured, conversational, even-keeled
    supportive  — encouraging, hype FOR others, positive replies
    snarky      — dry/sarcastic humor specifically beyond the genre baseline
    quiet       — short, infrequent, mostly reactive
    analytical  — technical commentary on gameplay/strategy
    unknown     — genuinely can't tell, or messages are too thin
- `interests`: 0 to 5 short tags the viewer has shown interest in (specific
  games, genres, hobbies, topics, communities). Lowercase. Examples:
  "speedrunning", "resident evil", "cats", "metalcore", "vintage cameras".
  Skip generic "twitch" / "chat" / "streaming" / "zombies" (zombies is
  the topic of the stream — not a viewer-specific interest signal).

If you have no signal for a field, leave it null / empty. Empty is
EXPECTED and NORMAL — do NOT fabricate.
""")


NOTE_EXTRACTION_SYSTEM = (_AUDIENCE_BLOCK + "\n" + _INPUT_FORMAT_BLOCK + """
TASK: extract short third-person notes about ONE chat viewer (the focal
user) from their own messages.

For each note, include `source_message_ids` — the specific focal-line ids
that support that note. The streamer uses this to trace any note back to
the exact line(s) it came from. Context lines (`[ctx id]`) are NOT
allowed in source_message_ids.

WHAT COUNTS AS A NOTE (any of these):
- Hard self-disclosure: pets, gear, location, jobs, family, games they
  play, hobbies. ("Has a cat named Loki.")
- Stated opinions or takes: positions they explicitly voiced.
  ("Defends Trump.", "Thinks Putin is a threat.", "Calls Hasan based.")
- Recurring references: a person, show, game, or topic they've brought
  up across multiple focal lines. ("Often references David Lynch.")
- Stated preferences: things they've explicitly liked, disliked, or
  championed. ("Hates the Resident Evil 4 remake.", "Champions the
  classic Silent Hill 2 ending.")

RULES:
- Extract 0 to 5 notes maximum. 0 is fine and common — don't fabricate.
- Each note is one short third-person sentence grounded in what the
  viewer ACTUALLY said. Do not infer beliefs they didn't state.
- A SARCASTIC statement is NOT a real opinion. If the viewer says
  "great, I love dying repeatedly to one zombie" right after a death
  context line, do NOT record "loves dying to zombies." Skip it.
- Pure reactions to the stream content (kills, deaths, jump-scares, RNG,
  the streamer's plays) are not notes. "BASED" or "LMAO" alone tells us
  nothing about the viewer.
- No personality judgments — describe what they SAID, not who they ARE.
  "Often makes political comments" is OK; "Is opinionated" is not.
- source_message_ids must reference focal `[id]` lines that actually
  appear in the input. If a note is supported by multiple lines, list
  them all (cap 5).
""")


TOPICS_SYSTEM = """You summarize what a Twitch chat is currently talking about.

RULES:
- Identify 3 to 5 main topics from the messages provided.
- One short line per topic.
- Cite which usernames are driving each topic.
- For each topic, pick ONE category that fits best:
    gaming    — game mechanics, runs, builds, in-game events
    personal  — life updates, pets, family, work, location
    meta      — stream meta, schedule, gear, OBS, broadcast tech
    tech      — hardware, software, programming
    off-topic — jokes, banter, memes, off-the-wall
    other     — everything else
- No editorializing. No inferred sentiment. Just topic + drivers + category.
- If chat is essentially silent or unfocused, return fewer topics or an empty list.
"""


class Summarizer:
    def __init__(self, repo: ChatterRepo, llm: OllamaClient, settings: Settings):
        self.repo = repo
        self.llm = llm
        self.settings = settings
        self._user_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._inflight: set[str] = set()
        # Topic threading runs as a side-effect of each new snapshot. Keep a
        # single shared instance so backfill state could live here later.
        self._threader = Threader(repo, llm, settings)

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

            # Look up reply parents per focal message so the LLM can interpret
            # short responses ("yes", "me too", "no way") in context.
            focal_ids = [mid for mid, _ in rows]
            full_msgs = await asyncio.to_thread(
                self.repo.get_messages_by_ids, focal_ids
            )
            by_id = {m.id: m for m in full_msgs}
            # Pull a chat-wide context window around each focal message —
            # 2 lines before, 2 after — so the LLM can spot sarcasm and
            # frame reactions around key moments. Returned union is
            # deduped + ordered by id; we render focal vs ctx differently.
            ctx_msgs = await asyncio.to_thread(
                self.repo.channel_context_around_ids, focal_ids,
                before=2, after=2,
            )
            focal_id_set = set(focal_ids)
            corpus_lines: list[str] = []
            for cm in ctx_msgs:
                if cm.id in focal_id_set:
                    m = by_id.get(cm.id) or cm
                    content = m.content
                    if m.reply_parent_body:
                        snippet = m.reply_parent_body[:160].replace('"', "'")
                        parent = m.reply_parent_login or "?"
                        corpus_lines.append(
                            f'[{cm.id}] (replying to {parent}: "{snippet}") {content}'
                        )
                    else:
                        corpus_lines.append(f"[{cm.id}] {content}")
                else:
                    snippet = (cm.content or "")[:200].replace("\n", " ")
                    corpus_lines.append(f"[ctx {cm.id}] {cm.name}: {snippet}")
            corpus = "\n".join(corpus_lines)
            prompt = (
                f"Focal viewer username: {display_name}\n\n"
                f"Chat transcript (focal lines = `[id]`, context lines = "
                f"`[ctx id] otherUser:`):\n{corpus}"
            )

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

            saved_notes = 0
            dropped_uncited = 0
            for entry in response.notes:
                try:
                    embedding = await self.llm.embed(entry.text)
                except Exception:
                    logger.exception("embed failed for note; storing without vector")
                    embedding = None
                # add_note with origin='llm' returns None when none of the
                # cited source ids resolve to messages this user actually
                # sent — that's the hallucination guard. Don't save.
                note_id = await asyncio.to_thread(
                    self.repo.add_note,
                    user_id,
                    entry.text,
                    embedding,
                    list(entry.source_message_ids),
                    origin="llm",
                )
                if note_id is None:
                    dropped_uncited += 1
                    logger.warning(
                        "summarizer: dropped uncited LLM note for %s: %r "
                        "(model cited ids=%s, none belong to this user)",
                        user_id, entry.text[:80], list(entry.source_message_ids),
                    )
                else:
                    saved_notes += 1

            # Soft-profile extraction — separate LLM call with a softer
            # rubric so we still build a useful "who is this" view even
            # for chatters whose hard-fact note count stays at zero.
            # Failures here are non-fatal; the notes pass already wrote.
            profile_summary = "no fields"
            try:
                profile = await self.llm.generate_structured(
                    prompt=prompt,
                    system_prompt=PROFILE_EXTRACTION_SYSTEM,
                    response_model=ProfileExtractionResponse,
                )
                await asyncio.to_thread(
                    self.repo.update_user_profile, user_id,
                    pronouns=profile.pronouns,
                    location=profile.location,
                    demeanor=profile.demeanor,
                    interests=list(profile.interests),
                )
                bits = []
                if profile.pronouns:  bits.append(f"pronouns={profile.pronouns}")
                if profile.location:  bits.append(f"location={profile.location}")
                if profile.demeanor:  bits.append(f"demeanor={profile.demeanor}")
                if profile.interests: bits.append(f"interests={len(profile.interests)}")
                profile_summary = ", ".join(bits) if bits else "no fields"
            except ValidationError:
                logger.exception("profile extraction validation failed for %s", user_id)
            except Exception:
                logger.exception("profile LLM generate failed for %s", user_id)

            last_id = rows[-1][0]
            await asyncio.to_thread(self.repo.set_watermark, user_id, last_id)
            logger.info(
                "summarized user=%s msgs=%d -> notes=%d (dropped_uncited=%d) "
                "profile=(%s) watermark=%d",
                display_name,
                len(rows),
                saved_notes,
                dropped_uncited,
                profile_summary,
                last_id,
            )

    # ---------------- background message-embedding indexer ----------------
    # Keeps vec_messages current as new chat arrives so /search has fresh
    # coverage. Pure local Ollama work — no external API quota at risk —
    # so we don't pause on OBS offline. Survives Ollama hiccups via the
    # standard exception swallow on the loop body.

    async def embed_loop(self) -> None:
        interval = max(5, self.settings.message_embed_interval_seconds)
        batch = max(1, self.settings.message_embed_batch_size)
        while True:
            try:
                await asyncio.sleep(interval)
                rows = await asyncio.to_thread(
                    self.repo.messages_missing_embedding_global, batch
                )
                if not rows:
                    continue
                wrote = 0
                for m in rows:
                    try:
                        vec = await self.llm.embed(m.content)
                    except Exception:
                        logger.exception("embed_loop: embed call failed for msg %d", m.id)
                        continue
                    await asyncio.to_thread(
                        self.repo.upsert_message_embedding, m.id, vec
                    )
                    wrote += 1
                if wrote:
                    indexed, total = await asyncio.to_thread(
                        self.repo.messages_embedding_coverage
                    )
                    logger.info(
                        "embed_loop: +%d → %d/%d indexed (%.1f%%)",
                        wrote, indexed, total,
                        100 * indexed / total if total else 0.0,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("embed_loop iteration failed")

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
        snapshot_id = await asyncio.to_thread(
            self.repo.add_topic_snapshot, summary, msg_range, topics_json
        )
        logger.info("topic snapshot saved range=%s topics=%d", msg_range, len(response.topics))

        # Cluster each topic into the thread index right after the snapshot
        # lands. Failure here is non-fatal — the snapshot is already saved
        # and the backfill on next bot start will pick it up.
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            await self._threader.cluster_snapshot(snapshot_id, ts, topics_json)
        except Exception:
            logger.exception("threader: cluster_snapshot raised — will be retried via backfill")


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
