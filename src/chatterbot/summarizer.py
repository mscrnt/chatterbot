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

WHAT COUNTS AS A NOTE — be generous; the streamer would rather skim 6
mediocre notes than miss the one that mattered. Capture any of:

- Hard self-disclosure: pets, gear, location, jobs, family, games they
  play, hobbies, schedule, health, education. ("Has a cat named Loki.",
  "Lives in Sydney.", "Works night shift.")
- Stated opinions or takes — political, cultural, gaming, tech. Even
  weak signals count if the chatter clearly meant them.
  ("Defends Trump.", "Thinks the FF7 remake split was a mistake.")
- Recurring references: a person, show, game, song, meme, or topic the
  chatter has brought up — even ONCE counts if it's specific enough to
  be a callback later. ("Mentioned David Lynch.", "Brought up
  Yakuza 0 unprompted.")
- Stated preferences: things they've liked, disliked, or championed.
  ("Hates the Resident Evil 4 remake.", "Loves Silent Hill 2.")
- Recurring chat patterns specific to this viewer: a catchphrase,
  recurring tease, in-joke they keep returning to. ("Constantly
  ribs the streamer about the missed parry from last stream.")
- Knowledge / expertise signals: technical comments, shop talk, lore
  deep-cuts that suggest the chatter knows what they're talking about.
  ("Knows Souls-game lore — quoted Praise the Sun phrasing.")

RULES:
- Extract up to 8 notes. **Aim for 2-4 on a chatter who's said
  anything substantive over 10+ messages**; 0 only when the messages
  are pure greetings / reactions. Don't withhold a note just because
  it feels small — a single grounded line is useful.
- Each note is one short third-person sentence grounded in what the
  viewer ACTUALLY said. Do not infer beliefs they didn't state.
- A SARCASTIC statement is NOT a real opinion. If the viewer says
  "great, I love dying repeatedly to one zombie" right after a death
  context line, do NOT record "loves dying to zombies." Skip it.
- Pure reactions to the stream content (kills, deaths, jump-scares, RNG,
  the streamer's plays) are not notes — "BASED" or "LMAO" alone tells
  us nothing about the viewer.
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
                # think=True: notes need to be cited correctly and survive
                # the hallucination guard. Extra latency is fine — this
                # runs off the message hot path.
                response = await self.llm.generate_structured(
                    prompt=prompt,
                    system_prompt=NOTE_EXTRACTION_SYSTEM,
                    response_model=NoteExtractionResponse,
                    think=True,
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
                # think=True for the same reason as the notes pass — a
                # mislabeled pronoun/location sticks around for sessions.
                profile = await self.llm.generate_structured(
                    prompt=prompt,
                    system_prompt=PROFILE_EXTRACTION_SYSTEM,
                    response_model=ProfileExtractionResponse,
                    think=True,
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

    # ---------------- end-of-stream recap ----------------
    # When OBS transitions from streaming → not-streaming, run a one-shot
    # LLM summary of the just-ended session: top topics, top engaged
    # chatters, things addressed, things still snoozed/missed. Saves to
    # `stream_recaps` for browsing on /insights.
    #
    # Lives behind an optional OBS handle — when None, the loop no-ops.

    RECAP_SYSTEM = """You write a short post-stream debrief for the streamer.

You are given the rough boundaries of a stream session and a list of the
most active topic threads + most active chatters during it. Produce a
recap they can scan in 30 seconds the next morning.

RULES:
- 5-8 short bullet points, plain text. No markdown headers.
- Lead with what was actually discussed (top topics).
- Mention 2-4 chatters who really engaged, by name.
- Flag anything the streamer asked to come back to that they didn't
  address (snoozed items still due) — only if the input includes them.
- No personality essays, no LLM-cheerleading. Just the debrief.
- This output renders to the streamer's private dashboard. Never to chat.
"""

    async def recap_loop(self, obs, on_stream_start=None) -> None:  # noqa: ANN001
        """Watch OBS state. On streaming → not-streaming transition, run
        a recap LLM call and persist the result. obs.status.streaming is
        the source of truth; we only fire when it goes True → False AND
        the OBS service is connected (so a disconnect doesn't fake an
        end-of-stream).

        Optional `on_stream_start` callback fires on the rising edge
        (not-streaming → streaming) so callers can hook session-reset
        behavior — e.g., wipe per-session insight state, clear the
        engaging-subjects blocklist, etc. Failures in the callback are
        logged but don't stop the loop.
        """
        if obs is None:
            return
        was_streaming = False
        stream_started_at: str | None = None
        first_msg_id: int | None = None
        from datetime import datetime, timezone
        while True:
            try:
                await asyncio.sleep(15)
                streaming = bool(obs.status.connected and obs.status.streaming)
                # Detect rising edge — stream went online.
                if streaming and not was_streaming:
                    stream_started_at = datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    )
                    first_msg_id = await asyncio.to_thread(
                        self.repo.latest_message_id
                    )
                    logger.info("recap_loop: stream started — anchor msg_id=%s", first_msg_id)
                    if on_stream_start is not None:
                        try:
                            res = on_stream_start()
                            if hasattr(res, "__await__"):
                                await res
                        except Exception:
                            logger.exception(
                                "recap_loop: on_stream_start callback raised",
                            )
                # Falling edge — stream went offline.
                if not streaming and was_streaming:
                    ended_at = datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    )
                    last_msg_id = await asyncio.to_thread(
                        self.repo.latest_message_id
                    )
                    logger.info(
                        "recap_loop: stream ended — generating recap for msgs %s-%s",
                        first_msg_id, last_msg_id,
                    )
                    try:
                        await self._generate_recap(
                            stream_started_at or ended_at,
                            ended_at,
                            first_msg_id,
                            last_msg_id,
                        )
                    except Exception:
                        logger.exception("recap_loop: recap generation failed")
                    stream_started_at = None
                    first_msg_id = None
                was_streaming = streaming
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("recap_loop iteration failed")

    async def _generate_recap(
        self,
        started_at: str,
        ended_at: str,
        message_id_lo: int | None,
        message_id_hi: int | None,
    ) -> None:
        """Build a prompt from session boundaries + recent threads + top
        chatters and call the LLM for the prose recap. Saves to
        stream_recaps."""
        # Top threads active during the session window.
        threads = await asyncio.to_thread(
            self.repo.list_threads, status_filter=None, query="", limit=10
        )
        # Top chatters during the session — reuse list_regulars over a
        # tight window. If start is unknown, fall back to last-2-hours.
        from datetime import datetime, timezone
        window = "-2 hours"
        try:
            if started_at:
                # Use the actual start as the boundary.
                window = started_at
        except Exception:
            pass
        try:
            top = await asyncio.to_thread(
                self.repo.list_regulars, since=window, limit=8,
            )
        except Exception:
            top = []
        # Snoozed-but-due items at end-of-stream — flag for the streamer.
        try:
            due = await asyncio.to_thread(self.repo.list_due_snoozes)
        except Exception:
            due = []

        thread_lines = [
            f"- {t.title} ({t.member_count}× seen, last {t.last_ts})"
            for t in threads[:8]
        ] or ["- (no threads tracked)"]
        chatter_lines = [
            f"- {r.user.name}: {r.msg_count} msgs"
            for r in top
        ] or ["- (no top chatters tracked)"]
        due_lines = [
            f"- snoozed ({d.kind} #{d.item_key}) — was due {d.due_ts}"
            for d in due[:5]
        ]

        prompt = (
            f"Stream session ended at {ended_at} (started ~{started_at}).\n\n"
            f"Top topic threads during the session:\n"
            + "\n".join(thread_lines)
            + "\n\nMost active chatters in that window:\n"
            + "\n".join(chatter_lines)
            + ("\n\nSnoozed items still due at session end:\n"
               + "\n".join(due_lines) if due_lines else "")
            + "\n\nWrite the streamer's debrief now."
        )
        text = ""
        try:
            # think=True — post-stream debrief is the slowest, most
            # accuracy-critical generation in the project. It runs once
            # per stream and the streamer reads it carefully.
            async for chunk in self.llm.stream_generate(
                prompt=prompt, system_prompt=self.RECAP_SYSTEM, think=True,
            ):
                text += chunk
        except Exception:
            logger.exception("recap_loop: LLM stream failed")
            return
        text = (text or "").strip()
        if not text:
            logger.info("recap_loop: LLM returned empty recap, skipping save")
            return

        # Compute KPI columns for the cross-stream delta strip.
        msg_count = unique_chatters = new_chatters = 0
        addressed_count = snoozed_count = 0
        try:
            with self.repo._cursor() as cur:  # noqa: SLF001
                if message_id_lo is not None and message_id_hi is not None:
                    cur.execute(
                        "SELECT COUNT(*), COUNT(DISTINCT user_id) "
                        "FROM messages WHERE id BETWEEN ? AND ? "
                        "AND is_emote_only = 0 AND spam_score < 0.5",
                        (message_id_lo, message_id_hi),
                    )
                    row = cur.fetchone()
                    msg_count = int(row[0] or 0)
                    unique_chatters = int(row[1] or 0)
                    cur.execute(
                        "SELECT COUNT(DISTINCT u.twitch_id) "
                        "FROM users u WHERE u.first_seen >= ? AND u.first_seen <= ?",
                        (started_at, ended_at),
                    )
                    new_chatters = int(cur.fetchone()[0] or 0)
            addressed_count = await asyncio.to_thread(
                self.repo.count_state_changes_since, started_at, state="addressed"
            )
            snoozed_count = await asyncio.to_thread(
                self.repo.count_state_changes_since, started_at, state="snoozed"
            )
        except Exception:
            logger.exception("recap_loop: stat collection failed")

        # Persist with stats, then clear ephemeral stream goals — they
        # were scoped to the just-ended session.
        with self.repo._cursor() as cur:  # noqa: SLF001
            cur.execute(
                """
                INSERT INTO stream_recaps(
                    started_at, ended_at, message_id_lo, message_id_hi, summary,
                    msg_count, unique_chatters, new_chatters,
                    addressed_count, snoozed_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    started_at, ended_at, message_id_lo, message_id_hi, text,
                    msg_count, unique_chatters, new_chatters,
                    addressed_count, snoozed_count,
                ),
            )
        try:
            await asyncio.to_thread(self.repo.clear_stream_goals)
        except Exception:
            logger.exception("recap_loop: clear_stream_goals failed")
        logger.info(
            "recap_loop: saved stream recap (%d chars, msgs=%d, addr=%d, snz=%d)",
            len(text), msg_count, addressed_count, snoozed_count,
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
                flooded = 0
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
                    # Copy-paste brigade detection — piggybacks on the
                    # embedding pass we already do. If 4+ OTHER users
                    # said something cosine-near-identical in the last
                    # 60 s, bump everyone in the cluster (focal + the
                    # near-dups). Threshold 0.85 keeps the cluster
                    # firmly above SPAM_THRESHOLD_DEFAULT (0.5) so all
                    # consumers filter it out.
                    try:
                        cluster = await asyncio.to_thread(
                            self.repo.find_near_duplicate_flood,
                            m.id, vec,
                        )
                        # Need at least 4 distinct *other* users (5 with the
                        # focal). Same user copy-pasting themselves is annoying
                        # but not brigading; we let the per-message detector
                        # handle that.
                        other_users = {uid for mid, uid in cluster if mid != m.id}
                        if len(other_users) >= 4:
                            ids = [mid for mid, _ in cluster]
                            bumped = await asyncio.to_thread(
                                self.repo.bump_spam_score,
                                ids, score=0.85, reason="near_dup_flood",
                            )
                            flooded += bumped
                            logger.info(
                                "embed_loop: near-dup flood detected — "
                                "bumped %d msgs (focal=%d, other_users=%d)",
                                bumped, m.id, len(other_users),
                            )
                    except Exception:
                        logger.exception(
                            "embed_loop: flood-check failed for msg %d", m.id,
                        )
                if wrote:
                    indexed, total = await asyncio.to_thread(
                        self.repo.messages_embedding_coverage
                    )
                    logger.info(
                        "embed_loop: +%d → %d/%d indexed (%.1f%%, flood-bumped=%d)",
                        wrote, indexed, total,
                        100 * indexed / total if total else 0.0,
                        flooded,
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
        last_processed_id = 0
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                # Skip the LLM call entirely when no new messages have
                # arrived since our last snapshot — same window in, same
                # topics out, just wasted inference. The bot wakes the
                # next iteration on schedule and re-checks.
                latest = await asyncio.to_thread(self.repo.latest_message_id)
                if latest <= last_processed_id:
                    continue
                await self._take_topic_snapshot(max_msgs)
                last_processed_id = latest
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
            # think=True — channel-wide topic snapshots feed into thread
            # clustering, so getting the labels right matters more than
            # finishing fast.
            response = await self.llm.generate_structured(
                prompt=f"Recent chat (oldest first):\n{formatted}",
                system_prompt=TOPICS_SYSTEM,
                response_model=TopicsResponse,
                think=True,
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
