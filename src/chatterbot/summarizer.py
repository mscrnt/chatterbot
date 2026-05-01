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

CHANNEL CONTEXT (game / title / tags) and STREAMER VOICE (last ~60s of
audio) may appear at the top of the prompt. Use them to:
  - Recognise in-game terms so they don't get extracted as interests.
    "Lighthouse" while Tarkov is the current game is a map, not the
    chatter's interest.
  - Filter out reactions: when STREAMER VOICE shows the streamer just
    said something and the chatter's message is a one-line agreement
    or short reply, that's a REACTION not their independent stance.
    Don't infer demeanor from reaction lines alone.

==================================================================
EXAMPLES
==================================================================

EXAMPLE 1 — GOOD (explicit pronoun + location, demeanor inferred)

Focal lines:
  [5101] ngl as a girl in chat she/her, this banter is wild
  [5102] from chicago it's been a long day
  [5103] love you all chat keep sending the support W

Good output:
  pronouns: "she/her"
  location: "Chicago"
  demeanor: "supportive"
  interests: []

Why good: pronouns + location are EXPLICIT (not inferred from
username). 'supportive' fits the encouraging tone at [5103].


EXAMPLE 2 — BAD (inferred from nothing, fabricated interests)

Focal lines:
  [5110] gn chat, sleepy
  [5111] o7

Bad output (DO NOT DO):
  pronouns: "they/them"            # username 'samuraicat42' is NOT a signal
  location: "Japan"                 # 'samurai' is NOT a location signal
  demeanor: "chill"                 # 2 messages is too thin
  interests: ["samurai", "cats"]   # username inferences, not stated

Good output:
  pronouns: null
  location: null
  demeanor: "quiet"   # short, infrequent, mostly reactive — fits
  interests: []


EXAMPLE 3 — GOOD (in-game term filtered from interests)

CHANNEL CONTEXT: streamer: xQc · playing: Escape from Tarkov
Focal lines:
  [5120] lighthouse is so unbalanced wtf
  [5121] been grinding this map all week, kappa is a meme

Good output:
  interests: ["tarkov"]   # the GAME is a real signal of interest
  demeanor: "analytical"

Why good: 'lighthouse' and 'kappa' are Tarkov-specific terms
(known from CHANNEL CONTEXT), so they don't show up as separate
interests. The chatter's clear engagement with Tarkov DOES count.
""")


NOTE_EXTRACTION_SYSTEM = (_AUDIENCE_BLOCK + "\n" + _INPUT_FORMAT_BLOCK + """
TASK: extract short third-person notes about ONE chat viewer (the focal
user) from their own messages.

For each note, include `source_message_ids` — the specific focal-line ids
that support that note. The streamer uses this to trace any note back to
the exact line(s) it came from. Context lines (`[ctx id]`) are NOT
allowed in source_message_ids.

CHANNEL CONTEXT may appear at the top of the prompt. It tells you what
the streamer is currently playing / streaming. Use it to recognise
in-game terms (map names, gear, mechanics) so they don't get extracted
as if they were the chatter's hobbies. If the chatter says "Lighthouse
is brutal" while the streamer is playing Tarkov, that's a comment on
the Tarkov map — NOT a fact about a place called Lighthouse. BUT — if
the chatter clearly references something OTHER than the current game
(yesterday's stream, an unrelated topic), don't force their words into
the current game.

STREAMER VOICE may also appear — a recap of the streamer's last ~60s
of audio. Chatter messages often REACT to what the streamer just said.
A chatter saying "yeah that boss is rough" right after the streamer
complained about the boss is a reaction, NOT the chatter's stated
opinion about the boss. Skip pure reactions; only extract when the
chatter brings their OWN content to the table.

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

==================================================================
EXAMPLES — study these, they show the difference between a useful
note and a hallucinated / context-blind one.
==================================================================

EXAMPLE 1 — GOOD (hard self-disclosure, single line)

CHANNEL CONTEXT: streamer: xQc · playing: Escape from Tarkov
Focal lines:
  [4501] just got my third kit wiped on Lighthouse this morning lol
  [4502] timezone is brutal tho, 3am here in Berlin

Good output:
  - "Lives in Berlin (3am their time)." [src: 4502]
  - "Plays Tarkov; got wiped on Lighthouse this morning." [src: 4501]

Why good: 'Berlin' is a self-disclosed location, not a Tarkov term.
'Lighthouse' is recognised (from CHANNEL CONTEXT) as a Tarkov map
so the note frames it as gameplay, not a place.


EXAMPLE 2 — BAD (hallucinated from context, sarcasm misread)

Focal lines:
  [4510] (replying to streamer: "this run is going great") oh yeah, GREAT run, 4 deaths in 5 min
  [4511] LULW

Bad output (DO NOT DO):
  - "Loves the streamer's run." [src: 4510]   # SARCASM — opposite meaning
  - "Reacts with LULW." [src: 4511]            # pure reaction — useless

Good output: { "notes": [] }   # No real signal here.


EXAMPLE 3 — GOOD (reaction to STREAMER VOICE filtered out)

STREAMER VOICE: The streamer is complaining about how clunky the new patch feels.
Focal lines:
  [4520] yeah the patch is rough
  [4521] ng4 felt better tbh, more responsive

Good output:
  - "Thinks NG4 felt more responsive than the current game." [src: 4521]

Why good: [4520] is a pure reaction to what the streamer just said
(skipped). [4521] is the chatter bringing their OWN comparison —
that's their take, worth a note.


EXAMPLE 4 — GOOD (recurring reference, callback potential)

Focal lines:
  [4530] is xqc gonna replay the missed parry clip again? 😂
  [4531] he was streaming this same boss yesterday and choked

Good output:
  - "Was here yesterday for the same boss; remembers the choke." [src: 4531]
  - "Refers to a 'missed parry clip' as a recurring stream meme." [src: 4530]

Why good: both are concrete recurring references the streamer can call back.
""")


TOPICS_SYSTEM = """You summarize what a Twitch chat is currently talking about.

CHANNEL CONTEXT may appear at the top of the prompt. Use it to:
  - Recognise in-game terms (map names, gear) so a topic about a Tarkov
    map gets categorised as `gaming` (not `personal` because a chatter
    name-dropped a place).
  - Disambiguate jargon — "kappa" in Tarkov is loot, in chat it's an
    emote. Read context.

DEFINITION OF A TOPIC:
- A SUBJECT that 2+ chatters have substantively engaged with. One
  person mentioning something once is NOT a topic.
- Specific, not vague. "Tarkov Lighthouse map balance" is a topic;
  "video games" is not.
- 4-10 word topic line, no fluff.

RULES:
- Identify 3 to 5 main topics from the messages provided.
- One short line per topic.
- Cite which usernames are driving each topic (people who actually
  said something on the topic — not just present in chat).
- For each topic, pick ONE category that fits best:
    gaming    — game mechanics, runs, builds, in-game events
    personal  — life updates, pets, family, work, location
    meta      — stream meta, schedule, gear, OBS, broadcast tech
    tech      — hardware, software, programming
    off-topic — jokes, banter, memes, off-the-wall
    other     — everything else
- No editorializing. No inferred sentiment. Just topic + drivers + category.
- If chat is essentially silent or unfocused, return fewer topics or an empty list.

==================================================================
EXAMPLES
==================================================================

EXAMPLE 1 — GOOD (specific, grounded, properly categorised)

CHANNEL CONTEXT: streamer: xQc · playing: Escape from Tarkov
Messages:
  alice: lighthouse is so unbalanced right now
  bob: yeah aim-down-sights speed is rough on that map
  alice: meanwhile rogue spawns are still busted
  carol: anyone else having stuttering issues since last patch?
  dave: yeah my fps tanks in interchange too

Good output:
  topics:
    - { topic: "Tarkov Lighthouse balance + rogue spawns",
        drivers: ["alice", "bob"], category: "gaming" }
    - { topic: "Post-patch performance / FPS issues",
        drivers: ["carol", "dave"], category: "tech" }


EXAMPLE 2 — BAD (vague, single-person, mis-categorised) — DO NOT DO

Same input, BAD output:
  topics:
    - { topic: "video games",                    # too vague
        drivers: ["alice", "bob", "carol", "dave"],
        category: "off-topic" }                  # wrong category
    - { topic: "Lighthouse",                     # treated as a place
        drivers: ["alice"], category: "personal" }  # one driver, wrong cat

Why bad: 'video games' has no specificity; 'Lighthouse' is a Tarkov
map (CHANNEL CONTEXT made that obvious) so it should be `gaming`,
not `personal`; one driver isn't a topic.


EXAMPLE 3 — GOOD (correct empty list)

Messages:
  alice: hi
  bob: lol
  carol: pog
  dave: !lurk

Good output: { topics: [] }   # pure greetings + bot commands, no topics.
"""


class Summarizer:
    def __init__(
        self, repo: ChatterRepo, llm: OllamaClient, settings: Settings,
        *, twitch_status=None,
    ):
        self.repo = repo
        self.llm = llm
        self.settings = settings
        # Optional[TwitchService] — when wired, every "thinking" call
        # in this module (notes, profile, topic snapshot, recap)
        # prefixes its prompt with the live Helix snapshot (game,
        # title, tags, viewer tier, uptime, streamer name). Same
        # treatment the transcript group-summary call already gets,
        # so notes for a Tarkov stream know "Lighthouse map" is a
        # game reference rather than a chatter's home town.
        self.twitch_status = twitch_status
        self._user_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._inflight: set[str] = set()
        # Topic threading runs as a side-effect of each new snapshot. Keep a
        # single shared instance so backfill state could live here later.
        self._threader = Threader(repo, llm, settings)

    def reconfigure(self, settings) -> None:
        """Pick up a fresh Settings snapshot from the lifecycle
        poller's reload pass. Propagates to the nested Threader so
        threading-related settings (centroid threshold, recap cadence,
        etc.) update too — without this propagation the child would
        keep using the boot-time Settings reference even after the
        parent swapped."""
        self.settings = settings
        # Threader reads tuning via `getattr(self.settings, ...)` at
        # use time, so a reference swap is enough.
        self._threader.settings = settings

    def _channel_context_block(self, *, authoritative: bool = False) -> str:
        """Render the live Helix snapshot as a CHANNEL CONTEXT
        preamble. Returns "" when twitch_status isn't wired or the
        streamer is offline so callers can concat unconditionally.

        `authoritative=False` for text-only calls (notes, profile,
        topic snapshot, recap). Reserve `authoritative=True` for
        calls that ALSO ship a screenshot — that flag adds the
        "screenshot may LOOK different — IGNORE THAT" framing,
        which doesn't apply when there's no image."""
        if self.twitch_status is None:
            return ""
        ts = getattr(self.twitch_status, "status", None)
        if ts is None:
            return ""
        try:
            return ts.format_for_llm(authoritative=authoritative)
        except AttributeError:
            return ""

    async def _latest_transcript_summary(self) -> str:
        """Most recent transcript group summary, formatted as a
        STREAMER VOICE preamble. Empty when whisper is disabled / no
        groups yet, so callers can concat unconditionally.

        Used by notes + profile so the LLM grounds chatter messages
        against what the streamer JUST said out loud — chatters often
        respond to the streamer's audio in ways that look like
        opinions but are really reactions."""
        try:
            groups = await asyncio.to_thread(
                self.repo.list_transcript_groups, limit=1,
            )
        except Exception:
            return ""
        if not groups or not groups[0].summary:
            return ""
        return (
            "STREAMER VOICE (most recent ~60s of audio summarised — "
            "what the streamer just said out loud; chatter messages "
            "may be reactions to this, NOT independent opinions):\n  "
            + groups[0].summary.strip()
            + "\n\n"
        )

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
            # Informed-call enrichment: channel context + streamer
            # voice. Same blocks the transcript group-summary call
            # gets — lets the LLM ground chatter words against what's
            # actually being streamed (Lighthouse is a Tarkov map,
            # not a place name) and tell reactions from independent
            # opinions (chatter agreeing with what the streamer just
            # said is a reaction, not a take).
            channel_context = self._channel_context_block(authoritative=False)
            transcript_block = await self._latest_transcript_summary()
            prompt = (
                channel_context
                + transcript_block
                + f"Focal viewer username: {display_name}\n\n"
                + "Chat transcript (focal lines = `[id]`, context lines = "
                + f"`[ctx id] otherUser:`):\n{corpus}"
            )

            try:
                # think=True + INFORMED_NUM_CTX: notes need to be
                # cited correctly and survive the hallucination
                # guard. Extra latency is fine — this runs off the
                # message hot path. 32k ctx headroom for the new
                # channel-context + streamer-voice preamble.
                from .insights import INFORMED_NUM_CTX
                response = await self.llm.generate_structured(
                    prompt=prompt,
                    system_prompt=NOTE_EXTRACTION_SYSTEM,
                    response_model=NoteExtractionResponse,
                    num_ctx=INFORMED_NUM_CTX,
                    think=True,
                    call_site="summarizer.note_extraction",
                    # Per-user pass — the focal chatter is the only
                    # user whose chat content is in this prompt
                    # by definition. Declaring them lets the dataset
                    # capture pipeline drop the event if they ever
                    # opt out (defense in depth on top of the
                    # is_opted_out gate above this call).
                    referenced_user_ids=[user_id],
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
                # think=True + INFORMED_NUM_CTX — same reasoning as
                # the notes pass: a mislabeled pronoun/location
                # sticks around for sessions, so accuracy beats
                # latency. The prompt already has channel context +
                # streamer voice from the notes pass above.
                from .insights import INFORMED_NUM_CTX
                profile = await self.llm.generate_structured(
                    prompt=prompt,
                    system_prompt=PROFILE_EXTRACTION_SYSTEM,
                    response_model=ProfileExtractionResponse,
                    num_ctx=INFORMED_NUM_CTX,
                    think=True,
                    call_site="summarizer.profile_extraction",
                    # Same per-user reasoning as note_extraction above.
                    referenced_user_ids=[user_id],
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

You are given the rough boundaries of a stream session, optional
CHANNEL CONTEXT (game / title / streamer name from Twitch's API), the
most active topic threads, the most active chatters, and any snoozed
items the streamer didn't get back to. Produce a recap they can scan
in 30 seconds the next morning.

CHANNEL CONTEXT, when present, names the streamer + what they were
playing. Use that name in your bullets ("xQc's Tarkov stream") rather
than the generic "the stream" or "the streamer". If absent, fall back
to "the stream".

RULES:
- 5-8 short bullet points, plain text. No markdown headers.
- Lead with the game / scope ("Tarkov stream, ~4h, 220 viewers peak")
  if CHANNEL CONTEXT gives you the data; otherwise just open with
  the top topic.
- Mention 2-4 chatters who really engaged, by name.
- Flag anything the streamer asked to come back to that they didn't
  address (snoozed items still due) — only if the input includes them.
- No personality essays, no LLM-cheerleading, no "great stream!"
  filler. Just the debrief.
- This output renders to the streamer's private dashboard. Never to chat.

==================================================================
EXAMPLE
==================================================================

CHANNEL CONTEXT: streamer: xQc · playing: Escape from Tarkov · live for 4h 12m
Threads: Lighthouse balance (24×), Tarkov post-patch perf (11×), …
Active chatters: alice 47 msgs, bob 31 msgs, carol 18 msgs
Snoozed (still due): "look at moonpie's clip suggestion"

Good output:
- xQc's Tarkov stream, ~4h. Lighthouse balance was the dominant
  thread — alice and bob carried it for a solid hour.
- Post-patch performance came up multiple times (carol led that
  thread). Worth pinning if next stream still has stutter.
- alice opened with a long Lighthouse rant, bob countered with the
  rogue-spawn complaint. Both regulars to call back.
- Snoozed but never addressed: moonpie's clip suggestion. First
  thing to check next stream.

Bad output (DO NOT DO):
- Great stream xQc! 🎉                       # cheerleading
- Lots of energy in chat today!              # vague essay
- Several people were talking about things.  # no specifics
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

        # Live channel context — game / title / streamer name. Useful
        # for the recap so it can lead with "xQc's Tarkov stream"
        # rather than the generic "the stream". The Helix poll may
        # have already flipped to is_live=False by the time the recap
        # fires (stream just ended), but cached broadcaster_login /
        # display_name still flow through, and game_name from the
        # last live poll is a reasonable proxy for "what they
        # played" — we accept that it'll be stale if they switched
        # games near the end.
        channel_context = self._channel_context_block(authoritative=False)
        prompt = (
            channel_context
            + f"Stream session ended at {ended_at} (started ~{started_at}).\n\n"
            + f"Top topic threads during the session:\n"
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

        # Channel context (game / title / tags / viewer tier / uptime)
        # from the live Helix poll, when wired. Empty string when the
        # bot starts without a TwitchService — caller can concat
        # unconditionally. authoritative=False since topic snapshots
        # don't ride with a screenshot (text-only call).
        channel_context = ""
        if self.twitch_status is not None:
            ts = getattr(self.twitch_status, "status", None)
            if ts is not None:
                try:
                    channel_context = ts.format_for_llm(authoritative=False)
                except AttributeError:
                    channel_context = ""

        try:
            # think=True + INFORMED_NUM_CTX — channel-wide topic
            # snapshots feed into thread clustering, so getting the
            # labels right matters more than finishing fast. The
            # 32k floor lets a hot chat fit the whole `recent_messages_for_topics`
            # window without truncating.
            from .insights import INFORMED_NUM_CTX
            from .llm.prompts import resolve_prompt
            response = await self.llm.generate_structured(
                prompt=(
                    channel_context
                    + f"Recent chat (oldest first):\n{formatted}"
                ),
                system_prompt=resolve_prompt(
                    "summarizer.topics_snapshot", self.repo,
                ),
                response_model=TopicsResponse,
                num_ctx=INFORMED_NUM_CTX,
                think=True,
                call_site="summarizer.topics_snapshot",
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
