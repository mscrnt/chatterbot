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
import uuid
from dataclasses import dataclass, field

from pydantic import ValidationError

from .config import Settings
from .llm.ollama_client import OllamaClient
from .llm.schemas import TalkingPointsResponse
from .repo import ChatterRepo

logger = logging.getLogger(__name__)


# Shared `num_ctx` floor for the four "informed" streamer-facing LLM calls:
# talking points, engaging subjects, thread recaps, and topic snapshots.
# These calls assemble the data the streamer reads on the dashboard, so
# they prioritise accuracy over latency — wide context (game, transcript,
# notes, chat) plus think=True. 32k handles a hot chat plus all the
# context blocks without truncation; Qwen 3.5 / Llama 3.x families all
# accept it. The Ollama think-mode floor is 16k; we double it here.
INFORMED_NUM_CTX = 32768


@dataclass
class _PersistentCluster:
    """One persistent engaging-subject cluster carried across refreshes
    of the extractor. Centroid is the running unit-vector mean of all
    messages ever attached to this cluster; new messages attach by
    cosine similarity above threshold rather than re-clustering from
    scratch each pass.

    Stable identity (cluster_id) means the LLM-assigned name/brief
    sticks across passes — we only relabel when the cluster has grown
    substantially or has no name yet. That fixes the "subjects rename
    every refresh" problem the streamer was seeing."""
    cluster_id: str
    centroid: list[float]   # current unit-vector centroid
    n: int                   # total messages ever attached
    name: str = ""
    brief: str = ""
    angles: list[str] = field(default_factory=list)
    is_sensitive: bool = False
    n_at_last_label: int = 0  # cluster size when the LLM last labeled
    last_added_ts: float = 0.0  # unix ts of most-recent attached msg
    msg_ids: list[int] = field(default_factory=list)  # capped list of recent attached msg ids


TALKING_POINTS_SYSTEM = """You help a Twitch streamer remember conversation hooks for chatters who are active in their chat right now. The streamer reads your output on a second monitor while streaming and uses it to engage with chat — your output never returns to chat itself.

You'll get a numbered list of active chatters, each with:
  - their username
  - extracted notes (facts they've previously stated about themselves)
  - a few of their most recent messages

You may also receive optional context blocks (use as silent grounding,
do not parrot back to the streamer):
  - CHANNEL CONTEXT — what the streamer is playing / streaming right now
  - STREAMER VOICE — a recent transcript of what the streamer just said
  - Current channel-wide topics — recent chat topics
  - An attached screenshot — what's on screen right now

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


@dataclass
class EngagingSubjectEntry:
    """One curated subject the LLM extracted from recent chat. Distinct
    from `topic_threads` (which cluster snapshots by embedding cosine):
    these come from a direct subject-extraction pass over messages, so
    the unit is "actual conversation subject" rather than "embedding-
    similar text".

    `brief` + `angles` ride along so the row can expand without another
    LLM call. `slug` is a deterministic short id derived from the name,
    used by the on-demand /insights/subject-messages route to identify
    a subject without leaking cache indices."""
    name: str
    drivers: list[str]
    msg_count: int
    brief: str = ""
    angles: list[str] = field(default_factory=list)
    slug: str = ""


@dataclass
class EngagingSubjectsCache:
    subjects: list[EngagingSubjectEntry]
    refreshed_at: float | None
    error: str | None = None


class InsightsService:
    """Caches the LLM-derived talking points; refreshed by a background task."""

    DEFAULT_REFRESH_SECONDS = 180  # 3 min
    ACTIVE_WINDOW_MINUTES = 10
    # Bumped from 6 — talking points is the highest-leverage call on
    # the dashboard, and INFORMED_NUM_CTX gives us the headroom to
    # carry richer per-chatter recent-message context without
    # truncating the prompt.
    RECENT_MESSAGES_PER_USER = 12
    # Active-chatter cap fed to the LLM. Bumped from 20 to 25 so a
    # busier chat still hits the talking-points pass without dropping
    # the long tail.
    ACTIVE_CHATTER_CAP = 25

    def __init__(
        self, repo: ChatterRepo, llm: OllamaClient, settings: Settings,
        *, twitch_status=None,  # Optional[TwitchService] — for channel context
    ):
        self.repo = repo
        self.llm = llm
        self.settings = settings
        # Used by the engaging-subjects extractor to inject "streamer
        # is currently playing X" into the prompt so the LLM can
        # disambiguate game-specific jargon. None when the dashboard
        # boots without TwitchService (e.g. no oauth token configured).
        self.twitch_status = twitch_status
        self._cache = InsightsCache(talking_points=[], refreshed_at=None, error=None)
        self._subjects_cache = EngagingSubjectsCache(
            subjects=[], refreshed_at=None, error=None,
        )
        self._lock = asyncio.Lock()
        self._subjects_lock = asyncio.Lock()
        # Persistent engaging-subject clusters carried across refreshes.
        # New messages append to existing clusters when the cosine
        # similarity to the centroid clears the threshold; only orphans
        # spawn new clusters. Names stick across passes unless the
        # cluster has substantially grown — keeps the dashboard from
        # renaming subjects every 3 min.
        self._clusters: dict[str, _PersistentCluster] = {}
        # msg_id -> cluster_id, so a window pull can render historical
        # cluster membership without re-running the cosine match.
        self._msg_to_cluster: dict[int, str] = {}
        # In-memory embedding cache for blocklisted subject names —
        # populated lazily during _refresh_engaging_subjects so new
        # subject names can be cosine-matched against rejected
        # historical names. Catches near-dupes ("FF8 remake" vs "Final
        # Fantasy 8 remake") that the literal slug/name blocklist
        # would miss. Keyed by slug so it stays in sync with the
        # blocklist's own dedupe key.
        self._blocklist_embed_cache: dict[str, list[float]] = {}

    @property
    def cache(self) -> InsightsCache:
        return self._cache

    @property
    def subjects_cache(self) -> EngagingSubjectsCache:
        return self._subjects_cache

    async def _latest_screenshot_grid(
        self, window_minutes: int = 10,
    ) -> str | None:
        """Build a base64 2x2 grid of the most recent OBS screenshots so
        the multimodal LLM has visual game context for talking-points,
        engaging-subjects, and thread-recap calls.

        Returns None when:
          - screenshot capture is disabled (interval = 0)
          - no shots in the last `window_minutes`
          - Pillow isn't installed / stitching fails

        Vision-incapable models silently ignore `images=`; the cost is
        only the wasted base64 payload, not a hard error. We still gate
        on `screenshot_interval_seconds` so non-whisper installs don't
        pay it at all."""
        try:
            interval = int(getattr(
                self.settings, "screenshot_interval_seconds", 0,
            ))
        except (TypeError, ValueError):
            interval = 0
        if interval <= 0:
            return None
        try:
            from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        except ImportError:
            return None
        end = _dt.now(_tz.utc)
        start = end - _td(minutes=max(1, int(window_minutes)))
        try:
            shots = await asyncio.to_thread(
                self.repo.screenshots_in_range,
                start.isoformat(timespec="seconds"),
                end.isoformat(timespec="seconds"),
                max_count=int(getattr(self.settings, "screenshot_grid_max", 4)),
            )
        except Exception:
            logger.exception("informed-call: screenshots_in_range failed")
            return None
        if not shots:
            return None
        from pathlib import Path
        data_dir = Path(self.settings.db_path).parent
        abs_paths = [str(data_dir / s.path) for s in shots]
        try:
            # Reuse the transcript service's stitcher — same 960x540
            # JPEG output the group-summary call uses, so the LLM sees
            # screenshots in a consistent format across all calls.
            from .transcript import _stitch_grid
            grid_bytes = await asyncio.to_thread(_stitch_grid, abs_paths)
        except Exception:
            logger.exception("informed-call: stitch_grid failed")
            return None
        if not grid_bytes:
            return None
        import base64 as _b64
        return _b64.b64encode(grid_bytes).decode("ascii")

    async def _latest_transcript_summary(self) -> str:
        """Most recent transcript group summary, formatted as a prompt
        block. Empty string when whisper is disabled / no groups yet,
        so callers can concat unconditionally."""
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
            "what the streamer just said out loud; chat may be "
            "reacting to this):\n  "
            + groups[0].summary.strip()
            + "\n\n"
        )

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
                self.ACTIVE_CHATTER_CAP,
            )
            if not active:
                self._cache = InsightsCache(
                    talking_points=[], refreshed_at=time.time(), error=None
                )
                return

            # Batch-fetch notes + recent messages for every active
            # chatter in two queries instead of 2N. Previously each
            # of ~20 chatters triggered two round-trips to the DB on
            # every 3-min refresh — collapsed here using the
            # canonical batch helpers in repo.py.
            user_ids = [u.twitch_id for u in active]
            notes_by_user, msgs_by_user = await asyncio.gather(
                asyncio.to_thread(self.repo.get_notes_for_users, user_ids),
                asyncio.to_thread(
                    self.repo.get_recent_messages_for_users,
                    user_ids,
                    per_user_limit=self.RECENT_MESSAGES_PER_USER,
                ),
            )

            # Build a numbered prompt block.
            blocks: list[str] = []
            id_by_index: dict[int, tuple[str, str]] = {}
            for i, user in enumerate(active, start=1):
                notes = notes_by_user.get(user.twitch_id, [])
                msgs = msgs_by_user.get(user.twitch_id, [])
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

            # Informed-call enrichment: what's the streamer playing,
            # what did they just say, and what's on screen. All three
            # are best-effort and skip silently when unavailable.
            # authoritative=True since we ALSO pass a screenshot — keeps
            # the LLM from second-guessing the Helix-supplied game name.
            channel_context = self._build_channel_context(authoritative=True)
            transcript_block = await self._latest_transcript_summary()
            grid_b64 = await self._latest_screenshot_grid(
                window_minutes=self.ACTIVE_WINDOW_MINUTES,
            )
            image_note = (
                "\n\nAn image is attached showing what is currently on "
                "screen. Use it ONLY as silent context (identify the "
                "game/app, ground vague chatter references). Do NOT "
                "describe the image or mention that it's attached."
            ) if grid_b64 else ""

            prompt = (
                channel_context
                + transcript_block
                + topic_block
                + f"Active chatters (last {self.ACTIVE_WINDOW_MINUTES} minutes):\n\n"
                + "\n\n".join(blocks)
                + image_note
            )

            try:
                # think=True + INFORMED_NUM_CTX + image — talking points
                # is the single most-read line on the dashboard while
                # the streamer is live, so we spend the latency budget
                # for accuracy. Cadence (~3 min) absorbs the cost.
                response = await self.llm.generate_structured(
                    prompt=prompt,
                    system_prompt=TALKING_POINTS_SYSTEM,
                    response_model=TalkingPointsResponse,
                    num_ctx=INFORMED_NUM_CTX,
                    images=[grid_b64] if grid_b64 else None,
                    think=True,
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

    # =========================================================
    # Thread recap loop — observational summaries for the Live
    # Conversations panel. Replaces per-chatter talking-points
    # as the primary "what's happening" surface.
    # =========================================================

    THREAD_RECAP_NUM_CTX = INFORMED_NUM_CTX
    THREAD_RECAP_TRUNCATE_PER_MSG = 200

    THREAD_RECAP_SYSTEM = """You're labeling clustered chat conversations on a Twitch streamer's dashboard.

For each numbered thread below, write a 1-2 sentence OBSERVATIONAL recap of what the chatters are actually discussing. The streamer reads these on a second monitor — they are NOT suggestions for what to talk about, they are descriptions of what's already in chat.

HARD RULES:
- Stay grounded: only paraphrase content that appears in the messages. Don't bridge, infer, or extrapolate.
- Never assert facts beyond what you can quote. If a chatter said "excited for the FF8 remake," do NOT write "an FF8 remake is coming" — write "chatters are talking about a possible FF8 remake."
- Never tell the streamer to do anything. No "ask them about X", no "you should mention Y." Pure description.
- If a thread's messages are too noisy / unrelated to summarise without inventing context, SKIP it (omit from your response).
- Reply with `thread_id` matching the integer key in the input and a 1-2 sentence `recap`.

Empty / partial replies are fine. Skipping is the right call when in doubt.
"""

    async def thread_recap_loop(
        self, interval_seconds: int | None = None,
    ) -> None:
        """Background task: periodically refresh recaps for active topic
        threads. No-op when interval is 0."""
        # Same booting delay as the talking-points refresh.
        await asyncio.sleep(15)
        while True:
            try:
                interval = max(60, int(
                    interval_seconds
                    or getattr(self.settings, "thread_recap_interval_seconds", 300)
                ))
                if interval <= 0:
                    return
                await self._refresh_thread_recaps()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("thread recap iteration failed")
            await asyncio.sleep(interval)

    async def _refresh_thread_recaps(self) -> None:
        """One pass: pull all active threads + their recent messages,
        send to LLM in one batch, write back recaps."""
        threads = await asyncio.to_thread(
            self.repo.list_threads,
            status_filter="active", query="", limit=20,
        )
        if not threads:
            return
        per_msg_cap = max(40, int(getattr(
            self.settings, "thread_recap_max_messages_per_thread", 30,
        )))

        # Build a numbered prompt — for each thread, include drivers +
        # recent messages clipped per-message so the LLM has signal but
        # we don't blow the context budget on a single hot thread.
        blocks: list[str] = []
        for t in threads:
            try:
                msgs = await asyncio.to_thread(
                    self.repo.get_thread_messages, t.id, per_msg_cap,
                )
            except Exception:
                logger.exception(
                    "thread-recap: get_thread_messages failed for %d", t.id,
                )
                continue
            if not msgs:
                continue
            user_id_to_name: dict[str, str] = {}
            for m in msgs:
                if m.user_id and m.user_id not in user_id_to_name:
                    u = await asyncio.to_thread(self.repo.get_user, m.user_id)
                    if u:
                        user_id_to_name[m.user_id] = u.name
            msg_lines = "\n      ".join(
                f"[{user_id_to_name.get(m.user_id, m.user_id)}] "
                f"{m.content[:self.THREAD_RECAP_TRUNCATE_PER_MSG]}"
                for m in msgs
            )
            drivers_str = ", ".join(t.drivers) if t.drivers else "—"
            blocks.append(
                f"THREAD {t.id} (current title: {t.title!r})\n"
                f"  drivers: {drivers_str}\n"
                f"  messages:\n      {msg_lines}"
            )
        if not blocks:
            return

        from .llm.schemas import ThreadRecapsResponse
        # Informed-call enrichment — channel context + screenshot give
        # the recap LLM the same grounding signals talking-points and
        # engaging-subjects already get.
        channel_context = self._build_channel_context(authoritative=True)
        grid_b64 = await self._latest_screenshot_grid(window_minutes=15)
        image_note = (
            "\n\nAn image is attached showing what is currently on "
            "screen. Use it ONLY as silent context to ground vague "
            "thread topics. Do NOT describe the image or mention that "
            "it's attached."
        ) if grid_b64 else ""
        prompt = (
            channel_context
            + "Active topic threads (numbered by `thread_id`):\n\n"
            + "\n\n".join(blocks)
            + image_note
            + "\n\nReturn one observational `recap` per thread you can "
            "ground in its messages. Skip noisy / unsummarisable ones."
        )
        try:
            # think=True — recaps are cached and persist; getting the
            # observational paraphrase right matters more than finishing
            # in seconds. ~5 min cadence absorbs the latency.
            response = await self.llm.generate_structured(
                prompt=prompt,
                system_prompt=self.THREAD_RECAP_SYSTEM,
                response_model=ThreadRecapsResponse,
                num_ctx=self.THREAD_RECAP_NUM_CTX,
                images=[grid_b64] if grid_b64 else None,
                think=True,
            )
        except ValidationError as e:
            logger.warning("thread-recap validation failed: %s", e)
            return
        except Exception:
            logger.exception("thread-recap LLM call failed")
            return

        valid_ids = {t.id for t in threads}
        written = 0
        for r in response.recaps:
            if r.thread_id not in valid_ids:
                continue
            try:
                await asyncio.to_thread(
                    self.repo.set_thread_recap, r.thread_id, r.recap,
                )
                written += 1
            except Exception:
                logger.exception(
                    "thread-recap: set_thread_recap failed for %d", r.thread_id,
                )
        logger.info(
            "thread recaps refreshed: %d threads in batch, %d recaps written",
            len(blocks), written,
        )

    # =========================================================
    # Engaging subjects loop — direct subject-extraction from
    # recent chat. Distinct from topic_threads (cosine clustering
    # of snapshots), which tend to lump multiple subjects together
    # within a window. This pass asks the LLM to SEPARATE distinct
    # subjects and to silently filter out religious / political /
    # controversial topics.
    # =========================================================

    # Bumped from 8192 to INFORMED_NUM_CTX so the prompt can carry
    # cluster blocks + driver notes + transcript section + known
    # subjects + channel context without truncation.
    SUBJECTS_NUM_CTX = INFORMED_NUM_CTX
    SUBJECTS_LOOKBACK_MINUTES = 20
    SUBJECTS_MAX_MESSAGES = 250
    # Streamer-controlled blocklist key in app_settings. Each entry:
    # {"slug": str, "name": str, "rejected_at": iso}. Fed into the
    # prompt as an "exclude these" list so subsequent LLM passes don't
    # re-extract a hallucinated subject the streamer already rejected.
    SUBJECTS_BLOCKLIST_KEY = "engaging_subjects_blocklist"

    SUBJECTS_SYSTEM = """You're identifying distinct conversation subjects in a Twitch chat for the streamer's dashboard.

Look at the recent messages and extract SEPARATE subjects — not vague themes. The streamer wants to see *what specific things* are being discussed so they can pivot toward the most engaging one.

DEFINITION OF A SUBJECT:
- Specific: "Ninja Gaiden 4 parry timing", not "video games"
- Distinct: "Resident Evil aim-parry strats" and "Resident Evil no-damage runs" can be separate subjects even if they share vocabulary; merge only when they're really the same conversation
- 4-8 word subject line, no fluff

DO NOT EMIT (mark `is_sensitive: true` and the dashboard filters them out):
- religion, faith traditions, religious holidays as moral commentary
- politics: parties, candidates, elections, policy debates
- controversies: war, abortion, gun control, immigration, race discourse, etc.
The streamer doesn't want these surfaced. If chatters are talking about them, just flag `is_sensitive: true` and the dashboard hides the row.

ALSO SKIP (don't emit at all — empty `name` is fine):
- Pure greeting / lurking ("hi", "lol", "first time here")
- One-off reactions with no follow-up
- Bot commands

For each remaining subject:
- name: the subject line
- drivers: distinct chatters actually engaged (NOT just present in the window)
- msg_count: rough number of messages on this subject in the window
- is_sensitive: false (you've already filtered the sensitive ones)
- brief: 1-2 sentence OBSERVATIONAL summary of what chatters are actually saying about it. Paraphrase. NO "you should…", NO "the streamer could…". Pure description.
- angles: up to 3 distinct SUB-ASPECTS that have come up *within this subject* in the messages. Example for subject "Resident Evil parry timing": ["aim-parry vs perfect parry", "comparison to Ninja Gaiden 4", "no-save-no-damage feasibility"]. These are sub-aspects observed in the messages, NOT recommendations.

Return AT MOST 8 subjects, sorted by len(drivers) desc. Empty list is the right answer when chat is too quiet or unfocused.

IMPORTANT: emit observation, not advice. No "you should…", no "the streamer could ask about…". Pure description of what people are talking about.

==================================================================
FEW-SHOT EXAMPLES — study these carefully, they show the difference
between a grounded extraction and a hallucinated one.
==================================================================

EXAMPLE 1 — GOOD (grounded, specific, observational)

Input messages:
  [alice] yo this parry timing in ng4 is brutal compared to ng2
  [bob] ng4 the windows feel way tighter, esp on izuna drop counters
  [alice] yeah and izuna's hyper-armor isn't carrying the way it used to
  [carol] anyone else getting wrecked by the boss on stage 3
  [bob] stage 3 boss is the wall fr

Good output:
{
  "subjects": [
    {
      "name": "Ninja Gaiden 4 parry timing vs NG2",
      "drivers": ["alice", "bob"],
      "msg_count": 3,
      "is_sensitive": false,
      "brief": "Alice and Bob are comparing parry windows in NG4 to NG2 and noting izuna drop counters feel tighter, with izuna's hyper-armor less reliable.",
      "angles": ["parry window tightness vs NG2", "izuna drop counters", "hyper-armor reliability"]
    },
    {
      "name": "NG4 stage 3 boss difficulty",
      "drivers": ["carol", "bob"],
      "msg_count": 2,
      "is_sensitive": false,
      "brief": "Carol asks if anyone else is struggling with the stage 3 boss; Bob agrees it's a wall.",
      "angles": ["stage 3 boss as a difficulty wall"]
    }
  ]
}

EXAMPLE 2 — BAD (hallucinated, vague, advisory) — DO NOT DO THIS

Same input as above, BAD output:
{
  "subjects": [
    {
      "name": "video games difficulty",          // too vague
      "drivers": ["alice", "bob", "carol"],
      "msg_count": 5,
      "brief": "You should ask them about the new NG4 DLC.",  // ADVICE + invented DLC
      "angles": ["challenging boss fights"]      // generic, not from messages
    }
  ]
}

Why it's bad: subject name is too vague; "DLC" is invented (not in messages); "you should" is advice not observation; angles are generic instead of message-specific.

EXAMPLE 3 — GOOD (correct skip when chat is just lurkers)

Input messages:
  [dave] hi
  [eve] just got here
  [frank] lol
  [dave] !lurk

Good output:
{ "subjects": [] }

(Empty list is correct — pure greetings and bot commands aren't subjects.)

EXAMPLE 4 — GOOD (filtering sensitive)

Input:
  [g] honestly the election was rigged
  [h] dont start
  [i] meanwhile the new patch dropped

Good output:
{
  "subjects": [
    {
      "name": "new game patch release",
      "drivers": ["i"],
      "msg_count": 1,
      "is_sensitive": false,
      "brief": "i mentions a new patch dropped.",
      "angles": []
    }
  ]
}

(Election thread is omitted — sensitive politics. Note: if you DO emit it for completeness, set `is_sensitive: true` and the dashboard will hide it. Either is acceptable; omitting is cleaner.)
"""

    async def engaging_subjects_loop(
        self, interval_seconds: int | None = None,
    ) -> None:
        """Background task: periodically refresh the engaging-subjects
        cache. No-op when the channel is quiet."""
        await asyncio.sleep(25)  # boot delay so the dashboard settles
        while True:
            try:
                interval = max(60, int(
                    interval_seconds
                    or getattr(self.settings, "engaging_subjects_interval_seconds", 180)
                ))
                if interval <= 0:
                    return
                await self._refresh_engaging_subjects()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("engaging-subjects iteration failed")
            await asyncio.sleep(interval)

    async def _ensure_blocklist_embeddings(self, blocklist: list[dict]) -> None:
        """Embed any blocklisted subject names that don't yet have a
        cached embedding. Runs at the start of each extraction pass so
        the embed cost is amortised. Failures are non-fatal — we just
        skip the cosine dedup for that name and fall back to literal
        slug / name matching."""
        for entry in blocklist:
            slug = (entry.get("slug") or "").lower()
            name = (entry.get("name") or "").strip()
            if not slug or not name:
                continue
            if slug in self._blocklist_embed_cache:
                continue
            try:
                vec = await self.llm.embed(name)
            except Exception:
                logger.exception(
                    "engaging-subjects: failed to embed blocklist entry %r",
                    name,
                )
                continue
            self._blocklist_embed_cache[slug] = vec
        # Drop cache entries for entries no longer in the blocklist
        # (clear_subject_blocklist resets it; reject can also age out
        # past the 50-cap).
        live_slugs = {(e.get("slug") or "").lower() for e in blocklist}
        for stale in list(self._blocklist_embed_cache):
            if stale not in live_slugs:
                self._blocklist_embed_cache.pop(stale, None)

    async def _is_blocked_by_embedding(
        self, name: str, *, threshold: float = 0.85,
    ) -> tuple[bool, str | None, float]:
        """Check whether `name` is semantically too close to any
        blocklisted subject. Returns (blocked, matched_slug,
        cosine_similarity). 0.85 is the default threshold —
        empirically distinguishes "FF8 remake" vs "Final Fantasy 8
        remake" (>0.85) from "FF8 remake" vs "FF14 raid" (<0.85).

        The cosine pass runs in `asyncio.to_thread` so a 50-entry
        blocklist matrix-multiply doesn't peg the event loop —
        previously this fired on every newly-extracted subject
        (~8/pass × every 3 min) and the 50× normalize+dot loop was
        contending with /health and other route handlers, causing
        the wide-jitter latency we measured."""
        if not self._blocklist_embed_cache or not name.strip():
            return False, None, 0.0
        try:
            import numpy as np
        except ImportError:
            return False, None, 0.0
        try:
            qv = await self.llm.embed(name.strip())
        except Exception:
            return False, None, 0.0

        cache_snapshot = list(self._blocklist_embed_cache.items())

        def _score() -> tuple[str | None, float]:
            # Build a single normalized matrix once, then matrix-multiply
            # against the query vector — one numpy call instead of 50
            # python-level cosines. Small enough to be irrelevant on its
            # own, but it stops the asyncio event loop from blocking on
            # CPU work that has nothing to do with the call's caller.
            q = np.asarray(qv, dtype=np.float32)
            qn = float(np.linalg.norm(q))
            if qn == 0.0:
                return None, 0.0
            q = q / qn
            slugs: list[str] = []
            vecs: list[list[float]] = []
            for slug, vec in cache_snapshot:
                if not vec:
                    continue
                slugs.append(slug)
                vecs.append(vec)
            if not vecs:
                return None, 0.0
            mat = np.asarray(vecs, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            # Avoid div-zero on degenerate rows.
            norms = np.where(norms == 0.0, 1.0, norms)
            mat = mat / norms
            sims = mat @ q                           # (N,)
            idx = int(np.argmax(sims))
            return slugs[idx], float(sims[idx])

        best_slug, best_sim = await asyncio.to_thread(_score)
        return (best_sim >= threshold, best_slug, best_sim)

    def _load_subject_blocklist(self) -> list[dict]:
        """Load the streamer-flagged subject blocklist from app_settings.
        Stored as a JSON array of {slug, name, rejected_at} dicts."""
        import json as _json
        raw = self.repo.get_app_setting(self.SUBJECTS_BLOCKLIST_KEY)
        if not raw:
            return []
        try:
            data = _json.loads(raw)
            return [d for d in data if isinstance(d, dict)]
        except (TypeError, ValueError):
            return []

    def reject_subject(self, slug: str, name: str) -> None:
        """Add a subject to the blocklist. The next extraction pass
        will see it in the prompt's exclusion list, and the cache-
        write filter is also keyed off it as a defense in depth."""
        import json as _json
        from datetime import datetime as _dt, timezone as _tz
        blocklist = self._load_subject_blocklist()
        slug_lc = (slug or "").lower()
        name_lc = (name or "").lower()
        # Idempotent — don't add twice.
        if any(
            (b.get("slug") or "").lower() == slug_lc
            or (b.get("name") or "").lower() == name_lc
            for b in blocklist
        ):
            return
        blocklist.append({
            "slug": slug_lc,
            "name": name.strip(),
            "rejected_at": _dt.now(_tz.utc).isoformat(timespec="seconds"),
        })
        # Cap at 50 entries so app_settings doesn't grow unbounded.
        blocklist = blocklist[-50:]
        self.repo.set_app_setting(self.SUBJECTS_BLOCKLIST_KEY, _json.dumps(blocklist))
        # Drop the rejected subject from the live cache immediately so
        # the streamer doesn't have to wait for the next refresh.
        self._subjects_cache = EngagingSubjectsCache(
            subjects=[
                s for s in self._subjects_cache.subjects
                if s.slug.lower() != slug_lc
                and s.name.lower() != name_lc
            ],
            refreshed_at=self._subjects_cache.refreshed_at,
            error=self._subjects_cache.error,
        )

    def clear_subject_blocklist(self) -> int:
        """Wipe all rejections — used when the streamer wants to reset
        (e.g., new stream, different topic). Returns the count cleared."""
        n = len(self._load_subject_blocklist())
        self.repo.set_app_setting(self.SUBJECTS_BLOCKLIST_KEY, "[]")
        return n

    def _build_channel_context(self, *, authoritative: bool = False) -> str:
        """Delegate to the shared TwitchStatus.format_for_llm so every
        informed call site sees the same Helix snapshot in the same
        shape. Returns "" when twitch_status isn't wired or the
        streamer is offline."""
        if self.twitch_status is None:
            return ""
        ts = getattr(self.twitch_status, "status", None)
        if ts is None:
            return ""
        try:
            return ts.format_for_llm(authoritative=authoritative)
        except AttributeError:
            # Defensive — older TwitchStatus shape without the helper.
            return ""


    def _load_streamer_facts(self) -> str:
        """Load streamer-authored channel facts from disk. Prepended to
        the engaging-subjects system prompt so subject extraction has
        channel-specific context (recurring bits, current arcs, inside
        jokes) that the LLM can't infer from chat alone.

        Missing or empty file returns "" — the prompt just falls back
        to the default behavior in that case."""
        from pathlib import Path
        path_str = getattr(self.settings, "streamer_facts_path", "") or ""
        if not path_str:
            return ""
        p = Path(path_str)
        if not p.is_absolute():
            # Resolve relative to the chatterbot project root (the
            # working directory the dashboard runs from).
            p = Path.cwd() / p
        try:
            text = p.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return ""
        # Cap at 4k chars so a runaway file can't blow the context budget.
        if len(text) > 4000:
            text = text[:4000] + "\n\n[…truncated…]"
        return text

    def _attach_or_create_clusters(
        self,
        items: list[tuple],  # list[(Message, list[float])]
        *, threshold: float, min_size: int, age_out_minutes: int,
    ) -> tuple[set[str], set[str]]:
        """Persistent cosine clustering. For each new (msg, vec):
          1. Look up an existing assignment; if present, skip.
          2. Else find the existing-cluster centroid with max cosine;
             if max sim ≥ threshold, attach (update centroid running
             mean + msg_ids + last_added_ts).
          3. Orphans (no match) go to a holding list.
        Then orphans get greedy-clustered among themselves; only groups
        of size ≥ `min_size` spawn NEW persistent clusters.

        Aging: clusters with last_added_ts older than `age_out_minutes`
        are dropped, along with their msg_ids in the lookup map.

        Returns:
          (live_cluster_ids, dirty_cluster_ids)
          - live: clusters with at least one msg in the current window
          - dirty: clusters that need (re)labeling — brand new, or
            n ≥ 2 * n_at_last_label (substantial growth)."""
        try:
            import numpy as np
        except ImportError:
            return set(), set()
        if not items:
            return set(), set()

        now_ts = time.time()
        # Aging — drop stale clusters before doing anything else so we
        # don't try to attach new messages to dead ones.
        cutoff = now_ts - (age_out_minutes * 60)
        stale = [
            cid for cid, c in self._clusters.items()
            if c.last_added_ts and c.last_added_ts < cutoff
        ]
        for cid in stale:
            for mid in self._clusters[cid].msg_ids:
                self._msg_to_cluster.pop(mid, None)
            self._clusters.pop(cid, None)

        # Cap _msg_to_cluster size — the dict grows unbounded otherwise.
        # Drop oldest message ids (smallest ints) when over the cap.
        max_map = 5000
        if len(self._msg_to_cluster) > max_map:
            keep = sorted(self._msg_to_cluster.keys())[-max_map:]
            keep_set = set(keep)
            self._msg_to_cluster = {
                k: v for k, v in self._msg_to_cluster.items() if k in keep_set
            }

        # Step 1: try to attach each not-yet-assigned message to an
        # existing cluster.
        cluster_ids = list(self._clusters.keys())
        if cluster_ids:
            centroids = np.array(
                [self._clusters[cid].centroid for cid in cluster_ids],
                dtype=np.float32,
            )
        else:
            centroids = np.zeros((0, 1), dtype=np.float32)

        orphan_msgs: list = []   # list[Message]
        orphan_vecs: list = []   # list[np.ndarray] (unit)
        attached_to: dict[int, str] = {}

        def _unit(v: list[float]) -> np.ndarray:
            arr = np.array(v, dtype=np.float32)
            n = float(np.linalg.norm(arr))
            return arr / n if n > 0 else arr

        for msg, vec in items:
            if msg.id in self._msg_to_cluster:
                continue  # already assigned on a prior pass
            uvec = _unit(vec)
            if cluster_ids and centroids.shape[0] > 0:
                sims = centroids @ uvec
                best = int(np.argmax(sims))
                if float(sims[best]) >= threshold:
                    cid = cluster_ids[best]
                    attached_to[msg.id] = cid
                    c = self._clusters[cid]
                    c.n += 1
                    # Running-mean update of the unit centroid.
                    new_centroid = (
                        (np.array(c.centroid, dtype=np.float32) * (c.n - 1)) + uvec
                    ) / c.n
                    cn = float(np.linalg.norm(new_centroid))
                    if cn > 0:
                        new_centroid = new_centroid / cn
                    c.centroid = new_centroid.tolist()
                    centroids[best] = new_centroid  # keep parallel array fresh
                    c.last_added_ts = now_ts
                    c.msg_ids.append(msg.id)
                    if len(c.msg_ids) > 200:
                        c.msg_ids = c.msg_ids[-200:]
                    self._msg_to_cluster[msg.id] = cid
                    continue
            orphan_msgs.append(msg)
            orphan_vecs.append(uvec)

        # Step 2: greedy-cluster the orphans among themselves; promote
        # groups of size ≥ min_size into NEW persistent clusters.
        new_cluster_ids: set[str] = set()
        if orphan_msgs:
            buckets: list[list[int]] = []
            bucket_centroids: list[np.ndarray] = []
            for i, uvec in enumerate(orphan_vecs):
                if not buckets:
                    buckets.append([i])
                    bucket_centroids.append(uvec.copy())
                    continue
                sims = np.array([float(c @ uvec) for c in bucket_centroids])
                best = int(np.argmax(sims))
                if float(sims[best]) >= threshold:
                    buckets[best].append(i)
                    k = len(buckets[best])
                    new_c = ((bucket_centroids[best] * (k - 1)) + uvec) / k
                    cn = float(np.linalg.norm(new_c))
                    if cn > 0:
                        new_c = new_c / cn
                    bucket_centroids[best] = new_c
                else:
                    buckets.append([i])
                    bucket_centroids.append(uvec.copy())

            for bi, idxs in enumerate(buckets):
                if len(idxs) < min_size:
                    continue
                cid = uuid.uuid4().hex[:12]
                new_cluster_ids.add(cid)
                self._clusters[cid] = _PersistentCluster(
                    cluster_id=cid,
                    centroid=bucket_centroids[bi].tolist(),
                    n=len(idxs),
                    last_added_ts=now_ts,
                    msg_ids=[orphan_msgs[i].id for i in idxs],
                )
                for i in idxs:
                    self._msg_to_cluster[orphan_msgs[i].id] = cid

        # Determine live + dirty clusters relative to the current items.
        live: set[str] = set()
        for m, _ in items:
            cid = self._msg_to_cluster.get(m.id)
            if cid:
                live.add(cid)
        dirty: set[str] = set(new_cluster_ids)
        for cid in live:
            c = self._clusters.get(cid)
            if c is None:
                continue
            if not c.name:
                dirty.add(cid)
            elif c.n >= max(c.n_at_last_label * 2, c.n_at_last_label + 4):
                dirty.add(cid)
        return live, dirty

    async def _refresh_engaging_subjects(self) -> None:
        async with self._subjects_lock:
            window_min = int(getattr(
                self.settings, "engaging_subjects_lookback_minutes",
                self.SUBJECTS_LOOKBACK_MINUTES,
            ))
            limit = int(getattr(
                self.settings, "engaging_subjects_max_messages",
                self.SUBJECTS_MAX_MESSAGES,
            ))
            try:
                items = await asyncio.to_thread(
                    self.repo.recent_messages,
                    limit=limit,
                    within_minutes=window_min,
                    with_embeddings=True,
                )
            except Exception:
                logger.exception(
                    "engaging-subjects: recent_messages failed",
                )
                return
            msgs = [m for m, _ in items]
            if len(msgs) < 5:
                # Too quiet to bother. Reset the cache so the panel
                # shows the empty state instead of stale subjects. Don't
                # nuke the persistent clusters though — a brief lull
                # doesn't mean the subjects are dead.
                self._subjects_cache = EngagingSubjectsCache(
                    subjects=[], refreshed_at=time.time(), error=None,
                )
                return

            cluster_threshold = float(getattr(
                self.settings, "engaging_subjects_cluster_threshold", 0.55,
            ))
            min_cluster_size = int(getattr(
                self.settings, "engaging_subjects_min_cluster_size", 3,
            ))
            # Persistent clusters age out at 2x the lookback window so a
            # brief lull doesn't kill them.
            live_ids, dirty_ids = self._attach_or_create_clusters(
                items,
                threshold=cluster_threshold,
                min_size=min_cluster_size,
                age_out_minutes=window_min * 2,
            )

            # Resolve user names once for prompt formatting + driver lists.
            user_id_to_name: dict[str, str] = {}
            for m in msgs:
                if m.user_id and m.user_id not in user_id_to_name:
                    u = await asyncio.to_thread(self.repo.get_user, m.user_id)
                    if u:
                        user_id_to_name[m.user_id] = u.name

            # Index window messages by cluster for prompt formatting +
            # cache rendering. Falls back to the persistent cluster's
            # broader msg_ids if needed.
            id_to_msg = {m.id: m for m in msgs}
            cluster_msgs: dict[str, list] = {cid: [] for cid in live_ids}
            for m in msgs:
                cid = self._msg_to_cluster.get(m.id)
                if cid in cluster_msgs:
                    cluster_msgs[cid].append(m)

            # Streamer voice — recent transcript chunks let the LLM tell
            # "chat reacting to what the streamer just said" from
            # "chat-driven subject". Also useful for grounding game-
            # specific jargon to what's actually on screen.
            try:
                transcripts = await asyncio.to_thread(
                    self.repo.recent_transcripts,
                    within_minutes=window_min, limit=80,
                )
            except Exception:
                transcripts = []
            transcript_section = ""
            if transcripts:
                tlines = [
                    f"  - {t.text[:240].strip()}"
                    for t in transcripts if t.text and t.text.strip()
                ]
                if tlines:
                    transcript_section = (
                        f"STREAMER VOICE (last {window_min} min, oldest first — "
                        "what the streamer has actually been saying out loud "
                        "on stream; chat may be reacting to this):\n"
                        + "\n".join(tlines[-60:])
                        + "\n\n"
                    )

            # Per-driver notes — for the most-active drivers in this
            # window, pull a few of their stored notes so the LLM has
            # background on who's talking and what they care about.
            from collections import Counter as _Counter
            driver_counts = _Counter(
                user_id_to_name[m.user_id]
                for m in msgs
                if m.user_id in user_id_to_name
            )
            max_drivers_with_notes = int(getattr(
                self.settings, "engaging_subjects_max_drivers_with_notes", 8,
            ))
            notes_per_driver = int(getattr(
                self.settings, "engaging_subjects_notes_per_driver", 2,
            ))
            top_drivers = [
                name for name, _ in driver_counts.most_common(max_drivers_with_notes)
            ]
            name_to_uid = {v: k for k, v in user_id_to_name.items()}
            driver_note_blocks: list[str] = []
            for name in top_drivers:
                uid = name_to_uid.get(name)
                if not uid:
                    continue
                try:
                    notes = await asyncio.to_thread(self.repo.get_notes, uid)
                except Exception:
                    continue
                if not notes:
                    continue
                top_notes = notes[:notes_per_driver]
                note_lines = "\n      - " + "\n      - ".join(
                    n.text[:160] for n in top_notes
                )
                driver_note_blocks.append(f"  {name}:{note_lines}")
            driver_notes_section = ""
            if driver_note_blocks:
                driver_notes_section = (
                    "DRIVER NOTES (background on the most-active chatters in "
                    "this window — use to ground subject extraction, not as "
                    "evidence by themselves):\n"
                    + "\n".join(driver_note_blocks)
                    + "\n\n"
                )

            # Active topic-thread titles — already-identified ongoing
            # subjects. The LLM gets them as "known subjects" so it
            # matches new messages against these rather than inventing
            # near-duplicates.
            try:
                active_threads = await asyncio.to_thread(
                    self.repo.list_threads,
                    status_filter="active", query="", limit=10,
                )
            except Exception:
                active_threads = []
            known_titles: list[str] = [t.title for t in active_threads if t.title]
            # Also include any ALREADY-NAMED persistent clusters in the
            # known list so the LLM doesn't pick a duplicative new label.
            for cid, c in self._clusters.items():
                if c.name and c.name not in known_titles:
                    known_titles.append(c.name)
            known_section = ""
            if known_titles:
                known_section = (
                    "KNOWN ACTIVE SUBJECTS (already-identified ongoing "
                    "conversations on this channel — match new messages to "
                    "these when applicable rather than inventing near-"
                    "duplicates; reuse the wording verbatim if it still "
                    "fits):\n  - "
                    + "\n  - ".join(known_titles[:20])
                    + "\n\n"
                )

            # Blocklist injection — the streamer has flagged these
            # subjects as hallucinated / wrong / irrelevant.
            blocklist = await asyncio.to_thread(self._load_subject_blocklist)
            blocklist_lines = ""
            if blocklist:
                names = [b.get("name", "") for b in blocklist if b.get("name")]
                if names:
                    blocklist_lines = (
                        "\n\nDO NOT EXTRACT (streamer flagged as wrong / "
                        "hallucinated / not actually a thing chatters are "
                        "discussing): "
                        + "; ".join(f'"{n}"' for n in names[-30:])
                        + ". If you see chatter messages mentioning anything "
                        "from this list, treat it as off-limits and don't "
                        "include it in your output."
                    )

            # Channel context — what the streamer is actually playing /
            # streaming right now. Helps the LLM ground game-specific
            # jargon, scale interpretation by chat size, and respect
            # streamer-declared content classifications. Sourced from
            # the Helix /streams + /channels poll on TwitchService;
            # silently skipped when Helix isn't connected.
            # authoritative=True — the engaging-subjects call now also
            # rides with a screenshot grid, same as talking-points.
            channel_context = self._build_channel_context(authoritative=True)

            # === LLM labeling pass — only for dirty clusters. ===
            # Carry-forward names mean stable identity; the LLM only
            # gets called when there's a genuinely new or substantially
            # grown cluster to label.
            response_subjects: list = []
            # Track the order in which we send clusters to the LLM so
            # we can match responses back positionally when the model
            # forgets to echo `cluster_id` (Qwen sometimes drops the
            # field even though the schema lists it as optional with
            # default "").
            sent_cluster_ids: list[str] = []
            if dirty_ids:
                cluster_blocks: list[str] = []
                # Sort by current window driver count desc so the LLM
                # processes the hottest clusters first.
                def _cluster_priority(cid: str) -> int:
                    return len({
                        m.user_id for m in cluster_msgs.get(cid, [])
                    })
                sorted_dirty = sorted(
                    dirty_ids, key=_cluster_priority, reverse=True,
                )
                for cid in sorted_dirty[:8]:  # cap LLM payload
                    block_msgs = cluster_msgs.get(cid, [])
                    if not block_msgs:
                        # Brand new cluster from orphans — fall back to
                        # the persistent msg_ids list so the LLM has
                        # SOMETHING to label.
                        block_msgs = [
                            id_to_msg[mid]
                            for mid in self._clusters[cid].msg_ids
                            if mid in id_to_msg
                        ]
                    if not block_msgs:
                        continue
                    prior_name = self._clusters[cid].name
                    name_hint = (
                        f" (currently labeled: {prior_name!r}; reuse if "
                        "still fits, rename only if drifted)"
                        if prior_name else " (NEW — needs a label)"
                    )
                    lines = [
                        f"    [{user_id_to_name.get(m.user_id, m.user_id)}] "
                        f"{m.content[:200]}"
                        for m in block_msgs[-30:]
                    ]
                    cluster_blocks.append(
                        f"  CLUSTER {cid}{name_hint} ({len(block_msgs)} msgs):\n"
                        + "\n".join(lines)
                    )
                    sent_cluster_ids.append(cid)

                if cluster_blocks:
                    prompt = (
                        channel_context
                        + transcript_section
                        + known_section
                        + driver_notes_section
                        + f"Recent chat clusters needing labels (over the last "
                        f"{window_min} min). Each block has a CLUSTER id — "
                        "ECHO that id back in your `cluster_id` field so the "
                        "dashboard can match your label to the cluster. "
                        "Reuse the existing label verbatim when it still fits "
                        "the messages; only rename if the conversation has "
                        "clearly drifted:\n\n"
                        + "\n\n".join(cluster_blocks)
                        + blocklist_lines
                        + "\n\nReturn one subject per cluster (or skip a "
                        "cluster if it's too noisy / sensitive)."
                    )

                    facts = await asyncio.to_thread(self._load_streamer_facts)
                    system_prompt = self.SUBJECTS_SYSTEM
                    if facts:
                        system_prompt = (
                            "STREAMER-AUTHORED CHANNEL FACTS (treat as ground "
                            "truth about the streamer / channel — chat "
                            "references to these are real, not hallucinations "
                            "to be flagged):\n\n"
                            + facts
                            + "\n\n==================================================================\n\n"
                            + self.SUBJECTS_SYSTEM
                        )

                    # Visual context — same screenshot grid the
                    # transcript group summary uses, so the LLM can
                    # ground game-specific jargon when chat references
                    # something on-screen.
                    grid_b64 = await self._latest_screenshot_grid(
                        window_minutes=window_min,
                    )
                    if grid_b64:
                        prompt += (
                            "\n\nAn image is attached showing what is "
                            "currently on screen. Use it ONLY as silent "
                            "context (identify the game/app, ground "
                            "vague subject names). Do NOT describe the "
                            "image or mention that it's attached."
                        )
                    try:
                        from .llm.schemas import EngagingSubjectsResponse
                        # think=True — engaging-subjects feeds the
                        # dashboard's main "what's chat talking about"
                        # panel; same accuracy-over-latency tradeoff as
                        # talking points. Cadence (~3 min) absorbs it.
                        response = await self.llm.generate_structured(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            response_model=EngagingSubjectsResponse,
                            num_ctx=self.SUBJECTS_NUM_CTX,
                            images=[grid_b64] if grid_b64 else None,
                            think=True,
                        )
                        response_subjects = response.subjects
                        if not response_subjects:
                            logger.info(
                                "engaging-subjects: LLM returned 0 subjects "
                                "for %d dirty cluster(s) — chat may be too "
                                "noisy/sensitive to label",
                                len(sorted_dirty[:8]),
                            )
                        else:
                            # Light diagnostic so we can spot when the LLM
                            # is producing labels but the apply step is
                            # dropping them on the floor (cluster_id
                            # mismatch, sensitivity flagging, blocklist).
                            logger.debug(
                                "engaging-subjects: LLM returned %d subject(s): %s",
                                len(response_subjects),
                                [
                                    {
                                        "name": s.name[:40],
                                        "cid": s.cluster_id[:12],
                                        "sens": s.is_sensitive,
                                        "drv": len(s.drivers),
                                    }
                                    for s in response_subjects[:8]
                                ],
                            )
                    except ValidationError as e:
                        logger.warning("engaging-subjects validation failed: %s", e)
                    except Exception:
                        logger.exception("engaging-subjects LLM call failed")

            # Apply LLM labels back onto the persistent clusters.
            # Match each response subject to a cluster:
            #   1. Prefer the echoed `cluster_id` field — most reliable.
            #   2. Fall back to positional matching against
            #      `sent_cluster_ids` when cluster_id is missing/empty.
            #      Qwen's structured output sometimes drops the field
            #      even though the schema includes it; positional is a
            #      strong second since the LLM tends to return subjects
            #      in input order.
            block_set_slug = {(b.get("slug") or "").lower() for b in blocklist}
            block_set_name = {(b.get("name") or "").lower() for b in blocklist}
            # Pre-embed every blocklisted name so we can cosine-dedup
            # the LLM's new picks against historical rejections — same
            # threshold even when the model rephrases ("FF8 remake" vs
            # "Final Fantasy 8 remake"). Cheap (capped at 50 entries,
            # cache reused across passes).
            await self._ensure_blocklist_embeddings(blocklist)
            applied = 0
            empty_id_count = 0
            for idx, s in enumerate(response_subjects):
                cid = (s.cluster_id or "").strip()
                if not cid:
                    empty_id_count += 1
                    if idx < len(sent_cluster_ids):
                        cid = sent_cluster_ids[idx]
                if not cid or cid not in self._clusters:
                    continue
                if s.is_sensitive:
                    self._clusters[cid].is_sensitive = True
                    self._clusters[cid].n_at_last_label = self._clusters[cid].n
                    continue
                clean_name = s.name.strip()
                if not clean_name:
                    continue
                if clean_name.lower() in block_set_name:
                    logger.info(
                        "engaging-subjects: dropped re-extracted blocklisted "
                        "subject %r", clean_name,
                    )
                    continue
                # Embedding-based near-dup gate. Catches the same
                # subject under a slightly different name. Threshold
                # is intentionally strict (0.85) so we don't smother
                # legitimate new subjects that share theme words.
                blocked_by_emb, matched_slug, sim = (
                    await self._is_blocked_by_embedding(clean_name)
                )
                if blocked_by_emb:
                    logger.info(
                        "engaging-subjects: dropped re-extracted "
                        "blocklisted subject %r (matched %r via "
                        "embedding sim=%.2f)",
                        clean_name, matched_slug, sim,
                    )
                    continue
                c = self._clusters[cid]
                c.name = clean_name
                c.brief = (s.brief or "").strip()
                c.angles = [
                    a.strip() for a in (s.angles or []) if a and a.strip()
                ][:3]
                c.is_sensitive = False
                c.n_at_last_label = c.n
                applied += 1
            if response_subjects and empty_id_count:
                logger.info(
                    "engaging-subjects: %d/%d responses missing cluster_id "
                    "(matched positionally); applied %d total",
                    empty_id_count, len(response_subjects), applied,
                )
            if response_subjects and applied == 0:
                # Bug bait — LLM gave us labels but every single one
                # got dropped (sensitivity flag / blocklist / cluster
                # mismatch). Worth a louder log so future regressions
                # don't silently empty the panel.
                logger.warning(
                    "engaging-subjects: %d response(s) returned but 0 applied — "
                    "all dropped via sensitivity/blocklist/missing-cluster. "
                    "First response: %r (sens=%s, cid=%r)",
                    len(response_subjects),
                    response_subjects[0].name[:80],
                    response_subjects[0].is_sensitive,
                    response_subjects[0].cluster_id[:32],
                )

            # Materialize the cache from live clusters (window-filtered).
            import hashlib as _h
            entries: list[EngagingSubjectEntry] = []
            for cid in live_ids:
                c = self._clusters.get(cid)
                if c is None or c.is_sensitive or not c.name:
                    continue
                slug = _h.sha1(c.name.lower().encode("utf-8")).hexdigest()[:12]
                if slug in block_set_slug or c.name.lower() in block_set_name:
                    continue
                window_msgs_for_cid = cluster_msgs.get(cid, [])
                drivers = []
                seen_drivers = set()
                # newest-first so the most-recent driver shows first.
                for m in reversed(window_msgs_for_cid):
                    name = user_id_to_name.get(m.user_id)
                    if name and name.lower() not in seen_drivers:
                        drivers.append(name)
                        seen_drivers.add(name.lower())
                if not drivers:
                    continue
                entries.append(EngagingSubjectEntry(
                    name=c.name,
                    drivers=drivers[:20],
                    msg_count=len(window_msgs_for_cid),
                    brief=c.brief,
                    angles=list(c.angles),
                    slug=slug,
                ))
            entries.sort(key=lambda e: (-len(e.drivers), -e.msg_count))
            entries = entries[:8]

            self._subjects_cache = EngagingSubjectsCache(
                subjects=entries, refreshed_at=time.time(), error=None,
            )
            logger.info(
                "engaging subjects refreshed: %d msgs, %d live clusters, "
                "%d dirty (relabeled), %d cached subjects",
                len(msgs), len(live_ids), len(dirty_ids), len(entries),
            )
