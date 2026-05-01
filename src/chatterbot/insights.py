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
class _PersistentSubject:
    """One persistent engaging-subject identity carried across
    refreshes of the extractor.

    Replaces the slice-2 `_PersistentCluster` model, which tried to
    cluster raw chat messages by embedding cosine and then label the
    clusters. That approach collapsed under centroid drift on short
    noisy chat strings — one mega-cluster ate everything, and the
    "extract distinct subjects" feature had no distinct clusters to
    extract from.

    The new model inverts the work split: the LLM does the topic
    modeling on raw chat (naming + grouping by cited message_ids)
    and embeddings drive cross-refresh identity matching by comparing
    the SUBJECT NAME's embedding against previously-seen subjects.
    Embeddings are now matching short clean strings (subject names),
    which is what they're good at.

    Identity invariant: same subject across refreshes → same
    `subject_id`, even if the LLM phrasing drifts slightly between
    runs. Names update in place; the id is forever.
    """
    subject_id: str            # uuid hex; stable across refreshes
    name: str
    name_embedding: list[float]  # unit-vector embedding of `name`
    brief: str = ""
    angles: list[str] = field(default_factory=list)
    is_sensitive: bool = False
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0    # most-recent refresh that surfaced this subject
    refresh_count: int = 0       # how many refreshes have surfaced it
    # Capped recent message_ids the LLM cited for this subject across
    # refreshes. Used to power the "see source messages" expansion
    # when the streamer clicks a subject row.
    msg_ids: list[int] = field(default_factory=list)
    # Per-subject talking points cache. Generated on demand when the
    # streamer opens the modal. Invalidated when the subject's
    # `last_seen_ts` advances (= the engaging-subjects refresh
    # surfaced this subject again with new context). Avoids paying
    # for an LLM call on every modal open while keeping the points
    # fresh as the conversation evolves.
    talking_points: list[str] = field(default_factory=list)
    talking_points_at: float = 0.0   # last_seen_ts at generation


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
    used by the on-demand /insights/subject/{slug}/expand route and the
    reject-subject flow to identify a subject without leaking cache
    indices.

    `msg_ids` is the list of message ids the LLM cited as supporting
    this subject in the most recent refresh. Powers the "see source
    messages" expansion — the dashboard hydrates these ids directly
    rather than guessing from driver names + lookback window. Empty
    list is safe (route falls back to driver-based lookup)."""
    name: str
    drivers: list[str]
    msg_count: int
    brief: str = ""
    angles: list[str] = field(default_factory=list)
    slug: str = ""
    msg_ids: list[int] = field(default_factory=list)


@dataclass
class EngagingSubjectsCache:
    subjects: list[EngagingSubjectEntry]
    refreshed_at: float | None
    error: str | None = None


@dataclass
class OpenQuestionDriver:
    """One chatter who asked (or co-asked) an open question. Mirrors the
    dict shape `repo.recent_questions` returns so the existing
    chat_questions.html template renders without changes."""
    name: str
    user_id: str
    ts: str


@dataclass
class OpenQuestionEntry:
    """One LLM-curated open question. Field names match the dict keys
    `chat_questions.html` already reads (`question`, `count`,
    `drivers`, `latest_ts`, `last_msg_id`) so the template doesn't
    need changes — the LLM just refines the heuristic candidates."""
    question: str
    count: int
    drivers: list[OpenQuestionDriver]
    latest_ts: str
    last_msg_id: int


@dataclass
class OpenQuestionsCache:
    questions: list[OpenQuestionEntry]
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
        self._questions_cache = OpenQuestionsCache(
            questions=[], refreshed_at=None, error=None,
        )
        self._lock = asyncio.Lock()
        self._subjects_lock = asyncio.Lock()
        self._questions_lock = asyncio.Lock()
        # Persistent engaging-subjects carried across refreshes,
        # keyed by stable subject_id (uuid hex). Cross-refresh
        # identity is resolved by cosine-matching the SUBJECT NAME's
        # embedding against existing entries — embeddings work well
        # on short clean strings (subject names) but poorly on raw
        # noisy chat (which is why slice-2's centroid clustering of
        # message embeddings collapsed into one mega-cluster).
        self._subjects: dict[str, _PersistentSubject] = {}
        # Watermark — the highest message id we've seen in a refresh.
        # When latest_message_id hasn't moved since the last pass,
        # we skip the LLM call entirely (same input → same output).
        self._subjects_last_msg_id: int = 0
        # In-memory embedding cache for blocklisted subject names —
        # populated lazily during _refresh_engaging_subjects so new
        # subject names can be cosine-matched against rejected
        # historical names. Catches near-dupes ("FF8 remake" vs "Final
        # Fantasy 8 remake") that the literal slug/name blocklist
        # would miss. Keyed by slug so it stays in sync with the
        # blocklist's own dedupe key.
        self._blocklist_embed_cache: dict[str, list[float]] = {}
        # Per-question answer-angles cache, keyed by last_msg_id (the
        # cluster's identity). Populated on first modal-open; reused
        # while the question is still surfaced (same identity). Pruned
        # in _refresh_open_questions when a question drops off the
        # cache so memory doesn't accumulate forever.
        self._question_angles_cache: dict[int, list[str]] = {}
        # Streamer-authored channel facts, cached with mtime so a hot
        # call site (every ~3 min) doesn't re-read the file every
        # pass. Edited file → next call picks up the change.
        self._facts_text: str = ""
        self._facts_mtime: float = 0.0
        # Stitched OBS screenshot grids cached by window_minutes with
        # a short TTL. The 5 generative LLM call sites each ask for a
        # grid per refresh / modal open; without a cache they each
        # do file IO + Pillow stitch independently even when their
        # windows overlap. TTL > screenshot_interval_seconds (default
        # 10s) so the cache survives long enough to dedupe one
        # refresh cycle's worth of calls but invalidates before
        # streamer-noticeable staleness.
        self._screenshot_grid_cache: dict[
            int, tuple[float, str | None],
        ] = {}

    @property
    def cache(self) -> InsightsCache:
        return self._cache

    @property
    def subjects_cache(self) -> EngagingSubjectsCache:
        return self._subjects_cache

    @property
    def open_questions_cache(self) -> OpenQuestionsCache:
        return self._questions_cache

    # Short TTL for the screenshot-grid cache. Longer than
    # screenshot_interval_seconds (default 10s) so the cache survives
    # one refresh cycle's worth of LLM calls; shorter than the
    # fastest LLM-loop interval (60s) so streamers don't see
    # multi-minute-stale visuals.
    SCREENSHOT_GRID_TTL_SECONDS = 20

    async def _latest_screenshot_grid(
        self, window_minutes: int = 10,
    ) -> str | None:
        """Build a base64 2x2 grid of the most recent OBS screenshots so
        the multimodal LLM has visual game context for talking-points,
        engaging-subjects, and thread-recap calls.

        Cached for `SCREENSHOT_GRID_TTL_SECONDS` keyed by
        `window_minutes` so repeat calls within one refresh cycle
        (or a modal open right after a panel refresh) skip the file
        IO + Pillow stitch.

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

        # Cache hit — return the previously-stitched grid (or None
        # for "no shots in window" outcomes; we cache negative
        # results too to avoid re-querying screenshots_in_range when
        # there genuinely aren't any shots).
        key = int(window_minutes)
        now = time.time()
        cached = self._screenshot_grid_cache.get(key)
        if cached is not None:
            cached_at, cached_b64 = cached
            if now - cached_at < self.SCREENSHOT_GRID_TTL_SECONDS:
                return cached_b64

        try:
            from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        except ImportError:
            return None
        end = _dt.now(_tz.utc)
        start = end - _td(minutes=max(1, key))
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
            self._screenshot_grid_cache[key] = (now, None)
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
            self._screenshot_grid_cache[key] = (now, None)
            return None
        import base64 as _b64
        encoded = _b64.b64encode(grid_bytes).decode("ascii")
        self._screenshot_grid_cache[key] = (now, encoded)
        return encoded

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
                from .llm.prompts import resolve_prompt
                response = await self.llm.generate_structured(
                    prompt=prompt,
                    system_prompt=await self._streamer_facts_prepended(
                        resolve_prompt(
                            "insights.talking_points", self.repo,
                        ),
                    ),
                    response_model=TalkingPointsResponse,
                    num_ctx=INFORMED_NUM_CTX,
                    images=[grid_b64] if grid_b64 else None,
                    think=True,
                    call_site="insights.talking_points",
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
            from .llm.prompts import resolve_prompt
            response = await self.llm.generate_structured(
                prompt=prompt,
                system_prompt=await self._streamer_facts_prepended(
                    resolve_prompt(
                        "insights.thread_recaps", self.repo,
                    ),
                ),
                response_model=ThreadRecapsResponse,
                num_ctx=self.THREAD_RECAP_NUM_CTX,
                images=[grid_b64] if grid_b64 else None,
                think=True,
                call_site="insights.thread_recaps",
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

    SUBJECTS_SYSTEM = """You're identifying distinct conversation subjects in a Twitch chat for the streamer's dashboard. The streamer reads your output on a second monitor while streaming — your output never goes back to chat.

You'll receive a NUMBERED LIST of recent chat messages (each line starts with `[id N] [chatter]`). Group them into 0-8 distinct subjects and CITE the supporting message ids for each.

DEFINITION OF A SUBJECT:
- Specific: "Ninja Gaiden 4 parry timing", not "video games"
- Distinct: "Resident Evil aim-parry strats" and "Resident Evil no-damage runs" can be separate subjects even if they share vocabulary; merge only when they're really the same conversation
- 4-8 word subject line, no fluff
- Each subject MUST cite at least 2 supporting message ids. A subject grounded in only one message is just a one-off.

DO NOT EMIT (mark `is_sensitive: true` and the dashboard filters them out):
- religion, faith traditions, religious holidays as moral commentary
- politics: parties, candidates, elections, policy debates
- controversies: war, abortion, gun control, immigration, race discourse, etc.
The streamer doesn't want these surfaced. If chatters are talking about them, just flag `is_sensitive: true` and the dashboard hides the row.

ALSO SKIP (don't emit — empty `subjects` list is fine):
- Pure greeting / lurking ("hi", "lol", "first time here")
- One-off reactions with no follow-up
- Bot commands

For each remaining subject return:
- name: the subject line
- drivers: distinct chatters actually engaged with this subject (their names from the input). Do NOT list chatters who only sent unrelated messages.
- msg_count: rough number of messages on this subject in the window
- is_sensitive: false (you've already filtered the sensitive ones; only set true if you DO emit a sensitive subject for completeness)
- brief: 1-2 sentence OBSERVATIONAL summary of what chatters are actually saying about it. Paraphrase. NO "you should…", NO "the streamer could…". Pure description.
- angles: up to 3 distinct SUB-ASPECTS that have come up *within this subject* in the messages. Example for "Resident Evil parry timing": ["aim-parry vs perfect parry", "comparison to Ninja Gaiden 4", "no-save-no-damage feasibility"]. Sub-aspects observed in the messages, NOT recommendations.
- message_ids: integer ids from the input lines that support this subject. The dashboard uses these ids to resolve who-said-what without parsing prose, so cite EVERY message that's clearly on this subject. Cite ids only — don't make them up.

If a `KNOWN ACTIVE SUBJECTS` list appears in the input, REUSE the wording verbatim when the same conversation continues. Pick a different name only when the subject has clearly drifted to a new sub-topic. Stable wording across refreshes lets the dashboard maintain identity.

Return AT MOST 8 subjects, sorted by len(drivers) desc. Empty `subjects` is the right answer when chat is too quiet or unfocused.

IMPORTANT: emit observation, not advice. No "you should…", no "the streamer could ask about…". Pure description of what people are talking about.

==================================================================
FEW-SHOT EXAMPLES — study these carefully, they show the difference
between a grounded extraction and a hallucinated one.
==================================================================

EXAMPLE 1 — GOOD (grounded, specific, observational)

Input messages:
  [id 101] [alice] yo this parry timing in ng4 is brutal compared to ng2
  [id 102] [bob] ng4 the windows feel way tighter, esp on izuna drop counters
  [id 103] [alice] yeah and izuna's hyper-armor isn't carrying the way it used to
  [id 104] [carol] anyone else getting wrecked by the boss on stage 3
  [id 105] [bob] stage 3 boss is the wall fr

Good output:
{
  "subjects": [
    {
      "name": "Ninja Gaiden 4 parry timing vs NG2",
      "drivers": ["alice", "bob"],
      "msg_count": 3,
      "is_sensitive": false,
      "brief": "Alice and Bob are comparing parry windows in NG4 to NG2 and noting izuna drop counters feel tighter, with izuna's hyper-armor less reliable.",
      "angles": ["parry window tightness vs NG2", "izuna drop counters", "hyper-armor reliability"],
      "message_ids": [101, 102, 103]
    },
    {
      "name": "NG4 stage 3 boss difficulty",
      "drivers": ["carol", "bob"],
      "msg_count": 2,
      "is_sensitive": false,
      "brief": "Carol asks if anyone else is struggling with the stage 3 boss; Bob agrees it's a wall.",
      "angles": ["stage 3 boss as a difficulty wall"],
      "message_ids": [104, 105]
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
      "angles": ["challenging boss fights"],     // generic, not from messages
      "message_ids": [101, 102, 103, 104, 105]   // lumps two distinct subjects
    }
  ]
}

Why it's bad: subject name is too vague; "DLC" is invented (not in messages); "you should" is advice not observation; angles are generic instead of message-specific; message_ids lump two distinct conversations into one row.

EXAMPLE 3 — GOOD (correct skip when chat is just lurkers)

Input messages:
  [id 200] [dave] hi
  [id 201] [eve] just got here
  [id 202] [frank] lol
  [id 203] [dave] !lurk

Good output:
{ "subjects": [] }

(Empty list is correct — pure greetings and bot commands aren't subjects.)

EXAMPLE 4 — GOOD (filtering sensitive)

Input:
  [id 300] [g] honestly the election was rigged
  [id 301] [h] dont start
  [id 302] [i] meanwhile the new patch dropped
  [id 303] [j] yeah finally fixed the lag

Good output:
{
  "subjects": [
    {
      "name": "new game patch fixing lag",
      "drivers": ["i", "j"],
      "msg_count": 2,
      "is_sensitive": false,
      "brief": "Chatters note the new patch dropped and observe it fixed lag they'd been seeing.",
      "angles": ["lag improvement"],
      "message_ids": [302, 303]
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
        # Negative supervision for the engaging-subjects extractor:
        # streamer judged this subject as wrong/hallucinated. Captured
        # for the personal training dataset so future fine-tunes can
        # learn what NOT to surface.
        self.repo._capture_streamer_action(
            action_kind="engaging_subject",
            item_key=slug_lc,
            action="rejected",
            note=name.strip(),
        )

    def clear_subject_blocklist(self) -> int:
        """Wipe all rejections — used when the streamer wants to reset
        (e.g., new stream, different topic). Returns the count cleared."""
        n = len(self._load_subject_blocklist())
        self.repo.set_app_setting(self.SUBJECTS_BLOCKLIST_KEY, "[]")
        # Reset signal — the streamer wants a fresh slate. Useful for
        # the dataset reader to know "everything before this point is
        # a different feedback regime."
        self.repo._capture_streamer_action(
            action_kind="engaging_subject_blocklist",
            item_key="",
            action="cleared",
            note=None,
        )
        return n

    # =========================================================
    # Per-subject talking points — on-demand modal LLM call.
    # Distinct from the per-chatter `TalkingPointsResponse` (which
    # produces hooks for individual chatters). This call asks the
    # model "what could the streamer say ABOUT this subject, given
    # the room's actual messages?" — output is 3-5 short
    # observational lines the streamer can offer back to chat.
    # =========================================================

    SUBJECT_TALKING_POINTS_NUM_CTX = INFORMED_NUM_CTX

    SUBJECT_TALKING_POINTS_SYSTEM = """You help a Twitch streamer think of things to say about a conversation chat is having right now. The streamer reads your output on a second monitor and uses it to engage with chat — your output never returns to chat itself.

You'll receive:
  - SUBJECT — the conversation chat is having (a short subject line)
  - BRIEF — a 1-2 sentence observational summary of what's been said
  - SUB-ASPECTS — up to 4 specific angles that have come up in the messages
  - MESSAGES — the verbatim chat messages that ground the subject (numbered)
  - Optional context blocks (use as silent grounding, do not parrot back):
    - CHANNEL CONTEXT — what the streamer is playing / streaming right now
    - STREAMER VOICE — a recent transcript of what the streamer just said
    - An attached screenshot — what's on screen right now

Produce 3-5 short TALKING POINTS the streamer could offer back to chat. Each is one line, under 25 words, written as something the streamer might naturally say (NOT as advice TO the streamer).

GOOD shape:
  - "I've found the parry timing way easier with claymore than with kunai."
  - "Ngl, I keep dying to that boss too — anyone running a different setup?"
  - "Worth noting the patch nerfed izuna's hyper-armor; that's why it feels off."

HARD RULES (violations make the dashboard worse, not better):
- ONLY paraphrase content visible in MESSAGES, BRIEF, SUB-ASPECTS, or the streamer's own transcripts. Do not bridge, infer, or extrapolate beyond that.
- NEVER invent products, releases, events, places, people, or facts that aren't directly attested. If chatters talk about an "FF8 remake," you do NOT know one exists — phrase as "they mentioned an FF8 remake" or "if there's an FF8 remake."
- Stay observational + grounded. Each point should be something the streamer can say WITHOUT putting words in chatters' mouths.
- Don't tell the streamer what to do ("you should ask…", "consider mentioning…"). Write the point AS the streamer would say it.
- Don't repeat the same point with different wording.
- Empty `points` list is acceptable when the subject is too thin to grounded-riff on (rare — usually the subject has enough material by the time it's surfaced).

When in doubt, fewer high-quality points beats more weak ones.
"""

    async def generate_subject_talking_points(
        self, slug: str, *, force: bool = False,
    ) -> tuple[list[str], str | None]:
        """Generate per-subject talking points for the modal.

        Returns `(points, error)` — `error` is None on success, a
        short string when the LLM call failed (caller surfaces it
        on the modal so the streamer knows why the section is
        empty rather than hallucinating placeholder content).

        Caches on the persistent subject's `talking_points` field.
        Cache invalidates when `last_seen_ts` has advanced past
        `talking_points_at` — i.e., the engaging-subjects refresh
        surfaced this subject with new chat context since we last
        generated. `force=True` bypasses the cache (used by a
        streamer-driven "regenerate" button if we add one later).

        Lookups by slug because that's what the dashboard knows;
        the slug is sha1(name.lower())[:12] which we hash on the
        fly to avoid storing slugs on `_PersistentSubject` (the
        name is the source of truth — slug is for transport)."""
        import hashlib as _h

        # Find the persistent subject whose name matches the slug.
        # Walk the dict — typically <50 entries, so O(n) lookup is
        # fine and avoids carrying a slug index that could go
        # stale during identity matching renames.
        match: _PersistentSubject | None = None
        for s in self._subjects.values():
            if _h.sha1(s.name.lower().encode("utf-8")).hexdigest()[:12] == slug:
                match = s
                break
        if match is None:
            return [], "subject not found"
        if match.is_sensitive:
            # Defense in depth — sensitive subjects shouldn't be
            # queryable through the modal.
            return [], "subject is filtered"

        # Cache hit: same slug, no new chat context since the last
        # generation. Re-opening the modal is free.
        if (
            not force
            and match.talking_points
            and match.talking_points_at >= match.last_seen_ts
        ):
            return list(match.talking_points), None

        # Hydrate the cited messages (current persistent msg_ids
        # cover historical refreshes too — gives the LLM more
        # signal than just the most-recent window).
        try:
            msgs = await asyncio.to_thread(
                self.repo.get_messages_by_ids, list(match.msg_ids[-50:]),
            )
        except Exception:
            logger.exception("subject talking points: hydrate failed")
            msgs = []

        if not msgs:
            return [], "no chat context to ground on"

        # Build the prompt blocks. Reuses the same context shape
        # the engaging-subjects pass uses — channel context +
        # streamer voice + numbered messages — so output style
        # stays consistent with the rest of the dashboard's LLM
        # surfaces.
        channel_context = self._build_channel_context(authoritative=False)
        transcript_block = await self._latest_transcript_summary()
        grid_b64 = await self._latest_screenshot_grid(window_minutes=20)

        message_lines = "\n".join(
            f"  [id {m.id}] [{m.name}] {m.content[:240]}"
            for m in msgs[-30:]
        )
        angles_lines = (
            "\n".join(f"  - {a}" for a in match.angles)
            if match.angles else "  (none extracted)"
        )

        prompt = (
            channel_context
            + transcript_block
            + f"SUBJECT: {match.name}\n\n"
            + f"BRIEF: {match.brief or '(no brief)'}\n\n"
            + f"SUB-ASPECTS:\n{angles_lines}\n\n"
            + f"MESSAGES (numbered):\n{message_lines}\n\n"
            + "Produce 3-5 short talking points the streamer could "
            "say back to chat about this subject. Each one paraphrases "
            "something visible in the input — no invented facts."
        )
        if grid_b64:
            prompt += (
                "\n\nAn image is attached showing what is currently "
                "on screen. Use it ONLY as silent context. Do NOT "
                "describe the image."
            )

        from .llm.schemas import SubjectTalkingPointsResponse
        from .llm.prompts import resolve_prompt
        try:
            response = await self.llm.generate_structured(
                prompt=prompt,
                system_prompt=await self._streamer_facts_prepended(
                    resolve_prompt(
                        "insights.subject_talking_points", self.repo,
                    ),
                ),
                response_model=SubjectTalkingPointsResponse,
                num_ctx=self.SUBJECT_TALKING_POINTS_NUM_CTX,
                images=[grid_b64] if grid_b64 else None,
                think=True,
                call_site="insights.subject_talking_points",
            )
        except ValidationError as e:
            logger.warning("subject-talking-points validation failed: %s", e)
            return [], "the model's response didn't validate"
        except Exception:
            logger.exception("subject-talking-points LLM call failed")
            return [], "LLM call failed (check server logs)"

        points = [p.strip() for p in response.points if p.strip()]
        match.talking_points = points
        match.talking_points_at = match.last_seen_ts or time.time()
        return list(points), None

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
        """Load streamer-authored channel facts from disk, cached with
        mtime so hot LLM call sites don't re-read the file every pass.
        Missing / empty / unreadable returns "" — caller falls back
        to the default prompt without a facts block.

        Used by `_streamer_facts_prepended` to inject channel-specific
        context (recurring bits, current arcs, inside jokes) into the
        system prompt of every LLM call where streamer-grounding helps:
        engaging-subjects, talking points, thread recaps, subject
        talking-points, and question answer-angles."""
        from pathlib import Path
        path_str = getattr(self.settings, "streamer_facts_path", "") or ""
        if not path_str:
            self._facts_text = ""
            self._facts_mtime = 0.0
            return ""
        p = Path(path_str)
        if not p.is_absolute():
            # Resolve relative to the chatterbot project root (the
            # working directory the dashboard runs from).
            p = Path.cwd() / p
        try:
            mtime = p.stat().st_mtime
        except OSError:
            self._facts_text = ""
            self._facts_mtime = 0.0
            return ""
        if mtime == self._facts_mtime and self._facts_text:
            return self._facts_text
        try:
            text = p.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return ""
        # Cap at 4k chars so a runaway file can't blow the context budget.
        if len(text) > 4000:
            text = text[:4000] + "\n\n[…truncated…]"
        self._facts_text = text
        self._facts_mtime = mtime
        return text

    async def _streamer_facts_prepended(self, system_prompt: str) -> str:
        """Prepend the STREAMER-AUTHORED CHANNEL FACTS block to a
        system prompt when streamer_facts.md is configured and
        non-empty. Returns the original prompt unchanged when no
        facts are present, so empty / missing files silently no-op
        without changing model behavior.

        Reads happen via `asyncio.to_thread` because callers are
        async and the cache-miss path hits disk; cache-hit is a
        sync dict load (cheap)."""
        facts = await asyncio.to_thread(self._load_streamer_facts)
        if not facts:
            return system_prompt
        return (
            "STREAMER-AUTHORED CHANNEL FACTS (treat as ground "
            "truth about the streamer / channel — chat references "
            "to these are real, not hallucinations to be flagged):\n\n"
            + facts
            + "\n\n==================================================================\n\n"
            + system_prompt
        )

    @staticmethod
    def _unit_vec(v: list[float]):
        """Unit-normalize a vector, returning the original on zero-
        norm (caller treats this as "no usable embedding")."""
        try:
            import numpy as np
        except ImportError:
            return None
        arr = np.array(v, dtype=np.float32)
        n = float(np.linalg.norm(arr))
        return (arr / n).tolist() if n > 0 else None

    async def _embed_subject_name(self, name: str) -> list[float] | None:
        """Embed a subject NAME for cross-refresh identity matching.
        Falls back to None on any failure — the caller treats that as
        "spawn a new persistent identity" rather than blocking on
        embedding hiccups."""
        try:
            raw = await self.llm.embed(name.strip())
        except Exception:
            return None
        return self._unit_vec(raw)

    def _match_subject_by_name_embedding(
        self, query_unit: list[float], threshold: float = 0.75,
    ) -> tuple[str | None, float]:
        """Find the persistent subject whose name-embedding is most
        similar to `query_unit`. Returns `(subject_id, sim)` if the
        best match clears `threshold`, otherwise `(None, best_sim)`
        so the caller can log near-misses if they want.

        Threshold 0.75 is empirically the right cut for nomic-embed-
        text on subject-line-length strings:
          - "RE9 parry timing" vs "Resident Evil 9 parry windows"  → ~0.82
          - "RE9 parry timing" vs "RE9 boss strats"                → ~0.55
          - "RE9 parry timing" vs "Final Fantasy speedrun routes"  → ~0.20

        Same-subject rephrasing clears 0.75 comfortably; genuinely-
        different subjects don't. Tunable via app_setting if needed."""
        if not self._subjects or query_unit is None:
            return None, 0.0
        try:
            import numpy as np
        except ImportError:
            return None, 0.0
        ids: list[str] = []
        vecs: list[list[float]] = []
        for sid, s in self._subjects.items():
            if s.name_embedding:
                ids.append(sid)
                vecs.append(s.name_embedding)
        if not vecs:
            return None, 0.0
        q = np.asarray(query_unit, dtype=np.float32)
        mat = np.asarray(vecs, dtype=np.float32)
        sims = mat @ q
        best = int(np.argmax(sims))
        best_sim = float(sims[best])
        if best_sim >= threshold:
            return ids[best], best_sim
        return None, best_sim

    def _age_out_persistent_subjects(self, *, max_idle_seconds: float) -> int:
        """Drop persistent subjects whose `last_seen_ts` is older
        than `max_idle_seconds`. Returns the number removed.

        Aging is by REFRESH presence, not message attachment: a
        subject ages out when N consecutive refreshes have failed to
        surface it, NOT when N minutes pass since the last cited
        message. This is more robust to noisy refresh cadence."""
        if not self._subjects:
            return 0
        now = time.time()
        stale = [
            sid for sid, s in self._subjects.items()
            if s.last_seen_ts and (now - s.last_seen_ts) > max_idle_seconds
        ]
        for sid in stale:
            self._subjects.pop(sid, None)
        return len(stale)

    async def _refresh_engaging_subjects(self) -> None:
        """One refresh pass over the engaging-subjects pipeline.

        High-level shape (LLM-first, embedding-for-identity):

          1. Pull the recent-messages window (no clustering).
          2. Skip the LLM entirely when latest_message_id hasn't
             moved since last pass — same input, same output.
          3. Build a numbered-messages prompt + the usual context
             blocks (channel + streamer voice + driver notes +
             known subjects + blocklist).
          4. Call the LLM. It returns subjects with cited
             message_ids (no cluster matching, no positional voodoo).
          5. For each returned subject: embed the NAME and look for
             a match in `_subjects` (cos sim ≥ 0.75) → reuse
             persistent ID, else spawn a new one.
          6. Materialize the cache from this-pass's surfaced
             subjects (with drivers resolved from cited message_ids
             → user_ids → names).
          7. Age out persistent subjects unseen for 2x the lookback
             window."""
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
                msgs = await asyncio.to_thread(
                    self.repo.recent_messages,
                    limit=limit, within_minutes=window_min,
                )
            except Exception:
                logger.exception(
                    "engaging-subjects: recent_messages failed",
                )
                return
            if len(msgs) < 5:
                # Too quiet to bother. Reset the cache so the panel
                # shows the empty state instead of stale subjects.
                # Persistent identities stay (they age out by
                # last_seen_ts, not message volume).
                self._subjects_cache = EngagingSubjectsCache(
                    subjects=[], refreshed_at=time.time(), error=None,
                )
                return

            # Skip-if-unchanged: when the highest message id hasn't
            # moved since our last refresh, the inputs are identical
            # and the LLM would produce identical output. Cheaper to
            # serve the existing cache. Same trick the talking-points
            # loop uses.
            current_max = max(m.id for m in msgs)
            if current_max == self._subjects_last_msg_id and self._subjects_cache.subjects:
                logger.debug(
                    "engaging-subjects: skip — latest_message_id "
                    "unchanged (%d), cached subjects=%d",
                    current_max, len(self._subjects_cache.subjects),
                )
                return
            self._subjects_last_msg_id = current_max

            # --- Resolve user names (one DB hit per distinct user) ---
            user_id_to_name: dict[str, str] = {}
            for m in msgs:
                if m.user_id and m.user_id not in user_id_to_name:
                    u = await asyncio.to_thread(self.repo.get_user, m.user_id)
                    if u:
                        user_id_to_name[m.user_id] = u.name
            id_to_msg = {m.id: m for m in msgs}

            # --- Streamer voice (transcript chunks) ---
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

            # --- Per-driver notes (background on most-active chatters) ---
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

            # --- Known active subjects (channel-wide topic threads
            # + still-alive persistent subjects) so the LLM matches
            # new messages to existing wording rather than inventing
            # near-duplicate names. ---
            try:
                active_threads = await asyncio.to_thread(
                    self.repo.list_threads,
                    status_filter="active", query="", limit=10,
                )
            except Exception:
                active_threads = []
            known_titles: list[str] = [t.title for t in active_threads if t.title]
            for s in self._subjects.values():
                if s.name and not s.is_sensitive and s.name not in known_titles:
                    known_titles.append(s.name)
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

            # --- Blocklist injection ---
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

            channel_context = self._build_channel_context(authoritative=True)

            # --- Numbered messages block. The LLM does the topic
            # modeling directly over these and cites message_ids
            # (the integer at the start of each line) for each
            # subject it returns. No clustering on our side. ---
            #
            # Cap at SUBJECTS_MAX_MESSAGES (default 250) which fits
            # comfortably in INFORMED_NUM_CTX (32k) — typical msg is
            # ~50 chars, so 250 msgs ≈ 12 KB plus the per-line `[id N]
            # [name]` prefix. Plenty of headroom for the surrounding
            # context blocks.
            messages_block = "\n".join(
                f"  [id {m.id}] [{user_id_to_name.get(m.user_id, m.user_id)}] "
                f"{m.content[:240]}"
                for m in msgs[-limit:]
            )

            prompt = (
                channel_context
                + transcript_section
                + known_section
                + driver_notes_section
                + f"RECENT CHAT (last {window_min} min, numbered by `id` — "
                "cite ids in your `message_ids` for each subject):\n\n"
                + messages_block
                + blocklist_lines
                + "\n\nReturn distinct subjects you can cite supporting "
                "message_ids for. 0-8 subjects; empty list is fine when "
                "chat is too unfocused."
            )

            from .llm.prompts import resolve_prompt
            system_prompt = await self._streamer_facts_prepended(
                resolve_prompt("insights.engaging_subjects", self.repo),
            )

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

            # --- LLM call ---
            response_subjects: list = []
            try:
                from .llm.schemas import EngagingSubjectsResponse
                response = await self.llm.generate_structured(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    response_model=EngagingSubjectsResponse,
                    num_ctx=self.SUBJECTS_NUM_CTX,
                    images=[grid_b64] if grid_b64 else None,
                    think=True,
                    call_site="insights.engaging_subjects",
                )
                response_subjects = response.subjects
            except ValidationError as e:
                logger.warning("engaging-subjects validation failed: %s", e)
            except Exception:
                logger.exception("engaging-subjects LLM call failed")

            # --- Apply: match each returned subject to a persistent
            # identity (or spawn a new one) and update its state. ---
            await self._ensure_blocklist_embeddings(blocklist)
            block_set_slug = {(b.get("slug") or "").lower() for b in blocklist}
            block_set_name = {(b.get("name") or "").lower() for b in blocklist}
            now_ts = time.time()
            applied: list[tuple[_PersistentSubject, list[int]]] = []
            dropped_blocklist = 0
            dropped_sensitive = 0
            dropped_orphan_msgs = 0
            valid_msg_ids = set(id_to_msg.keys())

            for s in response_subjects:
                clean_name = (s.name or "").strip()
                if not clean_name:
                    continue
                # Filter out sensitive/blocked before doing any
                # embedding work — cheap shortcuts first.
                if s.is_sensitive:
                    dropped_sensitive += 1
                    continue
                if clean_name.lower() in block_set_name:
                    dropped_blocklist += 1
                    logger.info(
                        "engaging-subjects: dropped blocklisted subject %r",
                        clean_name,
                    )
                    continue
                # Embedding-based blocklist near-dup check (catches
                # rephrased rejections like "FF8 remake" vs "Final
                # Fantasy 8 remake"). 0.85 is intentionally strict.
                blocked_by_emb, matched_slug, sim = (
                    await self._is_blocked_by_embedding(clean_name)
                )
                if blocked_by_emb:
                    dropped_blocklist += 1
                    logger.info(
                        "engaging-subjects: dropped near-duplicate of "
                        "blocklisted subject %r (matched %r via embedding "
                        "sim=%.2f)", clean_name, matched_slug, sim,
                    )
                    continue
                # Filter cited message_ids to ones actually in the
                # current window. The LLM is supposed to cite only
                # input ids but a hallucinated id should be ignored
                # rather than crash.
                cited = [
                    int(mid) for mid in (s.message_ids or [])
                    if isinstance(mid, int) and mid in valid_msg_ids
                ]
                if not cited:
                    # Subject with zero grounded message_ids is too
                    # weak to surface — there's nothing for the
                    # streamer to click into. The system prompt
                    # tells the LLM "at least 2 supporting ids"
                    # already; this is the final guard.
                    dropped_orphan_msgs += 1
                    continue

                # --- Persistent identity match by name embedding ---
                name_unit = await self._embed_subject_name(clean_name)
                identity_threshold = float(getattr(
                    self.settings, "engaging_subjects_identity_threshold", 0.75,
                ))
                matched_id, _ = self._match_subject_by_name_embedding(
                    name_unit or [], threshold=identity_threshold,
                )

                if matched_id and matched_id in self._subjects:
                    subj = self._subjects[matched_id]
                    # Reuse existing identity. Update name to the
                    # current LLM phrasing (often slight evolution),
                    # refresh brief/angles, bump counters.
                    subj.name = clean_name
                    subj.brief = (s.brief or "").strip()
                    subj.angles = [
                        a.strip() for a in (s.angles or []) if a and a.strip()
                    ][:4]
                    if name_unit is not None:
                        subj.name_embedding = name_unit
                    subj.is_sensitive = False
                    subj.last_seen_ts = now_ts
                    subj.refresh_count += 1
                    # Append cited message_ids; cap to keep the
                    # persistent record bounded.
                    existing = set(subj.msg_ids)
                    for mid in cited:
                        if mid not in existing:
                            subj.msg_ids.append(mid)
                            existing.add(mid)
                    if len(subj.msg_ids) > 200:
                        subj.msg_ids = subj.msg_ids[-200:]
                else:
                    # Spawn new persistent identity. If we couldn't
                    # embed the name, fall back to an empty embedding
                    # — next refresh's identity match will retry.
                    sid = uuid.uuid4().hex[:12]
                    subj = _PersistentSubject(
                        subject_id=sid,
                        name=clean_name,
                        name_embedding=name_unit or [],
                        brief=(s.brief or "").strip(),
                        angles=[
                            a.strip() for a in (s.angles or []) if a and a.strip()
                        ][:4],
                        is_sensitive=False,
                        first_seen_ts=now_ts,
                        last_seen_ts=now_ts,
                        refresh_count=1,
                        msg_ids=list(cited),
                    )
                    self._subjects[sid] = subj
                applied.append((subj, cited))

            # --- Age out persistent subjects unseen for 2× window ---
            aged = self._age_out_persistent_subjects(
                max_idle_seconds=window_min * 60 * 2,
            )

            # --- Materialize the cache from THIS pass's surfaced
            # subjects, with drivers resolved from cited message_ids. ---
            import hashlib as _h
            entries: list[EngagingSubjectEntry] = []
            for subj, cited in applied:
                slug = _h.sha1(subj.name.lower().encode("utf-8")).hexdigest()[:12]
                if slug in block_set_slug or subj.name.lower() in block_set_name:
                    continue
                # Drivers: walk cited message_ids → user_ids →
                # names. Newest cited first so the most-recent
                # contributor shows first in the UI.
                drivers: list[str] = []
                seen_drivers: set[str] = set()
                for mid in reversed(cited):
                    msg = id_to_msg.get(mid)
                    if msg is None:
                        continue
                    name = user_id_to_name.get(msg.user_id)
                    if name and name.lower() not in seen_drivers:
                        drivers.append(name)
                        seen_drivers.add(name.lower())
                if not drivers:
                    continue
                entries.append(EngagingSubjectEntry(
                    name=subj.name,
                    drivers=drivers[:20],
                    msg_count=len(cited),
                    brief=subj.brief,
                    angles=list(subj.angles),
                    slug=slug,
                    msg_ids=list(cited),
                ))
            entries.sort(key=lambda e: (-len(e.drivers), -e.msg_count))
            entries = entries[:8]

            self._subjects_cache = EngagingSubjectsCache(
                subjects=entries, refreshed_at=time.time(), error=None,
            )

            # Single info log per refresh — covers every gate so a
            # future "why is the panel empty" investigation can be
            # answered from this line alone.
            logger.info(
                "engaging subjects refreshed: %d msgs → LLM returned %d, "
                "applied %d (dropped %d sensitive, %d blocklisted, "
                "%d ungrounded), cached %d, persistent %d (aged out %d)",
                len(msgs), len(response_subjects), len(applied),
                dropped_sensitive, dropped_blocklist, dropped_orphan_msgs,
                len(entries), len(self._subjects), aged,
            )

    # =========================================================
    # Open chat questions — LLM filter pass over the heuristic
    # `recent_questions` candidates. The repo helper clusters by
    # token overlap and excludes @-mentions / Twitch reply rows;
    # this pass uses full chat + transcript context to drop
    # already-answered, rhetorical, or directed-at-other-chatter
    # questions that the heuristic can't catch.
    # =========================================================

    OPEN_QUESTIONS_NUM_CTX = INFORMED_NUM_CTX
    OPEN_QUESTIONS_LOOKBACK_MINUTES = 15
    OPEN_QUESTIONS_MAX_CANDIDATES = 12
    OPEN_QUESTIONS_MAX_CONTEXT_MSGS = 200

    OPEN_QUESTIONS_SYSTEM = """You're filtering chat questions for a Twitch streamer's dashboard. The dashboard surfaces OPEN questions chat is asking the streamer (or the room) so the streamer can answer the ones that matter. Your output never returns to chat.

You'll receive:
  - A numbered list of CANDIDATE QUESTIONS — each is the representative wording from a token-overlap cluster of similar questions chat asked recently. Each candidate has an integer id (`[N]`), the askers, and the timestamp of the most recent ask.
  - The full window of recent CHAT MESSAGES (oldest → newest) so you can see what was said before AND after each question — including any answers already given.
  - STREAMER VOICE — what the streamer said out loud during the same window. The streamer often answers questions verbally without typing.
  - CHANNEL CONTEXT — what's being streamed right now (helps you tell genuine questions from rhetorical reactions).

Return ONLY candidates that are still GENUINELY OPEN.

DROP a candidate when ANY of these apply:
- It's already been answered in chat (another chatter or the streamer typed an answer after the ask).
- The streamer answered it out loud — STREAMER VOICE shows them addressing the topic after the ask.
- It's directed at a SPECIFIC OTHER CHATTER, not the streamer or the room (e.g., "@bob you good?", "tom how was your run"). The SQL filter catches `@`-prefixed and Twitch-reply rows but contextual @-replies still slip through.
- It's rhetorical / not seeking an answer ("can you believe that?", "wait what?", "really??").
- It's pure game commentary or a reaction ("how did he die there?", "why is this boss so hard?") — not a question for the streamer.
- It's a bot command, meta-noise, or untranslatable spam.

KEEP a candidate when:
- It's a question chat is asking THE STREAMER or THE ROOM broadly.
- No clear answer has been given yet — neither in chat nor in the streamer's voice.
- The streamer would benefit from a nudge to address it.

For each kept candidate:
  - candidate_id: ECHO the integer `[N]` from the input. The dashboard uses this to re-attach the original askers + timestamps. If you can't ground the question in a specific candidate, drop it — do NOT invent ids.
  - question: the wording chat is asking. You may LIGHTLY clean it (capitalize the first letter, fix obvious typos, drop trailing filler) and you may merge two candidates with the same intent into one entry by picking either id and using a unified wording. You may NOT invent a question that no chatter actually asked.

Return AT MOST 8 questions. Order them most-recently-asked first when in doubt.

When uncertain, drop. Chat will re-ask anything that matters; surfacing an already-answered question wastes the streamer's attention.
"""

    async def open_questions_loop(
        self, interval_seconds: int | None = None,
    ) -> None:
        """Background task: periodically refresh the open-questions
        cache. No-op when the channel is quiet."""
        await asyncio.sleep(20)
        while True:
            try:
                interval = max(60, int(
                    interval_seconds
                    or getattr(self.settings, "open_questions_interval_seconds", 180)
                ))
                if interval <= 0:
                    return
                await self._refresh_open_questions()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("open-questions iteration failed")
            await asyncio.sleep(interval)

    async def _refresh_open_questions(self) -> None:
        async with self._questions_lock:
            window_min = int(getattr(
                self.settings, "open_questions_lookback_minutes",
                self.OPEN_QUESTIONS_LOOKBACK_MINUTES,
            ))
            try:
                candidates = await asyncio.to_thread(
                    self.repo.recent_questions,
                    within_minutes=window_min,
                    limit=self.OPEN_QUESTIONS_MAX_CANDIDATES,
                )
            except Exception:
                logger.exception("open-questions: recent_questions failed")
                return
            if not candidates:
                # Nothing to filter — flush the cache so the panel
                # shows the empty state instead of stale entries.
                self._questions_cache = OpenQuestionsCache(
                    questions=[], refreshed_at=time.time(), error=None,
                )
                return

            # Pull surrounding chat (full window, oldest-first) so the
            # LLM can spot answers given AFTER each question. Cap is
            # generous enough to cover busy windows but bounded so the
            # prompt stays inside INFORMED_NUM_CTX.
            try:
                chat_msgs = await asyncio.to_thread(
                    self.repo.recent_messages,
                    limit=self.OPEN_QUESTIONS_MAX_CONTEXT_MSGS,
                    within_minutes=window_min,
                )
            except Exception:
                logger.exception("open-questions: recent_messages failed")
                chat_msgs = []

            # Streamer voice — they often answer questions verbally
            # without typing. Same lookback window so the LLM can match
            # transcripts to specific asks by timestamp.
            try:
                transcripts = await asyncio.to_thread(
                    self.repo.recent_transcripts,
                    within_minutes=window_min, limit=80,
                )
            except Exception:
                transcripts = []

            # Build the prompt. Each candidate gets its [N] id (=
            # last_msg_id) so the LLM can echo it back.
            cand_blocks: list[str] = []
            cand_by_id: dict[int, dict] = {}
            for c in candidates:
                cid = int(c["last_msg_id"])
                cand_by_id[cid] = c
                askers = ", ".join(d["name"] for d in c["drivers"]) or "?"
                cand_blocks.append(
                    f"  [{cid}] (asked by {askers}; latest ask {c['latest_ts']}; "
                    f"×{c['count']} similar)\n      {c['question'][:240]}"
                )

            chat_lines = [
                f"  [{m.id}] {m.ts} <{m.name}> {m.content[:200]}"
                for m in chat_msgs
            ]
            chat_section = (
                f"RECENT CHAT (oldest → newest, last {window_min} min — use "
                "to spot answers given AFTER each candidate's latest ask):\n"
                + "\n".join(chat_lines)
                + "\n\n"
                if chat_lines else ""
            )

            transcript_section = ""
            if transcripts:
                tlines = [
                    f"  - {t.text[:240].strip()}"
                    for t in transcripts if t.text and t.text.strip()
                ]
                if tlines:
                    transcript_section = (
                        f"STREAMER VOICE (last {window_min} min, oldest first "
                        "— what the streamer said out loud; if they verbally "
                        "answered a candidate, drop it):\n"
                        + "\n".join(tlines[-60:])
                        + "\n\n"
                    )

            channel_context = self._build_channel_context(authoritative=False)

            prompt = (
                channel_context
                + chat_section
                + transcript_section
                + "CANDIDATE QUESTIONS (from token-overlap clustering of "
                "recent chat — each line: `[id] (askers; latest ts; count) "
                "question`). Echo the `[id]` back as `candidate_id` for the "
                "ones you keep:\n\n"
                + "\n\n".join(cand_blocks)
                + "\n\nReturn only the candidates that are still genuinely "
                "open. Drop already-answered, directed-at-other-chatter, "
                "rhetorical, and reaction questions."
            )

            from .llm.schemas import OpenQuestionsResponse
            try:
                from .llm.prompts import resolve_prompt
                response = await self.llm.generate_structured(
                    prompt=prompt,
                    system_prompt=resolve_prompt(
                        "insights.open_questions", self.repo,
                    ),
                    response_model=OpenQuestionsResponse,
                    num_ctx=self.OPEN_QUESTIONS_NUM_CTX,
                    think=True,
                    call_site="insights.open_questions",
                )
            except ValidationError as e:
                logger.warning("open-questions validation failed: %s", e)
                self._questions_cache = OpenQuestionsCache(
                    questions=[], refreshed_at=time.time(),
                    error=f"validation failed: {e!s}",
                )
                return
            except Exception as e:
                logger.exception("open-questions LLM call failed")
                self._questions_cache = OpenQuestionsCache(
                    questions=[], refreshed_at=time.time(), error=str(e),
                )
                return

            # Re-attach drivers from the original cluster. Drop any
            # candidate_id the LLM hallucinated.
            entries: list[OpenQuestionEntry] = []
            seen_ids: set[int] = set()
            for q in response.questions:
                if q.candidate_id in seen_ids:
                    continue  # LLM merged two cands onto one id; keep first
                cand = cand_by_id.get(int(q.candidate_id))
                if cand is None:
                    continue
                seen_ids.add(int(q.candidate_id))
                drivers = [
                    OpenQuestionDriver(
                        name=d["name"], user_id=d["user_id"], ts=d["ts"],
                    )
                    for d in cand["drivers"]
                ]
                entries.append(OpenQuestionEntry(
                    question=q.question.strip() or cand["question"],
                    count=int(cand["count"]),
                    drivers=drivers,
                    latest_ts=cand["latest_ts"],
                    last_msg_id=int(cand["last_msg_id"]),
                ))
            # Sort newest-ask first (matches the heuristic's tiebreak).
            entries.sort(key=lambda e: e.latest_ts, reverse=True)
            entries = entries[:8]

            self._questions_cache = OpenQuestionsCache(
                questions=entries, refreshed_at=time.time(), error=None,
            )
            # Prune the answer-angles cache to only entries still
            # surfaced. Bounds the cache to current panel size +
            # release LLM-generated content for clusters that have
            # since been answered or aged out.
            live_ids = {e.last_msg_id for e in entries}
            stale = [k for k in self._question_angles_cache if k not in live_ids]
            for k in stale:
                self._question_angles_cache.pop(k, None)
            logger.info(
                "open questions refreshed: %d candidates → %d open "
                "(LLM kept %d, dropped %d)",
                len(candidates), len(entries),
                len(response.questions),
                max(0, len(response.questions) - len(entries)),
            )

    # =========================================================
    # Per-question answer angles — async-loaded by the open-question
    # modal. Generates 3-5 short bullets the streamer could offer
    # back to chat. Cached on _question_angles_cache by last_msg_id;
    # cache invalidates when the question drops off the panel.
    # =========================================================

    QUESTION_ANSWER_ANGLES_NUM_CTX = INFORMED_NUM_CTX

    QUESTION_ANSWER_ANGLES_SYSTEM = """You're helping a Twitch streamer figure out angles for answering one specific chat question. Your output is shown ONLY on the streamer's private dashboard — it never returns to chat. The streamer reads your bullets and chooses how to actually respond out loud.

You'll receive:
  - QUESTION — the wording chat is asking.
  - VERBATIM ASKS — the actual chat messages from the chatters who asked it (so you can read tone + specifics, not just the cleaned-up wording).
  - RECENT CHAT — the surrounding window, oldest → newest.
  - STREAMER VOICE — what the streamer has said out loud recently.
  - CHANNEL CONTEXT — what's currently being streamed.

Produce 3-5 SHORT angles (one line each) the streamer could OFFER BACK. Angles ≠ scripts:
- Each angle is one direction the answer could go ("share what your current setup is", "ask chat what they'd recommend", "give the short answer + a fun aside about X").
- Phrase as a starter / direction, not a full sentence the streamer has to read aloud verbatim.
- Stay grounded in the streamer's actual context — don't invent facts. If the streamer's voice has already touched the topic, mention that as one angle.
- Mix shapes when possible: a direct answer angle, a turn-it-back-to-chat angle, a tangent angle.

If the question is genuinely unanswerable from the available context (asking about something the streamer hasn't shown / talked about), say so as one of the angles ("acknowledge you don't have a take on this yet") instead of fabricating.

Return AT MOST 5 angles. 3 is a fine answer when the question is narrow.
"""

    async def generate_question_answer_angles(
        self, last_msg_id: int, *, force: bool = False,
    ) -> tuple[list[str], str | None]:
        """Generate per-question answer angles for the open-question
        modal.

        Returns `(angles, error)` — `error` is None on success, a
        short string when the LLM call failed (caller surfaces it on
        the modal so the streamer knows why the section is empty).

        Cached on `_question_angles_cache` keyed by last_msg_id. Same
        question identity (same last_msg_id) re-opens for free. The
        cache is pruned in `_refresh_open_questions` when the question
        drops off the panel."""
        # Cache hit — re-opens within the same question identity are
        # free, no LLM call.
        if not force and last_msg_id in self._question_angles_cache:
            return list(self._question_angles_cache[last_msg_id]), None

        # Find the entry. The questions cache is small (<= 8) so a
        # linear walk is fine.
        entry: OpenQuestionEntry | None = next(
            (q for q in self._questions_cache.questions
             if q.last_msg_id == last_msg_id),
            None,
        )
        if entry is None:
            return [], "question not found in cache"

        # Pull surrounding context. Same lookback as the filter pass
        # so the LLM sees the same chat the filter saw + a bit more.
        window_min = int(getattr(
            self.settings, "open_questions_lookback_minutes",
            self.OPEN_QUESTIONS_LOOKBACK_MINUTES,
        ))
        try:
            chat_msgs = await asyncio.to_thread(
                self.repo.recent_messages,
                limit=120, within_minutes=window_min,
            )
        except Exception:
            logger.exception("question-angles: recent_messages failed")
            chat_msgs = []
        try:
            transcripts = await asyncio.to_thread(
                self.repo.recent_transcripts,
                within_minutes=window_min, limit=60,
            )
        except Exception:
            transcripts = []

        # Verbatim asks — pull messages from the cluster's drivers
        # that contain '?' within the window. Heuristic but good
        # enough; the cluster identity is by token-overlap not by
        # exact id list.
        driver_ids = {d.user_id for d in entry.drivers}
        verbatim_asks: list = []
        for m in chat_msgs:
            if m.user_id in driver_ids and "?" in (m.content or ""):
                verbatim_asks.append(m)
        # Cap so the prompt stays bounded on a long-running question.
        verbatim_asks = verbatim_asks[-12:]

        verbatim_block = ""
        if verbatim_asks:
            lines = [
                f"  - <{m.name}> {m.ts} {m.content[:240].strip()}"
                for m in verbatim_asks
            ]
            verbatim_block = "VERBATIM ASKS:\n" + "\n".join(lines) + "\n\n"

        chat_lines = [
            f"  [{m.id}] {m.ts} <{m.name}> {m.content[:200]}"
            for m in chat_msgs
        ]
        chat_section = (
            f"RECENT CHAT (oldest → newest, last {window_min} min):\n"
            + "\n".join(chat_lines)
            + "\n\n"
            if chat_lines else ""
        )

        transcript_section = ""
        if transcripts:
            tlines = [
                f"  - {t.text[:240].strip()}"
                for t in transcripts if t.text and t.text.strip()
            ]
            if tlines:
                transcript_section = (
                    f"STREAMER VOICE (last {window_min} min, oldest first):\n"
                    + "\n".join(tlines[-50:])
                    + "\n\n"
                )

        channel_context = self._build_channel_context(authoritative=False)
        grid_b64 = await self._latest_screenshot_grid(window_minutes=15)

        prompt = (
            channel_context
            + transcript_section
            + chat_section
            + verbatim_block
            + f"QUESTION: {entry.question}\n\n"
            + "Produce 3-5 short angles the streamer could offer back. "
            "Each one is a direction, not a script."
        )
        if grid_b64:
            prompt += (
                "\n\nAn image is attached showing what is currently "
                "on screen. Use it ONLY as silent context. Do NOT "
                "describe the image."
            )

        from .llm.schemas import QuestionAnswerAnglesResponse
        from .llm.prompts import resolve_prompt
        try:
            response = await self.llm.generate_structured(
                prompt=prompt,
                system_prompt=await self._streamer_facts_prepended(
                    resolve_prompt(
                        "insights.question_answer_angles", self.repo,
                    ),
                ),
                response_model=QuestionAnswerAnglesResponse,
                num_ctx=self.QUESTION_ANSWER_ANGLES_NUM_CTX,
                images=[grid_b64] if grid_b64 else None,
                think=True,
                call_site="insights.question_answer_angles",
            )
        except ValidationError as e:
            logger.warning("question-angles validation failed: %s", e)
            return [], "the model's response didn't validate"
        except Exception:
            logger.exception("question-angles LLM call failed")
            return [], "LLM call failed (check server logs)"

        angles = [a.strip() for a in response.angles if a.strip()]
        self._question_angles_cache[last_msg_id] = angles
        return list(angles), None
