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
from dataclasses import dataclass, field

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
    RECENT_MESSAGES_PER_USER = 6

    def __init__(self, repo: ChatterRepo, llm: OllamaClient, settings: Settings):
        self.repo = repo
        self.llm = llm
        self.settings = settings
        self._cache = InsightsCache(talking_points=[], refreshed_at=None, error=None)
        self._subjects_cache = EngagingSubjectsCache(
            subjects=[], refreshed_at=None, error=None,
        )
        self._lock = asyncio.Lock()
        self._subjects_lock = asyncio.Lock()

    @property
    def cache(self) -> InsightsCache:
        return self._cache

    @property
    def subjects_cache(self) -> EngagingSubjectsCache:
        return self._subjects_cache

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

    # =========================================================
    # Thread recap loop — observational summaries for the Live
    # Conversations panel. Replaces per-chatter talking-points
    # as the primary "what's happening" surface.
    # =========================================================

    THREAD_RECAP_NUM_CTX = 16384
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
        prompt = (
            "Active topic threads (numbered by `thread_id`):\n\n"
            + "\n\n".join(blocks)
            + "\n\nReturn one observational `recap` per thread you can "
            "ground in its messages. Skip noisy / unsummarisable ones."
        )
        try:
            response = await self.llm.generate_structured(
                prompt=prompt,
                system_prompt=self.THREAD_RECAP_SYSTEM,
                response_model=ThreadRecapsResponse,
                num_ctx=self.THREAD_RECAP_NUM_CTX,
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

    SUBJECTS_NUM_CTX = 8192
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
                msgs = await asyncio.to_thread(
                    self.repo.recent_messages, limit=limit,
                    within_minutes=window_min,
                )
            except Exception:
                logger.exception("engaging-subjects: recent_messages failed")
                return
            if len(msgs) < 5:
                # Too quiet to bother. Reset the cache so the panel
                # shows the empty state instead of stale subjects.
                self._subjects_cache = EngagingSubjectsCache(
                    subjects=[], refreshed_at=time.time(), error=None,
                )
                return

            # Build a numbered prompt block; clip per-message to keep
            # the context budget sane.
            user_id_to_name: dict[str, str] = {}
            for m in msgs:
                if m.user_id and m.user_id not in user_id_to_name:
                    u = await asyncio.to_thread(self.repo.get_user, m.user_id)
                    if u:
                        user_id_to_name[m.user_id] = u.name
            lines = [
                f"  [{user_id_to_name.get(m.user_id, m.user_id)}] {m.content[:200]}"
                for m in msgs
            ]
            # Blocklist injection — the streamer has flagged these
            # subjects as hallucinated / wrong / irrelevant. The LLM
            # is told to NOT extract them again. We also filter at
            # cache-write time as belt-and-suspenders.
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
                        "include it in your output. The streamer has already "
                        "decided this isn't a useful subject."
                    )
            prompt = (
                f"Recent chat ({len(msgs)} msgs over the last "
                f"{window_min} min, oldest first):\n"
                + "\n".join(lines)
                + blocklist_lines
                + "\n\nReturn the distinct conversation subjects, sorted by "
                "engagement (driver count). Skip sensitive topics."
            )
            try:
                from .llm.schemas import EngagingSubjectsResponse
                response = await self.llm.generate_structured(
                    prompt=prompt,
                    system_prompt=self.SUBJECTS_SYSTEM,
                    response_model=EngagingSubjectsResponse,
                    num_ctx=self.SUBJECTS_NUM_CTX,
                )
            except ValidationError as e:
                logger.warning("engaging-subjects validation failed: %s", e)
                self._subjects_cache = EngagingSubjectsCache(
                    subjects=[], refreshed_at=time.time(),
                    error=f"validation failed: {e!s}",
                )
                return
            except Exception as e:
                logger.exception("engaging-subjects LLM call failed")
                self._subjects_cache = EngagingSubjectsCache(
                    subjects=[], refreshed_at=time.time(), error=str(e),
                )
                return

            import hashlib as _h
            block_set_slug = {(b.get("slug") or "").lower() for b in blocklist}
            block_set_name = {(b.get("name") or "").lower() for b in blocklist}
            entries: list[EngagingSubjectEntry] = []
            for s in response.subjects:
                if s.is_sensitive:
                    continue
                clean_name = s.name.strip()
                if not clean_name:
                    continue
                # Slug: deterministic 12-char hash of the name. Used by
                # the on-demand /insights/subject-messages route so the
                # UI doesn't have to send the full name in URLs.
                slug = _h.sha1(clean_name.lower().encode("utf-8")).hexdigest()[:12]
                # Blocklist filter (the LLM might re-extract a flagged
                # subject anyway — defense in depth).
                if slug in block_set_slug or clean_name.lower() in block_set_name:
                    logger.info(
                        "engaging-subjects: dropped re-extracted blocklisted "
                        "subject %r", clean_name,
                    )
                    continue
                entries.append(EngagingSubjectEntry(
                    name=clean_name,
                    drivers=[d for d in s.drivers if d],
                    msg_count=int(s.msg_count or 0),
                    brief=(s.brief or "").strip(),
                    angles=[a.strip() for a in (s.angles or []) if a and a.strip()][:3],
                    slug=slug,
                ))
            self._subjects_cache = EngagingSubjectsCache(
                subjects=entries, refreshed_at=time.time(), error=None,
            )
            logger.info(
                "engaging subjects refreshed: %d msgs in window, "
                "%d subjects (after sensitivity filter)",
                len(msgs), len(entries),
            )
