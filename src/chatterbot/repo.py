"""ChatterRepo — the single SQLite access point.

Both the TUI and the FastAPI dashboard go through this module. No direct SQL
should appear anywhere else.

Tables:
  - users                : per-chatter identity (twitch_id, name, opt_out)
  - messages             : full chat log, retained indefinitely
  - notes                : LLM-extracted facts about a chatter, with embeddings
  - events               : StreamElements events (tip / sub / cheer / raid / follow)
  - topic_snapshots      : channel-wide "what's chat talking about" rollups
  - summarization_state  : per-user watermark of the highest message_id summarized

Plus a sqlite-vec virtual table mirroring note embeddings (and a separate one
for messages, used by the dashboard's per-user RAG endpoint).

Hard rule reminder:
  No method on this repo is intended to feed a chat-facing LLM prompt.
  Notes / events / topic snapshots are streamer-eyes-only.
  The RAG-search helpers exist for the streamer-only dashboard.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import sqlite_vec


# ------------------------------- dataclasses -------------------------------


@dataclass
class User:
    twitch_id: str
    name: str
    first_seen: str
    last_seen: str
    opt_out: bool
    # Denormalized per-message badge snapshot — last-seen state from IRCv3
    # tags. These reflect the chatter's status the last time they spoke in
    # chat, not their current Twitch state (silent un-subs etc. are missed).
    sub_tier: str | None = None      # '1000' | '2000' | '3000' | 'Prime' | None
    sub_months: int = 0
    is_mod: bool = False
    is_vip: bool = False
    is_founder: bool = False
    # Cross-platform identity. `source` ∈ {twitch, youtube, discord}; the
    # twitch_id PK is namespaced for non-Twitch (e.g. 'yt:UCxxx', 'dc:1234').
    source: str = "twitch"
    # If non-null, this user has been merged into another (the parent).
    # Aggregations should follow the link before reporting on this row.
    merged_into: str | None = None
    # Soft-profile fields built by the LLM profile extractor. The notes
    # surface continues to carry hard, cited facts; this is the squishier
    # "who is this person" view. Each is None until the LLM sees a clear
    # signal in the user's chat history.
    pronouns: str | None = None       # free text — "she/her", "they/them"
    location: str | None = None       # free text — "Sydney" or "Australia"
    demeanor: str | None = None       # constrained enum, see schemas.Demeanor
    interests: list[str] | None = None  # deduped, capped, ordered by recency
    profile_updated_at: str | None = None
    is_starred: bool = False          # streamer-personal favorite
    followed_at: str | None = None    # ISO-UTC; populated by HelixSyncService


@dataclass
class Reminder:
    """Streamer-set reminder attached to a chatter. Fires (auto-stamps
    `fired_at`) when that chatter next sends a message; the dashboard
    surfaces a fired-not-dismissed reminder until the streamer dismisses it."""

    id: int
    user_id: str
    user_name: str | None
    text: str
    created_at: str
    fired_at: str | None
    dismissed: bool


@dataclass
class InsightState:
    """Per-card state on the Insights page. Lets the streamer mark an
    item as addressed-on-stream, snooze it, pin it, or skip it. Items
    not in the table render as 'open' (default).

    `kind` ∈ {talking_point, thread, anniversary, newcomer, regular, lapsed}
    `item_key` is whatever uniquely identifies that card within its kind
    (thread_id for threads, hash of user_id+point for talking points, etc.)
    `state` ∈ {addressed, snoozed, pinned, skipped}
    `due_ts` is set only when state == 'snoozed' — the moment the card
    should resurface.
    `note` is the optional 'what I said on stream' memory aid the streamer
    can capture when dismissing — they speak on stream, never type to chat.
    """

    kind: str
    item_key: str
    state: str
    due_ts: str | None
    note: str | None
    updated_at: str


@dataclass
class TranscriptChunk:
    """One VAD-bounded utterance the streamer spoke on stream, transcribed
    by whisper. When the embedding cosine-matches an open insight card
    above threshold, matched_* + similarity are populated AND that card
    auto-flips to 'addressed' with this chunk's text as its note."""

    id: int
    ts: str
    duration_ms: int
    text: str
    matched_kind: str | None = None
    matched_item_key: str | None = None
    similarity: float | None = None


@dataclass
class TranscriptScreenshot:
    """One JPEG captured from OBS at `ts`. Path is relative to
    `Settings.db_path`'s parent directory."""
    id: int
    ts: str
    path: str
    scene_name: str | None


@dataclass
class TranscriptGroup:
    """A window of consecutive whisper utterances summarised by the LLM
    into a single observational line. Replaces per-utterance lines in the
    live transcript strip — less distracting, more skimmable.

    `context_message_ids` records the chat message IDs that were passed
    to the LLM as the CHAT DURING THIS WINDOW context block. Persisted
    with the group at write-time so the transcript-group modal can
    show the streamer EXACTLY the chat the LLM saw — not a re-queried
    approximation. Empty list for groups created before this field
    was added (graceful degrade in the UI)."""

    id: int
    start_ts: str
    end_ts: str
    first_chunk_id: int
    last_chunk_id: int
    summary: str
    created_at: str
    context_message_ids: list[int] = field(default_factory=list)


@dataclass
class StreamRecap:
    id: int
    started_at: str
    ended_at: str
    message_id_lo: int | None
    message_id_hi: int | None
    summary: str
    msg_count: int = 0
    unique_chatters: int = 0
    new_chatters: int = 0
    addressed_count: int = 0
    snoozed_count: int = 0


@dataclass
class StreamSession:
    """Computed session — a contiguous chunk of chat with no >GAP-min silence."""

    id: int           # pseudo-id (1..N), most recent first
    start_ts: str
    end_ts: str
    message_count: int
    unique_chatters: int
    top_chatter: str | None


@dataclass
class Note:
    id: int
    user_id: str
    ts: str
    text: str
    # 'manual' for streamer-typed, 'llm' for summarizer-extracted. Lets the
    # UI label provenance honestly instead of inferring from the presence
    # of source citations (a uncited LLM note ≠ a manual note).
    origin: str = "manual"


@dataclass
class NoteWithSources:
    """A note plus the message rows the LLM cited as supporting it.
    Manual notes (and pre-migration LLM notes) have an empty sources list."""

    note: Note
    sources: list[Message]


@dataclass
class Alias:
    name: str
    first_seen: str
    last_seen_as: str


@dataclass
class Message:
    id: int
    user_id: str
    name: str
    ts: str
    content: str
    # Native-reply context, populated when the chatter used the platform's
    # Reply feature (Twitch IRCv3 reply tags or Discord message references).
    # None for plain messages.
    reply_parent_login: str | None = None
    reply_parent_body: str | None = None
    # Cross-platform attribution. Pulled from users.source via JOIN. Default
    # 'twitch' so old call-sites that build Message() directly stay sane.
    source: str = "twitch"
    # Spam score in [0.0, 1.0] (0.0 = clean) and a list of reason codes
    # that fired. Score is the max across signals at ingest, optionally
    # bumped by the post-embedding flood detector. Consumers filter by
    # threshold; see `spam.py` for `SPAM_THRESHOLD_DEFAULT` /
    # `SPAM_THRESHOLD_LLM`.
    spam_score: float = 0.0
    spam_reasons: list[str] = field(default_factory=list)


@dataclass
class Event:
    id: int
    user_id: str | None
    twitch_name: str
    type: str
    amount: float | None
    currency: str | None
    message: str | None
    ts: str


@dataclass
class Incident:
    id: int
    user_id: str | None
    message_id: int | None
    ts: str
    severity: int               # 1 (minor) | 2 (warning) | 3 (serious)
    categories: list[str]       # parsed from JSON column
    rationale: str | None
    status: str                 # 'open' | 'reviewed' | 'dismissed'


@dataclass
class IncidentRow:
    """Joined incident with the offender's name and the offending message
    content — used by the moderation list view and per-user incident panel."""

    incident: Incident
    user_name: str | None
    message_content: str | None
    message_reply_parent_login: str | None = None
    message_reply_parent_body: str | None = None


@dataclass
class TopicSnapshot:
    id: int
    ts: str
    summary: str
    message_id_range: str | None
    topics_json: str | None = None  # parsed lazily by callers if present


@dataclass
class TopicThread:
    """A topic that has been observed across one or more snapshots — the
    streamer's mental model of "ongoing conversation in chat." `status` is
    computed (active = last_ts within 1h; dormant = within 24h; archived =
    older). `drivers` is the union across all members."""

    id: int
    title: str           # canonical (latest member's title wins)
    first_ts: str
    last_ts: str
    drivers: list[str]
    member_count: int
    status: str          # 'active' | 'dormant' | 'archived'
    category: str | None # latest member's category if any (Phase 4 tags)
    recap: str | None = None              # LLM-generated 1-2 sentence summary
    recap_updated_at: str | None = None


@dataclass
class TopicThreadMember:
    thread_id: int
    snapshot_id: int
    topic_index: int
    drivers: list[str]
    ts: str


@dataclass
class UserEventSummary:
    """Aggregated event totals for a user, for the per-user TUI / dashboard panel."""

    total_tip_amount: float
    tip_currency: str | None
    last_tip_ts: str | None
    total_bits: int
    sub_months: int
    sub_tier: str | None
    last_sub_ts: str | None


@dataclass
class ChatterRow:
    """Joined row for the chatters list view."""

    user: User
    note_count: int
    msg_count: int
    last_message_ts: str | None
    engagement_score: int = 0  # 0-100 composite, computed in list_chatters


def _engagement_score(
    *, msg_count: int, note_count: int, sub_tier: str | None,
    sub_months: int, is_mod: bool, is_vip: bool, is_starred: bool,
    last_seen: str | None,
) -> int:
    """Composite 0-100 engagement signal. Weighted toward facts the LLM
    has already extracted (notes), platform investment (sub/mod/vip),
    and recency. Capped so a single mega-chatter doesn't dominate.

    Tuned by hand against typical Twitch distributions; if you're
    seeing everyone clustered at one end, dial the weights here."""
    raw = 0
    raw += min(msg_count, 200)            # cap at 200 messages worth
    raw += note_count * 8                 # each cited fact is real signal
    if sub_tier:
        raw += 25 + min(sub_months, 24) * 2  # subs + tenure
    if is_mod:
        raw += 30
    if is_vip:
        raw += 20
    if is_starred:
        raw += 25
    # Recency decay — fade out chatters who haven't been seen recently.
    if last_seen:
        try:
            last = datetime.fromisoformat(last_seen)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            days = max(0, (datetime.now(timezone.utc) - last).days)
            raw -= min(days * 2, 60)
        except (TypeError, ValueError):
            pass
    return max(0, min(100, raw // 4))


# ------------------------------- helpers -----------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _blob_to_vec(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ------------------------------- repo --------------------------------------


class ChatterRepo:
    def __init__(self, db_path: str, embed_dim: int = 768):
        self.db_path = db_path
        self.embed_dim = embed_dim
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # Thread-local connections so each `asyncio.to_thread` worker (and
        # the bot, summarizer, transcript service) get their own SQLite
        # handle. The previous single-connection-with-Lock design serialised
        # ALL queries — readers blocked on writers, even though WAL would
        # allow them to run concurrently — and made /insights take 10-20 s
        # under load. Now the lock is gone; SQLite's WAL-level concurrency
        # does the right thing.
        self._tl = threading.local()
        # Initialise the schema on a primary connection. Subsequent
        # connections (lazy-created per thread) reuse the existing tables.
        self._init_conn()  # populates self._tl.conn for *this* thread
        self._init_schema()
        # In-memory app_settings cache (see _ensure_app_settings_cache).
        # Background loops hammer get_app_setting() once per tick for
        # things like watermarks and the engaging-subjects blocklist;
        # every call hit a 1-row SELECT before. Cache is process-local
        # and TTL'd so cross-process writes (bot ↔ dashboard) reconcile
        # within ~60 s.
        self._app_settings_cache: dict[str, str | None] = {}
        self._app_settings_cache_loaded_at: float = 0.0
        self._app_settings_lock = threading.Lock()
        # Personal-dataset DEK held in process memory while capture is
        # unlocked. None when the streamer hasn't unlocked yet (or has
        # capture disabled). Accessed via dataset_dek() / set_dataset_dek
        # so the hot capture-path is one attribute read.
        self._dataset_dek: bytes | None = None
        self._dataset_dek_lock = threading.Lock()

    def _init_conn(self) -> sqlite3.Connection:
        """Create + configure a SQLite connection for the current thread.
        Loaded once per thread; subsequent _cursor() calls reuse it."""
        conn = sqlite3.connect(self.db_path, check_same_thread=True)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        # Wait up to 5 s before raising SQLITE_BUSY when another writer
        # holds the lock. Hot streams have continuous writes (chat +
        # transcripts + insight states); without this, dashboard reads
        # would occasionally fail rather than queue briefly.
        conn.execute("PRAGMA busy_timeout=5000")
        # Reduced fsync — WAL still durable across crashes; this just
        # avoids fsync on every COMMIT, which is costly on bind-mounted
        # volumes.
        conn.execute("PRAGMA synchronous=NORMAL")
        self._tl.conn = conn
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._tl, "conn", None)
        if conn is None:
            conn = self._init_conn()
        return conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        finally:
            cur.close()

    def _init_schema(self) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    twitch_id  TEXT PRIMARY KEY,
                    name       TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen  TEXT NOT NULL,
                    opt_out    INTEGER NOT NULL DEFAULT 0,
                    sub_tier   TEXT,
                    sub_months INTEGER NOT NULL DEFAULT 0,
                    is_mod     INTEGER NOT NULL DEFAULT 0,
                    is_vip     INTEGER NOT NULL DEFAULT 0,
                    is_founder INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            # Idempotent badge-column migration for older DBs.
            cur.execute("PRAGMA table_info(users)")
            _ucols = {r["name"] for r in cur.fetchall()}
            for col, decl in (
                ("sub_tier",   "TEXT"),
                ("sub_months", "INTEGER NOT NULL DEFAULT 0"),
                ("is_mod",     "INTEGER NOT NULL DEFAULT 0"),
                ("is_vip",     "INTEGER NOT NULL DEFAULT 0"),
                ("is_founder", "INTEGER NOT NULL DEFAULT 0"),
                # Cross-platform identity. `source` ∈ {twitch, youtube, discord}.
                # The `twitch_id` PK is platform-namespaced for non-Twitch
                # rows (e.g. 'yt:UCxxx', 'dc:1234567890') so collisions across
                # platforms are impossible.
                ("source",      "TEXT NOT NULL DEFAULT 'twitch'"),
                # When a user is merged into another, this points at the
                # canonical user's twitch_id (the parent). All FKs are also
                # rewritten to the parent at merge time, so the child row
                # exists only for "merged from X" provenance.
                ("merged_into", "TEXT"),
                # LLM-built profile fields. Updated per summarization batch
                # via a separate softer-rubric extractor. None means we
                # haven't seen a clear signal yet — the profile pass never
                # overwrites a known value with null.
                ("pronouns",            "TEXT"),
                ("location",            "TEXT"),
                ("demeanor",            "TEXT"),  # constrained enum, see schemas.py
                ("interests",           "TEXT"),  # JSON array, deduped
                ("profile_updated_at",  "TEXT"),
                # Streamer-personal favorite flag — separate from Twitch's
                # native VIP role. Drives the gold-bordered live-widget
                # surfacing + the "starred chatters present" insights row.
                ("is_starred",          "INTEGER NOT NULL DEFAULT 0"),
                # When this user followed the broadcaster's channel.
                # Populated by HelixSyncService (channel-followers poll).
                # NULL = unknown / not following at the time of last
                # sync. Surfaced as a "follower for Nd" pill on profiles.
                ("followed_at",         "TEXT"),
            ):
                if col not in _ucols:
                    cur.execute(f"ALTER TABLE users ADD COLUMN {col} {decl}")
            # ============================================================
            # USER PROFILE EXTENSION TABLE
            # ------------------------------------------------------------
            # Sparse / streamer-set / LLM-extracted fields live here so
            # the hot core users table stays narrow. Pattern: every read
            # that needs profile data LEFT JOINs; every write that
            # touches profile data UPSERTs into user_profiles by user_id.
            # The migration below copies any pre-split data over and
            # drops the old columns so there's a single source of truth.
            # ============================================================
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id            TEXT PRIMARY KEY
                        REFERENCES users(twitch_id) ON DELETE CASCADE,
                    pronouns           TEXT,
                    location           TEXT,
                    demeanor           TEXT,
                    interests          TEXT,
                    profile_updated_at TEXT,
                    is_starred         INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_profiles_starred "
                "ON user_profiles(is_starred) WHERE is_starred = 1"
            )

            # One-time migration: detect old profile columns still on
            # users, copy non-null rows into user_profiles, then drop
            # the columns. Re-running on an already-migrated DB is a
            # no-op because the columns are gone.
            cur.execute("PRAGMA table_info(users)")
            _ucols2 = {r["name"] for r in cur.fetchall()}
            _profile_cols = (
                "pronouns", "location", "demeanor", "interests",
                "profile_updated_at", "is_starred",
            )
            _to_move = [c for c in _profile_cols if c in _ucols2]
            if _to_move:
                # Copy from users → user_profiles. INSERT OR REPLACE so
                # a partial migration (or re-run during dev) is safe.
                copy_cols = ", ".join(_to_move)
                # is_starred has a NOT NULL default 0 on the new table;
                # if it's not in _to_move (older schema lacked it),
                # the SELECT NULL → DEFAULT 0 path keeps things correct.
                # All other moved cols accept NULL so a sparse SELECT
                # works without coercion.
                cur.execute(
                    f"""
                    INSERT OR REPLACE INTO user_profiles
                        (user_id, {copy_cols})
                    SELECT twitch_id, {copy_cols}
                    FROM users
                    WHERE {' OR '.join(f'{c} IS NOT NULL' for c in _to_move if c != 'is_starred')}
                       OR is_starred = 1
                    """
                    if "is_starred" in _to_move
                    else f"""
                    INSERT OR REPLACE INTO user_profiles
                        (user_id, {copy_cols})
                    SELECT twitch_id, {copy_cols}
                    FROM users
                    WHERE {' OR '.join(f'{c} IS NOT NULL' for c in _to_move)}
                    """
                )
                # Drop the old columns. Requires SQLite 3.35+ (March
                # 2021); chatterbot ships against Python 3.11+ which
                # carries 3.40+. If a deployment somehow has older
                # SQLite, the ALTER raises and we surface the error.
                for col in _to_move:
                    cur.execute(f"ALTER TABLE users DROP COLUMN {col}")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id             TEXT NOT NULL REFERENCES users(twitch_id) ON DELETE CASCADE,
                    ts                  TEXT NOT NULL,
                    content             TEXT NOT NULL,
                    reply_parent_login  TEXT,
                    reply_parent_body   TEXT
                )
                """
            )
            # Idempotent migration for older DBs that pre-date the reply columns.
            cur.execute("PRAGMA table_info(messages)")
            _mcols = {r["name"] for r in cur.fetchall()}
            if "reply_parent_login" not in _mcols:
                cur.execute("ALTER TABLE messages ADD COLUMN reply_parent_login TEXT")
            if "reply_parent_body" not in _mcols:
                cur.execute("ALTER TABLE messages ADD COLUMN reply_parent_body TEXT")
            # Emote-only flag — set when the message contains nothing but
            # native emotes + whitespace. The bot derives this from the
            # IRCv3 `emotes` tag at write time (ground truth — no guessing).
            # Summarizer + moderator skip these; the live widget still shows
            # them so hype is visible.
            if "is_emote_only" not in _mcols:
                cur.execute(
                    "ALTER TABLE messages ADD COLUMN is_emote_only "
                    "INTEGER NOT NULL DEFAULT 0"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_emote_only "
                    "ON messages(is_emote_only)"
                )
            # Spam scoring — score (REAL 0..1) plus a JSON array of
            # reason codes that fired. Score lets each consumer pick
            # its own threshold (stats / word cloud at ~0.5, LLM
            # prompts at ~0.2). Reasons array is for transparency in
            # audit views — "filtered: 12 repetition, 2 caps_flood".
            # See `spam.py` for the detector contract.
            if "spam_score" not in _mcols:
                cur.execute(
                    "ALTER TABLE messages ADD COLUMN spam_score "
                    "REAL NOT NULL DEFAULT 0"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_spam_score "
                    "ON messages(spam_score)"
                )
            if "spam_reasons" not in _mcols:
                cur.execute(
                    "ALTER TABLE messages ADD COLUMN spam_reasons TEXT"
                )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    TEXT NOT NULL REFERENCES users(twitch_id) ON DELETE CASCADE,
                    ts         TEXT NOT NULL,
                    text       TEXT NOT NULL,
                    embedding  BLOB,
                    origin     TEXT NOT NULL DEFAULT 'manual'
                )
                """
            )
            # Idempotent migration: 'origin' distinguishes LLM-extracted
            # notes from streamer-typed ones. Default 'manual' is the
            # conservative (don't-lie-about-existing-data) choice; new
            # writes get the explicit origin from add_note's caller.
            cur.execute("PRAGMA table_info(notes)")
            _ncols = {r["name"] for r in cur.fetchall()}
            if "origin" not in _ncols:
                cur.execute(
                    "ALTER TABLE notes ADD COLUMN origin TEXT NOT NULL "
                    "DEFAULT 'manual'"
                )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT REFERENCES users(twitch_id) ON DELETE SET NULL,
                    twitch_name TEXT NOT NULL,
                    type        TEXT NOT NULL,
                    amount      REAL,
                    currency    TEXT,
                    message     TEXT,
                    ts          TEXT NOT NULL,
                    raw_json    TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS topic_snapshots (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts               TEXT NOT NULL,
                    summary          TEXT NOT NULL,
                    message_id_range TEXT,
                    topics_json      TEXT
                )
                """
            )
            # Migrate older DBs that pre-date the topics_json column.
            cur.execute("PRAGMA table_info(topic_snapshots)")
            _cols = {r["name"] for r in cur.fetchall()}
            if "topics_json" not in _cols:
                cur.execute("ALTER TABLE topic_snapshots ADD COLUMN topics_json TEXT")
            # Topic threads — recurring conversations across snapshots.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS topic_threads (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    title     TEXT NOT NULL,
                    first_ts  TEXT NOT NULL,
                    last_ts   TEXT NOT NULL,
                    category  TEXT,
                    embedding BLOB,
                    recap     TEXT,
                    recap_updated_at TEXT
                )
                """
            )
            # Migrate older DBs that pre-date the recap columns.
            cur.execute("PRAGMA table_info(topic_threads)")
            _cols = {r["name"] for r in cur.fetchall()}
            if "recap" not in _cols:
                cur.execute("ALTER TABLE topic_threads ADD COLUMN recap TEXT")
            if "recap_updated_at" not in _cols:
                cur.execute(
                    "ALTER TABLE topic_threads ADD COLUMN recap_updated_at TEXT"
                )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS topic_thread_members (
                    thread_id   INTEGER NOT NULL REFERENCES topic_threads(id) ON DELETE CASCADE,
                    snapshot_id INTEGER NOT NULL REFERENCES topic_snapshots(id) ON DELETE CASCADE,
                    topic_index INTEGER NOT NULL,
                    drivers     TEXT NOT NULL,
                    ts          TEXT NOT NULL,
                    PRIMARY KEY (thread_id, snapshot_id, topic_index)
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_threads_last_ts ON topic_threads(last_ts)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_thread_members_snap "
                "ON topic_thread_members(snapshot_id)"
            )
            # vec_threads uses cosine distance — the threader's similarity
            # threshold is expressed in cosine-distance terms (1 - cosine_sim).
            # vec0 defaults to L2 if the metric isn't specified, which made
            # the 0.30 threshold unreachable in practice. The matching
            # one-shot L2 → cosine migration runs after app_settings is
            # created (sentinel: vec_threads_metric_version = cosine_v1).
            cur.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_threads USING vec0(
                    thread_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{self.embed_dim}] distance_metric=cosine
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS summarization_state (
                    user_id                  TEXT PRIMARY KEY
                                             REFERENCES users(twitch_id) ON DELETE CASCADE,
                    last_summarized_msg_id   INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            # Reminders — streamer-set, fired when the user next chats.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT NOT NULL REFERENCES users(twitch_id) ON DELETE CASCADE,
                    text        TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    fired_at    TEXT,
                    dismissed   INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_reminders_user_pending "
                "ON reminders(user_id, fired_at, dismissed)"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    key        TEXT PRIMARY KEY,
                    value      TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            # Personal training-dataset capture index. Each row points at
            # one encrypted record inside a shard file under
            # data/dataset/shards/. The actual ciphertext never lives in
            # SQLite — keeps chatters.db backup size sane and lets
            # `rm -rf data/dataset/` cleanly destroy the dataset.
            #
            # ts is in cleartext on purpose: AES-GCM binds it as
            # associated-data so the reader needs it to decrypt, and
            # filtering an export by date range shouldn't require
            # touching the encryption pipeline.
            #
            # schema_version is per-row so the reader can dispatch to
            # the right migration when the on-disk event shape evolves.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_events (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts             TEXT NOT NULL,
                    event_kind     TEXT NOT NULL,
                    shard_path     TEXT NOT NULL,
                    byte_offset    INTEGER NOT NULL,
                    byte_length    INTEGER NOT NULL,
                    schema_version INTEGER NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_dataset_events_ts "
                "ON dataset_events(ts)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_dataset_events_kind "
                "ON dataset_events(event_kind)"
            )
            # Username history: a row per (user_id, observed_name). Twitch IDs
            # never change but display names do, so this gives us a complete
            # rename trail and lets the streamer search by an old handle.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_aliases (
                    user_id      TEXT NOT NULL REFERENCES users(twitch_id) ON DELETE CASCADE,
                    name         TEXT NOT NULL,
                    first_seen   TEXT NOT NULL,
                    last_seen_as TEXT NOT NULL,
                    PRIMARY KEY (user_id, name)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_aliases_name ON user_aliases(LOWER(name))"
            )
            # Moderation incidents (opt-in MOD_MODE_ENABLED). Advisory only —
            # the bot never auto-actions; the streamer reviews via the
            # dashboard's Moderation tab.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT REFERENCES users(twitch_id) ON DELETE CASCADE,
                    message_id  INTEGER REFERENCES messages(id) ON DELETE SET NULL,
                    ts          TEXT NOT NULL,
                    severity    INTEGER NOT NULL,
                    categories  TEXT NOT NULL,
                    rationale   TEXT,
                    status      TEXT NOT NULL DEFAULT 'open'
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_incidents_user ON incidents(user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_incidents_ts ON incidents(ts)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status)")
            # Per-item state for Insights cards. Lets the streamer mark a
            # talking point / thread / anniversary as addressed-on-stream,
            # snooze it for 10m / 30m / 1h / 24h, pin it to the top of its
            # section, or skip it. `note` is the optional "what I said on
            # stream" memory aid the streamer captures when dismissing.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS insight_states (
                    kind        TEXT NOT NULL,
                    item_key    TEXT NOT NULL,
                    state       TEXT NOT NULL,
                    due_ts      TEXT,
                    note        TEXT,
                    updated_at  TEXT NOT NULL,
                    PRIMARY KEY (kind, item_key)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_insight_states_due "
                "ON insight_states(due_ts) WHERE due_ts IS NOT NULL"
            )
            # Append-only audit log of insight_state transitions. The
            # streamer can review "what did I act on" + the recap can
            # cite concrete addressed-counts from this rather than the
            # text of the transcript.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS insight_state_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT NOT NULL,
                    kind        TEXT NOT NULL,
                    item_key    TEXT NOT NULL,
                    state       TEXT NOT NULL,
                    due_ts      TEXT,
                    note        TEXT
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_insight_history_ts "
                "ON insight_state_history(ts)"
            )
            # Live transcript chunks from the OBS audio relay → whisper
            # pipeline. Each row is one VAD-bounded utterance the streamer
            # spoke on stream. matched_* fields are populated when the
            # chunk's embedding cosine-matches an open insight card above
            # the configured threshold; the match auto-sets that card's
            # state to 'addressed'.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS transcript_chunks (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts                TEXT NOT NULL,
                    duration_ms       INTEGER NOT NULL DEFAULT 0,
                    text              TEXT NOT NULL,
                    matched_kind      TEXT,
                    matched_item_key  TEXT,
                    similarity        REAL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_transcript_ts "
                "ON transcript_chunks(ts)"
            )
            # Per-window LLM-summarized groups of transcript chunks.
            # Replaces the per-utterance live strip on the dashboard
            # with a less distracting "summary every N seconds" view.
            # The grouper service writes one row per pass.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS transcript_groups (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_ts        TEXT NOT NULL,
                    end_ts          TEXT NOT NULL,
                    first_chunk_id  INTEGER NOT NULL,
                    last_chunk_id   INTEGER NOT NULL,
                    summary         TEXT NOT NULL,
                    created_at      TEXT NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_transcript_groups_end "
                "ON transcript_groups(end_ts)"
            )
            # Idempotent column add for context_message_ids — JSON array
            # of chat message IDs the LLM saw when summarising. Older
            # DBs predate this column and need an ALTER; sqlite has no
            # IF NOT EXISTS for ADD COLUMN, so we check the schema first.
            cur.execute("PRAGMA table_info(transcript_groups)")
            tg_cols = {row["name"] for row in cur.fetchall()}
            if "context_message_ids" not in tg_cols:
                cur.execute(
                    "ALTER TABLE transcript_groups "
                    "ADD COLUMN context_message_ids TEXT"
                )
            # OBS screenshots paired with transcript chunks, captured by
            # the screenshot loop every N seconds while whisper + OBS
            # are both active. Each transcript group queries this table
            # by ts range and shows up to 4 evenly-spaced screenshots
            # so the streamer can scrub the visual context that went
            # with what they were saying.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS transcript_screenshots (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT NOT NULL,
                    path        TEXT NOT NULL,
                    scene_name  TEXT
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_transcript_screenshots_ts "
                "ON transcript_screenshots(ts)"
            )
            # Vector index over transcript chunk embeddings — drives the
            # bidirectional link: a chat message can find the recent
            # streamer utterance that's semantically close to it. Cosine
            # metric so the threshold means cosine-distance directly.
            cur.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_transcripts USING vec0(
                    chunk_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{self.embed_dim}] distance_metric=cosine
                )
                """
            )
            # End-of-stream LLM-generated recaps. Triggered when OBS
            # transitions from streaming → not-streaming. One row per
            # detected stream session.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS stream_recaps (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at      TEXT NOT NULL,
                    ended_at        TEXT NOT NULL,
                    message_id_lo   INTEGER,
                    message_id_hi   INTEGER,
                    summary         TEXT NOT NULL
                )
                """
            )
            # Idempotent migration to add stream-stat columns for the
            # cross-stream KPI deltas. Each is filled at recap time.
            cur.execute("PRAGMA table_info(stream_recaps)")
            _rcols = {r["name"] for r in cur.fetchall()}
            for col, decl in (
                ("msg_count",         "INTEGER NOT NULL DEFAULT 0"),
                ("unique_chatters",   "INTEGER NOT NULL DEFAULT 0"),
                ("new_chatters",      "INTEGER NOT NULL DEFAULT 0"),
                ("addressed_count",   "INTEGER NOT NULL DEFAULT 0"),
                ("snoozed_count",     "INTEGER NOT NULL DEFAULT 0"),
            ):
                if col not in _rcols:
                    cur.execute(
                        f"ALTER TABLE stream_recaps ADD COLUMN {col} {decl}"
                    )
            # One-time backfill so existing users get their current name as an alias
            # without needing a fresh observation. Idempotent.
            cur.execute(
                """
                INSERT OR IGNORE INTO user_aliases(user_id, name, first_seen, last_seen_as)
                SELECT twitch_id, name, first_seen, last_seen FROM users
                """
            )
            # vec_threads cosine-metric migration. Older DBs created the
            # virtual table under the default L2 metric, which made the
            # threader's 0.30 distance threshold unreachable — every
            # snapshot's topics created brand-new threads instead of
            # joining recurring ones. Sentinel-gated so this runs once
            # per DB. The bot's threader.backfill() then re-clusters
            # all snapshots into a fresh thread index against cosine.
            cur.execute(
                "SELECT value FROM app_settings WHERE key = 'vec_threads_metric_version'"
            )
            sentinel = cur.fetchone()
            if not sentinel or sentinel["value"] != "cosine_v1":
                cur.execute("DROP TABLE IF EXISTS vec_threads")
                cur.execute(
                    f"""
                    CREATE VIRTUAL TABLE vec_threads USING vec0(
                        thread_id INTEGER PRIMARY KEY,
                        embedding FLOAT[{self.embed_dim}] distance_metric=cosine
                    )
                    """
                )
                cur.execute("DELETE FROM topic_thread_members")
                cur.execute("DELETE FROM topic_threads")
                cur.execute(
                    """
                    INSERT INTO app_settings(key, value, updated_at)
                    VALUES ('vec_threads_metric_version', 'cosine_v1', ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                    (_now_iso(),),
                )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts)")
            # Compound index for the regulars / lapsed / high-impact
            # queries that filter by both user_id AND a ts window
            # (`WHERE m.user_id = u.twitch_id AND datetime(m.ts) >= …`).
            # Single-column indexes can't serve both halves — sqlite
            # picks one and full-scans the other condition. With this
            # index, a 50k-message DB drops those queries from
            # 50-150ms to <5ms.
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_user_ts "
                "ON messages(user_id, ts)"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_notes_user ON notes(user_id)")
            # Link a note back to the message(s) the LLM cited as supporting
            # it — provenance for "where did this fact come from?" Many-to-
            # many because the model often aggregates several lines into one
            # note. Manual notes (added from the dashboard) have no rows here.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS note_sources (
                    note_id    INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
                    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                    PRIMARY KEY (note_id, message_id)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_note_sources_note ON note_sources(note_id)"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_name ON events(LOWER(twitch_name))")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_topic_snapshots_ts ON topic_snapshots(ts)"
            )
            cur.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_notes USING vec0(
                    note_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{self.embed_dim}]
                )
                """
            )
            # Mirrors message embeddings for the dashboard RAG. Sparse: we only embed
            # messages on-demand, when they're surfaced via the Ask-Qwen feature.
            cur.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages USING vec0(
                    message_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{self.embed_dim}]
                )
                """
            )
    # ============================ WRITE surface (bot) ======================

    def upsert_user(self, twitch_id: str, name: str, *, source: str = "twitch") -> None:
        now = _now_iso()
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO users(twitch_id, name, first_seen, last_seen, opt_out, source)
                VALUES (?, ?, ?, ?, 0, ?)
                ON CONFLICT(twitch_id) DO UPDATE SET
                    name = excluded.name,
                    last_seen = excluded.last_seen
                """,
                (twitch_id, name, now, now, source),
            )
            # Record every name we see this user under. Twitch IDs are stable;
            # display names are not.
            cur.execute(
                """
                INSERT INTO user_aliases(user_id, name, first_seen, last_seen_as)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, name) DO UPDATE SET
                    last_seen_as = excluded.last_seen_as
                """,
                (twitch_id, name, now, now),
            )
            # Backfill orphan SE events seen before this name appeared in chat.
            # Match against any historical alias of this user, not just the
            # current one — a tip credited to "alice_old" should land on the
            # same profile as alice_new.
            cur.execute(
                """
                UPDATE events
                SET user_id = ?
                WHERE user_id IS NULL
                  AND LOWER(twitch_name) IN (
                    SELECT LOWER(name) FROM user_aliases WHERE user_id = ?
                  )
                """,
                (twitch_id, twitch_id),
            )

    def update_user_badges(
        self,
        twitch_id: str,
        *,
        sub_tier: str | None,
        sub_months: int,
        is_mod: bool,
        is_vip: bool,
        is_founder: bool,
    ) -> None:
        """Snap the chatter's role/sub state from IRCv3 tags onto users.
        Called by the bot on every message — gives us at-a-glance "who's
        currently subbed / a mod / etc." without needing Helix scopes."""
        with self._cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET sub_tier   = ?,
                    sub_months = MAX(sub_months, ?),
                    is_mod     = ?,
                    is_vip     = ?,
                    is_founder = is_founder OR ?
                WHERE twitch_id = ?
                """,
                (
                    sub_tier,
                    int(sub_months),
                    1 if is_mod else 0,
                    1 if is_vip else 0,
                    1 if is_founder else 0,
                    twitch_id,
                ),
            )

    def update_user_profile(
        self,
        twitch_id: str,
        *,
        pronouns: str | None = None,
        location: str | None = None,
        demeanor: str | None = None,
        interests: list[str] | None = None,
        max_interests: int = 12,
    ) -> None:
        """Merge LLM-extracted profile signals into the user_profiles
        row. Never overwrites a known value with None — the extractor
        only emits a field when it sees a clear signal, so silence on
        a field means "no new information," not "clear it." Interests
        are union-merged with existing entries (case-insensitive
        dedup) and capped.

        Writes go to the user_profiles extension table; the parent
        users row must already exist (the chat-message ingest path
        upserts it before any profile work happens)."""
        with self._cursor() as cur:
            # Read prior values from user_profiles (sparse — the row
            # may not exist yet for a newly-seen user).
            cur.execute(
                "SELECT pronouns, location, demeanor, interests "
                "FROM user_profiles WHERE user_id = ?",
                (twitch_id,),
            )
            row = cur.fetchone()
            prior_pronouns  = row["pronouns"]  if row else None
            prior_location  = row["location"]  if row else None
            prior_demeanor  = row["demeanor"]  if row else None
            prior_interests = row["interests"] if row else None

            new_pronouns = pronouns.strip() if pronouns and pronouns.strip() else prior_pronouns
            new_location = location.strip() if location and location.strip() else prior_location
            new_demeanor = demeanor.strip() if demeanor and demeanor.strip() else prior_demeanor
            existing_interests: list[str] = []
            if prior_interests:
                try:
                    parsed = json.loads(prior_interests)
                    if isinstance(parsed, list):
                        existing_interests = [str(x) for x in parsed if x]
                except (TypeError, ValueError):
                    pass
            new_interests = list(existing_interests)
            if interests:
                seen_lower = {x.lower() for x in new_interests}
                for entry in interests:
                    if not entry:
                        continue
                    e = entry.strip()
                    if not e or e.lower() in seen_lower:
                        continue
                    new_interests.append(e)
                    seen_lower.add(e.lower())
                # Newest first when over cap — keep most recent signals.
                if len(new_interests) > max_interests:
                    new_interests = new_interests[-max_interests:]
            # UPSERT the profile row. ON CONFLICT preserves is_starred
            # so a streamer-set favorite isn't clobbered by an LLM
            # profile pass.
            cur.execute(
                """
                INSERT INTO user_profiles
                    (user_id, pronouns, location, demeanor, interests, profile_updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    pronouns           = excluded.pronouns,
                    location           = excluded.location,
                    demeanor           = excluded.demeanor,
                    interests          = excluded.interests,
                    profile_updated_at = excluded.profile_updated_at
                """,
                (
                    twitch_id,
                    new_pronouns, new_location, new_demeanor,
                    json.dumps(new_interests) if new_interests else None,
                    _now_iso(),
                ),
            )

    def insert_message(
        self,
        user_id: str,
        content: str,
        *,
        reply_parent_login: str | None = None,
        reply_parent_body: str | None = None,
        is_emote_only: bool = False,
        spam_score: float = 0.0,
        spam_reasons: list[str] | None = None,
    ) -> int:
        """Insert a chat message and return its rowid. `reply_parent_*` are
        populated when the platform reports the message used a native Reply.
        `is_emote_only` is set when the message text is just emotes +
        whitespace; summarizer and moderator skip those.

        `spam_score` (0.0..1.0) and `spam_reasons` come from the
        ingest-time spam detector (see `spam.py`). Stored as a score so
        each consumer can pick its own threshold; reasons array is for
        audit transparency."""
        from .spam import encode_reasons
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages(user_id, ts, content,
                                     reply_parent_login, reply_parent_body,
                                     is_emote_only,
                                     spam_score, spam_reasons)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id, _now_iso(), content,
                    reply_parent_login, reply_parent_body,
                    1 if is_emote_only else 0,
                    float(spam_score),
                    encode_reasons(list(spam_reasons or [])),
                ),
            )
            return int(cur.lastrowid)

    def merge_users(self, child_id: str, parent_id: str) -> dict[str, int]:
        """Merge `child_id` into `parent_id`. Rewrites every FK to point at
        the parent, sets `merged_into` on the child, and folds the child's
        aliases under the parent. The child row stays around for "merged
        from X" provenance — it just no longer owns any data.

        Idempotent for any (child, parent) pair where child still exists.
        Refuses if either is missing or if parent is itself merged.

        Returns a dict of how many rows were rewritten per table."""
        if child_id == parent_id:
            raise ValueError("cannot merge a user into itself")
        with self._cursor() as cur:
            cur.execute(
                "SELECT twitch_id, merged_into FROM users WHERE twitch_id IN (?, ?)",
                (child_id, parent_id),
            )
            rows = {r["twitch_id"]: r for r in cur.fetchall()}
            if child_id not in rows or parent_id not in rows:
                raise ValueError("child or parent not found")
            if rows[parent_id]["merged_into"]:
                raise ValueError(
                    f"parent {parent_id} is itself merged into "
                    f"{rows[parent_id]['merged_into']}"
                )

            counts: dict[str, int] = {}
            for table in ("messages", "notes", "events", "reminders", "incidents"):
                cur.execute(
                    f"UPDATE {table} SET user_id = ? WHERE user_id = ?",
                    (parent_id, child_id),
                )
                counts[table] = int(cur.rowcount)

            # Watermark — keep the larger of the two (we've definitely
            # summarized everything up to it on the parent side).
            cur.execute(
                "SELECT user_id, last_summarized_msg_id FROM summarization_state "
                "WHERE user_id IN (?, ?)",
                (child_id, parent_id),
            )
            wms = {r["user_id"]: int(r["last_summarized_msg_id"]) for r in cur.fetchall()}
            new_wm = max(wms.get(child_id, 0), wms.get(parent_id, 0))
            if new_wm:
                cur.execute(
                    """
                    INSERT INTO summarization_state(user_id, last_summarized_msg_id)
                    VALUES (?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        last_summarized_msg_id = excluded.last_summarized_msg_id
                    """,
                    (parent_id, new_wm),
                )
            cur.execute(
                "DELETE FROM summarization_state WHERE user_id = ?", (child_id,)
            )

            # Aliases — insert child's under parent (skip on conflict), then
            # drop the child's rows. The child's display name itself becomes
            # an alias of the parent.
            cur.execute(
                """
                INSERT OR IGNORE INTO user_aliases(user_id, name, first_seen, last_seen_as)
                SELECT ?, name, first_seen, last_seen_as FROM user_aliases WHERE user_id = ?
                """,
                (parent_id, child_id),
            )
            cur.execute(
                """
                INSERT OR IGNORE INTO user_aliases(user_id, name, first_seen, last_seen_as)
                SELECT ?, name, first_seen, last_seen FROM users WHERE twitch_id = ?
                """,
                (parent_id, child_id),
            )
            cur.execute(
                "DELETE FROM user_aliases WHERE user_id = ?", (child_id,)
            )

            # Anything that pointed at the child via its old id should keep
            # working through the merged_into hop.
            cur.execute(
                "UPDATE users SET merged_into = ? WHERE twitch_id = ?",
                (parent_id, child_id),
            )
            return counts

    def list_merged_children(self, parent_id: str) -> list[User]:
        """Users that were merged INTO this one. Used by the user page to
        show 'merged from: X (Discord), Y (YouTube)'."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT twitch_id, name, first_seen, last_seen, opt_out, "
                "sub_tier, sub_months, is_mod, is_vip, is_founder, "
                "source, merged_into "
                "FROM users WHERE merged_into = ? ORDER BY source, name",
                (parent_id,),
            )
            return [
                User(
                    twitch_id=r["twitch_id"], name=r["name"],
                    first_seen=r["first_seen"], last_seen=r["last_seen"],
                    opt_out=bool(r["opt_out"]),
                    sub_tier=r["sub_tier"],
                    sub_months=int(r["sub_months"] or 0),
                    is_mod=bool(r["is_mod"]),
                    is_vip=bool(r["is_vip"]),
                    is_founder=bool(r["is_founder"]),
                    source=r["source"] or "twitch",
                    merged_into=r["merged_into"],
                )
                for r in cur.fetchall()
            ]

    def search_users_for_merge(
        self, query: str, *, exclude_id: str, limit: int = 10
    ) -> list[User]:
        """Name-search across un-merged users for the merge picker. Excludes
        the user being merged FROM and any other already-merged children."""
        q = f"%{query.strip().lower()}%"
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT u.twitch_id, u.name, u.first_seen, u.last_seen,
                                u.opt_out, u.sub_tier, u.sub_months,
                                u.is_mod, u.is_vip, u.is_founder,
                                u.source, u.merged_into
                FROM users u
                LEFT JOIN user_aliases a ON a.user_id = u.twitch_id
                WHERE u.twitch_id != ?
                  AND u.merged_into IS NULL
                  AND (LOWER(u.name) LIKE ? OR LOWER(a.name) LIKE ?)
                ORDER BY u.last_seen DESC
                LIMIT ?
                """,
                (exclude_id, q, q, int(limit)),
            )
            return [
                User(
                    twitch_id=r["twitch_id"], name=r["name"],
                    first_seen=r["first_seen"], last_seen=r["last_seen"],
                    opt_out=bool(r["opt_out"]),
                    sub_tier=r["sub_tier"],
                    sub_months=int(r["sub_months"] or 0),
                    is_mod=bool(r["is_mod"]),
                    is_vip=bool(r["is_vip"]),
                    is_founder=bool(r["is_founder"]),
                    source=r["source"] or "twitch",
                    merged_into=r["merged_into"],
                )
                for r in cur.fetchall()
            ]

    def latest_message_id(self) -> int:
        """The highest message id in the table, or 0 if empty. Used by the
        topics + insights loops as a freshness check — if this hasn't
        advanced since their last run, there's no point re-summarizing the
        same window."""
        with self._cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(id), 0) AS m FROM messages")
            return int(cur.fetchone()["m"])

    # ============================================================
    # CHANGE-WATERMARK QUERIES (for the /events/stream multiplexed SSE)
    # ------------------------------------------------------------
    # Each channel of the dashboard's event bus polls a tiny "version
    # stamp" SQL — typically a MAX(id) or MAX(ts) — and emits an SSE
    # event when the value changes. Polling the DB every 1.5 s for 4
    # of these queries is microseconds; the win is that downstream
    # HTMX-rendered panels stop re-rendering on a fixed Xs cadence.
    # See web/app.py /events/stream for the consumers.
    # ============================================================

    def latest_topic_thread_version(self) -> str:
        """Compact version stamp for the topic_threads list. Captures
        any add / title-update / recap-update / member-add. Returned as
        a string so the SSE poller can compare via ==."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT
                  COUNT(*)                               AS n,
                  COALESCE(MAX(t.last_ts), '')           AS lts,
                  COALESCE(MAX(t.recap_updated_at), '')  AS rua
                FROM topic_threads t
                """
            )
            r = cur.fetchone()
            return f"{int(r['n'])}|{r['lts']}|{r['rua']}"

    def latest_transcript_chunk_id(self) -> int:
        """Highest transcript chunk id, or 0 if empty. Watermark for
        the live transcript strip's SSE channel."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(id), 0) AS m FROM transcript_chunks"
            )
            return int(cur.fetchone()["m"])

    def latest_event_id(self) -> int:
        """Highest StreamElements / Twitch event id. Drives the events
        list channel; bumps on any new tip / sub / cheer / follow."""
        with self._cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(id), 0) AS m FROM events")
            return int(cur.fetchone()["m"])

    def latest_user_change_version(self) -> str:
        """Coarse "anything changed in users table" version. Drives the
        chatters-list channel. Captures new users (count) AND any
        last_seen update (max), so badge changes / new arrivals all
        fire."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS n, COALESCE(MAX(last_seen), '') AS m FROM users"
            )
            r = cur.fetchone()
            return f"{int(r['n'])}|{r['m']}"

    def is_opted_out(self, twitch_id: str) -> bool:
        with self._cursor() as cur:
            cur.execute("SELECT opt_out FROM users WHERE twitch_id = ?", (twitch_id,))
            row = cur.fetchone()
            return bool(row and row["opt_out"])

    def any_opted_out(self, user_ids: Iterable[str]) -> bool:
        """True if ANY user in the list has opt_out=1. Single SQL
        round trip via IN(...) so the dataset capture filter can
        decide whether to drop an event without N round trips.

        Empty input returns False (no users to check). Unknown user
        ids return False — they aren't in the table so there's no
        opt-out to honour."""
        ids = [u for u in user_ids if u]
        if not ids:
            return False
        placeholders = ",".join("?" for _ in ids)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT 1 FROM users WHERE opt_out = 1 "
                f"AND twitch_id IN ({placeholders}) LIMIT 1",
                ids,
            )
            return cur.fetchone() is not None

    # ============================ Summarizer-pipeline surface ==============

    def get_watermark(self, user_id: str) -> int:
        with self._cursor() as cur:
            cur.execute(
                "SELECT last_summarized_msg_id FROM summarization_state WHERE user_id = ?",
                (user_id,),
            )
            row = cur.fetchone()
            return int(row["last_summarized_msg_id"]) if row else 0

    def set_watermark(self, user_id: str, last_msg_id: int) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO summarization_state(user_id, last_summarized_msg_id)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    last_summarized_msg_id = excluded.last_summarized_msg_id
                """,
                (user_id, last_msg_id),
            )

    def messages_since_watermark(self, user_id: str) -> list[tuple[int, str]]:
        wm = self.get_watermark(user_id)
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT id, content
                FROM messages
                WHERE user_id = ? AND id > ? AND is_emote_only = 0 AND spam_score < 0.5
                ORDER BY id ASC
                """,
                (user_id, wm),
            )
            return [(int(r["id"]), r["content"]) for r in cur.fetchall()]

    def unsummarized_count(self, user_id: str) -> int:
        wm = self.get_watermark(user_id)
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM messages "
                "WHERE user_id = ? AND id > ? AND is_emote_only = 0 AND spam_score < 0.5",
                (user_id, wm),
            )
            return int(cur.fetchone()["c"])

    def users_with_unsummarized_count_at_least(self, threshold: int) -> list[str]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT m.user_id
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                LEFT JOIN summarization_state s ON s.user_id = m.user_id
                WHERE u.opt_out = 0
                  AND m.is_emote_only = 0 AND m.spam_score < 0.5
                  AND m.id > COALESCE(s.last_summarized_msg_id, 0)
                GROUP BY m.user_id
                HAVING COUNT(*) >= ?
                """,
                (threshold,),
            )
            return [r["user_id"] for r in cur.fetchall()]

    def users_with_idle_unsummarized(self, idle_minutes: int) -> list[str]:
        """Users who have unsummarized messages and whose latest message is older
        than `idle_minutes`."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT m.user_id
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                LEFT JOIN summarization_state s ON s.user_id = m.user_id
                WHERE u.opt_out = 0
                  AND m.is_emote_only = 0 AND m.spam_score < 0.5
                  AND m.id > COALESCE(s.last_summarized_msg_id, 0)
                GROUP BY m.user_id
                HAVING MAX(m.ts) <= datetime('now', ?)
                """,
                (f"-{int(idle_minutes)} minutes",),
            )
            return [r["user_id"] for r in cur.fetchall()]

    def add_note(
        self,
        user_id: str,
        text: str,
        embedding: list[float] | None,
        source_message_ids: list[int] | None = None,
        *,
        origin: str = "manual",
    ) -> int | None:
        """Insert a note. `origin` ∈ {'manual', 'llm'} marks provenance so
        the dashboard can label it correctly.

        For LLM notes, `source_message_ids` MUST resolve to at least one
        message owned by this user — that's the hallucination guard. If
        none validate, the note is dropped entirely and None is returned
        (the LLM is making things up and we won't store it).

        Manual notes never need sources; they save unconditionally.
        """
        valid: list[int] = []
        if source_message_ids:
            with self._cursor() as cur:
                placeholders = ",".join("?" for _ in source_message_ids)
                cur.execute(
                    f"SELECT id FROM messages WHERE user_id = ? AND id IN ({placeholders})",
                    (user_id, *source_message_ids),
                )
                valid = [int(r["id"]) for r in cur.fetchall()]

        if origin == "llm" and not valid:
            # Hallucination guard: the LLM emitted a note it can't tie to
            # any actual message this user sent. Drop it.
            return None

        blob = _vec_to_blob(embedding) if embedding else None
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO notes(user_id, ts, text, embedding, origin) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, _now_iso(), text, blob, origin),
            )
            note_id = int(cur.lastrowid)
            if blob is not None:
                cur.execute(
                    "INSERT INTO vec_notes(note_id, embedding) VALUES (?, ?)",
                    (note_id, blob),
                )
            for mid in valid:
                cur.execute(
                    "INSERT OR IGNORE INTO note_sources(note_id, message_id) "
                    "VALUES (?, ?)",
                    (note_id, mid),
                )
        # Capture for the streamer-personal training dataset. Origin
        # rides along in the action label so the reader can tell
        # human-authored notes from LLM-extracted ones — both are
        # useful supervision signals but they mean different things
        # for fine-tuning.
        self._capture_streamer_action(
            action_kind="note",
            item_key=str(note_id),
            action=f"created:{origin}",
            note=text,
        )
        return note_id

    def arrival_signals_for_users(
        self, user_ids: Iterable[str], *, returning_min_hours: int = 6
    ) -> dict[str, str]:
        """Return per-user arrival signal: 'first_timer', 'returning', or
        absent. `first_timer` = first_seen within last 24h. `returning` = the
        previous message from this user is at least `returning_min_hours` old.
        Pure read; used by the live widget to highlight notable arrivals."""
        ids = [u for u in user_ids if u]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        out: dict[str, str] = {}
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT u.twitch_id, u.first_seen,
                       (julianday('now') - julianday(u.first_seen)) * 24 AS first_age_h,
                       (
                         SELECT (julianday('now') - julianday(m2.ts)) * 24
                         FROM messages m2
                         WHERE m2.user_id = u.twitch_id
                         ORDER BY m2.id DESC LIMIT 1 OFFSET 1
                       ) AS prev_age_h
                FROM users u
                WHERE u.twitch_id IN ({placeholders})
                """,
                tuple(ids),
            )
            for r in cur.fetchall():
                first_age = r["first_age_h"]
                prev_age = r["prev_age_h"]
                if first_age is not None and float(first_age) <= 24.0:
                    out[r["twitch_id"]] = "first_timer"
                elif prev_age is not None and float(prev_age) >= returning_min_hours:
                    out[r["twitch_id"]] = "returning"
        return out

    def recent_global_messages_after_id(
        self, after_id: int, *, limit: int = 50,
    ) -> list[Message]:
        """Messages with id > after_id, newest first. Used by the
        live-chat SSE stream — server polls this with the connection's
        watermark every ~1 s and emits whatever's new. Bounded by
        `limit` so a chatty surge doesn't blow the SSE event size.
        Empty list (no new messages) is the common case; SSE sends
        nothing in that case beyond the heartbeat."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT m.id, m.user_id, u.name, u.source,
                       m.ts, m.content, m.reply_parent_login, m.reply_parent_body
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                WHERE m.id > ?
                ORDER BY m.id DESC
                LIMIT ?
                """,
                (int(after_id), int(limit)),
            )
            return [
                Message(
                    id=int(r["id"]),
                    user_id=r["user_id"],
                    name=r["name"],
                    ts=r["ts"],
                    content=r["content"],
                    reply_parent_login=r["reply_parent_login"],
                    reply_parent_body=r["reply_parent_body"],
                    source=r["source"] or "twitch",
                )
                for r in cur.fetchall()
            ]

    def recent_global_messages(self, limit: int = 30) -> list[Message]:
        """Latest messages across all sources, newest first.

        Used by the dashboard's live-chat widget. Opted-out users contribute
        nothing because their messages aren't inserted in the first place.
        Returns rows from every platform (Twitch / YouTube / Discord) in one
        chronological bucket.
        """
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT m.id, m.user_id, u.name, u.source,
                       m.ts, m.content, m.reply_parent_login, m.reply_parent_body
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                ORDER BY m.id DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [
                Message(
                    id=int(r["id"]),
                    user_id=r["user_id"],
                    name=r["name"],
                    ts=r["ts"],
                    content=r["content"],
                    reply_parent_login=r["reply_parent_login"],
                    reply_parent_body=r["reply_parent_body"],
                    source=r["source"] or "twitch",
                )
                for r in cur.fetchall()
            ]

    # ============================================================
    # CANONICAL "CLEAN MESSAGE" FILTER
    # ------------------------------------------------------------
    # The single source of truth for "messages we want the LLM /
    # word-cloud / topic clusterer to see". Any read path that feeds
    # downstream NLP work should compose this WHERE-clause snippet
    # via _CLEAN_MSG_WHERE so future filter changes (a new spam
    # signal, a new opt_out variant) touch one place instead of N.
    # ============================================================
    _CLEAN_MSG_WHERE = (
        "u.opt_out = 0 "
        "AND m.is_emote_only = 0 "
        "AND m.spam_score < 0.5"
    )

    def recent_messages(
        self,
        *,
        limit: int = 250,
        within_minutes: int = 20,
        with_embeddings: bool = False,
    ) -> list[Message] | list[tuple[Message, list[float]]]:
        """Recent clean (non-emote, non-spam, opt-in) messages within
        the lookback window, oldest-first. Default returns
        `list[Message]`; when `with_embeddings=True`, each row is
        paired with its `vec_messages` vector and rows lacking an
        embedding (still in the index queue) are skipped.

        Used by the engaging-subjects extractor (with_embeddings=True
        for cluster-first labeling) and any other path that needs
        clean recent text. Filter logic comes from `_CLEAN_MSG_WHERE`
        — change spam / opt-out / emote rules there once."""
        join = (
            "JOIN vec_messages v ON v.message_id = m.id"
            if with_embeddings else ""
        )
        select_extra = ", v.embedding" if with_embeddings else ""
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT m.id, m.user_id, u.name, u.source, m.ts, m.content,
                       m.reply_parent_login, m.reply_parent_body
                       {select_extra}
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                {join}
                WHERE {self._CLEAN_MSG_WHERE}
                  AND datetime(m.ts) >= datetime('now', ?)
                ORDER BY m.id DESC
                LIMIT ?
                """,
                (f"-{int(within_minutes)} minutes", int(limit)),
            )
            rows = cur.fetchall()
        if with_embeddings:
            out_paired: list[tuple[Message, list[float]]] = []
            for r in rows:
                msg = Message(
                    id=int(r["id"]), user_id=r["user_id"], name=r["name"],
                    source=r["source"] or "twitch",
                    ts=r["ts"], content=r["content"],
                    reply_parent_login=r["reply_parent_login"],
                    reply_parent_body=r["reply_parent_body"],
                )
                try:
                    vec = _blob_to_vec(r["embedding"])
                except Exception:
                    continue
                out_paired.append((msg, vec))
            out_paired.reverse()
            return out_paired
        out = [
            Message(
                id=int(r["id"]), user_id=r["user_id"], name=r["name"],
                source=r["source"] or "twitch",
                ts=r["ts"], content=r["content"],
                reply_parent_login=r["reply_parent_login"],
                reply_parent_body=r["reply_parent_body"],
            )
            for r in rows
        ]
        out.reverse()
        return out

    def recent_messages_with_embeddings(
        self, *, limit: int = 250, within_minutes: int = 20,
    ) -> list[tuple[Message, list[float]]]:
        """Compatibility shim — prefer `recent_messages(...,
        with_embeddings=True)` directly. Kept so existing callers
        don't break during the staged rollout."""
        return self.recent_messages(
            limit=limit, within_minutes=within_minutes,
            with_embeddings=True,
        )  # type: ignore[return-value]

    def recent_messages_for_topics(self, limit: int) -> list[tuple[int, str, str]]:
        """Latest `limit` messages across all opted-in users, oldest-first."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT m.id, u.name, m.content
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                WHERE u.opt_out = 0
                ORDER BY m.id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = [(int(r["id"]), r["name"], r["content"]) for r in cur.fetchall()]
        rows.reverse()
        return rows

    def add_topic_snapshot(
        self,
        summary: str,
        message_id_range: str | None,
        topics_json: str | None = None,
    ) -> int:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO topic_snapshots(ts, summary, message_id_range, topics_json)
                VALUES (?, ?, ?, ?)
                """,
                (_now_iso(), summary, message_id_range, topics_json),
            )
            return int(cur.lastrowid)

    # ============================ Events (StreamElements + YouTube) =======

    def record_event_for_user_id(
        self,
        user_id: str,
        display_name: str,
        event_type: str,
        *,
        amount: float | None = None,
        currency: str | None = None,
        message: str | None = None,
        raw: dict[str, Any] | None = None,
    ) -> int:
        """Direct event insert by user_id. Used by cross-platform listeners
        (e.g. YouTube super-chats / memberships) that already know the
        canonical user_id. The streamelements path uses `record_event` and
        looks up by name — that doesn't work cross-platform because a
        YouTube display name could collide with a Twitch chatter's name."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO events(user_id, twitch_name, type, amount, currency, message, ts, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    display_name,
                    event_type,
                    amount,
                    currency,
                    message,
                    _now_iso(),
                    json.dumps(raw) if raw is not None else None,
                ),
            )
            return int(cur.lastrowid)

    def record_event(
        self,
        twitch_name: str,
        event_type: str,
        amount: float | None = None,
        currency: str | None = None,
        message: str | None = None,
        raw: dict[str, Any] | None = None,
    ) -> int:
        with self._cursor() as cur:
            user_id = None
            if twitch_name:
                cur.execute(
                    "SELECT twitch_id FROM users WHERE LOWER(name) = LOWER(?) LIMIT 1",
                    (twitch_name,),
                )
                row = cur.fetchone()
                if row:
                    user_id = row["twitch_id"]
            cur.execute(
                """
                INSERT INTO events(user_id, twitch_name, type, amount, currency, message, ts, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    twitch_name,
                    event_type,
                    amount,
                    currency,
                    message,
                    _now_iso(),
                    json.dumps(raw) if raw is not None else None,
                ),
            )
            return int(cur.lastrowid)

    # ============================ READ surface (TUI / dashboard) ===========

    # ============================ Reminders ================================

    def add_reminder(self, user_id: str, text: str) -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO reminders(user_id, text, created_at) VALUES (?, ?, ?)",
                (user_id, text.strip(), _now_iso()),
            )
            return int(cur.lastrowid)

    def fire_pending_reminders(self, user_id: str) -> int:
        """Stamp every pending reminder for this user as fired-now. Returns
        the count of newly-fired reminders. Called by the bot when this user
        next speaks."""
        with self._cursor() as cur:
            cur.execute(
                """
                UPDATE reminders
                SET fired_at = ?
                WHERE user_id = ? AND fired_at IS NULL AND dismissed = 0
                """,
                (_now_iso(), user_id),
            )
            return int(cur.rowcount)

    def dismiss_reminder(self, reminder_id: int) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE reminders SET dismissed = 1 WHERE id = ?", (reminder_id,)
            )

    def delete_reminder(self, reminder_id: int) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))

    def get_reminders_for_user(self, user_id: str) -> list[Reminder]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT r.id, r.user_id, u.name AS user_name, r.text, r.created_at,
                       r.fired_at, r.dismissed
                FROM reminders r
                JOIN users u ON u.twitch_id = r.user_id
                WHERE r.user_id = ?
                ORDER BY r.dismissed ASC, r.created_at DESC
                """,
                (user_id,),
            )
            return [
                Reminder(
                    id=int(r["id"]), user_id=r["user_id"], user_name=r["user_name"],
                    text=r["text"], created_at=r["created_at"],
                    fired_at=r["fired_at"], dismissed=bool(r["dismissed"]),
                )
                for r in cur.fetchall()
            ]

    def list_fired_reminders(self) -> list[Reminder]:
        """All reminders that have fired and not been dismissed yet —
        the inbox the streamer needs to clear."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT r.id, r.user_id, u.name AS user_name, r.text, r.created_at,
                       r.fired_at, r.dismissed
                FROM reminders r
                JOIN users u ON u.twitch_id = r.user_id
                WHERE r.fired_at IS NOT NULL AND r.dismissed = 0
                ORDER BY r.fired_at DESC
                """
            )
            return [
                Reminder(
                    id=int(r["id"]), user_id=r["user_id"], user_name=r["user_name"],
                    text=r["text"], created_at=r["created_at"],
                    fired_at=r["fired_at"], dismissed=bool(r["dismissed"]),
                )
                for r in cur.fetchall()
            ]

    def count_fired_reminders(self) -> int:
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM reminders "
                "WHERE fired_at IS NOT NULL AND dismissed = 0"
            )
            return int(cur.fetchone()["c"])

    def count_pending_reminders_for_user(self, user_id: str) -> int:
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM reminders "
                "WHERE user_id = ? AND fired_at IS NULL AND dismissed = 0",
                (user_id,),
            )
            return int(cur.fetchone()["c"])

    # ============================ Insights state machine ===================
    # Per-card state for the Insights page. The streamer's primary surface
    # for managing what they've addressed on stream vs what to come back to.

    # 'auto_pending' is set by the transcript service when whisper hears
    # the streamer say something matching an open card (cosine sim
    # ≥ whisper_match_threshold). The UI shows a confirm/reject prompt;
    # an auto_confirm_loop flips remaining pendings → addressed after a
    # short timeout so passive workflow still works.
    _INSIGHT_STATES = frozenset({
        "addressed", "snoozed", "pinned", "skipped", "auto_pending",
    })

    def list_pending_auto_addresses(
        self, *, older_than_seconds: int = 60,
    ) -> list[InsightState]:
        """Auto-pending entries whose `updated_at` is older than the
        confirmation window. Drives the auto_confirm_loop."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=int(older_than_seconds))
        ).isoformat(timespec="seconds")
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT kind, item_key, state, due_ts, note, updated_at
                FROM insight_states
                WHERE state = 'auto_pending' AND updated_at <= ?
                ORDER BY updated_at ASC
                """,
                (cutoff,),
            )
            return [
                InsightState(
                    kind=r["kind"], item_key=r["item_key"], state=r["state"],
                    due_ts=r["due_ts"], note=r["note"],
                    updated_at=r["updated_at"],
                )
                for r in cur.fetchall()
            ]

    def set_insight_state(
        self,
        kind: str,
        item_key: str,
        state: str,
        *,
        due_ts: str | None = None,
        note: str | None = None,
    ) -> None:
        """Upsert insight card state. Pass state='open' (or any value not
        in _INSIGHT_STATES) to clear the row entirely. Every transition
        is appended to insight_state_history for the audit trail."""
        now = _now_iso()
        if state not in self._INSIGHT_STATES:
            with self._cursor() as cur:
                cur.execute(
                    "DELETE FROM insight_states WHERE kind = ? AND item_key = ?",
                    (kind, item_key),
                )
                cur.execute(
                    "INSERT INTO insight_state_history"
                    "(ts, kind, item_key, state, due_ts, note) "
                    "VALUES (?, ?, ?, 'open', NULL, NULL)",
                    (now, kind, item_key),
                )
            self._capture_streamer_action(
                action_kind="insight_state",
                item_key=f"{kind}:{item_key}",
                action="open",
                note=None,
            )
            return
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO insight_states(kind, item_key, state, due_ts, note, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(kind, item_key) DO UPDATE SET
                    state = excluded.state,
                    due_ts = excluded.due_ts,
                    note = COALESCE(excluded.note, insight_states.note),
                    updated_at = excluded.updated_at
                """,
                (kind, item_key, state, due_ts, note, now),
            )
            cur.execute(
                "INSERT INTO insight_state_history"
                "(ts, kind, item_key, state, due_ts, note) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (now, kind, item_key, state, due_ts, note),
            )
        # Capture AFTER the SQL commits — a row that wasn't actually
        # written must not leave a STREAMER_ACTION trail. Compose the
        # key as `<insight_kind>:<item_key>` so the dataset reader can
        # filter by kind without parsing.
        self._capture_streamer_action(
            action_kind="insight_state",
            item_key=f"{kind}:{item_key}",
            action=state,
            note=note,
        )

    def list_state_history(
        self, *, since: str | None = None, limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Recent insight-state transitions for the audit trail page.
        `since` is an ISO timestamp; defaults to the last 24h."""
        if not since:
            since = (
                datetime.now(timezone.utc) - timedelta(hours=24)
            ).isoformat(timespec="seconds")
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT ts, kind, item_key, state, due_ts, note
                FROM insight_state_history
                WHERE ts >= ?
                ORDER BY ts DESC LIMIT ?
                """,
                (since, int(limit)),
            )
            return [dict(r) for r in cur.fetchall()]

    def count_state_changes_since(self, since: str, *, state: str | None = None) -> int:
        """How many state transitions happened since `since`. Optional
        `state` filter (e.g. 'addressed') so the recap can cite "you
        addressed N items this stream"."""
        with self._cursor() as cur:
            params: list[Any] = [since]
            extra = ""
            if state:
                extra = " AND state = ?"
                params.append(state)
            cur.execute(
                f"SELECT COUNT(*) AS c FROM insight_state_history "
                f"WHERE ts >= ? {extra}",
                params,
            )
            return int(cur.fetchone()["c"])

    def get_insight_states(self, kind: str) -> dict[str, InsightState]:
        """All current states for a given kind, keyed by item_key. The
        Insights renderer fetches once per kind and applies the state to
        each card via dict lookup."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT kind, item_key, state, due_ts, note, updated_at "
                "FROM insight_states WHERE kind = ?",
                (kind,),
            )
            return {
                r["item_key"]: InsightState(
                    kind=r["kind"], item_key=r["item_key"], state=r["state"],
                    due_ts=r["due_ts"], note=r["note"],
                    updated_at=r["updated_at"],
                )
                for r in cur.fetchall()
            }

    def list_recent_transcript_matches(
        self, *, limit: int = 20, window_minutes: int = 60,
    ) -> list[dict]:
        """Recent LLM-transcript matches for the Insights "Recent matches"
        panel. Reads `insight_state_history` for state='auto_pending' rows
        within the window — each is one moment-of-match — then joins the
        current state from `insight_states` so the row badge can show
        whether the card is still pending or has since been addressed /
        skipped / etc.

        Returns dicts with: ts, kind, item_key, evidence, title, href,
        state. `evidence` is the note with the leading "(auto) " stripped.
        Title + href resolution differs by kind:

          - thread       → topic_threads.title, /modals/thread/{id}
          - talking_point → users.name (parsed from item_key prefix),
                            /users/{user_id}
        """
        out: list[dict] = []
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT h.ts, h.kind, h.item_key, h.note,
                       s.state AS current_state
                FROM insight_state_history h
                LEFT JOIN insight_states s
                  ON s.kind = h.kind AND s.item_key = h.item_key
                WHERE h.state = 'auto_pending'
                  AND datetime(h.ts) >= datetime('now', ?)
                  AND h.note IS NOT NULL
                ORDER BY h.ts DESC
                LIMIT ?
                """,
                (f"-{int(window_minutes)} minutes", int(limit)),
            )
            rows = cur.fetchall()
            # Pre-resolve the threads + users we need so we don't run
            # a sub-query per row.
            thread_ids: set[int] = set()
            user_ids: set[str] = set()
            for r in rows:
                if r["kind"] == "thread":
                    try:
                        thread_ids.add(int(r["item_key"]))
                    except ValueError:
                        pass
                elif r["kind"] == "talking_point":
                    uid = (r["item_key"] or "").split(":", 1)[0]
                    if uid:
                        user_ids.add(uid)
            thread_titles: dict[int, str] = {}
            if thread_ids:
                placeholders = ",".join("?" for _ in thread_ids)
                cur.execute(
                    f"SELECT id, title FROM topic_threads WHERE id IN ({placeholders})",
                    list(thread_ids),
                )
                thread_titles = {int(r["id"]): r["title"] for r in cur.fetchall()}
            user_names: dict[str, str] = {}
            if user_ids:
                placeholders = ",".join("?" for _ in user_ids)
                cur.execute(
                    f"SELECT twitch_id, name FROM users WHERE twitch_id IN ({placeholders})",
                    list(user_ids),
                )
                user_names = {r["twitch_id"]: r["name"] for r in cur.fetchall()}
            for r in rows:
                note = r["note"] or ""
                evidence = note[7:].strip() if note.startswith("(auto) ") else note.strip()
                title: str
                href: str | None
                if r["kind"] == "thread":
                    try:
                        tid = int(r["item_key"])
                    except ValueError:
                        tid = -1
                    title = thread_titles.get(tid, f"thread #{r['item_key']}")
                    href = f"/modals/thread/{tid}" if tid >= 0 else None
                elif r["kind"] == "talking_point":
                    uid = (r["item_key"] or "").split(":", 1)[0]
                    title = user_names.get(uid, uid or "(unknown chatter)")
                    href = f"/users/{uid}" if uid else None
                else:
                    title = f"{r['kind']} card"
                    href = None
                out.append({
                    "ts": r["ts"],
                    "kind": r["kind"],
                    "item_key": r["item_key"],
                    "evidence": evidence,
                    "title": title,
                    "href": href,
                    "state": r["current_state"] or "auto_pending",
                })
        return out

    def count_due_snoozes(self) -> int:
        """Snoozed cards whose due_ts has passed — drives the nav pill so
        the streamer sees pending follow-ups at a glance."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM insight_states "
                "WHERE state = 'snoozed' AND due_ts IS NOT NULL AND due_ts <= ?",
                (_now_iso(),),
            )
            return int(cur.fetchone()["c"])

    def list_due_snoozes(self) -> list[InsightState]:
        """Detailed list of cards whose snooze has fired — used by the
        browser-notification opt-in to know what to show."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT kind, item_key, state, due_ts, note, updated_at "
                "FROM insight_states "
                "WHERE state = 'snoozed' AND due_ts IS NOT NULL AND due_ts <= ? "
                "ORDER BY due_ts ASC",
                (_now_iso(),),
            )
            return [
                InsightState(
                    kind=r["kind"], item_key=r["item_key"], state=r["state"],
                    due_ts=r["due_ts"], note=r["note"],
                    updated_at=r["updated_at"],
                )
                for r in cur.fetchall()
            ]

    # ============================ Transcript chunks =======================

    def add_transcript_chunk(
        self,
        *,
        text: str,
        duration_ms: int = 0,
        matched_kind: str | None = None,
        matched_item_key: str | None = None,
        similarity: float | None = None,
        embedding: list[float] | None = None,
        ts: str | None = None,
    ) -> int:
        """Persist one whisper-transcribed utterance + any auto-match
        state. The embedding (when provided) is mirrored into
        `vec_transcripts` so a chat message can later find a recent
        utterance it semantically responded to.

        `ts` defaults to wall-clock now. Callers can pass an explicit
        ISO timestamp when the audio's actual capture time is known
        and differs from arrival time — e.g. when audio was buffered
        on the OBS-side audio_client during a dashboard outage. Live
        ingest (no outage) just falls through to now()."""
        ts_value = ts or _now_iso()
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO transcript_chunks(
                    ts, duration_ms, text,
                    matched_kind, matched_item_key, similarity
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_value, int(duration_ms), text,
                    matched_kind, matched_item_key, similarity,
                ),
            )
            chunk_id = int(cur.lastrowid)
            if embedding:
                blob = _vec_to_blob(embedding)
                cur.execute(
                    "INSERT INTO vec_transcripts(chunk_id, embedding) "
                    "VALUES (?, ?)",
                    (chunk_id, blob),
                )
            return chunk_id

    def get_transcript_chunk(self, chunk_id: int) -> TranscriptChunk | None:
        with self._cursor() as cur:
            cur.execute(
                "SELECT id, ts, duration_ms, text, "
                "       matched_kind, matched_item_key, similarity "
                "FROM transcript_chunks WHERE id = ?",
                (int(chunk_id),),
            )
            r = cur.fetchone()
            if not r:
                return None
            return TranscriptChunk(
                id=int(r["id"]), ts=r["ts"],
                duration_ms=int(r["duration_ms"] or 0),
                text=r["text"],
                matched_kind=r["matched_kind"],
                matched_item_key=r["matched_item_key"],
                similarity=float(r["similarity"]) if r["similarity"] is not None else None,
            )

    def transcript_context_around(
        self, chunk_id: int, *, before: int = 3, after: int = 3,
    ) -> dict[str, Any]:
        """Focal transcript chunk + N adjacent ones for the context modal."""
        focal = self.get_transcript_chunk(chunk_id)
        if not focal:
            return {"focal": None, "before": [], "after": []}
        with self._cursor() as cur:
            cur.execute(
                "SELECT id, ts, duration_ms, text, "
                "       matched_kind, matched_item_key, similarity "
                "FROM transcript_chunks WHERE id < ? ORDER BY id DESC LIMIT ?",
                (int(chunk_id), int(before)),
            )
            before_rows = [
                TranscriptChunk(
                    id=int(r["id"]), ts=r["ts"],
                    duration_ms=int(r["duration_ms"] or 0),
                    text=r["text"],
                    matched_kind=r["matched_kind"],
                    matched_item_key=r["matched_item_key"],
                    similarity=float(r["similarity"]) if r["similarity"] is not None else None,
                )
                for r in cur.fetchall()
            ]
            before_rows.reverse()
            cur.execute(
                "SELECT id, ts, duration_ms, text, "
                "       matched_kind, matched_item_key, similarity "
                "FROM transcript_chunks WHERE id > ? ORDER BY id ASC LIMIT ?",
                (int(chunk_id), int(after)),
            )
            after_rows = [
                TranscriptChunk(
                    id=int(r["id"]), ts=r["ts"],
                    duration_ms=int(r["duration_ms"] or 0),
                    text=r["text"],
                    matched_kind=r["matched_kind"],
                    matched_item_key=r["matched_item_key"],
                    similarity=float(r["similarity"]) if r["similarity"] is not None else None,
                )
                for r in cur.fetchall()
            ]
        return {"focal": focal, "before": before_rows, "after": after_rows}

    def search_transcripts(
        self,
        query_embedding: list[float],
        *,
        k: int = 20,
    ) -> list[tuple[TranscriptChunk, float]]:
        """KNN search across ALL embedded transcript chunks. Returns
        (chunk, distance) pairs sorted nearest-first; cosine distance,
        so 0.0 = identical and ~1.0 = unrelated.

        For RETRIEVAL only — used by /search and the streamer-history
        evidence panels. Live LLM calls (talking points, group
        summaries, engaging subjects) explicitly DO NOT pull from this
        method; they have their own time-windowed sources so historical
        utterances can't pollute current-stream context."""
        blob = _vec_to_blob(query_embedding)
        # Pull a wider window from the index than the requested k so
        # post-filtering (when callers want it) still has matches.
        ann_k = max(int(k) * 2, 40)
        with self._cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT v.chunk_id, v.distance, t.id, t.ts,
                           t.duration_ms, t.text,
                           t.matched_kind, t.matched_item_key, t.similarity
                    FROM vec_transcripts v
                    JOIN transcript_chunks t ON t.id = v.chunk_id
                    WHERE v.embedding MATCH ? AND k = ?
                    ORDER BY v.distance ASC
                    LIMIT ?
                    """,
                    (blob, ann_k, int(k)),
                )
                rows = cur.fetchall()
            except sqlite3.OperationalError:
                return []
        return [
            (
                TranscriptChunk(
                    id=int(r["id"]), ts=r["ts"],
                    duration_ms=int(r["duration_ms"] or 0),
                    text=r["text"],
                    matched_kind=r["matched_kind"],
                    matched_item_key=r["matched_item_key"],
                    similarity=(
                        float(r["similarity"])
                        if r["similarity"] is not None else None
                    ),
                ),
                float(r["distance"]),
            )
            for r in rows
        ]

    def transcripts_missing_embedding(
        self, limit: int = 200,
    ) -> list[TranscriptChunk]:
        """Transcript chunks that don't yet have a vec_transcripts row.
        The live ingest path embeds at write time, but historical rows
        from before vec_transcripts was wired up — or rows where the
        embed call failed at ingest — sit here until the backfill loop
        gets to them. Oldest-first so backfill makes monotonic progress
        through the archive."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT t.id, t.ts, t.duration_ms, t.text,
                       t.matched_kind, t.matched_item_key, t.similarity
                FROM transcript_chunks t
                LEFT JOIN vec_transcripts v ON v.chunk_id = t.id
                WHERE v.chunk_id IS NULL
                  AND t.text IS NOT NULL AND length(trim(t.text)) > 0
                ORDER BY t.id ASC
                LIMIT ?
                """,
                (int(limit),),
            )
            return [
                TranscriptChunk(
                    id=int(r["id"]), ts=r["ts"],
                    duration_ms=int(r["duration_ms"] or 0),
                    text=r["text"],
                    matched_kind=r["matched_kind"],
                    matched_item_key=r["matched_item_key"],
                    similarity=(
                        float(r["similarity"])
                        if r["similarity"] is not None else None
                    ),
                )
                for r in cur.fetchall()
            ]

    def upsert_transcript_embedding(
        self, chunk_id: int, embedding: list[float],
    ) -> None:
        """Write / overwrite the embedding for one chunk. Used by the
        backfill loop; the ingest path writes inline via
        add_transcript_chunk(embedding=...)."""
        blob = _vec_to_blob(embedding)
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM vec_transcripts WHERE chunk_id = ?",
                (int(chunk_id),),
            )
            cur.execute(
                "INSERT INTO vec_transcripts(chunk_id, embedding) "
                "VALUES (?, ?)",
                (int(chunk_id), blob),
            )

    def transcripts_embedding_coverage(self) -> tuple[int, int]:
        """(indexed, total_text). Diagnostic for /search-style coverage
        readouts. `total_text` excludes empty / whitespace-only chunks."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM transcript_chunks "
                "WHERE text IS NOT NULL AND length(trim(text)) > 0"
            )
            total = int(cur.fetchone()["c"])
            cur.execute("SELECT COUNT(*) AS c FROM vec_transcripts")
            indexed = int(cur.fetchone()["c"])
        return indexed, total

    def find_related_transcript(
        self,
        query_embedding: list[float],
        *,
        max_age_minutes: int = 5,
        threshold: float = 0.40,
    ) -> tuple[TranscriptChunk, float] | None:
        """KNN-search vec_transcripts for the most-similar recent
        utterance to the query embedding. Used to link a chat message
        back to the streamer utterance it likely responded to."""
        blob = _vec_to_blob(query_embedding)
        cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=int(max_age_minutes))
        ).isoformat(timespec="seconds")
        with self._cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT v.chunk_id, v.distance, t.id, t.ts,
                           t.duration_ms, t.text,
                           t.matched_kind, t.matched_item_key, t.similarity
                    FROM vec_transcripts v
                    JOIN transcript_chunks t ON t.id = v.chunk_id
                    WHERE v.embedding MATCH ? AND k = 5
                      AND t.ts >= ?
                    ORDER BY v.distance ASC LIMIT 1
                    """,
                    (blob, cutoff),
                )
                r = cur.fetchone()
            except sqlite3.OperationalError:
                return None
        if not r:
            return None
        # vec0 cosine returns cosine DISTANCE (1 - cos). Convert to
        # similarity for the caller, gate on threshold.
        cosine_sim = 1.0 - float(r["distance"])
        if cosine_sim < threshold:
            return None
        chunk = TranscriptChunk(
            id=int(r["id"]), ts=r["ts"],
            duration_ms=int(r["duration_ms"] or 0),
            text=r["text"],
            matched_kind=r["matched_kind"],
            matched_item_key=r["matched_item_key"],
            similarity=float(r["similarity"]) if r["similarity"] is not None else None,
        )
        return chunk, cosine_sim

    # ---- transcript groups (LLM-summarized windows) --------------------

    def add_transcript_group(
        self, *, start_ts: str, end_ts: str,
        first_chunk_id: int, last_chunk_id: int, summary: str,
        context_message_ids: list[int] | None = None,
    ) -> int:
        """Persist one summarised group. `context_message_ids` records
        the chat messages the LLM saw as the CHAT DURING THIS WINDOW
        block — the modal hydrates those exact rows so the streamer
        sees the same chat the LLM did, not a re-queried approximation."""
        ctx_json = (
            json.dumps([int(i) for i in context_message_ids])
            if context_message_ids else None
        )
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO transcript_groups(
                    start_ts, end_ts, first_chunk_id, last_chunk_id,
                    summary, created_at, context_message_ids
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (start_ts, end_ts, int(first_chunk_id), int(last_chunk_id),
                 summary.strip(), _now_iso(), ctx_json),
            )
            return int(cur.lastrowid)

    # Sentinel summary text the grouper writes when the LLM gave up on
    # a window but the watermark still needs to advance. Filtered out
    # of the strip render by default so they don't crowd the view.
    PLACEHOLDER_GROUP_SUMMARY = "(no coherent summary available)"

    def list_transcript_groups(
        self, *, limit: int = 15, include_placeholders: bool = False,
    ) -> list[TranscriptGroup]:
        """Recent groups, newest first. Drives the live transcript strip.
        Placeholder rows (the LLM gave up but we needed to advance the
        watermark) are hidden by default — they exist for the system,
        not for the streamer."""
        if include_placeholders:
            sql = (
                "SELECT id, start_ts, end_ts, first_chunk_id, last_chunk_id, "
                "       summary, created_at "
                "FROM transcript_groups ORDER BY id DESC LIMIT ?"
            )
            params: tuple = (int(limit),)
        else:
            sql = (
                "SELECT id, start_ts, end_ts, first_chunk_id, last_chunk_id, "
                "       summary, created_at "
                "FROM transcript_groups "
                "WHERE summary IS NOT NULL AND summary != '' AND summary != ? "
                "ORDER BY id DESC LIMIT ?"
            )
            params = (self.PLACEHOLDER_GROUP_SUMMARY, int(limit))
        with self._cursor() as cur:
            cur.execute(sql, params)
            return [
                TranscriptGroup(
                    id=int(r["id"]), start_ts=r["start_ts"], end_ts=r["end_ts"],
                    first_chunk_id=int(r["first_chunk_id"]),
                    last_chunk_id=int(r["last_chunk_id"]),
                    summary=r["summary"], created_at=r["created_at"],
                )
                for r in cur.fetchall()
            ]

    def get_transcript_group(self, group_id: int) -> TranscriptGroup | None:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT id, start_ts, end_ts, first_chunk_id, last_chunk_id,
                       summary, created_at, context_message_ids
                FROM transcript_groups WHERE id = ?
                """,
                (int(group_id),),
            )
            r = cur.fetchone()
            if not r:
                return None
            ctx_ids: list[int] = []
            raw = r["context_message_ids"] if "context_message_ids" in r.keys() else None
            if raw:
                try:
                    ctx_ids = [int(i) for i in json.loads(raw) if i is not None]
                except (TypeError, ValueError):
                    ctx_ids = []
            return TranscriptGroup(
                id=int(r["id"]), start_ts=r["start_ts"], end_ts=r["end_ts"],
                first_chunk_id=int(r["first_chunk_id"]),
                last_chunk_id=int(r["last_chunk_id"]),
                summary=r["summary"], created_at=r["created_at"],
                context_message_ids=ctx_ids,
            )

    def latest_transcript_group_last_chunk_id(self) -> int:
        """Watermark for the grouper loop — returns the largest chunk id
        already grouped, or 0 if no groups yet."""
        with self._cursor() as cur:
            r = cur.execute(
                "SELECT COALESCE(MAX(last_chunk_id), 0) AS x FROM transcript_groups"
            ).fetchone()
            return int(r["x"]) if r else 0

    # ---- transcript screenshots -----------------------------------------

    def add_transcript_screenshot(
        self, *, ts: str, path: str, scene_name: str | None,
    ) -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO transcript_screenshots(ts, path, scene_name) "
                "VALUES (?, ?, ?)",
                (ts, path, scene_name),
            )
            return int(cur.lastrowid)

    def screenshots_in_range(
        self, start_ts: str, end_ts: str, *, max_count: int = 4,
    ) -> list[TranscriptScreenshot]:
        """All screenshots between [start_ts, end_ts] inclusive, picked
        evenly so we never blow past `max_count`. Comparison is done via
        `datetime()` so the trailing `+00:00` on stored ISO timestamps
        doesn't break against SQLite's space-separated `datetime('now')`
        outputs."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT id, ts, path, scene_name
                FROM transcript_screenshots
                WHERE datetime(ts) >= datetime(?)
                  AND datetime(ts) <= datetime(?)
                ORDER BY ts ASC
                """,
                (start_ts, end_ts),
            )
            rows = [
                TranscriptScreenshot(
                    id=int(r["id"]), ts=r["ts"], path=r["path"],
                    scene_name=r["scene_name"],
                )
                for r in cur.fetchall()
            ]
        if len(rows) <= max_count:
            return rows
        # Evenly spaced subsample: indices 0 and N-1 always present so
        # the strip captures the start + end of the window.
        n = len(rows)
        k = max(1, int(max_count))
        idx_set = {0, n - 1}
        if k > 2:
            for i in range(1, k - 1):
                idx_set.add(int(round(i * (n - 1) / (k - 1))))
        return [rows[i] for i in sorted(idx_set)][:k]

    def delete_screenshots_older_than(self, cutoff_iso: str) -> list[str]:
        """Delete rows older than `cutoff_iso` and return the file paths
        that should now be removed from disk. Caller does the unlink so
        we don't mix DB and FS concerns inside the cursor lock."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT path FROM transcript_screenshots WHERE datetime(ts) < datetime(?)",
                (cutoff_iso,),
            )
            paths = [r["path"] for r in cur.fetchall()]
            cur.execute(
                "DELETE FROM transcript_screenshots WHERE datetime(ts) < datetime(?)",
                (cutoff_iso,),
            )
        return paths

    def transcript_chunks_in_id_range(
        self, first_id: int, last_id: int,
    ) -> list[TranscriptChunk]:
        """Pull chunks by inclusive id range — used by the group-detail
        modal to expand a summary back to its underlying utterances."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT id, ts, duration_ms, text,
                       matched_kind, matched_item_key, similarity
                FROM transcript_chunks
                WHERE id >= ? AND id <= ? ORDER BY id ASC
                """,
                (int(first_id), int(last_id)),
            )
            return [
                TranscriptChunk(
                    id=int(r["id"]), ts=r["ts"],
                    duration_ms=int(r["duration_ms"] or 0),
                    text=r["text"],
                    matched_kind=r["matched_kind"],
                    matched_item_key=r["matched_item_key"],
                    similarity=float(r["similarity"]) if r["similarity"] is not None else None,
                )
                for r in cur.fetchall()
            ]

    def list_transcripts_after_id(
        self, after_id: int, *, limit: int = 200,
    ) -> list[TranscriptChunk]:
        """Chunks with id > after_id, oldest first. Drives the batched
        LLM matching loop, which advances a watermark via app_settings
        each time it processes a window."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT id, ts, duration_ms, text,
                       matched_kind, matched_item_key, similarity
                FROM transcript_chunks WHERE id > ?
                ORDER BY id ASC LIMIT ?
                """,
                (int(after_id), int(limit)),
            )
            return [
                TranscriptChunk(
                    id=int(r["id"]), ts=r["ts"],
                    duration_ms=int(r["duration_ms"] or 0),
                    text=r["text"],
                    matched_kind=r["matched_kind"],
                    matched_item_key=r["matched_item_key"],
                    similarity=float(r["similarity"]) if r["similarity"] is not None else None,
                )
                for r in cur.fetchall()
            ]

    def list_transcript_chunks(self, *, limit: int = 30) -> list[TranscriptChunk]:
        """Recent utterances, newest first. Drives the live transcript
        strip on Insights."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT id, ts, duration_ms, text,
                       matched_kind, matched_item_key, similarity
                FROM transcript_chunks
                ORDER BY id DESC LIMIT ?
                """,
                (int(limit),),
            )
            return [
                TranscriptChunk(
                    id=int(r["id"]), ts=r["ts"],
                    duration_ms=int(r["duration_ms"] or 0),
                    text=r["text"],
                    matched_kind=r["matched_kind"],
                    matched_item_key=r["matched_item_key"],
                    similarity=float(r["similarity"]) if r["similarity"] is not None else None,
                )
                for r in cur.fetchall()
            ]

    def recent_transcripts(
        self, *, within_minutes: int = 20, limit: int = 80,
    ) -> list[TranscriptChunk]:
        """Streamer-voice utterances within the lookback window, oldest
        first. Used by the engaging-subjects extractor to ground chat
        subjects against what the streamer has actually been saying out
        loud — so the LLM can tell "chat reacting to the streamer" from
        "chat-driven subject"."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT id, ts, duration_ms, text,
                       matched_kind, matched_item_key, similarity
                FROM transcript_chunks
                WHERE datetime(ts) >= datetime('now', ?)
                ORDER BY id DESC LIMIT ?
                """,
                (f"-{int(within_minutes)} minutes", int(limit)),
            )
            rows = cur.fetchall()
        out = [
            TranscriptChunk(
                id=int(r["id"]), ts=r["ts"],
                duration_ms=int(r["duration_ms"] or 0),
                text=r["text"],
                matched_kind=r["matched_kind"],
                matched_item_key=r["matched_item_key"],
                similarity=float(r["similarity"]) if r["similarity"] is not None else None,
            )
            for r in rows
        ]
        out.reverse()
        return out

    # ============================ Stream recap ============================

    def add_stream_recap(
        self, *, started_at: str, ended_at: str,
        message_id_lo: int | None, message_id_hi: int | None,
        summary: str,
    ) -> int:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO stream_recaps(
                    started_at, ended_at, message_id_lo, message_id_hi, summary
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (started_at, ended_at, message_id_lo, message_id_hi, summary),
            )
            return int(cur.lastrowid)

    def list_stream_recaps(self, limit: int = 10) -> list[StreamRecap]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT id, started_at, ended_at,
                       message_id_lo, message_id_hi, summary
                FROM stream_recaps
                ORDER BY ended_at DESC LIMIT ?
                """,
                (int(limit),),
            )
            return [
                StreamRecap(
                    id=int(r["id"]), started_at=r["started_at"],
                    ended_at=r["ended_at"],
                    message_id_lo=int(r["message_id_lo"]) if r["message_id_lo"] is not None else None,
                    message_id_hi=int(r["message_id_hi"]) if r["message_id_hi"] is not None else None,
                    summary=r["summary"],
                )
                for r in cur.fetchall()
            ]

    # ============================ Thread velocity =========================

    def thread_velocity(self, *, window_minutes: int = 5) -> dict[int, str]:
        """Per-thread message-rate trend over the last 2*window minutes.
        Returns {thread_id: arrow} where arrow ∈ {↑↑, ↑, →, ↓}.

        Compares the count of thread_member messages in the most recent
        window to the prior window. A thread with no recent activity gets
        no entry (display falls back to no arrow).

        We approximate "thread message rate" via topic_thread_members.ts
        — each member is a snapshot occurrence of that thread, which is
        a useful proxy. (True per-message attribution would require
        clustering every chat line, which is too costly.)
        """
        bucket = max(60, int(window_minutes) * 60)
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT thread_id,
                       SUM(CASE WHEN ts >= datetime('now', '-{bucket} seconds')
                                THEN 1 ELSE 0 END) AS recent,
                       SUM(CASE WHEN ts <  datetime('now', '-{bucket} seconds')
                                AND  ts >= datetime('now', '-{2*bucket} seconds')
                                THEN 1 ELSE 0 END) AS prior
                FROM topic_thread_members
                GROUP BY thread_id
                """
            )
            out: dict[int, str] = {}
            for r in cur.fetchall():
                recent = int(r["recent"] or 0)
                prior = int(r["prior"] or 0)
                if recent == 0 and prior == 0:
                    continue
                if prior == 0:
                    out[int(r["thread_id"])] = "↑↑"
                elif recent >= prior * 2:
                    out[int(r["thread_id"])] = "↑↑"
                elif recent > prior:
                    out[int(r["thread_id"])] = "↑"
                elif recent == prior:
                    out[int(r["thread_id"])] = "→"
                else:
                    out[int(r["thread_id"])] = "↓"
            return out

    # ============================ Activity pulse ==========================

    def messages_per_minute(self, minutes: int = 60) -> list[tuple[str, int]]:
        """One row per minute over the last N minutes (oldest first), with
        the count of non-emote messages in that minute. Drives the
        sparkline at the top of /insights — instant 'is the room alive?'
        signal.

        Single index range scan: reads only messages within the window,
        groups by minute in SQL, then densifies in Python so the chart's
        x-axis stays continuous. The earlier RECURSIVE-CTE-with-correlated-
        subqueries approach was running 60 separate scans per call (~10 s
        on this DB), which made /insights feel broken.
        """
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT strftime('%Y-%m-%dT%H:%M:00', ts) AS slot, COUNT(*) AS c
                FROM messages
                WHERE is_emote_only = 0 AND spam_score < 0.5
                  AND ts >= datetime('now', ?)
                GROUP BY slot
                """,
                (f"-{int(minutes)} minutes",),
            )
            counts = {r["slot"]: int(r["c"]) for r in cur.fetchall()}
            # Anchor the densification on the same `now` SQLite saw so we
            # don't drift off-minute due to wall-clock skew.
            now_iso = cur.execute(
                "SELECT strftime('%Y-%m-%dT%H:%M:00', datetime('now'))"
            ).fetchone()[0]
        # Build the contiguous 60-slot list ourselves. Cheaper and clearer
        # than a RECURSIVE CTE.
        from datetime import datetime as _dt, timedelta as _td
        end = _dt.fromisoformat(now_iso)
        out: list[tuple[str, int]] = []
        for i in range(int(minutes), -1, -1):
            slot_dt = end - _td(minutes=i)
            slot = slot_dt.strftime("%Y-%m-%dT%H:%M:00")
            out.append((slot, counts.get(slot, 0)))
        return out

    # ============================ Direct mentions =========================

    # First N chars of the channel handle become a "possible mention"
    # match — covers chatters who shortcut the name, e.g. typing
    # `pcplay` for `pcplaysgames`. Set to 6 by default; degrades to
    # exact-match for handles shorter than this (so `xqc` doesn't
    # become a 3-char prefix that hits half the dictionary).
    _MENTION_PREFIX_LEN = 6
    _MIN_PREFIX_HANDLE_LEN = 4  # below this, prefix-match is too noisy

    def recent_questions(
        self,
        *,
        within_minutes: int = 15,
        limit: int = 8,
        min_chars: int = 8,
    ) -> list[dict]:
        """Cluster recent chat questions and return the most-asked
        ones, newest-first within ties on count. A "question" is any
        clean message containing `?` and at least `min_chars`
        characters.

        Clustering is token-set Jaccard (>=0.6) over short stopword-
        filtered tokens, so 'whats a good route' and 'whats the route'
        merge into one row showing both askers. Returns dicts shaped
        for direct render in the Questions panel:

            {
              question: str,                         # representative text
              count: int,                            # askers
              drivers: list[{name, user_id, ts}],
              latest_ts: str,
              last_msg_id: int,
            }

        Used by /insights → Questions panel. Different from
        recent_direct_mentions which captures @<channel> talking-to-you
        messages — most chat questions don't @-mention the streamer."""
        import re
        word_re = re.compile(r"[a-z][a-z']{2,}")
        # Minimal stopword set focused on chat-question filler. Keep
        # question words (what/when/where/why/how/which) — they help
        # cluster "what's a route" vs "any route" against "where's
        # the route" without false-merging unrelated questions.
        stopwords = frozenset({
            "the", "and", "for", "you", "are", "was", "were", "with",
            "this", "that", "have", "has", "had", "but", "not", "all",
            "any", "can", "will", "would", "could", "should", "your",
            "their", "them", "they", "his", "her", "him", "she", "yes",
            "yeah", "yep", "nah", "got", "get", "going", "gonna", "wanna",
            "say", "said", "see", "saw", "look", "tho", "lol", "lmao",
            "kek", "fr", "ngl", "btw", "imo", "idk", "tbh", "ok",
            "okay", "just", "like", "really",
        })

        def _tokens(text: str) -> set[str]:
            return {
                w for w in word_re.findall(text.lower())
                if w not in stopwords
            }

        # Pull candidates: clean filter + has '?' + reasonable length
        # + NOT a reply or @-mention (those questions are addressed
        # to a specific person and the dashboard's "Talking to you"
        # panel handles them — this panel is for OPEN questions chat
        # is asking the room).
        # SQL excludes:
        #   - rows with reply_parent_login set (Twitch's native reply
        #     feature → it's directed at someone)
        #   - content starting with '@' (chat convention for "I'm
        #     replying to a specific user")
        # Pull a wider candidate set (200) than the cap so a hot
        # chat window has plenty of material to cluster + so the
        # LLM filter pass downstream has signal to work with.
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT m.id, m.user_id, u.name, u.source, m.ts, m.content
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                WHERE {self._CLEAN_MSG_WHERE}
                  AND datetime(m.ts) >= datetime('now', ?)
                  AND m.content LIKE '%?%'
                  AND length(m.content) >= ?
                  AND m.reply_parent_login IS NULL
                  AND m.content NOT LIKE '@%'
                ORDER BY m.id DESC
                LIMIT 200
                """,
                (f"-{int(within_minutes)} minutes", int(min_chars)),
            )
            rows = list(cur.fetchall())

        # Cluster by Szymkiewicz-Simpson overlap coefficient
        # (intersect / smaller set). Jaccard penalises asymmetry —
        # one short question vs one long question — and short-vs-
        # short questions hit a 0.4 ceiling even when they're
        # obviously the same ask. Overlap recognises 'good route'
        # appearing in {good, route, game} ∩ {whats, good, route,
        # start} as 2/3 = 0.67 → cluster. Threshold 0.5 = "more than
        # half of the smaller question's distinctive tokens overlap"
        # — empirically the right cut for chat-style questions.
        # Walking newest → oldest so the representative text stays
        # current.
        clusters: list[dict] = []
        for r in rows:
            content = (r["content"] or "").strip()
            toks = _tokens(content)
            if not toks:
                continue
            best_idx, best_sim = -1, 0.0
            for i, c in enumerate(clusters):
                if not c["_tokens"]:
                    continue
                inter = len(toks & c["_tokens"])
                smaller = min(len(toks), len(c["_tokens"]))
                sim = inter / smaller if smaller else 0.0
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            driver = {
                "name": r["name"], "user_id": r["user_id"], "ts": r["ts"],
            }
            if best_sim >= 0.5 and best_idx >= 0:
                c = clusters[best_idx]
                # Dedupe driver list by user_id — one chatter asking
                # the same thing 3 times shouldn't inflate the count.
                if not any(d["user_id"] == driver["user_id"] for d in c["drivers"]):
                    c["drivers"].append(driver)
                    c["count"] += 1
                # Widen the cluster's vocabulary so subsequent merges
                # have a richer signature.
                c["_tokens"] |= toks
            else:
                clusters.append({
                    "question": content,
                    "_tokens": toks,
                    "count": 1,
                    "drivers": [driver],
                    "latest_ts": r["ts"],
                    "last_msg_id": int(r["id"]),
                })

        # Sort: most-asked first; tiebreak on most-recent.
        clusters.sort(key=lambda c: c["latest_ts"], reverse=True)
        clusters.sort(key=lambda c: c["count"], reverse=True)
        return [
            {k: v for k, v in c.items() if k != "_tokens"}
            for c in clusters[:int(limit)]
        ]

    def recent_direct_mentions(
        self, *, limit: int = 8, lookback_minutes: int = 30,
    ) -> list[Message]:
        """Recent messages where the chatter is *demonstrably* addressing
        the streamer. Three signals — all word-boundary-aware so we
        don't false-positive on substrings:

          1. `@<channel>` or bare `<channel>` at a word boundary:
             `@pcplaysgames` and `pcplaysgames is cool` both match;
             `@pcplaysgames123`, `xpcplaysgames`, `dudeplaysgames` do
             not.
          2. First-N-chars prefix at a word boundary: `pcplay` /
             `pcplays` / `pcplaysgames` all match for channel
             `pcplaysgames`. Common when chatters shortcut the name.
             Will false-positive on real words sharing the prefix
             (e.g. `pcplaying`); the modal's help banner frames this
             as "possible mention". For handles shorter than
             `_MIN_PREFIX_HANDLE_LEN`, prefix mode is disabled and we
             require an exact match.
          3. Twitch IRCv3 reply tag pointing at the streamer: when a
             chatter clicks "Reply" on the streamer's own message,
             `reply_parent_login` lands on the row. Strongest signal.

        Drops the old `?`-anywhere heuristic — it flooded the tab with
        random "wait what?" reactions. Without a configured
        `twitch_channel` setting, returns [].
        """
        import re
        twitch_channel = (self.get_app_setting("twitch_channel") or "").strip().lower()
        if not twitch_channel:
            return []

        # Prefix used for fuzzy matching. Short handles get exact-match
        # mode (no suffix expansion); long handles allow `[a-zA-Z0-9_]*`
        # after the first N chars so chatters who shortcut the name
        # still register.
        fuzzy = len(twitch_channel) >= self._MIN_PREFIX_HANDLE_LEN
        prefix = (
            twitch_channel[:min(self._MENTION_PREFIX_LEN, len(twitch_channel))]
            if fuzzy else twitch_channel
        )

        # SQL prefilter: any row containing the prefix as a substring
        # OR a reply tag matching the streamer. Word-boundary precision
        # happens in Python because SQLite LIKE doesn't do `\b`.
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT m.id, m.user_id, u.name, u.source, m.ts, m.content,
                       m.reply_parent_login, m.reply_parent_body
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                WHERE m.is_emote_only = 0 AND m.spam_score < 0.5
                  AND u.opt_out = 0
                  AND m.ts >= datetime('now', ?)
                  AND (
                       LOWER(m.content) LIKE ?
                    OR LOWER(IFNULL(m.reply_parent_login, '')) = ?
                  )
                ORDER BY m.id DESC
                LIMIT ?
                """,
                (
                    f"-{int(lookback_minutes)} minutes",
                    f"%{prefix}%",
                    twitch_channel,
                    # Pull a generous candidate set — post-filter shrinks it.
                    int(limit) * 4,
                ),
            )
            rows = cur.fetchall()

        # Word-boundary post-filter. Twitch usernames are alnum +
        # underscore; we hand-roll the boundary instead of `\b` because
        # `\b` doesn't break on `_`. In fuzzy mode the trailing
        # `[a-zA-Z0-9_]*` lets the prefix expand into the rest of the
        # handle so `pcplay` / `pcplays` / `pcplaysgames` all match for
        # channel `pcplaysgames`. In exact mode (short handles) we
        # require the match to end right at the handle boundary so
        # `xqcL` / `xqcow` don't false-positive on `xqc`.
        suffix_re = r"[a-zA-Z0-9_]*" if fuzzy else r""
        mention_re = re.compile(
            rf"(?:^|[^a-zA-Z0-9_])"        # left word boundary
            rf"@?{re.escape(prefix)}{suffix_re}"
            rf"(?![a-zA-Z0-9_])",          # right word boundary
            re.IGNORECASE,
        )
        out: list[Message] = []
        for r in rows:
            reply_login = (r["reply_parent_login"] or "").strip().lower()
            content = r["content"] or ""
            if reply_login == twitch_channel or mention_re.search(content):
                out.append(Message(
                    id=int(r["id"]), user_id=r["user_id"], name=r["name"],
                    ts=r["ts"], content=content,
                    reply_parent_login=r["reply_parent_login"],
                    reply_parent_body=r["reply_parent_body"],
                    source=r["source"] or "twitch",
                ))
                if len(out) >= int(limit):
                    break
        return out

    # ============================ Activity / arrival signals ===============

    def user_message_stats(self, user_id: str) -> dict[str, Any]:
        """Aggregate snapshot used by the activity-badge helper:
            msg_count, first_seen, last_seen, prev_seen (penultimate msg ts).
        prev_seen is what lets us detect "back after a long gap"."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM messages WHERE user_id = u.twitch_id) AS msg_count,
                  u.first_seen, u.last_seen,
                  (SELECT ts FROM messages WHERE user_id = u.twitch_id
                    ORDER BY id DESC LIMIT 1 OFFSET 1) AS prev_seen
                FROM users u
                WHERE u.twitch_id = ?
                """,
                (user_id,),
            )
            r = cur.fetchone()
            if not r:
                return {}
            return {
                "msg_count": int(r["msg_count"] or 0),
                "first_seen": r["first_seen"],
                "last_seen": r["last_seen"],
                "prev_seen": r["prev_seen"],
            }

    # ============================ Stream sessions =========================

    def stream_sessions(
        self, *, gap_minutes: int = 30, limit: int = 12
    ) -> list[StreamSession]:
        """Bucket the entire message log into stream sessions — contiguous
        chunks separated by silences of at least `gap_minutes`. Returns the
        most recent `limit` sessions, newest first.

        Computed at query time via SQLite window functions; no state to
        maintain. For each session also computes the top chatter."""
        with self._cursor() as cur:
            cur.execute(
                f"""
                WITH gapped AS (
                  SELECT id, user_id, ts,
                         (julianday(ts) - julianday(LAG(ts) OVER (ORDER BY id)))
                           * 24 * 60 AS gap_min
                  FROM messages
                ),
                flagged AS (
                  SELECT id, user_id, ts,
                         CASE WHEN gap_min IS NULL OR gap_min > {int(gap_minutes)}
                              THEN 1 ELSE 0 END AS is_new
                  FROM gapped
                ),
                sessioned AS (
                  SELECT id, user_id, ts,
                         SUM(is_new) OVER (ORDER BY id) AS sid
                  FROM flagged
                )
                SELECT sid,
                       MIN(ts)            AS start_ts,
                       MAX(ts)            AS end_ts,
                       COUNT(*)           AS msgs,
                       COUNT(DISTINCT user_id) AS uniques
                FROM sessioned
                GROUP BY sid
                ORDER BY sid DESC
                LIMIT ?
                """,
                (limit,),
            )
            session_rows = [dict(r) for r in cur.fetchall()]
            if not session_rows:
                return []
            # Top chatter per session — separate query, joined in Python.
            sids = [r["sid"] for r in session_rows]
            sids_csv = ",".join(str(int(s)) for s in sids)
            cur.execute(
                f"""
                WITH gapped AS (
                  SELECT id, user_id, ts,
                         (julianday(ts) - julianday(LAG(ts) OVER (ORDER BY id)))
                           * 24 * 60 AS gap_min
                  FROM messages
                ),
                flagged AS (
                  SELECT id, user_id, ts,
                         CASE WHEN gap_min IS NULL OR gap_min > {int(gap_minutes)}
                              THEN 1 ELSE 0 END AS is_new
                  FROM gapped
                ),
                sessioned AS (
                  SELECT id, user_id, ts,
                         SUM(is_new) OVER (ORDER BY id) AS sid
                  FROM flagged
                )
                SELECT s.sid, u.name, COUNT(*) AS c
                FROM sessioned s JOIN users u ON u.twitch_id = s.user_id
                WHERE s.sid IN ({sids_csv})
                GROUP BY s.sid, s.user_id
                """
            )
            rows = cur.fetchall()
        # Reduce to top chatter per session.
        top_by_sid: dict[int, tuple[str, int]] = {}
        for r in rows:
            sid = int(r["sid"])
            cur_top = top_by_sid.get(sid)
            if cur_top is None or int(r["c"]) > cur_top[1]:
                top_by_sid[sid] = (r["name"], int(r["c"]))
        out: list[StreamSession] = []
        for r in session_rows:
            sid = int(r["sid"])
            top = top_by_sid.get(sid)
            out.append(
                StreamSession(
                    id=sid, start_ts=r["start_ts"], end_ts=r["end_ts"],
                    message_count=int(r["msgs"]),
                    unique_chatters=int(r["uniques"]),
                    top_chatter=top[0] if top else None,
                )
            )
        return out

    # ============================ Anniversaries ===========================

    def users_with_anniversary_today(self) -> list[tuple[User, str]]:
        """Users whose first_seen anniversary lands on today's date (UTC),
        at one of: 1y, 2y, 3y, 6mo, 3mo. Returns (user, milestone_label)."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT twitch_id, name, first_seen, last_seen, opt_out, sub_tier, sub_months, is_mod, is_vip, is_founder,
                       date(first_seen) AS first_date,
                       julianday('now') - julianday(first_seen) AS days_since
                FROM users
                WHERE opt_out = 0
                  AND first_date IS NOT NULL
                  AND strftime('%m-%d', first_seen) = strftime('%m-%d', 'now')
                """
            )
            rows = cur.fetchall()
        out: list[tuple[User, str]] = []
        for r in rows:
            try:
                days = float(r["days_since"] or 0)
            except (TypeError, ValueError):
                continue
            label: str | None = None
            if 360 <= days < 370:
                label = "1 year"
            elif 720 <= days < 730:
                label = "2 years"
            elif 1080 <= days < 1090:
                label = "3 years"
            else:
                continue
            out.append(
                (
                    User(
                        twitch_id=r["twitch_id"], name=r["name"],
                        first_seen=r["first_seen"], last_seen=r["last_seen"],
                        opt_out=False,
                                            sub_tier=r["sub_tier"],
                        sub_months=int(r["sub_months"] or 0),
                        is_mod=bool(r["is_mod"]),
                        is_vip=bool(r["is_vip"]),
                        is_founder=bool(r["is_founder"]),
                    ),
                    label,
                )
            )
        # Also: half-year and quarter-year — check today vs first_seen + 6 mo / 3 mo.
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT twitch_id, name, first_seen, last_seen, opt_out, sub_tier, sub_months, is_mod, is_vip, is_founder
                FROM users
                WHERE opt_out = 0
                  AND date(first_seen, '+6 months') = date('now')
                """
            )
            for r in cur.fetchall():
                out.append(
                    (
                        User(
                            twitch_id=r["twitch_id"], name=r["name"],
                            first_seen=r["first_seen"], last_seen=r["last_seen"],
                            opt_out=False,
                                                    sub_tier=r["sub_tier"],
                            sub_months=int(r["sub_months"] or 0),
                            is_mod=bool(r["is_mod"]),
                            is_vip=bool(r["is_vip"]),
                            is_founder=bool(r["is_founder"]),
                        ),
                        "6 months",
                    )
                )
            cur.execute(
                """
                SELECT twitch_id, name, first_seen, last_seen, opt_out, sub_tier, sub_months, is_mod, is_vip, is_founder
                FROM users
                WHERE opt_out = 0
                  AND date(first_seen, '+3 months') = date('now')
                """
            )
            for r in cur.fetchall():
                out.append(
                    (
                        User(
                            twitch_id=r["twitch_id"], name=r["name"],
                            first_seen=r["first_seen"], last_seen=r["last_seen"],
                            opt_out=False,
                                                    sub_tier=r["sub_tier"],
                            sub_months=int(r["sub_months"] or 0),
                            is_mod=bool(r["is_mod"]),
                            is_vip=bool(r["is_vip"]),
                            is_founder=bool(r["is_founder"]),
                        ),
                        "3 months",
                    )
                )
        return out

    # ============================ Stats queries (Stats tab) ================
    # All read-only aggregates; no LLM. Powers the dashboard's Stats page.

    def stats_totals(self) -> dict[str, int]:
        """Top-of-page big-number cards: total chatters / messages / notes /
        events / topic snapshots / incidents (incidents are 0 if mod's off)."""
        out: dict[str, int] = {}
        with self._cursor() as cur:
            for label, q in (
                ("chatters",     "SELECT COUNT(*) FROM users"),
                ("messages",     "SELECT COUNT(*) FROM messages"),
                ("notes",        "SELECT COUNT(*) FROM notes"),
                ("events",       "SELECT COUNT(*) FROM events"),
                ("topic_snaps",  "SELECT COUNT(*) FROM topic_snapshots"),
                ("incidents",    "SELECT COUNT(*) FROM incidents"),
            ):
                try:
                    out[label] = int(cur.execute(q).fetchone()[0])
                except sqlite3.Error:
                    out[label] = 0
        return out

    def stats_event_totals(self) -> dict[str, float | int]:
        """Lifetime tips ($), bits, sub months — for the donations donut + tile."""
        with self._cursor() as cur:
            row = cur.execute(
                """
                SELECT
                  COALESCE(SUM(CASE WHEN type='tip'   THEN amount END), 0) AS tip_total,
                  COALESCE(SUM(CASE WHEN type='cheer' THEN amount END), 0) AS bits_total,
                  COALESCE(SUM(CASE WHEN type='sub'   THEN amount END), 0) AS sub_months,
                  COUNT(DISTINCT CASE WHEN type='tip'   THEN twitch_name END) AS unique_tippers,
                  COUNT(DISTINCT CASE WHEN type='cheer' THEN twitch_name END) AS unique_cheerers,
                  COUNT(DISTINCT CASE WHEN type='sub'   THEN twitch_name END) AS unique_subbers
                FROM events
                """
            ).fetchone()
            return {
                "tip_total":       float(row["tip_total"] or 0.0),
                "bits_total":      int(row["bits_total"] or 0),
                "sub_months":      int(row["sub_months"] or 0),
                "unique_tippers":  int(row["unique_tippers"] or 0),
                "unique_cheerers": int(row["unique_cheerers"] or 0),
                "unique_subbers":  int(row["unique_subbers"] or 0),
            }

    def stats_messages_per_day(self, days: int = 30) -> list[tuple[str, int]]:
        """Daily message counts for the last `days`. Returns (YYYY-MM-DD, n)
        oldest-first. Days with zero messages are emitted explicitly."""
        with self._cursor() as cur:
            rows = cur.execute(
                """
                SELECT date(ts) AS d, COUNT(*) AS c
                FROM messages
                WHERE ts >= datetime('now', ?)
                GROUP BY date(ts)
                """,
                (f"-{int(days)} days",),
            ).fetchall()
        # Densify so the chart x-axis is continuous.
        from datetime import date, timedelta
        counts = {r["d"]: int(r["c"]) for r in rows}
        today = date.today()
        out: list[tuple[str, int]] = []
        for i in range(days - 1, -1, -1):
            d = today - timedelta(days=i)
            key = d.isoformat()
            out.append((key, counts.get(key, 0)))
        return out

    def stats_messages_per_hour(self) -> list[tuple[int, int]]:
        """Aggregate message counts by hour-of-day (0..23). Helps the streamer
        see when their chat is most alive."""
        with self._cursor() as cur:
            rows = cur.execute(
                """
                SELECT CAST(strftime('%H', ts) AS INTEGER) AS h, COUNT(*) AS c
                FROM messages
                GROUP BY h
                """
            ).fetchall()
        counts = {int(r["h"]): int(r["c"]) for r in rows}
        return [(h, counts.get(h, 0)) for h in range(24)]

    def stats_top_chatters_lifetime(self, limit: int = 10) -> list[tuple[str, int]]:
        """Top N chatters by total messages (lifetime). (name, msg_count)."""
        with self._cursor() as cur:
            rows = cur.execute(
                """
                SELECT u.name, COUNT(m.id) AS c
                FROM users u
                JOIN messages m ON m.user_id = u.twitch_id
                GROUP BY u.twitch_id
                ORDER BY c DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [(r["name"], int(r["c"])) for r in rows]

    def stats_top_supporters(self, limit: int = 5) -> list[tuple[str, float]]:
        """Top tippers by lifetime $ amount. (name, tip_total)."""
        with self._cursor() as cur:
            rows = cur.execute(
                """
                SELECT twitch_name, COALESCE(SUM(amount), 0) AS total
                FROM events
                WHERE type = 'tip'
                GROUP BY twitch_name
                ORDER BY total DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [(r["twitch_name"], float(r["total"] or 0.0)) for r in rows]

    def stats_new_chatters_per_week(self, weeks: int = 12) -> list[tuple[str, int]]:
        """First-seen counts grouped by ISO week, oldest-first, densified."""
        with self._cursor() as cur:
            rows = cur.execute(
                """
                SELECT strftime('%Y-W%W', first_seen) AS w, COUNT(*) AS c
                FROM users
                WHERE first_seen >= datetime('now', ?)
                GROUP BY w
                """,
                (f"-{int(weeks * 7)} days",),
            ).fetchall()
        counts = {r["w"]: int(r["c"]) for r in rows}
        # Build the last `weeks` weekly buckets (oldest first).
        from datetime import date, timedelta
        today = date.today()
        out: list[tuple[str, int]] = []
        for i in range(weeks - 1, -1, -1):
            d = today - timedelta(weeks=i)
            key = d.strftime("%Y-W%W")
            out.append((key, counts.get(key, 0)))
        return out

    def stats_longest_message(self) -> tuple[str, str, int] | None:
        """Pull the single longest message ever logged. (name, content, len).
        Filters out emote-only and spam (score >= 0.5) messages so the
        "longest" stat doesn't get hijacked by emote walls or
        copy-paste floods."""
        with self._cursor() as cur:
            row = cur.execute(
                """
                SELECT u.name, m.content, length(m.content) AS L
                FROM messages m JOIN users u ON u.twitch_id = m.user_id
                WHERE m.is_emote_only = 0 AND m.spam_score < 0.5
                ORDER BY L DESC LIMIT 1
                """
            ).fetchone()
            if not row:
                return None
            return (row["name"], row["content"], int(row["L"]))

    def stats_avg_message_length(self) -> float:
        """Average message length, excluding emote-only and spammy
        messages so a single 1k-char repetition flood doesn't shift
        the channel-wide average."""
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT AVG(length(content)) AS a FROM messages "
                "WHERE is_emote_only = 0 AND spam_score < 0.5"
            ).fetchone()
            return float(row["a"] or 0.0)

    def stats_most_chatty_hour(self) -> int | None:
        """Hour of day (0..23) with the highest aggregate message count."""
        rows = self.stats_messages_per_hour()
        if not any(c for _, c in rows):
            return None
        return max(rows, key=lambda r: r[1])[0]

    def stats_busiest_day(self) -> tuple[str, int] | None:
        """Single day with the most messages ever. (date, count)."""
        with self._cursor() as cur:
            row = cur.execute(
                """
                SELECT date(ts) AS d, COUNT(*) AS c
                FROM messages GROUP BY d
                ORDER BY c DESC LIMIT 1
                """
            ).fetchone()
            if not row or not row["d"]:
                return None
            return (row["d"], int(row["c"]))

    def stats_oldest_chatter(self) -> tuple[str, str] | None:
        """Earliest first_seen — your "founder." (name, first_seen)."""
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT name, first_seen FROM users "
                "ORDER BY first_seen ASC LIMIT 1"
            ).fetchone()
            return (row["name"], row["first_seen"]) if row else None

    def stats_transcripts(self, *, match_threshold: float = 0.55) -> dict[str, float | int | str | None]:
        """Aggregate stats over `transcript_chunks` for the stats page. Empty
        DBs (whisper never ran) return zeros, not None."""
        out: dict[str, float | int | str | None] = {
            "chunks": 0,
            "total_seconds": 0.0,
            "matched": 0,
            "auto_addressed": 0,
            "near_miss": 0,
            "mean_similarity": None,
            "max_similarity": None,
            "first_ts": None,
            "last_ts": None,
        }
        try:
            with self._cursor() as cur:
                row = cur.execute(
                    """
                    SELECT
                      COUNT(*)                                              AS chunks,
                      COALESCE(SUM(duration_ms), 0) / 1000.0                AS total_seconds,
                      SUM(CASE WHEN similarity IS NOT NULL THEN 1 ELSE 0 END) AS matched,
                      SUM(CASE WHEN similarity >= ? THEN 1 ELSE 0 END)      AS auto_addressed,
                      SUM(CASE WHEN similarity IS NOT NULL AND similarity >= 0.30
                               AND similarity < ? THEN 1 ELSE 0 END)        AS near_miss,
                      AVG(similarity)                                       AS mean_sim,
                      MAX(similarity)                                       AS max_sim,
                      MIN(ts)                                               AS first_ts,
                      MAX(ts)                                               AS last_ts
                    FROM transcript_chunks
                    """,
                    (float(match_threshold), float(match_threshold)),
                ).fetchone()
        except sqlite3.Error:
            return out
        if not row or not row["chunks"]:
            return out
        out["chunks"] = int(row["chunks"])
        out["total_seconds"] = float(row["total_seconds"] or 0.0)
        out["matched"] = int(row["matched"] or 0)
        out["auto_addressed"] = int(row["auto_addressed"] or 0)
        out["near_miss"] = int(row["near_miss"] or 0)
        out["mean_similarity"] = float(row["mean_sim"]) if row["mean_sim"] is not None else None
        out["max_similarity"] = float(row["max_sim"]) if row["max_sim"] is not None else None
        out["first_ts"] = row["first_ts"]
        out["last_ts"] = row["last_ts"]
        return out

    def stats_transcripts_per_day(self, days: int = 30) -> list[tuple[str, int]]:
        """Daily utterance counts for the last `days`. Days with zero are
        emitted explicitly so the chart x-axis stays continuous."""
        try:
            with self._cursor() as cur:
                rows = cur.execute(
                    """
                    SELECT date(ts) AS d, COUNT(*) AS c
                    FROM transcript_chunks
                    WHERE ts >= datetime('now', ?)
                    GROUP BY date(ts)
                    """,
                    (f"-{int(days)} days",),
                ).fetchall()
        except sqlite3.Error:
            rows = []
        from datetime import date, timedelta
        counts = {r["d"]: int(r["c"]) for r in rows}
        today = date.today()
        out: list[tuple[str, int]] = []
        for i in range(days - 1, -1, -1):
            d = today - timedelta(days=i)
            key = d.isoformat()
            out.append((key, counts.get(key, 0)))
        return out

    def stats_longest_utterance(self) -> tuple[str, int] | None:
        """Single utterance with the most characters. (text, length).
        Returns None if there are no transcripts yet."""
        try:
            with self._cursor() as cur:
                row = cur.execute(
                    "SELECT text, length(text) AS n FROM transcript_chunks "
                    "ORDER BY n DESC LIMIT 1"
                ).fetchone()
        except sqlite3.Error:
            return None
        if not row or not row["n"]:
            return None
        return (row["text"], int(row["n"]))

    # ============================ Insights queries =========================
    # Streamer-only views: who's chatting now, who are the regulars, who's
    # lapsed, who's new today. All read-only / aggregate; no LLM calls here.

    # ============================ Streamer-personal favorites =============

    def set_user_starred(self, twitch_id: str, starred: bool) -> None:
        """Toggle the streamer-personal star flag. Stored on the
        user_profiles extension table (UPSERT) so a user with no
        prior profile data still gets a star without needing a
        separate "create profile" step."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_profiles (user_id, is_starred)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET is_starred = excluded.is_starred
                """,
                (twitch_id, 1 if starred else 0),
            )

    def count_starred(self) -> int:
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM user_profiles WHERE is_starred = 1"
            )
            return int(cur.fetchone()["c"])

    def list_starred_active(
        self, *, within_minutes: int = 30, limit: int = 12,
    ) -> list[User]:
        """Starred chatters whose most recent message is within the
        window — they're here right now, the streamer should notice."""
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT u.twitch_id, u.name, u.first_seen, u.last_seen, u.opt_out,
                       u.sub_tier, u.sub_months, u.is_mod, u.is_vip, u.is_founder,
                       u.source, u.merged_into, u.followed_at,
                       p.pronouns, p.location, p.demeanor, p.interests,
                       p.profile_updated_at,
                       COALESCE(p.is_starred, 0) AS is_starred
                FROM users u
                JOIN user_profiles p ON p.user_id = u.twitch_id
                WHERE p.is_starred = 1
                  AND u.opt_out = 0
                  AND EXISTS (
                    SELECT 1 FROM messages m
                    WHERE m.user_id = u.twitch_id
                      AND m.ts >= datetime('now', '-{int(within_minutes)} minutes')
                  )
                ORDER BY u.last_seen DESC LIMIT ?
                """,
                (int(limit),),
            )
            return [self._row_to_user(r) for r in cur.fetchall()]

    def _row_to_user(self, r) -> User:  # noqa: ANN001
        interests_raw = r["interests"] if "interests" in r.keys() else None
        interests: list[str] | None = None
        if interests_raw:
            try:
                parsed = json.loads(interests_raw)
                if isinstance(parsed, list):
                    interests = [str(x) for x in parsed if x]
            except (TypeError, ValueError):
                pass
        return User(
            twitch_id=r["twitch_id"], name=r["name"],
            first_seen=r["first_seen"], last_seen=r["last_seen"],
            opt_out=bool(r["opt_out"]),
            sub_tier=r["sub_tier"] if "sub_tier" in r.keys() else None,
            sub_months=int(r["sub_months"] or 0) if "sub_months" in r.keys() else 0,
            is_mod=bool(r["is_mod"]) if "is_mod" in r.keys() else False,
            is_vip=bool(r["is_vip"]) if "is_vip" in r.keys() else False,
            is_founder=bool(r["is_founder"]) if "is_founder" in r.keys() else False,
            source=(r["source"] if "source" in r.keys() else None) or "twitch",
            merged_into=r["merged_into"] if "merged_into" in r.keys() else None,
            pronouns=r["pronouns"] if "pronouns" in r.keys() else None,
            location=r["location"] if "location" in r.keys() else None,
            demeanor=r["demeanor"] if "demeanor" in r.keys() else None,
            interests=interests,
            profile_updated_at=(
                r["profile_updated_at"] if "profile_updated_at" in r.keys() else None
            ),
            is_starred=bool(r["is_starred"]) if "is_starred" in r.keys() else False, followed_at=(r["followed_at"] if "followed_at" in r.keys() else None),
        )

    # ============================ Neglected lurkers =======================

    def list_neglected_lurkers(
        self, *, active_within_minutes: int = 30, neglected_for_days: int = 7,
        limit: int = 8,
    ) -> list[User]:
        """Chatters active in the last `active_within_minutes` whose last
        addressed-state was more than `neglected_for_days` ago — or who
        have never been acknowledged at all. Different from lapsed: these
        are people in your chat *right now* that you keep missing.

        Excludes the broadcaster and bots (heuristic: is_mod and is_founder
        commonly flag the broadcaster's own account)."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=int(neglected_for_days))
        ).isoformat(timespec="seconds")
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT u.twitch_id, u.name, u.first_seen, u.last_seen, u.opt_out,
                       u.sub_tier, u.sub_months, u.is_mod, u.is_vip, u.is_founder,
                       u.source, u.merged_into, u.followed_at,
                       p.pronouns, p.location, p.demeanor, p.interests,
                       p.profile_updated_at,
                       COALESCE(p.is_starred, 0) AS is_starred
                FROM users u
                LEFT JOIN user_profiles p ON p.user_id = u.twitch_id
                WHERE u.opt_out = 0
                  AND u.merged_into IS NULL
                  AND EXISTS (
                    SELECT 1 FROM messages m
                    WHERE m.user_id = u.twitch_id
                      AND m.ts >= datetime('now', '-{int(active_within_minutes)} minutes')
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM insight_state_history h
                    WHERE h.item_key = u.twitch_id
                      AND h.state = 'addressed'
                      AND h.ts >= ?
                  )
                ORDER BY u.last_seen DESC
                LIMIT ?
                """,
                (cutoff, int(limit)),
            )
            return [self._row_to_user(r) for r in cur.fetchall()]

    # ============================ Conversation continuity =================

    def last_callback_for_users(
        self, user_ids: Iterable[str], *, max_age_days: int = 90,
    ) -> dict[str, str]:
        """Most recent note text per user — used by the live widget to
        surface a one-line callback when a returning chatter speaks.
        Skips notes older than `max_age_days` (stale callbacks misfire)."""
        ids = [u for u in user_ids if u]
        if not ids:
            return {}
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=int(max_age_days))
        ).isoformat(timespec="seconds")
        placeholders = ",".join("?" for _ in ids)
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT user_id, text FROM notes WHERE id IN (
                    SELECT MAX(id) FROM notes
                    WHERE user_id IN ({placeholders}) AND ts >= ?
                    GROUP BY user_id
                )
                """,
                tuple(ids) + (cutoff,),
            )
            return {r["user_id"]: r["text"] for r in cur.fetchall()}

    def list_active_chatters(self, window_minutes: int = 10, limit: int = 30) -> list[User]:
        """Users with at least one message in the last `window_minutes`."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT u.twitch_id, u.name, u.first_seen, u.last_seen, u.opt_out, u.sub_tier, u.sub_months, u.is_mod, u.is_vip, u.is_founder
                FROM users u
                JOIN messages m ON m.user_id = u.twitch_id
                WHERE u.opt_out = 0
                  AND m.ts >= datetime('now', ?)
                ORDER BY u.last_seen DESC
                LIMIT ?
                """,
                (f"-{int(window_minutes)} minutes", limit),
            )
            return [
                User(
                    twitch_id=r["twitch_id"], name=r["name"],
                    first_seen=r["first_seen"], last_seen=r["last_seen"],
                    opt_out=False,
                    sub_tier=r["sub_tier"], sub_months=int(r["sub_months"] or 0),
                    is_mod=bool(r["is_mod"]), is_vip=bool(r["is_vip"]),
                    is_founder=bool(r["is_founder"]),
                )
                for r in cur.fetchall()
            ]

    def list_regulars(
        self, *, since: str | None = "-7 days", limit: int = 10
    ) -> list[ChatterRow]:
        """Top chatters by message count.

        `since` is either a SQLite `datetime('now', ...)` modifier
        (e.g. '-7 days', '-30 days', 'start of year') or None for lifetime
        (no date filter)."""
        if since is None:
            where = "WHERE u.opt_out = 0"
            params: list[Any] = [limit]
        else:
            where = "WHERE u.opt_out = 0 AND m.ts >= datetime('now', ?)"
            params = [since, limit]
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT u.twitch_id, u.name, u.first_seen, u.last_seen, u.opt_out, u.sub_tier, u.sub_months, u.is_mod, u.is_vip, u.is_founder,
                       COUNT(m.id) AS msg_count,
                       (SELECT COUNT(*) FROM notes n WHERE n.user_id = u.twitch_id) AS note_count,
                       MAX(m.ts) AS last_msg_ts
                FROM users u
                JOIN messages m ON m.user_id = u.twitch_id
                {where}
                GROUP BY u.twitch_id
                ORDER BY msg_count DESC
                LIMIT ?
                """,
                params,
            )
            return [
                ChatterRow(
                    user=User(
                        twitch_id=r["twitch_id"], name=r["name"],
                        first_seen=r["first_seen"], last_seen=r["last_seen"],
                        opt_out=False,
                                            sub_tier=r["sub_tier"],
                        sub_months=int(r["sub_months"] or 0),
                        is_mod=bool(r["is_mod"]),
                        is_vip=bool(r["is_vip"]),
                        is_founder=bool(r["is_founder"]),
                    ),
                    note_count=int(r["note_count"]),
                    msg_count=int(r["msg_count"]),
                    last_message_ts=r["last_msg_ts"],
                )
                for r in cur.fetchall()
            ]

    def list_lapsed_regulars(
        self,
        *,
        active_since: str | None = "-30 days",
        lapsed_for: str = "-7 days",
        limit: int = 10,
    ) -> list[ChatterRow]:
        """Chatters who were active within `active_since` but quiet for at
        least `lapsed_for`. The "lapsed" definition stays a relative window;
        the "active" lookback is the streamer's lever."""
        if active_since is None:
            where = "WHERE u.opt_out = 0"
            having = "HAVING MAX(m.ts) <= datetime('now', ?)"
            params: list[Any] = [lapsed_for, limit]
        else:
            where = "WHERE u.opt_out = 0 AND m.ts >= datetime('now', ?)"
            having = "HAVING MAX(m.ts) <= datetime('now', ?)"
            params = [active_since, lapsed_for, limit]
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT u.twitch_id, u.name, u.first_seen, u.last_seen, u.opt_out, u.sub_tier, u.sub_months, u.is_mod, u.is_vip, u.is_founder,
                       COUNT(m.id) AS msg_count,
                       (SELECT COUNT(*) FROM notes n WHERE n.user_id = u.twitch_id) AS note_count,
                       MAX(m.ts) AS last_msg_ts
                FROM users u
                JOIN messages m ON m.user_id = u.twitch_id
                {where}
                GROUP BY u.twitch_id
                {having}
                ORDER BY msg_count DESC
                LIMIT ?
                """,
                params,
            )
            return [
                ChatterRow(
                    user=User(
                        twitch_id=r["twitch_id"], name=r["name"],
                        first_seen=r["first_seen"], last_seen=r["last_seen"],
                        opt_out=False,
                                            sub_tier=r["sub_tier"],
                        sub_months=int(r["sub_months"] or 0),
                        is_mod=bool(r["is_mod"]),
                        is_vip=bool(r["is_vip"]),
                        is_founder=bool(r["is_founder"]),
                    ),
                    note_count=int(r["note_count"]),
                    msg_count=int(r["msg_count"]),
                    last_message_ts=r["last_msg_ts"],
                )
                for r in cur.fetchall()
            ]

    def list_first_timers_today(self, limit: int = 20) -> list[User]:
        """Users whose first_seen is in the last 24h."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT twitch_id, name, first_seen, last_seen, opt_out, sub_tier, sub_months, is_mod, is_vip, is_founder
                FROM users
                WHERE opt_out = 0
                  AND first_seen >= datetime('now', '-24 hours')
                ORDER BY first_seen DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [
                User(
                    twitch_id=r["twitch_id"], name=r["name"],
                    first_seen=r["first_seen"], last_seen=r["last_seen"],
                    opt_out=False,
                    sub_tier=r["sub_tier"], sub_months=int(r["sub_months"] or 0),
                    is_mod=bool(r["is_mod"]), is_vip=bool(r["is_vip"]),
                    is_founder=bool(r["is_founder"]),
                )
                for r in cur.fetchall()
            ]

    def count_first_timers_today(self) -> int:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS c FROM users
                WHERE opt_out = 0
                  AND first_seen >= datetime('now', '-24 hours')
                """
            )
            return int(cur.fetchone()["c"])

    # -- Newcomers nav-pill acknowledgment ---------------------------------

    _NEWCOMERS_ACK_KEY = "newcomers_acknowledged_at"

    def get_newcomers_ack(self) -> str | None:
        return self.get_app_setting(self._NEWCOMERS_ACK_KEY)

    def set_newcomers_ack(self, ts: str | None = None) -> None:
        """Record the moment the streamer last reviewed newcomers (visiting
        the Insights tab counts). Called by /insights to clear the pill."""
        self.set_app_setting(self._NEWCOMERS_ACK_KEY, ts or _now_iso())

    # ============================ Stream goals ============================

    _GOALS_KEY = "stream_goals"

    def get_stream_goals(self) -> dict:
        """Current stream's goals. JSON-encoded in app_settings under
        key `stream_goals`. Reset when OBS confirms stream offline.

        Shape: {'targets': [{'kind': 'address_first_timers', 'count': 3}, ...],
                'set_at': iso_ts}.
        """
        raw = self.get_app_setting(self._GOALS_KEY)
        if not raw:
            return {"targets": [], "set_at": None}
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return {"targets": [], "set_at": None}
            data.setdefault("targets", [])
            data.setdefault("set_at", None)
            return data
        except (TypeError, ValueError):
            return {"targets": [], "set_at": None}

    def set_stream_goals(self, targets: list[dict]) -> None:
        payload = {"targets": targets, "set_at": _now_iso()}
        self.set_app_setting(self._GOALS_KEY, json.dumps(payload))

    def clear_stream_goals(self) -> None:
        self.delete_app_setting(self._GOALS_KEY)

    def count_first_timers_unacked(self) -> int:
        """Newcomers in the last 24h whose first_seen is NEWER than the
        streamer's last acknowledgment. This is what the nav pill displays."""
        ack = self.get_newcomers_ack()
        with self._cursor() as cur:
            if ack:
                cur.execute(
                    """
                    SELECT COUNT(*) AS c FROM users
                    WHERE opt_out = 0
                      AND first_seen >= datetime('now', '-24 hours')
                      AND first_seen > ?
                    """,
                    (ack,),
                )
            else:
                cur.execute(
                    """
                    SELECT COUNT(*) AS c FROM users
                    WHERE opt_out = 0
                      AND first_seen >= datetime('now', '-24 hours')
                    """
                )
            return int(cur.fetchone()["c"])

    def list_opt_out_users(self) -> list[User]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT twitch_id, name, first_seen, last_seen, opt_out, sub_tier, sub_months, is_mod, is_vip, is_founder "
                "FROM users WHERE opt_out = 1 ORDER BY LOWER(name)"
            )
            return [
                User(
                    twitch_id=r["twitch_id"], name=r["name"],
                    first_seen=r["first_seen"], last_seen=r["last_seen"],
                    opt_out=True,
                                    sub_tier=r["sub_tier"],
                    sub_months=int(r["sub_months"] or 0),
                    is_mod=bool(r["is_mod"]),
                    is_vip=bool(r["is_vip"]),
                    is_founder=bool(r["is_founder"]),
                )
                for r in cur.fetchall()
            ]

    def get_event(self, event_id: int) -> Event | None:
        with self._cursor() as cur:
            cur.execute(
                "SELECT id, user_id, twitch_name, type, amount, currency, message, ts, raw_json "
                "FROM events WHERE id = ?",
                (event_id,),
            )
            r = cur.fetchone()
            if not r:
                return None
            ev = _event_from_row(r)
            ev_raw = r["raw_json"]
            setattr(ev, "raw_json", ev_raw)
            return ev

    def note_count(self, user_id: str) -> int:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM notes WHERE user_id = ?", (user_id,))
            return int(cur.fetchone()["c"])

    def count_user_events(self, user_id: str) -> int:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM events WHERE user_id = ?", (user_id,))
            return int(cur.fetchone()["c"])

    def list_chatters(
        self,
        *,
        query: str = "",
        limit: int = 100,
        offset: int = 0,
        sort: str = "last_seen",
    ) -> list[ChatterRow]:
        """Joined chatter list with note + message counts. Searches both the
        current `users.name` and the full `user_aliases` history, so an old
        handle still finds the renamed account."""
        # 'engagement' sort is computed in Python after the SELECT —
        # SQL can't easily express the composite formula. We pull a wider
        # window and re-sort, so it's accurate within the result page.
        sort_in_python = sort == "engagement"
        order = {
            "last_seen": "u.last_seen DESC",
            "name": "LOWER(u.name) ASC",
            "messages": "msg_count DESC",
            "notes": "note_count DESC",
            "engagement": "u.last_seen DESC",  # placeholder, re-sorted below
        }.get(sort, "u.last_seen DESC")
        like = f"%{query.lower()}%" if query else None
        params: list[Any] = []
        where = ""
        if like:
            where = (
                "WHERE LOWER(u.name) LIKE ? "
                "   OR EXISTS (SELECT 1 FROM user_aliases a "
                "              WHERE a.user_id = u.twitch_id AND LOWER(a.name) LIKE ?)"
            )
            params.extend([like, like])
        # When sorting by engagement we need a wider candidate set since
        # SQL ordering won't match — pull at least 500 then trim.
        sql_limit = max(500, limit + offset) if sort_in_python else limit
        sql_offset = 0 if sort_in_python else offset
        params.extend([sql_limit, sql_offset])
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT u.twitch_id, u.name, u.first_seen, u.last_seen, u.opt_out,
                       u.sub_tier, u.sub_months, u.is_mod, u.is_vip, u.is_founder,
                       u.followed_at,
                       COALESCE(p.is_starred, 0) AS is_starred,
                       (SELECT COUNT(*) FROM notes n WHERE n.user_id = u.twitch_id) AS note_count,
                       (SELECT COUNT(*) FROM messages m WHERE m.user_id = u.twitch_id) AS msg_count,
                       (SELECT MAX(ts) FROM messages m WHERE m.user_id = u.twitch_id) AS last_msg_ts
                FROM users u
                LEFT JOIN user_profiles p ON p.user_id = u.twitch_id
                {where}
                ORDER BY {order}
                LIMIT ? OFFSET ?
                """,
                params,
            )
            rows = [
                ChatterRow(
                    user=User(
                        twitch_id=r["twitch_id"],
                        name=r["name"],
                        first_seen=r["first_seen"],
                        last_seen=r["last_seen"],
                        opt_out=bool(r["opt_out"]),
                        sub_tier=r["sub_tier"],
                        sub_months=int(r["sub_months"] or 0),
                        is_mod=bool(r["is_mod"]),
                        is_vip=bool(r["is_vip"]),
                        is_founder=bool(r["is_founder"]),
                        is_starred=bool(r["is_starred"]),
                        followed_at=r["followed_at"] if "followed_at" in r.keys() else None,
                    ),
                    note_count=int(r["note_count"]),
                    msg_count=int(r["msg_count"]),
                    last_message_ts=r["last_msg_ts"],
                    engagement_score=_engagement_score(
                        msg_count=int(r["msg_count"]),
                        note_count=int(r["note_count"]),
                        sub_tier=r["sub_tier"],
                        sub_months=int(r["sub_months"] or 0),
                        is_mod=bool(r["is_mod"]),
                        is_vip=bool(r["is_vip"]),
                        is_starred=bool(r["is_starred"]),
                        last_seen=r["last_seen"],
                    ),
                )
                for r in cur.fetchall()
            ]
            if sort_in_python:
                rows.sort(key=lambda r: r.engagement_score, reverse=True)
                rows = rows[offset:offset + limit]
            return rows

    def neighbours_in_chatters(
        self,
        twitch_id: str,
        *,
        query: str = "",
        sort: str = "last_seen",
        scan_cap: int = 5000,
    ) -> tuple[str | None, str | None]:
        """Find the previous and next chatter relative to `twitch_id` in the
        same filtered + sorted list the dashboard shows. Used by the
        user-detail page to render ← prev / next → buttons that stay in
        the streamer's context.

        Walks `list_chatters` once up to `scan_cap` and returns the
        neighbours' twitch_ids by index. Cap exists so a 50k-user channel
        doesn't load everything for one navigation; if the user is beyond
        the cap, prev/next collapse to None and the streamer just lands
        on the chatters page.
        """
        rows = self.list_chatters(
            query=query, sort=sort, limit=scan_cap, offset=0,
        )
        for i, row in enumerate(rows):
            if row.user.twitch_id == twitch_id:
                prev_id = rows[i - 1].user.twitch_id if i > 0 else None
                next_id = rows[i + 1].user.twitch_id if i + 1 < len(rows) else None
                return prev_id, next_id
        return None, None

    def count_chatters(self, query: str = "") -> int:
        if query:
            like = f"%{query.lower()}%"
            with self._cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) AS c FROM users u
                    WHERE LOWER(u.name) LIKE ?
                       OR EXISTS (SELECT 1 FROM user_aliases a
                                  WHERE a.user_id = u.twitch_id AND LOWER(a.name) LIKE ?)
                    """,
                    (like, like),
                )
                return int(cur.fetchone()["c"])
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM users")
            return int(cur.fetchone()["c"])

    def get_user(self, twitch_id: str) -> User | None:
        with self._cursor() as cur:
            cur.execute(
                "SELECT u.twitch_id, u.name, u.first_seen, u.last_seen, u.opt_out, "
                "u.sub_tier, u.sub_months, u.is_mod, u.is_vip, u.is_founder, "
                "u.source, u.merged_into, u.followed_at, "
                "p.pronouns, p.location, p.demeanor, p.interests, "
                "p.profile_updated_at, "
                "COALESCE(p.is_starred, 0) AS is_starred "
                "FROM users u "
                "LEFT JOIN user_profiles p ON p.user_id = u.twitch_id "
                "WHERE u.twitch_id = ?",
                (twitch_id,),
            )
            r = cur.fetchone()
            if not r:
                return None
            interests_raw = r["interests"]
            interests: list[str] | None = None
            if interests_raw:
                try:
                    parsed = json.loads(interests_raw)
                    if isinstance(parsed, list):
                        interests = [str(x) for x in parsed if x]
                except (TypeError, ValueError):
                    interests = None
            return User(
                twitch_id=r["twitch_id"],
                name=r["name"],
                first_seen=r["first_seen"],
                last_seen=r["last_seen"],
                opt_out=bool(r["opt_out"]),
                sub_tier=r["sub_tier"],
                sub_months=int(r["sub_months"] or 0),
                is_mod=bool(r["is_mod"]),
                is_vip=bool(r["is_vip"]),
                is_founder=bool(r["is_founder"]),
                source=r["source"] or "twitch",
                merged_into=r["merged_into"],
                pronouns=r["pronouns"],
                location=r["location"],
                demeanor=r["demeanor"],
                interests=interests,
                profile_updated_at=r["profile_updated_at"],
                is_starred=bool(r["is_starred"]),
                followed_at=r["followed_at"] if "followed_at" in r.keys() else None,
            )

    # ============================================================
    # CANONICAL BATCH HELPERS
    # ------------------------------------------------------------
    # The following three functions are the **gold-standard pattern**
    # for fetching repo data by user_id when the caller has more than
    # one user in hand. Use them — never write `for uid in ids:
    # repo.get_user(uid)` style loops. Each batch helper:
    #
    #   - Issues ONE SQL statement with a parameterised IN-list, not N.
    #   - Returns a `dict[user_id, ...]` so callers can reorder freely
    #     and skip the missing keys without extra branches.
    #   - Trims the SELECT to only what the dataclass needs.
    #
    # When you find yourself adding a new "for uid in ids: repo.X(uid)"
    # call site, add a `repo.X_for_users(ids)` sibling here instead and
    # link to this comment from its docstring. The N+1 → 1 win
    # compounds: cache-friendly SQL, fewer Python round-trips, fewer
    # SQLite GIL-released waits.
    # ============================================================

    def get_users_by_ids(self, ids: list[str] | set[str]) -> dict[str, User]:
        """Fetch many users in one query. See "Canonical batch helpers"
        comment above. Empty input → empty dict; missing ids are
        silently absent from the result (caller checks with `.get`).

        Used by the live-chat SSE rendering and any other place that
        needs to attach user metadata to a batch of messages."""
        ids = list(set(ids))
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with self._cursor() as cur:
            cur.execute(
                f"SELECT u.twitch_id, u.name, u.first_seen, u.last_seen, u.opt_out, "
                f"u.sub_tier, u.sub_months, u.is_mod, u.is_vip, u.is_founder, "
                f"u.source, u.merged_into, u.followed_at, "
                f"p.pronouns, p.location, p.demeanor, p.interests, "
                f"p.profile_updated_at, "
                f"COALESCE(p.is_starred, 0) AS is_starred "
                f"FROM users u "
                f"LEFT JOIN user_profiles p ON p.user_id = u.twitch_id "
                f"WHERE u.twitch_id IN ({placeholders})",
                ids,
            )
            out: dict[str, User] = {}
            for r in cur.fetchall():
                interests_raw = r["interests"]
                interests: list[str] | None = None
                if interests_raw:
                    try:
                        parsed = json.loads(interests_raw)
                        if isinstance(parsed, list):
                            interests = [str(x) for x in parsed if x]
                    except (TypeError, ValueError):
                        interests = None
                out[r["twitch_id"]] = User(
                    twitch_id=r["twitch_id"],
                    name=r["name"],
                    first_seen=r["first_seen"],
                    last_seen=r["last_seen"],
                    opt_out=bool(r["opt_out"]),
                    sub_tier=r["sub_tier"],
                    sub_months=int(r["sub_months"] or 0),
                    is_mod=bool(r["is_mod"]),
                    is_vip=bool(r["is_vip"]),
                    is_founder=bool(r["is_founder"]),
                    source=r["source"] or "twitch",
                    merged_into=r["merged_into"],
                    pronouns=r["pronouns"],
                    location=r["location"],
                    demeanor=r["demeanor"],
                    interests=interests,
                    profile_updated_at=r["profile_updated_at"],
                    is_starred=bool(r["is_starred"]),
                    followed_at=r["followed_at"] if "followed_at" in r.keys() else None,
                )
            return out

    def get_notes_for_users(
        self, ids: list[str] | set[str],
    ) -> dict[str, list[Note]]:
        """Fetch every note for a batch of users in one query.
        Returns `{user_id: [Note, ...]}` newest-first per user. See
        "Canonical batch helpers" above. Used by the talking-points
        refresh; each active chatter contributes a notes section to
        the prompt and we previously paid one DB round-trip per
        chatter."""
        ids = list(set(ids))
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with self._cursor() as cur:
            cur.execute(
                f"SELECT id, user_id, ts, text, origin FROM notes "
                f"WHERE user_id IN ({placeholders}) ORDER BY ts DESC",
                ids,
            )
            out: dict[str, list[Note]] = {uid: [] for uid in ids}
            for r in cur.fetchall():
                out.setdefault(r["user_id"], []).append(Note(
                    id=int(r["id"]), user_id=r["user_id"], ts=r["ts"],
                    text=r["text"], origin=r["origin"] or "manual",
                ))
            return out

    def get_recent_messages_for_users(
        self, ids: list[str] | set[str], *, per_user_limit: int = 10,
    ) -> dict[str, list[Message]]:
        """Fetch the most recent N non-emote, non-spam messages per
        user for a batch of users — in ONE query using a window
        function, not N. Returns `{user_id: [Message, ...]}` newest-
        first per user. See "Canonical batch helpers" above.

        Used by the talking-points refresh, which needs each active
        chatter's recent context to ground the LLM's hook line."""
        ids = list(set(ids))
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        # ROW_NUMBER over (PARTITION BY user_id ORDER BY id DESC) is
        # the classic SQLite pattern for "top N per group". Beats
        # N separate LIMIT queries by orders of magnitude.
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT id, user_id, name, source, ts, content,
                       reply_parent_login, reply_parent_body
                FROM (
                    SELECT m.id, m.user_id, u.name, u.source,
                           m.ts, m.content,
                           m.reply_parent_login, m.reply_parent_body,
                           ROW_NUMBER() OVER (
                               PARTITION BY m.user_id
                               ORDER BY m.id DESC
                           ) AS rn
                    FROM messages m
                    JOIN users u ON u.twitch_id = m.user_id
                    WHERE m.user_id IN ({placeholders})
                      AND m.is_emote_only = 0
                      AND m.spam_score < 0.5
                )
                WHERE rn <= ?
                ORDER BY user_id, id DESC
                """,
                (*ids, int(per_user_limit)),
            )
            out: dict[str, list[Message]] = {uid: [] for uid in ids}
            for r in cur.fetchall():
                out.setdefault(r["user_id"], []).append(Message(
                    id=int(r["id"]), user_id=r["user_id"], name=r["name"],
                    ts=r["ts"], content=r["content"],
                    reply_parent_login=r["reply_parent_login"],
                    reply_parent_body=r["reply_parent_body"],
                    source=r["source"] or "twitch",
                ))
            return out

    def get_user_aliases(self, user_id: str) -> list[Alias]:
        """All names this user has been observed under, newest-active first."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT name, first_seen, last_seen_as
                FROM user_aliases
                WHERE user_id = ?
                ORDER BY last_seen_as DESC
                """,
                (user_id,),
            )
            return [
                Alias(name=r["name"], first_seen=r["first_seen"], last_seen_as=r["last_seen_as"])
                for r in cur.fetchall()
            ]

    def get_note_sources(self, note_id: int) -> list[Message]:
        """Hydrate the source messages cited for one note, oldest-first."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT m.id, m.user_id, u.name, m.ts, m.content,
                       m.reply_parent_login, m.reply_parent_body
                FROM note_sources ns
                JOIN messages m ON m.id = ns.message_id
                JOIN users u    ON u.twitch_id = m.user_id
                WHERE ns.note_id = ?
                ORDER BY m.id ASC
                """,
                (note_id,),
            )
            return [
                Message(
                    id=int(r["id"]), user_id=r["user_id"], name=r["name"],
                    ts=r["ts"], content=r["content"],
                    reply_parent_login=r["reply_parent_login"],
                    reply_parent_body=r["reply_parent_body"],
                )
                for r in cur.fetchall()
            ]

    def get_notes_with_sources(self, user_id: str) -> list[NoteWithSources]:
        """All notes for a user, each bundled with its cited source messages.
        Convenience wrapper for the user-detail render path."""
        notes = self.get_notes(user_id)
        return [
            NoteWithSources(note=n, sources=self.get_note_sources(n.id))
            for n in notes
        ]

    def get_notes_filtered(
        self,
        user_id: str,
        *,
        query: str = "",
        origin: str = "all",
        limit: int = 20,
        offset: int = 0,
    ) -> list[NoteWithSources]:
        """Filtered + paginated notes for the user-detail page.

        `origin` is one of 'all' | 'manual' | 'llm' | 'suspect'.
        - 'manual' = origin column = 'manual'
        - 'llm'    = origin column = 'llm' AND has at least one cited source
        - 'suspect' = origin column = 'llm' AND no cited sources
                      (pre-hallucination-guard era — flag for review)

        `query` is a case-insensitive LIKE substring match against the
        note text.
        """
        params: list = [user_id]
        clauses: list[str] = ["n.user_id = ?"]

        if query and query.strip():
            clauses.append("LOWER(n.text) LIKE ?")
            params.append(f"%{query.strip().lower()}%")

        join_sql = ""
        having_sql = ""
        if origin == "manual":
            clauses.append("(n.origin IS NULL OR n.origin = 'manual')")
        elif origin == "llm":
            clauses.append("n.origin = 'llm'")
            join_sql = "LEFT JOIN note_sources ns ON ns.note_id = n.id"
            having_sql = "HAVING COUNT(ns.message_id) > 0"
        elif origin == "suspect":
            clauses.append("n.origin = 'llm'")
            join_sql = "LEFT JOIN note_sources ns ON ns.note_id = n.id"
            having_sql = "HAVING COUNT(ns.message_id) = 0"

        where_sql = " AND ".join(clauses)
        group_sql = "GROUP BY n.id" if join_sql else ""
        sql = f"""
            SELECT n.id, n.user_id, n.ts, n.text, n.origin
            FROM notes n
            {join_sql}
            WHERE {where_sql}
            {group_sql}
            {having_sql}
            ORDER BY n.ts DESC
            LIMIT ? OFFSET ?
        """
        params.extend([int(limit), int(offset)])

        with self._cursor() as cur:
            rows = cur.execute(sql, params).fetchall()
            notes = [
                Note(
                    id=int(r["id"]), user_id=r["user_id"], ts=r["ts"],
                    text=r["text"], origin=r["origin"] or "manual",
                )
                for r in rows
            ]
        return [
            NoteWithSources(note=n, sources=self.get_note_sources(n.id))
            for n in notes
        ]

    def count_notes_filtered(
        self,
        user_id: str,
        *,
        query: str = "",
        origin: str = "all",
    ) -> int:
        """Match the filter logic of get_notes_filtered for pagination math."""
        params: list = [user_id]
        clauses: list[str] = ["n.user_id = ?"]

        if query and query.strip():
            clauses.append("LOWER(n.text) LIKE ?")
            params.append(f"%{query.strip().lower()}%")

        join_sql = ""
        having_sql = ""
        if origin == "manual":
            clauses.append("(n.origin IS NULL OR n.origin = 'manual')")
        elif origin == "llm":
            clauses.append("n.origin = 'llm'")
            join_sql = "LEFT JOIN note_sources ns ON ns.note_id = n.id"
            having_sql = "HAVING COUNT(ns.message_id) > 0"
        elif origin == "suspect":
            clauses.append("n.origin = 'llm'")
            join_sql = "LEFT JOIN note_sources ns ON ns.note_id = n.id"
            having_sql = "HAVING COUNT(ns.message_id) = 0"

        where_sql = " AND ".join(clauses)

        if join_sql:
            sql = f"""
                SELECT COUNT(*) FROM (
                  SELECT n.id
                  FROM notes n
                  {join_sql}
                  WHERE {where_sql}
                  GROUP BY n.id
                  {having_sql}
                )
            """
        else:
            sql = f"SELECT COUNT(*) FROM notes n WHERE {where_sql}"

        with self._cursor() as cur:
            return int(cur.execute(sql, params).fetchone()[0])

    def set_user_profile(
        self,
        twitch_id: str,
        *,
        pronouns: str | None,
        location: str | None,
        demeanor: str | None,
        interests: list[str] | None,
    ) -> None:
        """Strict-set semantics for the manual edit form: empty string ⇒ NULL,
        empty list ⇒ NULL. Distinct from update_user_profile (which has
        merge-only semantics for the LLM extractor)."""
        norm_pronouns = pronouns.strip() if pronouns and pronouns.strip() else None
        norm_location = location.strip() if location and location.strip() else None
        norm_demeanor = demeanor.strip() if demeanor and demeanor.strip() else None
        cleaned: list[str] = []
        if interests:
            seen: set[str] = set()
            for entry in interests:
                if not entry:
                    continue
                e = entry.strip()
                if not e or e.lower() in seen:
                    continue
                cleaned.append(e)
                seen.add(e.lower())
        norm_interests = json.dumps(cleaned) if cleaned else None
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_profiles
                    (user_id, pronouns, location, demeanor, interests, profile_updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    pronouns           = excluded.pronouns,
                    location           = excluded.location,
                    demeanor           = excluded.demeanor,
                    interests          = excluded.interests,
                    profile_updated_at = excluded.profile_updated_at
                """,
                (
                    twitch_id,
                    norm_pronouns, norm_location, norm_demeanor,
                    norm_interests, _now_iso(),
                ),
            )

    def get_notes(self, user_id: str) -> list[Note]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT id, user_id, ts, text, origin FROM notes "
                "WHERE user_id = ? ORDER BY ts DESC",
                (user_id,),
            )
            return [
                Note(
                    id=int(r["id"]), user_id=r["user_id"], ts=r["ts"],
                    text=r["text"], origin=r["origin"] or "manual",
                )
                for r in cur.fetchall()
            ]

    def get_note(self, note_id: int) -> Note | None:
        with self._cursor() as cur:
            cur.execute(
                "SELECT id, user_id, ts, text, origin FROM notes WHERE id = ?",
                (note_id,),
            )
            r = cur.fetchone()
            if not r:
                return None
            return Note(
                id=int(r["id"]), user_id=r["user_id"], ts=r["ts"],
                text=r["text"], origin=r["origin"] or "manual",
            )

    def update_note(self, note_id: int, text: str) -> None:
        # Edits invalidate the embedding; the dashboard can re-embed lazily if it
        # wants. We zero it here so vec_notes lookups don't return stale matches.
        with self._cursor() as cur:
            cur.execute(
                "UPDATE notes SET text = ?, embedding = NULL WHERE id = ?",
                (text, note_id),
            )
            cur.execute("DELETE FROM vec_notes WHERE note_id = ?", (note_id,))
        # Streamer correcting an LLM-extracted note is the highest-
        # signal supervision example we have — fine-tuning gold.
        self._capture_streamer_action(
            action_kind="note", item_key=str(note_id),
            action="updated", note=text,
        )

    def delete_note(self, note_id: int) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM vec_notes WHERE note_id = ?", (note_id,))
            cur.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        # Negative supervision: the streamer rejected this note.
        self._capture_streamer_action(
            action_kind="note", item_key=str(note_id),
            action="deleted", note=None,
        )

    def set_user_followed_at(self, twitch_id: str, followed_at: str | None) -> None:
        """Update users.followed_at without touching anything else.
        Idempotent — setting the same value is a no-op. Called by the
        HelixSyncService after each /channels/followers poll."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE users SET followed_at = ? WHERE twitch_id = ?",
                (followed_at, twitch_id),
            )

    def upsert_chatter_minimal(
        self, twitch_id: str, name: str, source: str = "twitch",
    ) -> bool:
        """Lightweight upsert used by Helix sync — inserts a stub row
        if the user has never spoken in chat, so Helix-only signals
        (sub list, follower list, vip/mod list) get persisted even
        before the chatter sends their first message. Returns True if
        a new row was created.

        Differs from `upsert_user` in that we don't touch first_seen /
        last_seen — those are chat-driven timestamps and shouldn't be
        falsely advanced just because Helix saw them.
        """
        with self._cursor() as cur:
            cur.execute(
                "SELECT 1 FROM users WHERE twitch_id = ? LIMIT 1",
                (twitch_id,),
            )
            if cur.fetchone():
                return False
            now = _now_iso()
            cur.execute(
                """
                INSERT INTO users(
                    twitch_id, name, first_seen, last_seen, opt_out, source
                ) VALUES (?, ?, ?, ?, 0, ?)
                """,
                (twitch_id, name, now, now, source),
            )
            return True

    def helix_apply_role_snapshot(
        self,
        *,
        vips:    dict[str, str] | None = None,  # twitch_id -> login
        mods:    dict[str, str] | None = None,
        subs:    dict[str, dict] | None = None,  # twitch_id -> {login, tier}
        followers: dict[str, dict] | None = None,  # twitch_id -> {login, followed_at}
    ) -> dict[str, int]:
        """Bulk apply a Helix snapshot to the users table. Each map is
        optional — pass only the slices you have scope for. Returns a
        per-slice count of rows touched (existing + inserted).

        Semantics:
          - vips/mods: the role flag is set TRUE for users in the snapshot
            and FALSE for users not in the snapshot but currently flagged
            (so role removals propagate). Subs follow the same pattern
            for sub_tier (None when no longer subbed). Followers update
            followed_at to whatever Helix returned; followers not in the
            current poll are left alone (we only ever poll the recent
            page, so absence isn't a signal of unfollow).
          - sub_months is preserved — Helix's `/subscriptions` doesn't
            return cumulative months, only the current tier. IRCv3 tags
            from incoming chat keep that field accurate.
          - is_founder is preserved — Helix has no notion of it; the
            badge comes from chat tags only.

        New rows are inserted via upsert_chatter_minimal so a chatter
        we've never had chat from still gets a record.
        """
        counts = {"vips": 0, "mods": 0, "subs": 0, "followers": 0}

        # Insert stubs for any user we've never seen in chat. Doing
        # this first means subsequent UPDATEs hit existing rows.
        for snap in (vips, mods, subs, followers):
            if not snap:
                continue
            for uid, info in snap.items():
                if isinstance(info, dict):
                    name = info.get("login") or info.get("name") or uid
                else:
                    name = info or uid
                self.upsert_chatter_minimal(uid, name)

        with self._cursor() as cur:
            if vips is not None:
                cur.execute("UPDATE users SET is_vip = 0 WHERE is_vip = 1")
                for uid in vips:
                    cur.execute(
                        "UPDATE users SET is_vip = 1 WHERE twitch_id = ?",
                        (uid,),
                    )
                    counts["vips"] += cur.rowcount
            if mods is not None:
                cur.execute("UPDATE users SET is_mod = 0 WHERE is_mod = 1")
                for uid in mods:
                    cur.execute(
                        "UPDATE users SET is_mod = 1 WHERE twitch_id = ?",
                        (uid,),
                    )
                    counts["mods"] += cur.rowcount
            if subs is not None:
                # Wipe sub_tier for everyone currently flagged; the
                # snapshot then re-asserts.
                cur.execute(
                    "UPDATE users SET sub_tier = NULL WHERE sub_tier IS NOT NULL"
                )
                for uid, info in subs.items():
                    cur.execute(
                        "UPDATE users SET sub_tier = ? WHERE twitch_id = ?",
                        (info.get("tier") or None, uid),
                    )
                    counts["subs"] += cur.rowcount
            if followers is not None:
                # Don't wipe — followers not in this page might just
                # be older. Only update what we observed.
                for uid, info in followers.items():
                    cur.execute(
                        "UPDATE users SET followed_at = COALESCE(followed_at, ?) "
                        "WHERE twitch_id = ?",
                        (info.get("followed_at"), uid),
                    )
                    counts["followers"] += cur.rowcount

        return counts

    def reset_session_addressed_states(self) -> int:
        """Clear all current 'addressed' insight_states. Used at the
        start of a new stream session — "I addressed this on stream"
        is a per-session signal; carrying it forward makes the new
        stream feel like everything's already handled.

        The audit trail in `insight_state_history` is preserved; only
        the current snapshot in `insight_states` is wiped. Streamer
        can still see "I addressed X yesterday" via /audit if they
        want.

        Returns the number of rows cleared.
        """
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM insight_states WHERE state = 'addressed'"
            )
            n = int(cur.fetchone()["c"])
            cur.execute("DELETE FROM insight_states WHERE state = 'addressed'")
            return n

    def merge_notes(
        self,
        note_ids: list[int],
        merged_text: str,
        *,
        embedding: list[float] | None = None,
    ) -> int | None:
        """Combine multiple notes into one umbrella note. The new note's
        `source_message_ids` is the union of every original's sources.
        Originals are deleted (DB + vec_notes). Returns the new note's
        id, or None if `note_ids` was empty.

        All notes must belong to the same user — caller is responsible
        for verifying ownership before invoking.
        """
        if not note_ids:
            return None
        clean_text = (merged_text or "").strip()
        if not clean_text:
            return None
        ids_set = list(dict.fromkeys(int(i) for i in note_ids))  # dedup, preserve order
        placeholders = ",".join("?" for _ in ids_set)

        with self._cursor() as cur:
            # Look up the user_id from any of the originals (they should
            # all match — caller's responsibility).
            cur.execute(
                f"SELECT DISTINCT user_id FROM notes WHERE id IN ({placeholders})",
                ids_set,
            )
            rows = cur.fetchall()
            if not rows:
                return None
            user_ids = {r["user_id"] for r in rows}
            if len(user_ids) > 1:
                raise ValueError(
                    f"merge_notes: notes span multiple users {user_ids}"
                )
            user_id = rows[0]["user_id"]

            # Union the source_message_ids across the originals.
            cur.execute(
                f"""
                SELECT DISTINCT message_id FROM note_sources
                WHERE note_id IN ({placeholders})
                """,
                ids_set,
            )
            source_msg_ids = [int(r["message_id"]) for r in cur.fetchall()]

            # Insert the merged note. origin='manual' since the streamer
            # curated the merged text; we don't want the hallucination
            # guard to drop it for lacking citations even if the union
            # ends up empty.
            cur.execute(
                """
                INSERT INTO notes(user_id, ts, text, embedding, origin)
                VALUES (?, ?, ?, ?, 'manual')
                """,
                (
                    user_id, _now_iso(), clean_text,
                    _vec_to_blob(embedding) if embedding else None,
                ),
            )
            new_id = int(cur.lastrowid)

            # Re-attach the unioned sources to the new note.
            for mid in source_msg_ids:
                cur.execute(
                    "INSERT OR IGNORE INTO note_sources(note_id, message_id) "
                    "VALUES (?, ?)",
                    (new_id, mid),
                )

            # Delete originals (vec + notes + their note_sources).
            cur.execute(
                f"DELETE FROM vec_notes WHERE note_id IN ({placeholders})",
                ids_set,
            )
            cur.execute(
                f"DELETE FROM note_sources WHERE note_id IN ({placeholders})",
                ids_set,
            )
            cur.execute(
                f"DELETE FROM notes WHERE id IN ({placeholders})",
                ids_set,
            )

            # If we have an embedding, write to vec_notes for RAG search.
            if embedding:
                try:
                    cur.execute(
                        "INSERT INTO vec_notes(note_id, embedding) VALUES (?, ?)",
                        (new_id, _vec_to_blob(embedding)),
                    )
                except Exception:
                    pass

        return new_id

    def get_messages(
        self,
        user_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
        query: str = "",
    ) -> list[Message]:
        params: list[Any] = [user_id]
        where = "WHERE m.user_id = ?"
        if query:
            where += " AND LOWER(m.content) LIKE ?"
            params.append(f"%{query.lower()}%")
        params.extend([limit, offset])
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT m.id, m.user_id, u.name, m.ts, m.content, m.reply_parent_login, m.reply_parent_body
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                {where}
                ORDER BY m.id DESC
                LIMIT ? OFFSET ?
                """,
                params,
            )
            return [
                Message(
                    id=int(r["id"]),
                    user_id=r["user_id"],
                    name=r["name"],
                    ts=r["ts"],
                    content=r["content"],
                    reply_parent_login=r["reply_parent_login"],
                    reply_parent_body=r["reply_parent_body"],
                )
                for r in cur.fetchall()
            ]

    def find_reply_parent_id(
        self, login: str, body: str, *, before_id: int | None = None,
    ) -> int | None:
        """Resolve the actual `messages.id` of a stored reply-parent so the
        UI can hop up the thread. We don't store parent ids — Twitch's
        IRCv3 reply tag only carries login + body — so we approximate by
        finding the most recent matching message from that user prior to
        the reply itself.

        `before_id`, when supplied, caps the lookup to messages older than
        the reply (the parent has to come BEFORE the reply that quotes it).
        Match is on user_login + content prefix (200-char body limit) with
        an exact equality first, falling back to a prefix match for cases
        where the body got truncated by the IRC tag encoding.
        """
        if not login or not body:
            return None
        body = body.strip()
        with self._cursor() as cur:
            params: list[Any] = [login.lower(), body]
            id_clause = ""
            if before_id is not None:
                id_clause = " AND m.id < ?"
                params.append(int(before_id))
            cur.execute(
                f"""
                SELECT m.id FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                WHERE LOWER(u.name) = ?
                  AND m.content = ?
                  {id_clause}
                ORDER BY m.id DESC LIMIT 1
                """,
                params,
            )
            r = cur.fetchone()
            if r:
                return int(r["id"])
            # Fallback: prefix match (handles truncation in the IRC tag).
            params2: list[Any] = [login.lower(), body[:60] + "%"]
            id_clause2 = ""
            if before_id is not None:
                id_clause2 = " AND m.id < ?"
                params2.append(int(before_id))
            cur.execute(
                f"""
                SELECT m.id FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                WHERE LOWER(u.name) = ?
                  AND m.content LIKE ?
                  {id_clause2}
                ORDER BY m.id DESC LIMIT 1
                """,
                params2,
            )
            r = cur.fetchone()
            return int(r["id"]) if r else None

    def get_message_context(
        self, message_id: int, *, before: int = 3, after: int = 3
    ) -> dict[str, Any]:
        """Return the focal message plus N channel-wide messages on each side
        for conversational context. Result:
            {"focal": Message|None,
             "before": list[Message],   # oldest first
             "after":  list[Message],   # oldest first
             "parent_ids": dict[int, int]}  # message_id → parent_message_id
              (only for rows whose reply-parent we could resolve, so the
              modal can render a "↑ jump to parent" link).
        Cross-user — the surrounding rows show who else was talking around
        that moment."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT m.id, m.user_id, u.name, u.source, m.ts, m.content,
                       m.reply_parent_login, m.reply_parent_body
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                WHERE m.id = ?
                """,
                (message_id,),
            )
            r = cur.fetchone()
            focal = (
                Message(
                    id=int(r["id"]), user_id=r["user_id"], name=r["name"],
                    ts=r["ts"], content=r["content"],
                    reply_parent_login=r["reply_parent_login"],
                    reply_parent_body=r["reply_parent_body"],
                    source=r["source"] or "twitch",
                )
                if r else None
            )
            if not focal:
                return {"focal": None, "before": [], "after": []}

            cur.execute(
                """
                SELECT m.id, m.user_id, u.name, u.source, m.ts, m.content,
                       m.reply_parent_login, m.reply_parent_body
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                WHERE m.id < ?
                ORDER BY m.id DESC
                LIMIT ?
                """,
                (message_id, int(before)),
            )
            before_rows = [
                Message(
                    id=int(r["id"]), user_id=r["user_id"], name=r["name"],
                    ts=r["ts"], content=r["content"],
                    reply_parent_login=r["reply_parent_login"],
                    reply_parent_body=r["reply_parent_body"],
                    source=r["source"] or "twitch",
                )
                for r in cur.fetchall()
            ]
            before_rows.reverse()  # oldest first

            cur.execute(
                """
                SELECT m.id, m.user_id, u.name, u.source, m.ts, m.content,
                       m.reply_parent_login, m.reply_parent_body
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                WHERE m.id > ?
                ORDER BY m.id ASC
                LIMIT ?
                """,
                (message_id, int(after)),
            )
            after_rows = [
                Message(
                    id=int(r["id"]), user_id=r["user_id"], name=r["name"],
                    ts=r["ts"], content=r["content"],
                    reply_parent_login=r["reply_parent_login"],
                    reply_parent_body=r["reply_parent_body"],
                    source=r["source"] or "twitch",
                )
                for r in cur.fetchall()
            ]

        # Resolve the actual message id of each reply-parent so the modal
        # can render a "↑ jump to parent" link that re-anchors on it.
        # Skipped for rows without a reply tag.
        parent_ids: dict[int, int] = {}
        for m in (focal, *before_rows, *after_rows):
            if m and m.reply_parent_login and m.reply_parent_body:
                pid = self.find_reply_parent_id(
                    m.reply_parent_login, m.reply_parent_body,
                    before_id=m.id,
                )
                if pid is not None:
                    parent_ids[m.id] = pid

        return {
            "focal": focal, "before": before_rows, "after": after_rows,
            "parent_ids": parent_ids,
        }

    def count_messages(self, user_id: str, query: str = "") -> int:
        params: list[Any] = [user_id]
        where = "WHERE user_id = ?"
        if query:
            where += " AND LOWER(content) LIKE ?"
            params.append(f"%{query.lower()}%")
        with self._cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS c FROM messages {where}", params)
            return int(cur.fetchone()["c"])

    def get_user_events(self, user_id: str, limit: int = 50) -> list[Event]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, twitch_name, type, amount, currency, message, ts
                FROM events
                WHERE user_id = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            return [_event_from_row(r) for r in cur.fetchall()]

    def get_user_event_summary(self, user_id: str) -> UserEventSummary:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN type='tip' THEN amount END), 0)        AS total_tip,
                    (SELECT currency FROM events
                       WHERE user_id=? AND type='tip' AND currency IS NOT NULL
                       ORDER BY ts DESC LIMIT 1)                                  AS last_currency,
                    (SELECT MAX(ts) FROM events WHERE user_id=? AND type='tip')   AS last_tip_ts,
                    COALESCE(SUM(CASE WHEN type='cheer' THEN amount END), 0)      AS total_bits,
                    COALESCE(MAX(CASE WHEN type='sub' THEN amount END), 0)        AS sub_months,
                    (SELECT currency FROM events
                       WHERE user_id=? AND type='sub'
                       ORDER BY ts DESC LIMIT 1)                                  AS last_sub_tier,
                    (SELECT MAX(ts) FROM events WHERE user_id=? AND type='sub')   AS last_sub_ts
                FROM events
                WHERE user_id = ?
                """,
                (user_id, user_id, user_id, user_id, user_id),
            )
            r = cur.fetchone()
            if not r:
                return UserEventSummary(0.0, None, None, 0, 0, None, None)
            return UserEventSummary(
                total_tip_amount=float(r["total_tip"] or 0.0),
                tip_currency=r["last_currency"],
                last_tip_ts=r["last_tip_ts"],
                total_bits=int(r["total_bits"] or 0),
                sub_months=int(r["sub_months"] or 0),
                sub_tier=r["last_sub_tier"],
                last_sub_ts=r["last_sub_ts"],
            )

    def list_events(
        self,
        *,
        type_filter: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Event]:
        params: list[Any] = []
        where = ""
        if type_filter:
            where = "WHERE type = ?"
            params.append(type_filter)
        params.extend([limit, offset])
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT id, user_id, twitch_name, type, amount, currency, message, ts
                FROM events
                {where}
                ORDER BY ts DESC
                LIMIT ? OFFSET ?
                """,
                params,
            )
            return [_event_from_row(r) for r in cur.fetchall()]

    def list_topic_snapshots(self, limit: int = 20) -> list[TopicSnapshot]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT id, ts, summary, message_id_range, topics_json
                FROM topic_snapshots
                ORDER BY ts DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [
                TopicSnapshot(
                    id=int(r["id"]),
                    ts=r["ts"],
                    summary=r["summary"],
                    message_id_range=r["message_id_range"],
                    topics_json=r["topics_json"],
                )
                for r in cur.fetchall()
            ]

    # ============================ Topic threads ===========================

    # Computed status thresholds (kept here so the SQL CASE matches
    # whatever Python uses to interpret it elsewhere if ever needed).
    THREAD_ACTIVE_HOURS = 1
    THREAD_DORMANT_HOURS = 24

    def find_thread_by_embedding(
        self, embedding: list[float], k: int = 1
    ) -> tuple[int, float] | None:
        """Nearest-neighbour search over thread title embeddings.
        Returns (thread_id, distance) of the closest, or None if empty.
        sqlite-vec returns cosine distance — smaller = more similar."""
        blob = _vec_to_blob(embedding)
        with self._cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT thread_id, distance
                    FROM vec_threads
                    WHERE embedding MATCH ? AND k = ?
                    ORDER BY distance ASC
                    LIMIT 1
                    """,
                    (blob, max(1, int(k))),
                )
                r = cur.fetchone()
            except sqlite3.OperationalError:
                return None
        if not r:
            return None
        return (int(r["thread_id"]), float(r["distance"]))

    def attach_topic_to_thread(
        self,
        *,
        thread_id: int,
        snapshot_id: int,
        topic_index: int,
        title: str,
        category: str | None,
        drivers: list[str],
        ts: str,
    ) -> None:
        """Append a snapshot/topic_index as a member of an existing thread,
        and update the thread's title (latest wins) + last_ts + category."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR IGNORE INTO topic_thread_members(
                    thread_id, snapshot_id, topic_index, drivers, ts
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (thread_id, snapshot_id, topic_index, json.dumps(drivers), ts),
            )
            cur.execute(
                """
                UPDATE topic_threads
                SET title = ?,
                    last_ts = MAX(last_ts, ?),
                    category = COALESCE(?, category)
                WHERE id = ?
                """,
                (title, ts, category, thread_id),
            )

    def create_topic_thread(
        self,
        *,
        snapshot_id: int,
        topic_index: int,
        title: str,
        category: str | None,
        drivers: list[str],
        ts: str,
        embedding: list[float],
    ) -> int:
        blob = _vec_to_blob(embedding)
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO topic_threads(title, first_ts, last_ts, category, embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                (title, ts, ts, category, blob),
            )
            tid = int(cur.lastrowid)
            cur.execute(
                "INSERT INTO vec_threads(thread_id, embedding) VALUES (?, ?)",
                (tid, blob),
            )
            cur.execute(
                """
                INSERT INTO topic_thread_members(
                    thread_id, snapshot_id, topic_index, drivers, ts
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (tid, snapshot_id, topic_index, json.dumps(drivers), ts),
            )
            return tid

    _STATUS_CASE = (
        f"CASE "
        f"  WHEN t.last_ts >= datetime('now', '-{THREAD_ACTIVE_HOURS} hours') "
        f"    THEN 'active' "
        f"  WHEN t.last_ts >= datetime('now', '-{THREAD_DORMANT_HOURS} hours') "
        f"    THEN 'dormant' "
        f"  ELSE 'archived' "
        f"END"
    )

    def list_threads(
        self,
        *,
        status_filter: str | None = None,  # 'active'|'dormant'|'archived' or None
        query: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> list[TopicThread]:
        """Threads with member counts and union-of-driver lists. Search by
        title or driver name when `query` is non-empty."""
        params: list[Any] = []
        clauses: list[str] = []
        if status_filter:
            clauses.append(f"{self._STATUS_CASE} = ?")
            params.append(status_filter)
        if query:
            like = f"%{query.lower()}%"
            clauses.append(
                "(LOWER(t.title) LIKE ? OR EXISTS ("
                "  SELECT 1 FROM topic_thread_members m"
                "  WHERE m.thread_id = t.id AND LOWER(m.drivers) LIKE ?))"
            )
            params.extend([like, like])
        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.extend([limit, offset])
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT t.id, t.title, t.first_ts, t.last_ts, t.category,
                       t.recap, t.recap_updated_at,
                       (SELECT COUNT(*) FROM topic_thread_members m WHERE m.thread_id = t.id) AS mc,
                       {self._STATUS_CASE} AS status
                FROM topic_threads t
                {where_sql}
                ORDER BY t.last_ts DESC
                LIMIT ? OFFSET ?
                """,
                params,
            )
            rows = [dict(r) for r in cur.fetchall()]
        # Drivers are pulled in a separate query per thread; releasing the
        # cursor lock before that prevents re-entry deadlocks.
        out: list[TopicThread] = []
        for r in rows:
            drivers = self._collect_thread_drivers(int(r["id"]))
            out.append(
                TopicThread(
                    id=int(r["id"]),
                    title=r["title"],
                    first_ts=r["first_ts"],
                    last_ts=r["last_ts"],
                    drivers=drivers,
                    member_count=int(r["mc"]),
                    status=r["status"],
                    category=r["category"],
                    recap=r["recap"],
                    recap_updated_at=r["recap_updated_at"],
                )
            )
        return out

    def count_threads(
        self, *, status_filter: str | None = None, query: str = ""
    ) -> int:
        params: list[Any] = []
        clauses: list[str] = []
        if status_filter:
            clauses.append(f"{self._STATUS_CASE} = ?")
            params.append(status_filter)
        if query:
            like = f"%{query.lower()}%"
            clauses.append(
                "(LOWER(t.title) LIKE ? OR EXISTS ("
                "  SELECT 1 FROM topic_thread_members m"
                "  WHERE m.thread_id = t.id AND LOWER(m.drivers) LIKE ?))"
            )
            params.extend([like, like])
        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) AS c FROM topic_threads t {where_sql}", params
            )
            return int(cur.fetchone()["c"])

    def list_high_impact_subjects(
        self,
        *,
        active_within_minutes: int = 30,
        lookback_days: int = 14,
        min_overlap: int = 2,
        limit: int = 6,
    ) -> list[dict]:
        """Subjects that would engage the chatters currently in chat.

        Cross-references chatters active in the last `active_within_minutes`
        against the historical drivers of every topic_thread (within the
        `lookback_days` window). Threads where the most current chatters
        have driven that thread historically rise to the top.

        Use case: the streamer wants to know which subject to pivot to
        for maximum live audience engagement, given who's actually
        watching right now.

        Returns dicts:
          thread            — TopicThread (with recap if any)
          overlap_drivers   — list of {name, twitch_id} for currently-
                              active chatters who have driven this
                              thread historically
          overlap_count     — len(overlap_drivers)
          unique_drivers    — total distinct drivers across all snapshots
                              (channel-wide reach, beyond the active set)
        """
        # Step 1 — pull the active set of chatter names (lowercased)
        # alongside their twitch_id, so we can de-dup correctly across
        # alias hits below.
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT u.twitch_id, u.name
                FROM users u
                JOIN messages m ON m.user_id = u.twitch_id
                WHERE u.opt_out = 0
                  AND datetime(m.ts) >= datetime('now', ?)
                """,
                (f"-{int(active_within_minutes)} minutes",),
            )
            active_rows = cur.fetchall()
        if not active_rows:
            return []
        active_names_lc = {r["name"].lower(): r["twitch_id"] for r in active_rows if r["name"]}
        if not active_names_lc:
            return []

        # Step 2 — for each thread in the lookback window, find drivers
        # whose name (case-insensitive) matches any active chatter.
        # `json_each` expands the drivers JSON array into rows so we
        # can join on it.
        sql = f"""
            SELECT t.id, t.title, t.first_ts, t.last_ts, t.category,
                   t.recap, t.recap_updated_at,
                   (SELECT COUNT(*) FROM topic_thread_members
                    WHERE thread_id = t.id) AS mc,
                   {self._STATUS_CASE} AS status,
                   COUNT(DISTINCT j.value) AS unique_drivers
            FROM topic_threads t
            JOIN topic_thread_members m ON m.thread_id = t.id
            JOIN json_each(m.drivers) AS j
            WHERE datetime(t.last_ts) >= datetime('now', ?)
            GROUP BY t.id
            ORDER BY unique_drivers DESC, t.last_ts DESC
            LIMIT 200
        """
        with self._cursor() as cur:
            cur.execute(sql, (f"-{int(lookback_days)} days",))
            thread_rows = cur.fetchall()

        # Step 3 — collect per-thread drivers in TWO batched queries
        # (was N+1: one per-thread query inside the for loop, plus a
        # second per-thread query via _collect_thread_drivers, so each
        # /insights load was firing 200+ extra cursors). Now we pull
        # all candidate threads' drivers in one shot and group in
        # Python — single round trip per call.
        candidate_ids = [int(t["id"]) for t in thread_rows]
        all_drivers_by_thread: dict[int, list[str]] = {
            tid: [] for tid in candidate_ids
        }
        if candidate_ids:
            placeholders = ",".join("?" * len(candidate_ids))
            with self._cursor() as cur:
                cur.execute(
                    f"""
                    SELECT m.thread_id, j.value AS name
                    FROM topic_thread_members m
                    JOIN json_each(m.drivers) AS j
                    WHERE m.thread_id IN ({placeholders})
                    ORDER BY m.ts ASC
                    """,
                    candidate_ids,
                )
                seen_pair: set[tuple[int, str]] = set()
                for r in cur.fetchall():
                    tid = int(r["thread_id"])
                    name = r["name"]
                    if not name:
                        continue
                    key = (tid, name.lower())
                    if key in seen_pair:
                        continue
                    seen_pair.add(key)
                    all_drivers_by_thread[tid].append(name)

        out: list[dict] = []
        for trow in thread_rows:
            tid = int(trow["id"])
            driver_names = all_drivers_by_thread.get(tid, [])
            overlap: list[dict] = []
            seen_uids: set[str] = set()
            for n in driver_names:
                uid = active_names_lc.get(n.lower())
                if uid and uid not in seen_uids:
                    seen_uids.add(uid)
                    overlap.append({"name": n, "twitch_id": uid})
            if len(overlap) < int(min_overlap):
                continue
            # Reuse the already-fetched driver set as the thread's
            # drivers list — same source as _collect_thread_drivers
            # would return, but without the second per-thread query.
            thread = TopicThread(
                id=tid,
                title=trow["title"],
                first_ts=trow["first_ts"],
                last_ts=trow["last_ts"],
                drivers=list(driver_names),
                member_count=int(trow["mc"]),
                status=trow["status"],
                category=trow["category"],
                recap=trow["recap"],
                recap_updated_at=trow["recap_updated_at"],
            )
            out.append({
                "thread": thread,
                "overlap_drivers": overlap,
                "overlap_count": len(overlap),
                "unique_drivers": int(trow["unique_drivers"]),
            })
            if len(out) >= int(limit):
                break
        # Sort: highest overlap first; tiebreak by total channel reach.
        out.sort(key=lambda d: (-d["overlap_count"], -d["unique_drivers"]))
        return out[:int(limit)]

    def subjects_engaging_chatter(
        self, twitch_id: str, *, limit: int = 10,
    ) -> list[dict]:
        """Topic threads this chatter has driven, ranked by drive count
        + recency. Used by the per-chatter modal to answer "what
        subjects engage this person?" — observational only, no LLM
        prescription.

        Resolves the chatter's current name + every historical alias,
        so a renamed user still gets credit for threads they drove
        under their old name.

        Returns dicts with: thread (TopicThread), drive_count,
        last_drove_at. Ordered most-driven first, then most-recent.
        """
        # Pull current name + every alias (lowercased) so renamed
        # chatters still match.
        names: set[str] = set()
        with self._cursor() as cur:
            cur.execute(
                "SELECT name FROM users WHERE twitch_id = ?",
                (twitch_id,),
            )
            r = cur.fetchone()
            if r and r["name"]:
                names.add(r["name"].lower())
            cur.execute(
                "SELECT name FROM user_aliases WHERE user_id = ?",
                (twitch_id,),
            )
            for row in cur.fetchall():
                if row["name"]:
                    names.add(row["name"].lower())
        if not names:
            return []

        placeholders = ",".join("?" for _ in names)
        sql = f"""
            SELECT t.id, t.title, t.first_ts, t.last_ts, t.category,
                   t.recap, t.recap_updated_at,
                   (SELECT COUNT(*) FROM topic_thread_members
                    WHERE thread_id = t.id) AS mc,
                   {self._STATUS_CASE} AS status,
                   COUNT(DISTINCT m.snapshot_id) AS drive_count,
                   MAX(m.ts) AS last_drove_at
            FROM topic_threads t
            JOIN topic_thread_members m ON m.thread_id = t.id
            JOIN json_each(m.drivers) AS j
            WHERE LOWER(j.value) IN ({placeholders})
            GROUP BY t.id
            ORDER BY drive_count DESC, last_drove_at DESC
            LIMIT ?
        """
        with self._cursor() as cur:
            cur.execute(sql, list(names) + [int(limit)])
            rows = [dict(r) for r in cur.fetchall()]

        out: list[dict] = []
        for row in rows:
            tid = int(row["id"])
            thread = TopicThread(
                id=tid,
                title=row["title"],
                first_ts=row["first_ts"],
                last_ts=row["last_ts"],
                drivers=self._collect_thread_drivers(tid),
                member_count=int(row["mc"]),
                status=row["status"],
                category=row["category"],
                recap=row["recap"],
                recap_updated_at=row["recap_updated_at"],
            )
            out.append({
                "thread": thread,
                "drive_count": int(row["drive_count"]),
                "last_drove_at": row["last_drove_at"],
            })
        return out

    def list_quiet_thread_cohorts(
        self,
        *,
        silence_minutes: int = 15,
        lookback_hours: int = 24,
        min_drivers: int = 2,
        limit: int = 8,
    ) -> list[dict]:
        """Topic threads whose driver chatters have all gone quiet.

        Detects "lapsed cohorts" — groups of chatters who shared an
        interest in a clustered conversation but have stopped speaking.
        The streamer can use this surface to decide whether to pivot
        back to a topic that would re-engage that group.

        Returns dicts with:
          thread          — TopicThread (with recap if populated)
          drivers         — list of {name, last_seen} dicts
          driver_count    — resolved driver count (matched to users.name)
          cohort_last_ts  — when the most-recent driver spoke
          minutes_quiet   — minutes since cohort_last_ts

        Sort: largest cohorts first, then longest-silent. The biggest
        "missing crowd" floats to the top.

        Limitations:
          - Driver name → user lookup is by current users.name only.
            Renamed chatters (history in user_aliases) won't resolve;
            their cohort weight is reduced. Acceptable for now.
          - Only considers active+dormant threads (within `lookback_hours`).
            Archived threads are intentionally skipped — they're done.
        """
        sql = f"""
            SELECT t.id, t.title, t.first_ts, t.last_ts, t.category,
                   t.recap, t.recap_updated_at,
                   (SELECT COUNT(*) FROM topic_thread_members
                    WHERE thread_id = t.id) AS mc,
                   {self._STATUS_CASE} AS status,
                   MAX(u.last_seen) AS cohort_last_ts,
                   COUNT(DISTINCT u.twitch_id) AS resolved_drivers
            FROM topic_threads t
            JOIN topic_thread_members m ON m.thread_id = t.id
            JOIN json_each(m.drivers) AS j
            JOIN users u ON LOWER(u.name) = LOWER(j.value)
            WHERE datetime(t.last_ts) >= datetime('now', ?)
            GROUP BY t.id
            HAVING datetime(MAX(u.last_seen)) < datetime('now', ?)
               AND COUNT(DISTINCT u.twitch_id) >= ?
            ORDER BY resolved_drivers DESC, cohort_last_ts ASC
            LIMIT ?
        """
        with self._cursor() as cur:
            cur.execute(
                sql,
                (
                    f"-{int(lookback_hours)} hours",
                    f"-{int(silence_minutes)} minutes",
                    int(min_drivers),
                    int(limit),
                ),
            )
            rows = [dict(r) for r in cur.fetchall()]

        # Batched driver lookup — was N+1: per-thread _collect_thread_drivers
        # plus per-driver users.last_seen lookups (so for 6 cohort threads
        # with 5 drivers each, that's ~36 cursor opens). Now: 2 batched
        # queries total (drivers across all threads, then users for all
        # named drivers in one IN clause).
        from datetime import datetime as _dt, timezone as _tz
        now_utc = _dt.now(_tz.utc)
        if not rows:
            return []
        candidate_ids = [int(r["id"]) for r in rows]

        drivers_by_thread: dict[int, list[str]] = {tid: [] for tid in candidate_ids}
        all_names: set[str] = set()
        with self._cursor() as cur:
            ph = ",".join("?" * len(candidate_ids))
            cur.execute(
                f"""
                SELECT thread_id, drivers
                FROM topic_thread_members
                WHERE thread_id IN ({ph})
                ORDER BY ts ASC
                """,
                candidate_ids,
            )
            seen_per_thread: dict[int, set[str]] = {tid: set() for tid in candidate_ids}
            for r in cur.fetchall():
                tid = int(r["thread_id"])
                try:
                    arr = json.loads(r["drivers"])
                except (TypeError, ValueError):
                    arr = []
                for name in arr:
                    if (
                        isinstance(name, str)
                        and name not in seen_per_thread[tid]
                    ):
                        seen_per_thread[tid].add(name)
                        drivers_by_thread[tid].append(name)
                        all_names.add(name)

        # Resolve every distinct driver name → last_seen in one IN query.
        last_seen_by_lower: dict[str, str] = {}
        if all_names:
            names_list = list(all_names)
            with self._cursor() as cur:
                ph = ",".join("?" * len(names_list))
                cur.execute(
                    f"SELECT name, last_seen FROM users "
                    f"WHERE LOWER(name) IN ({','.join(['LOWER(?)'] * len(names_list))})",
                    names_list,
                )
                # The query's IN clause uses LOWER(?) per slot so
                # parameterisation expands cleanly. Map by lowercased
                # name so we can look up by the driver's display-cased
                # name without a second normalize.
                for r in cur.fetchall():
                    nm = r["name"]
                    if nm:
                        last_seen_by_lower[nm.lower()] = r["last_seen"]

        out: list[dict] = []
        for row in rows:
            tid = int(row["id"])
            driver_names = drivers_by_thread.get(tid, [])
            driver_info: list[dict] = []
            for name in driver_names:
                ls = last_seen_by_lower.get(name.lower())
                if ls is not None:
                    driver_info.append({"name": name, "last_seen": ls})
            try:
                cohort_ts = (row["cohort_last_ts"] or "").replace(" ", "T")
                last = _dt.fromisoformat(cohort_ts)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=_tz.utc)
                minutes_quiet = max(0, int((now_utc - last).total_seconds() / 60))
            except (TypeError, ValueError):
                minutes_quiet = 0
            thread = TopicThread(
                id=tid,
                title=row["title"],
                first_ts=row["first_ts"],
                last_ts=row["last_ts"],
                drivers=driver_names,
                member_count=int(row["mc"]),
                status=row["status"],
                category=row["category"],
                recap=row["recap"],
                recap_updated_at=row["recap_updated_at"],
            )
            out.append({
                "thread": thread,
                "drivers": driver_info,
                "driver_count": int(row["resolved_drivers"]),
                "cohort_last_ts": row["cohort_last_ts"],
                "minutes_quiet": minutes_quiet,
            })
        return out

    def _collect_thread_drivers(self, thread_id: int) -> list[str]:
        """Union of all driver names across the thread's members, preserving
        first-seen order for a stable display."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT drivers FROM topic_thread_members
                WHERE thread_id = ? ORDER BY ts ASC
                """,
                (thread_id,),
            )
            seen: list[str] = []
            seen_set: set[str] = set()
            for r in cur.fetchall():
                try:
                    arr = json.loads(r["drivers"])
                except (TypeError, ValueError):
                    arr = []
                for name in arr:
                    if isinstance(name, str) and name not in seen_set:
                        seen_set.add(name)
                        seen.append(name)
            return seen

    def get_thread(self, thread_id: int) -> TopicThread | None:
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT t.id, t.title, t.first_ts, t.last_ts, t.category,
                       t.recap, t.recap_updated_at,
                       (SELECT COUNT(*) FROM topic_thread_members m WHERE m.thread_id = t.id) AS mc,
                       {self._STATUS_CASE} AS status
                FROM topic_threads t
                WHERE t.id = ?
                """,
                (thread_id,),
            )
            r = cur.fetchone()
            if not r:
                return None
            row = dict(r)
        # Re-acquire after releasing — _collect_thread_drivers takes the lock.
        return TopicThread(
            id=int(row["id"]),
            title=row["title"],
            first_ts=row["first_ts"],
            last_ts=row["last_ts"],
            drivers=self._collect_thread_drivers(thread_id),
            member_count=int(row["mc"]),
            status=row["status"],
            category=row["category"],
            recap=row["recap"],
            recap_updated_at=row["recap_updated_at"],
        )

    def get_thread_members(self, thread_id: int) -> list[TopicThreadMember]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT thread_id, snapshot_id, topic_index, drivers, ts
                FROM topic_thread_members
                WHERE thread_id = ?
                ORDER BY ts ASC
                """,
                (thread_id,),
            )
            out: list[TopicThreadMember] = []
            for r in cur.fetchall():
                try:
                    drivers = json.loads(r["drivers"])
                except (TypeError, ValueError):
                    drivers = []
                out.append(
                    TopicThreadMember(
                        thread_id=int(r["thread_id"]),
                        snapshot_id=int(r["snapshot_id"]),
                        topic_index=int(r["topic_index"]),
                        drivers=[str(d) for d in drivers],
                        ts=r["ts"],
                    )
                )
            return out

    def set_thread_recap(self, thread_id: int, recap: str) -> None:
        """Persist the LLM-generated thread recap. The InsightsService
        recap loop calls this after batch-generating recaps for active
        threads."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE topic_threads SET recap = ?, recap_updated_at = ? "
                "WHERE id = ?",
                (recap.strip() or None, _now_iso(), int(thread_id)),
            )

    def get_thread_messages(
        self, thread_id: int, limit: int = 200
    ) -> list[Message]:
        """Union of messages from all member-snapshots' id ranges, filtered to
        members' driver names. Deduped by message id, sorted oldest-first."""
        members = self.get_thread_members(thread_id)
        if not members:
            return []
        all_drivers: list[str] = []
        seen: set[str] = set()
        for m in members:
            for d in m.drivers:
                if d not in seen:
                    seen.add(d)
                    all_drivers.append(d)
        msg_seen: set[int] = set()
        out: list[Message] = []
        for m in members:
            snap = self.get_topic_snapshot(m.snapshot_id)
            if not snap or not snap.message_id_range:
                continue
            try:
                a, b = snap.message_id_range.split("-", 1)
                first_id, last_id = int(a), int(b)
            except (ValueError, AttributeError):
                continue
            chunk = self.messages_in_id_range_for_names(
                first_id, last_id, all_drivers, limit=limit,
            )
            for msg in chunk:
                if msg.id in msg_seen:
                    continue
                msg_seen.add(msg.id)
                out.append(msg)
            if len(out) >= limit:
                break
        out.sort(key=lambda m: m.id)
        return out[:limit]

    def snapshots_without_threads(self) -> list[TopicSnapshot]:
        """Snapshots that have topics_json but no thread members yet — fed to
        the Threader's backfill at bot startup."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT id, ts, summary, message_id_range, topics_json
                FROM topic_snapshots
                WHERE topics_json IS NOT NULL
                  AND id NOT IN (SELECT DISTINCT snapshot_id FROM topic_thread_members)
                ORDER BY id ASC
                """
            )
            return [
                TopicSnapshot(
                    id=int(r["id"]),
                    ts=r["ts"],
                    summary=r["summary"],
                    message_id_range=r["message_id_range"],
                    topics_json=r["topics_json"],
                )
                for r in cur.fetchall()
            ]

    # ---------------- Word cloud (Stats tab) ----------------

    _WORDCLOUD_STOPWORDS = frozenset(
        # English stopwords
        "a an the and or but if while of in on at by for with about against "
        "between into through during before after above below from up down "
        "out off over under again further then once here there when where "
        "why how all any both each few more most other some such no nor not "
        "only own same so than too very can will just don should now is are "
        "was were be been being have has had do does did would could may "
        "might must shall need to as because until i me my myself we our "
        "ours ourselves you your yours yourself yourselves he him his "
        "himself she her hers herself it its itself they them their theirs "
        "themselves what which who whom this that these those am ".split()
    ) | frozenset(
        # Chat noise / filler
        "lol lmao lmfao rofl kek kekw kekl lulw lul pog pogchamp poggers "
        "gg ez wp hi hello hey yo sup wsg gn bye cya thanks thx ty np "
        "yes yeah yep yup nope nah ok okay alright sure cool nice good bad "
        "really actually basically literally definitely probably maybe "
        "guys guy dude man bro lady stream streamer chat people thing "
        "things stuff way back even still right left big small new old "
        "got get make made see seen say said know knew think thought "
        "want need feel felt look looking looked come came take took use "
        "used go going gone went well like also much many lot lots one two "
        "three first last great cool fine pretty kind sort lot ".split()
    ) | frozenset(
        # Common contractions w/o apostrophe
        "im ive ill id youre youve youll youd hes shes its were theyre "
        "theyve theyll theyd dont doesnt didnt wont wouldnt couldnt shouldnt "
        "isnt arent wasnt werent havent hasnt hadnt cant cannot lets ".split()
    )

    def stats_top_words(
        self,
        *,
        limit: int = 100,
        min_count: int = 3,
        lookback_days: int | None = 30,
    ) -> list[tuple[str, int]]:
        """Naive top-words for the Stats tab word cloud. Strips URLs and
        @mentions, lowercases, drops stopwords + chat filler, requires
        3-20 char alphabetic tokens. Skips messages flagged as emote-only
        so 'bawkCrazy' style hype spam doesn't dominate.

        Bounded to the last `lookback_days` by default (30 d) — the word
        cloud is more useful as "what's chat been on lately" than as
        a lifetime average, and bounding the scan keeps the cost flat
        as the DB grows. Pass `lookback_days=None` for the lifetime
        view."""
        import re as _re
        from collections import Counter as _Counter
        url_re = _re.compile(r"https?://\S+", _re.IGNORECASE)
        mention_re = _re.compile(r"@\S+")
        word_re = _re.compile(r"\b[a-z]{3,20}\b")
        counter: _Counter[str] = _Counter()
        sql = (
            "SELECT content FROM messages "
            "WHERE is_emote_only = 0 AND spam_score < 0.5"
        )
        params: tuple = ()
        if lookback_days is not None and lookback_days > 0:
            sql += " AND ts >= datetime('now', ?)"
            params = (f"-{int(lookback_days)} days",)
        with self._cursor() as cur:
            cur.execute(sql, params)
            for r in cur.fetchall():
                text = (r["content"] or "").lower()
                text = url_re.sub("", text)
                text = mention_re.sub("", text)
                for w in word_re.findall(text):
                    if w in self._WORDCLOUD_STOPWORDS:
                        continue
                    counter[w] += 1
        return [(w, c) for w, c in counter.most_common(limit) if c >= min_count]

    def stats_top_words_transcripts(
        self,
        *,
        limit: int = 100,
        min_count: int = 2,
        lookback_days: int | None = 30,
    ) -> list[tuple[str, int]]:
        """Top-words across whisper transcripts — same tokenisation as
        chat, but min_count defaults lower since transcript volume is
        much smaller than chat volume on most streams. Bounded to the
        last `lookback_days` for the same reason as `stats_top_words`."""
        import re as _re
        from collections import Counter as _Counter
        word_re = _re.compile(r"\b[a-z]{3,20}\b")
        counter: _Counter[str] = _Counter()
        sql = "SELECT text FROM transcript_chunks"
        params: tuple = ()
        if lookback_days is not None and lookback_days > 0:
            sql += " WHERE ts >= datetime('now', ?)"
            params = (f"-{int(lookback_days)} days",)
        try:
            with self._cursor() as cur:
                cur.execute(sql, params)
                for r in cur.fetchall():
                    text = (r["text"] or "").lower()
                    for w in word_re.findall(text):
                        if w in self._WORDCLOUD_STOPWORDS:
                            continue
                        counter[w] += 1
        except sqlite3.Error:
            return []
        return [(w, c) for w, c in counter.most_common(limit) if c >= min_count]

    def get_topic_snapshot(self, snapshot_id: int) -> TopicSnapshot | None:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT id, ts, summary, message_id_range, topics_json
                FROM topic_snapshots WHERE id = ?
                """,
                (snapshot_id,),
            )
            r = cur.fetchone()
            if not r:
                return None
            return TopicSnapshot(
                id=int(r["id"]),
                ts=r["ts"],
                summary=r["summary"],
                message_id_range=r["message_id_range"],
                topics_json=r["topics_json"],
            )

    def messages_for_names_within(
        self,
        names: list[str],
        *,
        within_minutes: int = 30,
        limit: int = 50,
    ) -> list[Message]:
        """Recent messages from any of `names` (matched against current
        users.name OR historical aliases) within the lookback window,
        oldest-first. Used by the engaging-subjects expand route to
        surface verbatim context for a subject."""
        if not names:
            return []
        placeholders = ",".join("?" for _ in names)
        lower_names = [n.lower() for n in names]
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT u.twitch_id
                FROM users u
                LEFT JOIN user_aliases a ON a.user_id = u.twitch_id
                WHERE LOWER(u.name) IN ({placeholders})
                   OR LOWER(a.name) IN ({placeholders})
                """,
                lower_names + lower_names,
            )
            twitch_ids = [r["twitch_id"] for r in cur.fetchall()]
            if not twitch_ids:
                return []
            id_placeholders = ",".join("?" for _ in twitch_ids)
            cur.execute(
                f"""
                SELECT m.id, m.user_id, u.name, u.source, m.ts, m.content,
                       m.reply_parent_login, m.reply_parent_body
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                WHERE m.user_id IN ({id_placeholders})
                  AND m.is_emote_only = 0 AND m.spam_score < 0.5
                  AND datetime(m.ts) >= datetime('now', ?)
                ORDER BY m.id DESC
                LIMIT ?
                """,
                twitch_ids + [f"-{int(within_minutes)} minutes", int(limit)],
            )
            rows = cur.fetchall()
        out = [
            Message(
                id=int(r["id"]), user_id=r["user_id"], name=r["name"],
                source=r["source"] or "twitch",
                ts=r["ts"], content=r["content"],
                reply_parent_login=r["reply_parent_login"],
                reply_parent_body=r["reply_parent_body"],
            )
            for r in rows
        ]
        out.reverse()
        return out

    def messages_in_id_range_for_names(
        self,
        first_id: int,
        last_id: int,
        names: list[str],
        limit: int = 100,
    ) -> list[Message]:
        """Pull messages within [first_id, last_id] sent by users currently or
        historically known under any of `names`. Used by the topic modal to
        show what those drivers actually said in that snapshot's window."""
        if not names:
            return []
        # Resolve every alias variant to the parent twitch_id so renames don't
        # silently lose context.
        placeholders = ",".join("?" for _ in names)
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT u.twitch_id
                FROM users u
                LEFT JOIN user_aliases a ON a.user_id = u.twitch_id
                WHERE LOWER(u.name) IN ({placeholders})
                   OR LOWER(a.name) IN ({placeholders})
                """,
                [n.lower() for n in names] + [n.lower() for n in names],
            )
            twitch_ids = [r["twitch_id"] for r in cur.fetchall()]
            if not twitch_ids:
                return []
            id_placeholders = ",".join("?" for _ in twitch_ids)
            cur.execute(
                f"""
                SELECT m.id, m.user_id, u.name, m.ts, m.content, m.reply_parent_login, m.reply_parent_body
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                WHERE m.id BETWEEN ? AND ?
                  AND m.user_id IN ({id_placeholders})
                ORDER BY m.id ASC
                LIMIT ?
                """,
                [first_id, last_id, *twitch_ids, limit],
            )
            return [
                Message(
                    id=int(r["id"]), user_id=r["user_id"], name=r["name"],
                    ts=r["ts"], content=r["content"],
                    reply_parent_login=r["reply_parent_login"],
                    reply_parent_body=r["reply_parent_body"],
                )
                for r in cur.fetchall()
            ]

    def find_user_by_alias_or_name(self, name: str) -> User | None:
        """Resolve a username to a user, transparently following renames."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT u.twitch_id, u.name, u.first_seen, u.last_seen, u.opt_out, u.sub_tier, u.sub_months, u.is_mod, u.is_vip, u.is_founder
                FROM users u
                WHERE LOWER(u.name) = LOWER(?)
                   OR EXISTS (
                       SELECT 1 FROM user_aliases a
                       WHERE a.user_id = u.twitch_id AND LOWER(a.name) = LOWER(?)
                   )
                LIMIT 1
                """,
                (name, name),
            )
            r = cur.fetchone()
            if not r:
                return None
            return User(
                twitch_id=r["twitch_id"], name=r["name"],
                first_seen=r["first_seen"], last_seen=r["last_seen"],
                opt_out=bool(r["opt_out"]),
                sub_tier=r["sub_tier"],
                sub_months=int(r["sub_months"] or 0),
                is_mod=bool(r["is_mod"]),
                is_vip=bool(r["is_vip"]),
                is_founder=bool(r["is_founder"]),
            )

    # ============================ Mutations from streamer surfaces ==========

    def forget_user(self, twitch_id: str) -> None:
        """Hard-delete every row touching a user (user / notes / messages / events /
        summarization_state / vec entries)."""
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM vec_notes WHERE note_id IN (SELECT id FROM notes WHERE user_id = ?)",
                (twitch_id,),
            )
            cur.execute(
                "DELETE FROM vec_messages WHERE message_id IN "
                "(SELECT id FROM messages WHERE user_id = ?)",
                (twitch_id,),
            )
            cur.execute("DELETE FROM events WHERE user_id = ?", (twitch_id,))
            # ON DELETE CASCADE handles notes / messages / summarization_state.
            cur.execute("DELETE FROM users WHERE twitch_id = ?", (twitch_id,))

    def set_opt_out(self, twitch_id: str, opt_out: bool) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE users SET opt_out = ? WHERE twitch_id = ?",
                (1 if opt_out else 0, twitch_id),
            )

    # ============================ RAG helpers (dashboard-only) =============
    # These power the dashboard "Ask Qwen about this user" feature. Both the
    # query and the answer render in the streamer's browser only.

    def upsert_message_embedding(
        self, message_id: int, embedding: list[float]
    ) -> None:
        blob = _vec_to_blob(embedding)
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM vec_messages WHERE message_id = ?", (message_id,)
            )
            cur.execute(
                "INSERT INTO vec_messages(message_id, embedding) VALUES (?, ?)",
                (message_id, blob),
            )

    def find_near_duplicate_flood(
        self,
        focal_message_id: int,
        embedding: list[float],
        *,
        within_seconds: int = 60,
        cosine_distance_max: float = 0.05,
        knn_top_k: int = 20,
    ) -> list[tuple[int, str]]:
        """Look for a copy-paste brigade around a freshly-embedded
        message. Returns `(message_id, user_id)` for the focal message
        plus every other message within `within_seconds` whose
        embedding is closer than `cosine_distance_max` (cosine
        distance, so 0.05 ≈ 0.95 similarity) and that came from a
        DIFFERENT user.

        Returns [] when the focal message itself can't be found (e.g.
        deleted) or when sqlite-vec isn't available. Caller decides
        whether the cluster size justifies bumping spam scores."""
        blob = _vec_to_blob(embedding)
        with self._cursor() as cur:
            # Resolve focal message metadata first so we can filter by
            # arrival time and exclude same-user matches.
            cur.execute(
                "SELECT user_id, ts FROM messages WHERE id = ?",
                (int(focal_message_id),),
            )
            row = cur.fetchone()
            if row is None:
                return []
            focal_user = row["user_id"]
            try:
                cur.execute(
                    """
                    SELECT v.message_id, v.distance, m.user_id, m.ts
                    FROM vec_messages v
                    JOIN messages m ON m.id = v.message_id
                    WHERE v.embedding MATCH ? AND k = ?
                      AND v.distance <= ?
                      AND datetime(m.ts) >= datetime('now', ?)
                      AND m.user_id != ?
                    ORDER BY v.distance ASC
                    """,
                    (
                        blob, int(knn_top_k),
                        float(cosine_distance_max),
                        f"-{int(within_seconds)} seconds",
                        focal_user,
                    ),
                )
                rows = cur.fetchall()
            except sqlite3.OperationalError:
                return []
        # Always include the focal message in the cluster — it's the
        # one that just arrived and triggered the check, and the
        # caller wants to bump it too if a flood is confirmed.
        out: list[tuple[int, str]] = [(int(focal_message_id), focal_user)]
        for r in rows:
            out.append((int(r["message_id"]), r["user_id"]))
        return out

    def bump_spam_score(
        self,
        message_ids: list[int],
        *,
        score: float,
        reason: str,
    ) -> int:
        """Raise `spam_score` toward `score` (only if higher) for the
        given message ids, and append `reason` to the JSON array in
        `spam_reasons` (idempotent — won't add duplicates).

        Used by the flood detector to retroactively flag messages
        that, on their own, looked clean but were part of a copy-
        paste brigade. Returns the count of rows updated."""
        if not message_ids:
            return 0
        from .spam import decode_reasons, encode_reasons
        score = max(0.0, min(1.0, float(score)))
        with self._cursor() as cur:
            placeholders = ",".join("?" * len(message_ids))
            cur.execute(
                f"SELECT id, spam_score, spam_reasons FROM messages "
                f"WHERE id IN ({placeholders})",
                [int(mid) for mid in message_ids],
            )
            rows = cur.fetchall()
            updated = 0
            for r in rows:
                cur_score = float(r["spam_score"] or 0.0)
                cur_reasons = decode_reasons(r["spam_reasons"])
                new_score = max(cur_score, score)
                if reason not in cur_reasons:
                    cur_reasons = cur_reasons + [reason]
                if new_score == cur_score and reason in decode_reasons(r["spam_reasons"]):
                    continue  # already at or above this score with this reason
                cur.execute(
                    "UPDATE messages SET spam_score = ?, spam_reasons = ? WHERE id = ?",
                    (new_score, encode_reasons(cur_reasons), int(r["id"])),
                )
                updated += 1
        return updated

    def messages_missing_embedding(self, user_id: str, limit: int) -> list[Message]:
        """Return messages for a user that don't yet have a vec_messages row."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT m.id, m.user_id, u.name, m.ts, m.content, m.reply_parent_login, m.reply_parent_body
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                LEFT JOIN vec_messages v ON v.message_id = m.id
                WHERE m.user_id = ? AND v.message_id IS NULL
                ORDER BY m.id DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            return [
                Message(
                    id=int(r["id"]),
                    user_id=r["user_id"],
                    name=r["name"],
                    ts=r["ts"],
                    content=r["content"],
                    reply_parent_login=r["reply_parent_login"],
                    reply_parent_body=r["reply_parent_body"],
                )
                for r in cur.fetchall()
            ]

    def search_global_messages(
        self,
        query_embedding: list[float],
        k: int = 20,
        *,
        exclude_emote_only: bool = True,
    ) -> list[tuple[Message, float]]:
        """Semantic search across every embedded message in the channel.
        Returns (Message, distance) pairs sorted nearest-first. Distance is
        the vec0 KNN score (lower = closer match)."""
        blob = _vec_to_blob(query_embedding)
        ann_k = max(k * 4, 80)
        emote_clause = "AND m.is_emote_only = 0 AND m.spam_score < 0.5" if exclude_emote_only else ""
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT m.id, m.user_id, u.name, u.source, m.ts, m.content,
                       m.reply_parent_login, m.reply_parent_body,
                       v.distance AS dist
                FROM vec_messages v
                JOIN messages m ON m.id = v.message_id
                JOIN users u    ON u.twitch_id = m.user_id
                WHERE v.embedding MATCH ? AND k = ?
                  AND u.opt_out = 0
                  {emote_clause}
                ORDER BY v.distance
                LIMIT ?
                """,
                (blob, ann_k, k),
            )
            return [
                (
                    Message(
                        id=int(r["id"]),
                        user_id=r["user_id"],
                        name=r["name"],
                        ts=r["ts"],
                        content=r["content"],
                        reply_parent_login=r["reply_parent_login"],
                        reply_parent_body=r["reply_parent_body"],
                        source=r["source"] or "twitch",
                    ),
                    float(r["dist"]),
                )
                for r in cur.fetchall()
            ]

    def messages_embedding_coverage(self) -> tuple[int, int]:
        """Returns (indexed, total_text). `total_text` excludes emote-only
        rows since those aren't candidates for embedding."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE is_emote_only = 0 AND spam_score < 0.5"
            )
            total = int(cur.fetchone()["c"])
            cur.execute("SELECT COUNT(*) AS c FROM vec_messages")
            indexed = int(cur.fetchone()["c"])
        return indexed, total

    def messages_missing_embedding_global(self, limit: int) -> list[Message]:
        """Like `messages_missing_embedding` but channel-wide. Used by the
        backfill script to populate the search index over historical chat."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT m.id, m.user_id, u.name, m.ts, m.content,
                       m.reply_parent_login, m.reply_parent_body
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                LEFT JOIN vec_messages v ON v.message_id = m.id
                WHERE v.message_id IS NULL
                  AND m.is_emote_only = 0 AND m.spam_score < 0.5
                  AND u.opt_out = 0
                ORDER BY m.id ASC
                LIMIT ?
                """,
                (limit,),
            )
            return [
                Message(
                    id=int(r["id"]),
                    user_id=r["user_id"],
                    name=r["name"],
                    ts=r["ts"],
                    content=r["content"],
                    reply_parent_login=r["reply_parent_login"],
                    reply_parent_body=r["reply_parent_body"],
                )
                for r in cur.fetchall()
            ]

    def search_user_messages(
        self, user_id: str, query_embedding: list[float], k: int = 10
    ) -> list[Message]:
        blob = _vec_to_blob(query_embedding)
        # vec0 KNN requires `k = ?` inside the MATCH clause; we then re-filter to
        # the requested user. Pull a wider window from the index so the per-user
        # filter still has matches when the user has few embedded messages.
        ann_k = max(k * 4, 40)
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT m.id, m.user_id, u.name, m.ts, m.content, m.reply_parent_login, m.reply_parent_body
                FROM vec_messages v
                JOIN messages m ON m.id = v.message_id
                JOIN users u    ON u.twitch_id = m.user_id
                WHERE v.embedding MATCH ? AND k = ?
                  AND m.user_id = ?
                ORDER BY v.distance
                LIMIT ?
                """,
                (blob, ann_k, user_id, k),
            )
            return [
                Message(
                    id=int(r["id"]),
                    user_id=r["user_id"],
                    name=r["name"],
                    ts=r["ts"],
                    content=r["content"],
                    reply_parent_login=r["reply_parent_login"],
                    reply_parent_body=r["reply_parent_body"],
                )
                for r in cur.fetchall()
            ]

    def search_user_notes(
        self, user_id: str, query_embedding: list[float], k: int = 5
    ) -> list[Note]:
        blob = _vec_to_blob(query_embedding)
        ann_k = max(k * 4, 20)
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT n.id, n.user_id, n.ts, n.text
                FROM vec_notes v
                JOIN notes n ON n.id = v.note_id
                WHERE v.embedding MATCH ? AND k = ?
                  AND n.user_id = ?
                ORDER BY v.distance
                LIMIT ?
                """,
                (blob, ann_k, user_id, k),
            )
            return [
                Note(id=int(r["id"]), user_id=r["user_id"], ts=r["ts"], text=r["text"])
                for r in cur.fetchall()
            ]

    # ============================ Moderation (opt-in) =====================

    _MOD_WATERMARK_KEY = "mod_review_watermark"

    def get_mod_watermark(self) -> int:
        v = self.get_app_setting(self._MOD_WATERMARK_KEY)
        try:
            return int(v) if v is not None else 0
        except (TypeError, ValueError):
            return 0

    def set_mod_watermark(self, msg_id: int) -> None:
        self.set_app_setting(self._MOD_WATERMARK_KEY, str(int(msg_id)))

    def messages_for_mod_review(self, limit: int) -> list[Message]:
        """Pull messages newer than the moderation watermark, oldest first.
        Skips emote-only messages — they're hype noise, not policy content."""
        wm = self.get_mod_watermark()
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT m.id, m.user_id, u.name, m.ts, m.content, m.reply_parent_login, m.reply_parent_body
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                WHERE m.id > ? AND m.is_emote_only = 0 AND m.spam_score < 0.5
                ORDER BY m.id ASC
                LIMIT ?
                """,
                (wm, limit),
            )
            return [
                Message(
                    id=int(r["id"]), user_id=r["user_id"], name=r["name"],
                    ts=r["ts"], content=r["content"],
                    reply_parent_login=r["reply_parent_login"],
                    reply_parent_body=r["reply_parent_body"],
                )
                for r in cur.fetchall()
            ]

    def recent_user_messages_with_context(
        self,
        user_id: str,
        *,
        user_limit: int = 10,
        ctx_before: int = 2,
        ctx_after: int = 2,
    ) -> tuple[list[Message], set[int]]:
        """Pull the user's most recent N messages plus surrounding chat
        context for each. Returns `(rows, focal_id_set)` where rows are
        oldest-first deduped across all sources, and focal_id_set marks
        which rows are the user's own (vs context lines from others).

        Used by the talking-point modal to show what the chatter is
        saying right now in the actual flow of chat."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT id FROM messages
                WHERE user_id = ? AND is_emote_only = 0 AND spam_score < 0.5
                ORDER BY id DESC LIMIT ?
                """,
                (user_id, int(user_limit)),
            )
            focal_ids = [int(r["id"]) for r in cur.fetchall()]
        if not focal_ids:
            return [], set()
        rows = self.channel_context_around_ids(
            focal_ids, before=ctx_before, after=ctx_after
        )
        return rows, set(focal_ids)

    def channel_context_around_ids(
        self,
        message_ids: list[int],
        *,
        before: int = 2,
        after: int = 2,
    ) -> list[Message]:
        """Pull a chat-wide context window around each focal message id.

        For every id in `message_ids`, fetch up to `before` and `after`
        adjacent messages across the whole channel (including the focal
        message itself). Returns the union, deduped, oldest-first. Skips
        emote-only rows so the context is actually informative.

        Used by the summarizer to give the LLM the surrounding chat for
        sarcasm + key-moment detection — a "lol kill them all" reaction
        reads very differently when the prior line was a clutch save vs
        a horror jump-scare.
        """
        if not message_ids:
            return []
        seen_ids: set[int] = set()
        with self._cursor() as cur:
            for mid in message_ids:
                cur.execute(
                    """
                    SELECT m.id FROM messages m
                    WHERE m.id < ? AND m.is_emote_only = 0 AND m.spam_score < 0.5
                    ORDER BY m.id DESC LIMIT ?
                    """,
                    (mid, int(before)),
                )
                seen_ids.update(int(r["id"]) for r in cur.fetchall())
                seen_ids.update(message_ids)
                cur.execute(
                    """
                    SELECT m.id FROM messages m
                    WHERE m.id > ? AND m.is_emote_only = 0 AND m.spam_score < 0.5
                    ORDER BY m.id ASC LIMIT ?
                    """,
                    (mid, int(after)),
                )
                seen_ids.update(int(r["id"]) for r in cur.fetchall())
            if not seen_ids:
                return []
            placeholders = ",".join("?" for _ in seen_ids)
            cur.execute(
                f"""
                SELECT m.id, m.user_id, u.name, m.ts, m.content,
                       m.reply_parent_login, m.reply_parent_body
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                WHERE m.id IN ({placeholders})
                ORDER BY m.id ASC
                """,
                tuple(seen_ids),
            )
            return [
                Message(
                    id=int(r["id"]),
                    user_id=r["user_id"],
                    name=r["name"],
                    ts=r["ts"],
                    content=r["content"],
                    reply_parent_login=r["reply_parent_login"],
                    reply_parent_body=r["reply_parent_body"],
                )
                for r in cur.fetchall()
            ]

    def get_messages_by_ids(self, ids: list[int]) -> list[Message]:
        """Hydrate full Message rows (including reply_parent_*) for a set of
        ids. Order is by id ASC. Used by the per-user summarizer so it can
        annotate each user line with its reply parent for context."""
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT m.id, m.user_id, u.name, m.ts, m.content, m.reply_parent_login, m.reply_parent_body
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                WHERE m.id IN ({placeholders})
                ORDER BY m.id ASC
                """,
                ids,
            )
            return [
                Message(
                    id=int(r["id"]), user_id=r["user_id"], name=r["name"],
                    ts=r["ts"], content=r["content"],
                    reply_parent_login=r["reply_parent_login"],
                    reply_parent_body=r["reply_parent_body"],
                )
                for r in cur.fetchall()
            ]

    def messages_before_id(self, before_id: int, limit: int) -> list[Message]:
        """Pull up to `limit` messages with id < before_id, returned oldest-first.
        Used as look-back context for the moderator (so a flag-worthy message
        right at the start of a batch isn't judged without its parent line)."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT m.id, m.user_id, u.name, m.ts, m.content, m.reply_parent_login, m.reply_parent_body
                FROM messages m
                JOIN users u ON u.twitch_id = m.user_id
                WHERE m.id < ?
                ORDER BY m.id DESC
                LIMIT ?
                """,
                (before_id, limit),
            )
            rows = [
                Message(
                    id=int(r["id"]), user_id=r["user_id"], name=r["name"],
                    ts=r["ts"], content=r["content"],
                    reply_parent_login=r["reply_parent_login"],
                    reply_parent_body=r["reply_parent_body"],
                )
                for r in cur.fetchall()
            ]
        rows.reverse()
        return rows

    def add_incident(
        self,
        *,
        user_id: str | None,
        message_id: int | None,
        severity: int,
        categories: list[str],
        rationale: str | None,
    ) -> int:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO incidents(user_id, message_id, ts, severity, categories, rationale, status)
                VALUES (?, ?, ?, ?, ?, ?, 'open')
                """,
                (
                    user_id,
                    message_id,
                    _now_iso(),
                    int(severity),
                    json.dumps(list(categories)),
                    rationale,
                ),
            )
            return int(cur.lastrowid)

    def list_incidents(
        self,
        *,
        status: str | None = "open",
        limit: int = 100,
        offset: int = 0,
    ) -> list[IncidentRow]:
        params: list[Any] = []
        where = []
        if status:
            where.append("i.status = ?")
            params.append(status)
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        params.extend([limit, offset])
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT i.id, i.user_id, i.message_id, i.ts, i.severity,
                       i.categories, i.rationale, i.status,
                       u.name AS user_name,
                       m.content AS message_content, m.reply_parent_login AS message_reply_parent_login, m.reply_parent_body AS message_reply_parent_body
                FROM incidents i
                LEFT JOIN users u    ON u.twitch_id = i.user_id
                LEFT JOIN messages m ON m.id        = i.message_id
                {where_sql}
                ORDER BY i.ts DESC
                LIMIT ? OFFSET ?
                """,
                params,
            )
            return [_incident_row_from_row(r) for r in cur.fetchall()]

    def count_incidents(self, status: str | None = None) -> int:
        params: list[Any] = []
        where_sql = ""
        if status:
            where_sql = "WHERE status = ?"
            params.append(status)
        with self._cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS c FROM incidents {where_sql}", params)
            return int(cur.fetchone()["c"])

    def get_user_incidents(self, user_id: str, limit: int = 25) -> list[IncidentRow]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT i.id, i.user_id, i.message_id, i.ts, i.severity,
                       i.categories, i.rationale, i.status,
                       u.name AS user_name,
                       m.content AS message_content, m.reply_parent_login AS message_reply_parent_login, m.reply_parent_body AS message_reply_parent_body
                FROM incidents i
                LEFT JOIN users u    ON u.twitch_id = i.user_id
                LEFT JOIN messages m ON m.id        = i.message_id
                WHERE i.user_id = ?
                ORDER BY i.ts DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            return [_incident_row_from_row(r) for r in cur.fetchall()]

    def update_incident_status(self, incident_id: int, status: str) -> None:
        if status not in ("open", "reviewed", "dismissed"):
            raise ValueError(f"invalid status: {status!r}")
        with self._cursor() as cur:
            cur.execute(
                "UPDATE incidents SET status = ? WHERE id = ?", (status, incident_id)
            )

    def get_incident(self, incident_id: int) -> IncidentRow | None:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT i.id, i.user_id, i.message_id, i.ts, i.severity,
                       i.categories, i.rationale, i.status,
                       u.name AS user_name,
                       m.content AS message_content, m.reply_parent_login AS message_reply_parent_login, m.reply_parent_body AS message_reply_parent_body
                FROM incidents i
                LEFT JOIN users u    ON u.twitch_id = i.user_id
                LEFT JOIN messages m ON m.id        = i.message_id
                WHERE i.id = ?
                """,
                (incident_id,),
            )
            r = cur.fetchone()
            return _incident_row_from_row(r) if r else None

    # ============================ App settings (dashboard editable) ========
    # The dashboard's /settings page edits Twitch + StreamElements credentials
    # here. The bot reads these on startup with .env values as fallback. A
    # restart of the bot is required to pick up changes; the dashboard surfaces
    # that prompt after every save.

    # ============================================================
    # APP-SETTINGS CACHE — read-through, write-through, TTL'd
    # ------------------------------------------------------------
    # Background loops (transcript watermark, engaging-subjects
    # blocklist, recap session goals, etc.) read from app_settings
    # once per tick. Without a cache that's one 1-row SELECT per call
    # — fine in isolation, but with 5+ loops at ~1 Hz it adds up.
    #
    # Pattern: lazy-load ALL keys in one SELECT on first access,
    # serve subsequent reads from a process-local dict. Writes hit
    # the DB and update the cache in lockstep ("write-through").
    # 60 s TTL bounds staleness in the bot↔dashboard split-process
    # case where the OTHER process might write a key we care about.
    # Within one process, cache and DB stay in sync exactly.
    # ============================================================

    _APP_SETTINGS_TTL_SECONDS = 60.0

    def _ensure_app_settings_cache(self) -> None:
        """Reload the cache when stale (or empty). Cheap full-scan
        — app_settings is small (~tens of rows) and beats N
        per-key SELECTs on hot read paths."""
        now = time.time()
        if now - self._app_settings_cache_loaded_at < self._APP_SETTINGS_TTL_SECONDS:
            return
        with self._cursor() as cur:
            cur.execute("SELECT key, value FROM app_settings")
            rows = cur.fetchall()
        with self._app_settings_lock:
            self._app_settings_cache = {r["key"]: r["value"] for r in rows}
            self._app_settings_cache_loaded_at = now

    def get_app_setting(self, key: str) -> str | None:
        self._ensure_app_settings_cache()
        with self._app_settings_lock:
            return self._app_settings_cache.get(key)

    def set_app_setting(self, key: str, value: str) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO app_settings(key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, _now_iso()),
            )
        # Write-through: keep the cache in sync with the DB so the
        # next get_app_setting() returns this value without a
        # round-trip.
        with self._app_settings_lock:
            self._app_settings_cache[key] = value

    def delete_app_setting(self, key: str) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM app_settings WHERE key = ?", (key,))
        with self._app_settings_lock:
            self._app_settings_cache.pop(key, None)

    def get_all_app_settings(self) -> dict[str, str]:
        self._ensure_app_settings_cache()
        with self._app_settings_lock:
            return {
                k: v for k, v in self._app_settings_cache.items()
                if v is not None
            }

    # ============================ dataset capture =========================
    # Opt-in personal training-dataset surface. The hot path is one
    # attribute read on `_dataset_dek` and one dict lookup on the cached
    # `app_settings` — when capture is off (the default) these helpers
    # cost nothing. Actual cipher / storage primitives live in
    # `chatterbot.dataset.*` and are lazy-imported there so the optional
    # `cryptography` + `zstandard` deps don't load unless capture is
    # used.

    def dataset_capture_enabled(self) -> bool:
        """Hot-path toggle. Reads the cached app_settings value — single
        dict lookup after the first call per process."""
        v = self.get_app_setting("dataset_capture_enabled")
        return (v or "").lower() == "true"

    def dataset_dek(self) -> bytes | None:
        """In-memory data-encryption key. Returns None when the streamer
        hasn't unlocked. The capture wrapper checks this on every event;
        no lock needed for the read because the reference itself is
        atomic in CPython."""
        return self._dataset_dek

    def set_dataset_dek(self, dek: bytes | None) -> None:
        """Install the unlocked DEK into process memory. Called once at
        bot/dashboard startup after the passphrase is verified, or
        cleared (None) when the streamer toggles capture off."""
        with self._dataset_dek_lock:
            self._dataset_dek = dek

    def insert_dataset_event(
        self,
        *,
        ts: str,
        event_kind: str,
        shard_path: str,
        byte_offset: int,
        byte_length: int,
        schema_version: int,
    ) -> int:
        """Append one row to the dataset_events index. Returns the new
        row id. Caller has already written the ciphertext to the shard
        — this just records the pointer."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO dataset_events
                  (ts, event_kind, shard_path, byte_offset, byte_length, schema_version)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ts, event_kind, shard_path, int(byte_offset), int(byte_length), int(schema_version)),
            )
            return int(cur.lastrowid or 0)

    def iter_dataset_events(
        self,
        *,
        since_ts: str | None = None,
        until_ts: str | None = None,
        kinds: list[str] | None = None,
    ) -> Iterable[dict]:
        """Yield index rows for the export pipeline, oldest-first.
        Filters are optional — `since_ts` / `until_ts` are ISO-UTC
        strings; `kinds` restricts to specific event_kind values."""
        clauses: list[str] = []
        params: list[Any] = []
        if since_ts:
            clauses.append("ts >= ?")
            params.append(since_ts)
        if until_ts:
            clauses.append("ts <= ?")
            params.append(until_ts)
        if kinds:
            placeholders = ",".join("?" * len(kinds))
            clauses.append(f"event_kind IN ({placeholders})")
            params.extend(kinds)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT id, ts, event_kind, shard_path, byte_offset,
                       byte_length, schema_version
                FROM dataset_events
                {where}
                ORDER BY id ASC
                """,
                params,
            )
            for r in cur.fetchall():
                yield {
                    "id": int(r["id"]),
                    "ts": r["ts"],
                    "event_kind": r["event_kind"],
                    "shard_path": r["shard_path"],
                    "byte_offset": int(r["byte_offset"]),
                    "byte_length": int(r["byte_length"]),
                    "schema_version": int(r["schema_version"]),
                }

    def _capture_streamer_action(
        self, *, action_kind: str, item_key: str, action: str,
        note: str | None = None,
    ) -> None:
        """Internal helper: forward to the dataset capture pipeline if
        attached. Keeps the import lazy so the optional `dataset`
        extra doesn't load on bot/dashboard runs without capture
        enabled. Errors swallowed inside `record_streamer_action_safe`
        — calling this from a mutator method must never surface a
        capture error to the caller."""
        try:
            from .dataset.capture import record_streamer_action_safe
            record_streamer_action_safe(
                self,
                kind=action_kind,
                item_key=item_key,
                action=action,
                note=note,
            )
        except Exception:
            # Defense-in-depth — record_streamer_action_safe already
            # swallows, but if importing the module itself fails (e.g.
            # the optional extra is missing), don't break the mutator.
            pass

    def dataset_event_count(self, *, kind: str | None = None) -> int:
        """Total event count, optionally filtered by kind. Used by the
        `chatterbot dataset info` CLI."""
        with self._cursor() as cur:
            if kind:
                cur.execute(
                    "SELECT COUNT(*) AS n FROM dataset_events WHERE event_kind = ?",
                    (kind,),
                )
            else:
                cur.execute("SELECT COUNT(*) AS n FROM dataset_events")
            row = cur.fetchone()
            return int(row["n"]) if row else 0

    # ============================ teardown =================================

    def close(self) -> None:
        # Best-effort: close the current thread's connection. Other
        # threads' thread-local connections will be closed when their
        # threads exit — Python finalizes thread-local storage on
        # thread teardown.
        conn = getattr(self._tl, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._tl.conn = None


def _incident_row_from_row(r) -> IncidentRow:
    try:
        cats = json.loads(r["categories"]) if r["categories"] else []
        if not isinstance(cats, list):
            cats = []
    except (TypeError, ValueError):
        cats = []
    inc = Incident(
        id=int(r["id"]),
        user_id=r["user_id"],
        message_id=int(r["message_id"]) if r["message_id"] is not None else None,
        ts=r["ts"],
        severity=int(r["severity"]),
        categories=[str(c) for c in cats],
        rationale=r["rationale"],
        status=r["status"],
    )
    return IncidentRow(
        incident=inc,
        user_name=r["user_name"],
        message_content=r["message_content"],
        message_reply_parent_login=r["message_reply_parent_login"],
        message_reply_parent_body=r["message_reply_parent_body"],
    )


def _event_from_row(r) -> Event:
    return Event(
        id=int(r["id"]),
        user_id=r["user_id"],
        twitch_name=r["twitch_name"],
        type=r["type"],
        amount=r["amount"],
        currency=r["currency"],
        message=r["message"],
        ts=r["ts"],
    )
