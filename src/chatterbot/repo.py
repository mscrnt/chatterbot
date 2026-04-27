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
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
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
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    @contextmanager
    def _cursor(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
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
            ):
                if col not in _ucols:
                    cur.execute(f"ALTER TABLE users ADD COLUMN {col} {decl}")
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
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    TEXT NOT NULL REFERENCES users(twitch_id) ON DELETE CASCADE,
                    ts         TEXT NOT NULL,
                    text       TEXT NOT NULL,
                    embedding  BLOB
                )
                """
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
                    embedding BLOB
                )
                """
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
            cur.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_threads USING vec0(
                    thread_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{self.embed_dim}]
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
            # One-time backfill so existing users get their current name as an alias
            # without needing a fresh observation. Idempotent.
            cur.execute(
                """
                INSERT OR IGNORE INTO user_aliases(user_id, name, first_seen, last_seen_as)
                SELECT twitch_id, name, first_seen, last_seen FROM users
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts)")
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
        """Merge LLM-extracted profile signals into the users row. Never
        overwrites a known value with None — the extractor only emits a
        field when it sees a clear signal, so silence on a field means
        "no new information," not "clear it." Interests are union-merged
        with existing entries (case-insensitive dedup) and capped."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT pronouns, location, demeanor, interests "
                "FROM users WHERE twitch_id = ?",
                (twitch_id,),
            )
            row = cur.fetchone()
            if not row:
                return
            new_pronouns = pronouns.strip() if pronouns and pronouns.strip() else row["pronouns"]
            new_location = location.strip() if location and location.strip() else row["location"]
            new_demeanor = demeanor.strip() if demeanor and demeanor.strip() else row["demeanor"]
            existing_interests: list[str] = []
            if row["interests"]:
                try:
                    parsed = json.loads(row["interests"])
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
            cur.execute(
                """
                UPDATE users SET
                    pronouns = ?, location = ?, demeanor = ?,
                    interests = ?, profile_updated_at = ?
                WHERE twitch_id = ?
                """,
                (
                    new_pronouns, new_location, new_demeanor,
                    json.dumps(new_interests) if new_interests else None,
                    _now_iso(),
                    twitch_id,
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
    ) -> int:
        """Insert a chat message and return its rowid. `reply_parent_*` are
        populated when the platform reports the message used a native Reply.
        `is_emote_only` is set when the message text is just emotes +
        whitespace; summarizer and moderator skip those."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages(user_id, ts, content,
                                     reply_parent_login, reply_parent_body,
                                     is_emote_only)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id, _now_iso(), content,
                    reply_parent_login, reply_parent_body,
                    1 if is_emote_only else 0,
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

    def is_opted_out(self, twitch_id: str) -> bool:
        with self._cursor() as cur:
            cur.execute("SELECT opt_out FROM users WHERE twitch_id = ?", (twitch_id,))
            row = cur.fetchone()
            return bool(row and row["opt_out"])

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
                WHERE user_id = ? AND id > ? AND is_emote_only = 0
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
                "WHERE user_id = ? AND id > ? AND is_emote_only = 0",
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
                  AND m.is_emote_only = 0
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
                  AND m.is_emote_only = 0
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
    ) -> int:
        """Insert a note. `source_message_ids` (when supplied) links this note
        back to the specific message ids the LLM cited as supporting it. Pass
        None or [] for manual notes — they show up in the dashboard with no
        provenance link."""
        blob = _vec_to_blob(embedding) if embedding else None
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO notes(user_id, ts, text, embedding) VALUES (?, ?, ?, ?)",
                (user_id, _now_iso(), text, blob),
            )
            note_id = int(cur.lastrowid)
            if blob is not None:
                cur.execute(
                    "INSERT INTO vec_notes(note_id, embedding) VALUES (?, ?)",
                    (note_id, blob),
                )
            if source_message_ids:
                # Filter to ids that actually belong to this user — defends
                # against hallucinated ids slipping through validation.
                placeholders = ",".join("?" for _ in source_message_ids)
                cur.execute(
                    f"SELECT id FROM messages WHERE user_id = ? AND id IN ({placeholders})",
                    (user_id, *source_message_ids),
                )
                valid = [int(r["id"]) for r in cur.fetchall()]
                for mid in valid:
                    cur.execute(
                        "INSERT OR IGNORE INTO note_sources(note_id, message_id) VALUES (?, ?)",
                        (note_id, mid),
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

    # ============================ Events (StreamElements) =================

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
        """Pull the single longest message ever logged. (name, content, len)."""
        with self._cursor() as cur:
            row = cur.execute(
                """
                SELECT u.name, m.content, length(m.content) AS L
                FROM messages m JOIN users u ON u.twitch_id = m.user_id
                ORDER BY L DESC LIMIT 1
                """
            ).fetchone()
            if not row:
                return None
            return (row["name"], row["content"], int(row["L"]))

    def stats_avg_message_length(self) -> float:
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT AVG(length(content)) AS a FROM messages"
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

    # ============================ Insights queries =========================
    # Streamer-only views: who's chatting now, who are the regulars, who's
    # lapsed, who's new today. All read-only / aggregate; no LLM calls here.

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

    def get_surface_ack(self, surface: str) -> str | None:
        """Last-read timestamp for a named dashboard surface (e.g. 'topics',
        'insights'). None means the streamer has never marked it read.

        Used to render a 'NEW' badge on items whose ts is newer than the
        ack — same pattern as `count_first_timers_unacked` but generic."""
        return self.get_app_setting(f"surface_acked_at:{surface}")

    def set_surface_ack(self, surface: str, ts: str | None = None) -> None:
        self.set_app_setting(f"surface_acked_at:{surface}", ts or _now_iso())

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
        order = {
            "last_seen": "u.last_seen DESC",
            "name": "LOWER(u.name) ASC",
            "messages": "msg_count DESC",
            "notes": "note_count DESC",
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
        params.extend([limit, offset])
        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT u.twitch_id, u.name, u.first_seen, u.last_seen, u.opt_out, u.sub_tier, u.sub_months, u.is_mod, u.is_vip, u.is_founder,
                       (SELECT COUNT(*) FROM notes n WHERE n.user_id = u.twitch_id) AS note_count,
                       (SELECT COUNT(*) FROM messages m WHERE m.user_id = u.twitch_id) AS msg_count,
                       (SELECT MAX(ts) FROM messages m WHERE m.user_id = u.twitch_id) AS last_msg_ts
                FROM users u
                {where}
                ORDER BY {order}
                LIMIT ? OFFSET ?
                """,
                params,
            )
            return [
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
                    ),
                    note_count=int(r["note_count"]),
                    msg_count=int(r["msg_count"]),
                    last_message_ts=r["last_msg_ts"],
                )
                for r in cur.fetchall()
            ]

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
                "SELECT twitch_id, name, first_seen, last_seen, opt_out, "
                "sub_tier, sub_months, is_mod, is_vip, is_founder, "
                "source, merged_into, "
                "pronouns, location, demeanor, interests, profile_updated_at "
                "FROM users WHERE twitch_id = ?",
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
            )

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

    def get_notes(self, user_id: str) -> list[Note]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT id, user_id, ts, text FROM notes WHERE user_id = ? ORDER BY ts DESC",
                (user_id,),
            )
            return [
                Note(id=int(r["id"]), user_id=r["user_id"], ts=r["ts"], text=r["text"])
                for r in cur.fetchall()
            ]

    def get_note(self, note_id: int) -> Note | None:
        with self._cursor() as cur:
            cur.execute(
                "SELECT id, user_id, ts, text FROM notes WHERE id = ?", (note_id,)
            )
            r = cur.fetchone()
            if not r:
                return None
            return Note(
                id=int(r["id"]), user_id=r["user_id"], ts=r["ts"], text=r["text"]
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

    def delete_note(self, note_id: int) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM vec_notes WHERE note_id = ?", (note_id,))
            cur.execute("DELETE FROM notes WHERE id = ?", (note_id,))

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

    def get_message_context(
        self, message_id: int, *, before: int = 3, after: int = 3
    ) -> dict[str, Any]:
        """Return the focal message plus N channel-wide messages on each side
        for conversational context. Result:
            {"focal": Message|None,
             "before": list[Message],   # oldest first
             "after":  list[Message]}   # oldest first
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
        return {"focal": focal, "before": before_rows, "after": after_rows}

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
        self, *, limit: int = 100, min_count: int = 3
    ) -> list[tuple[str, int]]:
        """Naive top-words for the Stats tab word cloud. Strips URLs and
        @mentions, lowercases, drops stopwords + chat filler, requires
        3-20 char alphabetic tokens. Skips messages flagged as emote-only
        so 'bawkCrazy' style hype spam doesn't dominate."""
        import re as _re
        from collections import Counter as _Counter
        url_re = _re.compile(r"https?://\S+", _re.IGNORECASE)
        mention_re = _re.compile(r"@\S+")
        word_re = _re.compile(r"\b[a-z]{3,20}\b")
        counter: _Counter[str] = _Counter()
        with self._cursor() as cur:
            cur.execute("SELECT content FROM messages WHERE is_emote_only = 0")
            for r in cur.fetchall():
                text = (r["content"] or "").lower()
                text = url_re.sub("", text)
                text = mention_re.sub("", text)
                for w in word_re.findall(text):
                    if w in self._WORDCLOUD_STOPWORDS:
                        continue
                    counter[w] += 1
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
        emote_clause = "AND m.is_emote_only = 0" if exclude_emote_only else ""
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
                "SELECT COUNT(*) AS c FROM messages WHERE is_emote_only = 0"
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
                  AND m.is_emote_only = 0
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
                WHERE m.id > ? AND m.is_emote_only = 0
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
                WHERE user_id = ? AND is_emote_only = 0
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
                    WHERE m.id < ? AND m.is_emote_only = 0
                    ORDER BY m.id DESC LIMIT ?
                    """,
                    (mid, int(before)),
                )
                seen_ids.update(int(r["id"]) for r in cur.fetchall())
                seen_ids.update(message_ids)
                cur.execute(
                    """
                    SELECT m.id FROM messages m
                    WHERE m.id > ? AND m.is_emote_only = 0
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

    def get_app_setting(self, key: str) -> str | None:
        with self._cursor() as cur:
            cur.execute("SELECT value FROM app_settings WHERE key = ?", (key,))
            row = cur.fetchone()
            return row["value"] if row else None

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

    def delete_app_setting(self, key: str) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM app_settings WHERE key = ?", (key,))

    def get_all_app_settings(self) -> dict[str, str]:
        with self._cursor() as cur:
            cur.execute("SELECT key, value FROM app_settings")
            return {r["key"]: r["value"] for r in cur.fetchall()}

    # ============================ teardown =================================

    def close(self) -> None:
        with self._lock:
            self._conn.close()


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
