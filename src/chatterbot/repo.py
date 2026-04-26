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


@dataclass
class Note:
    id: int
    user_id: str
    ts: str
    text: str


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
class TopicSnapshot:
    id: int
    ts: str
    summary: str
    message_id_range: str | None
    topics_json: str | None = None  # parsed lazily by callers if present


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
                    opt_out    INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    TEXT NOT NULL REFERENCES users(twitch_id) ON DELETE CASCADE,
                    ts         TEXT NOT NULL,
                    content    TEXT NOT NULL
                )
                """
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
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS summarization_state (
                    user_id                  TEXT PRIMARY KEY
                                             REFERENCES users(twitch_id) ON DELETE CASCADE,
                    last_summarized_msg_id   INTEGER NOT NULL DEFAULT 0
                )
                """
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

    def upsert_user(self, twitch_id: str, name: str) -> None:
        now = _now_iso()
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO users(twitch_id, name, first_seen, last_seen, opt_out)
                VALUES (?, ?, ?, ?, 0)
                ON CONFLICT(twitch_id) DO UPDATE SET
                    name = excluded.name,
                    last_seen = excluded.last_seen
                """,
                (twitch_id, name, now, now),
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

    def insert_message(self, user_id: str, content: str) -> int:
        """Insert a chat message and return its rowid."""
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO messages(user_id, ts, content) VALUES (?, ?, ?)",
                (user_id, _now_iso(), content),
            )
            return int(cur.lastrowid)

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
                WHERE user_id = ? AND id > ?
                ORDER BY id ASC
                """,
                (user_id, wm),
            )
            return [(int(r["id"]), r["content"]) for r in cur.fetchall()]

    def unsummarized_count(self, user_id: str) -> int:
        wm = self.get_watermark(user_id)
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE user_id = ? AND id > ?",
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
                  AND m.id > COALESCE(s.last_summarized_msg_id, 0)
                GROUP BY m.user_id
                HAVING MAX(m.ts) <= datetime('now', ?)
                """,
                (f"-{int(idle_minutes)} minutes",),
            )
            return [r["user_id"] for r in cur.fetchall()]

    def add_note(self, user_id: str, text: str, embedding: list[float] | None) -> int:
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
            return note_id

    def recent_global_messages(self, limit: int = 30) -> list[Message]:
        """Latest messages across the whole channel, newest first.

        Used by the dashboard's live-chat widget. Opted-out users contribute
        nothing because their messages aren't inserted in the first place.
        """
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT m.id, m.user_id, u.name, m.ts, m.content
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

    def list_opt_out_users(self) -> list[User]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT twitch_id, name, first_seen, last_seen, opt_out "
                "FROM users WHERE opt_out = 1 ORDER BY LOWER(name)"
            )
            return [
                User(
                    twitch_id=r["twitch_id"], name=r["name"],
                    first_seen=r["first_seen"], last_seen=r["last_seen"],
                    opt_out=True,
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
                SELECT u.twitch_id, u.name, u.first_seen, u.last_seen, u.opt_out,
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
                "SELECT twitch_id, name, first_seen, last_seen, opt_out FROM users WHERE twitch_id = ?",
                (twitch_id,),
            )
            r = cur.fetchone()
            if not r:
                return None
            return User(
                twitch_id=r["twitch_id"],
                name=r["name"],
                first_seen=r["first_seen"],
                last_seen=r["last_seen"],
                opt_out=bool(r["opt_out"]),
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
                SELECT m.id, m.user_id, u.name, m.ts, m.content
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
                )
                for r in cur.fetchall()
            ]

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
                SELECT m.id, m.user_id, u.name, m.ts, m.content
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
                )
                for r in cur.fetchall()
            ]

    def find_user_by_alias_or_name(self, name: str) -> User | None:
        """Resolve a username to a user, transparently following renames."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT u.twitch_id, u.name, u.first_seen, u.last_seen, u.opt_out
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
                SELECT m.id, m.user_id, u.name, m.ts, m.content
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
                SELECT m.id, m.user_id, u.name, m.ts, m.content
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
