# Architecture

This doc covers the high-level shape of chatterbot — the architectural
constraints that drove the design, the stack, the SQLite schema, and the
data pipeline from a chat message arriving to a streamer-facing insight
showing up on the dashboard.

## Contents

- [The hard rule](#the-hard-rule)
- [Processes](#processes)
- [Stack](#stack)
- [Schema](#schema)
- [Pipeline](#pipeline)
- [Cross-process notification](#cross-process-notification)

---

## The hard rule

> **chatterbot never reads its own output back into a chat-facing prompt.**

Notes, profiles, topic snapshots, talking points, engaging subjects — all
streamer-only. They live in `data/chatters.db` and surface on the
streamer's dashboard / TUI. They are never piped to a system that produces
text the streamer's chat will see.

This is structural, not stylistic. There's no "trust" boundary to be
violated by adding a future feature that quotes a stored note in chat —
the systems aren't connected, and the dashboard's RAG endpoints
(`/users/<id>/ask`, the search index, etc.) all live behind dashboard auth,
not in the bot's chat-output path.

The bot's chat-output surface is currently zero. It listens, it never
posts. Even if a future contributor adds chat-output functionality (for a
mod-mode feature or a !command), that path must NOT consume the
profile-knowledge surfaces. See the bigger comment block at the top of
[`repo.py`](../src/chatterbot/repo.py) for the rule's restatement.

## Processes

chatterbot runs as **two cooperating processes** sharing a single SQLite DB
through WAL mode:

1. **`chatterbot bot`** — silent listener. Connects to Twitch (and
   optionally YouTube / Discord / StreamElements), persists every clean
   message, runs background loops for note extraction, profile extraction,
   topic snapshots, mod classification, and transcript ingestion.

2. **`chatterbot dashboard`** — FastAPI web app + insights pipeline. Runs
   the LLM-driven panels (talking points, engaging subjects, open
   questions, thread recaps), the per-subject talking-points modal, the
   audit trail, the search index, and the per-user profile pages.

WAL mode means readers never block writers and vice versa. Both processes
hit the same `data/chatters.db` file; the dashboard's loops produce
in-memory caches keyed off the streamer's last action, the bot's loops
produce DB rows.

```
                   ┌────────────────────┐
   Twitch chat ───►│  chatterbot bot    │── inserts ──► chatters.db (WAL)
   YouTube live ──►│  (silent listener) │                        ▲
   StreamElements ►│                    │                        │
                   └────────────────────┘                        │
                                                                 │
                   ┌────────────────────┐                        │
   Streamer ──────►│  chatterbot        │── reads ───────────────┘
   browser         │  dashboard         │── writes app_settings ─┘
                   │  (FastAPI)         │
                   └────────────────────┘
                            │
                            ▼
                   Streamer's second monitor
```

There's also an optional **`chatterbot tui`** — a Textual streamer-only
viewer for situations where the dashboard isn't practical (over SSH, on a
secondary machine without a browser). Same DB, no LLM-driven panels.

A small **internal-notify HTTP channel** lets the bot push events to the
dashboard with ~10ms latency instead of waiting for a watermark poll. See
[Cross-process notification](#cross-process-notification) below.

## Stack

| Layer       | Choice              | Why                                       |
| ----------- | ------------------- | ----------------------------------------- |
| Language    | Python 3.11+        | type hints, structural pattern matching   |
| Web         | FastAPI             | async-native, structured-output friendly  |
| Templating  | Jinja2 + HTMX + Alpine.js | server-rendered with progressive enhancement; no JS build pipeline |
| Storage     | SQLite + WAL        | one file, multi-process safe, zero ops    |
| Vector idx  | [sqlite-vec](https://github.com/asg017/sqlite-vec) | embeddings live next to relational rows |
| LLM         | Ollama / Anthropic / OpenAI | switchable; embeddings always Ollama |
| Whisper     | faster-whisper      | local, GPU-accelerated, batchable         |
| Twitch      | TwitchIO            | the canonical Python IRC client           |
| YouTube     | manual API polls    | no library; live-chat API is small        |
| Tests       | pytest + pytest-asyncio | mock at the LLMProvider Protocol layer; no HTTP mocking |
| CI          | GitHub Actions      | unit suite runs in ~50s on ubuntu-latest  |

The dashboard JS is intentionally minimal — HTMX + Alpine.js loaded from
CDN, no build step. Tailwind is build-once
(`scripts/tailwind_build.sh` → `web/static/css/output.css`) so the dev
loop is "edit Python, hard-refresh." See the Dashboard section of
[development.md](development.md) for the full dev workflow.

## Schema

Every persistent table lives in `data/chatters.db`. The schema is built
on first connection by [`repo.py:_init_schema`](../src/chatterbot/repo.py#L465-L1117)
and migrated additively (PRAGMA table_info → ALTER TABLE pattern) on
subsequent connections. No migration tool — additive only, no destructive
schema changes after data lands.

### Core tables

| Table                  | What's in it                                                  |
| ---------------------- | ------------------------------------------------------------- |
| `users`                | One row per chatter. `twitch_id` PK; `name` + `opt_out` + soft profile fields (pronouns / location / demeanor / interests). |
| `messages`             | Full chat log, retained indefinitely. `id` (autoincrement) + `user_id` FK + `ts` + `content` + `reply_parent_*` + spam fields. |
| `notes`                | LLM-extracted facts about a chatter. `user_id` FK + `text` + `embedding` blob + `origin` ('manual' / 'llm'). |
| `note_sources`         | Which message_ids the LLM cited as supporting each note (provenance). |
| `events`               | StreamElements tip / sub / cheer / raid / follow events. |
| `topic_snapshots`      | Channel-wide "what's chat talking about" rollups, every ~3 min. |
| `topic_threads`        | Cosine-clustered topic-snapshot groupings; one persistent thread per concept. |
| `app_settings`         | KV store for dashboard-managed config (Twitch creds, integration toggles, prompt customizations). |

### Specialized tables

| Table                       | What's in it                                                  |
| --------------------------- | ------------------------------------------------------------- |
| `incidents`                 | Moderation classifier flags (advisory only — no auto-actions). |
| `insight_states`            | Per-card state on the Insights page (dismissed / addressed / snoozed / pinned). |
| `insight_state_history`     | Append-only audit trail of every state transition.            |
| `transcripts`               | Streamer-voice utterances from the Whisper pipeline.          |
| `transcript_groups`         | LLM-summarised ~60s windows of streamer voice + chat context. |
| `screenshots`               | OBS screenshot metadata; raw JPEGs live on disk under `data/screenshots/`. |
| `reminders`                 | Streamer-set reminders attached to chatters (fire when chatter speaks again). |
| `dataset_events`            | Index pointing at encrypted records in `data/dataset/shards/*.cbds.bin`. |

### Vector indexes (sqlite-vec virtual tables)

| Vector table     | Mirrors                                       |
| ---------------- | --------------------------------------------- |
| `vec_notes`      | `notes.embedding` for note-level RAG          |
| `vec_messages`   | `messages.content` embeddings for per-user search |
| `vec_threads`    | Topic-thread centroids for thread-clustering  |
| `vec_transcripts`| Streamer-voice utterances for `/search` "Streamer voice" tab |

All vector tables use the same 768-dim geometry from
`nomic-embed-text`. Switching embedding models would require
re-embedding every row.

## Pipeline

What happens when a chat message arrives:

```
Twitch IRC ─► bot.py:on_message ─► spam.score_message
                                  │
                                  ▼
                       repo.insert_message (writes row + sets watermark)
                                  │
                                  ▼
                       summarizer.maybe_summarize_user
                       (fire if user has N unsummarized messages)
                                  │
                                  ▼ (async, off the hot path)
                       Note + profile extraction LLM call
                       (think=True, INFORMED_NUM_CTX 32k)
                                  │
                                  ▼
                       repo.add_note + update_user_profile
                                  │
                                  ▼
                       internal-notify HTTP push to dashboard
```

In parallel, the bot's `topics_loop` runs every ~3 min and writes a
channel-wide `topic_snapshots` row. The dashboard's loops
(`refresh_loop`, `engaging_subjects_loop`, `open_questions_loop`,
`thread_recap_loop`, `context_snapshot_loop`, `retention_loop`) read
recent messages + their own caches and produce in-memory results that
the templates render directly.

Heavy LLM calls run with **`think=True` + `num_ctx=32768`** — the
INFORMED_NUM_CTX floor in [`insights.py`](../src/chatterbot/insights.py).
Streamer-facing accuracy beats latency for these calls; cadence (~3 min)
absorbs the cost.

## Cross-process notification

The bot doesn't share memory with the dashboard, so the dashboard would
otherwise have to poll for "is there new chat to render." Instead the
bot fires a tiny HTTP POST at `dashboard_internal_url + /internal/notify`
whenever something changes (new chat, new event, new transcript).

The dashboard's `/internal/notify` route validates an `X-Internal-Secret`
header (set by `internal_notify_secret` in `app_settings`) and emits an
SSE event on a multiplexed channel. The dashboard's templates subscribe
via `hx-trigger="<channel> from:body"` — so a panel that needs to
re-fetch on new chat has a one-line attribute, no manual JS.

When the bus is configured the dashboard updates within ~10ms of a chat
event. With it disabled (empty `dashboard_internal_url`), the dashboard
falls back to a 10s watermark poll. Configuration knobs live in
[`config.py`](../src/chatterbot/config.py) (`dashboard_internal_url` +
`internal_notify_secret`).

See [development.md → Tests](development.md#tests) for how the suite
exercises the bus without standing up two real processes.
