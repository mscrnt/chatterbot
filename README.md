# chatterbot

[![Buy me a coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=flat&logo=buymeacoffee&logoColor=000)](https://buymeacoffee.com/mscrnt)

Silent chatter-profiling sidecar for streamers. Listens to Twitch chat (and,
optionally via stubs, YouTube and Discord), tracks StreamElements donation
events, OBS live state, and builds lightweight per-chatter profiles via local
LLM summarization. The streamer reads this through a terminal UI **or** a
browser dashboard. The bot itself **never** writes to chat.

## Quickstart

You need Python 3.11+, [`uv`](https://github.com/astral-sh/uv), and a local
[Ollama](https://ollama.com) reachable on the network.

```bash
# 1. clone + install
git clone https://github.com/mscrnt/chatterbot.git
cd chatterbot
uv sync

# 2. pull the Ollama models (on your Ollama host)
ollama pull qwen3.5:9b
ollama pull nomic-embed-text

# 3. configure
cp .env.example .env
# Edit .env: set TWITCH_BOT_NICK, TWITCH_OAUTH_TOKEN (oauth:...),
# TWITCH_CHANNEL, and OLLAMA_HOST. Get a token at
# https://twitchtokengenerator.com (Bot Chat Token, scope chat:read).

# 4. run both bot + dashboard in one shell
make all
```

Open the dashboard at <http://127.0.0.1:8765>. Bot logs land in `logs/bot.log`.
Ctrl+C in the dashboard terminal stops both.

> Need separate processes? `make bot` and `make dashboard` (and optionally
> `make tui`) all run independently and share the SQLite DB via WAL.

## Architectural rule (non-negotiable)

Profile data, event data, message logs, and topic summaries must **never** enter
any LLM prompt that produces a Twitch-chat-facing response. The store is
write-only from the bot's perspective and read-only from the streamer's
surfaces. There is no `!ask`-style command and no chat-side bot persona.

The dashboard's "Ask Qwen about this user" RAG is fine: the answer renders in
the streamer's browser only and never returns to Twitch chat.

If you find yourself adding a code path where notes, events, messages, or topic
summaries get loaded into a prompt that produces a chat message — stop. That's
the rule.

## Stack

- Python 3.11+, [`uv`](https://github.com/astral-sh/uv) for deps
- Twitch chat: `twitchio` 2.x (read-only IRC listener)
- StreamElements realtime via socket.io v3 (`python-socketio`)
- LLM: local Ollama at `192.168.69.229:11434`, `qwen3.5:9b` for summarization,
  `nomic-embed-text` for embeddings (`think: false` always passed)
- Storage: SQLite at `data/chatters.db` with [`sqlite-vec`](https://github.com/asg017/sqlite-vec), WAL mode
- TUI: [Textual](https://textual.textualize.io/)
- Dashboard: FastAPI + Jinja2 + HTMX + Tailwind (server-rendered, no SPA)

## Schema

```
users(twitch_id PK, name, first_seen, last_seen, opt_out,
      sub_tier, sub_months, is_mod, is_vip, is_founder,        -- IRCv3 badge snapshot
      source DEFAULT 'twitch', merged_into NULL)               -- cross-platform identity
messages(id PK, user_id FK, ts, content,
         reply_parent_login, reply_parent_body)                -- Twitch native-reply context
notes(id PK, user_id FK, ts, text, embedding BLOB)             -- LLM-extracted facts
note_sources(note_id FK, message_id FK)                        -- which messages a note cites
events(id PK, user_id FK NULL, twitch_name, type,              -- StreamElements events
       amount, currency, message, ts, raw_json)
topic_snapshots(id PK, ts, summary, message_id_range, topics_json)
topic_threads(id PK, title, category, status, ...)             -- clustered topics across snapshots
topic_thread_members(thread_id, snapshot_id, ...)              -- per-snapshot membership
reminders(id PK, user_id FK, text, created_at, fired_at, dismissed)
incidents(id PK, user_id FK, message_id FK, severity, categories, status)  -- mod mode only
user_aliases(user_id FK, name, first_seen, last_seen_as)       -- rename trail
summarization_state(user_id PK FK, last_summarized_msg_id)
app_settings(key PK, value, updated_at)                        -- dashboard-managed overrides
```

Messages are retained indefinitely. The summarizer advances a per-user
watermark instead of deleting rows. `events.user_id` may be NULL until that
viewer's name appears in chat, at which point the orphan rows are back-filled.

**Cross-platform identity.** The `twitch_id` PK is platform-namespaced for
non-Twitch rows (`yt:UCxxx`, `dc:1234567890`) so collisions across services
are impossible. The streamer can merge any two users from the dashboard;
`merge_users()` rewrites every FK to point at the canonical parent and sets
`merged_into` on the orphan child for "merged from" provenance.

## Pipeline

1. On every chat message: upsert user, snapshot IRCv3 badge state
   (sub tier/months, mod, vip, founder), persist Twitch native-reply context,
   insert into `messages`, fire any pending reminders for that chatter, link
   orphan SE events for their name to the user.
2. Per-user summarizer ships unsummarized messages (since the watermark) to
   Ollama once a user accumulates `SUMMARIZE_AFTER_MESSAGES` (default 20) of
   them, **or** has been idle for `SUMMARIZE_IDLE_MINUTES` (default 10).
   Survivors land in `notes` with embeddings + back-references to the
   specific messages they cite; the watermark advances.
3. Channel-topic summarizer takes a snapshot of "what's chat talking about"
   every `TOPICS_INTERVAL_MINUTES` (default 5) over the last
   `TOPICS_MAX_MESSAGES` (default 200) messages. The threader clusters each
   topic into a long-running `topic_threads` row via embedding cosine
   distance (≤ 0.30) so recurring conversations are surfaced as
   active / dormant / archived buckets.
4. StreamElements realtime listener persists tip / sub / cheer / follow / raid
   events into `events` (full payload retained in `raw_json`).
5. Twitch Helix poller (auto-derives `client_id` from the existing OAuth
   token) refreshes viewer count + stream thumbnail every 60 s for the
   dashboard nav.
6. OBS poller (opt-in via `OBS_ENABLED=true`) reads streaming / recording /
   scene state every ~10 s for a LIVE / REC pill in the nav.
7. Moderation classifier (opt-in via `MOD_MODE_ENABLED=true`) batches recent
   messages through a strict-rubric LLM every `MOD_REVIEW_INTERVAL_MINUTES`
   and persists flagged ones as `incidents` for streamer review. Advisory
   only — the bot never takes chat action.
8. `opt_out=1` users: no summarization, no new notes. Their watermark still
   advances so we don't re-evaluate the same messages forever.

> N=20 / M_idle=10 / M_topics=5 are starting points. Tune via env vars after
> observing real cadence.

## Run model

Three independent processes share the SQLite DB via WAL.

```bash
cp .env.example .env  # fill in TWITCH_*, STREAMELEMENTS_*

uv sync
make bot         # listener + summarizer + SE socket
make tui         # streamer-only Textual viewer
make dashboard   # streamer-only FastAPI dashboard
```

Pull the Ollama models on the host:

```bash
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

### Sharing one Ollama with another app

If you also run [streamlored](https://github.com/mscrnt/streamlored) (or any
second LLM client) against the same Ollama host, set the host's
`OLLAMA_NUM_PARALLEL` to at least `2` so requests from the two apps don't
queue up serially:

```bash
# on the Ollama host (Linux/macOS)
OLLAMA_NUM_PARALLEL=2 ollama serve

# or via systemd: edit /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_NUM_PARALLEL=2"
```

A 9B model + an embeddings model + two-way parallelism is roughly **8–10 GB
VRAM** on a CUDA card, or fits comfortably in 64 GB unified memory on an
M-series Mac.

If you want chatterbot's moderation classifier to ride a smaller / faster
model than note extraction (since moderation runs every few minutes while
notes are rare), set `OLLAMA_MOD_MODEL=qwen3.5:4b` in `.env` and pull that
model on the Ollama host. Note extraction, topics, and the dashboard's
"Ask Qwen" RAG continue to use `OLLAMA_MODEL`.

## TUI

Three tabs: **Chatters** (search + per-user profile + events + recent messages
stacked), **Live Topics**, **Events**. Keybindings:

- `1` / `2` / `3` switch tab
- `r` refresh
- `/` focus search
- `f` forget user (delete every row touching them)
- `o` toggle opt-out
- `d` delete the most-recent note for the selected user
- `q` quit

## Dashboard

Bind `127.0.0.1:8765` by default. Mobile-first responsive layout. The nav
surfaces live status pills (OBS LIVE/REC, Twitch viewer count + stream
thumbnail, fired-reminders bell, newcomers-today counter).

Top-level pages:

- `/` — chatters list with search-as-you-type, sort, paginate, activity
  badges (first-timer / regular / frequent / quiet)
- `/users/{id}` — profile: badges (sub tier/months, mod, VIP, founder, source),
  notes (with citation links to source messages), reminders, donations &
  events log, paginated + searchable message history, "Ask Qwen about this
  user" streaming RAG, merge button, "merged from" provenance list
- `/topics` — bucketed by status (active / dormant / archived); the "Right
  now" snapshot's bullets are clickable per-topic detail modals; threads
  cluster recurring conversations
- `/insights` — community/engagement helper: live talking points for active
  chatters, anniversaries today, newcomers, regulars, lapsed regulars,
  configurable time window
- `/stats` — totals, "did you know" facts, charts (messages/day, hour-of-day,
  top chatters, new-per-week, support breakdown), word cloud, recent
  bucketed stream sessions
- `/live` — full-page live chat feed (also embedded as a floating widget
  globally when `live_widget_enabled` is on); first-timer / returning
  highlights; clicking a row jumps to the chatter with a context modal
  showing 3 messages before/after
- `/reminders` — fired-but-not-dismissed inbox
- `/events` — StreamElements feed, filter by type
- `/moderation` — opt-in incident queue (review / dismiss)
- `/settings` — tabbed: Twitch · OBS · StreamElements · YouTube · Discord ·
  Moderation · Dashboard UI · Diagnostics

Notable HTMX endpoints:

- `GET  /modals/message/{id}` — message-context modal (3 before/after)
- `GET  /modals/merge/{id}` / `POST /users/{id}/merge` — cross-platform merge
- `POST /users/{id}/reminders` — add a reminder; fires on their next message
- `GET  /users/{id}/ask?q=…` — SSE LLM tokens with note + message citations
- `GET  /diagnose` — privacy-safe `.cbreport` bundle for bug reports

### Optional basic auth

Setting `DASHBOARD_BASIC_AUTH_USER` *and* `DASHBOARD_BASIC_AUTH_PASS` enables
HTTP basic auth. Use this when binding to anything other than `127.0.0.1` (e.g.
`0.0.0.0` so you can view the dashboard from a phone on the same LAN).

### Tailwind

`base.html` ships with the Tailwind Play CDN by default — zero build step
needed. For a precompiled stylesheet:

```bash
npm install                  # one-time
make tailwind                # writes src/chatterbot/web/static/css/output.css
```

Then in [base.html](src/chatterbot/web/templates/base.html), replace the
`<script src="https://cdn.tailwindcss.com"></script>` line with:

```html
<link rel="stylesheet" href="/static/css/output.css">
```

## Optional integrations

Off by default. Each is independently toggled in **/settings** or via env.

- **OBS** — `OBS_ENABLED=true` + host/port/password. Read-only WebSocket peek
  at streaming / recording / scene state.
- **StreamElements** — `STREAMELEMENTS_ENABLED=true` + JWT + channel id.
  Pulls tip / sub / cheer / raid / follow events.
- **Twitch Helix viewer count + thumbnail** — automatic when
  `TWITCH_OAUTH_TOKEN` is present. The poller calls
  `https://id.twitch.tv/oauth2/validate` to derive the matching `client_id`,
  so no extra setup needed.
- **Moderation classifier** — `MOD_MODE_ENABLED=true`. Advisory only.
- **YouTube ingestion** — STUB. Module exists at
  `src/chatterbot/youtube.py` with the wiring contract; no API polling yet.
- **Discord ingestion** — STUB. Module exists at
  `src/chatterbot/discord_bot.py` with the wiring contract; no gateway
  connection yet.

The two stubs let you reserve credentials in `/settings` today; when each is
implemented, ingested users are persisted with `source='youtube'` /
`source='discord'` and namespaced ids (`yt:UCxxx` / `dc:1234567890`). The
**Merge** button on a chatter's page folds them into the canonical Twitch
profile.

### Backfilling Twitch badges

`scripts/backfill_badges.py` calls Helix `/channels/vips`,
`/moderation/moderators`, and `/subscriptions` to fill in `is_vip`, `is_mod`,
`sub_tier` for chatters who haven't spoken since you upgraded. Each endpoint
needs its own scope on `TWITCH_OAUTH_TOKEN`
(`channel:read:vips`, `moderation:read`, `channel:read:subscriptions`); the
script gracefully skips any it can't access.

```bash
uv run python scripts/backfill_badges.py
```

## Layout

```
chatterbot/
├── data/                          # SQLite DB lives here
├── scripts/
│   ├── backfill_badges.py         # Helix → users.is_vip / is_mod / sub_tier
│   └── seed_demo.py               # synthetic data for screenshots
├── src/chatterbot/
│   ├── main.py                    # CLI: `chatterbot bot|tui|dashboard`
│   ├── config.py                  # pydantic-settings + DB-overrides layer
│   ├── repo.py                    # ChatterRepo — single SQLite access point
│   ├── bot.py                     # TwitchIO listener (write-only)
│   ├── twitch.py                  # Helix viewer count + thumbnail poller
│   ├── obs.py                     # OBS WebSocket status poller (opt-in)
│   ├── youtube.py                 # YouTube live-chat listener (STUB)
│   ├── discord_bot.py             # Discord listener (STUB)
│   ├── streamelements.py          # SE realtime socket.io listener
│   ├── summarizer.py              # per-user + topics LLM loops
│   ├── threader.py                # topic-snapshot → topic-thread clustering
│   ├── moderator.py               # advisory-only mod classifier (opt-in)
│   ├── insights.py                # talking-points + regulars/lapsed
│   ├── diagnose.py                # `.cbreport` bundle builder
│   ├── tui.py                     # Textual streamer UI
│   ├── llm/ollama_client.py       # Ollama wrapper (think: false)
│   └── web/
│       ├── app.py                 # FastAPI dashboard
│       ├── auth.py                # optional basic auth
│       ├── rag.py                 # per-user "Ask Qwen" RAG
│       ├── templates/             # Jinja2 + HTMX
│       └── static/                # JS (SSE consumer) + CSS
├── pyproject.toml
├── package.json                   # optional Tailwind CLI deps
├── tailwind.config.js
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── .env.example
└── README.md
```
