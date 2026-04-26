# chatterbot

[![Buy me a coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=flat&logo=buymeacoffee&logoColor=000)](https://buymeacoffee.com/mscrnt)

Silent Twitch chatter-profiling sidecar. Listens to chat, tracks StreamElements
events, and builds lightweight per-chatter profiles via local LLM
summarization. The streamer reads this through a terminal UI **or** a browser
dashboard. The bot itself **never** writes to chat.

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
users(twitch_id PK, name, first_seen, last_seen, opt_out BOOL DEFAULT 0)
messages(id PK, user_id FK, ts, content)                  -- full chat log, retained
notes(id PK, user_id FK, ts, text, embedding BLOB)        -- LLM-extracted facts
events(id PK, user_id FK NULL, twitch_name, type,         -- StreamElements events
       amount, currency, message, ts, raw_json)
topic_snapshots(id PK, ts, summary, message_id_range)     -- channel topic rollups
summarization_state(user_id PK FK, last_summarized_msg_id)
```

Messages are retained indefinitely. The summarizer advances a per-user
watermark instead of deleting rows. `events.user_id` may be NULL until that
viewer's name appears in chat, at which point the orphan rows are back-filled.

## Pipeline

1. On every chat message: upsert user, insert into `messages`. Orphan SE events
   for that name are linked to the user.
2. Per-user summarizer ships unsummarized messages (since the watermark) to
   Ollama once a user accumulates `SUMMARIZE_AFTER_MESSAGES` (default 20) of
   them, **or** has been idle for `SUMMARIZE_IDLE_MINUTES` (default 10).
   Survivors land in `notes` with embeddings; the watermark advances.
3. Channel-topic summarizer takes a snapshot of "what's chat talking about"
   every `TOPICS_INTERVAL_MINUTES` (default 5) over the last
   `TOPICS_MAX_MESSAGES` (default 200) messages.
4. StreamElements realtime listener persists tip / sub / cheer / follow / raid
   events into `events` (full payload retained in `raw_json`).
5. `opt_out=1` users: no summarization, no new notes. Their watermark still
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

Bind `127.0.0.1:8765` by default. Mobile-first responsive layout.

Routes:

- `GET /` — chatters list (search-as-you-type, sort, paginate)
- `GET /users/{twitch_id}` — profile (notes, donations & events totals,
  paginated message history) + "Ask Qwen about this user" RAG
- `GET /users/{twitch_id}/ask?q=…` — SSE stream of LLM tokens with citations
- `POST /users/{twitch_id}/forget` — delete every row touching the user
- `POST /users/{twitch_id}/opt-out` — toggle
- `PATCH /notes/{id}` / `DELETE /notes/{id}` — edit / delete notes inline
- `GET /topics` — auto-refreshing topic snapshots (`hx-trigger="every 30s"`)
- `GET /events` — events feed, filter by type

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

## Layout

```
chatterbot/
├── data/                          # SQLite DB lives here
├── src/chatterbot/
│   ├── main.py                    # CLI: `chatterbot bot|tui|dashboard`
│   ├── config.py                  # pydantic-settings
│   ├── repo.py                    # ChatterRepo — single SQLite access point
│   ├── bot.py                     # TwitchIO listener (write-only)
│   ├── streamelements.py          # SE realtime socket.io listener
│   ├── summarizer.py              # per-user + topics LLM loops
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
