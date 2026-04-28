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
# TWITCH_CHANNEL, and OLLAMA_HOST. See "Twitch credentials" below.

# 4. run both bot + dashboard in one shell
make all
```

Open the dashboard at <http://127.0.0.1:8765>. Bot logs land in `logs/bot.log`.
Ctrl+C in the dashboard terminal stops both.

> Need separate processes? `make bot` and `make dashboard` (and optionally
> `make tui`) all run independently and share the SQLite DB via WAL.

## Twitch credentials

Two paths — pick one:

### Quick path (no app registration)

If you just want to read chat and get up and running fast, generate a
**Bot Chat Token** at <https://twitchtokengenerator.com> with scope
`chat:read`. Paste the resulting `oauth:…` string into `TWITCH_OAUTH_TOKEN`
and you're done. This works because the generator gives you a token signed
against *its* public Client ID.

Trade-offs:

- The Client ID isn't yours; the generator can revoke it at any time and
  every chatterbot install would have to re-issue.
- No path to add Helix scopes for the moderator-roster sync (VIP / mod /
  sub / follower lookups) — that side of the dashboard stays empty.
- No refresh flow.

If `chat:read` plus event ingestion is enough for you, this is fine. If you
want the full dashboard surface, do the recommended path below.

### Recommended path (your own app)

Register a Twitch developer app to get a stable Client ID and the option to
issue tokens with the moderator scopes the [HelixSyncService][hs] uses.

#### 1. Register a developer app

Sign in at <https://dev.twitch.tv/console/apps> and create a new application
(<https://dev.twitch.tv/docs/authentication/register-app>):

- **Name** — anything, e.g. `chatterbot-<your-handle>`.
- **OAuth Redirect URLs** — `http://localhost` is fine for personal use.
- **Category** — *Chat Bot*.
- **Client Type** — *Confidential* (gives you a Client Secret).

Twitch then gives you a **Client ID** and lets you generate a **Client
Secret**. Keep the secret somewhere private — it's only used if you later
want to refresh tokens automatically (chatterbot itself doesn't refresh
today; long-lived user tokens are the simplest path).

#### 2. Get an OAuth user token

Issue a user-access token against your Client ID with the scopes you need.
The minimum is `chat:read`; the moderator-roster sync ([HelixSyncService][hs])
adds optional scopes that unlock VIP / mod / sub / follower lookups:

| Capability | Scope |
|---|---|
| Read chat (required) | `chat:read` |
| VIP roster sync | `channel:read:vips` |
| Moderator roster sync | `moderation:read` |
| Subscriber roster + tier sync | `channel:read:subscriptions` |
| Follower list + follow dates | `moderator:read:followers` |

The scopes are checked once at startup; endpoints you lack scope for are
silently skipped — chatterbot still runs fine on a `chat:read`-only token.

For a one-time setup, easiest is the **Implicit Grant Flow** described at
<https://dev.twitch.tv/docs/authentication/getting-tokens-oauth/#implicit-grant-flow>:

```
https://id.twitch.tv/oauth2/authorize
  ?client_id=YOUR_CLIENT_ID
  &redirect_uri=http://localhost
  &response_type=token
  &scope=chat:read+channel:read:vips+moderation:read+channel:read:subscriptions+moderator:read:followers
```

Paste that URL in your browser, approve, and copy the `access_token=…`
fragment from the redirect URL.

#### 3. Drop credentials into `.env`

```env
TWITCH_BOT_NICK=your_login_name
TWITCH_OAUTH_TOKEN=oauth:your_access_token_here
TWITCH_CHANNEL=channel_to_listen_to     # can be your own or someone else's
TWITCH_CLIENT_ID=your_client_id          # optional today; reserved for refresh
TWITCH_CLIENT_SECRET=your_client_secret  # optional today; keep private
```

`TWITCH_OAUTH_TOKEN` and `TWITCH_CHANNEL` are independent: a personal token
can read any public channel's chat. The dashboard will log
`token owner=<login>, watching=<channel>` so the relationship is visible.

[hs]: src/chatterbot/helix_sync.py

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
transcript_chunks(id PK, ts, duration_ms, text,                -- whisper utterances
                  matched_kind, matched_item_key, similarity)
vec_transcripts(chunk_id PK, embedding)                        -- cosine vector index
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
8. Real-time whisper transcription (opt-in via `WHISPER_ENABLED=true`). The
   `obs_scripts/audio_client.py` helper captures system audio on the
   streamer's machine, resamples to 16 kHz mono, and POSTs 1 s PCM chunks to
   `/audio/ingest`. The bot buffers `WHISPER_BUFFER_SECONDS` (default 5 s) of
   audio, runs `faster-whisper` with VAD-filtered segmentation
   (`WHISPER_MIN_SILENCE_MS` controls how long a pause must last before
   splitting an utterance — default 5000 ms groups whole thoughts), embeds
   each utterance, and cosine-matches against open insight cards. A match
   above `WHISPER_MATCH_THRESHOLD` (default 0.55) flips the card to
   `auto_pending`; the streamer confirms or rejects, or it auto-promotes to
   `addressed` after 60 s. Each chat message also gets a reverse-lookup
   against recent transcripts so the message-context modal can surface
   "likely a response to what you just said on stream."
9. `opt_out=1` users: no summarization, no new notes. Their watermark still
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
  Moderation · Dashboard UI · Whisper · Diagnostics

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
- **Whisper transcription** — `WHISPER_ENABLED=true` plus the `[whisper]`
  extras (`uv sync --extra whisper`). The bot transcribes via
  `faster-whisper`, embeds each utterance, and auto-marks insight cards as
  `addressed` when you speak about them. First model load downloads
  ~75 MB–1 GB depending on `WHISPER_MODEL` (`tiny.en`/`base.en`/`small.en`/
  `medium.en`). Two ways to feed audio in:

    * **OBS script (recommended).** Add `obs_scripts/chatterbot_audio_relay.py`
      via *Tools → Scripts*. Pick a mic (via `sounddevice` / PortAudio) and,
      on Windows, any output device for WASAPI loopback (via `soundcard`)
      to capture browser / game / system audio without a virtual cable. Both
      sources stream to `/audio/ingest` simultaneously as separate
      subprocesses; the OBS script auto-restarts any that die. Click
      *Refresh devices* once on first run to bootstrap a Windows venv
      (`.venv-win`).
    * **Standalone client.** Run `obs_scripts/audio_client.py` directly
      from a terminal. Same pipeline, single device, useful when OBS isn't
      involved. `--loopback` for WASAPI loopback on Windows.
- **YouTube ingestion** — `YOUTUBE_ENABLED=true` + API key + channel ID.
  Polls `liveChatMessages` and persists chat into `messages` and
  super-chats / new-members / member-milestones into `events`.
  Adaptive backoff (configurable `youtube_min_poll_seconds` /
  `youtube_max_poll_seconds`, defaults 10/30) keeps a 6-hour active
  stream around 7-10K quota units per day — within the free tier.
  Raise the minimum poll interval if you stream past 6 hours, or
  request a quota bump in Google Cloud Console. OBS-aware: skips the
  100-unit `search.list` discovery while OBS reports offline.
- **Discord ingestion** — STUB. Module exists at
  `src/chatterbot/discord_bot.py` with the wiring contract; no gateway
  connection yet.

Cross-platform users land in the same chatters table with
`source='youtube'` / `source='discord'` and namespaced ids
(`yt:UCxxx` / `dc:1234567890`) so collisions with Twitch user IDs are
impossible. The **Merge** button on a chatter's page folds them into
the canonical profile.

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
│   ├── youtube.py                 # YouTube live-chat listener (data-API v3, adaptive polling)
│   ├── discord_bot.py             # Discord listener (STUB)
│   ├── streamelements.py          # SE realtime socket.io listener
│   ├── summarizer.py              # per-user + topics LLM loops
│   ├── threader.py                # topic-snapshot → topic-thread clustering
│   ├── moderator.py               # advisory-only mod classifier (opt-in)
│   ├── insights.py                # talking-points + regulars/lapsed
│   ├── transcript.py              # whisper buffer + VAD + cosine matcher (opt-in)
│   ├── diagnose.py                # `.cbreport` bundle builder
│   ├── tui.py                     # Textual streamer UI
│   ├── llm/ollama_client.py       # Ollama wrapper (think: false)
│   └── web/
│       ├── app.py                 # FastAPI dashboard
│       ├── auth.py                # optional basic auth
│       ├── rag.py                 # per-user "Ask Qwen" RAG
│       ├── templates/             # Jinja2 + HTMX
│       └── static/                # JS (SSE consumer) + CSS
├── obs_scripts/
│   ├── chatterbot_audio_relay.py  # OBS script: coordinates audio_client subprocesses
│   ├── audio_client.py            # standalone audio capture → /audio/ingest
│   └── audio_client.bat           # Windows venv bootstrapper for audio_client.py
├── pyproject.toml
├── package.json                   # optional Tailwind CLI deps
├── tailwind.config.js
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── .env.example
└── README.md
```
