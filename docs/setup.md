# Setup

Everything you need to get chatterbot running on your machine. The tool ships
as a Python package + a docker-compose file; either path works. The bot stays
silent on Twitch by default — it never posts to chat without an opt-in toggle.

## Contents

- [System requirements](#system-requirements)
- [Install](#install)
- [Twitch credentials](#twitch-credentials)
- [LLM provider switch (Ollama / Claude / OpenAI)](#llm-provider-switch)
- [Optional integrations](#optional-integrations)
- [Common pitfalls](#common-pitfalls)

---

## System requirements

| What                | Why                                                |
| ------------------- | -------------------------------------------------- |
| Python 3.11+        | type-hint syntax, async features in stdlib         |
| SQLite 3.40+        | for [`sqlite-vec`](https://github.com/asg017/sqlite-vec) embedding index — usually ships with Python |
| 8 GB RAM            | comfortable headroom for Ollama + dashboard        |
| Ollama (local)      | embeddings always run locally regardless of LLM provider |
| GPU (recommended)   | non-trivial for live transcription with Whisper    |

The bot itself is light — most of the resource cost is the LLM. If you point
chatterbot at a remote Ollama or use Claude/OpenAI for generation, the local
machine only runs embeddings + the dashboard. See
[architecture.md](architecture.md#stack) for what each component costs.

## Install

Two paths.

### uv (recommended)

```bash
git clone https://github.com/mscrnt/chatterbot
cd chatterbot
uv sync                          # base install
uv sync --extra whisper          # if you want streamer-voice transcripts
uv sync --extra dataset          # if you want personal training-dataset capture
uv sync --extra dev              # if you'll be running tests
```

The base install is enough to run the bot + dashboard + TUI against a Twitch
chat. Everything else is opt-in.

### pip

```bash
pip install -e .
pip install -e ".[whisper,dataset,dev]"   # all extras
```

Same shape. Editable install (`-e`) so a `git pull` is enough to update.

### Docker

The repo ships a `docker-compose.yml` that runs bot + dashboard as separate
services sharing the SQLite DB through a volume. See
[architecture.md](architecture.md#processes) for why bot and dashboard are
separate processes.

```bash
docker compose up -d
```

Configuration goes in `.env`; see [Twitch credentials](#twitch-credentials)
below.

## Twitch credentials

Two paths — the quick path uses a public token generator, the recommended
path is your own dev app + a refreshable user token.

### Quick path (no app registration)

1. Log into the Twitch account you want the bot to use (most people make a
   second free account just for the bot).
2. Visit [twitchtokengenerator.com](https://twitchtokengenerator.com), pick
   the "Bot Chat Token" preset, click **Generate**.
3. Copy the access token (starts with `oauth:` — we strip the prefix
   automatically on read).

Drop it into `.env`:

```bash
TWITCH_BOT_NICK=your_bot_nick
TWITCH_OAUTH_TOKEN=oauth:your_token_here
TWITCH_CHANNEL=channel_to_watch
```

This works but the token is long-lived and tied to a third-party site. For
anything beyond local hacking, register your own app (next section).

### Recommended path (your own app)

The bot stores tokens in `data/chatters.db` via the dashboard's settings UI
(see [`config.py`](../src/chatterbot/config.py) for the editable keys list).
A refreshable user token means you don't have to regenerate every two months.

#### 1. Register a developer app

Go to [dev.twitch.tv/console](https://dev.twitch.tv/console) → **Register
Your Application**. Fill in:

- **Name** — anything (e.g. `chatterbot-local`)
- **OAuth Redirect URLs** — `http://localhost`
- **Category** — `Chat Bot`

After registering, you'll get a **Client ID**. Click **New Secret** to
generate a **Client Secret**. Treat both like passwords.

#### 2. Get an OAuth user token

Use any standard OAuth tool — see the chatterbot README's
[Twitch credentials → Recommended path](../README.md) for one workflow.

The scopes you need: `chat:read`, `chat:edit`, `moderator:read:followers`
(if you want follower events), `channel:read:subscriptions` (if you want
sub events), `user:read:chat`, `user:write:chat`. The dashboard works fine
without optional ones — features that need a missing scope quietly skip.

#### 3. Drop credentials into `.env` or settings

`.env`:

```bash
TWITCH_BOT_NICK=your_bot_nick
TWITCH_OAUTH_TOKEN=oauth:user_access_token
TWITCH_CHANNEL=channel_to_watch
TWITCH_CLIENT_ID=your_client_id
TWITCH_CLIENT_SECRET=your_client_secret
```

Or visit `/settings` in the dashboard once it's running — every Twitch
credential field is editable there.

## LLM provider switch

chatterbot supports three LLM backends for generation:

| Provider  | Cost        | Latency       | Setup                          |
| --------- | ----------- | ------------- | ------------------------------ |
| Ollama    | free        | model-dependent | run [Ollama](https://ollama.com) locally or remote |
| Anthropic | API tokens  | ~1-3s         | `ANTHROPIC_API_KEY` env var    |
| OpenAI    | API tokens  | ~1-3s         | `OPENAI_API_KEY` env var       |

**Embeddings always run on local Ollama** regardless of the generation
provider. The `vec_messages` / `vec_threads` indexes are locked to
`nomic-embed-text`'s 768-dim geometry; switching embedding models would
require re-embedding every row. The provider factory at
[`llm/providers.py`](../src/chatterbot/llm/providers.py) wires this — every
non-Ollama client takes an Ollama instance via `embed_via=` and delegates
embeddings to it.

To switch providers, set `LLM_PROVIDER` in `.env` (or via `/settings`):

```bash
LLM_PROVIDER=ollama       # default
# LLM_PROVIDER=anthropic
# LLM_PROVIDER=openai
```

Each provider has its own `*_MODEL` knob:

```bash
OLLAMA_MODEL=qwen3.5:9b           # default
ANTHROPIC_MODEL=claude-opus-4-7   # default when LLM_PROVIDER=anthropic
OPENAI_MODEL=gpt-4o               # default when LLM_PROVIDER=openai
```

See [`config.py`](../src/chatterbot/config.py) for every provider knob.

## Optional integrations

All integrations are opt-in. Each has an `*_ENABLED` toggle in `.env` /
`/settings` that gates everything else.

### StreamElements (tip / sub / cheer / raid / follow events)

```bash
STREAMELEMENTS_ENABLED=true
STREAMELEMENTS_JWT=your_jwt
STREAMELEMENTS_CHANNEL_ID=your_channel_id
```

Get the JWT from [streamelements.com → Account → Show Secrets](https://streamelements.com/dashboard/account/channels).
The channel ID is in the URL of any of your StreamElements pages.

### YouTube live chat

Reads YouTube live chat alongside Twitch and treats both as one big chat
stream.

```bash
YOUTUBE_ENABLED=true
YOUTUBE_API_KEY=your_api_key
YOUTUBE_CHANNEL_ID=your_channel_id
```

The API key needs YouTube Data API v3 access from a Google Cloud project.

### Discord (work in progress)

```bash
DISCORD_ENABLED=true
DISCORD_BOT_TOKEN=your_bot_token
DISCORD_CHANNEL_IDS=comma,separated,ids
```

This is wired but not yet functional — safe to leave off.

### OBS (live state + screenshots)

The dashboard polls OBS via [obs-websocket](https://github.com/obsproject/obs-websocket)
to know when you're live and to grab screenshots for AI context. See
[whisper.md](whisper.md) for what screenshots feed into.

```bash
OBS_ENABLED=true
OBS_HOST=localhost
OBS_PORT=4455
OBS_PASSWORD=your_obs_password
```

## Common pitfalls

- **`chatters.db is locked`** — bot and dashboard share the SQLite file via
  WAL mode, and they should coexist fine. If you see lock errors, check
  whether a stale process is hanging onto the DB
  ([`data/.bot.pid`](../data/.bot.pid) holds the bot's PID — left over from
  a hard kill).
- **Twitch token expired** — Twitch user tokens last 60 days. The bot
  silently keeps trying to reconnect; check `logs/bot.log` for
  `authentication failed` lines.
- **Empty engaging-subjects panel** — chat is too quiet OR your model
  doesn't support `format=` JSON-schema constraints. See
  [insights.md → engaging subjects](insights.md#engaging-subjects).
- **Whisper not transcribing** — install the whisper extra
  (`uv sync --extra whisper`), set `WHISPER_ENABLED=true`, point the OBS
  audio relay at your dashboard. See [whisper.md](whisper.md#audio-relay).
