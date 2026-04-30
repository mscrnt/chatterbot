# chatterbot

[![Buy me a coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=flat&logo=buymeacoffee&logoColor=000)](https://buymeacoffee.com/mscrnt)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Tests](https://github.com/mscrnt/chatterbot/actions/workflows/tests.yml/badge.svg)](.github/workflows/tests.yml)

Silent chatter-profiling sidecar for streamers. Listens to Twitch chat
(and optionally YouTube live chat, StreamElements events, Discord) and
builds a streamer-only profile + insights surface. Never posts to chat.
Locally-hosted; no data leaves your machine unless you point chatterbot
at a hosted LLM.

> **Streamer-only output surface.** Everything chatterbot extracts —
> notes, profiles, topic snapshots, talking points — is read on the
> streamer's dashboard. None of it is ever piped back into a system
> that produces text the streamer's chat will see. See
> [the hard rule](docs/architecture.md#the-hard-rule).

---

## Contents

- [Disclaimers](#disclaimers)
- [What it does](#what-it-does)
- [System requirements](#system-requirements)
- [Quickstart](#quickstart)
- [Documentation](#documentation)
- [Stack](#stack)
- [Contributing](#contributing)
- [Filing bugs](#filing-bugs)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Disclaimers

- **Local-first by design.** chatterbot stores everything in a SQLite
  file on your machine (`data/chatters.db`). Switching the LLM
  provider to Anthropic / OpenAI sends prompts to those services per
  their privacy policies; embeddings always run on your local Ollama.
- **Chat-side silence.** The bot reads chat. It does not post,
  whisper, or take Twitch mod actions. The optional moderation
  classifier is **advisory only** — flagged incidents land on your
  dashboard for review; nothing auto-actions.
- **No moderation guarantees.** Even with mod-mode enabled, the
  classifier is best-effort and intended as a second pair of eyes
  for the streamer / human moderators. It will miss things and it
  will false-positive. Don't use it as your primary moderation.
- **Chatter privacy.** Every chatter has an `opt_out` flag respected
  at SQL query time across the codebase. The optional dataset-
  capture system (off by default) honours opt-out at capture time
  too — opted-out chatters never enter the encrypted store. See
  [docs/dataset.md](docs/dataset.md).
- **Encryption boundaries.** The personal training-dataset feature
  encrypts at rest; the `chatters.db` file itself is not encrypted.
  If you sync `data/` to cloud backup, you're moving plaintext chat
  logs with it.

## What it does

- **Profiles every chatter** — extracts hard facts (notes with
  source-message provenance) and soft profile fields (pronouns,
  location, demeanor, interests) via batched LLM calls. All
  streamer-only.
- **Surfaces conversation hooks** — per-chatter talking points, plus
  a separate panel of distinct **engaging subjects** chat is
  discussing right now (with click-through to AI-suggested things to
  say back).
- **Catches open questions** — a two-stage filter pulls the chat
  questions still genuinely waiting on an answer. Drops rhetorical
  / already-answered / directed-at-someone-else asks.
- **Transcribes streamer voice** (opt-in) — local Whisper pipeline
  via OBS audio relay, with periodic LLM-summarised "what was
  happening" windows that include OBS screenshots.
- **Customisable prompts** — every personality-driven LLM panel can
  be tuned with a Factory / Guided / Custom mode picker on
  `/settings → Prompts`.
- **Personal training dataset** (opt-in, encrypted) — captures
  every LLM call + streamer dashboard action, encrypted at rest with
  a streamer-controlled passphrase. Exportable as a `.cbds` bundle
  for fine-tuning later.

## System requirements

| What                | Why                                                |
| ------------------- | -------------------------------------------------- |
| Python 3.11+        | type-hint syntax, async stdlib                     |
| SQLite 3.40+        | for sqlite-vec embedding index (usually shipped)   |
| 8 GB RAM            | comfortable headroom for Ollama + dashboard        |
| Ollama (local)      | embeddings always run locally regardless of LLM provider |
| GPU (recommended)   | non-trivial for live transcription with Whisper    |

The bot itself is light — most of the resource cost is the LLM. See
[docs/setup.md](docs/setup.md) for full install + configuration.

## Quickstart

```bash
# 1. clone + install
git clone https://github.com/mscrnt/chatterbot
cd chatterbot
uv sync

# 2. pull Ollama models (on your Ollama host)
ollama pull qwen3.5:9b
ollama pull nomic-embed-text

# 3. configure
cp .env.example .env
# Edit .env: set TWITCH_BOT_NICK, TWITCH_OAUTH_TOKEN (oauth:...),
# TWITCH_CHANNEL, OLLAMA_HOST.

# 4. run bot + dashboard in one shell
make all
# or run them separately:
chatterbot bot &
chatterbot dashboard
```

Dashboard at <http://localhost:8765>. Twitch credentials are
editable from `/settings` once it's up.

For the deeper install path (own Twitch app, refreshable user token,
optional integrations), see [docs/setup.md](docs/setup.md).

## Documentation

| Doc                                              | Covers                                                   |
| ------------------------------------------------ | -------------------------------------------------------- |
| [docs/setup.md](docs/setup.md)                   | install, Twitch credentials, LLM provider switch, optional integrations |
| [docs/architecture.md](docs/architecture.md)     | the architectural rule, processes, stack, schema, pipeline |
| [docs/dashboard.md](docs/dashboard.md)           | tour of every dashboard tab + modal                      |
| [docs/tui.md](docs/tui.md)                       | the streamer-only Textual viewer                         |
| [docs/whisper.md](docs/whisper.md)               | streamer-voice transcription + group summaries           |
| [docs/insights.md](docs/insights.md)             | the LLM panels: talking points, engaging subjects, etc.  |
| [docs/prompts.md](docs/prompts.md)               | the streamer-customizable prompt system                  |
| [docs/dataset.md](docs/dataset.md)               | opt-in encrypted training-dataset capture                |
| [docs/moderation.md](docs/moderation.md)         | opt-in advisory moderation classifier                    |
| [docs/development.md](docs/development.md)       | dev environment, tests, CI, conventions, contributing    |

## Stack

| Layer       | Choice                                                 |
| ----------- | ------------------------------------------------------ |
| Language    | Python 3.11+                                           |
| Web         | FastAPI                                                |
| Templates   | Jinja2 + HTMX + Alpine.js (no build step)              |
| Storage     | SQLite + WAL + [sqlite-vec](https://github.com/asg017/sqlite-vec) |
| LLM         | Ollama (default), Anthropic, OpenAI                    |
| Transcripts | faster-whisper (opt-in)                                |
| Twitch IRC  | TwitchIO                                               |
| Tests       | pytest + pytest-asyncio (~210 tests, ~50s on CI)       |

## Contributing

PRs welcome. Smaller is better — one focused commit per slice with
its own tests. See
[docs/development.md → Pull requests](docs/development.md#pull-requests)
for the cadence + checklist.

Quick start for contributors:

```bash
git clone https://github.com/mscrnt/chatterbot
cd chatterbot
uv sync --extra dataset --extra dev
uv run pytest
```

A few project conventions worth knowing about:

- **Comments are sparse and explain *why*, not *what*.** Identifier
  names should carry meaning; comments are for hidden constraints
  and non-obvious invariants. See
  [docs/development.md → Code style](docs/development.md#code-style).
- **Migrations are additive only.** No destructive schema changes
  after data lands. The `PRAGMA table_info` + `ALTER TABLE` pattern
  in [`repo.py`](src/chatterbot/repo.py) is the model.
- **Test the wire shape, not the implementation.** Tests assert on
  request/response shapes and cache invariants, not internal
  cluster IDs. See [`tests/`](tests/) for examples.

## Filing bugs

Use the dashboard's diagnostics workflow — no log-spelunking required:

1. `/settings → Diagnostics → Download bundle`
2. (Optional) tick **Anonymize chatter names** if you want to share
   24h activity shape without identities
3. Click **Open new GitHub issue**
4. Drag the `.cbreport` bundle into the issue body

The bundle includes log tails, system info, DB row counts, app
settings (with secrets masked), insight-state shape, and dataset-
capture status. It excludes chat content, secrets, and (by default)
chatter usernames. See
[docs/development.md → Filing bugs](docs/development.md#filing-bugs).

## License

[MIT](LICENSE). Use it however you want, but the warranty's the
warranty (none).

## Acknowledgments

- **[TwitchIO](https://github.com/PythonistaGuild/TwitchIO)** —
  the canonical Python Twitch IRC client. Heart of the listener.
- **[Ollama](https://ollama.com)** — local LLM inference that
  works.
- **[sqlite-vec](https://github.com/asg017/sqlite-vec)** —
  embedding index that lives next to relational rows in the same
  SQLite file. Removes a whole class of operational pain.
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** —
  the only Whisper variant fast enough to run live on a single
  consumer GPU.
- **[FastAPI](https://fastapi.tiangolo.com)** + **[HTMX](https://htmx.org)**
  + **[Alpine.js](https://alpinejs.dev)** + **[Tailwind](https://tailwindcss.com)**
  — the dashboard stack. No build step, no SPA, no JS framework
  treadmill.

If you ship a meaningful improvement, send a PR; if you ship
something built on top, drop a star. Thanks.
