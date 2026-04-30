# Dashboard

The streamer-only FastAPI dashboard. Everything chatterbot has profiled,
classified, transcribed, or summarised surfaces here. Run it via
`chatterbot dashboard` (or `make dashboard`); the default URL is
`http://localhost:8765`.

## Contents

- [Layout](#layout)
- [Insights](#insights)
- [Chatters](#chatters)
- [Search](#search)
- [Settings](#settings)
- [Live chat dock](#live-chat-dock)
- [Modals](#modals)
- [Optional auth](#optional-auth)

---

## Layout

The dashboard is built with Jinja2 + HTMX + Alpine.js. No JS build step,
no SPA — every page is server-rendered, panels refresh via HTMX
swaps. See [architecture.md → Stack](architecture.md#stack) for the
rationale.

```
┌── nav: Insights · Chatters · Stats · Events · Search · [Moderation] · [Dataset] ──┐
│                                                                                   │
│   <main content>                                                                  │
│                                                                                   │
└── chat dock (always-present footer; collapsible) ─────────────────────────────────┘
```

Top-level routes are wired in
[`web/app.py`](../src/chatterbot/web/app.py). Templates live in
[`web/templates/`](../src/chatterbot/web/templates/) and mostly extend
`base.html`.

## Insights

`/insights` (also `/`) — the main streamer monitor. Multiple panels stack
vertically:

- **Talking points** — one short conversation hook per active chatter.
  See [insights.md → Talking points](insights.md#talking-points).
- **Engaging subjects** — distinct conversation subjects chat is
  discussing right now. Click any card to open a modal with chat
  context + AI talking points + a dismiss button. See
  [insights.md → Engaging subjects](insights.md#engaging-subjects).
- **Questions in chat** — open questions chat is asking. See
  [insights.md → Open questions](insights.md#open-questions).
- **Live conversations / Thread recaps** — observational summaries of
  active topic threads.
- **Talking to you** — `@`-mentions and direct addresses to the streamer.
- **Active but not engaged** — regulars currently in chat the streamer
  hasn't talked to in a week.
- **Newcomers / Regulars / Lapsed** — ambient context about who's around.
- **Recaps** — at-a-glance KPI strip from the last few stream-end recaps.

Each panel has its own background loop (refresh cadence ranges from 3-5
minutes). The dashboard caches the LLM output in memory; templates
re-render when HTMX subscribers receive an SSE event from the
[cross-process bus](architecture.md#cross-process-notification).

Per-card actions — dismiss, addressed, snooze, pin — go through
[`repo.set_insight_state`](../src/chatterbot/repo.py) and write to
`insight_states` + `insight_state_history`. The audit page at
`/audit` shows every transition.

## Chatters

`/chatters` lists every chatter the bot has profiled, sorted by recency
of activity. Click any row to open `/users/<twitch_id>`:

- **Profile** — pronouns, location, demeanor, interests (the soft
  profile fields the LLM extracts).
- **Notes** — every fact the LLM has cited about this chatter, with
  source-message links for provenance.
- **Recent messages** — last N messages, with the streamer's "Talking
  to you" highlights pre-applied.
- **Ask Qwen** — RAG endpoint streaming a one-shot answer about this
  chatter, grounded in their notes + recent messages.
- **Reminders** — set a reminder that fires when this chatter speaks
  again.

The notes pane supports hand-authored notes too (origin marked as
`manual` vs `llm` for provenance). Edit / delete via the per-note
modal.

## Search

`/search` is a multi-tab semantic search:

- **Notes** — search across every chatter's notes via `vec_notes`.
- **Messages** — per-user message embedding via `vec_messages`.
- **Threads** — topic-thread centroids via `vec_threads`.
- **Streamer voice** — Whisper transcript chunks via `vec_transcripts`.
  See [whisper.md → Embedding backfill](whisper.md#embedding-backfill).

Each tab is server-rendered Jinja with HTMX-debounced `keyup` triggers
on the search input — no client-side JS for the search itself.

## Settings

`/settings` is the dashboard's configuration tab. Most knobs are KV
entries in `app_settings` (see
[`config.py:EDITABLE_SETTING_KEYS`](../src/chatterbot/config.py)),
rendered via the metadata in
[`web/settings_meta.py`](../src/chatterbot/web/settings_meta.py).

Tabs:

- **Connections** — Twitch / OBS / StreamElements / YouTube / Discord
- **AI brain** — LLM provider, model overrides, concurrency
- **Whisper** — audio buffer length, chat-lag calibration
- **Insights** — cadence + size knobs for each insights loop
- **Advanced** — internal bus, polling cadence, retention windows
- **Prompts** — streamer-customizable LLM prompts. See
  [prompts.md](prompts.md).
- **Diagnostics** — generates a `.cbreport` bundle for bug reports.

### Diagnostic bundle

`Settings → Diagnostics → Download bundle` produces a privacy-safe zip
that includes:

- Tool version + git SHA + system info
- DB row counts + activity-window timestamps
- App-settings dump (secrets masked by name pattern)
- Insight-state counts
- Personal-dataset capture status (counts only, never decrypts)
- Log tails (last ~1 MB per file)
- Optional: anonymised 24-hour chatter activity (off by default)

See [development.md → Filing bugs](development.md#filing-bugs) for the
full bundle workflow.

## Live chat dock

Every page (except `/live`) has a docked live-chat footer at the bottom
of the viewport. Three states cycled by clicking the chevron:

- **Collapsed** (~30px header bar)
- **Slim** (~33vh body)
- **Expanded** (~50vh body)

Plus drag-to-resize on the handle bar — pulling below ~60px snaps
closed. Toggleable off entirely via `live_widget_enabled` in settings.
Implementation in [`templates/base.html`](../src/chatterbot/web/templates/base.html).

## Modals

Several pages open modals via HTMX into `#modal-root`:

- **Engaging subject modal** — opened from any subject card. Renders
  brief + angles + AI talking points (lazy-loaded) + chat context +
  driver pills. See [insights.md → Subject talking-points modal](insights.md#per-subject-talking-points-modal).
- **Thread modal** — opened from "Live conversations" panel. Shows
  thread members + drivers + verbatim messages, with a "Pull this
  thread back up" button that streams a contextual recap.
- **Note modals** — edit / delete / merge.
- **Transcript group modal** — what the streamer just said (60s window)
  with the chat context the LLM saw at summary time.
- **User merge modal** — manually merge two chatter profiles (alias
  reconciliation).

Each modal is its own template under `templates/modals/`. The shell
component is [`components/_modal.html`](../src/chatterbot/web/templates/components/_modal.html);
modals extend it and override `modal_body`.

## Optional auth

Dashboard binds to `127.0.0.1:8765` by default — local-only. To bind to
a non-loopback interface, also set basic auth so anyone on the network
can't read the dashboard:

```bash
DASHBOARD_HOST=0.0.0.0
DASHBOARD_BASIC_AUTH_ENABLED=true
DASHBOARD_BASIC_AUTH_USER=streamer
DASHBOARD_BASIC_AUTH_PASS=long_random_string
```

Without basic auth, the dashboard logs a warning at startup but still
boots. See [`web/auth.py`](../src/chatterbot/web/auth.py) for the
middleware.

The `/internal/notify` endpoint (for the bot↔dashboard bus) is
auth-gated separately by `internal_notify_secret` regardless of the
dashboard's user-facing auth.
