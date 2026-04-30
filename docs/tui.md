# TUI viewer

The Textual-based streamer-only viewer. Same DB as the dashboard, but
runs entirely in your terminal — useful when the dashboard isn't
practical (over SSH, on a secondary machine without a browser, on a
slow link).

## Contents

- [What it shows](#what-it-shows)
- [Run it](#run-it)
- [Keybindings](#keybindings)
- [What it doesn't do](#what-it-doesnt-do)

---

## What it shows

The TUI surfaces a focused subset of what's on the dashboard's
Insights / Chatters tabs:

- Live chat tail with sender + content
- Per-chatter notes browser (jump to any chatter, page through their
  notes + recent messages)
- Per-chatter profile sidebar (pronouns / location / demeanor)
- Manual note authoring (the same `notes` table the dashboard reads
  from — both UIs see new notes immediately)

Implementation in [`tui.py`](../src/chatterbot/tui.py). The Textual
app is structured as one screen per concern (chat / chatters / notes)
with j/k row navigation and slash-prefixed commands.

## Run it

```bash
chatterbot tui
# or:
python -m chatterbot tui
```

The TUI shares the SQLite DB with the bot + dashboard via WAL, so all
three can run simultaneously. The bot writes; the dashboard + TUI
read; manual notes the streamer writes via either UI land in the same
table.

## Keybindings

The TUI uses Textual's standard navigation conventions:

| Key       | Action                                               |
| --------- | ---------------------------------------------------- |
| `j` / `k` | Move down / up in the active list                    |
| `/`       | Focus the search / chatter-jump field                |
| `Enter`   | Open the focused row (chat → user / user → notes)    |
| `Esc`     | Close the active modal / clear focus                 |
| `n`       | New manual note for the focused chatter              |
| `q`       | Quit                                                 |

Full list lives in [`tui.py`](../src/chatterbot/tui.py).

## What it doesn't do

The TUI is intentionally a read-mostly window into the same data the
dashboard surfaces. It does NOT run:

- The LLM-driven panels (talking points, engaging subjects, open
  questions, thread recaps) — those need the dashboard's background
  loops.
- Whisper transcription / OBS polling.
- Settings editing (use `/settings` on the dashboard or edit
  `app_settings` rows directly).
- The personal-dataset capture pipeline.

If you need any of those, run the dashboard. The TUI is for "I want
to see chat + my notes from a tmux pane while I'm doing something
else."
