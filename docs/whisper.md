# Whisper / Streamer voice

Real-time transcription of the streamer's audio (game audio + mic),
plus LLM-summarised "what was happening" windows that pair the
transcripts with OBS screenshots.

Whisper is **opt-in** — install the `whisper` extra
(`uv sync --extra whisper`) and toggle `whisper_enabled` in
`/settings`. Without it, none of this fires; the dashboard works
fine on text alone.

## Contents

- [Why](#why)
- [Pipeline overview](#pipeline-overview)
- [Audio relay](#audio-relay)
- [Transcription](#transcription)
- [Group summaries](#group-summaries)
- [Screenshots](#screenshots)
- [Embedding backfill](#embedding-backfill)
- [LLM-match loop](#llm-match-loop)
- [Tuning](#tuning)

---

## Why

Without transcripts, the LLM-driven panels see chat but not the
streamer's side of the conversation. That's why panels sometimes
spurious-classify "what's chat reacting to": chat is reacting to
something the streamer just said, but the LLM has no record of it.

Whisper closes that loop. Every panel that can take grounding
context (talking points, engaging subjects, open questions, etc.)
prepends a "STREAMER VOICE" block from `recent_transcripts(...)` so
the LLM can disambiguate "alice is talking about a route" vs "alice
is reacting to the streamer talking about a route."

## Pipeline overview

```
OBS audio source ─► obs_scripts/audio_client.py
                    (relays raw PCM via HTTP POST to the dashboard)
                          │
                          ▼
                 dashboard /audio endpoint
                          │
                          ▼
              transcript.py:_TranscriptBuffer
              (~10s rolling window, VAD-filtered)
                          │
                          ▼
              faster-whisper (lazy-loaded model)
                          │
                          ▼
              repo.insert_transcript ─► transcripts table
                          │
                          ▼
              vec_transcripts (background embed loop)
                          │
                          ▼
              transcript_groups (every ~60s, LLM-summarised
              with attached OBS screenshot grid)
```

Implementation in [`transcript.py`](../src/chatterbot/transcript.py).

## Audio relay

The actual audio capture happens in OBS, not in the dashboard. A small
Python script (`obs_scripts/audio_client.py`) runs as an OBS Python
script with a microphone source attached, captures raw PCM frames,
and POSTs them to the dashboard's `/audio` endpoint with an
`X-Captured-At` header so the server knows when the audio was actually
recorded (not when it arrived).

Why this shape:

- Audio capture in OBS uses the same audio sources OBS already
  manages (game audio, mic, browser source, etc.) — no separate
  driver dance.
- HTTP POST is dead simple and survives OBS reload (the script
  reconnects automatically via backoff in the audio_client).
- The streamer can disable transcription in OBS without touching the
  dashboard — turn off the Python script.

The dashboard endpoint accumulates frames into a per-stream rolling
buffer (`whisper_buffer_seconds`, default 10). When the buffer is
full + has voice activity (VAD), the buffer flushes to whisper.

## Transcription

faster-whisper is lazy-loaded on first audio chunk so the dashboard
boots fast and only pays the model-download cost when transcription
actually needs it. The model size is configurable via `whisper_model`
(default `medium.en`).

Each transcript row carries:

- `ts` — when the audio was captured (`X-Captured-At`)
- `text` — what whisper produced
- `seg_start_ms` / `seg_end_ms` — segment offsets within the buffer

Streamer-style speech (fast, emotional, mumbled, yelled) historically
trips whisper. The relevant tuning knobs:

- `whisper_no_speech_threshold` (raise to 0.4 from the 0.6 default)
- `whisper_log_prob_threshold` (lower to -1.5 from the -1.0 default)
- `whisper_vad_threshold` (lower to 0.3 from the 0.5 default)
- `whisper_initial_prompt_enabled` + `whisper_initial_prompt_extra`
  — feeds the top ~80 chat-derived terms into the model's initial
  prompt so jargon-heavy speech transcribes more reliably

See `/settings → Whisper` for every knob, with inline help text
explaining the trade-offs.

## Group summaries

Every ~60s of accumulated transcripts get bundled into a
`transcript_groups` row with an LLM-summarised paragraph describing
what was happening. The summary includes:

- The verbatim utterances (newest first)
- A 2x2 grid of OBS screenshots taken during the same window
- The chat messages the LLM saw at summary time (persisted as
  `context_message_ids` so the modal can re-display the exact
  same chat slice)

The LLM call lives in
[`transcript.py:_run_group_summary`](../src/chatterbot/transcript.py#L1514-L1620).
Customizable via the `transcript.group_summary` prompt entry — see
[prompts.md](prompts.md).

The dashboard's transcript strip on `/insights` shows each group as a
short row; clicking opens a modal with the full utterance list +
screenshot grid + linked chat context.

## Screenshots

OBS screenshot capture runs on a separate background loop
(`transcript_service.screenshot_loop()`). Every
`screenshot_interval_seconds` (default 10s), the dashboard requests
a frame from OBS via `obs-websocket` and saves it to
`data/screenshots/<date>/<ts>.jpg`.

The 2x2 grid stitched into group summaries uses the most-recent N
screenshots (default 4) within the group's time window. Stitching
is done with Pillow in
[`transcript.py:_stitch_grid`](../src/chatterbot/transcript.py#L42-L85)
— output is a
single ~960x540 JPEG that gets base64'd into the LLM's `images=`
parameter.

Vision-incapable LLMs silently ignore `images=`; the screenshot
cost is just the wasted base64 payload, not a hard error. Worth
disabling on text-only models (set `screenshot_interval_seconds=0`).

## Embedding backfill

The `transcripts` table fills as audio arrives, but the
`vec_transcripts` index doesn't auto-populate — there's a separate
`transcript_embed_backfill_loop` that walks unindexed transcripts and
runs them through the local Ollama embedding model in batches.

This powers the `/search → Streamer voice` tab. The backfill loop
runs in the dashboard process (cheap, embedding-only, doesn't need
the full whisper pipeline).

## LLM-match loop

Periodically the dashboard takes the most recent transcript chunks +
the open insight cards (talking points, threads) and asks the LLM
"did the streamer demonstrably engage with any of these cards?"
Matching cards get auto-flipped to `addressed` state with the cited
utterance as the audit-trail note.

This is the `transcript.llm_match` call site in
[`transcript.py`](../src/chatterbot/transcript.py). Streamers who
prefer to manually flag every card as addressed can set
`whisper_llm_match_enabled=false`. The matcher is intentionally
**not** in the streamer-customizable prompts list — it's a
mechanical state-transition matcher; tweaking it via UI risks
silently dropping confirmed engagements.

## Tuning

Streamer-voice transcription quality varies hugely by mic + game
audio mix + vocal style. Some practical knobs:

- **Initial prompt** — `whisper_initial_prompt_enabled=true` pulls
  the top ~80 chat words from the last 24h and injects them as a
  warmup. Drastically helps with jargon-heavy speech.
- **Buffer length** — `whisper_buffer_seconds`. 10s is the default.
  Shorter = faster turnaround, but worse model context. Longer =
  better transcription but slower-to-display.
- **Chat-lag calibration** — `chat_lag_seconds` is how many seconds
  of delay the dashboard assumes between streamer audio and chat
  reactions (so the LLM-match loop pairs them correctly). The
  background `chat_lag_calibration_loop` auto-tunes this every ~10
  min via cross-correlation. See
  [`transcript.py`](../src/chatterbot/transcript.py).

For a deep-dive on the whisper tuning slice, search the git log for
"Whisper tuning for streamer-style speech" — that commit lists the
specific defaults we landed on for fast/emotional/mumbled streamer
speech.
