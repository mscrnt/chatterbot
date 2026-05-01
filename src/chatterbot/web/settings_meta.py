"""Settings page metadata — labels, tooltips, help text, input types,
sections, and conditional dependencies for every editable setting.

The settings page renders strictly from this metadata. Adding a new
editable setting means:
  1. Add the key to `EDITABLE_SETTING_KEYS` in config.py
  2. Add a Field entry here with label / help / tooltip / type
  3. Reference the key in a Section's `fields` list

Help text is written for a streamer who isn't a software engineer.
Avoid jargon. Lead with what the knob does for the streamer's
experience, not the implementation. Mention the default value at the
end of the help text so they can revert by eye.

`depends_on` lets a field hide itself unless another field has a
particular value — used for provider-specific API keys and for
hiding integration fields when their `*_enabled` toggle is off.
"""

from __future__ import annotations

from typing import Any, Literal


# ---------- Field metadata ----------
# Each key in EDITABLE_SETTING_KEYS appears here with full metadata.
# Format:
#   "key": {
#       "label": str,       # short human-readable label
#       "tooltip": str,     # 1-sentence hover hint
#       "help": str,        # 1-3 sentence plain-English explanation
#       "type": str,        # "text" | "number" | "bool" | "select" | "secret" | "textarea"
#       "options": list,    # for type=select
#       "min": number,      # for type=number
#       "max": number,
#       "step": number,
#       "suffix": str,      # unit label rendered to the right of the input
#       "placeholder": str,
#       "depends_on": tuple,# (other_key, value) — show only when other_key == value
#       "advanced": bool,   # hide under "Show advanced" disclosure within sub-card
#   }


FIELDS: dict[str, dict[str, Any]] = {
    # ============================================================
    # CONNECTIONS → TWITCH
    # ============================================================
    "twitch_channel": {
        "label": "Channel to watch",
        "tooltip": "The Twitch channel chatterbot listens to.",
        "help": "Type the streamer's Twitch username (no @, no URL — just the name). This is the channel whose chat the bot reads. Usually your own username.",
        "type": "text",
        "placeholder": "yourchannelname",
    },
    "twitch_bot_nick": {
        "label": "Bot username",
        "tooltip": "The Twitch account the bot logs in as.",
        "help": "The Twitch username the bot itself uses. Most people make a second free Twitch account just for the bot, but you can use your main one. The bot doesn't post in chat by default — this just identifies the connection.",
        "type": "text",
        "placeholder": "mybotname",
    },
    "twitch_oauth_token": {
        "label": "Bot OAuth token",
        "tooltip": "Login token for the bot account.",
        "help": "Paste an OAuth token here so the bot can connect. The easy way: log into the bot account on Twitch in your browser, visit twitchtokengenerator.com, copy the token. Starts with 'oauth:' (we strip that automatically). Treat it like a password — never share it.",
        "type": "secret",
    },
    "twitch_client_id": {
        "label": "Client ID",
        "tooltip": "Your Twitch developer app's Client ID.",
        "help": "Twitch needs this so chatterbot can ask their API about the channel (current game, viewer count, tags, etc). Get it at dev.twitch.tv → Console → Register Your Application. Free, takes 2 minutes. Without this you lose 'currently playing' info on the dashboard.",
        "type": "text",
    },
    "twitch_client_secret": {
        "label": "Client Secret",
        "tooltip": "Your Twitch developer app's Client Secret.",
        "help": "Pairs with the Client ID above. Same place: dev.twitch.tv → Console → your app → Manage → New Secret. Treat it like a password.",
        "type": "secret",
    },

    # ============================================================
    # CONNECTIONS → OBS
    # ============================================================
    "obs_enabled": {
        "label": "Connect to OBS",
        "tooltip": "Lets chatterbot peek at OBS for 'is the stream live?' and 'what scene is up?'.",
        "help": "When on, chatterbot reads (read-only) from OBS WebSocket — knows when you go live, what scene you're on, and can grab screenshots. Speeds up the dashboard's 'are we streaming?' detection and powers the 'screenshot grid' that ships alongside transcript summaries. Off by default.",
        "type": "bool",
    },
    "obs_host": {
        "label": "OBS host",
        "tooltip": "Where OBS WebSocket is running. Usually 'localhost'.",
        "help": "The IP or hostname of the machine running OBS. Almost always 'localhost' (same machine as chatterbot). Change only if OBS is on a different computer on your LAN.",
        "type": "text",
        "placeholder": "localhost",
        "depends_on": ("obs_enabled", True),
    },
    "obs_port": {
        "label": "OBS port",
        "tooltip": "OBS WebSocket port. Default 4455.",
        "help": "OBS WebSocket's port. Default is 4455 — only change if you set a custom port in OBS → Tools → WebSocket Server Settings.",
        "type": "number",
        "min": 1, "max": 65535, "step": 1,
        "placeholder": "4455",
        "depends_on": ("obs_enabled", True),
    },
    "obs_password": {
        "label": "OBS password",
        "tooltip": "Password set in OBS WebSocket Settings. Leave blank if no password.",
        "help": "If you enabled authentication in OBS WebSocket Settings, paste the password here. Leave blank if your OBS WebSocket is unauthenticated (fine for localhost). Treat like a password.",
        "type": "secret",
        "depends_on": ("obs_enabled", True),
    },

    # ============================================================
    # CONNECTIONS → STREAMELEMENTS
    # ============================================================
    "streamelements_enabled": {
        "label": "Connect to StreamElements",
        "tooltip": "Pulls tip / sub / cheer / raid / follow events.",
        "help": "When on, chatterbot fetches your channel's events from StreamElements (donations, subs, raids, etc) and shows them on the dashboard. Off by default — only turn on if you use StreamElements for events.",
        "type": "bool",
    },
    "streamelements_jwt": {
        "label": "StreamElements JWT",
        "tooltip": "Account JWT token from streamelements.com.",
        "help": "Get this from StreamElements → your dashboard → Account Settings → Show secrets → JWT Token. Long string. Treat like a password.",
        "type": "secret",
        "depends_on": ("streamelements_enabled", True),
    },
    "streamelements_channel_id": {
        "label": "Channel ID",
        "tooltip": "Your StreamElements channel ID (not your name).",
        "help": "A long alphanumeric ID — NOT your channel name. Find it in the same StreamElements settings page, or in the URL of your activity feed page.",
        "type": "text",
        "depends_on": ("streamelements_enabled", True),
    },

    # ============================================================
    # CONNECTIONS → YOUTUBE
    # ============================================================
    "youtube_enabled": {
        "label": "Read YouTube live chat",
        "tooltip": "Pulls live-chat messages from a YouTube stream.",
        "help": "When on, chatterbot reads live chat messages from your YouTube live stream alongside Twitch. Treats both as one big chat for notes, recaps, and engaging-subjects. Off by default.",
        "type": "bool",
    },
    "youtube_api_key": {
        "label": "YouTube API key",
        "tooltip": "Read-only API key from Google Cloud Console.",
        "help": "Free API key from console.cloud.google.com → API & Services → Credentials → Create API Key. Restrict it to 'YouTube Data API v3' for safety. Treat like a password.",
        "type": "secret",
        "depends_on": ("youtube_enabled", True),
    },
    "youtube_channel_id": {
        "label": "YouTube channel ID",
        "tooltip": "Your YouTube channel ID (starts with UC...).",
        "help": "Looks like 'UCxxxxxxxxxxxxxxxxxxxxxx'. Find it at youtube.com/account_advanced or in the channel page URL.",
        "type": "text",
        "depends_on": ("youtube_enabled", True),
    },

    # ============================================================
    # CONNECTIONS → DISCORD
    # ============================================================
    "discord_enabled": {
        "label": "Connect to Discord (WIP)",
        "tooltip": "Stub — listener wired but no gateway connection yet.",
        "help": "Placeholder for a future Discord-bridge feature. Turning this on doesn't do anything useful yet — the gateway connection isn't implemented. Safe to leave off.",
        "type": "bool",
    },
    "discord_bot_token": {
        "label": "Discord bot token",
        "tooltip": "Bot token from discord.com/developers/applications.",
        "help": "Discord bot token. Treat like a password. Won't actually be used until the gateway listener ships.",
        "type": "secret",
        "depends_on": ("discord_enabled", True),
    },
    "discord_channel_ids": {
        "label": "Channel IDs",
        "tooltip": "Comma-separated Discord channel IDs to watch.",
        "help": "Which Discord channels the bot should listen in. Comma-separated numeric IDs. Right-click a channel in Discord → Copy Channel ID (you need Developer Mode on).",
        "type": "text",
        "placeholder": "123456789012345678, 234567890123456789",
        "depends_on": ("discord_enabled", True),
    },

    # ============================================================
    # AI BRAIN
    # ============================================================
    "llm_provider": {
        "label": "AI provider",
        "tooltip": "Which LLM service handles thinking-heavy calls.",
        "help": "Picks which AI handles things like notes, recaps, and chat-subject extraction. Ollama runs locally on your machine (free, slower without a GPU). Anthropic = Claude (paid API, very capable). OpenAI = GPT (paid API). Embeddings ALWAYS run on local Ollama no matter what — that's locked. Restart the dashboard after changing this.",
        "type": "select",
        "options": ["ollama", "anthropic", "openai"],
    },
    "anthropic_api_key": {
        "label": "Claude API key",
        "tooltip": "Anthropic API key from console.anthropic.com.",
        "help": "Get this from console.anthropic.com → API Keys. Required when 'AI provider' is set to anthropic. Charged per token used. Treat like a password.",
        "type": "secret",
        "depends_on": ("llm_provider", "anthropic"),
    },
    "anthropic_model": {
        "label": "Claude model",
        "tooltip": "Which Claude variant to use.",
        "help": "The model name Anthropic should run for chatterbot's calls. 'claude-opus-4-7' is the most capable; smaller variants are cheaper and faster. Check console.anthropic.com for the current list.",
        "type": "text",
        "placeholder": "claude-opus-4-7",
        "depends_on": ("llm_provider", "anthropic"),
    },
    "openai_api_key": {
        "label": "OpenAI API key",
        "tooltip": "OpenAI API key from platform.openai.com.",
        "help": "Get this from platform.openai.com → API Keys. Required when 'AI provider' is set to openai. Charged per token used. Treat like a password.",
        "type": "secret",
        "depends_on": ("llm_provider", "openai"),
    },
    "openai_model": {
        "label": "OpenAI model",
        "tooltip": "Default OpenAI model.",
        "help": "Which OpenAI model to use for normal calls. 'gpt-4o' is the standard default. Check platform.openai.com for the current list.",
        "type": "text",
        "placeholder": "gpt-4o",
        "depends_on": ("llm_provider", "openai"),
    },

    # ============================================================
    # VOICE → AUDIO BASICS
    # ============================================================
    "whisper_enabled": {
        "label": "Transcribe my voice",
        "tooltip": "Turns on real-time stream transcription via OBS audio.",
        "help": "When on, the OBS audio relay script feeds your mic into a local Whisper model that turns it into text. Powers the Stream timeline on /insights and lets the dashboard auto-check off chat cards when you address them out loud. Off by default — needs the OBS script set up.",
        "type": "bool",
    },
    "whisper_model": {
        "label": "Whisper model size",
        "tooltip": "Bigger = more accurate, slower, more disk + RAM.",
        "help": "Which Whisper model size to download and use. 'tiny.en' is fastest (~75MB, fine for clean audio). 'base.en' is the sweet spot for most streams. 'medium.en' is the most accurate but ~1.5GB and needs a decent CPU/GPU. Downloaded on first use.",
        "type": "select",
        "options": ["tiny.en", "base.en", "small.en", "medium.en"],
        "depends_on": ("whisper_enabled", True),
    },
    "whisper_buffer_seconds": {
        "label": "Audio buffer length",
        "tooltip": "How much audio to accumulate before transcribing.",
        "help": "Whisper transcribes in chunks. Longer buffer = better context (more accurate) but slower to appear on the dashboard. 5 seconds is the sweet spot. Drop to 2-3 for snappier feedback; raise to 8-10 for noisy audio.",
        "type": "number",
        "min": 1, "max": 30, "step": 0.5,
        "suffix": "seconds",
        "depends_on": ("whisper_enabled", True),
    },
    "whisper_min_silence_ms": {
        "label": "Silence between sentences",
        "tooltip": "Pause length that ends one transcript chunk and starts the next.",
        "help": "How long a silence must be (in milliseconds) before Whisper splits into a new utterance. Lower = more fragmented sentences; higher = full thoughts grouped together. Default 5000ms (5s) groups whole thoughts. Drop to 500-1000 if you want tighter per-clause splits.",
        "type": "number",
        "min": 100, "max": 10000, "step": 100,
        "suffix": "ms",
        "depends_on": ("whisper_enabled", True),
    },

    # ============================================================
    # VOICE → ACCURACY TUNING (decoder + vocabulary bias)
    # ============================================================
    "whisper_beam_size": {
        "label": "Decoder beam size",
        "tooltip": "How many alternative transcriptions Whisper considers (1 = greedy, 5 = thorough).",
        "help": "Higher = more accurate on hard audio (yelling, mumbling, stammering, fast speech) because Whisper can recover from a wrong first guess. Lower = faster. Default 3 is the sweet spot for streamer-style speech; bump to 5 if you stream very emotional content and have GPU headroom.",
        "type": "number",
        "min": 1, "max": 10, "step": 1,
        "suffix": "1-10",
        "depends_on": ("whisper_enabled", True),
    },
    "whisper_no_speech_threshold": {
        "label": "Silence detection strictness",
        "tooltip": "How sure Whisper has to be that audio is silent before dropping it (0-1).",
        "help": "Whisper's default of 0.6 sometimes misclassifies sustained yelling as silence and drops it. Lowered to 0.4 here so emotional moments stay in the transcript. Raise toward 0.6 if you see Whisper hallucinating text on truly quiet passages.",
        "type": "number",
        "min": 0, "max": 1, "step": 0.05,
        "suffix": "0-1",
        "depends_on": ("whisper_enabled", True),
    },
    "whisper_log_prob_threshold": {
        "label": "Quality fallback threshold",
        "tooltip": "Below this avg log-probability, Whisper retries with higher randomness.",
        "help": "Whisper's default of -1.0 triggers a retry-with-randomness fallback on any low-confidence segment. Emotional / distorted audio has worse log-prob naturally, so the default fires too often and produces 'you you you' style hallucinations. -1.5 here keeps the fallback for genuinely garbled audio only.",
        "type": "number",
        "min": -3.0, "max": 0.0, "step": 0.1,
        "suffix": "(negative)",
        "depends_on": ("whisper_enabled", True),
    },
    "whisper_vad_threshold": {
        "label": "VAD onset sensitivity",
        "tooltip": "How loud audio has to be before VAD flags it as speech (0-1, lower = more sensitive).",
        "help": "Voice-activity-detection threshold. Whisper's default of 0.5 sometimes skips the first word of a yelled clause because of breath noise on the onset. 0.3 catches those moments. Raise toward 0.5 if VAD is firing on background music / game audio.",
        "type": "number",
        "min": 0.1, "max": 0.9, "step": 0.05,
        "suffix": "0-1",
        "depends_on": ("whisper_enabled", True),
    },
    "whisper_initial_prompt_enabled": {
        "label": "Bias Whisper with stream vocabulary",
        "tooltip": "Auto-build a vocabulary hint from the streamer's name, current game, active chatters, top chat words, and streamer_facts.md.",
        "help": "Whisper accepts an 'initial prompt' it treats as vocabulary bias — words present in the prompt are far more likely to be transcribed correctly. When on, the dashboard auto-builds this from runtime context every ~30s: streamer name, current game, active chatters' handles, the top words from chat in the last week (same source as the wordcloud), and streamer_facts.md contents. Massive accuracy boost on niche game terms, character names, and chatter handles. Recommended on. Free at runtime.",
        "type": "bool",
        "depends_on": ("whisper_enabled", True),
    },
    "whisper_initial_prompt_extra": {
        "label": "Additional vocabulary hint",
        "tooltip": "Free-form text appended to the auto-built prompt — list any specific terms you want Whisper to recognise.",
        "help": "Optional. Comma-separated terms or short sentences appended to the auto-built initial prompt. Use for streamer-specific lingo not captured in streamer_facts.md or the live channel context. Keep brief — anything beyond ~80 names/terms gets ignored by Whisper anyway.",
        "type": "text",
        "placeholder": "e.g. 'Ratones, GIGACAEDREL, Caps, Hans Sama'",
        "depends_on": ("whisper_initial_prompt_enabled", True),
    },

    # ============================================================
    # VOICE → CARD MATCHING
    # ============================================================
    "whisper_match_threshold": {
        "label": "Match strictness",
        "tooltip": "How sure Whisper has to be before auto-checking a chat card (0-1).",
        "help": "When you talk during stream, the dashboard compares what you said against open chat cards. This setting is how confident the match has to be before auto-checking the card off. Higher = stricter (fewer false matches but you might miss real ones). Lower = looser. Default 0.55.",
        "type": "number",
        "min": 0, "max": 1, "step": 0.05,
        "suffix": "0-1",
        "depends_on": ("whisper_enabled", True),
    },
    "whisper_unnamed_match_threshold": {
        "label": "Strictness when no name was said",
        "tooltip": "Higher bar when you didn't say the chatter's name.",
        "help": "Naming the chatter (\"that's right Aquanote, …\") is the strongest signal you're addressing them. When you DON'T name them, we use this stricter threshold to avoid mistakenly checking off vague utterances. Default 0.80.",
        "type": "number",
        "min": 0, "max": 1, "step": 0.05,
        "suffix": "0-1",
        "depends_on": ("whisper_enabled", True),
    },
    "whisper_auto_confirm_seconds": {
        "label": "Auto-confirm timer",
        "tooltip": "How long an auto-pending card waits before becoming 'addressed' on its own.",
        "help": "When Whisper auto-pends a card (thinks you addressed it), the card waits this long for you to confirm or reject. If you don't, it self-promotes to 'addressed'. Default 300s (5 min). Lower = less time to override; higher = more review window.",
        "type": "number",
        "min": 30, "max": 1800, "step": 10,
        "suffix": "seconds",
        "depends_on": ("whisper_enabled", True),
    },
    "whisper_llm_match_enabled": {
        "label": "Use AI for smarter matching",
        "tooltip": "AI re-checks transcript windows against cards in batches.",
        "help": "On top of plain similarity matching, the dashboard can ask the AI 'did the streamer actually engage with any of these cards?' in batches. AI is better at context (knows most utterances are game reactions, not chat-directed). When on, this becomes the PRIMARY auto-pending mechanism. Recommended on.",
        "type": "bool",
        "depends_on": ("whisper_enabled", True),
    },
    "whisper_llm_match_interval_seconds": {
        "label": "AI match interval",
        "tooltip": "How often the AI re-checks the transcript window.",
        "help": "How often (in seconds) the AI matcher runs. Each pass looks at every transcript chunk added since the last pass. Higher = saves AI throughput; lower = tighter feedback on the dashboard. Default 90s.",
        "type": "number",
        "min": 30, "max": 600, "step": 10,
        "suffix": "seconds",
        "depends_on": ("whisper_llm_match_enabled", True),
    },
    "whisper_llm_match_min_chunks": {
        "label": "Min chunks before AI runs",
        "tooltip": "Don't run the AI matcher on tiny windows.",
        "help": "Skip the AI matcher when fewer than this many transcript chunks have arrived since last pass. Avoids burning AI calls on a single stray utterance. Default 3.",
        "type": "number",
        "min": 1, "max": 20, "step": 1,
        "suffix": "chunks",
        "depends_on": ("whisper_llm_match_enabled", True),
    },
    "whisper_llm_match_confidence": {
        "label": "AI match confidence floor",
        "tooltip": "Minimum confidence the AI must report to auto-pend a card (0-1).",
        "help": "The AI emits a 0-1 confidence per match. Below this floor we log it but don't auto-pend. Default 0.65 — lower for more permissive auto-pending, higher to require near-certainty.",
        "type": "number",
        "min": 0, "max": 1, "step": 0.05,
        "suffix": "0-1",
        "depends_on": ("whisper_llm_match_enabled", True),
    },

    # ============================================================
    # VOICE → GROUP SUMMARIES
    # ============================================================
    "whisper_group_interval_seconds": {
        "label": "Summary interval",
        "tooltip": "How often the Stream timeline rolls up new utterances into one summarised moment.",
        "help": "How often (in seconds) the dashboard summarises the latest transcript chunks into a single observational moment on the Stream timeline. Default 60s. Lower = more granular but more AI calls; higher = fewer rolled-up moments.",
        "type": "number",
        "min": 15, "max": 600, "step": 5,
        "suffix": "seconds",
        "depends_on": ("whisper_enabled", True),
    },
    "whisper_group_min_chunks": {
        "label": "Min chunks for a summary",
        "tooltip": "Skip summarising tiny windows.",
        "help": "Don't bother summarising windows with fewer than this many transcript chunks. Avoids 'one-word reaction → an entire summary line' noise. Default 2.",
        "type": "number",
        "min": 1, "max": 20, "step": 1,
        "suffix": "chunks",
        "depends_on": ("whisper_enabled", True),
    },

    # ============================================================
    # VOICE → SCREENSHOTS (paired with summaries)
    # ============================================================
    "screenshot_interval_seconds": {
        "label": "Screenshot every",
        "tooltip": "How often to grab an OBS screenshot. 0 disables.",
        "help": "Whisper + OBS together: chatterbot grabs a screenshot every this-many seconds while you're live. Up to 4 from each summary's window are stitched into a 2x2 grid and given to the AI alongside the transcript text. Set to 0 to disable screenshot capture. Default 15s.",
        "type": "number",
        "min": 0, "max": 600, "step": 1,
        "suffix": "seconds (0 = off)",
    },
    "screenshot_max_age_hours": {
        "label": "Keep screenshots for",
        "tooltip": "How long old screenshots stay on disk. 0 = forever.",
        "help": "Screenshots older than this many hours get deleted to keep disk usage in check. Default 0 = keep forever — captures are content-hash deduped and stored as WebP, so disk growth is bounded even without an age limit. Raise to opt back into time-based cleanup.",
        "type": "number",
        "min": 0, "max": 720, "step": 1,
        "suffix": "hours (0 = forever)",
    },
    "screenshot_jpeg_quality": {
        "label": "Capture JPEG quality",
        "tooltip": "Quality OBS uses for the in-memory capture before WebP transcode (5-100).",
        "help": "OBS captures the frame as JPEG; we transcode to WebP before persisting. This is the input quality for the transcode — keep it high (80+) to avoid double-degrading. Default 85.",
        "type": "number",
        # min must be on the step grid for HTML form validity.
        "min": 5, "max": 100, "step": 5,
        "suffix": "5-100",
    },
    "screenshot_webp_quality": {
        "label": "Stored WebP quality",
        "tooltip": "Quality of the WebP file written to disk (5-100).",
        "help": "Lower = smaller files, blurrier. Higher = bigger files, sharper. WebP packs ~25-35% smaller than JPEG at the same visual quality so this can sit lower than the JPEG knob did. Default 65.",
        "type": "number",
        "min": 5, "max": 100, "step": 5,
        "suffix": "5-100",
    },
    "screenshot_width": {
        "label": "Screenshot width",
        "tooltip": "Pixels wide. Smaller = much smaller files.",
        "help": "How wide each screenshot is captured (height auto-scales). Default 480px is plenty for the AI to identify what's on screen without storing huge files.",
        "type": "number",
        "min": 240, "max": 1920, "step": 60,
        "suffix": "px",
    },
    "screenshot_grid_max": {
        "label": "Screenshots per summary grid",
        "tooltip": "Max screenshots stitched into a 2x2 grid for the AI.",
        "help": "Up to this many screenshots from a summary's time window get stitched into a single grid for the AI. Default 4 keeps a clean 2x2 layout. More = more visual context but a bigger image payload.",
        "type": "number",
        "min": 1, "max": 9, "step": 1,
        "suffix": "screenshots",
    },

    # ============================================================
    # VOICE → CHAT-LAG (managed by calibrator panel above)
    # ============================================================
    "chat_lag_seconds": {
        "label": "Chat lag offset",
        "tooltip": "How far behind your audio your chat reactions arrive.",
        "help": "Use the Chat-lag calibration panel above instead of editing this directly — it auto-detects the right value. Roughly: ~6s for Twitch Low Latency, ~12s for Standard.",
        "type": "number",
        "min": 0, "max": 60, "step": 1,
        "suffix": "seconds",
        "advanced": True,
    },
    "chat_lag_auto_tune_interval_seconds": {
        "label": "Auto-tune interval",
        "tooltip": "How often the dashboard auto-recalibrates chat lag.",
        "help": "Use the toggle in the calibration panel above instead of editing this. 0 disables auto-tune; 600s = re-check every 10 min.",
        "type": "number",
        "min": 0, "max": 3600, "step": 30,
        "suffix": "seconds (0 = off)",
        "advanced": True,
    },

    # ============================================================
    # INSIGHTS → ENGAGING SUBJECTS
    # ============================================================
    "engaging_subjects_interval_seconds": {
        "label": "Refresh interval",
        "tooltip": "How often the AI re-extracts engaging subjects from chat.",
        "help": "How often (in seconds) the AI re-runs the engaging-subjects pass. Default 180s (3 min). Faster = fresher subjects but more AI calls.",
        "type": "number",
        "min": 60, "max": 1800, "step": 30,
        "suffix": "seconds",
    },
    "engaging_subjects_lookback_minutes": {
        "label": "Lookback window",
        "tooltip": "How far back to scan chat each pass.",
        "help": "How many minutes of recent chat the engaging-subjects extractor looks at each pass. Default 20. Wider = catches longer-running threads; narrower = stays current.",
        "type": "number",
        "min": 5, "max": 120, "step": 5,
        "suffix": "minutes",
    },
    "engaging_subjects_max_messages": {
        "label": "Max messages per pass",
        "tooltip": "Cap on chat messages per extraction.",
        "help": "Hard cap on how many recent chat messages the extractor sends to the AI in one go. Higher = more context but bigger AI calls. Default 250.",
        "type": "number",
        "min": 50, "max": 1000, "step": 25,
        "suffix": "messages",
    },
    "engaging_subjects_min_cluster_size": {
        "label": "Min cluster size",
        "tooltip": "How many similar messages to count as a 'subject'.",
        "help": "Engaging subjects come from clustering similar chat messages. A cluster needs at least this many messages to count as a subject (otherwise it's just noise). Default 3.",
        "type": "number",
        "min": 2, "max": 10, "step": 1,
        "suffix": "messages",
        "advanced": True,
    },
    "engaging_subjects_notes_per_driver": {
        "label": "Notes per driver",
        "tooltip": "How many notes about each chatter to include as context.",
        "help": "When the AI is naming a subject, it gets background notes about the most-active chatters in the cluster. This sets how many notes per chatter. Default 2. Higher = better grounding, longer prompts.",
        "type": "number",
        "min": 0, "max": 10, "step": 1,
        "suffix": "notes",
        "advanced": True,
    },
    "engaging_subjects_max_drivers_with_notes": {
        "label": "Max drivers with notes",
        "tooltip": "How many distinct chatters get note-context.",
        "help": "Cap on how many of a cluster's most-active chatters get note-context attached. Default 8. Higher = richer context, much longer AI prompts.",
        "type": "number",
        "min": 1, "max": 30, "step": 1,
        "suffix": "chatters",
        "advanced": True,
    },

    # ============================================================
    # INSIGHTS → QUIET COHORT
    # ============================================================
    "quiet_cohort_silence_minutes": {
        "label": "Silence threshold",
        "tooltip": "How long every driver in a thread must have been silent.",
        "help": "Surfaces topic threads where ALL the chatters who used to drive it have gone quiet. This is how long every driver must have been silent to count. Default 15 minutes.",
        "type": "number",
        "min": 5, "max": 240, "step": 5,
        "suffix": "minutes",
    },
    "quiet_cohort_lookback_hours": {
        "label": "Look how far back",
        "tooltip": "Threads outside this window aren't considered.",
        "help": "Only threads that were active in this window count for the quiet-cohort scan. Default 24h. Older 'cold' threads are ignored.",
        "type": "number",
        "min": 1, "max": 168, "step": 1,
        "suffix": "hours",
    },
    "quiet_cohort_min_drivers": {
        "label": "Min drivers",
        "tooltip": "How many distinct chatters a thread needs to qualify.",
        "help": "Don't show 'quiet cohorts' that were really just one person. Default 2 — at least two chatters had to be driving it.",
        "type": "number",
        "min": 1, "max": 10, "step": 1,
        "suffix": "chatters",
    },
    "quiet_cohort_limit": {
        "label": "Max shown",
        "tooltip": "How many quiet cohorts to surface at most.",
        "help": "Cap on how many quiet-cohort cards appear on the engagement view. Default 6.",
        "type": "number",
        "min": 1, "max": 20, "step": 1,
        "suffix": "cards",
    },

    # ============================================================
    # INSIGHTS → HIGH-IMPACT SUBJECTS
    # ============================================================
    "high_impact_active_within_minutes": {
        "label": "Active chatter window",
        "tooltip": "Who counts as 'currently in chat' for high-impact ranking.",
        "help": "A chatter counts as currently active if they messaged within this many minutes. The high-impact panel ranks topics by how many of THESE chatters historically drove them. Default 30 min.",
        "type": "number",
        "min": 5, "max": 240, "step": 5,
        "suffix": "minutes",
    },
    "high_impact_lookback_days": {
        "label": "Historical lookback",
        "tooltip": "How far back to look for past driver patterns.",
        "help": "How many days of history to scan when checking 'has this currently-active chatter ever driven this topic'. Default 14 days. Longer = better history match for older streamers.",
        "type": "number",
        "min": 1, "max": 90, "step": 1,
        "suffix": "days",
    },
    "high_impact_min_overlap": {
        "label": "Min driver overlap",
        "tooltip": "How many currently-active chatters must historically have driven a topic.",
        "help": "A topic only shows up as 'high impact' if at least this many of the chatters currently in your chat have driven it before. Default 2.",
        "type": "number",
        "min": 1, "max": 10, "step": 1,
        "suffix": "chatters",
    },
    "high_impact_limit": {
        "label": "Max shown",
        "tooltip": "Cap on high-impact cards.",
        "help": "How many high-impact subject cards to show at most. Default 6.",
        "type": "number",
        "min": 1, "max": 20, "step": 1,
        "suffix": "cards",
    },

    # ============================================================
    # INSIGHTS → THREAD RECAPS
    # ============================================================
    "thread_recap_interval_seconds": {
        "label": "Recap interval",
        "tooltip": "How often the AI writes recaps for active conversation threads.",
        "help": "How often (in seconds) the AI re-recaps each active topic-thread on the engagement view. Default 300s (5 min). Lower = fresher recaps but more AI calls.",
        "type": "number",
        "min": 60, "max": 1800, "step": 30,
        "suffix": "seconds",
    },
    "thread_recap_max_messages_per_thread": {
        "label": "Max messages per thread",
        "tooltip": "Cap on chat the recap AI sees per thread.",
        "help": "How many recent messages from each thread the AI sees when writing the recap. Default 30. Higher = more context, longer AI calls.",
        "type": "number",
        "min": 5, "max": 100, "step": 5,
        "suffix": "messages",
    },

    # ============================================================
    # INSIGHTS → MISC
    # ============================================================
    "streamer_facts_path": {
        "label": "Streamer facts file",
        "tooltip": "Path to a markdown file with channel-specific facts.",
        "help": "A markdown file you write that contains stable facts about your channel — recurring bits, current arcs, inside jokes. The AI prepends it to extraction prompts so it stops 'hallucinating' things you've corrected. Default path is data/streamer_facts.md. Edit the contents directly from Settings → Prompts → Channel facts.",
        "type": "text",
        "placeholder": "data/streamer_facts.md",
    },
    "insights_modal_prewarm_top_n": {
        "label": "Pre-warm modal answers (top N)",
        "tooltip": "Eagerly generate AI bullets for the top N panel entries so first modal-open is instant.",
        "help": "When the engaging-subjects or open-questions panel refreshes, the dashboard can pre-generate the modal contents for the top N entries so opening any of them feels instant instead of spinning while the LLM thinks. 0 disables; 3 is a good default. Higher values spend more LLM cost on entries you might not click.",
        "type": "number",
        "min": 0, "max": 8, "step": 1,
    },
    "live_widget_enabled": {
        "label": "Show live chat widget",
        "tooltip": "Floating chat widget on most pages.",
        "help": "Shows a small live chat widget in the bottom corner of most dashboard pages. Off if you find it distracting; doesn't affect anything else.",
        "type": "bool",
    },
    "mod_mode_enabled": {
        "label": "Enable moderation classifier",
        "tooltip": "Opt-in advisory classifier — the bot never takes chat action.",
        "help": "When on, the AI batches recent chat through a strict-rubric classifier and saves any flagged messages as incidents on the /moderation page. ADVISORY ONLY — the bot never times anyone out or sends chat replies. Off by default.",
        "type": "bool",
    },

    # ============================================================
    # ADVANCED → POLLING
    # ============================================================
    "youtube_min_poll_seconds": {
        "label": "YouTube min poll interval",
        "tooltip": "Floor for how often we ask YouTube for new chat.",
        "help": "Adaptive polling — when chat is active we hit the API every this-many seconds. Lower = snappier but uses more daily quota. Default 10s puts a 6-hour stream comfortably under 10K daily quota units.",
        "type": "number",
        "min": 5, "max": 60, "step": 1,
        "suffix": "seconds",
        "advanced": True,
    },
    "youtube_max_poll_seconds": {
        "label": "YouTube max poll interval",
        "tooltip": "Ceiling — slowest we'll poll when chat is quiet.",
        "help": "When chat is quiet we double the poll interval up to this ceiling. Default 30s. Higher = saves more quota during dead air.",
        "type": "number",
        "min": 10, "max": 300, "step": 5,
        "suffix": "seconds",
        "advanced": True,
    },

    # ============================================================
    # ADVANCED → AI PROVIDER TUNING
    # ============================================================
    "anthropic_thinking_budget_tokens": {
        "label": "Claude thinking budget",
        "tooltip": "Tokens Claude can spend on extended thinking per call.",
        "help": "Some calls turn on Claude's extended-thinking mode (better reasoning at the cost of latency). This is the token budget for that. Default 4096. Higher = deeper thinking, more expensive.",
        "type": "number",
        "min": 512, "max": 32768, "step": 256,
        "suffix": "tokens",
        "advanced": True,
    },
    "openai_reasoning_model": {
        "label": "OpenAI reasoning model",
        "tooltip": "Which OpenAI model to use for thinking-heavy calls.",
        "help": "When a call asks for reasoning (e.g. notes, recaps), this model is used instead of the default. Leave blank to fall back to the default OpenAI model. Examples: 'o4-mini', 'gpt-5-thinking'.",
        "type": "text",
        "placeholder": "(leave blank to use default model)",
        "advanced": True,
    },
    "openai_organization": {
        "label": "OpenAI organization",
        "tooltip": "Optional OpenAI org ID.",
        "help": "If your OpenAI key belongs to an organization (most personal keys don't), paste the org ID here. Leave blank otherwise.",
        "type": "text",
        "placeholder": "(usually blank)",
        "advanced": True,
    },

    # ============================================================
    # ADVANCED → INTERNAL BUS
    # ============================================================
    "dashboard_internal_url": {
        "label": "Dashboard internal URL",
        "tooltip": "Where the bot reaches the dashboard for push notifications.",
        "help": "Cross-process bus — the bot pings the dashboard whenever new chat arrives so the live page updates instantly (~10ms) instead of polling. Default 'http://dashboard:8765' (the docker-compose service name). Empty disables the push, falling back to a 10s poll loop. Only touch this if you're running a custom setup.",
        "type": "text",
        "placeholder": "http://dashboard:8765",
        "advanced": True,
    },
    "internal_notify_secret": {
        "label": "Internal bus secret",
        "tooltip": "Shared secret between bot and dashboard.",
        "help": "Both halves share this secret to authenticate push notifications. Empty disables auth (fine for localhost / dev). Set anything you like — must match between bot and dashboard processes.",
        "type": "secret",
        "advanced": True,
    },
    # ============================================================
    # PERSONAL TRAINING DATASET (opt-in, off by default)
    # ============================================================
    "dataset_capture_enabled": {
        "label": "Capture training dataset",
        "tooltip": "Save every AI prompt + your dashboard actions, encrypted, for future fine-tuning.",
        "help": "Off by default. When ON (and you've run setup with a passphrase), every AI prompt + your dismiss/snooze/correct actions get encrypted and saved to data/dataset/. Down the line you can export the bundle and use it to fine-tune your own model. Setup, status, and export live on the Dataset page (look for it in the nav). Capture also needs CHATTERBOT_DATASET_PASSPHRASE in the bot/dashboard environment to unlock — without it the toggle is on but nothing gets written.",
        "type": "bool",
    },
}


# ---------- Section hierarchy ----------
# 5 main sections, each with sub-cards. Sub-cards are collapsible
# (open by default for the first one in each section). The streamer
# sees a tab strip for the 5 main sections plus a "Diagnostics" tab.

SECTIONS: list[dict[str, Any]] = [
    {
        "id": "connections",
        "title": "Connections",
        "icon": "fa-solid fa-plug",
        "blurb": (
            "Where chatterbot reads chat, audio, and events from. "
            "Twitch is required; everything else is opt-in."
        ),
        "subcards": [
            {
                "id": "twitch",
                "title": "Twitch (required)",
                "icon": "fa-brands fa-twitch",
                "blurb": (
                    "The bot's Twitch account + which channel it watches. "
                    "Restart the bot after changing these."
                ),
                "fields": [
                    "twitch_channel", "twitch_bot_nick",
                    "twitch_oauth_token",
                    "twitch_client_id", "twitch_client_secret",
                ],
                "default_open": True,
            },
            {
                "id": "obs",
                "title": "OBS",
                "icon": "fa-solid fa-circle-dot",
                "blurb": (
                    "Lets the dashboard know when you're live + grabs "
                    "screenshots for AI context. Read-only."
                ),
                "fields": [
                    "obs_enabled", "obs_host", "obs_port", "obs_password",
                ],
            },
            {
                "id": "streamelements",
                "title": "StreamElements",
                "icon": "fa-solid fa-coins",
                "blurb": (
                    "Pulls tip / sub / cheer / raid / follow events into "
                    "the dashboard."
                ),
                "fields": [
                    "streamelements_enabled", "streamelements_jwt",
                    "streamelements_channel_id",
                ],
            },
            {
                "id": "youtube",
                "title": "YouTube live chat",
                "icon": "fa-brands fa-youtube",
                "blurb": (
                    "Reads YouTube live chat alongside Twitch and treats "
                    "both as one big chat."
                ),
                "fields": [
                    "youtube_enabled", "youtube_api_key",
                    "youtube_channel_id",
                ],
            },
            {
                "id": "discord",
                "title": "Discord (work in progress)",
                "icon": "fa-brands fa-discord",
                "blurb": (
                    "Stub — wired but not yet functional. Safe to leave "
                    "off."
                ),
                "fields": [
                    "discord_enabled", "discord_bot_token",
                    "discord_channel_ids",
                ],
            },
        ],
    },
    {
        "id": "ai",
        "title": "AI brain",
        "icon": "fa-solid fa-robot",
        "blurb": (
            "Which AI service handles thinking-heavy calls (notes, recaps, "
            "engaging-subjects). Embeddings always run on local Ollama."
        ),
        "subcards": [
            {
                "id": "ai-provider",
                "title": "Provider",
                "icon": "fa-solid fa-microchip",
                "blurb": (
                    "Pick one. Restart the dashboard after changing this."
                ),
                "fields": ["llm_provider"],
                "default_open": True,
            },
            {
                "id": "ai-anthropic",
                "title": "Anthropic (Claude) credentials",
                "icon": "fa-solid fa-key",
                "blurb": "Required when 'AI provider' is set to anthropic.",
                "fields": ["anthropic_api_key", "anthropic_model"],
            },
            {
                "id": "ai-openai",
                "title": "OpenAI credentials",
                "icon": "fa-solid fa-key",
                "blurb": "Required when 'AI provider' is set to openai.",
                "fields": ["openai_api_key", "openai_model"],
            },
        ],
    },
    {
        "id": "voice",
        "title": "Voice & screen",
        "icon": "fa-solid fa-microphone",
        "blurb": (
            "Live transcription via Whisper + OBS screenshot capture. "
            "Powers the Stream timeline on /insights and auto-checks "
            "off chat cards when you address them out loud."
        ),
        "subcards": [
            {
                "id": "voice-basics",
                "title": "Audio basics",
                "icon": "fa-solid fa-headphones",
                "blurb": (
                    "Turn on transcription and pick a model size. "
                    "Bigger model = more accurate, slower, more disk."
                ),
                "fields": [
                    "whisper_enabled", "whisper_model",
                    "whisper_buffer_seconds", "whisper_min_silence_ms",
                ],
                "default_open": True,
            },
            {
                "id": "voice-accuracy",
                "title": "Accuracy tuning",
                "icon": "fa-solid fa-wand-sparkles",
                "blurb": (
                    "Tune Whisper for fast, emotional, mumbled, or "
                    "yelled streamer speech. The vocabulary-bias "
                    "feature auto-builds from runtime context (your "
                    "name, current game, active chatters, top chat "
                    "words, streamer_facts.md) and dramatically "
                    "improves accuracy on niche game terms and "
                    "chatter handles."
                ),
                "fields": [
                    "whisper_initial_prompt_enabled",
                    "whisper_initial_prompt_extra",
                    "whisper_beam_size",
                    "whisper_no_speech_threshold",
                    "whisper_log_prob_threshold",
                    "whisper_vad_threshold",
                ],
            },
            {
                "id": "voice-matching",
                "title": "Card matching",
                "icon": "fa-solid fa-bullseye",
                "blurb": (
                    "When you say something matching a chat card, the "
                    "dashboard auto-marks it 'addressed'. Tune strictness here."
                ),
                "fields": [
                    "whisper_match_threshold",
                    "whisper_unnamed_match_threshold",
                    "whisper_auto_confirm_seconds",
                    "whisper_llm_match_enabled",
                    "whisper_llm_match_interval_seconds",
                    "whisper_llm_match_min_chunks",
                    "whisper_llm_match_confidence",
                ],
            },
            {
                "id": "voice-summaries",
                "title": "Live summary cadence",
                "icon": "fa-solid fa-list",
                "blurb": (
                    "How often your audio gets rolled up into a summarised "
                    "moment on the Stream timeline."
                ),
                "fields": [
                    "whisper_group_interval_seconds",
                    "whisper_group_min_chunks",
                ],
            },
            {
                "id": "voice-screenshots",
                "title": "Screenshot capture",
                "icon": "fa-solid fa-camera",
                "blurb": (
                    "Grabs screenshots while you stream and ships them "
                    "to the AI alongside transcript summaries for visual "
                    "context. Set capture interval to 0 to disable."
                ),
                "fields": [
                    "screenshot_interval_seconds",
                    "screenshot_max_age_hours",
                    "screenshot_jpeg_quality",
                    "screenshot_webp_quality",
                    "screenshot_width",
                    "screenshot_grid_max",
                ],
            },
            {
                "id": "voice-chatlag",
                "title": "Chat-lag offset",
                "icon": "fa-solid fa-stopwatch",
                "blurb": (
                    "Use the Chat-lag calibration panel above for this. "
                    "These are the underlying values it manages — only "
                    "edit by hand if you really know what you're doing."
                ),
                "fields": [
                    "chat_lag_seconds",
                    "chat_lag_auto_tune_interval_seconds",
                ],
            },
        ],
    },
    {
        "id": "insights",
        "title": "Insights",
        "icon": "fa-solid fa-lightbulb",
        "blurb": (
            "What shows up on the dashboard's insight panels — engaging "
            "subjects, quiet cohorts, recaps, and per-feature toggles."
        ),
        "subcards": [
            {
                "id": "insights-subjects",
                "title": "Engaging subjects",
                "icon": "fa-solid fa-comments",
                "blurb": (
                    "The 'what's chat actually talking about right now' panel."
                ),
                "fields": [
                    "engaging_subjects_interval_seconds",
                    "engaging_subjects_lookback_minutes",
                    "engaging_subjects_max_messages",
                    "engaging_subjects_min_cluster_size",
                    "engaging_subjects_notes_per_driver",
                    "engaging_subjects_max_drivers_with_notes",
                ],
                "default_open": True,
            },
            {
                "id": "insights-quiet",
                "title": "Quiet cohorts",
                "icon": "fa-solid fa-moon",
                "blurb": (
                    "Surfaces topic threads where every driver has gone "
                    "quiet — chatters you can pivot back to and re-engage."
                ),
                "fields": [
                    "quiet_cohort_silence_minutes",
                    "quiet_cohort_lookback_hours",
                    "quiet_cohort_min_drivers",
                    "quiet_cohort_limit",
                ],
            },
            {
                "id": "insights-highimpact",
                "title": "High-impact subjects",
                "icon": "fa-solid fa-bolt",
                "blurb": (
                    "Topics ranked by how many CURRENTLY-active chatters "
                    "have historically driven them — your highest-leverage "
                    "pivots."
                ),
                "fields": [
                    "high_impact_active_within_minutes",
                    "high_impact_lookback_days",
                    "high_impact_min_overlap",
                    "high_impact_limit",
                ],
            },
            {
                "id": "insights-recaps",
                "title": "Thread recaps",
                "icon": "fa-solid fa-bookmark",
                "blurb": (
                    "Observational 1-2 sentence summaries of each active "
                    "topic-thread. Cached and refreshed in the background."
                ),
                "fields": [
                    "thread_recap_interval_seconds",
                    "thread_recap_max_messages_per_thread",
                ],
            },
            {
                "id": "insights-livewidget",
                "title": "Live chat widget",
                "icon": "fa-solid fa-comment-dots",
                "blurb": (
                    "The floating chat widget in the bottom corner of "
                    "most dashboard pages. Turn off to hide it."
                ),
                "fields": ["live_widget_enabled"],
            },
            {
                "id": "insights-moderation",
                "title": "Moderation classifier",
                "icon": "fa-solid fa-shield-halved",
                "blurb": (
                    "Opt-in advisory classifier. The bot never times "
                    "anyone out or sends chat replies — flagged "
                    "messages just appear on /moderation for review."
                ),
                "fields": ["mod_mode_enabled"],
            },
            {
                "id": "insights-facts",
                "title": "Streamer facts file",
                "icon": "fa-solid fa-file-lines",
                "blurb": (
                    "Optional markdown file you write with channel-"
                    "specific context (recurring bits, current arcs, "
                    "inside jokes). The AI reads it before extracting "
                    "subjects so it stops 'hallucinating' things you "
                    "don't actually do."
                ),
                "fields": ["streamer_facts_path", "insights_modal_prewarm_top_n"],
            },
        ],
    },
    {
        "id": "advanced",
        "title": "Advanced",
        "icon": "fa-solid fa-gear",
        "blurb": (
            "Backend tuning + plumbing. Defaults are sensible for most "
            "setups — only touch if you have a specific reason."
        ),
        "subcards": [
            {
                "id": "advanced-polling",
                "title": "API polling",
                "icon": "fa-solid fa-arrows-rotate",
                "blurb": (
                    "How often we hit external APIs. Tuning these affects "
                    "quota usage on long streams."
                ),
                "fields": [
                    "youtube_min_poll_seconds",
                    "youtube_max_poll_seconds",
                ],
            },
            {
                "id": "advanced-ai",
                "title": "AI provider tuning",
                "icon": "fa-solid fa-sliders",
                "blurb": (
                    "Fine-tuning for Claude / OpenAI behaviour on "
                    "thinking-heavy calls."
                ),
                "fields": [
                    "anthropic_thinking_budget_tokens",
                    "openai_reasoning_model",
                    "openai_organization",
                ],
            },
            {
                "id": "advanced-bus",
                "title": "Internal bus",
                "icon": "fa-solid fa-bolt",
                "blurb": (
                    "Cross-process push from the bot to the dashboard. "
                    "Defaults work for the docker-compose setup; only "
                    "change for custom layouts."
                ),
                "fields": [
                    "dashboard_internal_url",
                    "internal_notify_secret",
                ],
            },
            {
                "id": "advanced-dataset",
                "title": "Personal training dataset (opt-in)",
                "icon": "fa-solid fa-database",
                "blurb": (
                    "Off by default. When on, encrypts every AI prompt + "
                    "your dashboard actions to disk so you can fine-tune "
                    "your own model later. Setup, status, and export live "
                    "on the Dataset page."
                ),
                "fields": ["dataset_capture_enabled"],
            },
        ],
    },
]


def field_meta(key: str) -> dict[str, Any]:
    """Look up a field's metadata, returning a defaults-filled dict
    even for keys that aren't yet in FIELDS (so a forgotten metadata
    entry renders as a plain text input rather than crashing the
    page). Update FIELDS to fix any 'fall-through' rendering."""
    base = FIELDS.get(key, {})
    return {
        "label": base.get("label") or key.replace("_", " "),
        "tooltip": base.get("tooltip", ""),
        "help": base.get("help", ""),
        "type": base.get("type", "text"),
        "options": base.get("options"),
        "min": base.get("min"),
        "max": base.get("max"),
        "step": base.get("step"),
        "suffix": base.get("suffix", ""),
        "placeholder": base.get("placeholder", ""),
        "depends_on": base.get("depends_on"),
        "advanced": bool(base.get("advanced", False)),
    }
