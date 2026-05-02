"""Configuration loaded from environment / .env, with optional DB overrides.

Two layers:

  1. Environment / .env file (the original baseline).
  2. The dashboard-managed `app_settings` SQLite table — overrides for Twitch
     and StreamElements credentials. The bot reads these on startup; a restart
     is required to pick up changes.

Anything not editable through the dashboard (Ollama host, DB path, cadence
knobs, dashboard host/port itself) lives in env-only.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


# Keys the dashboard's /settings page is allowed to write into app_settings.
# Anything outside this set stays env-only.
EDITABLE_SETTING_KEYS: tuple[str, ...] = (
    "twitch_bot_nick",
    "twitch_oauth_token",
    "twitch_channel",
    "twitch_client_id",
    "twitch_client_secret",
    "streamelements_enabled",
    "streamelements_jwt",
    "streamelements_channel_id",
    "mod_mode_enabled",
    "obs_enabled",
    "obs_host",
    "obs_port",
    "obs_password",
    "youtube_enabled",
    "youtube_api_key",
    "youtube_channel_id",
    "youtube_min_poll_seconds",
    "youtube_max_poll_seconds",
    "discord_enabled",
    "discord_bot_token",
    "discord_channel_ids",
    "live_widget_enabled",
    "whisper_enabled",
    "whisper_model",
    "whisper_buffer_seconds",
    "whisper_match_threshold",
    "whisper_min_silence_ms",
    "whisper_beam_size",
    "whisper_no_speech_threshold",
    "whisper_log_prob_threshold",
    "whisper_vad_threshold",
    "whisper_initial_prompt_enabled",
    "whisper_initial_prompt_extra",
    "whisper_unnamed_match_threshold",
    "whisper_llm_match_enabled",
    "whisper_llm_match_interval_seconds",
    "whisper_llm_match_min_chunks",
    "whisper_llm_match_confidence",
    "whisper_auto_confirm_seconds",
    "whisper_group_interval_seconds",
    "whisper_group_min_chunks",
    "thread_recap_interval_seconds",
    "thread_recap_max_messages_per_thread",
    "chat_lag_seconds",
    "chat_lag_auto_tune_interval_seconds",
    "screenshot_interval_seconds",
    "screenshot_max_age_hours",
    "screenshot_jpeg_quality",
    "screenshot_webp_quality",
    "screenshot_phash_distance",
    "screenshot_width",
    "screenshot_grid_max",
    "quiet_cohort_silence_minutes",
    "quiet_cohort_lookback_hours",
    "quiet_cohort_min_drivers",
    "quiet_cohort_limit",
    "engaging_subjects_interval_seconds",
    "engaging_subjects_lookback_minutes",
    "engaging_subjects_max_messages",
    "high_impact_active_within_minutes",
    "high_impact_lookback_days",
    "high_impact_min_overlap",
    "high_impact_limit",
    "engaging_subjects_min_cluster_size",
    "engaging_subjects_notes_per_driver",
    "engaging_subjects_max_drivers_with_notes",
    "streamer_facts_path",
    "insights_modal_prewarm_top_n",
    "whisper_perfect_pass_enabled",
    "whisper_perfect_pass_model",
    "whisper_perfect_pass_beam_size",
    "whisper_perfect_pass_best_of",
    "whisper_perfect_pass_confidence_threshold",
    "whisper_perfect_pass_interval_seconds",
    "whisper_perfect_pass_hallucination_filter",
    "whisper_perfect_pass_hallucination_filter_strict",
    "whisper_perfect_pass_grace_seconds",
    "audio_clip_storage_enabled",
    "audio_clip_retention_hours",
    # ---------- LLM provider switch ----------
    "llm_provider",
    "anthropic_api_key",
    "anthropic_model",
    "anthropic_thinking_budget_tokens",
    "openai_api_key",
    "openai_model",
    "openai_reasoning_model",
    "openai_organization",
    # ---------- Cross-process bus ----------
    "dashboard_internal_url",
    "internal_notify_secret",
    # ---------- Personal training dataset (opt-in capture) ----------
    # Single user-facing toggle. The wrapped DEK / fingerprint /
    # other dataset_* keys are managed by the dataset CLI + the
    # /dataset page directly via repo.set_app_setting and aren't
    # editable through the standard /settings form (they need
    # passphrase derivation that the form pipeline can't model).
    "dataset_capture_enabled",
)

# Subset that should be rendered as password inputs. Blank submissions for
# these preserve the existing value rather than clearing it.
SECRET_SETTING_KEYS: frozenset[str] = frozenset(
    {
        "twitch_oauth_token",
        "twitch_client_secret",
        "streamelements_jwt",
        "obs_password",
        "youtube_api_key",
        "discord_bot_token",
        "anthropic_api_key",
        "openai_api_key",
        "internal_notify_secret",
    }
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Twitch
    twitch_bot_nick: str = ""
    twitch_oauth_token: str = ""
    twitch_channel: str = ""
    twitch_client_id: str = ""
    twitch_client_secret: str = ""

    # Ollama
    ollama_host: str = "192.168.69.229"
    ollama_port: int = 11434
    ollama_model: str = "qwen3.5:9b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_embed_dim: int = 768
    # Optional model override for the moderator. Empty string = use ollama_model.
    # Useful when the moderation classifier (high-frequency, cheap calls) should
    # ride a smaller / faster model than note extraction (rare, quality-critical).
    ollama_mod_model: str = ""
    # How many generation calls run concurrently against Ollama. Single-slot
    # by default — Ollama serialises on the GPU anyway, and a fair semaphore
    # at the client surface gives background loops a proper FIFO queue
    # rather than blocking randomly inside the network. Embeddings always
    # bypass this cap.
    ollama_max_concurrent_generations: int = 1

    # ---------- LLM provider selection ----------
    # Which backend handles generation calls (notes, recaps,
    # engaging-subjects, etc.). Embeddings ALWAYS run on Ollama
    # regardless — vec_messages / vec_threads are locked to
    # nomic-embed-text's 768-dim geometry.
    #   ollama     — local, default. Free, slow on CPU, fast on GPU.
    #   anthropic  — Claude. Requires anthropic_api_key.
    #   openai     — OpenAI. Requires openai_api_key.
    llm_provider: str = "ollama"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-4-7"
    # Token budget when think=True is passed (extended-thinking
    # mode). Generation calls that opt in get this much room for
    # the reasoning trace before the answer counts against
    # max_tokens.
    anthropic_thinking_budget_tokens: int = 4096

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    # Used only when think=True is passed and the caller doesn't
    # override the model. Empty falls back to `openai_model`.
    openai_reasoning_model: str = ""
    openai_organization: str = ""

    # Cross-process notification bus. The bot fires HTTP POSTs to
    # `dashboard_internal_url + /internal/notify` whenever something
    # changes (new chat message, new event, etc.) so the dashboard's
    # SSE stream pushes to clients with ~10 ms latency instead of
    # waiting for the watermark poll. Both halves run inside the
    # same docker-compose network by default.
    #   dashboard_internal_url — where the bot reaches the dashboard.
    #     Empty disables push notifications; the dashboard falls
    #     back to its watermark poll loop and everything still works.
    #   internal_notify_secret — shared secret. The dashboard
    #     rejects /internal/notify calls without a matching
    #     X-Internal-Secret header. Empty disables auth (dev only).
    dashboard_internal_url: str = "http://dashboard:8765"
    internal_notify_secret: str = ""

    # Personal training dataset capture. Off by default. When on AND
    # a wrapped DEK has been generated (CLI: `chatterbot dataset
    # setup`, or via /dataset in the dashboard) AND the bot/dashboard
    # process has CHATTERBOT_DATASET_PASSPHRASE in its env, every
    # structured LLM call + dashboard mutation is encrypted and
    # appended to data/dataset/shards/ for later fine-tuning. See
    # src/chatterbot/dataset/.
    dataset_capture_enabled: bool = False

    # OBS (read-only status: live state + current scene). Disabled by default.
    obs_enabled: bool = False
    obs_host: str = "localhost"
    obs_port: int = 4455
    obs_password: str = ""

    # StreamElements
    streamelements_enabled: bool = False
    streamelements_jwt: str = ""
    streamelements_channel_id: str = ""

    # YouTube live-chat ingestion via the YouTube Data API v3.
    # Read-only API-key auth. The listener pages liveChatMessages and
    # persists into messages + events alongside Twitch.
    youtube_enabled: bool = False
    youtube_api_key: str = ""
    youtube_channel_id: str = ""
    # Adaptive poll cadence to keep daily quota within the free 10,000
    # units. Empty polls double the interval (capped at max); non-empty
    # polls reset to the minimum. The listener also honors the server's
    # pollingIntervalMillis as a floor. Default min=10s / max=30s puts
    # a 6-hour active stream under ~10K units.
    youtube_min_poll_seconds: int = 10
    youtube_max_poll_seconds: int = 30

    # Discord — STUB. The listener exists but no gateway connection yet.
    discord_enabled: bool = False
    discord_bot_token: str = ""
    discord_channel_ids: str = ""  # comma-separated

    # Dashboard
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8765
    dashboard_basic_auth_user: str = ""
    dashboard_basic_auth_pass: str = ""

    # Storage
    db_path: str = "data/chatters.db"

    # Summarizer cadence (starting points; tune after observation)
    # Run note extraction once a chatter accumulates this many
    # unsummarized messages. Lower = more notes per chatter (more
    # LLM calls); higher = sparser but cheaper. Pair with
    # `summarize_idle_minutes` so a chatter who quiets down before
    # hitting the threshold still gets summarized.
    summarize_after_messages: int = 10
    summarize_idle_minutes: int = 10
    idle_sweep_interval_seconds: int = 60

    # Channel topic snapshots (starting points; tune)
    topics_interval_minutes: int = 5
    topics_max_messages: int = 200

    # Message embedding indexer — feeds /search semantic-search index.
    # Runs alongside the summarizer; pure local Ollama work, no external
    # quota at risk. Set interval=0 to disable.
    message_embed_interval_seconds: int = 30
    message_embed_batch_size: int = 25

    # Real-time whisper transcription pipeline. The OBS audio relay
    # script POSTs PCM chunks to /audio/ingest; the service buffers,
    # transcribes via faster-whisper, embeds each chunk, and cosine-
    # matches against open insight cards. Match above threshold flips
    # the card to 'addressed' automatically.
    whisper_enabled: bool = False
    whisper_model: str = "base.en"          # tiny.en | base.en | small.en | medium.en
    whisper_buffer_seconds: float = 5.0     # accumulate this much audio before transcribing
    whisper_match_threshold: float = 0.55   # cosine sim needed to auto-address
    whisper_compute_type: str = "auto"      # auto | int8 | float16 | float32
    # How long a silence must last before whisper splits into a new
    # utterance. Lower = more fragmented sentences; higher = full
    # thoughts grouped together. 5000 ms (5s) groups whole thoughts
    # for natural conversational pacing; drop to 500-1000 if you want
    # tighter per-clause splits.
    whisper_min_silence_ms: int = 5000
    # Beam-search width for the whisper decode. 1 = greedy (faster,
    # worse on hard audio: yelling, mumbling, stammered words).
    # 3 = sweet spot for streamer-style speech — recovers from a
    # bad first guess without doubling latency. 5 = slower but
    # squeezes a little more accuracy on very emotional content.
    whisper_beam_size: int = 3
    # Lowered from whisper's 0.6 default. Sustained yelling can
    # register as silence on the no-speech-prob detector and get
    # dropped — a conservative threshold keeps emotional moments
    # in the transcript. Raise (toward 0.6) if you're seeing
    # whisper hallucinate text on truly silent passages.
    whisper_no_speech_threshold: float = 0.4
    # Lowered from whisper's -1.0 default for the same reason —
    # emotional / distorted audio has worse avg log-prob and the
    # default triggers a temperature-backoff fallback loop that
    # produces "you you you" hallucinations.
    whisper_log_prob_threshold: float = -1.5
    # VAD threshold (0-1, lower = more sensitive). High-energy
    # breath noise can mask speech onset on yelling; 0.3 catches
    # those clauses where 0.5 would skip the first word.
    whisper_vad_threshold: float = 0.3
    # When on, build an `initial_prompt` for whisper from runtime
    # context: streamer display-name, current game, recently-active
    # chatter handles, and the contents of streamer_facts.md. Whisper
    # treats this as vocabulary bias — accuracy on niche game terms
    # (Tarkov maps, RE bosses, LoL champion names) and chatter
    # handles improves dramatically. Free at runtime; toggle off
    # only if you suspect it's causing unwanted bias.
    whisper_initial_prompt_enabled: bool = True
    # Optional extra text appended to the auto-built initial_prompt.
    # Use for streamer-specific terms not in streamer_facts.md or
    # the live channel context.
    whisper_initial_prompt_extra: str = ""
    # Talking-point cards are short and generic enough that vague
    # utterances ("why that happened") repeatedly clear the regular
    # threshold against unrelated cards. To gate that, we apply a
    # higher bar when the streamer hasn't named the chatter in the
    # utterance. Naming them ("that's right aquanote1, …") is the
    # strongest signal of real engagement, so a name-mention drops
    # the bar back to whisper_match_threshold. Set to a value ≥ 1.0
    # to require name mention always; set equal to whisper_match_threshold
    # to restore old uniform behavior.
    whisper_unnamed_match_threshold: float = 0.80
    # Batched LLM matcher. Aggregates whisper transcripts over a window
    # and asks the LLM "did the streamer actually engage with any of
    # these cards?" using a streamer-aware prompt that knows most
    # utterances are game reactions / thinking aloud, not chat-directed.
    # When enabled, this is the *primary* auto-pending mechanism;
    # per-utterance cosine still runs for the live transcript strip
    # icons + chat-↔-transcript reverse lookup but doesn't write
    # auto_pending itself.
    whisper_llm_match_enabled: bool = True
    # Window between LLM passes (seconds). Each pass processes every
    # transcript chunk added since the last pass — set higher to save
    # Ollama throughput, lower for tighter feedback.
    whisper_llm_match_interval_seconds: int = 90
    # Minimum chunks accumulated before a pass runs. Avoids wasting LLM
    # calls on tiny windows (one stray utterance every 90s).
    whisper_llm_match_min_chunks: int = 3
    # Confidence floor for auto-pending. The LLM emits a 0..1 confidence
    # per match; below this we log but don't write auto_pending.
    whisper_llm_match_confidence: float = 0.65
    # Auto-pending cards self-promote to 'addressed' after this many
    # seconds without explicit confirm/reject. The original 60 s was too
    # tight — by the time you opened the dashboard the card was already
    # gone. 300 s (5 min) is the new default; raise if you want more
    # review time, lower if you want auto-pendings to disappear faster.
    whisper_auto_confirm_seconds: int = 300
    # Transcript-group summariser: replaces the per-utterance live strip
    # with one line per window. Set interval=0 to disable.
    whisper_group_interval_seconds: int = 60
    whisper_group_min_chunks: int = 2

    # Twitch broadcast latency. The streamer's mic captures into OBS
    # in real time, but viewers don't hear it until ~3-15s later
    # depending on Twitch's "Low Latency" vs "Standard" mode + CDN +
    # player buffer + viewer reaction time. Chat is reacting to what
    # they HEARD, so when we pair transcript chunks with chat for the
    # group-summary call, we offset the chat window backwards by this
    # many seconds. 0 = pair by wall-clock (correct for the test setup
    # where chatterbot ingests playback audio); 5-8 typical for
    # streamers using Low Latency; 12-15 for Standard/DVR.
    # The /settings → Whisper page exposes a calibration tool that
    # auto-detects this from cross-correlation of transcript text vs
    # chat token overlap.
    chat_lag_seconds: int = 6
    # How often the background auto-tuner re-runs the calibration.
    # 0 disables. Default 600 (10 min) — frequent enough to converge
    # within the first few panels of a stream, infrequent enough to
    # not dominate CPU when chat is hot.
    chat_lag_auto_tune_interval_seconds: int = 600

    # Topic-thread recap loop. Periodically summarises each active
    # thread's recent messages into a 1-2 sentence observational line
    # for the engagement-view "Live conversations" panel. Set
    # interval=0 to disable.
    thread_recap_interval_seconds: int = 300
    thread_recap_max_messages_per_thread: int = 30

    # OBS screenshot capture for transcript groups. When both whisper
    # and OBS are enabled, captures the current program scene every
    # `screenshot_interval_seconds`. Up to 4 screenshots from the
    # group's time window are stitched into a 2x2 grid and passed to
    # the LLM alongside the transcript text on each group summary —
    # the model gets visual context too, not just audio. Set
    # interval=0 to disable screenshot capture entirely.
    # Chat-only fallback group summaries — fire when there's chat
    # but no audio in the window so the Stream timeline isn't empty
    # during whisper-off stretches. Audio remains primary when it
    # exists; this is a fallback only.
    chat_only_summary_enabled: bool = True
    chat_only_summary_min_messages: int = 5
    chat_only_summary_window_minutes: int = 10

    # Capture interval. 10s with the 6-cell grid + scene-change
    # dedup gives the LLM finer-grained visual context for active
    # streams without blowing up payload during paused / static
    # scenes (the dedup collapses near-duplicate frames before
    # stitching).
    screenshot_interval_seconds: int = 10
    # 0 = keep forever (default). Captures are content-hash deduped
    # and stored as WebP so disk growth is bounded; raise above 0 to
    # opt back into age-based deletion.
    screenshot_max_age_hours: int = 0
    # JPEG quality for the OBS-side capture; we transcode to WebP
    # locally before persisting, so this is just the input quality
    # for the transcode (kept high to not double-degrade).
    screenshot_jpeg_quality: int = 85
    screenshot_width: int = 480
    # WebP quality for the persisted file. 60-70 is the sweet spot
    # for visual context to a multimodal LLM — smaller than JPEG
    # at the same quality, dedup-friendly via content hash.
    screenshot_webp_quality: int = 65
    # Perceptual-hash Hamming-distance threshold for adjacent-frame
    # dedup at stitch time. 0 disables dedup; higher = more
    # aggressive (drops more "similar" frames). 6 is conservative —
    # only kills near-pixel-identical frames (paused / static scenes).
    screenshot_phash_distance: int = 6

    # Perfect-pass transcription (slice 12). The first pass runs at
    # `whisper_model` for live signal (card-matching, the strip).
    # The perfect pass re-transcribes low-confidence chunks with
    # accuracy-tuned settings (beam=5, best_of=5,
    # condition_on_previous_text=True, biased initial_prompt with
    # previous chunks + streamer_facts vocabulary). Same model
    # by default — empty `whisper_perfect_pass_model` reuses the
    # first-pass model so no extra VRAM. Set to `large-v3` for the
    # additional ~5-10% accuracy bump on hard cases at the cost of
    # doubling VRAM.
    whisper_perfect_pass_enabled: bool = True
    whisper_perfect_pass_model: str = ""  # "" = same as whisper_model
    whisper_perfect_pass_beam_size: int = 5
    whisper_perfect_pass_best_of: int = 5
    # Chunks with avg_logprob < this threshold are queued for re-pass.
    # NULL avg_logprob (legacy rows or whisper didn't return it) is
    # also queued so the perfect pass benefits everything captured
    # before the slice landed.
    whisper_perfect_pass_confidence_threshold: float = -0.5
    # Cadence the perfect-pass loop polls the queue at. The loop
    # processes one chunk per tick + sleeps between, so this is
    # effectively "minimum seconds between perfect-pass GPU bursts."
    whisper_perfect_pass_interval_seconds: int = 5
    # Filter perfect-pass output against a list of canonical whisper
    # hallucinations ("I'll see you in the next video", "Thanks for
    # watching", "[Music]", etc.). When the refined text introduces
    # one of these phrases that wasn't in the first-pass text,
    # the refine is rejected and the first-pass text is kept.
    # Default on. Disable if your channel content genuinely uses
    # those phrases (e.g. you DO end streams with "see you next
    # time" and the filter is rejecting your refines).
    whisper_perfect_pass_hallucination_filter: bool = True
    # Strict mode adds outros / CTAs ("thanks for watching",
    # "subscribe to my channel", "see you in the next video") to the
    # filter. Off by default because real streamers say these
    # legitimately and rejecting them throws away genuine refines.
    # Opt in if your channel never produces those phrases (e.g.
    # competitive / esports content where stream outros are silent).
    # condition_on_previous_text=True on the perfect pass already
    # helps mid-stream context resist these; strict mode is the
    # streamer-side override for the remainder.
    whisper_perfect_pass_hallucination_filter_strict: bool = False
    # Group summaries read transcript_chunks.text directly; if the
    # summary fires before the perfect pass has refined the window's
    # chunks, the LLM sees first-pass text (potentially with whisper
    # hallucinations the perfect pass would have caught). This grace
    # period defers the summary until refine-eligible chunks in the
    # window are either refined OR the youngest chunk in the window
    # is older than `grace` seconds (cap so a perfect-pass crash
    # can't block summaries forever). 0 disables the gate. 240s
    # default ≈ a comfortable margin over the observed ~170s lag.
    whisper_perfect_pass_grace_seconds: int = 240

    # Audio-clip storage (slice 12) — persists the WAV bytes for each
    # captured chunk so the perfect-pass loop can re-transcribe and
    # future features (replay-on-click, multimodal LLM input) have
    # the source bytes available. Content-hashed under
    # `data/audio_clips/` (same scheme as screenshots). Disabling
    # storage also disables the perfect pass since it has nothing
    # to re-transcribe.
    audio_clip_storage_enabled: bool = True
    # 0 = keep forever (default). Captures are content-hash deduped
    # so disk growth is bounded by unique audio content; raise above
    # 0 to opt back into age-based deletion.
    audio_clip_retention_hours: int = 0
    # Maximum screenshots stitched into the per-group grid. 6 uses a
    # 3x2 layout (1440x540 canvas, cells stay 480x270 — same
    # legibility as the 4-cell layout). With phash dedup the average
    # payload is ~unchanged; only grids with 6 genuinely-different
    # frames pay the +50% byte cost.
    screenshot_grid_max: int = 6

    # Quiet-cohort detection on the engagement view. Surfaces topic
    # threads whose driver chatters have all gone silent — clusters
    # of people the streamer can pivot back toward to re-engage.
    # `silence_minutes`: how long every driver in the thread must
    # have been silent before the thread shows as quiet.
    # `lookback_hours`: only consider threads that were active in this
    # window; archived ones don't count.
    # `min_drivers`: don't surface single-person "cohorts."
    quiet_cohort_silence_minutes: int = 15
    quiet_cohort_lookback_hours: int = 24
    quiet_cohort_min_drivers: int = 2
    quiet_cohort_limit: int = 6

    # Engaging-subjects extractor on InsightsService. A separate
    # subject-level pass over recent chat messages — distinct from
    # topic_threads (cosine clustering) which often lump multiple
    # subjects within the same time window. The LLM is prompted to
    # silently filter religion / politics / controversy out.
    engaging_subjects_interval_seconds: int = 180
    engaging_subjects_lookback_minutes: int = 20
    engaging_subjects_max_messages: int = 250

    # "What to say" / high-impact subjects panel — ranks topic_threads
    # by how many of the chatters CURRENTLY in chat have historically
    # driven that thread. Streamer pivots to a high-rank subject for
    # maximum engagement of the live audience.
    high_impact_active_within_minutes: int = 30
    high_impact_lookback_days: int = 14
    high_impact_min_overlap: int = 2
    high_impact_limit: int = 6

    # Engaging-subjects extractor: pre-cluster messages by embedding
    # cosine similarity before sending to the LLM. Each cluster gets
    # one subject in the output. Threshold tuning:
    #   - higher (0.65+) → tighter clusters, more separate subjects
    #   - lower (0.45-)  → looser clusters, more merging
    # Tiny clusters (< min_cluster_size) are dropped as noise.
    # Set cluster_threshold=0 to disable clustering and fall back
    # to the single-pass extractor.
    engaging_subjects_cluster_threshold: float = 0.55
    engaging_subjects_min_cluster_size: int = 3
    # Per-driver context: how many recent notes (per chatter) to inject
    # into the prompt as "who they are" hints. 0 disables.
    engaging_subjects_notes_per_driver: int = 2
    engaging_subjects_max_drivers_with_notes: int = 8
    # Streamer-authored facts file. Loaded if present and prepended
    # to extraction prompts as channel context. Lets the streamer
    # correct hallucinations at source ("there is no FF8 remake").
    streamer_facts_path: str = "data/streamer_facts.md"

    # Modal-output pre-warming — when an insights panel refreshes,
    # eagerly generate the modal contents (subject talking-points /
    # question answer-angles) for the top N entries so the
    # streamer's first modal-open is instant. 0 disables; higher
    # spends more LLM cost for snappier modal opens.
    insights_modal_prewarm_top_n: int = 3

    # ---------- int8 vector-storage pilot ----------
    # When True, ranking-only RAG reads target the int8-quantized
    # mirrors (`vec_messages_q8`, `vec_notes_q8`, `vec_threads_q8`,
    # `vec_transcripts_q8`) instead of their FLOAT[768] originals.
    # Writes always dual-populate both tables so the flag can be
    # flipped on/off without losing data. Per-table quantization
    # scales are auto-computed and persisted to `app_settings` on
    # first startup with the flag on. Threshold-bound reads
    # (find_near_duplicate_flood, count_transcripts_matching_embedding,
    # find_recent_transcript_for_message) intentionally stay on
    # float32 because their distance thresholds are tuned to the
    # float32 metric scale. Env-only — not exposed in the dashboard
    # form. ~4x storage reduction, ~99% top-10 recall in our eval
    # (see scripts/bench_embedding_quantization.py).
    use_int8_embeddings: bool = False

    # Moderation mode — opt-in. When enabled, the bot batches recent
    # messages through a strict-rubric LLM classifier and persists
    # flagged ones as incidents for streamer review. Advisory only —
    # the bot never takes chat action.
    mod_mode_enabled: bool = False
    mod_review_interval_minutes: int = 5
    mod_review_max_messages: int = 100

    # Dashboard UI toggles
    live_widget_enabled: bool = True

    # Run mode (used by main.py / docker entrypoint)
    run_mode: str = "bot"

    @property
    def ollama_base_url(self) -> str:
        return f"http://{self.ollama_host}:{self.ollama_port}"

    @property
    def dashboard_basic_auth_enabled(self) -> bool:
        return bool(self.dashboard_basic_auth_user and self.dashboard_basic_auth_pass)


def _coerce(key: str, value: str) -> Any:
    if key == "whisper_buffer_seconds":
        try:
            return float(value)
        except (TypeError, ValueError):
            return 5.0
    if key == "whisper_match_threshold":
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.55
    if key == "engaging_subjects_cluster_threshold":
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.55
    if key == "whisper_unnamed_match_threshold":
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.80
    if key == "whisper_llm_match_confidence":
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.65
    if key == "whisper_no_speech_threshold":
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.4
    if key == "whisper_log_prob_threshold":
        try:
            return float(value)
        except (TypeError, ValueError):
            return -1.5
    if key == "whisper_vad_threshold":
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.3
    if key == "whisper_beam_size":
        try:
            return max(1, min(10, int(value)))
        except (TypeError, ValueError):
            return 3
    if key == "whisper_perfect_pass_confidence_threshold":
        try:
            return float(value)
        except (TypeError, ValueError):
            return -0.5
    if key in (
        "whisper_llm_match_interval_seconds", "whisper_llm_match_min_chunks",
        "whisper_auto_confirm_seconds",
        "whisper_group_interval_seconds", "whisper_group_min_chunks",
        "thread_recap_interval_seconds", "thread_recap_max_messages_per_thread",
        "chat_lag_seconds", "chat_lag_auto_tune_interval_seconds",
        "youtube_min_poll_seconds", "youtube_max_poll_seconds",
        "screenshot_interval_seconds", "screenshot_max_age_hours",
        "screenshot_jpeg_quality", "screenshot_webp_quality",
        "screenshot_phash_distance",
        "screenshot_width", "screenshot_grid_max",
        "quiet_cohort_silence_minutes", "quiet_cohort_lookback_hours",
        "quiet_cohort_min_drivers", "quiet_cohort_limit",
        "engaging_subjects_interval_seconds",
        "engaging_subjects_lookback_minutes",
        "engaging_subjects_max_messages",
        "high_impact_active_within_minutes",
        "high_impact_lookback_days",
        "high_impact_min_overlap",
        "high_impact_limit",
        "engaging_subjects_min_cluster_size",
        "engaging_subjects_notes_per_driver",
        "engaging_subjects_max_drivers_with_notes",
        "anthropic_thinking_budget_tokens",
        "whisper_perfect_pass_beam_size",
        "whisper_perfect_pass_best_of",
        "whisper_perfect_pass_interval_seconds",
        "whisper_perfect_pass_grace_seconds",
        "audio_clip_retention_hours",
    ):
        try:
            return int(value)
        except (TypeError, ValueError):
            return {
                "whisper_llm_match_interval_seconds": 90,
                "whisper_llm_match_min_chunks": 3,
                "whisper_auto_confirm_seconds": 300,
                "whisper_group_interval_seconds": 60,
                "whisper_group_min_chunks": 2,
                "thread_recap_interval_seconds": 300,
                "thread_recap_max_messages_per_thread": 30,
                "chat_lag_seconds": 6,
                "chat_lag_auto_tune_interval_seconds": 600,
                "youtube_min_poll_seconds": 10,
                "youtube_max_poll_seconds": 30,
                "screenshot_interval_seconds": 10,
                "screenshot_max_age_hours": 0,
                "screenshot_jpeg_quality": 85,
                "screenshot_webp_quality": 65,
                "screenshot_phash_distance": 6,
                "screenshot_width": 480,
                "screenshot_grid_max": 6,
                "quiet_cohort_silence_minutes": 15,
                "quiet_cohort_lookback_hours": 24,
                "quiet_cohort_min_drivers": 2,
                "quiet_cohort_limit": 6,
                "engaging_subjects_interval_seconds": 180,
                "engaging_subjects_lookback_minutes": 20,
                "engaging_subjects_max_messages": 250,
                "high_impact_active_within_minutes": 30,
                "high_impact_lookback_days": 14,
                "high_impact_min_overlap": 2,
                "high_impact_limit": 6,
                "engaging_subjects_min_cluster_size": 3,
                "engaging_subjects_notes_per_driver": 2,
                "engaging_subjects_max_drivers_with_notes": 8,
                "anthropic_thinking_budget_tokens": 4096,
                "whisper_perfect_pass_beam_size": 5,
                "whisper_perfect_pass_best_of": 5,
                "whisper_perfect_pass_interval_seconds": 5,
                "whisper_perfect_pass_grace_seconds": 240,
                "audio_clip_retention_hours": 0,
            }[key]
    if key in (
        "streamelements_enabled", "mod_mode_enabled",
        "obs_enabled", "live_widget_enabled",
        "youtube_enabled", "discord_enabled",
        "whisper_enabled", "whisper_llm_match_enabled",
        "whisper_initial_prompt_enabled",
        "whisper_perfect_pass_enabled",
        "whisper_perfect_pass_hallucination_filter",
        "whisper_perfect_pass_hallucination_filter_strict",
        "audio_clip_storage_enabled",
    ):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if key == "obs_port":
        try:
            return int(value)
        except (TypeError, ValueError):
            return 4455
    if key == "whisper_min_silence_ms":
        try:
            return int(value)
        except (TypeError, ValueError):
            return 5000
    return value


def _load_db_overrides(db_path: str) -> dict[str, Any]:
    """Read app_settings from SQLite if present. Returns {} if the file or
    table doesn't exist yet."""
    if not Path(db_path).exists():
        return {}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT key, value FROM app_settings")
        rows = cur.fetchall()
        conn.close()
    except sqlite3.Error:
        return {}
    out: dict[str, Any] = {}
    for r in rows:
        if r["key"] in EDITABLE_SETTING_KEYS:
            out[r["key"]] = _coerce(r["key"], r["value"])
    return out


def get_settings() -> Settings:
    """Load env / .env, then layer dashboard-managed DB overrides on top."""
    base = Settings()
    overrides = _load_db_overrides(base.db_path)
    if not overrides:
        return base
    return base.model_copy(update=overrides)


def get_settings_with_sources() -> tuple[Settings, set[str]]:
    """Like get_settings(), but also returns the set of keys that came from
    the DB (used by the dashboard to badge each field's source)."""
    base = Settings()
    overrides = _load_db_overrides(base.db_path)
    if not overrides:
        return base, set()
    return base.model_copy(update=overrides), set(overrides.keys())
