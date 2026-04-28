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
    "screenshot_interval_seconds",
    "screenshot_max_age_hours",
    "screenshot_jpeg_quality",
    "screenshot_width",
    "screenshot_grid_max",
    "quiet_cohort_silence_minutes",
    "quiet_cohort_lookback_hours",
    "quiet_cohort_min_drivers",
    "quiet_cohort_limit",
    "engaging_subjects_interval_seconds",
    "engaging_subjects_lookback_minutes",
    "engaging_subjects_max_messages",
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
    summarize_after_messages: int = 20
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
    screenshot_interval_seconds: int = 15
    screenshot_max_age_hours: int = 24
    screenshot_jpeg_quality: int = 60
    screenshot_width: int = 480
    # Maximum screenshots stitched into the per-group grid. 4 keeps a
    # 2x2 layout that's still legible when a group's window contains
    # many shots.
    screenshot_grid_max: int = 4

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
    if key in (
        "whisper_llm_match_interval_seconds", "whisper_llm_match_min_chunks",
        "whisper_auto_confirm_seconds",
        "whisper_group_interval_seconds", "whisper_group_min_chunks",
        "thread_recap_interval_seconds", "thread_recap_max_messages_per_thread",
        "youtube_min_poll_seconds", "youtube_max_poll_seconds",
        "screenshot_interval_seconds", "screenshot_max_age_hours",
        "screenshot_jpeg_quality", "screenshot_width", "screenshot_grid_max",
        "quiet_cohort_silence_minutes", "quiet_cohort_lookback_hours",
        "quiet_cohort_min_drivers", "quiet_cohort_limit",
        "engaging_subjects_interval_seconds",
        "engaging_subjects_lookback_minutes",
        "engaging_subjects_max_messages",
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
                "youtube_min_poll_seconds": 10,
                "youtube_max_poll_seconds": 30,
                "screenshot_interval_seconds": 15,
                "screenshot_max_age_hours": 24,
                "screenshot_jpeg_quality": 60,
                "screenshot_width": 480,
                "screenshot_grid_max": 4,
                "quiet_cohort_silence_minutes": 15,
                "quiet_cohort_lookback_hours": 24,
                "quiet_cohort_min_drivers": 2,
                "quiet_cohort_limit": 6,
                "engaging_subjects_interval_seconds": 180,
                "engaging_subjects_lookback_minutes": 20,
                "engaging_subjects_max_messages": 250,
            }[key]
    if key in (
        "streamelements_enabled", "mod_mode_enabled",
        "obs_enabled", "live_widget_enabled",
        "youtube_enabled", "discord_enabled",
        "whisper_enabled", "whisper_llm_match_enabled",
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
