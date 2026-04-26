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
)

# Subset that should be rendered as password inputs. Blank submissions for
# these preserve the existing value rather than clearing it.
SECRET_SETTING_KEYS: frozenset[str] = frozenset(
    {
        "twitch_oauth_token",
        "twitch_client_secret",
        "streamelements_jwt",
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

    # StreamElements
    streamelements_enabled: bool = False
    streamelements_jwt: str = ""
    streamelements_channel_id: str = ""

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

    # Run mode (used by main.py / docker entrypoint)
    run_mode: str = "bot"

    @property
    def ollama_base_url(self) -> str:
        return f"http://{self.ollama_host}:{self.ollama_port}"

    @property
    def dashboard_basic_auth_enabled(self) -> bool:
        return bool(self.dashboard_basic_auth_user and self.dashboard_basic_auth_pass)


def _coerce(key: str, value: str) -> Any:
    if key == "streamelements_enabled":
        return value.strip().lower() in ("true", "1", "yes", "on")
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
