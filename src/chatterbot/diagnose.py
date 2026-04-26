"""Diagnostic bundle builder.

When something breaks, the streamer runs `chatterbot diagnose` (or clicks the
button on the settings page) and gets a `chatterbot-diagnose-<ts>.cbreport`
file — a renamed zip — that they can attach to a bug report.

PRIVACY DEFAULTS — minimal-by-default. The bundle deliberately excludes:

  - chat message bodies, note text, event payloads (raw_json), usernames,
    twitch_ids, the SQLite DB itself
  - OAuth tokens, JWTs, basic-auth credentials, the .env file's values

It DOES include:

  - rotating log file tails (which may contain a stack trace + the offending
    Python module/line, but no chat content because we never log that)
  - system info (Python / OS / sqlite versions, installed packages)
  - DB row counts and schema version (no per-row data)
  - .env keys with present/missing + value lengths only
  - app version + git SHA if the directory is a git checkout
  - Ollama reachability check at the configured base URL

Pass `with_recent_activity=True` to additionally include a `recent_activity.json`
file with the last N hours' usernames + per-user message *counts* (still no
message bodies). That's an opt-in for reproducing timing-related bugs.
"""

from __future__ import annotations

import io
import json
import os
import platform
import sqlite3
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx

from . import __version__
from .config import EDITABLE_SETTING_KEYS, Settings
from .logging_setup import LOG_DIR


# Cap each log file's tail size in the bundle so a noisy week doesn't ship
# a giant attachment.
LOG_TAIL_BYTES = 1 * 1024 * 1024  # 1 MB tail per log file


def build_diagnostic_bundle(
    out_path: Path,
    settings: Settings,
    *,
    with_recent_activity: bool = False,
) -> Path:
    """Write the bundle. Returns out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(_meta(), indent=2))
        zf.writestr("system.txt", _system_info())
        zf.writestr("packages.txt", _packages())
        zf.writestr("env.txt", _env_summary(settings))
        zf.writestr("db_stats.json", json.dumps(_db_stats(settings), indent=2))
        zf.writestr("ollama.json", json.dumps(_ollama_probe(settings), indent=2))

        for log_file in _log_files():
            tail = _tail_bytes(log_file, LOG_TAIL_BYTES)
            zf.writestr(f"logs/{log_file.name}", tail)

        if with_recent_activity:
            zf.writestr(
                "recent_activity.json",
                json.dumps(_recent_activity(settings), indent=2),
            )

        zf.writestr("README.txt", _bundle_readme(with_recent_activity))
    return out_path


def default_bundle_filename() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"chatterbot-diagnose-{ts}.cbreport"


# Where bug reports go. The repo is hard-coded to mscrnt/chatterbot — fork
# users can override by setting an env var if/when that ever matters.
ISSUE_REPO = os.environ.get("CHATTERBOT_ISSUE_REPO", "mscrnt/chatterbot")


def make_github_issue_url() -> str:
    """Build a github.com/.../issues/new URL with a prefilled title + body.

    GitHub's web UI doesn't support attaching files via URL params, so the
    body explicitly tells the streamer to drag the .cbreport they just
    downloaded into the textarea. Includes a system-info footer so the
    maintainer has versions even before the .cbreport lands."""
    from urllib.parse import urlencode

    sysinfo = (
        f"chatterbot {__version__} · "
        f"python {sys.version.split()[0]} · "
        f"{platform.system().lower()} {platform.release()} · "
        f"sqlite {sqlite3.sqlite_version}"
    )
    body = (
        "## What happened?\n\n"
        "<!-- describe the bug -->\n\n"
        "## What did you expect?\n\n"
        "<!-- what should have happened instead? -->\n\n"
        "## Steps to reproduce\n\n"
        "1. \n2. \n3. \n\n"
        "## Diagnostic bundle\n\n"
        "📎 **Please drag the `.cbreport` file you just downloaded into this "
        "textarea so it gets attached to this issue.**\n\n"
        "---\n"
        f"_{sysinfo}_\n"
    )
    params = urlencode({"title": "Bug report: ", "body": body, "labels": "bug"})
    return f"https://github.com/{ISSUE_REPO}/issues/new?{params}"


# ----------------------------- builders ----------------------------------

def _meta() -> dict[str, Any]:
    return {
        "tool": "chatterbot",
        "version": __version__,
        "git_sha": _git_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node() or "unknown",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _system_info() -> str:
    parts = [
        f"chatterbot={__version__}",
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"machine={platform.machine()}",
        f"node={platform.node() or 'unknown'}",
        f"sqlite_runtime={sqlite3.sqlite_version}",
        f"sqlite_module={sqlite3.version}",
        f"cwd={os.getcwd()}",
    ]
    return "\n".join(parts) + "\n"


def _packages() -> str:
    rows: list[str] = []
    for dist in metadata.distributions():
        try:
            rows.append(f"{dist.metadata['Name']}=={dist.version}")
        except Exception:
            continue
    rows.sort(key=str.lower)
    return "\n".join(rows) + "\n"


def _env_summary(settings: Settings) -> str:
    """List every editable + infra key with set/unset and value length only."""
    lines = ["# .env keys — values redacted, length only", ""]

    def emit(key: str, value: Any) -> str:
        if isinstance(value, bool):
            return f"{key}: {'set' if value else 'unset'} (bool)"
        if value is None or value == "":
            return f"{key}: unset"
        return f"{key}: set ({len(str(value))} chars)"

    lines.append("# editable via /settings or .env")
    for key in EDITABLE_SETTING_KEYS:
        lines.append(emit(key, getattr(settings, key)))
    lines.append("")
    lines.append("# infra (env-only)")
    for key in (
        "ollama_host",
        "ollama_port",
        "ollama_model",
        "ollama_embed_model",
        "ollama_embed_dim",
        "dashboard_host",
        "dashboard_port",
        "dashboard_basic_auth_user",
        "dashboard_basic_auth_pass",
        "db_path",
        "summarize_after_messages",
        "summarize_idle_minutes",
        "idle_sweep_interval_seconds",
        "topics_interval_minutes",
        "topics_max_messages",
        "run_mode",
    ):
        lines.append(emit(key, getattr(settings, key, None)))
    return "\n".join(lines) + "\n"


def _db_stats(settings: Settings) -> dict[str, Any]:
    db_path = Path(settings.db_path)
    if not db_path.exists():
        return {"db_path": str(db_path), "exists": False}
    out: dict[str, Any] = {"db_path": str(db_path), "exists": True}
    try:
        out["size_bytes"] = db_path.stat().st_size
    except OSError:
        out["size_bytes"] = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        tables = ["users", "messages", "notes", "events",
                  "topic_snapshots", "summarization_state",
                  "user_aliases", "app_settings", "vec_notes", "vec_messages"]
        counts: dict[str, int | None] = {}
        for t in tables:
            try:
                counts[t] = int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
            except sqlite3.Error:
                counts[t] = None
        out["row_counts"] = counts
        # Activity windows (no content).
        for label, q in (
            ("oldest_message_ts", "SELECT MIN(ts) FROM messages"),
            ("newest_message_ts", "SELECT MAX(ts) FROM messages"),
            ("newest_event_ts", "SELECT MAX(ts) FROM events"),
            ("newest_note_ts", "SELECT MAX(ts) FROM notes"),
            ("newest_topic_ts", "SELECT MAX(ts) FROM topic_snapshots"),
        ):
            try:
                row = conn.execute(q).fetchone()
                out[label] = row[0] if row else None
            except sqlite3.Error:
                out[label] = None
        conn.close()
    except sqlite3.Error as e:
        out["error"] = str(e)
    return out


def _ollama_probe(settings: Settings) -> dict[str, Any]:
    """Quick reachability probe — no payload data leaves the bundle."""
    info: dict[str, Any] = {
        "base_url": settings.ollama_base_url,
        "configured_model": settings.ollama_model,
        "configured_embed_model": settings.ollama_embed_model,
    }
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=4.0)
        info["status_code"] = r.status_code
        if r.status_code == 200:
            data = r.json()
            info["models_available"] = [
                m.get("name") for m in data.get("models", []) if isinstance(m, dict)
            ]
    except Exception as e:
        info["error"] = type(e).__name__ + ": " + str(e)
    return info


def _log_files() -> list[Path]:
    if not LOG_DIR.exists():
        return []
    # Include all *.log* files (rotated copies have .log.1, .log.2, etc.).
    files = sorted(LOG_DIR.iterdir())
    return [f for f in files if f.is_file() and f.suffix in (".log",) or ".log" in f.name]


def _tail_bytes(path: Path, n: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError:
        return b""
    if size <= n:
        return path.read_bytes()
    with path.open("rb") as f:
        f.seek(size - n)
        # Skip to next newline so we don't open mid-line.
        chunk = f.read()
        nl = chunk.find(b"\n")
        return chunk[nl + 1 :] if nl != -1 else chunk


def _recent_activity(settings: Settings) -> dict[str, Any]:
    """Opt-in extra: usernames + per-user message *counts* over the last
    24h. Still no message bodies, no note text, no events. The streamer
    explicitly opts into sharing usernames by selecting `--with-recent-activity`."""
    db_path = Path(settings.db_path)
    if not db_path.exists():
        return {"exists": False}
    out: dict[str, Any] = {"exists": True, "window_hours": 24}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT u.name, COUNT(m.id) AS msg_count
            FROM messages m JOIN users u ON u.twitch_id = m.user_id
            WHERE m.ts >= datetime('now', '-24 hours')
            GROUP BY u.twitch_id
            ORDER BY msg_count DESC
            """
        ).fetchall()
        out["per_user_message_counts"] = [dict(r) for r in rows]
        conn.close()
    except sqlite3.Error as e:
        out["error"] = str(e)
    return out


def _bundle_readme(with_recent_activity: bool) -> str:
    return f"""chatterbot diagnostic bundle
============================

Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}
Privacy mode: {'with-recent-activity (opt-in)' if with_recent_activity else 'minimal (default)'}

Files:
  meta.json          - tool version, git SHA, host, platform, Python version
  system.txt         - OS / Python / SQLite versions
  packages.txt       - installed Python packages
  env.txt            - .env keys with set/unset + value length (NO VALUES)
  db_stats.json      - row counts per table + activity window timestamps
  ollama.json        - configured Ollama URL + reachability probe
  logs/*.log         - rotating log tails (last ~1 MB each, no chat content)
{'  recent_activity.json - usernames + per-user message COUNTS over last 24h (opt-in)' if with_recent_activity else ''}

NOT included by default:
  - chat message bodies, note text, event payloads, usernames, twitch_ids
  - the SQLite database itself
  - any secrets (OAuth tokens, JWTs, basic-auth creds)

If you'd like the maintainer to debug something timing-related and you're OK
sharing usernames + their message counts (no message bodies), regenerate with
`chatterbot diagnose --with-recent-activity` or tick the box on the
/settings page before downloading.
"""
