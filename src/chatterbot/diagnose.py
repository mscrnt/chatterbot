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
    anonymize_recent_activity: bool = False,
) -> Path:
    """Write the bundle. Returns out_path.

    `with_recent_activity=True` adds the `recent_activity.json` slice
    (last 24h chatters + per-user message counts).
    `anonymize_recent_activity=True` runs the dataset redactor over
    that file so usernames become stable `<USER_NNN>` tokens. Lets
    the streamer share activity SHAPE without identities. No-op
    when `with_recent_activity=False`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(_meta(), indent=2))
        zf.writestr("system.txt", _system_info())
        zf.writestr("packages.txt", _packages())
        zf.writestr("env.txt", _env_summary(settings))
        zf.writestr("db_stats.json", json.dumps(_db_stats(settings), indent=2))
        zf.writestr("ollama.json", json.dumps(_ollama_probe(settings), indent=2))
        # Dashboard-managed app_settings — filtered to drop secrets.
        # Catches the dataset_*, engaging_subjects_*, retention, and
        # other dashboard knobs that don't live on the Settings class.
        zf.writestr(
            "app_settings.json",
            json.dumps(_app_settings_dump(settings), indent=2),
        )
        # Insight-card state shape (counts only, no content). Tells
        # the maintainer "the streamer has been clicking through
        # cards" without leaking what the cards said.
        zf.writestr(
            "insight_states.json",
            json.dumps(_insight_states_summary(settings), indent=2),
        )
        # Personal training-dataset capture status (counts only,
        # never decrypts). Surfaces "capture is on but DEK isn't
        # loaded" cases that would otherwise need log digging.
        zf.writestr(
            "dataset.json",
            json.dumps(_dataset_status_dump(settings), indent=2),
        )

        for log_file in _log_files():
            tail = _tail_bytes(log_file, LOG_TAIL_BYTES)
            zf.writestr(f"logs/{log_file.name}", tail)

        if with_recent_activity:
            activity = _recent_activity(settings)
            if anonymize_recent_activity:
                activity = _anonymize_recent_activity(settings, activity)
            zf.writestr(
                "recent_activity.json",
                json.dumps(activity, indent=2),
            )

        zf.writestr(
            "README.txt",
            _bundle_readme(with_recent_activity, anonymize_recent_activity),
        )
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
                  "user_aliases", "app_settings", "vec_notes", "vec_messages",
                  # Dataset capture index. Helps maintainers see at a
                  # glance whether capture has been recording — count
                  # without decrypting anything.
                  "dataset_events",
                  # Insight-state history (per-card transitions) is
                  # cheap to count and surfaces dashboard usage shape.
                  "insight_states", "insight_state_history"]
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


# Substring patterns we refuse to put into `app_settings.json`. Any
# key containing one of these in its name (case-insensitive) gets
# replaced with a length-only summary. Keeps the bundle safely
# shareable on a public GitHub issue.
_APP_SETTINGS_SECRET_PATTERNS: tuple[str, ...] = (
    "secret", "token", "key", "passphrase", "dek", "wrapped",
    "password", "credential", "jwt", "oauth", "api_key",
)


def _redact_secret_value(value: str) -> str:
    """Replace a secret with a "set (N chars)" marker. Length is
    safe to share — it can hint at the SHAPE of the value (a 36-char
    JWT vs an 8-char short token) without leaking any byte."""
    return f"<redacted: {len(value)} chars>"


def _is_secret_key(key: str) -> bool:
    k = key.lower()
    return any(p in k for p in _APP_SETTINGS_SECRET_PATTERNS)


def _app_settings_dump(settings: Settings) -> dict[str, Any]:
    """Full app_settings KV dump with secrets redacted by name match.

    The dashboard writes a lot of state to `app_settings` that
    isn't on the Settings class — engaging_subjects_blocklist,
    dataset_*, chat_lag_*, retention knobs. env.txt covers Settings
    only, so this fills the gap. Filter is name-based + tolerant:
    a new "*_secret" key added later automatically gets redacted."""
    db_path = Path(settings.db_path)
    if not db_path.exists():
        return {"exists": False}
    out: dict[str, Any] = {"exists": True, "values": {}, "redacted": []}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
        for r in rows:
            k = r["key"]
            v = r["value"] or ""
            if _is_secret_key(k):
                out["values"][k] = _redact_secret_value(v)
                out["redacted"].append(k)
            else:
                # Cap value length so a runaway streamer-facts blob
                # (which can be ~4 KB) doesn't bloat the bundle. The
                # full value is in app_settings on disk anyway; we
                # only need a shape-of for diagnostics.
                if len(v) > 500:
                    out["values"][k] = v[:500] + f"...[{len(v) - 500} more chars]"
                else:
                    out["values"][k] = v
        conn.close()
    except sqlite3.Error as e:
        out["error"] = str(e)
    return out


def _insight_states_summary(settings: Settings) -> dict[str, Any]:
    """Counts of insight cards by (kind, state). No content — just
    shapes. Catches "the dashboard isn't surfacing X" reports where
    the cause turns out to be "you've dismissed every card of that
    kind already."

    `insight_state_history` is also surfaced as a per-state count
    over the last 7 days so the maintainer can see whether the
    streamer is actively engaging with cards or whether the audit
    trail is empty."""
    db_path = Path(settings.db_path)
    if not db_path.exists():
        return {"exists": False}
    out: dict[str, Any] = {"exists": True}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # Current state (live insight_states table).
        rows = conn.execute(
            "SELECT kind, state, COUNT(*) AS n FROM insight_states "
            "GROUP BY kind, state ORDER BY kind, state"
        ).fetchall()
        out["live_by_kind_state"] = [
            {"kind": r["kind"], "state": r["state"], "count": int(r["n"])}
            for r in rows
        ]
        # Last 7 days of transitions — gives the maintainer a sense
        # of cadence ("did the streamer just dismiss 200 cards?").
        rows = conn.execute(
            "SELECT state, COUNT(*) AS n FROM insight_state_history "
            "WHERE ts >= datetime('now', '-7 days') "
            "GROUP BY state ORDER BY state"
        ).fetchall()
        out["transitions_last_7d_by_state"] = [
            {"state": r["state"], "count": int(r["n"])}
            for r in rows
        ]
        conn.close()
    except sqlite3.Error as e:
        out["error"] = str(e)
    return out


def _dataset_status_dump(settings: Settings) -> dict[str, Any]:
    """Personal training-dataset capture status — counts only,
    never decrypts. Surfaces the most common debug case:
    "I enabled capture but nothing's getting saved" → maintainer
    sees `enabled: true, unlocked: false` → instant root cause:
    missing CHATTERBOT_DATASET_PASSPHRASE in the running process'
    env.

    Pure SQL + filesystem reads — doesn't import the dataset
    module so a bundle generated on an install without the
    optional `dataset` extra still produces a clean dataset.json
    instead of crashing the bundle build."""
    db_path = Path(settings.db_path)
    if not db_path.exists():
        return {"exists": False}
    out: dict[str, Any] = {"exists": True}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # Toggle + key state straight from app_settings.
        kv = {
            r["key"]: r["value"]
            for r in conn.execute(
                "SELECT key, value FROM app_settings WHERE key LIKE 'dataset_%'"
            ).fetchall()
        }
        out["enabled"] = (kv.get("dataset_capture_enabled") or "").lower() == "true"
        out["configured"] = bool(kv.get("dataset_key_wrapped"))
        out["fingerprint"] = kv.get("dataset_key_fingerprint") or ""
        out["retention_days"] = kv.get("dataset_retention_days")
        out["retention_max_mb"] = kv.get("dataset_retention_max_mb")

        # Event counts by kind + total bytes (index-only — never
        # touches shard files).
        try:
            rows = conn.execute(
                "SELECT event_kind, COUNT(*) AS n, "
                "COALESCE(SUM(byte_length), 0) AS sz "
                "FROM dataset_events GROUP BY event_kind"
            ).fetchall()
            out["events_by_kind"] = [
                {"kind": r["event_kind"], "count": int(r["n"]),
                 "encrypted_bytes": int(r["sz"])}
                for r in rows
            ]
            row = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(byte_length), 0) AS sz, "
                "MIN(ts) AS oldest, MAX(ts) AS newest FROM dataset_events"
            ).fetchone()
            out["total_events"] = int(row["n"]) if row else 0
            out["total_encrypted_bytes"] = int(row["sz"]) if row else 0
            out["oldest_event_ts"] = row["oldest"] if row else None
            out["newest_event_ts"] = row["newest"] if row else None
        except sqlite3.Error:
            # Table might not exist on a very old DB that pre-dates
            # the migration.
            out["events_by_kind"] = []
            out["total_events"] = 0
            out["total_encrypted_bytes"] = 0
        conn.close()
    except sqlite3.Error as e:
        out["error"] = str(e)

    # Shard file footprint — we can stat the directory without the
    # crypto extra installed, so this works even on a base install.
    try:
        shards_root = Path(settings.db_path).parent / "dataset" / "shards"
        if shards_root.exists():
            files = sorted(shards_root.glob("*.cbds.bin"))
            out["shard_count"] = len(files)
            out["shard_total_bytes"] = sum(f.stat().st_size for f in files)
        else:
            out["shard_count"] = 0
            out["shard_total_bytes"] = 0
    except OSError:
        pass
    return out


def _anonymize_recent_activity(
    settings: Settings, activity: dict[str, Any],
) -> dict[str, Any]:
    """Run the dataset redactor over a `_recent_activity` payload,
    replacing every chatter username with a stable `<USER_NNN>`
    token. Lets the streamer share the SHAPE of activity (volume,
    distribution, tail length) without identifying anyone.

    No-op if the dataset extra isn't installed (cryptography +
    zstandard) — the redactor module itself doesn't need crypto,
    but the import path goes through `chatterbot.dataset` which
    has the soft-fail guard. Returns the original on any error so
    a degraded bundle still ships."""
    try:
        from .dataset import redactor as _redactor
        from .repo import ChatterRepo
    except Exception:
        return activity

    rows = activity.get("per_user_message_counts") or []
    if not rows:
        return activity

    db_path = Path(settings.db_path)
    if not db_path.exists():
        return activity

    repo = None
    try:
        repo = ChatterRepo(
            str(db_path),
            embed_dim=settings.ollama_embed_dim,
            use_int8_embeddings=settings.use_int8_embeddings,
        )
        # The activity rows have name only — resolve to user_ids via
        # a small alias query so the redactor's user_aliases path can
        # cover renames.
        names = [str(r.get("name") or "") for r in rows if r.get("name")]
        if not names:
            return activity
        with repo._cursor() as cur:  # noqa: SLF001 — internal diagnose helper
            placeholders = ",".join("?" * len(names))
            cur.execute(
                f"SELECT DISTINCT user_id FROM user_aliases "
                f"WHERE name IN ({placeholders})",
                names,
            )
            user_ids = [r["user_id"] for r in cur.fetchall()]
        plan = _redactor.build_plan(repo, user_ids)
        out = dict(activity)
        out["per_user_message_counts"] = [
            {
                **r,
                "name": plan.name_to_token.get(
                    str(r.get("name") or "").lower(),
                    str(r.get("name") or ""),
                ),
            }
            for r in rows
        ]
        out["anonymized"] = True
        return out
    except Exception:
        return activity
    finally:
        if repo is not None:
            try:
                repo.close()
            except Exception:
                pass


def _bundle_readme(
    with_recent_activity: bool,
    anonymize_recent_activity: bool = False,
) -> str:
    privacy_mode = (
        "with-recent-activity" + (
            " (anonymised)" if anonymize_recent_activity else " (opt-in)"
        )
        if with_recent_activity
        else "minimal (default)"
    )
    return f"""chatterbot diagnostic bundle
============================

Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}
Privacy mode: {privacy_mode}

Files:
  meta.json            - tool version, git SHA, host, platform, Python version
  system.txt           - OS / Python / SQLite versions
  packages.txt         - installed Python packages
  env.txt              - .env / Settings keys with set/unset + value length (NO VALUES)
  db_stats.json        - row counts per table + activity window timestamps
  app_settings.json    - dashboard-managed KV (secrets redacted by name pattern)
  insight_states.json  - card-state counts (kind/state shape, no content)
  dataset.json         - personal-dataset capture status (counts only, never decrypts)
  ollama.json          - configured Ollama URL + reachability probe
  logs/*.log           - rotating log tails (last ~1 MB each, no chat content)
{'  recent_activity.json - usernames + per-user message COUNTS over last 24h (opt-in)' + (' [ANONYMISED via <USER_NNN> tokens]' if anonymize_recent_activity else '') if with_recent_activity else ''}

NOT included by default:
  - chat message bodies, note text, event payloads, usernames, twitch_ids
  - the SQLite database itself
  - any secrets (OAuth tokens, JWTs, basic-auth creds, dataset DEK / passphrase)

If you'd like the maintainer to debug something timing-related and you're OK
sharing usernames + their message counts (no message bodies), regenerate with
`chatterbot diagnose --with-recent-activity` or tick the box on the
/settings page before downloading. Pair it with the "Anonymise" checkbox
to share activity SHAPE without identities.
"""
