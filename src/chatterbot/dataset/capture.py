"""Capture entry points used by the LLM provider wrappers and (slice 3)
the streamer-action repo writes.

Hot-path contract: when capture is OFF, `record_llm_call(...)` is a single
attribute access on the repo. The encrypt + compress + write pipeline only
runs when (a) the toggle is on AND (b) the DEK has been unlocked into
process memory.

Capture is async-friendly: callers `await record_llm_call(...)` from
async code and we hop the encrypt+write into `asyncio.to_thread` so we
never block the event loop on disk I/O. Failures inside capture are
*never* propagated to the LLM call — they log + drop. The dataset is a
side-effect; missing one event must never cost the streamer a real LLM
response.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

# NOTE: no module-top imports of `cipher` or `zstandard`. capture.py is
# imported by `chatterbot.dataset.cli` (specifically `cmd_info`) which
# must be parseable without the optional `dataset` extra installed.
# Heavy / extra-only imports live inside the helpers that need them.
from .storage import ShardWriter

if TYPE_CHECKING:
    from ..repo import ChatterRepo

logger = logging.getLogger(__name__)


# Schema version code knows about. Bump when the on-disk event dict
# loses or renames a field; new fields with sensible defaults don't need
# a bump (forward-additive). Stored alongside every row in
# dataset_events.schema_version.
CAPTURE_SCHEMA_VERSION = 1

# Event kinds. Slice 1 only emits LLM_CALL; the others are wired in
# later slices but the constant lives here so the writer is one place
# and the schema is documented end-to-end.
EVENT_LLM_CALL = "llm_call"
EVENT_STREAMER_ACTION = "streamer_action"
EVENT_CONTEXT_SNAPSHOT = "context_snapshot"


# Stable kind identifiers for STREAMER_ACTION events. Anything that
# the streamer does on the dashboard that the dataset reader might
# care about gets one of these. Pinning the strings here means a
# future export pipeline can filter on them without grepping
# production code for the literal.
ACTION_KIND_INSIGHT_STATE = "insight_state"          # dismiss / addressed / snooze / pin / skip
ACTION_KIND_NOTE = "note"                            # add / update / delete
ACTION_KIND_SUBJECT = "engaging_subject"             # reject one
ACTION_KIND_SUBJECT_BLOCKLIST = "engaging_subject_blocklist"  # clear all


# Process-wide singleton — one ShardWriter, guarded by a lock so the
# (rare) parallel capture writes from different background tasks
# serialize cleanly. Lazy-initialised on first capture so no-capture
# processes never touch disk.
_writer_lock = threading.Lock()
_writer: ShardWriter | None = None


def _get_writer(data_root: Path) -> ShardWriter:
    global _writer
    with _writer_lock:
        if _writer is None:
            _writer = ShardWriter(data_root)
        return _writer


def reset_writer() -> None:
    """Test-only: drop the singleton so a fresh data_root is picked up.
    Closes the active file handle if one is open."""
    global _writer
    with _writer_lock:
        if _writer is not None:
            _writer.close()
        _writer = None


# ---- compression ----


def _compress(payload: bytes) -> bytes:
    """zstd level-3 — good ratio at ~500 MB/s. Level 3 is the zstd
    library's default and well-balanced for this shape (mostly small
    JSON repeating keys)."""
    import zstandard as zstd
    return zstd.ZstdCompressor(level=3).compress(payload)


def _decompress(payload: bytes) -> bytes:
    import zstandard as zstd
    return zstd.ZstdDecompressor().decompress(payload)


# ---- public capture API ----


def record_streamer_action_safe(
    repo: "ChatterRepo | None",
    *,
    kind: str,
    item_key: str,
    action: str,
    note: str | None = None,
) -> None:
    """Sync capture helper for STREAMER_ACTION events. Hooks at the
    repo / insights mutation chokepoints (set_insight_state, note
    CRUD, reject_subject, clear_subject_blocklist) call this right
    after the SQL write succeeds.

    Sync on purpose — streamer actions fire at human cadence
    (a handful per minute at most), so the encrypt-and-write doesn't
    need to leave the calling thread. Avoids the "fire-and-forget
    asyncio.create_task from a sync method" footgun (no event loop
    in CLI / TUI / sync test contexts).

    Hot-path no-op when:
      - repo is None (capture not wired into this repo)
      - dataset_capture_enabled() is false
      - DEK isn't loaded into process memory
    Errors swallowed at debug level — capture must never break the
    streamer-facing mutation it's observing.
    """
    if repo is None:
        return
    try:
        if not repo.dataset_capture_enabled():
            return
        dek = repo.dataset_dek()
        if dek is None:
            _warn_no_dek_once()
            return
        payload = {
            "v": CAPTURE_SCHEMA_VERSION,
            "kind": EVENT_STREAMER_ACTION,
            "action_kind": kind,
            "item_key": item_key,
            "action": action,
            "note": note,
        }
        _write_event_sync(repo, dek, EVENT_STREAMER_ACTION, payload)
    except Exception:
        logger.debug(
            "dataset capture: streamer-action write failed for kind=%r action=%r",
            kind, action, exc_info=True,
        )


async def record_llm_call_safe(
    repo: "ChatterRepo | None",
    *,
    call_site: str,
    model_id: str,
    provider: str,
    system_prompt: str | None,
    prompt: str,
    response_text: str,
    response_schema_name: str,
    num_ctx: int | None,
    num_predict: int | None,
    think: bool,
    latency_ms: int,
    error: str | None = None,
    referenced_user_ids: list[str] | None = None,
) -> None:
    """Drop-in wrapper for `record_llm_call` that swallows errors and
    no-ops when `repo is None`. Provider clients call this from their
    `generate_structured` finally-block so the capture path is one
    line per provider and can never break the LLM call.

    Three things this guarantees the caller doesn't have to:
      1. `repo is None` (capture never attached) → silent return.
      2. Any exception inside the capture pipeline → log at DEBUG and
         continue. Capture is a side-effect; failures are non-fatal.
      3. The actual capture call is `await`ed only when capture is
         going to do something — saves an event-loop hop in the
         common no-capture-attached case.

    `referenced_user_ids` lists every chatter the prompt references
    by user_id. The capture pipeline drops the event entirely if any
    listed user has `users.opt_out=1`, so opted-out chatters never
    enter the dataset on disk. Default `None` means "I don't know /
    don't care" — capture proceeds unconditionally. Most call sites
    that build prompts from chatter content already filter on
    opt_out=0 at SQL query time; this is defense in depth.
    """
    if repo is None:
        return
    try:
        await record_llm_call(
            repo,
            call_site=call_site,
            model_id=model_id,
            provider=provider,
            system_prompt=system_prompt,
            prompt=prompt,
            response_text=response_text,
            response_schema_name=response_schema_name,
            num_ctx=num_ctx,
            num_predict=num_predict,
            think=think,
            latency_ms=latency_ms,
            error=error,
            referenced_user_ids=referenced_user_ids,
        )
    except Exception:
        # Capture must never propagate. The capture pipeline already
        # logs at WARNING when individual writes fail; this is the
        # last-resort net for everything else (import errors, repo
        # disconnects, etc).
        logger.debug("dataset capture: record_llm_call_safe swallowed", exc_info=True)


async def record_llm_call(
    repo: "ChatterRepo",
    *,
    call_site: str,
    model_id: str,
    provider: str,
    system_prompt: str | None,
    prompt: str,
    response_text: str,
    response_schema_name: str,
    num_ctx: int | None,
    num_predict: int | None,
    think: bool,
    latency_ms: int,
    error: str | None = None,
    referenced_user_ids: list[str] | None = None,
) -> None:
    """Persist one LLM_CALL event. Hot-path no-op when capture is off.

    `call_site` is a free-form identifier we pass through from the
    caller (e.g. "summarizer.note_extraction") so the export bundle can
    say which prompt site each row came from without parsing prompts.

    `provider` is "ollama" / "anthropic" / "openai" — slice 1 only
    wires Ollama. The schema is uniform across providers.

    Failures here log + drop. Capture must never break the LLM call.
    """
    # Hot-path gate. `dataset_capture_enabled()` reads the cached
    # app_settings dict — single dict lookup. `dataset_dek` is a plain
    # attribute on the repo; None when the streamer hasn't unlocked.
    if not repo.dataset_capture_enabled():
        return
    dek = repo.dataset_dek()
    if dek is None:
        # Toggle is on but the DEK isn't in memory. Log once per process
        # so we don't spam — the streamer either hasn't unlocked yet
        # (env var not set) or the unlock failed silently.
        _warn_no_dek_once()
        return

    # Opt-out filter at capture time, not export time. A chatter who
    # opts out today should never have a prompt mentioning them
    # written to encrypted disk — even with a valid DEK, the act of
    # capturing them violates the consent we promised. Cheap to skip
    # here; expensive to scan-and-rewrite shards later.
    if referenced_user_ids:
        try:
            if await asyncio.to_thread(repo.any_opted_out, referenced_user_ids):
                logger.debug(
                    "dataset capture: dropped %s event — referenced an "
                    "opted-out user", call_site,
                )
                return
        except Exception:
            # If the opt-out check itself errors, fail closed (drop the
            # event). Better to lose one row than to capture an
            # opted-out user against the streamer's consent.
            logger.exception(
                "dataset capture: opt-out check failed for call_site=%r — "
                "dropping event defensively", call_site,
            )
            return

    payload = {
        "v": CAPTURE_SCHEMA_VERSION,
        "kind": EVENT_LLM_CALL,
        "call_site": call_site,
        "provider": provider,
        "model_id": model_id,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "response_text": response_text,
        "response_schema_name": response_schema_name,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "think": bool(think),
        "latency_ms": int(latency_ms),
        "error": error,
        # List of user_ids the caller declared the prompt references.
        # Persisted so a future redaction pass can scrub specific
        # chatters from old captures without re-running the original
        # opt-out check.
        "referenced_user_ids": list(referenced_user_ids or []),
    }

    try:
        await asyncio.to_thread(_write_event_sync, repo, dek, EVENT_LLM_CALL, payload)
    except Exception:
        logger.exception("dataset capture: write failed for call_site=%r", call_site)


_warned_no_dek = False


def _warn_no_dek_once() -> None:
    global _warned_no_dek
    if _warned_no_dek:
        return
    _warned_no_dek = True
    logger.warning(
        "dataset capture is enabled but no DEK is loaded — events are "
        "being dropped. Set CHATTERBOT_DATASET_PASSPHRASE in the "
        "process environment, or run `chatterbot dataset setup` first."
    )


# ---- sync write path (runs in a worker thread) ----


def _write_event_sync(
    repo: "ChatterRepo", dek: bytes, kind: str, payload: dict[str, Any],
) -> None:
    """Serialise → compress → encrypt → append to shard → write index row.
    Runs on a worker thread so the asyncio loop stays responsive."""
    from . import cipher  # lazy — keeps capture.py importable without the dataset extra
    ts_iso = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    payload_with_ts = {"ts": ts_iso, **payload}
    raw = json.dumps(payload_with_ts, separators=(",", ":")).encode("utf-8")
    compressed = _compress(raw)
    nonce, ct = cipher.encrypt_event(dek, compressed, ts_iso=ts_iso)

    data_root = Path(repo.db_path).parent
    writer = _get_writer(data_root)
    shard_path, offset, length = writer.append(nonce, ct)

    # Index row. Path is stored relative to the data dir so the index
    # stays portable across machines (export hands the receiver both
    # the shard files and the index).
    try:
        rel_path = shard_path.relative_to(data_root)
    except ValueError:
        rel_path = shard_path
    repo.insert_dataset_event(
        ts=ts_iso,
        event_kind=kind,
        shard_path=str(rel_path),
        byte_offset=offset,
        byte_length=length,
        schema_version=CAPTURE_SCHEMA_VERSION,
    )


# ---- read path (used by export / verify in slice 1's CLI) ----


def decrypt_event(
    dek: bytes, ts_iso: str, nonce: bytes, ciphertext: bytes,
) -> dict[str, Any]:
    """Inverse of the write pipeline: AES-GCM decrypt → zstd decompress
    → JSON parse. Used by the export and verify CLIs."""
    from . import cipher  # lazy — see _write_event_sync
    plaintext = cipher.decrypt_event(dek, nonce, ciphertext, ts_iso=ts_iso)
    decompressed = _decompress(plaintext)
    return json.loads(decompressed.decode("utf-8"))


# ---- safe accessors / cleanup ----


def close_writer() -> None:
    """Flush + close the shared writer's file handle. Called from the bot
    / dashboard shutdown paths so no captured data is lost in a clean
    exit. Safe to call when the writer was never opened."""
    global _writer
    with _writer_lock:
        if _writer is not None:
            _writer.close()
