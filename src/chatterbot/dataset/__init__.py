"""Personal training-dataset capture (opt-in).

Off by default. When the streamer runs `chatterbot dataset setup`, a 32-byte
data-encryption key (DEK) is generated, wrapped under a passphrase-derived
key (Argon2id → KEK), and the wrapped key is persisted in `app_settings`.
The plaintext DEK is never written to disk.

When capture is enabled (`dataset_capture_enabled=true` AND the DEK has been
unlocked into process memory via `repo.set_dataset_dek`), the LLM provider
wrappers and the streamer-action repo writes record one event per touch:

  - LLM_CALL       — every `generate_structured()` invocation: prompt,
                     response, model_id, latency, schema name, call_site.
  - STREAMER_ACTION — every `set_insight_state` / `reject_subject` /
                     note CRUD touch (slice 3 — not yet wired in slice 1).
  - CONTEXT_SNAPSHOT — periodic snapshot of the recent message / transcript
                     window so bundles are self-contained (slice 5).

Each event is JSON-serialised, zstd-compressed, then AES-GCM-encrypted with
a fresh nonce; only the ciphertext lives on disk. The dashboard's `/dataset`
status page (slice 4) and the `chatterbot dataset export` CLI walk the
ciphertext index to produce a portable `.cbds` bundle for fine-tuning later.

Slice 1 wires:
  - cipher.py   — Argon2id KDF + AES-GCM wrap/unwrap helpers
  - storage.py  — append-only encrypted shard writer + sqlite index
  - capture.py  — `record_llm_call(...)` entry point + opt-in gate

The actual call-site wiring lives in `llm/ollama_client.py` (one wrapped
call) and the CLI commands live in `main.py`.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# Lazy-import facade — these modules pull in `cryptography` and `zstandard`,
# which are extra deps. Importing this package without the `dataset` extra
# installed is fine as long as nothing actually calls into the modules.
__all__ = ["cipher", "storage", "capture", "try_unlock_at_startup"]


# Env var the bot/dashboard read at startup to unlock the DEK without
# prompting interactively. Slice 1 ships with this as the only unlock
# path; slice 4 adds an interactive dashboard prompt that delivers the
# DEK via the existing internal-notify channel.
PASSPHRASE_ENV_VAR = "CHATTERBOT_DATASET_PASSPHRASE"


def try_unlock_at_startup(repo, llm) -> bool:
    """Best-effort unlock during bot/dashboard startup.

    Order of checks (each failure logs + returns False):
      1. dataset_capture_enabled toggle is on
      2. dataset_key_wrapped exists in app_settings
      3. CHATTERBOT_DATASET_PASSPHRASE is set in the environment
      4. unwrap_dek succeeds (passphrase is correct)
      5. fingerprint matches the recorded value (sanity check that the
         streamer didn't paste in a wrapped key from a different DEK)

    On success: installs the DEK on `repo`, calls
    `llm.attach_dataset_capture(repo)` if the client supports it, and
    returns True. Any failure is non-fatal — the bot keeps running with
    capture silently off.

    Importing this function does NOT pull in `cryptography` / `zstandard`
    — those load only inside the body, on the success path. A streamer
    who hasn't installed the `dataset` extra and hasn't enabled capture
    pays nothing.
    """
    try:
        if not repo.dataset_capture_enabled():
            return False
    except Exception:
        logger.exception("dataset: capture-enabled check failed")
        return False

    wrapped_raw = repo.get_app_setting("dataset_key_wrapped")
    if not wrapped_raw:
        logger.warning(
            "dataset capture is enabled but no wrapped DEK is stored — "
            "run `chatterbot dataset setup` to initialise it."
        )
        return False

    passphrase = os.environ.get(PASSPHRASE_ENV_VAR, "")
    if not passphrase:
        logger.warning(
            "dataset capture is enabled but %s is not set in the "
            "environment — events will be dropped until the DEK is "
            "unlocked.", PASSPHRASE_ENV_VAR,
        )
        return False

    try:
        from . import cipher
    except ImportError:
        logger.warning(
            "dataset capture is enabled but the `dataset` extra is not "
            "installed (`uv sync --extra dataset`). Capture stays off."
        )
        return False

    try:
        wrapped = cipher.WrappedDEK.from_json(wrapped_raw)
    except Exception:
        logger.exception("dataset: stored wrapped DEK is malformed")
        return False

    try:
        dek = cipher.unwrap_dek(wrapped, passphrase)
    except Exception:
        logger.error(
            "dataset: passphrase did not unwrap the stored DEK — "
            "check %s. Capture stays off.", PASSPHRASE_ENV_VAR,
        )
        return False

    expected_fp = (repo.get_app_setting("dataset_key_fingerprint") or "").strip()
    actual_fp = cipher.fingerprint_dek(dek)
    if expected_fp and expected_fp != actual_fp:
        logger.error(
            "dataset: DEK fingerprint mismatch (expected %s, got %s) — "
            "the wrapped key may have been replaced. Capture stays off.",
            expected_fp, actual_fp,
        )
        return False

    repo.set_dataset_dek(dek)
    if hasattr(llm, "attach_dataset_capture"):
        llm.attach_dataset_capture(repo)
    logger.info(
        "dataset capture: unlocked (fingerprint %s) — events will be "
        "encrypted to data/dataset/shards/", actual_fp,
    )
    return True
