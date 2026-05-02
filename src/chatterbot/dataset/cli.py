"""CLI subcommands for the personal-dataset capture system.

Slice 1 commands:
  - `chatterbot dataset setup`   — generate a DEK, wrap it under a
                                    streamer-supplied passphrase, persist
                                    the wrapped form. Prints a recovery
                                    string the streamer can save offline.
  - `chatterbot dataset enable`  — flip dataset_capture_enabled=true
                                    (no-op if already on).
  - `chatterbot dataset disable` — flip dataset_capture_enabled=false.
                                    Does NOT delete the DEK or any past
                                    events; flipping back on resumes
                                    capture under the same key.
  - `chatterbot dataset info`    — read-only snapshot: enabled/unlocked/
                                    fingerprint/event count by kind.
                                    Does NOT decrypt anything.
  - `chatterbot dataset export`  — produce a `.cbds` bundle containing
                                    every event in the index, decrypted
                                    once with the streamer's passphrase
                                    and then re-encrypted under a fresh
                                    bundle key (also passphrase-wrapped).
  - `chatterbot dataset verify`  — decrypt a `.cbds` bundle in-place and
                                    confirm every record parses cleanly.
                                    Prints the event-kind histogram.

Future slices add `destroy`, `recover`, `redact`, dashboard surfaces.

Each command is a small function called from `main.py`'s argparse
dispatch. Keeping them here (not in main.py) means the lazy `dataset`
extra deps don't load on `chatterbot bot` / `chatterbot dashboard`
unless the streamer explicitly invokes a dataset command.
"""

from __future__ import annotations

import getpass
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# NOTE: do NOT import `cipher` (or anything that pulls `cryptography` /
# `zstandard`) at module top — main.py registers this CLI's subparser
# on every `chatterbot ...` run, including `bot` and `dashboard`. The
# `dataset` extra is optional, so a base install must be able to *parse*
# this module without those deps present. Each command body lazy-imports
# what it actually needs, and `_require_dataset_extra` gives a single
# friendly error when the deps are missing.
logger = logging.getLogger(__name__)


def _require_dataset_extra() -> None:
    """Friendly error when a dataset subcommand runs but the optional
    `dataset` extra (cryptography + zstandard) isn't installed. Called
    at the top of every command that actually touches crypto or
    compression. Exits the process — there's no useful fallback."""
    missing = []
    try:
        import cryptography  # noqa: F401
    except ImportError:
        missing.append("cryptography")
    try:
        import zstandard  # noqa: F401
    except ImportError:
        missing.append("zstandard")
    if missing:
        print(
            "the dataset capture system needs the optional `dataset` extra:\n"
            f"  missing: {', '.join(missing)}\n"
            "install with:\n"
            "  uv sync --extra dataset\n"
            "or:\n"
            "  pip install chatterbot[dataset]",
            file=sys.stderr,
        )
        sys.exit(2)


# ---- helpers ----


def _open_repo(settings):
    """Build a ChatterRepo against the configured DB. Imported here (not
    at module import time) so `chatterbot dataset --help` doesn't require
    the full project to spin up."""
    from ..repo import ChatterRepo
    return ChatterRepo(
        settings.db_path,
        embed_dim=settings.ollama_embed_dim,
        use_int8_embeddings=settings.use_int8_embeddings,
    )


def _prompt_passphrase(prompt: str = "passphrase: ", *, confirm: bool = False) -> str:
    """Read a passphrase from stdin without echoing. With `confirm=True`,
    re-prompts and verifies a match."""
    p = getpass.getpass(prompt)
    if not p:
        print("error: empty passphrase", file=sys.stderr)
        sys.exit(2)
    if confirm:
        again = getpass.getpass("confirm passphrase: ")
        if p != again:
            print("error: passphrases don't match", file=sys.stderr)
            sys.exit(2)
    return p


# ---- setup ----


def cmd_setup(settings, *, force: bool = False) -> int:
    """Initialise the dataset DEK + wrap it with a passphrase.

    Idempotent unless `--force`: if a wrapped DEK already exists, refuse
    to overwrite (would lock the streamer out of past events). The
    streamer can still rotate via a later `dataset rotate` command (out
    of scope for slice 1)."""
    _require_dataset_extra()
    from . import cipher
    repo = _open_repo(settings)
    try:
        existing = repo.get_app_setting("dataset_key_wrapped")
        if existing and not force:
            print(
                "dataset is already initialised (a wrapped DEK is stored).\n"
                "  fingerprint: " + (repo.get_app_setting("dataset_key_fingerprint") or "?") + "\n"
                "Pass --force to wipe and start over (this WILL lock you out of any "
                "events captured under the previous key).",
                file=sys.stderr,
            )
            return 1

        passphrase = _prompt_passphrase("new dataset passphrase: ", confirm=True)
        print("\nDeriving key — this takes a few seconds...", flush=True)

        dek = cipher.generate_dek()
        wrapped = cipher.wrap_dek(dek, passphrase)
        fingerprint = cipher.fingerprint_dek(dek)

        repo.set_app_setting("dataset_key_wrapped", wrapped.to_json())
        repo.set_app_setting("dataset_key_fingerprint", fingerprint)
        # Sensible defaults — capture is OFF by default. Streamer flips
        # it via `dataset enable` after this so they don't accidentally
        # start collecting before they're ready.
        if repo.get_app_setting("dataset_capture_enabled") is None:
            repo.set_app_setting("dataset_capture_enabled", "false")

        recovery = cipher.dek_to_recovery_string(dek)
        print(
            f"\nDEK initialised — fingerprint {fingerprint}.\n\n"
            "RECOVERY STRING (save this somewhere offline — a password "
            "manager, printed page, etc):\n\n"
            f"  {recovery}\n\n"
            "If you forget your passphrase, this string lets you recover "
            "your captured data. Without either, past events are "
            "permanently unrecoverable — that's the encryption working "
            "as intended.\n\n"
            "Next steps:\n"
            "  1. Save the recovery string above somewhere safe.\n"
            "  2. Run `chatterbot dataset enable` to start capturing.\n"
            "  3. Set CHATTERBOT_DATASET_PASSPHRASE in your bot/dashboard "
            "environment so they can unlock the DEK at startup.\n"
        )
        return 0
    finally:
        repo.close()


# ---- enable / disable ----


def cmd_enable(settings) -> int:
    repo = _open_repo(settings)
    try:
        if not repo.get_app_setting("dataset_key_wrapped"):
            print(
                "no wrapped DEK is stored — run `chatterbot dataset setup` first.",
                file=sys.stderr,
            )
            return 1
        repo.set_app_setting("dataset_capture_enabled", "true")
        print("dataset capture: enabled.")
        print(
            "Reminder: the bot/dashboard processes need "
            "CHATTERBOT_DATASET_PASSPHRASE in their environment to actually "
            "unlock the DEK and start writing events."
        )
        return 0
    finally:
        repo.close()


def cmd_disable(settings) -> int:
    repo = _open_repo(settings)
    try:
        repo.set_app_setting("dataset_capture_enabled", "false")
        print(
            "dataset capture: disabled. Past events are still encrypted "
            "on disk and can be exported; flipping back on resumes capture."
        )
        return 0
    finally:
        repo.close()


# ---- info ----


def cmd_info(settings) -> int:
    """Read-only status — never decrypts anything. Safe to share output.
    Doesn't require the dataset extra: walks SQLite only, which is
    always available."""
    repo = _open_repo(settings)
    try:
        enabled = repo.dataset_capture_enabled()
        wrapped_present = bool(repo.get_app_setting("dataset_key_wrapped"))
        fingerprint = repo.get_app_setting("dataset_key_fingerprint") or "—"

        total = repo.dataset_event_count()
        from .capture import (
            EVENT_LLM_CALL,
            EVENT_STREAMER_ACTION,
            EVENT_CONTEXT_SNAPSHOT,
        )
        per_kind = {
            EVENT_LLM_CALL: repo.dataset_event_count(kind=EVENT_LLM_CALL),
            EVENT_STREAMER_ACTION: repo.dataset_event_count(kind=EVENT_STREAMER_ACTION),
            EVENT_CONTEXT_SNAPSHOT: repo.dataset_event_count(kind=EVENT_CONTEXT_SNAPSHOT),
        }

        # Total bytes — sum byte_length over the index (cheap; one SQL
        # query). Doesn't include the file headers, but those are tiny.
        with repo._cursor() as cur:  # noqa: SLF001 — small CLI helper
            cur.execute("SELECT COALESCE(SUM(byte_length), 0) AS total_bytes FROM dataset_events")
            row = cur.fetchone()
            total_bytes = int(row["total_bytes"]) if row else 0

        print("Personal training-dataset status:")
        print(f"  capture enabled : {'yes' if enabled else 'no'}")
        print(f"  wrapped DEK     : {'present' if wrapped_present else 'NOT INITIALISED'}")
        print(f"  fingerprint     : {fingerprint}")
        print(f"  events          : {total:,}")
        for kind, n in per_kind.items():
            if n:
                print(f"    {kind:18s}: {n:,}")
        print(f"  encrypted bytes : {total_bytes:,}")
        return 0
    finally:
        repo.close()


# ---- export ----


def cmd_export(
    settings, out_path: Path, *,
    since: str | None = None,
    until: str | None = None,
    redact_users: bool = False,
) -> int:
    """Decrypt every indexed event under the streamer's passphrase, then
    repackage them into a single passphrase-protected `.cbds` bundle.

    Bundle layout (tar):
      manifest.json       — cleartext: schema_version, date range, event
                            counts, KDF params, fingerprint, ts
      payload.bin         — AES-GCM(bundle_dek, zstd(NDJSON of events))
      bundle_dek.wrapped  — bundle_dek wrapped under streamer's KEK

    Cleartext manifest is intentional: a fine-tune service can inspect
    the bundle's shape without decrypting. The events themselves stay
    encrypted until the receiver enters the passphrase.

    `redact_users=True` runs the export-time redactor: every chatter
    referenced via `referenced_user_ids` (or appearing in snapshot
    messages) is replaced with a stable per-bundle anon token like
    `<USER_001>`. Manifest gets `redacted: true` so a downstream
    consumer can tell anonymised bundles from raw ones."""
    _require_dataset_extra()
    from . import cipher
    from .storage import read_record
    repo = _open_repo(settings)
    try:
        wrapped_raw = repo.get_app_setting("dataset_key_wrapped")
        if not wrapped_raw:
            print("no wrapped DEK — run `dataset setup` first.", file=sys.stderr)
            return 1
        wrapped = cipher.WrappedDEK.from_json(wrapped_raw)
        passphrase = _prompt_passphrase("dataset passphrase: ")
        try:
            dek = cipher.unwrap_dek(wrapped, passphrase)
        except Exception:
            print("wrong passphrase.", file=sys.stderr)
            return 2

        from .capture import decrypt_event

        # Walk the index, decrypt each record, accumulate as parsed
        # dicts (so the redactor can run before serialisation if
        # requested). Two-pass — first decrypt all, then optionally
        # redact, then serialise.
        data_root = Path(repo.db_path).parent
        decoded: list[dict] = []
        kinds_count: dict[str, int] = {}
        first_ts: str | None = None
        last_ts: str | None = None
        for row in repo.iter_dataset_events(since_ts=since, until_ts=until):
            shard_path = data_root / row["shard_path"]
            # Be tolerant of slightly different relative-path roots
            # (e.g. shard_path persisted with the data dir baked in).
            if not shard_path.exists():
                shard_path = Path(row["shard_path"])
            try:
                rec = read_record(shard_path, row["byte_offset"], row["byte_length"])
            except FileNotFoundError:
                logger.warning("shard missing for event id=%s: %s", row["id"], shard_path)
                continue
            try:
                payload = decrypt_event(dek, row["ts"], rec.nonce, rec.ciphertext)
            except Exception:
                logger.warning("decrypt failed for event id=%s — skipping", row["id"])
                continue
            decoded.append(payload)
            kinds_count[row["event_kind"]] = kinds_count.get(row["event_kind"], 0) + 1
            if first_ts is None:
                first_ts = row["ts"]
            last_ts = row["ts"]

        if not decoded:
            print("no events in the requested range — nothing to export.", file=sys.stderr)
            return 1

        # Optional redaction pass — replace every covered chatter
        # name with a stable `<USER_NNN>` token. `build_plan_for_export`
        # does both passes (explicit user_ids + @-mention sweep
        # against user_aliases) so chatters that appear in chat
        # PROSE — not just in declared metadata — also get redacted.
        redaction_meta: dict | None = None
        if redact_users:
            from . import redactor as _redactor
            plan = _redactor.build_plan_for_export(repo, decoded)
            decoded = [_redactor.redact_event(plan, ev) for ev in decoded]
            redaction_meta = {
                "applied": True,
                # `user_names_with_at_mentions` documents the
                # expanded coverage so a downstream consumer can
                # tell this bundle apart from one redacted with the
                # narrower v1 strategy.
                "strategy": "user_names_with_at_mentions",
                "anon_user_count": len(plan.id_to_token),
            }

        events: list[bytes] = [
            json.dumps(ev, separators=(",", ":")).encode("utf-8")
            for ev in decoded
        ]
        ndjson = b"\n".join(events)
        import zstandard as zstd
        compressed = zstd.ZstdCompressor(level=10).compress(ndjson)

        bundle_dek = cipher.generate_dek()
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import os as _os
        bundle_nonce = _os.urandom(12)
        bundle_ct = AESGCM(bundle_dek).encrypt(
            bundle_nonce, compressed, associated_data=b"chatterbot/dataset/bundle/v1",
        )
        bundle_wrapped = cipher.wrap_dek(bundle_dek, passphrase)

        manifest = {
            "format": "cbds",
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "first_event_ts": first_ts,
            "last_event_ts": last_ts,
            "event_count": sum(kinds_count.values()),
            "event_kinds": kinds_count,
            "fingerprint": cipher.fingerprint_dek(dek),
            "kdf_params": wrapped.kdf_params,
            # Redaction status — "redacted: false" by default so a
            # downstream consumer's policy can refuse-to-train when
            # the field is missing OR explicitly false. `meta` carries
            # the strategy details when redaction was applied.
            "redacted": bool(redact_users),
            "redaction": redaction_meta,
        }

        # Single tar containing manifest (cleartext) + payload (encrypted)
        # + bundle_dek.wrapped (KEK-wrapped, lets the receiver decrypt).
        import io
        import tarfile
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out_path, "w") as tar:
            def _add(name: str, data: bytes):
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

            _add("manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
            _add("bundle_dek.wrapped", bundle_wrapped.to_json().encode("utf-8"))
            _add("payload.nonce", bundle_nonce)
            _add("payload.bin", bundle_ct)

        size = out_path.stat().st_size
        print(
            f"exported {len(events):,} events to {out_path} ({size:,} bytes).\n"
            f"  date range: {first_ts} → {last_ts}\n"
            f"  kinds     : {kinds_count}\n"
            f"  fingerprint: {manifest['fingerprint']}"
        )
        return 0
    finally:
        repo.close()


# ---- verify ----


def cmd_verify(bundle_path: Path) -> int:
    """Open a `.cbds` bundle and decrypt + parse every event. Prints the
    histogram. Exits non-zero if anything fails to decrypt or parse."""
    _require_dataset_extra()
    from . import cipher
    if not bundle_path.exists():
        print(f"no such bundle: {bundle_path}", file=sys.stderr)
        return 1
    import tarfile

    with tarfile.open(bundle_path, "r") as tar:
        names = tar.getnames()
        required = {"manifest.json", "bundle_dek.wrapped", "payload.nonce", "payload.bin"}
        missing = required - set(names)
        if missing:
            print(f"bundle missing required entries: {missing}", file=sys.stderr)
            return 2

        def _read(name: str) -> bytes:
            f = tar.extractfile(name)
            assert f is not None
            return f.read()

        manifest = json.loads(_read("manifest.json"))
        wrapped = cipher.WrappedDEK.from_json(_read("bundle_dek.wrapped").decode("utf-8"))
        nonce = _read("payload.nonce")
        ct = _read("payload.bin")

    print(f"manifest: {json.dumps(manifest, indent=2)}\n")
    passphrase = _prompt_passphrase("bundle passphrase: ")
    try:
        bundle_dek = cipher.unwrap_dek(wrapped, passphrase)
    except Exception:
        print("wrong passphrase.", file=sys.stderr)
        return 2

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    try:
        compressed = AESGCM(bundle_dek).decrypt(
            nonce, ct, associated_data=b"chatterbot/dataset/bundle/v1",
        )
    except Exception:
        print("decrypt failed — bundle is corrupted or KEK is wrong.", file=sys.stderr)
        return 2

    import zstandard as zstd
    ndjson = zstd.ZstdDecompressor().decompress(compressed)
    line_count = 0
    parsed_kinds: dict[str, int] = {}
    for line in ndjson.split(b"\n"):
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            print(f"event #{line_count} failed to parse as JSON", file=sys.stderr)
            return 3
        kind = event.get("kind", "?")
        parsed_kinds[kind] = parsed_kinds.get(kind, 0) + 1
        line_count += 1

    print(
        f"verified {line_count:,} events.\n"
        f"  kinds (parsed): {parsed_kinds}\n"
        f"  matches manifest: {parsed_kinds == manifest.get('event_kinds')}"
    )
    return 0


# ---- argparse plumbing ----


def register_subcommands(subparsers) -> None:
    """Register the `setup / enable / disable / info / export / verify`
    subcommands onto the given subparsers object. Called from main.py
    after it constructs an isolated `chatterbot dataset` parser —
    keeping registration here means the command list lives in one
    place even if the dispatch story changes later."""
    p_setup = subparsers.add_parser("setup", help="initialise + wrap a DEK with a passphrase")
    p_setup.add_argument(
        "--force", action="store_true",
        help="overwrite an existing DEK (DESTROYS access to past events)",
    )
    p_setup.set_defaults(_dataset_handler=lambda args, settings: cmd_setup(settings, force=args.force))

    p_enable = subparsers.add_parser("enable", help="flip dataset_capture_enabled=true")
    p_enable.set_defaults(_dataset_handler=lambda args, settings: cmd_enable(settings))

    p_disable = subparsers.add_parser("disable", help="flip dataset_capture_enabled=false")
    p_disable.set_defaults(_dataset_handler=lambda args, settings: cmd_disable(settings))

    p_info = subparsers.add_parser("info", help="read-only status (never decrypts)")
    p_info.set_defaults(_dataset_handler=lambda args, settings: cmd_info(settings))

    p_export = subparsers.add_parser("export", help="export an encrypted .cbds bundle")
    p_export.add_argument(
        "--out", type=Path, required=True,
        help="output bundle path (e.g. ./my-dataset.cbds)",
    )
    p_export.add_argument("--since", type=str, default=None, help="ISO-UTC lower bound")
    p_export.add_argument("--until", type=str, default=None, help="ISO-UTC upper bound")
    p_export.add_argument(
        "--redact-users", action="store_true",
        help="anonymise chatter usernames in the bundle (replaces "
             "names with stable <USER_NNN> tokens, marks the manifest "
             "as redacted)",
    )
    p_export.set_defaults(_dataset_handler=lambda args, settings: cmd_export(
        settings, args.out,
        since=args.since, until=args.until,
        redact_users=args.redact_users,
    ))

    p_verify = subparsers.add_parser("verify", help="decrypt + validate a .cbds bundle")
    p_verify.add_argument("bundle", type=Path, help="path to a .cbds bundle")
    p_verify.set_defaults(_dataset_handler=lambda args, settings: cmd_verify(args.bundle))


def dispatch(args, settings) -> int:
    """Run the resolved subcommand handler. Returns the exit code."""
    handler = getattr(args, "_dataset_handler", None)
    if handler is None:
        print("no dataset subcommand specified", file=sys.stderr)
        return 2
    return int(handler(args, settings) or 0)
