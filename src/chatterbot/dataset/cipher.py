"""Crypto primitives for the personal-dataset capture system.

Two-tier key hierarchy (envelope encryption):

  passphrase ──Argon2id──▶ KEK ──AES-GCM-wrap──▶ DEK
                                                  │
                                                  └─AES-GCM─▶ event ciphertext

  - DEK (32 bytes) is generated once at `chatterbot dataset setup`. Used
    directly to encrypt every captured event. Never written to disk in
    plaintext — only its KEK-wrapped form is persisted.
  - KEK is derived from the streamer's passphrase via Argon2id with
    per-install salt + KDF parameters stored in `app_settings`. Tunable
    cost — bumping `time_cost` / `memory_kib` in a future release doesn't
    invalidate existing wrapped DEKs (the wrapper stores params alongside
    the ciphertext).

Threat model defended:
  - Casual disk theft / cloud-backup leak (KEK never on disk; DEK only
    encrypted on disk).
  - Streamer accidentally screenshots their data dir on stream (file is
    ciphertext + metadata; reveals only file size and timestamp).

Out of scope:
  - Live process memory dump (DEK is in process RAM while capture is on —
    inherent to any local-encryption design).
  - Side-channel attacks against the streamer's machine.

Recovery: passphrase forgotten → optional recovery key. The CLI prints the
DEK as a base32 string at setup time and offers to redirect it to a file.
The streamer can later re-wrap that DEK under a new passphrase.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
from dataclasses import dataclass
from typing import Final

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id


# AES-GCM nonce length. NIST SP 800-38D recommends 96 bits / 12 bytes for
# random nonces; we use os.urandom for every encryption so collision risk
# is bounded by the birthday paradox at ~2^48 messages — 280 trillion
# events. Fine for our scale.
_NONCE_LEN: Final[int] = 12

# DEK / KEK length. AES-256 is overkill for our threat model but pyca's
# AESGCM only accepts 16/24/32-byte keys and 32 is the conventional pick.
_KEY_LEN: Final[int] = 32

# Argon2id defaults. OWASP 2024 baseline is `t=2, m=19 MiB, p=1`; we lean
# heavier (`t=3, m=256 MiB, p=1`) because passphrase unlock is a one-time
# cost per process restart, not a per-request cost. Tunable via the kdf
# params dict that ships alongside each wrapped DEK.
_DEFAULT_KDF_PARAMS: Final[dict] = {
    "kdf": "argon2id",
    "time_cost": 3,
    "memory_kib": 262144,   # 256 MiB
    "parallelism": 1,
    "version": 19,           # Argon2 version 0x13 (1.3)
}


# ---- DEK / KEK / passphrase plumbing ----


@dataclass
class WrappedDEK:
    """The encrypted form of the data-encryption key as it lives in
    `app_settings`. Persisted as JSON: salt + nonce + ciphertext + KDF
    params. Reading this back + the streamer's passphrase is enough to
    recover the plaintext DEK."""
    salt_b64: str
    nonce_b64: str
    ciphertext_b64: str
    kdf_params: dict

    def to_json(self) -> str:
        return json.dumps({
            "salt": self.salt_b64,
            "nonce": self.nonce_b64,
            "ct": self.ciphertext_b64,
            "kdf": self.kdf_params,
        }, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> "WrappedDEK":
        d = json.loads(raw)
        return cls(
            salt_b64=d["salt"],
            nonce_b64=d["nonce"],
            ciphertext_b64=d["ct"],
            kdf_params=d["kdf"],
        )


def generate_dek() -> bytes:
    """Generate a fresh random DEK. Called once at `dataset setup`."""
    return secrets.token_bytes(_KEY_LEN)


def derive_kek(passphrase: str, salt: bytes, params: dict | None = None) -> bytes:
    """Derive a 32-byte KEK from the streamer's passphrase via Argon2id.

    `salt` is per-install (16 random bytes generated at setup and stored
    alongside the wrapped DEK). `params` overrides the default KDF cost —
    used when re-deriving against a previously-wrapped DEK whose params
    differ from the current default."""
    p = {**_DEFAULT_KDF_PARAMS, **(params or {})}
    if p.get("kdf") != "argon2id":
        raise ValueError(f"unsupported kdf {p.get('kdf')!r}")
    kdf = Argon2id(
        salt=salt,
        length=_KEY_LEN,
        iterations=int(p["time_cost"]),
        lanes=int(p["parallelism"]),
        memory_cost=int(p["memory_kib"]),
    )
    return kdf.derive(passphrase.encode("utf-8"))


def wrap_dek(dek: bytes, passphrase: str) -> WrappedDEK:
    """Encrypt the DEK under a passphrase-derived KEK. Returns the bundle
    that should be persisted to `app_settings.dataset_key_wrapped`. Salt
    + KDF params travel with the ciphertext so unwrap can be done without
    extra inputs."""
    if len(dek) != _KEY_LEN:
        raise ValueError(f"dek must be {_KEY_LEN} bytes, got {len(dek)}")
    salt = os.urandom(16)
    params = dict(_DEFAULT_KDF_PARAMS)
    kek = derive_kek(passphrase, salt, params)
    nonce = os.urandom(_NONCE_LEN)
    ct = AESGCM(kek).encrypt(nonce, dek, associated_data=b"chatterbot/dataset/dek")
    return WrappedDEK(
        salt_b64=base64.b64encode(salt).decode("ascii"),
        nonce_b64=base64.b64encode(nonce).decode("ascii"),
        ciphertext_b64=base64.b64encode(ct).decode("ascii"),
        kdf_params=params,
    )


def unwrap_dek(wrapped: WrappedDEK, passphrase: str) -> bytes:
    """Recover the plaintext DEK using the streamer's passphrase. Raises
    `cryptography.exceptions.InvalidTag` on wrong passphrase — caller
    should catch and surface as "wrong passphrase" to the streamer."""
    salt = base64.b64decode(wrapped.salt_b64)
    nonce = base64.b64decode(wrapped.nonce_b64)
    ct = base64.b64decode(wrapped.ciphertext_b64)
    kek = derive_kek(passphrase, salt, wrapped.kdf_params)
    return AESGCM(kek).decrypt(nonce, ct, associated_data=b"chatterbot/dataset/dek")


def fingerprint_dek(dek: bytes) -> str:
    """Short non-reversible identifier for the DEK — first 8 hex chars of
    the SHA-256 of the key. Stored in `app_settings.dataset_key_fingerprint`
    so we can fail loudly if a writer tries to use a different DEK than
    the one events were written under."""
    import hashlib
    return hashlib.sha256(dek).hexdigest()[:16]


# ---- per-event encrypt / decrypt ----


def encrypt_event(dek: bytes, plaintext: bytes, *, ts_iso: str) -> tuple[bytes, bytes]:
    """Encrypt a single event payload. Returns (nonce, ciphertext). The
    timestamp is bound into AES-GCM's associated data so an attacker
    can't reorder rows by swapping ciphertexts between timestamps."""
    if len(dek) != _KEY_LEN:
        raise ValueError(f"dek must be {_KEY_LEN} bytes, got {len(dek)}")
    nonce = os.urandom(_NONCE_LEN)
    ct = AESGCM(dek).encrypt(nonce, plaintext, associated_data=ts_iso.encode("utf-8"))
    return nonce, ct


def decrypt_event(dek: bytes, nonce: bytes, ciphertext: bytes, *, ts_iso: str) -> bytes:
    """Inverse of encrypt_event. Caller must pass the same `ts_iso` that
    was used at encrypt time; the index table stores `ts` in cleartext
    so the reader can rebuild the AAD without needing to decrypt first."""
    return AESGCM(dek).decrypt(nonce, ciphertext, associated_data=ts_iso.encode("utf-8"))


# ---- recovery key formatting ----


def dek_to_recovery_string(dek: bytes) -> str:
    """Format the DEK as a human-printable base32 string with hyphens
    every 4 chars. This is what the CLI prints at setup time so the
    streamer can write it down as a passphrase-recovery fallback."""
    s = base64.b32encode(dek).decode("ascii").rstrip("=")
    return "-".join(s[i:i + 4] for i in range(0, len(s), 4))


def dek_from_recovery_string(s: str) -> bytes:
    """Inverse of dek_to_recovery_string. Strips hyphens + whitespace +
    case-folds the input, then base32-decodes. Useful for the eventual
    `dataset recover` command (out of scope for slice 1)."""
    cleaned = "".join(c for c in s.upper() if c.isalnum())
    pad_needed = (-len(cleaned)) % 8
    cleaned += "=" * pad_needed
    return base64.b32decode(cleaned)
