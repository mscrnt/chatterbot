"""Crypto primitives — round-trip + tamper-detection tests.

Cipher is the foundation of the entire dataset capture system; if it
silently regresses we'd corrupt every capture going forward. These
tests are short on purpose — each one exercises ONE invariant.
"""

from __future__ import annotations

import pytest

from chatterbot.dataset import cipher


def test_dek_is_32_bytes():
    """DEK is always 32 random bytes (AES-256). If this changes the
    `cryptography.hazmat` AESGCM constructor will reject it at
    runtime, but we want to fail at unit-test time, not at the first
    capture call on a streamer's machine."""
    dek = cipher.generate_dek()
    assert len(dek) == 32


def test_dek_is_random():
    """Two consecutive generate_dek calls must not return the same
    bytes. Catches accidentally seeding `secrets` with a constant."""
    a = cipher.generate_dek()
    b = cipher.generate_dek()
    assert a != b


def test_wrap_unwrap_roundtrip():
    """A DEK wrapped under a passphrase decrypts back to the original
    bytes — the basic envelope-encryption invariant."""
    dek = cipher.generate_dek()
    wrapped = cipher.wrap_dek(dek, "my passphrase")
    recovered = cipher.unwrap_dek(wrapped, "my passphrase")
    assert recovered == dek


def test_unwrap_with_wrong_passphrase_raises():
    """The whole point of encryption: a wrong passphrase produces an
    AESGCM `InvalidTag`, not silently-wrong bytes."""
    from cryptography.exceptions import InvalidTag
    dek = cipher.generate_dek()
    wrapped = cipher.wrap_dek(dek, "correct")
    with pytest.raises(InvalidTag):
        cipher.unwrap_dek(wrapped, "wrong")


def test_wrapped_dek_json_roundtrip():
    """The wrapped DEK persists in `app_settings` as a JSON string.
    `to_json` → `from_json` must be lossless."""
    dek = cipher.generate_dek()
    wrapped = cipher.wrap_dek(dek, "x")
    serialised = wrapped.to_json()
    restored = cipher.WrappedDEK.from_json(serialised)
    # Round-trip via the wrapped form — nothing else exposes equality
    # on WrappedDEK, so we verify by unwrapping with the same
    # passphrase and comparing DEKs.
    assert cipher.unwrap_dek(restored, "x") == dek


def test_fingerprint_is_stable_per_dek():
    """Same DEK → same fingerprint. Different DEK → different
    fingerprint. The fingerprint is the only sanity check the bot has
    that the wrapped key in app_settings matches the DEK in memory."""
    a = cipher.generate_dek()
    b = cipher.generate_dek()
    assert cipher.fingerprint_dek(a) == cipher.fingerprint_dek(a)
    assert cipher.fingerprint_dek(a) != cipher.fingerprint_dek(b)
    assert len(cipher.fingerprint_dek(a)) == 16  # 8 hex bytes


def test_event_encrypt_decrypt_roundtrip():
    """The per-event encrypt path is what runs on every captured LLM
    call. Round-tripping with the same timestamp + DEK must return
    the exact bytes."""
    dek = cipher.generate_dek()
    plaintext = b"hello chatterbot"
    nonce, ct = cipher.encrypt_event(dek, plaintext, ts_iso="2026-04-30T00:00:00Z")
    recovered = cipher.decrypt_event(
        dek, nonce, ct, ts_iso="2026-04-30T00:00:00Z",
    )
    assert recovered == plaintext


def test_event_decrypt_rejects_tampered_timestamp():
    """The timestamp is bound into AES-GCM's associated-data field. An
    attacker who reorders rows in the index table (different ts) but
    keeps the ciphertext intact must get a decryption failure — not
    silent corruption."""
    from cryptography.exceptions import InvalidTag
    dek = cipher.generate_dek()
    nonce, ct = cipher.encrypt_event(
        dek, b"payload", ts_iso="2026-04-30T00:00:00Z",
    )
    with pytest.raises(InvalidTag):
        cipher.decrypt_event(
            dek, nonce, ct, ts_iso="2099-01-01T00:00:00Z",
        )


def test_recovery_string_roundtrip():
    """The recovery string the CLI prints at setup must be losslessly
    decodable back to the same DEK. Catches accidentally chopping off
    base32 padding or hyphenation drift."""
    dek = cipher.generate_dek()
    s = cipher.dek_to_recovery_string(dek)
    # Hyphens are formatting, not semantic — strip + lowercase should
    # also still recover (matches `dek_from_recovery_string`'s
    # tolerance to user-mangled input).
    assert cipher.dek_from_recovery_string(s) == dek
    assert cipher.dek_from_recovery_string(s.lower().replace("-", " ")) == dek


def test_kdf_params_are_carried_with_wrapped_dek():
    """Bumping KDF cost in a future release shouldn't lock streamers
    out of their existing wrapped DEKs — the params travel with the
    ciphertext. Verify the wrapped form actually contains them."""
    dek = cipher.generate_dek()
    wrapped = cipher.wrap_dek(dek, "x")
    assert wrapped.kdf_params.get("kdf") == "argon2id"
    assert wrapped.kdf_params.get("time_cost") >= 1
    assert wrapped.kdf_params.get("memory_kib") >= 1024
