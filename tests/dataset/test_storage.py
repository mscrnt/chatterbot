"""Shard storage — append + read invariants, header validation.

ShardWriter is an append-only writer of length-prefixed encrypted
records. These tests pin its on-disk format so a future "let's switch
to NDJSON" or "let's add a checksum" change can't silently break old
shards. Real DEKs aren't needed here — we treat the encrypted bytes
as opaque blobs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chatterbot.dataset.storage import (
    ShardWriter,
    read_record,
    shards_dir,
    SHARD_SUBDIR,
)


def test_writer_creates_shards_dir(tmp_path: Path):
    """A fresh data dir with no shards/ subdir gets one on first
    append. Streamers shouldn't have to mkdir manually."""
    writer = ShardWriter(tmp_path)
    writer.append(nonce=os.urandom(12), ciphertext=b"x")
    assert (tmp_path / SHARD_SUBDIR).is_dir()
    writer.close()


def test_append_roundtrip(tmp_path: Path):
    """Write three records, read each back by its returned offset +
    length. Bytes must match exactly."""
    writer = ShardWriter(tmp_path)
    payloads = [
        (os.urandom(12), b"first record"),
        (os.urandom(12), b"second record, slightly longer"),
        (os.urandom(12), b""),  # empty ciphertext is legal too
    ]
    locs = []
    for nonce, ct in payloads:
        path, offset, length = writer.append(nonce, ct)
        locs.append((path, offset, length))
    writer.close()

    for (nonce, ct), (path, offset, length) in zip(payloads, locs):
        rec = read_record(path, offset, length)
        assert rec.nonce == nonce
        assert rec.ciphertext == ct


def test_read_record_validates_length(tmp_path: Path):
    """The index table stores `byte_length` separately from the
    record's own header lengths; if the two ever disagree, the
    reader must refuse rather than serving truncated data."""
    writer = ShardWriter(tmp_path)
    path, offset, length = writer.append(os.urandom(12), b"hello")
    writer.close()
    # Lie about the length — shorter than what the header advertises.
    with pytest.raises(ValueError, match="record-length mismatch"):
        read_record(path, offset, length - 4)


def test_read_record_short_read_raises(tmp_path: Path):
    """Pointing past EOF returns a truncated read; the parser must
    surface that as ValueError, not a silent zero-byte record."""
    writer = ShardWriter(tmp_path)
    path, _, _ = writer.append(os.urandom(12), b"hi")
    writer.close()
    with pytest.raises(ValueError, match="short read"):
        read_record(path, byte_offset=10**6, byte_length=128)


def test_shards_dir_idempotent(tmp_path: Path):
    """Calling `shards_dir` twice on the same root should not raise —
    it's expected to no-op when the directory already exists."""
    a = shards_dir(tmp_path)
    b = shards_dir(tmp_path)
    assert a == b
    assert a.is_dir()


def test_writer_resumes_existing_shard(tmp_path: Path):
    """A new ShardWriter on a data dir with existing shards picks up
    where the previous writer left off — doesn't start a new file
    just because the process restarted. Crucial for crash recovery
    so we don't end up with hundreds of tiny shard files."""
    w1 = ShardWriter(tmp_path)
    w1.append(os.urandom(12), b"first")
    w1.close()

    w2 = ShardWriter(tmp_path)
    path2, offset2, length2 = w2.append(os.urandom(12), b"second")
    w2.close()

    # Both records should live in the same shard file.
    shards = sorted((tmp_path / SHARD_SUBDIR).glob("*.cbds.bin"))
    assert len(shards) == 1
    rec = read_record(path2, offset2, length2)
    assert rec.ciphertext == b"second"
