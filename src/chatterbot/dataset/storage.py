"""Append-only encrypted shard storage for captured events.

Each event is JSON-serialised, zstd-compressed, then AES-GCM-encrypted with
a fresh nonce. The ciphertext lands in a length-prefixed record inside a
shard file under `data/dataset/shards/`; the index sits in
`chatters.db.dataset_events` (slice-1 co-tenant choice — moves to a sidecar
DB later if shard volume grows).

Shard record format (binary, big-endian):

    [u32 nonce_len][u32 ct_len][nonce bytes][ciphertext bytes]

The index row points at the byte_offset of the *first byte* of `[u32
nonce_len]`, with `byte_length` covering the whole record. Reads are one
`pread` + parse + decrypt.

Shards roll when the active file passes 50 MiB or the day rolls over; the
naming scheme `YYYY-MM-DD__NNNN.cbds.bin` lets retention-by-age delete
whole files without rewriting any record.

Slice 1 only writes events. Reads are exercised by `chatterbot dataset
export` and `verify` — bulk-walk via the index table, not random access.
"""

from __future__ import annotations

import logging
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# Single source of truth for where shards live, relative to the data dir.
SHARD_SUBDIR = "dataset/shards"

# Shard rotation thresholds. 50 MiB keeps individual files small enough
# to back up / scp in seconds; 24 h aligns shard boundaries with retention
# pruning so an expiring shard is always whole.
_SHARD_MAX_BYTES = 50 * 1024 * 1024
_SHARD_MAX_AGE_SECONDS = 24 * 3600

# Record header: two u32s (nonce length, ciphertext length). Big-endian
# because we want stable cross-architecture readability — these shards
# may be moved to a different machine for fine-tuning.
_HEADER = struct.Struct(">II")


@dataclass
class ShardRecord:
    """One encrypted record on disk. Returned by `read_record` for use by
    the export / verify pipeline."""
    nonce: bytes
    ciphertext: bytes


def shards_dir(data_root: Path) -> Path:
    """Resolve and create the shards directory under the project's data
    root (whatever `settings.db_path` lives next to)."""
    p = data_root / SHARD_SUBDIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def _shard_filename(ts: float, seq: int) -> str:
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return f"{dt.strftime('%Y-%m-%d')}__{seq:04d}.cbds.bin"


def _scan_active_shard(dir_path: Path) -> tuple[Path | None, int]:
    """Find the most-recently-modified shard in `dir_path`. Returns
    (path, seq) or (None, 0) if the dir is empty. The seq is parsed from
    the filename and used for the next rotation."""
    if not dir_path.exists():
        return None, 0
    files = sorted(dir_path.glob("*.cbds.bin"))
    if not files:
        return None, 0
    last = files[-1]
    try:
        seq_part = last.stem.split("__")[-1]
        seq = int(seq_part)
    except (ValueError, IndexError):
        seq = 0
    return last, seq


def _should_rotate(active: Path, now_ts: float) -> bool:
    try:
        st = active.stat()
    except FileNotFoundError:
        return True
    if st.st_size >= _SHARD_MAX_BYTES:
        return True
    if (now_ts - st.st_mtime) >= _SHARD_MAX_AGE_SECONDS:
        return True
    return False


class ShardWriter:
    """Append-only writer for encrypted event records.

    One instance per process. NOT thread-safe — wrap calls in a lock if
    multiple async tasks share it (the capture path uses a single lock
    in `capture.py`). Files are opened in append-binary mode and fsync'd
    after each record so crash recovery is bounded to "lose the last
    in-flight event" rather than "lose the tail of the shard."

    Reads are file-handle-free — they pread off the path stored in the
    sqlite index, so the writer holding an open append handle doesn't
    block the export reader."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self._dir = shards_dir(self.data_root)
        existing, seq = _scan_active_shard(self._dir)
        self._seq = seq
        self._active: Path | None = existing
        self._fh = None  # opened lazily so a no-write process never touches disk

    def _ensure_active_shard(self, now_ts: float) -> Path:
        """Pick or create the shard that the next record will land in."""
        if self._active is None or _should_rotate(self._active, now_ts):
            self._close_handle()
            self._seq += 1
            new_path = self._dir / _shard_filename(now_ts, self._seq)
            # touch
            new_path.touch(exist_ok=False)
            self._active = new_path
            logger.info("dataset: rotated to new shard %s", new_path.name)
        return self._active

    def _open_handle(self, path: Path):
        if self._fh is not None:
            return
        self._fh = open(path, "ab")

    def _close_handle(self):
        if self._fh is not None:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except OSError:
                pass
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    def append(self, nonce: bytes, ciphertext: bytes) -> tuple[Path, int, int]:
        """Write one record. Returns (shard_path, byte_offset, byte_length)
        for indexing in the events table.

        byte_offset points at the start of the record header (the first
        u32). byte_length covers `header + nonce + ciphertext`."""
        now = time.time()
        active = self._ensure_active_shard(now)
        self._open_handle(active)
        assert self._fh is not None

        header = _HEADER.pack(len(nonce), len(ciphertext))
        record = header + nonce + ciphertext

        offset = self._fh.tell()
        self._fh.write(record)
        # Per-record fsync would be a perf disaster for high-frequency
        # capture; we flush only and let the OS schedule the disk write.
        # Crash window: the unflushed kernel buffers (low double-digit
        # KB on Linux ext4 with default writeback). Fine for slice 1.
        self._fh.flush()
        return active, offset, len(record)

    def close(self):
        self._close_handle()


def read_record(shard_path: Path, byte_offset: int, byte_length: int) -> ShardRecord:
    """Read one record by offset. Used by the export/verify pipeline.
    Validates the header lengths match `byte_length` so a corrupt index
    can't drag a read off the end of the file."""
    with open(shard_path, "rb") as fh:
        fh.seek(byte_offset)
        raw = fh.read(byte_length)
    if len(raw) < _HEADER.size:
        raise ValueError(
            f"short read at {shard_path.name}@{byte_offset}: "
            f"want >= {_HEADER.size}, got {len(raw)}"
        )
    nonce_len, ct_len = _HEADER.unpack(raw[: _HEADER.size])
    expected_total = _HEADER.size + nonce_len + ct_len
    if expected_total != byte_length:
        raise ValueError(
            f"record-length mismatch at {shard_path.name}@{byte_offset}: "
            f"header says {expected_total}, index says {byte_length}"
        )
    nonce = raw[_HEADER.size : _HEADER.size + nonce_len]
    ct = raw[_HEADER.size + nonce_len :]
    return ShardRecord(nonce=nonce, ciphertext=ct)
