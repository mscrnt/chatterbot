"""Measure compression headroom on a TEXT column.

Standalone harness — does not touch the schema, does not write back.
Reads the chosen `(table, column)` from a target DB, splits into a
train / test partition with a fixed seed, and reports several
compression strategies side-by-side so we can decide whether per-row
zstd + dictionary + dedupe is worth a real implementation.

Usage:
    uv run python scripts/bench_message_compression.py data/jynxzi_chatters.db
    uv run python scripts/bench_message_compression.py data/jynxzi_chatters.db \\
        --table notes --column text

Strategies measured (per row, unless noted):
    raw                      — sum of UTF-8 byte lengths (baseline).
    dedup-only               — distinct(content) bytes; counts what
                               an exact-duplicate-merge schema saves
                               WITHOUT any compression.
    zstd no-dict             — zstd compress each row alone.
    zstd with-dict           — zstd compress each row with a per-channel
                               trained dictionary (the headline number).
    zstd bulk no-dict        — concatenate all rows, compress once.
                               Reference floor — not random-accessible
                               so not a real storage strategy, but tells
                               us the entropy ceiling.

Why train/test split? Training the dict on the same rows you measure
against overstates the win. We hold out 80% as test.
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import zstandard as zstd

TRAIN_FRACTION = 0.20  # fraction of rows reserved for dict training
DICT_SIZE_BYTES = 110 * 1024
ZSTD_LEVEL = 19  # high but reasonable; storage-side is offline anyway
SEED = 42


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def fmt_ratio(raw: int, compressed: int) -> str:
    if compressed <= 0:
        return "n/a"
    return f"{raw / compressed:.2f}x"


def load_rows(db_path: Path, table: str, column: str) -> list[str]:
    """Pull every non-null row of `table.column`. SQLite handles 1M+
    rows in seconds — no need to stream."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL"
        )
        rows = [r[0] for r in cur.fetchall() if r[0] is not None]
    finally:
        conn.close()
    return rows


def measure(rows: list[str]) -> None:
    rng = random.Random(SEED)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    train_n = int(len(rows) * TRAIN_FRACTION)
    train_idx = set(indices[:train_n])
    train_rows = [rows[i] for i in indices[:train_n]]
    test_rows = [rows[i] for i in indices[train_n:]]

    encoded_test = [r.encode("utf-8") for r in test_rows]
    raw_total = sum(len(b) for b in encoded_test)

    sizes = [len(b) for b in encoded_test]
    p50 = int(statistics.median(sizes))
    p95 = int(sorted(sizes)[int(len(sizes) * 0.95)]) if sizes else 0

    print(f"\nMessages: {len(rows):,} total "
          f"(train: {len(train_rows):,}, test: {len(test_rows):,})")
    print(f"Raw bytes (test):      {fmt_bytes(raw_total):>12}")
    print(f"Per-message size:      median={p50} B, p95={p95} B")

    # --- dedup-only -------------------------------------------------
    # What a pure dedupe-with-counter schema saves before any actual
    # compression is applied. distinct(content) bytes = canonical
    # storage; the count column is a few bytes per distinct row.
    distinct = {}
    for b in encoded_test:
        distinct[b] = distinct.get(b, 0) + 1
    dedup_bytes = sum(len(b) for b in distinct)
    distinct_count = len(distinct)
    print(f"\nDistinct rows (test):  {distinct_count:,} "
          f"({distinct_count / len(test_rows):.1%} unique)")
    print(f"Dedup-only bytes:      {fmt_bytes(dedup_bytes):>12} "
          f"({fmt_ratio(raw_total, dedup_bytes)})")

    # --- zstd per-row, no dict --------------------------------------
    cctx_plain = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    t0 = time.monotonic()
    plain_total = sum(len(cctx_plain.compress(b)) for b in encoded_test)
    t_plain = time.monotonic() - t0
    print(f"\nzstd-no-dict per row:  {fmt_bytes(plain_total):>12} "
          f"({fmt_ratio(raw_total, plain_total)}) "
          f"in {t_plain:.1f}s")

    # --- zstd per-row, with trained dict ----------------------------
    train_blobs = [r.encode("utf-8") for r in train_rows]
    t0 = time.monotonic()
    dictionary = zstd.train_dictionary(DICT_SIZE_BYTES, train_blobs)
    t_train = time.monotonic() - t0
    dict_bytes = len(dictionary.as_bytes())
    cctx_dict = zstd.ZstdCompressor(level=ZSTD_LEVEL, dict_data=dictionary)
    t0 = time.monotonic()
    dict_total = sum(len(cctx_dict.compress(b)) for b in encoded_test)
    t_dict = time.monotonic() - t0
    print(f"zstd-with-dict row:    {fmt_bytes(dict_total):>12} "
          f"({fmt_ratio(raw_total, dict_total)}) "
          f"in {t_dict:.1f}s "
          f"[dict: {fmt_bytes(dict_bytes)}, train {t_train:.1f}s]")

    # Effective ratio after amortising dictionary size across the table.
    eff_total = dict_total + dict_bytes
    print(f"  + dict overhead:     {fmt_bytes(eff_total):>12} "
          f"({fmt_ratio(raw_total, eff_total)})")

    # --- bulk reference ---------------------------------------------
    bulk_blob = b"\n".join(encoded_test)
    t0 = time.monotonic()
    bulk_total = len(cctx_plain.compress(bulk_blob))
    t_bulk = time.monotonic() - t0
    print(f"\nzstd bulk (reference): {fmt_bytes(bulk_total):>12} "
          f"({fmt_ratio(raw_total, bulk_total)}) "
          f"in {t_bulk:.1f}s "
          f"— not random-accessible, entropy floor only")

    # --- combined: dedup THEN zstd-with-dict ------------------------
    # Real-world stack: store distinct rows compressed with the dict.
    distinct_blobs = list(distinct.keys())
    t0 = time.monotonic()
    distinct_compressed = sum(len(cctx_dict.compress(b)) for b in distinct_blobs)
    t_combo = time.monotonic() - t0
    combo_total = distinct_compressed + dict_bytes
    print(f"\ndedup + zstd-w-dict:   {fmt_bytes(combo_total):>12} "
          f"({fmt_ratio(raw_total, combo_total)}) "
          f"in {t_combo:.1f}s "
          f"— headline number")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=Path, help="path to *_chatters.db")
    parser.add_argument("--table", default="messages")
    parser.add_argument("--column", default="content")
    args = parser.parse_args()

    if not args.db.is_file():
        print(f"error: {args.db} is not a file", file=sys.stderr)
        return 1

    rows = load_rows(args.db, args.table, args.column)
    if not rows:
        print(f"error: {args.db} has no rows in {args.table}.{args.column}",
              file=sys.stderr)
        return 1

    print(f"DB: {args.db}")
    print(f"Source: {args.table}.{args.column}")
    measure(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
