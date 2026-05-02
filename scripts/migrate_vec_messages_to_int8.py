"""Pilot: build a parallel `vec_messages_q8` populated from the existing
`vec_messages` float32 vectors, and report recall parity.

This script is non-destructive — it does NOT touch `vec_messages`. It
creates `vec_messages_q8` (INT8[768], cosine), drops/recreates if it
already exists, and runs a held-out recall@10 check by using N sample
rows as queries against both the float32 and int8 indexes.

Quantization scheme: GLOBAL SCALAR. Compute one float scale = (max |v|
across the entire corpus) / 127, then `code = round(v / scale).clip
(-128, 127).astype(int8)`. The scale is persisted in app_settings under
`vec_messages_q8_scale` for the read path to use later.

Usage:
    uv run python scripts/migrate_vec_messages_to_int8.py \\
        data/jynxzi_chatters.db
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import sqlite_vec

DIM = 768
N_QUERIES = 200
TOP_K = 10
SEED = 42


def open_db(p: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(p))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_float32_vectors(conn) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids, vectors) as parallel arrays. Vectors as (N, 768) float32."""
    cur = conn.execute("SELECT message_id, embedding FROM vec_messages")
    ids: list[int] = []
    blobs: list[bytes] = []
    for r in cur:
        ids.append(r["message_id"])
        blobs.append(r["embedding"])
    if not blobs:
        raise SystemExit("vec_messages is empty")
    expected = DIM * 4
    arr = np.empty((len(blobs), DIM), dtype=np.float32)
    for i, b in enumerate(blobs):
        if len(b) != expected:
            raise SystemExit(f"row {i}: expected {expected} bytes, got {len(b)}")
        arr[i] = np.frombuffer(b, dtype=np.float32)
    return np.array(ids, dtype=np.int64), arr


def quantize_global_int8(vecs: np.ndarray, scale: float) -> np.ndarray:
    """Per-component scalar quantization with a single global scale.
    Returns (N, 768) int8 array."""
    return np.round(vecs / scale).clip(-128, 127).astype(np.int8)


def build_q8_table(conn, ids: np.ndarray, codes: np.ndarray) -> None:
    """Create-or-replace `vec_messages_q8` and bulk-insert the codes."""
    conn.execute("DROP TABLE IF EXISTS vec_messages_q8")
    # Match the existing vec_messages distance_metric (L2 default). If
    # we later change the production table to cosine, change this in
    # lockstep.
    conn.execute(
        f"CREATE VIRTUAL TABLE vec_messages_q8 "
        f"USING vec0(message_id INTEGER PRIMARY KEY, "
        f"embedding INT8[{DIM}])"
    )
    # Bind in batches; sqlite-vec insert is per-row.
    cur = conn.cursor()
    t0 = time.monotonic()
    for i in range(len(ids)):
        cur.execute(
            "INSERT INTO vec_messages_q8(message_id, embedding) "
            "VALUES (?, vec_int8(?))",
            (int(ids[i]), codes[i].tobytes()),
        )
    conn.commit()
    print(f"  inserted {len(ids):,} rows in {time.monotonic() - t0:.1f}s")


def topk_via_match(conn, table: str, query_blob: bytes, query_fn: str, k: int) -> list[int]:
    """Run a vec MATCH query, return the message_ids of the top-k."""
    rows = conn.execute(
        f"SELECT message_id FROM {table} "
        f"WHERE embedding MATCH {query_fn}(?) AND k = ? "
        f"ORDER BY distance",
        (query_blob, k),
    ).fetchall()
    return [r["message_id"] for r in rows]


def measure_recall(
    conn, ids: np.ndarray, vecs_f32: np.ndarray, codes_i8: np.ndarray,
) -> tuple[float, float]:
    """Hold out N_QUERIES rows. Use each as a query against the live
    vec_messages (float32) for ground truth, and against the new
    vec_messages_q8 (int8) for the candidate. Return (recall@10, recall@1).
    Excludes the query itself from each result set."""
    rng = np.random.default_rng(SEED)
    qi = rng.choice(len(ids), size=min(N_QUERIES, len(ids)), replace=False)

    rec10 = []
    rec1 = []
    for i in qi:
        q_f32 = vecs_f32[i].tobytes()
        q_i8 = codes_i8[i].tobytes()
        # +1 because the query row is itself the closest match; we'll drop it.
        truth = topk_via_match(conn, "vec_messages", q_f32, "vec_f32", TOP_K + 1)
        pred = topk_via_match(conn, "vec_messages_q8", q_i8, "vec_int8", TOP_K + 1)
        truth = [t for t in truth if t != int(ids[i])][:TOP_K]
        pred = [p for p in pred if p != int(ids[i])][:TOP_K]
        if not truth:
            continue
        overlap10 = len(set(truth) & set(pred)) / len(truth)
        rec10.append(overlap10)
        rec1.append(1.0 if pred and pred[0] in truth[:1] else 0.0)
    return float(np.mean(rec10)), float(np.mean(rec1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("db", type=Path)
    args = ap.parse_args()

    conn = open_db(args.db)
    print(f"DB: {args.db}")

    # 1. Load float32 vectors.
    t0 = time.monotonic()
    ids, vecs = fetch_float32_vectors(conn)
    print(f"Loaded {len(ids):,} vectors in {time.monotonic() - t0:.1f}s")

    # 2. Compute global scale from p99.5 of |component|. Clipping
    # ≤0.5% of components in exchange for ~5-10x finer resolution on
    # the bulk of values. (No unit-normalize: vec_messages uses L2,
    # and L2 is magnitude-sensitive — normalizing would corrupt
    # distances. Verified the existing CREATE VIRTUAL TABLE has no
    # `distance_metric=` clause, so vec0 defaults to L2.)
    abs_vals = np.abs(vecs)
    abs_max = float(abs_vals.max())
    abs_p995 = float(np.quantile(abs_vals, 0.995))
    scale = abs_p995 / 127.0
    print(f"|component|: max={abs_max:.4f}  p99.5={abs_p995:.4f}  "
          f"→  scale = {scale:.8f}")

    # 3. Quantize.
    codes = quantize_global_int8(vecs, scale)
    nonzero_clip = int(np.sum((codes == 127) | (codes == -128)))
    print(f"Quantized; clipped values: {nonzero_clip:,} / "
          f"{codes.size:,} ({100 * nonzero_clip / codes.size:.3f}%)")

    # 4. Build the q8 table.
    print("Building vec_messages_q8...")
    build_q8_table(conn, ids, codes)

    # 5. Persist scale so the read path can dequantize.
    # `app_settings` already exists in chatters DBs with a NOT-NULL
    # `updated_at`. Match the existing schema by supplying it.
    from datetime import datetime, timezone
    conn.execute(
        "INSERT OR REPLACE INTO app_settings(key, value, updated_at) "
        "VALUES (?, ?, ?)",
        (
            "vec_messages_q8_scale",
            str(scale),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    print(f"Stored scale in app_settings.vec_messages_q8_scale")

    # 6. Storage delta.
    raw_bytes = len(ids) * DIM * 4
    q8_bytes = len(ids) * DIM
    print(f"\nStorage:")
    print(f"  vec_messages (float32):    {raw_bytes / 1024 / 1024:.1f} MB")
    print(f"  vec_messages_q8 (int8):    {q8_bytes / 1024 / 1024:.1f} MB")
    print(f"  ratio: {raw_bytes / q8_bytes:.2f}x")

    # 7. Recall parity.
    print(f"\nRunning recall parity (n={N_QUERIES} queries, k={TOP_K})...")
    t0 = time.monotonic()
    r10, r1 = measure_recall(conn, ids, vecs, codes)
    dt = time.monotonic() - t0
    print(f"  recall@10: {r10:.3f}")
    print(f"  recall@1:  {r1:.3f}")
    print(f"  ({dt:.1f}s)")

    if r10 >= 0.95:
        print("\n✓ Recall holds. Safe to proceed to read-path migration.")
    elif r10 >= 0.85:
        print("\n△ Recall acceptable but not great. Consider per-vector "
              "scaling with auxiliary columns before committing.")
    else:
        print("\n✗ Recall is too low. Do NOT migrate reads. Investigate "
              "before proceeding (try per-vector scale, or rotate before "
              "quantizing).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
