"""Measure embedding-quantization storage ratio vs recall@10.

Loads `notes.embedding` BLOBs (float32, dim=768) from a target DB. Holds
out 200 vectors as queries; the rest is the corpus. Computes ground-
truth top-10 nearest neighbors with float32 cosine, then evaluates
each quantization scheme by re-running top-10 against the quantized
corpus and reporting Jaccard-overlap recall@10.

Schemes:
    float32           — baseline (sanity check, recall=1.0).
    int8 scalar       — per-vector min/max → uint8. 4x. Trivial impl.
    binary            — sign(v - mean(v)) → 1 bit/dim. 32x. Hamming
                        distance correlates with cosine on most
                        sentence-embedding models.
    PQ M=96           — product quantization, 96 sub-codebooks of 256
                        codes each. 32x bytes/vec, plus a one-time
                        codebook (~768 KB) that amortises over the
                        corpus. The strongest accuracy/ratio combo
                        for million-row vec stores in the literature.
    PQ M=192          — same scheme, finer split. 16x bytes/vec.

Usage:
    uv run python scripts/bench_embedding_quantization.py \\
        data/jynxzi_chatters.db
"""

from __future__ import annotations

import argparse
import sqlite3
import struct
import sys
import time
from pathlib import Path

import numpy as np

DIM = 768
N_QUERIES = 200
TOP_K = 10
PQ_TRAIN_ITERS = 12
PQ_CODEBOOK_SIZE = 256  # 8-bit code per sub-vector
SEED = 42


# ----------------------------------------------------------------- io

def load_embeddings(db_path: Path) -> np.ndarray:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT embedding FROM notes WHERE embedding IS NOT NULL"
        )
        blobs = [r[0] for r in cur.fetchall() if r[0] is not None]
    finally:
        conn.close()
    if not blobs:
        raise SystemExit("no embeddings found in notes.embedding")
    expected = DIM * 4
    arr = np.empty((len(blobs), DIM), dtype=np.float32)
    for i, b in enumerate(blobs):
        if len(b) != expected:
            raise SystemExit(f"row {i}: expected {expected} bytes, got {len(b)}")
        arr[i] = np.frombuffer(b, dtype=np.float32)
    return arr


def normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


# --------------------------------------------------------- recall metric

def topk_cosine(queries: np.ndarray, corpus: np.ndarray, k: int) -> np.ndarray:
    sims = queries @ corpus.T  # both unit-normalized → cosine
    return np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]


def recall(truth: np.ndarray, pred: np.ndarray) -> float:
    n, k = truth.shape
    return float(np.mean([
        len(set(truth[i]) & set(pred[i])) / k for i in range(n)
    ]))


# -------------------------------------------------------- quantizers

def int8_scalar(corpus: np.ndarray) -> tuple[int, callable]:
    """Per-vector scalar quantization. Bytes/vec = dim + 8 (min/max)."""
    cmin = corpus.min(axis=1, keepdims=True)
    cmax = corpus.max(axis=1, keepdims=True)
    scale = (cmax - cmin) / 255.0
    scale[scale == 0] = 1.0
    q = ((corpus - cmin) / scale).round().clip(0, 255).astype(np.uint8)

    def search(queries: np.ndarray, k: int) -> np.ndarray:
        deq = q.astype(np.float32) * scale + cmin
        deq = normalize(deq)
        return topk_cosine(queries, deq, k)

    return DIM + 8, search


def binary_quant(corpus: np.ndarray) -> tuple[int, callable]:
    """Sign-based 1-bit quantization. Bytes/vec = ceil(dim/8)."""
    centered = corpus - corpus.mean(axis=1, keepdims=True)
    bits = (centered > 0).astype(np.uint8)
    bytes_per_vec = (DIM + 7) // 8

    def search(queries: np.ndarray, k: int) -> np.ndarray:
        # Same binarisation for queries; use Hamming similarity as proxy.
        q_bits = ((queries - queries.mean(axis=1, keepdims=True)) > 0).astype(np.uint8)
        # Hamming similarity = dim - hamming_distance.
        # Equivalent rank to negative XOR popcount.
        sims = q_bits @ bits.T - (1 - q_bits) @ bits.T
        return np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]

    return bytes_per_vec, search


def kmeans_np(x: np.ndarray, k: int, iters: int, rng: np.random.Generator) -> np.ndarray:
    """Minimal Lloyd's k-means. Returns (k, dim) centroids."""
    n, _ = x.shape
    init = rng.choice(n, size=k, replace=False)
    centroids = x[init].copy()
    for _ in range(iters):
        # Assign
        d2 = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        # Update
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = x[mask].mean(axis=0)
    return centroids


def product_quant(corpus: np.ndarray, M: int, rng: np.random.Generator) -> tuple[int, int, callable]:
    """Product quantization. Splits dim into M sub-spaces, learns a 256-
    code codebook per sub-space. Returns (bytes_per_vec, codebook_bytes,
    search_fn). Search uses asymmetric distance computation (ADC):
    queries are NOT quantized, only the corpus is."""
    if DIM % M:
        raise SystemExit(f"DIM ({DIM}) not divisible by M ({M})")
    sub_dim = DIM // M
    sub = corpus.reshape(-1, M, sub_dim)  # (N, M, sub_dim)
    codebooks = np.empty((M, PQ_CODEBOOK_SIZE, sub_dim), dtype=np.float32)
    codes = np.empty((corpus.shape[0], M), dtype=np.uint8)
    for m in range(M):
        cb = kmeans_np(sub[:, m, :], PQ_CODEBOOK_SIZE, PQ_TRAIN_ITERS, rng)
        codebooks[m] = cb
        # Assign each vector's m-th sub-block to nearest centroid.
        d2 = ((sub[:, m, :, None] - cb.T[None, :, :]) ** 2).sum(axis=1)
        codes[:, m] = d2.argmin(axis=1).astype(np.uint8)

    bytes_per_vec = M
    codebook_bytes = M * PQ_CODEBOOK_SIZE * sub_dim * 4

    def search(queries: np.ndarray, k: int) -> np.ndarray:
        q_sub = queries.reshape(-1, M, sub_dim)
        # Asymmetric distance: precompute query-to-centroid table then
        # sum over sub-spaces using the corpus's quantization codes.
        # dist_table[q, m, c] = ||q_sub[q,m] - cb[m,c]||^2
        dist_table = np.empty((queries.shape[0], M, PQ_CODEBOOK_SIZE), dtype=np.float32)
        for m in range(M):
            diff = q_sub[:, m, :, None] - codebooks[m].T[None, :, :]
            dist_table[:, m, :] = (diff ** 2).sum(axis=1)
        # For each query, gather the per-sub-space distance for each
        # corpus vector's codes and sum across M.
        dists = np.zeros((queries.shape[0], corpus.shape[0]), dtype=np.float32)
        for m in range(M):
            dists += dist_table[:, m, :][:, codes[:, m]]
        return np.argpartition(dists, kth=k - 1, axis=1)[:, :k]

    return bytes_per_vec, codebook_bytes, search


# ----------------------------------------------------------------- main

def fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)

    print(f"DB: {args.db}")
    raw = load_embeddings(args.db)
    print(f"Loaded {raw.shape[0]:,} vectors of dim {raw.shape[1]}")
    raw = normalize(raw)

    # Train/test split.
    idx = rng.permutation(raw.shape[0])
    qi, ci = idx[:N_QUERIES], idx[N_QUERIES:]
    queries, corpus = raw[qi], raw[ci]
    n_corpus = corpus.shape[0]

    t0 = time.monotonic()
    truth = topk_cosine(queries, corpus, TOP_K)
    t_truth = time.monotonic() - t0
    raw_bytes = n_corpus * DIM * 4
    print(f"Corpus: {n_corpus:,} vec  Queries: {N_QUERIES}  k={TOP_K}")
    print(f"Float32 baseline: {fmt(raw_bytes)}  truth-build {t_truth:.1f}s\n")

    print(f"{'scheme':<22} {'bytes/vec':>10} {'corpus':>10} "
          f"{'+codebook':>10} {'ratio':>7} {'recall@10':>10} {'time':>6}")
    print("-" * 82)

    def report(name, bpv, code_bytes, search_fn):
        t0 = time.monotonic()
        pred = search_fn(queries, TOP_K)
        t = time.monotonic() - t0
        corpus_bytes = bpv * n_corpus
        total_bytes = corpus_bytes + code_bytes
        ratio = raw_bytes / total_bytes if total_bytes else 0
        r = recall(truth, pred)
        cb = fmt(code_bytes) if code_bytes else "—"
        print(f"{name:<22} {bpv:>10} {fmt(corpus_bytes):>10} "
              f"{cb:>10} {ratio:>6.2f}x {r:>9.3f} {t:>5.1f}s")

    bpv, fn = int8_scalar(corpus)
    report("int8 scalar", bpv, 0, fn)

    bpv, fn = binary_quant(corpus)
    report("binary (sign)", bpv, 0, fn)

    for M in (192, 96, 64):
        bpv, code_bytes, fn = product_quant(corpus, M, rng)
        report(f"PQ M={M}", bpv, code_bytes, fn)

    return 0


if __name__ == "__main__":
    sys.exit(main())
