"""Backfill: embed historical chat messages into vec_messages so global
semantic search (`/search`) returns more than the recent slice.

The bot only embeds messages on-demand for the per-user "Ask Qwen" RAG
flow, so most rows in `messages` start with no entry in `vec_messages`.
This script walks the table, batches uncovered rows, calls Ollama's
embeddings API for each, and inserts.

Costs vary by model — `nomic-embed-text` typically returns in <100 ms
per call on a CUDA host. 10k messages = ~15 min.

Usage
-----
  uv run python scripts/backfill_message_embeddings.py --db data/foo.db
  uv run python scripts/backfill_message_embeddings.py --db data/foo.db --batch 50 --max 5000

`--max` caps the work for a single run (resume by re-running). Skip
already-embedded rows automatically — the script is idempotent.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chatterbot.config import get_settings  # noqa: E402
from chatterbot.llm.ollama_client import OllamaClient  # noqa: E402
from chatterbot.repo import ChatterRepo  # noqa: E402


async def main_async(args) -> int:
    settings = get_settings()
    db_path = Path(args.db) if args.db else Path(settings.db_path)
    if not db_path.exists():
        print(f"✗ DB not found: {db_path}")
        return 1

    repo = ChatterRepo(str(db_path))
    llm = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        embed_model=settings.ollama_embed_model,
    )

    if not await llm.health_check():
        print(f"✗ Ollama unreachable at {settings.ollama_base_url}")
        return 1

    indexed, total = repo.messages_embedding_coverage()
    print(
        f"→ {db_path.name}: {indexed:,} / {total:,} text messages embedded "
        f"({100 * indexed / total:.1f}%)"
        if total else f"→ {db_path.name}: empty"
    )
    if indexed >= total:
        print("Already complete. Nothing to do.")
        return 0

    cap = args.max if args.max > 0 else (total - indexed)
    print(f"  embedding up to {cap:,} more in batches of {args.batch}…")

    written = 0
    while written < cap:
        remaining = cap - written
        batch_size = min(args.batch, remaining)
        rows = repo.messages_missing_embedding_global(batch_size)
        if not rows:
            break
        for m in rows:
            try:
                vec = await llm.embed(m.content)
            except Exception as e:
                print(f"  ! embed failed for msg {m.id}: {type(e).__name__}: {e}")
                continue
            repo.upsert_message_embedding(m.id, vec)
            written += 1
        print(f"  wrote {written:,} / {cap:,}")

    indexed, total = repo.messages_embedding_coverage()
    print(
        f"\n✓ done. coverage now: {indexed:,} / {total:,} "
        f"({100 * indexed / total:.1f}%)"
        if total else "\n✓ done."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db",
        help="Path to the SQLite DB. Defaults to settings.db_path.",
    )
    ap.add_argument(
        "--batch", type=int, default=50,
        help="How many messages to fetch per outer loop (default 50).",
    )
    ap.add_argument(
        "--max", type=int, default=0,
        help="Cap on total messages embedded this run (0 = no cap, embed all).",
    )
    args = ap.parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\ninterrupted — partial progress saved.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
