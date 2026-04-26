"""One-time backfill: flag historical emote-only messages.

The bot only started recording the IRCv3 `emotes` tag after a recent
schema change, so existing rows have `is_emote_only = 0` even when the
message was just emote spam (e.g. `bawkCrazy bawkCrazy bawkCrazy`).
This script applies a structural heuristic to flag those rows so the
word cloud, summarizer, and moderator skip them retroactively.

Heuristic
---------
A message is flagged iff every whitespace-separated token is
"emote-shaped" — meaning either:
  - has internal uppercase (CamelCase: `bawkCrazy`, `KappaPride`)
  - is all-uppercase length >= 3 (`KEKW`, `LUL`, `OMEGALUL`)

A single-token message is flagged only if it's CamelCase (so a one-off
"WUT" reaction isn't suppressed). Multi-token messages with mixed
all-caps and CamelCase still flag.

This intentionally MISSES single-word title-case globals like `Kappa`,
`Pog`, `Kreygasm` — the cost of leaving them in the word cloud is low
compared to the risk of false-positives on real one-word reactions.

Usage
-----
  uv run python scripts/backfill_emote_only.py --db data/bawkbasoup_chatters.db
  uv run python scripts/backfill_emote_only.py --db <path> --apply

Defaults to a dry run. Pass `--apply` to actually write. Idempotent —
re-running won't re-flag rows that are already flagged, and it never
unflags anything.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path


# CamelCase = lowercase followed by uppercase somewhere in the token.
# Anchored at start to make sure we're describing the whole token shape.
_CAMEL_RE = re.compile(r"^[A-Za-z]*[a-z][A-Z][A-Za-z]*$")
_ALLCAPS_RE = re.compile(r"^[A-Z]{3,}$")
_LETTERS_RE = re.compile(r"^[A-Za-z]{2,30}$")


def _is_emote_token(tok: str) -> bool:
    if not _LETTERS_RE.match(tok):
        return False
    return bool(_CAMEL_RE.match(tok) or _ALLCAPS_RE.match(tok))


def is_emote_only(content: str) -> bool:
    if not content:
        return False
    tokens = content.split()
    if not tokens:
        return False
    if not all(_is_emote_token(t) for t in tokens):
        return False
    has_camel = any(_CAMEL_RE.match(t) for t in tokens)
    # CamelCase anywhere is the strongest emote signal — `bawkCrazy`,
    # `KappaPride` etc. don't appear as English. Flag immediately.
    if has_camel:
        return True
    # Otherwise everything is all-caps. To avoid catching shouty English
    # ("THE RUN", "OMG LOL", "HOLY"), require at least one token to be
    # ≥6 chars — long all-caps words are emote-shaped (OMEGALUL, POGGERS,
    # JEBAITED, KEKWAIT) much more often than English shouting.
    return any(len(t) >= 6 for t in tokens)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db", required=True,
        help="Path to the SQLite DB (e.g. data/bawkbasoup_chatters.db)",
    )
    ap.add_argument(
        "--apply", action="store_true",
        help="Actually write. Without this, runs in dry-run mode.",
    )
    ap.add_argument(
        "--sample", type=int, default=20,
        help="How many flagged samples to print for sanity check (default 20)",
    )
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"✗ DB not found: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Make sure the column exists (older DBs that haven't booted the
    # current ChatterRepo won't have it yet).
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(messages)")}
    if "is_emote_only" not in cols:
        print(
            "✗ messages.is_emote_only column missing — boot ChatterRepo "
            "(e.g. start the bot or dashboard once) to run the schema migration."
        )
        return 1

    rows = conn.execute(
        "SELECT id, content FROM messages WHERE is_emote_only = 0"
    ).fetchall()
    total = len(rows)
    print(f"→ scanning {total:,} unflagged message(s)…")

    to_flag: list[int] = []
    samples: list[str] = []
    for r in rows:
        if is_emote_only(r["content"] or ""):
            to_flag.append(int(r["id"]))
            if len(samples) < args.sample:
                samples.append(r["content"])

    pct = (len(to_flag) / total * 100) if total else 0.0
    print(f"  flagged: {len(to_flag):,} of {total:,} ({pct:.1f}%)")

    if samples:
        print("\nSample of flagged messages:")
        for s in samples:
            preview = s if len(s) <= 80 else (s[:77] + "…")
            print(f"  • {preview}")

    if not to_flag:
        print("\nNothing to do.")
        return 0

    if not args.apply:
        print(
            f"\n(dry run — pass --apply to write {len(to_flag):,} updates)"
        )
        return 0

    # Chunk the UPDATEs to keep the SQL string sane on big DBs.
    BATCH = 500
    updated = 0
    cur = conn.cursor()
    for i in range(0, len(to_flag), BATCH):
        chunk = to_flag[i : i + BATCH]
        placeholders = ",".join("?" for _ in chunk)
        cur.execute(
            f"UPDATE messages SET is_emote_only = 1 WHERE id IN ({placeholders})",
            chunk,
        )
        updated += cur.rowcount
    conn.commit()
    print(f"\n✓ updated {updated:,} row(s) → is_emote_only = 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
