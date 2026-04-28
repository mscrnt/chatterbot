"""Spam scoring for incoming chat messages.

Score-based, not boolean. Each signal contributes to a `spam_score` in
[0.0, 1.0] and emits a reason code; the message's final score is the
max across all signals. Storing the score (not a flag) lets each
consumer pick its own threshold — stats / word cloud can tolerate 0.5,
LLM prompts want 0.2, an audit dashboard wants everything.

The detector is a pure function of `(content, account_metadata)` —
testable, no side effects, no DB calls. The flood detector that
piggybacks on the existing embedding pipeline lives separately
(`apply_near_duplicate_flood` in repo.py) and bumps scores after the
fact when copy-paste brigades are detected.

Anti-patterns this module deliberately avoids:
- A boolean flag (`is_spam`) — bakes one threshold in; future tuning
  costs a migration.
- Detection in SQL `WHERE` clauses — scans every read forever.
- Coupling to one consumer's needs ("flag it because the word cloud
  doesn't like it") — the detector is consumer-agnostic.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass


# Default threshold consumers can use for "ignore obvious spam". LLM
# prompts should pick something stricter (~0.2). Stored in code, not
# config — calling code names its own number, the detector itself is
# threshold-agnostic.
SPAM_THRESHOLD_DEFAULT = 0.5
SPAM_THRESHOLD_LLM = 0.2


@dataclass(frozen=True)
class _Signal:
    code: str
    score: float  # 0.0..1.0


def score_message(
    content: str,
    *,
    account_age_days: int | None = None,
) -> tuple[float, list[str]]:
    """Score a message for spammy traits. Returns `(score, reasons)`.

    `score` is the max across all triggered signals (0.0 = clean).
    `reasons` is the list of reason codes that fired above 0.3, for
    transparency in the dashboard / audit views.

    Signals (each is independent and can fire in combination):
      - `repetition`: low unique-token ratio. Catches `emote1 emote2`
        repeated 14 times (the user's reported case).
      - `compression`: zlib compresses repetitive text to a tiny
        fraction. Catches char-level loops like `aaaaaaaaa` that the
        token ratio misses.
      - `caps_flood`: long message that's mostly UPPERCASE.
      - `symbol_flood`: long message that's mostly punctuation /
        non-alphanumeric (often emoji+symbol bombs).
      - `long_msg_new_account`: very long message from an account
        that's less than a day old. Account-age signal; only
        contributes when the caller passes `account_age_days`.

    `account_age_days=None` means we don't know — that's fine; the
    signal just doesn't fire. Detector degrades cleanly when metadata
    is missing.
    """
    if not content:
        return 0.0, []
    signals: list[_Signal] = []

    # 1. Token repetition ratio. Skip very short messages — chat is
    # full of legit short replies and we don't want to flag "lol lol".
    tokens = content.split()
    if len(tokens) >= 10:
        unique_ratio = len({t.lower() for t in tokens}) / len(tokens)
        # Map: ratio 0.0 → score 1.0, ratio 0.5 → score 0.0. Below
        # 50% unique on a 10+ token message is almost always spam.
        if unique_ratio < 0.5:
            score = min(1.0, (0.5 - unique_ratio) * 2.0)
            signals.append(_Signal("repetition", score))

    # 2. Compression ratio — catches char-level repetition that the
    # token-based check misses (e.g. `aaaaaaaa`, `EEEEEEEE`). Only
    # meaningful at length, where compression has something to chew on.
    if len(content) >= 40:
        try:
            payload = content.encode("utf-8")
            comp_ratio = len(zlib.compress(payload)) / max(1, len(payload))
            # Below 0.3 means the message compresses to <30% of its
            # size — highly repetitive. 0.3 → score 0.0, 0.0 → 1.0.
            if comp_ratio < 0.3:
                score = min(1.0, (0.3 - comp_ratio) / 0.3)
                signals.append(_Signal("compression", score))
        except Exception:
            # zlib never raises on bytes input, but we never want a
            # spam-detector exception to poison message ingest.
            pass

    # 3. Caps flood — long message that's mostly uppercase. We only
    # care about LETTERS, so a message of `!!!!!!!!` doesn't trigger
    # this (it'll trigger symbol_flood instead).
    if len(content) >= 30:
        letters = [c for c in content if c.isalpha()]
        if len(letters) >= 20:
            caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if caps_ratio > 0.7:
                score = min(1.0, (caps_ratio - 0.7) / 0.3)
                signals.append(_Signal("caps_flood", score))

    # 4. Symbol flood — non-alnum, non-space chars dominate. Catches
    # `:):):)` strings and emoji bombs. Twitch native emotes are
    # already filtered upstream by `is_emote_only`.
    if len(content) >= 30:
        non_text = sum(1 for c in content if not c.isalnum() and not c.isspace())
        sym_ratio = non_text / len(content)
        if sym_ratio > 0.6:
            score = min(1.0, (sym_ratio - 0.6) / 0.4)
            signals.append(_Signal("symbol_flood", score))

    # 5. Long message + brand-new account — bot pattern. Mid-strength
    # signal so it can combine with others; not a kill switch on its
    # own (lots of legit users send wall-of-text intros).
    if (
        account_age_days is not None
        and account_age_days < 1
        and len(content) > 200
    ):
        signals.append(_Signal("long_msg_new_account", 0.6))

    if not signals:
        return 0.0, []
    score = max(s.score for s in signals)
    # Reasons: every signal that fired above 0.3 gets surfaced. The
    # dashboard can show "filtered: 12 repetition, 2 caps_flood" so
    # the streamer can audit.
    reasons = [s.code for s in signals if s.score >= 0.3]
    return score, reasons


def encode_reasons(reasons: list[str]) -> str | None:
    """Serialize reason codes for storage. Empty list → NULL so the
    column compresses nicely."""
    return json.dumps(reasons) if reasons else None


def decode_reasons(raw: str | None) -> list[str]:
    """Parse `spam_reasons` blob back into a list. Returns [] for
    NULL / empty / malformed."""
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x) for x in data if x]
    except (TypeError, ValueError):
        pass
    return []
