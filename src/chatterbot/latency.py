"""Twitch broadcast-latency calibration.

Cross-correlates transcript text against chat content at various time
offsets to auto-detect the streamer's broadcast latency.

Why it works: chat parrots streamer words. Game names, the people
they mention by name, "yes / no / lol / W / L", any slightly
distinctive token. So when we score `weighted overlap of chunk_tokens
∩ chat_tokens` across a window of transcript chunks paired with chat
messages shifted by `offset_seconds`, the offset that maximises the
score is the broadcast latency.

Scoring uses IDF-style weighting computed against the chat corpus:
rare tokens (proper nouns, game names) count strongly; chat staples
("POG", "lol", "everyone") get near-zero weight. This sharpens the
correlation peak dramatically on short / greeting-heavy windows
where plain set-overlap would be flat.

The function returns flat scores across 0-18 s in 2 s steps so the UI
can render a bar chart, AND a small "evidence" list of the top-K
chunk-chat pairs that drove the best offset's score, so the streamer
can verify the match by eye instead of trusting a black-box number.

No streamer action required.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import Any

# Tokens shorter than this are too noisy ("a", "i", "ok"). Tokens
# longer than ~24 chars are usually emote spam ("OMEGALULiguana").
_MIN_TOKEN_LEN = 3
_MAX_TOKEN_LEN = 24

# Common english stopwords — chat and streamer audio share these
# uniformly across all offsets, so filtering them out sharpens the
# correlation peak. The IDF weighting catches the rest of the chat
# staples that aren't "english" stopwords (POG, lol, etc.).
_STOPWORDS = frozenset({
    "the", "and", "for", "you", "are", "was", "were", "with", "this",
    "that", "have", "has", "had", "but", "not", "all", "any", "can",
    "will", "would", "could", "should", "what", "when", "where", "who",
    "why", "how", "which", "from", "into", "out", "off", "over",
    "under", "about", "your", "their", "them", "they", "his", "her",
    "him", "she", "yes", "yeah", "no", "nope", "ok", "okay", "just",
    "like", "really", "got", "get", "going", "gonna", "wanna", "say",
    "said", "see", "saw", "seen", "look", "way", "lot", "much", "more",
    "less", "than", "then", "now", "here", "there", "still", "even",
    "ever", "never", "again", "always", "some", "many", "few",
    "lol", "lmao", "kek", "wat", "tho", "imo", "fr", "ngl", "btw",
    "ya", "yo", "hey", "yep", "wtf", "idk", "tbh", "rn",
})

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenise(text: str) -> set[str]:
    """Lowercase, strip non-alphanumerics, drop stopwords + too-short
    + too-long tokens. Returns a SET so each unique token contributes
    once per message — chat repetition would otherwise dominate."""
    if not text:
        return set()
    out: set[str] = set()
    for tok in _TOKEN_RE.findall(text.lower()):
        if (
            _MIN_TOKEN_LEN <= len(tok) <= _MAX_TOKEN_LEN
            and tok not in _STOPWORDS
        ):
            out.add(tok)
    return out


def _parse_iso(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(
            tzinfo=None,
        )
    except (TypeError, ValueError):
        return None


# Search 0-18 s in 2 s steps. Wider than typical Low Latency (~3-5 s)
# but narrower than Standard mode (~12-15 s) so the bar chart stays
# legible. Bumpable if a streamer is on DVR mode (~20+ s).
OFFSET_CANDIDATES = list(range(0, 20, 2))

# Window around `chunk_ts + offset` to look for chat. ±5 s is the
# sweet spot — wide enough to catch the natural read-react-type
# spread (most chat lands within 5 s of when they heard the audio),
# narrow enough that adjacent offsets capture DIFFERENT chat messages
# rather than blurring into one big plateau. Earlier 15 s window
# made the curve flat because every candidate offset's window
# overlapped the true reaction time. With 2 s offset steps + 5 s
# window, neighboring offsets share half their window and the peak
# is well-defined.
_CHAT_WINDOW_SEC = 5

# How many chunk-chat pairs to surface as evidence. Three is enough
# to convey "yes, real matches" without a wall of text.
_EVIDENCE_K = 5


def _build_idf_weights(
    msg_token_sets: list[set[str]],
) -> dict[str, float]:
    """Inverse document frequency over the chat corpus only.

    Streamer-said-it doesn't make a token common; chat-said-it-many-
    times does. So we compute df against chat messages only.

    weight = max(0, log(N / df) - 1)

    The `-1` floor turns "appears in roughly 1/e of messages" into
    weight 0. Any token more common than that contributes nothing —
    they're chat staples. Genuinely rare tokens (proper nouns, game
    names, distinctive content) rise above the noise floor.
    """
    n_msgs = max(1, len(msg_token_sets))
    df: dict[str, int] = {}
    for toks in msg_token_sets:
        for t in toks:
            df[t] = df.get(t, 0) + 1
    weights: dict[str, float] = {}
    for tok, count in df.items():
        # log natural — log(N/df) for token in every message = 0,
        # token in 1/N messages = log(N).
        w = math.log(n_msgs / count) - 1.0
        if w > 0:
            weights[tok] = w
    return weights


def calibrate_chat_lag(
    repo,
    *,
    lookback_minutes: int = 5,
    offset_candidates: list[int] | None = None,
) -> dict[str, Any]:
    """Score each candidate offset for how well it aligns transcript
    text against chat text, weighted by token rarity.

    Returns:
        {
          "ok": bool,                # False on too-quiet / no signal
          "reason": str | None,
          "samples": {"chunks": int, "messages": int},
          "lookback_minutes": int,
          "offsets": [
            {"seconds": int, "score": float, "matched_tokens": int},
            ...
          ],
          "best_offset": int | None,         # peak score
          "second_best_offset": int | None,  # for confidence check
          "evidence": [
            {
              "chunk_ts": str,           # HH:MM:SS
              "chunk_text": str,         # streamer utterance
              "chat": [
                {"ts": str, "name": str, "content": str,
                 "matched": list[str]}
              ],
              "weight": float,           # IDF sum across pairs
            },
            ...
          ],
          "weighted_token_universe": int,  # how many distinctive tokens scored
        }
    """
    candidates = list(offset_candidates or OFFSET_CANDIDATES)

    # Pull data. recent_transcripts + recent_messages both already
    # exist on the repo and respect the standard clean-message filter.
    try:
        chunks = repo.recent_transcripts(
            within_minutes=lookback_minutes, limit=2000,
        )
    except Exception:
        chunks = []
    try:
        msgs = repo.recent_messages(
            limit=4000, within_minutes=lookback_minutes,
        )
    except Exception:
        msgs = []

    n_chunks = len(chunks)
    n_msgs = len(msgs)
    samples = {"chunks": n_chunks, "messages": n_msgs}

    empty_offsets = [
        {"seconds": s, "score": 0.0, "matched_tokens": 0}
        for s in candidates
    ]
    empty_result = {
        "ok": False,
        "samples": samples,
        "lookback_minutes": lookback_minutes,
        "offsets": empty_offsets,
        "best_offset": None,
        "second_best_offset": None,
        "evidence": [],
        "weighted_token_universe": 0,
        "confidence": 0.0,
    }

    # Quiet-channel guard — without enough cross-content the curve is
    # flat noise and "best offset" is meaningless. The streamer
    # should fall back to the manual default.
    if n_chunks < 6 or n_msgs < 15:
        empty_result["reason"] = (
            "not enough activity in the lookback window "
            f"({n_chunks} transcript chunks, {n_msgs} chat msgs)"
        )
        return empty_result

    # Pre-tokenise once. Each chunk / message becomes (datetime, token_set,
    # original) so we can reach back for the original text when
    # building the evidence list.
    chunk_pairs: list[tuple[datetime, set[str], Any]] = []
    for c in chunks:
        ts = _parse_iso(getattr(c, "ts", None))
        if ts is None:
            continue
        toks = _tokenise(getattr(c, "text", "") or "")
        if toks:
            chunk_pairs.append((ts, toks, c))

    msg_pairs: list[tuple[datetime, set[str], Any]] = []
    msg_token_sets: list[set[str]] = []
    for m in msgs:
        ts = _parse_iso(getattr(m, "ts", None))
        if ts is None:
            continue
        toks = _tokenise(getattr(m, "content", "") or "")
        if toks:
            msg_pairs.append((ts, toks, m))
            msg_token_sets.append(toks)

    if not chunk_pairs or not msg_pairs:
        return {
            "ok": False,
            "reason": "no tokenisable content after filtering",
            "samples": samples,
            "lookback_minutes": lookback_minutes,
            "offsets": empty_offsets,
            "best_offset": None,
            "second_best_offset": None,
            "evidence": [],
            "weighted_token_universe": 0,
        }

    # IDF weights from the chat corpus. Rare tokens get high weight,
    # common tokens get zero weight (they're filtered out implicitly).
    idf = _build_idf_weights(msg_token_sets)
    universe = len(idf)
    if universe < 5:
        return {
            "ok": False,
            "reason": (
                "chat is too uniform to extract distinctive tokens "
                f"({universe} weighted tokens after IDF filter)"
            ),
            "samples": samples,
            "lookback_minutes": lookback_minutes,
            "offsets": empty_offsets,
            "best_offset": None,
            "second_best_offset": None,
            "evidence": [],
            "weighted_token_universe": universe,
        }

    # Sort messages by timestamp so we can binary-window them per
    # chunk.
    msg_pairs.sort(key=lambda p: p[0])
    msg_tslist = [p[0] for p in msg_pairs]

    import bisect

    results: list[dict[str, Any]] = []
    half = timedelta(seconds=_CHAT_WINDOW_SEC)
    # Per-offset evidence buffer for whichever offset wins. Keyed by
    # candidate offset, value is list[(weight, chunk, [(msg, shared_toks)])]
    # so we don't compute evidence twice — we cherry-pick the winning
    # offset's buffer at the end.
    evidence_per_offset: dict[int, list[tuple[float, Any, list[tuple[Any, list[str]]]]]] = {
        s: [] for s in candidates
    }

    for off in candidates:
        delta = timedelta(seconds=off)
        total_weighted = 0.0
        total_matched = 0
        contributing_chunks = 0
        for c_ts, c_toks, c_obj in chunk_pairs:
            lo = c_ts + delta - half
            hi = c_ts + delta + half
            i_lo = bisect.bisect_left(msg_tslist, lo)
            i_hi = bisect.bisect_right(msg_tslist, hi)
            if i_lo >= i_hi:
                continue
            chunk_weight = 0.0
            chunk_matched = 0
            chunk_evidence: list[tuple[Any, list[str]]] = []
            for m_ts, m_toks, m_obj in msg_pairs[i_lo:i_hi]:
                shared = c_toks & m_toks
                if not shared:
                    continue
                # Only weighted tokens (non-zero IDF) contribute. Stops
                # chat staples from dominating common-greeting windows.
                weighted_shared = [
                    t for t in shared if idf.get(t, 0.0) > 0
                ]
                if not weighted_shared:
                    continue
                pair_weight = sum(idf[t] for t in weighted_shared)
                chunk_weight += pair_weight
                chunk_matched += len(weighted_shared)
                chunk_evidence.append((m_obj, weighted_shared))
            if chunk_weight > 0:
                contributing_chunks += 1
                total_weighted += chunk_weight
                total_matched += chunk_matched
                # Buffer this chunk's evidence so we can surface it
                # if `off` ends up being the winning offset.
                evidence_per_offset[off].append(
                    (chunk_weight, c_obj, chunk_evidence),
                )
        score = (
            total_weighted / contributing_chunks
            if contributing_chunks else 0.0
        )
        results.append({
            "seconds": off,
            "score": round(score, 3),
            "matched_tokens": total_matched,
        })

    # Pick best + second-best for confidence check. Plateau handling:
    # when several adjacent offsets score within 2% of the max
    # (because the chat reaction-time spread fills more than one
    # window), pick the MEDIAN of the plateau rather than the first
    # one. The median is the centre of mass of the chat reaction
    # distribution, which is what we actually want to apply.
    max_score = max((r["score"] for r in results), default=0.0)
    if max_score > 0:
        plateau = [
            r for r in results
            if r["score"] >= max_score * 0.98
        ]
        plateau.sort(key=lambda r: r["seconds"])
        best = plateau[len(plateau) // 2]
    else:
        best = None
    # Second-best = highest-scoring offset OUTSIDE the plateau, so
    # the confidence margin reflects "how much better than the next
    # peak" rather than "how much better than my own neighbour".
    plateau_offsets = (
        {r["seconds"] for r in plateau} if max_score > 0 else set()
    )
    non_plateau = sorted(
        (r for r in results if r["seconds"] not in plateau_offsets),
        key=lambda r: r["score"], reverse=True,
    )
    second = non_plateau[0] if non_plateau else None

    ok = bool(best and best["matched_tokens"] > 0)

    # Confidence = how much the peak beats the runner-up that lives
    # OUTSIDE the plateau. 0..1; values above ~0.15 are trustworthy.
    # On greetings-only / low-signal windows the whole curve is flat
    # and confidence collapses to near-zero, which the UI surfaces as
    # an amber "low signal — wait for more chat" hint.
    confidence = 0.0
    if best is not None and best["score"] > 0:
        if second is not None:
            confidence = max(0.0, (best["score"] - second["score"]) / best["score"])
        else:
            confidence = 1.0

    # Build the evidence list for the winning offset only — top-K
    # chunk-chat pairs sorted by weight, so the streamer sees the
    # strongest matches first.
    evidence: list[dict[str, Any]] = []
    if ok and best is not None:
        winning_buffer = evidence_per_offset.get(best["seconds"], [])
        winning_buffer.sort(key=lambda r: r[0], reverse=True)
        for chunk_weight, c_obj, chat_pairs in winning_buffer[:_EVIDENCE_K]:
            chat_payload = []
            # Sort chat pairs by their own weight desc — strongest
            # match first. Matched-token list capped at 6 for display.
            chat_pairs_sorted = sorted(
                chat_pairs,
                key=lambda cp: sum(idf.get(t, 0.0) for t in cp[1]),
                reverse=True,
            )
            for m_obj, shared in chat_pairs_sorted[:5]:
                chat_payload.append({
                    "ts": _short_clock(getattr(m_obj, "ts", "")),
                    "name": getattr(m_obj, "name", "") or "?",
                    "content": (getattr(m_obj, "content", "") or "")[:200],
                    "matched": shared[:6],
                })
            evidence.append({
                "chunk_ts": _short_clock(getattr(c_obj, "ts", "")),
                "chunk_text": (getattr(c_obj, "text", "") or "")[:300],
                "chat": chat_payload,
                "weight": round(chunk_weight, 2),
            })

    return {
        "ok": ok,
        "reason": None if ok else "no token overlap at any offset",
        "samples": samples,
        "lookback_minutes": lookback_minutes,
        "offsets": results,
        "best_offset": best["seconds"] if ok else None,
        "second_best_offset": (
            second["seconds"] if (ok and second) else None
        ),
        "evidence": evidence,
        "weighted_token_universe": universe,
        "confidence": round(confidence, 3),
    }


def _short_clock(ts: str) -> str:
    """ISO timestamp → HH:MM:SS for compact display in the
    evidence list. Empty / unparsable falls through unchanged."""
    if not ts:
        return ""
    if len(ts) >= 19 and ts[10] in (" ", "T"):
        return ts[11:19]
    return ts
