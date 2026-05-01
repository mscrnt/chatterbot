"""Real-time whisper transcription + auto-match service.

The OBS audio relay script POSTs raw 16 kHz mono float32 PCM chunks to
the dashboard's `/audio/ingest` endpoint. This service:

  1. Buffers chunks per session.
  2. Every `whisper_buffer_seconds`, runs faster-whisper with VAD on the
     accumulated buffer (skips silence, keeps speech).
  3. For each transcript utterance, embeds the text via Ollama and
     cosine-matches against open insight cards (talking points + topic
     threads). When sim ≥ `whisper_match_threshold`, auto-sets the
     card's state to 'addressed' with the transcript as the note.
  4. Persists every chunk to `transcript_chunks` for the live strip
     on Insights and the audit trail.

`faster-whisper` is heavy (model files are 75 MB → 3 GB depending on
size). Lazy-imported so the dashboard boots fine when the feature is
disabled. The whisper model is loaded once on first audio chunk.

Hard rule reminder: this is streamer-only. Transcripts of the streamer's
voice render to the streamer's private dashboard. They never enter any
chat-facing prompt.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import struct
import time
from dataclasses import dataclass, field

import numpy as np


_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_TRAIL_DIGITS_RE = re.compile(r"\d+$")


def _stitch_grid(image_paths: list[str], max_size: tuple[int, int] = (960, 540)) -> bytes | None:
    """Stitch up to 4 JPEGs into a 2x2 grid (or 1x1 / 1x2 for fewer).
    Returns JPEG bytes ready to base64-encode for the multimodal LLM
    call. Returns None if no images load — not an error, just "skip
    the image attachment for this group."

    Lazy-imports Pillow so the dashboard boots fine without it (the
    feature is opt-in via screenshot_interval_seconds > 0)."""
    try:
        from PIL import Image
    except ImportError:
        logger.warning(
            "transcript: Pillow not installed — skipping screenshot grid. "
            "Run `uv sync --extra whisper` to enable."
        )
        return None
    paths = [p for p in image_paths if p]
    if not paths:
        return None
    n = min(4, len(paths))
    if n == 1:
        cols, rows = 1, 1
    elif n == 2:
        cols, rows = 2, 1
    else:  # 3 or 4 — always 2x2 with last cell blank if N=3
        cols, rows = 2, 2
    cell_w = max_size[0] // cols
    cell_h = max_size[1] // rows
    canvas = Image.new("RGB", (cell_w * cols, cell_h * rows), (10, 10, 14))
    for i, path in enumerate(paths[:n]):
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            continue
        im.thumbnail((cell_w, cell_h))
        col = i % cols
        row = i // cols
        x = col * cell_w + (cell_w - im.width) // 2
        y = row * cell_h + (cell_h - im.height) // 2
        canvas.paste(im, (x, y))
    import io as _io
    buf = _io.BytesIO()
    canvas.save(buf, format="JPEG", quality=70, optimize=True)
    return buf.getvalue()


def _encode_webp(jpeg_bytes: bytes, quality: int) -> bytes | None:
    """Transcode JPEG bytes to WebP for compact, content-deduplicable
    long-term storage. WebP lands ~25-35% smaller than JPEG at the
    same visual quality, which matters since screenshots are kept
    forever (no age-based prune); content-hash dedup (paused stream
    on the same scene) compounds the savings further.

    Returns None on any PIL failure so the caller falls back to
    skipping the capture for that interval — never crashes the loop."""
    try:
        from PIL import Image
        from io import BytesIO
        img = Image.open(BytesIO(jpeg_bytes))
        # Drop alpha if present — JPEGs don't carry it but defensive
        # if a future capture path does, since WebP would otherwise
        # paint a black background through the alpha channel.
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        out = BytesIO()
        # method=4 is the libwebp default — balanced speed/quality.
        # Higher methods (5-6) shave a few percent off filesize at
        # ~2x encode time; not worth it on a 480px frame.
        img.save(out, format="WEBP", quality=int(quality), method=4)
        return out.getvalue()
    except ImportError:
        logger.warning(
            "transcript: Pillow not installed — webp transcode skipped. "
            "Run `uv sync --extra whisper` to enable.",
        )
        return None
    except Exception:
        logger.exception("transcript: webp transcode failed")
        return None


def _content_hashed_relpath(content: bytes, ext: str = ".webp") -> str:
    """Build the on-disk relative path for a content-addressed
    screenshot. Layout `<sha[:2]>/<sha[2:4]>/<sha><ext>` keeps any
    one directory bounded to ~256 entries even at huge scale (uniform
    hash distribution), avoiding the directory-bloat problem that
    flat layouts hit after a few months of streaming.

    Uses sha256 hex; the full hex digest is the filename so the
    streamer can spot-check identity without a separate manifest."""
    import hashlib
    sha = hashlib.sha256(content).hexdigest()
    return f"{sha[:2]}/{sha[2:4]}/{sha}{ext}"


def _utterance_mentions_chatter(utterance: str, names: list[str]) -> bool:
    """Heuristic: did the streamer name the chatter in this utterance?

    Whisper transcribes phonetically and inserts whitespace freely, so
    "aquanote1" can come back as "aqua note one" or "aqua note 1". We
    normalize both sides to lowercase alphanumerics-only and check
    substring containment. We also try the digit-stripped variant for
    handles like "asmongold123" → look for "asmongold". Names shorter
    than 4 chars after stripping are rejected as too noisy to gate on.
    """
    if not utterance or not names:
        return False
    u_norm = _ALNUM_RE.sub("", utterance.lower())
    if not u_norm:
        return False
    for name in names:
        if not name:
            continue
        n_norm = _ALNUM_RE.sub("", name.lower())
        if len(n_norm) >= 4 and n_norm in u_norm:
            return True
        n_stripped = _TRAIL_DIGITS_RE.sub("", n_norm)
        if len(n_stripped) >= 4 and n_stripped in u_norm:
            return True
    return False

from .config import Settings
from .insights import INFORMED_NUM_CTX
from .llm.ollama_client import OllamaClient
from .repo import ChatterRepo

logger = logging.getLogger(__name__)


SAMPLE_RATE = 16000  # whisper's required input rate
SAMPLE_BYTES = 4     # float32


@dataclass
class _IngestBuffer:
    """In-memory audio buffer. Float32 mono PCM at 16 kHz.

    Tracks the capture time of the OLDEST chunk in the current buffer
    (`first_captured_at`) so the resulting transcript chunk can be
    timestamped against actual audio time, not wall-clock arrival
    time. This matters when the OBS-side audio_client buffered chunks
    during a dashboard outage: those chunks should appear on the
    timeline at their original capture moment, not at the moment they
    finally got POSTed."""

    samples: list[np.ndarray] = field(default_factory=list)
    total_samples: int = 0
    last_flush: float = field(default_factory=time.time)
    # ISO string of the earliest captured-at the audio_client reported
    # for chunks in this buffer. None when no captured-at was sent
    # (older audio_client versions / no header) — caller falls back
    # to wall-clock now in that case.
    first_captured_at: str | None = None

    def append(
        self, pcm: np.ndarray, *, captured_at: str | None = None,
    ) -> None:
        self.samples.append(pcm)
        self.total_samples += pcm.shape[0]
        # First captured-at after a flush wins. Subsequent appends in
        # the same window keep the earliest reference (so the
        # resulting transcript chunk lines up with when the streamer
        # started speaking, not when whisper finished processing).
        if captured_at and self.first_captured_at is None:
            self.first_captured_at = captured_at

    def flush(self) -> tuple[np.ndarray, str | None]:
        if not self.samples:
            return np.zeros(0, dtype=np.float32), None
        out = np.concatenate(self.samples).astype(np.float32, copy=False)
        captured_at = self.first_captured_at
        self.samples.clear()
        self.total_samples = 0
        self.last_flush = time.time()
        self.first_captured_at = None
        return out, captured_at


class TranscriptService:
    """Owns the whisper model + ingest buffer + match loop. Thread-safe
    enough for the dashboard's single-process usage."""

    def __init__(
        self, repo: ChatterRepo, llm: OllamaClient, settings: Settings,
        *, obs=None, twitch_status=None,
    ):
        self.repo = repo
        self.llm = llm
        self.settings = settings
        self.obs = obs  # OBSStatusService — used for screenshot capture
        # Optional[TwitchService] — feeds the live Helix snapshot
        # (game_name, title, viewer_count, etc.) into the group-summary
        # prompt as authoritative channel context, so the LLM stops
        # guessing the game from the screenshot pixels.
        self.twitch_status = twitch_status
        self._buffer = _IngestBuffer()
        self._lock = asyncio.Lock()
        # Cached initial_prompt for whisper. Rebuilt every
        # _initial_prompt_ttl_s seconds so it stays current as
        # chatters come and go and the streamer changes games. Null
        # when whisper_initial_prompt_enabled=False.
        self._initial_prompt: str | None = None
        self._initial_prompt_built_at: float = 0.0
        self._initial_prompt_ttl_s: float = 30.0
        # streamer_facts.md mtime — separate from the prompt cache
        # so we re-read the file only when it actually changes.
        self._streamer_facts_text: str = ""
        self._streamer_facts_mtime: float = 0.0
        # Cached top-words list from the wordcloud query. Heavier
        # query (full message scan + regex), so a 5-min TTL is plenty
        # — chat vocabulary doesn't shift faster than that.
        self._top_chat_words: list[str] = []
        self._top_chat_words_built_at: float = 0.0
        self._top_chat_words_ttl_s: float = 300.0
        self._model = None  # lazy-loaded WhisperModel
        self._model_load_attempted = False
        self._model_load_error: str | None = None
        # Cached embeddings of open insight cards. Refreshed every minute
        # so adds/removes propagate. Keyed by (kind, item_key) → vector.
        self._card_embeds: dict[tuple[str, str], list[float]] = {}
        # Original text behind each cached embedding — used for the
        # debug log line so the streamer can see WHY a transcript chunk
        # closely matched a card.
        self._card_embed_texts: dict[tuple[str, str], str] = {}
        # For talking_point cards: the names the streamer might call the
        # chatter by (display name + historical aliases). When
        # `whisper_require_chatter_name` is on, we hard-gate the match
        # on at least one of these appearing in the utterance — vague
        # statements like "why that happened" otherwise mop up generic
        # talking-points.
        self._card_chatter_names: dict[tuple[str, str], list[str]] = {}
        self._card_embeds_refreshed_at: float = 0.0
        self._card_embeds_lock = asyncio.Lock()
        # Optional callback that returns the current talking points.
        # Wired by the dashboard so we can index them too.
        self._talking_points_provider = None
        # Counter incremented every time we write 'auto_pending'. The
        # /transcript route reads this and emits HX-Trigger so the
        # insights body re-renders the moment a new pending appears.
        self._pending_count = 0

    @property
    def enabled(self) -> bool:
        return bool(self.settings.whisper_enabled)

    async def _ensure_model(self):
        """Lazy-load the whisper model on first ingest. Heavy import +
        download — keep it off the dashboard boot path."""
        if self._model is not None or self._model_load_attempted:
            return self._model
        self._model_load_attempted = True
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            self._model_load_error = (
                f"faster-whisper not installed: {e}. "
                "Add it to pyproject.toml and `uv sync`."
            )
            logger.error("transcript: %s", self._model_load_error)
            return None
        compute = self.settings.whisper_compute_type
        try:
            # Run model load in a thread so we don't block the event loop.
            self._model = await asyncio.to_thread(
                WhisperModel,
                self.settings.whisper_model,
                device="auto",
                compute_type=compute,
            )
            logger.info(
                "transcript: loaded whisper model=%s compute=%s",
                self.settings.whisper_model, compute,
            )
        except Exception as e:
            self._model_load_error = f"{type(e).__name__}: {e}"
            logger.exception("transcript: failed to load whisper model")
        return self._model

    async def ingest_chunk(
        self,
        payload: bytes,
        sample_rate: int,
        *,
        captured_at: str | None = None,
    ) -> None:
        """Append a raw PCM chunk to the buffer. The OBS script sends
        16 kHz mono float32; if a different rate is supplied, we
        resample.

        `captured_at` is the ISO timestamp the audio was originally
        captured at on the OBS side (sent via `X-Captured-At` header).
        Optional — falls back to wall-clock now during transcription
        when absent. Lets the dashboard place buffered audio at the
        right point on the timeline after a long outage."""
        if not self.enabled:
            return
        if sample_rate != SAMPLE_RATE:
            # Linear resample is fine for VAD/whisper preprocessing —
            # whisper itself does its own internal mel-spec conversion.
            x = np.frombuffer(payload, dtype=np.float32)
            ratio = SAMPLE_RATE / float(sample_rate)
            n_out = int(x.shape[0] * ratio)
            if n_out <= 0:
                return
            idx = (np.arange(n_out) / ratio).astype(np.int64)
            idx = np.clip(idx, 0, x.shape[0] - 1)
            x = x[idx]
        else:
            x = np.frombuffer(payload, dtype=np.float32)

        async with self._lock:
            self._buffer.append(x, captured_at=captured_at)
            should_flush = (
                self._buffer.total_samples
                >= int(SAMPLE_RATE * self.settings.whisper_buffer_seconds)
            )

        if should_flush:
            asyncio.create_task(self._transcribe_and_match())

    def _load_streamer_facts_text(self) -> str:
        """Read the streamer-authored facts file with an mtime cache.
        Returns the file's contents (capped to ~1500 chars) or '' if
        the file is missing / unreadable / disabled.

        Same file the engaging-subjects extractor + insights service
        read; here we use it as vocabulary bias for whisper. Cache
        survives across transcribe calls so the disk read happens
        at most once per file change."""
        from pathlib import Path
        path_str = getattr(self.settings, "streamer_facts_path", "") or ""
        if not path_str:
            return ""
        p = Path(path_str)
        if not p.is_absolute():
            p = Path.cwd() / p
        try:
            mtime = p.stat().st_mtime
        except OSError:
            self._streamer_facts_text = ""
            self._streamer_facts_mtime = 0.0
            return ""
        if mtime != self._streamer_facts_mtime:
            try:
                text = p.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                text = ""
            # Cap at 1500 chars so the prompt stays in vocabulary-hint
            # territory rather than crowding the audio context.
            self._streamer_facts_text = text[:1500]
            self._streamer_facts_mtime = mtime
        return self._streamer_facts_text

    def _build_initial_prompt(self) -> str | None:
        """Construct an `initial_prompt` for faster-whisper from
        runtime context. Whisper treats this as vocabulary bias —
        words present in the prompt are more likely to be transcribed
        correctly when they appear in the audio. Massive accuracy
        boost on niche terms (game-specific vocabulary, chatter
        handles, the streamer's own name).

        Cached for `_initial_prompt_ttl_s` seconds so we don't rebuild
        on every 5 s buffer flush. Returns None when the feature is
        disabled or the prompt would be empty.

        Bounded at ~250 chars total — too long and whisper starts
        treating it as a transcript prefix instead of vocabulary
        seed, which produces hallucinated continuations of the
        prompt text."""
        if not bool(getattr(
            self.settings, "whisper_initial_prompt_enabled", True,
        )):
            return None
        now = time.time()
        if (
            self._initial_prompt is not None
            and (now - self._initial_prompt_built_at) < self._initial_prompt_ttl_s
        ):
            return self._initial_prompt

        parts: list[str] = []

        # Streamer name + current game from the live Helix poll.
        ts = getattr(self.twitch_status, "status", None) if self.twitch_status else None
        if ts is not None:
            name = (
                (getattr(ts, "broadcaster_display_name", None) or "").strip()
                or (getattr(ts, "broadcaster_login", None) or "").strip()
            )
            game = (getattr(ts, "game_name", None) or "").strip()
            if name and game:
                parts.append(f"Streamer {name} is playing {game}.")
            elif name:
                parts.append(f"Streamer: {name}.")
            elif game:
                parts.append(f"Currently playing {game}.")

        # Active chatter handles — Whisper biases toward names it sees,
        # so even a comma-separated list works. Cap at 25 names to
        # keep the prompt under whisper's effective vocab budget.
        try:
            active = self.repo.list_active_chatters(20, 25)
            names = [u.name for u in active if u.name]
            if names:
                parts.append("Chatters: " + ", ".join(names) + ".")
        except Exception:
            pass

        # Top chat-frequent words from the last week — game terms,
        # character names, recurring topics chat returns to. Reuses
        # the same stats_top_words query that backs /stats/wordcloud.
        # Cached separately (5min) since the query is heavier than
        # the chatter-list lookup.
        if (now - self._top_chat_words_built_at) >= self._top_chat_words_ttl_s:
            try:
                words = self.repo.stats_top_words(
                    limit=60, min_count=3, lookback_days=7,
                )
                # Drop words shorter than 4 chars — whisper doesn't
                # need help with "yes" / "the" / etc, and short tokens
                # bias too aggressively toward common syllables.
                self._top_chat_words = [
                    w for w, _c in words if len(w) >= 4
                ][:60]
            except Exception:
                self._top_chat_words = []
            self._top_chat_words_built_at = now
        if self._top_chat_words:
            parts.append("Common terms: " + ", ".join(self._top_chat_words) + ".")

        # Streamer-authored facts (terms / inside jokes / recurring bits).
        facts = self._load_streamer_facts_text()
        if facts:
            parts.append(facts)

        # Streamer-supplied free-form extra (settings field).
        extra = (
            getattr(self.settings, "whisper_initial_prompt_extra", "") or ""
        ).strip()
        if extra:
            parts.append(extra)

        prompt = " ".join(parts).strip()
        if not prompt:
            self._initial_prompt = None
        else:
            # Hard cap: whisper's prompt tokenizer cuts off around 200
            # tokens; anything beyond that is dropped. 1200 chars
            # ≈ 250 tokens, comfortably within budget.
            self._initial_prompt = prompt[:1200]
        self._initial_prompt_built_at = now
        return self._initial_prompt

    async def _transcribe_and_match(self) -> None:
        """Pop the buffer, transcribe the audio, embed each segment,
        match against open insight cards, persist. Runs as a fire-and-
        forget task so the ingest endpoint stays snappy.

        The buffer's earliest captured-at (when the OBS audio_client
        sent X-Captured-At) is used as the resulting transcript
        chunk's `ts` so buffered audio from a dashboard outage lands
        at its actual capture moment, not at the moment whisper
        finished processing it. None falls back to wall-clock now in
        add_transcript_chunk."""
        async with self._lock:
            audio, buffer_captured_at = self._buffer.flush()
        if audio.shape[0] == 0:
            return
        model = await self._ensure_model()
        if model is None:
            return

        # Whisper transcribe. VAD filter strips silence inside the buffer
        # so we don't end up with "you you you" hallucinations on quiet
        # patches. Run synchronously in a thread — whisper is CPU/GPU
        # bound and isn't async-friendly.
        min_silence_ms = max(100, int(self.settings.whisper_min_silence_ms or 5000))
        # Streamer-friendly tuning. Defaults are calibrated for fast,
        # emotional, often-mumbled speech — see the field comments in
        # config.py for the why behind each. All knobs are user-
        # editable in /settings → Voice & screen → Card matching /
        # Audio basics.
        beam = max(1, int(getattr(self.settings, "whisper_beam_size", 3) or 3))
        no_speech = float(getattr(
            self.settings, "whisper_no_speech_threshold", 0.4,
        ))
        log_prob = float(getattr(
            self.settings, "whisper_log_prob_threshold", -1.5,
        ))
        vad_thr = float(getattr(self.settings, "whisper_vad_threshold", 0.3))
        prompt = self._build_initial_prompt()

        def _run() -> list[tuple[str, float, float]]:
            try:
                segments, _info = model.transcribe(
                    audio,
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": min_silence_ms,
                        "threshold": vad_thr,
                    },
                    beam_size=beam,
                    no_speech_threshold=no_speech,
                    log_prob_threshold=log_prob,
                    condition_on_previous_text=False,
                    initial_prompt=prompt,
                )
                return [(s.text.strip(), s.start, s.end) for s in segments]
            except Exception:
                logger.exception("transcript: whisper.transcribe failed")
                return []

        segments = await asyncio.to_thread(_run)
        if not segments:
            return

        # Refresh card embeddings if stale.
        await self._refresh_card_embeds_if_stale()

        threshold = float(self.settings.whisper_match_threshold)
        unnamed_threshold = float(
            getattr(self.settings, "whisper_unnamed_match_threshold", 0.80)
        )
        for text, start_s, end_s in segments:
            text = (text or "").strip()
            if not text or len(text) < 4:
                continue
            duration_ms = max(0, int((end_s - start_s) * 1000))
            best_kind, best_key, best_sim, best_text, qv = (
                await self._best_candidate(text)
            )
            # For talking_point cards, scale the bar by whether the
            # streamer named the chatter. Naming them ("that's right
            # aquanote1, …") is the strongest signal of actual
            # engagement — without it, the points are short and generic
            # enough that vague utterances ("why that happened") clear
            # 0.55 against unrelated cards. Setting unnamed_threshold
            # above 1.0 disables unnamed matching entirely; setting it
            # equal to threshold restores the old uniform behavior.
            named = (
                best_kind == "talking_point"
                and _utterance_mentions_chatter(
                    text,
                    self._card_chatter_names.get((best_kind, best_key), []),
                )
            )
            if best_kind == "talking_point" and not named:
                effective_threshold = unnamed_threshold
            else:
                effective_threshold = threshold
            crossed = best_sim is not None and best_sim >= effective_threshold
            # Always store the best candidate AND the embedding so the
            # chat ↔ transcript reverse lookup (a chat message can find
            # a recent utterance it semantically answered) works.
            chunk_id = await asyncio.to_thread(
                self.repo.add_transcript_chunk,
                text=text, duration_ms=duration_ms,
                matched_kind=best_kind, matched_item_key=best_key,
                similarity=best_sim, embedding=qv,
                ts=buffer_captured_at,
            )
            # When the batched LLM matcher is active, it owns auto-pending
            # writes — per-utterance cosine still runs to populate
            # transcript_chunks.matched_* (drives the strip icons + the
            # chat→transcript reverse lookup) but we don't write
            # auto_pending here. The LLM pass periodically reviews the
            # window with full streamer-aware context and flips cards
            # then. Two writers to the same insight_state would race.
            llm_match_enabled = bool(getattr(
                self.settings, "whisper_llm_match_enabled", True,
            ))
            if llm_match_enabled:
                # Per-utterance cosine still ran (it annotated the chunk
                # row + the strip icons), but the LLM batch is the
                # auto-pending writer. Skip the per-utterance branches
                # below — and don't log a line per crossed utterance,
                # because at ~1 line/sec while talking it's pure noise.
                # The per-pass summary line already tells you what
                # mattered.
                continue
            if crossed:
                # Don't auto-flip to 'addressed' immediately — flip to
                # 'auto_pending' so the streamer can confirm/reject in
                # the UI before it sticks. The auto_confirm_loop
                # promotes any unconfirmed pendings to 'addressed' after
                # the timeout, so passive workflow still works.
                try:
                    await asyncio.to_thread(
                        self.repo.set_insight_state,
                        best_kind, best_key, "auto_pending",
                        note=f"(auto) {text}",
                    )
                    self._pending_count += 1
                    # Drop the just-pendinged card from the in-memory
                    # match pool so the next utterance can't keep
                    # matching it (the streamer is presumably still
                    # talking about the same thing — let *other* cards
                    # win their similarity contest). The next periodic
                    # refresh would do this too, but it's up to 60s out;
                    # without this the live transcript spams near-miss
                    # logs against the same card the whole window.
                    self._card_embeds.pop((best_kind, best_key), None)
                    self._card_embed_texts.pop((best_kind, best_key), None)
                    logger.info(
                        "transcript: AUTO-PENDING %s/%s sim=%.3f "
                        "spoken=%r → card=%r",
                        best_kind, best_key, best_sim,
                        text[:80], (best_text or "")[:80],
                    )
                except Exception:
                    logger.exception("transcript: auto-pending write failed")
            elif best_sim is not None:
                # Tag whether the unnamed-threshold gate was the one
                # blocking — makes it obvious in logs why otherwise
                # decent-similarity utterances aren't auto-pending.
                gate = (
                    "unnamed-threshold"
                    if best_kind == "talking_point" and not named
                    else "threshold"
                )
                logger.info(
                    "transcript: near-miss sim=%.3f (%s=%.2f) "
                    "spoken=%r ≈ %s/%s=%r",
                    best_sim, gate, effective_threshold,
                    text[:80], best_kind, best_key, (best_text or "")[:80],
                )
            else:
                logger.info(
                    "transcript: no cards indexed yet — chunk #%d: %r",
                    chunk_id, text[:80],
                )

    async def _refresh_card_embeds_if_stale(self, *, max_age_s: float = 60) -> None:
        async with self._card_embeds_lock:
            if (time.time() - self._card_embeds_refreshed_at) < max_age_s:
                return
            await self._refresh_card_embeds_locked()

    async def _refresh_card_embeds_locked(self) -> None:
        """Build the cosine-search target set from currently-open cards.
        Skipped cards (state=skipped) are excluded; addressed/snoozed
        also excluded so we don't re-flip them. Includes:
          - active + dormant topic threads (their titles)
          - talking points from the latest insights cache
        """
        targets: list[tuple[str, str, str]] = []  # (kind, item_key, text)

        # Topic threads: title is the searchable text, item_key = id.
        try:
            threads = await asyncio.to_thread(
                self.repo.list_threads, status_filter=None, query="", limit=200,
            )
            t_states = await asyncio.to_thread(
                self.repo.get_insight_states, "thread"
            )
        except Exception:
            threads = []
            t_states = {}
        for t in threads:
            s = t_states.get(str(t.id))
            # auto_pending also skipped: the card is awaiting confirm/reject
            # or the 60s auto-promote. Keeping it in the pool means every
            # subsequent utterance keeps "matching" the same card (logging
            # AUTO-PENDING once, then a stream of near-misses against it),
            # which makes the live transcript feel like a broken record.
            if s and s.state in ("addressed", "snoozed", "skipped", "auto_pending"):
                continue
            if t.status == "archived":
                continue
            targets.append(("thread", str(t.id), t.title))

        # Talking points: the LLM's "active right now" suggestions. The
        # cache is built by InsightsService — we just read it. item_key
        # has to match the hash format the dashboard renderer uses so
        # auto-address writes flip the right card.
        try:
            from .insights import InsightsService  # type: ignore  # circular-safe in lazy import
            # The dashboard process owns the InsightsService instance;
            # we don't have a direct handle, so we fall back to repo
            # for the current cache by walking through the insights
            # ack/cache surface. Simplest path: read from the same
            # talking_points list the dashboard renders.
        except Exception:
            pass
        try:
            tps = list(self._talking_points_provider() if self._talking_points_provider else [])
        except Exception:
            logger.exception("transcript: talking_points_provider raised")
            tps = []
        import hashlib as _h
        new_chatter_names: dict[tuple[str, str], list[str]] = {}
        for tp in tps:
            point = (tp.point or "").strip()
            if not point:
                continue
            uid = tp.user_id
            digest = _h.sha1(f"{uid}|{point}".encode("utf-8")).hexdigest()[:16]
            item_key = f"{uid}:{digest}"
            s_states = await asyncio.to_thread(
                self.repo.get_insight_states, "talking_point"
            )
            s = s_states.get(item_key)
            # auto_pending also skipped: the card is awaiting confirm/reject
            # or the 60s auto-promote. Keeping it in the pool means every
            # subsequent utterance keeps "matching" the same card (logging
            # AUTO-PENDING once, then a stream of near-misses against it),
            # which makes the live transcript feel like a broken record.
            if s and s.state in ("addressed", "snoozed", "skipped", "auto_pending"):
                continue
            # Search text: blend chatter name + point so "ask alice
            # about her cat" can match either side.
            targets.append(("talking_point", item_key, f"{tp.name}: {point}"))
            # Stash every name the streamer might call this chatter by,
            # for the require-name gate at match time. Aliases handle the
            # rename trail (zackrawrr → asmongold etc).
            names: list[str] = [str(tp.name)] if tp.name else []
            try:
                aliases = await asyncio.to_thread(self.repo.get_user_aliases, str(uid))
                for a in aliases:
                    if a.name and a.name not in names:
                        names.append(a.name)
            except Exception:
                pass
            new_chatter_names[("talking_point", item_key)] = names

        # Resolve embeddings — re-use the cached embedding when we have
        # one for that key; otherwise call Ollama.
        new_embeds: dict[tuple[str, str], list[float]] = {}
        for kind, key, text in targets:
            cached = self._card_embeds.get((kind, key))
            if cached is not None:
                new_embeds[(kind, key)] = cached
                continue
            try:
                vec = await self.llm.embed(text)
                new_embeds[(kind, key)] = vec
            except Exception:
                logger.exception("transcript: embed failed for %s/%s", kind, key)
        self._card_embeds = new_embeds
        self._card_embed_texts = {(k, key): t for k, key, t in targets}
        self._card_chatter_names = new_chatter_names
        self._card_embeds_refreshed_at = time.time()
        logger.info(
            "transcript: refreshed %d card embeddings (%d threads, %d talking-points)",
            len(new_embeds),
            sum(1 for k, _ in new_embeds.keys() if k == "thread"),
            sum(1 for k, _ in new_embeds.keys() if k == "talking_point"),
        )

    async def _best_candidate(
        self, text: str,
    ) -> tuple[str | None, str | None, float | None, str | None, list[float] | None]:
        """Embed transcript text and return the closest card by cosine
        similarity, REGARDLESS of threshold. Returns
        (kind, item_key, similarity, target_text, query_embedding).

        The embedding is also returned so the caller can persist it
        into vec_transcripts — used for the chat ↔ transcript reverse
        lookup (chat message can find a recent utterance it answered)."""
        try:
            qv = await self.llm.embed(text)
        except Exception:
            logger.exception("transcript: query embed failed")
            return None, None, None, None, None
        if not self._card_embeds:
            return None, None, None, None, qv
        q = np.asarray(qv, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn <= 0:
            return None, None, None, None, qv
        best_kind: str | None = None
        best_key: str | None = None
        best_sim = -1.0
        for (kind, key), vec in self._card_embeds.items():
            v = np.asarray(vec, dtype=np.float32)
            vn = float(np.linalg.norm(v))
            if vn <= 0:
                continue
            sim = float(np.dot(q, v) / (qn * vn))
            if sim > best_sim:
                best_sim = sim
                best_kind = kind
                best_key = key
        if best_kind is None:
            return None, None, None, None, qv
        target_text = self._card_embed_texts.get((best_kind, best_key))
        return best_kind, best_key, best_sim, target_text, qv

    LLM_MATCH_WATERMARK_KEY = "transcript_llm_watermark"
    # When the bundled LLM matcher runs, give it 16k context — comfortably
    # holds 30+ minutes of speech transcripts plus a healthy candidate list.
    # Qwen 3 supports 256K natively; we just need enough headroom that we
    # never silently truncate a long stream window.
    LLM_MATCH_NUM_CTX = 16384
    LLM_MATCH_FETCH_LIMIT = 400

    LLM_MATCH_SYSTEM = """You are watching the live voice transcript of a Twitch streamer.

IMPORTANT — whisper output is imperfect. Expect misheard words, garbled proper nouns and game/character names, missing punctuation, run-on phrases, and the streamer trailing off or overlapping with game audio. When matching utterances to chatter names or topic threads, allow for phonetic spellings ("aqua note one" can be "aquanote1"; "purry" near gameplay context is likely "parry"). If a word seems garbled, infer the most plausible reading from context before deciding whether it's a match.


The streamer is most likely:
  - playing a game and reacting to gameplay
  - reacting to media (videos, articles, news)
  - thinking aloud / monologuing about whatever's on their mind

Most utterances are NOT directed at chat. They're game callouts, swearing at a boss, narrating a video, or stream-of-consciousness. Your job is NOT to find loose thematic overlap with the listed cards — that produces false positives every few seconds.

You are looking for the rare, clear cases where the streamer has actually engaged with one of the listed insight cards. Two patterns count:

  1. The streamer addresses a chatter directly by name. Example utterances: "thanks aquanote1", "good point bon3sy3", "yeah you're right kid_dingo, that did happen". The chatter's name (or a clear phonetic spelling — Whisper transcribes phonetically and may insert spaces or substitute digits with words) must appear in the utterance for a chatter card to match.

  2. The streamer has a substantive on-topic discussion that clearly aligns with one of the listed topic threads — multiple sentences engaging with that exact theme. A passing word that happens to share vocabulary is NOT a match.

When in doubt, return NO match. Empty `matches` list is the expected, common output. Each `evidence` field must quote the exact utterance fragment that justifies the match (verbatim, ≤ 200 chars). `confidence` ∈ [0, 1] — only emit ≥ 0.6 for cases you're genuinely sure of."""

    def _bundle_card_lines(self) -> tuple[list[str], dict[int, tuple[str, str]]]:
        """Build the numbered candidate-cards block for the matcher prompt
        and a `card_id -> (kind, item_key)` lookup so we can resolve the
        LLM's chosen ids back to insight states.

        Reuses the same `_card_embed_texts` / `_card_chatter_names` caches
        the per-utterance matcher uses, so we don't double-fetch from the
        repo. The cosine refresh runs every 60s already.
        """
        lines: list[str] = []
        lookup: dict[int, tuple[str, str]] = {}
        next_id = 1
        # Talking-point cards first — they're per-chatter and get the
        # name-based matching pattern.
        for (kind, key), text in self._card_embed_texts.items():
            if kind != "talking_point":
                continue
            names = self._card_chatter_names.get((kind, key)) or []
            primary = names[0] if names else ""
            tail = (
                f" (also known as: {', '.join(names[1:])})"
                if len(names) > 1 else ""
            )
            lines.append(
                f"  TP-{next_id}  chatter='{primary}'{tail}  point: {text}"
            )
            lookup[next_id] = (kind, key)
            next_id += 1
        # Topic threads — broader-topic cards.
        for (kind, key), text in self._card_embed_texts.items():
            if kind != "thread":
                continue
            lines.append(f"  TP-{next_id}  topic-thread: {text}")
            lookup[next_id] = (kind, key)
            next_id += 1
        return lines, lookup

    async def _run_llm_match(self) -> int:
        """One pass of the batched LLM matcher. Returns the number of
        cards flipped to `auto_pending` (0 in the common case where the
        streamer was just gaming / monologuing)."""
        # Make sure the cosine target pool is fresh — we share its cache
        # for the candidate-cards prompt block.
        await self._refresh_card_embeds_if_stale()
        if not self._card_embed_texts:
            return 0

        watermark_str = await asyncio.to_thread(
            self.repo.get_app_setting, self.LLM_MATCH_WATERMARK_KEY,
        )
        try:
            watermark = int(watermark_str) if watermark_str else 0
        except ValueError:
            watermark = 0

        chunks = await asyncio.to_thread(
            self.repo.list_transcripts_after_id,
            watermark, limit=self.LLM_MATCH_FETCH_LIMIT,
        )
        min_chunks = int(getattr(self.settings, "whisper_llm_match_min_chunks", 3))
        if len(chunks) < min_chunks:
            return 0

        card_lines, card_lookup = self._bundle_card_lines()
        if not card_lines:
            # Nothing to match against — still advance the watermark so we
            # don't re-process this window when cards eventually appear.
            await asyncio.to_thread(
                self.repo.set_app_setting,
                self.LLM_MATCH_WATERMARK_KEY, str(chunks[-1].id),
            )
            return 0

        # Number utterances; keep the ids out of the LLM's response space
        # (it returns card_id, not utterance index — we extract evidence
        # quotes verbatim).
        utterance_lines = [f"  - {c.text}" for c in chunks]
        prompt = (
            "CANDIDATE CARDS:\n"
            + "\n".join(card_lines)
            + "\n\nSTREAMER UTTERANCES (oldest first, "
            + f"{len(chunks)} chunks over the last window):\n"
            + "\n".join(utterance_lines)
            + "\n\nReturn the cards (by card_id) that the streamer demonstrably "
            "engaged with. Empty list if none — that's the expected common case. "
            "For each match include a verbatim `evidence` quote from the utterances "
            "above and a `confidence` in [0,1]."
        )

        try:
            from .llm.schemas import TranscriptMatchResponse
            response = await self.llm.generate_structured(
                prompt=prompt,
                system_prompt=self.LLM_MATCH_SYSTEM,
                response_model=TranscriptMatchResponse,
                num_ctx=self.LLM_MATCH_NUM_CTX,
                call_site="transcript.llm_match",
            )
        except Exception:
            logger.exception("transcript: llm match call failed")
            return 0

        confidence_floor = float(
            getattr(self.settings, "whisper_llm_match_confidence", 0.65)
        )
        flipped = 0
        for m in response.matches:
            pair = card_lookup.get(m.card_id)
            if pair is None:
                logger.info(
                    "transcript: llm-match dropped — bad card_id %s "
                    "(evidence=%r conf=%.2f)",
                    m.card_id, m.evidence[:80], m.confidence,
                )
                continue
            if m.confidence < confidence_floor:
                logger.info(
                    "transcript: llm-match below floor — %s/%s conf=%.2f "
                    "(floor=%.2f) evidence=%r",
                    pair[0], pair[1], m.confidence, confidence_floor,
                    m.evidence[:80],
                )
                continue
            kind, key = pair
            try:
                await asyncio.to_thread(
                    self.repo.set_insight_state,
                    kind, key, "auto_pending",
                    note=f"(auto) {m.evidence}",
                )
                self._pending_count += 1
                # Same eviction trick as the per-utterance path — drop
                # the card from the in-memory pool so it doesn't keep
                # surfacing in the strip's near-miss icons.
                self._card_embeds.pop((kind, key), None)
                self._card_embed_texts.pop((kind, key), None)
                flipped += 1
                logger.info(
                    "transcript: llm AUTO-PENDING %s/%s conf=%.2f "
                    "evidence=%r",
                    kind, key, m.confidence, m.evidence[:120],
                )
            except Exception:
                logger.exception(
                    "transcript: llm auto-pending write failed for %s/%s",
                    kind, key,
                )

        # Advance the watermark unconditionally — if we re-processed the
        # same window we'd get the same answer.
        await asyncio.to_thread(
            self.repo.set_app_setting,
            self.LLM_MATCH_WATERMARK_KEY, str(chunks[-1].id),
        )
        logger.info(
            "transcript: llm match pass — %d chunks (id %d → %d), "
            "%d cards flipped, %d candidates",
            len(chunks), chunks[0].id, chunks[-1].id, flipped, len(card_lines),
        )
        return flipped

    async def llm_match_loop(self) -> None:
        """Background task: run the batched LLM matcher every
        `whisper_llm_match_interval_seconds`. No-op when whisper is
        disabled or the LLM matcher itself is disabled in settings."""
        # Small startup delay so the dashboard finishes booting and we
        # don't compete with the talking-points cache's first refresh.
        await asyncio.sleep(20)
        while True:
            try:
                interval = max(15, int(getattr(
                    self.settings, "whisper_llm_match_interval_seconds", 90,
                )))
                await asyncio.sleep(interval)
                if not self.enabled:
                    continue
                if not bool(getattr(self.settings, "whisper_llm_match_enabled", True)):
                    continue
                await self._run_llm_match()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("transcript: llm_match_loop iteration failed")

    # =========================================================
    # Transcript group summariser — replaces the per-utterance
    # live strip. Every N seconds, take new chunks and emit one
    # 1-2 sentence observational line for the group.
    # =========================================================

    # Bumped to INFORMED_NUM_CTX — the prompt now carries the full
    # CHANNEL CONTEXT block (game / title / tags / viewer tier /
    # uptime) plus up to ~400 utterance lines plus a base64 image
    # payload. 8k truncated long groups; 32k gives headroom.
    GROUP_SUMMARY_NUM_CTX = INFORMED_NUM_CTX

    def _build_channel_context(self) -> str:
        """Delegate to TwitchStatus.format_for_llm with
        authoritative=True so the group-summary prompt gets the full
        Helix snapshot (game, title, tags, content labels, viewer
        tier, uptime) and explicit "do not guess from screenshot"
        framing. Returns "" when the Helix poll isn't connected."""
        if self.twitch_status is None:
            return ""
        ts = getattr(self.twitch_status, "status", None)
        if ts is None:
            return ""
        try:
            return ts.format_for_llm(authoritative=True)
        except AttributeError:
            return ""
    GROUP_SUMMARY_TRUNCATE_PER_CHUNK = 200

    GROUP_SUMMARY_SYSTEM = """You're labeling a window of streamer voice transcripts on a Twitch dashboard. An attached image shows what was on the streamer's OBS scene during this same window — use it as SILENT CONTEXT, not as content.

IMPORTANT — whisper output is imperfect. The transcripts come from real-time speech-to-text on streamer audio, so expect:
  - misheard words (homophones, proper nouns garbled, game/character names mangled)
  - missing or wrong punctuation
  - run-on phrases with no clear sentence boundaries
  - the streamer trailing off mid-thought, mumbling, or overlapping with game audio
Use the image and context to interpret what was *probably* said. If "purry" appears next to a Resident Evil scene, it's "parry." If "yogi" turns up while Yoshi is on screen, it's Yoshi. Don't quote literally — paraphrase with the most plausible reading.


Write a 2-4 sentence OBSERVATIONAL recap of WHAT THE STREAMER SAID and (briefly) HOW CHAT RESPONDED. The audio is primary. The image and chat block exist to help you ground the audio:
  - resolve pronouns / vague nouns ("the boss" → "the Malenia boss in Elden Ring"),
  - name the game / app / scene if it's relevant to what was said,
  - back up an ambiguous word the streamer used,
  - identify chat replies the streamer is reacting to (when a "CHAT DURING THIS WINDOW" block is present).

PEOPLE ON STREAM:
- Name people the streamer is talking to / talking about by their actual name when audible ("xQc tells Train to stop trolling …", not "the streamer tells someone to stop trolling …"). Whisper sometimes garbles names — match against the chat block when uncertain.

CHAT REACTION (optional, when worth a sentence):
- If the chat block shows a clear collective reaction worth one sentence (chat memes, agrees, pushes back, asks a follow-up question), include it as a brief trailing sentence: "Chat hypes the play / pushes back on his take / asks why he stopped."
- Skip the chat sentence when chat is just generic emote spam or unrelated background.

STREAMER NAME:
- The prompt may begin with a "STREAMER NAME: <name>" line. If present, refer to them by that name in your summary ("xQc explains …", "Aiko notes …") rather than the generic "the streamer". Reads more naturally on the dashboard.
- When NO "STREAMER NAME:" line is present, "the streamer" is the right fallback.

GAME / APP IDENTIFICATION (THIS IS THE #1 RULE — read it twice):
- The prompt may begin with a "KNOWN GAME: <name>" line. If present, that name comes from Twitch's live API and IS THE GAME. Use it verbatim.
- The screenshot will sometimes LOOK like a different game (similar art style, similar UI, the streamer is on a menu screen, the cam is overlapping gameplay). IGNORE THAT. The Twitch API knows what game is on the channel; the screenshot is a single frame of unknown context. If they disagree, the API wins. Always.
- Never write "judging by the graphics", "appears to be", "looks like Fall Guys / Valheim / etc.", or any phrasing that implies you're guessing the game from pixels.
- When NO "KNOWN GAME:" line is present, you may infer the game from the screenshot — but state it plainly ("playing Valheim"), not as a guess.

NEVER describe the image directly. Do NOT mention:
  - the image itself ("the screenshot shows", "in the image"),
  - the image format ("four-panel grid", "2x2 grid", "panel", "thumbnail"),
  - the cam ("webcam inset", "the streamer is visible in"),
  - the layout ("alongside a UI displaying", "next to the cam").

Bad example (your output keeps doing this — STOP):
  "The streamer is playing a co-op game (Valheim, judging by the graphics) and..."
  "In a four-panel grid of the game Parkitect, the streamer explores a park scene while discussing digital tickets."

Good example (image used silently to identify the game; summary about what was said):
  "The streamer talks about a new feature in Parkitect that lets guests buy tickets in advance."

HARD RULES:
- KNOWN GAME line, if present, OVERRIDES the screenshot. Period.
- STREAMER NAME line, if present, replaces "the streamer" in your output.
- Audio is primary. Lead with what was said. Use the image and chat block as supporting context.
- 2-4 sentences. Long enough to capture multi-topic windows; short enough to scan at a glance. A 1-sentence summary is fine when the window is single-topic.
- Name people the streamer is talking to / about, when audible.
- Brief chat reaction (one sentence) is welcome when chat is engaged with the streamer's bit. Skip it when chat is just emote spam.
- No advice ("you should…").
- Don't invent products, events, plot points, or characters that aren't in the audio, image, or chat.
- Skip filler: empty `summary` is fine when the utterances are one-word reactions or noise.

Reply with `summary` = the line(s) (or empty string).
"""

    async def transcript_embed_backfill_loop(self) -> None:
        """Background task: embed historical transcript chunks that
        don't have a vec_transcripts row.

        The live ingest path writes embeddings inline, so this only
        catches rows from BEFORE vec_transcripts was wired up plus any
        row where the inline embed call failed. Runs slowly (small
        batch every 30 s) since we're racing nothing — we just want
        the search index to fill in over time. Sleeps when there's
        nothing to do.

        Persistent embeddings are RETRIEVAL-only (the streamer's
        /search page + the chatter / thread modal evidence panels).
        Live LLM calls remain time-windowed and do NOT pull from
        this index — guard against historical-context pollution by
        not introducing one."""
        await asyncio.sleep(30)  # let the dashboard settle on boot
        while True:
            try:
                pending = await asyncio.to_thread(
                    self.repo.transcripts_missing_embedding, 25,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("transcript-embed backfill: query failed")
                pending = []

            if not pending:
                # Index is caught up — sleep longer until new chunks
                # arrive (the live ingest path will keep things current
                # in the meantime).
                await asyncio.sleep(300)
                continue

            for chunk in pending:
                try:
                    vec = await self.llm.embed(chunk.text)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "transcript-embed backfill: embed failed for "
                        "chunk %d", chunk.id,
                    )
                    continue
                try:
                    await asyncio.to_thread(
                        self.repo.upsert_transcript_embedding,
                        chunk.id, vec,
                    )
                except Exception:
                    logger.exception(
                        "transcript-embed backfill: upsert failed for "
                        "chunk %d", chunk.id,
                    )
            try:
                indexed, total = await asyncio.to_thread(
                    self.repo.transcripts_embedding_coverage,
                )
                logger.info(
                    "transcript-embed backfill: %d/%d indexed (this "
                    "batch wrote up to %d)",
                    indexed, total, len(pending),
                )
            except Exception:
                pass
            # Pace ourselves so we don't hog the embed endpoint.
            await asyncio.sleep(30)

    async def chat_lag_calibration_loop(self) -> None:
        """Background task: periodically auto-tune `chat_lag_seconds`.

        Runs every ~10 min with a 15-min lookback. Silently updates
        the setting when the cross-correlation is confident (best
        score has a clear margin over the runner-up AND we sampled
        enough activity to trust it). Otherwise skips the iteration
        — better to keep the manual / previously-applied value than
        chase noise.

        The streamer sees the auto-tuned value + timestamp in the
        calibrator panel on /settings → Whisper. Set
        `chat_lag_auto_tune_interval_seconds` to 0 to disable."""
        from .latency import calibrate_chat_lag

        await asyncio.sleep(120)  # let the dashboard settle + accumulate data
        while True:
            # Read fresh — the auto-tune toggle in the calibrator
            # writes app_settings, and we want a flip from on→off (or
            # interval change) to take effect within one cycle, not
            # require a dashboard restart.
            interval_raw = await asyncio.to_thread(
                self.repo.get_app_setting,
                "chat_lag_auto_tune_interval_seconds",
            )
            try:
                interval = int(
                    interval_raw if interval_raw is not None
                    else getattr(
                        self.settings,
                        "chat_lag_auto_tune_interval_seconds", 600,
                    )
                )
            except (TypeError, ValueError):
                interval = 600
            if interval <= 0:
                # Disabled — sleep a fixed window then re-check, so
                # toggling back on doesn't need a restart either.
                await asyncio.sleep(60)
                continue
            try:
                result = await asyncio.to_thread(
                    calibrate_chat_lag, self.repo, lookback_minutes=15,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("chat-lag auto-calibration iteration failed")
                result = None

            if result and result.get("ok") and result.get("best_offset") is not None:
                # Confidence gate: the winning offset must beat the
                # runner-up by at least 10% AND we must have non-trivial
                # sample sizes. Loose enough to converge once chat picks
                # up; tight enough that a flat curve can't sneak through.
                offsets = result.get("offsets") or []
                samples = result.get("samples") or {}
                scores = sorted(
                    (o.get("score", 0.0) for o in offsets), reverse=True,
                )
                best_score = scores[0] if scores else 0.0
                second_score = scores[1] if len(scores) > 1 else 0.0
                samples_ok = (
                    samples.get("chunks", 0) >= 15
                    and samples.get("messages", 0) >= 50
                )
                margin_ok = (
                    second_score == 0.0
                    or best_score >= second_score * 1.1
                )
                if samples_ok and margin_ok:
                    new_value = int(result["best_offset"])
                    try:
                        current = int(self.repo.get_app_setting(
                            "chat_lag_seconds",
                        ) or getattr(self.settings, "chat_lag_seconds", 6))
                    except (TypeError, ValueError):
                        current = getattr(self.settings, "chat_lag_seconds", 6)
                    if new_value != current:
                        try:
                            from datetime import datetime as _dt
                            ts_iso = _dt.utcnow().isoformat(timespec="seconds")
                            self.repo.set_app_setting(
                                "chat_lag_seconds", str(new_value),
                            )
                            self.repo.set_app_setting(
                                "chat_lag_auto_tuned_at", ts_iso,
                            )
                            self.repo.set_app_setting(
                                "chat_lag_auto_tuned_value", str(new_value),
                            )
                            logger.info(
                                "chat-lag auto-tune: %ds → %ds "
                                "(samples chunks=%d msgs=%d, best=%.2f vs "
                                "second=%.2f)",
                                current, new_value,
                                samples.get("chunks", 0),
                                samples.get("messages", 0),
                                best_score, second_score,
                            )
                        except Exception:
                            logger.exception("chat-lag auto-tune: persist failed")
                    else:
                        logger.debug(
                            "chat-lag auto-tune: already at %ds (no change)",
                            current,
                        )
                else:
                    logger.debug(
                        "chat-lag auto-tune: skipped — samples_ok=%s "
                        "margin_ok=%s (best=%.2f second=%.2f chunks=%d msgs=%d)",
                        samples_ok, margin_ok, best_score, second_score,
                        samples.get("chunks", 0), samples.get("messages", 0),
                    )

            await asyncio.sleep(interval)

    async def transcript_group_loop(self) -> None:
        """Background task: every `whisper_group_interval_seconds`,
        pull new chunks since the last group's watermark and write one
        LLM-summarised group row. No-op when disabled or whisper off."""
        await asyncio.sleep(20)
        while True:
            try:
                interval = max(15, int(getattr(
                    self.settings, "whisper_group_interval_seconds", 60,
                )))
                if interval <= 0:
                    return
                await asyncio.sleep(interval)
                if not self.enabled:
                    continue
                await self._run_group_summary()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("transcript: group_loop iteration failed")

    async def build_group_summary_prompt(
        self, chunks: list, *, include_image: bool = True,
    ) -> dict:
        """Assemble the exact prompt + screenshot grid + system prompt
        the group-summary LLM call uses, without firing the LLM. Used
        by `_run_group_summary` (the live caller) and by the debug
        route `/debug/transcript-prompt` (so the streamer can inspect
        what the model is actually being told).

        Returns a dict so the debug route can render structured fields:
            {
              "system_prompt": str,
              "user_prompt": str,
              "screenshot_count": int,
              "screenshot_grid_b64": str | None,  # only when include_image
              "channel_context": str,             # the rendered block
              "known_game": str,                  # "" when Helix offline
              "model": str,
              "num_ctx": int,
              "think": bool,
            }
        """
        # Per-utterance lines — clip each utterance defensively.
        lines = [
            f"  [{c.ts[11:16] if len(c.ts) >= 16 else c.ts}] "
            f"{c.text[:self.GROUP_SUMMARY_TRUNCATE_PER_CHUNK]}"
            for c in chunks
        ]

        # Screenshots in the chunk window — up to `screenshot_grid_max`
        # stitched into a 2x2 grid. Skipped when include_image=False
        # (the debug route can ask for prompt-only).
        grid_b64: str | None = None
        screenshot_count = 0
        if include_image and chunks:
            try:
                shots = await asyncio.to_thread(
                    self.repo.screenshots_in_range,
                    chunks[0].ts, chunks[-1].ts,
                    max_count=int(getattr(self.settings, "screenshot_grid_max", 4)),
                )
            except Exception:
                logger.exception("transcript: screenshots_in_range failed")
                shots = []
            if shots:
                from pathlib import Path
                data_dir = Path(self.settings.db_path).parent
                abs_paths = [str(data_dir / s.path) for s in shots]
                try:
                    grid_bytes = await asyncio.to_thread(_stitch_grid, abs_paths)
                except Exception:
                    logger.exception("transcript: stitch_grid failed")
                    grid_bytes = None
                if grid_bytes:
                    import base64 as _b64
                    grid_b64 = _b64.b64encode(grid_bytes).decode("ascii")
                    screenshot_count = len(shots)

        # Known game from Helix — pinned into the image_note so the
        # constraint sits right next to the image reference.
        known_game = ""
        try:
            ts = getattr(self.twitch_status, "status", None)
            if ts is not None and getattr(ts, "is_live", False):
                known_game = (getattr(ts, "game_name", None) or "").strip()
        except Exception:
            known_game = ""
        if screenshot_count:
            game_pin = (
                f" The KNOWN GAME is {known_game!r} (from the Twitch "
                f"API at the top of the prompt). DO NOT rename it based "
                f"on what the image looks like — even if the screenshot "
                f"looks like a different game, the API is authoritative."
            ) if known_game else ""
            image_note = (
                "\n\nAn image is attached showing what was on screen "
                "during this window. Use it ONLY as silent context: "
                "resolve vague nouns the streamer used, verify an "
                "ambiguous word. Do NOT describe the image, its layout, "
                "or mention that it's attached. The summary should "
                "read as if you only heard the audio, with the image "
                "silently helping you understand it." + game_pin
            )
        else:
            image_note = ""

        channel_context = self._build_channel_context()

        # Pull chat that overlaps the chunk window so the LLM has
        # context for what the streamer is reacting to. Soft window:
        # `recent_messages(within_minutes=N)` looks back from "now",
        # so we size N to comfortably cover [chunks[0].ts, chunks[-1].ts]
        # plus a few minutes of pre-roll (chat triggers a streamer
        # reaction that lasts a while). Capped to 60 messages so a
        # very busy chat doesn't blow the context budget.
        # Twitch broadcast latency offset. Streamer's mic captures into
        # OBS in real time, but viewers don't hear it until ~3-15s
        # later (Low Latency vs Standard mode + CDN + player buffer +
        # viewer reaction). Chat is reacting to what they HEARD, so
        # the chat window is offset BACKWARDS from the audio window
        # by `chat_lag_seconds`. Default 6s ≈ Low Latency Twitch.
        # 0 disables the offset (correct for the test setup where
        # chatterbot ingests the same playback audio chat is reacting
        # to). Calibrate via /settings → Whisper.
        #
        # Read fresh from app_settings rather than `self.settings` so
        # that "Apply Ns" in the calibrator + the auto-tune loop both
        # take effect on the NEXT group summary, not after a restart.
        chat_lag_raw = await asyncio.to_thread(
            self.repo.get_app_setting, "chat_lag_seconds",
        )
        try:
            chat_lag = max(0, int(
                chat_lag_raw if chat_lag_raw is not None
                else getattr(self.settings, "chat_lag_seconds", 6)
            ))
        except (TypeError, ValueError):
            chat_lag = 6

        chat_block = ""
        try:
            from datetime import datetime as _dt, timedelta as _td
            now = _dt.utcnow()
            try:
                first_ts = _dt.fromisoformat(
                    chunks[0].ts.replace("Z", "+00:00")
                ).replace(tzinfo=None)
                last_ts = _dt.fromisoformat(
                    chunks[-1].ts.replace("Z", "+00:00")
                ).replace(tzinfo=None)
                window_min = max(3, int((now - first_ts).total_seconds() / 60) + 2)
            except Exception:
                first_ts = None
                last_ts = None
                window_min = 5
            window_min = min(window_min, 12)
            raw_chat_msgs = await asyncio.to_thread(
                self.repo.recent_messages,
                limit=120, within_minutes=window_min,
            )
            # Filter to the lag-adjusted window: chat that arrived
            # between (chunks[0].ts + chat_lag - 30s) and
            # (chunks[-1].ts + chat_lag + 30s). The +chat_lag shifts
            # the chat window FORWARD relative to audio because chat
            # arrives AFTER the audio it's reacting to. The ±30s
            # padding catches messages straddling the boundary.
            chat_msgs: list = []
            if first_ts is not None and last_ts is not None and chat_lag > 0:
                lag_low = first_ts + _td(seconds=chat_lag - 30)
                lag_high = last_ts + _td(seconds=chat_lag + 30)
                for m in raw_chat_msgs:
                    try:
                        m_ts = _dt.fromisoformat(
                            m.ts.replace("Z", "+00:00")
                        ).replace(tzinfo=None)
                    except Exception:
                        continue
                    if lag_low <= m_ts <= lag_high:
                        chat_msgs.append(m)
                # Cap at 60 to keep prompt size bounded.
                chat_msgs = chat_msgs[-60:]
            else:
                chat_msgs = raw_chat_msgs[-60:]
        except Exception:
            logger.exception("transcript: chat fetch for group summary failed")
            chat_msgs = []
        if chat_msgs:
            chat_lines = []
            for m in chat_msgs:
                ts_short = m.ts[11:16] if len(m.ts) >= 16 else m.ts
                # Light reply-context — when the chatter is replying
                # to someone, give the LLM the parent so it can tell
                # "alice asks bob about X" from "alice asks streamer
                # about X".
                reply_prefix = ""
                if m.reply_parent_login and m.reply_parent_body:
                    reply_prefix = (
                        f"@{m.reply_parent_login}({m.reply_parent_body[:40]}) "
                    )
                chat_lines.append(
                    f"  [{ts_short}] {m.name}: {reply_prefix}"
                    f"{m.content[:200]}"
                )
            chat_block = (
                f"\n\nCHAT DURING THIS WINDOW ({len(chat_lines)} "
                f"messages, oldest first — use as SECONDARY context "
                "to ground reactions; the streamer's audio is still "
                "primary):\n"
                + "\n".join(chat_lines)
            )

        user_prompt = (
            channel_context
            + f"STREAMER UTTERANCES ({len(chunks)} lines, "
            f"oldest first):\n"
            + "\n".join(lines)
            + chat_block
            + image_note
            + "\n\nReturn a 2-4 sentence observational `summary`, or "
            "an empty string if there's nothing coherent to summarise."
        )

        # IDs of the chat messages that landed in the prompt — caller
        # persists this with the resulting transcript_group so the
        # detail modal can re-display the exact same chat the LLM
        # saw, not a re-queried approximation.
        chat_message_ids = [int(m.id) for m in chat_msgs if getattr(m, "id", None)]

        from .llm.prompts import resolve_prompt
        return {
            "system_prompt": resolve_prompt(
                "transcript.group_summary", self.repo,
            ),
            "user_prompt": user_prompt,
            "screenshot_count": screenshot_count,
            "screenshot_grid_b64": grid_b64,
            "channel_context": channel_context,
            "known_game": known_game,
            "chat_message_ids": chat_message_ids,
            "model": getattr(self.llm, "model", ""),
            "num_ctx": self.GROUP_SUMMARY_NUM_CTX,
            "think": True,
        }

    async def _run_group_summary(self) -> int:
        """One pass — pull all chunks > last group's last_chunk_id,
        summarise, persist. Returns number of groups created (0 or 1)."""
        watermark = await asyncio.to_thread(
            self.repo.latest_transcript_group_last_chunk_id
        )
        chunks = await asyncio.to_thread(
            self.repo.list_transcripts_after_id, watermark, limit=400,
        )
        min_chunks = max(1, int(getattr(
            self.settings, "whisper_group_min_chunks", 2,
        )))
        if len(chunks) < min_chunks:
            return 0

        bundle = await self.build_group_summary_prompt(chunks)
        prompt = bundle["user_prompt"]
        grid_b64 = bundle["screenshot_grid_b64"]
        screenshot_count = bundle["screenshot_count"]
        known_game = bundle["known_game"]
        channel_context = bundle["channel_context"]
        chat_message_ids = bundle.get("chat_message_ids") or []

        # Diagnostic — log whether the LLM is going in with the
        # KNOWN GAME pin or not, plus screenshot count. When the LLM
        # mis-identifies a game, this is the first place to look:
        #   - "no helix" → twitch_status not wired or offline
        #   - "no game"  → Helix is up but game_name missing
        #   - "game=X"   → LLM was told and ignored it (prompt issue)
        if logger.isEnabledFor(logging.INFO):
            if not channel_context:
                helix_state = "no helix"
            elif known_game:
                helix_state = f"game={known_game!r}"
            else:
                helix_state = "no game"
            logger.info(
                "transcript group summary: %d chunks, %d screenshot(s), "
                "%s, prompt=%d chars",
                len(chunks), screenshot_count, helix_state, len(prompt),
            )

        try:
            # think=True — group summary runs after a window of audio
            # has elapsed (not realtime) and the streamer reads it on
            # the dashboard, so accuracy beats latency.
            from .llm.schemas import TranscriptGroupSummaryResponse
            from .llm.prompts import resolve_prompt
            response = await self.llm.generate_structured(
                prompt=prompt,
                system_prompt=resolve_prompt(
                    "transcript.group_summary", self.repo,
                ),
                response_model=TranscriptGroupSummaryResponse,
                num_ctx=self.GROUP_SUMMARY_NUM_CTX,
                images=[grid_b64] if grid_b64 else None,
                think=True,
                call_site="transcript.group_summary",
            )
        except Exception:
            logger.exception("transcript: group summary call failed")
            return 0

        summary = (response.summary or "").strip()
        if not summary:
            # The LLM said this window had nothing summarisable. Still
            # advance the watermark by inserting an empty group? No —
            # that pollutes the strip with empty rows. Just don't emit
            # anything; the watermark stays where it was, and the next
            # pass will retry with the same chunks plus more — by then
            # there may be enough signal to summarise.
            logger.info(
                "transcript: group skipped (LLM returned empty) — %d chunks",
                len(chunks),
            )
            # If chunks are growing without bound (LLM consistently empty),
            # eventually we need to advance to avoid retrying forever.
            # Force advance after 5x the threshold.
            if len(chunks) >= 5 * min_chunks:
                logger.info(
                    "transcript: force-advancing watermark — %d chunks "
                    "without summary", len(chunks),
                )
                await asyncio.to_thread(
                    self.repo.add_transcript_group,
                    start_ts=chunks[0].ts, end_ts=chunks[-1].ts,
                    first_chunk_id=chunks[0].id, last_chunk_id=chunks[-1].id,
                    # Use the repo's sentinel so list_transcript_groups
                    # filters this row out of the live strip by default.
                    # The row exists only to advance the watermark.
                    summary=self.repo.PLACEHOLDER_GROUP_SUMMARY,
                )
                return 1
            return 0

        await asyncio.to_thread(
            self.repo.add_transcript_group,
            start_ts=chunks[0].ts, end_ts=chunks[-1].ts,
            first_chunk_id=chunks[0].id, last_chunk_id=chunks[-1].id,
            summary=summary,
            context_message_ids=chat_message_ids,
        )
        logger.info(
            "transcript: group written — %d chunks (id %d → %d): %r",
            len(chunks), chunks[0].id, chunks[-1].id, summary[:80],
        )
        return 1

    # =========================================================
    # OBS screenshot capture — runs every N seconds while whisper +
    # OBS are both enabled. Stored as JPEGs on disk so the per-group
    # render can pick up to N evenly spaced frames from the window
    # and stitch them into a grid for the multimodal LLM call.
    # =========================================================

    # Prune cadence is wall-clock based (60 min) so it fires even when
    # the loop is idle (whisper off, OBS down, etc.). Old files from
    # prior streams shouldn't sit around just because nothing's being
    # captured right now.
    SCREENSHOT_PRUNE_INTERVAL_SECONDS = 3600

    def _screenshot_dir(self):
        from pathlib import Path
        return Path(self.settings.db_path).parent / "transcript_screenshots"

    async def screenshot_loop(self) -> None:
        """Background task: every `screenshot_interval_seconds` capture
        the current OBS program scene and write to disk + DB. Periodic
        pruning trims rows + files older than `screenshot_max_age_hours`
        to keep disk usage bounded.

        Gating: whisper enabled + OBS enabled + OBS reachable. We do
        NOT gate on streaming/recording state — the streamer might
        capture audio (and want paired screenshots) without OBS's
        output pipeline being live (testing, local recording with the
        recording output off, second-monitor preview, etc.).
        """
        await asyncio.sleep(20)  # wait for boot to settle
        if self.obs is None:
            logger.info("transcript: screenshot_loop disabled — no OBS handle")
            return
        passes = 0
        skipped_disabled = 0
        skipped_disconnected = 0
        last_prune_at = 0.0
        from pathlib import Path
        shot_dir = self._screenshot_dir()
        shot_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "transcript: screenshot_loop starting (interval=%ds, dir=%s)",
            int(getattr(self.settings, "screenshot_interval_seconds", 15)),
            shot_dir,
        )
        # Startup prune — handles old files from prior runs that aged
        # out while the dashboard was down, plus any orphaned files
        # whose DB rows got cleared between sessions.
        try:
            await self._prune_screenshots()
            last_prune_at = time.time()
        except Exception:
            logger.exception("transcript: startup prune failed")
        while True:
            try:
                interval = max(5, int(getattr(
                    self.settings, "screenshot_interval_seconds", 15,
                )))
                if interval <= 0:
                    return
                await asyncio.sleep(interval)
                # Wall-clock pruning fires regardless of whether we're
                # capturing, so old files from prior streams don't sit
                # around while the streamer is offline.
                if time.time() - last_prune_at >= self.SCREENSHOT_PRUNE_INTERVAL_SECONDS:
                    try:
                        await self._prune_screenshots()
                    except Exception:
                        logger.exception("transcript: scheduled prune failed")
                    last_prune_at = time.time()
                if not self.enabled:
                    skipped_disabled += 1
                    if skipped_disabled in (1, 60):
                        logger.info(
                            "transcript: screenshot_loop idle — whisper disabled "
                            "(skipped %d cycles)", skipped_disabled,
                        )
                    continue
                if not getattr(self.settings, "obs_enabled", False):
                    skipped_disabled += 1
                    if skipped_disabled in (1, 60):
                        logger.info(
                            "transcript: screenshot_loop idle — obs disabled "
                            "(skipped %d cycles)", skipped_disabled,
                        )
                    continue
                obs_status = getattr(self.obs, "status", None)
                if obs_status is not None and not obs_status.connected:
                    skipped_disconnected += 1
                    if skipped_disconnected in (1, 60):
                        logger.info(
                            "transcript: screenshot_loop waiting — OBS not "
                            "connected (skipped %d cycles)", skipped_disconnected,
                        )
                    continue
                # Reset the skip counters once we're actively capturing.
                skipped_disabled = skipped_disconnected = 0
                shot = await asyncio.to_thread(
                    self.obs.take_screenshot_sync,
                    image_format="jpg",
                    width=int(getattr(self.settings, "screenshot_width", 480)),
                    quality=int(getattr(self.settings, "screenshot_jpeg_quality", 85)),
                )
                if shot is None:
                    if passes == 0:
                        logger.warning(
                            "transcript: take_screenshot_sync returned None — "
                            "OBS up but screenshot call failed (check OBS "
                            "WebSocket logs)"
                        )
                    continue
                jpeg_bytes, scene_name = shot
                # Transcode JPEG → WebP for persisted storage. WebP
                # is meaningfully smaller at the same visual quality,
                # which compounds with content-hash dedup to keep
                # disk growth bounded under the keep-forever policy.
                webp_quality = int(getattr(
                    self.settings, "screenshot_webp_quality", 65,
                ))
                webp_bytes = await asyncio.to_thread(
                    _encode_webp, jpeg_bytes, webp_quality,
                )
                if webp_bytes is None:
                    # Pillow missing or encode failed — skip persisting
                    # this interval. Modal grid will still work using
                    # the screenshots already on disk.
                    continue
                from datetime import datetime as _dt2, timezone as _tz2
                ts = _dt2.now(_tz2.utc).isoformat(timespec="seconds")
                # Content-addressed layout: <sha[:2]>/<sha[2:4]>/<sha>.webp.
                # Identical bytes (a paused stream on the same scene)
                # collapse to one on-disk file; multiple DB rows can
                # reference it. The orphan sweep walks recursively so
                # the dedup-shared file isn't accidentally pruned when
                # one row's row-level lifecycle ends.
                sub_path = _content_hashed_relpath(webp_bytes, ext=".webp")
                fpath = shot_dir / sub_path
                dedup_hit = False
                try:
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    if fpath.exists():
                        dedup_hit = True
                    else:
                        fpath.write_bytes(webp_bytes)
                except OSError:
                    logger.exception("transcript: write screenshot failed")
                    continue
                rel_path = f"transcript_screenshots/{sub_path}"
                await asyncio.to_thread(
                    self.repo.add_transcript_screenshot,
                    ts=ts, path=rel_path, scene_name=scene_name,
                )
                passes += 1
                if passes in (1, 5, 30, 100) or passes % 240 == 0:
                    logger.info(
                        "transcript: screenshot #%d saved (scene=%r, "
                        "%d KB jpeg → %d KB webp, dedup=%s)",
                        passes, scene_name,
                        len(jpeg_bytes) // 1024, len(webp_bytes) // 1024,
                        "hit" if dedup_hit else "miss",
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("transcript: screenshot_loop iteration failed")

    async def _prune_screenshots(self) -> None:
        """Optionally delete DB rows + image files older than the
        configured TTL, then sweep the screenshot dir for orphan
        files no longer referenced in the DB.

        `screenshot_max_age_hours` <= 0 means "keep forever" — only
        the orphan sweep runs. With content-hash dedup + WebP, disk
        growth stays bounded under that policy; the streamer can
        opt back in to age-based deletion by raising the setting."""
        try:
            max_age_h = int(getattr(
                self.settings, "screenshot_max_age_hours", 0,
            ))
        except (TypeError, ValueError):
            max_age_h = 0
        from pathlib import Path
        data_dir = Path(self.settings.db_path).parent

        aged_deleted = 0
        if max_age_h > 0:
            from datetime import datetime as _dt, timezone as _tz, timedelta as _td
            cutoff = (_dt.now(_tz.utc) - _td(hours=max_age_h)).isoformat(timespec="seconds")
            # Age-based delete: drop DB rows and unlink files. With
            # dedup, multiple rows can share a file — only unlink
            # when no remaining row still references this path.
            try:
                paths = await asyncio.to_thread(
                    self.repo.delete_screenshots_older_than, cutoff,
                )
            except Exception:
                logger.exception("transcript: screenshot prune DB step failed")
                return
            try:
                still_ref = set(await asyncio.to_thread(
                    self._list_referenced_screenshot_paths,
                ))
            except Exception:
                still_ref = set()
            for rel in paths:
                if rel in still_ref:
                    continue
                try:
                    p = data_dir / rel
                    if p.exists():
                        p.unlink()
                        aged_deleted += 1
                except OSError:
                    pass

        # Orphan sweep — find files in the screenshot dir not
        # referenced in the DB. Walks recursively so the new
        # nested `<sha[:2]>/<sha[2:4]>/<sha>.webp` layout is
        # covered as well as legacy flat-dir files. Compares full
        # relative paths (not basenames) so two unrelated files
        # with the same name in different dedup buckets can't
        # accidentally save each other.
        shot_dir = self._screenshot_dir()
        orphan_deleted = 0
        if shot_dir.exists():
            try:
                referenced_paths = await asyncio.to_thread(
                    self._list_referenced_screenshot_paths,
                )
                referenced: set[str] = set()
                for rel in referenced_paths:
                    # DB paths are stored as
                    # `transcript_screenshots/<...>` so strip the
                    # prefix to get the path relative to shot_dir.
                    if rel.startswith("transcript_screenshots/"):
                        referenced.add(rel[len("transcript_screenshots/"):])
                    else:
                        referenced.add(rel)
                for f in shot_dir.rglob("*"):
                    if not f.is_file():
                        continue
                    rel = f.relative_to(shot_dir).as_posix()
                    if rel in referenced:
                        continue
                    try:
                        f.unlink()
                        orphan_deleted += 1
                    except OSError:
                        pass
            except Exception:
                logger.exception("transcript: orphan screenshot sweep failed")

        if aged_deleted or orphan_deleted:
            logger.info(
                "transcript: pruned %d aged + %d orphan screenshots "
                "(TTL %s)",
                aged_deleted, orphan_deleted,
                f"{max_age_h}h" if max_age_h > 0 else "off",
            )

    def _list_referenced_screenshot_paths(self) -> list[str]:
        """Synchronous helper for the orphan sweep — returns every path
        currently referenced in the transcript_screenshots table."""
        with self.repo._cursor() as cur:
            cur.execute("SELECT path FROM transcript_screenshots")
            return [r["path"] for r in cur.fetchall()]

    AUTO_CONFIRM_INTERVAL = 15

    async def auto_confirm_loop(self) -> None:
        """Promote auto_pending insight states to 'addressed' after
        `whisper_auto_confirm_seconds` without explicit confirm/reject.
        Runs forever; no-op when whisper is disabled."""
        while True:
            try:
                await asyncio.sleep(self.AUTO_CONFIRM_INTERVAL)
                if not self.enabled:
                    continue
                window = max(
                    self.AUTO_CONFIRM_INTERVAL,
                    int(getattr(self.settings, "whisper_auto_confirm_seconds", 300)),
                )
                pendings = await asyncio.to_thread(
                    self.repo.list_pending_auto_addresses,
                    older_than_seconds=window,
                )
                for p in pendings:
                    await asyncio.to_thread(
                        self.repo.set_insight_state,
                        p.kind, p.item_key, "addressed",
                        note=p.note,
                    )
                    logger.info(
                        "transcript: auto-confirm timeout — %s/%s → addressed",
                        p.kind, p.item_key,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("transcript: auto_confirm_loop iteration failed")

    def status(self) -> dict:
        """Snapshot for the Settings → Diagnostics health panel."""
        return {
            "enabled": self.enabled,
            "model_loaded": self._model is not None,
            "model_error": self._model_load_error,
            "model": self.settings.whisper_model,
            "buffered_samples": self._buffer.total_samples,
            "card_embeds": len(self._card_embeds),
            "pending_count": self._pending_count,
        }
