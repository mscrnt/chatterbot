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
from .llm.ollama_client import OllamaClient
from .repo import ChatterRepo

logger = logging.getLogger(__name__)


SAMPLE_RATE = 16000  # whisper's required input rate
SAMPLE_BYTES = 4     # float32


@dataclass
class _IngestBuffer:
    """In-memory audio buffer. Float32 mono PCM at 16 kHz."""

    samples: list[np.ndarray] = field(default_factory=list)
    total_samples: int = 0
    last_flush: float = field(default_factory=time.time)

    def append(self, pcm: np.ndarray) -> None:
        self.samples.append(pcm)
        self.total_samples += pcm.shape[0]

    def flush(self) -> np.ndarray:
        if not self.samples:
            return np.zeros(0, dtype=np.float32)
        out = np.concatenate(self.samples).astype(np.float32, copy=False)
        self.samples.clear()
        self.total_samples = 0
        self.last_flush = time.time()
        return out


class TranscriptService:
    """Owns the whisper model + ingest buffer + match loop. Thread-safe
    enough for the dashboard's single-process usage."""

    def __init__(
        self, repo: ChatterRepo, llm: OllamaClient, settings: Settings,
        *, obs=None,
    ):
        self.repo = repo
        self.llm = llm
        self.settings = settings
        self.obs = obs  # OBSStatusService — used for screenshot capture
        self._buffer = _IngestBuffer()
        self._lock = asyncio.Lock()
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

    async def ingest_chunk(self, payload: bytes, sample_rate: int) -> None:
        """Append a raw PCM chunk to the buffer. The OBS script sends
        16 kHz mono float32; if a different rate is supplied, we resample."""
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
            self._buffer.append(x)
            should_flush = (
                self._buffer.total_samples
                >= int(SAMPLE_RATE * self.settings.whisper_buffer_seconds)
            )

        if should_flush:
            asyncio.create_task(self._transcribe_and_match())

    async def _transcribe_and_match(self) -> None:
        """Pop the buffer, transcribe the audio, embed each segment,
        match against open insight cards, persist. Runs as a fire-and-
        forget task so the ingest endpoint stays snappy."""
        async with self._lock:
            audio = self._buffer.flush()
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

        def _run() -> list[tuple[str, float, float]]:
            try:
                segments, _info = model.transcribe(
                    audio,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": min_silence_ms},
                    beam_size=1,
                    condition_on_previous_text=False,
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

    GROUP_SUMMARY_NUM_CTX = 8192
    GROUP_SUMMARY_TRUNCATE_PER_CHUNK = 200

    GROUP_SUMMARY_SYSTEM = """You're labeling a window of streamer voice transcripts on a Twitch dashboard, with an attached screenshot grid showing what was on the streamer's OBS scene during that same window.

Write ONE 1-2 sentence OBSERVATIONAL summary that combines BOTH sources:
  - what the streamer SAID (the utterances), and
  - what was VISIBLE on screen (the screenshot grid — gameplay HUD, cam shot, on-screen text, scene name, characters, or whatever the scene contains).

The screenshots are ground truth — you may describe what's in them as confidently as the utterances. Together they're complementary; "the streamer reacts to a boss fight in [game]" is more useful than just "the streamer reacts to something."

HARD RULES:
- Pure description. Don't tell the streamer anything ("you should…", advice).
- Stay grounded — don't invent products, events, plot points, or characters that aren't visible/audible. If the gameplay shows a generic menu or cam, just say so.
- Skip filler: if both the utterances AND the image are uninformative (one-word reactions, blank scene, etc.), return an empty `summary` string.

Reply with `summary` = the line (or empty string).
"""

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

        # Build the prompt — clip each utterance defensively.
        lines = [
            f"  [{c.ts[11:16] if len(c.ts) >= 16 else c.ts}] "
            f"{c.text[:self.GROUP_SUMMARY_TRUNCATE_PER_CHUNK]}"
            for c in chunks
        ]

        # Pull screenshots in this window's time range and stitch up to
        # `screenshot_grid_max` into a 2x2 grid. The multimodal LLM gets
        # this image alongside the transcript text — visual context for
        # the audio. No screenshots? Pure text call as before.
        grid_b64 = None
        screenshot_count = 0
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

        image_note = (
            f"\n\nThe attached image is a {screenshot_count}-cell grid of "
            "OBS scene screenshots from this same window, oldest top-left, "
            "newest bottom-right. Treat what's in the image as ground truth: "
            "describe what's visible (game / scene / characters / on-screen "
            "text / cam) alongside what the streamer said."
        ) if screenshot_count else ""

        prompt = (
            f"STREAMER UTTERANCES ({len(chunks)} lines, "
            f"oldest first):\n"
            + "\n".join(lines)
            + image_note
            + "\n\nReturn ONE observational `summary` line, or empty "
            "string if there's nothing coherent to summarise."
        )

        try:
            from .llm.schemas import TranscriptGroupSummaryResponse
            response = await self.llm.generate_structured(
                prompt=prompt,
                system_prompt=self.GROUP_SUMMARY_SYSTEM,
                response_model=TranscriptGroupSummaryResponse,
                num_ctx=self.GROUP_SUMMARY_NUM_CTX,
                images=[grid_b64] if grid_b64 else None,
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
                    quality=int(getattr(self.settings, "screenshot_jpeg_quality", 60)),
                )
                if shot is None:
                    if passes == 0:
                        logger.warning(
                            "transcript: take_screenshot_sync returned None — "
                            "OBS up but screenshot call failed (check OBS "
                            "WebSocket logs)"
                        )
                    continue
                image_bytes, scene_name = shot
                from datetime import datetime as _dt2, timezone as _tz2
                ts = _dt2.now(_tz2.utc).isoformat(timespec="seconds")
                fname = ts.replace(":", "-").replace("+", "_") + ".jpg"
                fpath = shot_dir / fname
                try:
                    fpath.write_bytes(image_bytes)
                except OSError:
                    logger.exception("transcript: write screenshot failed")
                    continue
                rel_path = f"transcript_screenshots/{fname}"
                await asyncio.to_thread(
                    self.repo.add_transcript_screenshot,
                    ts=ts, path=rel_path, scene_name=scene_name,
                )
                passes += 1
                if passes in (1, 5, 30, 100) or passes % 240 == 0:
                    logger.info(
                        "transcript: screenshot #%d saved (scene=%r, %d KB)",
                        passes, scene_name, len(image_bytes) // 1024,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("transcript: screenshot_loop iteration failed")

    async def _prune_screenshots(self) -> None:
        """Delete DB rows + JPEG files older than the configured TTL,
        then sweep the screenshot dir for any files no longer
        referenced in the DB (orphans from manual DB clears, crashes
        between insert and write, container migrations, etc.)."""
        max_age_h = max(1, int(getattr(
            self.settings, "screenshot_max_age_hours", 24,
        )))
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        cutoff = (_dt.now(_tz.utc) - _td(hours=max_age_h)).isoformat(timespec="seconds")

        # Step 1 — age-based delete: drops DB rows and returns the
        # paths so we can unlink the files.
        try:
            paths = await asyncio.to_thread(
                self.repo.delete_screenshots_older_than, cutoff,
            )
        except Exception:
            logger.exception("transcript: screenshot prune DB step failed")
            return
        from pathlib import Path
        data_dir = Path(self.settings.db_path).parent
        aged_deleted = 0
        for rel in paths:
            try:
                p = data_dir / rel
                if p.exists():
                    p.unlink()
                    aged_deleted += 1
            except OSError:
                pass

        # Step 2 — orphan sweep: find files in the screenshot dir
        # whose paths aren't referenced in the DB any more, and
        # remove them. Use a set for O(1) membership.
        shot_dir = self._screenshot_dir()
        orphan_deleted = 0
        if shot_dir.exists():
            try:
                referenced_paths = await asyncio.to_thread(
                    self._list_referenced_screenshot_paths,
                )
                referenced = {Path(rel).name for rel in referenced_paths}
                for f in shot_dir.iterdir():
                    if not f.is_file():
                        continue
                    if f.name in referenced:
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
                "(TTL %dh)",
                aged_deleted, orphan_deleted, max_age_h,
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
