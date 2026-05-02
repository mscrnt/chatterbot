"""Tests for slice 12 — perfect-pass transcription + audio storage.

Covers the schema migration (new columns are nullable + default),
the repo helpers that drive the perfect-pass queue, and the audio-
clip persistence helper. Doesn't actually run faster-whisper —
exercising the model would need real audio + GPU + a multi-hundred-MB
download. The transcribe-side logic is tested at the boundary:
"given chunks pending, the helpers find them; given a fresh-text +
embedding, the update writes through correctly."""
from __future__ import annotations

from pathlib import Path

import pytest

# numpy is brought in by faster-whisper / sounddevice, but the tests
# only need it for the audio-clip persistence helper. Skip cleanly
# when it's missing (dev extra has it via Pillow's chain anyway).
np = pytest.importorskip("numpy")

from chatterbot.repo import ChatterRepo
from chatterbot.transcript import (
    _looks_like_whisper_hallucination,
    _persist_audio_clip,
    _should_defer_group_summary,
)


# ---- schema migration ----

def test_new_columns_are_nullable(tmp_repo: ChatterRepo):
    """Adding a chunk WITHOUT the slice-12 fields still works —
    legacy callers that don't know about avg_logprob / audio_path
    aren't broken."""
    chunk_id = tmp_repo.add_transcript_chunk(text="hello world")
    chunk = tmp_repo.get_transcript_chunk(chunk_id)
    assert chunk is not None
    assert chunk.text == "hello world"
    assert chunk.avg_logprob is None
    assert chunk.audio_path is None
    assert chunk.transcribed_v2_at is None


def test_new_columns_round_trip(tmp_repo: ChatterRepo):
    """avg_logprob + audio_path persist + come back through reads."""
    chunk_id = tmp_repo.add_transcript_chunk(
        text="hello world",
        avg_logprob=-0.42,
        audio_path="audio_clips/ab/cd/abcdef.wav",
    )
    chunk = tmp_repo.get_transcript_chunk(chunk_id)
    assert chunk.avg_logprob == pytest.approx(-0.42)
    assert chunk.audio_path == "audio_clips/ab/cd/abcdef.wav"
    assert chunk.transcribed_v2_at is None  # not yet refined


# ---- chunks_pending_perfect_pass ----

def test_pending_includes_low_confidence(tmp_repo: ChatterRepo):
    """A chunk with audio_path + avg_logprob below the threshold +
    no v2 stamp shows up in the queue."""
    tmp_repo.add_transcript_chunk(
        text="muffled", avg_logprob=-1.0,
        audio_path="audio_clips/aa/bb/aabbcc.wav",
    )
    pending = tmp_repo.chunks_pending_perfect_pass(
        confidence_threshold=-0.5, limit=10,
    )
    assert len(pending) == 1
    assert pending[0].text == "muffled"


def test_pending_includes_null_logprob(tmp_repo: ChatterRepo):
    """Legacy chunks (NULL avg_logprob) but with audio_path get
    queued — the perfect pass benefits everything that has audio,
    not just chunks captured after the slice landed."""
    tmp_repo.add_transcript_chunk(
        text="legacy", avg_logprob=None,
        audio_path="audio_clips/aa/bb/legacy.wav",
    )
    pending = tmp_repo.chunks_pending_perfect_pass(
        confidence_threshold=-0.5, limit=10,
    )
    assert len(pending) == 1


def test_pending_excludes_high_confidence(tmp_repo: ChatterRepo):
    """A chunk with audio + high confidence (above threshold) does
    NOT enter the queue. The first-pass text is good enough."""
    tmp_repo.add_transcript_chunk(
        text="crystal clear", avg_logprob=-0.1,
        audio_path="audio_clips/aa/bb/clear.wav",
    )
    pending = tmp_repo.chunks_pending_perfect_pass(
        confidence_threshold=-0.5, limit=10,
    )
    assert pending == []


def test_pending_excludes_no_audio(tmp_repo: ChatterRepo):
    """A chunk WITHOUT audio_path can't be re-transcribed (the bytes
    are gone), so it's excluded regardless of confidence."""
    tmp_repo.add_transcript_chunk(
        text="no audio", avg_logprob=-1.0, audio_path=None,
    )
    pending = tmp_repo.chunks_pending_perfect_pass(
        confidence_threshold=-0.5, limit=10,
    )
    assert pending == []


def test_pending_excludes_already_refined(tmp_repo: ChatterRepo):
    """Once a chunk has been refined (transcribed_v2_at stamped),
    it doesn't re-enter the queue. The loop is supposed to be a
    one-time-per-chunk operation."""
    chunk_id = tmp_repo.add_transcript_chunk(
        text="muffled v1", avg_logprob=-1.0,
        audio_path="audio_clips/aa/bb/refined.wav",
    )
    tmp_repo.update_chunk_text_v2(chunk_id, "muffled v2 (refined)")
    pending = tmp_repo.chunks_pending_perfect_pass(
        confidence_threshold=-0.5, limit=10,
    )
    assert pending == []


def test_pending_oldest_first(tmp_repo: ChatterRepo):
    """Queue drains in capture order so a slow GPU never starves
    older chunks of their refine pass."""
    ids = []
    for i in range(3):
        ids.append(tmp_repo.add_transcript_chunk(
            text=f"chunk {i}", avg_logprob=-1.0,
            audio_path=f"audio_clips/aa/bb/{i}.wav",
        ))
    pending = tmp_repo.chunks_pending_perfect_pass(
        confidence_threshold=-0.5, limit=10,
    )
    assert [c.id for c in pending] == ids


# ---- update_chunk_text_v2 ----

def test_update_writes_text_and_stamp(tmp_repo: ChatterRepo):
    """update_chunk_text_v2 rewrites text AND stamps
    transcribed_v2_at — the stamp is what removes the chunk from
    the queue, so missing it would create an infinite loop."""
    chunk_id = tmp_repo.add_transcript_chunk(
        text="muffled v1", avg_logprob=-1.0,
        audio_path="audio_clips/aa/bb/refined.wav",
    )
    tmp_repo.update_chunk_text_v2(chunk_id, "crystal clear v2")
    chunk = tmp_repo.get_transcript_chunk(chunk_id)
    assert chunk.text == "crystal clear v2"
    assert chunk.transcribed_v2_at is not None


def test_update_replaces_embedding(tmp_repo: ChatterRepo):
    """When a perfect pass finds different text, the search index
    needs to follow. update_chunk_text_v2 with new_embedding
    replaces the vec_transcripts row so search returns the
    refined text, not the first-pass text."""
    chunk_id = tmp_repo.add_transcript_chunk(
        text="muffled", avg_logprob=-1.0,
        audio_path="audio_clips/aa/bb/refined.wav",
        embedding=[0.1] * tmp_repo.embed_dim,
    )
    new_emb = [0.9] * tmp_repo.embed_dim
    tmp_repo.update_chunk_text_v2(
        chunk_id, "refined text", new_embedding=new_emb,
    )
    # Cosine search for the new embedding should return this chunk.
    hits = tmp_repo.search_transcripts(new_emb, k=5)
    assert any(h[0].id == chunk_id for h in hits)


# ---- list_referenced_audio_paths (orphan-sweep input) ----

def test_referenced_paths_dedups(tmp_repo: ChatterRepo):
    """Same audio_path on two chunks (content-hash dedup case)
    surfaces as one entry — the orphan sweeper iterates this list
    and a duplicate would just slow the comparison without changing
    correctness."""
    p = "audio_clips/dd/ee/ffaabb.wav"
    tmp_repo.add_transcript_chunk(text="a", audio_path=p)
    tmp_repo.add_transcript_chunk(text="b", audio_path=p)
    paths = tmp_repo.list_referenced_audio_paths()
    assert paths == [p]


def test_referenced_paths_excludes_null(tmp_repo: ChatterRepo):
    """Chunks without audio don't pollute the referenced-paths
    list. The sweep operates over actual file references."""
    tmp_repo.add_transcript_chunk(text="no audio", audio_path=None)
    tmp_repo.add_transcript_chunk(
        text="with audio", audio_path="audio_clips/aa/bb/c.wav",
    )
    paths = tmp_repo.list_referenced_audio_paths()
    assert paths == ["audio_clips/aa/bb/c.wav"]


# ---- _persist_audio_clip ----

def test_persist_audio_clip_writes_wav(tmp_path: Path):
    """The persistence helper writes a real WAV the standard
    library can read back. Round-tripping the data confirms the
    conversion (float32 → int16 → file → int16) is lossless at
    the bit level we care about."""
    sr = 16000
    audio = (np.random.rand(sr * 2).astype(np.float32) * 2 - 1) * 0.5
    rel = _persist_audio_clip(audio, sr, tmp_path)
    assert rel is not None
    assert rel.startswith("audio_clips/")
    fpath = tmp_path / rel
    assert fpath.is_file()

    import wave
    with wave.open(str(fpath), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2  # int16
        assert w.getframerate() == sr
        assert w.getnframes() == sr * 2


def test_persist_audio_clip_dedups_identical(tmp_path: Path):
    """Two passes with byte-identical audio resolve to the same
    on-disk file via content-hash. A streamer paused on the same
    scene producing identical chunks doesn't accumulate duplicate
    audio files."""
    sr = 16000
    audio = np.full(sr, 0.25, dtype=np.float32)
    rel1 = _persist_audio_clip(audio, sr, tmp_path)
    rel2 = _persist_audio_clip(audio, sr, tmp_path)
    assert rel1 == rel2
    # And only ONE file exists under audio_clips/
    files = list((tmp_path / "audio_clips").rglob("*.wav"))
    assert len(files) == 1


# ---- _looks_like_whisper_hallucination (slice 13) ----
#
# The filter is INTENTIONALLY conservative — it only rejects refines
# that introduce caption-track artifacts that real streamer speech
# never produces. Outros and CTAs that streamers genuinely say
# ("thanks for watching", "subscribe", "bye bye") are NOT filtered;
# rejecting them would silently throw away legitimate refines.

@pytest.mark.parametrize("orig,refined,expected", [
    # Caption-track artifacts — refine is rejected.
    (
        "the next round is starting",
        "Subtitles by the Amara.org community",
        True,
    ),
    (
        "you can hear the song faintly",
        "[Music]",
        True,
    ),
    (
        "the audio is muffled here",
        "[Applause]",
        True,
    ),
    (
        "nothing important",
        "Captions by ChannelOne",
        True,
    ),
    (
        "we're moving on",
        "Korean subtitles",
        True,
    ),
    # Streamer-legit outros / CTAs — NOT filtered. These are real
    # speech a streamer can say.
    (
        "alright that's the stream",
        "Thanks for watching everybody!",
        False,
    ),
    (
        "alright let's keep going",
        "Subscribe to my channel for more!",
        False,
    ),
    (
        "I'll see what you mean",
        "I'll see you in the next video. Bye bye.",
        False,
    ),
    (
        "alright that's it",
        "see you next time, bye!",
        False,
    ),
    # Single-word refines — NOT filtered. "you", "bye", "thank you"
    # are all real speech; rejecting them throws away legit short
    # utterances.
    ("a longer thought here", "Bye.", False),
    ("a longer thought here", "you", False),
    ("a longer thought here", "thank you", False),
    # Caption-track artifact already in the original — refine
    # repeating it isn't a NEW hallucination, allowed through.
    (
        "[Music] playing in the background",
        "[Music] continues throughout",
        False,
    ),
    # Genuine refine with no hallucination signature — allowed.
    (
        "purry deployed",
        "Pulse deployed",
        False,
    ),
    # Same text on both sides — allowed (passthrough refine).
    ("nothing to refine here", "nothing to refine here", False),
    # Case-insensitivity sanity on the artifacts that DO trigger.
    (
        "alright let's keep going",
        "[MUSIC PLAYING]",
        True,
    ),
])
def test_hallucination_filter(orig: str, refined: str, expected: bool):
    assert _looks_like_whisper_hallucination(orig, refined) is expected


# ---- _should_defer_group_summary (slice 15) ----
#
# The gate's job: hold off the group-summary LLM call until the
# perfect-pass loop has had a chance to refine chunks in the
# window. The grace cap prevents indefinite blocking when refines
# never arrive (loop crash, GPU starved, etc).

class _StubChunk:
    """Minimal duck-typed chunk for the deferral helper. Real
    TranscriptChunk has more fields; the helper only reads three."""
    def __init__(self, ts: str, audio_path=None, avg_logprob=None,
                 transcribed_v2_at=None):
        self.ts = ts
        self.audio_path = audio_path
        self.avg_logprob = avg_logprob
        self.transcribed_v2_at = transcribed_v2_at


def _now_iso(offset_seconds: float = 0.0) -> str:
    """ISO-format timestamp `offset_seconds` away from now (negative
    = past). Matches the format `transcript_chunks.ts` uses."""
    from datetime import datetime, timedelta, timezone
    dt = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return dt.isoformat(timespec="seconds")


def test_defer_when_pending_and_window_fresh():
    """Recent unrefined chunk with low confidence + audio = defer."""
    chunks = [
        _StubChunk(
            ts=_now_iso(-30),  # 30s old
            audio_path="audio_clips/aa/bb/x.wav",
            avg_logprob=-1.0,
            transcribed_v2_at=None,
        ),
    ]
    defer, count, _age = _should_defer_group_summary(
        chunks, grace_seconds=240, confidence_threshold=-0.5,
    )
    assert defer is True
    assert count == 1


def test_no_defer_when_window_older_than_grace():
    """Window crosses grace cap → fire even with pending chunks. The
    perfect-pass loop has clearly had its time; one straggler
    shouldn't block summaries forever."""
    chunks = [
        _StubChunk(
            ts=_now_iso(-300),  # 5 min old
            audio_path="audio_clips/aa/bb/x.wav",
            avg_logprob=-1.0,
            transcribed_v2_at=None,
        ),
    ]
    defer, _count, _age = _should_defer_group_summary(
        chunks, grace_seconds=240, confidence_threshold=-0.5,
    )
    assert defer is False


def test_no_defer_when_all_refined():
    """Every refine-eligible chunk has been refined → fire now."""
    chunks = [
        _StubChunk(
            ts=_now_iso(-30),
            audio_path="audio_clips/aa/bb/x.wav",
            avg_logprob=-1.0,
            transcribed_v2_at=_now_iso(-10),
        ),
    ]
    defer, count, _age = _should_defer_group_summary(
        chunks, grace_seconds=240, confidence_threshold=-0.5,
    )
    assert defer is False
    assert count == 0


def test_no_defer_for_high_confidence_chunks():
    """Chunks with avg_logprob >= threshold never enter the perfect-
    pass queue, so waiting for them would wait forever. The gate
    must not block on high-confidence chunks even when they have
    audio + no v2 stamp."""
    chunks = [
        _StubChunk(
            ts=_now_iso(-30),
            audio_path="audio_clips/aa/bb/x.wav",
            avg_logprob=-0.1,  # well above -0.5 threshold
            transcribed_v2_at=None,
        ),
    ]
    defer, _count, _age = _should_defer_group_summary(
        chunks, grace_seconds=240, confidence_threshold=-0.5,
    )
    assert defer is False


def test_defer_for_null_logprob():
    """NULL avg_logprob (legacy chunks or whisper didn't return
    one) IS refine-eligible — `chunks_pending_perfect_pass` queues
    them. The gate must reflect the same behavior."""
    chunks = [
        _StubChunk(
            ts=_now_iso(-30),
            audio_path="audio_clips/aa/bb/x.wav",
            avg_logprob=None,
            transcribed_v2_at=None,
        ),
    ]
    defer, count, _age = _should_defer_group_summary(
        chunks, grace_seconds=240, confidence_threshold=-0.5,
    )
    assert defer is True
    assert count == 1


def test_no_defer_when_grace_zero():
    """grace_seconds=0 disables the gate — opt-out."""
    chunks = [
        _StubChunk(
            ts=_now_iso(-1),  # 1s old, very fresh
            audio_path="audio_clips/aa/bb/x.wav",
            avg_logprob=-1.0,
            transcribed_v2_at=None,
        ),
    ]
    defer, _count, _age = _should_defer_group_summary(
        chunks, grace_seconds=0, confidence_threshold=-0.5,
    )
    assert defer is False


def test_no_defer_for_chunks_without_audio():
    """A chunk without audio_path can't be refined no matter how
    long we wait. Pre-slice-12 chunks AND post-slice-12 chunks
    captured with audio storage off both fall in this bucket; the
    gate must not block on them."""
    chunks = [
        _StubChunk(
            ts=_now_iso(-30),
            audio_path=None,
            avg_logprob=-1.0,
            transcribed_v2_at=None,
        ),
    ]
    defer, _count, _age = _should_defer_group_summary(
        chunks, grace_seconds=240, confidence_threshold=-0.5,
    )
    assert defer is False


def test_no_defer_on_empty_chunks():
    """Empty input — nothing to defer."""
    defer, count, age = _should_defer_group_summary(
        [], grace_seconds=240, confidence_threshold=-0.5,
    )
    assert defer is False
    assert count == 0
    assert age == 0.0


def test_no_defer_on_malformed_timestamp():
    """If the youngest chunk's timestamp won't parse, the gate
    treats the window as old enough to fire (fail-open) rather than
    blocking on bad data."""
    chunks = [
        _StubChunk(
            ts="not-an-iso-timestamp",
            audio_path="audio_clips/aa/bb/x.wav",
            avg_logprob=-1.0,
            transcribed_v2_at=None,
        ),
    ]
    defer, _count, _age = _should_defer_group_summary(
        chunks, grace_seconds=240, confidence_threshold=-0.5,
    )
    assert defer is False


def test_persist_audio_clip_path_layout(tmp_path: Path):
    """Path is the content-hash layout `<sha[:2]>/<sha[2:4]>/<sha>.wav`
    so any one directory is bounded to ~256 entries even at huge
    scale. Mirrors the screenshot scheme."""
    sr = 16000
    audio = np.full(sr, 0.5, dtype=np.float32)
    rel = _persist_audio_clip(audio, sr, tmp_path)
    parts = Path(rel).parts
    # ['audio_clips', '<2-hex>', '<2-hex>', '<full-hex>.wav']
    assert parts[0] == "audio_clips"
    assert len(parts[1]) == 2
    assert len(parts[2]) == 2
    assert parts[3].endswith(".wav")
    # The filename's stem is the full sha256 hex (64 chars).
    assert len(Path(parts[3]).stem) == 64
