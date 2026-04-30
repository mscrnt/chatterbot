"""Engaging-subjects pipeline tests — pin the LLM-first / embedding-
for-identity contract.

Slice-2 used online cosine clustering on raw chat embeddings and
collapsed everything into one mega-cluster on real chat. The slice-7
rewrite inverts the work split: the LLM does topic modeling on
numbered chat messages and cites supporting `message_ids`; the
dashboard then matches each returned subject to a persistent
identity by embedding the SUBJECT NAME (not the messages) and
comparing against existing entries.

These tests pin the new contracts:
  - Empty / sub-threshold input → empty cache, no LLM call.
  - LLM returns multiple subjects → all of them land (slice-2's
    big bug was dropping subjects 2..N because of positional
    matching against a single dirty cluster).
  - Same subject across two refreshes → same persistent
    `subject_id` even when the LLM rephrases the name slightly.
  - Distinct subjects across refreshes → distinct ids.
  - Sensitive flag drops without leaking into the cache.
  - Blocklist drops by literal name.
  - Subjects with no grounded message_ids drop (anti-hallucination).
  - Skip-if-unchanged: identical inputs don't fire a second LLM call.
  - Aging: persistent subjects unseen for 2× window get pruned.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from chatterbot.insights import (
    EngagingSubjectEntry,
    EngagingSubjectsCache,
    InsightsService,
)
from chatterbot.llm.schemas import EngagingSubject, EngagingSubjectsResponse


def _ts(minutes_ago: int) -> str:
    """ISO-UTC timestamp `minutes_ago` minutes back, in SQLite's
    default datetime() shape. Same helper used by the
    open-questions tests."""
    dt = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


@pytest.fixture
def svc(tmp_repo, mock_llm) -> InsightsService:
    """A fresh InsightsService bound to the tmp_repo + mock_llm.
    SimpleNamespace settings — the engaging-subjects path only
    reads a handful of attrs and falls back to defaults via
    `getattr`, so we don't need a full Settings object."""
    settings = SimpleNamespace(
        db_path="ignored",
        screenshot_interval_seconds=0,
    )
    return InsightsService(tmp_repo, mock_llm, settings)


def _seed_chat(make_message, *content: str, name: str = "alice"):
    """Drop a list of messages into the repo as the same chatter,
    timestamps spaced over the last 5 minutes so they all clear
    the SUBJECTS_LOOKBACK_MINUTES window. Returns the inserted ids."""
    return [
        make_message(name=name, content=c, ts=_ts(5 - i))
        for i, c in enumerate(content)
    ]


# ---- empty / sub-threshold input ----


async def test_too_quiet_resets_cache_no_llm_call(svc, mock_llm, make_message):
    """Fewer than 5 messages is "too quiet to bother." Cache resets
    to empty, no LLM call fires."""
    # Pre-seed a stale cache so we can prove it actually got reset.
    svc._subjects_cache = EngagingSubjectsCache(
        subjects=[EngagingSubjectEntry(
            name="stale", drivers=["x"], msg_count=1,
        )], refreshed_at=0.0,
    )
    # Only 3 messages — below the floor.
    for i in range(3):
        make_message(name="alice", content=f"hi {i}", ts=_ts(2))

    await svc._refresh_engaging_subjects()

    assert svc._subjects_cache.subjects == []
    assert mock_llm.calls == []


# ---- multi-subject responses ----


async def test_multi_subject_response_all_applied(svc, mock_llm, make_message):
    """The slice-2 bug: LLM returned multiple subjects but only the
    first matched a sent cluster, so subjects 2..N got dropped. In
    the new pipeline there are no clusters — every returned subject
    that cites grounded message_ids lands."""
    ids = _seed_chat(
        make_message,
        "ng4 parry timing is brutal compared to ng2",
        "ng4 windows feel tighter on izuna drops",
        "izuna's hyper-armor isn't carrying",
        "anyone else getting wrecked on stage 3 boss",
        "stage 3 boss is the wall fr",
        "thread B msg one",
        "thread B msg two",
    )

    mock_llm.queue_response(
        call_site="insights.engaging_subjects",
        response=EngagingSubjectsResponse(subjects=[
            EngagingSubject(
                name="NG4 parry timing vs NG2",
                drivers=["alice"], msg_count=3,
                brief="Comparing parry windows in NG4 to NG2.",
                angles=["window tightness", "izuna drops"],
                message_ids=ids[0:3],
            ),
            EngagingSubject(
                name="NG4 stage 3 boss difficulty",
                drivers=["alice"], msg_count=2,
                brief="Stage 3 boss as a difficulty wall.",
                angles=[],
                message_ids=ids[3:5],
            ),
        ]),
    )

    await svc._refresh_engaging_subjects()

    cached = svc.subjects_cache.subjects
    names = {s.name for s in cached}
    assert "NG4 parry timing vs NG2" in names
    assert "NG4 stage 3 boss difficulty" in names
    assert len(cached) == 2


# ---- persistent identity matching ----


async def test_same_subject_across_refreshes_keeps_id(svc, mock_llm, make_message):
    """Two refreshes, same subject name → one persistent identity
    (same `subject_id` in `_subjects`). The LLM's name embedding
    is what drives the match."""
    ids = _seed_chat(
        make_message, "talking about route X tips", "route X strats",
        "more route X talk", "route X again", "route X questions",
    )

    # First refresh — spawn the persistent identity.
    mock_llm.queue_response(
        call_site="insights.engaging_subjects",
        response=EngagingSubjectsResponse(subjects=[EngagingSubject(
            name="Route X strategy",
            drivers=["alice"], msg_count=4,
            brief="Discussing route X.", angles=[], message_ids=ids,
        )]),
    )
    await svc._refresh_engaging_subjects()
    first_ids = list(svc._subjects.keys())
    assert len(first_ids) == 1
    persistent_id = first_ids[0]

    # New messages so latest_message_id moves forward (otherwise
    # the skip-if-unchanged short-circuit fires). Need 5+ messages
    # in the window for the pipeline to run at all — the existing
    # `ids` from the first seed are still in the lookback, so just
    # adding 2 here would also work, but we go with 5 fresh ones
    # for clarity.
    new_ids = _seed_chat(
        make_message, "route X update", "still on route X",
        "route X v2", "another route X", "more route X",
        name="bob",
    )

    # Second refresh — LLM returns the SAME-named subject.
    mock_llm.queue_response(
        call_site="insights.engaging_subjects",
        response=EngagingSubjectsResponse(subjects=[EngagingSubject(
            name="Route X strategy",  # identical name
            drivers=["alice", "bob"], msg_count=2,
            brief="More route X talk.", angles=[], message_ids=new_ids,
        )]),
    )
    await svc._refresh_engaging_subjects()

    # Same persistent identity reused.
    assert list(svc._subjects.keys()) == [persistent_id]
    assert svc._subjects[persistent_id].refresh_count == 2


async def test_rephrased_subject_matches_existing_identity(svc, mock_llm, make_message):
    """The MockLLMClient's embed() is content-aware (sha256 →
    floats), not semantic. So "RE9 parry timing" and "Resident
    Evil 9 parry windows" produce roughly-orthogonal random
    vectors, sometimes with NEGATIVE cosine.

    To test the identity-matching MECHANISM (regardless of the
    embedder's semantic ability), drop the identity threshold to
    -1.1 — every non-empty subject set then matches. If the
    matching code path breaks (e.g., a refactor stops indexing
    every subject), this test still catches it."""
    svc.settings = SimpleNamespace(
        engaging_subjects_identity_threshold=-1.1,
        screenshot_interval_seconds=0, db_path="ignored",
    )
    ids = _seed_chat(
        make_message, "msg one", "msg two", "msg three", "msg four", "msg five",
    )

    mock_llm.queue_response(
        call_site="insights.engaging_subjects",
        response=EngagingSubjectsResponse(subjects=[EngagingSubject(
            name="RE9 parry timing",
            drivers=["alice"], msg_count=3,
            brief="…", angles=[], message_ids=ids,
        )]),
    )
    await svc._refresh_engaging_subjects()
    persistent_id = list(svc._subjects.keys())[0]

    # New messages so the watermark moves + the 5-msg floor clears
    # for the second refresh too.
    new_ids = _seed_chat(
        make_message, "more talk", "still talk", "again",
        "and again", "yet again", name="bob",
    )

    mock_llm.queue_response(
        call_site="insights.engaging_subjects",
        response=EngagingSubjectsResponse(subjects=[EngagingSubject(
            name="Resident Evil 9 parry windows",  # rephrased
            drivers=["bob"], msg_count=2,
            brief="…", angles=[], message_ids=new_ids,
        )]),
    )
    await svc._refresh_engaging_subjects()

    # Threshold 0.0 → every name matches → identity reused.
    assert list(svc._subjects.keys()) == [persistent_id]
    # Name updated to the latest phrasing.
    assert svc._subjects[persistent_id].name == "Resident Evil 9 parry windows"


# ---- filtering ----


async def test_sensitive_subjects_dropped(svc, mock_llm, make_message):
    """LLM flags a subject as `is_sensitive=True` → never appears
    in the cache, no persistent identity created."""
    ids = _seed_chat(make_message, "msg one", "msg two", "msg three", "msg four", "msg five")
    mock_llm.queue_response(
        call_site="insights.engaging_subjects",
        response=EngagingSubjectsResponse(subjects=[
            EngagingSubject(
                name="legit subject", drivers=["alice"], msg_count=2,
                brief="…", message_ids=ids[:2],
            ),
            EngagingSubject(
                name="political controversy",
                is_sensitive=True,
                drivers=["alice"], msg_count=2,
                brief="…", message_ids=ids[2:4],
            ),
        ]),
    )
    await svc._refresh_engaging_subjects()
    names = {s.name for s in svc.subjects_cache.subjects}
    assert "legit subject" in names
    assert "political controversy" not in names
    # Persistent identity NOT created for the sensitive one.
    persistent_names = {s.name for s in svc._subjects.values()}
    assert "political controversy" not in persistent_names


async def test_blocklist_drops_extracted_subject(svc, mock_llm, make_message, tmp_repo):
    """Streamer rejected "Hallucinated thing" via reject_subject.
    Next refresh's LLM returns the same name → drop without
    creating an identity."""
    svc.reject_subject(slug="abc", name="Hallucinated thing")

    ids = _seed_chat(make_message, "msg one", "msg two", "msg three", "msg four", "msg five")
    mock_llm.queue_response(
        call_site="insights.engaging_subjects",
        response=EngagingSubjectsResponse(subjects=[EngagingSubject(
            name="Hallucinated thing",  # exact match against blocklist
            drivers=["alice"], msg_count=3,
            brief="…", message_ids=ids[:3],
        )]),
    )
    await svc._refresh_engaging_subjects()
    assert svc.subjects_cache.subjects == []
    # No persistent identity spawned for the blocked subject.
    assert all(
        s.name != "Hallucinated thing" for s in svc._subjects.values()
    )


# ---- anti-hallucination ----


async def test_subject_with_no_message_ids_dropped(svc, mock_llm, make_message):
    """A subject the LLM returned without citing any input message
    ids is too weak to surface. Dropped from cache + no persistent
    identity."""
    _seed_chat(make_message, "msg one", "msg two", "msg three", "msg four", "msg five")
    mock_llm.queue_response(
        call_site="insights.engaging_subjects",
        response=EngagingSubjectsResponse(subjects=[EngagingSubject(
            name="Ungrounded subject",
            drivers=["alice"], msg_count=0,
            brief="…", message_ids=[],  # no citations
        )]),
    )
    await svc._refresh_engaging_subjects()
    assert svc.subjects_cache.subjects == []
    assert all(
        s.name != "Ungrounded subject" for s in svc._subjects.values()
    )


async def test_hallucinated_message_id_filtered(svc, mock_llm, make_message):
    """LLM cites a message_id that isn't in the current window.
    The id is silently dropped; if at least one VALID id remains,
    the subject still lands."""
    real_ids = _seed_chat(make_message, "msg one", "msg two", "msg three", "msg four", "msg five")
    mock_llm.queue_response(
        call_site="insights.engaging_subjects",
        response=EngagingSubjectsResponse(subjects=[EngagingSubject(
            name="Mostly grounded",
            drivers=["alice"], msg_count=2,
            brief="…",
            message_ids=[real_ids[0], 999_999, real_ids[1]],  # 999_999 is fake
        )]),
    )
    await svc._refresh_engaging_subjects()
    # Subject lands — fake id dropped, real ones kept.
    assert any(
        s.name == "Mostly grounded" for s in svc.subjects_cache.subjects
    )
    # Cached subject's msg_ids contain ONLY the real ones.
    landed = next(
        s for s in svc.subjects_cache.subjects if s.name == "Mostly grounded"
    )
    assert 999_999 not in landed.msg_ids


# ---- skip-if-unchanged ----


async def test_skip_when_latest_message_id_unchanged(svc, mock_llm, make_message):
    """Same input twice → second refresh short-circuits and serves
    the cached result without firing the LLM. Mirrors the
    talking-points refresh's watermark optimisation."""
    ids = _seed_chat(make_message, "m1", "m2", "m3", "m4", "m5")
    mock_llm.queue_response(
        call_site="insights.engaging_subjects",
        response=EngagingSubjectsResponse(subjects=[EngagingSubject(
            name="A subject", drivers=["alice"], msg_count=5,
            brief="…", message_ids=ids,
        )]),
    )
    await svc._refresh_engaging_subjects()
    assert len(mock_llm.calls) == 1

    # No new messages — second refresh must NOT call the LLM.
    await svc._refresh_engaging_subjects()
    assert len(mock_llm.calls) == 1


# ---- aging ----


async def test_persistent_subject_ages_out(svc, mock_llm, make_message):
    """A persistent subject not surfaced for 2× the lookback
    window gets pruned. We simulate the time delta by
    mutating `last_seen_ts` directly rather than wall-clocking."""
    ids = _seed_chat(make_message, "m1", "m2", "m3", "m4", "m5")
    mock_llm.queue_response(
        call_site="insights.engaging_subjects",
        response=EngagingSubjectsResponse(subjects=[EngagingSubject(
            name="A subject", drivers=["alice"], msg_count=5,
            brief="…", message_ids=ids,
        )]),
    )
    await svc._refresh_engaging_subjects()
    assert len(svc._subjects) == 1
    persistent_id = list(svc._subjects.keys())[0]

    # Backdate the subject's last_seen_ts past the aging cutoff.
    # SUBJECTS_LOOKBACK_MINUTES default is 20, so 2× = 40 min;
    # set last_seen 60 min in the past.
    svc._subjects[persistent_id].last_seen_ts = time.time() - 60 * 60

    # Drive a refresh with new messages BUT the LLM returns no
    # subjects (chat moved on) — the aging pass should still run.
    new_ids = _seed_chat(make_message, "n1", "n2", "n3", "n4", "n5", name="bob")
    mock_llm.queue_response(
        call_site="insights.engaging_subjects",
        response=EngagingSubjectsResponse(subjects=[]),
    )
    await svc._refresh_engaging_subjects()
    assert persistent_id not in svc._subjects


# ---- cache shape contract for the template ----


async def test_cache_entry_carries_msg_ids_for_expand_route(svc, mock_llm, make_message):
    """The `/insights/subject/{slug}/expand` route hydrates messages
    from `EngagingSubjectEntry.msg_ids`. Pin that the cache entry
    actually carries those ids after a successful refresh."""
    ids = _seed_chat(make_message, "m1", "m2", "m3", "m4", "m5")
    mock_llm.queue_response(
        call_site="insights.engaging_subjects",
        response=EngagingSubjectsResponse(subjects=[EngagingSubject(
            name="A subject", drivers=["alice"], msg_count=3,
            brief="…", message_ids=ids[:3],
        )]),
    )
    await svc._refresh_engaging_subjects()
    cached = svc.subjects_cache.subjects[0]
    assert cached.msg_ids == ids[:3]
    # Slug is still derived from the name, same shape as before.
    assert cached.slug
    assert len(cached.slug) == 12
