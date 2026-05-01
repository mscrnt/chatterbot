"""open_questions LLM-filter pass — cache shape, LLM-id matching,
empty-pool short-circuit.

The production flow:
  1. repo.recent_questions returns heuristic candidate clusters
  2. _refresh_open_questions builds a prompt naming each cluster by
     `last_msg_id` (the integer the LLM echoes back)
  3. LLM returns OpenQuestionsResponse with `candidate_id` per kept
     question
  4. We re-attach drivers from the original cluster onto an
     OpenQuestionEntry

The shape contract MUST stay stable because chat_questions.html reads
it positionally (q.question / q.count / q.drivers / q.latest_ts /
q.last_msg_id). These tests pin both the shape and the matching
logic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from chatterbot.insights import (
    InsightsService,
    OpenQuestionEntry,
    OpenQuestionsCache,
)
from chatterbot.llm.schemas import OpenQuestion, OpenQuestionsResponse


def _ts(minutes_ago: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _service(tmp_repo, mock_llm) -> InsightsService:
    """Build an InsightsService with a SimpleNamespace settings — only
    the attrs the open-questions path reads need to exist (and
    `getattr` defaults handle missing ones)."""
    settings = SimpleNamespace(
        db_path="ignored",
        screenshot_interval_seconds=0,
        # Disable modal-output pre-warming in tests — see the
        # parallel comment in test_engaging_subjects.py.
        insights_modal_prewarm_top_n=0,
    )
    return InsightsService(tmp_repo, mock_llm, settings)


# ---- empty-pool short-circuit ----


async def test_empty_candidate_pool_clears_cache(tmp_repo, mock_llm):
    """When no chat questions exist, _refresh_open_questions must NOT
    call the LLM (would waste a 30s think pass on nothing) and must
    flush the cache so the panel renders empty."""
    svc = _service(tmp_repo, mock_llm)
    # Pre-seed a stale cache to verify the flush.
    svc._questions_cache = OpenQuestionsCache(
        questions=[
            OpenQuestionEntry(
                question="stale", count=1, drivers=[],
                latest_ts="x", last_msg_id=1,
            ),
        ],
        refreshed_at=0.0,
    )
    await svc._refresh_open_questions()
    assert svc.open_questions_cache.questions == []
    assert svc.open_questions_cache.refreshed_at is not None
    # No LLM call fired.
    assert mock_llm.calls == []


# ---- happy path: LLM keeps both candidates ----


async def test_llm_response_re_attaches_drivers(
    tmp_repo, mock_llm, make_message,
):
    """The LLM only returns `candidate_id` + `question`; the service
    must re-look-up the drivers/count/timestamp from the original
    heuristic cluster. Pinning this so a refactor that drops the
    re-attachment step gets caught."""
    mid_a = make_message(content="any tips for boss 3?", name="alice", ts=_ts(2))
    mid_b = make_message(content="when does the new patch drop?", name="bob", ts=_ts(1))

    svc = _service(tmp_repo, mock_llm)

    # The LLM wrapper produces ANY question via candidate_id echo. We
    # queue a response that refines wording slightly (legal under the
    # prompt's "lightly clean" allowance) for one and keeps the other
    # verbatim — exercises both paths.
    mock_llm.queue_response(
        call_site="insights.open_questions",  # production code currently doesn't pass call_site
        response=OpenQuestionsResponse(questions=[
            OpenQuestion(candidate_id=mid_a, question="Any tips for boss 3?"),
            OpenQuestion(candidate_id=mid_b, question="When does the new patch drop?"),
        ]),
    )

    await svc._refresh_open_questions()
    cache = svc.open_questions_cache
    assert len(cache.questions) == 2

    by_id = {q.last_msg_id: q for q in cache.questions}
    assert mid_a in by_id and mid_b in by_id

    q_a = by_id[mid_a]
    assert q_a.question == "Any tips for boss 3?"
    assert q_a.count == 1
    assert len(q_a.drivers) == 1
    assert q_a.drivers[0].name == "alice"


# ---- LLM filtering: drops a candidate by omitting its id ----


async def test_dropped_candidate_does_not_appear_in_cache(
    tmp_repo, mock_llm, make_message,
):
    """When the LLM omits a candidate_id (its way of saying "this is
    rhetorical / answered / off-topic"), the cache MUST NOT include
    that cluster. This is the whole point of the filter pass."""
    mid_a = make_message(content="when's the next stream?", name="alice", ts=_ts(2))
    mid_b = make_message(content="rhetorical question right?", name="bob", ts=_ts(1))

    svc = _service(tmp_repo, mock_llm)
    mock_llm.queue_response(
        call_site="insights.open_questions",
        response=OpenQuestionsResponse(questions=[
            OpenQuestion(candidate_id=mid_a, question="when's the next stream?"),
            # mid_b deliberately omitted — LLM filtered it.
        ]),
    )

    await svc._refresh_open_questions()
    cache = svc.open_questions_cache
    assert len(cache.questions) == 1
    assert cache.questions[0].last_msg_id == mid_a


# ---- hallucinated candidate_id: silently dropped ----


async def test_hallucinated_candidate_id_is_dropped(
    tmp_repo, mock_llm, make_message,
):
    """If the LLM invents a candidate_id that wasn't in the input,
    the service must drop it (not crash, not display a broken row).
    Defends against output drift on weaker models."""
    mid_a = make_message(content="real question?", name="alice", ts=_ts(2))

    svc = _service(tmp_repo, mock_llm)
    mock_llm.queue_response(
        call_site="insights.open_questions",
        response=OpenQuestionsResponse(questions=[
            OpenQuestion(candidate_id=mid_a, question="real question?"),
            OpenQuestion(candidate_id=999_999, question="hallucinated"),
        ]),
    )

    await svc._refresh_open_questions()
    cache = svc.open_questions_cache
    assert len(cache.questions) == 1
    assert cache.questions[0].last_msg_id == mid_a


# ---- LLM error handling: cache reflects error, not stale data ----


async def test_validation_error_records_error_in_cache(
    tmp_repo, mock_llm, make_message,
):
    """If the LLM call raises ValidationError (Pydantic mismatch), the
    cache must clear AND set `error` so the dashboard surfaces a
    diagnostic instead of silently rendering stale data. The mock
    raises by having no queued response → AssertionError, which the
    service should catch in its broad exception handler."""
    make_message(content="will it run?", name="alice", ts=_ts(2))

    svc = _service(tmp_repo, mock_llm)
    # No queued response → mock raises AssertionError on the call.
    await svc._refresh_open_questions()

    cache = svc.open_questions_cache
    assert cache.questions == []
    assert cache.error is not None


# ---- de-duplication: LLM merging two cands onto one id ----


async def test_duplicate_candidate_id_in_response_kept_once(
    tmp_repo, mock_llm, make_message,
):
    """If the LLM returns the same candidate_id twice (rare, but
    possible when it merges near-dupes), only the first must appear
    in the cache — duplicates would render as two identical rows."""
    mid = make_message(content="how to parry?", name="alice", ts=_ts(2))

    svc = _service(tmp_repo, mock_llm)
    mock_llm.queue_response(
        call_site="insights.open_questions",
        response=OpenQuestionsResponse(questions=[
            OpenQuestion(candidate_id=mid, question="How to parry?"),
            OpenQuestion(candidate_id=mid, question="how do you parry"),
        ]),
    )

    await svc._refresh_open_questions()
    cache = svc.open_questions_cache
    assert len(cache.questions) == 1
    # First wins.
    assert cache.questions[0].question == "How to parry?"


# ---- cache-shape contract for the template ----


async def test_cache_entry_has_all_template_fields(
    tmp_repo, mock_llm, make_message,
):
    """chat_questions.html reads `q.question / q.count / q.drivers /
    q.latest_ts / q.last_msg_id` and `d.name / d.user_id / d.ts`.
    Pin every field — a regression that drops one would render a
    broken anchor href in the dashboard."""
    # Content needs to clear `recent_questions(min_chars=8)`, so use
    # a real-shape question rather than the bare "route?" we tested
    # the cluster shape against.
    mid = make_message(content="any good route tips?", name="alice", ts=_ts(2))

    svc = _service(tmp_repo, mock_llm)
    mock_llm.queue_response(
        call_site="insights.open_questions",
        response=OpenQuestionsResponse(questions=[
            OpenQuestion(candidate_id=mid, question="Any good route tips?"),
        ]),
    )
    await svc._refresh_open_questions()
    q = svc.open_questions_cache.questions[0]

    # Top-level cluster fields the template reads.
    assert isinstance(q.question, str)
    assert isinstance(q.count, int)
    assert isinstance(q.last_msg_id, int)
    assert isinstance(q.latest_ts, str)
    assert isinstance(q.drivers, list)
    # Per-driver fields the pill renderer reads.
    d = q.drivers[0]
    assert d.name == "alice"
    assert isinstance(d.user_id, str) and d.user_id
    assert isinstance(d.ts, str) and d.ts
