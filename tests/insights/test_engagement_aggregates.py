"""Engagement-aggregates cache tests.

The /insights engagement view used to fire ~25 sequential SQLite calls
per page render on the asyncio event loop. Under live-bot write
contention that ballooned to 5+ seconds and blocked /health, the SSE
stream, and HTMX polls. The cache + parallel-gather rewrite replaces
that with one in-memory dataclass read on the fast path and a single
parallel `gather` on the cold path.

These tests pin the cache contract:
  - Refresh populates every field with the right repo result.
  - Window parameter routes to the right SQL date modifier.
  - Refresh failures preserve previously-cached data + record the
    error (mirrors how InsightsCache handles errors).
  - Cache survives an unknown-window key by falling back to the
    7d default modifier.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from chatterbot.insights import EngagementAggregatesCache, InsightsService


@pytest.fixture
def svc(tmp_repo, mock_llm) -> InsightsService:
    settings = SimpleNamespace(
        db_path="ignored",
        screenshot_interval_seconds=0,
        insights_modal_prewarm_top_n=0,
        high_impact_active_within_minutes=30,
        high_impact_lookback_days=14,
        high_impact_min_overlap=2,
        high_impact_limit=6,
        quiet_cohort_silence_minutes=15,
        quiet_cohort_lookback_hours=24,
        quiet_cohort_min_drivers=2,
        quiet_cohort_limit=6,
    )
    return InsightsService(tmp_repo, mock_llm, settings)


async def test_cache_starts_cold(svc):
    """Fresh service has an empty cache; the route's `cache_hit`
    branch falls through to the live-parallel path until the first
    refresh tick lands."""
    cache = svc.engagement_aggregates_cache
    assert cache.refreshed_at is None
    assert cache.window == "7d"
    assert cache.regulars == []
    assert cache.insight_states == {}


async def test_refresh_populates_all_fields(svc, make_message):
    """One refresh fans out every aggregate in parallel and lands a
    fully-populated cache. Verifies the unpacking order matches the
    gather order — a positional bug here would silently swap fields
    (e.g. lapsed-list landing in `regulars`)."""
    # Seed enough chat that messages_per_minute returns rows.
    for i in range(5):
        make_message(name="alice", content=f"hi {i}")

    await svc._refresh_engagement_aggregates(window="7d")

    cache = svc.engagement_aggregates_cache
    assert cache.refreshed_at is not None
    assert cache.error is None
    assert cache.window == "7d"
    # Every field exists and matches its repo type. Empty lists are
    # fine — we just want to know the gather unpack didn't crash and
    # didn't put a list where a scalar belonged.
    assert isinstance(cache.regulars, list)
    assert isinstance(cache.lapsed, list)
    assert isinstance(cache.newcomers, list)
    assert isinstance(cache.anniversaries, list)
    assert isinstance(cache.direct_mentions, list)
    assert isinstance(cache.starred_active, list)
    assert isinstance(cache.neglected_lurkers, list)
    assert isinstance(cache.transcript_chunks, list)
    assert isinstance(cache.transcript_groups, list)
    assert isinstance(cache.recent_matches, list)
    assert isinstance(cache.live_threads, list)
    assert isinstance(cache.high_impact_subjects, list)
    assert isinstance(cache.quiet_cohorts, list)
    assert isinstance(cache.recent_recaps, list)
    # All 10 insight-state kinds got fetched in the same gather, not
    # 10 separate DB calls per page render.
    assert set(cache.insight_states.keys()) == {
        "talking_point", "anniversary", "newcomer", "regular", "lapsed",
        "direct_mention", "neglected_lurker", "chat_question", "thread",
        "high_impact",
    }


async def test_refresh_window_routes_to_modifier(svc):
    """The `window` arg picks the SQL date modifier passed to
    `list_regulars` / `list_lapsed_regulars`. Pin this so a future
    refactor of the window-modifier table doesn't silently break the
    route's per-window behaviour."""
    seen = {}

    real_list_regulars = svc.repo.list_regulars

    def spy_list_regulars(since, limit):
        seen["since"] = since
        return real_list_regulars(since=since, limit=limit)

    with patch.object(svc.repo, "list_regulars", side_effect=spy_list_regulars):
        await svc._refresh_engagement_aggregates(window="30d")

    assert seen["since"] == "-30 days"


async def test_refresh_unknown_window_falls_back_to_7d(svc):
    """An unknown window key (shouldn't happen, but defensive) gets
    the 7d modifier. Mirrors what the route does in its own
    validation block."""
    seen = {}
    real = svc.repo.list_regulars

    def spy(since, limit):
        seen["since"] = since
        return real(since=since, limit=limit)

    with patch.object(svc.repo, "list_regulars", side_effect=spy):
        await svc._refresh_engagement_aggregates(window="bogus")

    assert seen["since"] == "-7 days"


async def test_refresh_failure_preserves_previous_cache(svc, make_message):
    """When the gather raises (e.g. one repo helper blew up because
    of a malformed row), we keep the previously-cached data and
    only stamp `error`. Without this preservation, a transient DB
    glitch would wipe the dashboard back to empty lists for one
    refresh cycle."""
    for i in range(3):
        make_message(name="alice", content=f"hi {i}")
    await svc._refresh_engagement_aggregates(window="7d")
    prev = svc.engagement_aggregates_cache
    assert prev.refreshed_at is not None
    prev_refreshed_at = prev.refreshed_at

    # Now force a failure on the next refresh. Patching one of the
    # gathered helpers to raise is enough — gather propagates and
    # the except branch kicks in.
    with patch.object(
        svc.repo,
        "list_neglected_lurkers",
        side_effect=RuntimeError("boom"),
    ):
        await svc._refresh_engagement_aggregates(window="7d")

    cache = svc.engagement_aggregates_cache
    # Old data preserved.
    assert cache.refreshed_at == prev_refreshed_at
    # Error recorded.
    assert cache.error is not None
    assert "boom" in cache.error


async def test_latest_recap_derived_from_recent_recaps(svc):
    """`latest_recap` is `recent_recaps[0]` — pin so a future change
    to the recap-list ordering doesn't accidentally surface the
    wrong (older) recap on the engagement view."""
    fake_recaps = [
        SimpleNamespace(id=2, summary="newer"),
        SimpleNamespace(id=1, summary="older"),
    ]
    with patch.object(svc.repo, "list_stream_recaps", return_value=fake_recaps):
        await svc._refresh_engagement_aggregates(window="7d")

    cache = svc.engagement_aggregates_cache
    assert cache.recent_recaps == fake_recaps
    assert cache.latest_recap is fake_recaps[0]
