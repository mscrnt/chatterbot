"""recent_questions clustering + filter tests.

The repo helper that backs the "Questions in chat" panel. Two
distinct things to verify:

  1. SQL-level filters (clean_msg_where, @-prefix exclusion,
     reply_parent_login exclusion, lookback window).
  2. Token-overlap clustering (Szymkiewicz-Simpson coefficient ≥ 0.5
     merges similar questions; below threshold splits them).

Each test seeds messages, calls `recent_questions(within_minutes=…)`,
and asserts on the cluster shape.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


def _ts(minutes_ago: int) -> str:
    """Produce an ISO-UTC timestamp `minutes_ago` minutes before now,
    matching SQLite's default datetime() format."""
    dt = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ---- SQL-level filtering ----


def test_drops_messages_without_question_mark(tmp_repo, make_message):
    """Only messages containing `?` are candidates."""
    make_message(content="just commenting here", ts=_ts(2))
    make_message(content="anyone got a question?", ts=_ts(2))
    qs = tmp_repo.recent_questions(within_minutes=15)
    assert len(qs) == 1
    assert "question" in qs[0]["question"]


def test_excludes_at_prefixed_messages(tmp_repo, make_message):
    """`@bob you good?` is directed at a specific chatter — the
    `Talking to you` panel handles those. recent_questions must NOT
    surface them."""
    make_message(content="@bob you good?", name="alice", ts=_ts(2))
    make_message(content="how do I unlock the boss?", name="alice", ts=_ts(2))
    qs = tmp_repo.recent_questions(within_minutes=15)
    assert len(qs) == 1
    assert qs[0]["question"].startswith("how do I unlock")


def test_excludes_reply_parent_login(tmp_repo, make_message):
    """Twitch's native reply feature sets reply_parent_login. Even
    without an `@` prefix, a reply IS directed at someone — drop it."""
    make_message(content="that route works?", reply_parent_login="bob", ts=_ts(2))
    make_message(content="what's the WR time?", ts=_ts(2))
    qs = tmp_repo.recent_questions(within_minutes=15)
    assert len(qs) == 1
    assert "WR" in qs[0]["question"]


def test_lookback_window_filters_out_old_messages(tmp_repo, make_message):
    """A message older than `within_minutes` must not appear. Catches
    a bug where the SQL forgot to bind the time bound."""
    make_message(content="old question?", ts=_ts(60))
    make_message(content="fresh question?", ts=_ts(2))
    qs = tmp_repo.recent_questions(within_minutes=15)
    assert len(qs) == 1
    assert qs[0]["question"] == "fresh question?"


def test_min_chars_filter(tmp_repo, make_message):
    """A bare `?` or `??` is too short to be a real question — the
    `min_chars` filter drops them."""
    make_message(content="?", ts=_ts(2))
    make_message(content="??", ts=_ts(2))
    make_message(content="how does parry work?", ts=_ts(2))
    qs = tmp_repo.recent_questions(within_minutes=15, min_chars=8)
    assert len(qs) == 1


# ---- token-overlap clustering ----


def test_similar_questions_cluster_together(tmp_repo, make_message):
    """`whats a good route` and `whats the route` share enough
    tokens to merge under Szymkiewicz-Simpson ≥ 0.5. Two askers,
    one cluster."""
    make_message(content="whats a good route?", name="alice", ts=_ts(2))
    make_message(content="whats the route?", name="bob", ts=_ts(1))
    qs = tmp_repo.recent_questions(within_minutes=15)
    assert len(qs) == 1
    cluster = qs[0]
    assert cluster["count"] == 2
    asker_names = {d["name"] for d in cluster["drivers"]}
    assert asker_names == {"alice", "bob"}


def test_distinct_questions_stay_separate(tmp_repo, make_message):
    """Genuinely different questions don't share enough tokens — must
    NOT merge."""
    make_message(content="how do you parry the boss?", name="alice", ts=_ts(2))
    make_message(content="when does the new patch drop?", name="bob", ts=_ts(2))
    qs = tmp_repo.recent_questions(within_minutes=15)
    assert len(qs) == 2


def test_same_chatter_asking_twice_doesnt_inflate_count(tmp_repo, make_message):
    """One chatter spamming the same question 3x must show as count=1
    in their cluster. Dedup is by user_id."""
    uid = "u_alice_static"
    make_message(user_id=uid, name="alice", content="whats a good route?", ts=_ts(3))
    make_message(user_id=uid, name="alice", content="whats the good route?", ts=_ts(2))
    make_message(user_id=uid, name="alice", content="good route anyone?", ts=_ts(1))
    qs = tmp_repo.recent_questions(within_minutes=15)
    assert len(qs) == 1
    assert qs[0]["count"] == 1


def test_results_sorted_by_count_desc(tmp_repo, make_message):
    """Ties broken on most-recent timestamp. Without tie-break the
    UI would order spuriously, confusing the streamer."""
    # Cluster A: one asker
    make_message(content="how does loot scale?", name="alice", ts=_ts(2))
    # Cluster B: two askers
    make_message(content="when is the marathon?", name="bob", ts=_ts(3))
    make_message(content="when does the marathon start?", name="carol", ts=_ts(2))
    qs = tmp_repo.recent_questions(within_minutes=15)
    assert len(qs) == 2
    # Cluster B (count=2) must come first.
    assert qs[0]["count"] == 2
    assert qs[1]["count"] == 1


# ---- driver shape ----


def test_cluster_drivers_carry_user_id_name_ts(tmp_repo, make_message):
    """Template renders pills linking to /users/<user_id>; missing
    fields would surface as bad anchor hrefs."""
    uid = "u_alice_x"
    make_message(user_id=uid, name="alice", content="whats up?", ts=_ts(2))
    qs = tmp_repo.recent_questions(within_minutes=15)
    drv = qs[0]["drivers"][0]
    assert drv["name"] == "alice"
    assert drv["user_id"] == uid
    assert drv["ts"]  # non-empty string


def test_cluster_includes_last_msg_id(tmp_repo, make_message):
    """The dismiss/snooze/addressed actions key on `last_msg_id` —
    must be the integer id of the most recent message in the
    cluster."""
    mid_old = make_message(content="any tips?", ts=_ts(5))
    mid_new = make_message(content="got any tips for this?", ts=_ts(1))
    qs = tmp_repo.recent_questions(within_minutes=15)
    assert len(qs) == 1
    # `last_msg_id` is the seed message we walked first (newest-first
    # iteration), which means the more recent insert.
    assert qs[0]["last_msg_id"] == mid_new
