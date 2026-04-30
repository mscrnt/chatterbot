"""Browser-facing routes for the engaging-subject modal.

Slice-8 added `/modals/subject/{slug}` (renders the modal body) and
`/insights/subject/{slug}/talking-points` (async-loaded points
partial). These tests use TestClient against the real ASGI app —
no HTTP mocking — and pre-populate `insights.subjects_cache` so
the routes have a subject to render.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chatterbot.config import Settings
from chatterbot.insights import (
    EngagingSubjectEntry,
    EngagingSubjectsCache,
    _PersistentSubject,
)
from chatterbot.repo import ChatterRepo


# ---- shared app fixture ----


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch):
    """Real FastAPI app + sqlite DB. The LLM provider is stubbed
    so create_app boots without a live Ollama. Returns (TestClient,
    repo, app) so tests can reach into the captured InsightsService
    via app.state if needed.

    NOTE: `insights` is constructed inside `create_app` and held by
    closure; we expose it via a tiny `_get_insights` helper that
    walks the FastAPI app's route table to find the closure."""
    db_path = tmp_path / "chatters.db"
    repo = ChatterRepo(str(db_path), embed_dim=768)

    class _StubLLM:
        def __init__(self):
            self._dataset_repo = None
            self.model = "stub"
            self.embed_model = "stub-embed"

        def attach_dataset_capture(self, repo):
            self._dataset_repo = repo

        async def health_check(self):
            return True

        async def embed(self, text):
            return [0.0] * 768

    stub_llm = _StubLLM()
    from chatterbot.llm import providers as _providers
    monkeypatch.setattr(
        _providers, "make_llm_client", lambda settings: stub_llm,
    )

    from chatterbot.web.app import create_app
    settings = Settings(_env_file=None)
    settings.db_path = str(db_path)
    app = create_app(repo, settings)
    client = TestClient(app)
    try:
        yield client, repo, app
    finally:
        client.close()
        repo.close()


def _seed_cached_subject(
    app, repo,
    *, slug: str = "abc123def456", name: str = "RE9 parry timing",
) -> tuple[str, list[int]]:
    """Drop a hand-built subject + the supporting messages into the
    insights cache so the modal route has something to render.
    Returns (slug, msg_ids) so tests can verify hydration."""
    # Pluck the InsightsService out of the app's state. The
    # web/app.py builds it inside create_app and registers
    # closures into the FastAPI app — we look it up via the
    # `_insights_service` reference create_app stashes on
    # `app.state` (added in slice-8 for testability).
    repo.upsert_user(twitch_id="u_alice", name="alice")
    msg_ids = []
    for c in ("parry timing 1", "parry timing 2", "parry timing 3"):
        mid = repo.insert_message(user_id="u_alice", content=c)
        msg_ids.append(mid)

    insights = app.state.insights_service
    insights._subjects_cache = EngagingSubjectsCache(
        subjects=[EngagingSubjectEntry(
            name=name, drivers=["alice"], msg_count=len(msg_ids),
            brief="Discussing parry timing.",
            angles=["windows", "izuna drops"],
            slug=slug, msg_ids=msg_ids,
        )],
        refreshed_at=0.0,
    )
    # Also drop a matching persistent subject so the talking-points
    # generator can find it via slug → name → sha1 lookup.
    insights._subjects[name] = _PersistentSubject(
        subject_id="x" * 12, name=name, name_embedding=[1.0] + [0.0] * 767,
        brief="Discussing parry timing.",
        angles=["windows", "izuna drops"],
        msg_ids=msg_ids,
        last_seen_ts=1.0,
    )
    return slug, msg_ids


# ---- /modals/subject/{slug} ----


def test_subject_modal_renders_full_body(app_client):
    client, repo, app = app_client
    slug, msg_ids = _seed_cached_subject(app, repo)

    r = client.get(f"/modals/subject/{slug}")
    assert r.status_code == 200
    body = r.text
    # Modal-shell pieces present (header + close button).
    assert "RE9 parry timing" in body
    assert "fa-fire" in body
    # Content: brief, angles, drivers, chat context, talking points placeholder.
    assert "Discussing parry timing." in body
    assert "izuna drops" in body
    # The cited messages were hydrated into the chat-context section.
    for c in ("parry timing 1", "parry timing 2", "parry timing 3"):
        assert c in body
    # Talking-points placeholder uses HTMX to lazy-load on intersect.
    assert f"/insights/subject/{slug}/talking-points" in body
    # Modal-side dismiss button targets the card row to swap it out.
    assert f"#subject-row-{slug}" in body


def test_subject_modal_404_for_unknown_slug(app_client):
    client, _, _ = app_client
    r = client.get("/modals/subject/nope")
    assert r.status_code == 404


def test_subject_modal_renders_without_persistent_subject(app_client):
    """Cache and persistent subjects can drift in rare race
    conditions (cache built from one refresh, persistent dict
    aged out before the streamer clicks). Modal body must still
    render — the talking-points partial will report the error
    when it lazy-loads, but the rest of the modal works."""
    client, repo, app = app_client
    repo.upsert_user(twitch_id="u_alice", name="alice")
    mid = repo.insert_message(user_id="u_alice", content="orphan msg")

    insights = app.state.insights_service
    insights._subjects_cache = EngagingSubjectsCache(
        subjects=[EngagingSubjectEntry(
            name="Orphan subject", drivers=["alice"], msg_count=1,
            brief="…", angles=[], slug="orph12345678", msg_ids=[mid],
        )],
        refreshed_at=0.0,
    )
    # Note: NO persistent subject seeded — the cache + persistent
    # are deliberately out of sync.

    r = client.get("/modals/subject/orph12345678")
    assert r.status_code == 200
    assert "Orphan subject" in r.text
    assert "orphan msg" in r.text


# ---- /insights/subject/{slug}/talking-points ----


def test_talking_points_route_renders_points_partial(app_client, monkeypatch):
    client, repo, app = app_client
    slug, _ = _seed_cached_subject(app, repo)

    # Stub the generator so we don't need a real LLM. The
    # talking-points route just calls into insights and renders
    # the partial — substituting the generator is the simplest
    # way to test the route shape end-to-end.
    insights = app.state.insights_service

    async def _fake(slug, *, force=False):
        return ["You could mention X", "Worth comparing Y"], None

    monkeypatch.setattr(insights, "generate_subject_talking_points", _fake)

    r = client.get(f"/insights/subject/{slug}/talking-points")
    assert r.status_code == 200
    body = r.text
    assert "You could mention X" in body
    assert "Worth comparing Y" in body
    # The "talking points" header pin so the modal's lazy-load
    # replacement preserves the section heading.
    assert "talking points" in body.lower()


def test_talking_points_route_renders_error_inline(app_client, monkeypatch):
    """When the generator returns `(empty, error)`, the route
    template renders the error as a small inline warning rather
    than showing nothing. Pin so a streamer always sees WHY the
    section is empty."""
    client, repo, app = app_client
    slug, _ = _seed_cached_subject(app, repo)
    insights = app.state.insights_service

    async def _fail(slug, *, force=False):
        return [], "the model response did not validate"

    monkeypatch.setattr(insights, "generate_subject_talking_points", _fail)

    r = client.get(f"/insights/subject/{slug}/talking-points")
    assert r.status_code == 200
    # The error message renders inline (Jinja autoescapes
    # apostrophes to entities, so we match a substring without).
    assert "did not validate" in r.text


def test_talking_points_route_swallows_internal_errors(app_client, monkeypatch):
    """If the generator raises (programmer error / repo glitch),
    the route catches it, logs, and still returns 200 with an
    "internal error" partial. The modal's UX must never break
    just because one section fails."""
    client, repo, app = app_client
    slug, _ = _seed_cached_subject(app, repo)
    insights = app.state.insights_service

    async def _boom(slug, *, force=False):
        raise RuntimeError("simulated explosion")

    monkeypatch.setattr(insights, "generate_subject_talking_points", _boom)

    r = client.get(f"/insights/subject/{slug}/talking-points")
    assert r.status_code == 200
    assert "internal error" in r.text.lower()
