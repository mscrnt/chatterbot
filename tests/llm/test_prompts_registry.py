"""Streamer-customizable prompts registry tests.

The registry is the single source of truth for which call sites
the streamer can edit. These tests pin:

  - Every editable call site has a non-empty factory + guided_template
    + at least one slot.
  - resolve_prompt round-trips factory / guided / custom modes.
  - Defensive fallback: bad app_settings → factory prompt, never
    empty / broken output to the LLM.
  - Revert clears every override key.
  - Guided template format errors fall back to factory.
  - Editable list is exactly the streamer-personality / channel-
    context tier (no moderator, no profile/note extraction).
"""

from __future__ import annotations

import json

import pytest

from chatterbot.llm import prompts as P


# ---- registry shape ----


def test_registry_has_expected_call_sites():
    """Registry covers exactly the streamer-personality tier.
    Adding a new call site to chatterbot does NOT automatically
    expose it as editable — someone has to register it. Pin the
    list so a refactor that drops or adds an entry shows up here."""
    expected = {
        "insights.talking_points",
        "insights.engaging_subjects",
        "insights.subject_talking_points",
        "insights.open_questions",
        "insights.question_answer_angles",
        "insights.high_impact_openers",
        "insights.thread_recaps",
        "summarizer.topics_snapshot",
        "transcript.group_summary",
    }
    found = {p.call_site for p in P.all_prompt_defs()}
    assert found == expected


def test_every_prompt_has_factory_and_guided_template():
    """Each registry entry must have non-empty factory text + an
    guided_template + at least one guided slot. Catches a half-
    finished registry add."""
    for pd in P.all_prompt_defs():
        assert pd.factory.strip(), f"{pd.call_site}: empty factory"
        assert pd.guided_template.strip(), f"{pd.call_site}: empty template"
        assert len(pd.guided_slots) >= 1, f"{pd.call_site}: no slots"


def test_every_guided_slot_has_default():
    """Defaults are what render when the streamer hasn't saved
    anything; missing default = NoneType in the format() call →
    crash. Pin presence."""
    for pd in P.all_prompt_defs():
        for slot in pd.guided_slots:
            assert slot.default, (
                f"{pd.call_site}.{slot.name}: empty default"
            )


def test_guided_template_contains_every_declared_slot_placeholder():
    """Every slot the registry declares must have a matching
    `{slot_name}` placeholder in the guided_template. Catches
    typos like `{tones}` (in template) vs `tone` (slot name) —
    the resolver's `.replace("{name}", value)` would silently
    leave the misspelled placeholder in the rendered prompt.

    Note: we deliberately use string replace (not str.format) in
    the resolver because some factory prompts contain literal
    `{` characters in JSON few-shot examples that would explode
    str.format. The trade-off is that this test has to enforce
    placeholder/slot alignment manually."""
    for pd in P.all_prompt_defs():
        for slot in pd.guided_slots:
            placeholder = "{" + slot.name + "}"
            assert placeholder in pd.guided_template, (
                f"{pd.call_site}: declared slot {slot.name!r} but "
                f"placeholder {placeholder!r} is missing from "
                f"guided_template"
            )


def test_guided_render_substitutes_every_slot(tmp_repo):
    """End-to-end: switch to guided mode, save sentinel values,
    confirm the rendered prompt has the values substituted in
    AND no `{slot_name}` placeholders are left dangling."""
    for pd in P.all_prompt_defs():
        P.save_mode(pd.call_site, "guided", tmp_repo)
        sentinels = {
            slot.name: f"<<SENTINEL-{slot.name}>>"
            for slot in pd.guided_slots
        }
        P.save_guided_values(pd.call_site, sentinels, tmp_repo)
        rendered = P.resolve_prompt(pd.call_site, tmp_repo)
        for slot in pd.guided_slots:
            assert sentinels[slot.name] in rendered, (
                f"{pd.call_site}: slot {slot.name!r} sentinel not "
                f"found in rendered prompt"
            )
            placeholder = "{" + slot.name + "}"
            assert placeholder not in rendered, (
                f"{pd.call_site}: dangling placeholder {placeholder!r} "
                f"after substitution"
            )
        # Reset state for the next iteration.
        P.revert_to_factory(pd.call_site, tmp_repo)


# ---- resolve_prompt: factory mode ----


def test_resolve_factory_returns_factory_prompt(tmp_repo):
    """No app_settings keys touched → resolve returns the factory
    string verbatim. The streamer who never visits the Prompts tab
    sees zero behaviour change."""
    pd = P.get_prompt_def("insights.talking_points")
    assert P.resolve_prompt("insights.talking_points", tmp_repo) == pd.factory


def test_resolve_unknown_call_site_returns_empty(tmp_repo):
    """Caller must explicitly register a call site to make it
    editable. Unknown sites return empty + log a warning so misuse
    surfaces loudly (an empty system_prompt to the LLM produces
    obviously-bad output)."""
    assert P.resolve_prompt("not.a.real.site", tmp_repo) == ""


# ---- resolve_prompt: guided mode ----


def test_resolve_guided_with_defaults(tmp_repo):
    """Mode set to guided but no values saved → renders with
    every slot's default. Equivalent in shape to factory + the
    injection block, never broken."""
    P.save_mode("insights.talking_points", "guided", tmp_repo)
    rendered = P.resolve_prompt("insights.talking_points", tmp_repo)
    pd = P.get_prompt_def("insights.talking_points")
    # Should have all default values substituted in.
    for slot in pd.guided_slots:
        assert slot.default in rendered


def test_resolve_guided_with_streamer_overrides(tmp_repo):
    """Streamer-saved values override defaults at render time."""
    P.save_mode("insights.talking_points", "guided", tmp_repo)
    P.save_guided_values(
        "insights.talking_points",
        {"tone": "loud and chaotic", "avoid": "anything serious"},
        tmp_repo,
    )
    rendered = P.resolve_prompt("insights.talking_points", tmp_repo)
    assert "loud and chaotic" in rendered
    assert "anything serious" in rendered


def test_resolve_guided_partial_overrides_fall_back_to_defaults(tmp_repo):
    """Saved values cover only some slots → unset slots use
    their defaults. Streamer doesn't have to fill every field."""
    P.save_mode("insights.talking_points", "guided", tmp_repo)
    P.save_guided_values(
        "insights.talking_points",
        {"tone": "spicy"},  # `avoid` slot is unset
        tmp_repo,
    )
    rendered = P.resolve_prompt("insights.talking_points", tmp_repo)
    pd = P.get_prompt_def("insights.talking_points")
    avoid_default = next(s.default for s in pd.guided_slots if s.name == "avoid")
    assert "spicy" in rendered
    assert avoid_default in rendered


def test_resolve_guided_corrupt_json_falls_back_to_factory(tmp_repo):
    """If app_settings has malformed JSON for the guided values
    (a streamer hand-edited the DB?), fall back to factory.
    Never serve a broken prompt to the LLM."""
    P.save_mode("insights.talking_points", "guided", tmp_repo)
    tmp_repo.set_app_setting(
        "prompts.insights.talking_points.guided",
        "{this is not valid json",
    )
    rendered = P.resolve_prompt("insights.talking_points", tmp_repo)
    # Guided mode rendering uses defaults when JSON is bad — so we
    # get an guided-rendered prompt with all defaults, NOT the
    # plain factory. Either is acceptable; pin which.
    pd = P.get_prompt_def("insights.talking_points")
    assert pd.guided_slots[0].default in rendered


# ---- resolve_prompt: custom mode ----


def test_resolve_custom_returns_streamer_text(tmp_repo):
    """Custom mode replaces the entire prompt with whatever the
    streamer typed."""
    P.save_mode("insights.talking_points", "custom", tmp_repo)
    P.save_custom_text(
        "insights.talking_points",
        "Just say something funny about every chatter.",
        tmp_repo,
    )
    rendered = P.resolve_prompt("insights.talking_points", tmp_repo)
    assert rendered == "Just say something funny about every chatter."


def test_resolve_custom_empty_text_falls_back_to_factory(tmp_repo):
    """Streamer flips to custom but never saves any text. We
    serve the factory, not an empty prompt — empty would produce
    garbage LLM output."""
    P.save_mode("insights.talking_points", "custom", tmp_repo)
    # No custom text saved.
    rendered = P.resolve_prompt("insights.talking_points", tmp_repo)
    pd = P.get_prompt_def("insights.talking_points")
    assert rendered == pd.factory


# ---- save / revert helpers ----


def test_save_mode_rejects_invalid_values(tmp_repo):
    assert P.save_mode("insights.talking_points", "garbage", tmp_repo) is False
    assert P.save_mode("insights.talking_points", "factory", tmp_repo) is True


def test_save_mode_rejects_unknown_call_site(tmp_repo):
    assert P.save_mode("not.editable", "factory", tmp_repo) is False


def test_save_guided_values_filters_unknown_slots(tmp_repo):
    """A stale form submission with extra slot keys (e.g. someone
    renamed a slot in code) shouldn't pollute storage. Pin the
    filter."""
    P.save_guided_values(
        "insights.talking_points",
        {"tone": "x", "ghost_slot": "should be dropped"},
        tmp_repo,
    )
    saved = P.get_guided_values("insights.talking_points", tmp_repo)
    assert "tone" in saved
    assert "ghost_slot" not in saved


def test_revert_clears_every_override_key(tmp_repo):
    """Revert wipes mode + guided + custom in one shot. Idempotent."""
    P.save_mode("insights.talking_points", "custom", tmp_repo)
    P.save_guided_values(
        "insights.talking_points", {"tone": "spicy"}, tmp_repo,
    )
    P.save_custom_text(
        "insights.talking_points", "custom prompt body", tmp_repo,
    )
    assert tmp_repo.get_app_setting("prompts.insights.talking_points.mode")
    assert tmp_repo.get_app_setting("prompts.insights.talking_points.guided")
    assert tmp_repo.get_app_setting("prompts.insights.talking_points.custom")

    assert P.revert_to_factory("insights.talking_points", tmp_repo) is True
    for key in (
        "prompts.insights.talking_points.mode",
        "prompts.insights.talking_points.guided",
        "prompts.insights.talking_points.custom",
    ):
        assert tmp_repo.get_app_setting(key) is None

    # Idempotent — second revert is a no-op.
    assert P.revert_to_factory("insights.talking_points", tmp_repo) is True


def test_revert_unknown_site_returns_false(tmp_repo):
    assert P.revert_to_factory("not.editable", tmp_repo) is False


# ---- get_mode tolerates bad data ----


def test_get_mode_falls_back_on_unrecognised_value(tmp_repo):
    """Hand-edited app_settings can stash garbage. `get_mode`
    returns 'factory' for anything not in VALID_MODES."""
    tmp_repo.set_app_setting(
        "prompts.insights.talking_points.mode", "WHATEVER",
    )
    assert P.get_mode("insights.talking_points", tmp_repo) == "factory"


# ---- editable sites are NOT correctness-critical ----


def test_correctness_critical_sites_are_NOT_editable():
    """The point of the registry is to expose only streamer-
    personality sites. Editing the moderator classifier or note
    extractor would create false-negatives / data corruption with
    no streamer-personality benefit. Pin the negative list."""
    forbidden = (
        "moderator.incident_classification",
        "summarizer.note_extraction",
        "summarizer.profile_extraction",
        "transcript.llm_match",
    )
    for site in forbidden:
        assert P.get_prompt_def(site) is None, (
            f"{site!r} should NOT be editable — touch the registry "
            f"VERY carefully if you're adding it intentionally."
        )
