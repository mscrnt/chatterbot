"""Coverage matrix — every `generate_structured(...)` call site in
production passes the expected `call_site=` string.

The capture system is only useful if the dataset can be sliced by
call site (e.g. "show me every prompt that went to
`insights.open_questions` last month"). A regression where someone
removes or typos `call_site=...` on one of these call sites would
silently dump those events into the `unknown` bucket and we'd lose
the ability to filter.

Strategy: scan the production source for every `generate_structured`
call and assert each one has a `call_site=` literal. This is a
static check, not a runtime call — fast and exhaustive.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterator

import pytest


# Where to scan. Anything inside this prefix that calls
# `generate_structured` MUST pass `call_site=`.
_PROD_ROOT = Path(__file__).parent.parent.parent / "src" / "chatterbot"

# Files we DON'T scan: the LLM client modules themselves (they DEFINE
# generate_structured rather than calling it from a production
# context), and the dataset module which only has docstring mentions.
_EXCLUDED_DIRS = {"llm", "dataset"}


def _iter_prod_py_files() -> Iterator[Path]:
    for p in _PROD_ROOT.rglob("*.py"):
        if any(part in _EXCLUDED_DIRS for part in p.relative_to(_PROD_ROOT).parts):
            continue
        yield p


# Extract every (file, lineno, call_site_value_or_None) for each
# generate_structured call. Single AST walk per file.
def _collect_call_sites() -> list[tuple[Path, int, str | None]]:
    out: list[tuple[Path, int, str | None]] = []
    for py in _iter_prod_py_files():
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Match `*.generate_structured(...)` — attribute call only.
            if not (isinstance(func, ast.Attribute)
                    and func.attr == "generate_structured"):
                continue
            cs = None
            for kw in node.keywords:
                if kw.arg == "call_site" and isinstance(kw.value, ast.Constant):
                    cs = kw.value.value
                    break
            out.append((py, node.lineno, cs))
    return out


# Fixed registry of expected call sites. New call sites must be added
# here so the test catches typos AND ensures we update test coverage
# when a new prompt site lands.
EXPECTED_CALL_SITES = {
    "summarizer.note_extraction",
    "summarizer.profile_extraction",
    "summarizer.topics_snapshot",
    "moderator.incident_classification",
    "insights.talking_points",
    "insights.thread_recaps",
    "insights.engaging_subjects",
    "insights.open_questions",
    "insights.question_answer_angles",
    "insights.subject_talking_points",
    "transcript.llm_match",
    "transcript.group_summary",
}


def test_every_production_call_site_passes_call_site_kwarg():
    """Static guard: every `generate_structured(...)` invocation in
    production code passes a literal `call_site=`. Catches a refactor
    that drops the kwarg on one site (which would silently dump that
    site's events into the `unknown` bucket)."""
    sites = _collect_call_sites()
    assert sites, "no generate_structured call sites found — scan path wrong?"
    missing = [(p, line) for (p, line, cs) in sites if cs is None]
    assert not missing, (
        "these generate_structured calls don't pass call_site=:\n"
        + "\n".join(f"  {p.relative_to(_PROD_ROOT)}:{line}" for p, line in missing)
    )


def test_call_sites_match_expected_registry():
    """The set of call_site values seen in production must EQUAL the
    expected registry above. Catches:
      - new call site added without updating EXPECTED_CALL_SITES
        (a hint that the test author should also add a mock-driven
        unit test for the new prompt)
      - typo'd call_site (e.g. "insights.open_question" missing the s)
      - call site removed without cleaning EXPECTED_CALL_SITES."""
    sites = _collect_call_sites()
    seen = {cs for (_, _, cs) in sites if cs is not None}

    extra = seen - EXPECTED_CALL_SITES
    missing = EXPECTED_CALL_SITES - seen
    assert not extra, (
        "production has call_site values not in EXPECTED_CALL_SITES — "
        f"add them: {sorted(extra)}"
    )
    assert not missing, (
        "EXPECTED_CALL_SITES references call_site values not in production "
        f"— remove or rename: {sorted(missing)}"
    )


def test_call_site_naming_convention():
    """Format is `<module>.<purpose>` — one dot, lowercase, no
    spaces. Pinning the convention so future additions stay
    consistent (otherwise the dashboard's planned per-site filter
    UI would have to special-case oddballs)."""
    pattern = re.compile(r"^[a-z][a-z_]*\.[a-z][a-z_]*$")
    for cs in EXPECTED_CALL_SITES:
        assert pattern.match(cs), (
            f"call_site {cs!r} doesn't match <module>.<purpose>: "
            "lowercase letters and underscores only, exactly one dot"
        )
