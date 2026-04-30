"""Streamer-customizable LLM prompts.

The dashboard's most personality-driven LLM call sites can be tuned
by the streamer through three modes:

  - **factory** (default): the carefully-tuned prompt that ships with
    chatterbot. Streamers who don't visit the Prompts settings tab
    see no behavior change.
  - **adlibs**: a small handful of structured questions ("What tone
    do you want?" / "What should be avoided?") whose answers slot
    into a pre-built template. Cheap personality dialing without
    rewriting prompt instructions.
  - **custom**: free-form full-text editor. Streamer rewrites the
    entire prompt. Fastest way to bend the system but also the
    easiest way to break it — the dashboard surfaces a "this prompt
    may be broken — revert?" affordance when the cache empties.

The registry below is the SINGLE source of truth for which call
sites are streamer-editable. Adding a new entry to `REGISTRY` makes
that call site appear in the Prompts settings tab automatically;
the call-site code switches from a hardcoded constant to
`resolve_prompt(call_site, repo)`.

Sites NOT in the registry stay hardcoded — those are correctness-
critical or mechanical (moderator classification, note extraction,
transcript-to-card matching). Touching them via the UI would
introduce subtle data corruption with no streamer-personality
benefit, so we deliberately skip them.

Storage shape in `app_settings`:

  prompts.<call_site>.mode      = "factory" | "adlibs" | "custom"
  prompts.<call_site>.adlibs    = JSON dict {slot_name: value}
  prompts.<call_site>.custom    = full prompt text

Empty / absent = factory mode. The streamer can revert any prompt
to factory at any time via a one-click button on its card.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..repo import ChatterRepo

logger = logging.getLogger(__name__)


# ---- registry types ----


@dataclass(frozen=True)
class AdlibSlot:
    """One adlib slot — a question the streamer answers, whose
    answer slots into the prompt's `adlib_template` at the
    `{name}` placeholder.

    `default` is rendered into the form as a placeholder + used
    when the streamer hasn't saved a value. It's also what the
    rendered prompt uses for an empty adlib config — so the
    factory experience and "adlibs with all defaults" are identical
    by construction.

    `multiline=True` → render as textarea (for free-text rules).
    `options` (when set) → render as select dropdown with these
    choices. `placeholder` is a hint shown in the input."""
    name: str
    question: str
    default: str
    placeholder: str = ""
    multiline: bool = False
    options: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PromptDef:
    """One streamer-editable prompt site. Carries the factory text,
    the adlib template (factory text with `{slot_name}` placeholders),
    and the metadata the settings UI needs to render its card."""
    call_site: str
    section: str               # "insights" | "channel" | "transcripts"
    title: str
    description: str           # plain-english what-this-does
    factory: str
    adlib_template: str
    adlib_slots: tuple[AdlibSlot, ...]


# ---- helpers used to compose adlib_template values ----


def _injection_block(label: str, content_placeholder: str) -> str:
    """Helper to format the streamer's adlib values as a small
    "STREAMER PREFERENCES" block. Hybrid approach: factory text
    stays canonical, the adlib values are appended/prepended as
    extra instructions rather than re-templating the whole prompt
    (which would force us to maintain a near-duplicate prompt body)."""
    return (
        f"\n\n=== STREAMER PREFERENCES — {label} ===\n"
        f"{content_placeholder}\n"
        "These preferences override the defaults above where they "
        "conflict; otherwise treat them as additional guidance.\n"
    )


# Stitched at bottom of file — registry assembled after the prompts
# import to avoid circular imports.

REGISTRY: dict[str, PromptDef] = {}


def _build_registry() -> None:
    """Lazy registry construction so the import order
    `prompts.py → insights.py → schemas.py → prompts.py`
    can't deadlock. Called the first time `resolve_prompt` runs."""
    global REGISTRY
    if REGISTRY:
        return
    from ..insights import InsightsService, TALKING_POINTS_SYSTEM
    from ..summarizer import TOPICS_SYSTEM
    from ..transcript import TranscriptService

    REGISTRY = {

        # =====================================================
        # CONVERSATION INSIGHTS
        # =====================================================

        "insights.talking_points": PromptDef(
            call_site="insights.talking_points",
            section="insights",
            title="Per-chatter talking points",
            description=(
                "Generates one short conversation hook per active "
                "chatter. The streamer reads these on a second "
                "monitor and uses them to engage with chat. Tuning "
                "this changes the TONE of the hooks (low-energy "
                "observational vs playful) and what topics they "
                "avoid."
            ),
            factory=TALKING_POINTS_SYSTEM,
            adlib_template=(
                TALKING_POINTS_SYSTEM
                + _injection_block(
                    "talking-points style",
                    "Tone — {tone}.\n"
                    "Things to avoid bringing up — {avoid}.\n"
                    "Hooks should still follow all the HARD RULES "
                    "above (grounded paraphrases only, no invented "
                    "facts).",
                )
            ),
            adlib_slots=(
                AdlibSlot(
                    name="tone",
                    question="How should the hooks sound?",
                    default="observational and low-energy — like a casual aside",
                    placeholder="e.g. playful and joking / dry sarcasm / warm and curious",
                ),
                AdlibSlot(
                    name="avoid",
                    question="Topics or phrases the hooks should AVOID",
                    default="(none)",
                    placeholder="e.g. anything political; references to drama; etc.",
                    multiline=True,
                ),
            ),
        ),

        "insights.engaging_subjects": PromptDef(
            call_site="insights.engaging_subjects",
            section="insights",
            title="Engaging subjects extraction",
            description=(
                "Identifies the distinct conversation subjects chat "
                "is currently discussing — what shows up in the "
                "Engaging subjects panel on the Insights page. "
                "Tuning this changes how SPECIFIC subject lines are "
                "and what extra topics the filter treats as off-limits."
            ),
            factory=InsightsService.SUBJECTS_SYSTEM,
            adlib_template=(
                InsightsService.SUBJECTS_SYSTEM
                + _injection_block(
                    "subject-extraction preferences",
                    "Subject specificity preference — {specificity}.\n"
                    "Additional topics to filter (beyond the default "
                    "religion/politics/controversies) — {extra_filter}.\n"
                    "Subjects must still cite at least 2 supporting "
                    "message_ids per the rules above.",
                )
            ),
            adlib_slots=(
                AdlibSlot(
                    name="specificity",
                    question="How specific should subject names be?",
                    default="very specific (4-8 word lines, e.g. 'NG4 parry timing vs NG2')",
                    options=(
                        "very specific (4-8 word lines, e.g. 'NG4 parry timing vs NG2')",
                        "moderately specific (5-10 words, slightly broader)",
                        "broad (themes / categories rather than specific runs)",
                    ),
                ),
                AdlibSlot(
                    name="extra_filter",
                    question="Topics to filter beyond the defaults",
                    default="(none — defaults are sufficient)",
                    placeholder="e.g. specific drama topics; competitor channel callouts; etc.",
                    multiline=True,
                ),
            ),
        ),

        "insights.subject_talking_points": PromptDef(
            call_site="insights.subject_talking_points",
            section="insights",
            title="Per-subject talking points",
            description=(
                "When the streamer opens an engaging-subject's modal, "
                "this prompt generates 3-5 short things they could "
                "say back to chat about it. Tuning this changes the "
                "VOICE — how the suggested lines are phrased."
            ),
            factory=InsightsService.SUBJECT_TALKING_POINTS_SYSTEM,
            adlib_template=(
                InsightsService.SUBJECT_TALKING_POINTS_SYSTEM
                + _injection_block(
                    "talking-point voice",
                    "Voice / phrasing style — {voice}.\n"
                    "Things to avoid bringing up — {avoid}.\n"
                    "Points must still paraphrase content visible in "
                    "the input — no invented facts.",
                )
            ),
            adlib_slots=(
                AdlibSlot(
                    name="voice",
                    question="How should the suggested lines sound?",
                    default=(
                        "first-person conversational — things the streamer "
                        "would naturally say while playing"
                    ),
                    placeholder=(
                        "e.g. confident hot-takes / curious questions back "
                        "to chat / dry observations"
                    ),
                ),
                AdlibSlot(
                    name="avoid",
                    question="Things the suggested lines should AVOID",
                    default="(none)",
                    placeholder="e.g. unrelated jokes; advice phrasing; etc.",
                    multiline=True,
                ),
            ),
        ),

        "insights.open_questions": PromptDef(
            call_site="insights.open_questions",
            section="insights",
            title="Open questions filter",
            description=(
                "Decides which chat questions are still 'open' "
                "(unanswered, asked of the room rather than a "
                "specific chatter, not rhetorical). Tuning changes "
                "how STRICT the filter is."
            ),
            factory=InsightsService.OPEN_QUESTIONS_SYSTEM,
            adlib_template=(
                InsightsService.OPEN_QUESTIONS_SYSTEM
                + _injection_block(
                    "open-questions filter strictness",
                    "Filter strictness — {strictness}.\n"
                    "Streamers who want the panel to surface MORE "
                    "questions should set this permissive; those who "
                    "want only the most clearly-open asks should set "
                    "it strict.",
                )
            ),
            adlib_slots=(
                AdlibSlot(
                    name="strictness",
                    question="How strict should the filter be?",
                    default="balanced — drop rhetorical and answered, keep ambiguous-but-likely-open",
                    options=(
                        "strict — only direct, unambiguous questions to the room",
                        "balanced — drop rhetorical and answered, keep ambiguous-but-likely-open",
                        "permissive — surface anything that could be a question worth answering",
                    ),
                ),
            ),
        ),

        "insights.thread_recaps": PromptDef(
            call_site="insights.thread_recaps",
            section="insights",
            title="Thread recaps",
            description=(
                "Writes one observational recap per active topic "
                "thread on the Live Conversations panel. Tuning this "
                "changes the TONE and LENGTH of each recap."
            ),
            factory=InsightsService.THREAD_RECAP_SYSTEM,
            adlib_template=(
                InsightsService.THREAD_RECAP_SYSTEM
                + _injection_block(
                    "recap tone + length",
                    "Recap tone — {tone}.\n"
                    "Recap length — {length}.\n"
                    "Recaps must still be observational paraphrase, "
                    "never advice or speculation.",
                )
            ),
            adlib_slots=(
                AdlibSlot(
                    name="tone",
                    question="How should the recaps sound?",
                    default="clinical and observational — what was said, nothing else",
                    placeholder=(
                        "e.g. warm and familiar / clinical / mildly playful"
                    ),
                ),
                AdlibSlot(
                    name="length",
                    question="How long should each recap be?",
                    default="1-2 sentences",
                    options=(
                        "1 sentence (terse)",
                        "1-2 sentences",
                        "2-3 sentences (more context)",
                    ),
                ),
            ),
        ),

        # =====================================================
        # CHANNEL CONTEXT
        # =====================================================

        "summarizer.topics_snapshot": PromptDef(
            call_site="summarizer.topics_snapshot",
            section="channel",
            title="Channel topic snapshots",
            description=(
                "Summarizes the channel-wide topics chat is touching "
                "on at a given moment. These feed the Topics view + "
                "are passed into other LLM prompts as context. "
                "Tuning changes how SPECIFIC the topic labels are."
            ),
            factory=TOPICS_SYSTEM,
            adlib_template=(
                TOPICS_SYSTEM
                + _injection_block(
                    "topic-snapshot specificity",
                    "Topic-label specificity — {specificity}.",
                )
            ),
            adlib_slots=(
                AdlibSlot(
                    name="specificity",
                    question="How specific should topic labels be?",
                    default="moderately specific (e.g. 'speedrun route discussion' rather than 'gaming')",
                    options=(
                        "broad (e.g. 'gaming', 'banter', 'meta')",
                        "moderately specific (e.g. 'speedrun route discussion')",
                        "very specific (e.g. 'NG4 stage 3 boss strategy comparison')",
                    ),
                ),
            ),
        ),

        # =====================================================
        # TRANSCRIPTS
        # =====================================================

        "transcript.group_summary": PromptDef(
            call_site="transcript.group_summary",
            section="transcripts",
            title="Streamer-voice group summaries",
            description=(
                "Summarizes ~60-second windows of streamer voice "
                "transcripts (with screenshot context) into one "
                "observational paragraph. Tuning changes the FOCUS — "
                "what the summary emphasises — and the tone."
            ),
            factory=TranscriptService.GROUP_SUMMARY_SYSTEM,
            adlib_template=(
                TranscriptService.GROUP_SUMMARY_SYSTEM
                + _injection_block(
                    "group-summary focus + tone",
                    "Focus — {focus}.\n"
                    "Tone — {tone}.\n"
                    "Summaries must still be observational and "
                    "grounded in the actual utterances + screenshot.",
                )
            ),
            adlib_slots=(
                AdlibSlot(
                    name="focus",
                    question="What should each summary emphasise?",
                    default="both what's happening on screen and what the streamer is talking about",
                    options=(
                        "what's happening on screen (gameplay, scene changes)",
                        "what the streamer is talking about (topic, mood)",
                        "both what's happening on screen and what the streamer is talking about",
                    ),
                ),
                AdlibSlot(
                    name="tone",
                    question="Tone preference",
                    default="neutral and descriptive",
                    placeholder="e.g. dry and clinical / warm and familiar / mildly playful",
                ),
            ),
        ),

    }


# ---- public surface ----


# Section ordering for the settings UI. Sections not in this list
# (if anyone adds one without updating here) sort to the end
# alphabetically.
SECTION_ORDER: tuple[str, ...] = ("insights", "channel", "transcripts")
SECTION_TITLES: dict[str, str] = {
    "insights": "Conversation insights",
    "channel": "Channel context",
    "transcripts": "Transcripts",
}


def all_prompt_defs() -> list[PromptDef]:
    """Return every editable prompt, sorted by section + title for
    stable UI rendering."""
    _build_registry()
    rank = {s: i for i, s in enumerate(SECTION_ORDER)}
    return sorted(
        REGISTRY.values(),
        key=lambda p: (rank.get(p.section, 999), p.title),
    )


def get_prompt_def(call_site: str) -> PromptDef | None:
    """Look up the registry entry for a call site. Returns None
    when the call site isn't streamer-editable (i.e., correctness-
    critical or mechanical). Callers handle None as "fall back to
    the hardcoded constant"."""
    _build_registry()
    return REGISTRY.get(call_site)


# ---- mode + saved value lookups ----


def _setting_keys(call_site: str) -> tuple[str, str, str]:
    """Stable app_settings key triple for one call site."""
    return (
        f"prompts.{call_site}.mode",
        f"prompts.{call_site}.adlibs",
        f"prompts.{call_site}.custom",
    )


VALID_MODES: frozenset[str] = frozenset({"factory", "adlibs", "custom"})


def get_mode(call_site: str, repo: "ChatterRepo") -> str:
    """Current mode for a call site. Returns "factory" when no
    setting is stored OR when the stored value is unrecognised
    (defensive — tampering / typo'd app_settings shouldn't break
    the LLM call)."""
    mode_key, _, _ = _setting_keys(call_site)
    raw = (repo.get_app_setting(mode_key) or "").lower().strip()
    return raw if raw in VALID_MODES else "factory"


def get_adlib_values(call_site: str, repo: "ChatterRepo") -> dict[str, str]:
    """Streamer-saved adlib values for this call site. Missing
    slots fall back to their defaults at render time, so an
    incomplete dict is fine."""
    _, adlib_key, _ = _setting_keys(call_site)
    raw = repo.get_app_setting(adlib_key) or ""
    if not raw:
        return {}
    try:
        v = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(v, dict):
        return {}
    # Coerce all values to str — JSON might've stored non-str if
    # someone hand-edited app_settings.
    return {str(k): str(val) for k, val in v.items()}


def get_custom_text(call_site: str, repo: "ChatterRepo") -> str:
    """Streamer-saved custom prompt text. Empty when none set."""
    _, _, custom_key = _setting_keys(call_site)
    return repo.get_app_setting(custom_key) or ""


# ---- the resolver — what production code actually calls ----


def resolve_prompt(call_site: str, repo: "ChatterRepo") -> str:
    """Return the effective system prompt for a call site, honouring
    the streamer's mode + saved values.

    Defensive: any error path (unknown call site, malformed
    app_settings, template format failure) falls back to the
    factory prompt rather than serving an empty / broken prompt to
    the LLM. Logs at warning so a misconfigured prompt doesn't
    silently produce garbage output.

    Production replacement pattern:

        # Before:
        system_prompt=self.SUBJECTS_SYSTEM,

        # After:
        system_prompt=resolve_prompt("insights.engaging_subjects", self.repo),
    """
    pd = get_prompt_def(call_site)
    if pd is None:
        # Not streamer-editable; caller should keep using the
        # hardcoded constant. Return empty string so a misuse
        # surfaces loudly (LLM with empty system prompt produces
        # obviously-bad output).
        logger.warning(
            "prompts: resolve_prompt called for non-editable site %r — "
            "caller should use the hardcoded constant", call_site,
        )
        return ""

    mode = get_mode(call_site, repo)
    if mode == "factory":
        return pd.factory

    if mode == "adlibs":
        saved = get_adlib_values(call_site, repo)
        slot_values = {
            slot.name: (saved.get(slot.name) or slot.default)
            for slot in pd.adlib_slots
        }
        # Targeted replace rather than `str.format()` — factory
        # prompts can contain literal `{` characters (e.g. the
        # JSON examples in the engaging-subjects few-shot block)
        # and `format()` would try to interpret those as
        # placeholders. We only substitute the well-known
        # `{slot_name}` markers our injection block defines.
        rendered = pd.adlib_template
        for name, value in slot_values.items():
            rendered = rendered.replace("{" + name + "}", value)
        return rendered

    if mode == "custom":
        custom = get_custom_text(call_site, repo).strip()
        if not custom:
            # Streamer in custom mode but never saved any text.
            # Treat as factory rather than serving an empty prompt.
            return pd.factory
        return custom

    # Defensive — get_mode() already filters to VALID_MODES, but
    # belt-and-suspenders.
    return pd.factory


# ---- save helpers (used by /settings/prompts routes) ----


def save_mode(call_site: str, mode: str, repo: "ChatterRepo") -> bool:
    """Set a call site's mode. Returns False when `mode` is invalid
    or `call_site` isn't editable; caller should surface a flash
    error in that case."""
    if mode not in VALID_MODES:
        return False
    if get_prompt_def(call_site) is None:
        return False
    mode_key, _, _ = _setting_keys(call_site)
    repo.set_app_setting(mode_key, mode)
    return True


def save_adlib_values(
    call_site: str, values: dict[str, str], repo: "ChatterRepo",
) -> bool:
    """Persist the streamer's adlib values as JSON. Filters to only
    the slot names the prompt actually defines, so a stale form
    submission with extra keys can't pollute storage."""
    pd = get_prompt_def(call_site)
    if pd is None:
        return False
    valid_names = {s.name for s in pd.adlib_slots}
    filtered = {
        k: str(v).strip() for k, v in values.items()
        if k in valid_names and v is not None
    }
    _, adlib_key, _ = _setting_keys(call_site)
    repo.set_app_setting(adlib_key, json.dumps(filtered, ensure_ascii=False))
    return True


def save_custom_text(call_site: str, text: str, repo: "ChatterRepo") -> bool:
    """Persist the streamer's full-custom prompt text."""
    if get_prompt_def(call_site) is None:
        return False
    _, _, custom_key = _setting_keys(call_site)
    repo.set_app_setting(custom_key, text)
    return True


def revert_to_factory(call_site: str, repo: "ChatterRepo") -> bool:
    """Wipe every streamer override for one call site. Resets mode
    to factory + clears adlib + custom values. Idempotent — safe
    to call when no overrides exist."""
    if get_prompt_def(call_site) is None:
        return False
    for key in _setting_keys(call_site):
        repo.delete_app_setting(key)
    return True
