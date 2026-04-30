# Prompts (Settings → Prompts)

The dashboard ships with carefully-tuned LLM prompts for every panel.
Streamers can customize them through the Prompts settings tab, with
three modes per prompt:

- **Factory** (default) — the ships-with-the-app prompt. Streamers
  who never visit this tab see zero behaviour change.
- **Guided** — answer 1-4 short questions; we slot the answers into a
  pre-built template appended to the factory prompt. Cheap personality
  dialing without rewriting instructions.
- **Custom** — full-text editor. Replaces the entire prompt.

## Contents

- [Editable prompts](#editable-prompts)
- [How modes work](#how-modes-work)
- [Adding a new editable prompt](#adding-a-new-editable-prompt)
- [Why some prompts are NOT editable](#why-some-prompts-are-not-editable)

---

## Editable prompts

Seven prompts span Conversation insights / Channel context / Transcripts:

| Prompt                                | What it drives                              |
| ------------------------------------- | ------------------------------------------- |
| `insights.talking_points`             | per-chatter conversation hooks              |
| `insights.engaging_subjects`          | distinct conversation subject extraction    |
| `insights.subject_talking_points`     | what the streamer could say about a subject |
| `insights.open_questions`             | which chat questions count as "still open"  |
| `insights.thread_recaps`              | observational summaries of topic threads    |
| `summarizer.topics_snapshot`          | channel-wide topic snapshots                |
| `transcript.group_summary`            | streamer-voice 60-second window summaries   |

Each has 3-4 Guided slots — see
[insights.md](insights.md) for what each panel does, and the slot
metadata in
[`llm/prompts.py:_build_registry`](../src/chatterbot/llm/prompts.py)
for the exact questions + defaults.

Slots are split into "always visible" (most-personality-driven, 2-3
per prompt) and `advanced=True` (mechanics knobs, hidden behind a
"Show more options" disclosure).

## How modes work

### Factory

The factory text is imported directly from the existing constants in
[`insights.py`](../src/chatterbot/insights.py),
[`summarizer.py`](../src/chatterbot/summarizer.py), and
[`transcript.py`](../src/chatterbot/transcript.py). The registry uses
the same string objects — no copy. When a streamer is in factory
mode, `resolve_prompt(call_site, repo)` returns those strings
verbatim:

```python
# Factory mode resolution (pseudocode):
mode = repo.get_app_setting(f"prompts.{call_site}.mode") or "factory"
if mode == "factory":
    return registry[call_site].factory  # the original constant
```

Real implementation:
[`llm/prompts.py:resolve_prompt`](../src/chatterbot/llm/prompts.py).

### Guided

Guided mode uses a string-substitution template (NOT `str.format` —
some factory prompts have literal `{` characters in JSON few-shot
examples, which `format` would interpret as placeholders).

The registry's `guided_template` is the factory text with a
"STREAMER PREFERENCES" injection block appended, containing
`{slot_name}` markers. The resolver fills them via targeted replace:

```python
# Guided mode resolution (pseudocode):
saved = json.loads(repo.get_app_setting(f"prompts.{call_site}.guided") or "{}")
rendered = pd.guided_template
for slot in pd.guided_slots:
    value = saved.get(slot.name) or slot.default
    rendered = rendered.replace("{" + slot.name + "}", value)
return rendered
```

Streamers who only fill some slots get defaults for the rest.

### Custom

Custom mode replaces the entire prompt with whatever the streamer
typed. Empty custom text falls back to factory at runtime — a half-
finished edit can't blank out the LLM call.

```python
# Custom mode resolution (pseudocode):
custom = (repo.get_app_setting(f"prompts.{call_site}.custom") or "").strip()
return custom if custom else pd.factory
```

### Storage

Three keys per call site in `app_settings`:

```
prompts.<call_site>.mode      = "factory" | "guided" | "custom"
prompts.<call_site>.guided    = JSON dict {slot: value}
prompts.<call_site>.custom    = full prompt text
```

Empty / absent = factory mode. The "Revert to factory" button on the
card wipes all three keys at once. Idempotent — re-clicking is a
no-op.

### When changes apply

Saved prompts take effect on the **next refresh** of the affected
feature (most loops are 3-5 min). The Save flash banner explicitly
mentions this so streamers don't expect instant updates.

No restart is required. The resolver runs per-LLM-call, so every
new call picks up the streamer's current saved values.

## Adding a new editable prompt

Three steps:

1. **Add a `PromptDef` to the registry** in
   [`llm/prompts.py:_build_registry`](../src/chatterbot/llm/prompts.py).
   Set `factory=` to the existing constant (import it from the call
   site's module). Build `guided_template` as `factory + injection_block`
   with `{slot_name}` placeholders. Define `guided_slots`.

2. **Switch the call site** to use `resolve_prompt` instead of the
   hardcoded constant:

   ```python
   # Before:
   system_prompt=self.MY_PROMPT_SYSTEM,

   # After:
   from .llm.prompts import resolve_prompt
   system_prompt=resolve_prompt("my.call_site", self.repo),
   ```

3. **Add the call site to the test registry**. The
   `EXPECTED_CALL_SITES` set in
   [`tests/dataset/test_call_sites.py`](../tests/dataset/test_call_sites.py)
   pins which sites pass `call_site=` through to dataset capture; if
   the new prompt's call site isn't there, the AST-walk test fails.

That's it. The new prompt automatically appears in the Prompts
settings tab — the UI iterates `all_prompt_defs()` from the
registry.

## Why some prompts are NOT editable

The registry deliberately covers only **streamer-personality** /
**channel-context** prompts. Three call sites stay non-editable:

| Site                                  | Why not                                       |
| ------------------------------------- | --------------------------------------------- |
| `moderator.incident_classification`   | Editing could create false negatives on harassment / threats. The streamer doesn't gain personality benefit; they risk the bot missing a real incident. |
| `summarizer.note_extraction`          | Tampering corrupts the chatters DB with hallucinated facts. Not a personality decision. |
| `summarizer.profile_extraction`       | Same — profile fields (pronouns / location / interests) are factual extractions. |
| `transcript.llm_match`                | Mechanical state-transition matcher. No personality dimension; tweaking it risks silently dropping confirmed engagements. |

A test pins this list so a future contributor adding a moderator-prompt
edit accidentally fails CI and has to think hard about it. See
[`tests/llm/test_prompts_registry.py:test_correctness_critical_sites_are_NOT_editable`](../tests/llm/test_prompts_registry.py).
