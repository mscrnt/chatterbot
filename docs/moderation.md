# Moderation

An opt-in advisory moderation classifier. Reviews recent chat in
batches and flags messages that violate harassment / hate-speech /
threat / spam rules. **Advisory only** — the bot never auto-actions
chat. The streamer reviews flagged incidents in a dedicated dashboard
tab.

## Contents

- [Off by default](#off-by-default)
- [Enable](#enable)
- [What it does](#what-it-does)
- [What it doesn't do](#what-it-doesnt-do)
- [Reviewing incidents](#reviewing-incidents)
- [Why this prompt is NOT customizable](#why-this-prompt-is-not-customizable)

---

## Off by default

`mod_mode_enabled=false` in [`config.py`](../src/chatterbot/config.py).
With it off, the moderator loop never starts and the dashboard's
Moderation tab is hidden from the nav.

This is intentional. Streamers running chatterbot are usually focused
on chatter profiling + insights, not moderation. Twitch's own
AutoMod + human moderators handle most needs already.

## Enable

```bash
MOD_MODE_ENABLED=true
```

Or toggle via `/settings → AI brain → mod mode`. Restart the bot
after enabling. The Moderation tab appears in the dashboard nav.

Optionally route the classifier to a different (smaller / faster) LLM:

```bash
OLLAMA_MOD_MODEL=qwen3.5:3b   # for example
```

The classifier is the highest-frequency LLM call in the bot — it
runs on every chat batch — so dedicated lightweight capacity helps.
Empty falls back to `OLLAMA_MODEL`.

## What it does

The moderator loop runs in the bot process. Periodically:

1. Pull recent unclassified messages from `messages`.
2. Format as a numbered review batch.
3. LLM call returns `ModerationBatchResponse` with a list of
   classifications: only the messages the model judged as actual
   violations, with `severity` (1=minor / 2=warning / 3=serious),
   `categories` (harassment / hate_speech / threats / spam /
   doxxing / other), and a short `rationale`.
4. Each classification writes a row to `incidents` with
   `status='open'`.
5. Watermark advances so the next pass starts where this one left off.

Implementation:
[`moderator.py`](../src/chatterbot/moderator.py).

## What it doesn't do

The moderator never:

- **Times out, bans, or deletes** any message. No Twitch mod actions
  fire from this bot. The bot account doesn't even need mod
  permissions.
- **Whispers / responds in chat** about flagged content.
- **Escalates** to anywhere outside the dashboard.

It only writes rows to `incidents`. The streamer reads those rows on
the dashboard's Moderation tab and decides what to do.

## Reviewing incidents

`/moderation` shows every open incident, newest first:

- Original message + chatter + timestamp
- Severity + categories the LLM flagged
- The rationale string
- "Show context" — recent messages from this chatter for situational
  awareness

Per-incident actions:

- **Mark reviewed** — flips `status='reviewed'`. Hides from the open
  list. Audit trail keeps the row.
- **Dismiss** — flips `status='dismissed'`. Same effect as reviewed
  but explicitly tells the dataset capture that this was a
  false-positive (negative supervision for a future fine-tune).

The dashboard's `/audit` page shows every incident-status transition
alongside insight-state changes.

## Why this prompt is NOT customizable

The moderator's system prompt (`MOD_REVIEW_SYSTEM` in
[`moderator.py`](../src/chatterbot/moderator.py)) is **deliberately
not in the streamer-customizable prompts list** — see
[prompts.md → Why some prompts are NOT editable](prompts.md#why-some-prompts-are-not-editable).

The prompt is correctness-critical: editing it via the UI could
create false negatives that miss harassment or hate speech. There's
no streamer-personality dimension to gain — what counts as
harassment is not a stylistic choice.

If you need to tune the moderator's behaviour:

- **Lower the noise floor** — the LLM already returns "no
  violations" on the vast majority of batches; if you're getting
  too many flags, check the model. A larger model is usually
  more reliable than a smaller one for classification tasks.
- **Adjust the chat-window** — `mod_review_max_messages_per_pass`
  controls how many messages per LLM call. Larger windows give
  more context but take longer.
- **Add custom rules in code** — if you need channel-specific
  rules (e.g. "ban competitor channel mentions"), edit the
  `MOD_REVIEW_SYSTEM` constant directly. There's no UI gate
  because there's no safe UI gate — but PRs adding a more careful
  customization mechanism are welcome (see
  [development.md](development.md)).

A regression test pins that this prompt stays non-customizable so
a future contributor accidentally exposing it via UI fails CI:
[`tests/llm/test_prompts_registry.py:test_correctness_critical_sites_are_NOT_editable`](../tests/llm/test_prompts_registry.py).
