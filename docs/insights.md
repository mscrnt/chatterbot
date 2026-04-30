# Insights

The streamer-facing LLM panels on `/insights`. Every panel is a
background loop in [`insights.py`](../src/chatterbot/insights.py) that
produces an in-memory cache the dashboard renders directly. None of
this output ever returns to chat (see
[architecture.md → The hard rule](architecture.md#the-hard-rule)).

## Contents

- [Talking points](#talking-points)
- [Engaging subjects](#engaging-subjects)
- [Per-subject talking-points modal](#per-subject-talking-points-modal)
- [Open questions](#open-questions)
- [Thread recaps](#thread-recaps)
- [Active but not engaged](#active-but-not-engaged)
- [Talking to you](#talking-to-you)
- [Card actions](#card-actions)

---

## Talking points

One short conversation hook per active chatter, refreshed every ~3 min.
Streamer reads on a second monitor while playing.

How it works:

1. Pull the active-chatter cap (~25) from the last 10 min.
2. For each chatter, batch-fetch their stored notes + recent messages.
3. Numbered prompt block: `[1] alice notes:... messages:...`
4. Per-call enrichment: channel context + streamer voice + screenshot
   grid + current topic snapshot.
5. LLM call with `think=True` + `INFORMED_NUM_CTX` (32k) returns
   `TalkingPointsResponse` — at most one hook per chatter, indexed by
   the number we sent.
6. Cache by `(user_id, name, point)` for the dashboard render.

Implementation:
[`insights.py:_refresh`](../src/chatterbot/insights.py).

Streamer rules baked into the system prompt:

- Only paraphrase content visible in the chatter's notes / messages.
- Never invent products / events / people not attested.
- Skipping is fine — empty `points` list is the right answer when nothing's
  groundable.

The prompt is streamer-customizable via the `insights.talking_points`
entry on the Prompts settings tab. See [prompts.md](prompts.md) for
how Factory / Guided / Custom modes work.

## Engaging subjects

The biggest LLM panel — distinct conversation subjects chat is currently
discussing. Cards on `/insights`; click any to open the
[per-subject modal](#per-subject-talking-points-modal).

Earlier versions clustered raw chat embeddings via online cosine and
labelled clusters. That collapsed into one mega-cluster on real chat
(centroid mean-drift on noisy short strings). Slice 7 inverted the work
split:

- **LLM does the topic modeling** on numbered chat messages and cites
  supporting `message_ids` for each subject.
- **Embeddings drive cross-refresh identity** by comparing the SUBJECT
  NAME (a short clean string) against existing persistent subjects via
  cosine ≥ 0.75.

Per refresh:

1. Pull recent messages (last 20 min, capped at 250).
2. Skip if `latest_message_id` hasn't moved since last pass (cheap
   short-circuit).
3. Build numbered-messages prompt + the usual context blocks.
4. LLM returns 0-8 subjects with cited `message_ids`.
5. For each: embed the name, look up in `_subjects` by cosine threshold
   → reuse persistent ID or spawn new.
6. Materialise the cache from this-pass's surfaced subjects.
7. Age out persistent subjects unseen for 2× the lookback window.

Cache shape (`EngagingSubjectEntry`): `name`, `drivers`, `msg_count`,
`brief`, `angles`, `slug`, `msg_ids`. The slug is a stable
sha1-of-name-lowercased prefix the dashboard uses for routing
(`/modals/subject/{slug}`, `/insights/subject/{slug}/reject`).

Implementation:
[`insights.py:_refresh_engaging_subjects`](../src/chatterbot/insights.py).

Streamer-customizable via the `insights.engaging_subjects` prompt entry.
See [prompts.md](prompts.md). The customizable knobs cover subject
specificity, extra filtering rules, minimum activity threshold, and
whether to surface meta-subjects (subjects about the streamer / channel
itself).

### Diagnostics

If the panel never populates, check the dashboard logs for the
single per-refresh summary line:

```
engaging subjects refreshed: N msgs → LLM returned X, applied Y
  (dropped A sensitive, B blocklisted, C ungrounded), cached Z,
  persistent K (aged out W)
```

Every gate is visible. If you ever see `cached 0` after `applied N`, it
means everything got dropped via blocklist / sensitivity / ungrounded
filter. Common causes:

- Threshold too aggressive — see the embedding identity-match value.
- Blocklist accumulated too many "wrong" rejections — clear via
  `Settings → reset blocklist`.

## Per-subject talking-points modal

Click any engaging-subject card. Opens a modal showing:

- Brief + sub-aspects (cached, instant render)
- **AI talking points** — async-loaded via HTMX. 3-5 short lines the
  streamer could say BACK to chat about this subject. Generated on
  first open; cached on the persistent subject until the next refresh
  surfaces this subject again with new context.
- Drivers as clickable pills → `/users/<id>`
- Cited chat messages, scrollable
- Footer: dismiss + close

The talking-points generator is `insights.subject_talking_points` in
the streamer-customizable prompt registry. Customizable knobs cover
voice / style, things to avoid, count of points, and self-disclosure
level. Implementation:
[`insights.py:generate_subject_talking_points`](../src/chatterbot/insights.py).

The modal template is
[`templates/modals/_engaging_subject.html`](../src/chatterbot/web/templates/modals/_engaging_subject.html).

## Open questions

Surfaces chat questions that are still genuinely **open** — asked of
the streamer or the room broadly, not yet answered, not directed at
another chatter.

Two-stage pipeline:

1. **Heuristic**: `repo.recent_questions` clusters `?`-bearing chat by
   token-overlap (Szymkiewicz-Simpson coefficient ≥ 0.5). SQL filters
   exclude Twitch reply rows + `@`-prefixed messages. Output:
   candidate question clusters with their askers.
2. **LLM filter**: `insights.open_questions_loop` runs the candidates
   through the LLM with full chat + transcript context. Drops:
   - Already-answered questions (in chat OR by the streamer out loud)
   - Directed-at-other-chatter asks that slipped past SQL
   - Rhetorical / pure-reaction / bot-command noise

Cache shape mirrors the heuristic dict (`question`, `count`, `drivers`,
`latest_ts`, `last_msg_id`) so the template renders without changes.

Implementation:
[`insights.py:_refresh_open_questions`](../src/chatterbot/insights.py).

Streamer-customizable via the `insights.open_questions` prompt entry.
Knobs cover filter strictness, streamer-relevance preference, and how
aggressively to drop already-answered questions.

## Thread recaps

The Live Conversations panel shows active topic threads (cosine-clustered
topic-snapshot groupings) with one observational LLM-written recap each.

The `thread_recap_loop` runs every ~5 min, picks active threads, and for
each one batches the recent driver messages into a single LLM call that
returns recaps. Recaps go to `topic_threads.recap`; the dashboard reads
from there.

Implementation:
[`insights.py:_refresh_thread_recaps`](../src/chatterbot/insights.py).

Streamer-customizable via the `insights.thread_recaps` prompt entry.
Knobs cover tone, length, and focus (content vs mood vs both).

## Active but not engaged

Regulars who are currently in chat (active in the last 30 min) and that
the streamer hasn't talked to in the last 7 days. Pure SQL — no LLM
call. Implementation:
[`repo.list_neglected_lurkers`](../src/chatterbot/repo.py).

## Talking to you

Recent `@`-mentions or direct addresses to the streamer. Detected via
three signals:

1. `@<channel>` or bare `<channel>` at a word boundary
2. First-N-chars prefix at a word boundary (covers "pcplays" for
   `pcplaysgames` etc.)
3. Twitch IRCv3 reply tag pointing at the streamer

Implementation:
[`repo.recent_direct_mentions`](../src/chatterbot/repo.py).

## Card actions

Every Insights card has a row of small icons (dismiss / addressed /
snooze / pin) at the right. These all flow through the same
endpoint:

```
POST /insights/state
  kind=<insight_kind>
  item_key=<id>
  state=<addressed|skipped|snoozed|pinned|open>
  due_ts=<iso, only for snoozed>
  note=<optional note>
```

Storage is `insight_states` (current state) + `insight_state_history`
(every transition). The audit page at `/audit` shows the full
transition log.

State semantics:

- **addressed** — streamer talked about it on stream. Card hides.
- **skipped** — not relevant. Card hides until the underlying item
  changes (e.g. for chat_question, the cluster's `last_msg_id`).
- **snoozed** — hide for N minutes; re-surfaces after `due_ts`.
- **pinned** — keep at the top of its section.
- **open** — clear all state. Default.

If whisper is enabled, the LLM-match loop auto-flips `addressed` for
talking-points + threads when it detects the streamer engaged with the
subject out loud — see [whisper.md → LLM-match loop](whisper.md#llm-match-loop).
