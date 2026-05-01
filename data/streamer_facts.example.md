# Streamer facts

This file is read by the AI features on the dashboard so they can
ground chat references in **what's actually true about this channel**
rather than what chat seems to imply. Five generative features read
it: engaging-subjects extraction, talking-points, thread recaps,
subject talking-points, and question answer-angles.

It's safe to leave it as-is — the defaults below are conservative and
won't hurt grounding for any streamer. Filling it in with channel-
specific facts dramatically improves accuracy on inside jokes,
recurring bits, and current arcs.

To customize: open Settings → Prompts → **Channel facts**, or edit
the file at the path configured in Settings → Insights → "Streamer
facts file". Mtime-cached — the next AI refresh (within ~3 min)
picks up your changes.

## About the streamer

> This section hasn't been customized. The AI will fall back to the
> Helix-supplied **STREAMER NAME** and **KNOWN GAME** pins it gets
> from other sources for identity grounding.
>
> *To customize:* describe yourself in 2-4 lines — the name you go
> by on stream, pronouns if you want them used, region / timezone if
> chat references "early stream" / "late stream", any handle
> shortenings or nicknames chat uses for you. Treat each line as a
> ground-truth fact the AI should believe over chat-implied
> alternatives.

## What the channel is

> *To customize:* one paragraph on what the channel actually is —
> primary game(s), category mix, any sponsorships chat may reference,
> typical schedule patterns. The AI uses this to disambiguate chat
> references like "the patch dropped" (which game?) or "the route"
> (which run?).

Until customized, the AI assumes:
- **Game terminology in chat is real game terminology**, not a
  real-world reference. Operator / character / item / map / boss /
  weapon names from games on this channel are ON-TOPIC content,
  even when they sound like real-world things.
- **In-game rage / loud reactions / clip-worthy moments are
  entertainment**, not signs of distress. Chat reactions to them
  are audience participation, not concerning behavior.
- **The streamer's mic + audio capture is the primary input** for
  any voice-aware features; chat is supplementary context that
  reacts to what was said / shown on stream.

## Recurring bits / inside jokes

> *To customize:* list each recurring bit, inside joke, pet name,
> emote-with-non-obvious-meaning, or running gag chat references
> regularly. Each entry should make it OBVIOUS that the thing is
> real and not a hallucination to flag. Examples of the pattern:
>
> - **"CATCHPHRASE"** — when this is yelled, it means X. Originates
>   from <event>.
> - **<Person>** — recurring co-streamer / friend. References to
>   "<nickname>" in chat are this specific person, not a stranger.
> - **<Bit name>** — what chat does when <trigger> happens.
>
> Until customized, the AI treats unrecognised proper nouns from
> chat as *probably* game-content references first, real-people
> references second, and won't flag them as hallucinations.

## Current series / arcs

> *To customize:* what you're actively grinding right now — current
> game / playthrough / challenge run / podcast season. Update this
> when it changes; the AI uses it to disambiguate "the new patch",
> "the boss", "the route", etc.
>
> Until customized, the AI relies on the live Helix **KNOWN GAME**
> pin for "what game is currently being streamed" and won't try to
> infer a multi-stream arc from chat alone.

## Known sensitive topics for this channel

The default religion / politics / controversy filter is in effect.

Default channel-content assumptions (apply to any gaming stream):
- **M-rated game content** (gore, violence, scary moments, weapon
  talk, in-game cursing) is on-topic. Chat reactions to it are
  entertainment, not concerning behavior.
- **In-game competitive trash-talk / gang-war / rivalry storylines**
  (PvP feuds, faction rivalries, in-game beef between streamers) is
  ON-TOPIC entertainment, not real-world conflict.
- **Loss-tilt / rage moments** are content. Don't filter clips or
  chat reactions to them as concerning.

> *To customize:* add anything channel-specific here — real-world
> topics you don't want the AI surfacing in talking-points, sponsors
> you can't mention by name, controversial subjects you've
> deliberately decided are off-limits for this channel. Conversely:
> if your channel's whole content IS political / religious / niche
> commentary, note it here so the default filter doesn't suppress
> the actual subject matter.
