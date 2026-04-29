# Streamer facts

Streamer-authored notes about *you* (the streamer), the channel, and any
recurring context the LLM should know when interpreting chat. The
engaging-subjects extractor reads this file at refresh time and prepends
it to the system prompt so subject extraction is grounded in what's
actually true about your channel rather than what chat seems to imply.

To use this file: copy it to `streamer_facts.md` in the same directory
and edit. The path is configurable in /settings → Insights →
"Streamer facts file" if you want to keep it elsewhere.

Lines starting with `#` are headings; everything else is prose. Keep
it short — this gets injected into every extraction call. A few
hundred words is plenty; a few thousand will start eating your
context budget on every refresh.

## About the streamer

(Add things like: name you go by on stream, real name if you want the
LLM to use it, pronouns, region/timezone, how long you've been
streaming, what kind of content you usually do, day job if relevant.)

## What the channel is

(Primary game(s) or content type, typical schedule, average concurrent
viewer count, sponsorships chat may reference. Helps the LLM know
whether "patch dropped" means the game you stream or something else.)

## Recurring bits / inside jokes

(Things chat references regularly that an outside LLM wouldn't
recognise: pet names, running jokes, lore from past streams, emotes
with non-obvious meaning, real people who keep coming up. Each entry
should make it OBVIOUS that the thing is real and not a hallucination
to flag.)

## Current series / arcs

(Games or topics you're currently in the middle of: "doing a no-hit
run of X", "playthrough series of Y", "weekly Z night". Helps the
LLM distinguish a one-off mention from an ongoing arc.)

## Known sensitive topics for this channel

(Anything beyond the default religion / politics / controversy filter
that you want treated as sensitive on YOUR channel specifically.
Conversely: if your channel's whole content IS political commentary,
note it here so the LLM doesn't filter the actual subject matter.
Leave blank if the default filter is enough.)
