"""Per-chatter "tell me more" RAG for the Insights modals.

One module, five kinds of insight cards. Each kind shares the same shape
(transcript + LLM analysis stream) but has its own framing prompt:

  - talking_point  — "what's this active chatter currently engaged with?"
  - anniversary    — "how to acknowledge their N-month/year milestone?"
  - newcomer       — "what kind of viewer is this so far?"
  - regular        — "what's this regular currently engaged with?"
  - lapsed         — "how to re-engage this lapsed regular naturally?"

Streamer-only output. Streams via SSE into the modal — never returns to
chat. Architectural rule: profile / event / topic / message data must
NEVER enter any LLM prompt that produces a chat-facing response. This
output is dashboard-only and that contract holds.

Free-form prose answers ride `OllamaClient.stream_generate()`; structured
shapes belong in `llm/schemas.py`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from ..llm.ollama_client import OllamaClient
from ..repo import ChatterRepo, Message

logger = logging.getLogger(__name__)


_AUDIENCE_BLOCK = """STREAM CONTEXT (read first):
This is a Twitch stream focused on zombie / horror / gore games. Audience
skews 18-35, predominantly male. Casual profanity, dark humor, gallows
reactions to violence, and snark are baseline-normal — not personality
red flags. Hype reactions to gory moments are dialect, not personality
data. A line that sounds positive after a death/fail context line is
likely sarcasm.
"""

_OUTPUT_RULES = """OUTPUT RULES:
- Total: 2-3 short sentences, max ~80 words. No bullet points.
- This output goes ONLY to the streamer's private dashboard. Never to chat.
- Don't put suggested phrasing in quotes — just describe the angle.
- If the evidence doesn't support the framing, say so plainly.
"""


_KIND_PROMPTS: dict[str, str] = {
    "adhoc": (
        _AUDIENCE_BLOCK + "\n"
        "INPUT: a viewer's recent messages (`[id]`) interleaved with chat-wide "
        "context lines (`[ctx id] otherUser:`). The streamer just noticed "
        "this person and wants a 1-line angle to engage them right now.\n\n"
        "TASK: 1 sentence on what they're currently engaged with (specific "
        "moment, joke, ongoing thread). Then 1 short sentence: a concrete "
        "thing the streamer could say that connects to it. Keep it casual.\n\n"
        + _OUTPUT_RULES
    ),
    "talking_point": (
        _AUDIENCE_BLOCK + "\n"
        "INPUT: a viewer's recent messages (`[id]`) interleaved with chat-wide "
        "context lines (`[ctx id] otherUser:`), plus a previously-generated "
        "talking point about this viewer.\n\n"
        "TASK: confirm or refine the talking point in 1-2 sentences (expand if "
        "supported, dismiss if not). Then 1 short sentence: a concrete thing "
        "the streamer could say to engage them naturally on this thread "
        "(acknowledgment, callback, or question).\n\n"
        + _OUTPUT_RULES
    ),
    "anniversary": (
        _AUDIENCE_BLOCK + "\n"
        "INPUT: a viewer hitting an anniversary milestone (e.g. '1 year' since "
        "their first message). You'll see their recent messages with chat "
        "context.\n\n"
        "TASK: 1 sentence on the vibe of who this person is based on their "
        "recent activity (reactive lurker? regular? big personality?). Then "
        "1-2 sentences on how the streamer could acknowledge the milestone in "
        "a low-key, non-awkward way that fits this chatter — NOT a generic "
        "'shout out'. If they just chat reactions to gameplay, say a simple "
        "name-mention is fine.\n\n"
        + _OUTPUT_RULES
    ),
    "newcomer": (
        _AUDIENCE_BLOCK + "\n"
        "INPUT: a brand-new viewer's first messages (within the last 24h) "
        "interleaved with chat-wide context lines so you can see what they "
        "were reacting to.\n\n"
        "TASK: 1 sentence on what kind of viewer they look like so far (hype, "
        "lurker who finally spoke, technical, banter-heavy, etc.). Then 1-2 "
        "sentences on the easiest natural way to make them feel welcome based "
        "on what they've actually said — not generic welcomes.\n\n"
        + _OUTPUT_RULES
    ),
    "regular": (
        _AUDIENCE_BLOCK + "\n"
        "INPUT: an established regular's recent messages interleaved with "
        "chat-wide context.\n\n"
        "TASK: 1 sentence on what they're currently engaged with right now "
        "(specific game moment, ongoing chat thread, something personal). "
        "Then 1-2 sentences on a callback or question the streamer could use "
        "that lands because the regular has context — not generic engagement.\n\n"
        + _OUTPUT_RULES
    ),
    "lapsed": (
        _AUDIENCE_BLOCK + "\n"
        "INPUT: a lapsed regular's most recent messages from when they were "
        "still active (may be days/weeks old) plus chat-wide context.\n\n"
        "TASK: 1 sentence on what this person used to talk about / their vibe "
        "based on the transcript. Then 1-2 sentences on the most genuine way "
        "to re-engage them next time they show up (a callback to a past "
        "thread, asking about something they cared about, etc). Don't suggest "
        "calling out their absence — that reads as guilt-tripping.\n\n"
        + _OUTPUT_RULES
    ),
}


# Display config consumed by the modal template + nav title. Keep here so
# adding a new kind only touches this file + the route enum.
KIND_DISPLAY: dict[str, dict[str, str]] = {
    "adhoc":         {"title": "Engaged by", "icon": "fa-bookmark",
                      "meta_label": "context"},
    "talking_point": {"title": "Active right now", "icon": "fa-fire",
                      "meta_label": "talking point"},
    "anniversary":   {"title": "Anniversary today", "icon": "fa-cake-candles",
                      "meta_label": "milestone"},
    "newcomer":      {"title": "New today", "icon": "fa-user-plus",
                      "meta_label": "first seen"},
    "regular":       {"title": "Regular", "icon": "fa-crown",
                      "meta_label": "window"},
    "lapsed":        {"title": "Lapsed regular", "icon": "fa-user-clock",
                      "meta_label": "last seen"},
}

VALID_KINDS = frozenset(_KIND_PROMPTS.keys())


class InsightContext:
    def __init__(
        self,
        kind: str,
        user_id: str,
        name: str,
        meta: str,
        messages: list[Message],
        focal_ids: set[int],
        stream: AsyncIterator[str],
    ):
        self.kind = kind
        self.user_id = user_id
        self.name = name
        self.meta = meta
        self.messages = messages
        self.focal_ids = focal_ids
        self.stream = stream


async def explain_insight(
    repo: ChatterRepo,
    llm: OllamaClient,
    kind: str,
    user_id: str,
    meta: str,
) -> InsightContext | None:
    if kind not in VALID_KINDS:
        return None
    user = await asyncio.to_thread(repo.get_user, user_id)
    if not user:
        return None
    rows, focal_ids = await asyncio.to_thread(
        repo.recent_user_messages_with_context, user_id,
    )
    prompt = _build_prompt(kind, user.name, meta, rows, focal_ids)
    stream = llm.stream_generate(
        prompt=prompt, system_prompt=_KIND_PROMPTS[kind],
    )
    return InsightContext(
        kind=kind, user_id=user_id, name=user.name, meta=meta,
        messages=rows, focal_ids=focal_ids, stream=stream,
    )


def _build_prompt(
    kind: str,
    name: str,
    meta: str,
    rows: list[Message],
    focal_ids: set[int],
) -> str:
    label = KIND_DISPLAY[kind]["meta_label"]
    parts = [f"Focal viewer: {name}"]
    if meta:
        parts.append(f"{label.capitalize()}: {meta}")
    parts.extend(["", "Recent chat transcript:"])
    if not rows:
        parts.append("- (no recent messages from this viewer)")
    else:
        for m in rows:
            tag = f"[{m.id}]" if m.id in focal_ids else f"[ctx {m.id}] {m.name}:"
            content = (m.content or "").replace("\n", " ")
            parts.append(f"{tag} {content}")
    parts.append("")
    parts.append("Now respond per the rules above.")
    return "\n".join(parts)
