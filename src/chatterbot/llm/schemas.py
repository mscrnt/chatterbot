"""Pydantic response schemas for every structured LLM call.

Every call to the LLM that expects parseable output goes through
`OllamaClient.generate_structured(..., response_model=Model)`. The model's
JSON Schema is passed to Ollama as the `format` parameter (Ollama 0.5+
constrains generation to match), and the response is validated on receipt
with the same pydantic model. One source of truth for both the prompt-time
constraint and the parse-time check.

Add new schemas here when adding new structured call sites — never reach for
manual `json.loads` + dict-walking in summarizer / RAG / etc.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints


# ---- per-user note extraction ----

NoteString = Annotated[
    str,
    StringConstraints(min_length=1, max_length=500, strip_whitespace=True),
]


class NoteEntry(BaseModel):
    """One extracted fact, plus the message_ids the model cited as supporting
    it. Provenance for "where did this fact come from?" — surfaced in the
    dashboard as expandable source messages on each note."""

    text: NoteString
    source_message_ids: list[int] = Field(default_factory=list, max_length=5)


class NoteExtractionResponse(BaseModel):
    """Reply for `summarizer._summarize_user`. The LLM is asked to extract 0-3
    factual notes about a single viewer. Empty list means "nothing notable",
    which is a valid (and common) result."""

    notes: list[NoteEntry] = Field(default_factory=list, max_length=3)


# ---- channel topic snapshotter ----

TopicTitle = Annotated[
    str,
    StringConstraints(min_length=1, max_length=200, strip_whitespace=True),
]
DriverName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=64, strip_whitespace=True),
]


TopicCategory = Literal[
    "gaming",     # game mechanics, runs, builds, in-game events
    "personal",   # life, pets, family, work, location
    "meta",       # stream meta, schedule, gear, OBS, technical broadcast
    "tech",       # hardware, software, programming
    "off-topic",  # jokes, banter, memes
    "other",
]


class TopicEntry(BaseModel):
    topic: TopicTitle
    drivers: list[DriverName] = Field(default_factory=list, max_length=10)
    # Phase-4 thread tag. Older topics_json rows that pre-date this field
    # validate cleanly because of the default.
    category: TopicCategory = "other"


class TopicsResponse(BaseModel):
    """Reply for `summarizer._take_topic_snapshot`. 0-5 topics summarizing
    what chat is currently discussing, each with the usernames driving it."""

    topics: list[TopicEntry] = Field(default_factory=list, max_length=5)


# ---- moderation incident classifier ----

ModCategory = Literal[
    "harassment",
    "hate_speech",
    "threats",
    "spam",
    "doxxing",
    "other",
]

Rationale = Annotated[
    str,
    StringConstraints(min_length=1, max_length=400, strip_whitespace=True),
]


class IncidentClassification(BaseModel):
    """One flagged message. Only emitted when is_violation=true."""

    message_id: int
    is_violation: bool
    severity: Literal[1, 2, 3]            # 1 minor / 2 warning / 3 serious
    categories: list[ModCategory] = Field(default_factory=list, max_length=4)
    rationale: Rationale


class ModerationBatchResponse(BaseModel):
    """Reply for `moderator._review_batch`. The classifier reviews a batch of
    chat messages and returns ONLY the ones that violate the rubric. Returning
    an empty list is the expected, common case when chat is calm."""

    classifications: list[IncidentClassification] = Field(default_factory=list)


# ---- engagement / "talking points" helper ----

TalkingPointText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=300, strip_whitespace=True),
]


class TalkingPoint(BaseModel):
    """One conversation hook for a single active chatter, keyed by the index
    we numbered them with in the prompt (1-based)."""

    chatter_index: int
    point: TalkingPointText


class TalkingPointsResponse(BaseModel):
    """Reply for `insights.InsightsService._refresh`. The model gets a numbered
    list of currently-active chatters with their notes + recent messages and
    returns one short conversation-hook per chatter for the streamer to use."""

    points: list[TalkingPoint] = Field(default_factory=list, max_length=20)
