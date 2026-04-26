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


class NoteExtractionResponse(BaseModel):
    """Reply for `summarizer._summarize_user`. The LLM is asked to extract 0-3
    factual notes about a single viewer. Empty list means "nothing notable",
    which is a valid (and common) result."""

    notes: list[NoteString] = Field(default_factory=list, max_length=3)


# ---- channel topic snapshotter ----

TopicTitle = Annotated[
    str,
    StringConstraints(min_length=1, max_length=200, strip_whitespace=True),
]
DriverName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=64, strip_whitespace=True),
]


class TopicEntry(BaseModel):
    topic: TopicTitle
    drivers: list[DriverName] = Field(default_factory=list, max_length=10)


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
