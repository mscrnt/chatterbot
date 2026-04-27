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
    """Reply for `summarizer._summarize_user`. The LLM extracts 0-5 short
    third-person notes about a single viewer — covering hard self-disclosure
    (pets, gear, location, jobs), stated opinions / takes, recurring
    references, and explicit preferences. Empty is valid and common."""

    notes: list[NoteEntry] = Field(default_factory=list, max_length=5)


# ---- per-user soft-profile extraction ----

# Constrained vocabulary for the demeanor field. Six buckets, plus 'unknown'
# as the "I genuinely couldn't tell" exit. Keep this list small — every new
# value the LLM has to choose from raises false-classification risk.
Demeanor = Literal[
    "hype",        # loud, lots of caps, reacts strongly to clutch moments
    "chill",       # measured, conversational, even-keeled
    "supportive",  # encouraging, hype FOR others, often replies with positivity
    "snarky",      # dry humor, sarcastic, jokes at situations
    "quiet",       # short messages, infrequent, mostly reactive
    "analytical",  # technical commentary on gameplay/strategy
    "unknown",
]

InterestString = Annotated[
    str,
    StringConstraints(min_length=2, max_length=40, strip_whitespace=True),
]
ProfileFreeText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=80, strip_whitespace=True),
]


class ProfileExtractionResponse(BaseModel):
    """Reply for the soft-profile extractor that runs alongside notes.

    Differs from notes by intent: notes capture only hard self-stated facts
    with citations. Profile captures squishier identity signals — pronouns
    when explicitly used, location when explicitly mentioned, demeanor as
    an inferred bucket, and interests as a deduped list.

    Every field is optional. None / empty means "no clear signal in this
    batch" — the merger never overwrites an existing value with absence,
    so partial signals across batches accumulate into a richer profile.
    """

    pronouns: ProfileFreeText | None = None
    location: ProfileFreeText | None = None
    demeanor: Demeanor | None = None
    interests: list[InterestString] = Field(default_factory=list, max_length=5)


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


# ---- batched whisper transcript matching ----

EvidenceText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=400, strip_whitespace=True),
]


class TranscriptMatch(BaseModel):
    """One matched insight card from the batched transcript-LLM matcher.

    `card_id` references the integer key we numbered the candidate cards
    with in the prompt (talking_points and threads share one numbering
    space, prefixed in the prompt as TP- and T-). `evidence` is the
    streamer-spoken phrase that justifies the match — surfaced in the
    Insight card's auto-pending note so the streamer can verify before
    confirming."""

    card_id: int
    evidence: EvidenceText
    confidence: float = Field(ge=0.0, le=1.0)


class TranscriptMatchResponse(BaseModel):
    """Reply for `transcript.TranscriptService._run_llm_match`. The model
    reads a window of streamer utterances and a list of open insight
    cards, and returns ONLY the cards the streamer demonstrably engaged
    with. Empty list is the expected, common case (most utterances are
    game commentary or thinking aloud, not chat-directed)."""

    matches: list[TranscriptMatch] = Field(default_factory=list, max_length=10)
