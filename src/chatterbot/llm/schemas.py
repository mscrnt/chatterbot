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
    """Reply for `summarizer._summarize_user`. The LLM extracts up to 8
    short third-person notes about a single viewer — covering hard
    self-disclosure (pets, gear, location, jobs), stated opinions /
    takes, recurring references, explicit preferences, and recurring
    chat patterns specific to this viewer. The prompt asks the model
    to err generous: 2-4 is typical on a chatter with substantive
    activity; 0 only on pure greeting / reaction streams."""

    notes: list[NoteEntry] = Field(default_factory=list, max_length=8)


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


# ---- topic-thread recaps (batch summarisation, observational) ----

ThreadRecapText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=400, strip_whitespace=True),
]


class ThreadRecap(BaseModel):
    """One recap for a single active topic thread. `thread_id` references
    the integer key the prompt numbered the threads with. `recap` is a
    1-2 sentence observational summary — what the chatters are actually
    discussing — never a suggestion to the streamer."""

    thread_id: int
    recap: ThreadRecapText


class ThreadRecapsResponse(BaseModel):
    """Reply for the batched thread recap call. The model reads a
    numbered list of active threads with their drivers + recent messages
    and returns one recap per thread it can ground in the messages.
    Empty / partial replies are fine — skip threads where the messages
    are too noisy to summarise without inventing facts."""

    recaps: list[ThreadRecap] = Field(default_factory=list, max_length=20)


# ---- transcript group summaries ----


class TranscriptGroupSummaryResponse(BaseModel):
    """Reply for the per-window transcript grouper. Reads a contiguous
    block of streamer utterances + chat that happened during the same
    window, and returns a 2-4 sentence observational recap. Empty
    string means "nothing summarisable in this window" — the grouper
    persists silently and skips.

    Bumped 400 → 1200 chars: 400 was clipping multi-topic windows mid-
    sentence on busy streams, and the dashboard's transcript strip
    handles 2-4 sentence rows fine."""

    summary: str = Field(default="", max_length=1200)


# ---- engaging subjects (per-message extraction with sensitivity gate) ----

EngagingSubjectName = Annotated[
    str,
    StringConstraints(min_length=3, max_length=120, strip_whitespace=True),
]


class EngagingSubject(BaseModel):
    """One distinct conversation subject the LLM extracted from a window
    of recent chat messages.

    `drivers` is the list of chatters actually engaging with this
    subject (not just present in chat). `msg_count` is approximate —
    how many recent messages the model judged to be on this subject.
    `is_sensitive` flags subjects involving religion, politics, or
    other controversies that the streamer probably wants the dashboard
    to silently filter out.

    `brief` is a 1-2 sentence observational summary of what chatters
    are actually saying about the subject — paraphrase, not advice.
    `angles` is up to 3 sub-aspects of the subject that have come up
    in the messages (e.g. for "Resident Evil parry timing": "aim-parry
    vs perfect parry", "comparison to Ninja Gaiden 4"). These ride
    along with the subject so they're available the moment the
    streamer expands the row — no on-demand LLM call needed."""

    name: EngagingSubjectName
    drivers: list[str] = Field(default_factory=list, max_length=20)
    msg_count: int = Field(ge=0, default=0)
    is_sensitive: bool = False
    brief: str = Field(default="", max_length=400)
    angles: list[str] = Field(default_factory=list, max_length=4)
    # Echoed-back identifier from the input prompt so we can match the
    # LLM's labels to the persistent cluster they belong to. Empty when
    # the LLM doesn't know about persistent clustering (legacy / one-off
    # extraction without prior state).
    cluster_id: str = Field(default="", max_length=64)


class EngagingSubjectsResponse(BaseModel):
    """Reply for the engaging-subjects extractor on InsightsService.
    Empty list is a valid output when chat is too quiet / unfocused
    to identify distinct subjects."""

    subjects: list[EngagingSubject] = Field(default_factory=list, max_length=12)
