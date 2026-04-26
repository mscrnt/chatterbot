"""Topic threading — cluster snapshot-topics into recurring conversation
threads via embedding similarity.

A thread is a topic that's been observed across one or more snapshots. The
Threader is called once per topic in a fresh snapshot:

  1. Embed the topic title via Ollama (`nomic-embed-text`).
  2. Search `vec_threads` for the nearest existing thread by cosine distance.
  3. If best distance < `(1 - SIMILARITY_THRESHOLD)`, attach this snapshot
     as another member of that thread (and update title + last_ts +
     category to the latest values).
  4. Otherwise create a new thread.

Also runs a one-shot **backfill** at bot startup that walks any
`topic_snapshots.topics_json` rows that don't yet have thread members and
clusters them in chronological order — so existing data isn't lost when
this feature ships.

Streamer-only: thread output renders only in the dashboard and never
returns to chat.
"""

from __future__ import annotations

import asyncio
import logging

from .config import Settings
from .llm.ollama_client import OllamaClient
from .llm.schemas import TopicsResponse
from .repo import ChatterRepo

logger = logging.getLogger(__name__)


# Cosine similarity threshold for "this topic is a continuation." Tuned for
# nomic-embed-text — short titles cluster tightly when they overlap on the
# noun phrase that matters. distance = 1 - cosine_sim, so 0.30 → sim ≥ 0.70.
SIMILARITY_DISTANCE_MAX = 0.30


class Threader:
    def __init__(self, repo: ChatterRepo, llm: OllamaClient, settings: Settings):
        self.repo = repo
        self.llm = llm
        self.settings = settings

    async def cluster_topic(
        self,
        snapshot_id: int,
        topic_index: int,
        title: str,
        category: str | None,
        drivers: list[str],
        ts: str,
    ) -> int:
        """Embed `title`, find or create the matching thread, attach this
        topic as a member. Returns the thread_id."""
        try:
            embedding = await self.llm.embed(title)
        except Exception:
            logger.exception("threader: failed to embed %r — skipping", title)
            return -1

        match = await asyncio.to_thread(self.repo.find_thread_by_embedding, embedding, 1)
        if match and match[1] <= SIMILARITY_DISTANCE_MAX:
            tid, distance = match
            await asyncio.to_thread(
                self.repo.attach_topic_to_thread,
                thread_id=tid, snapshot_id=snapshot_id, topic_index=topic_index,
                title=title, category=category, drivers=drivers, ts=ts,
            )
            logger.info(
                "threader: attached '%s' to thread %d (distance=%.3f)",
                title, tid, distance,
            )
            return tid

        tid = await asyncio.to_thread(
            self.repo.create_topic_thread,
            snapshot_id=snapshot_id, topic_index=topic_index, title=title,
            category=category, drivers=drivers, ts=ts, embedding=embedding,
        )
        logger.info("threader: new thread %d '%s'", tid, title)
        return tid

    async def cluster_snapshot(self, snapshot_id: int, ts: str, topics_json: str) -> None:
        """Cluster every topic in a fresh snapshot. Tolerant of malformed
        JSON — bad snapshots are logged and skipped."""
        try:
            parsed = TopicsResponse.model_validate_json(topics_json)
        except Exception:
            logger.exception("threader: bad topics_json on snapshot %d", snapshot_id)
            return
        for idx, entry in enumerate(parsed.topics):
            try:
                await self.cluster_topic(
                    snapshot_id, idx, entry.topic, entry.category,
                    list(entry.drivers), ts,
                )
            except Exception:
                logger.exception(
                    "threader: cluster failed for snapshot %d topic %d",
                    snapshot_id, idx,
                )

    async def backfill(self) -> int:
        """Cluster any pre-existing snapshots that don't have thread members
        yet. Idempotent — safe to call on every bot startup. Returns the
        number of snapshots backfilled."""
        snapshots = await asyncio.to_thread(self.repo.snapshots_without_threads)
        if not snapshots:
            return 0
        logger.info("threader: backfilling %d snapshots into threads", len(snapshots))
        for snap in snapshots:
            if not snap.topics_json:
                continue
            try:
                await self.cluster_snapshot(snap.id, snap.ts, snap.topics_json)
            except Exception:
                logger.exception("threader: backfill iteration failed for %d", snap.id)
        return len(snapshots)
