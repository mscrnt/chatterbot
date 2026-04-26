"""Seed a chatterbot DB with synthetic data so the dashboard has something to render.

Used only for local screenshot / responsive-layout verification. Safe to delete.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chatterbot.repo import ChatterRepo  # noqa: E402


CHATTERS = [
    ("11001", "alice_dev"),
    ("11002", "bob_speedruns"),
    ("11003", "charlie_modder"),
    ("11004", "deidre_lurker"),
    ("11005", "evan_cat_dad"),
    ("11006", "fern"),
    ("11007", "gus_loud"),
    ("11008", "hattie_quiet"),
]

SAMPLE_MESSAGES = [
    "loving the run so far",
    "is this the bad ending?",
    "my cat Loki keeps walking on my keyboard lol",
    "have you tried the % run? way more chaotic",
    "this boss took me 4 hours when I ran it",
    "hi from melbourne btw, just got home from work",
    "I main keyboard, switched from controller last year",
    "you should react to that retrospective video",
    "I just got a new GPU, finally upgraded from a 1060",
    "RE6 was actually fun stop bullying it",
    "anyone else playing on a CRT? feels right for this game",
    "I work in support and this is my unwind stream",
    "got my dog bonk in my lap right now <3",
]

SAMPLE_NOTES = [
    "Has a cat named Loki.",
    "Speedruns RE2; PB roughly 1:35.",
    "Lives in Melbourne.",
    "Switched from controller to keyboard last year.",
    "Owns a dog named bonk.",
    "Recently upgraded GPU from a 1060.",
    "Works in tech support.",
    "Plays on a CRT for retro games.",
]


def main(db_path: str = "data/demo.db") -> None:
    Path(db_path).unlink(missing_ok=True)
    repo = ChatterRepo(db_path)
    rng = random.Random(42)
    base = datetime.now(timezone.utc) - timedelta(days=2)

    for tid, name in CHATTERS:
        repo.upsert_user(tid, name)
        n_msgs = rng.randint(3, 40)
        for _ in range(n_msgs):
            content = rng.choice(SAMPLE_MESSAGES)
            repo.insert_message(tid, content)
        for _ in range(rng.randint(0, 3)):
            note = rng.choice(SAMPLE_NOTES)
            repo.add_note(tid, note, embedding=None)

    # Events
    for _ in range(8):
        tid, name = rng.choice(CHATTERS)
        repo.record_event(
            twitch_name=name, event_type="tip", amount=rng.choice([5.0, 10.0, 25.0, 100.0]),
            currency="USD", message=rng.choice(["thanks for the stream!", None, "GG"]),
        )
    for _ in range(4):
        tid, name = rng.choice(CHATTERS)
        repo.record_event(
            twitch_name=name, event_type="cheer", amount=rng.choice([100, 500, 1000]),
        )
    for _ in range(3):
        tid, name = rng.choice(CHATTERS)
        repo.record_event(
            twitch_name=name, event_type="sub",
            amount=rng.choice([1, 3, 12]), currency="1000",
            message=rng.choice([None, "[gift from anon]"]),
        )
    repo.record_event(
        twitch_name="some_unknown_viewer", event_type="tip",
        amount=20.0, currency="USD", message="orphan tip",
    )
    repo.record_event(twitch_name="raid_friend", event_type="raid", amount=42)
    repo.record_event(twitch_name="newfollower42", event_type="follow")

    # Topic snapshots
    repo.add_topic_snapshot(
        "\u2022 Speedrun route discussion (bob_speedruns, alice_dev)\n"
        "\u2022 Cat / pet stories (evan_cat_dad, fern)\n"
        "\u2022 Hardware: GPU upgrades, CRTs (gus_loud, hattie_quiet)",
        "1-200",
    )
    repo.add_topic_snapshot(
        "\u2022 Boss strategy debate (charlie_modder, alice_dev)\n"
        "\u2022 Off-topic: weather (deidre_lurker)\n"
        "\u2022 New game suggestions (multiple)",
        "201-380",
    )

    print(f"seeded {db_path}")
    repo.close()


if __name__ == "__main__":
    main()
