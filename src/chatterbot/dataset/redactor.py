"""Export-time redactor.

Scrubs chatter usernames out of decrypted event payloads at export
time so a streamer can hand the resulting bundle to a fine-tuning
service without leaking chatter identities. Capture-time data stays
verbatim — keeping originals lets the streamer change their mind
about what to share later.

Scope (intentionally narrow for v1):

  - Only redacts names that the capture event itself declared via
    `referenced_user_ids`. The opposite policy (heuristic name
    matching across the whole prompt) is fragile — common nicknames
    like "alice" appearing in unrelated chat would false-match. We
    can add a more aggressive mode later.
  - Replaces every alias of each declared user_id (current name +
    `user_aliases` history) with a stable per-bundle token like
    `<USER_001>`. Tokens are renumbered per export so two exports
    of the same data produce different mappings.
  - Patches text fields: `prompt`, `system_prompt`, `response_text`,
    plus any nested string list inside snapshot events.
  - Replaces the user_id itself in `referenced_user_ids` so the
    payload no longer round-trips back to the chatters table.

Manifest gets an explicit `redacted: true` so a downstream consumer
can refuse-to-train on un-redacted data if their policy requires it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..repo import ChatterRepo


@dataclass
class RedactionPlan:
    """Per-export anonymisation plan: maps every covered user_id +
    every alias to a stable anon token. Built once per export, used
    on every event."""
    # user_id → "<USER_001>"
    id_to_token: dict[str, str]
    # case-folded name → "<USER_001>". Includes every alias for
    # every covered user.
    name_to_token: dict[str, str]
    # compiled regex matching any covered name (word-boundary, case
    # insensitive) — used to replace inline mentions in prose.
    name_pattern: re.Pattern[str] | None


def build_plan(repo: "ChatterRepo", user_ids: list[str]) -> RedactionPlan:
    """Resolve every name + alias for the given user_ids into a
    stable token map. Tokens are deterministic for one export run
    so the same chatter shows the same token across every event in
    the bundle."""
    if not user_ids:
        return RedactionPlan(
            id_to_token={}, name_to_token={}, name_pattern=None,
        )

    # Sort for determinism within one export — same list of users
    # always produces the same token assignment.
    unique_ids = sorted({u for u in user_ids if u})
    id_to_token: dict[str, str] = {}
    name_to_token: dict[str, str] = {}
    all_names: list[str] = []

    for idx, uid in enumerate(unique_ids, start=1):
        token = f"<USER_{idx:03d}>"
        id_to_token[uid] = token
        # Every alias for this user maps to the same token. Aliases
        # are case-insensitive in the matcher; we store them lower-
        # cased here.
        try:
            with repo._cursor() as cur:  # noqa: SLF001 — internal redactor
                cur.execute(
                    "SELECT name FROM user_aliases WHERE user_id = ?",
                    (uid,),
                )
                aliases = [r["name"] for r in cur.fetchall()]
        except Exception:
            aliases = []
        # Also pull the canonical name from users in case it's not
        # in aliases (defensive — upsert_user always adds it but a
        # bug elsewhere shouldn't leak names).
        try:
            user = repo.get_user(uid)
            if user and user.name and user.name not in aliases:
                aliases.append(user.name)
        except Exception:
            pass
        for name in aliases:
            if not name:
                continue
            name_to_token[name.lower()] = token
            all_names.append(re.escape(name))

    # Build one big alternation pattern, sorted longest-first so
    # "alice_streamer" matches before "alice" doesn't strip the
    # prefix of the longer name.
    pattern: re.Pattern[str] | None = None
    if all_names:
        all_names.sort(key=len, reverse=True)
        # Word boundary on both sides — won't mangle e.g. "alicelike"
        # when redacting "alice". Twitch usernames allow only
        # [a-zA-Z0-9_], so \b is the right boundary.
        pattern = re.compile(
            r"\b(" + "|".join(all_names) + r")\b",
            re.IGNORECASE,
        )

    return RedactionPlan(
        id_to_token=id_to_token,
        name_to_token=name_to_token,
        name_pattern=pattern,
    )


def redact_text(plan: RedactionPlan, text: str) -> str:
    """Replace every covered name in `text` with its anon token.
    Single regex pass; case-insensitive; preserves surrounding
    punctuation. No-op when the plan has no names."""
    if not text or plan.name_pattern is None:
        return text

    def _sub(m: re.Match[str]) -> str:
        return plan.name_to_token.get(m.group(0).lower(), m.group(0))

    return plan.name_pattern.sub(_sub, text)


def redact_event(plan: RedactionPlan, event: dict[str, Any]) -> dict[str, Any]:
    """Apply the redaction plan to one decoded event payload.
    Returns a NEW dict — caller's input is not mutated. Skips
    events the plan can't act on (e.g. snapshots whose chatters
    aren't in the plan's user_ids list)."""
    if not plan.id_to_token and not plan.name_to_token:
        return event

    out = dict(event)

    # 1) Replace user_ids in `referenced_user_ids` with their tokens.
    refs = out.get("referenced_user_ids")
    if isinstance(refs, list) and refs:
        out["referenced_user_ids"] = [
            plan.id_to_token.get(u, u) for u in refs
        ]

    # 2) Patch every text-bearing field. Skip-safe — missing keys
    # stay missing.
    for field in ("prompt", "system_prompt", "response_text", "note"):
        v = out.get(field)
        if isinstance(v, str) and v:
            out[field] = redact_text(plan, v)

    # 3) STREAMER_ACTION events: redact item_key only when the
    # action_kind is "note" or "engaging_subject" — those pass user-
    # facing strings; insight_state's keys are internal IDs.
    action_kind = out.get("action_kind")
    if action_kind in ("note", "engaging_subject"):
        ik = out.get("item_key")
        if isinstance(ik, str):
            out["item_key"] = redact_text(plan, ik)

    # 4) CONTEXT_SNAPSHOT events: redact chatter names inside the
    # nested snapshot dict. We touch only the obvious string fields
    # so the structure round-trips cleanly.
    snapshot = out.get("snapshot")
    if isinstance(snapshot, dict):
        new_snapshot = dict(snapshot)
        msgs = new_snapshot.get("messages")
        if isinstance(msgs, list):
            new_msgs = []
            for m in msgs:
                if not isinstance(m, dict):
                    new_msgs.append(m)
                    continue
                m2 = dict(m)
                if "user_id" in m2 and m2["user_id"] in plan.id_to_token:
                    m2["user_id"] = plan.id_to_token[m2["user_id"]]
                if "name" in m2 and isinstance(m2["name"], str):
                    m2["name"] = plan.name_to_token.get(
                        m2["name"].lower(), m2["name"],
                    )
                if "content" in m2 and isinstance(m2["content"], str):
                    m2["content"] = redact_text(plan, m2["content"])
                if "reply_parent_login" in m2 and isinstance(
                    m2["reply_parent_login"], str,
                ):
                    m2["reply_parent_login"] = plan.name_to_token.get(
                        m2["reply_parent_login"].lower(),
                        m2["reply_parent_login"],
                    )
                new_msgs.append(m2)
            new_snapshot["messages"] = new_msgs
        # Threads carry a `drivers` list of names — redact each.
        threads = new_snapshot.get("threads")
        if isinstance(threads, list):
            new_threads = []
            for t in threads:
                if not isinstance(t, dict):
                    new_threads.append(t)
                    continue
                t2 = dict(t)
                drivers = t2.get("drivers")
                if isinstance(drivers, list):
                    t2["drivers"] = [
                        plan.name_to_token.get(d.lower(), d)
                        if isinstance(d, str) else d
                        for d in drivers
                    ]
                if isinstance(t2.get("recap"), str):
                    t2["recap"] = redact_text(plan, t2["recap"])
                new_threads.append(t2)
            new_snapshot["threads"] = new_threads
        out["snapshot"] = new_snapshot

    return out


def collect_user_ids_in_events(events: list[dict[str, Any]]) -> list[str]:
    """Walk a batch of events and return every user_id we'd want to
    redact. Used by `cmd_export` to build the plan in one pass
    before re-walking events to apply it."""
    seen: set[str] = set()
    for ev in events:
        refs = ev.get("referenced_user_ids")
        if isinstance(refs, list):
            for u in refs:
                if isinstance(u, str) and u:
                    seen.add(u)
        # Snapshot events list user_ids inline.
        snap = ev.get("snapshot")
        if isinstance(snap, dict):
            for m in snap.get("messages") or []:
                if isinstance(m, dict):
                    uid = m.get("user_id")
                    if isinstance(uid, str) and uid:
                        seen.add(uid)
    return sorted(seen)
