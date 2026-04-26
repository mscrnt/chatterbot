"""Backfill chatters.db with Twitch badge state via Helix.

Pulls VIPs, moderators, and subscribers from Helix and merges them into the
users table. Each endpoint requires a scope on TWITCH_OAUTH_TOKEN; missing
scopes are reported as warnings and skipped — the rest still applies.

Required scopes (all on the broadcaster's own token):
  channel:read:vips           → /helix/channels/vips
  moderation:read             → /helix/moderation/moderators
  channel:read:subscriptions  → /helix/subscriptions

Idempotent: re-running just refreshes the snapshot.

Usage (from repo root):
  uv run python scripts/backfill_badges.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chatterbot.config import get_settings  # noqa: E402
from chatterbot.repo import ChatterRepo  # noqa: E402

HELIX = "https://api.twitch.tv/helix"
VALIDATE = "https://id.twitch.tv/oauth2/validate"


def _validate(token: str) -> tuple[str, str, list[str]] | None:
    """Returns (client_id, login, scopes) or None on failure."""
    r = httpx.get(VALIDATE, headers={"Authorization": f"OAuth {token}"}, timeout=8.0)
    if r.status_code != 200:
        print(f"  ✗ token validate returned {r.status_code}")
        return None
    j = r.json() or {}
    return j.get("client_id"), j.get("login"), j.get("scopes") or []


def _get_broadcaster_id(client: httpx.Client, login: str) -> str | None:
    r = client.get(f"{HELIX}/users", params={"login": login})
    r.raise_for_status()
    data = r.json().get("data") or []
    return data[0]["id"] if data else None


def _paged(client: httpx.Client, path: str, params: dict) -> list[dict]:
    """Walk a Helix cursor-paginated endpoint, returning all items."""
    out: list[dict] = []
    cursor: str | None = None
    while True:
        p = dict(params)
        if cursor:
            p["after"] = cursor
        r = client.get(f"{HELIX}{path}", params=p)
        if r.status_code in (401, 403):
            scope_err = (r.json() or {}).get("message", r.text)
            raise PermissionError(scope_err)
        r.raise_for_status()
        j = r.json()
        out.extend(j.get("data") or [])
        cursor = (j.get("pagination") or {}).get("cursor")
        if not cursor:
            break
    return out


def main() -> int:
    settings = get_settings()
    repo = ChatterRepo(settings.db_path)

    raw = (settings.twitch_oauth_token or "").removeprefix("oauth:").strip()
    if not raw:
        print("✗ TWITCH_OAUTH_TOKEN is empty — nothing to do.")
        return 1

    print("→ validating token…")
    v = _validate(raw)
    if not v:
        return 1
    client_id, login, scopes = v
    print(f"  ok — login={login}, scopes={','.join(scopes) or '(none)'}")

    headers = {"Authorization": f"Bearer {raw}", "Client-Id": client_id}
    with httpx.Client(headers=headers, timeout=10.0) as client:
        bid = _get_broadcaster_id(client, login)
        if not bid:
            print(f"✗ couldn't resolve broadcaster_id for login={login}")
            return 1
        print(f"  broadcaster_id={bid}")

        vip_ids: dict[str, str] = {}
        mod_ids: dict[str, str] = {}
        sub_rows: dict[str, dict] = {}

        # VIPs — { user_id, user_login, user_name }
        if "channel:read:vips" in scopes or "moderator:read:vips" in scopes:
            try:
                rows = _paged(
                    client, "/channels/vips",
                    {"broadcaster_id": bid, "first": 100},
                )
                vip_ids = {r["user_id"]: r["user_login"] for r in rows}
                print(f"  ✓ VIPs: {len(vip_ids)}")
            except PermissionError as e:
                print(f"  ✗ VIPs blocked by scope: {e}")
        else:
            print("  · skipping VIPs (need channel:read:vips)")

        # Moderators
        if "moderation:read" in scopes or "channel:manage:moderators" in scopes:
            try:
                rows = _paged(
                    client, "/moderation/moderators",
                    {"broadcaster_id": bid, "first": 100},
                )
                mod_ids = {r["user_id"]: r["user_login"] for r in rows}
                print(f"  ✓ Moderators: {len(mod_ids)}")
            except PermissionError as e:
                print(f"  ✗ Moderators blocked by scope: {e}")
        else:
            print("  · skipping Moderators (need moderation:read)")

        # Subscribers — also gives us tier + plan_name
        if "channel:read:subscriptions" in scopes:
            try:
                rows = _paged(
                    client, "/subscriptions",
                    {"broadcaster_id": bid, "first": 100},
                )
                # Normalize tier: Helix returns '1000' / '2000' / '3000' / 'Prime'
                for r in rows:
                    sub_rows[r["user_id"]] = {
                        "login": r["user_login"],
                        "tier": str(r.get("tier") or ""),
                        # Note: /subscriptions doesn't return cumulative months;
                        # only the broadcaster's own /helix/subscriptions/user
                        # call does, and per-user. Leave months at 0 here so
                        # we don't overwrite higher counts already learned
                        # from chat IRC tags.
                    }
                print(f"  ✓ Subs: {len(sub_rows)}")
            except PermissionError as e:
                print(f"  ✗ Subs blocked by scope: {e}")
        else:
            print("  · skipping Subs (need channel:read:subscriptions)")

    if not (vip_ids or mod_ids or sub_rows):
        print("\nNothing to apply — your token has no badge-relevant scopes.")
        print(
            "Generate a new token at https://twitchtokengenerator.com with the\n"
            "scopes listed in the script header, then re-run."
        )
        return 0

    # Union the user_ids we touched. For each, also grab the login so we can
    # upsert into users (Helix gives both).
    by_id: dict[str, str] = {}
    by_id.update(vip_ids)
    by_id.update(mod_ids)
    for uid, info in sub_rows.items():
        by_id[uid] = info["login"]

    print(f"\n→ applying badges to {len(by_id)} chatters…")
    touched_existing = 0
    inserted = 0
    for uid, uname in by_id.items():
        existed = repo.get_user(uid) is not None
        # Upsert so users we've never had chat from still get a row. The
        # bot's next message from them will refresh first/last_seen accurately.
        repo.upsert_user(uid, uname)
        # Founder isn't returned by Helix; preserve whatever's already on file.
        existing = repo.get_user(uid)
        repo.update_user_badges(
            uid,
            sub_tier=(sub_rows.get(uid) or {}).get("tier") or existing.sub_tier,
            sub_months=existing.sub_months,
            is_mod=(uid in mod_ids) or existing.is_mod,
            is_vip=(uid in vip_ids) or existing.is_vip,
            is_founder=existing.is_founder,
        )
        if existed:
            touched_existing += 1
        else:
            inserted += 1

    print(f"  ✓ updated {touched_existing} existing · {inserted} newly inserted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
