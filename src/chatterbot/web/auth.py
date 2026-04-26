"""Optional HTTP basic auth for the dashboard.

Disabled when DASHBOARD_BASIC_AUTH_USER/PASS are blank. The dashboard binds to
127.0.0.1 by default, so auth is only needed when the streamer wants to expose
it on LAN (e.g. to view from a phone).
"""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from ..config import Settings


_security = HTTPBasic(auto_error=False)


def make_auth_dependency(settings: Settings):
    if not settings.dashboard_basic_auth_enabled:
        async def _noop() -> None:
            return None
        return _noop

    expected_user = settings.dashboard_basic_auth_user
    expected_pass = settings.dashboard_basic_auth_pass

    async def _check(
        creds: HTTPBasicCredentials | None = Depends(_security),
    ) -> None:
        if creds is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="auth required",
                headers={"WWW-Authenticate": "Basic"},
            )
        ok_user = secrets.compare_digest(creds.username, expected_user)
        ok_pass = secrets.compare_digest(creds.password, expected_pass)
        if not (ok_user and ok_pass):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="bad credentials",
                headers={"WWW-Authenticate": "Basic"},
            )

    return _check
