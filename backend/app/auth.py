"""Password login backed by a signed, HTTP-only session cookie.

The app is for two people who share one password, so there is no user table.
Whoever logs in picks a display name; that name scopes their progress, drafts,
and run history.
"""

import hmac
import time

from fastapi import Depends, HTTPException, Request, Response, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from .config import Settings, get_settings

COOKIE_NAME = "li_session"


def _serializer(settings: Settings) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.session_secret, salt="learn-inference")


def verify_password(candidate: str, settings: Settings) -> bool:
    return hmac.compare_digest(candidate, settings.app_password)


def issue_session(response: Response, name: str, settings: Settings) -> None:
    token = _serializer(settings).dumps({"name": name, "at": time.time()})
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=settings.session_days * 86400,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )


def clear_session(response: Response, settings: Settings) -> None:
    response.delete_cookie(COOKIE_NAME, path="/", samesite="lax",
                           secure=settings.cookie_secure, httponly=True)


def read_session(request: Request, settings: Settings) -> str | None:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    try:
        data = _serializer(settings).loads(token, max_age=settings.session_days * 86400)
    except (BadSignature, SignatureExpired):
        return None
    name = data.get("name")
    return name if isinstance(name, str) and name else None


def current_user(
    request: Request, settings: Settings = Depends(get_settings)
) -> str:
    """FastAPI dependency that returns the display name or raises 401."""
    name = read_session(request, settings)
    if not name:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not signed in"
        )
    return name
