"""Resolve the caller from the Supabase access token the app already sends.

The API uses the service-role key, which bypasses row-level security.
The body or path user id is never trusted on its own.
"""

import asyncio

from fastapi import HTTPException

from db.supabase import supabase


async def resolve_user_id(
    authorization: str | None,
    claimed_user_id: str | None,
    *,
    allow_anonymous: bool,
) -> str:
    """Return the user id this request may read or write.

    No token: anonymous, and only when the route allows it.
    A token is checked with Supabase. A claimed id other than that user
    is rejected, so a caller cannot pull someone else's ring data.
    """
    header = (authorization or "").strip()
    if not header:
        if allow_anonymous:
            return "anonymous"
        raise HTTPException(status_code=401, detail="Sign in required")

    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token_user = await _user_id_for_token(token.strip())
    claimed = (claimed_user_id or "").strip()
    if claimed and claimed not in ("anonymous", token_user):
        raise HTTPException(status_code=403, detail="You can only access your own data")
    return token_user


async def _user_id_for_token(token: str) -> str:
    def _lookup():
        return supabase.auth.get_user(token)

    try:
        result = await asyncio.to_thread(_lookup)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = getattr(result, "user", None)
    user_id = getattr(user, "id", None) if user is not None else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    return str(user_id)
