"""Conversation ownership and prior turns for Assist."""

import asyncio
import uuid

from fastapi import HTTPException

from agent.biometrics import prior_turns
from db.supabase import supabase


def _parse_conversation_id(conversation_id: str) -> str:
    try:
        return str(uuid.UUID(conversation_id))
    except (TypeError, ValueError):
        raise HTTPException(status_code=404, detail="Conversation not found")


async def require_owned_conversation(conversation_id: str, user_id: str) -> str:
    """Return the id only when this user owns the conversation."""
    conversation_id = _parse_conversation_id(conversation_id)

    def _query():
        return (
            supabase.table("conversations")
            .select("user_id")
            .eq("id", conversation_id)
            .limit(1)
            .execute()
        )

    try:
        result = await asyncio.to_thread(_query)
    except Exception as exc:
        print(f"[MedXAI] conversation lookup failed: {type(exc).__name__}")
        raise HTTPException(status_code=404, detail="Conversation not found")

    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if str(rows[0].get("user_id")) != user_id:
        raise HTTPException(status_code=403, detail="You can only access your own data")
    return conversation_id


async def load_prior_turns(conversation_id: str, user_id: str) -> list[tuple[str, str]]:
    """Earlier turns for this user. Empty when history storage is unavailable."""
    try:
        conversation_id = await require_owned_conversation(conversation_id, user_id)
    except HTTPException:
        raise

    def _query():
        return (
            supabase.table("messages")
            .select("role,content")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=True)
            .limit(8)
            .execute()
        )

    try:
        result = await asyncio.to_thread(_query)
    except Exception as exc:
        print(f"[MedXAI] history load skipped: {type(exc).__name__}")
        return []
    rows = list(reversed(result.data or []))
    return prior_turns(rows)
