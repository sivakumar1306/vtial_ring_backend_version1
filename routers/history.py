import asyncio

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from db.supabase import supabase
from services.chat_history import require_owned_conversation
from services.identity import resolve_user_id
from typing import Optional

router = APIRouter()

class SaveMessageRequest(BaseModel):
    user_id: str
    conversation_id: Optional[str] = None
    user_message: str
    assistant_reply: str
    title: Optional[str] = None

@router.post("/history/save")
async def save_message(
    request: SaveMessageRequest,
    authorization: str | None = Header(default=None),
):
    user_id = await resolve_user_id(
        authorization,
        request.user_id,
        allow_anonymous=False,
    )
    conversation_id = request.conversation_id
    try:
        if conversation_id:
            conversation_id = await require_owned_conversation(conversation_id, user_id)
        else:
            title = request.title or request.user_message[:50]

            def _insert_conversation():
                return supabase.table("conversations").insert({
                    "user_id": user_id,
                    "title": title,
                }).execute()

            conv = await asyncio.to_thread(_insert_conversation)
            conversation_id = conv.data[0]["id"]

        def _insert_both():
            supabase.table("messages").insert({
                "conversation_id": conversation_id,
                "role": "user",
                "content": request.user_message,
            }).execute()
            supabase.table("messages").insert({
                "conversation_id": conversation_id,
                "role": "assistant",
                "content": request.assistant_reply,
            }).execute()

        await asyncio.to_thread(_insert_both)
        return {"conversation_id": conversation_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not save this conversation") from exc

@router.get("/history/{user_id}")
async def get_conversations(
    user_id: str,
    authorization: str | None = Header(default=None),
):
    user_id = await resolve_user_id(authorization, user_id, allow_anonymous=False)

    def _query():
        return (
            supabase.table("conversations")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )

    try:
        result = await asyncio.to_thread(_query)
        return {"conversations": result.data or []}
    except Exception:
        return {"conversations": []}

@router.get("/history/messages/{conversation_id}")
async def get_messages(
    conversation_id: str,
    authorization: str | None = Header(default=None),
):
    user_id = await resolve_user_id(authorization, None, allow_anonymous=False)
    try:
        conversation_id = await require_owned_conversation(conversation_id, user_id)
    except HTTPException:
        raise

    def _query():
        return (
            supabase.table("messages")
            .select("*")
            .eq("conversation_id", conversation_id)
            .order("created_at")
            .execute()
        )

    try:
        result = await asyncio.to_thread(_query)
        return {"messages": result.data or []}
    except Exception:
        return {"messages": []}
