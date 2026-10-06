import asyncio
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agent.graph import run_agent
from db.supabase import supabase
from services.chat_history import require_owned_conversation
from services.identity import resolve_user_id
from typing import Optional, Any

load_dotenv()

router = APIRouter()

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    user_id: str = "anonymous"
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    intent: str = "agent"
    conversation_id: Optional[str] = None
    card: Optional[dict[str, Any]] = None

async def _save_turn(
    user_id: str,
    conversation_id: str | None,
    user_message: str,
    reply: str,
    card: dict | None,
) -> str | None:
    """Persist the turn. Returns the conversation id, or None if storage failed."""

    def _insert_conversation():
        return supabase.table("conversations").insert({
            "user_id": user_id,
            "title": user_message[:50],
        }).execute()

    if not conversation_id:
        conv = await asyncio.to_thread(_insert_conversation)
        conversation_id = conv.data[0]["id"]

    def _insert_user_message():
        return supabase.table("messages").insert({
            "conversation_id": conversation_id,
            "role": "user",
            "content": user_message,
        }).execute()

    await asyncio.to_thread(_insert_user_message)

    assistant_msg = {
        "conversation_id": conversation_id,
        "role": "assistant",
        "content": reply,
    }
    if card:
        assistant_msg["card"] = card

    def _insert_assistant_message():
        return supabase.table("messages").insert(assistant_msg).execute()

    await asyncio.to_thread(_insert_assistant_message)
    return conversation_id

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, authorization: str | None = Header(default=None)):
    # The token decides who this is. A body user_id for someone else is rejected
    # before any biometric row is read. No token stays anonymous and unsaved.
    user_id = await resolve_user_id(
        authorization,
        request.user_id,
        allow_anonymous=True,
    )
    print(f"[MedXAI] user: {user_id} | chars: {len(request.message)}")

    conversation_id = request.conversation_id
    if user_id == "anonymous":
        conversation_id = None
    elif conversation_id:
        try:
            conversation_id = await require_owned_conversation(conversation_id, user_id)
        except HTTPException as exc:
            if exc.status_code == 403:
                raise
            # Unknown id, or history tables not created yet: answer anyway
            # and start a fresh conversation on save.
            conversation_id = None

    try:
        reply, card = await run_agent(request.message, user_id, conversation_id)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[MedXAI] chat failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail="The assistant is unavailable right now. Please try again.",
        )

    if user_id != "anonymous":
        try:
            conversation_id = await _save_turn(
                user_id, conversation_id, request.message, reply, card
            )
        except Exception as exc:
            print(f"[MedXAI] History save failed: {type(exc).__name__}")

    return ChatResponse(
        reply=reply,
        intent="agent",
        conversation_id=conversation_id,
        card=card,
    )
