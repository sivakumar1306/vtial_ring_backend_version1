import asyncio
from fastapi import APIRouter
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agent.graph import run_agent
from db.supabase import supabase
from typing import Optional, Any

# Supabase database migration note:
# ALTER TABLE messages ADD COLUMN IF NOT EXISTS card jsonb DEFAULT NULL;

load_dotenv()

router = APIRouter()

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=1000)
    user_id: str = "anonymous"
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    intent: str = "agent"
    conversation_id: Optional[str] = None
    card: Optional[dict[str, Any]] = None

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        print(f"[MedXAI] user: {request.user_id} | message: {request.message[:60]}")
        reply, card = await run_agent(request.message, request.user_id)

        # save to conversation history
        conversation_id = request.conversation_id
        try:
            if request.user_id != "anonymous":
                # Each of these is a blocking supabase.execute() call, run
                # directly inside this async route — exactly like the
                # /insights endpoint before its fix, this froze FastAPI's
                # entire event loop (and every other concurrent user's
                # request) for the duration of each insert. Wrapping in
                # asyncio.to_thread moves them to a background thread so the
                # event loop stays free.
                if not conversation_id:
                    def _insert_conversation():
                        return supabase.table("conversations").insert({
                            "user_id": request.user_id,
                            "title": request.message[:50]
                        }).execute()

                    conv = await asyncio.to_thread(_insert_conversation)
                    conversation_id = conv.data[0]["id"]

                def _insert_user_message():
                    return supabase.table("messages").insert({
                        "conversation_id": conversation_id,
                        "role": "user",
                        "content": request.message
                    }).execute()

                await asyncio.to_thread(_insert_user_message)

                assistant_msg = {
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": reply
                }
                if card:
                    assistant_msg["card"] = card

                def _insert_assistant_message():
                    return supabase.table("messages").insert(assistant_msg).execute()

                await asyncio.to_thread(_insert_assistant_message)
        except Exception as e:
            print(f"[MedXAI] History save failed: {e}")

        return ChatResponse(reply=reply, intent="agent", conversation_id=conversation_id, card=card)

    except Exception as e:
        return ChatResponse(reply=f"Error: {str(e)}", intent="error")