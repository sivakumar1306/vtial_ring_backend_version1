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
                if not conversation_id:
                    conv = supabase.table("conversations").insert({
                        "user_id": request.user_id,
                        "title": request.message[:50]
                    }).execute()
                    conversation_id = conv.data[0]["id"]

                supabase.table("messages").insert({
                    "conversation_id": conversation_id,
                    "role": "user",
                    "content": request.message
                }).execute()

                assistant_msg = {
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": reply
                }
                if card:
                    assistant_msg["card"] = card

                supabase.table("messages").insert(assistant_msg).execute()
        except Exception as e:
            print(f"[MedXAI] History save failed: {e}")

        return ChatResponse(reply=reply, intent="agent", conversation_id=conversation_id, card=card)

    except Exception as e:
        return ChatResponse(reply=f"Error: {str(e)}", intent="error")