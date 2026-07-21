from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from db.supabase import supabase
from typing import Optional
import uuid

router = APIRouter()

class SaveMessageRequest(BaseModel):
    user_id: str
    conversation_id: Optional[str] = None
    user_message: str
    assistant_reply: str
    title: Optional[str] = None

@router.post("/history/save")
async def save_message(request: SaveMessageRequest):
    try:
        conversation_id = request.conversation_id

        # create new conversation if no id provided
        if not conversation_id:
            title = request.title or request.user_message[:50]
            conv = supabase.table("conversations").insert({
                "user_id": request.user_id,
                "title": title
            }).execute()
            conversation_id = conv.data[0]["id"]

        # save user message
        supabase.table("messages").insert({
            "conversation_id": conversation_id,
            "role": "user",
            "content": request.user_message
        }).execute()

        # save assistant reply
        supabase.table("messages").insert({
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": request.assistant_reply
        }).execute()

        return {"conversation_id": conversation_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/history/{user_id}")
async def get_conversations(user_id: str):
    try:
        result = supabase.table("conversations")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .execute()
        return {"conversations": result.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/history/messages/{conversation_id}")
async def get_messages(conversation_id: str):
    try:
        result = supabase.table("messages")\
            .select("*")\
            .eq("conversation_id", conversation_id)\
            .order("created_at")\
            .execute()
        return {"messages": result.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))