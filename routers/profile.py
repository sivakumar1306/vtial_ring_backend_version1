import asyncio

from fastapi import APIRouter, Header
from pydantic import BaseModel
from typing import List, Optional
from db.supabase import supabase
from services.identity import resolve_user_id

router = APIRouter()

class ProfileRequest(BaseModel):
    user_id: str
    name: Optional[str] = ""
    age: Optional[int] = None
    gender: Optional[str] = ""
    blood_type: Optional[str] = ""
    conditions: Optional[List[str]] = []
    medications: Optional[List[str]] = []
    allergies: Optional[List[str]] = []

@router.post("/profile")
async def save_profile(request: ProfileRequest, authorization: str | None = Header(default=None)):
    user_id = await resolve_user_id(authorization, request.user_id, allow_anonymous=False)
    data = {
        "user_id": user_id,
        "age": request.age,
        "blood_type": request.blood_type,
        "conditions": request.conditions,
        "medications": request.medications,
        "allergies": request.allergies,
    }

    def _save():
        existing = supabase.table("health_profiles")\
            .select("id")\
            .eq("user_id", user_id)\
            .execute()
        if existing.data:
            supabase.table("health_profiles")\
                .update(data)\
                .eq("user_id", user_id)\
                .execute()
        else:
            supabase.table("health_profiles")\
                .insert(data)\
                .execute()

    try:
        await asyncio.to_thread(_save)
        return {"message": "Profile saved successfully"}
    except Exception:
        return {"error": "Could not save profile"}