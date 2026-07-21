from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from db.supabase import supabase

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
async def save_profile(request: ProfileRequest):
    try:
        existing = supabase.table("health_profiles")\
            .select("id")\
            .eq("user_id", request.user_id)\
            .execute()

        data = {
            "user_id": request.user_id,
            "age": request.age,
            "blood_type": request.blood_type,
            "conditions": request.conditions,
            "medications": request.medications,
            "allergies": request.allergies,
        }

        if existing.data:
            supabase.table("health_profiles")\
                .update(data)\
                .eq("user_id", request.user_id)\
                .execute()
        else:
            supabase.table("health_profiles")\
                .insert(data)\
                .execute()

        return {"message": "Profile saved successfully"}
    except Exception as e:
        return {"error": str(e)}