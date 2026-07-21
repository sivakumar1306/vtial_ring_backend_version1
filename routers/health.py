from fastapi import APIRouter
from db.supabase import supabase

router = APIRouter()

@router.get("/health-data/{user_id}")
async def get_health_data(user_id: str):
    try:
        # latest reading
        latest = supabase.table("ring_data")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("recorded_at", desc=True)\
            .limit(1)\
            .execute()

        # last 7 days for weekly avg
        weekly = supabase.table("ring_data")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("recorded_at", desc=True)\
            .limit(7)\
            .execute()

        weekly_avg = None
        if weekly.data:
            days = weekly.data
            weekly_avg = {
                "avg_sleep": sum(d.get("sleep_duration_minutes", 0) for d in days) / len(days),
                "avg_hr": sum(d.get("heart_rate_avg", 0) for d in days) / len(days),
                "avg_steps": sum(d.get("steps", 0) for d in days) / len(days),
                "avg_spo2": sum(d.get("spo2_avg", 0) for d in days) / len(days),
                "avg_sleep_score": sum(d.get("sleep_score", 0) for d in days) / len(days),
            }

        return {
            "latest": latest.data[0] if latest.data else None,
            "weekly_avg": weekly_avg,
        }
    except Exception as e:
        return {"error": str(e)}