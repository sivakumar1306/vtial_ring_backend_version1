from langchain_core.tools import tool
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from db.supabase import supabase

# All measured_at / date-ish timestamps from Supabase are stored in UTC.
# Convert to IST before handing them to the LLM so replies show the user's
# actual local time instead of raw UTC (e.g. "15:51" showing when it's
# really 21:21 for the user).
def _to_ist(iso_str) -> str:
    if not iso_str:
        return "unknown time"
    try:
        dt = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
        return dt.astimezone(ZoneInfo("Asia/Kolkata")).strftime("%H:%M on %d %B %Y")
    except Exception:
        return str(iso_str)

@tool
def search_medical_knowledge(query: str) -> str:
    """Search the medical knowledge base for information about symptoms, conditions, treatments, and medications. Automatically expands knowledge base if needed."""
    try:
        import asyncio
        from services.rag import retrieve_context_with_expansion
        results = asyncio.run(retrieve_context_with_expansion(query, match_count=3))
        if not results:
            return "No relevant medical information found."
        output = ""
        for r in results:
            output += f"[Source: {r['source']}]\n{r['content']}\n\n"
        return output.strip()
    except Exception as e:
        return f"Medical knowledge search failed: {str(e)}"

@tool
def get_patient_data(user_id: str) -> str:
    """Get the patient's health profile and recent smart ring biometric data. Input should be the user's UUID string."""
    try:
        user_id = user_id.strip().strip('"').strip("'")
        context = ""

        profile = supabase.table("user_profiles")\
            .select("*")\
            .eq("id", user_id)\
            .maybe_single()\
            .execute()
        if profile.data:
            p = profile.data
            context += f"""
PATIENT PROFILE:
- Name: {p.get('full_name', 'unknown')}
- Age: {p.get('age', 'unknown')}
- Gender: {p.get('gender', 'unknown')}
- Height: {p.get('height_cm', 'unknown')} cm
- Weight: {p.get('weight_kg', 'unknown')} kg
"""

        def latest(table, order_col="date"):
            r = supabase.table(table)\
                .select("*")\
                .eq("user_id", user_id)\
                .order(order_col, desc=True)\
                .limit(3)\
                .execute()
            return r.data or []

        # Live/current reading — from the raw per-measurement table, NOT the daily
        # aggregate (user_hr only has avg/min/max per day, no single "current" value).
        # Isolated in its own try/except so a failure here can't wipe out the rest
        # of the patient context (profile, sleep, etc.) via the outer except below.
        try:
            current_hr = supabase.table("user_hr_readings")\
                .select("*")\
                .eq("user_id", user_id)\
                .order("measured_at", desc=True)\
                .limit(1)\
                .execute()
            if current_hr.data:
                c = current_hr.data[0]
                measured_at_str = c.get("measured_at")
                staleness_note = ""
                try:
                    measured_dt = datetime.fromisoformat(measured_at_str.replace("Z", "+00:00"))
                    age_hours = (datetime.now(timezone.utc) - measured_dt).total_seconds() / 3600
                    if age_hours > 3:
                        staleness_note = f" [STALE: this reading is {age_hours:.1f} hours old, NOT real-time]"
                except Exception:
                    pass
                measured_at_local = _to_ist(measured_at_str)
                context += f"\nCURRENT HEART RATE (most recent single reading in DB): {c.get('value_bpm')} bpm, measured at {measured_at_local}{staleness_note}\n"
            else:
                context += "\nCURRENT HEART RATE: NO reading found in database for this user.\n"
        except Exception as e:
            context += f"\nCURRENT HEART RATE: query failed ({str(e)}). Do NOT guess a value.\n"

        sleep = latest("user_sleep")
        if sleep:
            context += "\nRECENT SLEEP:\n"
            for day in reversed(sleep):
                total_min = day.get("total_duration") or 0
                context += f"- {day.get('date')}: {total_min // 60}h {total_min % 60}m, score {day.get('sleep_score')}/100\n"

        hr = latest("user_hr")
        if hr:
            context += "\nHISTORICAL DAILY HEART RATE (NOT the current/live reading):\n"
            for day in reversed(hr):
                context += f"- {day.get('date')}: avg {day.get('avg_hr')} bpm (min {day.get('min_hr')}, max {day.get('max_hr')})\n"

        hrv = latest("user_hrv")
        if hrv:
            context += "\nRECENT HRV:\n"
            for day in reversed(hrv):
                context += f"- {day.get('date')}: avg {day.get('avg_hrv')} ms\n"

        spo2 = latest("user_spo2")
        if spo2:
            context += "\nRECENT SPO2:\n"
            for day in reversed(spo2):
                context += f"- {day.get('date')}: avg {day.get('avg_spo2')}%\n"

        steps = latest("user_steps")
        if steps:
            context += "\nRECENT STEPS:\n"
            for day in reversed(steps):
                context += f"- {day.get('date')}: {day.get('steps')} steps, {day.get('calories')} kcal\n"

        bp = supabase.table("user_bp")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("measured_at", desc=True)\
            .limit(1)\
            .execute()
        if bp.data:
            b = bp.data[0]
            context += f"\nLATEST BLOOD PRESSURE: {b.get('systolic')}/{b.get('diastolic')} (measured {_to_ist(b.get('measured_at'))})\n"

        result = context.strip() if context else "No patient data found."
        print(f"[get_patient_data] user_id={user_id}\n---TOOL OUTPUT SENT TO LLM---\n{result}\n---END TOOL OUTPUT---")
        return result
    except Exception as e:
        error_msg = f"Failed to fetch patient data: {str(e)}"
        print(f"[get_patient_data] ERROR for user_id={user_id}: {error_msg}")
        return error_msg

@tool
def check_emergency(message: str) -> str:
    """Check if the message contains emergency or life-threatening symptoms that require immediate medical attention."""
    emergency_keywords = [
        "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
        "heart attack", "stroke", "unconscious", "unresponsive", "seizure",
        "severe bleeding", "overdose", "suicidal", "suicide", "kill myself",
        "severe headache", "sudden confusion", "face drooping", "arm weakness",
        "slurred speech", "severe allergic", "anaphylaxis", "stopped breathing"
    ]
    msg_lower = message.lower()
    triggered = [kw for kw in emergency_keywords if kw in msg_lower]
    if triggered:
        return f"EMERGENCY DETECTED: {', '.join(triggered)}. This requires IMMEDIATE medical attention. Call emergency services (112 in India) or go to the nearest emergency room NOW. Do not wait."
    return "No emergency detected."

@tool
def analyze_symptoms(symptoms: str) -> str:
    """Analyze a list of symptoms and return a structured breakdown with possible conditions to investigate."""
    try:
        common_patterns = {
            "thirst,urination,fatigue,blurry vision": "Pattern suggests possible blood sugar issues — consider diabetes screening",
            "chest pain,shortness of breath,sweating": "Pattern suggests possible cardiac issue — seek immediate evaluation",
            "fever,cough,sore throat,runny nose": "Pattern consistent with upper respiratory infection",
            "headache,fever,stiff neck,sensitivity to light": "Pattern warrants urgent evaluation — possible meningitis",
            "fatigue,weight gain,cold intolerance,dry skin": "Pattern suggests possible thyroid dysfunction",
            "anxiety,rapid heartbeat,sweating,trembling": "Pattern consistent with anxiety or panic disorder",
        }
        symptom_lower = symptoms.lower()
        analysis = f"Symptoms reported: {symptoms}\n\n"
        matched = False
        for pattern, suggestion in common_patterns.items():
            pattern_words = pattern.split(",")
            if sum(1 for word in pattern_words if word in symptom_lower) >= 2:
                analysis += f"⚠️ {suggestion}\n"
                matched = True
        if not matched:
            analysis += "No strong pattern match found. Recommend consulting a healthcare provider for proper evaluation.\n"
        analysis += "\n⚠️ This is not a diagnosis. Always consult a qualified medical professional."
        return analysis
    except Exception as e:
        return f"Symptom analysis failed: {str(e)}"