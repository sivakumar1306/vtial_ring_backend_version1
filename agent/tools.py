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

# IMPORTANT: this must stay `async def`, not a sync function wrapped in
# asyncio.run(). rag.py's retrieve_context_with_expansion() fires off a
# background asyncio.create_task() to expand the knowledge base without
# blocking the user's reply. asyncio.run() creates a brand-new event loop,
# runs the coroutine to completion, and immediately DESTROYS that loop the
# instant it returns — which cancels any task scheduled on it that hasn't
# finished yet. That would silently kill the background expansion task
# before it ever did any work, completely defeating that fix (and likely
# print "Task was destroyed but it is pending!" warnings in the logs).
# By making this an async tool, LangChain's agent.ainvoke() awaits it
# directly on the real, persistent FastAPI/uvicorn event loop instead — so
# the background task actually survives and completes after this call returns.
@tool
async def search_medical_knowledge(query: str) -> str:
    """Search the medical knowledge base for information about symptoms, conditions, treatments, and medications. Automatically expands knowledge base if needed."""
    try:
        from services.rag import retrieve_context_with_expansion
        results = await retrieve_context_with_expansion(query, match_count=3)
        if not results:
            return "No relevant medical information found."
        output = ""
        for r in results:
            output += f"[Source: {r['source']}]\n{r['content']}\n\n"
        return output.strip()
    except Exception as e:
        return f"Medical knowledge search failed: {str(e)}"

# Note on this file's blocking supabase.table(...).execute() calls (below):
# get_patient_data is a synchronous @tool. When the agent is invoked via
# agent.ainvoke() (see graph.py's run_agent), LangChain automatically runs
# sync tool functions in a background thread pool rather than on the main
# event loop — so these blocking DB calls do NOT freeze other concurrent
# FastAPI requests the way the un-wrapped calls in health.py's /insights
# route previously did (those were called directly inside an async route
# handler, with nothing offloading them to a thread). No asyncio wrapping is
# needed here; left as plain synchronous Supabase calls, matching how @tool
# functions are meant to be written. (search_medical_knowledge above is the
# one exception that needed to be async, for the reason explained there.)
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
                .neq("source", "demo_seed")\
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

        temp = supabase.table("user_temp")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("measured_at", desc=True)\
            .limit(1)\
            .execute()
        if temp.data:
            t = temp.data[0]
            context += f"\nLATEST TEMPERATURE: {t.get('value_c')} °C (measured {_to_ist(t.get('measured_at'))})\n"
        else:
            context += "\nLATEST TEMPERATURE: NO reading found in database for this user.\n"

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