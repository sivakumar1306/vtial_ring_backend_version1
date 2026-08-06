import asyncio
import time
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from agent.tools import (
    search_medical_knowledge,
    get_patient_data,
    check_emergency,
    analyze_symptoms
)
import os
import re
from datetime import datetime as dt, timedelta
from dotenv import load_dotenv
from typing import Any, Optional
from db.supabase import supabase

load_dotenv()

SYSTEM_PROMPT = """You are MedXAI, an intelligent AI health assistant connected to a patient's smart ring data.

You have 4 tools available:
1. check_emergency — ALWAYS call this first for any health complaint or symptom
2. get_patient_data — call this when the user asks about their personal health, biometrics, or ring data
3. search_medical_knowledge — call this for general medical questions, conditions, symptoms, treatments
4. analyze_symptoms — call this when the user lists multiple symptoms together

Important rules:
- Always call check_emergency first if the message mentions any physical symptom or complaint
- Users may make spelling mistakes or typos — always interpret their intent charitably and respond helpfully. For example "dibeties" means "diabetes", "symtoms" means "symptoms", "herat" means "heart". Never reject a message due to spelling.
- If a tool call fails and the user is asking a GENERAL medical question (e.g. "what causes a headache"), you may still answer from general medical knowledge.
- If a tool call fails and the user is asking about THEIR OWN biometric data (heart rate, blood pressure, SpO2, sleep, steps, HRV, weight, etc.), you must NOT use general medical knowledge to answer. Say plainly that you could not retrieve their data right now and to try again shortly. Never substitute a plausible-sounding number.
- Never diagnose — only provide health insights and guidance
- Always recommend seeing a doctor for serious concerns
- CRITICAL — DATA ACCURACY: You must ALWAYS call get_patient_data before answering ANY question about the user's own biometrics, even if you think you already know the answer from earlier in the conversation. Only state numeric values that appear VERBATIM in that tool's output. Never estimate, round, infer, average, or invent a number that isn't explicitly present in the tool result. If you cannot find a requested value anywhere in the tool output, say so explicitly instead of producing a number.
- When asked for the "current" or "live" heart rate specifically, use ONLY the value labeled "CURRENT HEART RATE" in the tool output. Do NOT substitute a value from "HISTORICAL DAILY HEART RATE" (those are daily avg/min/max, not current). If that reading is marked [STALE], say clearly that it's not real-time and state its actual age/date — do not present it as "current" without that caveat. If the tool says no reading was found, say so plainly instead of guessing.
- Do not fabricate field labels or stats (e.g. "resting average", "recent max") that are not literally present in the tool output.
- Before sending your final reply, silently check every number you are about to state against the tool output. If a number cannot be found verbatim in the tool output, delete it and say the data is unavailable instead.

RESPONSE FORMAT — STRICTLY FOLLOW THIS:
- Use precise clinical/medical terminology (e.g. "tachycardia" instead of "fast heart rate", "hyperglycemia" instead of "high blood sugar"). Add a brief plain-language clarification in parentheses the first time you use an uncommon term.
- Do NOT use any markdown formatting — no asterisks, no bold, no headers, no numbering. Plain text only.
- Start with one short summary line (no label, no prefix — just the sentence).
- Give AT MOST 4 bullet points total (not counting the mandatory disclaimer bullet). If more metrics are relevant than that, group/merge related ones into a single bullet (e.g. combine HR+HRV+SpO2 into one "vitals are in normal range" bullet) rather than listing each one separately.
- Do not list every historical day's data — summarize the trend across the days (e.g. "sleep score improved from 63 to 89 over the week") in one bullet instead of one bullet per day.
- Every number stated must still come verbatim from tool output — summarizing must never introduce averages or values not present in the tool output.
- Follow with bullet points using a plain hyphen "-" at the start of each line. Keep each bullet under 15 words.
- Do not use section labels like "Summary:" or "Findings:" — just a summary sentence, then bullets.
- Be empathetic in tone even while being concise.
- Always end with this exact line as the final bullet: "This is general health information, not medical advice."

Example format:
No signs of fever based on current data.
- Current vitals normal: HR 90 bpm, SpO2 97%
- Temperature: 36.6 °C (afebrile)
- Monitor for chills, body aches, or fatigue
- Consult a doctor if fever develops or persists
- This is general health information, not medical advice.

Example when asked specifically for current/live heart rate and the reading is marked stale:
No real-time heart rate reading is available right now.
- Last recorded reading was 84 bpm on 21 July, 2026
- That is 4 days old, not a live measurement
- Open the ring app to sync or take a fresh reading
- Consult a doctor if you feel unwell
- This is general health information, not medical advice.
"""

# Cached at module level instead of recreated on every /chat request — building
# a fresh ChatMistralAI client + react-agent graph per call was wasted work on
# every single request for no benefit, since none of it depends on per-request
# state (message/user_id are only passed in at invoke time, not construction time).
_AGENT = None


def get_medxai_agent():
    global _AGENT
    if _AGENT is None:
        t0 = time.monotonic()
        llm = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral-small-latest",
            temperature=0.1,
        )
        tools = [
            check_emergency,
            get_patient_data,
            search_medical_knowledge,
            analyze_symptoms
        ]
        _AGENT = create_react_agent(llm, tools)
        print(f"[TIMING] Agent construction (first call only, cached after): {time.monotonic() - t0:.2f}s")
    return _AGENT


# ── Card builders for each vital ────────────────────────────────────────────
#
# Every supabase.table(...).execute() call below is synchronous/blocking —
# the Supabase Python client has no native async mode. Run inside FastAPI's
# single-threaded event loop directly, a blocking DB call here freezes the
# *entire server* for every other concurrent request (insights, chat, history,
# everyone) until it returns — not just this one. asyncio.to_thread() runs the
# blocking call on a background thread instead, so the event loop stays free
# to serve other requests while this one waits on the DB.

_WEEKDAY_ABBR = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

def _fmt_date(d: dt) -> str:
    return d.strftime('%Y-%m-%d')

async def get_sleep_card_data(user_id: str) -> Optional[dict[str, Any]]:
    try:
        if not user_id or user_id == "anonymous":
            return {
                "type": "sleep_highlights",
                "data": {
                    "time_awake_min": 10,
                    "light_sleep_min": 63,
                    "deep_sleep_min": 250,
                    "total_label": "5 hours and 13 minutes"
                }
            }

        def _query():
            return supabase.table("user_sleep")\
                .select("*")\
                .eq("user_id", user_id)\
                .order("date", desc=True)\
                .limit(1)\
                .execute()

        r = await asyncio.to_thread(_query)

        if r.data:
            row = r.data[0]
            total_val = row.get("total_duration") or 0
            if total_val > 1440:
                total_min = total_val // 60
            else:
                total_min = total_val

            hours = total_min // 60
            minutes = total_min % 60
            total_label = f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"

            time_awake_min = int(total_min * 0.05)
            deep_sleep_min = int(total_min * 0.25)
            light_sleep_min = total_min - time_awake_min - deep_sleep_min

            return {
                "type": "sleep_highlights",
                "data": {
                    "time_awake_min": time_awake_min,
                    "light_sleep_min": light_sleep_min,
                    "deep_sleep_min": deep_sleep_min,
                    "total_label": total_label
                }
            }
    except Exception as e:
        print(f"Error fetching sleep card data: {e}")

    return {
        "type": "sleep_highlights",
        "data": {
            "time_awake_min": 10,
            "light_sleep_min": 63,
            "deep_sleep_min": 250,
            "total_label": "5 hours and 13 minutes"
        }
    }

async def get_hr_card_data(user_id: str) -> Optional[dict[str, Any]]:
    demo = {
        "type": "heart_rate_trend",
        "data": {
            "avg": 78, "min": 58, "max": 112, "unit": "bpm",
            "values": [72, 75, 80, 77, 82, 79, 78],
            "labels": _WEEKDAY_ABBR,
        }
    }
    if not user_id or user_id == "anonymous":
        return demo
    try:
        now = dt.utcnow()

        def _query():
            return supabase.table("user_hr").select("*").eq("user_id", user_id)\
                .gte("date", _fmt_date(now - timedelta(days=6)))\
                .lte("date", _fmt_date(now)).order("date", desc=False).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return demo
        by_date = {r["date"]: r for r in rows}
        values, labels = [], []
        for i in range(7):
            day = now - timedelta(days=6 - i)
            row = by_date.get(_fmt_date(day))
            values.append(int(row["avg_hr"]) if row and row.get("avg_hr") else 0)
            labels.append(_WEEKDAY_ABBR[day.weekday()])
        non_zero = [v for v in values if v > 0]
        mins = [int(r["min_hr"]) for r in rows if r.get("min_hr")]
        maxs = [int(r["max_hr"]) for r in rows if r.get("max_hr")]
        return {
            "type": "heart_rate_trend",
            "data": {
                "avg": round(sum(non_zero) / len(non_zero)) if non_zero else 0,
                "min": min(mins) if mins else 0,
                "max": max(maxs) if maxs else 0,
                "unit": "bpm",
                "values": values,
                "labels": labels,
            }
        }
    except Exception as e:
        print(f"Error fetching HR card data: {e}")
        return demo

async def get_spo2_card_data(user_id: str) -> Optional[dict[str, Any]]:
    demo = {
        "type": "spo2_trend",
        "data": {
            "avg": 97, "min": 94, "max": 99, "unit": "%",
            "values": [96, 97, 98, 97, 95, 98, 97],
            "labels": _WEEKDAY_ABBR,
        }
    }
    if not user_id or user_id == "anonymous":
        return demo
    try:
        now = dt.utcnow()

        def _query():
            return supabase.table("user_spo2").select("*").eq("user_id", user_id)\
                .gte("date", _fmt_date(now - timedelta(days=6)))\
                .lte("date", _fmt_date(now)).order("date", desc=False).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return demo
        by_date = {r["date"]: r for r in rows}
        values, labels = [], []
        for i in range(7):
            day = now - timedelta(days=6 - i)
            row = by_date.get(_fmt_date(day))
            values.append(int(row["avg_spo2"]) if row and row.get("avg_spo2") else 0)
            labels.append(_WEEKDAY_ABBR[day.weekday()])
        non_zero = [v for v in values if v > 0]
        mins = [int(r["min_spo2"]) for r in rows if r.get("min_spo2")]
        maxs = [int(r["max_spo2"]) for r in rows if r.get("max_spo2")]
        return {
            "type": "spo2_trend",
            "data": {
                "avg": round(sum(non_zero) / len(non_zero)) if non_zero else 0,
                "min": min(mins) if mins else 0,
                "max": max(maxs) if maxs else 0,
                "unit": "%",
                "values": values,
                "labels": labels,
            }
        }
    except Exception as e:
        print(f"Error fetching SpO2 card data: {e}")
        return demo

async def get_hrv_card_data(user_id: str) -> Optional[dict[str, Any]]:
    demo = {
        "type": "hrv_trend",
        "data": {
            "avg": 52, "min": 30, "max": 78, "unit": "ms",
            "values": [45, 50, 55, 48, 60, 52, 52],
            "labels": _WEEKDAY_ABBR,
        }
    }
    if not user_id or user_id == "anonymous":
        return demo
    try:
        now = dt.utcnow()

        def _query():
            return supabase.table("user_hrv").select("*").eq("user_id", user_id)\
                .gte("date", _fmt_date(now - timedelta(days=6)))\
                .lte("date", _fmt_date(now)).order("date", desc=False).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return demo
        by_date = {r["date"]: r for r in rows}
        values, labels = [], []
        for i in range(7):
            day = now - timedelta(days=6 - i)
            row = by_date.get(_fmt_date(day))
            values.append(int(row["avg_hrv"]) if row and row.get("avg_hrv") else 0)
            labels.append(_WEEKDAY_ABBR[day.weekday()])
        non_zero = [v for v in values if v > 0]
        mins = [int(r["min_hrv"]) for r in rows if r.get("min_hrv")]
        maxs = [int(r["max_hrv"]) for r in rows if r.get("max_hrv")]
        return {
            "type": "hrv_trend",
            "data": {
                "avg": round(sum(non_zero) / len(non_zero)) if non_zero else 0,
                "min": min(mins) if mins else 0,
                "max": max(maxs) if maxs else 0,
                "unit": "ms",
                "values": values,
                "labels": labels,
            }
        }
    except Exception as e:
        print(f"Error fetching HRV card data: {e}")
        return demo

async def get_bp_card_data(user_id: str) -> Optional[dict[str, Any]]:
    demo = {
        "type": "bp_trend",
        "data": {
            "sbp_avg": 118, "dbp_avg": 76,
            "sbp_values": [115, 120, 118, 122, 117, 119, 118],
            "dbp_values": [74, 78, 76, 80, 75, 77, 76],
            "labels": _WEEKDAY_ABBR,
        }
    }
    if not user_id or user_id == "anonymous":
        return demo
    try:
        now = dt.utcnow()

        def _query():
            return supabase.table("user_bp").select("*").eq("user_id", user_id)\
                .gte("measured_at", (now - timedelta(days=6)).isoformat())\
                .order("measured_at", desc=False).limit(7).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return demo
        sbp_values = [int(r.get("systolic") or 0) for r in rows]
        dbp_values = [int(r.get("diastolic") or 0) for r in rows]
        labels = []
        for r in rows:
            try:
                d = dt.fromisoformat(str(r.get("measured_at")).replace("Z", "+00:00"))
                labels.append(_WEEKDAY_ABBR[d.weekday()])
            except Exception:
                labels.append("")
        sbp_nz = [v for v in sbp_values if v > 0]
        dbp_nz = [v for v in dbp_values if v > 0]
        return {
            "type": "bp_trend",
            "data": {
                "sbp_avg": round(sum(sbp_nz) / len(sbp_nz)) if sbp_nz else 0,
                "dbp_avg": round(sum(dbp_nz) / len(dbp_nz)) if dbp_nz else 0,
                "sbp_values": sbp_values,
                "dbp_values": dbp_values,
                "labels": labels,
            }
        }
    except Exception as e:
        print(f"Error fetching BP card data: {e}")
        return demo

async def get_steps_card_data(user_id: str) -> Optional[dict[str, Any]]:
    demo = {
        "type": "steps_trend",
        "data": {
            "avg": 6400, "unit": "steps",
            "values": [5200, 7100, 6800, 4900, 8200, 6300, 6400],
            "labels": _WEEKDAY_ABBR,
        }
    }
    if not user_id or user_id == "anonymous":
        return demo
    try:
        now = dt.utcnow()

        def _query():
            return supabase.table("user_steps").select("*").eq("user_id", user_id)\
                .gte("date", _fmt_date(now - timedelta(days=6)))\
                .lte("date", _fmt_date(now)).order("date", desc=False).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return demo
        by_date = {r["date"]: r for r in rows}
        values, labels = [], []
        for i in range(7):
            day = now - timedelta(days=6 - i)
            row = by_date.get(_fmt_date(day))
            values.append(int(row["steps"]) if row and row.get("steps") else 0)
            labels.append(_WEEKDAY_ABBR[day.weekday()])
        non_zero = [v for v in values if v > 0]
        return {
            "type": "steps_trend",
            "data": {
                "avg": round(sum(non_zero) / len(non_zero)) if non_zero else 0,
                "unit": "steps",
                "values": values,
                "labels": labels,
            }
        }
    except Exception as e:
        print(f"Error fetching steps card data: {e}")
        return demo

async def run_agent(message: str, user_id: str) -> tuple[str, Optional[dict[str, Any]]]:
    try:
        t_start = time.monotonic()

        agent = get_medxai_agent()
        t_agent_ready = time.monotonic()

        full_message = f"{message}\n\n[user_id: {user_id}]"
        result = await agent.ainvoke({
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=full_message)
            ]
        })
        t_invoke_done = time.monotonic()

        reply = "I was unable to generate a response. Please try again."
        # get last AI message
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content and msg.type == "ai":
                reply = msg.content
                break

        msg_lower = message.lower()
        card = None
        if "sleep" in msg_lower:
            card = await get_sleep_card_data(user_id)
        elif any(k in msg_lower for k in ["blood pressure", "systolic", "diastolic"]) or re.search(r'\bbp\b', msg_lower):
            card = await get_bp_card_data(user_id)
        elif any(k in msg_lower for k in ["spo2", "sp02", "blood oxygen", "oxygen level", "oxygen saturation"]):
            card = await get_spo2_card_data(user_id)
        elif any(k in msg_lower for k in ["hrv", "heart rate variability", "variability"]):
            card = await get_hrv_card_data(user_id)
        elif any(k in msg_lower for k in ["heart rate", "pulse", "bpm", "snore", "snoring"]):
            card = await get_hr_card_data(user_id)
        elif any(k in msg_lower for k in ["steps", "walked", "walking", "step count"]):
            card = await get_steps_card_data(user_id)
        t_card_done = time.monotonic()

        # TEMP DEBUG — remove once the /chat latency source is confirmed.
        # Breaks down where the total request time is actually going:
        #  - "agent setup": only non-zero on the very first /chat call in this
        #    container's lifetime (LLM client + tool schema binding); cached
        #    after that, so this should read ~0.00s on every subsequent call.
        #  - "agent.ainvoke (LLM tool-routing + reasoning)": the actual round
        #    trip(s) to Mistral — deciding which tool(s) to call, waiting on
        #    tool results, then generating the final reply. This is the one
        #    to watch; if this number matches the overall slow request time,
        #    the bottleneck is the LLM call itself, not our code.
        #  - "card builder": should be near-instant (simple indexed DB reads).
        print(
            f"[TIMING] agent setup: {t_agent_ready - t_start:.2f}s | "
            f"agent.ainvoke (LLM tool-routing + reasoning): {t_invoke_done - t_agent_ready:.2f}s | "
            f"card builder: {t_card_done - t_invoke_done:.2f}s | "
            f"TOTAL: {t_card_done - t_start:.2f}s"
        )

        return reply, card
    except Exception as e:
        return f"Agent error: {str(e)}", None