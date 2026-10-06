import asyncio
import time
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from agent.biometrics import (
    cycle_snapshot,
    last_7_local_dates,
    local_date_key,
    local_today,
    message_text,
    select_card_kind,
    sleep_card_data,
    weekday_label,
    window_start_utc_iso,
)
from agent.tools import (
    search_medical_knowledge,
    get_patient_data,
    check_emergency,
    analyze_symptoms
)
import os
from dotenv import load_dotenv
from typing import Any, Optional
from db.supabase import supabase

load_dotenv()

SYSTEM_PROMPT = """You are MedXAI, an intelligent AI health assistant connected to a patient's smart ring data.

The latest user message already contains everything you may cite:
- PATIENT DATA: this person's profile and ring biometrics, fetched for this request. This is the only source of numeric readings.
- MEDICAL EXCERPTS: optional passages from the medical knowledge base. They are included for general health questions and omitted for ring-only questions.
- Earlier chat turns, when present, are conversation context only. If they disagree with PATIENT DATA, trust PATIENT DATA.

Do not call tools. Do not write tool-call syntax, function names, or JSON tool requests. Answer in plain text.

Important rules:
- Users may make spelling mistakes or typos — always interpret their intent charitably and respond helpfully. For example "dibeties" means "diabetes", "symtoms" means "symptoms", "herat" means "heart". Never reject a message due to spelling.
- If MEDICAL EXCERPTS are absent, you may give brief general health information and make clear it is not from this patient's records.
- If PATIENT DATA says a reading was not found, or no ring biometric data is present, respond clearly: "Not enough continuous biometric data is available yet. Please wear your ring continuously to record readings." Never substitute a plausible-sounding number.
- Never diagnose — only provide health insights and guidance
- Always recommend seeing a doctor for serious concerns
- CRITICAL — DATA ACCURACY: You must ALWAYS call get_patient_data before answering ANY question about the user's own biometrics, even if you think you already know the answer from earlier in the conversation. Only state numeric values that appear VERBATIM in that tool's output. Never estimate, round, infer, average, or invent a number that isn't explicitly present in the tool result. If you cannot find a requested value anywhere in the tool output, say so explicitly instead of producing a number.
- When asked for the "current" or "live" heart rate specifically, use ONLY the value labeled "CURRENT HEART RATE" in the tool output. Do NOT substitute a value from "HISTORICAL DAILY HEART RATE" (those are daily avg/min/max, not current). If that reading is marked [STALE], say clearly that it's not real-time and state its actual age/date — do not present it as "current" without that caveat. If the tool says no reading was found, say so plainly instead of guessing.
- Do not fabricate field labels or stats (e.g. "resting average", "recent max") that are not literally present in the tool output.
- Before sending your final reply, silently check every number you are about to state against the tool output. If a number cannot be found verbatim in the tool output, delete it and say the data is unavailable instead.
- NEVER pair a denial (any phrasing like "I cannot find", "I don't have", "no reading is available", "I cannot retrieve") with a specific real value in the same reply. If you have a real value to report, report it — do not deny having it. If you truly have no value, do not state a number at all. Check this before every reply: if your reply contains both a specific number and a denial phrase about that same metric, delete the denial and keep only the value with its staleness caveat.
- STRICT SCOPING: when the user asks about ONE specific metric by name (e.g. "how is my blood pressure"), your entire reply must be about that metric ONLY. Do not mention any other metric's tool output (heart rate, temperature, HRV, SpO2, sleep, steps) even if it's present in what get_patient_data returned — ignore that other data entirely for this reply. The only exception is a genuine emergency flagged by check_emergency.
- If the user's question does not name a specific metric (e.g. "how am I doing", "how's my health"), you may give a brief multi-metric overview — but if they name one metric, stay scoped to that one.

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
# a fresh ChatGroq client + react-agent graph per call was wasted work on
# every single request for no benefit, since none of it depends on per-request
# state (message/user_id are only passed in at invoke time, not construction time).
_AGENT = None
_LLM = None

def get_medxai_llm():
    global _LLM
    if _LLM is None:
        _LLM = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY") or os.getenv("MISTRAL_API_KEY"),
            model="openai/gpt-oss-120b",
            temperature=0.1,
        )
    return _LLM

def get_medxai_agent():
    global _AGENT
    if _AGENT is None:
        t0 = time.monotonic()
        llm = get_medxai_llm()
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

async def get_sleep_card_data(user_id: str) -> Optional[dict[str, Any]]:
    try:
        if not user_id or user_id == "anonymous":
            return None

        def _query():
            return supabase.table("user_sleep")\
                .select("*")\
                .eq("user_id", user_id)\
                .order("date", desc=True)\
                .limit(1)\
                .execute()

        r = await asyncio.to_thread(_query)

        if r.data:
            return sleep_card_data(r.data[0])
    except Exception as e:
        print(f"Error fetching sleep card data: {e}")

    return None

def _date_window() -> tuple[str, str]:
    days = last_7_local_dates()
    return days[0].isoformat(), days[-1].isoformat()


def _series_for(by_date: dict, value_fn) -> tuple[list, list]:
    values, labels = [], []
    for day in last_7_local_dates():
        values.append(value_fn(by_date.get(day.isoformat())))
        labels.append(weekday_label(day))
    return values, labels


async def get_hr_card_data(user_id: str) -> Optional[dict[str, Any]]:
    if not user_id or user_id == "anonymous":
        return None
    try:
        start, end = _date_window()

        def _query():
            return supabase.table("user_hr").select("*").eq("user_id", user_id)\
                .gte("date", start)\
                .lte("date", end).order("date", desc=False).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return None
        by_date = {str(r.get("date"))[:10]: r for r in rows if r.get("date")}
        values, labels = _series_for(
            by_date,
            lambda row: int(row["avg_hr"]) if row and row.get("avg_hr") else 0,
        )
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
        return None

async def get_spo2_card_data(user_id: str) -> Optional[dict[str, Any]]:
    if not user_id or user_id == "anonymous":
        return None
    try:
        start, end = _date_window()

        def _query():
            return supabase.table("user_spo2").select("*").eq("user_id", user_id)\
                .gte("date", start)\
                .lte("date", end).order("date", desc=False).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return None
        by_date = {str(r.get("date"))[:10]: r for r in rows if r.get("date")}
        values, labels = _series_for(
            by_date,
            lambda row: int(row["avg_spo2"]) if row and row.get("avg_spo2") else 0,
        )
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
        return None

async def get_hrv_card_data(user_id: str) -> Optional[dict[str, Any]]:
    if not user_id or user_id == "anonymous":
        return None
    try:
        start, end = _date_window()

        def _query():
            return supabase.table("user_hrv").select("*").eq("user_id", user_id)\
                .gte("date", start)\
                .lte("date", end).order("date", desc=False).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return None
        by_date = {str(r.get("date"))[:10]: r for r in rows if r.get("date")}
        values, labels = _series_for(
            by_date,
            lambda row: int(row["avg_hrv"]) if row and row.get("avg_hrv") else 0,
        )
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
        return None

async def get_bp_card_data(user_id: str) -> Optional[dict[str, Any]]:
    # No fabricated series. A missing reading returns None, and the chat
    # screen skips the card. A fake 7-day trend is worse than no card.
    if not user_id or user_id == "anonymous":
        return None
    try:
        cutoff = window_start_utc_iso(6)

        def _query():
            return supabase.table("user_bp").select("*").eq("user_id", user_id)\
                .gte("measured_at", cutoff)\
                .order("measured_at", desc=False).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return None

        # Multiple BP readings can land on the same calendar day. Keep only
        # the latest reading per local date so the trend spans distinct days.
        by_date: dict[str, dict] = {}
        for r in rows:
            date_key = local_date_key(r.get("measured_at"))
            if not date_key:
                continue
            existing = by_date.get(date_key)
            if existing is None or str(r.get("measured_at")) > str(existing.get("measured_at")):
                by_date[date_key] = r

        sbp_values, labels = _series_for(
            by_date,
            lambda row: int(row["systolic"]) if row and row.get("systolic") else 0,
        )
        dbp_values, _ = _series_for(
            by_date,
            lambda row: int(row["diastolic"]) if row and row.get("diastolic") else 0,
        )

        sbp_nz = [v for v in sbp_values if v > 0]
        dbp_nz = [v for v in dbp_values if v > 0]
        if not sbp_nz and not dbp_nz:
            return None

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
        return None

async def get_steps_card_data(user_id: str) -> Optional[dict[str, Any]]:
    if not user_id or user_id == "anonymous":
        return None
    try:
        start, end = _date_window()

        def _query():
            return supabase.table("user_steps").select("*").eq("user_id", user_id)\
                .gte("date", start)\
                .lte("date", end).order("date", desc=False).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return None
        by_date = {str(r.get("date"))[:10]: r for r in rows if r.get("date")}
        values, labels = _series_for(
            by_date,
            lambda row: int(row["steps"]) if row and row.get("steps") else 0,
        )
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
        return None

async def get_temperature_card_data(user_id: str) -> Optional[dict[str, Any]]:
    if not user_id or user_id == "anonymous":
        return None
    try:
        cutoff_str = window_start_utc_iso(6)

        def _query():
            return supabase.table("user_temp").select("*").eq("user_id", user_id)\
                .gte("measured_at", cutoff_str)\
                .order("measured_at", desc=False).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return None

        by_date: dict[str, list[float]] = {}
        for r in rows:
            date_key = local_date_key(r.get("measured_at"))
            if not date_key:
                continue
            val = r.get("value_c")
            if val is not None:
                by_date.setdefault(date_key, []).append(float(val))

        values, labels = _series_for(
            by_date,
            lambda day_vals: round(sum(day_vals) / len(day_vals), 1) if day_vals else 0.0,
        )

        non_zero = [v for v in values if v > 0]
        if len(non_zero) < 2:
            return None

        all_vals = [float(r["value_c"]) for r in rows if r.get("value_c") is not None]
        return {
            "type": "temperature_trend",
            "data": {
                "avg": round(sum(non_zero) / len(non_zero), 1),
                "min": min(all_vals) if all_vals else 0.0,
                "max": max(all_vals) if all_vals else 0.0,
                "unit": "°C",
                "values": values,
                "labels": labels,
            }
        }
    except Exception as e:
        print(f"Error fetching temperature card data: {e}")
        return None

async def get_stress_card_data(user_id: str) -> Optional[dict[str, Any]]:
    if not user_id or user_id == "anonymous":
        return None
    try:
        cutoff_str = window_start_utc_iso(6)

        def _query():
            return supabase.table("user_stress").select("*").eq("user_id", user_id)\
                .gte("measured_at", cutoff_str)\
                .order("measured_at", desc=False).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return None

        by_date: dict[str, list[int]] = {}
        for r in rows:
            date_key = local_date_key(r.get("measured_at"))
            if not date_key:
                continue
            val = r.get("stress_value")
            if val is not None:
                by_date.setdefault(date_key, []).append(int(val))

        values, labels = _series_for(
            by_date,
            lambda day_vals: round(sum(day_vals) / len(day_vals)) if day_vals else 0,
        )

        non_zero = [v for v in values if v > 0]
        if len(non_zero) < 2:
            return None

        all_vals = [int(r["stress_value"]) for r in rows if r.get("stress_value") is not None]
        return {
            "type": "stress_trend",
            "data": {
                "avg": round(sum(non_zero) / len(non_zero)),
                "min": min(all_vals) if all_vals else 0,
                "max": max(all_vals) if all_vals else 0,
                "unit": "",
                "values": values,
                "labels": labels,
            }
        }
    except Exception as e:
        print(f"Error fetching stress card data: {e}")
        return None

async def get_cycle_card_data(user_id: str) -> Optional[dict[str, Any]]:
    if not user_id or user_id == "anonymous":
        return None
    try:
        def _query():
            # Same view the app's Home cycle card reads (derived from flow days).
            return supabase.table("derived_user_cycles").select("*").eq("user_id", user_id)\
                .order("period_start", desc=True).limit(1).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return None

        c = rows[0]
        snap = cycle_snapshot(
            c.get("period_start"),
            c.get("cycle_length"),
            c.get("period_length"),
            local_today(),
        )
        return {"type": "cycle_trend", "data": snap}
    except Exception as e:
        print(f"Error fetching cycle card data: {e}")
        return None

_CARD_BUILDERS = {
    "sleep": get_sleep_card_data,
    "bp": get_bp_card_data,
    "spo2": get_spo2_card_data,
    "hrv": get_hrv_card_data,
    "hr": get_hr_card_data,
    "steps": get_steps_card_data,
    "temperature": get_temperature_card_data,
    "stress": get_stress_card_data,
    "cycle": get_cycle_card_data,
}

_EXCERPT_TIMEOUT_S = 12


async def _medical_excerpts(message: str) -> str:
    """Knowledge-base passages for non-biometric questions.

    The embedding lookup runs off the event loop and is capped so a cold
    model load cannot consume the whole chat timeout. A miss still answers
    from PATIENT DATA and general guidance.
    """
    try:
        from services.rag import excerpts_text, retrieve_for_chat, schedule_expansion

        results = await asyncio.wait_for(
            asyncio.to_thread(retrieve_for_chat, message, 3),
            timeout=_EXCERPT_TIMEOUT_S,
        )
    except Exception as exc:
        print(f"[MedXAI] medical knowledge skipped: {type(exc).__name__}")
        return ""
    if results is None:
        return ""
    try:
        await schedule_expansion(message, results)
    except Exception as exc:
        print(f"[MedXAI] knowledge expansion skipped: {type(exc).__name__}")
    return excerpts_text(results)


async def run_agent(
    message: str,
    user_id: str,
    conversation_id: str | None = None,
) -> tuple[str, Optional[dict[str, Any]]]:
    t_start = time.monotonic()

    # 1. Fast-path emergency check in Python (0.001s, 0 LLM calls)
    emerg_res = check_emergency.invoke(message)
    if "EMERGENCY DETECTED" in emerg_res:
        reply = (
            "This needs emergency care right now.\n"
            "- Call emergency services (112 in India) or go to the nearest emergency room\n"
            "- Do not wait for the symptoms to pass\n"
            "- This is general health information, not medical advice."
        )
        return reply, None

    # 2. Patient biometrics. Anonymous callers never hit another user's rows.
    if not user_id or user_id == "anonymous":
        patient_data = (
            "No biometric or ring data is available because this person is not signed in."
        )
    else:
        patient_data = await asyncio.to_thread(get_patient_data.invoke, user_id)

    # 3. Prior turns, the vital card, and knowledge excerpts together.
    from services.chat_history import load_prior_turns

    prior_task = None
    if conversation_id and user_id and user_id != "anonymous":
        prior_task = load_prior_turns(conversation_id, user_id)

    card_builder = _CARD_BUILDERS.get(select_card_kind(message))
    card_task = card_builder(user_id) if card_builder else None
    excerpt_task = _medical_excerpts(message)

    tasks = [excerpt_task]
    if prior_task:
        tasks.append(prior_task)
    if card_task:
        tasks.append(card_task)
    results = await asyncio.gather(*tasks)
    excerpts = results[0]
    index = 1
    prior: list[tuple[str, str]] = []
    if prior_task:
        prior = results[index]
        index += 1
    card = results[index] if card_task else None

    sections = [f"PATIENT DATA:\n{patient_data}"]
    if excerpts:
        sections.append(f"MEDICAL EXCERPTS:\n{excerpts}")
    sections.append(f"USER QUESTION:\n{message}")
    full_user_content = "\n\n".join(sections)

    llm_messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
            + "\n- Direct response mode: answer in plain text from PATIENT DATA and MEDICAL EXCERPTS. Do not output tool calls."
        )
    ]
    for role, content in prior:
        if role == "assistant":
            llm_messages.append(AIMessage(content=content))
        else:
            llm_messages.append(HumanMessage(content=content))
    llm_messages.append(HumanMessage(content=full_user_content))

    llm = get_medxai_llm()
    llm_res = await llm.ainvoke(llm_messages)
    reply = message_text(llm_res.content)
    if not reply:
        raise RuntimeError("The model returned an empty reply")

    print(f"[TIMING] 1-SHOT OPTIMIZED CHAT TOTAL: {time.monotonic() - t_start:.2f}s")
    return reply, card