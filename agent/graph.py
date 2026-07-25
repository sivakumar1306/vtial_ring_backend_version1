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
- Follow with bullet points using a plain hyphen "-" at the start of each line. Keep each bullet under 15 words.
- Do not use section labels like "Summary:" or "Findings:" — just a summary sentence, then bullets.
- Be empathetic in tone even while being concise.
- Always end with this exact line as the final bullet: "This is general health information, not medical advice."

Example format:
No signs of fever based on current data.
- Current heart rate: 90 bpm (within normal range)
- SpO2: 97% (normal oxygen saturation)
- No temperature reading available
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

def create_medxai_agent():
    llm = ChatMistralAI(
        api_key=os.getenv("MISTRAL_API_KEY"),
        model="mistral-large-latest",
        temperature=0.1,
    )
    tools = [
        check_emergency,
        get_patient_data,
        search_medical_knowledge,
        analyze_symptoms
    ]
    agent = create_react_agent(llm, tools)
    return agent

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

        r = supabase.table("user_sleep")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("date", desc=True)\
            .limit(1)\
            .execute()

        if r.data:
            row = r.data[0]
            total_val = row.get("total_duration") or 0
            # If total_val is in seconds, convert to minutes (e.g. 7h = 25200s)
            if total_val > 1440:
                total_min = total_val // 60
            else:
                total_min = total_val

            hours = total_min // 60
            minutes = total_min % 60
            total_label = f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"

            # Since get_patient_data tool's return context doesn't expose stages,
            # we stub using the sleep duration field as instructed:
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

async def run_agent(message: str, user_id: str) -> tuple[str, Optional[dict[str, Any]]]:
    try:
        agent = create_medxai_agent()
        full_message = f"{message}\n\n[user_id: {user_id}]"
        result = await agent.ainvoke({
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=full_message)
            ]
        })

        reply = "I was unable to generate a response. Please try again."
        # get last AI message
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content and msg.type == "ai":
                reply = msg.content
                break

        is_sleep_query = "sleep" in message.lower()
        card = None
        if is_sleep_query:
            card = await get_sleep_card_data(user_id)

        return reply, card
    except Exception as e:
        return f"Agent error: {str(e)}", None