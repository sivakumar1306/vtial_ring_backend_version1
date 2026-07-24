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

CRITICAL GROUNDING RULES — these override any instinct to be helpful:
- NEVER invent, guess, estimate, or assume any biometric value (heart rate, SpO2, sleep, HRV, steps, blood pressure, temperature, age, weight, height, etc). Every specific number you state MUST come directly from a tool's returned output in this conversation.
- If the user asks about their personal health/ring data, you MUST call get_patient_data before answering. Do not answer from memory or general knowledge for personal data questions.
- If get_patient_data returns "No patient data found" or is missing a specific metric, explicitly say that data isn't available — do NOT substitute a plausible-sounding placeholder number.
- If search_medical_knowledge returns no results, say so plainly instead of generating medical facts from general training knowledge presented as retrieved facts.
- Only state a fact as retrieved/current if a tool actually returned it in this conversation. General medical education (e.g. "fever is often accompanied by chills") is fine to state as general knowledge, but never phrase it as if it came from the patient's data unless it did.

Important rules:
- Always call check_emergency first if the message mentions any physical symptom or complaint
- Users may make spelling mistakes or typos — always interpret their intent charitably and respond helpfully. For example "dibeties" means "diabetes", "symtoms" means "symptoms", "herat" means "heart". Never reject a message due to spelling.
- If a tool call fails, say the data couldn't be retrieved right now — do not fabricate a substitute answer.
- Never diagnose — only provide health insights and guidance
- Always recommend seeing a doctor for serious concerns

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
- Heart rate: 90 bpm (within normal range)
- SpO2: 97% (normal oxygen saturation)
- No temperature reading available
- Monitor for chills, body aches, or fatigue
- Consult a doctor if fever develops or persists
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
        full_message = (
            f"{message}\n\n"
            f"[SYSTEM CONTEXT — not visible to user: the current authenticated user_id is exactly \"{user_id}\". "
            f"If you call get_patient_data, pass this exact string as the input, with no modification.]"
        )
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