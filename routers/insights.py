from fastapi import APIRouter
from datetime import datetime, timedelta
from typing import Optional
import json
import os
from db.supabase import supabase
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage

router = APIRouter()

_RANGE_DAYS = {"7D": 7, "30D": 30, "90D": 90}


def _fmt_date(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def _avg_by_date(table: str, value_col: str, user_id: str, start: datetime, end: datetime) -> tuple[Optional[float], int]:
    """Average a daily-aggregate column (date-keyed table) over [start, end]."""
    try:
        rows = supabase.table(table).select(f"date,{value_col}")\
            .eq("user_id", user_id)\
            .gte("date", _fmt_date(start))\
            .lte("date", _fmt_date(end))\
            .execute().data or []
        vals = [r[value_col] for r in rows if r.get(value_col) not in (None, 0)]
        if not vals:
            return None, 0
        return sum(vals) / len(vals), len(vals)
    except Exception as e:
        print(f"[_avg_by_date] {table}.{value_col} error: {e}")
        return None, 0


def _avg_sleep_minutes(user_id: str, start: datetime, end: datetime) -> tuple[Optional[float], int]:
    """Sleep duration storage is ambiguous (seconds vs minutes per historical
    rows — see get_sleep_card_data's same >1440 heuristic), so normalize here."""
    try:
        rows = supabase.table("user_sleep").select("date,total_duration")\
            .eq("user_id", user_id)\
            .gte("date", _fmt_date(start))\
            .lte("date", _fmt_date(end))\
            .execute().data or []
        mins = []
        for r in rows:
            total_val = r.get("total_duration") or 0
            if total_val <= 0:
                continue
            mins.append(total_val // 60 if total_val > 1440 else total_val)
        if not mins:
            return None, 0
        return sum(mins) / len(mins), len(mins)
    except Exception as e:
        print(f"[_avg_sleep_minutes] error: {e}")
        return None, 0


def _avg_bp(user_id: str, start: datetime, end: datetime) -> tuple[Optional[float], Optional[float], int]:
    try:
        rows = supabase.table("user_bp").select("systolic,diastolic,measured_at")\
            .eq("user_id", user_id)\
            .gte("measured_at", start.isoformat())\
            .lte("measured_at", end.isoformat())\
            .execute().data or []
        sbp = [r["systolic"] for r in rows if r.get("systolic")]
        dbp = [r["diastolic"] for r in rows if r.get("diastolic")]
        if not sbp or not dbp:
            return None, None, 0
        return sum(sbp) / len(sbp), sum(dbp) / len(dbp), len(rows)
    except Exception as e:
        print(f"[_avg_bp] error: {e}")
        return None, None, 0


def _collect_metrics(user_id: str, start: datetime, end: datetime) -> dict:
    hr_avg, hr_n = _avg_by_date("user_hr", "avg_hr", user_id, start, end)
    hrv_avg, hrv_n = _avg_by_date("user_hrv", "avg_hrv", user_id, start, end)
    spo2_avg, spo2_n = _avg_by_date("user_spo2", "avg_spo2", user_id, start, end)
    steps_avg, steps_n = _avg_by_date("user_steps", "steps", user_id, start, end)
    sleep_avg, sleep_n = _avg_sleep_minutes(user_id, start, end)
    sbp_avg, dbp_avg, bp_n = _avg_bp(user_id, start, end)
    return {
        "heart_rate": {"avg": hr_avg, "n": hr_n, "unit": "bpm"},
        "hrv": {"avg": hrv_avg, "n": hrv_n, "unit": "ms"},
        "spo2": {"avg": spo2_avg, "n": spo2_n, "unit": "%"},
        "steps": {"avg": steps_avg, "n": steps_n, "unit": "steps"},
        "sleep": {"avg": sleep_avg, "n": sleep_n, "unit": "min"},
        "blood_pressure": {"sbp": sbp_avg, "dbp": dbp_avg, "n": bp_n, "unit": "mmHg"},
    }


def _delta(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous is None:
        return None
    return round(current - previous, 1)


INSIGHTS_SYSTEM_PROMPT = """You are generating a short, precise health insight summary for a smart ring app's home screen.

You will be given ONLY real, pre-computed numeric averages and deltas for the user's biometrics over a period, comparing the current period to the immediately preceding period of equal length. You must NOT invent, estimate, or round any number that isn't given to you. If a metric has no data (null), do not mention it.

Respond with STRICT JSON only — no markdown, no code fences, no commentary before or after. Match this exact shape:

{
  "headline_parts": [
    {"text": "Your sleep is steadier, but blood pressure is ", "highlight": false},
    {"text": "high", "highlight": true}
  ],
  "subtext": "One or two plain-language sentences elaborating on the headline, grounded only in the given numbers.",
  "updates": [
    {"key": "sleep", "title": "Sleep", "delta_label": "+22 min", "subtitle": "22 minutes more sleep on average", "alarming": false},
    {"key": "blood_pressure", "title": "Blood pressure", "delta_label": "+6", "subtitle": "Systolic trending up across the period", "alarming": true}
  ]
}

Rules:
- "key" must be one of: sleep, heart_rate, hrv, spo2, blood_pressure, steps
- Pick the 1-2 most meaningful changes (largest, most clinically relevant deltas) for "updates" — do not include a metric with no real change or with null/insufficient data.
- "alarming": true only for changes in a concerning direction — heart rate or blood pressure rising notably, SpO2 dropping, HRV dropping, sleep dropping notably. Positive or neutral changes get alarming: false.
- "delta_label" should be a short signed number/unit string like "+6", "-3 bpm", "+22 min" — derived only from the given delta values, never invented.
- The headline should read like a short, human sentence (max ~15 words), split into headline_parts so one clause can be highlighted (the concerning or most notable part) — set highlight: true on that clause only.
- If there isn't enough data for any metric, headline_parts should read something like "Not enough data yet to show trends" (highlight: false) with an empty updates array and a brief explanatory subtext.
- Never mention units or numbers you were not given verbatim in the metrics."""


def _build_metrics_text(range_label: str, current: dict, previous: dict) -> str:
    lines = [f"Period: last {range_label}, compared to the preceding {range_label} of equal length.\n"]
    for key, label in [
        ("heart_rate", "Heart rate"), ("hrv", "HRV"), ("spo2", "SpO2"),
        ("blood_pressure", "Blood pressure"), ("sleep", "Sleep"), ("steps", "Steps"),
    ]:
        cur = current[key]
        prev = previous[key]
        if key == "blood_pressure":
            if cur["n"] == 0:
                lines.append(f"{label}: no data")
                continue
            sbp_delta = _delta(cur["sbp"], prev["sbp"])
            dbp_delta = _delta(cur["dbp"], prev["dbp"])
            lines.append(
                f"{label}: current avg {cur['sbp']:.0f}/{cur['dbp']:.0f} mmHg "
                f"(previous period {prev['sbp']:.0f}/{prev['dbp']:.0f} mmHg" if prev["n"] else f"{label}: current avg {cur['sbp']:.0f}/{cur['dbp']:.0f} mmHg (no previous-period data"
            )
            if sbp_delta is not None:
                lines[-1] += f", systolic delta {sbp_delta:+.1f}, diastolic delta {dbp_delta:+.1f})"
            else:
                lines[-1] += ")"
            continue

        if cur["n"] == 0:
            lines.append(f"{label}: no data")
            continue
        delta = _delta(cur["avg"], prev["avg"])
        text = f"{label}: current avg {cur['avg']:.1f} {cur['unit']} ({cur['n']} samples)"
        if prev["n"] and delta is not None:
            text += f", previous period avg {prev['avg']:.1f} {cur['unit']}, delta {delta:+.1f} {cur['unit']}"
        else:
            text += ", no previous-period data to compare"
        lines.append(text)
    return "\n".join(lines)


def _fallback_insights(current: dict, previous: dict) -> dict:
    """Deterministic, non-LLM fallback if the model call fails or returns
    unparseable output — built only from real numbers, never invented."""
    candidates = []
    for key, label, unit, fmt in [
        ("heart_rate", "Heart rate", "bpm", "{:+.0f}"),
        ("hrv", "HRV", "ms", "{:+.0f}"),
        ("spo2", "SpO2", "%", "{:+.0f}"),
        ("sleep", "Sleep", "min", "{:+.0f}"),
        ("steps", "Steps", "steps", "{:+.0f}"),
    ]:
        cur, prev = current[key], previous[key]
        if cur["n"] == 0 or prev["n"] == 0:
            continue
        d = _delta(cur["avg"], prev["avg"])
        if d is None or abs(d) < 0.5:
            continue
        alarming = (key in ("heart_rate",) and d > 0) or (key in ("spo2", "hrv", "sleep") and d < 0)
        candidates.append({
            "key": key, "title": label,
            "delta_label": fmt.format(d),
            "subtitle": f"{label} averaged {cur['avg']:.0f} {unit} this period vs {prev['avg']:.0f} {unit} previously",
            "alarming": alarming,
            "abs_delta": abs(d),
        })

    bp_cur, bp_prev = current["blood_pressure"], previous["blood_pressure"]
    if bp_cur["n"] and bp_prev["n"]:
        sbp_d = _delta(bp_cur["sbp"], bp_prev["sbp"])
        if sbp_d is not None and abs(sbp_d) >= 1:
            candidates.append({
                "key": "blood_pressure", "title": "Blood pressure",
                "delta_label": f"{sbp_d:+.0f}",
                "subtitle": "Systolic trending up" if sbp_d > 0 else "Systolic trending down",
                "alarming": sbp_d > 0,
                "abs_delta": abs(sbp_d),
            })

    if not candidates:
        return {
            "headline_parts": [{"text": "Not enough data yet to show trends", "highlight": False}],
            "subtext": "Keep wearing your ring and check back soon.",
            "updates": [],
        }

    candidates.sort(key=lambda c: c["abs_delta"], reverse=True)
    top = candidates[:2]
    worst = next((c for c in top if c["alarming"]), None)

    if worst:
        headline_parts = [
            {"text": f"Your {worst['title'].lower()} needs attention", "highlight": False},
        ]
        subtext = f"{worst['subtitle']}. Other metrics look steady."
    else:
        best = top[0]
        headline_parts = [
            {"text": f"Your {best['title'].lower()} is trending well", "highlight": True},
        ]
        subtext = f"{best['subtitle']}."

    for c in top:
        c.pop("abs_delta", None)

    return {"headline_parts": headline_parts, "subtext": subtext, "updates": top}


def _call_llm_for_insights(range_label: str, current: dict, previous: dict) -> dict:
    metrics_text = _build_metrics_text(range_label, current, previous)
    llm = ChatMistralAI(
        api_key=os.getenv("MISTRAL_API_KEY"),
        model="mistral-large-latest",
        temperature=0.2,
    )
    result = llm.invoke([
        SystemMessage(content=INSIGHTS_SYSTEM_PROMPT),
        HumanMessage(content=metrics_text),
    ])
    raw = result.content.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM response")
    parsed = json.loads(raw[start:end + 1])
    if "headline_parts" not in parsed or "subtext" not in parsed:
        raise ValueError("Malformed insights JSON from LLM")
    parsed.setdefault("updates", [])
    return parsed


@router.get("/insights/{user_id}")
async def get_insights(user_id: str, range: str = "30D"):
    days = _RANGE_DAYS.get(range, 30)
    now = datetime.utcnow()
    cur_start = now - timedelta(days=days)
    prev_start = now - timedelta(days=days * 2)
    prev_end = cur_start

    current = _collect_metrics(user_id, cur_start, now)
    previous = _collect_metrics(user_id, prev_start, prev_end)

    try:
        return _call_llm_for_insights(range, current, previous)
    except Exception as e:
        print(f"[get_insights] LLM insight generation failed, using fallback: {e}")
        return _fallback_insights(current, previous)