from fastapi import APIRouter
from datetime import datetime, timedelta, timezone
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


# ── Dynamic insights (headline + key-change cards) ─────────────────────────
#
# Replaces the previously hardcoded "Your recovery is steady, but blood
# pressure is high" copy and the fixed Sleep/Blood pressure cards in the
# Flutter AI-assist header. Everything below is computed from real rows in
# the DB for the requested range, split into an "earlier half" and a "later
# half" so we can measure a genuine trend rather than a single snapshot.

RANGE_DAYS = {"7D": 7, "30D": 30, "90D": 90}

def _rows_in_range(table: str, user_id: str, days: int, date_col: str):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    r = supabase.table(table)\
        .select("*")\
        .eq("user_id", user_id)\
        .gte(date_col, cutoff)\
        .order(date_col, desc=False)\
        .execute()
    return r.data or []

def _split_avg(rows, field):
    """Average `field` over the first half vs second half of `rows` (chronological)."""
    vals = [r.get(field) for r in rows if r.get(field) is not None]
    if len(vals) < 2:
        return None, None
    mid = len(vals) // 2
    first_half = vals[:mid] if mid > 0 else vals[:1]
    second_half = vals[mid:]
    if not first_half or not second_half:
        return None, None
    return sum(first_half) / len(first_half), sum(second_half) / len(second_half)

def _bp_rows_in_range(user_id: str, days: int):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    r = supabase.table("user_bp")\
        .select("*")\
        .eq("user_id", user_id)\
        .gte("measured_at", cutoff)\
        .order("measured_at", desc=False)\
        .execute()
    return r.data or []

@router.get("/insights/{user_id}")
async def get_insights(user_id: str, range: str = "30D"):
    days = RANGE_DAYS.get(range, 30)
    try:
        changes = []  # each: dict(key,title,delta_label,subtitle,alarming,direction)

        # --- Sleep (user_sleep: date, total_duration [minutes], sleep_score) ---
        sleep_rows = _rows_in_range("user_sleep", user_id, days, "date")
        first, second = _split_avg(sleep_rows, "total_duration")
        if first is not None:
            delta_min = round(second - first)
            if abs(delta_min) >= 15:
                improving = delta_min > 0
                changes.append({
                    "key": "sleep",
                    "title": "Sleep",
                    "delta_label": f"{'+' if delta_min >= 0 else ''}{delta_min} min",
                    "subtitle": f"{abs(delta_min)} minutes {'more' if improving else 'less'} sleep on average",
                    "alarming": (not improving) and abs(delta_min) >= 30,
                    "improving": improving,
                })

        # --- HRV (user_hrv: date, avg_hrv ms) — lower HRV trend is the concern ---
        hrv_rows = _rows_in_range("user_hrv", user_id, days, "date")
        first, second = _split_avg(hrv_rows, "avg_hrv")
        if first is not None:
            delta_hrv = round(second - first)
            if abs(delta_hrv) >= 4:
                improving = delta_hrv > 0
                changes.append({
                    "key": "hrv",
                    "title": "HRV",
                    "delta_label": f"{'+' if delta_hrv >= 0 else ''}{delta_hrv} ms",
                    "subtitle": f"Heart rate variability {'improved' if improving else 'dropped'} across the period",
                    "alarming": (not improving) and abs(delta_hrv) >= 10,
                    "improving": improving,
                })

        # --- Resting heart rate (user_hr: date, avg_hr) ---
        # "alarming" is based purely on the CURRENT resting HR value crossing
        # a clinically meaningful threshold (>=100 bpm resting = tachycardia
        # range), not on how much it moved during the period. A rapid swing
        # that stays within a normal resting range (e.g. 80 -> 90 bpm) is a
        # trend worth surfacing, but it should not be flagged as "needs
        # attention" the same way an actually-elevated reading is.
        hr_rows = _rows_in_range("user_hr", user_id, days, "date")
        first, second = _split_avg(hr_rows, "avg_hr")
        if first is not None:
            delta_hr = round(second - first)
            latest_avg_hr = hr_rows[-1].get("avg_hr") if hr_rows else None
            alarming = latest_avg_hr is not None and latest_avg_hr >= 100
            if abs(delta_hr) >= 5 or alarming:
                improving = delta_hr <= 0  # lower resting HR trend = improving
                changes.append({
                    "key": "heart_rate",
                    "title": "Heart rate",
                    "delta_label": f"{'+' if delta_hr >= 0 else ''}{delta_hr} bpm",
                    "subtitle": (
                        f"Resting heart rate elevated, averaging {round(latest_avg_hr)} bpm recently"
                        if alarming else
                        f"Resting heart rate trending {'up' if delta_hr > 0 else 'down'} across the period"
                    ),
                    "alarming": alarming,
                    "improving": improving,
                })

        # --- Blood pressure (user_bp: measured_at, systolic, diastolic) ---
        # Same principle as heart rate: "alarming" reflects the CURRENT
        # reading crossing a hypertensive threshold, not the size of the
        # trend/delta by itself.
        bp_rows = _bp_rows_in_range(user_id, days)
        first_s, second_s = _split_avg(bp_rows, "systolic")
        if first_s is not None:
            delta_s = round(second_s - first_s)
            latest_systolic = bp_rows[-1].get("systolic") if bp_rows else None
            latest_diastolic = bp_rows[-1].get("diastolic") if bp_rows else None
            hypertensive = (latest_systolic and latest_systolic >= 130) or (latest_diastolic and latest_diastolic >= 80)
            if abs(delta_s) >= 5 or hypertensive:
                improving = delta_s <= 0
                alarming = hypertensive
                changes.append({
                    "key": "blood_pressure",
                    "title": "Blood pressure",
                    "delta_label": f"{'+' if delta_s >= 0 else ''}{delta_s}",
                    "subtitle": (
                        f"Systolic trending up across the period, latest {latest_systolic}/{latest_diastolic}"
                        if alarming else
                        "Systolic trending down across the period"
                    ),
                    "alarming": alarming,
                    "improving": improving,
                })

        # --- SpO2 (user_spo2: date, avg_spo2) — alarming if genuinely low ---
        spo2_rows = _rows_in_range("user_spo2", user_id, days, "date")
        if spo2_rows:
            latest_spo2 = spo2_rows[-1].get("avg_spo2")
            if latest_spo2 is not None and latest_spo2 < 94:
                changes.append({
                    "key": "spo2",
                    "title": "Blood oxygen",
                    "delta_label": f"{round(latest_spo2)}%",
                    "subtitle": f"Recent SpO2 reading of {round(latest_spo2)}% is below the normal range",
                    "alarming": True,
                    "improving": False,
                })

        # --- Steps (user_steps: date, steps) ---
        steps_rows = _rows_in_range("user_steps", user_id, days, "date")
        first, second = _split_avg(steps_rows, "steps")
        if first and first > 0:
            pct = round(((second - first) / first) * 100)
            if abs(pct) >= 20:
                improving = pct > 0
                changes.append({
                    "key": "steps",
                    "title": "Activity",
                    "delta_label": f"{'+' if pct >= 0 else ''}{pct}%",
                    "subtitle": f"Daily steps {'up' if improving else 'down'} {abs(pct)}% across the period",
                    "alarming": False,
                    "improving": improving,
                })

        # If anything is alarming, only alarming card(s) are surfaced — matches
        # the product requirement that an alarming metric should stand alone
        # rather than compete for attention with routine trend cards.
        alarming_changes = [c for c in changes if c["alarming"]]
        if alarming_changes:
            updates = alarming_changes
        else:
            # Otherwise show up to 2 most notable (by absolute magnitude isn't
            # tracked numerically per-type here, so we keep insertion order,
            # which already favors sleep/hrv/hr/bp/spo2/steps in clinical
            # relevance order) non-alarming changes.
            updates = changes[:2]

        # --- Headline + subtext ---
        if not changes:
            headline_parts = [{"text": "Not enough data yet to summarize your trends.", "highlight": False}]
            subtext = "Keep wearing your ring and check back after a few more days of readings."
        elif alarming_changes:
            worst = alarming_changes[0]
            headline_parts = [
                {"text": f"Your {worst['title'].lower()} needs attention", "highlight": False},
            ]
            other_improving = [c["title"] for c in changes if c.get("improving") and not c["alarming"]]
            if other_improving:
                subtext = f"{', '.join(other_improving)} improved this period, but {worst['title'].lower()} needs closer attention. See the details below."
            else:
                subtext = f"{worst['title']} needs closer attention this period. See the details below."
        else:
            improving = [c["title"] for c in changes if c.get("improving")]
            declining = [c["title"] for c in changes if not c.get("improving")]
            if improving and not declining:
                headline_parts = [{"text": "Your recovery is trending in the right direction", "highlight": False}]
            elif improving and declining:
                headline_parts = [
                    {"text": f"{', '.join(improving)} improved, but ", "highlight": False},
                    {"text": declining[0].lower(), "highlight": True},
                    {"text": " needs a closer look", "highlight": False},
                ]
            else:
                headline_parts = [{"text": "A few metrics need a closer look", "highlight": False}]
            subtext = "Select any change below for more detail." if changes else ""

        return {
            "range": range,
            "headline_parts": headline_parts,
            "subtext": subtext,
            "updates": [
                {
                    "key": c["key"],
                    "title": c["title"],
                    "delta_label": c["delta_label"],
                    "subtitle": c["subtitle"],
                    "alarming": c["alarming"],
                }
                for c in updates
            ],
        }
    except Exception as e:
        return {"error": str(e)}