"""Pure helpers for Assist replies. No database or model imports."""

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# Assist shows times in India Standard Time, matching the emergency copy
# ("112 in India") and the timestamps handed to the model.
LOCAL_TZ = ZoneInfo("Asia/Kolkata")

_WEEKDAY_ABBR = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_CHARS = 800


def local_now() -> datetime:
    return datetime.now(LOCAL_TZ)


def local_today() -> date:
    return local_now().date()


def last_7_local_dates() -> list[date]:
    today = local_today()
    return [today - timedelta(days=6 - i) for i in range(7)]


def weekday_label(day: date) -> str:
    return _WEEKDAY_ABBR[day.weekday()]


def local_date_key(iso_str) -> str | None:
    """Calendar date in LOCAL_TZ for a measured_at timestamp."""
    if not iso_str:
        return None
    try:
        measured = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
    except ValueError:
        return None
    if measured.tzinfo is None:
        measured = measured.replace(tzinfo=timezone.utc)
    return measured.astimezone(LOCAL_TZ).date().isoformat()


def window_start_utc_iso(days: int = 6) -> str:
    """UTC timestamp for local midnight `days` ago, inclusive of today."""
    start = (local_now() - timedelta(days=days)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return start.astimezone(timezone.utc).isoformat()


def seconds_to_minutes(value) -> int:
    """Ring sleep columns are seconds. Insights divides by 60; chat must too."""
    try:
        seconds = int(value or 0)
    except (TypeError, ValueError):
        return 0
    if seconds <= 0:
        return 0
    return seconds // 60


def format_hm(total_minutes: int) -> str:
    hours, minutes = divmod(max(int(total_minutes), 0), 60)
    return f"{hours}h {minutes}m"


def sleep_card_data(row: dict) -> dict | None:
    """Chat sleep card from the real stage columns, in minutes."""
    if not row:
        return None
    total = seconds_to_minutes(row.get("total_duration"))
    awake = seconds_to_minutes(row.get("awake_duration"))
    light = seconds_to_minutes(row.get("light_duration"))
    deep = seconds_to_minutes(row.get("deep_duration"))
    rem = seconds_to_minutes(row.get("rem_duration"))
    if total <= 0 and awake + light + deep + rem <= 0:
        return None
    hours, minutes = divmod(total, 60)
    hour_word = "hour" if hours == 1 else "hours"
    minute_word = "minute" if minutes == 1 else "minutes"
    return {
        "type": "sleep_highlights",
        "data": {
            "time_awake_min": awake,
            "light_sleep_min": light,
            "deep_sleep_min": deep,
            "rem_sleep_min": rem,
            "total_label": f"{hours} {hour_word} and {minutes} {minute_word}",
        },
    }


def cycle_snapshot(period_start, cycle_length, period_length, today: date) -> dict:
    """Same day count as the app's CycleEngine.cycleDay: no modulo wrap.

    Phase names match CycleEngine.phaseName: Menstruation, Follicular,
    Ovulating, Luteal. A start date in the future yields current_day <= 0
    and phase '---'.
    """
    cycle_len = int(cycle_length or 28)
    period_len = int(period_length or 5)
    if cycle_len <= 0:
        cycle_len = 28
    if period_len <= 0:
        period_len = 5

    if isinstance(period_start, datetime):
        start = period_start.date()
        start_label = start.isoformat()
    elif isinstance(period_start, date):
        start = period_start
        start_label = start.isoformat()
    else:
        start_label = str(period_start or "")[:10]
        start = datetime.strptime(start_label, "%Y-%m-%d").date()

    delta = (today - start).days
    current_day = delta + 1
    if current_day <= 0:
        phase = "---"
        days_until = -delta
    elif current_day <= period_len:
        phase = "Menstruation"
        days_until = cycle_len - delta
    elif current_day <= 13:
        phase = "Follicular"
        days_until = cycle_len - delta
    elif current_day <= 16:
        phase = "Ovulating"
        days_until = cycle_len - delta
    else:
        phase = "Luteal"
        days_until = cycle_len - delta

    return {
        "period_start": start_label,
        "cycle_length": cycle_len,
        "period_length": period_len,
        "current_day": current_day,
        "phase": phase,
        "days_until_next": days_until,
    }


def message_text(content) -> str:
    """Plain text from a chat-model message. Drops reasoning blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                kind = block.get("type")
                if kind in (None, "text") and block.get("text"):
                    parts.append(str(block["text"]))
            else:
                text = getattr(block, "text", None)
                kind = getattr(block, "type", None)
                if text and kind in (None, "text"):
                    parts.append(str(text))
        return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    return str(content).strip()


def prior_turns(rows: list) -> list[tuple[str, str]]:
    """Last few user/assistant turns. Skips saved failure strings."""
    cleaned: list[tuple[str, str]] = []
    for row in rows or []:
        role = row.get("role")
        if role not in ("user", "assistant"):
            continue
        content = str(row.get("content") or "").strip()
        if not content:
            continue
        if content.startswith("Error:") or content.startswith("Agent error:"):
            continue
        if len(content) > MAX_HISTORY_CHARS:
            content = content[:MAX_HISTORY_CHARS] + "…"
        cleaned.append((role, content))
    return cleaned[-MAX_HISTORY_MESSAGES:]


_CARD_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("sleep", (r"\bsleep\b", r"\bslept\b", r"\binsomnia\b")),
    ("bp", (r"\bblood pressure\b", r"\bsystolic\b", r"\bdiastolic\b", r"\bbp\b")),
    (
        "spo2",
        (
            r"\bspo2\b",
            r"\bsp02\b",
            r"\bblood oxygen\b",
            r"\boxygen saturation\b",
            r"\boxygen level\b",
        ),
    ),
    ("hrv", (r"\bhrv\b", r"\bheart rate variability\b")),
    ("hr", (r"\bheart rate\b", r"\bpulse\b", r"\bbpm\b")),
    ("steps", (r"\bsteps\b", r"\bstep count\b", r"\bwalked\b", r"\bwalking\b")),
    ("temperature", (r"\btemperature\b", r"\bbody temp\b", r"\bfever\b", r"\btemp\b")),
    ("stress", (r"\bstress\b", r"\banxiety\b", r"\bstressed\b")),
    (
        "cycle",
        (
            r"\bmenstrual\b",
            r"\bmenstruation\b",
            r"\bovulation\b",
            r"\bpms\b",
            r"\bfertile\b",
            r"\bmy periods?\b",
            r"\bnext period\b",
            r"\bperiod start\b",
            r"\bmy cycle\b",
            r"\bwomen'?s health\b",
        ),
    ),
]


def select_card_kind(message: str) -> str | None:
    """One vital card, most specific match first. Whole words only."""
    import re

    msg = (message or "").lower()
    for kind, patterns in _CARD_RULES:
        if any(re.search(pattern, msg) for pattern in patterns):
            return kind
    return None
