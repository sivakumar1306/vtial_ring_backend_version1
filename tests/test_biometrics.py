"""Assist biometric formatting. No network, no model."""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.biometrics import (
    cycle_snapshot,
    format_hm,
    local_date_key,
    message_text,
    prior_turns,
    seconds_to_minutes,
    select_card_kind,
    sleep_card_data,
)


def test_sleep_seconds_become_hours_and_minutes():
    # 8 hours stored as seconds must not be reported as 480 hours.
    assert seconds_to_minutes(28800) == 480
    assert format_hm(480) == "8h 0m"
    # A 20-minute nap is 1200 seconds, not 1200 minutes.
    assert seconds_to_minutes(1200) == 20
    assert format_hm(20) == "0h 20m"


def test_sleep_card_uses_real_stages():
    card = sleep_card_data({
        "total_duration": 28800,
        "awake_duration": 1800,
        "light_duration": 14400,
        "deep_duration": 7200,
        "rem_duration": 5400,
    })
    data = card["data"]
    assert data["time_awake_min"] == 30
    assert data["light_sleep_min"] == 240
    assert data["deep_sleep_min"] == 120
    assert data["rem_sleep_min"] == 90
    assert data["total_label"] == "8 hours and 0 minutes"
    assert sleep_card_data({"total_duration": 0}) is None


def test_cycle_day_matches_the_app_engine():
    # CycleEngine.cycleDay is the calendar difference plus one, not modulo.
    snap = cycle_snapshot("2026-09-01", 28, 5, date(2026, 9, 10))
    assert snap["current_day"] == 10
    assert snap["phase"] == "Follicular"
    assert snap["days_until_next"] == 19

    assert cycle_snapshot("2026-09-01", 28, 5, date(2026, 9, 3))["phase"] == "Menstruation"
    assert cycle_snapshot("2026-09-01", 28, 5, date(2026, 9, 14))["phase"] == "Ovulating"
    assert cycle_snapshot("2026-09-01", 28, 5, date(2026, 9, 20))["phase"] == "Luteal"
    assert cycle_snapshot("2026-09-10", 28, 5, date(2026, 9, 1))["phase"] == "---"


def test_card_routing_uses_whole_words():
    assert select_card_kind("how did I sleep") == "sleep"
    assert select_card_kind("how is my heart rate variability") == "hrv"
    assert select_card_kind("how is my heart rate") == "hr"
    assert select_card_kind("I was snoring last night") is None
    assert select_card_kind("what is a time period") is None
    assert select_card_kind("when is my period") == "cycle"
    assert select_card_kind("contemporary design") is None
    assert select_card_kind("my body temperature") == "temperature"
    assert select_card_kind("I feel stressed") == "stress"


def test_timestamps_bucket_on_india_time():
    # 20:00 UTC is 01:30 the next day in IST.
    assert local_date_key("2026-10-05T20:00:00Z") == "2026-10-06"


def test_prior_turns_drop_saved_errors_and_cap_length():
    rows = [{"role": "assistant", "content": "Error: boom"}]
    rows += [{"role": "user", "content": f"m{i}"} for i in range(10)]
    turns = prior_turns(rows)
    assert len(turns) == 8
    assert all(not content.startswith("Error:") for _, content in turns)
    long = prior_turns([{"role": "user", "content": "x" * 900}])
    assert len(long[0][1]) == 801


def test_message_text_skips_reasoning_blocks():
    raw = [
        {"type": "reasoning", "text": "hidden"},
        {"type": "text", "text": "Heart rate is 72 bpm."},
    ]
    assert message_text(raw) == "Heart rate is 72 bpm."
    assert message_text("  plain  ") == "plain"
