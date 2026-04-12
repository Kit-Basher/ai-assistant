from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any


DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
RANGE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\s*(?:to|through|-)\s*(\d{4}-\d{2}-\d{2})")


@dataclass
class TimeframeResult:
    ok: bool
    clarify: bool
    label: str
    start_date: str | None
    end_date: str | None
    start_ts: str | None
    end_ts: str | None
    reason: str | None = None


def parse_timeframe(text: str, db: Any, timezone: str) -> TimeframeResult:
    lowered = (text or "").lower()

    latest_date = None
    latest_ts = None
    if db:
        latest_date = db.get_latest_snapshot_local_date_any()
        latest_ts = db.get_latest_snapshot_taken_at_any()

    if not latest_date:
        return TimeframeResult(False, False, "no snapshots", None, None, None, None, "no_snapshots")

    if "last week" in lowered:
        end_date = _previous_week_end(latest_date, timezone)
        start_date = (end_date - timedelta(days=6)).isoformat()
        return TimeframeResult(True, False, "last week", start_date, end_date.isoformat(), None, None)

    if "recently" in lowered:
        if not latest_ts:
            return TimeframeResult(False, False, "recently", None, None, None, None, "no_snapshots")
        end_dt = datetime.fromisoformat(latest_ts)
        start_dt = end_dt - timedelta(hours=72)
        return TimeframeResult(
            True,
            False,
            "last 72 hours",
            start_dt.date().isoformat(),
            end_dt.date().isoformat(),
            start_dt.isoformat(),
            end_dt.isoformat(),
        )

    if "lately" in lowered:
        end_date = latest_date
        start_date = (datetime.fromisoformat(end_date).date() - timedelta(days=6)).isoformat()
        return TimeframeResult(True, False, "last 7 days", start_date, end_date, None, None)

    match = RANGE_RE.search(lowered)
    if match:
        start_date = match.group(1)
        end_date = match.group(2)
        return TimeframeResult(True, False, f"{start_date} to {end_date}", start_date, end_date, None, None)

    date_match = DATE_RE.search(lowered)
    if date_match:
        day = date_match.group(1)
        return TimeframeResult(True, False, f"on {day}", day, day, None, None)

    if _ambiguous_timeframe(lowered):
        return TimeframeResult(False, True, "", None, None, None, None, "ambiguous")

    end_date = latest_date
    start_date = (datetime.fromisoformat(end_date).date() - timedelta(days=6)).isoformat()
    return TimeframeResult(
        True,
        False,
        f"last 7 days ending {end_date}",
        start_date,
        end_date,
        None,
        None,
    )


def _ambiguous_timeframe(lowered: str) -> bool:
    if "last" in lowered and "last week" not in lowered and "last 7" not in lowered and "last 72" not in lowered:
        return True
    if "past" in lowered and "past week" not in lowered and "past 7" not in lowered and "past 72" not in lowered:
        return True
    return False


def _previous_week_end(latest_date_str: str, timezone: str) -> datetime:
    tzinfo = ZoneInfo(timezone)
    latest_date = datetime.fromisoformat(latest_date_str).date()
    latest_dt = datetime(latest_date.year, latest_date.month, latest_date.day, tzinfo=tzinfo)
    weekday = latest_dt.weekday()  # Monday=0
    current_week_start = latest_dt - timedelta(days=weekday)
    previous_week_end = current_week_start - timedelta(days=1)
    return previous_week_end
