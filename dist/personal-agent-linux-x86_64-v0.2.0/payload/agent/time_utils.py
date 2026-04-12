from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


def parse_local_datetime(dt_str: str, tz_name: str) -> datetime:
    # Expected format: YYYY-MM-DD HH:MM
    local = datetime.strptime(dt_str.strip(), "%Y-%m-%d %H:%M")
    return local.replace(tzinfo=ZoneInfo(tz_name))


def to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(ZoneInfo("UTC")).isoformat()
