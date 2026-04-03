from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def diff_disk_reports(old_snapshot: dict[str, Any], new_snapshot: dict[str, Any]) -> dict[str, Any]:
    if not old_snapshot or not new_snapshot:
        return {"has_changes": False, "grew": [], "shrank": [], "top_growth": []}
    if old_snapshot == new_snapshot:
        return {"has_changes": False, "grew": [], "shrank": [], "top_growth": []}
    return {"has_changes": True, "grew": [], "shrank": [], "top_growth": []}


def time_since(iso_ts: str | None) -> str:
    if not iso_ts:
        return "unknown"
    try:
        dt = datetime.fromisoformat(str(iso_ts))
    except ValueError:
        return "unknown"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta = now - dt
    days = delta.days
    seconds = delta.seconds
    if days > 0:
        return f"{days}d ago"
    hours = seconds // 3600
    if hours > 0:
        return f"{hours}h ago"
    minutes = seconds // 60
    if minutes > 0:
        return f"{minutes}m ago"
    return "just now"
