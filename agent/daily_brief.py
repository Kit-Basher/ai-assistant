from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class DailyBriefDecision:
    should_send: bool
    local_date: str
    reason: str


def parse_hhmm(value: str) -> tuple[int, int] | None:
    text = (value or "").strip()
    parts = text.split(":")
    if len(parts) != 2:
        return None
    if not parts[0].isdigit() or not parts[1].isdigit():
        return None
    hh = int(parts[0])
    mm = int(parts[1])
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        return None
    return hh, mm


def should_send_daily_brief(
    now_utc: datetime,
    timezone_name: str,
    enabled: bool,
    local_time_hhmm: str,
    last_sent_local_date: str | None,
    quiet_mode: bool = False,
    disk_delta_mb: float | None = None,
    disk_delta_threshold_mb: float = 250.0,
    service_unhealthy: bool = False,
    only_send_if_service_unhealthy: bool = False,
    has_due_open_loops: bool = False,
) -> DailyBriefDecision:
    local_now = now_utc.astimezone(ZoneInfo(timezone_name))
    local_date = local_now.date().isoformat()
    if not enabled:
        return DailyBriefDecision(False, local_date, "disabled")
    parsed = parse_hhmm(local_time_hhmm)
    if not parsed:
        return DailyBriefDecision(False, local_date, "invalid_time")
    hh, mm = parsed
    if (local_now.hour, local_now.minute) < (hh, mm):
        return DailyBriefDecision(False, local_date, "before_time")
    if (last_sent_local_date or "").strip() == local_date:
        return DailyBriefDecision(False, local_date, "already_sent_today")
    if only_send_if_service_unhealthy and not service_unhealthy:
        return DailyBriefDecision(False, local_date, "service_healthy_gate")
    if quiet_mode:
        disk_notable = False
        if disk_delta_mb is not None:
            disk_notable = abs(float(disk_delta_mb)) >= float(disk_delta_threshold_mb)
        if not (disk_notable or service_unhealthy or has_due_open_loops):
            return DailyBriefDecision(False, local_date, "quiet_no_signals")
    return DailyBriefDecision(True, local_date, "send")
