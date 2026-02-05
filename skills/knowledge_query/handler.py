from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from skills.storage_governor import collector as storage_collector


SUPPORTED_EXAMPLES = [
    "latest disk report",
    "what changed this week?",
    "largest directory growth in /home",
    "any anomalies lately?",
]


class ParsedIntent:
    def __init__(
        self,
        intent: str | None,
        reason: str | None,
        start_date: str | None,
        end_date: str | None,
        path_filter: str | None,
        mount_filter: str | None,
    ) -> None:
        self.intent = intent
        self.reason = reason
        self.start_date = start_date
        self.end_date = end_date
        self.path_filter = path_filter
        self.mount_filter = mount_filter


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _debug_enabled() -> bool:
    return os.getenv("KNOWLEDGE_QUERY_DEBUG", "").strip() == "1"


def _basename(path: str) -> str:
    if path == "/":
        return "/"
    return path.rstrip("/").split("/")[-1]


def _resolve_mount_key(db: Any, mountpoint: str) -> tuple[str, str | None]:
    latest_taken_at = db.get_latest_disk_snapshot_taken_at()
    if not latest_taken_at:
        return mountpoint, None
    candidates = db.list_disk_snapshot_mountpoints_for_taken_at(latest_taken_at)
    if mountpoint in candidates:
        return mountpoint, None
    base = _basename(mountpoint)
    if base == "/":
        return mountpoint, None
    matches = [m for m in candidates if _basename(m) == base]
    if len(matches) == 1:
        return matches[0], "matched_by_basename"
    return mountpoint, "mount_key_not_found"

def _now_local(tz_name: str) -> date:
    tz = ZoneInfo(tz_name or "UTC")
    return datetime.now(tz).date()


def _start_of_week(today: date) -> date:
    return today - timedelta(days=today.weekday())


def _parse_date_range(text: str, tz_name: str) -> tuple[str, str] | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})\s*(?:to|-|–)\s*(\d{4}-\d{2}-\d{2})", text)
    if not match:
        return None
    start = date.fromisoformat(match.group(1))
    end = date.fromisoformat(match.group(2))
    if start > end:
        start, end = end, start
    return start.isoformat(), end.isoformat()


def _parse_time_window(text: str, tz_name: str) -> tuple[str, str]:
    today = _now_local(tz_name)
    if "today" in text:
        return today.isoformat(), today.isoformat()
    if "yesterday" in text:
        day = today - timedelta(days=1)
        return day.isoformat(), day.isoformat()
    if "this week" in text:
        start = _start_of_week(today)
        return start.isoformat(), today.isoformat()
    if "last week" in text:
        start = _start_of_week(today) - timedelta(days=7)
        end = start + timedelta(days=6)
        return start.isoformat(), end.isoformat()

    match = re.search(r"last\s+(\d{1,2})\s+days", text)
    if match:
        days = max(1, min(90, int(match.group(1))))
        start = today - timedelta(days=days - 1)
        return start.isoformat(), today.isoformat()

    range_match = _parse_date_range(text, tz_name)
    if range_match:
        return range_match

    start = today - timedelta(days=6)
    return start.isoformat(), today.isoformat()


def _parse_path_filter(text: str) -> tuple[str | None, str | None]:
    if "/var/log" in text:
        return "/var/log", "/"
    if "/home" in text:
        return "/home", "/"
    if "downloads" in text:
        return "Downloads", "/"
    if "/data2" in text:
        return "/data2", "/data2"
    if "/data" in text:
        return "/data", "/data"
    if "/" in text:
        return "/", "/"
    return None, None


def _match_intents(text: str) -> list[str]:
    intents = []
    if any(phrase in text for phrase in ("latest", "last snapshot", "last report", "since last snapshot")):
        intents.append("latest_snapshot_summary")
    if any(phrase in text for phrase in ("what changed", "changes", "change over", "changed")):
        intents.append("time_window_storage_changes")
    if any(phrase in text for phrase in ("grew", "growth", "largest", "biggest")) and any(
        word in text for word in ("folder", "directory", "path", "dirs", "directories")
    ):
        intents.append("top_growth_paths")
    if any(phrase in text for phrase in ("anomal", "flag", "unusual")):
        intents.append("anomalies_in_window")
    return list(dict.fromkeys(intents))


def parse_intent(query: str, tz_name: str) -> ParsedIntent:
    text = _normalize(query)
    intents = _match_intents(text)
    if len(intents) != 1:
        return ParsedIntent(None, "ambiguous_or_unknown", None, None, None, None)

    start_date, end_date = _parse_time_window(text, tz_name)
    path_filter, mount_filter = _parse_path_filter(text)
    return ParsedIntent(intents[0], None, start_date, end_date, path_filter, mount_filter)


def _bytes_to_human(num_bytes: int) -> str:
    if num_bytes < 0:
        return "0B"
    units = ["B", "K", "M", "G", "T", "P", "E"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}B"
            formatted = f"{value:.1f}".rstrip("0").rstrip(".")
            return f"{formatted}{unit}"
        value /= 1024
    return f"{int(value)}B"


def _render_report(intent: ParsedIntent, facts: dict[str, Any], limits: dict[str, Any]) -> str:
    lines = []
    if intent.intent:
        lines.append(f"Knowledge query: {intent.intent}")
    if intent.start_date and intent.end_date:
        lines.append(f"Window: {intent.start_date} to {intent.end_date}")
    is_time_window = intent.intent == "time_window_storage_changes"
    if is_time_window:
        snapshot_count = facts.get("snapshot_count")
        latest_ts = facts.get("latest_snapshot_ts")
        if snapshot_count is not None:
            lines.append(f"Snapshots in window: {snapshot_count}")
        if latest_ts:
            lines.append(f"Latest snapshot timestamp: {latest_ts}")

    if facts.get("available") is False:
        lines.append("Data availability: not available")
        reason = facts.get("reason")
        if reason:
            lines.append(f"- reason: {reason}")
        if not is_time_window:
            return "\n".join(lines)

    if intent.intent == "latest_snapshot_summary":
        lines.append("Latest disk snapshots:")
        mounts = facts.get("mounts", [])
        if not mounts:
            lines.append("- no snapshots available")
        for row in mounts:
            used = _bytes_to_human(int(row["used_bytes"]))
            total = _bytes_to_human(int(row["total_bytes"]))
            delta = row.get("delta_used")
            delta_reason = row.get("delta_reason")
            delta_str = _bytes_to_human(abs(int(delta))) if delta is not None else ""
            sign = "+" if delta is not None and int(delta) >= 0 else "-"
            if delta is None:
                delta_out = f"not available ({delta_reason or 'missing_previous_snapshot'})"
            else:
                delta_out = f"{sign}{delta_str}"
            lines.append(
                f"- {row['mountpoint']} used {used} / {total} (delta {delta_out} since previous)"
            )
    elif intent.intent == "time_window_storage_changes":
        lines.append("Storage changes:")
        mounts = facts.get("mounts", [])
        if not mounts:
            lines.append("- no snapshots available")
        if facts.get("available") is False:
            reason = facts.get("reason") or "missing_data"
            lines.append(f"- not available ({reason})")
        for row in mounts:
            delta = row.get("delta_used")
            delta_reason = row.get("delta_reason")
            delta_str = _bytes_to_human(abs(int(delta))) if delta is not None else ""
            sign = "+" if delta is not None and int(delta) >= 0 else "-"
            if delta is None:
                reason_label = delta_reason or "missing_data"
                delta_out = f"not available ({reason_label})"
            else:
                delta_out = f"{sign}{delta_str}"
            latest_used = row.get("latest_used_bytes")
            latest_total = row.get("latest_total_bytes")
            latest_pct = row.get("latest_used_percent")
            latest_ts = row.get("latest_taken_at")
            latest_suffix = ""
            if latest_used is not None and latest_total is not None:
                used = _bytes_to_human(int(latest_used))
                total = _bytes_to_human(int(latest_total))
                if latest_pct is None:
                    latest_suffix = f"; latest {used} / {total}"
                else:
                    latest_suffix = f"; latest {used} / {total} ({latest_pct:.1f}%)"
                if latest_ts:
                    latest_suffix += f" at {latest_ts}"
            lines.append(f"- {row['mountpoint']} used change: {delta_out}{latest_suffix}")
    elif intent.intent == "top_growth_paths":
        lines.append("Top path growth:")
        items = facts.get("top_paths", [])
        if not items:
            lines.append("- no growth data available")
        for item in items:
            delta = _bytes_to_human(abs(int(item["delta_bytes"])))
            sign = "+" if int(item["delta_bytes"]) >= 0 else "-"
            lines.append(f"- {item['path']}: {sign}{delta}")
    elif intent.intent == "anomalies_in_window":
        lines.append("Anomalies:")
        anomalies = facts.get("anomalies", [])
        if not anomalies:
            lines.append("- none recorded")
        for flag in anomalies:
            lines.append(f"- {flag}")

    notes = limits.get("notes") or []
    if notes:
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines)


def _latest_snapshot_summary(db: Any, mount_filter: str | None) -> dict[str, Any]:
    mounts = storage_collector.DEFAULT_MOUNTPOINTS
    if mount_filter:
        mounts = [m for m in mounts if m == mount_filter]
    results = []
    for mount in mounts:
        resolved_mount, resolution_note = _resolve_mount_key(db, mount)
        latest = db.get_latest_disk_snapshot(resolved_mount)
        if not latest:
            continue
        prev = db.get_previous_disk_snapshot(resolved_mount, latest.get("taken_at"))
        delta = None
        delta_reason = None
        if prev:
            delta = int(latest["used_bytes"]) - int(prev["used_bytes"])
        else:
            delta_reason = "missing_previous_snapshot"
        results.append(
            {
                "mountpoint": mount,
                "resolved_mountpoint": resolved_mount if resolved_mount != mount else None,
                "mount_resolution": resolution_note,
                "taken_at": latest.get("taken_at"),
                "used_bytes": int(latest["used_bytes"]),
                "total_bytes": int(latest["total_bytes"]),
                "delta_used": delta,
                "delta_reason": delta_reason,
            }
        )
    if not results:
        return {"available": False, "reason": "no_snapshots"}
    return {"available": True, "mounts": results}


def _storage_changes_in_window(
    db: Any, start_date: str, end_date: str, mount_filter: str | None
) -> dict[str, Any]:
    mounts = storage_collector.DEFAULT_MOUNTPOINTS
    if mount_filter:
        mounts = [m for m in mounts if m == mount_filter]
    results = []
    notes: list[str] = []
    window_stats = db.get_disk_snapshot_window_stats(start_date, end_date)
    snapshot_count = int(window_stats.get("snapshot_count") or 0)
    earliest_date = window_stats.get("earliest_snapshot_date")
    latest_date = window_stats.get("latest_snapshot_date")
    earliest_ts = window_stats.get("earliest_taken_at")
    latest_ts = window_stats.get("latest_taken_at")

    if snapshot_count < 2:
        notes.append("insufficient_snapshots_in_window")

    for mount in mounts:
        resolved_mount, resolution_note = _resolve_mount_key(db, mount)
        delta = None
        delta_reason = None
        if snapshot_count >= 2 and earliest_date and latest_date:
            start_row = db.get_disk_snapshot_for_mount_and_date(resolved_mount, earliest_date)
            end_row = db.get_disk_snapshot_for_mount_and_date(resolved_mount, latest_date)
            if start_row and end_row:
                delta = int(end_row["used_bytes"]) - int(start_row["used_bytes"])
            else:
                if not start_row and not end_row:
                    delta_reason = "missing_start_end_snapshot"
                elif not start_row:
                    delta_reason = "missing_start_snapshot"
                else:
                    delta_reason = "missing_end_snapshot"
        else:
            delta_reason = "insufficient_snapshots_in_window"

        latest = db.get_latest_disk_snapshot(resolved_mount)
        latest_used = latest_total = latest_free = latest_pct = latest_taken_at = None
        if latest:
            latest_used = int(latest["used_bytes"])
            latest_total = int(latest["total_bytes"])
            latest_free = int(latest["free_bytes"])
            latest_taken_at = latest.get("taken_at")
            if latest_total > 0:
                latest_pct = (latest_used / latest_total) * 100

        results.append(
            {
                "mountpoint": mount,
                "resolved_mountpoint": resolved_mount if resolved_mount != mount else None,
                "mount_resolution": resolution_note,
                "delta_used": delta,
                "delta_reason": delta_reason,
                "latest_used_bytes": latest_used,
                "latest_total_bytes": latest_total,
                "latest_free_bytes": latest_free,
                "latest_used_percent": latest_pct,
                "latest_taken_at": latest_taken_at,
            }
        )

    if snapshot_count == 0:
        return {
            "available": False,
            "reason": "no_snapshots",
            "snapshot_count": snapshot_count,
            "earliest_snapshot_ts": earliest_ts,
            "latest_snapshot_ts": latest_ts,
            "notes": notes,
            "mounts": results,
        }
    return {
        "available": True,
        "mounts": results,
        "snapshot_count": snapshot_count,
        "earliest_snapshot_ts": earliest_ts,
        "latest_snapshot_ts": latest_ts,
        "earliest_snapshot_date": earliest_date,
        "latest_snapshot_date": latest_date,
        "notes": notes,
    }


def _top_growth_paths(
    db: Any, start_date: str, end_date: str, path_filter: str | None
) -> dict[str, Any]:
    scopes = ["root_top", "home_top"]
    if path_filter:
        if path_filter.startswith("/home") or "Downloads" in path_filter:
            scopes = ["home_top"]
        elif path_filter.startswith("/"):
            scopes = ["root_top"]
    best: list[dict[str, Any]] = []
    for scope in scopes:
        dates = db.list_dir_size_sample_dates(scope, start_date, end_date)
        if len(dates) < 2:
            continue
        first_date = dates[0]
        last_date = dates[-1]
        first = db.list_dir_size_samples_for_date(scope, first_date)
        last = db.list_dir_size_samples_for_date(scope, last_date)
        if not first or not last:
            continue
        first_map = {row["path"]: int(row["bytes"]) for row in first}
        last_map = {row["path"]: int(row["bytes"]) for row in last}
        for path, last_val in last_map.items():
            if path_filter and path_filter not in path:
                continue
            if path not in first_map:
                continue
            delta = int(last_val) - int(first_map[path])
            best.append({"path": path, "delta_bytes": delta, "scope": scope})
    if not best:
        return {"available": False, "reason": "dir_growth_not_stored"}
    best.sort(key=lambda item: item["delta_bytes"], reverse=True)
    return {"available": True, "top_paths": best[:5]}


def _anomalies_in_window(db: Any, start_date: str, end_date: str, user_id: str | None) -> dict[str, Any]:
    if not user_id:
        return {"available": False, "reason": "missing_user_id"}
    try:
        rows = db.get_anomalies(user_id, start_date, end_date)
    except Exception as exc:
        message = str(exc)
        if "anomaly_events" in message.lower():
            return {"available": False, "reason": "anomaly_events_table_missing"}
        return {"available": False, "reason": "anomaly_query_failed"}
    return {"available": True, "anomalies": rows}


def knowledge_query(context: dict[str, Any], query: str) -> dict[str, Any]:
    db = (context or {}).get("db")
    tz_name = (context or {}).get("timezone") or "UTC"
    user_id = (context or {}).get("user_id")
    if not db:
        text = "Database not available."
        return {"text": text, "data": {"intent": None, "facts": {"available": False}}}

    parsed = parse_intent(query, tz_name)
    if not parsed.intent:
        text = (
            "I can answer a few types of questions. Which one do you mean?\n"
            "Examples: " + "; ".join(SUPPORTED_EXAMPLES)
        )
        return {
            "text": text,
            "data": {"intent": None, "facts": {}, "limits": {"notes": ["clarification_required"]}},
        }

    facts: dict[str, Any]
    if parsed.intent == "latest_snapshot_summary":
        facts = _latest_snapshot_summary(db, parsed.mount_filter)
    elif parsed.intent == "time_window_storage_changes":
        facts = _storage_changes_in_window(db, parsed.start_date or "", parsed.end_date or "", parsed.mount_filter)
    elif parsed.intent == "top_growth_paths":
        facts = _top_growth_paths(db, parsed.start_date or "", parsed.end_date or "", parsed.path_filter)
    else:
        facts = _anomalies_in_window(
            db,
            parsed.start_date or "",
            parsed.end_date or "",
            str(user_id) if user_id is not None else None,
        )

    limits = {
        "time_window": {"start": parsed.start_date, "end": parsed.end_date},
        "rows": len(facts.get("mounts", [])) + len(facts.get("top_paths", [])),
        "notes": [],
    }
    if facts.get("available") is False:
        limits["notes"].append(facts.get("reason"))
    if facts.get("notes"):
        limits["notes"].extend([note for note in facts.get("notes") if note])

    report_text = _render_report(parsed, facts, limits)
    debug: dict[str, Any] | None = None
    if _debug_enabled() and parsed.intent == "time_window_storage_changes":
        window_start = parsed.start_date
        window_end = parsed.end_date
        window_stats = db.get_disk_snapshot_window_stats(window_start or "", window_end or "")
        latest_taken_at = db.get_latest_disk_snapshot_taken_at()
        latest_mounts = (
            db.list_disk_snapshot_mountpoints_for_taken_at(latest_taken_at)
            if latest_taken_at
            else []
        )
        earliest_date = window_stats.get("earliest_snapshot_date")
        latest_date = window_stats.get("latest_snapshot_date")
        mount_rows = []
        mount_keys = facts.get("mounts", [])
        for row in mount_keys:
            mountpoint = row.get("mountpoint")
            start_row = (
                db.get_disk_snapshot_for_mount_and_date(mountpoint, earliest_date)
                if earliest_date
                else None
            )
            end_row = (
                db.get_disk_snapshot_for_mount_and_date(mountpoint, latest_date)
                if latest_date
                else None
            )
            mount_rows.append(
                {
                    "mountpoint": mountpoint,
                    "has_start_snapshot": bool(start_row),
                    "has_end_snapshot": bool(end_row),
                }
            )
        debug = {
            "window_start": window_start,
            "window_end": window_end,
            "snapshot_count_in_window": int(window_stats.get("snapshot_count") or 0),
            "earliest_snapshot_ts_in_window": window_stats.get("earliest_taken_at"),
            "latest_snapshot_ts_in_window": window_stats.get("latest_taken_at"),
            "mount_keys_in_latest_snapshot": latest_mounts[:10],
            "mount_keys_requested": [row.get("mountpoint") for row in mount_keys],
            "mount_snapshot_matches": mount_rows,
            "db_methods_used": [
                "get_disk_snapshot_window_stats",
                "get_disk_snapshot_for_mount_and_date",
                "get_latest_disk_snapshot",
                "get_latest_disk_snapshot_taken_at",
                "list_disk_snapshot_mountpoints_for_taken_at",
            ],
        }
    return {
        "text": report_text,
        "data": {
            "intent": {
                "name": parsed.intent,
                "start_date": parsed.start_date,
                "end_date": parsed.end_date,
                "path_filter": parsed.path_filter,
                "mount_filter": parsed.mount_filter,
            },
            "facts": facts,
            "limits": limits,
            "debug": debug,
        },
    }
